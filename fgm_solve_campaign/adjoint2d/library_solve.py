"""Shape-fidelity solve across the standardized shape library.

One shape per invocation. Arms, all scored under the SAME objective

    J(s, t_stop) = sum over the WHOLE domain of (phi(x, t_stop) - chi_part(x))^2

with each arm at its OWN optimal stop, t_stop = argmin over that arm's own
stored trajectory of J (the "J-stop" convention of
`VERIFICATION_PRINTABILITY_REPORT.md`):

  U_uniform     uniform saturation s = 1
  HIST_*        every ACTUAL stored 4-bits-per-pixel dopant map for this shape,
                from both the old {0.30 .. 0.85} grid campaign
                (`outputs_eqs/geometry_dual_readstate`) and the calibration
                campaign (`outputs_eqs/fgm_calibrated_control`), loaded through
                the PRODUCTION loader and scored in the permittivity-co-varying
                channel they were run in. Two boundary conventions are scored
                for each map: as stored (saturation zero outside the part) and
                the prototype convention (nominal 1 outside the part). The best
                of all of them by J is the historical baseline. The old grid is
                NOT assumed to lose: on the square it wins.
  A1_cont       the solved per-cell map, box [0, 1], single printing pass
  A1_4bpp       that map on the printer's 16-level grid, re-run through the real
                forward. THE deliverable arm.
  A1_2bpp       the same on the 4-level grid
  A15_*         box [0, 1.5], a SECOND printing pass, run only when the
                single-pass deliverable arm leaves more than 15 percent of the
                part unmelted

Every scored run reports the standing energy-residual gate
(`adjoint2d.energy_gate`, threshold 5 percent of integrated dose) evaluated AT
that arm's own stop index, alongside the three clip gates.

Run:
  ./.venv312/bin/python -m adjoint2d.library_solve <shape> <outdir> [budget]
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import adjoint, control, energy_gate as eg, forward as fwd, gradops
from . import printability as pq
from . import shape_objective as so
from .pins import build_case, load_cfg
from .shape_solve import run_adjoint
from .verify_hist import CFGD, FCC, GDR, load_stored_map

SHAPES = (
    "square", "circle", "hexagon", "triangle", "equilateral_triangle",
    "L_shape", "H_shape", "T_shape", "cross", "diamond", "ellipse",
    "octagon", "pentagon", "rectangle", "rounded_rect", "star", "star6",
    "trapezoid",
)

GT_LOGO_SKIP_REASON = (
    "gt_logo is SKIPPED: its geometry is rasterized from an image and "
    "`rfam_eqs_coupled.make_domain` raises ModuleNotFoundError: No module named "
    "'cv2' in the .venv312 interpreter. Verified by calling build_case on "
    "outputs_eqs/fgm_calibrated_control/configs/gt_logo_m0p6054.yaml."
)

PATIENCE = 250
BUDGET_FORWARD_EQUIVALENTS = 40.0
DOUBLE_PASS_UNDER_MELT_TRIGGER_PCT = 15.0
TIE_BAND_J = 0.05          # relative J
TIE_BAND_IOU = 0.02        # IoU points
SOLVED_IOU = 0.95


# ---------------------------------------------------------------------------
# pure logic (unit tested)
# ---------------------------------------------------------------------------

def needs_double_pass(under_melt_pct: float) -> bool:
    """The L-class trigger: more than 15 percent of the part left unmelted."""
    return bool(float(under_melt_pct) > DOUBLE_PASS_UNDER_MELT_TRIGGER_PCT)


def best_arm_by_J(arms: dict) -> str | None:
    if not arms:
        return None
    return min(arms, key=lambda k: arms[k]["J"])


def classify(iou_solved: float, J_solved: float, iou_hist: float, J_hist: float) -> str:
    """The four-way ladder.

    SOLVED is absolute (the melted region IS the nominal part to within 5 IoU
    points). The other three are COMPARATIVE against the best historical mask,
    with a tie band of 5 percent on J and 2 IoU points. When the two metrics
    disagree in direction beyond the band the verdict is MATCHED, because a
    disagreement is not evidence of a win in either direction.
    """
    if float(iou_solved) >= SOLVED_IOU:
        return "SOLVED"
    dJ = (float(J_hist) - float(J_solved)) / max(abs(float(J_hist)), 1e-30)
    dI = float(iou_solved) - float(iou_hist)
    if dJ >= TIE_BAND_J and dI >= -TIE_BAND_IOU:
        return "IMPROVED"
    if dI >= TIE_BAND_IOU and dJ >= -TIE_BAND_J:
        return "IMPROVED"
    if abs(dJ) < TIE_BAND_J and abs(dI) < TIE_BAND_IOU:
        return "MATCHED"
    if dJ <= -TIE_BAND_J and dI <= TIE_BAND_IOU:
        return "NOT RESCUED"
    if dI <= -TIE_BAND_IOU and dJ <= TIE_BAND_J:
        return "NOT RESCUED"
    return "MATCHED"


def shape_config(shape: str) -> Path:
    """The calibrated configuration for this shape.

    COMPUTED: within a shape, every `fgm_calibrated_control` config differs from
    every other in exactly one line, the path of the stored dopant map, which
    the prototype never reads (it builds its own design variable). The drive
    voltage, geometry, thermal, densification and material blocks are identical.
    Verified by `diff` on square and star. The first name in sorted order is
    therefore a deterministic and physically arbitrary choice.
    """
    cands = sorted(CFGD.glob(f"{shape}_m*.yaml"))
    if not cands:
        raise FileNotFoundError(f"no calibrated config for {shape!r}")
    return cands[0]


def stored_mask_catalog(shape: str) -> list[tuple[str, Path]]:
    """Every distinct stored 4-bits-per-pixel dopant map for this shape.

    Both campaigns are scanned. Duplicate files (the same bytes written into two
    map directories) are collapsed so a mask is not scored twice.
    """
    out: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for camp, tag, root in (("geometry_dual_readstate", "oldgrid", GDR),
                            ("fgm_calibrated_control", "cal", FCC)):
        for p in sorted((root / shape).glob("map_m*/*.npz")):
            digest = hashlib.md5(p.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            out.append((f"{tag}_{p.parent.name}_{p.stem.split('_')[-1]}", p))
    return out


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score(case, s: np.ndarray, eps_covary: bool = False) -> dict:
    """One forward run, scored at its own J-stop, with every standing gate."""
    tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=PATIENCE,
                     eps_covary=eps_covary)
    m = so.full_metrics(tr, case)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["eps_covary"] = bool(eps_covary)
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_max_in_part"] = float(np.max(s[case.part_mask]))
    m["energy_gate"] = eg.gate_from_trajectory(tr, m["t_stop_index"])
    return m


def historical_scan(case, cfg: dict, catalog: list[tuple[str, Path]]) -> dict:
    pm = case.part_mask
    arms: dict[str, dict] = {}
    for label, path in catalog:
        s = load_stored_map(case, path, cfg)
        for conv, sat in (("asstored", s), ("outside1", np.where(pm, s, 1.0))):
            name = f"hist_{label}_{conv}_eps"
            m = score(case, sat, eps_covary=True)
            m["arm"] = name
            m["map_npz"] = str(path)
            m["convention"] = conv
            arms[name] = m
    return arms


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def solve_arm(case, ops, tag: str, box, n_evals: int) -> tuple[dict, np.ndarray, list[dict]]:
    rows, store = run_adjoint(case, ops, n_evals, box)
    if not rows:
        raise RuntimeError(f"{tag}: no evaluations completed")
    best = min(rows, key=lambda r: r["J"])
    s_best = store[str(best["eval_index"])]
    return best, s_best, rows


def quantized_variants(s_cont: np.ndarray, pm: np.ndarray, sat_max: float) -> dict:
    return {
        "4bpp": pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=sat_max),
        "2bpp": pq.quantize_in_part(s_cont, pm, bpp=2, sat_max=sat_max),
    }


def main(shape: str, outdir: str, budget: float = BUDGET_FORWARD_EQUIVALENTS) -> dict:
    if shape not in SHAPES:
        raise ValueError(f"{shape!r} is not in the standardized library; {GT_LOGO_SKIP_REASON}"
                         if shape == "gt_logo" else f"{shape!r} unknown")
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    cfg_path = shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)

    res: dict = {
        "shape": shape,
        "config": str(cfg_path),
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "n_part_cells": case.n_part,
        "budget_forward_equivalents": float(budget),
        "stop_convention": "t_stop = argmin over the arm's own trajectory of J",
        "arms": {},
        "hist_scan": {},
    }
    maps_store: dict[str, np.ndarray] = {}

    # --- cost model, measured on this shape -------------------------------
    s_u = np.ones(pm.shape)
    t_a = time.perf_counter()
    tr_u = fwd.forward(case, s_u, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=PATIENCE)
    t_b = time.perf_counter()
    st_u = so.optimal_stop(tr_u, case)
    _J, seed = so.shape_J_and_seed(tr_u.T_at_end(st_u.index), case)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {st_u.index: seed}, grad_ops=ops)
    t_d = time.perf_counter()
    ratio = (t_d - t_c) / max(t_b - t_a, 1e-9)
    n_evals = control.max_gradient_evals(float(budget), ratio)
    res["cost"] = {"forward_s": t_b - t_a, "adjoint_s": t_d - t_c, "ratio": ratio,
                   "n_gradient_evals": n_evals}

    m_u = score(case, s_u)
    m_u["arm"] = "U_uniform"
    res["arms"]["U_uniform"] = m_u
    maps_store["U_uniform"] = s_u
    print(f"[{shape}] uniform J {m_u['J']:.2f} IoU {m_u['IoU']:.4f} "
          f"under {m_u['part_under_melt_pct']:.2f}% "
          f"Eres {m_u['energy_gate']['rel_residual_at_index']*100:.2f}%", flush=True)

    # --- historical scan ---------------------------------------------------
    catalog = stored_mask_catalog(shape)
    res["n_stored_masks_scanned"] = len(catalog)
    hist = historical_scan(case, cfg, catalog)
    res["hist_scan"] = hist
    best_hist_name = best_arm_by_J(hist)
    res["best_hist_arm"] = best_hist_name
    res["arms"]["HIST_best"] = dict(hist[best_hist_name], arm="HIST_best",
                                    source_arm=best_hist_name)
    print(f"[{shape}] best historical of {len(hist)} arms: {best_hist_name} "
          f"J {hist[best_hist_name]['J']:.2f} IoU {hist[best_hist_name]['IoU']:.4f}",
          flush=True)

    # --- single pass solve, box [0, 1] -------------------------------------
    best_a1, s_a1, rows_a1 = solve_arm(case, ops, "A1", (0.0, 1.0), n_evals)
    m = score(case, s_a1)
    m["arm"] = "A1_cont"
    res["arms"]["A1_cont"] = m
    res["A1_rows"] = rows_a1
    maps_store["A1_cont"] = s_a1

    for name, sq in quantized_variants(s_a1, pm, 1.0).items():
        mq = score(case, sq)
        mq["arm"] = f"A1_{name}"
        mq.update({f"census_{k}": v for k, v in
                   pq.level_census(sq, pm, bpp=2 if name == "2bpp" else 4).items()})
        res["arms"][f"A1_{name}"] = mq
        maps_store[f"A1_{name}"] = sq
        print(f"[{shape}] A1_{name} J {mq['J']:.2f} IoU {mq['IoU']:.4f} "
              f"under {mq['part_under_melt_pct']:.2f}% "
              f"Eres {mq['energy_gate']['rel_residual_at_index']*100:.2f}%", flush=True)

    # --- double pass only for the L-class ----------------------------------
    under_deliverable = res["arms"]["A1_4bpp"]["part_under_melt_pct"]
    res["double_pass_triggered"] = needs_double_pass(under_deliverable)
    res["double_pass_trigger_under_pct"] = float(under_deliverable)
    if res["double_pass_triggered"]:
        best_a15, s_a15, rows_a15 = solve_arm(case, ops, "A15", (0.0, 1.5), n_evals)
        m15 = score(case, s_a15)
        m15["arm"] = "A15_cont"
        res["arms"]["A15_cont"] = m15
        res["A15_rows"] = rows_a15
        maps_store["A15_cont"] = s_a15
        for name, sq in quantized_variants(s_a15, pm, 1.5).items():
            mq = score(case, sq)
            mq["arm"] = f"A15_{name}"
            mq.update({f"census_{k}": v for k, v in
                       pq.level_census(sq, pm, bpp=2 if name == "2bpp" else 4).items()})
            res["arms"][f"A15_{name}"] = mq
            maps_store[f"A15_{name}"] = sq
            print(f"[{shape}] A15_{name} J {mq['J']:.2f} IoU {mq['IoU']:.4f} "
                  f"under {mq['part_under_melt_pct']:.2f}%", flush=True)

    # --- verdict -----------------------------------------------------------
    d = res["arms"]["A1_4bpp"]
    h = res["arms"]["HIST_best"]
    res["verdict"] = {
        "deliverable_arm": "A1_4bpp",
        "class": classify(d["IoU"], d["J"], h["IoU"], h["J"]),
        "beats_hist_on_J": bool(d["J"] < h["J"]),
        "beats_hist_on_IoU": bool(d["IoU"] > h["IoU"]),
        "dJ_rel": (h["J"] - d["J"]) / max(abs(h["J"]), 1e-30),
        "dIoU": d["IoU"] - h["IoU"],
        "reaches_nominal": bool(d["IoU"] >= SOLVED_IOU),
    }
    res["energy_gate_violations"] = [
        a for a, m in res["arms"].items() if not m["energy_gate"]["PASS"]]
    res["wall_s"] = time.perf_counter() - t_start

    np.savez_compressed(out / f"{shape}_maps.npz", part_mask=pm, x=case.x, y=case.y,
                        **maps_store)
    (out / f"{shape}.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] verdict {res['verdict']['class']}  dJ {res['verdict']['dJ_rel']*100:+.1f}%  "
          f"dIoU {res['verdict']['dIoU']:+.4f}  wall {res['wall_s']:.1f} s", flush=True)
    return res


if __name__ == "__main__":
    _shape = sys.argv[1]
    _outdir = sys.argv[2]
    _budget = float(sys.argv[3]) if len(sys.argv) > 3 else BUDGET_FORWARD_EQUIVALENTS
    main(_shape, _outdir, _budget)
