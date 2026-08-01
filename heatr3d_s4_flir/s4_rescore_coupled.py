#!/usr/bin/env python3
"""S4 RE-SCORE with the in-march EQS re-solve + sigma(T) coupling.

Answers ONE question from S4_GATE_REPORT.md sec 4.2 / sec 7 item 1-2: does
un-freezing Q_rf (and adding sigma(T) feedback) move the LATE-TIME TOPOLOGY of
the prediction toward what the two untuned FLIR runs actually show (case A: a
centre-hot plateau; case B: narrow diagonal bands)?

DISCIPLINE (read this before touching the coefficients)
-------------------------------------------------------
* The pre-registration (README.md) is READ-ONLY. Cases, registration operator,
  metrics and thresholds are unchanged and are evaluated through the same
  `s4_flir_lib` functions. The measured side is not re-decoded at all: it is
  loaded verbatim from the committed `results.json` + `fields.npz`.
* The coupling coefficients are NOT FITTED TO THE FLIR FRAMES. Fitting the
  coupling to the very data used to score it would be circular. They are a small
  EXPLORATORY sweep, fixed here before any score is read:
    a = 0.0    -> mechanism 1 alone (Q_rf un-frozen, no sigma feedback). Control.
    a = +0.002 -> the only nonzero value that exists anywhere in this repo:
                  configs/_archive_old/rfam_eqs_comsol_mimic.yaml l. 83
                  (sigma_temp_coeff_per_K: 0.002, sigma_ref_temp_c: 23.0).
                  Archived, never validated.
    a = +0.010 -> 1 %/K, the order-of-magnitude upper end for a thermally
                  activated (hopping) carbon-black/PA12 composite below melt.
    a = -0.010 -> the opposite sign, the PTC branch: thermal expansion breaking
                  the percolation network above the melt drops conductivity.
  Both signs are run because the physically plausible sign for a carbon-doped
  semicrystalline polymer is NOT settled below vs above melt. Reporting only the
  branch that helps would be the same circularity by another route.
* sigma_density_coeff is held at 0.0 and that is not a choice: the pre-registered
  S4 march runs `densify=False`, so rho_rel is constant at p.rho_rel for the
  whole march and the density term is PROVABLY INERT. Mechanism 3 of sec 4.2
  (densification coupling) is therefore NOT tested here.

DEVIATIONS from the original s4_run.py driver, all recorded in the report:
  R1  20 checkpoints instead of 40. Each chained segment now begins with a full
      EQS solve (30 s at this grid), and the segment length is set EQUAL to
      eqs_update_interval_s so that pre-loop solve IS the scheduled re-solve --
      no solve is wasted. The UNCOUPLED baseline is re-run at the same 20
      checkpoints so the comparison isolates coupling, not time resolution.
  R2  the single fitted scalar is REFITTED with coupling on, by the same
      pre-registered procedure (match the 60 s free-face mean rise), because the
      coupled model is a different model. Both powers are reported.
  R3  two TOPOLOGY diagnostics (centre-minus-ring, diagonal-minus-quadrant) are
      added. They are NOT pre-registered and NOT gating; they exist because the
      gate report's failure is stated in topological words and r alone cannot say
      which way the field moved.

Usage:
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      ./.venv312/bin/python -u heatr3d_s4_flir/s4_rescore_coupled.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from heatr3d import Grid, Params, run  # noqa: E402
import s4_flir_lib as L  # noqa: E402
import s4_run as S  # noqa: E402  (build_part, constants; nothing is mutated)

logger = logging.getLogger("s4rescore")

# exploratory sweep, fixed before any score is read (see the module docstring)
SIGMA_TEMP_COEFFS = (0.0, 0.002, 0.010, -0.010)
TARGET_INTERVAL_S = 30.0      # nominal EQS re-solve cadence (= checkpoint spacing)
SCORE_CASES = ("A", "B")      # the two untuned anchors; C stays the control
PLANE = "planeP"              # plane S is VOID (gate report sec 4.4)

# pre-registered thresholds (README sec 5)
THRESH = dict(M1_r=0.50, M1_strong=0.70, M2_margin=0.15, M3_chamfer_mm=6.0)


# --------------------------------------------------------------------------- #
# topology diagnostics (deviation R3: NOT pre-registered, NOT gating)
# --------------------------------------------------------------------------- #
def _unit_grid(ng: int) -> Tuple[np.ndarray, np.ndarray]:
    c = np.linspace(-0.5, 0.5, ng)
    return np.meshgrid(c, c, indexing="ij")


def centre_minus_ring(field: np.ndarray) -> float:
    """mean(central disc r<0.20) - mean(ring 0.33<r<0.50), on the normalized rise
    field. POSITIVE = a centre-hot plateau (case A's measured late topology);
    NEGATIVE = a cool-centre ring (what the frozen-Q_rf prediction gives)."""
    u, v = _unit_grid(field.shape[0])
    r = np.hypot(u, v)
    return float(field[r < 0.20].mean() - field[(r > 0.33) & (r < 0.50)].mean())


def diagonal_minus_quadrant(field: np.ndarray) -> float:
    """mean(|u|-|v| within 0.08 of 0, i.e. the two diagonals) - mean(the four
    quadrant interiors). POSITIVE = bright diagonal bands crossing at the centre
    (case B's measured late topology, the archive's literal 'X')."""
    u, v = _unit_grid(field.shape[0])
    diag = np.abs(np.abs(u) - np.abs(v)) < 0.08
    quad = (np.abs(np.abs(u) - np.abs(v)) > 0.20) & (np.hypot(u, v) < 0.45)
    return float(field[diag].mean() - field[quad].mean())


def topology(field: np.ndarray, amb: float) -> Dict[str, float]:
    f = L.normalize_rise(field, amb)
    return dict(centre_minus_ring=centre_minus_ring(f),
                diagonal_minus_quadrant=diagonal_minus_quadrant(f))


# --------------------------------------------------------------------------- #
# coupled march (chained, one global re-solve schedule via t_start_s)
# --------------------------------------------------------------------------- #
def march(grid: Grid, part: np.ndarray, p: Params, n_seg: int, seg_s: float,
          qrf: np.ndarray | None = None) -> Dict:
    """n_seg chained segments of seg_s seconds. qrf=None -> heatr3d solves the
    EQS itself (coupled path); qrf given -> the frozen-drive baseline."""
    ny = part.shape[1]
    y_free = int(np.nonzero(part.any(axis=(0, 2)))[0].max())
    T0 = None
    t_list, planeP = [], []
    part_mean, part_max, f_mean, f_max = [], [], [], []
    e_in = e_st = e_loss = 0.0
    clamp = cfl = False
    nsub = 1
    n_solve = n_skip = 0
    for k in range(n_seg):
        res = run(grid, part, p, max_time_s=seg_s, phi_target=1.5,
                  T0_override=T0, qrf_override=qrf, t_start_s=k * seg_s)
        T0 = res.T_final
        e_in += res.energy_in_j
        e_st += res.energy_stored_j
        e_loss += res.energy_loss_j
        clamp |= bool(res.clamp_bound)
        cfl |= bool(res.cfl_violated)
        nsub = max(nsub, int(res.n_substeps_used))
        n_solve += int(res.n_eqs_solves)
        n_skip += int(res.n_eqs_resolves_skipped)
        T = res.T_final
        t_list.append((k + 1) * seg_s)
        planeP.append(T[:, y_free, :].copy())
        part_mean.append(float(T[part].mean()))
        part_max.append(float(T[part].max()))
        fp, pm = T[:, y_free, :], part[:, y_free, :]
        f_mean.append(float(fp[pm].mean()))
        f_max.append(float(fp[pm].max()))
    resid = (e_in - e_st - e_loss) / max(e_in, 1e-30)
    return dict(t_s=np.array(t_list), planeP=np.array(planeP),
                part_mean=np.array(part_mean), part_max=np.array(part_max),
                face_mean=np.array(f_mean), face_max=np.array(f_max),
                energy_in_j=e_in, energy_stored_j=e_st, energy_loss_j=e_loss,
                energy_residual_frac=resid, clamp_bound=clamp, cfl_violated=cfl,
                n_substeps_used=nsub, n_eqs_solves=n_solve,
                n_eqs_resolves_skipped=n_skip, y_free=y_free)


def fit_power_coupled(grid: Grid, part: np.ndarray, p: Params, seg_s: float,
                      target_rise_c: float, t_fit_s: float) -> Tuple[float, float, List]:
    """The SAME pre-registered fit (one scalar, matched to the measured free-face
    mean rise at t_fit_s), driving power through Params.power_density_w_per_m3 so
    the coupled EQS path controls the absorbed power itself."""
    y_free = int(np.nonzero(part.any(axis=(0, 2)))[0].max())
    n_seg = max(1, int(round(t_fit_s / seg_s)))
    hist: List[Dict] = []

    def rise_at(dens: float) -> float:
        pp = dataclasses.replace(p, power_density_w_per_m3=dens)
        r = march(grid, part, pp, n_seg, t_fit_s / n_seg)
        val = float(r["face_mean"][-1]) - pp.preheat_c
        hist.append(dict(power_density=dens, rise_c=val))
        return val

    d0 = p.power_density_w_per_m3
    r0 = rise_at(d0)
    d1 = d0 * target_rise_c / max(r0, 1e-9)
    r1 = rise_at(d1)
    for _ in range(3):
        if abs(r1 - target_rise_c) / max(target_rise_c, 1e-9) < 0.005:
            break
        den = r1 - r0
        if abs(den) < 1e-12:
            break
        d2 = d1 + (target_rise_c - r1) * (d1 - d0) / den
        d0, r0, d1 = d1, r1, max(d2, 1e-9)
        r1 = rise_at(d1)
    return d1, r1, hist


# --------------------------------------------------------------------------- #
# scoring (identical operator to s4_run.py, planeP only)
# --------------------------------------------------------------------------- #
def register_and_score(pr: Dict, amb_p: float, meas: Dict, amb_m: float,
                       meas_C_095: np.ndarray, amb_C: float) -> Dict:
    theta_p = L.normalized_rise_curve(pr["t_s"], pr["face_mean"], amb_p)
    imgs = pr["planeP"]
    gmax = imgs.max(axis=(1, 2))
    rise = gmax - amb_p
    ref_i = int(np.argmax(rise >= 0.6 * rise.max()))
    roi = L.largest_component_roi(imgs[ref_i], amb_p, frac=0.5)
    rect = L.min_area_rect(roi)
    out: Dict = dict(fields={})
    for fr in S.MATCH_FRACS:
        key = f"{fr:.2f}"
        tt = L.time_at_fraction(pr["t_s"], theta_p, fr)
        j = int(np.argmin(np.abs(pr["t_s"] - tt)))
        pf = L.resample_unit_square(imgs[j], rect, ng=S.NG)
        mf = meas[key]
        a = L.normalize_rise(mf, amb_m)
        b = L.normalize_rise(pf, amb_p)
        out[key] = dict(
            r=L.pattern_correlation(a, b),
            chamfer_mm=L.chamfer_mm(a, b),
            argmax_offset_mm=L.argmax_offset_mm(a, b),
            cX_measured=L.corner_edge_contrast(mf),
            cX_predicted=L.corner_edge_contrast(pf),
            t_pred_s=float(pr["t_s"][j]),
            topology_pred=topology(pf, amb_p),
            topology_meas=topology(mf, amb_m),
        )
        out["fields"][key] = pf
    # M2 discrimination: same prediction vs the TUNED control's 95 % field
    r_tuned = L.pattern_correlation(
        L.normalize_rise(meas_C_095, amb_C),
        L.normalize_rise(out["fields"]["0.95"], amb_p))
    out["M2_r_vs_tuned"] = r_tuned
    out["M2_margin"] = out["0.95"]["r"] - r_tuned
    out["rect"] = dataclasses.asdict(rect)
    out["ref_t_s"] = float(pr["t_s"][ref_i])
    return out


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coeffs", type=float, nargs="*", default=list(SIGMA_TEMP_COEFFS))
    ap.add_argument("--cases", nargs="*", default=list(SCORE_CASES))
    ap.add_argument("--out", type=Path, default=HERE / "results_coupled.json")
    ap.add_argument("--fields-out", type=Path, default=HERE / "fields_coupled.npz")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    base = json.loads((HERE / "results.json").read_text())
    fields = np.load(HERE / "fields.npz")
    grid, part = S.build_part()
    qrf_ref, part_c, h = S.load_qrf()
    assert np.array_equal(part, part_c) and abs(h - grid.h) < 1e-12
    part_vol = float(int(part.sum()) * grid.dV)
    amb_C = base["cases"]["C"]["measured"]["e095"]["ambient_c"]
    meas_C_095 = fields["meas_C_0.95"]

    out: Dict = dict(
        meta=dict(
            purpose="S4 re-score with in-march EQS re-solve + sigma(T) coupling",
            pre_registration="README.md (unchanged, read-only)",
            measured_side="loaded verbatim from results.json + fields.npz "
                          "(no re-decode of any .seq)",
            coefficients_are_exploratory_not_fitted=True,
            sigma_temp_coeffs=list(args.coeffs),
            sigma_density_coeff=0.0,
            sigma_density_coeff_note="provably inert: the pre-registered march "
                                     "runs densify=False, so rho_rel is constant",
            target_eqs_interval_s=TARGET_INTERVAL_S,
            thresholds=THRESH,
            deviations=["R1 20 checkpoints (was 40), segment == re-solve interval; "
                        "uncoupled baseline re-run at 20 for an apples-to-apples "
                        "comparison",
                        "R2 the one fitted scalar is refitted with coupling on, by "
                        "the same pre-registered 60 s procedure",
                        "R3 two non-pre-registered, non-gating topology diagnostics"],
        ),
        cases={},
    )

    for ck in args.cases:
        m = base["cases"][ck]["measured"]["e095"]
        amb = m["ambient_c"]
        duration = m["duration_s"]
        n_seg = max(1, int(round(duration / TARGET_INTERVAL_S)))
        seg_s = duration / n_seg
        meas_fields = {f"{fr:.2f}": fields[f"meas_{ck}_{fr:.2f}"] for fr in S.MATCH_FRACS}
        p_case = dataclasses.replace(
            Params(), phase_update="enthalpy", t_pc_c=185.0, dt_s=S.DT_S,
            ambient_c=amb, preheat_c=amb,
            # T_ref = the run's own ambient, so the coupling factor is EXACTLY 1
            # at t = 0 and the coupled run starts from the uncoupled field.
            sigma_ref_temp_c=amb,
        )
        entry: Dict = dict(
            ambient_c=amb, duration_s=duration, n_seg=n_seg, seg_s=seg_s,
            published=base["cases"][ck]["scores"], variants={},
        )
        logger.info("case %s: %.1f s in %d segments of %.2f s", ck, duration, n_seg, seg_s)

        # ---- uncoupled baseline at the SAME checkpoint count (deviation R1) ---
        t0 = time.time()
        p_w = base["gates"][ck]["power_w"]
        pr = march(grid, part, p_case, n_seg, seg_s,
                   qrf=qrf_ref * (p_w / float(qrf_ref.sum() * grid.dV)))
        sc = register_and_score(pr, amb, meas_fields, amb, meas_C_095, amb_C)
        entry["variants"]["baseline_frozen"] = _pack(pr, sc, p_w, None, time.time() - t0)
        logger.info("  baseline (frozen Q_rf, published power %.2f W): "
                    "r60=%+.3f r95=%+.3f chamfer95=%.2f mm  [%.0f s]",
                    p_w, sc["0.60"]["r"], sc["0.95"]["r"], sc["0.95"]["chamfer_mm"],
                    time.time() - t0)

        for a in args.coeffs:
            t0 = time.time()
            p_c = dataclasses.replace(
                p_case, eqs_update_interval_s=seg_s, sigma_temp_coeff_per_K=float(a),
                sigma_density_coeff=0.0,
                power_density_w_per_m3=p_w / part_vol,     # start the fit at the
            )                                              # published power
            dens, r_fit, hist = fit_power_coupled(grid, part, p_c, seg_s,
                                                  m["rise_at_fit_c"], S.T_FIT_S)
            p_c = dataclasses.replace(p_c, power_density_w_per_m3=dens)
            pw_c = dens * part_vol
            logger.info("  a=%+.4f /K: fitted %.3f W (target rise %.2f C, got %.2f C)",
                        a, pw_c, m["rise_at_fit_c"], r_fit)
            pr = march(grid, part, p_c, n_seg, seg_s)
            sc = register_and_score(pr, amb, meas_fields, amb, meas_C_095, amb_C)
            entry["variants"][f"a{a:+.4f}"] = _pack(pr, sc, pw_c, hist, time.time() - t0)
            logger.info("  a=%+.4f /K: r60=%+.3f r95=%+.3f chamfer95=%.2f mm "
                        "cmr(pred/meas)=%+.3f/%+.3f dmq=%+.3f/%+.3f  "
                        "solves=%d skipped=%d  [%.0f s]",
                        a, sc["0.60"]["r"], sc["0.95"]["r"], sc["0.95"]["chamfer_mm"],
                        sc["0.95"]["topology_pred"]["centre_minus_ring"],
                        sc["0.95"]["topology_meas"]["centre_minus_ring"],
                        sc["0.95"]["topology_pred"]["diagonal_minus_quadrant"],
                        sc["0.95"]["topology_meas"]["diagonal_minus_quadrant"],
                        pr["n_eqs_solves"], pr["n_eqs_resolves_skipped"],
                        time.time() - t0)
            args.out.write_text(json.dumps(_strip(out | dict(cases=out["cases"] | {ck: entry})),
                                           indent=1, default=float))
        out["cases"][ck] = entry
        # save after EVERY case: these runs are hours long and were killed once
        np.savez_compressed(
            args.fields_out,
            **{f"{c2}_{vk}_{fr:.2f}": np.asarray(v["fields"][f"{fr:.2f}"])
               for c2, ce in out["cases"].items() for vk, v in ce["variants"].items()
               for fr in S.MATCH_FRACS})
        args.out.write_text(json.dumps(_strip(out), indent=1, default=float))
    logger.info("wrote %s", args.out)


def _pack(pr: Dict, sc: Dict, power_w: float, hist, wall_s: float) -> Dict:
    return dict(
        power_w=power_w, fit_hist=hist, wall_s=wall_s,
        gates=dict(energy_residual_frac=pr["energy_residual_frac"],
                   energy_gate_pass=bool(abs(pr["energy_residual_frac"]) <= 1e-2),
                   clamp_bound=pr["clamp_bound"], cfl_violated=pr["cfl_violated"],
                   n_substeps_used=pr["n_substeps_used"]),
        n_eqs_solves=pr["n_eqs_solves"], n_eqs_resolves_skipped=pr["n_eqs_resolves_skipped"],
        face_max_end_c=float(pr["face_max"][-1]), part_max_end_c=float(pr["part_max"][-1]),
        t_s=pr["t_s"].tolist(), face_mean=pr["face_mean"].tolist(),
        part_max=pr["part_max"].tolist(), face_max=pr["face_max"].tolist(),
        scores={k: v for k, v in sc.items() if k != "fields"},
        fields={k: np.asarray(v) for k, v in sc["fields"].items()},
    )


def _strip(o):
    """JSON view: drop the raw 64x64 fields (they go to fields_coupled.npz)."""
    if isinstance(o, dict):
        return {k: _strip(v) for k, v in o.items() if k != "fields"}
    if isinstance(o, (list, tuple)):
        return [_strip(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


if __name__ == "__main__":
    main()
