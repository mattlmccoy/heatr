"""Shape-fidelity SOLVE as a first-class FGM creation mode for HEATR 2-D.

FGM = functionally graded material. This entry point runs the production-recipe
gradient solve of the melt-region shape-fidelity objective

    J(s, t_stop) = sum over the domain of (phi(x, t_stop) - chi(x))^2

using ``fgm_solve_campaign/adjoint2d`` as a LIBRARY (imported, never copied),
and emits the result as a 4 bits per pixel printable map in the production
``fgm_generator.py`` npz format so the existing engine injection path
(``rfam_eqs_coupled._FgmFeedback.from_config``, rfam_eqs_coupled.py:366-380)
and the graphical user interface can consume it unchanged.

Production recipe (MULTISTART_REPORT.md and TOPOPT_REPORT.md verdicts, frozen
in FROZEN_CONVENTIONS_2D.md):
  * FILTERED full-depth SINGLE start (normalized-convolution Gaussian filter,
    adjoint2d/design_filter.py, via ``topopt.design_to_map`` at beta = 0);
  * filter radius 1.0 mm, a PHYSICAL length, converted per grid by
    ``topopt.sigma_cells_for``;
  * NO smoothed-Heaviside projection by default (beta = 0; at beta <= 0
    ``design_to_map`` is bit-identical to ``design_filter.apply_filter``);
  * warm start only where a strong historical mask exists
    (``fgm_solve_campaign/out_lib/<shape>.json`` HIST_best, injected FILTERED,
    the ``ms_solve.build_starts`` convention);
  * conductivity-only actuator channel (eps_covary = False); the permittivity
    channel stays behind ``eps_channel_model_only`` and defaults OFF because it
    has no deployment path (FROZEN_CONVENTIONS_2D.md section 7);
  * deliverable is the 4 bits per pixel quantized map via the production
    quantizer (``printability.quantize_in_part``), re-run through the real
    forward. Never the continuous map alone.

Grid qualifier, quoted wherever a number is: every fidelity number carries its
grid; a map solved at grid 120 is not established at any other grid unless the
grid hold-out gate has been run (FROZEN_CONVENTIONS_2D.md section 8).

Run:
  ./.venv312/bin/python scripts/solve_fgm.py \
      --config outputs_eqs/fgm_calibrated_control/configs/square_m0p0500.yaml \
      --output-dir outputs_eqs/runs/square/fgm_solve/<name> [--budget 40]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DIR = ROOT / "fgm_solve_campaign"


def _ensure_import_paths() -> None:
    """Make ``rfam_eqs_coupled`` (repo root) and ``adjoint2d`` importable."""
    for p in (str(ROOT), str(CAMPAIGN_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)

# ---------------------------------------------------------------------------
# fgm_solve config block (pure logic, unit tested in test_solve_fgm.py)
# ---------------------------------------------------------------------------

_ALLOWED_KEYS = {
    "budget_forward_equivalents",
    "filter_radius_mm",
    "warm_start",
    "eps_channel_model_only",
    "bpp",
}


@dataclass(frozen=True)
class FgmSolveConfig:
    """The ``fgm_solve`` block of a HEATR 2-D shape configuration.

    warm_start is one of:
      "auto"  use the best stored historical mask recorded by the library
              campaign when one exists, else start cold (logged loudly);
      "cold"  always the uniform full-depth start;
      a path  a stored dopant-map npz to inject (filtered) as the start.
    """

    budget_forward_equivalents: float = 40.0
    filter_radius_mm: float = 1.0
    warm_start: str = "auto"
    eps_channel_model_only: bool = False
    bpp: int = 4


def parse_fgm_solve_block(cfg: dict[str, Any],
                          budget_override: float | None = None) -> FgmSolveConfig:
    """Parse and validate the optional ``fgm_solve`` block of a config dict.

    Args:
        cfg: the full simulation config dict (the block may be absent).
        budget_override: command-line budget that wins over the block value.

    Returns:
        A frozen, validated FgmSolveConfig.

    Raises:
        ValueError: on a malformed block, unknown key, or invalid value.
    """
    raw = cfg.get("fgm_solve", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"fgm_solve must be a mapping, got {type(raw).__name__}")
    unknown = set(raw) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unknown fgm_solve key(s): {sorted(unknown)}; "
                         f"allowed: {sorted(_ALLOWED_KEYS)}")

    budget = float(raw.get("budget_forward_equivalents", 40.0))
    if budget_override is not None:
        budget = float(budget_override)
    if budget <= 0:
        raise ValueError(f"fgm_solve budget must be > 0, got {budget}")

    radius_mm = float(raw.get("filter_radius_mm", 1.0))
    if radius_mm <= 0:
        raise ValueError(f"fgm_solve filter_radius_mm must be > 0, got {radius_mm}")

    warm = str(raw.get("warm_start", "auto"))
    if not warm:
        raise ValueError("fgm_solve warm_start must be 'auto', 'cold', or a path")

    bpp = int(raw.get("bpp", 4))
    if bpp not in (2, 4):
        raise ValueError(f"fgm_solve bpp must be 2 or 4, got {bpp}")

    return FgmSolveConfig(
        budget_forward_equivalents=budget,
        filter_radius_mm=radius_mm,
        warm_start=warm,
        eps_channel_model_only=bool(raw.get("eps_channel_model_only", False)),
        bpp=bpp,
    )


# ---------------------------------------------------------------------------
# production map emitter
# ---------------------------------------------------------------------------

def emit_production_npz(sat, x, y, out_path: str | Path, bpp: int = 4,
                        dpi: int = 720, run_name: str = "solve") -> Path:
    """Write the solved map in the production ``fgm_generator.py`` npz format.

    Key set and dtypes reproduce the emitter at fgm_generator.py:679-696
    exactly (level_map, sat_map, x_mm, y_mm, width_mm, height_mm, bpp,
    n_levels, magnitude, baseline_saturation, dead_band, proxy_field, invert,
    dpi), so the engine injection path (rfam_eqs_coupled.py:366-380), the
    Results-tab re-simulate button, and fgm_to_rip all consume it unchanged.

    ``magnitude`` is 1.0 and ``baseline_saturation`` 0.5 because those are the
    values at which the injection loader applies NO rescale
    (rfam_eqs_coupled.py:383-388); the solve's map is already final.
    ``proxy_field`` is the literal string "solve": this map came from the
    adjoint gradient solve, not from any proxy-field inversion.
    """
    import numpy as np

    _ensure_import_paths()
    from adjoint2d import printability as pq

    sat = np.asarray(sat, dtype=np.float32)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx_m = float(x[1] - x[0])
    dy_m = float(y[1] - y[0])
    # The printer-resolution integer level map, same code path the production
    # quantizer module reproduces from fgm_generator.py:593-606.
    level_map_dpi = pq.printer_level_map(sat, dx_m=dx_m, dy_m=dy_m,
                                         dpi=int(dpi), bpp=int(bpp))
    x_mm = x * 1000.0
    y_mm = y * 1000.0
    width_mm = float(x_mm[-1] - x_mm[0])
    height_mm = float(y_mm[-1] - y_mm[0])

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        level_map=level_map_dpi,
        sat_map=sat,
        x_mm=x_mm,
        y_mm=y_mm,
        width_mm=np.array(width_mm, dtype=np.float64),
        height_mm=np.array(height_mm, dtype=np.float64),
        bpp=np.array(int(bpp), dtype=np.int32),
        n_levels=np.array(1 << int(bpp), dtype=np.int32),
        magnitude=np.array(1.0, dtype=np.float32),
        baseline_saturation=np.array(0.5, dtype=np.float32),
        dead_band=np.array(0.0, dtype=np.float32),
        proxy_field=np.array("solve"),
        invert=np.array(False),
        dpi=np.array(int(dpi), dtype=np.int32),
    )
    return out_path


def emit_map_pngs(npz_path: str | Path) -> dict[str, str]:
    """Emit the standard FGM map figure pair next to the production npz.

    Reproduces the fgm_generator.py PNG convention (fgm_generator.py:702-744),
    the same artifacts the other FGM (functionally graded material) creation
    modes ship with every map:
      * ``<stem>_preview.png``: white = max ink, black = no ink, flipped
        vertically so the physical top of the part appears at the image top
        (fields row 0 is the physical bottom; PNG row 0 is the image top);
      * ``<stem>_meteor_import.png``: the exact pixel inversion, black = max
        ink, the file Meteor RIP (raster image processor) imports directly.
    """
    import numpy as np
    from PIL import Image

    npz_path = Path(npz_path)
    with np.load(npz_path, allow_pickle=True) as d:
        level_map = d["level_map"]
        bpp = int(d["bpp"])
    max_val = (1 << bpp) - 1
    scale = 255.0 / max(max_val, 1)
    vis_preview = np.flipud(level_map.astype(np.float32) * scale).astype(np.uint8)
    png_path = npz_path.with_name(f"{npz_path.stem}_preview.png")
    Image.fromarray(vis_preview, "L").save(str(png_path))
    meteor_png_path = npz_path.with_name(f"{npz_path.stem}_meteor_import.png")
    Image.fromarray((255 - vis_preview), "L").save(str(meteor_png_path))
    return {"png_path": str(png_path), "meteor_png_path": str(meteor_png_path)}


# ---------------------------------------------------------------------------
# production verification pass (the full standard figure suite)
# ---------------------------------------------------------------------------

# The standard per-run file set a production rfam_eqs_coupled.py run emits;
# captured from a REAL v2.0.0 engine run of a solved 4 bits per pixel map
# (outputs_eqs/fgm_solve_showcase/triangle_A1_4bpp_stop270/). This is exactly
# what the Results tab renders.
PRODUCTION_SUITE_FILES = {
    "electric_fields.png", "thermal_fields_final.png", "rf_summary_v5.png",
    "paper_style_report.png", "validation_report.png", "time_series.png",
    "time_series.json", "fields.npz", "summary.json", "used_config.yaml",
    "density_evolution.gif", "electric_field_evolution.gif",
    "thermal_evolution.gif", "report_manifest.json",
}


def build_production_verify_config(cfg: dict[str, Any], npz_path: str | Path,
                                   t_stop_s: float) -> dict[str, Any]:
    """Build the engine config for the production verification run.

    Follows the showcase precedent
    (outputs_eqs/fgm_solve_showcase/triangle_solved_A1_4bpp_stop270.yaml): the
    shape's calibrated config unchanged, except
      * ``fgm_feedback`` injects the deliverable map through the direct hook
        ``sat_map_npz_direct`` (the loader reads the npz ``sat_map`` key at
        simulation resolution, rfam_eqs_coupled.py:390-418; ``sat_max`` 1.0
        keeps the printable clamp, ``iterate`` false keeps the map frozen);
      * ``thermal.n_steps`` is set to the solve's optimal stop
        (round(t_stop_s / dt_s); the showcase ran 540 = 270 s / 0.5 s);
      * the ``fgm_solve`` block is stripped (the engine does not read it).

    Args:
        cfg: the solve's own loaded config dict (not mutated).
        npz_path: path to the deliverable production map npz.
        t_stop_s: the solve's optimal stop time in seconds.

    Returns:
        A new config dict ready to be written as yaml for rfam_eqs_coupled.py.

    Raises:
        ValueError: if the config carries no positive thermal.dt_s.
    """
    import copy

    out = copy.deepcopy(cfg)
    out.pop("fgm_solve", None)
    dt_s = float(out.get("thermal", {}).get("dt_s", 0.0) or 0.0)
    if dt_s <= 0.0:
        raise ValueError("production verify needs a positive thermal.dt_s "
                         "in the config to convert the stop time to n_steps")
    out.setdefault("thermal", {})["n_steps"] = max(int(round(float(t_stop_s) / dt_s)), 1)
    out["fgm_feedback"] = {
        "enabled": True,
        "sat_map_npz_direct": str(npz_path),
        "sat_max": 1.0,
        "iterate": False,
    }
    return out


def run_production_verify(out: Path, cfg: dict[str, Any], npz_path: Path,
                          m_q: dict, case, chi, log) -> dict:
    """Run the REAL engine on the deliverable map and compare at the stop.

    Invokes rfam_eqs_coupled.py as a subprocess into ``<out>/production_verify/``
    so the run emits the entire standard per-run figure suite (the set the
    Results tab renders, PRODUCTION_SUITE_FILES), stamped with the engine
    version. Then compares the production run's final temperature field,
    scored with the SAME objective functions the solve used, against the
    solve's own deliverable numbers. Prior gates (adjoint2d gate L0) proved
    the two marches bit-identical, so the deltas here double as the
    end-to-end integration check; anything above 1 percent is flagged loudly.
    """
    import subprocess

    import numpy as np
    import yaml

    from adjoint2d import topopt_objective as tobj

    verify_dir = out / "production_verify"
    verify_dir.mkdir(parents=True, exist_ok=True)
    vcfg = build_production_verify_config(cfg, npz_path, float(m_q["t_stop_s"]))
    cfg_yaml = out / "production_verify_config.yaml"
    cfg_yaml.write_text(yaml.safe_dump(vcfg, sort_keys=True))
    log(f"production verify: engine run of the deliverable map, "
        f"n_steps {vcfg['thermal']['n_steps']} "
        f"(stop {float(m_q['t_stop_s']):.1f} s) -> {verify_dir.name}/")

    log_path = out / "production_verify.log"
    with log_path.open("w") as fh:
        proc = subprocess.run(
            [sys.executable, str(ROOT / "rfam_eqs_coupled.py"),
             "--config", str(cfg_yaml), "--output-dir", str(verify_dir)],
            cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise RuntimeError(
            f"production verify engine run FAILED (exit {proc.returncode}); "
            f"see {log_path}")

    missing = sorted(f for f in PRODUCTION_SUITE_FILES
                     if not (verify_dir / f).exists())
    if missing:
        raise RuntimeError(
            f"production verify run completed but the standard suite is "
            f"incomplete; missing: {missing} (see {log_path})")

    # Score the production run's final T with the solve's own objective
    # functions (fields.npz T is the engine state at the stop; gate L0 proved
    # the marches bit-identical, gate_l0.py field_max_abs_diff).
    with np.load(verify_dir / "fields.npz") as d:
        T_prod = np.asarray(d["T"], dtype=float)
    J_prod, _seed = tobj.J_and_seed(T_prod, case, chi)
    iou_prod = float(tobj.metrics(T_prod, case, chi)["IoU"])

    engine_version = "unknown"
    try:
        used = yaml.safe_load((verify_dir / "used_config.yaml").read_text())
        engine_version = str(used.get("engine_version", "unknown"))
    except Exception:  # noqa: BLE001 - version stamp is informational
        pass

    dJ = float(J_prod) - float(m_q["J"])
    dJ_rel = abs(dJ) / max(abs(float(m_q["J"])), 1e-30)
    dIoU = iou_prod - float(m_q["IoU"])
    agree = (dJ_rel <= 0.01) and (abs(dIoU) <= 0.01)
    if agree:
        log(f"production verify AGREES with the solve: "
            f"J {J_prod:.4f} vs {m_q['J']:.4f} (rel {dJ_rel:.2e}), "
            f"IoU {iou_prod:.4f} vs {m_q['IoU']:.4f} (d {dIoU:+.2e}), "
            f"engine v{engine_version}")
    else:
        log(f"WARNING: production verify DISAGREES with the solve beyond "
            f"1 percent: J {J_prod:.4f} vs {m_q['J']:.4f} "
            f"(rel {dJ_rel:.3e}), IoU {iou_prod:.4f} vs {m_q['IoU']:.4f} "
            f"(d {dIoU:+.3e}). Prior gates matched to the 1e-9 to 1e-4 "
            f"class; investigate before trusting either number.")

    return {
        "dir": str(verify_dir),
        "engine_version": engine_version,
        "n_steps": int(vcfg["thermal"]["n_steps"]),
        "t_stop_s": float(m_q["t_stop_s"]),
        "suite_files_present": sorted(PRODUCTION_SUITE_FILES),
        "J_production": float(J_prod),
        "J_solve": float(m_q["J"]),
        "dJ": dJ,
        "dJ_rel": dJ_rel,
        "IoU_production": iou_prod,
        "IoU_solve": float(m_q["IoU"]),
        "dIoU": dIoU,
        "agrees_within_1_percent": agree,
    }


# ---------------------------------------------------------------------------
# warm start resolution
# ---------------------------------------------------------------------------

def _resolve_warm_start(sc: FgmSolveConfig, case, cfg: dict, shape: str, log):
    """The production warm-start rule: warm only from a strong historical mask.

    Returns (v0, meta). The warm start injects the FILTERED historical mask by
    construction (v0 is a design variable; the filter is applied inside the
    solve chain), the ``ms_solve.build_starts`` convention.
    """
    import numpy as np
    from adjoint2d import multistart as ms
    from adjoint2d.verify_hist import load_stored_map

    pm = case.part_mask
    cold = np.ones(pm.shape)
    if sc.warm_start == "cold":
        return cold, {"start": "cold uniform full depth (requested)"}

    if sc.warm_start != "auto":
        p = Path(sc.warm_start)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            raise FileNotFoundError(f"fgm_solve warm_start map not found: {p}")
        s_h = load_stored_map(case, p, cfg)
        log(f"warm start from configured map {p.name}")
        return ms.start_from_map(s_h, pm, (0.0, 1.0)), {
            "start": "warm from configured path", "map_npz": str(p)}

    # auto: the best stored historical mask recorded by the library campaign
    lib_json = CAMPAIGN_DIR / "out_lib" / f"{shape}.json"
    if lib_json.exists():
        j = json.loads(lib_json.read_text())
        h = j.get("arms", {}).get("HIST_best")
        if h and h.get("map_npz") and Path(h["map_npz"]).exists():
            s_h = load_stored_map(case, Path(h["map_npz"]), cfg)
            if h.get("convention") == "outside1":
                s_h = np.where(pm, s_h, 1.0)
            log(f"warm start AUTO: library HIST_best "
                f"(IoU {h.get('IoU', float('nan')):.4f} in its own channel)")
            return ms.start_from_map(s_h, pm, (0.0, 1.0)), {
                "start": "warm from library HIST_best",
                "map_npz": h["map_npz"], "convention": h.get("convention"),
                "note": "historical arm was scored in the permittivity-"
                        "co-varying channel; used only as a start point"}
    log("warm start AUTO: no stored historical mask found in out_lib; "
        "starting COLD (stated loudly, not silently)")
    return cold, {"start": "cold (auto found no historical mask)"}


# ---------------------------------------------------------------------------
# the solve
# ---------------------------------------------------------------------------

def run_solve(config_path: str | Path, output_dir: str | Path,
              budget_override: float | None = None,
              skip_verify: bool = False,
              intake_v2_json: str | Path | None = None) -> dict:
    """Run the production-recipe shape-fidelity solve on one shape config."""
    _ensure_import_paths()
    import numpy as np
    from scipy.optimize import minimize

    from adjoint2d import adjoint, chi_area, control as ctl, energy_gate as eg
    from adjoint2d import forward as fwd, gradops
    from adjoint2d import library_solve as lib
    from adjoint2d import ms_solve as msv
    from adjoint2d import printability as pq
    from adjoint2d import topopt
    from adjoint2d import topopt_objective as tobj
    from adjoint2d.pins import build_case, load_cfg

    t_start = time.perf_counter()
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg_path = Path(config_path).resolve()
    cfg = load_cfg(cfg_path)
    sc = parse_fgm_solve_block(cfg, budget_override=budget_override)
    budget = sc.budget_forward_equivalents

    def log(msg: str) -> None:
        print(f"[solve_fgm] {msg}", flush=True)

    # Shape name lives at geometry.part.shape in the production config schema
    # (see outputs_eqs/fgm_calibrated_control/configs/*.yaml); multi-part
    # configs use geometry.parts, for which the first part names the run.
    geo = cfg.get("geometry", {})
    shape = str(geo.get("part", {}).get("shape")
                or (geo.get("parts") or [{}])[0].get("shape")
                or "unknown")
    case = build_case(cfg)
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    chi, chi_info = chi_area.chi_from_cfg(cfg, case.x, case.y)
    radius_m = sc.filter_radius_mm * 1e-3
    sigma_cells = topopt.sigma_cells_for(radius_m, case.dx)
    eps_covary = bool(sc.eps_channel_model_only)
    beta = 0.0  # production recipe: NO smoothed-Heaviside projection

    log(f"shape {shape}, grid {pm.shape[0]}, dx {case.dx*1e3:.4f} mm, "
        f"filter radius {sc.filter_radius_mm:.2f} mm = {sigma_cells:.3f} cells, "
        f"beta 0 (no projection), channel "
        f"{'PERMITTIVITY CO-VARYING (MODEL ONLY)' if eps_covary else 'conductivity only (deployable)'}")
    if eps_covary:
        log("WARNING: eps_channel_model_only is ON. This is a MODEL result "
            "with no deployment path (FROZEN_CONVENTIONS_2D.md section 7).")
    if budget < 20.0:
        log(f"NOTE: budget {budget:g} forward-equivalents is below the "
            f"campaign standard 40. This is a reduced-budget run (smoke "
            f"test class), NOT a quality solve.")

    def run_forward(s):
        return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                           shape_stop_patience=lib.PATIENCE,
                           eps_covary=eps_covary)

    def score(s):
        tr = run_forward(s)
        m = tobj.full_metrics(tr, case, chi)
        i = int(m["t_stop_index"])
        m["P_abs_W_per_m"] = tr.P_abs_B
        m["sat_mean_in_part"] = float(np.mean(s[pm]))
        m["sat_min_in_part"] = float(np.min(s[pm]))
        m["sat_max_in_part"] = float(np.max(s[pm]))
        m["energy_gate"] = eg.gate_from_trajectory(tr, i)
        m["J_raster_chi"] = tobj.optimal_stop(tr, case, pm.astype(float)).J
        m["eps_covary"] = eps_covary
        phi, _ = tobj.phi_field(tr.T_at_end(i), case)
        melted = (phi >= tobj.MELT_LEVEL)
        del tr
        return m, melted

    # --- cost model; the uniform forward doubles as the uniform reference ----
    s_u = np.ones(pm.shape)
    t_a = time.perf_counter()
    tr_u = run_forward(s_u)
    t_b = time.perf_counter()
    st_u = tobj.optimal_stop(tr_u, case, chi)
    _Ju, seed_u = tobj.J_and_seed(tr_u.T_at_end(st_u.index), case, chi)
    t_c = time.perf_counter()
    adjoint.gradient(case, s_u, tr_u, {st_u.index: seed_u}, grad_ops=ops,
                     eps_covary=eps_covary)
    t_d = time.perf_counter()
    m_u = tobj.full_metrics(tr_u, case, chi)
    m_u["P_abs_W_per_m"] = tr_u.P_abs_B
    m_u["energy_gate"] = eg.gate_from_trajectory(tr_u, int(m_u["t_stop_index"]))
    m_u["J_raster_chi"] = tobj.optimal_stop(tr_u, case, pm.astype(float)).J
    del tr_u
    ratio, ratio_info = msv.budget_ratio(shape, (t_d - t_c) / max(t_b - t_a, 1e-9), log)
    pool = max(int(ctl.max_gradient_evals(float(budget), ratio)), 1)
    log(f"pool {pool} gradient evaluations (budget {budget:g} forward-"
        f"equivalents, adjoint-to-forward ratio {ratio:.2f})")
    log(f"U_uniform J {m_u['J']:.2f} IoU {m_u['IoU']:.4f} "
        f"P {m_u['P_abs_W_per_m']:.1f} W/m")

    # --- warm start -----------------------------------------------------------
    v0_full, start_meta = _resolve_warm_start(sc, case, cfg, shape, log)

    # --- filtered beta = 0 solve, single start, L-BFGS-B -----------------------
    rows: list[dict] = []
    store: dict[int, np.ndarray] = {}
    idx = np.flatnonzero(pm.ravel())
    box = (0.0, 1.0)

    def unpack(vec):
        v = np.ones(pm.shape)
        v.ravel()[idx] = vec
        return v

    def fun(vec):
        if len(rows) >= pool:
            raise StopIteration
        v = unpack(vec)
        s = topopt.design_to_map(v, pm, dx=case.dx, radius_m=radius_m, beta=beta)
        tr = run_forward(s)
        st = tobj.optimal_stop(tr, case, chi)
        J, seed = tobj.J_and_seed(tr.T_at_end(st.index), case, chi)
        g_s = adjoint.gradient(case, s, tr, {st.index: seed}, grad_ops=ops,
                               eps_covary=eps_covary)
        g = topopt.design_vjp(g_s, v, pm, dx=case.dx, radius_m=radius_m,
                              beta=beta)
        iou = float(tobj.metrics(tr.T_at_end(st.index), case, chi)["IoU"])
        del tr
        row = {"eval_index": len(rows) + 1, "J": float(J),
               "IoU": iou, "t_stop_index": int(st.index),
               "t_stop_s": float(st.time_s),
               "t_stop_at_horizon": bool(st.at_horizon),
               "grad_norm": float(np.linalg.norm(g[pm]))}
        rows.append(row)
        store[row["eval_index"]] = v.copy()
        # Campaign convention (topopt_solve): the uniform cost-model run is
        # free; forward-equivalents count the gradient evaluations only.
        fe = ctl.forward_equivalents(len(rows), len(rows), ratio)
        print(f"SOLVE_PROGRESS eval={row['eval_index']} pool={pool} "
              f"fe_spent={fe:.1f} fe_budget={budget:g} J={J:.2f} IoU={iou:.4f}",
              flush=True)
        return float(J), g.ravel()[idx].astype(float)

    v0 = np.clip(np.asarray(v0_full, dtype=float).ravel()[idx], *box)
    stop_reason = "budget"
    try:
        minimize(fun, v0, jac=True, method="L-BFGS-B", bounds=[box] * len(idx),
                 options={"maxiter": 10_000, "maxfun": 10_000,
                          "ftol": 1e-16, "gtol": 1e-16})
        stop_reason = "L-BFGS-B converged"
    except StopIteration:
        pass
    if not rows:
        raise RuntimeError("the solve produced no evaluation")
    best = min(rows, key=lambda r: r["J"])
    v_best = store[best["eval_index"]]
    log(f"solve done: {len(rows)} evaluations ({stop_reason}), best J "
        f"{best['J']:.2f} at eval {best['eval_index']}")

    # --- deliverable: 4 bits per pixel through the production quantizer -------
    s_cont = topopt.design_to_map(v_best, pm, dx=case.dx, radius_m=radius_m,
                                  beta=beta)
    m_c, _melt_c = score(s_cont)
    s_q = pq.quantize_in_part(s_cont, pm, bpp=sc.bpp, sat_max=1.0)
    m_q, melted_q = score(s_q)
    m_q.update({f"census_{k}": v for k, v in
                pq.level_census(s_q, pm, bpp=sc.bpp).items()})
    log(f"SOLVE_{sc.bpp}bpp J {m_q['J']:.2f} IoU {m_q['IoU']:.4f} "
        f"grow {m_q['bed_melt_pct_of_part']:.2f}% "
        f"under {m_q['part_under_melt_pct']:.2f}% "
        f"stop {m_q['t_stop_s']:.1f} s"
        f"{' HORIZON' if m_q['t_stop_at_horizon'] else ''}")

    # --- outputs ---------------------------------------------------------------
    run_name = out.name
    npz_path = emit_production_npz(
        s_q, case.x, case.y, out / f"fgm_{run_name}_solve_{sc.bpp}bpp.npz",
        bpp=sc.bpp, run_name=run_name)
    log(f"production map npz -> {npz_path.name}")
    map_pngs = emit_map_pngs(npz_path)
    log(f"map figures -> {Path(map_pngs['png_path']).name} + "
        f"{Path(map_pngs['meteor_png_path']).name}")

    # Campaign convention: budget counts gradient evaluations only; the cost
    # model and the two deliverable scoring forwards are reported separately.
    fe_spent = float((1.0 + ratio) * len(rows))
    fe_overhead = float((1.0 + ratio) + 2.0)  # cost-model run + 2 scoring forwards
    n_grid = int(pm.shape[0])
    res = {
        "entry_point": "scripts/solve_fgm.py",
        "shape": shape,
        "config": str(cfg_path),
        "recipe": {
            "start": start_meta,
            "filter_radius_mm": sc.filter_radius_mm,
            "sigma_cells": float(sigma_cells),
            "beta": beta,
            "projection": "none (beta = 0; production recipe)",
            "channel": ("permittivity co-varying (MODEL ONLY, no deployment "
                        "path)" if eps_covary else
                        "conductivity only (deployable)"),
            "bpp": sc.bpp,
            "quantizer": "printability.quantize_in_part (production convention,"
                         " nominal 1.0 outside the part)",
        },
        "grid_qualifier": (
            f"All numbers in this file are at grid {n_grid}. A map solved at "
            f"one grid is not established at another; this entry point runs "
            f"NO grid hold-out (FROZEN_CONVENTIONS_2D.md section 8 Gate A)."),
        "budget_forward_equivalents": float(budget),
        "budget_note": (None if budget >= 40.0 else
                        "REDUCED BUDGET (below the campaign standard 40 "
                        "forward-equivalents): smoke-test class, not a "
                        "quality solve"),
        "cost": {"ratio": ratio, "ratio_info": ratio_info,
                 "pool_gradient_evals": pool,
                 "n_evals_used": len(rows),
                 "forward_equivalents_spent_evals": fe_spent,
                 "forward_equivalents_overhead": fe_overhead,
                 "stop_reason": stop_reason},
        "chi": chi_info,
        "arms": {"U_uniform": m_u, "SOLVE_cont": m_c,
                 f"SOLVE_{sc.bpp}bpp": m_q},
        "deliverable_arm": f"SOLVE_{sc.bpp}bpp",
        "vs_uniform": {"dJ_rel": (m_u["J"] - m_q["J"]) / max(abs(m_u["J"]), 1e-30),
                       "dIoU": m_q["IoU"] - m_u["IoU"]},
        "rows": rows,
        "map_npz": str(npz_path),
        "map_pngs": map_pngs,
        "production_verify": None,
        "wall_s": time.perf_counter() - t_start,
    }
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))

    _write_map_melt_png(out / "solve_map_melt.png", case, s_q, melted_q, m_q,
                        shape, n_grid, sc)
    np.savez_compressed(out / "solve_maps.npz", part_mask=pm.astype(np.uint8),
                        chi_area=chi.astype(np.float32),
                        SOLVE_cont=s_cont.astype(np.float32),
                        **{f"SOLVE_{sc.bpp}bpp": s_q.astype(np.float32)},
                        x=case.x, y=case.y)
    log(f"results.json + solve_map_melt.png + solve_maps.npz -> {out}")

    # --- production verification pass: the full standard figure suite --------
    if skip_verify:
        res["production_verify"] = {"skipped": True,
                                    "reason": "--skip-verify was passed"}
        log("production verify SKIPPED (--skip-verify): no standard figure "
            "suite for this run; re-run without the flag for the deliverable "
            "report set")
    else:
        res["production_verify"] = run_production_verify(
            out, cfg, npz_path, m_q, case, chi, log)

    res["wall_s"] = time.perf_counter() - t_start
    (out / "results.json").write_text(json.dumps(res, indent=2, default=float))

    # --- print-package manifest fragment (FROZEN schema 2.0.0, section 7b) --
    # The 2-D lane's contribution to the Studio print package: engine
    # version stamp, plan block, correction provenance with the explicit
    # three-state transfer record, the voltage power block, and the real
    # (or explicitly not-run) production_verify record.
    from scripts import package_fragment as pfrag
    frag_path = pfrag.emit_fragment(out, results=res, cfg=cfg,
                                    map_npz_path=npz_path, method="solve",
                                    intake_v2_json=intake_v2_json)
    log(f"print-package fragment (schema 2.0.0) -> {frag_path.name}")

    log(f"done in {res['wall_s']:.0f} s")
    return res


def _write_map_melt_png(path: Path, case, s_q, melted, m_q: dict, shape: str,
                        n_grid: int, sc: FgmSolveConfig) -> None:
    """The cheap two-panel check figure: delivered map and melted region."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    pm = case.part_mask
    ext = [case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), dpi=180)
    a = axes[0]
    shown = np.where(pm, s_q, np.nan)
    im = a.imshow(shown, origin="lower", extent=ext, cmap="viridis",
                  vmin=0.0, vmax=1.0, interpolation="nearest")
    a.contour(pm.astype(float), levels=[0.5], colors="w", linewidths=0.8,
              extent=ext, origin="lower")
    a.set_title(f"Delivered {sc.bpp} bpp dopant map (part only)")
    fig.colorbar(im, ax=a, shrink=0.85, label="saturation")
    b = axes[1]
    b.imshow(melted.astype(float), origin="lower", extent=ext, cmap="Reds",
             vmin=0, vmax=1, interpolation="nearest")
    b.contour(pm.astype(float), levels=[0.5], colors="k", linewidths=0.8,
              extent=ext, origin="lower")
    b.set_title(f"Melted region at stop {m_q['t_stop_s']:.0f} s "
                f"(IoU {m_q['IoU']:.4f} at grid {n_grid})")
    for ax in axes:
        ax.set_xlabel("x [mm]")
        ax.set_ylabel("y [mm]")
    fig.suptitle(f"{shape}: shape-fidelity SOLVE deliverable "
                 f"(grid {n_grid}; numbers carry this grid)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# command line
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help="shape configuration yaml (calibrated per-shape "
                         "config; may carry an fgm_solve block)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--budget", type=float, default=None,
                    help="forward-equivalents budget override "
                         "(campaign standard 40)")
    ap.add_argument("--skip-verify", action="store_true",
                    help="skip the production verification pass (the real-"
                         "engine run of the deliverable map that emits the "
                         "full standard figure suite into production_verify/ "
                         "and checks the solve numbers). Default is ON; use "
                         "this flag only for fast iterations.")
    ap.add_argument("--intake-v2-json", default=None,
                    help="path to the geometry intake actuator classifier "
                         "version 2 record for this shape, ONLY when the run "
                         "came through the intake path; the print-package "
                         "fragment then carries the classifier "
                         "recommendation. Default: the fragment records the "
                         "classifier as absent with the stated reason.")
    args = ap.parse_args(argv)
    run_solve(args.config, args.output_dir, budget_override=args.budget,
              skip_verify=args.skip_verify,
              intake_v2_json=args.intake_v2_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
