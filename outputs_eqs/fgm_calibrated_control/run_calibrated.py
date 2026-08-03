"""Runner: gain-calibrated one-shot FGM control on the trusted 2-D engine.

Protocol frozen in PREREGISTRATION.md. Configuration is byte-identical to
`outputs_eqs/geometry_dual_readstate/run_sweep.py` (voltage drive at the stored per-shape
v_cal, grid 120, as-sized, horizon 1500 steps, proportional inverse map at bpp 4,
baseline_saturation 0.5, dead_band 0.05), so the 4 stored arms at m in {0.30, 0.50, 0.70,
0.85} are exact cache hits and the new numbers are directly comparable to
`outputs_eqs/geometry_dual_readstate/`.

Fit metric  = heating-peak sigma_T (what the line search minimizes).
Hold-out    = melt-onset sigma_T at phi_bar = 0.90 (reported, never used to select).

Usage:
    ./.venv312/bin/python outputs_eqs/fgm_calibrated_control/run_calibrated.py verify
    ./.venv312/bin/python outputs_eqs/fgm_calibrated_control/run_calibrated.py run square circle ...
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from glob import glob
from pathlib import Path
from typing import Optional

import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "outputs_eqs/geometry_dual_readstate"))

import dual_readstate as dr        # noqa: E402
import fgm_generator              # noqa: E402
import gain_calibration as gc     # noqa: E402

DRS = REPO / "outputs_eqs/geometry_dual_readstate"
RUNS = HERE / "runs"
CFGS = HERE / "configs"
PY = sys.executable
SOLVER = str(REPO / "rfam_eqs_coupled.py")

HORIZON = 1500
DOMAIN = (0.05, 2.50)
MAX_NEW_EVALS = 4          # 4 cached + 4 new = 8 evaluations, the pre-registered budget
WARM_MAGS = (0.30, 0.50, 0.70, 0.85)

# v2 (unseeded). The domain is extended past the v1 cap of 2.50 because the
# measured triangle global minimum sits at m = 2.402, right at that edge.
V2_DOMAIN = (0.05, 6.00)
V2_N_COARSE = 7
V2_MAX_REFINE = 3


def tag_for(m: float) -> str:
    return f"m{m:.4f}".replace(".", "p")


def load_shape_record(shape: str) -> dict:
    return json.loads((DRS / f"{shape}.json").read_text())


def base_config(shape: str, rec: dict) -> dict:
    return yaml.safe_load((REPO / rec["base_config"]).read_text())


def build_map(shape: str, m: float, baseline_dir: Path) -> str:
    map_dir = RUNS / shape / f"map_{tag_for(m)}"
    map_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(glob(str(map_dir / "*.npz")))
    if existing:
        return existing[0]
    res = fgm_generator.generate_fgm(
        str(baseline_dir), bpp=4, proxy_field="T_phi90", invert=True,
        magnitude=m, baseline_saturation=0.5, dead_band=0.05,
        emit_formats=("npz",), output_dir=str(map_dir),
    )
    return res.get("npz_path") or sorted(glob(str(map_dir / "*.npz")))[0]


def solve(shape: str, m: float, rec: dict, out_dir: Path) -> float:
    """Run one forward solve for gain m. Returns wall seconds (0.0 if resumed)."""
    ts = out_dir / "time_series.json"
    if ts.exists():
        return 0.0
    cfg = base_config(shape, rec)
    cfg["electric"]["enforce_generator_power"] = False
    cfg["electric"]["voltage_v"] = float(rec["v_cal"])
    cfg["thermal"]["n_steps"] = HORIZON
    cfg["fgm_feedback"] = {
        "enabled": True,
        "saturation_map_npz": build_map(shape, m, DRS / "runs" / shape / "baseline"),
        "magnitude": 1.0, "baseline_saturation": 0.5,
        "invert": True, "iterate": False, "proxy_field": "T_phi90",
    }
    CFGS.mkdir(parents=True, exist_ok=True)
    cfg_path = CFGS / f"{shape}_{tag_for(m)}.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    r = subprocess.run([PY, SOLVER, "--config", str(cfg_path), "--output-dir", str(out_dir)],
                       cwd=str(REPO), capture_output=True, text=True)
    dt = time.time() - t0
    if r.returncode != 0 or not ts.exists():
        raise RuntimeError(f"solver failed {shape} m={m}:\n{r.stdout[-1500:]}\n{r.stderr[-1500:]}")
    return dt


def read_arm(run_dir: Path) -> dict:
    ts = dr.load_ts(run_dir)
    rs = dr.extract_read_states(ts)
    rec = dict(rs)
    rec["run_dir"] = str(run_dir)
    if rs["melt_reached"]:
        rec["residual_frac_melt"] = dr.residual_frac_at(ts, rs["melt_onset_idx"])
        rec["power_melt_W_per_m"] = dr.power_at(ts, rs["melt_onset_idx"])
    if rs["heating_peak_idx"] is not None:
        rec["residual_frac_peak"] = dr.residual_frac_at(ts, rs["heating_peak_idx"])
        rec["power_peak_W_per_m"] = dr.power_at(ts, rs["heating_peak_idx"])
    rec["frac_dT_clipped_final"] = ts["frac_cells_dT_clipped"][-1]
    return rec


# ---------------------------------------------------------------------------

def verify_cache(shape: str = "square", m: float = 0.85) -> dict:
    """Re-run one stored grid point and compare, so cache reuse is evidence-backed."""
    rec = load_shape_record(shape)
    out = RUNS / shape / f"verify_{tag_for(m)}"
    dt = solve(shape, m, rec, out)
    fresh = read_arm(out)
    stored = next(a for a in rec["fgm_arms"] if abs(a["m"] - m) < 1e-9)
    res = {
        "shape": shape, "m": m, "wall_s": dt,
        "stored_heating_peak": stored["heating_peak_sigma_T"],
        "fresh_heating_peak": fresh["heating_peak_sigma_T"],
        "stored_melt_onset": stored["melt_onset_sigma_T"],
        "fresh_melt_onset": fresh["melt_onset_sigma_T"],
    }
    res["abs_diff_heating_peak"] = abs(res["fresh_heating_peak"] - res["stored_heating_peak"])
    res["abs_diff_melt_onset"] = abs(res["fresh_melt_onset"] - res["stored_melt_onset"])
    (HERE / "cache_verification.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


def process_shape(shape: str) -> dict:
    out_json = HERE / f"{shape}.json"
    if out_json.exists() and json.loads(out_json.read_text()).get("complete"):
        print(f"[skip] {shape} complete", flush=True)
        return json.loads(out_json.read_text())

    print(f"\n=== {shape} ===", flush=True)
    rec = load_shape_record(shape)
    arms: dict[float, dict] = {}
    timings: dict[str, float] = {}

    # warm start from the stored 4-point grid (exact cache hits, zero new solves)
    for a in rec["fgm_arms"]:
        arms[float(a["m"])] = a
    warm = {float(m): arms[m]["heating_peak_sigma_T"] for m in sorted(arms)}

    def objective(m: float) -> Optional[float]:
        out = RUNS / shape / f"fgm_{tag_for(m)}"
        dt = solve(shape, m, rec, out)
        timings[tag_for(m)] = dt
        arm = read_arm(out)
        arm["m"] = m
        arms[m] = arm
        hp = arm["heating_peak_sigma_T"]
        print(f"  m={m:.4f}  heating_peak={hp}  melt_onset={arm['melt_onset_sigma_T']}"
              f"  melt={arm['melt_reached']}  ({dt:.0f}s)", flush=True)
        return hp

    search = gc.calibrate_gain(objective, warm_start=warm,
                               domain=DOMAIN, max_new_evals=MAX_NEW_EVALS)

    cands = [gc.GainCandidate(gain=m,
                              fit_score=arms[m]["heating_peak_sigma_T"],
                              holdout_score=arms[m]["melt_onset_sigma_T"],
                              melt_reached=bool(arms[m]["melt_reached"]))
             for m in sorted(arms)]
    sel = gc.select_on_holdout(cands)

    out = {
        "shape": shape,
        "base_config": rec["base_config"],
        "v_cal": rec["v_cal"],
        "horizon_steps": HORIZON,
        "domain": list(DOMAIN),
        "fit_metric": "heating_peak_sigma_T",
        "holdout_metric": "melt_onset_sigma_T",
        "baseline": rec["baseline"],
        "best_of_four_in_sample": rec.get("best_fgm"),
        "warm_start_gains": sorted(warm),
        "new_gains": list(search.new_gains),
        "n_new_solves": search.n_new_evals,
        "n_total_evals": search.n_total_evals,
        "stop_reason": search.stop_reason,
        "timings_s": timings,
        "arms": {f"{m:.4f}": arms[m] for m in sorted(arms)},
        "selected_gain": sel.selected.gain if sel.selected else None,
        "selected_fit_heating_peak": sel.selected.fit_score if sel.selected else None,
        "selected_holdout_melt_onset": sel.reported_holdout,
        "unconstrained_gain": sel.unconstrained.gain if sel.unconstrained else None,
        "n_infeasible": sel.n_infeasible,
        "infeasible_gains": list(sel.infeasible_gains),
        "status": sel.status,
        "complete": True,
    }
    out_json.write_text(json.dumps(out, indent=2))
    print(f"  [done] {shape} selected m={out['selected_gain']} "
          f"holdout={out['selected_holdout_melt_onset']} status={sel.status}", flush=True)
    return out


def process_shape_v2(shape: str) -> dict:
    """v2: UNSEEDED coarse scan across the full domain, then refine.

    v1 warm-started from the stored grid {0.30..0.85}. The triangle fit metric is
    bimodal (local maximum m = 0.638), so that warm start sat entirely on one side
    of a local maximum and the search walked to the domain floor. v2 takes no warm
    start. Everything else (config pins, fit metric, hold-out, feasibility rule)
    is unchanged from PREREGISTRATION.md.
    """
    out_json = HERE / f"{shape}.v2.json"
    if out_json.exists() and json.loads(out_json.read_text()).get("complete"):
        print(f"[skip] {shape} v2 complete", flush=True)
        return json.loads(out_json.read_text())

    print(f"\n=== {shape} (v2 unseeded) ===", flush=True)
    rec = load_shape_record(shape)
    arms: dict[float, dict] = {}
    timings: dict[str, float] = {}

    def objective(m: float) -> Optional[float]:
        out = RUNS / shape / f"fgm_{tag_for(m)}"
        dt = solve(shape, m, rec, out)
        timings[tag_for(m)] = dt
        arm = read_arm(out)
        arm["m"] = m
        arms[m] = arm
        hp = arm["heating_peak_sigma_T"]
        print(f"  m={m:.4f}  heating_peak={hp}  melt_onset={arm['melt_onset_sigma_T']}"
              f"  melt={arm['melt_reached']}  ({dt:.0f}s{' cached' if dt == 0 else ''})",
              flush=True)
        return hp

    search = gc.calibrate_gain_unseeded(objective, domain=V2_DOMAIN,
                                        n_coarse=V2_N_COARSE, max_refine=V2_MAX_REFINE)
    cands = [gc.GainCandidate(gain=m,
                              fit_score=arms[m]["heating_peak_sigma_T"],
                              holdout_score=arms[m]["melt_onset_sigma_T"],
                              melt_reached=bool(arms[m]["melt_reached"]))
             for m in sorted(arms)]
    sel = gc.select_on_holdout(cands)

    n_cached = sum(1 for t in timings.values() if t == 0.0)
    out = {
        "shape": shape, "variant": "v2_unseeded",
        "base_config": rec["base_config"], "v_cal": rec["v_cal"],
        "horizon_steps": HORIZON, "domain": list(V2_DOMAIN),
        "n_coarse": V2_N_COARSE, "max_refine": V2_MAX_REFINE,
        "fit_metric": "heating_peak_sigma_T",
        "holdout_metric": "melt_onset_sigma_T",
        "baseline": rec["baseline"],
        "best_of_four_in_sample": rec.get("best_fgm"),
        "evaluated_gains": sorted(arms),
        "n_evaluations": len(arms),
        "n_new_solves": len(arms) - n_cached,
        "n_cache_hits": n_cached,
        "stop_reason": search.stop_reason,
        "timings_s": timings,
        "arms": {f"{m:.4f}": arms[m] for m in sorted(arms)},
        "selected_gain": sel.selected.gain if sel.selected else None,
        "selected_fit_heating_peak": sel.selected.fit_score if sel.selected else None,
        "selected_holdout_melt_onset": sel.reported_holdout,
        "unconstrained_gain": sel.unconstrained.gain if sel.unconstrained else None,
        "n_infeasible": sel.n_infeasible,
        "infeasible_gains": list(sel.infeasible_gains),
        "status": sel.status,
        "complete": True,
    }
    out_json.write_text(json.dumps(out, indent=2))
    print(f"  [done v2] {shape} selected m={out['selected_gain']} "
          f"holdout={out['selected_holdout_melt_onset']} status={sel.status} "
          f"new_solves={out['n_new_solves']}", flush=True)
    return out


def main() -> None:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return
    if args[0] == "verify":
        verify_cache(*(args[1:2] or ["square"]))
        return
    if args[0] == "run2":
        RUNS.mkdir(parents=True, exist_ok=True)
        for shape in args[1:]:
            try:
                process_shape_v2(shape)
            except Exception as e:  # noqa: BLE001 - keep the campaign resumable
                print(f"[ERROR] {shape}: {e}", flush=True)
                (HERE / f"{shape}.v2.ERROR.txt").write_text(str(e))
        return
    RUNS.mkdir(parents=True, exist_ok=True)
    for shape in args[1:] if args[0] == "run" else args:
        try:
            process_shape(shape)
        except Exception as e:  # noqa: BLE001 - keep the campaign resumable
            print(f"[ERROR] {shape}: {e}", flush=True)
            (HERE / f"{shape}.ERROR.txt").write_text(str(e))


if __name__ == "__main__":
    main()
