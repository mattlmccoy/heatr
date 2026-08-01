#!/usr/bin/env python3
"""LEVEL 2 of the continuous-rotation campaign: verification on the REAL engine.

Every arm below is a run of the production two-dimensional engine
(`rfam_eqs_coupled.run_sim`) through its own turntable machinery
(`rfam_eqs_coupled.py:2795-2837`), with the dopant map CO-ROTATED at every
rotation event by the run-script glue in `scripts.analysis.turntable_glue`. No
engine file is edited; the glue is monkeypatches held for the duration of one
call and every switch defaults to the engine's shipped behaviour.

Objective and stop convention, carried on every number:

    J_phi(t) = sum over the WHOLE domain of (phi(x, t) - chi_part(x; theta(t)))^2

with chi_part the part mask the engine has rasterized at the orientation in
force at time t, so the nominal target co-rotates with the part. Every metric
is read at that arm's OWN J-stop, argmin of J_phi over its own trajectory on a
1500-step horizon (dt 0.5 s, 750 s). A minimum on the last stored step is
flagged HORIZON and makes that J a BOUND. Melted region phi >= 0.5. GRID 120.

Arms:
  S_uniform    static, no dopant map
  S_map0       static at 0 degrees, the best stored zero-degree solved map
  S_joint      static at the joint campaign's winning angle, its winning map
  R_uniform    rotating, no dopant map
  R_map0       rotating, the zero-degree map co-rotated
  R_joint      rotating, the joint winner map rotated back into the part frame
  R_avg        rotating, the LEVEL-1 averaged-kernel map. THE test.
  R_avg_4bpp   the same at 4 bits per pixel, the printable version
  R_avg_norot  R_avg with the co-rotation switched OFF, which is the size of
               the bug that motivated the glue
  R_eps_fix    R_uniform with the permittivity field also re-rasterized at
               rotation events, which is a SECOND engine gap this campaign found
  R_null90     rotating in 90-degree steps at the same event rate. On a
               four-fold symmetric part the mask is invariant and the remap
               lands exactly on grid points, so any difference from the static
               arm is PURE numerical remap error and nothing else.

Run:
  ./.venv312/bin/python scripts/analysis/run_rot_verify.py <shape> [period_s ...]
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import library_solve as lib                      # noqa: E402
from adjoint2d import printability as pq                        # noqa: E402
from adjoint2d.pins import load_cfg                             # noqa: E402
from scripts.analysis.orientation_map_rotation import rotate_sat_map   # noqa: E402
from scripts.analysis.turntable_glue import run_rotating        # noqa: E402

N_STEPS = 1500
DT_S = 0.5
STEP_DEG = 15.0
PRIMARY_PERIOD_S = 24.0        # the fastest turn the 0.5 s step can resolve at
                               # 15-degree increments (one event every 2 steps)
OUT_ROOT = REPO / "fgm_solve_campaign/out_rot"
OUT_LIB = REPO / "fgm_solve_campaign/out_lib"
OUT_MS = REPO / "fgm_solve_campaign/out_ms"
OUT_JOINT = REPO / "fgm_solve_campaign/out_joint"

JOINT_BEST_ANGLE = {"T_shape": 90.0, "L_shape": 135.0, "cross": 45.0, "star": 18.0}


# ---------------------------------------------------------------------------
# map sources
# ---------------------------------------------------------------------------

def stored_zero_map(shape: str) -> tuple[np.ndarray, dict] | tuple[None, None]:
    best = None
    for arm, npz, key, jp in (("A1_cont", OUT_LIB / f"{shape}_maps.npz", "A1_cont",
                               OUT_LIB / f"{shape}.json"),
                              ("MS_cont", OUT_MS / f"{shape}_maps.npz", "MS_cont",
                               OUT_MS / f"{shape}.json")):
        if not (npz.exists() and jp.exists()):
            continue
        arms = json.loads(jp.read_text()).get("arms", {})
        if arm not in arms:
            continue
        J = float(arms[arm]["J"])
        if best is None or J < best[0]:
            best = (J, arm, npz, key)
    if best is None:
        return None, None
    J, arm, npz, key = best
    return np.clip(np.asarray(np.load(npz)[key], dtype=float), 0.0, 1.0), {
        "arm": arm, "npz": str(npz), "key": key, "prototype_static_J": J}


def joint_winner_map(shape: str) -> tuple[np.ndarray, float, dict] | tuple[None, None, None]:
    """The joint campaign's winning map, in its own LAB frame, and its angle."""
    d = OUT_JOINT / f"{shape}_sigma_refine"
    f = d / "results_refine.json"
    if not f.exists():
        return None, None, None
    r = json.loads(f.read_text())
    ang = float(r["refined_best_angle_deg"])
    tag = f"ang{ang:07.2f}".replace(".", "p")
    npz = d / "fields" / f"{tag}.npz"
    if not npz.exists():
        return None, None, None
    m = np.clip(np.asarray(np.load(npz)["sat_cont"], dtype=float), 0.0, 1.0)
    return m, ang, {"npz": str(npz), "angle_deg": ang,
                    "prototype_joint_J": float(r["J_at_refined_best"]),
                    "prototype_joint_IoU": float(r["IoU_at_refined_best"])}


def rotavg_maps(shape: str) -> tuple[np.ndarray, np.ndarray, dict] | tuple[None, None, None]:
    npz = OUT_ROOT / f"{shape}_rotavg_maps.npz"
    js = OUT_ROOT / f"{shape}_rotavg.json"
    if not (npz.exists() and js.exists()):
        return None, None, None
    d = np.load(npz)
    r = json.loads(js.read_text())
    meta = {"npz": str(npz),
            "averaged_kernel_J_cont": float(r["arms"]["AVG_cont"]["J"]),
            "averaged_kernel_J_4bpp": float(r["arms"]["AVG_4bpp"]["J"]),
            "averaged_kernel_IoU_cont": float(r["arms"]["AVG_cont"]["IoU"]),
            "n_angles": int(r["n_angles"])}
    return (np.asarray(d["sat_cont"], dtype=float),
            np.asarray(d["sat_4bpp"], dtype=float), meta)


# ---------------------------------------------------------------------------
# config builders
# ---------------------------------------------------------------------------

def base_cfg(shape: str, rotation_deg: float = 0.0) -> dict:
    cfg = load_cfg(lib.shape_config(shape))
    cfg = copy.deepcopy(cfg)
    cfg["thermal"]["n_steps"] = N_STEPS
    cfg["geometry"]["part"]["rotation_deg"] = float(rotation_deg)
    cfg.pop("fgm_feedback", None)
    cfg.pop("turntable", None)
    return cfg


def with_turntable(cfg: dict, period_s: float, step_deg: float = STEP_DEG) -> dict:
    cfg = copy.deepcopy(cfg)
    events_per_turn = 360.0 / float(step_deg)
    interval_s = float(period_s) / events_per_turn
    interval_steps = max(1, round(interval_s / DT_S))
    n_rot = int(np.ceil(N_STEPS / interval_steps)) + 2
    cfg["turntable"] = {"enabled": True, "rotation_deg": float(step_deg),
                        "total_rotations": int(n_rot),
                        "rotation_interval_s": float(interval_steps * DT_S)}
    return cfg


def realized_period_s(period_s: float, step_deg: float = STEP_DEG) -> float:
    interval_steps = max(1, round((float(period_s) * float(step_deg) / 360.0) / DT_S))
    return float(interval_steps * DT_S * 360.0 / float(step_deg))


# ---------------------------------------------------------------------------
# one arm
# ---------------------------------------------------------------------------

def run_arm(name: str, cfg: dict, sat, corotate: bool, corotate_eps: bool,
            extra: dict, log) -> dict:
    t0 = time.perf_counter()
    res = run_rotating(cfg, sat_map=sat, corotate=corotate,
                       corotate_eps=corotate_eps, record=True, quiet=True)
    m = res.at_stop()
    m["arm"] = name
    m["n_outer"] = len(res.steps)
    m["n_rotation_events"] = len(res.rotation_events)
    m["total_rotation_deg"] = (float(res.cumulative_angles_deg[-1])
                               if res.cumulative_angles_deg else 0.0)
    m["J_at_end"] = float(res.steps[-1]["J"])
    m["mean_rho_rel_part_at_end"] = float(res.steps[-1]["mean_rho_rel_part"])
    m["over_ceiling_250c"] = bool(m["max_T_part_c"] > 250.0)
    m["wall_s"] = time.perf_counter() - t0
    m.update(extra)
    log(f"  {name:14s} J {m['J']:8.2f}  IoU {m['IoU']:.4f}  "
        f"grow {m['bed_melt_pct_of_part']:6.2f}%  under {m['part_under_melt_pct']:6.2f}%  "
        f"rho {m['mean_rho_rel_part']:.4f}  stop {m['t_stop_s']:6.1f} s"
        f"{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
        f"maxT {m['max_T_part_c']:6.1f} C{'  CEILING' if m['over_ceiling_250c'] else ''}  "
        f"Egate {'PASS' if m['energy_gate_PASS'] else 'FAIL'} "
        f"({100 * m['energy_residual_rel']:.2f}%)  "
        f"evts {m['n_rotation_events']:4d}  [{m['wall_s']:.0f} s]")
    # Fields are returned AT THE ARM'S OWN J-STOP, which is where every metric
    # above is read; the end-of-horizon field is kept separately and is only a
    # diagnostic.
    return {"metrics": m, "J_curve": res.J_curve.astype(np.float32),
            "phi_final": (res.phi_at_stop if res.phi_at_stop is not None
                          else res.final_phi).astype(np.float32),
            "part_mask_final": (res.mask_at_stop if res.mask_at_stop is not None
                                else res.final_part_mask),
            "phi_end_of_horizon": res.final_phi.astype(np.float32),
            "sat_final": (None if res.final_sat_map is None
                          else res.final_sat_map.astype(np.float32))}


def main(shape: str, periods: list[float] | None = None,
         with_controls: bool = False) -> dict:
    t0 = time.perf_counter()
    periods = periods or [PRIMARY_PERIOD_S]

    def log(msg):
        print(f"[{shape}/rotver] {msg}", flush=True)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / "verify_fields").mkdir(parents=True, exist_ok=True)

    sat0, meta0 = stored_zero_map(shape)
    sat_j, ang_j, meta_j = joint_winner_map(shape)
    sat_a, sat_a4, meta_a = rotavg_maps(shape)
    if sat_a is None:
        raise FileNotFoundError(
            f"Level 1 has not been run for {shape}: "
            f"{OUT_ROOT / (shape + '_rotavg_maps.npz')} missing")
    log(f"maps: zero-degree {meta0 is not None}, joint {meta_j is not None} "
        f"(angle {ang_j}), averaged-kernel yes ({meta_a['n_angles']} angles)")

    rows: list[dict] = []
    fields: dict[str, np.ndarray] = {}

    def record(tag, out):
        rows.append(out["metrics"])
        fields[f"phi_{tag}"] = out["phi_final"]
        fields[f"mask_{tag}"] = out["part_mask_final"]
        if out["sat_final"] is not None:
            fields[f"sat_{tag}"] = out["sat_final"]

    # -- static references, same engine, same conventions --------------------
    log("static references (turntable off)")
    record("S_uniform", run_arm("S_uniform", base_cfg(shape, 0.0), None, False, False,
                                {"rotating": False, "rotation_deg": 0.0}, log))
    if sat0 is not None:
        record("S_map0", run_arm("S_map0", base_cfg(shape, 0.0), sat0, False, False,
                                 {"rotating": False, "rotation_deg": 0.0,
                                  "map_source": meta0}, log))
    if sat_j is not None:
        record("S_joint", run_arm("S_joint", base_cfg(shape, ang_j), sat_j, False, False,
                                  {"rotating": False, "rotation_deg": ang_j,
                                   "map_source": meta_j}, log))

    # -- rotating arms -------------------------------------------------------
    for P in periods:
        Pr = realized_period_s(P)
        cfgP = with_turntable(base_cfg(shape, 0.0), P)
        tagP = f"P{int(round(Pr))}"
        log(f"rotating, requested period {P:.1f} s, realized {Pr:.1f} s, "
            f"{STEP_DEG:.0f}-degree steps every "
            f"{cfgP['turntable']['rotation_interval_s']:.1f} s")
        ex = {"rotating": True, "period_s": Pr, "requested_period_s": float(P),
              "step_deg": STEP_DEG}
        record(f"R_uniform_{tagP}",
               run_arm(f"R_uniform_{tagP}", cfgP, None, True, False, dict(ex), log))
        record(f"R_avg_{tagP}",
               run_arm(f"R_avg_{tagP}", cfgP, sat_a, True, False,
                       dict(ex, map_source=meta_a), log))
        if P == periods[0]:
            if sat0 is not None:
                record(f"R_map0_{tagP}",
                       run_arm(f"R_map0_{tagP}", cfgP, sat0, True, False,
                               dict(ex, map_source=meta0), log))
            if sat_j is not None:
                # the joint map lives in the LAB frame at its winning angle;
                # rotate it BACK into the part frame so the turntable carries it
                sat_j_part = rotate_sat_map(sat_j, -float(ang_j), outside=1.0)
                record(f"R_joint_{tagP}",
                       run_arm(f"R_joint_{tagP}", cfgP, sat_j_part, True, False,
                               dict(ex, map_source=meta_j,
                                    note="joint map rotated back into the part frame"), log))
            record(f"R_avg4bpp_{tagP}",
                   run_arm(f"R_avg4bpp_{tagP}", cfgP, sat_a4, True, False,
                           dict(ex, map_source=meta_a, printable="4 bits per pixel"), log))

    # -- controls ------------------------------------------------------------
    if with_controls:
        P = periods[0]
        Pr = realized_period_s(P)
        tagP = f"P{int(round(Pr))}"
        cfgP = with_turntable(base_cfg(shape, 0.0), P)
        log("controls")
        record("C_avg_norotate",
               run_arm("C_avg_norotate", cfgP, sat_a, False, False,
                       {"rotating": True, "period_s": Pr,
                        "note": "co-rotation OFF: the engine as shipped"}, log))
        record("C_uniform_epsfix",
               run_arm("C_uniform_epsfix", cfgP, None, True, True,
                       {"rotating": True, "period_s": Pr,
                        "note": "permittivity field also re-rasterized at events"}, log))
        # numerical remap control: 90-degree steps at the same event rate
        cfg90 = copy.deepcopy(cfgP)
        cfg90["turntable"]["rotation_deg"] = 90.0
        record("C_null90",
               run_arm("C_null90", cfg90, None, True, False,
                       {"rotating": True, "step_deg": 90.0,
                        "note": "90-degree steps: exact pixel permutation, so any "
                                "gap to the static arm on a four-fold symmetric "
                                "part is pure remap numerical error"}, log))

    np.savez_compressed(OUT_ROOT / "verify_fields" / f"{shape}_fields.npz", **fields)
    out = {"shape": shape, "engine": "rfam_eqs_coupled.run_sim (production 2-D)",
           "glue": "scripts.analysis.turntable_glue (monkeypatch, no engine edit)",
           "n_steps": N_STEPS, "dt_s": DT_S, "grid": 120,
           "periods_requested_s": [float(p) for p in periods],
           "periods_realized_s": [realized_period_s(p) for p in periods],
           "joint_best_angle_deg": ang_j,
           "stop_convention": ("t_stop = argmin of J_phi over the arm's own "
                               "trajectory; 1500-step horizon (750 s); melted "
                               "region phi >= 0.5; grid 120; the nominal target "
                               "co-rotates with the part"),
           "rows": rows, "wall_s": time.perf_counter() - t0}
    (OUT_ROOT / f"{shape}_verify.json").write_text(json.dumps(out, indent=1, default=float))
    log(f"DONE {len(rows)} arms, wall {out['wall_s']:.0f} s")
    return out


if __name__ == "__main__":
    _args = sys.argv[1:]
    _controls = "--controls" in _args
    _args = [a for a in _args if a != "--controls"]
    _shape = _args[0]
    _periods = [float(a) for a in _args[1:]] or None
    main(_shape, _periods, with_controls=_controls)
