"""solve3d Phase A anchor cases + the MEASURED parity tolerances (plan Task 1).

RUNS IN THE geo-prewarp VENV (imports heatr3d, which needs scipy):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ./.venv312/bin/python -m solve3d.cases --measure

heatr3d.py is READ-ONLY here. Every heatr3d interaction goes through its public
API (`make_geometry`, `build_gamma`, `solve_eqs_3d`, `compute_qrf_3d`, `run`)
and its documented validation hooks (`qrf_override`, `T0_override`, `t_start_s`).

WHY THE TOLERANCES ARE MEASURED, NOT ASSERTED
---------------------------------------------
Phase A asks whether a dolfinx forward reproduces heatr3d's coupled forward.
"Reproduces" only has meaning relative to how much heatr3d disagrees with
ITSELF between its two certified anchor grids (n=64 and n=96; n=96 is the
EQS-01 full-physics ceiling, heatr3d.EQS_MAX_GRID_FULL_PHYSICS). That
self-discretization spread is measured here and frozen into
solve3d/results/parity_tolerances.json at 1.5x before any dolfinx number
exists. Nothing downstream may widen it.

ANCHOR CASES (extrusions, so the case is exactly z-invariant and a conforming
FEM mesh has no z structure to disagree about):
  * circle -- heatr3d.make_geometry(..., "cylinder", diam=20 mm), full height
  * square -- heatr3d.make_geometry(..., "square", diam=20 mm, zspan=L),
    i.e. a FULL-HEIGHT prism, the same convention
    heatr3d_d1_spike/run_heatr3d_reference.py used.
Drive: qrf_gradient="masked" ONLY (the corrected part-confined stencil; spec
sec 4 forbids building on legacy Q).
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np

import heatr3d

from solve3d.gates import curve_rel_l2

logger = logging.getLogger(__name__)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PART_DIAM_M = 0.020
ANCHOR_GRIDS = (64, 96)
SAMPLE_DT_S = 10.0                  # part-mean heating-curve sample interval
MAX_TIME_S = 1500.0
PHI_TARGET = 0.90
TOLERANCE_SAFETY = 1.5              # tolerances = this x the measured spread


# --------------------------------------------------------------------------- #
# Geometry + drive
# --------------------------------------------------------------------------- #
def make_part(grid: heatr3d.Grid, shape: str) -> np.ndarray:
    """The Phase A anchor geometries, as full-height extrusions."""
    if shape == "circle":
        return heatr3d.make_geometry(grid, "cylinder", diam=PART_DIAM_M)
    if shape == "square":
        return heatr3d.make_geometry(grid, "square", diam=PART_DIAM_M,
                                     zspan=grid.L)
    raise ValueError(f"unknown anchor shape {shape!r}")


def compute_drive(grid: heatr3d.Grid, part: np.ndarray, p: heatr3d.Params,
                  T: np.ndarray | None = None,
                  rho_rel: np.ndarray | None = None) -> dict:
    """heatr3d's own EQS solve + corrected (masked-gradient) Q_rf.

    When p.sigma_temp_coeff_per_K / p.sigma_density_coeff are nonzero and T /
    rho_rel are given, heatr3d.apply_sigma_coupling is applied first -- exactly
    what heatr3d.run does at a scheduled re-solve."""
    gamma = heatr3d.build_gamma(part, p)
    if T is not None:
        if rho_rel is None:
            rho_rel = np.full(part.shape, p.rho_rel)
        gamma = heatr3d.apply_sigma_coupling(gamma, part, T, rho_rel, p)
    t0 = time.perf_counter()
    V = heatr3d.solve_eqs_3d(gamma, grid, p)
    wall_eqs = time.perf_counter() - t0
    Q = heatr3d.compute_qrf_3d(V, gamma, grid, p, doped=part, premix=False,
                               qrf_gradient="masked")
    return {"gamma": gamma, "V": V, "Qrf": Q, "wall_eqs_s": wall_eqs,
            "p_total_w": float(Q.sum() * grid.dV)}


# --------------------------------------------------------------------------- #
# Segmented march (samples the part-mean heating curve)
# --------------------------------------------------------------------------- #
def march_sampled(grid: heatr3d.Grid, part: np.ndarray, p: heatr3d.Params,
                  Qrf: np.ndarray, max_time_s: float = MAX_TIME_S,
                  phi_target: float = PHI_TARGET,
                  sample_dt_s: float = SAMPLE_DT_S) -> dict:
    """Chain heatr3d.run through T0_override so the part-mean T(t) curve can be
    sampled WITHOUT modifying heatr3d.py.

    Exactness: with densify=False the whole solver state is T (rho_rel is held
    at p.rho_rel and phase is a pure function of T), and qrf_override freezes
    the drive, so segmenting the march is arithmetically the identical march.
    Pinned by test_cases.test_segmented_march_reproduces_a_monolithic_run at a
    relative 1e-12.
    """
    n_seg_steps = int(round(sample_dt_s / p.dt_s))
    if abs(n_seg_steps * p.dt_s - sample_dt_s) > 1e-12:
        raise ValueError("sample_dt_s must be an exact multiple of p.dt_s")
    T = np.full(part.shape, p.preheat_c, dtype=np.float64)
    t_now = 0.0
    curve_t = [0.0]
    curve_T = [float(T[part].mean())]
    curve_phi = [float(heatr3d.phase_fraction(T, p)[0][part].mean())]
    e_in = e_stored = e_loss = 0.0
    clamp_bound = False
    out: dict = {"reached": False, "t90_s": float("nan")}
    while t_now < max_time_s - 1e-12:
        seg = min(sample_dt_s, max_time_s - t_now)
        res = heatr3d.run(grid, part, p, qrf_override=Qrf, max_time_s=seg,
                          phi_target=phi_target, T0_override=T,
                          t_start_s=t_now)
        e_in += res.energy_in_j
        e_stored += res.energy_stored_j
        e_loss += res.energy_loss_j
        clamp_bound = clamp_bound or bool(res.clamp_bound)
        T = res.T_final
        if res.reached:
            out.update({"reached": True, "t90_s": t_now + res.t_phi90_s,
                        "T_phi90": res.T_phi90.copy(),
                        "sigma_T_c": float(res.sigma_T),
                        "T_max_c": float(res.T_max_c)})
            curve_t.append(t_now + res.t_phi90_s)
            curve_T.append(float(res.T_phi90[part].mean()))
            curve_phi.append(float(heatr3d.phase_fraction(res.T_phi90, p)[0][part].mean()))
            break
        t_now += seg
        curve_t.append(t_now)
        curve_T.append(float(T[part].mean()))
        curve_phi.append(float(heatr3d.phase_fraction(T, p)[0][part].mean()))
    if not out["reached"]:
        logger.warning("march_sampled: phi_target=%.2f never reached in %.1f s",
                       phi_target, max_time_s)
        out.update({"T_phi90": T.copy(),
                    "sigma_T_c": float(T[part].std()),
                    "T_max_c": float(T[part].max())})
    out.update({
        "curve_t_s": curve_t, "curve_part_mean_T_c": curve_T,
        "curve_part_mean_phi": curve_phi,
        "T_final": T,
        "energy_in_j": e_in, "energy_stored_j": e_stored,
        "energy_loss_j": e_loss,
        "energy_residual_frac": (e_in - e_stored - e_loss) / max(e_in, 1e-30),
        "clamp_bound": clamp_bound,
    })
    return out


# --------------------------------------------------------------------------- #
# Task 1: measure heatr3d's own self-discretization spread
# --------------------------------------------------------------------------- #
def run_anchor(shape: str, n: int, p: heatr3d.Params, save_npz: bool = True) -> dict:
    grid = heatr3d.Grid(n=n)
    part = make_part(grid, shape)
    drive = compute_drive(grid, part, p)
    t0 = time.perf_counter()
    m = march_sampled(grid, part, p, drive["Qrf"])
    wall_march = time.perf_counter() - t0
    rec = {
        "shape": shape, "n": n, "h_m": grid.h, "L_m": grid.L,
        "n_voxels_in_part": int(part.sum()),
        "part_volume_m3": float(part.sum() * grid.dV),
        "p_total_w": drive["p_total_w"],
        "wall_eqs_s": drive["wall_eqs_s"], "wall_march_s": wall_march,
        "t90_s": m["t90_s"], "reached": bool(m["reached"]),
        "sigma_T_c": m["sigma_T_c"], "T_max_c": m["T_max_c"],
        "energy_residual_frac": m["energy_residual_frac"],
        "clamp_bound": m["clamp_bound"],
        "curve_t_s": m["curve_t_s"],
        "curve_part_mean_T_c": m["curve_part_mean_T_c"],
        "curve_part_mean_phi": m["curve_part_mean_phi"],
    }
    if save_npz:
        RESULTS.mkdir(parents=True, exist_ok=True)
        npz = RESULTS / f"anchor_heatr3d_{shape}_n{n}.npz"
        np.savez_compressed(
            npz, x=grid.x, y=grid.y, z=grid.z, h=grid.h, L=grid.L, n=n,
            part=part, Qrf=drive["Qrf"], V=drive["V"],
            T_phi90=m["T_phi90"], T_final=m["T_final"],
            curve_t_s=np.asarray(m["curve_t_s"]),
            curve_part_mean_T_c=np.asarray(m["curve_part_mean_T_c"]),
            curve_part_mean_phi=np.asarray(m["curve_part_mean_phi"]),
            t90_s=m["t90_s"], sigma_T_c=m["sigma_T_c"])
        rec["npz"] = npz.name
    return rec


def measure_self_spread(shape: str = "circle",
                        grids: tuple[int, ...] = ANCHOR_GRIDS) -> dict:
    """Run the anchor at each certified grid and report heatr3d's own spread."""
    p = heatr3d.Params(phase_update="enthalpy")     # coupled defaults OFF
    runs = {}
    for n in grids:
        print(f"[task1] heatr3d {shape} n={n} ...", flush=True)
        runs[f"n{n}"] = run_anchor(shape, n, p)
        print(json.dumps({k: v for k, v in runs[f"n{n}"].items()
                          if not k.startswith("curve_")}, indent=1), flush=True)
    lo, hi = f"n{grids[0]}", f"n{grids[-1]}"
    a, b = runs[lo], runs[hi]                       # b = finer = reference
    curve = curve_rel_l2(a["curve_t_s"], a["curve_part_mean_T_c"],
                         b["curve_t_s"], b["curve_part_mean_T_c"])
    spread = {
        "reference_grid": hi,
        "t90_rel_spread": abs(a["t90_s"] - b["t90_s"]) / b["t90_s"],
        "curve_rel_l2_spread": curve["rel_l2"],
        "sigma_T_rel_spread": abs(a["sigma_T_c"] - b["sigma_T_c"]) / abs(b["sigma_T_c"]),
        "curve_detail": curve,
    }
    return {"shape": shape, "phase_update": "enthalpy",
            "qrf_gradient": "masked", "coupling": "off (defaults)",
            "grids": list(grids), "runs": runs, "spread": spread}


def write_tolerances(path: Path | None = None) -> dict:
    """Task 1 deliverable: the FROZEN Phase A parity gates."""
    meas = measure_self_spread()
    s = meas["spread"]
    doc = {
        "what": "Phase A parity tolerances, MEASURED from heatr3d's own "
                "n=64-vs-n=96 self-discretization spread on the extruded-circle "
                "anchor (enthalpy scheme, corrected/masked Q, coupling OFF) and "
                "frozen at %.1fx before any dolfinx number existed. Downstream "
                "tasks READ this file; widening it is forbidden."
                % TOLERANCE_SAFETY,
        "safety_factor": TOLERANCE_SAFETY,
        "measurement": meas,
        "tolerances": {
            "t90_rel": TOLERANCE_SAFETY * s["t90_rel_spread"],
            "curve_rel_l2": TOLERANCE_SAFETY * s["curve_rel_l2_spread"],
            "sigma_T_rel": TOLERANCE_SAFETY * s["sigma_T_rel_spread"],
        },
        "raw_spread": {k: v for k, v in s.items() if k != "curve_detail"},
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = path or (RESULTS / "parity_tolerances.json")
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["tolerances"], indent=1))
    return doc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true",
                    help="run Task 1 and write results/parity_tolerances.json")
    ap.add_argument("--anchor", nargs=2, metavar=("SHAPE", "N"),
                    help="run one anchor case and cache its npz")
    args = ap.parse_args()
    if args.measure:
        write_tolerances()
    if args.anchor:
        shape, n = args.anchor[0], int(args.anchor[1])
        rec = run_anchor(shape, n, heatr3d.Params(phase_update="enthalpy"))
        print(json.dumps({k: v for k, v in rec.items()
                          if not k.startswith("curve_")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
