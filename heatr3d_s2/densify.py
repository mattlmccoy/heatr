"""S2 Task 4: the densify=True coupled march (Matt's explicit assignment).

    ./.venv312/bin/python -m heatr3d_s2.densify

WHY THIS EXISTS. `sigma_density_coeff` has been provably INERT in every prior
study, and the reason is structural, not accidental: it enters
heatr3d.apply_sigma_coupling only through the factor
(1 + b * (rho_rel - rho_ref)), and with densify=False the relative density
NEVER LEAVES rho_ref, so that factor is exactly 1.0 for any b. Every previous
campaign ran densify=False. This task turns densification on and exercises the
term for the first time.

THE CONTROL IS THE INTERESTING PART. Two things must both be true:
  * with densify=FALSE, changing b must change NOTHING, bit-for-bit -- that is
    the prior inertness, proven here rather than asserted;
  * with densify=TRUE, changing b must change something -- otherwise the term
    is still unreachable and the assignment has not been carried out.
Both are measured.

ANTI-CIRCULARITY (the standing S4 rule): the coefficient arms are the ones
pre-registered in Task 0 and are NOT tuned to match any FLIR observation.
Nothing here is fitted to scored frames. The only value with any provenance in
this repo is 0.6 (configs/_archive_old/rfam_eqs_comsol_mimic.yaml l.83-85); it
is EXPLORATORY and is labelled so everywhere it appears.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np

import heatr3d
from heatr3d_s2 import harness

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = RESULTS / "densify_coupled.json"


def _prereg() -> dict:
    return json.loads((RESULTS / "s2_preregistration.json").read_text())["densify_march"]


def _params(b: float, densify_on: bool, pr: dict) -> heatr3d.Params:
    """Coupling params. With densify OFF the re-solve schedule is still armed,
    so the ONLY difference between the two control arms is b itself."""
    return heatr3d.Params(
        phase_update="enthalpy",
        eqs_update_interval_s=float(pr["eqs_update_interval_s"]),
        sigma_temp_coeff_per_K=float(pr["sigma_temp_coeff_per_K"]),
        sigma_density_coeff=float(b))


def _surface_interior_split(T: np.ndarray, part: np.ndarray,
                            grid: heatr3d.Grid, band_voxels: float = 1.5) -> dict:
    """Surface-minus-interior mean part temperature -- the S4 'mechanism 3'
    topology observable."""
    X, Y, _ = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    r = np.sqrt(X ** 2 + Y ** 2)
    interior = part & ((0.010 - r) > band_voxels * grid.h)
    surface = part & ~interior
    return {"surface_mean_c": float(T[surface].mean()),
            "interior_mean_c": float(T[interior].mean()),
            "surface_minus_interior_c":
                float(T[surface].mean() - T[interior].mean()),
            "n_surface": int(surface.sum()), "n_interior": int(interior.sum())}


def run_arm(b: float, densify_on: bool, n: int, max_time_s: float,
            pr: dict) -> dict:
    p = _params(b, densify_on, pr)
    grid = heatr3d.Grid(n=n)
    part = harness.make_part(grid, pr["shape"])
    t0 = time.perf_counter()
    res = heatr3d.run(grid, part, p, max_time_s=max_time_s, phi_target=0.90,
                      densify=densify_on, qrf_gradient="masked")
    wall = time.perf_counter() - t0
    T = res.T_phi90
    rec = {"sigma_density_coeff": float(b), "densify": bool(densify_on),
           "n": n, "max_time_s": max_time_s,
           "t90_s": float(res.t_phi90_s), "reached": bool(res.reached),
           "sigma_T_c": float(res.sigma_T), "T_max_c": float(res.T_max_c),
           "T_mean_c": float(T[part].mean()),
           "n_eqs_solves": int(res.n_eqs_solves),
           "n_eqs_resolves_skipped": int(res.n_eqs_resolves_skipped),
           "wall_s": wall,
           "gates": {"energy_residual_frac": float(res.energy_residual_frac),
                     "clamp_bound": bool(res.clamp_bound),
                     "cfl_violated": bool(res.cfl_violated)},
           "topology": _surface_interior_split(T, part, grid)}
    if res.rho_final is not None:
        rho = res.rho_final[part]
        rec["rho_final"] = {"mean": float(rho.mean()), "min": float(rho.min()),
                            "max": float(rho.max()), "std": float(rho.std())}
    rec["_T"] = T
    return rec


def build(spot_check_grids=(48, 64)) -> dict:
    pr = _prereg()
    n, tmax = int(pr["grid"]), 600.0
    doc = {"what": "S2 Task 4: densify=True coupled march; first exercise of "
                   "sigma_density_coeff",
           "registration": pr, "max_time_s": tmax,
           "controls": {}, "arms": {}, "spot_check": {}}

    # ---- control 1: with densify OFF the coefficient must be INERT ------- #
    off = {}
    for b in (0.0, 0.6):
        r = run_arm(b, False, n, tmax, pr)
        off[str(b)] = r
    a, c = off["0.0"], off["0.6"]
    dT = float(np.max(np.abs(a.pop("_T") - c.pop("_T"))))
    doc["controls"]["densify_off_inertness"] = {
        "arms": {k: {kk: vv for kk, vv in v.items() if kk != "_T"}
                 for k, v in off.items()},
        "max_abs_dT_c": dT,
        "sigma_T_rel_diff": abs(a["sigma_T_c"] - c["sigma_T_c"]) / a["sigma_T_c"],
        "t90_identical": a["t90_s"] == c["t90_s"],
        "inert": bool(dT == 0.0),
        "why": "with densify=False rho never leaves rho_ref, so the coupling "
               "factor (1 + b (rho - rho_ref)) is exactly 1.0 for ANY b. This "
               "is the prior inertness, proven rather than asserted."}

    # ---- the arms: densify ON -------------------------------------------- #
    ref_T = None
    for b in pr["sigma_density_coeff_arms"]:
        r = run_arm(float(b), True, n, tmax, pr)
        T = r.pop("_T")
        if float(b) == 0.0:
            ref_T = T
        else:
            r["vs_zero_arm"] = {
                "max_abs_dT_c": float(np.max(np.abs(T - ref_T))) if ref_T is not None else None,
                "mean_abs_dT_c": float(np.mean(np.abs(T - ref_T))) if ref_T is not None else None}
        doc["arms"][str(b)] = r
    z = doc["arms"]["0.0"]
    for b, r in doc["arms"].items():
        if b == "0.0":
            continue
        r["delta_vs_zero"] = {
            "t90_s": r["t90_s"] - z["t90_s"],
            "sigma_T_c": r["sigma_T_c"] - z["sigma_T_c"],
            "surface_minus_interior_c":
                r["topology"]["surface_minus_interior_c"]
                - z["topology"]["surface_minus_interior_c"],
            "rho_mean": (r.get("rho_final", {}).get("mean", float("nan"))
                         - z.get("rho_final", {}).get("mean", float("nan")))}
    doc["question_a_density_coupling_is_reachable"] = {
        "answer": any(abs(r.get("vs_zero_arm", {}).get("max_abs_dT_c") or 0.0) > 0.0
                      for b, r in doc["arms"].items() if b != "0.0"),
        "note": "the term is reachable iff a nonzero b moves the field with "
                "densify ON; with densify OFF it provably cannot"}

    # ---- question (b): spot check, TWO grids, labelled as such ----------- #
    for g in spot_check_grids:
        r = run_arm(float(pr["sigma_density_coeff_arms"][2]), True, g, tmax, pr)
        r.pop("_T", None)
        doc["spot_check"][str(g)] = r
    doc["spot_check_caveat"] = ("TWO grids at ONE shape is a SPOT CHECK, not a "
                                "convergence claim: the pre-registration "
                                "requires 3 grids for a band and this "
                                "deliberately does not meet it")
    RESULTS.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    return doc


if __name__ == "__main__":
    d = build()
    print(json.dumps({"inert_with_densify_off":
                      d["controls"]["densify_off_inertness"]["inert"],
                      "reachable_with_densify_on":
                      d["question_a_density_coupling_is_reachable"]["answer"]},
                     indent=1))
