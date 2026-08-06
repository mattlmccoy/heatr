"""solve3d Stage A: the densify FORWARD (thermal-ceiling spec, L2 forward half).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.densify_forward --gate

WHY THIS EXISTS (spec sec 2): the degradation-ceiling peak is an END-STATE
quantity -- it occurs at the rho=rho_target densification stop, not at the
melt-onset envelope the Phase C/E solve reads. To gate on it the solve3d
forward must march densify to rho_target so the trajectory reaches that peak.
That march is here.

WHAT IT ADDS: nothing to the OFF path. `march_densify` delegates to
`forward.march_enthalpy(densify=True, ...)`, whose densify branch is guarded so
`densify=False` is BIT-IDENTICAL to the Phase A forward (pinned by
test_densify_forward). The densification physics is the single ported function
`forward.densify_rate` (a bit-for-bit port of heatr3d.densify_rate).

EQUIVALENCE GATE (plan Task 1): the port is judged against heatr3d's OWN densify
march on the extruded-circle anchor, using the Phase A cross-family parity band
machinery (solve3d.gates.combine_spreads / solve3d.crossfamily). The heatr3d
reference is generated in-process (the spike env imports heatr3d) so the gate is
one self-contained short small-case run. Nothing in heatr3d.py is modified;
every interaction is through its public API.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import forward as fwd
from solve3d import gates

RESULTS = Path(__file__).resolve().parent / "results"

# Short small-case gate configuration. Deliberately small and warm-driven so the
# densify path is exercised end to end in a couple of minutes (compute-schedule
# convention: Task-1 equivalence is a SHORT run, not the heavy Task-4 solve).
GATE_GRID_N = 40                    # heatr3d voxel grid
GATE_STOP_MEAN_RHO = 0.68           # a modest densification stop (from rho0=0.55)
GATE_MAX_TIME_S = 400.0
GATE_DRIVE_A = 6.0                  # power multiplier: melt fast, then densify
GATE_SAMPLE_DT_S = 10.0


# --------------------------------------------------------------------------- #
# Public densify march + full forward
# --------------------------------------------------------------------------- #
def march_densify(msh, p: fwd.ForwardParams, stop_mean_rho: float,
                  mats=None, q_dg0=None, q_uniform=None, in_part=None,
                  max_time_s: float = 1500.0, phi_target: float = 0.90,
                  L: float = fwd.L_DOMAIN,
                  sample_dt_s: float | None = None) -> dict:
    """Densify march to a target MEAN relative density.

    A thin delegator to forward.march_enthalpy(densify=True, ...). Kept as a
    named entry so the Stage A stack (drive selection, dopant solve) reads the
    densify end-state through one door, and so the bit-identity test has a
    stable symbol to pin.
    """
    return fwd.march_enthalpy(
        msh, p, in_part=in_part, q_uniform=q_uniform, q_dg0=q_dg0, mats=mats,
        max_time_s=max_time_s, phi_target=phi_target, L=L,
        sample_dt_s=sample_dt_s, densify=True, stop_mean_rho=stop_mean_rho)


def run_densify_forward(shape: str, target_nodes_in_part: int, lc0: float,
                        stop_mean_rho: float,
                        p: fwd.ForwardParams | None = None,
                        max_time_s: float = 1500.0, phi_target: float = 0.90,
                        sample_dt_s: float = GATE_SAMPLE_DT_S,
                        L: float = fwd.L_DOMAIN, petsc_options=None) -> dict:
    """EQS solve -> densify march (the solve3d side of the equivalence gate and
    the forward the Stage A drive sweep / dopant solve drive at their chosen a).

    Coupling OFF only (Stage A scope): the ceiling is nearly dopant-independent
    and the densify port is the thing under test, so the in-march EQS re-solve
    is not exercised here.
    """
    p = p or fwd.ForwardParams()
    case = fwd.eqs_case(shape, target_nodes_in_part, lc0, p=p,
                        petsc_options=petsc_options)
    msh, mats, q_fn = case["msh"], case["mats"], case["q"]
    t0 = time.perf_counter()
    march = march_densify(msh, p, stop_mean_rho, mats=mats, q_dg0=q_fn,
                          max_time_s=max_time_s, phi_target=phi_target,
                          L=L, sample_dt_s=sample_dt_s)
    march["wall_march_s"] = time.perf_counter() - t0
    out = {k: v for k, v in case.items() if k not in ("msh", "mats", "q")}
    out.update(march)
    out["shape"] = shape
    return out


# --------------------------------------------------------------------------- #
# heatr3d reference densify march (voxel engine, its own EQS + densify)
# --------------------------------------------------------------------------- #
def _heatr3d_densify_monolithic(shape: str, n: int, stop_mean_rho: float,
                                p_kwargs: dict, max_time_s: float) -> dict:
    """One heatr3d densify run (rho_rel is an internal evolving field, so it
    cannot be segment-chained the way the melt-onset curve sampler does). We
    read the trajectory-independent scalars the gate needs: final part-mean rho,
    the end-state part-mean T, and the end-state part peak T_end_max. The rho
    trajectory is compared at the single end point; the T CURVE equivalence is
    already covered by the Phase A melt-onset gate on the same anchor.
    """
    import heatr3d

    p = heatr3d.Params(phase_update="enthalpy", **p_kwargs)
    grid = heatr3d.Grid(n=n)
    if shape == "circle":
        part = heatr3d.make_geometry(grid, "cylinder", diam=0.020)
    elif shape == "square":
        part = heatr3d.make_geometry(grid, "square", diam=0.020, zspan=grid.L)
    else:
        raise ValueError(f"unknown anchor shape {shape!r}")
    res = heatr3d.run(grid, part, p, densify=True, stop_mean_rho=stop_mean_rho,
                      max_time_s=max_time_s, qrf_gradient="masked")
    rho_f = res.rho_final
    return {
        "engine": "heatr3d", "shape": shape, "n": n,
        "n_voxels_in_part": int(part.sum()),
        "part_mean_rho": float(rho_f[part].mean()),
        "T_end_max_c": float(res.T_final[part].max()),
        "part_mean_T_end_c": float(res.T_final[part].mean()),
        "t90_s": float(res.t_phi90_s), "reached": bool(res.reached),
        "exposure_s": float(res.exposure_s),
        "energy_residual_frac": float(res.energy_residual_frac),
        "clamp_bound": bool(res.clamp_bound),
    }


# --------------------------------------------------------------------------- #
# Densify parity tolerances: heatr3d's OWN grid self-spread (cases.py method)
# --------------------------------------------------------------------------- #
DENSIFY_TOL_GRIDS = (40, 64)        # coarse gate grid, finer reference
DENSIFY_TOL_SAFETY = 1.5
_DENSIFY_QUANTS = ("part_mean_rho", "T_end_max_c", "part_mean_T_end_c")


def measure_densify_self_spread(grids=DENSIFY_TOL_GRIDS,
                                stop_mean_rho: float = GATE_STOP_MEAN_RHO,
                                max_time_s: float = GATE_MAX_TIME_S,
                                drive_a: float = GATE_DRIVE_A) -> dict:
    """The densify CROSS-FAMILY parity band on the extruded circle, measured the
    same way Phase A did (cases.measure_self_spread + crossfamily.py): BOTH
    engines' own two-grid self-spreads combined by the triangle-inequality sum
    rule at 1.5x (gates.combine_spreads).

    Why cross-family and not just heatr3d's own spread: Phase A's close-out
    proved a heatr3d-only band is unachievable by any correct dolfinx
    implementation -- the two families converge to the same continuum answer but
    from different discretization errors, so the bound on |A - B| is the SUM of
    both. T_end_max is a MAX (a single hottest cell), so its own grid spread is
    the widest; a borrowed sigma_T std band is the wrong yardstick for it, which
    is why this is measured. Downstream READS this file; widening is forbidden.
    """
    base = fwd.ForwardParams()
    pw = base.power_density_w_per_m3 * float(drive_a)
    p_solve = fwd.ForwardParams(power_density_w_per_m3=pw)

    heatr_runs, solve_runs = {}, {}
    for n in grids:
        print(f"[densify-band] heatr3d circle n={n} ...", flush=True)
        heatr_runs[f"n{n}"] = _heatr3d_densify_monolithic(
            "circle", n, stop_mean_rho, {"power_density_w_per_m3": pw},
            max_time_s)
        target_nodes = heatr_runs[f"n{n}"]["n_voxels_in_part"]
        print(f"[densify-band] solve3d circle matched-nodes={target_nodes} "
              f"(heatr3d n={n}) ...", flush=True)
        s = run_densify_forward("circle", target_nodes, base_lc_for_grid(n),
                                stop_mean_rho, p=p_solve, max_time_s=max_time_s,
                                sample_dt_s=GATE_SAMPLE_DT_S)
        solve_runs[f"n{n}"] = {
            "part_mean_rho": float(s["part_mean_rho"]),
            "T_end_max_c": float(s["T_end_max_c"]),
            "part_mean_T_end_c": float(s["part_mean_T_c"]),
            "n_nodes_in_part": int(s["n_nodes_in_part"]),
        }

    lo, hi = f"n{grids[0]}", f"n{grids[-1]}"      # hi = finer = reference
    h_spread = {q: abs(float(heatr_runs[lo][q]) - float(heatr_runs[hi][q]))
                / abs(float(heatr_runs[hi][q])) for q in _DENSIFY_QUANTS}
    d_spread = {q: abs(float(solve_runs[lo][q]) - float(solve_runs[hi][q]))
                / abs(float(solve_runs[hi][q])) for q in _DENSIFY_QUANTS}
    tolerances = {q: gates.combine_spreads(h_spread[q], d_spread[q])
                  for q in _DENSIFY_QUANTS}
    doc = {
        "what": "Stage A densify CROSS-FAMILY parity tolerances on the "
                "extruded-circle anchor (enthalpy, masked Q, coupling off, "
                "drive_a=%g, stop_mean_rho=%g). tolerance = 1.5 x "
                "(heatr3d_spread + solve3d_spread), the Phase A cross-family "
                "rule (gates.combine_spreads); each spread is that engine's own "
                "n=%d-vs-n=%d grid self-spread. Frozen before the port is "
                "judged; widening forbidden."
                % (drive_a, stop_mean_rho, grids[0], grids[-1]),
        "combination_rule": gates.COMBINATION_RULE,
        "safety_factor": gates.CROSS_FAMILY_SAFETY,
        "grids": list(grids), "reference_grid": hi,
        "config": {"drive_a": drive_a, "stop_mean_rho": stop_mean_rho,
                   "max_time_s": max_time_s, "power_density_w_per_m3": pw},
        "heatr3d_runs": heatr_runs, "solve3d_runs": solve_runs,
        "heatr3d_spread": h_spread, "solve3d_spread": d_spread,
        "combined_spread": {q: h_spread[q] + d_spread[q] for q in _DENSIFY_QUANTS},
        "tolerances": tolerances,
    }
    gates.write_json("densify_parity_tolerances.json", doc)
    print(json.dumps({"heatr3d_spread": h_spread, "solve3d_spread": d_spread,
                      "tolerances": tolerances}, indent=1))
    return doc


# --------------------------------------------------------------------------- #
# Equivalence gate
# --------------------------------------------------------------------------- #
def equivalence_gate(n: int = GATE_GRID_N,
                     stop_mean_rho: float = GATE_STOP_MEAN_RHO,
                     max_time_s: float = GATE_MAX_TIME_S,
                     drive_a: float = GATE_DRIVE_A,
                     mutate_drop_liquid: bool = False) -> dict:
    """Run heatr3d densify (reference) and solve3d densify (port) on the
    extruded-circle anchor and judge the port against the Phase A cross-family
    parity band. `mutate_drop_liquid` drops the densify-rate liquid branch in
    the solve3d port to prove the gate REJECTS a broken densification rate
    (plan Task 1 mutation check)."""
    base = fwd.ForwardParams()
    pw = base.power_density_w_per_m3 * float(drive_a)
    p = fwd.ForwardParams(power_density_w_per_m3=pw)
    p_kwargs = {"power_density_w_per_m3": pw}

    ref = _heatr3d_densify_monolithic("circle", n, stop_mean_rho, p_kwargs,
                                      max_time_s)

    # Match solve3d's in-part NODE count to heatr3d's in-part VOXEL count.
    target_nodes = ref["n_voxels_in_part"]
    lc0 = float(base_lc_for_grid(n))

    _orig = fwd.densify_rate
    if mutate_drop_liquid:
        fwd.densify_rate = _densify_rate_no_liquid
    try:
        port = run_densify_forward("circle", target_nodes, lc0, stop_mean_rho,
                                   p=p, max_time_s=max_time_s,
                                   sample_dt_s=GATE_SAMPLE_DT_S)
    finally:
        fwd.densify_rate = _orig

    tol_path = RESULTS / "densify_parity_tolerances.json"
    if not tol_path.exists():
        raise FileNotFoundError(
            f"{tol_path} not found. Run `--measure-band` first: the densify "
            "tolerances are MEASURED from heatr3d's own grid spread, never "
            "borrowed from the melt-onset sigma_T band (a std is not the right "
            "yardstick for a peak).")
    tolerances = json.loads(tol_path.read_text())["tolerances"]

    port_mean_T_end = float(port["part_mean_T_c"])
    rel = {
        "part_mean_rho": gates.rel_spread(port["part_mean_rho"],
                                          ref["part_mean_rho"]),
        "T_end_max_c": gates.rel_spread(port["T_end_max_c"], ref["T_end_max_c"]),
        "part_mean_T_end_c": gates.rel_spread(port_mean_T_end,
                                              ref["part_mean_T_end_c"]),
    }
    port_val = {"part_mean_rho": port["part_mean_rho"],
                "T_end_max_c": port["T_end_max_c"],
                "part_mean_T_end_c": port_mean_T_end}
    checks = {
        q: {"port": port_val[q], "ref": ref[q], "rel": rel[q],
            "tolerance": float(tolerances[q]),
            "pass": bool(rel[q] <= float(tolerances[q]))}
        for q in _DENSIFY_QUANTS
    }
    all_pass = bool(all(c["pass"] for c in checks.values()))
    doc = {
        "what": "Stage A Task 1 densify-forward equivalence gate: solve3d "
                "march_densify vs heatr3d's own densify march on the "
                "extruded-circle anchor, judged against the MEASURED densify "
                "parity band (densify_parity_tolerances.json = 1.5x heatr3d's "
                "own grid self-spread; cases.py method). rho, T_end_max and "
                "part-mean T are the end-state scalars; the melt-onset T CURVE "
                "equivalence is already covered by the Phase A gate on the same "
                "anchor.",
        "band_source": "densify_parity_tolerances.json (1.5x heatr3d grid self-spread)",
        "tolerances": tolerances,
        "config": {"grid_n": n, "stop_mean_rho": stop_mean_rho,
                   "max_time_s": max_time_s, "drive_a": drive_a,
                   "power_density_w_per_m3": pw,
                   "solve3d_target_nodes_in_part": target_nodes,
                   "solve3d_nodes_in_part": int(port["n_nodes_in_part"]),
                   "mutate_drop_liquid": bool(mutate_drop_liquid)},
        "reference_heatr3d": ref,
        "port_solve3d": {
            "part_mean_rho": port["part_mean_rho"],
            "T_end_max_c": port["T_end_max_c"],
            "true_peak_T_c": port["true_peak_T_c"],
            "part_mean_T_end_c": port_mean_T_end,
            "reached_rho": port["reached_rho"], "exposure_s": port["exposure_s"],
            "energy_residual_frac": port["energy_residual_frac"],
            "clamp_bound": port["clamp_bound"],
            "n_nodes_in_part": int(port["n_nodes_in_part"]),
        },
        "checks": checks,
        "all_pass": all_pass,
    }
    return doc


def ceiling_band_check(n: int = GATE_GRID_N,
                       stop_mean_rho: float = GATE_STOP_MEAN_RHO,
                       max_time_s: float = GATE_MAX_TIME_S,
                       drive_a: float = GATE_DRIVE_A,
                       band_rel: float = 0.05) -> dict:
    """Task 2 acceptance: the KS aggregate tracks the true max within the
    pre-registered 5% band on BOTH a synthetic field AND a real densify march,
    and the ceiling verdict reads the TRUE max. Writes ceiling_ks_band.json."""
    from solve3d import ceiling as cl

    base = fwd.ForwardParams()
    pw = base.power_density_w_per_m3 * float(drive_a)
    p = fwd.ForwardParams(power_density_w_per_m3=pw)

    ref = _heatr3d_densify_monolithic("circle", n, stop_mean_rho,
                                      {"power_density_w_per_m3": pw}, max_time_s)
    port = run_densify_forward("circle", ref["n_voxels_in_part"],
                               base_lc_for_grid(n), stop_mean_rho, p=p,
                               max_time_s=max_time_s, sample_dt_s=GATE_SAMPLE_DT_S)
    obs = cl.observe(port, ceiling_c=250.0, melt_onset_c=185.0, warn_c=235.0)

    rng = np.random.default_rng(7)
    synth = np.concatenate([rng.normal(235.0, 5.0, 4000),
                            rng.normal(248.0, 1.5, 200)])
    syn = cl.peak_temp(synth)

    doc = {
        "what": "Stage A Task 2 KS-vs-true-max tracking band. The KS aggregate "
                "is the smooth gradient proxy ONLY; T_ceiling_ok is on the "
                "TRUE max. Verified within the pre-registered 5% band on a "
                "synthetic field and a real densify march.",
        "band_rel_registered": band_rel,
        "rho_per_c": cl.KS_RHO_PER_C,
        "synthetic": {"true_max_c": syn["true_max_c"],
                      "ks_aggregate_c": syn["ks_aggregate_c"],
                      "gap_rel": syn["gap_rel"],
                      "pass": bool(syn["gap_rel"] <= band_rel)},
        "real_march": {"true_max_c": obs["peak"]["true_max_c"],
                       "ks_aggregate_c": obs["peak"]["ks_aggregate_c"],
                       "gap_rel": obs["peak"]["gap_rel"],
                       "pass": bool(obs["peak"]["gap_rel"] <= band_rel),
                       "true_peak_matches_march": obs["true_peak_matches_march"],
                       "T_ceiling_ok": obs["ceiling"]["T_ceiling_ok"],
                       "ceiling_read_on": obs["ceiling"]["peak_source"],
                       "melt_complete": obs["completeness"]["complete"],
                       "min_in_part_peak_c": obs["completeness"]["min_in_part_peak_c"]},
    }
    doc["all_pass"] = bool(doc["synthetic"]["pass"]
                           and doc["real_march"]["pass"]
                           and doc["real_march"]["true_peak_matches_march"])
    gates.write_json("ceiling_ks_band.json", doc)
    print(json.dumps({"all_pass": doc["all_pass"],
                      "synthetic_gap_rel": doc["synthetic"]["gap_rel"],
                      "real_gap_rel": doc["real_march"]["gap_rel"],
                      "T_ceiling_ok": doc["real_march"]["T_ceiling_ok"],
                      "ceiling_read_on": doc["real_march"]["ceiling_read_on"]},
                     indent=1))
    return doc


def _densify_rate_no_liquid(T, phi, rho_rel, p):
    """MUTATION: the densify rate with the liquid viscous-capillary branch
    dropped. Must fail the equivalence gate (Task 1 mutation check)."""
    Tk = np.maximum(np.array(T, dtype=float, copy=True) + 273.15, 1.0)
    rho_term = np.power(np.clip(1.0 - rho_rel, 0.0, 1.0), p.dens_rho_exp)
    kss = p.dens_k0_ss * np.exp(-p.dens_ea_ss / (fwd.R_GAS * Tk))
    ss_drive = np.power(np.clip(1.0 - phi, 0.0, 1.0), p.dens_phi_solid_exp)
    return kss * ss_drive * rho_term          # liquid branch DROPPED


def base_lc_for_grid(n: int) -> float:
    """A gmsh seed length near the heatr3d voxel size h = L / n, so match_lc
    starts close to the in-part node target."""
    return fwd.L_DOMAIN / float(n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure-band", action="store_true",
                    help="measure heatr3d's own densify grid self-spread and "
                         "freeze densify_parity_tolerances.json (run first)")
    ap.add_argument("--ceiling-band", action="store_true",
                    help="verify the KS-vs-true-max tracking band on a real "
                         "densify march + a synthetic field (Task 2)")
    ap.add_argument("--gate", action="store_true",
                    help="run the densify-forward equivalence gate")
    ap.add_argument("--mutate", action="store_true",
                    help="run the gate with the liquid densify term dropped "
                         "(must FAIL)")
    args = ap.parse_args()
    if args.measure_band:
        measure_densify_self_spread()
    if args.ceiling_band:
        ceiling_band_check()
    if args.gate or args.mutate:
        doc = equivalence_gate(mutate_drop_liquid=bool(args.mutate))
        name = ("densify_equivalence_mutation.json" if args.mutate
                else "densify_equivalence.json")
        gates.write_json(name, doc)
        print(json.dumps({"all_pass": doc["all_pass"],
                          "checks": {k: {"rel": v["rel"],
                                         "tol": v["tolerance"], "pass": v["pass"]}
                                     for k, v in doc["checks"].items()}},
                         indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
