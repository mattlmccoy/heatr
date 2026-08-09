"""Generalize the B1-B4 ceiling-coupled dopant loop to CUBE + PYRAMID.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.geom_generalize --shape cube --gate
    ... --shape pyramid --gate
    ... --shape cube --probe --drives 0.20,0.24,0.28

This is a DRIVER, not new physics. It reuses the FD-gated square machinery with
only the geometry swapped (density_adjoint.build_coarse_case(shape=...) routes
cube/pyramid to the Phase E conforming mesh). The square loop is closed
cross-engine; the point here is to prove the SAME closed loop is a general method.

THE IRON LAW (per geometry, before any solve): a correct density co-state on the
square is NOT proof on the cube/pyramid -- different mesh, different element
quality. So on EACH new solid we re-run, at the frozen 1e-6 tol with NO widening:

  1. the combined AL gradient FD gate (stage_b4.fd_gate_reconfirm) -- the exact
     gradient the heavy solve uses, re-gated on this mesh; the _drop_al_term
     mutation must bite.
  2. the density co-state's own FD gate (density_adjoint.fd_gate) -- the
     drop-lambda_rho mutation must bite (the density co-state is load-bearing).
  3. the _march-vs-production fidelity check (density_adjoint.march_fidelity_check)
     -- the gate forward must match the arbiter forward on this mesh.

If any of these fails on a geometry, that geometry is BLOCKED (no solve).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from solve3d import density_adjoint as da, stage_a, stage_b4 as b4

RESULTS = Path(__file__).resolve().parent / "results"

# GATE DRIVE (a device, not the solve drive). The coarse gate case must sit in
# the MELTING + DENSIFYING regime, or the density co-state (lambda_rho) is
# trivially inactive and its drop-lambda_rho mutation cannot bite -- so the gate
# would not actually test the density path on the new mesh. The square's 0.40x
# baseline melts the square coarse case (207 C, max_rho 0.95) but only reaches
# 143 C on the coarse CUBE (no melt) because the cube absorbs less RF per volume
# at the same absolute power density. MEASURED coarse-case sweep: at 2.0x the
# 0.40x baseline (1,273,240 W/m^3) the cube reaches 222.6 C / max_rho 0.999 and
# the pyramid 197.2 C / max_rho 0.932 -- both melt and densify strongly with ALL
# clips inactive and max_rho just under 1.0 (no rho-saturation kink), the same
# active regime the square gate uses. This gate drive activates lambda_rho; the
# SOLVE drive is a separate, backed-off value chosen by the drive probe.
GATE_DRIVE_MULT = 2.0


def _gate_power_density() -> float:
    return GATE_DRIVE_MULT * float(stage_a.recommended_power_settings(0.40)
                                   ["power_density_w_per_m3"])


def run_fd_gate(shape: str) -> dict:
    """Re-gate the combined AL gradient, the density co-state, and the march
    fidelity on the coarse `shape` mesh at the frozen 1e-6 tol, in the DENSIFYING
    gate regime (GATE_DRIVE_MULT) so lambda_rho is exercised. Writes three
    per-geometry artifacts and a combined verdict."""
    t0 = time.perf_counter()
    pw = _gate_power_density()
    print(f"[geom {shape}] gate drive = {GATE_DRIVE_MULT}x 0.40x-baseline "
          f"= {pw:.0f} W/m^3 (densifying regime, activates lambda_rho)", flush=True)
    print(f"[geom {shape}] combined AL gradient FD gate (fd_gate_reconfirm) ...",
          flush=True)
    al = b4.fd_gate_reconfirm(shape=shape, power_density=pw)
    b4.stage_b._write_json(RESULTS / f"geom_al_fd_gate_{shape}.json", al)
    print(f"[geom {shape}] AL gate: worst_rel_err={al['worst_rel_err']:.3e} "
          f"launch_ok={al['launch_ok']} mutation_bites={al['mutation_bites']} "
          f"n_design={al['n_design']} n_part_nodes={al['n_part_nodes']}", flush=True)

    print(f"[geom {shape}] density co-state FD gate (drop-lambda_rho mutation) ...",
          flush=True)
    dcase = da.build_coarse_case(shape=shape, power_density=pw)
    dens = da.fd_gate(case=dcase, out_name=f"geom_density_fd_gate_{shape}.json")
    print(f"[geom {shape}] density gate: worst_rel_err={dens['worst_rel_err']:.3e} "
          f"passed={dens['fd_gate_passed']} mutation_bites={dens['mutation_bites']} "
          f"launch_ok={dens['launch_ok']}", flush=True)

    print(f"[geom {shape}] _march-vs-production fidelity ...", flush=True)
    fcase = da.build_coarse_case(shape=shape, power_density=pw)
    fid = da.march_fidelity_check(case=fcase,
                                  out_name=f"geom_march_fidelity_{shape}.json")
    print(f"[geom {shape}] fidelity: rel_T={fid['rel_T_in_part']:.3e} "
          f"agree={fid['agree']}", flush=True)

    green = bool(al["launch_ok"] and al["mutation_bites"]
                 and dens["launch_ok"] and dens["mutation_bites"]
                 and fid["agree"])
    verdict = {
        "shape": shape,
        "iron_law": "combined AL gate + density co-state gate + march fidelity, "
                    "re-run on THIS mesh at the frozen 1e-6 tol (no widening)",
        "gate_drive_mult_of_0p40x": GATE_DRIVE_MULT,
        "gate_power_density_w_per_m3": pw,
        "gate_regime_note": "densifying regime so lambda_rho is load-bearing; the "
                            "SOLVE drive is the backed-off value from the probe",
        "al_gate": {"worst_rel_err": al["worst_rel_err"], "tol": al["tol"],
                    "launch_ok": al["launch_ok"],
                    "mutation_bites": al["mutation_bites"]},
        "density_gate": {"worst_rel_err": dens["worst_rel_err"],
                         "pass_rel_err": dens["pass_rel_err"],
                         "fd_gate_passed": dens["fd_gate_passed"],
                         "mutation_worst_rel_err_drop_lambda_rho":
                             dens["mutation_worst_rel_err_drop_lambda_rho"],
                         "mutation_bites": dens["mutation_bites"],
                         "launch_ok": dens["launch_ok"]},
        "march_fidelity": {"rel_T_in_part": fid["rel_T_in_part"],
                           "rel_mean_rho": fid["rel_mean_rho"],
                           "agree": fid["agree"]},
        "fd_gate_green": green,
        "n_design": al["n_design"], "n_part_nodes": al["n_part_nodes"],
        "wall_s": round(time.perf_counter() - t0, 1),
    }
    b4.stage_b._write_json(RESULTS / f"geom_fd_gate_verdict_{shape}.json", verdict)
    tag = "GREEN -- ready for the drive probe" if green else "BLOCKED"
    print(f"[geom {shape}] FD-GATE {tag} (wall {verdict['wall_s']}s)", flush=True)
    return verdict


def run_probe(shape: str, drives: tuple) -> dict:
    """The B4 drive-backoff probe on the fine hold-out for `shape`: uniform peak
    at each candidate drive, pick the backed-off drive whose uniform peak lands a
    few C under T_eff=235. REFUSES unless the FD gate is green for this geometry."""
    v = RESULTS / f"geom_fd_gate_verdict_{shape}.json"
    if not v.exists() or not json.loads(v.read_text()).get("fd_gate_green"):
        raise RuntimeError(
            f"run_probe refused ({shape}): FD gate not green. Run --gate first "
            "and confirm fd_gate_green before probing a shape's drives.")
    return b4.drive_probe(candidates=tuple(drives), shape=shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True, choices=["cube", "pyramid", "square"])
    ap.add_argument("--gate", action="store_true", help="run the per-geometry FD gate")
    ap.add_argument("--probe", action="store_true", help="run the drive-backoff probe")
    ap.add_argument("--drives", type=str, default="0.20,0.24,0.28",
                    help="comma-separated drive multipliers for --probe")
    a = ap.parse_args()
    if a.gate:
        print(json.dumps(run_fd_gate(a.shape), indent=1))
    if a.probe:
        drives = tuple(float(x) for x in a.drives.split(","))
        print(json.dumps(run_probe(a.shape, drives), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
