"""Export the Phase C cylinder read-state fields for the dissertation figure.

RENDERING SUPPORT ONLY. No solve is re-run and no solve3d solver code is
modified. This runs ONE forward evaluation per arm on the STORED maps
(uniform ones; solve3d/results/phase_c_map_solve_filter_only_asymmetric_scaled.npz)
because Phase C's scorer recorded scalars but never wrote the nodal read state.

GATE: every re-computed scalar must reproduce the recorded Phase C JSON to
rtol 1e-9; the script raises if it does not, so a drifted environment cannot
silently produce a figure that disagrees with PHASE_C_REPORT.md.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      heatr3d_d1_spike/env/bin/python -m fgm_solve_campaign.figs_3d.export_phase_c_fields
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from solve3d import forward as fwd, gates, objective as obj, phase_c_run as pcr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "solve3d" / "results"
OUT = Path(__file__).resolve().parent
RTOL = 1e-9


def _read_state(tc, s_map: np.ndarray) -> dict:
    """Reproduce phase_c_run.score_arm's READ RULE exactly and keep the field."""
    t0 = time.perf_counter()
    tr = tc.forward(tc.design_to_sigma(s_map))
    wall = time.perf_counter() - t0
    tc.set_objective("symmetric")
    Js = tc.J_trajectory(tr)
    k = int(np.argmin(Js))
    tc.set_objective("asymmetric")
    Ja = tc.J_trajectory(tr)
    ka = int(np.argmin(Ja))
    T_read = tc.state_at(tr, ka)
    phi = fwd.phase_fraction(T_read, tc.p)[0]
    vol, chi = tc.vol_nodal, tc.m_nodal
    split = obj.split_asymmetric(phi, chi, vol)

    pts, shp, h = gates.eval_grid_points()
    W = fwd.functionspace(tc.msh, ("Lagrange", 1))
    Tf = fwd.fem.Function(W)
    Tf.x.array[:] = T_read.astype(fwd.dolfinx.default_scalar_type)
    Te, missed = fwd.eval_at(Tf, tc.msh, pts)
    Te = Te.reshape(shp)

    return {
        "J_symmetric": float(Js[k]), "argmin_symmetric": k,
        "J_asymmetric": float(Ja[ka]), "argmin_asymmetric": ka,
        "t_stop_s": float(ka * tc.p.dt_s),
        "J_out_of_bounds": split["J_out_of_bounds"],
        "J_in_bounds_deficit": split["J_in_bounds_deficit"],
        "out_of_part_melt_fraction_of_part":
            split["out_of_bounds_melt_fraction_of_part"],
        "in_bounds_below_floor_fraction": split["in_bounds_below_floor_fraction"],
        "wall_forward_s": wall, "eval_missed": int(missed),
        "_T_read": T_read, "_phi": phi, "_T_eval": Te,
    }


def _check(name: str, got: dict, recorded: dict) -> list[str]:
    lines = []
    for key in ("J_symmetric", "J_asymmetric", "argmin_symmetric",
                "argmin_asymmetric", "J_out_of_bounds", "J_in_bounds_deficit",
                "out_of_part_melt_fraction_of_part",
                "in_bounds_below_floor_fraction"):
        a, b = float(got[key]), float(recorded[key])
        rel = abs(a - b) / max(abs(b), 1e-300)
        ok = rel <= RTOL
        lines.append(f"  {name:<8s} {key:<36s} {a!r} vs {b!r}  rel={rel:.3e} "
                     f"{'OK' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit(f"REPRODUCTION GATE FAILED: {name}/{key} rel={rel:.3e}\n"
                             + "\n".join(lines))
    return lines


def main() -> int:
    base = json.loads((RESULTS / "phase_c_baselines.json").read_text())
    solves = json.loads((RESULTS / "phase_c_solves.json").read_text())
    rec = {"uniform": base["arms"]["uniform_baseline"],
           "solved": solves["arms"]["solve_filter_only_asymmetric_scaled"]}

    tc = pcr.build_solve_case()
    npart = int(tc.eqs.part.size)
    z = np.load(RESULTS / "phase_c_map_solve_filter_only_asymmetric_scaled.npz")
    maps = {"uniform": np.ones(npart), "solved": np.asarray(z["s_map"], float)}

    report, payload = [], {}
    for name, s_map in maps.items():
        got = _read_state(tc, s_map)
        report += _check(name, got, rec[name])
        payload[f"{name}_T_read"] = got.pop("_T_read")
        payload[f"{name}_phi"] = got.pop("_phi")
        payload[f"{name}_T_eval"] = got.pop("_T_eval")
        payload[f"{name}_scalars"] = json.dumps(got)
        print(f"[{name}] forward {got['wall_forward_s']:.1f} s, "
              f"J_asym {got['J_asymmetric']:.6e}, argmin {got['argmin_asymmetric']}")

    x, y, zc, h = gates.eval_grid_axes()
    payload.update(
        eval_x=x, eval_y=y, eval_z=zc, eval_h=h,
        part_mask_xy=gates.nominal_part_mask("circle"),
        node_xyz=np.asarray(tc.msh.geometry.x, float),
        chi_nodal=np.asarray(tc.m_nodal, float),
        vol_nodal=np.asarray(tc.vol_nodal, float),
        s_map_solved=maps["solved"],
        centroids=np.asarray(z["centroids"], float),
        cell_volumes=np.asarray(z["volumes"], float),
    )
    np.savez_compressed(OUT / "phase_c_cylinder_fields.npz", **payload)
    (OUT / "phase_c_cylinder_fields_gate.txt").write_text(
        "Phase C field re-export reproduction gate (rtol %g)\n" % RTOL
        + "\n".join(report) + "\nALL PASS\n")
    print("\n".join(report))
    print("wrote", OUT / "phase_c_cylinder_fields.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
