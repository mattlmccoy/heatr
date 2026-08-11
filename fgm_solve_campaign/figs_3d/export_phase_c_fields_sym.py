"""Export the read-state fields for the SYMMETRIZED Phase C cylinder map.

RENDERING SUPPORT ONLY. Same read rule and same reproduction gate as
export_phase_c_fields.py, but for the symmetrized arm. The uniform arm's fields
are copied from the existing phase_c_cylinder_fields.npz (same mesh, same case,
same map of ones), so this costs ONE forward, not two.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" \
      heatr3d_d1_spike/env/bin/python \
      fgm_solve_campaign/figs_3d/export_phase_c_fields_sym.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solve3d import phase_c_run as pcr
from fgm_solve_campaign.figs_3d.export_phase_c_fields import _read_state, _check

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "solve3d" / "results"
OUT = Path(__file__).resolve().parent
ARM = "symmetrized_filter_only_asymmetric_scaled"


def main() -> int:
    rec = json.loads((RESULTS / "phase_c_solves.json").read_text())["arms"][ARM]
    z = np.load(RESULTS / f"phase_c_map_{ARM}.npz")
    s_map = np.asarray(z["s_map"], float)

    tc = pcr.build_solve_case()
    got = _read_state(tc, s_map)
    report = _check("sym", got, rec)

    old = np.load(OUT / "phase_c_cylinder_fields.npz", allow_pickle=False)
    payload = {k: old[k] for k in ("uniform_T_read", "uniform_phi",
                                   "uniform_T_eval", "uniform_scalars",
                                   "eval_x", "eval_y", "eval_z", "eval_h",
                                   "part_mask_xy", "node_xyz", "chi_nodal",
                                   "vol_nodal", "cell_volumes", "centroids")}
    payload["solved_T_read"] = got.pop("_T_read")
    payload["solved_phi"] = got.pop("_phi")
    payload["solved_T_eval"] = got.pop("_T_eval")
    payload["solved_scalars"] = json.dumps(got)
    payload["s_map_solved"] = s_map
    payload["s_map_presymmetrization"] = np.asarray(z["s_map_source"], float)

    np.savez_compressed(OUT / "phase_c_cylinder_fields_sym.npz", **payload)
    (OUT / "phase_c_cylinder_fields_sym_gate.txt").write_text(
        "Phase C SYMMETRIZED-arm field export reproduction gate (rtol 1e-9)\n"
        + "\n".join(report) + "\nALL PASS\n")
    print("\n".join(report))
    print("wrote", OUT / "phase_c_cylinder_fields_sym.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
