"""Mean-shift control: score a SPATIALLY UNIFORM map at the symmetrized map's
own volume-weighted mean, on both meshes.

CAMPAIGN SCRIPT. No solve3d solver module is modified. This separates the two
things the symmetrized arm's margin could be made of:

  LEVEL    simply putting less dopant everywhere (mean 0.9334 instead of 1.0),
  SHAPE    the spatial structure on top of that level.

The control is the level with no shape. Whatever it wins is not attributable to
the design's structure. It is scored through the EXACT same read rule as every
other arm (phase_c_run.score_arm: envelope argmin of the asymmetric objective).

On the hold-out mesh the constant is used directly rather than transferred,
because phase_c_run.transfer_map_across_meshes is a row-normalized convolution
and therefore reproduces a constant EXACTLY. That is asserted numerically here
on a random subset of destination cells rather than taken on trust.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH="$PWD" \
      heatr3d_d1_spike/env/bin/python \
      scripts/analysis/score_mean_shift_control.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from solve3d import phase_c_run as pcr

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "solve3d" / "results"
SYM_ARM = "symmetrized_filter_only_asymmetric_scaled"
OUT = ROOT / "scripts" / "analysis" / "mean_shift_control.json"


def _ckpt(doc: dict, stage: str, payload) -> None:
    doc[stage] = {"t": time.strftime("%Y-%m-%d %H:%M:%S"), "payload": payload}
    OUT.write_text(json.dumps(doc, indent=1, default=str))
    print(f"[ckpt] {stage}", flush=True)


def main() -> int:
    z = np.load(RESULTS / f"phase_c_map_{SYM_ARM}.npz")
    s_sym = np.asarray(z["s_map"], float)
    vols = np.asarray(z["volumes"], float)
    s_bar = float(np.average(s_sym, weights=vols))

    doc = {"what": "mean-shift control for the Phase C symmetrized cylinder map",
           "level": s_bar,
           "level_definition": "volume-weighted mean of "
                               f"phase_c_map_{SYM_ARM}.npz s_map over part cells"}
    _ckpt(doc, "level", {"s_bar": s_bar})

    # ---------------- solve mesh ---------------- #
    tc = pcr.build_solve_case()
    n = int(tc.eqs.part.size)
    rec_c = pcr.score_arm(tc, np.full(n, s_bar), "mean_shift_control_solve_mesh")
    _ckpt(doc, "solve_mesh", rec_c)

    # ---------------- hold-out mesh ---------------- #
    tc_f = pcr.build_solve_case(pcr.SCORE_MESH)
    n_f = int(tc_f.eqs.part.size)

    # measurement, not assumption: the cross-mesh transfer reproduces a constant
    import dolfinx
    mp_f = np.asarray(dolfinx.mesh.compute_midpoints(
        tc_f.msh, tc_f.msh.topology.dim,
        np.arange(tc_f.ncells, dtype=np.int32)))[tc_f.eqs.part]
    rng = np.random.default_rng(0)
    sub = rng.choice(n_f, size=min(2000, n_f), replace=False)
    tr = pcr.transfer_map_across_meshes(
        np.asarray(z["centroids"], float), vols, np.full(s_sym.size, s_bar),
        mp_f[sub], float(pcr.prereg()["design_chain"]["filter_radius_m"]))
    err = float(np.max(np.abs(tr["map"] - s_bar)))
    _ckpt(doc, "transfer_constant_check",
          {"n_checked": int(sub.size), "max_abs_dev_from_constant": err,
           "verdict": "constant reproduced exactly" if err < 1e-12
                      else "NOT exact, investigate"})
    if err >= 1e-9:
        raise SystemExit(f"transfer does not reproduce a constant: {err:.3e}")

    rec_f = pcr.score_arm(tc_f, np.full(n_f, s_bar),
                          "mean_shift_control_score_mesh")
    _ckpt(doc, "score_mesh", rec_f)

    # ---------------- the split ---------------- #
    base = json.loads((RESULTS / "phase_c_baselines.json").read_text())["arms"]
    gate = json.loads((RESULTS / "phase_c_gate.json").read_text())["arms"][SYM_ARM]
    h = gate["mesh_holdout"]
    split = {}
    for mesh, uni, ctrl, sym in (
            ("solve_mesh", base["uniform_baseline"]["J_asymmetric"],
             rec_c["J_asymmetric"], h["solved_at_solve_mesh"]),
            ("score_mesh", h["uniform_at_score_mesh"],
             rec_f["J_asymmetric"], h["solved_at_score_mesh"])):
        tot = (uni - sym) / uni
        lvl = (uni - ctrl) / uni
        split[mesh] = {
            "J_uniform_s1": uni, "J_uniform_at_level": ctrl,
            "J_symmetrized": sym,
            "total_margin_rel": tot, "level_margin_rel": lvl,
            "shape_margin_rel": tot - lvl,
            "level_share_of_total": lvl / tot if tot else float("nan"),
            "shape_share_of_total": (tot - lvl) / tot if tot else float("nan"),
        }
    _ckpt(doc, "split", split)
    print(json.dumps(split, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
