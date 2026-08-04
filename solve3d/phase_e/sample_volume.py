"""Sample a stored nodal read state onto a regular 3-D grid, for rendering.

    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.sample_volume --shape pyramid

WHY THIS EXISTS (data contract, checked before it was written). The Phase E
`field_<shape>_<arm>.npz` artifacts hold:

    T_read   (n_nodes,)   the FULL 3-D nodal temperature at the arm's envelope
                          read state -- a real volume, not sampled planes
    T_eval   (5, 200, 200) the same state on five evaluation planes
    s_map    (n_design,)  the dopant saturation the arm was run with

There is no stored phi volume, but none is needed: `forward.phase_fraction` is
POINTWISE in T (phi = clip((T - 180)/10 + 0.5, 0, 1)), which is exactly what
`gates.phase_fraction_phi` and the scorer use. So T_read determines a genuine
full 3-D melt-fraction volume, and a figure coloured by it is coloured by real
melt fraction -- not by a proxy and not by the dopant map.

No physics is run here. The mesh is rebuilt (fixed gmsh seed, deterministic)
only so the stored nodal vector can be evaluated at grid points; the design map
is transferred by nearest design cell, which is a read-out, not a resampling
that feeds any metric.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parent / "results"
DEFAULT_N = 96


def sample(shape: str, arms: tuple[str, ...], n: int = DEFAULT_N) -> Path:
    from solve3d import forward as fwd
    from solve3d.phase_e import geometry as geo, run as R

    half = (geo.PYR_H_M if shape == "pyramid" else geo.CUBE_A_M) / 2.0
    pad = 1.02 * half
    ax = np.linspace(-pad, pad, n)
    X, Y, Z = np.meshgrid(ax, ax, ax, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    inside = geo.in_part_predicate(shape)(pts.T).reshape(X.shape)

    tc = R.build_case(shape)
    W = fwd.functionspace(tc.msh, ("Lagrange", 1))
    out = {"axis_m": ax, "inside": inside, "shape": shape}

    for arm in arms:
        z = np.load(RESULTS / f"field_{shape}_{arm}.npz")
        Tf = fwd.fem.Function(W)
        Tf.x.array[:] = np.asarray(z["T_read"], float).astype(
            fwd.dolfinx.default_scalar_type)
        Te, n_missed = fwd.eval_at(Tf, tc.msh, pts)   # eval_at returns a COUNT
        Te = np.asarray(Te, float).reshape(X.shape)
        out[f"T__{arm}"] = Te
        out[f"missed__{arm}"] = int(n_missed)
        out[f"missed_in_part__{arm}"] = int(np.isnan(Te[inside]).sum())
        out[f"s_map__{arm}"] = np.asarray(z["s_map"], float)

    # the design map on the same grid, by nearest design cell (display only)
    from scipy.spatial import cKDTree
    import dolfinx
    cen = np.asarray(dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)))[tc.eqs.part]
    _, idx = cKDTree(cen).query(pts)
    out["design_nearest_index"] = idx.reshape(X.shape).astype(np.int32)

    from solve3d.phase_e import geometry as _g
    out["nominal_base_side_m"] = np.float64(
        _g.PYR_B_M if shape == "pyramid" else _g.CUBE_A_M)
    out["nominal_height_m"] = np.float64(
        _g.PYR_H_M if shape == "pyramid" else _g.CUBE_A_M)
    p = RESULTS / f"vol_{shape}.npz"
    np.savez_compressed(p, **out)
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--n", type=int, default=DEFAULT_N)
    ap.add_argument("--arms", default="uniform_baseline,solve_filter_only")
    a = ap.parse_args()
    p = sample(a.shape, tuple(a.arms.split(",")), a.n)
    z = np.load(p)
    print("wrote", p, "grid", z["inside"].shape,
          "in-part cells", int(z["inside"].sum()))
    for k in z.files:
        if k.startswith("missed"):
            print("  ", k, int(z[k]))
        if k.startswith("T__"):
            t = z[k][z["inside"]]
            print("  ", k, "in-part T range %.2f .. %.2f C" % (t.min(), t.max()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
