"""Deck figure: the SOLVED Tamper, uniform vs solved.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.render_deck_tamper

RENDERING ONLY. Every number is read from solve3d/phase_e/results/
phase_e_tamper.json and the saved field npz files; nothing is recomputed here,
so the figure cannot disagree with the record.

COORDINATES ARE REBUILT, NOT STORED. score_arm saves nodal fields (phi, T) but
not nodal coordinates. The mesh build is deterministic -- same STL, same lc,
same seed, same chamber -- so it is rebuilt here and its geometry read. A
shape check against the saved arrays is what makes that safe rather than
hopeful: if the rebuild ever stopped matching, the sizes would disagree and
this refuses instead of plotting one run's field on another run's mesh.

PLAN VIEW, THROUGH-THICKNESS MEAN, not a slice. The Tamper is a flanged
tamper -- a 44 mm flange over a ~25 mm stem -- so any single z-slice
misrepresents it (at mid-height you see only the stem). A thin band also has
too few nodes to interpolate and renders as triangulation facets rather than
physics. The through-thickness mean uses every node, is smooth without
smoothing, and is what a layer-wise printer acts on.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))

import matplotlib                                    # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.gridspec import GridSpec             # noqa: E402
from matplotlib.colors import ListedColormap         # noqa: E402

RESULTS = HERE / "results"
OUT = RESULTS / "fig_deck_tamper_solved.png"
CMAP_S = ListedColormap(plt.cm.viridis(np.linspace(0.18, 1.0, 256)))
CMAP_PHI = ListedColormap(plt.cm.inferno(np.linspace(0.06, 1.0, 256)))
DPI = 180


def _doc() -> dict:
    return json.loads((RESULTS / "phase_e_tamper.json").read_text())


def _rebuild():
    """The deterministic mesh the arms were scored on."""
    from solve3d import stl_mesh
    from solve3d.phase_e import run_tamper as rt
    msh, info = stl_mesh.build_mesh_from_stl(
        rt.TAMPER_STL, lc_part=rt.LC_PART_M, with_chamber=True, L=None)
    return msh, info


def _nodal_xyz(msh) -> np.ndarray:
    return np.asarray(msh.geometry.x)


def _phi_on_part_cells(msh, info, phi_nodal: np.ndarray) -> np.ndarray:
    """Nodal phi averaged onto the PART CELLS.

    The saved phi is nodal, and a plan view of it is too sparse to read: only
    2054 of 10221 nodes are in-part, which over a 44 mm disc leaves most bins
    empty and renders as speckle. The part has 20106 CELLS -- ten times the
    sampling -- and they are the cells the design map lives on, so both panels
    end up on the same support and can be compared square for square.
    """
    from solve3d import forward as fwd
    W = fwd.functionspace(msh, ("Lagrange", 1))
    cells = fwd.cell_average_p1(fwd.p1_cell_dofs(W), np.asarray(phi_nodal))
    return cells[np.asarray(info.part_cells, dtype=np.int64)]


def _plan_mean(ax, xyz, vals, cmap, vmin, vmax, title, mask=None,
               bin_mm=1.3):
    """PLAN VIEW: the through-thickness MEAN over the part, binned on (x, y).

    Not a single z-slice, and the reason is the part. The Tamper is a flanged
    tamper -- a 44 mm flange over a ~25 mm stem -- so ANY single slice
    misrepresents it: at mid-height you see only the stem, at the flange only
    the head. A thin band also leaves too few nodes to interpolate, which
    showed up as triangulation facets rather than physics.

    The through-thickness mean uses every node in the part, is smooth without
    smoothing, and is the quantity a layer-wise printer actually acts on. Bins
    with no part in them stay blank rather than being filled by extrapolation.
    """
    from scipy.stats import binned_statistic_2d
    m = np.ones(xyz.shape[0], dtype=bool) if mask is None else np.asarray(mask)
    x, y, v = xyz[m, 0] * 1e3, xyz[m, 1] * 1e3, np.asarray(vals)[m]
    r = float(max(np.abs(x).max(), np.abs(y).max())) * 1.02
    n = max(8, int(round(2.0 * r / float(bin_mm))))
    edges = np.linspace(-r, r, n + 1)
    G, _, _, _ = binned_statistic_2d(x, y, v, statistic="mean",
                                     bins=[edges, edges])
    im = ax.imshow(G.T, origin="lower", extent=[-r, r, -r, r], cmap=cmap,
                   vmin=vmin, vmax=vmax, interpolation="bilinear")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    return im


def main() -> int:
    doc = _doc()
    arms = doc["arms"]
    if "solve_filter_only" not in arms:
        print("solve arm not recorded yet; nothing to render")
        return 1
    uni, sol = arms["uniform_baseline"], arms["solve_filter_only"]
    mesh_rec = arms["_mesh"]

    fu = np.load(RESULTS / "field_tamper_uniform_baseline.npz")
    fs = np.load(RESULTS / "field_tamper_solve_filter_only.npz")
    msh, info = _rebuild()
    xyz = _nodal_xyz(msh)
    if xyz.shape[0] != fu["phi"].size:
        raise SystemExit(
            f"mesh rebuild does not match the saved fields "
            f"({xyz.shape[0]} nodes vs {fu['phi'].size}); refusing to plot "
            "one run's field on another run's mesh")
    ctr = fs["centroids"]
    if not np.allclose(ctr, fu["centroids"]):
        raise SystemExit("the two arms were scored on different cells")
    phi_u = _phi_on_part_cells(msh, info, fu["phi"])
    phi_s = _phi_on_part_cells(msh, info, fs["phi"])
    if phi_u.size != ctr.shape[0]:
        raise SystemExit(
            f"cell-averaged phi ({phi_u.size}) does not match the saved "
            f"centroids ({ctr.shape[0]}); refusing to plot a mismatched mesh")

    fig = plt.figure(figsize=(13.0, 7.4))
    gs = GridSpec(2, 3, figure=fig, height_ratios=[1.0, 1.0],
                  hspace=0.30, wspace=0.22)

    # --- top left: the solved dopant map ---------------------------------- #
    ax = fig.add_subplot(gs[0, 0])
    sc = _plan_mean(ax, ctr, fs["s_map"], CMAP_S, 0.0, 1.0,
                    "Solved dopant map\n(through-thickness mean)")
    ax.set_ylabel("y (mm)", fontsize=8)
    fig.colorbar(sc, ax=ax, fraction=0.046).set_label("saturation", size=8)

    # --- top middle/right: melt fraction, uniform vs solved ---------------- #
    for k, (pv, rec, lab) in enumerate(
            ((phi_u, uni, "Uniform doping"), (phi_s, sol, "Solved"))):
        ax = fig.add_subplot(gs[0, 1 + k])
        sc = _plan_mean(ax, ctr, pv, CMAP_PHI, 0.0, 1.0,
                        f"{lab}: melt fraction, through-thickness mean\n"
                        f"t_stop = {rec['t_stop_s']:.1f} s, "
                        f"part mean phi = {rec['part_mean_phi']:.3f}")
        if k == 1:
            fig.colorbar(sc, ax=ax, fraction=0.046).set_label("phi", size=8)

    # --- bottom: what it bought ------------------------------------------- #
    ax = fig.add_subplot(gs[1, :2])
    keys = [("J_asymmetric", "objective J", True),
            ("J_in_bounds_deficit", "in-part deficit", True),
            ("J_out_of_bounds", "out-of-part melt", True),
            ("out_of_part_melt_fraction_of_part", "bed melt / part vol", True),
            ("sigma_T_diagnostic_c", "sigma_T (C, diagnostic)", False)]
    labs, us, ss = [], [], []
    for key, lab, _lower in keys:
        labs.append(lab)
        us.append(float(uni[key]))
        ss.append(float(sol[key]))
    x = np.arange(len(labs))
    rel = [(s / u - 1.0) * 100.0 if u else np.nan for u, s in zip(us, ss)]
    cols = ["#3b7dd8" if r <= 0 else "#c0392b" for r in rel]
    ax.bar(x, rel, color=cols, width=0.55)
    ax.axhline(0.0, color="0.3", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labs, fontsize=8, rotation=12, ha="right")
    ax.set_ylabel("solved vs uniform (%)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title("Change from the uniform baseline "
                 "(negative is better for the first four)", fontsize=9)
    for xi, r in zip(x, rel):
        ax.annotate(f"{r:+.1f}%", (xi, r), ha="center", fontsize=7,
                    va="bottom" if r >= 0 else "top")

    # --- bottom right: the provenance block ------------------------------- #
    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    g = sol.get("standing_gates", {})
    lines = [
        "RUN",
        f"  part        Part Studio 1 - Tamper.stl",
        f"  run id      {mesh_rec.get('run_id', 'n/a')}",
        f"  chamber     {mesh_rec['chamber']['tag']} "
        f"({mesh_rec['L_chamber_m'] * 1e3:.0f} mm, {mesh_rec['chamber']['mode']})",
        f"  mesh        {mesh_rec['n_cells']} cells, "
        f"{mesh_rec['n_part_cells']} in part",
        f"  part vol    {mesh_rec['part_volume_rel_err_vs_stl']:+.2e} vs STL",
        f"  L0          xy {mesh_rec['precomp']['xy_scale']:.4f}, "
        f"z {mesh_rec['precomp']['z_scale']:.4f}",
        "",
        "SOLVE",
        f"  budget      {sol.get('gradient_evaluations_used')} gradient evals"
        f" (= 40 fwd-equiv)",
        f"  objective   asymmetric, envelope read",
        f"  J uniform   {uni['J_asymmetric']:.6e}",
        f"  J solved    {sol['J_asymmetric']:.6e}",
        f"  improvement {(sol['J_asymmetric'] / uni['J_asymmetric'] - 1) * 100:+.2f}%",
        "",
        "STANDING GATES (solved arm)",
        f"  physical    {g.get('forward_physical')}",
        f"  clamp       {g.get('clamp_bound')}",
        f"  cfl         {g.get('cfl_violated')}",
        f"  energy res  {g.get('energy_residual_frac'):.2e}",
        f"  peak T      {g.get('peak_T_c'):.1f} C "
        f"(ceiling {g.get('ceiling_c'):.0f}, over: {g.get('peak_over_ceiling')})",
    ]
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=7.2,
            family="monospace", transform=ax.transAxes)

    fig.suptitle("Solved dopant grading on an arbitrary user part "
                 "(sim-only, dolfinx forward)", fontsize=11)
    fig.savefig(OUT, dpi=DPI, bbox_inches="tight")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
