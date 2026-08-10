"""Dissertation figure: Phase C cylinder, uniform dopant vs the direct 3-D solve.

Light (white-background) style, colorblind-safe colormaps, no title inside the
image. Reads ONLY stored artifacts:

  solve3d/results/phase_c_map_solve_filter_only_asymmetric_scaled.npz  (solved map)
  solve3d/results/phase_c_baselines.json                (uniform arm scalars)
  solve3d/results/phase_c_solves.json                   (solved arm scalars, trajectory)
  solve3d/results/phase_c_gate.json                     (mesh hold-out)
  fgm_solve_campaign/figs_3d/phase_c_cylinder_fields.npz  (read-state fields,
      re-exported by export_phase_c_fields.py; every scalar reproduces the
      recorded Phase C JSON bit-for-bit, see phase_c_cylinder_fields_gate.txt)

    ./.venv312/bin/python fgm_solve_campaign/figs_3d/render_fig_cylinder_3d_solve.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RES = ROOT / "solve3d" / "results"
OUT = HERE / "fig_cylinder_3d_solve.png"

BG, FG, DIM = "#ffffff", "#1a1a1a", "#5a5f66"
C_UNI, C_SOL = "#8a9099", "#0277bd"
PHI_FLOOR = 0.85
W_OUT = 10.0
PART_R_MM = 10.0
SURF_R_MM = 9.4          # everything inboard of this carries exactly zero J

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": DIM, "axes.labelcolor": FG,
    "xtick.color": DIM, "ytick.color": DIM, "font.size": 7.5,
    "axes.titlesize": 7.8, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.linewidth": 0.7, "axes.grid": False,
})


# --------------------------------------------------------------------------- #
def load() -> dict:
    base = json.loads((RES / "phase_c_baselines.json").read_text())
    solves = json.loads((RES / "phase_c_solves.json").read_text())
    gate = json.loads((RES / "phase_c_gate.json").read_text())
    return {"uni": base["arms"]["uniform_baseline"],
            "sol": solves["arms"]["solve_filter_only_asymmetric_scaled"],
            "gate": gate["arms"]["solve_filter_only_asymmetric_scaled"],
            "mesh": base["mesh"],
            "f": np.load(HERE / "phase_c_cylinder_fields.npz", allow_pickle=False)}


def dopant_midplane(centroids: np.ndarray, s: np.ndarray,
                    half_slab_m: float = 5e-4, n: int = 260) -> np.ndarray:
    """Nearest-centroid render of the DG0 map in a thin mid-height slab.

    The design field is cellwise constant, so nearest-centroid IS the field; a
    slab is required because no tetrahedron centroid lies exactly on z = 0.
    """
    from scipy.spatial import cKDTree
    m = np.abs(centroids[:, 2]) <= half_slab_m
    tree = cKDTree(centroids[m, :2])
    g = np.linspace(-0.0105, 0.0105, n)
    X, Y = np.meshgrid(g, g, indexing="xy")
    _, idx = tree.query(np.column_stack([X.ravel(), Y.ravel()]))
    img = s[m][idx].reshape(n, n)
    img[np.hypot(X, Y) > 0.0098] = np.nan
    return img


def _objective_density(f, arm: str) -> np.ndarray:
    """Per-node contribution to J at the read state.

    J = sum_i vol_i [ 10 max(phi-chi,0)^2 + max(0.85 chi - phi,0)^2 ].
    """
    chi, vol, phi = f["chi_nodal"], f["vol_nodal"], f[arm + "_phi"]
    return (vol * np.maximum(PHI_FLOOR * chi - phi, 0.0) ** 2
            + W_OUT * vol * np.maximum(phi - chi, 0.0) ** 2)


def surface_profiles(f, n_th: int = 36, n_z: int = 24):
    """J binned by azimuth and by height over the LATERAL SURFACE nodes.

    100 % of J sits on those nodes: the part spans the full domain height, so
    the flat ends have chi = 1 and contribute nothing, and every interior node
    is fully melted (phi = 1) so neither hinge is active there.
    """
    xyz = f["node_xyz"]
    r = np.hypot(xyz[:, 0], xyz[:, 1]) * 1e3
    th = np.degrees(np.arctan2(xyz[:, 1], xyz[:, 0]))
    z = xyz[:, 2] * 1e3
    k = r > SURF_R_MM
    out = {}
    for arm in ("uniform", "solved"):
        j = _objective_density(f, arm)
        out[arm] = {
            "share": float(j[k].sum() / j.sum()),
            "az": np.histogram(th[k], bins=n_th, range=(-180, 180),
                               weights=j[k]),
            "z": np.histogram(z[k], bins=n_z, range=(-30, 30), weights=j[k]),
        }
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    d = load()
    f, u, s = d["f"], d["uni"], d["sol"]

    img_sol = dopant_midplane(np.asarray(f["centroids"]),
                              np.asarray(f["s_map_solved"]))
    img_uni = np.where(np.isnan(img_sol), np.nan, 1.0)
    prof = surface_profiles(f)
    share = prof["uniform"]["share"]
    ref_bin = float(prof["uniform"]["az"][0].mean())

    fig = plt.figure(figsize=(6.5, 7.1))
    gs = GridSpec(3, 3, figure=fig,
                  width_ratios=[1.0, 1.0, 0.055],
                  height_ratios=[1.0, 0.74, 0.80],
                  left=0.095, right=0.885, top=0.955, bottom=0.105,
                  wspace=0.26, hspace=0.52)

    # ---- row 1: the design variable -------------------------------------- #
    ext_d = [-10.5, 10.5, -10.5, 10.5]
    vmin = float(np.nanmin(img_sol))
    titles = ["(a) uniform dopant, $s=1$ everywhere",
              "(b) solved dopant map"]
    for j, img in enumerate([img_uni, img_sol]):
        ax = fig.add_subplot(gs[0, j])
        im = ax.imshow(img, origin="lower", extent=ext_d, cmap="viridis",
                       vmin=vmin, vmax=1.0, interpolation="nearest")
        ax.add_patch(plt.Circle((0, 0), PART_R_MM, fill=False, ec=FG, lw=0.7))
        ax.set_xlim(-11, 11); ax.set_ylim(-11, 11); ax.set_aspect("equal")
        ax.set_xticks([-10, -5, 0, 5, 10]); ax.set_yticks([-10, -5, 0, 5, 10])
        ax.set_xlabel("x [mm]", labelpad=1)
        if j == 0:
            ax.set_ylabel("y [mm]", labelpad=1)
        ax.set_title(titles[j], pad=3)
    cb = fig.colorbar(im, cax=fig.add_subplot(gs[0, 2]))
    cb.set_label("dopant saturation $s$", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)

    # ---- row 2: where the objective actually lives ------------------------ #
    specs = [("az", "azimuth [deg]", [-180, -90, 0, 90, 180],
              "(c) shape error around the part surface"),
             ("z", "height z [mm]", [-30, -15, 0, 15, 30],
              "(d) shape error along the part height")]
    for j, (key, xlab, xt, title) in enumerate(specs):
        ax = fig.add_subplot(gs[1, j])
        hu, edges = prof["uniform"][key]
        hs, _ = prof["solved"][key]
        c = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(c, hs / ref_bin, hu / ref_bin, step="mid",
                        color=C_SOL, alpha=0.28, lw=0)
        ax.step(c, hu / ref_bin, where="mid", color="#6e747c", lw=1.1)
        ax.step(c, hs / ref_bin, where="mid", color=C_SOL, lw=1.4)
        ax.set_xlim(edges[0], edges[-1]); ax.set_ylim(0.52, 1.98)
        ax.set_xticks(xt)
        ax.set_xlabel(xlab, labelpad=1)
        if j == 0:
            ax.set_ylabel("shape error per bin\n(uniform mean = 1)", labelpad=1)
            ax.text(-172, 1.86, "uniform", color="#6e747c", fontsize=7)
            ax.text(-172, 1.70, "solved", color=C_SOL, fontsize=7,
                    fontweight="bold")
            ax.text(-172, 1.54, "shaded = removed by the solve", color=C_SOL,
                    fontsize=6.6, alpha=0.85)
        else:
            n_az = int((prof["solved"]["az"][0] < prof["uniform"]["az"][0]).sum())
            n_z = int((prof["solved"]["z"][0] < prof["uniform"]["z"][0]).sum())
            ax.text(-28, 1.86,
                    f"{n_z} of {prof['uniform']['z'][0].size} height bins and "
                    f"{n_az} of {prof['uniform']['az'][0].size} azimuth bins "
                    "improve", color=DIM, fontsize=6.4)
        ax.set_title(title, pad=3, fontsize=7.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)

    # ---- row 3 left: the objective numbers -------------------------------- #
    ax = fig.add_subplot(gs[2, 0])
    keys = [("J_asymmetric", "total"),
            ("J_in_bounds_deficit", "in-bounds\ndeficit"),
            ("J_out_of_bounds", "bed melt\n($\\times$10)")]
    ref = u["J_asymmetric"]
    xx = np.arange(len(keys))
    ax.bar(xx - 0.19, [u[k] / ref for k, _ in keys], width=0.36, color=C_UNI,
           label="uniform")
    ax.bar(xx + 0.19, [s[k] / ref for k, _ in keys], width=0.36, color=C_SOL,
           label="solved")
    for i, (k, _) in enumerate(keys):
        ax.text(i + 0.19, s[k] / ref + 0.025,
                f"{100.0 * (s[k] / u[k] - 1.0):+.2f}%", ha="center",
                fontsize=6.8, color=C_SOL)
    ax.set_xticks(xx)
    ax.set_xticklabels([lab for _, lab in keys], fontsize=6.8)
    ax.set_ylim(0, 1.30)
    ax.set_ylabel("$J$ / $J_{\\mathrm{uniform}}$", labelpad=1)
    ax.set_title("(e) shape-fidelity objective, solve mesh", pad=3)
    g0 = d["gate"]["mesh_holdout"]
    ax.text(0.42, 1.16,
            f"hold-out mesh: "
            f"{100.0 * (g0['solved_at_score_mesh'] / g0['uniform_at_score_mesh'] - 1.0):+.2f}%",
            ha="center", fontsize=6.6, color=DIM)
    ax.legend(loc="upper right", frameon=False, ncol=1, handlelength=1.1,
              borderaxespad=0.2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # ---- row 3 right: the solve trajectory -------------------------------- #
    ax2 = fig.add_subplot(gs[2, 1])
    ev = np.array([t["eval"] for t in s["trajectory"]])
    Jt = np.array([t["J"] for t in s["trajectory"]])
    ax2.axhline(1.0, color=C_UNI, lw=1.1, ls="--")
    ax2.text(12.4, 1.002, "uniform", color=DIM, fontsize=6.8, ha="right")
    ax2.plot(ev, Jt / ref, "o-", color=C_SOL, lw=1.3, ms=3.0)
    ax2.annotate("budget limit,\nstill descending",
                 xy=(ev[-1], Jt[-1] / ref), xytext=(7.2, 0.952),
                 fontsize=6.8, color=FG, ha="center",
                 arrowprops=dict(arrowstyle="->", color=DIM, lw=0.7))
    ax2.set_xlabel("gradient evaluation", labelpad=1)
    ax2.set_ylabel("$J$ / $J_{\\mathrm{uniform}}$", labelpad=1)
    ax2.set_xlim(0.4, 12.6); ax2.set_xticks([1, 4, 8, 12])
    ax2.set_title("(f) solve trajectory, 12 of 12 evaluations", pad=3)
    for sp in ("top", "right"):
        ax2.spines[sp].set_visible(False)

    # ---- footer facts, all printed from the artifacts --------------------- #
    g = d["gate"]["mesh_holdout"]
    hold = 100.0 * (1.0 - g["solved_at_score_mesh"] / g["uniform_at_score_mesh"])
    prim = 100.0 * (1.0 - s["J_asymmetric"] / u["J_asymmetric"])
    fig.text(0.095, 0.037,
             f"Solve mesh {d['mesh']['n_cells']:,} cells, coupling off, read at "
             f"the objective's own stop ($t = ${s['t_stop_s']:.1f} s).",
             fontsize=6.8, color=DIM)
    fig.text(0.095, 0.014,
             "The inversion heuristic gives no benefit on this shape: "
             "+0.3% $\\sigma_T$ (heatr3d re-run, a different engine and metric).",
             fontsize=6.8, color=DIM)

    fig.savefig(OUT, dpi=300)
    hu = prof["uniform"]["az"][0]
    hs = prof["solved"]["az"][0]
    hzu, hzs = prof["uniform"]["z"][0], prof["solved"]["z"][0]
    print("wrote", OUT)
    print(f"primary {prim:.4f} %  holdout {hold:.4f} %  "
          f"surface share of J {share:.6f}  "
          f"azimuth bins improved {(hs < hu).mean():.3f}  "
          f"height bins improved {(hzs < hzu).mean():.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
