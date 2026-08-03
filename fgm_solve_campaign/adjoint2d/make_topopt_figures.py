"""Figures for the topology-optimization pass. Five, each saying one thing.

  fig_topopt_parameterization  what the parameterization IS: the projection
      ladder, the physical filter on two grids, and the grid-independent target
      against the binary raster it replaces
  fig_topopt_gate              the finite-difference gate, all layers, all
      probes, the V curves and the transpose exactness
  fig_topopt_continuation      the beta continuation: objective and
      non-discreteness against evaluation, stage by stage
  fig_topopt_maps              the delivered maps and what melts
  fig_topopt_acceptance        the two acceptance gates, against the multi-start
      filtered arms and the unfiltered originals
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import chi_area, topopt
from .pins import build_case, load_cfg
from . import library_solve as lib

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out_topopt"
OUT_MS = ROOT / "out_ms"
OUT_ROBUST = ROOT / "out_robust"
FIGS = ROOT / "figs_topopt"
DPI = 180
SHAPES = ("square", "circle", "trapezoid", "triangle", "diamond", "rectangle")
BETA_COLOR = {1.0: "#4C72B0", 2.0: "#55A868", 4.0: "#DD8452",
              8.0: "#C44E52", 16.0: "#8172B3", 0.0: "#777777"}


def _load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def collect() -> list[dict]:
    rows = []
    for sh in SHAPES:
        rows.append({"shape": sh,
                     "solve": _load(OUT / f"{sh}.json"),
                     "ctl": _load(OUT / f"{sh}_control_filteronly.json"),
                     "rb": _load(OUT / f"{sh}_robust.json"),
                     "ctl_rb": _load(OUT / f"{sh}_control_filteronly_robust.json"),
                     "ms": _load(OUT_MS / f"{sh}.json"),
                     "ms_rb": _load(OUT_MS / f"{sh}_robust.json"),
                     "rb_old": _load(OUT_ROBUST / f"{sh}_rim.json"),
                     "grid_old": _load(OUT_ROBUST / f"{sh}_grid.json")})
    return rows


# ---------------------------------------------------------------------------

def fig_parameterization() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    ax = axes[0, 0]
    u = np.linspace(0, 1, 501)
    for b in topopt.BETA_SCHEDULE:
        ax.plot(u, topopt.project(u, b), lw=2.0, color=BETA_COLOR[b],
                label=f"beta = {b:g}")
    ax.plot(u, u, "k--", lw=1.0, label="beta = 0 (identity)")
    ax.axvline(topopt.ETA, color="0.5", lw=0.8)
    ax.set_xlabel("filtered design v_f")
    ax.set_ylabel("physical saturation s")
    ax.set_title("A  smoothed-Heaviside projection, threshold eta = 0.5\n"
                 "the frozen continuation ladder", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for n, style in ((120, "-"), (160, "--")):
        dxg = 0.06 / (n - 1)
        sig = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, dxg)
        r_mm = np.arange(-8, 9) * dxg * 1e3
        k = np.exp(-0.5 * (np.arange(-8, 9) / sig) ** 2)
        k /= k.sum()
        ax.plot(r_mm, k, style, marker="o", ms=3.5, lw=1.6,
                label=f"grid {n}: dx = {dxg*1e3:.3f} mm, sigma = {sig:.3f} cells")
    ax.axvline(topopt.FILTER_RADIUS_M * 1e3, color="crimson", lw=1.2)
    ax.axvline(-topopt.FILTER_RADIUS_M * 1e3, color="crimson", lw=1.2,
               label=f"radius {topopt.FILTER_RADIUS_M*1e3:.2f} mm (frozen)")
    ax.set_xlabel("distance from the cell, mm")
    ax.set_ylabel("filter weight per cell (each curve sums to 1)")
    ax.set_title("B  the radius is a LENGTH, so the same physical feature\n"
                 "is filtered on both grids", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    cfg = load_cfg(lib.shape_config("circle"))
    case = build_case(cfg)
    chi, _ = chi_area.chi_from_cfg(cfg, case.x, case.y)
    ax = axes[1, 0]
    # A 45-degree arc of the circle, where the raster staircase is worst.
    i0 = int(np.argmin(np.abs(case.x - 0.00707)))
    j0 = int(np.argmin(np.abs(case.y - 0.00707)))
    sl = slice(j0 - 6, j0 + 6), slice(i0 - 6, i0 + 6)
    ax.imshow(chi[sl], cmap="viridis", vmin=0, vmax=1, origin="lower",
              interpolation="nearest")
    for (j, i), val in np.ndenumerate(chi[sl]):
        ax.text(i, j, f"{val:.2f}", ha="center", va="center", fontsize=6.0,
                color="w" if val < 0.55 else "k")
    ax.contour(case.part_mask[sl].astype(float), levels=[0.5], colors="crimson",
               linewidths=2.0, origin="lower")
    d = float(np.sum(np.abs(case.part_mask[sl].astype(float) - chi[sl])))
    ax.set_title("C  circle, 45-degree arc at grid 120. Colour and numbers are the\n"
                 f"area-fill target chi; red is the binary raster it replaces. The two\n"
                 f"targets disagree by {d:.2f} cells of area in this 12 by 12 window.",
                 fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1, 1]
    grids = (80, 100, 120, 140, 160, 200)
    exact = np.pi * (0.010 ** 2)
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    poly = np.column_stack([0.010 * np.cos(t), 0.010 * np.sin(t)])
    e_chi, e_bin = [], []
    for g in grids:
        xg = np.linspace(-0.03, 0.03, g)
        dA = (xg[1] - xg[0]) ** 2
        e_chi.append(100.0 * (chi_area.area_fill_poly(poly, xg, xg).sum() * dA - exact) / exact)
        e_bin.append(100.0 * (chi_area.area_fill_poly(poly, xg, xg, n_sub=1).sum() * dA
                              - exact) / exact)
    ax.plot(grids, e_bin, "o-", color="#C44E52", label="binary raster target")
    ax.plot(grids, e_chi, "s-", color="#4C72B0", label="area-fill target chi")
    ax.axhline(0.0, color="k", lw=0.8)
    for g in (120, 160):
        ax.axvline(g, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel("grid cells per side")
    ax.set_ylabel("target area error against the closed form, percent")
    ax.set_title("D  a 10 mm circle: the raster target moves with the grid,\n"
                 "the area-fill target does not", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = FIGS / "fig_topopt_parameterization.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_gate() -> Path:
    gates = [(s, _load(OUT / f"gate_topopt_{s}.json")) for s in ("square", "circle")]
    gates = [(s, g) for s, g in gates if g]
    if not gates:
        return None
    nl = max(len(g["layers"]) for _s, g in gates)
    nr = len(gates)
    fig = plt.figure(figsize=(4.0 * nl, 4.2 * nr + 4.6))
    gs = fig.add_gridspec(nr + 1, nl, height_ratios=[1] * nr + [1.1])
    for r, (sh, g) in enumerate(gates):
        for c, layer in enumerate(g["layers"]):
            ax = fig.add_subplot(gs[r, c])
            for pname, pr in layer["probes"].items():
                eps = [row["eps"] for row in pr["sweep"]]
                err = [max(row["rel_err"], 1e-16) for row in pr["sweep"]]
                ax.loglog(eps, err, "o-", ms=3.2, lw=1.2, label=pname)
            ax.axhline(1e-6, color="green", lw=1.0, ls="--")
            ax.axhline(1e-5, color="orange", lw=1.0, ls=":")
            ax.set_xlabel("epsilon")
            if c == 0:
                ax.set_ylabel(f"{sh}\nrelative error")
            ax.set_title(f"{layer['layer']}\n{layer['n_probes_pass_1e-6']}/"
                         f"{layer['n_probes']} at 1e-6, "
                         f"{layer['n_probes_pass_1e-5']}/{layer['n_probes']} at 1e-5",
                         fontsize=9)
            ax.grid(alpha=0.3, which="both")
            if r == 0 and c == 0:
                ax.legend(fontsize=6.5, loc="lower left")

    # The diagnosis panel: the ABSOLUTE error is flat, so every large relative
    # error is a small denominator and not a wrong gradient.
    ax = fig.add_subplot(gs[nr, :])
    mk = {"P1_target_only_dJds": "o", "P2_filter_only_dJdv": "s",
          "P3_projection_beta1_dJdv": "^", "P4_projection_beta16_dJdv": "D"}
    col = {"square": "#4C72B0", "circle": "#DD8452"}
    for sh, g in gates:
        for layer in g["layers"]:
            for pname, pr in layer["probes"].items():
                ax.loglog(abs(pr["analytic"]), pr["best_abs_err"],
                          mk.get(layer["layer"], "o"), color=col.get(sh, "k"),
                          ms=6, alpha=0.8, mfc="none" if sh == "circle" else None)
    a = np.logspace(-2, 3, 50)
    ax.loglog(a, 1e-6 * a, "g--", lw=1.2, label="relative error 1e-6")
    ax.loglog(a, 1e-5 * a, color="orange", ls=":", lw=1.4,
              label="relative error 1e-5, the campaign subgradient standard")
    ax.axhspan(1e-6, 1e-4, color="0.85", alpha=0.5,
               label="the measured absolute-error band of this forward")
    ax.set_xlabel("magnitude of the analytic directional derivative")
    ax.set_ylabel("best absolute finite-difference error")
    ax.set_title("Diagnosis: the absolute error is flat across every layer and both "
                 "shapes, between 4e-07 and 1.3e-04.\nA probe fails the relative "
                 "standard when its own derivative is small, not when the gradient "
                 "is wrong. Filled = square, open = circle; "
                 "o P1, s P2, triangle P3, diamond P4.", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    tc = "; ".join(f"{sh}: max {max(t['max_rel_err'] for t in g['transpose_consistency'].values()):.2e}"
                   for sh, g in gates)
    fig.suptitle("Finite-difference gate on the composed topology-optimization "
                 "gradient. Green 1e-6, orange the 1e-5 subgradient standard.\n"
                 f"Chain-rule transpose exactness (dot-product identity), worst over "
                 f"beta 0, 1 and 16 -- {tc}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIGS / "fig_topopt_gate.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_continuation(data) -> Path:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.0))
    for k, d in enumerate(data):
        ax = axes[k // 3][k % 3]
        j = d["solve"]
        if j is None:
            ax.axis("off")
            continue
        rows = j["rows"]
        xs = [r["eval_index"] for r in rows]
        ax.plot(xs, [r["J"] for r in rows], "o-", color="#333333", ms=4, lw=1.4,
                label="continuation, J")
        b0 = None
        for r in rows:
            if r["beta"] != b0:
                ax.axvline(r["eval_index"] - 0.5, color=BETA_COLOR[r["beta"]],
                           lw=1.2, alpha=0.8)
                ax.text(r["eval_index"] - 0.4, ax.get_ylim()[1],
                        f"b={r['beta']:g}", fontsize=7,
                        color=BETA_COLOR[r["beta"]], va="top")
                b0 = r["beta"]
        if d["ctl"]:
            cr = d["ctl"]["rows"]
            ax.plot([r["eval_index"] for r in cr], [r["J"] for r in cr], "s--",
                    color="#DD8452", ms=3.5, lw=1.2,
                    label="control, beta 0 at full budget")
        ax.axhline(j["arms"]["U_uniform"]["J"], color="#4C72B0", lw=1.0, ls=":",
                   label="uniform")
        ax2 = ax.twinx()
        ax2.plot(xs, [r["non_discreteness"] for r in rows], "^-", color="#55A868",
                 ms=3.2, lw=1.0, alpha=0.75)
        ax2.set_ylabel("non-discreteness M_nd", color="#55A868", fontsize=8)
        ax2.tick_params(axis="y", labelcolor="#55A868", labelsize=7)
        ax2.set_ylim(0, 1)
        ax.set_title(d["shape"], fontsize=10)
        ax.set_xlabel("gradient evaluation")
        ax.set_ylabel("J against the area-fill target")
        ax.grid(alpha=0.3)
        if k == 0:
            ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("Beta continuation at a 40 forward-equivalent budget. Vertical "
                 "lines are stage boundaries; green is how binary the map is "
                 "(0 = fully binary).", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    p = FIGS / "fig_topopt_continuation.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_maps(data) -> Path:
    """Row 1 the delivered map, row 2 what melts, row 3 where the error is.

    Row 2 and 3 re-run the forward once per shape at the delivered map, which is
    six forward runs and is the only place in this figure set that computes
    physics.
    """
    from . import topopt_objective as tobj
    from . import topopt_solve as tos
    fig, axes = plt.subplots(3, len(data), figsize=(3.1 * len(data), 9.4))
    im0 = im2 = None
    for k, d in enumerate(data):
        npz = OUT / f"{d['shape']}_maps.npz"
        if d["solve"] is None or not npz.exists():
            for r in range(3):
                axes[r][k].axis("off")
            continue
        with np.load(npz) as z:
            s = np.asarray(z["TO_4bpp"], dtype=float)
            chi = np.asarray(z["chi_area"], dtype=float)
            pm = np.asarray(z["part_mask"]).astype(bool)
        j = d["solve"]
        case = build_case(load_cfg(lib.shape_config(d["shape"])))
        tr = tos.run_forward(case, s)
        st = tobj.optimal_stop(tr, case, chi)
        phi = tobj.phi_of(tr.T_at_end(st.index), case)
        del tr

        im0 = axes[0][k].imshow(np.where(pm, s, np.nan), cmap="magma", vmin=0, vmax=1,
                                origin="lower", interpolation="bilinear")
        axes[0][k].set_title(f"{d['shape']}\n4-bpp map, M_nd "
                             f"{j['arms']['TO_4bpp']['non_discreteness']:.2f}",
                             fontsize=9)
        axes[1][k].imshow(phi, cmap="inferno", vmin=0, vmax=1, origin="lower",
                          interpolation="bilinear")
        axes[1][k].contour(chi, levels=[0.5], colors="#39FF88", linewidths=1.1)
        axes[1][k].set_title(f"melt at stop, {st.time_s:.0f} s\nIoU "
                             f"{j['arms']['TO_4bpp']['IoU']:.4f} / area "
                             f"{j['arms']['TO_4bpp']['IoU_area']:.4f}", fontsize=8)
        im2 = axes[2][k].imshow(phi - chi, cmap="RdBu_r", vmin=-1, vmax=1,
                                origin="lower", interpolation="bilinear")
        axes[2][k].contour(chi, levels=[0.5], colors="k", linewidths=0.8)
        axes[2][k].set_title(f"melt minus target\ngrow "
                             f"{j['arms']['TO_4bpp']['bed_melt_pct_of_part']:.1f} %, under "
                             f"{j['arms']['TO_4bpp']['part_under_melt_pct']:.1f} %",
                             fontsize=8)
        for r in range(3):
            axes[r][k].set_xticks([]); axes[r][k].set_yticks([])
    if im0 is not None:
        fig.colorbar(im0, ax=axes[0].tolist(), fraction=0.02, label="saturation")
        fig.colorbar(im2, ax=axes[2].tolist(), fraction=0.02,
                     label="melt fraction minus target")
    fig.suptitle("Delivered maps at grid 120, what they melt, and where the residual "
                 "error sits. Green and black outline the area-fill target.",
                 fontsize=11)
    p = FIGS / "fig_topopt_maps.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_acceptance(data) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(19.5, 5.8))
    dm = {d["shape"]: _load(OUT / f"{d['shape']}_dosematch.json") for d in data}

    ax = axes[0]
    labels, groups = [], []
    for d in data:
        if d["rb"] is None:
            continue
        rb, cb = d["rb"], d["ctl_rb"]

        def best160(r):
            g = r["gate_A_grid"]
            return max(g["TO_4bpp_160_maptransfer_recal"]["IoU"],
                       g["TO_4bpp_160_designtransfer_recal"]["IoU"])
        labels.append(d["shape"])
        groups.append({
            "uniform at 160": rb["gate_A_grid"]["U_uniform_160_recal"]["IoU"],
            "continuation at 120": d["solve"]["arms"]["TO_4bpp"]["IoU"],
            "continuation at 160": best160(rb),
            "control, no projection, at 120":
                (d["ctl"] or {}).get("arms", {}).get("TO_4bpp", {}).get("IoU", np.nan),
            "control, no projection, at 160": best160(cb) if cb else np.nan})
    keys = list(groups[0])
    cols = ["#C44E52", "#9BB7D4", "#4C72B0", "#A8D5A2", "#55A868"]
    x = np.arange(len(labels))
    w = 0.8 / len(keys)
    for i, key in enumerate(keys):
        ax.bar(x + i * w - 0.4 + w / 2, [g[key] for g in groups], w, label=key,
               color=cols[i])
    ax.axhline(0.95, color="crimson", lw=1.4, ls="--", label="SOLVED, IoU 0.95")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("intersection over union against the binary part mask")
    ax.set_ylim(0.55, 1.03)
    ax.set_title("A  Gate A, grid hold-out. Solve at 120, score at 160 with the voltage\n"
                 "recalibrated so the uniform arm absorbs 500 W/m there. The 160 bar is\n"
                 "the better of the map and design transfers.", fontsize=9.5)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    names, unf, ms_, ct_, to_ = [], [], [], [], []
    for d in data:
        if d["rb"] is None:
            continue
        names.append(d["shape"])
        unf.append(abs(100 * d["rb_old"]["verdict"]["dJ_rel_r1.0"]) if d["rb_old"] else np.nan)
        if d["ms_rb"]:
            bj = d["ms_rb"]["baseline_120"]["MS_4bpp"]["J"]
            ms_.append(abs(100 * (d["ms_rb"]["rim"]["r1.0"]["J"] - bj) / abs(bj)))
        else:
            ms_.append(np.nan)
        to_.append(abs(100 * d["rb"]["gate_B_sub_radius"]["r1.0"]["dJ_rel"]))
        ct_.append(abs(100 * d["ctl_rb"]["gate_B_sub_radius"]["r1.0"]["dJ_rel"])
                   if d["ctl_rb"] else np.nan)
    x = np.arange(len(names))
    for i, (vals, lab, col) in enumerate((
            (unf, "unfiltered single start (out_robust)", "#C44E52"),
            (ms_, "filtered multi-start, 0.76 mm radius (out_ms)", "#DD8452"),
            (ct_, "1.00 mm radius, NO projection (control, this pass)", "#55A868"),
            (to_, "1.00 mm radius WITH projection (this pass)", "#4C72B0"))):
        ax.bar(x + i * 0.2 - 0.3, vals, 0.2, color=col, label=lab)
        for xi, v in zip(x + i * 0.2 - 0.3, vals):
            if np.isfinite(v):
                ax.text(xi, v * 1.15, f"{v:.0f}", ha="center", fontsize=6.5)
    ax.axhline(10, color="green", lw=1.4, ls="--", label="the 10 percent tolerance")
    ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=20)
    ax.set_ylabel("absolute change in J at a one-cell (0.50 mm) blur, percent")
    ax.set_title("B  Gate B at a fixed 0.50 mm blur, four arms. The projection is what\n"
                 "removes the sub-radius sensitivity; the radius alone does not.\n"
                 "Missing bars were not run in that pass.", fontsize=9.5)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(alpha=0.3, axis="y", which="both")

    ax = axes[2]
    from matplotlib.lines import Line2D
    colr = {}
    for d in data:
        if d["rb"] is None:
            continue
        sh = d["shape"]
        rr = [m for m in sorted(d["rb"]["gate_B_sub_radius"].values(),
                                key=lambda m: m["gaussian_sigma_cells"])
              if m["below_filter_radius"]]
        xs = [0.0] + [m["gaussian_sigma_mm"] for m in rr]
        line, = ax.plot(xs, [0.0] + [100 * m["dJ_rel"] for m in rr], "o-", ms=4, lw=1.8)
        colr[sh] = line.get_color()
        b2 = sorted(d["rb"]["gate_B2_design_blur"].values(),
                    key=lambda m: m["gaussian_sigma_cells"])
        ax.plot(xs, [0.0] + [100 * m["dJ_rel"] for m in b2], "s--", color=colr[sh],
                ms=3.6, lw=1.2)
        if dm[sh]:
            b3 = sorted(dm[sh]["arms"].values(),
                        key=lambda m: m["gaussian_sigma_cells"])
            ax.plot(xs, [0.0] + [100 * m["dJ_rel"] for m in b3], "^:",
                    color=colr[sh], ms=4, lw=1.2)
    ax.axhspan(-10, 10, color="green", alpha=0.10)
    ax.axhline(0, color="k", lw=0.8)
    h1 = [Line2D([], [], color=c, lw=2, label=sh) for sh, c in colr.items()]
    h2 = [Line2D([], [], color="0.3", ls="-", marker="o", label="map space"),
          Line2D([], [], color="0.3", ls="--", marker="s", label="design space"),
          Line2D([], [], color="0.3", ls=":", marker="^", label="map space, dose matched")]
    ax.legend(handles=h1 + h2, fontsize=7, ncol=2, loc="upper left")
    ax.set_xlabel("blur standard deviation, mm (all below the 1.00 mm radius)")
    ax.set_ylabel("change in J, percent")
    ax.set_title("C  the three forms of Gate B. Two shapes leave the band and both are\n"
                 "named: trapezoid in every form, circle only when dose matched.",
                 fontsize=10)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    p = FIGS / "fig_topopt_acceptance.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


def main(which: str = "all") -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    data = collect()
    made = []
    if which in ("all", "param"):
        made.append(fig_parameterization())
    if which in ("all", "gate"):
        made.append(fig_gate())
    if which in ("all", "cont"):
        made.append(fig_continuation(data))
    if which in ("all", "maps"):
        made.append(fig_maps(data))
    if which in ("all", "accept"):
        made.append(fig_acceptance(data))
    for p in made:
        print(p if p else "SKIPPED (no inputs)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
