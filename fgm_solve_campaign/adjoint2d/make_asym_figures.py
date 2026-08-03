"""Figures for the DENSE-IF-AND-ONLY-IF-IN-BOUNDS solve.

  fig_asym_<shape>.png   one composite per shape. Row 1 the dopant map of every
      arm; row 2 the melt fraction at that arm's own J_asym stop, where any
      colour outside the cyan nominal boundary IS the hard out-of-bounds term;
      row 3 the relative density at the same stop, where any colour below the
      dashed floor contour IS the soft in-bounds term. Row 4 spans: the
      objective decomposition against time, and this shape's arms in the trade
      plane.

  fig_asym_census.png    the five-shape verdict: the objective decomposition by
      arm, every arm in the trade plane, what the asymmetric solve moves
      against the melt-solved map, and the two optimizers head to head.

  fig_asym_trade.png     THE trade curve. Sweeping the out-of-bounds price
      w_out over a stored pair of curves recovers the exact stop at every
      price from one forward run, so this is the measured exchange rate between
      shape (growth) and in-bounds density, per shape, with no extra solve.

  fig_asym_floor_hexagon.png   the floor sweep, f = 0.80, 0.85, 0.90.

CONVENTIONS drawn on every number. The stop is the argmin over that arm's own
stored trajectory of J_asym. The melted region is melt fraction >= 0.5. Growth
is melted bed cells as a percentage of the part cell count; under is unmelted
part cells on the same normalization. Grid 120 x 120 throughout.

Run:
  ./.venv312/bin/python -m adjoint2d.make_asym_figures shape <shape> <out> <figs>
  ./.venv312/bin/python -m adjoint2d.make_asym_figures census <out> <figs>
  ./.venv312/bin/python -m adjoint2d.make_asym_figures trade <out> <figs>
  ./.venv312/bin/python -m adjoint2d.make_asym_figures floor <out> <figs>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from . import asym_objective as ao, asym_solve as asol, shape_objective as so
from .library_solve import shape_config
from .make_figures import crop_box
from .pins import build_case, load_cfg

DPI = 180
SHAPES = ("square", "hexagon", "cross", "triangle", "L_shape")
ARM_ORDER = [
    ("U_uniform", "uniform s = 1\nno grading"),
    ("PHI4_prev", "MELT-solved map, 4 bits per pixel\nthe previous deliverable"),
    ("ASYM_lbfgsb_4bpp", "ASYM-solved, L-BFGS-B\n4 bits per pixel"),
    ("ASYM_mma_4bpp", "ASYM-solved, MMA\n4 bits per pixel"),
]
ARM_COLOR = {"U_uniform": "#8c8c8c", "PHI4_prev": "#1f77b4",
             "ASYM_lbfgsb_4bpp": "#d62728", "ASYM_mma_4bpp": "#2ca02c"}


def load(outdir: Path, stem: str) -> dict:
    return json.loads((outdir / f"{stem}.json").read_text())


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def _map_panel(ax, s, pm, extent, title):
    im = ax.imshow(np.where(pm, s, np.nan), origin="lower", extent=extent,
                   vmin=0.0, vmax=1.0, cmap="viridis", interpolation="nearest")
    xs = np.linspace(extent[0], extent[1], pm.shape[1])
    ys = np.linspace(extent[2], extent[3], pm.shape[0])
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#ffffff", linewidths=1.0)
    ax.set_title(title, fontsize=7.0, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def _field_panel(ax, f, pm, extent, title, cmap, front, vmin, vmax):
    im = ax.imshow(f, origin="lower", extent=extent, vmin=vmin, vmax=vmax,
                   cmap=cmap, interpolation="bilinear")
    xs = np.linspace(extent[0], extent[1], pm.shape[1])
    ys = np.linspace(extent[2], extent[3], pm.shape[0])
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#00e5ff", linewidths=1.4)
    if np.nanmax(f) > front > np.nanmin(f):
        ax.contour(xs, ys, f, levels=[front], colors="#ffffff", linewidths=1.0,
                   linestyles="--")
    ax.set_title(title, fontsize=7.0, linespacing=1.35)
    ax.set_xticks([]); ax.set_yticks([])
    return im


# ---------------------------------------------------------------------------
# the measured trade curve, recomputed from one forward run per arm
# ---------------------------------------------------------------------------

def measured_trade(case, s, floor, w_list=asol.TRADE_W_OUT) -> list[dict]:
    """For each out-of-bounds price, the stop it implies and the outcome there.

    One forward run covers the whole sweep, because both parts of J_asym are
    read at the same index and only the weight changes.
    """
    tr = asol.run_forward(case, s)
    jt, jo, ji = ao.curves(tr, case, floor=floor, w_out=1.0, w_in=1.0)
    rows = []
    for w in w_list:
        i = int(np.argmin(float(w) * jo + ji))
        m = ao.region_metrics(tr.T_at_end(i), tr.rho_at_end(i), case, floor=floor)
        rows.append({"w_out": float(w), "index": i,
                     "time_s": float(tr.time_s[i]), **m})
    del tr
    return rows


# ---------------------------------------------------------------------------
# per-shape composite
# ---------------------------------------------------------------------------

def shape_figure(shape: str, outdir: Path, figdir: Path, stem: str | None = None) -> Path:
    stem = stem or shape
    res = load(outdir, stem)
    floor = float(res["floor_rho_rel"])
    rho_lo = float(res["rho_floor"])
    case = build_case(load_cfg(shape_config(shape)))
    pm = np.asarray(case.part_mask, dtype=bool)
    extent = (case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3)
    box = crop_box(pm, case.x, case.y)
    with np.load(outdir / f"{stem}_maps.npz") as d:
        maps = {k: np.asarray(d[k], dtype=float) for k in d.files}
    cols = [(k, lab, maps[k]) for k, lab in ARM_ORDER if k in maps and k in res["arms"]]

    fig = plt.figure(figsize=(2.95 * len(cols), 12.6))
    gs = fig.add_gridspec(4, 2 * len(cols), height_ratios=[1.0, 1.0, 1.0, 0.72],
                          hspace=0.34, wspace=0.55)
    axes = [[fig.add_subplot(gs[i, 2 * j:2 * j + 2]) for j in range(len(cols))]
            for i in range(3)]
    im_map = im_phi = im_rho = None
    curves_store = {}
    for j, (key, label, s) in enumerate(cols):
        m = res["arms"][key]
        i_stop = int(m["asym_stop_index"])
        im_map = _map_panel(axes[0][j], s, pm, extent,
                            f"{label}\nmean s in part {float(np.mean(s[pm])):.3f}   "
                            f"roughness {m['map_roughness']:.4f}")
        tr = asol.run_forward(case, s, eps_covary=bool(m.get("eps_covary", False)))
        jt, jo, ji = ao.curves(tr, case, floor=floor)
        curves_store[key] = (np.asarray(tr.time_s, dtype=float), jt, jo, ji, i_stop)
        phi = so.phi_field(tr.T_at_end(i_stop), case)[0]
        rho = np.where(pm, np.asarray(tr.rho_at_end(i_stop), dtype=float), np.nan)
        del tr
        im_phi = _field_panel(
            axes[1][j], phi, pm, extent,
            f"melt fraction at the J_asym stop {m['asym_stop_s']:.0f} s"
            f"{' (AT HORIZON)' if m['asym_stop_at_horizon'] else ''}\n"
            f"OUT of bounds {m['J_asym_out']:.4f} of J_asym {m['J_asym']:.4f}\n"
            f"growth {m['growth_pct']:.2f} %     under {m['under_pct']:.2f} %\n"
            f"melt IoU {m['IoU']:.4f}", "inferno", 0.5, 0.0, 1.0)
        im_rho = _field_panel(
            axes[2][j], rho, pm, extent,
            f"relative density at the SAME stop\n"
            f"IN bounds deficit {m['J_asym_in']:.4f} of J_asym {m['J_asym']:.4f}\n"
            f"mean {m['mean_rho_rel_part']:.3f}   min {m['min_rho_rel_part']:.3f}\n"
            f"at or above the floor {100 * m['frac_part_at_or_above_floor']:.1f} % "
            f"of the part", "cividis", floor, rho_lo, 1.0)
        for i in range(3):
            axes[i][j].set_xlim(box[0], box[1]); axes[i][j].set_ylim(box[2], box[3])

    fig.colorbar(im_map, ax=axes[0], fraction=0.02, pad=0.01,
                 label="binder saturation s")
    fig.colorbar(im_phi, ax=axes[1], fraction=0.02, pad=0.01, label="melt fraction")
    fig.colorbar(im_rho, ax=axes[2], fraction=0.02, pad=0.01,
                 label=f"relative density, {rho_lo:.2f} to 1")

    # --- row 4 left: the objective against time -----------------------------
    axc = fig.add_subplot(gs[3, 0:len(cols)])
    for key, (t, jt, jo, ji, i_stop) in curves_store.items():
        c = ARM_COLOR.get(key, "#333333")
        axc.plot(t, jt, color=c, lw=1.6, label=key)
        axc.plot(t, jo, color=c, lw=0.9, ls="--", alpha=0.75)
        axc.plot(t, ji, color=c, lw=0.9, ls=":", alpha=0.75)
        axc.plot(t[i_stop], jt[i_stop], "o", color=c, ms=5.5)
    axc.set_xlabel("time, s", fontsize=8)
    axc.set_ylabel("J_asym per part cell", fontsize=8)
    axc.set_title("solid J_asym; dashed the OUT-of-bounds term, which rises;\n"
                  "dotted the IN-bounds deficit, which falls. The dot is the "
                  "argmin stop.", fontsize=7.5)
    axc.tick_params(labelsize=7)
    axc.legend(fontsize=6.5, ncol=2)
    axc.grid(alpha=0.25)

    # --- row 4 right: the trade plane ---------------------------------------
    axt = fig.add_subplot(gs[3, len(cols):])
    for key, _lab, _s in cols:
        m = res["arms"][key]
        axt.scatter(m["growth_pct"], 100 * m["frac_part_at_or_above_floor"],
                    s=70, color=ARM_COLOR.get(key, "#333"), label=key, zorder=3)
    axt.axvline(asol.GROWTH_TOL_PCT, color="k", lw=0.8, ls="--")
    axt.axhline(100 * asol.FLOOR_FRAC_TOL, color="k", lw=0.8, ls="--")
    axt.scatter([0], [100], marker="*", s=220, color="#ffb000", zorder=4,
                edgecolor="k", linewidth=0.5)
    axt.set_xlabel("growth: melted bed as a percent of the part", fontsize=8)
    axt.set_ylabel(f"percent at or above the {floor:.2f} floor", fontsize=8)
    axt.set_title("the specification plane. The star at (0, 100) is DENSE IF AND\n"
                  "ONLY IF IN BOUNDS; dashed are the acceptance tolerances.",
                  fontsize=7.5)
    axt.tick_params(labelsize=7)
    axt.grid(alpha=0.25)
    axt.legend(fontsize=6.5)

    v = res["verdict"]
    fig.suptitle(
        f"{shape}   DENSE IF AND ONLY IF IN BOUNDS, grid 120 x 120, floor "
        f"{floor:.2f} relative density, w_out {res['w_out']:.2f}, design filter "
        f"sigma {res['sigma_cells']} cells, "
        f"{res['budget_forward_equivalents_per_optimizer']:.0f} forward-equivalents "
        f"per optimizer.\nEvery panel is read at that arm's OWN J_asym stop "
        f"(argmin over its own trajectory). Best deliverable: "
        f"{v['best_deliverable']}. Energy-gate violations: "
        f"{res['energy_gate_violations'] or 'none'}.", fontsize=9.5)
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / f"fig_asym_{stem}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# census
# ---------------------------------------------------------------------------

def census_figure(outdir: Path, figdir: Path, shapes=SHAPES) -> Path:
    res = {s: load(outdir, s) for s in shapes if (outdir / f"{s}.json").exists()}
    keys = [k for k, _ in ARM_ORDER]
    fig, ax = plt.subplots(2, 2, figsize=(15.0, 10.0))

    # A: the decomposition
    a = ax[0, 0]
    w = 0.2
    xs = np.arange(len(res))
    for i, k in enumerate(keys):
        out = [res[s]["arms"][k]["J_asym_out"] if k in res[s]["arms"] else np.nan
               for s in res]
        inn = [res[s]["arms"][k]["J_asym_in"] if k in res[s]["arms"] else np.nan
               for s in res]
        a.bar(xs + (i - 1.5) * w, out, w, color=ARM_COLOR[k], label=f"{k} out")
        a.bar(xs + (i - 1.5) * w, inn, w, bottom=out, color=ARM_COLOR[k],
              alpha=0.42, hatch="//", label=f"{k} in")
    a.set_xticks(xs); a.set_xticklabels(list(res), fontsize=8)
    a.set_ylabel("J_asym per part cell", fontsize=9)
    a.set_title("A. what the objective is made of. Solid is the HARD "
                "out-of-bounds term,\nhatched is the SOFT in-bounds deficit "
                "above the floor hinge.", fontsize=9)
    a.legend(fontsize=6, ncol=2)
    a.grid(alpha=0.25, axis="y")

    # B: the specification plane
    b = ax[0, 1]
    for s in res:
        for k in keys:
            if k not in res[s]["arms"]:
                continue
            m = res[s]["arms"][k]
            b.scatter(m["growth_pct"], 100 * m["frac_part_at_or_above_floor"],
                      s=62, color=ARM_COLOR[k], zorder=3)
            b.annotate(s[:4], (m["growth_pct"], 100 * m["frac_part_at_or_above_floor"]),
                       fontsize=5.5, xytext=(3, 3), textcoords="offset points")
    b.scatter([0], [100], marker="*", s=260, color="#ffb000", edgecolor="k",
              linewidth=0.5, zorder=4)
    b.axvline(asol.GROWTH_TOL_PCT, color="k", lw=0.8, ls="--")
    b.axhline(100 * asol.FLOOR_FRAC_TOL, color="k", lw=0.8, ls="--")
    b.set_xlabel("growth: melted bed as a percent of the part", fontsize=9)
    b.set_ylabel("percent of the part at or above the density floor", fontsize=9)
    b.set_title("B. the specification plane, every arm of every shape.\nThe star "
                "is DENSE IF AND ONLY IF IN BOUNDS. Nothing reaches it.",
                fontsize=9)
    b.grid(alpha=0.25)
    for k in keys:
        b.scatter([], [], color=ARM_COLOR[k], label=k)
    b.legend(fontsize=6.5)

    # C: what the asymmetric solve moves against the melt-solved map
    c = ax[1, 0]
    labels, dgrowth, dfloor = [], [], []
    for s in res:
        arms = res[s]["arms"]
        if "PHI4_prev" not in arms:
            continue
        best = min([k for k in arms if k.startswith("ASYM_") and k.endswith("4bpp")],
                   key=lambda k: arms[k]["J_asym"], default=None)
        if best is None:
            continue
        labels.append(f"{s}\n{best.split('_')[1]}")
        dgrowth.append(arms[best]["growth_pct"] - arms["PHI4_prev"]["growth_pct"])
        dfloor.append(100 * (arms[best]["frac_part_at_or_above_floor"]
                             - arms["PHI4_prev"]["frac_part_at_or_above_floor"]))
    xs = np.arange(len(labels))
    c.bar(xs - 0.2, dgrowth, 0.4, color="#d62728", label="change in growth, points")
    c.bar(xs + 0.2, dfloor, 0.4, color="#2ca02c",
          label="change in percent of the part above the floor")
    c.axhline(0, color="k", lw=0.8)
    c.set_xticks(xs); c.set_xticklabels(labels, fontsize=7)
    c.set_title("C. the asymmetric solve against the MELT-solved map.\nNegative "
                "growth and positive floor coverage are both improvements.",
                fontsize=9)
    c.legend(fontsize=7)
    c.grid(alpha=0.25, axis="y")

    # D: the two optimizers
    d = ax[1, 1]
    labels, jl, jm = [], [], []
    for s in res:
        arms = res[s]["arms"]
        if "ASYM_lbfgsb_4bpp" in arms and "ASYM_mma_4bpp" in arms:
            labels.append(s)
            jl.append(arms["ASYM_lbfgsb_4bpp"]["J_asym"])
            jm.append(arms["ASYM_mma_4bpp"]["J_asym"])
    xs = np.arange(len(labels))
    d.bar(xs - 0.2, jl, 0.4, color=ARM_COLOR["ASYM_lbfgsb_4bpp"], label="L-BFGS-B")
    d.bar(xs + 0.2, jm, 0.4, color=ARM_COLOR["ASYM_mma_4bpp"], label="MMA")
    for i, (x, y) in enumerate(zip(jl, jm)):
        d.annotate(f"{100 * (y - x) / max(x, 1e-30):+.1f} %", (xs[i], max(x, y)),
                   ha="center", va="bottom", fontsize=7)
    d.set_xticks(xs); d.set_xticklabels(labels, fontsize=8)
    d.set_ylabel("J_asym of the 4-bits-per-pixel deliverable", fontsize=9)
    d.set_title("D. the two optimizers at a matched 40 forward-equivalents.\n"
                "The percentage is MMA against L-BFGS-B; negative is MMA winning.",
                fontsize=9)
    d.legend(fontsize=7.5)
    d.grid(alpha=0.25, axis="y")

    any_res = next(iter(res.values()))
    fig.suptitle(
        f"DENSE IF AND ONLY IF IN BOUNDS, five shapes, grid 120 x 120, floor "
        f"{any_res['floor_rho_rel']:.2f} relative density, w_out "
        f"{any_res['w_out']:.2f}, every arm at its own J_asym stop.", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / "fig_asym_census.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# the trade curve
# ---------------------------------------------------------------------------

def trade_figure(outdir: Path, figdir: Path, shapes=SHAPES) -> Path:
    fig, ax = plt.subplots(1, 2, figsize=(14.0, 5.8))
    store = {}
    for s in shapes:
        if not (outdir / f"{s}.json").exists():
            continue
        res = load(outdir, s)
        floor = float(res["floor_rho_rel"])
        case = build_case(load_cfg(shape_config(s)))
        pm = np.asarray(case.part_mask, dtype=bool)
        arms = res["arms"]
        best = min([k for k in arms if k.startswith("ASYM_") and k.endswith("4bpp")],
                   key=lambda k: arms[k]["J_asym"])
        with np.load(outdir / f"{s}_maps.npz") as d:
            smap = np.asarray(d[best], dtype=float)
        rows = measured_trade(case, smap, floor)
        store[s] = rows
        g = [r["growth_pct"] for r in rows]
        rho = [r["mean_rho_rel_part"] for r in rows]
        ln, = ax[0].plot(g, rho, "o-", ms=4, lw=1.4, label=f"{s} ({best.split('_')[1]})")
        for r in rows:
            if r["w_out"] in (0.25, 1.0, 5.0, 20.0):
                ax[0].annotate(f"{r['w_out']:g}", (r["growth_pct"],
                                                   r["mean_rho_rel_part"]),
                               fontsize=6, xytext=(3, -7), textcoords="offset points",
                               color=ln.get_color())
        ax[1].plot([r["w_out"] for r in rows], [r["time_s"] for r in rows], "o-",
                   ms=4, lw=1.4, color=ln.get_color(), label=s)
        del case, pm

    any_floor = float(load(outdir, next(iter(store)))["floor_rho_rel"])
    ax[0].axhline(any_floor, color="k", ls="--", lw=0.9)
    ax[0].annotate(f"the {any_floor:.2f} density floor", (0.02, any_floor + 0.004),
                   xycoords=("axes fraction", "data"), fontsize=7.5)
    ax[0].set_xlabel("growth: melted bed as a percent of the part", fontsize=9)
    ax[0].set_ylabel("mean in-bounds relative density at the implied stop", fontsize=9)
    ax[0].set_title("THE TRADE. Each point is one out-of-bounds price w_out "
                    "(labelled) on the SAME solved map;\nraising the price stops "
                    "earlier, which buys shape and pays for it in density.",
                    fontsize=9)
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.25)
    ax[1].set_xscale("log")
    ax[1].set_xlabel("out-of-bounds price w_out", fontsize=9)
    ax[1].set_ylabel("the stop the price implies, s", fontsize=9)
    ax[1].set_title("the same sweep as a stop time. The whole trade is bought and "
                    "sold\nin the READ STATE, which is why the price sets the "
                    "bake.", fontsize=9)
    ax[1].legend(fontsize=7); ax[1].grid(alpha=0.25)
    fig.tight_layout()
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / "fig_asym_trade.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    (figdir / "trade_rows.json").write_text(json.dumps(store, indent=2, default=float))
    return p


# ---------------------------------------------------------------------------
# the floor sweep
# ---------------------------------------------------------------------------

def floor_figure(outdir: Path, figdir: Path, shape: str = "hexagon",
                 stems=("hexagon_f080", "hexagon", "hexagon_f090")) -> Path:
    have = [st for st in stems if (outdir / f"{st}.json").exists()]
    case = build_case(load_cfg(shape_config(shape)))
    pm = np.asarray(case.part_mask, dtype=bool)
    extent = (case.x[0] * 1e3, case.x[-1] * 1e3, case.y[0] * 1e3, case.y[-1] * 1e3)
    box = crop_box(pm, case.x, case.y)
    fig, ax = plt.subplots(2, len(have) + 1, figsize=(3.1 * (len(have) + 1), 7.2))
    # the REFERENCE map is the default-floor run, named explicitly rather than
    # taken as whichever stem happens to be processed first
    ref_stem = shape if (outdir / f"{shape}.json").exists() else have[0]
    ref_res = load(outdir, ref_stem)
    ref_arms = ref_res["arms"]
    ref_best = min([k for k in ref_arms if k.startswith("ASYM_") and k.endswith("4bpp")],
                   key=lambda k: ref_arms[k]["J_asym"])
    with np.load(outdir / f"{ref_stem}_maps.npz") as d:
        base = np.asarray(d[ref_best], dtype=float)
    ref_floor = float(ref_res["floor_rho_rel"])
    for j, st in enumerate(have):
        res = load(outdir, st)
        arms = res["arms"]
        best = min([k for k in arms if k.startswith("ASYM_") and k.endswith("4bpp")],
                   key=lambda k: arms[k]["J_asym"])
        with np.load(outdir / f"{st}_maps.npz") as d:
            smap = np.asarray(d[best], dtype=float)
        m = arms[best]
        _map_panel(ax[0, j], smap, pm, extent,
                   f"floor {res['floor_rho_rel']:.2f}   ({best.split('_')[1]})\n"
                   f"mean s {float(np.mean(smap[pm])):.3f}   "
                   f"root-mean-square distance from the\n{ref_floor:.2f} map "
                   f"{float(np.sqrt(np.mean((smap[pm] - base[pm]) ** 2))):.4f}")
        ax[0, j].set_xlim(box[0], box[1]); ax[0, j].set_ylim(box[2], box[3])
        tr = asol.run_forward(case, smap)
        i = int(m["asym_stop_index"])
        rho = np.where(pm, np.asarray(tr.rho_at_end(i), dtype=float), np.nan)
        del tr
        _field_panel(ax[1, j], rho, pm, extent,
                     f"relative density at the stop {m['asym_stop_s']:.0f} s\n"
                     f"mean {m['mean_rho_rel_part']:.3f}   growth "
                     f"{m['growth_pct']:.2f} %   IoU {m['IoU']:.4f}\n"
                     f"above ITS floor "
                     f"{100 * m['frac_part_at_or_above_floor']:.1f} %",
                     "cividis", float(res["floor_rho_rel"]), float(res["rho_floor"]), 1.0)
        ax[1, j].set_xlim(box[0], box[1]); ax[1, j].set_ylim(box[2], box[3])

    s1, s2 = ax[0, -1], ax[1, -1]
    fl = [float(load(outdir, st)["floor_rho_rel"]) for st in have]
    for k, lab, col in (("growth_pct", "growth, percent of the part", "#d62728"),
                        ("under_pct", "under-melt, percent of the part", "#9467bd")):
        vals = []
        for st in have:
            arms = load(outdir, st)["arms"]
            b = min([a for a in arms if a.startswith("ASYM_") and a.endswith("4bpp")],
                    key=lambda a: arms[a]["J_asym"])
            vals.append(arms[b][k])
        s1.plot(fl, vals, "o-", color=col, label=lab)
    s1.set_xlabel("density floor, relative density", fontsize=8)
    s1.set_title("what the floor choice moves", fontsize=8.5)
    s1.legend(fontsize=6.5); s1.grid(alpha=0.25); s1.tick_params(labelsize=7)
    for k, lab, col in (("mean_rho_rel_part", "mean in-bounds relative density", "#1f77b4"),
                        ("IoU", "melt IoU at the stop", "#2ca02c")):
        vals = []
        for st in have:
            arms = load(outdir, st)["arms"]
            b = min([a for a in arms if a.startswith("ASYM_") and a.endswith("4bpp")],
                    key=lambda a: arms[a]["J_asym"])
            vals.append(arms[b][k])
        s2.plot(fl, vals, "o-", color=col, label=lab)
    s2.set_xlabel("density floor, relative density", fontsize=8)
    s2.legend(fontsize=6.5); s2.grid(alpha=0.25); s2.tick_params(labelsize=7)
    s2.set_title("and what it does not", fontsize=8.5)

    fig.suptitle(f"{shape}: the density floor swept 0.80, 0.85, 0.90 relative "
                 f"density. Grid 120 x 120, w_out 1.00, 40 forward-equivalents "
                 f"per optimizer, both optimizers run and the better kept.",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    figdir.mkdir(parents=True, exist_ok=True)
    p = figdir / f"fig_asym_floor_{shape}.png"
    fig.savefig(p, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "shape":
        print(shape_figure(sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]),
                           sys.argv[5] if len(sys.argv) > 5 else None))
    elif mode == "census":
        print(census_figure(Path(sys.argv[2]), Path(sys.argv[3])))
    elif mode == "trade":
        print(trade_figure(Path(sys.argv[2]), Path(sys.argv[3])))
    elif mode == "floor":
        print(floor_figure(Path(sys.argv[2]), Path(sys.argv[3])))
    else:
        raise SystemExit(f"unknown mode {mode!r}")
