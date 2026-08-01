#!/usr/bin/env python3
"""Figures for the continuous-rotation campaign.

  fig_rot_kernels.png        the averaged heating kernel against the static
                             zero-degree kernel, one row per shape, uniform map
  fig_rot_maps_<shape>.png   the Level-1 solved map, its melt field under the
                             averaged kernel, and the static zero-degree map
                             beside it for contrast
  fig_rot_verify_<shape>.png melt against nominal under the TRUE rotating
                             engine, one panel per arm, with the J_phi
                             trajectories underneath
  fig_rot_summary.png        J_phi and IoU per arm per shape, static against
                             rotating

Run:
  ./.venv312/bin/python scripts/analysis/make_rot_figures.py [shape ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                        # noqa: E402

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import forward as fwd                    # noqa: E402
from adjoint2d import library_solve as lib              # noqa: E402
from adjoint2d.pins import build_case, load_cfg         # noqa: E402

OUT = REPO / "fgm_solve_campaign/out_rot"
FIGS = REPO / "fgm_solve_campaign/figs_rot"
SHAPES = ("T_shape", "L_shape", "cross", "star", "square")
DPI = 180


def _outline(ax, mask, color="w", lw=1.0):
    ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw)


def _im(ax, a, title, cmap="magma", vmin=None, vmax=None):
    h = ax.imshow(a, origin="lower", cmap=cmap, interpolation="bilinear",
                  vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    return h


def static_kernel(shape: str) -> tuple[np.ndarray, np.ndarray]:
    """State-B heating at zero degrees under a uniform map, for contrast."""
    case = build_case(load_cfg(lib.shape_config(shape)))
    s = np.ones(case.part_mask.shape)
    st = fwd.solve_electric(case, fwd.sigma_state_b(case, s)[0], fwd.eps_field(case))
    return st.Qrf, case.part_mask


def _radial_profile(Q, pm, nbin=24):
    ny, nx = pm.shape
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    cy = float(np.mean(jj[pm]))
    cx = float(np.mean(ii[pm]))
    r = np.sqrt((jj - cy) ** 2 + (ii - cx) ** 2)
    rmax = float(r[pm].max())
    edges = np.linspace(0.0, rmax, nbin + 1)
    mid, val = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = pm & (r >= a) & (r < b)
        if m.sum() >= 3:
            mid.append(0.5 * (a + b))
            val.append(float(np.mean(Q[m])))
    return np.asarray(mid), np.asarray(val)


def fig_kernels(shapes) -> Path:
    """Interiors, not edges.

    The boundary cells of the electro-quasi-static solve carry Q_rf one to two
    orders of magnitude above the interior, so a linear full-range colour scale
    renders every part solid black and says nothing. Every panel here is scaled
    to the 99th percentile INSIDE the part and the difference panel to the 98th
    percentile of its own magnitude, with the radial profile beside it because
    the physical claim to be checked is a RADIAL one.
    """
    have = [s for s in shapes if (OUT / f"{s}_rotavg_maps.npz").exists()]
    fig, axes = plt.subplots(len(have), 4, figsize=(10.0, 2.4 * len(have)),
                             constrained_layout=True, squeeze=False)
    scale = 1e-6
    for r, sh in enumerate(have):
        d = np.load(OUT / f"{sh}_rotavg_maps.npz")
        pm = d["part_mask"].astype(bool)
        Qavg = np.asarray(d["Q_avg_uniform_B"], dtype=float)
        Qst, _pm0 = static_kernel(sh)
        vmax = float(np.percentile(np.concatenate([Qst[pm], Qavg[pm]]), 99.0)) * scale
        h = _im(axes[r][0], np.where(pm, Qst, np.nan) * scale,
                f"{sh}: static 0 degrees", vmin=0, vmax=vmax)
        _outline(axes[r][0], pm, color="k", lw=0.6)
        _im(axes[r][1], np.where(pm, Qavg, np.nan) * scale,
            f"{sh}: rotationally averaged", vmin=0, vmax=vmax)
        _outline(axes[r][1], pm, color="k", lw=0.6)
        plt.colorbar(h, ax=axes[r][1], fraction=0.046,
                     label="Q_rf  MW per cubic metre")
        rel = np.where(pm, (Qavg - Qst) / max(float(np.mean(Qst[pm])), 1e-30), np.nan)
        lim = float(np.percentile(np.abs(rel[pm]), 98.0))
        h2 = _im(axes[r][2], rel, "averaged minus static,\nin units of the mean static Q_rf",
                 cmap="RdBu_r", vmin=-lim, vmax=lim)
        _outline(axes[r][2], pm, color="k", lw=0.6)
        plt.colorbar(h2, ax=axes[r][2], fraction=0.046)
        a = axes[r][3]
        for Q, lab, c in ((Qst, "static 0 degrees", "0.35"),
                          (Qavg, "rotationally averaged", "crimson")):
            mid, val = _radial_profile(Q, pm)
            a.plot(mid * 0.5, val * scale, "-o", ms=2.5, color=c, label=lab)
        a.set_xlabel("distance from the part centroid, mm", fontsize=7)
        a.set_ylabel("mean Q_rf, MW per m3", fontsize=7)
        a.tick_params(labelsize=6)
        a.grid(alpha=0.3)
        a.set_yscale("log")
        if r == 0:
            a.legend(fontsize=6)
        a.set_title("radial profile (log)", fontsize=8)
    fig.suptitle("Rotationally averaged heating kernel against the static kernel, "
                 "uniform dopant map, part frame, grid 120. Colour scales are "
                 "clipped at the 99th percentile inside the part.", fontsize=9)
    p = FIGS / "fig_rot_kernels.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def fig_maps(shape: str) -> Path | None:
    f = OUT / f"{shape}_rotavg_maps.npz"
    j = OUT / f"{shape}_rotavg.json"
    if not (f.exists() and j.exists()):
        return None
    d = np.load(f)
    r = json.loads(j.read_text())
    pm = d["part_mask"].astype(bool)
    fig, ax = plt.subplots(2, 3, figsize=(7.6, 5.2), constrained_layout=True)

    h = _im(ax[0][0], np.where(pm, d["sat_static0"], np.nan),
            f"stored static 0 deg map\n(prototype static J "
            f"{r['warm_start']['static_J_at_zero_deg']:.1f})",
            cmap="viridis", vmin=0, vmax=1)
    _outline(ax[0][0], pm, color="k", lw=0.6)
    _im(ax[0][1], np.where(pm, d["sat_cont"], np.nan),
        f"averaged-kernel solved map\n(J {r['arms']['AVG_cont']['J']:.1f}, "
        f"IoU {r['arms']['AVG_cont']['IoU']:.3f})", cmap="viridis", vmin=0, vmax=1)
    _outline(ax[0][1], pm, color="k", lw=0.6)
    _im(ax[0][2], np.where(pm, d["sat_4bpp"], np.nan),
        f"the same at 4 bits per pixel\n(J {r['arms']['AVG_4bpp']['J']:.1f})",
        cmap="viridis", vmin=0, vmax=1)
    _outline(ax[0][2], pm, color="k", lw=0.6)
    plt.colorbar(h, ax=ax[0][2], fraction=0.046, label="dopant saturation")

    for c, (key, lab) in enumerate((("phi_uniform", "uniform"),
                                    ("phi_static0", "static 0 deg map"),
                                    ("phi_cont", "averaged-kernel map"))):
        arm = {"phi_uniform": "AVG_uniform", "phi_static0": "AVG_static0deg_map",
               "phi_cont": "AVG_cont"}[key]
        m = r["arms"][arm]
        _im(ax[1][c], d[key], f"melt at stop, {lab}\nJ {m['J']:.1f}  "
            f"IoU {m['IoU']:.3f}  grow {m['bed_melt_pct_of_part']:.1f}%",
            cmap="inferno", vmin=0, vmax=1)
        _outline(ax[1][c], pm, color="c", lw=0.8)
    fig.suptitle(f"{shape}: Level 1, the rotationally averaged kernel "
                 f"({r['n_angles']} angles, {r['step_deg']:.0f}-degree step). "
                 f"Cyan is the nominal part.", fontsize=9)
    p = FIGS / f"fig_rot_maps_{shape}.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def fig_verify(shape: str) -> Path | None:
    j = OUT / f"{shape}_verify.json"
    f = OUT / "verify_fields" / f"{shape}_fields.npz"
    if not (j.exists() and f.exists()):
        return None
    r = json.loads(j.read_text())
    d = np.load(f)
    rows = {m["arm"]: m for m in r["rows"]}
    order = [a for a in rows if not a.startswith("C_")]
    n = len(order)
    ncol = min(n, 5)
    nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.0 * ncol, 2.5 * nrow),
                           constrained_layout=True, squeeze=False)
    for k, arm in enumerate(order):
        a = ax[k // ncol][k % ncol]
        phi = d[f"phi_{arm}"]
        mask = d[f"mask_{arm}"].astype(bool)
        m = rows[arm]
        _im(a, phi, f"{arm}\nJ {m['J']:.0f}  IoU {m['IoU']:.3f}\n"
                    f"grow {m['bed_melt_pct_of_part']:.0f}%  "
                    f"under {m['part_under_melt_pct']:.0f}%",
            cmap="inferno", vmin=0, vmax=1)
        _outline(a, mask, color="c", lw=0.8)
    for k in range(n, nrow * ncol):
        ax[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"{shape}: melt fraction at the END OF THE HORIZON under the TRUE "
                 f"rotating engine. Cyan is the nominal part at that instant. "
                 f"Metrics are read at each arm's own J-stop.", fontsize=8)
    p = FIGS / f"fig_rot_verify_{shape}.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def fig_summary(shapes) -> Path | None:
    """NOT a deliverable. Kept because it is referenced in the report as the
    figure that was CUT: it collapsed rotating arms measured at different
    rotation periods into a single bar, which is not a comparison. The headline
    figure replaces it with one explicit period per shape."""
    have = [s for s in shapes if (OUT / f"{s}_verify.json").exists()]
    if not have:
        return None
    keys = ["S_uniform", "S_map0", "S_joint", "R_uniform", "R_map0",
            "R_joint", "R_avg", "R_avg4bpp"]
    fig, ax = plt.subplots(2, 1, figsize=(9.0, 6.2), constrained_layout=True)
    w = 0.10
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for a, metric, lab in ((ax[0], "J", "J_phi at the arm's own stop (lower better)"),
                           (ax[1], "IoU", "melt IoU at the arm's own stop (higher better)")):
        for i, k in enumerate(keys):
            vals, xs = [], []
            for s_i, sh in enumerate(have):
                rows = {m["arm"].split("_P")[0] if m["arm"].startswith("R_") else m["arm"]: m
                        for m in json.loads((OUT / f"{sh}_verify.json").read_text())["rows"]}
                if k in rows:
                    vals.append(rows[k][metric])
                    xs.append(s_i + (i - len(keys) / 2) * w)
            a.bar(xs, vals, width=w, label=k, color=colors[i % 10])
        a.set_xticks(range(len(have)))
        a.set_xticklabels(have)
        a.set_ylabel(lab, fontsize=8)
        a.grid(axis="y", alpha=0.3)
    ax[0].legend(fontsize=7, ncol=4)
    fig.suptitle("Continuous rotation against static, TRUE rotating engine, grid 120",
                 fontsize=9)
    p = FIGS / "fig_rot_summary.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def fig_headline(shapes) -> Path | None:
    """The one figure that answers the question, at each arm's OWN J-stop.

    Rows are shapes; columns are the four arms the verdict compares. The melt
    field shown IS the field the numbers describe, because the glue snapshots
    it at the running J minimum rather than at the horizon.
    """
    have = [s for s in shapes if (OUT / f"{s}_headline.json").exists()]
    if not have:
        return None
    arms = ["S_uniform", "S_best_static", "R_uniform", "R_avg", "I90_avg4angle"]
    titles = {"S_uniform": "static, uniform dopant",
              "S_best_static": "best STATIC arm\n(joint angle plus map)",
              "R_uniform": "rotating, uniform dopant",
              "R_avg": "rotating, SOLVED map\n(24-angle kernel)",
              "I90_avg4angle": "90-degree INDEXING,\nSOLVED 4-angle map"}
    fig, ax = plt.subplots(len(have), 5, figsize=(11.4, 2.55 * len(have)),
                           constrained_layout=True, squeeze=False)
    for r, sh in enumerate(have):
        d = dict(np.load(OUT / "verify_fields" / f"{sh}_headline.npz"))
        meta = json.loads((OUT / f"{sh}_headline.json").read_text())
        rows = {m["arm"]: m for m in meta["rows"]}
        fi = OUT / "verify_fields" / f"{sh}_index90.npz"
        ji = OUT / f"{sh}_index90.json"
        if fi.exists() and ji.exists():
            d.update(dict(np.load(fi)))
            rows.update({m["arm"]: m for m in json.loads(ji.read_text())["rows"]})
        best = min([a for a in arms if a in rows], key=lambda k: rows[k]["J"])
        for c, arm in enumerate(arms):
            if arm not in rows:
                ax[r][c].axis("off")
                continue
            m = rows[arm]
            a = ax[r][c]
            _im(a, d[f"phi_{arm}"], "", cmap="inferno", vmin=0, vmax=1)
            _outline(a, d[f"mask_{arm}"].astype(bool), color="c", lw=0.9)
            hz = " bound" if m["t_stop_at_horizon"] else ""
            a.set_title(f"{titles[arm] if r == 0 else ''}\n"
                        f"J {m['J']:.0f}{hz}   IoU {m['IoU']:.3f}\n"
                        f"grow {m['bed_melt_pct_of_part']:.0f}%  "
                        f"under {m['part_under_melt_pct']:.0f}%",
                        fontsize=7.5,
                        color=("darkgreen" if arm == best else "black"),
                        fontweight=("bold" if arm == best else "normal"))
            if c == 0:
                a.set_ylabel(sh, fontsize=9)
        ax[r][2].set_xlabel(f"period {meta['period_s']:.0f} s", fontsize=7)
        ax[r][3].set_xlabel(f"period {meta['period_s']:.0f} s", fontsize=7)
        ax[r][4].set_xlabel("index every 2.0 s", fontsize=7)
    fig.suptitle("Does continuous rotation beat the best static arm? Melt fraction at "
                 "each arm's OWN J-stop, true rotating engine, grid 120.\n"
                 "Cyan is the nominal part at that instant. Green bold is the winner "
                 "on J_phi. Every arm shown passes the 5 percent energy gate.",
                 fontsize=9)
    p = FIGS / "fig_rot_headline.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def _all_rows(shape) -> list[dict]:
    rows = []
    for f in (OUT / f"{shape}_verify.json", OUT / f"{shape}_speed.json"):
        if f.exists():
            rows += json.loads(f.read_text())["rows"]
    seen, out = set(), []
    for r in rows:
        if r["arm"] not in seen:
            seen.add(r["arm"])
            out.append(r)
    return out


def fig_speed(shapes) -> Path | None:
    """The two error sources of the continuous-rotation idea, on one abscissa.

    LEFT: J_phi against the rotation period. The quasi-static approximation
    improves as the period shortens; the engine's rotation-event remap error
    grows as it shortens. Where the two cross is the best achievable period.
    RIGHT: the energy-residual gate against the number of rotation events, with
    the 90-degree control, whose remap is an exact pixel permutation, marked.
    """
    have = [s for s in shapes if _all_rows(s)]
    if not have:
        return None
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    cols = dict(zip(SHAPES, plt.cm.tab10(np.linspace(0, 1, 10))))
    for sh in have:
        rows = _all_rows(sh)
        for arm_pref, ls, mk in (("R_avg_P", "-", "o"), ("R_uniform_P", "--", "s")):
            pts = sorted([(r["period_s"], r["J"], r["energy_gate_PASS"]) for r in rows
                          if r["arm"].startswith(arm_pref) and "period_s" in r])
            if not pts:
                continue
            p, j, ok = zip(*pts)
            ax[0].plot(p, j, ls, color=cols[sh], lw=1.2, alpha=0.9,
                       label=f"{sh} {'solved map' if 'avg' in arm_pref else 'uniform'}")
            ax[0].scatter(p, j, marker=mk, s=34, color=cols[sh],
                          edgecolors=["k" if o else "r" for o in ok],
                          linewidths=[0.6 if o else 1.6 for o in ok], zorder=3)
        f1 = OUT / f"{sh}_rotavg.json"
        if f1.exists():
            jq = float(json.loads(f1.read_text())["arms"]["AVG_cont"]["J"])
            ax[0].axhline(jq, color=cols[sh], lw=0.7, alpha=0.45, ls=":")
        sj = [r["J"] for r in rows if r["arm"] == "S_joint"]
        if sj:
            ax[0].axhline(sj[0], color=cols[sh], lw=0.9, alpha=0.8, ls="-.")
        for r in rows:
            if "n_rotation_events" in r and r["n_rotation_events"]:
                is90 = r["arm"] == "C_null90"
                ax[1].scatter(r["n_rotation_events"], 100 * r["energy_residual_rel"],
                              marker="*" if is90 else "o", s=150 if is90 else 26,
                              color=cols[sh], edgecolors="k",
                              linewidths=1.2 if is90 else 0.4, zorder=3 if is90 else 2)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("rotation period, s (log)")
    ax[0].set_ylabel("J_phi at the arm's own stop")
    ax[0].set_title("dotted: the Level-1 quasi-static prediction.  dash-dot: the joint\n"
                    "static winner.  red edge: the energy gate FAILS.", fontsize=8)
    ax[0].legend(fontsize=6, ncol=2)
    ax[0].grid(alpha=0.3)
    ax[1].axhline(5.0, color="k", lw=1.0, ls="--")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("number of rotation events over the horizon (log)")
    ax[1].set_ylabel("energy-balance residual, percent of dose")
    ax[1].set_title("stars are the 90-degree control: the remap is an exact pixel\n"
                    "permutation, so it carries no interpolation error.  "
                    "dashed line is the 5 percent gate.", fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.suptitle("Rotation speed: the quasi-static approximation against the engine's "
                 "remap error, true rotating engine, grid 120", fontsize=9)
    p = FIGS / "fig_rot_speed.png"
    fig.savefig(p, dpi=DPI)
    plt.close(fig)
    return p


def main(shapes=None):
    FIGS.mkdir(parents=True, exist_ok=True)
    shapes = list(shapes) if shapes else list(SHAPES)
    made = []
    made.append(fig_kernels(shapes))
    for sh in shapes:
        for fn in (fig_maps, fig_verify):
            p = fn(sh)
            if p is not None:
                made.append(p)
    for fn in (fig_speed, fig_headline):
        p = fn(shapes)
        if p is not None:
            made.append(p)
    for p in made:
        print(p)
    return made


if __name__ == "__main__":
    main(sys.argv[1:] or None)
