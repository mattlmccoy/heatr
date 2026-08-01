#!/usr/bin/env python3
"""Figures for the JOINT per-angle dopant-map re-solve.

One composite per shape and actuator:

  row 1  J_phi against orientation, three curves at the same abscissae: the
         fixed-map sweep's uniform arm, the fixed-map sweep's graded arm (the
         zero-degree map rigidly rotated) and this campaign's per-angle
         re-solve. Stars mark each curve's argmin. This is the visual test of
         the registered prediction that the best angle MOVES once the map is
         re-solved.
  row 2  at the joint optimum: the delivered 4-bits-per-pixel dopant map, the
         melt-fraction field, and for contrast the fixed-map graded melt field
         at the sweep's own best angle.

Conventions carried from the library and orientation figures: cyan is the
rotated nominal part, dashed white is the melt front phi = 0.5, every field is
read at that arm's OWN J-stop.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402
import numpy as np                # noqa: E402

REPO = Path(__file__).resolve().parents[2]
JOINT = REPO / "fgm_solve_campaign/out_joint"
SWEEP = REPO / "outputs_eqs/orientation_optimization"
FIGS = REPO / "fgm_solve_campaign/figs_joint"
DPI = 180
EXTENT = (-30.0, 30.0, -30.0, 30.0)   # mm
LIM = 18.0


def sweep_rows(shape: str, arm: str) -> list[dict]:
    res = json.load(open(SWEEP / shape / "results.json"))
    rows = [r for r in res["rows"] if r["arm"] == arm]
    return sorted(rows, key=lambda r: r["angle_deg"])


def sweep_field(shape: str, angle: float, arm: str):
    tag = f"ang{angle:07.2f}_{arm}".replace(".", "p")
    return np.load(SWEEP / shape / "fields" / f"{tag}.npz")


def joint_field(shape: str, actuator: str, angle: float):
    tag = f"ang{angle:07.2f}".replace(".", "p")
    return np.load(JOINT / f"{shape}_{actuator}" / "fields" / f"{tag}.npz")


def _melt_panel(ax, phi, pm, title):
    ny, nx = phi.shape
    xs = np.linspace(EXTENT[0], EXTENT[1], nx)
    ys = np.linspace(EXTENT[2], EXTENT[3], ny)
    ax.imshow(phi, origin="lower", extent=EXTENT, vmin=0.0, vmax=1.0,
              cmap="inferno", interpolation="bilinear")
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#00e5ff",
               linewidths=1.4)
    ax.contour(xs, ys, phi, levels=[0.5], colors="#ffffff", linewidths=1.0,
               linestyles="--")
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xticks([])
    ax.set_yticks([])


def _map_panel(fig, ax, sat, pm, title):
    ny, nx = sat.shape
    xs = np.linspace(EXTENT[0], EXTENT[1], nx)
    ys = np.linspace(EXTENT[2], EXTENT[3], ny)
    shown = np.where(pm.astype(bool), sat, np.nan)
    im = ax.imshow(shown, origin="lower", extent=EXTENT, vmin=0.0, vmax=1.0,
                   cmap="viridis", interpolation="nearest")
    ax.contour(xs, ys, pm.astype(float), levels=[0.5], colors="#00e5ff",
               linewidths=1.2)
    ax.set_title(title, fontsize=8.5)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03,
                 label="dopant saturation s")


def make_figure(shape: str, actuator: str = "sigma") -> Path:
    res = json.load(open(JOINT / f"{shape}_{actuator}" / "results.json"))
    rows = sorted(res["rows"], key=lambda r: r["angle_deg"])
    ang_j = [r["angle_deg"] for r in rows]
    J_joint = [r["JOINT_4bpp"]["J"] for r in rows]
    J_joint_c = [r["JOINT_cont"]["J"] for r in rows]

    u = sweep_rows(shape, "uniform")
    g = sweep_rows(shape, "graded")
    ang_s = [r["angle_deg"] for r in u]
    J_u = [r["J"] for r in u]
    J_g = [r["J"] for r in g]

    a_joint = res["joint_best_angle_deg"]
    a_fixed = res["fixed_map_sweep_best_angle_deg"]
    move = res["angle_move_deg"]

    fig = plt.figure(figsize=(11.4, 8.6), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0])

    ax = fig.add_subplot(gs[0, :])
    ax.plot(ang_s, J_u, "o--", color="#888888", lw=1.4, ms=5,
            label="fixed map, uniform s = 1 (sweep)")
    ax.plot(ang_s, J_g, "s--", color="#d62728", lw=1.6, ms=6,
            label="fixed map, zero-degree map rigidly rotated (sweep)")
    ax.plot(ang_j, J_joint, "o-", color="#1f77b4", lw=2.2, ms=7,
            label="JOINT: map re-solved at every angle (4 bits per pixel)")
    ax.plot(ang_j, J_joint_c, ":", color="#1f77b4", lw=1.2, alpha=0.7,
            label="JOINT, continuous map before quantization")
    pr = JOINT / f"{shape}_{actuator}_refine" / "results_refine.json"
    if pr.exists():
        rf = json.loads(pr.read_text())
        ra = [r["angle_deg"] for r in sorted(rf["rows"], key=lambda x: x["angle_deg"])]
        rJ = [r["JOINT_4bpp"]["J"] for r in sorted(rf["rows"],
                                                   key=lambda x: x["angle_deg"])]
        ax.plot(ra, rJ, "D", color="#2ca02c", ms=9, mec="k", mew=0.6, zorder=6,
                label="JOINT depth check, 40 forward-equivalents per start")
    ax.plot([a_fixed], [J_g[ang_s.index(a_fixed)]], "*", color="#d62728",
            ms=20, mec="k", mew=0.6, zorder=5)
    ax.plot([a_joint], [J_joint[ang_j.index(a_joint)]], "*", color="#1f77b4",
            ms=22, mec="k", mew=0.6, zorder=5)
    verdict = ("the best angle MOVED" if abs(move) > 1e-9
               else "the best angle DID NOT move")
    depth = ""
    if pr.exists():
        rf2 = json.loads(pr.read_text())
        depth = (f"\ndepth check at 40 forward-equivalents per start on "
                 f"{', '.join(f'{a:g}' for a in rf2['angles_refined_deg'])} deg: "
                 f"best {rf2['refined_best_angle_deg']:g} deg, "
                 f"scan argmin {'HOLDS' if rf2['argmin_survives_depth'] else 'REVERSES'}")
    ax.set_title(
        f"{shape} ({'conductivity only' if actuator == 'sigma' else 'permittivity co-varying, MODEL ONLY'}): "
        f"{verdict} at the 15 forward-equivalent scan, {move:+.1f} deg "
        f"(fixed-map best {a_fixed:g} deg, joint best {a_joint:g} deg){depth}",
        fontsize=10.5)
    ax.set_xlabel("part rotation angle, degrees")
    ax.set_ylabel("J_phi at each arm's own J-stop")
    ax.set_xticks(ang_s)
    ax.grid(alpha=0.25)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.22 * (hi - lo))
    ax.legend(fontsize=8.5, loc="upper center", ncol=2, framealpha=0.95)

    d = joint_field(shape, actuator, a_joint)
    pm = d["part_mask"]
    r_joint = next(r for r in rows if r["angle_deg"] == a_joint)
    m = r_joint["JOINT_4bpp"]
    ax1 = fig.add_subplot(gs[1, 0])
    _map_panel(fig, ax1, d["sat_4bpp"], pm,
               f"JOINT solved map at {a_joint:g} deg\n"
               f"start {r_joint['winner_start']}, "
               f"{m.get('census_n_levels_used', 0)} printer levels")
    ax2 = fig.add_subplot(gs[1, 1])
    _melt_panel(ax2, d["phi_4bpp"], pm,
                f"JOINT melt at {a_joint:g} deg\n"
                f"J {m['J']:.1f}  IoU {m['IoU']:.4f}  stop {m['t_stop_s']:.0f} s"
                f"{' (H)' if m['t_stop_at_horizon'] else ''}\n"
                f"grow {m['bed_melt_pct_of_part']:.1f}%  "
                f"under {m['part_under_melt_pct']:.1f}%  "
                f"maxT {m['max_T_at_stop_c']:.0f} C")
    gs_row = next(r for r in g if r["angle_deg"] == a_fixed)
    ds = sweep_field(shape, a_fixed, "graded")
    ax3 = fig.add_subplot(gs[1, 2])
    _melt_panel(ax3, ds["phi_stop"], ds["part_mask"],
                f"FIXED-map graded melt at its own best {a_fixed:g} deg\n"
                f"J {gs_row['J']:.1f}  IoU {gs_row['IoU']:.4f}  "
                f"stop {gs_row['t_stop_s']:.0f} s"
                f"{' (H)' if gs_row['t_stop_at_horizon'] else ''}\n"
                f"grow {gs_row['bed_melt_pct_of_part']:.1f}%  "
                f"under {gs_row['part_under_melt_pct']:.1f}%")

    fig.suptitle(
        "Joint orientation and dopant-map optimization, grid 120, box [0, 1], "
        "design filter sigma 1.5 cells. Cyan is the rotated nominal part; "
        "dashed white is the melt front phi = 0.5.", fontsize=9.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / f"fig_joint_{shape}_{actuator}.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


def make_summary(pairs) -> Path:
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.0 * len(pairs), 3.9),
                             layout="constrained")
    axes = np.atleast_1d(axes)
    for ax, (shape, actuator) in zip(axes, pairs):
        res = json.load(open(JOINT / f"{shape}_{actuator}" / "results.json"))
        rows = sorted(res["rows"], key=lambda r: r["angle_deg"])
        ang = [r["angle_deg"] for r in rows]
        Jj = [r["JOINT_4bpp"]["J"] for r in rows]
        g = sweep_rows(shape, "graded")
        ax.plot([r["angle_deg"] for r in g], [r["J"] for r in g], "s--",
                color="#d62728", ms=5, lw=1.4, label="fixed map (sweep)")
        ax.plot(ang, Jj, "o-", color="#1f77b4", ms=6, lw=2.0, label="joint re-solve")
        ax.axvline(res["fixed_map_sweep_best_angle_deg"], color="#d62728",
                   ls=":", lw=1.2)
        ax.axvline(res["joint_best_angle_deg"], color="#1f77b4", ls=":", lw=1.2)
        pr = JOINT / f"{shape}_{actuator}_refine" / "results_refine.json"
        tail = ""
        if pr.exists():
            rf = json.loads(pr.read_text())
            ra = sorted(rf["rows"], key=lambda x: x["angle_deg"])
            ax.plot([r["angle_deg"] for r in ra],
                    [r["JOINT_4bpp"]["J"] for r in ra], "D", color="#2ca02c",
                    ms=7, mec="k", mew=0.5, zorder=6, label="depth check")
            tail = (f"\ndepth check best {rf['refined_best_angle_deg']:g} deg "
                    f"({'holds' if rf['argmin_survives_depth'] else 'REVERSES'})")
        ax.set_title(f"{shape} / {actuator}\nmove {res['angle_move_deg']:+.1f} deg, "
                     f"IoU {res['IoU_at_joint_best']:.4f}{tail}", fontsize=9.0)
        ax.set_xlabel("angle, degrees")
        ax.set_xticks(ang)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("J_phi at own J-stop")
    axes[0].legend(fontsize=8)
    fig.suptitle("Does the best orientation move when the dopant map is "
                 "re-solved at every angle? Grid 120, J_phi at each arm's own "
                 "J-stop.", fontsize=10.5)
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "fig_joint_summary.png"
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)
    return out


if __name__ == "__main__":
    args = sys.argv[1:]
    pairs = ([tuple(a.split(":")) for a in args] if args else
             [("T_shape", "sigma"), ("L_shape", "sigma"), ("cross", "sigma"),
              ("star", "sigma"), ("cross", "eps")])
    for sh, act in pairs:
        make_figure(sh, act)
    make_summary([p for p in pairs])
