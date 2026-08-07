"""Deck figure: Tamper.stl with and without dopant correction.

    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.render_tamper_result

HONEST SCOPE. Tamper is a thin plate (x,y +-22.7 mm, z +-5.5 mm), so a TOP-DOWN
(x,y) view of the per-cell dopant map faithfully shows the whole correction (the
z variation is small). This figure therefore shows the ACTUATOR that produced the
win - the dopant map the solve applied - not a 3-D melt-body isosurface. The
melt-body "part that would form" render needs node coordinates that the field
export did not save (only cell centroids); that is a mesh re-export, noted on the
figure. Left panel: real per-cell dopant (map_tamper_solve_filter_only.npz,
s_map on centroids), binned top-down. Right panel: the scalar outcome from
phase_e_tamper.json (uniform_baseline vs solve_filter_only). No physics is run
here; this is a render of committed solve artifacts.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "deck_figures_3d"))
import style3d as st  # noqa: E402

RESULTS = Path(__file__).resolve().parent / "results"
MM = 1000.0


def binned_topdown(centroids: np.ndarray, values: np.ndarray, nbin: int = 130):
    x = centroids[:, 0] * MM
    y = centroids[:, 1] * MM
    xe = np.linspace(x.min(), x.max(), nbin + 1)
    ye = np.linspace(y.min(), y.max(), nbin + 1)
    ssum, _, _ = np.histogram2d(x, y, bins=[xe, ye], weights=values)
    cnt, _, _ = np.histogram2d(x, y, bins=[xe, ye])
    with np.errstate(invalid="ignore"):
        avg = np.where(cnt > 0, ssum / np.maximum(cnt, 1), np.nan)
    return avg.T, (xe[0], xe[-1], ye[0], ye[-1])


def main() -> int:
    mp = np.load(RESULTS / "map_tamper_solve_filter_only.npz")
    doc = json.loads((RESULTS / "phase_e_tamper.json").read_text())
    u = doc["arms"]["uniform_baseline"]
    s = doc["arms"]["solve_filter_only"]

    centroids = mp["centroids"]
    s_solved = mp["s_map"]
    grid, extent = binned_topdown(centroids, s_solved)

    fig = plt.figure(figsize=(12.6, 6.2))
    st.title_block(
        fig,
        "Tamper: with and without dopant correction",
        "solved dopant map beats uniform by 45% - less melt outside bounds, more density inside",
    )

    # --- Left: the applied correction (top-down dopant map) ---
    axL = fig.add_axes([0.035, 0.13, 0.37, 0.68])
    axL.set_facecolor(st.BG)
    im = axL.imshow(grid, origin="lower", extent=extent, cmap="inferno",
                    vmin=0.0, vmax=1.0, interpolation="bilinear", aspect="equal")
    axL.set_title("Dopant correction, top view (Tamper is a ring)", color=st.FG, fontsize=13, pad=8)
    axL.set_xlabel("x [mm]", color=st.DIM, fontsize=10)
    axL.set_ylabel("y [mm]", color=st.DIM, fontsize=10)
    axL.tick_params(colors=st.DIM, labelsize=8)
    for sp in axL.spines.values():
        sp.set_color(st.DIM)
    # colorbar in its own slim axis, hugging the left panel (no collision downstream)
    cax = fig.add_axes([0.415, 0.13, 0.012, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("dopant s  (1 = uniform)", color=st.DIM, fontsize=9)
    cb.ax.yaxis.set_tick_params(color=st.DIM, labelsize=8)
    plt.setp(plt.getp(cb.ax.axes, "yticklabels"), color=st.DIM)

    # --- Right: the outcome (without vs with correction) ---
    axR = fig.add_axes([0.655, 0.18, 0.30, 0.58])
    axR.set_facecolor(st.BG)
    metrics = [
        ("melt outside\nbounds [% part]",
         u["out_of_part_melt_fraction_of_part"] * 100.0,
         s["out_of_part_melt_fraction_of_part"] * 100.0, "lower is better"),
        ("in-bounds below\ndensity floor [%]",
         u["in_bounds_below_floor_fraction"] * 100.0,
         s["in_bounds_below_floor_fraction"] * 100.0, "lower is better"),
        ("mean melt\nfraction [%]",
         u["part_mean_phi"] * 100.0,
         s["part_mean_phi"] * 100.0, "higher is better"),
    ]
    ypos = np.arange(len(metrics))[::-1]
    h = 0.34
    for i, (lab, uv, sv, _) in zip(ypos, metrics):
        axR.barh(i + h / 2, uv, height=h, color=st.WARM, label="without correction" if i == ypos[0] else None)
        axR.barh(i - h / 2, sv, height=h, color=st.GOOD, label="with correction" if i == ypos[0] else None)
        axR.text(uv, i + h / 2, f" {uv:.1f}", va="center", ha="left", color=st.WARM, fontsize=9)
        axR.text(sv, i - h / 2, f" {sv:.1f}", va="center", ha="left", color=st.GOOD, fontsize=9)
    axR.set_yticks(ypos)
    axR.set_yticklabels([m[0] for m in metrics], color=st.FG, fontsize=9.5)
    axR.set_xlabel("percent", color=st.DIM, fontsize=10)
    axR.set_xlim(0, 104)
    axR.tick_params(colors=st.DIM, labelsize=8)
    for sp in axR.spines.values():
        sp.set_color(st.DIM)
    axR.set_title("The result (optimized stop)", color=st.FG, fontsize=13, pad=8)
    axR.legend(loc="lower right", facecolor=st.PANEL, edgecolor=st.DIM, labelcolor=st.FG, fontsize=9)
    axR.text(0.5, -0.30,
             f"objective J {u['J_asymmetric']:.3e} -> {s['J_asymmetric']:.3e}  (-45.0%);  "
             f"peak {u['part_max_T_c']:.0f} -> {s['part_max_T_c']:.0f} C, both under 250",
             transform=axR.transAxes, color=st.DIM, fontsize=9, ha="center", va="top")

    fig.text(0.035, 0.028,
             "Actuator (dopant) shown top-down on the thin Tamper plate; the 3-D melt-body render is pending a mesh re-export. "
             "Real solve3d Phase-E arms, committed.",
             fontsize=7.5, color=st.DIM, ha="left", va="bottom")

    out = RESULTS / "fig_deck_tamper_result.png"
    fig.savefig(out, dpi=st.DPI)
    print(f"wrote {out}")
    print(f"grid non-nan bins: {np.isfinite(grid).sum()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
