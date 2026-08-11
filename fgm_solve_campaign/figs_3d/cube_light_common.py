"""Shared light-style plumbing for the two Phase E CUBE dissertation figures.

White background, no title inside the image, colourblind-safe hues, DejaVu
(matplotlib default) text. Same conventions as
``render_fig_cylinder_3d_solve.py``.

This module reads ONLY stored Phase E artifacts:

  solve3d/phase_e/results/phase_e_cube.json            per-arm scalars
  solve3d/phase_e/results/map_cube_solve_filter_only.npz   delivered map
  solve3d/phase_e/results/vol_cube.npz                 96^3 sampled read state
  solve3d/results/symmetry_retro_3d.json               symmetry-gate retro-check

No solve3d code is modified, no forward is re-run, no optimizer is invoked.
Every nodal field the figures need is already on disk (``vol_cube.npz`` holds
``T__<arm>`` for all three arms), so the Phase C export/forward-rerun pattern
is not needed here.

REPRODUCTION GATE. ``gate()`` recomputes, from the stored arrays, every scalar
the figures draw that also exists in the recorded JSON, and raises unless the
bit-identical checks come back at relative error 0 and the grid-versus-FEM
cross-checks stay inside their stated tolerance. Renderers call it before they
draw.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PE = ROOT / "solve3d" / "phase_e" / "results"
SYM_JSON = ROOT / "solve3d" / "results" / "symmetry_retro_3d.json"
GATE_TXT = HERE / "fig_cube_3d_gate.txt"

MM = 1000.0
ARMS = ("uniform_baseline", "heuristic_grading_law", "solve_filter_only")
ARM_LABEL = {"uniform_baseline": "no correction",
             "heuristic_grading_law": "hand-built grading law",
             "solve_filter_only": "direct solve"}

# ---- light palette -------------------------------------------------------- #
BG, FG, DIM = "#ffffff", "#1a1a1a", "#5a5f66"
NOMINAL = "#1a1a1a"        # nominal-geometry wireframe / outline, black
MELT_IN = "#e8b04b"        # melted material inside the nominal bounds (amber)
MELT_OUT = "#a02b1d"       # melted material outside them (dark red; differs in
                           # lightness as well as hue, so it survives greyscale)
C_UNI, C_SOL, C_HEU = "#8a9099", "#0277bd", "#b26a00"

RC = {
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": FG, "axes.edgecolor": DIM, "axes.labelcolor": FG,
    "xtick.color": DIM, "ytick.color": DIM, "font.size": 7.5,
    "axes.titlesize": 8.0, "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "legend.fontsize": 6.8,
    "axes.linewidth": 0.7, "axes.grid": False, "font.family": "sans-serif",
}


def use_light_style() -> None:
    """Apply the light rcParams.

    Must be called AFTER importing the solve3d.phase_e render helpers, because
    ``deck_figures_3d/style3d.py`` mutates rcParams to the dark deck theme at
    import time. Only pure geometry helpers are borrowed from those modules.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(matplotlib.rcParamsDefault)
    plt.rcParams.update(RC)


def phase_helpers():
    """Import the pure geometry/field helpers from the Phase E deck renderers."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from solve3d.phase_e import render_deck_cutaway3d as C     # noqa: E402
    from solve3d.phase_e import render_deck_loop3d as L        # noqa: E402
    from solve3d.phase_e import render_deck_melt_body as MB    # noqa: E402
    return C, L, MB


def light_3d_axes(ax) -> None:
    """Strip an mplot3d axes to a white-background, frameless look."""
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.line.set_color((0, 0, 0, 0))
        axis.set_ticklabels([])
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0)
        axis.set_ticks([])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)


def cube_wire(ax, half_mm: float, color: str = NOMINAL, lw: float = 0.8,
              alpha: float = 0.85) -> None:
    """The twelve edges of the nominal cube, drawn analytically."""
    for sx in (-1, 1):
        for sy in (-1, 1):
            ax.plot([sx * half_mm] * 2, [sy * half_mm] * 2,
                    [-half_mm, half_mm], color=color, lw=lw, alpha=alpha)
    for zz in (-half_mm, half_mm):
        ax.plot([-half_mm, half_mm, half_mm, -half_mm, -half_mm],
                [-half_mm, -half_mm, half_mm, half_mm, -half_mm], [zz] * 5,
                color=color, lw=lw, alpha=alpha)


# ---- data ----------------------------------------------------------------- #
def load() -> dict:
    doc = json.loads((PE / "phase_e_cube.json").read_text())
    vol = np.load(PE / "vol_cube.npz")
    mp = np.load(PE / "map_cube_solve_filter_only.npz")
    sym = json.loads(SYM_JSON.read_text())
    return {"doc": doc, "vol": vol, "map": mp, "sym": sym,
            "half_mm": float(vol["nominal_base_side_m"]) / 2.0 * MM}


def phi_of_T(T: np.ndarray) -> np.ndarray:
    """forward.phase_fraction, pointwise in T. Identical to solve3d.gates."""
    return np.clip((np.asarray(T, float) - 180.0) / 10.0 + 0.5, 0.0, 1.0)


# ---- reproduction gate ---------------------------------------------------- #
def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b) if b else abs(a - b)


def gate(d: dict, rtol_exact: float = 1e-12, rtol_grid: float = 0.03,
         write: bool = True) -> list[tuple[str, float, float, float, bool]]:
    """Recompute the drawn scalars from the stored arrays and check them.

    Two classes of check, kept apart on purpose:

    * EXACT: quantities the figure recomputes from the same arrays the scorer
      used (the delivered map and its cell volumes). These must reproduce the
      recorded JSON at relative error 0.
    * GRID: the volume-weighted mean melt fraction recomputed on the 96^3
      render grid against the FEM volume-weighted scorer. These are different
      quadratures of the same field, so they agree to a tolerance, not
      bit-for-bit. The tolerance is stated, and the measured gap is written to
      the gate file rather than hidden.
    """
    doc, mp, vol = d["doc"], d["map"], d["vol"]
    sol = doc["arms"]["solve_filter_only"]
    s, v = np.asarray(mp["s_map"], float), np.asarray(mp["volumes"], float)
    rows: list[tuple[str, float, float, float, bool]] = []

    def chk(name, got, want, tol):
        r = _rel(float(got), float(want))
        rows.append((name, float(got), float(want), r, r <= tol))

    chk("map_mean_volume_weighted", np.average(s, weights=v),
        sol["map_stats"]["mean"], rtol_exact)
    chk("map_min", s.min(), sol["map_stats"]["min"], rtol_exact)
    chk("map_max", s.max(), sol["map_stats"]["max"], rtol_exact)
    chk("n_design_cells", s.size, 24042, 0.0)

    uni = doc["arms"]["uniform_baseline"]
    heu = doc["arms"]["heuristic_grading_law"]
    # The two headline deltas, against PHASE_E_OPENER_REPORT.md section 7a
    # ("-46.09 %", "+50.00 % worse"). The reference values below are those
    # percentages at full precision, so this also gates the report table.
    chk("dJ_solve_percent", 100.0 * (sol["J_asymmetric"] / uni["J_asymmetric"] - 1.0),
        -46.0861458043, 1e-9)
    chk("dJ_heuristic_percent",
        100.0 * (heu["J_asymmetric"] / uni["J_asymmetric"] - 1.0),
        49.9975737291, 1e-9)

    ins = np.asarray(vol["inside"], bool)
    for arm in ARMS:
        chk(f"grid_mean_phi[{arm}]", phi_of_T(vol[f"T__{arm}"])[ins].mean(),
            doc["arms"][arm]["part_mean_phi"], rtol_grid)
        chk(f"sampler_missed_in_part[{arm}]",
            float(vol[f"missed_in_part__{arm}"]), 0.0, 0.0)

    sym = d["sym"]["results"]["cube"]
    chk("symmetry_frac_xz", sym["frac_xz_corrected"], 0.9121, 1e-4)
    chk("symmetry_frac_xyz", sym["frac_full_xyz"], 0.9063, 1e-4)

    if write:
        lines = ["Phase E cube light-figure reproduction gate",
                 "recomputed from stored artifacts at render time; no solve, "
                 "no forward re-run",
                 f"exact rtol {rtol_exact:g}, grid-vs-FEM rtol {rtol_grid:g}",
                 ""]
        w = max(len(r[0]) for r in rows)
        for nm, got, want, r, ok in rows:
            lines.append(f"{nm:<{w}}  recomputed {got: .12e}  stored "
                         f"{want: .12e}  rel {r: .3e}  "
                         f"{'PASS' if ok else 'FAIL'}")
        GATE_TXT.write_text("\n".join(lines) + "\n")

    bad = [r for r in rows if not r[4]]
    if bad:
        raise SystemExit("reproduction gate FAILED: "
                         + "; ".join(f"{r[0]} rel {r[3]:.3e}" for r in bad))
    return rows


if __name__ == "__main__":
    for nm, got, want, r, ok in gate(load()):
        print(f"{'PASS' if ok else 'FAIL'}  {nm:<34} rel {r:.3e}")
    print("gate file:", GATE_TXT)
