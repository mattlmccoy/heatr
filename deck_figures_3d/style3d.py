"""Shared deck style for the true-3-D solve figures.

Same visual language as deck_gifs/src/style.py: near-black background,
Space Mono for all text (registered from ~/Library/Fonts, it is not in the
matplotlib system list), inferno for heat, thin cyan nominal outlines.
Rendering only; no physics is computed anywhere in this package.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np                   # noqa: E402
from matplotlib import font_manager  # noqa: E402
import matplotlib.pyplot as plt      # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent

BG = "#0e1013"          # near-black, not pure
PANEL = "#14171c"       # slightly lifted panel background
FG = "#e6e6e6"          # primary text
DIM = "#9aa0a6"         # secondary text
ACCENT = "#4dd0e1"      # nominal-outline cyan
WARM = "#ff8a65"        # warning / spill accent
GOOD = "#aed581"        # success accent
MELT = "#ffb74d"        # melt-region solid
SHELL = "#f4e04d"       # phi 0.8 translucent shell

CMAP_T = "inferno"
CMAP_PHI = "inferno"
CMAP_Q = "inferno"

DPI = 200


def register_space_mono() -> str:
    found = False
    for p in sorted(Path("~/Library/Fonts").expanduser().glob("SpaceMono-*.ttf")):
        font_manager.fontManager.addfont(str(p))
        found = True
    return "Space Mono" if found else "monospace"


FAMILY = register_space_mono()

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": FG,
    "axes.edgecolor": DIM,
    "axes.labelcolor": FG,
    "xtick.color": DIM,
    "ytick.color": DIM,
    "font.family": FAMILY,
    "font.size": 10,
})


def dark_3d_axes(ax) -> None:
    """Strip a mplot3d axes down to the dark deck look."""
    ax.set_facecolor(BG)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((0, 0, 0, 0))
        axis.line.set_color((1, 1, 1, 0.12))
        axis.set_ticklabels([])
        axis._axinfo["tick"]["inward_factor"] = 0.0
        axis._axinfo["tick"]["outward_factor"] = 0.0
        axis._axinfo["grid"]["color"] = (1, 1, 1, 0.05)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])


def shade_faces(verts: np.ndarray, faces: np.ndarray, base_rgb, light=(0.4, -0.6, 0.7),
                lo: float = 0.35, hi: float = 1.0) -> np.ndarray:
    """Simple Lambert shading for a marching-cubes mesh; returns per-face RGB."""
    tri = verts[faces]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(n, axis=1)
    norm[norm == 0] = 1.0
    n = n / norm[:, None]
    li = np.asarray(light, dtype=float)
    li = li / np.linalg.norm(li)
    lam = np.abs(n @ li)
    w = lo + (hi - lo) * lam
    base = np.asarray(matplotlib.colors.to_rgb(base_rgb))
    return np.clip(w[:, None] * base[None, :], 0, 1)


def title_block(fig, title: str, subtitle: str, x: float = 0.035, y: float = 0.96) -> None:
    fig.text(x, y, title, fontsize=17, color=FG, ha="left", va="top",
             fontweight="bold")
    fig.text(x, y - 0.052, subtitle, fontsize=10.5, color=DIM, ha="left", va="top")


def phase_fraction_phi(T: np.ndarray, t_pc_c: float = 180.0,
                       dt_pc_c: float = 10.0) -> np.ndarray:
    """heatr3d melt fraction, identical to solve3d.gates.phase_fraction_phi."""
    return np.clip((np.asarray(T, dtype=float) - t_pc_c) / dt_pc_c + 0.5, 0.0, 1.0)
