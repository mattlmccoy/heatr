"""Shared slide-deck style for the FGM solve story GIFs.

Dark near-black background, Space Mono for captions, thin light annotations,
repo visualization standard colormaps: inferno for temperature, magma for
relative density, viridis for dopant saturation.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

BG = "#0e1013"          # near-black, not pure
PANEL = "#14171c"       # slightly lifted panel background
FG = "#e6e6e6"          # primary text
DIM = "#9aa0a6"         # secondary text
ACCENT = "#4dd0e1"      # nominal-outline cyan
WARM = "#ff8a65"        # warning / stop accent
GOOD = "#aed581"        # success accent

CMAP_T = "inferno"      # temperature
CMAP_RHO = "magma"      # relative density
CMAP_SAT = "viridis"    # dopant saturation
CMAP_Q = "inferno"      # heating power density

FPS = 12
DPI = 100


def register_space_mono() -> str:
    """Register Space Mono from the user font library; return the family name."""
    found = False
    for p in sorted(Path("~/Library/Fonts").expanduser().glob("SpaceMono-*.ttf")):
        font_manager.fontManager.addfont(str(p))
        found = True
    if not found:
        return "monospace"
    return "Space Mono"


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


def crop_indices(pm: np.ndarray, pad: int = 12) -> tuple[int, int, int, int]:
    r = np.flatnonzero(pm.any(axis=1))
    c = np.flatnonzero(pm.any(axis=0))
    return (max(int(r[0]) - pad, 0), min(int(r[-1]) + pad + 1, pm.shape[0]),
            max(int(c[0]) - pad, 0), min(int(c[-1]) + pad + 1, pm.shape[1]))


def field_axes(ax, title: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=11, color=FG, pad=8)


def slim_colorbar(fig, im, ax, label: str):
    cb = fig.colorbar(im, ax=ax, fraction=0.043, pad=0.025)
    cb.ax.tick_params(labelsize=7, colors=DIM, length=2)
    cb.outline.set_edgecolor(DIM)
    cb.outline.set_linewidth(0.4)
    cb.set_label(label, fontsize=8, color=DIM)
    return cb
