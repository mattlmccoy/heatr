"""Deck thumbnail: the cross, STATIC orientation + UNIFORM dopant, melt at stop.

The uncorrected baseline of the rotation story. One square panel, dark deck
style: the nominal cross as a cyan outline over a dim slate fill, the melt
fraction at the arm's own optimal stop burned in with the deck's inferno
melt/temperature colormap, and nothing else but the IoU.

DATA, not recomputed. The field is the STORED campaign result
`fgm_solve_campaign/out_rot_ladder/cross_ladder_maps.npz`, key
`g120_phi_STATIC_uniform`, written by `scripts/analysis/run_rot_grid_ladder.py`
as that arm's `_phi_at_stop` -- the melt fraction at the argmin of J over the
arm's OWN trajectory (t_stop 226.5 s, NOT the 750 s horizon). The IoU and J are
recomputed here from that field and gated against
`ROTATING_GRID_LADDER_REPORT.md` Section 3.1 (IoU 0.5515, J 463.25) before the
figure is written, so a wrong arm or a wrong grid cannot render silently.

HONESTY, for anyone quoting this slide. Grid 120 is the grid every dopant map in
this campaign was solved on, and on the cross it is simultaneously the ladder
MINIMUM of the static uniform arm and the ladder MAXIMUM of the rotating uniform
arm. The static-vs-rotating contrast is therefore at its grid-120-maximal here.
Rankings hold at all seven grids; the magnitudes are grid-120 numbers.

Run:  ./.venv312/bin/python deck_gifs/src/r7_cross_static_uniform.py
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import style  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]
MAPS = REPO / "fgm_solve_campaign/out_rot_ladder/cross_ladder_maps.npz"
OUT = REPO / "deck_gifs/fig_cross_static_uniform_thumb.png"

GRID = 120
ARM = "STATIC_uniform"
MELT_THRESHOLD = 0.5

# ROTATING_GRID_LADDER_REPORT.md Section 3.1, row: cross | 120 | STATIC_uniform
REPORT_IOU = 0.5515
REPORT_J = 463.25

PART_FILL = "#212934"   # dim slate: the nominal part, so cold limbs still read
PX = 600


@dataclass(frozen=True)
class Arm:
    """One stored ladder arm: melt fraction at its own stop, plus its geometry."""

    phi: np.ndarray
    part_mask: np.ndarray
    chi: np.ndarray
    x: np.ndarray
    y: np.ndarray


def load_arm(grid: int = GRID, arm: str = ARM) -> Arm:
    """Load the stored melt-at-stop field for one ladder arm."""
    try:
        d = np.load(MAPS)
    except FileNotFoundError:
        logger.error("stored ladder maps not found: %s", MAPS)
        raise
    g = f"g{grid}"
    return Arm(
        phi=np.asarray(d[f"{g}_phi_{arm}"], dtype=float),
        part_mask=np.asarray(d[f"{g}_part_mask"], dtype=bool),
        chi=np.asarray(d[f"{g}_chi"], dtype=float),
        x=np.asarray(d[f"{g}_x"], dtype=float),
        y=np.asarray(d[f"{g}_y"], dtype=float),
    )


def iou_binary(phi: np.ndarray, part_mask: np.ndarray,
               thr: float = MELT_THRESHOLD) -> float:
    """Intersection over union of the melted region (phi >= thr) with the part."""
    melt = phi >= thr
    union = int((melt | part_mask).sum())
    if union == 0:
        raise ValueError("empty union; melt and part mask are both empty")
    return float((melt & part_mask).sum()) / union


def j_whole_domain(phi: np.ndarray, chi: np.ndarray) -> float:
    """J = sum over the WHOLE domain of (phi - chi)^2, the campaign objective."""
    return float(((phi - chi) ** 2).sum())


def limb_melt_fractions(phi: np.ndarray, part_mask: np.ndarray,
                        thr: float = MELT_THRESHOLD) -> tuple[float, float]:
    """Melted fraction of the horizontal limbs and of the vertical limbs.

    The cross is partitioned from the mask itself, with no hard-coded geometry:
    the columns spanned by the tallest column are the vertical bar, the rows
    spanned by the widest row are the horizontal bar. A limb is the part of the
    bar outside the central overlap block.

    Returns:
        (horizontal_limb_melted_fraction, vertical_limb_melted_fraction).
    """
    col_h = part_mask.sum(axis=0)
    row_w = part_mask.sum(axis=1)
    vbar_cols = col_h >= col_h.max()          # the vertical bar's columns
    hbar_rows = row_w >= row_w.max()          # the horizontal bar's rows
    horiz = part_mask & ~vbar_cols[None, :]   # left and right limbs
    vert = part_mask & ~hbar_rows[:, None]    # top and bottom limbs
    melt = phi >= thr
    if horiz.sum() == 0 or vert.sum() == 0:
        raise ValueError("limb partition is empty; mask is not a cross")
    return (float((melt & horiz).sum()) / int(horiz.sum()),
            float((melt & vert).sum()) / int(vert.sum()))


def render(arm: Arm, iou: float, out: Path = OUT, px: int = PX) -> Path:
    """Write the square dark-style thumbnail."""
    r0, r1, c0, c1 = style.crop_indices(arm.part_mask, pad=6)
    phi = arm.phi[r0:r1, c0:c1]
    pm = arm.part_mask[r0:r1, c0:c1].astype(float)

    dpi = 100
    fig = plt.figure(figsize=(px / dpi, px / dpi), dpi=dpi)
    fig.patch.set_facecolor("black")
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_facecolor("black")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # 1. the nominal part as a dim slate fill, so the COLD limbs are visible
    part_rgba = np.zeros(pm.shape + (4,))
    part_rgba[..., :3] = np.array([0x21, 0x29, 0x34]) / 255.0
    part_rgba[..., 3] = pm
    ax.imshow(part_rgba, origin="lower", interpolation="bilinear")

    # 2. melt fraction at the stop, deck melt colormap, faded out where cold
    cmap = plt.get_cmap(style.CMAP_T)
    melt_rgba = cmap(np.clip(phi, 0.0, 1.0))
    melt_rgba[..., 3] = np.clip(phi, 0.0, 1.0) ** 0.5
    ax.imshow(melt_rgba, origin="lower", interpolation="bilinear")

    # 3. nominal outline, deck cyan
    ax.contour(pm, levels=[0.5], colors=[style.ACCENT], linewidths=[1.6])

    ax.text(0.965, 0.030, f"IoU {iou:.2f}", transform=ax.transAxes,
            ha="right", va="bottom", color=style.FG, fontsize=13,
            family=style.FAMILY)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, facecolor="black")
    plt.close(fig)
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    arm = load_arm()
    iou = iou_binary(arm.phi, arm.part_mask)
    j = j_whole_domain(arm.phi, arm.chi)
    horiz, vert = limb_melt_fractions(arm.phi, arm.part_mask)

    if abs(iou - REPORT_IOU) >= 5e-4 or abs(j - REPORT_J) / REPORT_J >= 5e-3:
        raise ValueError(
            f"stored arm does not reproduce the report: IoU {iou:.4f} vs "
            f"{REPORT_IOU}, J {j:.2f} vs {REPORT_J}")

    logger.info("cross grid %d %s: IoU %.4f  J %.2f  (report %.4f / %.2f) PASS",
                GRID, ARM, iou, j, REPORT_IOU, REPORT_J)
    logger.info("limb melt fraction: horizontal %.3f  vertical %.3f", horiz, vert)
    logger.info("wrote %s", render(arm, iou))


if __name__ == "__main__":
    main()
