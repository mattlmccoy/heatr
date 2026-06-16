#!/usr/bin/env python3
"""Compute the equal-cross-sectional-area linear size for each FGM shape.

Target area A* = area of a 20 mm-diameter circle = pi*(10mm)^2 ~= 314.159 mm^2.

For each shape we reproduce the solver's exact mask (make_shape + polygon_mask on
the 120x120 / 0.06 m grid), measure the filled area at the base width, then solve
for the width that yields A* (area scales with width^2 at fixed aspect ratio).

Reports the scaled width (mm), the resulting filled area, the thinnest feature
proxy (min filled-row/col run as a resolution check), and the cell size.
"""
import math
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rfam_eqs_coupled import _single_part_mask_and_fill  # noqa: E402

A_STAR_MM2 = math.pi * (10.0 ** 2)  # 314.159 mm^2

# shape -> (base_config_name, base_width_m, base_height_m, rotation_deg)
# square has no dedicated config; build from circle base with shape override.
SHAPES = {
    "circle":               ("shape_circle_6min.yaml", 0.020, 0.020, 0.0),
    "square":               ("shape_circle_6min.yaml", 0.020, 0.020, 0.0),
    "hexagon":              ("shape_hexagon_6min.yaml", 0.020, 0.020, 0.0),
    "pentagon":             ("shape_pentagon_6min.yaml", 0.020, 0.020, 0.0),
    "ellipse":              ("shape_ellipse_6min.yaml", 0.020, 0.012, 0.0),
    "octagon":              ("shape_octagon_6min.yaml", 0.020, 0.020, 0.0),
    "triangle":             ("shape_triangle_6min.yaml", 0.020, 0.020, 0.0),
    "diamond":              ("shape_diamond_6min.yaml", 0.020, 0.020, 0.0),
    "equilateral_triangle": ("shape_equilateral_triangle_6min.yaml", 0.020, 0.020, 0.0),
    "star":                 ("shape_star_6min.yaml", 0.022, 0.022, 0.0),
    "L_shape":              ("shape_L_shape_6min.yaml", 0.020, 0.028, 0.0),
    "cross":                ("shape_cross_6min.yaml", 0.022, 0.022, 0.0),
    "T_shape":              ("shape_T_shape_6min.yaml", 0.024, 0.028, 0.0),
}


def grid(chamber_x: float = 0.06, chamber_y: float = 0.06, nx: int = 120, ny: int = 120):
    x = np.linspace(-chamber_x / 2, chamber_x / 2, nx)
    y = np.linspace(-chamber_y / 2, chamber_y / 2, ny)
    return x, y


def filled_area_mm2(shape: str, w_m: float, h_m: float, rot: float, x, y) -> tuple[float, int]:
    part = {"shape": shape, "width": w_m, "height": h_m,
            "rotation_deg": rot, "center_x": 0.0, "center_y": 0.0,
            "n_circle_pts": 800}
    _poly, mask, _fill = _single_part_mask_and_fill(x, y, part)
    dx = float(x[1] - x[0]); dy = float(y[1] - y[0])
    cell_mm2 = dx * dy * 1e6
    n = int(mask.sum())
    return n * cell_mm2, n


def thin_feature_mm(mask_w_m: float, shape: str, h_m: float, rot: float, x, y) -> float:
    """Min filled run length (mm) across rows and cols — crude thinnest-feature proxy."""
    part = {"shape": shape, "width": mask_w_m, "height": h_m,
            "rotation_deg": rot, "center_x": 0.0, "center_y": 0.0,
            "n_circle_pts": 800}
    _poly, mask, _fill = _single_part_mask_and_fill(x, y, part)
    dx = float(x[1] - x[0]) * 1e3
    runs = []
    for row in mask:
        if row.any():
            runs.append(int(row.sum()))
    for col in mask.T:
        if col.any():
            runs.append(int(col.sum()))
    return (min(runs) * dx) if runs else 0.0


def main() -> None:
    x, y = grid()
    dx_mm = float(x[1] - x[0]) * 1e3
    print(f"# Target A* = {A_STAR_MM2:.2f} mm^2 (20 mm-dia circle)   cell = {dx_mm:.3f} mm")
    print(f"{'shape':22s} {'base_w_mm':>9s} {'base_A_mm2':>10s} {'scale':>6s} "
          f"{'new_w_mm':>8s} {'new_h_mm':>8s} {'new_A_mm2':>9s} {'thin_mm':>7s}")
    results = {}
    for shape, (_cfg, w0, h0, rot) in SHAPES.items():
        a0, _n = filled_area_mm2(shape, w0, h0, rot, x, y)
        if a0 <= 0:
            print(f"{shape:22s}  AREA=0 (mask build failed)")
            continue
        # Iterate scale so the discrete filled area converges to A* (<0.5%).
        scale = math.sqrt(A_STAR_MM2 / a0)
        a_new = a0
        for _ in range(8):
            w_try = w0 * scale
            h_try = h0 * scale
            a_new, _ = filled_area_mm2(shape, w_try, h_try, rot, x, y)
            if abs(a_new - A_STAR_MM2) / A_STAR_MM2 < 0.005:
                break
            scale *= math.sqrt(A_STAR_MM2 / a_new)
        w_new = w0 * scale
        h_new = h0 * scale
        thin = thin_feature_mm(w_new, shape, h_new, rot, x, y)
        results[shape] = {
            "base_w_mm": w0 * 1e3, "base_A_mm2": a0, "scale": scale,
            "new_w_mm": w_new * 1e3, "new_h_mm": h_new * 1e3,
            "new_A_mm2": a_new, "thin_mm": thin,
        }
        print(f"{shape:22s} {w0*1e3:9.3f} {a0:10.2f} {scale:6.3f} "
              f"{w_new*1e3:8.3f} {h_new*1e3:8.3f} {a_new:9.2f} {thin:7.3f}")
    import json
    (ROOT / "scripts_p0_rerun" / "equal_ca_scale.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
