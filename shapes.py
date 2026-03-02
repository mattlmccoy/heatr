"""shapes.py — Geometry primitives for the RFAM EQS simulation.

Provides make_shape() which returns an (N, 2) closed polygon array in metres,
centered at the origin. All widths/heights are in metres.

Supported shape names (case-insensitive):
  square, rectangle / rect, circle, ellipse, triangle,
  equilateral_triangle, diamond / rhombus, hexagon / hex,
  octagon, pentagon, star / star5, star6, star8,
  cross / plus, rounded_square / rounded_rect,
  L_shape / L, T_shape / T, arrow, trapezoid,
  from_svg / svg  (pass svg_path= kwarg)
  from_image / bitmap  (pass image_path= kwarg)
  polygon / poly  (pass polygon_points= kwarg — list of [x,y] in metres)
"""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
from matplotlib.path import Path as MplPath


# ─────────────────────────────────────────────────────────────────────────────
# Core utilities
# ─────────────────────────────────────────────────────────────────────────────

def rotate(poly: np.ndarray, theta_rad: float) -> np.ndarray:
    """Rotate polygon CCW by theta_rad around the origin."""
    c, s = math.cos(theta_rad), math.sin(theta_rad)
    return poly @ np.array([[c, -s], [s, c]], dtype=float).T


def _close(poly: np.ndarray) -> np.ndarray:
    if not np.allclose(poly[0], poly[-1]):
        poly = np.vstack([poly, poly[0]])
    return poly


def resample_polygon(poly: np.ndarray, n: int) -> np.ndarray:
    """Resample polygon to n evenly-spaced points along its perimeter."""
    c = _close(poly)
    seg = c[1:] - c[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = s[-1]
    if total <= 0:
        return np.repeat(poly[:1], n, axis=0)
    t_vals = np.linspace(0.0, total, n, endpoint=False)
    out = np.zeros((n, 2), dtype=float)
    j = 0
    for i, t in enumerate(t_vals):
        while j < len(seg_len) - 1 and s[j + 1] < t:
            j += 1
        dt = (t - s[j]) / max(seg_len[j], 1e-12)
        out[i] = c[j] + dt * seg[j]
    return out


def polygon_mask(poly: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rasterise polygon onto (x, y) grid. Returns boolean mask [ny, nx]."""
    xx, yy = np.meshgrid(x, y, indexing="xy")
    pts = np.column_stack([xx.ravel(), yy.ravel()])
    mask = MplPath(poly).contains_points(pts)
    return mask.reshape(xx.shape)


def _center_and_scale(poly: np.ndarray,
                       target_width: float,
                       target_height: Optional[float] = None) -> np.ndarray:
    """Center polygon at origin and scale so its bounding box fits target_width × target_height."""
    lo = poly.min(axis=0)
    hi = poly.max(axis=0)
    span = hi - lo
    poly = poly - (lo + span / 2.0)          # center
    sx = target_width / max(span[0], 1e-12)
    sy = (target_height / max(span[1], 1e-12)) if target_height is not None else sx
    poly = poly * np.array([sx, sy])
    return poly


# ─────────────────────────────────────────────────────────────────────────────
# Primitive shape constructors
# ─────────────────────────────────────────────────────────────────────────────

def make_rect(width: float, height: float) -> np.ndarray:
    w, h = width / 2.0, height / 2.0
    return np.array([[-w, -h], [w, -h], [w, h], [-w, h]], dtype=float)


def make_regular_ngon(radius: float, n_sides: int, phase_deg: float = 0.0) -> np.ndarray:
    ang = np.linspace(0.0, 2.0 * math.pi, n_sides, endpoint=False) + math.radians(phase_deg)
    return np.column_stack([radius * np.cos(ang), radius * np.sin(ang)])


def make_triangle(width: float, height: float) -> np.ndarray:
    """Isosceles triangle pointing up."""
    w, h = width / 2.0, height / 2.0
    return np.array([[-w, -h], [w, -h], [0.0, h]], dtype=float)


def make_equilateral_triangle(side: float) -> np.ndarray:
    """Equilateral triangle with given side length, pointing up."""
    h = side * math.sqrt(3.0) / 2.0
    return make_triangle(side, h)


def make_star(outer_r: float,
              inner_r: Optional[float] = None,
              n_points: int = 5,
              phase_deg: float = 90.0) -> np.ndarray:
    """N-pointed star. inner_r defaults to the 'natural' star ratio."""
    if inner_r is None:
        # Natural ratio for a regular star polygon
        inner_r = outer_r * math.cos(math.radians(180.0 / n_points)) / math.cos(math.radians(90.0 / n_points))
        inner_r = max(inner_r, outer_r * 0.25)
    outer_ang = (np.linspace(0, 2 * math.pi, n_points, endpoint=False)
                 + math.radians(phase_deg))
    inner_ang = outer_ang + math.pi / n_points
    outer_pts = np.column_stack([outer_r * np.cos(outer_ang), outer_r * np.sin(outer_ang)])
    inner_pts = np.column_stack([inner_r * np.cos(inner_ang), inner_r * np.sin(inner_ang)])
    pts = np.empty((2 * n_points, 2), dtype=float)
    pts[0::2] = outer_pts
    pts[1::2] = inner_pts
    return pts


def make_cross(width: float, thickness: float) -> np.ndarray:
    """Plus / cross shape. width = total arm span, thickness = arm width."""
    t, w = thickness / 2.0, width / 2.0
    return np.array([
        [-t, -w], [-t, -t], [-w, -t], [-w,  t],
        [-t,  t], [-t,  w], [ t,  w], [ t,  t],
        [ w,  t], [ w, -t], [ t, -t], [ t, -w],
    ], dtype=float)


def make_rounded_rect(width: float, height: float,
                       radius: float, n_arc: int = 10) -> np.ndarray:
    """Rectangle with rounded corners. radius clipped to half the shorter side."""
    radius = min(radius, min(width, height) / 2.0 - 1e-9)
    w, h = width / 2.0 - radius, height / 2.0 - radius
    corners = [
        (-w, -h, math.pi,        1.5 * math.pi),
        ( w, -h, 1.5 * math.pi, 2.0 * math.pi),
        ( w,  h, 0.0,            0.5 * math.pi),
        (-w,  h, 0.5 * math.pi, math.pi),
    ]
    pts = []
    for cx, cy, a0, a1 in corners:
        angs = np.linspace(a0, a1, n_arc + 1, endpoint=True)
        pts.extend([[cx + radius * math.cos(a), cy + radius * math.sin(a)] for a in angs])
    return np.array(pts, dtype=float)


def make_L_shape(width: float, height: float, thickness: float) -> np.ndarray:
    """L-shape. thickness = arm thickness. Centered at bounding-box centroid."""
    t = thickness
    poly = np.array([
        [0.0,    0.0],
        [width,  0.0],
        [width,  t],
        [t,      t],
        [t,      height],
        [0.0,    height],
    ], dtype=float)
    return poly - np.array([width / 2.0, height / 2.0])


def make_T_shape(width: float, height: float, thickness: float) -> np.ndarray:
    """T-shape (top bar horizontal). Centered at bounding-box centroid."""
    t = thickness
    stem_x = (width - t) / 2.0
    poly = np.array([
        [0.0,   0.0],
        [width, 0.0],
        [width, t],
        [stem_x + t, t],
        [stem_x + t, height],
        [stem_x, height],
        [stem_x, t],
        [0.0,   t],
    ], dtype=float)
    return poly - np.array([width / 2.0, height / 2.0])


def make_H_shape(width: float, height: float, leg_width: float) -> np.ndarray:
    """H-shape (letter H).

    Parameters
    ----------
    width     : total bounding-box width [m]
    height    : total bounding-box height [m]
    leg_width : width of each vertical leg [m]
    crossbar  : crossbar height is 30% of total height (centred)

    Vertex order: CCW from bottom-left, 12 vertices, centred at origin.
    """
    hw   = width  / 2.0
    hh   = height / 2.0
    g    = hw - leg_width          # x-coord of inner vertical edges (±g)
    cb   = height * 0.15           # half-height of crossbar (±cb → total = 30% of height)
    poly = np.array([
        [-hw, -hh],   #  0 outer bottom-left
        [-g,  -hh],   #  1 inner bottom-left
        [-g,  -cb],   #  2 inner, bottom of crossbar (left)
        [+g,  -cb],   #  3 inner, bottom of crossbar (right)
        [+g,  -hh],   #  4 inner bottom-right
        [+hw, -hh],   #  5 outer bottom-right
        [+hw, +hh],   #  6 outer top-right
        [+g,  +hh],   #  7 inner top-right
        [+g,  +cb],   #  8 inner, top of crossbar (right)
        [-g,  +cb],   #  9 inner, top of crossbar (left)
        [-g,  +hh],   # 10 inner top-left
        [-hw, +hh],   # 11 outer top-left
    ], dtype=float)
    return poly


def make_arrow(length: float, head_width: float,
               shaft_width: float, head_length: float) -> np.ndarray:
    """Right-pointing arrow. Centered at bounding-box centroid."""
    sl = length - head_length
    sw = shaft_width / 2.0
    hw = head_width / 2.0
    poly = np.array([
        [0.0,  -sw],
        [sl,   -sw],
        [sl,   -hw],
        [length, 0.0],
        [sl,    hw],
        [sl,    sw],
        [0.0,   sw],
    ], dtype=float)
    return poly - np.array([length / 2.0, 0.0])


def make_trapezoid(width_bottom: float, width_top: float, height: float) -> np.ndarray:
    wb, wt, h = width_bottom / 2.0, width_top / 2.0, height / 2.0
    return np.array([[-wb, -h], [wb, -h], [wt, h], [-wt, h]], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# SVG path import
# ─────────────────────────────────────────────────────────────────────────────

def _parse_svg_numbers(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", s)]


def _cubic_bezier(p0, p1, p2, p3, n: int = 12) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def _quadratic_bezier(p0, p1, p2, n: int = 8) -> np.ndarray:
    t = np.linspace(0, 1, n, endpoint=False)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def _svg_arc(x1, y1, rx, ry, phi_deg, large_arc, sweep, x2, y2, n=16):
    """Convert SVG arc to polyline points (endpoint parameterization → center form)."""
    phi = math.radians(phi_deg)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (x1 - x2) / 2.0, (y1 - y2) / 2.0
    x1p =  cp * dx + sp * dy
    y1p = -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = (x1p / rx) ** 2 + (y1p / ry) ** 2
    if lam > 1.0:
        rx *= math.sqrt(lam); ry *= math.sqrt(lam)
    num = max(0.0, rx**2 * ry**2 - rx**2 * y1p**2 - ry**2 * x1p**2)
    den = rx**2 * y1p**2 + ry**2 * x1p**2
    sq = math.sqrt(num / max(den, 1e-12))
    if large_arc == sweep:
        sq = -sq
    cxp =  sq * rx * y1p / ry
    cyp = -sq * ry * x1p / rx
    cx = cp * cxp - sp * cyp + (x1 + x2) / 2.0
    cy = sp * cxp + cp * cyp + (y1 + y2) / 2.0

    def ang(ux, uy, vx, vy):
        a = math.atan2(ux * vy - uy * vx, ux * vx + uy * vy)
        return a

    theta1 = ang(1, 0, (x1p - cxp) / rx, (y1p - cyp) / ry)
    dtheta = ang((x1p - cxp) / rx, (y1p - cyp) / ry,
                 (-x1p - cxp) / rx, (-y1p - cyp) / ry)
    if not sweep and dtheta > 0:
        dtheta -= 2 * math.pi
    if sweep and dtheta < 0:
        dtheta += 2 * math.pi

    ts = np.linspace(theta1, theta1 + dtheta, n, endpoint=False)
    xs = cx + rx * np.cos(ts) * cp - ry * np.sin(ts) * sp
    ys = cy + rx * np.cos(ts) * sp + ry * np.sin(ts) * cp
    return np.column_stack([xs, ys])


def _parse_svg_path_d(d: str, curve_pts: int = 12) -> list[np.ndarray]:
    """Parse SVG 'd' attribute → list of sub-path polygon arrays."""
    # Tokenise: split on command letters, keep letters as tokens
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", d)
    subpaths, pts = [], []
    x = y = 0.0
    x0 = y0 = 0.0    # subpath start
    last_ctrl = None
    last_cmd = None
    i = 0

    def consume(n):
        nonlocal i
        vs = [float(tokens[i + k]) for k in range(n)]
        i += n
        return vs

    while i < len(tokens):
        tok = tokens[i]
        if re.match(r"[MmLlHhVvCcSsQqTtAaZz]", tok):
            cmd = tok
            i += 1
        else:
            # implicit repeat of last command
            cmd = last_cmd if last_cmd else "L"

        if cmd in ("M", "m"):
            if pts:
                subpaths.append(np.array(pts, dtype=float)); pts = []
            dx, dy = consume(2)
            x, y = (dx, dy) if cmd == "M" else (x + dx, y + dy)
            x0, y0 = x, y
            pts = [[x, y]]
            last_cmd = "L" if cmd == "M" else "l"
        elif cmd in ("Z", "z"):
            if pts:
                pts.append([x0, y0])
                subpaths.append(np.array(pts, dtype=float)); pts = []
            x, y = x0, y0
            last_cmd = cmd
        elif cmd in ("L", "l"):
            dx, dy = consume(2)
            x, y = (dx, dy) if cmd == "L" else (x + dx, y + dy)
            pts.append([x, y])
            last_cmd = cmd
        elif cmd in ("H", "h"):
            dx, = consume(1)
            x = dx if cmd == "H" else x + dx
            pts.append([x, y])
            last_cmd = cmd
        elif cmd in ("V", "v"):
            dy, = consume(1)
            y = dy if cmd == "V" else y + dy
            pts.append([x, y])
            last_cmd = cmd
        elif cmd in ("C", "c"):
            cx1, cy1, cx2, cy2, ex, ey = consume(6)
            if cmd == "c":
                cx1 += x; cy1 += y; cx2 += x; cy2 += y; ex += x; ey += y
            seg = _cubic_bezier(np.array([x, y]), np.array([cx1, cy1]),
                                 np.array([cx2, cy2]), np.array([ex, ey]),
                                 n=curve_pts)
            pts.extend(seg.tolist())
            last_ctrl = np.array([cx2, cy2])
            x, y = ex, ey
            last_cmd = cmd
        elif cmd in ("S", "s"):
            cx2, cy2, ex, ey = consume(4)
            if cmd == "s":
                cx2 += x; cy2 += y; ex += x; ey += y
            if last_ctrl is not None and last_cmd.lower() in ("c", "s"):
                cx1 = 2 * x - last_ctrl[0]; cy1 = 2 * y - last_ctrl[1]
            else:
                cx1, cy1 = x, y
            seg = _cubic_bezier(np.array([x, y]), np.array([cx1, cy1]),
                                 np.array([cx2, cy2]), np.array([ex, ey]),
                                 n=curve_pts)
            pts.extend(seg.tolist())
            last_ctrl = np.array([cx2, cy2])
            x, y = ex, ey
            last_cmd = cmd
        elif cmd in ("Q", "q"):
            cx1, cy1, ex, ey = consume(4)
            if cmd == "q":
                cx1 += x; cy1 += y; ex += x; ey += y
            seg = _quadratic_bezier(np.array([x, y]), np.array([cx1, cy1]),
                                     np.array([ex, ey]), n=curve_pts)
            pts.extend(seg.tolist())
            last_ctrl = np.array([cx1, cy1])
            x, y = ex, ey
            last_cmd = cmd
        elif cmd in ("T", "t"):
            ex, ey = consume(2)
            if cmd == "t":
                ex += x; ey += y
            if last_ctrl is not None and last_cmd.lower() in ("q", "t"):
                cx1 = 2 * x - last_ctrl[0]; cy1 = 2 * y - last_ctrl[1]
            else:
                cx1, cy1 = x, y
            seg = _quadratic_bezier(np.array([x, y]), np.array([cx1, cy1]),
                                     np.array([ex, ey]), n=curve_pts)
            pts.extend(seg.tolist())
            last_ctrl = np.array([cx1, cy1])
            x, y = ex, ey
            last_cmd = cmd
        elif cmd in ("A", "a"):
            rx, ry, phi, la, sw, ex, ey = consume(7)
            if cmd == "a":
                ex += x; ey += y
            try:
                seg = _svg_arc(x, y, rx, ry, phi, int(la), int(sw), ex, ey, n=curve_pts)
                pts.extend(seg.tolist())
            except Exception:
                pass
            x, y = ex, ey
            last_cmd = cmd
        else:
            i += 1  # skip unknown

    if pts:
        subpaths.append(np.array(pts, dtype=float))
    return [s for s in subpaths if len(s) >= 3]


def make_shape_from_svg(svg_path: str,
                         target_width_m: float,
                         target_height_m: Optional[float] = None,
                         n_pts: int = 500,
                         curve_pts: int = 20,
                         path_index: int = 0,
                         raster_res: int = 1024) -> np.ndarray:
    """
    Load an SVG file and return a polygon scaled to target_width_m (metres),
    centered at origin.

    Handles multi-path SVGs (e.g. logos with multiple colour regions) by
    rasterising ALL paths onto a high-res bitmap, taking their union, then
    extracting the largest outer contour with cv2. This correctly reproduces
    complex outlines like the GT Yellow Jackets logo.

    Parameters
    ----------
    svg_path        : path to .svg file
    target_width_m  : desired bounding-box width in metres
    target_height_m : desired bounding-box height (None = preserve aspect ratio)
    n_pts           : polygon resolution after resampling
    curve_pts       : bezier segments per curve during parsing
    path_index      : reserved (always uses largest contour)
    raster_res      : internal bitmap resolution for rasterisation
    """
    import cv2
    from xml.etree import ElementTree as ET

    svg_path = Path(svg_path)
    if not svg_path.exists():
        raise FileNotFoundError(f"SVG not found: {svg_path}")

    tree = ET.parse(svg_path)
    root = tree.getroot()

    # Parse viewBox to get SVG coordinate extent
    vb = root.get("viewBox", "")
    if vb:
        vb_vals = [float(v) for v in vb.split()]
        svg_x0, svg_y0, svg_w, svg_h = vb_vals
    else:
        svg_w = float(root.get("width", 100))
        svg_h = float(root.get("height", 100))
        svg_x0, svg_y0 = 0.0, 0.0

    # Collect all path 'd' attributes
    all_d = []
    for elem in root.iter():
        tag = elem.tag.split("}")[-1]
        if tag == "path":
            d = elem.get("d", "")
            if d:
                all_d.append(d)

    if not all_d:
        raise ValueError(f"No <path> elements found in {svg_path}")

    # Rasterise all paths onto a bitmap using matplotlib (handles winding correctly)
    scale = raster_res / max(svg_w, svg_h)
    bmp = np.zeros((raster_res, raster_res), dtype=np.uint8)

    for d in all_d:
        subpaths = _parse_svg_path_d(d, curve_pts=curve_pts)
        for sp in subpaths:
            if len(sp) < 3:
                continue
            # Map SVG coords → pixel coords (y flipped)
            px = ((sp[:, 0] - svg_x0) * scale).astype(np.int32)
            py = ((svg_h - (sp[:, 1] - svg_y0)) * scale).astype(np.int32)
            px = np.clip(px, 0, raster_res - 1)
            py = np.clip(py, 0, raster_res - 1)
            pts_cv = np.column_stack([px, py]).reshape(-1, 1, 2)
            cv2.fillPoly(bmp, [pts_cv], 255)

    # Find the largest outer contour
    contours, _ = cv2.findContours(bmp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No contours found after rasterising {svg_path}")

    cnt = max(contours, key=cv2.contourArea)
    poly = cnt.squeeze().astype(float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("Could not extract a 2-D contour from SVG")

    # Convert pixel → SVG coords, flip y back to our convention
    poly[:, 0] = poly[:, 0] / scale + svg_x0
    poly[:, 1] = svg_h - poly[:, 1] / scale + svg_y0
    poly[:, 1] = -poly[:, 1]   # flip y for simulation convention

    # Scale, center, resample
    poly = _center_and_scale(poly, target_width_m, target_height_m)
    poly = resample_polygon(poly, n_pts)
    return poly


def make_shape_from_image(image_path: str,
                           target_width_m: float,
                           target_height_m: Optional[float] = None,
                           threshold: int = 128,
                           n_pts: int = 500) -> np.ndarray:
    """
    Rasterise a bitmap (PNG/JPG/etc) to a polygon mask by finding the largest
    contour of the dark/opaque region. Scales to target_width_m.
    """
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is not None and img.ndim == 3:
            # Use alpha channel if available
            img = img[:, :, 3] if img.shape[2] == 4 else img[:, :, 0]
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError(f"No contours found in {image_path}")

    # Largest contour
    cnt = max(contours, key=cv2.contourArea)
    poly = cnt.squeeze().astype(float)
    if poly.ndim != 2 or poly.shape[1] != 2:
        raise ValueError("Could not extract a 2-D contour")

    poly[:, 1] = -poly[:, 1]   # flip y (image coords)
    poly = _center_and_scale(poly, target_width_m, target_height_m)
    poly = resample_polygon(poly, n_pts)
    return poly


# ─────────────────────────────────────────────────────────────────────────────
# Main dispatch
# ─────────────────────────────────────────────────────────────────────────────

def make_shape(shape: str,
               width: float,
               height: Optional[float] = None,
               n_circle_pts: int = 220,
               # Extra kwargs for special shapes
               thickness: Optional[float] = None,
               inner_radius_ratio: Optional[float] = None,
               n_points: Optional[int] = None,
               corner_radius: Optional[float] = None,
               svg_path: Optional[str] = None,
               image_path: Optional[str] = None,
               **kwargs) -> np.ndarray:
    """
    Return a centred (N, 2) polygon for the named shape (coords in metres).

    Parameters
    ----------
    shape   : shape name string (see module docstring)
    width   : bounding-box width [m]
    height  : bounding-box height [m] (defaults to width for symmetric shapes)
    n_circle_pts : resolution for curved shapes
    thickness    : arm/stem thickness for cross, L, T shapes (default width/4)
    inner_radius_ratio : inner/outer ratio for star shapes (default ~0.382)
    n_points     : number of star points (default 5)
    corner_radius: corner radius for rounded_rect (default width/6)
    svg_path     : path to SVG file (required for shape='from_svg')
    image_path   : path to image file (required for shape='from_image')
    """
    if height is None:
        height = width
    s = shape.strip().lower().replace("-", "_").replace(" ", "_")

    # ── rectangles ──────────────────────────────────────────────────────────
    if s in {"square", "cube"}:
        return make_rect(width, width)
    if s in {"rect", "rectangle"}:
        return make_rect(width, height)

    # ── circles / ellipses ──────────────────────────────────────────────────
    if s in {"circle", "disk", "cylinder", "sphere"}:
        return make_regular_ngon(width / 2.0, max(24, n_circle_pts), phase_deg=90.0)
    if s in {"ellipse", "oval"}:
        poly = make_regular_ngon(1.0, max(24, n_circle_pts), phase_deg=90.0)
        poly[:, 0] *= width / 2.0
        poly[:, 1] *= height / 2.0
        return poly

    # ── triangles ────────────────────────────────────────────────────────────
    if s in {"triangle", "tri", "isosceles", "pyramid", "tri_prism"}:
        return make_triangle(width, height)
    if s in {"equilateral_triangle", "equilateral", "equitri"}:
        return make_equilateral_triangle(width)

    # ── polygons ─────────────────────────────────────────────────────────────
    if s in {"diamond", "rhombus"}:
        return rotate(make_rect(width, height), math.pi / 4.0)
    if s in {"hexagon", "hex"}:
        return make_regular_ngon(width / 2.0, 6, phase_deg=30.0)
    if s in {"pentagon"}:
        return make_regular_ngon(width / 2.0, 5, phase_deg=90.0)
    if s in {"octagon"}:
        return make_regular_ngon(width / 2.0, 8, phase_deg=22.5)
    if s in {"trapezoid"}:
        return make_trapezoid(width_bottom=width, width_top=0.5 * width, height=height)

    # ── stars ────────────────────────────────────────────────────────────────
    if s in {"star", "star5"}:
        ir = (width / 2.0) * (inner_radius_ratio or 0.382)
        return make_star(width / 2.0, ir, n_points=n_points or 5, phase_deg=90.0)
    if s in {"star6"}:
        ir = (width / 2.0) * (inner_radius_ratio or 0.5)
        return make_star(width / 2.0, ir, n_points=6, phase_deg=90.0)
    if s in {"star8"}:
        ir = (width / 2.0) * (inner_radius_ratio or 0.414)
        return make_star(width / 2.0, ir, n_points=8, phase_deg=22.5)

    # ── compound shapes ──────────────────────────────────────────────────────
    if s in {"cross", "plus"}:
        t = thickness or width / 3.0
        return make_cross(width, t)
    if s in {"rounded_square", "rounded_rect", "rounded_rectangle"}:
        r = corner_radius or width / 6.0
        return make_rounded_rect(width, height, r, n_arc=max(6, n_circle_pts // 16))
    if s in {"l_shape", "l"}:
        t = thickness or width / 3.0
        return make_L_shape(width, height, t)
    if s in {"t_shape", "t"}:
        t = thickness or width / 4.0
        return make_T_shape(width, height, t)
    if s in {"h_shape", "h"}:
        t = thickness or width * 0.30   # leg width; default = 30% of total width
        return make_H_shape(width, height, t)
    if s in {"arrow"}:
        hl = height or width * 0.4
        hw = width * 0.5
        sw = hw * 0.4
        return make_arrow(width, hw, sw, hl)

    # ── inline polygon (used by rfam_prewarp.py for iterative injection) ─────
    if s in {"polygon", "poly"}:
        pts = kwargs.get("polygon_points")
        if pts is None:
            raise ValueError("shape='polygon' requires polygon_points= kwarg "
                             "(list of [x,y] in metres)")
        pts = np.array(pts, dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError("polygon_points must be an (N, 2) array of [x, y] pairs")
        if not np.allclose(pts[0], pts[-1]):   # ensure closed ring
            pts = np.vstack([pts, pts[0]])
        return pts

    # ── SVG / image import ───────────────────────────────────────────────────
    if s in {"from_svg", "svg"}:
        if svg_path is None:
            raise ValueError("shape='from_svg' requires svg_path= kwarg")
        return make_shape_from_svg(svg_path, width,
                                    target_height_m=height if height != width else None,
                                    n_pts=n_circle_pts,
                                    **{k: v for k, v in kwargs.items()
                                       if k in ("curve_pts", "path_index")})
    if s in {"from_image", "bitmap", "image"}:
        if image_path is None:
            raise ValueError("shape='from_image' requires image_path= kwarg")
        return make_shape_from_image(image_path, width,
                                      target_height_m=height if height != width else None,
                                      n_pts=n_circle_pts,
                                      **{k: v for k, v in kwargs.items()
                                         if k in ("threshold",)})

    raise ValueError(
        f"Unknown shape: '{shape}'. Supported: square, rect, circle, ellipse, "
        "triangle, equilateral_triangle, diamond, hexagon, pentagon, octagon, "
        "trapezoid, star/star5/star6/star8, cross, rounded_rect, L_shape, "
        "T_shape, H_shape, arrow, from_svg, from_image, polygon"
    )
