"""GEOMETRY INTAKE: any imported outline becomes a solvable HEATR 2-D case.

WHAT THIS IS FOR. Every actuator the campaign developed (the calibrated drive,
the filtered gradient solve, the rotationally averaged kernel, the dwell
schedule and its turntable program) is written against a `Case` built from a
production configuration. Until now those configurations came from the
eighteen-shape standardized library, so every result was, strictly, a result
about those eighteen shapes. This module removes that restriction: it accepts

  (a) a binary MASK array,
  (b) a POLYGON vertex list,
  (c) a PNG (portable network graphics) mask file,

and emits, at a requested grid, the two objects the solve stack needs and the
one object the rest of the pipeline needs:

  chi         the grid-independent sub-cell AREA FILL target indicator, built
              by `chi_area` (one implementation, reused, never re-derived);
  part_mask   the production binary raster, built by the production domain
              builder from the same polygon, so the target and the material
              can never disagree about what the part is;
  cfg         a production configuration with `geometry.part.shape = polygon`,
              which `rfam_eqs_coupled.make_shape` has supported all along
              (shapes.py:855-866), so the engine, `scripts/solve_fgm.py`, the
              rotation machinery and the graphical user interface consume an
              imported geometry with no code change at all.

ACRONYMS on first use: PNG portable network graphics; STL stereolithography;
EQS electro-quasi-static; FGM functionally graded material; IoU intersection
over union; bpp bits per pixel; W/m watts per metre of depth.

AREA-FILL CONVENTION, stated here because a second lane has to mirror it.
`area_fill(polygon, x, y)` returns `[len(y), len(x)]` cell-average indicator
values in [0, 1]. Cells are centred on the grid points. Inside-ness is the
EVEN-ODD rule, so vertex WINDING IS IRRELEVANT. Sub-cell sampling is a regular
`n_sub` by `n_sub` point grid at the production offsets of
`rfam_eqs_coupled._subpixel_fill_fraction`, with `CHI_N_SUB = 32`, evaluated
only in a band of one half cell diagonal around the boundary (proven bit
identical to the unrestricted production sampler in `test_chi_area.py`). The
executable form of this convention, parameterized over the fill implementation
so the three-dimensional lane can run it against a VOLUME fill, is
`adjoint2d/tests/fill_contract.py`; `assert_extrusion_slice_reduction` states
the reduction a volume fill must satisfy on an interior extrusion slice.

LAYER-WISE AGGREGATION CONTRACT, for the STL-to-layer-wise consumers: see
`LAYER_AGGREGATION_CONTRACT` below. In one line, a build emits ONE turntable
program, the per-layer recommendations are combined DOSE-WEIGHTED by default,
the WORST layer is computed and reported as a disagreement flag, and the whole
per-layer aggregation is a HEURISTIC pending a full three-dimensional
verification run, because z-coupling is real.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from . import chi_area
from . import geometry_contour as gc
from .pins import build_case, load_cfg

__all__ = ["IntakeError", "Intake", "Geometry", "area_fill", "grid_axes",
           "from_polygon", "from_mask", "from_png", "build_config",
           "CHI_N_SUB", "LAYER_AGGREGATION_CONTRACT", "DEFAULT_TEMPLATE"]

CHI_N_SUB = chi_area.CHI_N_SUB
DEFAULT_GRID = 120
DEFAULT_CHAMBER_M = 0.06

LAYER_AGGREGATION_CONTRACT = """\
AGGREGATION CONTRACT for layer-wise consumers (the STL-to-layer-wise pipeline).

A build is one physical exposure of a stack of layers on ONE turntable, so the
deliverable is a PER-BUILD turntable program and a per-build actuator
recommendation, not one per layer. The per-layer intake results are combined as
follows.

1. DOSE-WEIGHTED AGGREGATION IS THE DEFAULT. Each layer's anisotropy spectrum
   and recommendation enter the build-level decision weighted by that layer's
   share of the delivered dose, approximated by its part area times its
   exposure time. A one-cell layer must not outvote the bulk of the build.
2. THE WORST LAYER IS ALWAYS COMPUTED AND REPORTED (the worst-layer flag). The
   maximum over layers of
   the residual anisotropy is carried alongside the dose-weighted value as a
   DISAGREEMENT FLAG. When the worst layer's recommended actuator class differs
   from the dose-weighted one, that is a STRONG DISAGREEMENT and it must be
   surfaced to the user rather than averaged away: it means the build contains
   a layer the chosen program cannot serve.
3. PER-LAYER AGGREGATION IS A HEURISTIC, NOT A VERIFIED MODEL. Every layer here
   is solved as an independent two-dimensional problem. Z-COUPLING IS REAL: the
   stl_compensation_tool analysis measured a sub-linear z-gain (roughly g to
   the power -0.7), which is direct evidence that layers are not independent.
   The aggregation is therefore pending a full three-dimensional verification
   run and must be quoted as a heuristic wherever it is used.
4. RESAMPLING. Stored FGM maps are printer-resolution rasters, not solve-grid
   rasters. Resample before reuse, by the production convention
   (`robust.resample_map`, bilinear then clip).
"""

DEFAULT_TEMPLATE = "square"     # which library config supplies the physics blocks


class IntakeError(RuntimeError):
    """The imported geometry cannot be turned into a solvable case."""


# ---------------------------------------------------------------------------
# the area fill, one implementation only
# ---------------------------------------------------------------------------

def area_fill(poly: np.ndarray, x: np.ndarray, y: np.ndarray,
              n_sub: int = CHI_N_SUB) -> np.ndarray:
    """Sub-cell area fill of one polygon. Delegates to `chi_area`.

    This is deliberately a thin delegation and not a second implementation:
    two copies of a fill rule drift, and the drift shows up as a target
    indicator that disagrees with the material fill fraction by a boundary
    cell. The convention it obeys is documented in this module's docstring and
    asserted, parameterized over the implementation, in
    `adjoint2d/tests/fill_contract.py`.
    """
    return chi_area.area_fill_poly(np.asarray(poly, dtype=float), x, y, n_sub=n_sub)


def area_fill_union(polys: Sequence[np.ndarray], x: np.ndarray, y: np.ndarray,
                    n_sub: int = CHI_N_SUB) -> np.ndarray:
    """Union fill over several disjoint polygons, by the production maximum rule."""
    return chi_area.area_fill_union(list(polys), x, y, n_sub=n_sub)


MAX_VERTICES_FOR_CROSSING_CHECK = 3000


def find_self_intersection(poly: np.ndarray) -> tuple[int, int] | None:
    """The first pair of NON-ADJACENT edges that cross, or None.

    WHY THIS IS CHECKED AT ALL. The area fill uses the even-odd rule, which is
    total: it returns an answer for a crossed outline instead of an error, and
    the answer is a plausible looking part that is NOT the one the user drew.
    Measured in this pass on a keyhole outline built as a head arc followed by
    stem corners: the two crossed where they joined and the even-odd rule cut a
    notch out of the head, which then propagated into the target indicator, the
    part mask, the calibration and the solve. Refusing is the only safe
    behaviour.

    Adjacent edges share an endpoint and are not a crossing. The test is the
    standard orientation test with a collinear-overlap case included.
    """
    p = np.asarray(poly, dtype=float)
    n = len(p)
    if n > MAX_VERTICES_FOR_CROSSING_CHECK:
        return None

    def orient(a, b, c):
        v = ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        return 0 if abs(v) < 1e-18 else (1 if v > 0 else -1)

    def on_seg(a, b, c):
        return (min(a[0], b[0]) - 1e-15 <= c[0] <= max(a[0], b[0]) + 1e-15 and
                min(a[1], b[1]) - 1e-15 <= c[1] <= max(a[1], b[1]) + 1e-15)

    for i in range(n):
        a1, a2 = p[i], p[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or (i + 1) % n == j:
                continue
            b1, b2 = p[j], p[(j + 1) % n]
            o1, o2 = orient(a1, a2, b1), orient(a1, a2, b2)
            o3, o4 = orient(b1, b2, a1), orient(b1, b2, a2)
            if o1 != o2 and o3 != o4:
                return (i, j)
            if o1 == 0 and on_seg(a1, a2, b1):
                return (i, j)
            if o2 == 0 and on_seg(a1, a2, b2):
                return (i, j)
    return None


def grid_axes(grid: int, chamber_m: float = DEFAULT_CHAMBER_M
              ) -> tuple[np.ndarray, np.ndarray]:
    """The production axes, `rfam_eqs_coupled.make_domain` lines 1624-1625."""
    n = int(grid)
    w = float(chamber_m)
    return (np.linspace(-0.5 * w, 0.5 * w, n), np.linspace(-0.5 * w, 0.5 * w, n))


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Geometry:
    """The imported outline, in metres, in the chamber frame."""

    polygons: tuple[np.ndarray, ...]
    source: str
    provenance: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return int(sum(len(p) for p in self.polygons))

    def bbox(self) -> tuple[float, float, float, float]:
        allp = np.vstack(self.polygons)
        return (float(allp[:, 0].min()), float(allp[:, 0].max()),
                float(allp[:, 1].min()), float(allp[:, 1].max()))


@dataclass(frozen=True)
class Intake:
    """Everything the solve stack needs for one imported geometry at one grid."""

    geometry: Geometry
    x: np.ndarray
    y: np.ndarray
    chi: np.ndarray
    part_mask: np.ndarray
    cfg: dict
    info: dict

    @property
    def grid(self) -> int:
        return int(len(self.x))

    @property
    def dx(self) -> float:
        return float(self.x[1] - self.x[0])


# ---------------------------------------------------------------------------
# configuration synthesis
# ---------------------------------------------------------------------------

def _template_cfg(template: str | Path | dict | None) -> dict:
    """The physics, material and thermal blocks come from a campaign config.

    Nothing about the physics is invented here: the intake changes the GEOMETRY
    block and nothing else, so an imported part runs the same frozen material,
    thermal, densification and electrode settings every library shape ran.
    """
    if isinstance(template, dict):
        return copy.deepcopy(template)
    from . import library_solve as lib
    path = (lib.shape_config(DEFAULT_TEMPLATE) if template is None
            else (lib.shape_config(str(template))
                  if not Path(str(template)).exists() else Path(str(template))))
    return load_cfg(path)


def build_config(polys: Sequence[np.ndarray], grid: int = DEFAULT_GRID,
                 chamber_m: float = DEFAULT_CHAMBER_M,
                 template: str | Path | dict | None = None,
                 voltage_v: float | None = None,
                 name: str = "imported") -> dict:
    """A production config whose part is the imported polygon.

    The `polygon` shape branch of `shapes.make_shape` (shapes.py:855-866) takes
    the vertex list straight through, so no new engine code is needed. The
    stored `fgm_feedback` block of the template is REMOVED: it points at a
    dopant map solved for a different part, and leaving it in would silently
    inject that map into any engine run of this config.
    """
    cfg = _template_cfg(template)
    cfg.pop("fgm_feedback", None)
    cfg.pop("fgm_solve", None)
    geom = cfg.setdefault("geometry", {})
    geom["chamber_x"] = float(chamber_m)
    geom["chamber_y"] = float(chamber_m)
    geom["grid_nx"] = int(grid)
    geom["grid_ny"] = int(grid)
    geom.pop("parts", None)
    pts = [np.asarray(p, dtype=float) for p in polys]
    if len(pts) != 1:
        # Multi-part intake is expressible (geometry.parts, union by maximum),
        # but every downstream rotation and dwell convention in the campaign
        # was measured on a single part. Refuse rather than quietly extend it.
        raise IntakeError(
            f"{len(pts)} disjoint components were imported; the single-part "
            "conventions of the rotation and dwell campaigns are not "
            "established for multi-part builds. Import them one at a time.")
    p = pts[0]
    bx = float(p[:, 0].max() - p[:, 0].min())
    by = float(p[:, 1].max() - p[:, 1].min())
    geom["part"] = {
        "shape": "polygon",
        "polygon_points": [[float(a), float(b)] for a, b in p],
        # width/height are required by `_single_part_mask_and_fill` and are
        # IGNORED by the polygon branch; they are set to the true bounding box
        # so any reader that prints them prints the truth.
        "width": bx, "height": by,
        "center_x": 0.0, "center_y": 0.0, "rotation_deg": 0.0,
        "imported_name": str(name),
    }
    elec = cfg.setdefault("electric", {})
    elec["voltage_mode"] = "grounded"
    elec["enforce_generator_power"] = False
    elec.pop("voltage_hi_v", None)
    elec.pop("voltage_lo_v", None)
    if voltage_v is not None:
        elec["voltage_v"] = float(voltage_v)
    return cfg


# ---------------------------------------------------------------------------
# the three intake routes
# ---------------------------------------------------------------------------

def _finish(polys: list[np.ndarray], grid: int, chamber_m: float,
            template, voltage_v, name: str, source: str,
            provenance: dict, n_sub: int) -> Intake:
    x, y = grid_axes(grid, chamber_m)
    half = 0.5 * float(chamber_m)
    allp = np.vstack(polys)
    if (allp[:, 0].min() < -half or allp[:, 0].max() > half
            or allp[:, 1].min() < -half or allp[:, 1].max() > half):
        raise IntakeError(
            f"the imported geometry leaves the {chamber_m*1e3:.0f} mm chamber: "
            f"x in [{allp[:,0].min()*1e3:.2f}, {allp[:,0].max()*1e3:.2f}] mm, "
            f"y in [{allp[:,1].min()*1e3:.2f}, {allp[:,1].max()*1e3:.2f}] mm. "
            "Scale it down or enlarge chamber_m.")
    cfg = build_config(polys, grid=grid, chamber_m=chamber_m, template=template,
                       voltage_v=voltage_v, name=name)
    case = build_case(cfg)
    chi = area_fill_union(polys, x, y, n_sub=n_sub)
    dA = float(x[1] - x[0]) * float(y[1] - y[0])
    info = {
        "name": name,
        "source": source,
        "grid": int(grid),
        "chamber_m": float(chamber_m),
        "dx_m": float(x[1] - x[0]),
        "n_sub": int(n_sub),
        "n_vertices": int(sum(len(p) for p in polys)),
        "area_chi_m2": float(chi.sum()) * dA,
        "area_raster_m2": float(case.part_mask.sum()) * dA,
        "n_part_cells": int(case.part_mask.sum()),
        "raster_vs_area": chi_area.raster_vs_area_delta(
            case.part_mask, chi, float(x[1] - x[0]), float(y[1] - y[0])),
        "voltage_v": float(cfg["electric"]["voltage_v"]),
        "voltage_is_calibrated": voltage_v is not None,
        **provenance,
    }
    # Mask and PNG routes come out of `geometry_contour` counter-clockwise by
    # construction, so nothing was flipped for them.
    info.setdefault("winding_normalized", False)
    if info["n_part_cells"] < 50:
        raise IntakeError(
            f"the imported geometry covers only {info['n_part_cells']} cells at "
            f"grid {grid}; that is too few to solve. Enlarge the part or the grid.")
    return Intake(geometry=Geometry(tuple(polys), source, provenance),
                  x=x, y=y, chi=chi, part_mask=case.part_mask, cfg=cfg, info=info)


def from_polygon(poly: np.ndarray | Iterable[np.ndarray],
                 grid: int = DEFAULT_GRID, chamber_m: float = DEFAULT_CHAMBER_M,
                 template: Any = None, voltage_v: float | None = None,
                 name: str = "imported", n_sub: int = CHI_N_SUB) -> Intake:
    """Route (b): an explicit vertex list, in METRES, in the chamber frame."""
    p = np.asarray(poly, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise IntakeError(
            f"a polygon must be an (N, 2) array of [x, y] pairs in metres, "
            f"got shape {p.shape}")
    if len(p) < 3:
        raise IntakeError(f"a polygon needs at least 3 vertices, got {len(p)}")
    if np.allclose(p[0], p[-1]) and len(p) > 3:
        p = p[:-1]              # the ring is implicit; a repeated last vertex
        # would be a zero-length edge in the even-odd test
    # Drop consecutive duplicate vertices. A repeated point is a zero-length
    # edge, which has no orientation and makes the crossing test report a false
    # positive between the two edges that straddle it.
    keep = np.ones(len(p), dtype=bool)
    keep[1:] = np.any(np.abs(np.diff(p, axis=0)) > 1e-15, axis=1)
    if not keep.all():
        p = p[keep]
    # WINDING. `geometry_contour` emits counter-clockwise loops and treats a
    # clockwise loop from a MASK import as an interior hole, because that is
    # what it means there. A hand-drawn or computer-aided-design outline carries
    # no such meaning: clockwise is simply the other traversal of the same
    # boundary, and the even-odd fill is winding invariant (proven in
    # `fill_contract.assert_winding_invariance`). Refusing would be gratuitous,
    # and silently passing a clockwise ring through would leave two conventions
    # alive downstream, so the ring is NORMALIZED to counter-clockwise here and
    # the fact is recorded in the provenance.
    flipped = bool(gc.signed_area(p) < 0.0)
    if flipped:
        p = p[::-1].copy()
    hit = find_self_intersection(p)
    if hit is not None:
        i, j = hit
        raise IntakeError(
            f"the imported outline self-intersects: edge {i} "
            f"({p[i]} to {p[(i+1) % len(p)]}) crosses edge {j} "
            f"({p[j]} to {p[(j+1) % len(p)]}). The even-odd fill rule would "
            "silently reinterpret the crossing as a notch or a hole and the "
            "pipeline would solve a part you did not draw. Fix the outline.")
    return _finish([p], grid, chamber_m, template, voltage_v, name,
                   "polygon", {"closed_ring_trimmed": True,
                               "winding_normalized": flipped,
                               "self_intersection_checked": len(p) <=
                               MAX_VERTICES_FOR_CROSSING_CHECK}, n_sub)


def from_mask(mask: np.ndarray, grid: int = DEFAULT_GRID,
              chamber_m: float = DEFAULT_CHAMBER_M,
              pixel_pitch_m: float | None = None,
              part_width_m: float | None = None,
              template: Any = None, voltage_v: float | None = None,
              name: str = "imported", n_sub: int = CHI_N_SUB,
              centre: bool = True) -> Intake:
    """Route (a): a binary mask array at its own resolution.

    Exactly one scale argument must be given, `pixel_pitch_m` (the physical
    size of one mask pixel) or `part_width_m` (the physical width of the mask's
    BOUNDING BOX, aspect ratio preserved). Requiring one and refusing both is
    deliberate: a silently assumed scale is the fastest way to solve the wrong
    part.
    """
    m = np.asarray(mask)
    if m.ndim != 2:
        raise IntakeError(f"a mask must be a 2-D array, got shape {m.shape}")
    m = m.astype(bool)
    if (pixel_pitch_m is None) == (part_width_m is None):
        raise IntakeError(
            "give exactly one of pixel_pitch_m or part_width_m; the intake "
            "will not assume a scale for an imported mask")
    cols = np.flatnonzero(m.any(axis=0))
    rows = np.flatnonzero(m.any(axis=1))
    if cols.size == 0:
        raise IntakeError("the imported mask is empty")
    w_px = float(cols[-1] - cols[0] + 1)
    pitch = (float(pixel_pitch_m) if pixel_pitch_m is not None
             else float(part_width_m) / w_px)
    polys = gc.mask_to_polygons(m, dx=pitch, dy=pitch, x0=0.0, y0=0.0)
    if centre:
        allp = np.vstack(polys)
        cx = 0.5 * (allp[:, 0].min() + allp[:, 0].max())
        cy = 0.5 * (allp[:, 1].min() + allp[:, 1].max())
        polys = [p - np.array([cx, cy]) for p in polys]
    prov = {"mask_shape": [int(m.shape[0]), int(m.shape[1])],
            "pixel_pitch_m": pitch,
            "mask_pixels_true": int(m.sum()),
            "scale_given_as": "pixel_pitch_m" if pixel_pitch_m is not None
                              else "part_width_m",
            "centred": bool(centre)}
    return _finish(polys, grid, chamber_m, template, voltage_v, name,
                   "mask", prov, n_sub)


def from_png(path: str | Path, grid: int = DEFAULT_GRID,
             chamber_m: float = DEFAULT_CHAMBER_M,
             pixel_pitch_m: float | None = None,
             part_width_m: float | None = None,
             threshold: float = 0.5, invert: bool = False,
             template: Any = None, voltage_v: float | None = None,
             name: str | None = None, n_sub: int = CHI_N_SUB,
             centre: bool = True) -> Intake:
    """Route (c): a PNG mask file. Bright pixels are the part unless inverted.

    ROW ORDER. PNG row 0 is the image TOP; a field array's row 0 is the
    physical BOTTOM (the convention `scripts/solve_fgm.emit_map_pngs` flips on
    the way out). The image is therefore flipped vertically on the way in, so
    what the user drew at the top of the picture is at the top of the part.
    """
    from PIL import Image

    p = Path(path)
    if not p.exists():
        raise IntakeError(f"PNG mask not found: {p}")
    img = np.asarray(Image.open(p).convert("L"), dtype=float) / 255.0
    m = np.flipud(img) >= float(threshold)
    if invert:
        m = ~m
    it = from_mask(m, grid=grid, chamber_m=chamber_m,
                   pixel_pitch_m=pixel_pitch_m, part_width_m=part_width_m,
                   template=template, voltage_v=voltage_v,
                   name=(name or p.stem), n_sub=n_sub, centre=centre)
    it.info["png_path"] = str(p)
    it.info["png_threshold"] = float(threshold)
    it.info["png_inverted"] = bool(invert)
    return it
