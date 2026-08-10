"""Shape-library generalization campaign harness.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.shape_campaign --shape cone --export-dg0
    .venv312/bin/python -m solve3d.shape_campaign --shape cone --measure-fidelity

REUSABLE FOR ALL 14 LIBRARY SHAPES, not hardcoded to the three de-risk targets.
Per shape it runs the campaign's per-shape pipeline steps 1-3 (SHAPE_LIBRARY_
CAMPAIGN.md); the heavy AL solve (step 4) is launched separately, permission-
gated.

STEP 1 -- THE FIDELITY PRE-GATE (the pyramid lesson, MANDATORY FIRST). Before a
grid-confounded heavy solve, measure the DG0->voxel transfer mass-move at the
candidate grids (n64, n80, n96) and pick the grid where it is faithful (under the
2 % Phase-C staircase gate that studio3d.transfer already enforces on the SOLVED
map). The pyramid was NON-MONOTONIC in n (n80 faithful, n64+n96 alias at the
apex), so we PROBE the grids, never assume the finest is best.

TWO-ENV BRIDGE, deliberately. The DG0 centroids come from the dolfinx conforming
tet mesh (spike env: dolfinx+gmsh, no trimesh). The voxel part comes from the
SAME heatr3d voxelizer the cross-engine verify uses (studio3d.runner.voxelize_stl,
.venv312: trimesh+heatr3d, no dolfinx). Neither env has both, so the pre-gate is
two stages joined by a small npz: export_dg0 (spike) writes the centroids; the
measure stage (.venv312) voxelizes + transfers + decides.

THE PROBE FIELD, and its honest limit. The transfer mass-move is IDENTICALLY zero
for a spatially-uniform field (mean in == mean out by construction), so a faithful
pre-gate must probe with a field that VARIES across the part like a shaped dopant
map does (the pyramid's shipped map spanned s in [0.008, 1.0], an O(1) range). We
probe a small basis normalised to [0, 1] -- the three axis ramps and the radial
field -- and gate on the WORST axis ramp per grid. Calibration against the pinned
pyramid: this worst-axis screen's argmin correctly SELECTS the faithful grid
(n80), the load-bearing output; the radial field is pathologically conservative
on compact convex parts and is reported but NOT gated on. The screen does NOT
reproduce the solved map's exact per-grid value (that needs the solved map, which
does not exist pre-solve) -- an honest limit stated, not hidden.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# candidate voxel grids probed by the fidelity pre-gate
CANDIDATE_GRIDS = (64, 80, 96)

# --------------------------------------------------------------------------- #
# BUILD-AXIS CONVENTION: natural-up -> +Y (2026-08-10). The EQS field + build
# axis is Y (forward._electrode_dofs electrodes at x[1]=+-L/2, open/convection
# face y=+L/2), so every part must stand with its natural "up" along +Y or it
# solves sideways. The phase_e OCC primitives are BUILT along +Y directly; the
# library STLs ship these bodies with their native axis along +Z, so the voxel
# side rotates the STL +Z->+Y (Rx -90 deg) to match the tet mesh. For the
# awkward shapes (toroid, flat_plane, l_extrusion, lattice) the principled
# default is "longest axis vertical / largest flat face down" -- each is FLAGGED
# in the tracker for Matt's review rather than guessed silently here.
STL_REORIENT = {
    # bodies of revolution, native axis +Z -> +Y
    "cone": "Rx-90", "cylinder": "Rx-90",
    # symmetric or already axis-along-Y in the STL -> identity
    "sphere": None,
    # polyhedra whose OCC is built axis-agnostic and STL already matches
    "cube": None, "pyramid": None,
}
# shapes whose build-axis default is a JUDGEMENT CALL, flagged for review
AWKWARD_SHAPES = ("toroid", "flat_plane", "l_extrusion", "lattice")
# the 2 % Phase-C staircase gate studio3d.transfer enforces on the transferred map
MASS_MOVE_GATE = 0.02
# axis-ramp fields the worst-axis screen gates on (radial is reported, not gated)
GATE_FIELDS = ("xramp", "yramp", "zramp")


# --------------------------------------------------------------------------- #
# Pure-logic layer (env-agnostic: no dolfinx, no trimesh)
# --------------------------------------------------------------------------- #
def _norm01(x: np.ndarray) -> np.ndarray:
    """Affine map to [0, 1]; a constant field maps to all zeros (no div-by-0)."""
    x = np.asarray(x, dtype=float)
    lo = float(x.min())
    span = float(x.max()) - lo
    if span <= 0.0:
        return np.zeros_like(x)
    return (x - lo) / span


def build_probe_fields(centroids: np.ndarray) -> Dict[str, np.ndarray]:
    """The probe basis for the fidelity pre-gate, each normalised to [0, 1].

    `centroids` is (N, 3) DG0 cell centroids in metres. The three axis ramps
    stress boundary aliasing along each principal direction (the pyramid's apex
    aliasing lives on its axis); the radial field (distance from the part
    centroid) is reported for transparency but not gated on."""
    c = np.asarray(centroids, dtype=float)
    ctr = c.mean(axis=0)
    return {
        "xramp": _norm01(c[:, 0]),
        "yramp": _norm01(c[:, 1]),
        "zramp": _norm01(c[:, 2]),
        "radial": _norm01(np.linalg.norm(c - ctr, axis=1)),
    }


def worst_axis_move(field_moves: Dict[str, float]) -> float:
    """The gated metric: the WORST mass-move over the axis ramps, radial ignored.

    A shaped dopant map's transfer error is dominated by field variation across
    the part in some direction; the worst axis ramp is the conservative proxy.
    Radial is excluded because it over-flags compact convex parts (calibrated on
    the pyramid, where radial reads ~3 % at every grid while the shipped map
    passed n80 at 1.59 %)."""
    return float(max(field_moves[f] for f in GATE_FIELDS))


def select_faithful_grid(grid_moves: Dict[int, Dict[str, float]],
                         gate: float = MASS_MOVE_GATE) -> dict:
    """Pick the faithful voxel grid, or flag the shape transfer-limited.

    `grid_moves` maps grid n -> {field -> mass_move_rel}. The chosen grid is the
    ARGMIN of the worst-axis metric (NOT the coarsest-under-gate: the pyramid's
    n64 is under a naive threshold yet its shipped map aliased there, so we take
    the most faithful grid, ties broken to the coarser mesh for compute). The
    shape is `transfer_limited` iff even that most-faithful grid is not under the
    gate -- an honest 'no candidate grid transfers faithfully', never a silent
    coarsening or a widened gate."""
    per_grid = {int(n): worst_axis_move(m) for n, m in grid_moves.items()}
    # argmin worst-axis; tie -> coarser (smaller n) grid
    chosen = min(per_grid, key=lambda n: (per_grid[n], n))
    chosen_move = per_grid[chosen]
    faithful = bool(chosen_move < gate)
    return {
        "chosen_grid": int(chosen),
        "chosen_worst_axis_move": float(chosen_move),
        "faithful": faithful,
        "transfer_limited": bool(not faithful),
        "gate": float(gate),
        "per_grid_worst_axis": per_grid,
    }


# --------------------------------------------------------------------------- #
# Stage 1 (spike env): export the DG0 centroids for a shape's coarse case
# --------------------------------------------------------------------------- #
def export_dg0(shape: str, out_path=None, power_density: float | None = None):
    """Build the SOLVE-mesh conforming case for `shape` and write its DG0
    design-cell centroids + volumes to an npz the measure stage reads.

    Built at the SOLVE-mesh resolution (stage_b.SOLVE_TARGET_NODES / SOLVE_LC0_M),
    NOT the coarse FD-gate mesh: the fidelity pre-gate must measure the transfer
    of the map at the resolution the heavy solve will actually emit and hand to
    the cross-engine verify (the pyramid's pinned 1.59 % is a solve-mesh number,
    8931 cells, not a coarse-mesh one). dolfinx only; no physics is solved --
    just the mesh + design chain (a mesh build, minutes)."""
    from pathlib import Path

    from solve3d import density_adjoint as da, stage_b

    case = da.build_coarse_case(
        shape=shape, power_density=power_density,
        target_nodes=stage_b.SOLVE_TARGET_NODES, lc0=stage_b.SOLVE_LC0_M)
    chain = case.chain
    centroids = np.asarray(chain.centroids, dtype=float)
    volumes = np.asarray(chain.volumes, dtype=float)
    if out_path is None:
        out_path = (Path(__file__).resolve().parent / "results"
                    / f"fidelity_dg0_{shape}.npz")
    out_path = Path(out_path)
    np.savez_compressed(out_path, centroids=centroids, volumes=volumes,
                        n_design=np.array(int(centroids.shape[0])),
                        shape=np.array(shape))
    return {"shape": shape, "n_design": int(centroids.shape[0]),
            "out": str(out_path),
            "bbox_m": [centroids.min(0).tolist(), centroids.max(0).tolist()]}


# --------------------------------------------------------------------------- #
# Stage 2 (.venv312): voxelize + measure the transfer mass-move per grid
# --------------------------------------------------------------------------- #
def _dg0_voxel_mass_move(centroids, values, volumes, part, chamber_m) -> float:
    """Non-gating measurement of the DG0->voxel in-part mass-move.

    REUSES studio3d.transfer.dg0_to_voxel when it passes (pins this number to the
    production transfer), and reproduces the SAME measured value from the raised
    TransferError message when the transfer is over-gate -- so the pre-gate can
    report the number at every grid, including the aliased ones the production
    gate refuses. The equality on a passing grid is asserted by the harness
    validation (pyramid n80)."""
    import re

    from studio3d import transfer as tr
    try:
        rec = tr.dg0_to_voxel(np.asarray(centroids, float),
                              np.asarray(values, float),
                              np.asarray(volumes, float), part, chamber_m)
        return float(rec["dopant_mass_move_rel"])
    except tr.TransferError as e:
        m = re.search(r"by ([0-9.]+) %", str(e))
        if m is None:
            raise
        return float(m.group(1)) / 100.0


def _reoriented_stl_path(shape: str, stl_path, tmpdir) -> str:
    """Return a path to the STL reoriented to the build-axis convention (natural
    up -> +Y). For bodies of revolution shipped native-+Z (cone/cylinder) this
    rotates the mesh +Z->+Y (Rx -90) and writes a temp STL so the voxel part
    matches the +Y-built tet mesh. Identity shapes return the original path."""
    import numpy as _np
    import trimesh
    from pathlib import Path

    kind = STL_REORIENT.get(shape, None)
    if kind is None:
        return str(stl_path)
    if kind != "Rx-90":
        raise ValueError(f"unknown STL reorientation {kind!r} for {shape!r}")
    mesh = trimesh.load_mesh(str(stl_path))
    Rx = trimesh.transformations.rotation_matrix(-_np.pi / 2.0, [1.0, 0.0, 0.0])
    mesh.apply_transform(Rx)                      # +Z -> +Y
    out = Path(tmpdir) / f"{shape}_reoriented_plusY.stl"
    mesh.export(str(out))
    return str(out)


def measure_fidelity(shape: str, dg0_path=None, grids=CANDIDATE_GRIDS,
                     chamber_m: float = 0.060, stl_path=None) -> dict:
    """Stage 2: at each candidate grid, voxelize the library STL (REORIENTED to
    the +Y build-axis convention, matching the tet mesh) with the SAME voxelizer
    the cross-engine verify uses, transfer each probe field, and select the
    faithful grid (or flag transfer_limited). Writes fidelity_<shape>.json.
    trimesh + heatr3d only."""
    import json
    import tempfile
    from pathlib import Path

    from studio3d.runner import voxelize_stl

    RESULTS = Path(__file__).resolve().parent / "results"
    ROOT = Path(__file__).resolve().parents[1]
    if dg0_path is None:
        dg0_path = RESULTS / f"fidelity_dg0_{shape}.npz"
    if stl_path is None:
        stl_path = ROOT / "shape_library_3d" / "stl" / f"{shape}.stl"
    d = np.load(dg0_path, allow_pickle=True)
    centroids, volumes = d["centroids"], d["volumes"]
    fields = build_probe_fields(centroids)

    tmpdir = tempfile.mkdtemp(prefix="fidelity_reorient_")
    stl_use = _reoriented_stl_path(shape, stl_path, tmpdir)
    grid_moves: Dict[int, Dict[str, float]] = {}
    part_vox: Dict[int, int] = {}
    for n in grids:
        part = voxelize_stl(stl_use, int(n), chamber_m=chamber_m)
        part_vox[int(n)] = int(part.sum())
        grid_moves[int(n)] = {
            name: _dg0_voxel_mass_move(centroids, vals, volumes, part, chamber_m)
            for name, vals in fields.items()}

    sel = select_faithful_grid(grid_moves, gate=MASS_MOVE_GATE)
    doc = {
        "what": "Shape-library campaign fidelity PRE-GATE (step 1): DG0->voxel "
                "transfer mass-move per candidate grid, gated on the worst axis "
                "ramp at the 2% Phase-C staircase threshold. Pick the faithful "
                "grid (argmin worst-axis) or flag transfer_limited. Probe fields "
                "vary O(1) across the part like a shaped dopant map; radial is "
                "reported, not gated (pathologically conservative on convex parts).",
        "stage": "fidelity_pre_gate",
        "shape": shape,
        "build_axis": "+Y",
        "stl_reorientation": STL_REORIENT.get(shape, None),
        "chamber_m": float(chamber_m),
        "n_design_cells": int(centroids.shape[0]),
        "candidate_grids": [int(n) for n in grids],
        "part_voxels": part_vox,
        "per_grid_mass_move_by_field": grid_moves,
        "gate_fields": list(GATE_FIELDS),
        "reported_not_gated_fields": ["radial"],
        "gate": MASS_MOVE_GATE,
        "selection": sel,
        "chosen_grid": sel["chosen_grid"],
        "transfer_limited": sel["transfer_limited"],
        "calibration_note": "worst-axis argmin reproduces the PINNED pyramid "
                            "faithful grid (n80); absolute per-grid value is a "
                            "conservative screen, not the solved-map mass-move.",
    }
    out = RESULTS / f"fidelity_{shape}.json"
    out.write_text(json.dumps(doc, indent=2))
    return doc


def main() -> int:
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--export-dg0", action="store_true",
                    help="Stage 1 (spike env): write the DG0 centroids npz")
    ap.add_argument("--measure-fidelity", action="store_true",
                    help="Stage 2 (.venv312): voxelize + measure + select grid")
    a = ap.parse_args()
    if a.export_dg0:
        print(json.dumps(export_dg0(a.shape), indent=1))
    if a.measure_fidelity:
        doc = measure_fidelity(a.shape)
        print(json.dumps({"shape": a.shape, "chosen_grid": doc["chosen_grid"],
                          "transfer_limited": doc["transfer_limited"],
                          "selection": doc["selection"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
