"""Support-aware map transfer (spec section 6).

Phase C measured the failure mode this module exists to prevent: a voxel
map's support is the staircase part and zero outside it, so interpolating
near the rim pulls in zeros and thins the map (7.15 % dopant loss vs 0.065 %
for a support-aware path, solve3d/PHASE_C_REPORT.md section 4).

Rule: EXTEND the field beyond its support (nearest inside-support value),
THEN interpolate, THEN re-mask to the target support. The acceptance gate is
the Phase C pre-registered threshold: the volume-normalized in-part dopant
level (mean saturation over the part) moves < 2 % across the transfer, or
the transfer raises TransferError. It never passes silently.

The transfer record's ``state`` is one of the three explicit manifest states
(spec 7b): "measured_and_passed" here; "measured_and_failed" is reported via
TransferError by the caller; "transfer_not_applicable" is for paths with no
volume-to-grid transfer step and is never implied by absence.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
from scipy import ndimage

MASS_MOVE_GATE = 0.02


class TransferError(ValueError):
    """A transfer that moved in-part dopant beyond the 2 % gate."""


def _extend_into(field: np.ndarray, support: np.ndarray) -> np.ndarray:
    """Fill everywhere outside ``support`` with the nearest in-support value."""
    if not support.any():
        raise TransferError("source support is empty")
    _, idx = ndimage.distance_transform_edt(~support, return_indices=True)
    return field[tuple(idx)]


def _mean_sat(sat: np.ndarray, part: np.ndarray) -> float:
    return float(sat[part].mean())


def _interp_to_grid(src: np.ndarray, n1: int) -> np.ndarray:
    """Trilinear sample of a cubic voxel field at the n1-grid cell centers.

    Both grids share the chamber frame: center c_k = (k + 0.5) * h - L/2, so
    the source index coordinate of a target center is (c + L/2)/h0 - 0.5.
    """
    n0 = src.shape[0]
    c1 = (np.arange(n1) + 0.5) / n1          # in units of L
    coord = c1 * n0 - 0.5
    X, Y, Z = np.meshgrid(coord, coord, coord, indexing="ij")
    return ndimage.map_coordinates(src, [X, Y, Z], order=1, mode="nearest")


def _gate(move_rel: float, context: str) -> None:
    if not np.isfinite(move_rel) or abs(move_rel) >= MASS_MOVE_GATE:
        raise TransferError(
            f"{context}: transfer moved in-part dopant by "
            f"{move_rel * 100:.2f} %, over the 2 % gate "
            "(Phase C staircase threshold). Refusing the transferred map.")


def voxel_to_voxel(sat0: np.ndarray, part0: np.ndarray, part1: np.ndarray,
                   chamber_m: float) -> Dict[str, Any]:
    """Transfer a voxel dopant volume onto a different-n voxel part."""
    n1 = part1.shape[0]
    mean0 = _mean_sat(sat0, part0)
    if mean0 <= 0:
        raise TransferError("source map has no in-part dopant")

    extended = _extend_into(np.asarray(sat0, float), np.asarray(part0, bool))
    sat1 = _interp_to_grid(extended, n1)
    sat1[~part1] = 0.0
    move_rel = abs(_mean_sat(sat1, part1) - mean0) / mean0

    # the naive zero-fill result, computed identically, kept for the record
    naive = _interp_to_grid(np.where(part0, sat0, 0.0).astype(float), n1)
    naive[~part1] = 0.0
    naive_move = abs(_mean_sat(naive, part1) - mean0) / mean0

    _gate(move_rel, "voxel_to_voxel")
    return {"sat": sat1, "state": "measured_and_passed",
            "method": "extend_before_interpolate_nearest_edt",
            "dopant_mass_move_rel": float(move_rel),
            "naive_mass_move_rel": float(naive_move),
            "gate": MASS_MOVE_GATE}


def stack_to_voxel(sat_stack: np.ndarray, mask_stack: np.ndarray,
                   z_mm: np.ndarray, part: np.ndarray,
                   chamber_m: float,
                   target_chamber_m: float | None = None) -> Dict[str, Any]:
    """2.5-D dopant_volume stack onto an (n, n, n) voxel part.

    TWO FRAMES, deliberately separate (TAMPER_DIAGNOSIS.md section 3f):

      chamber_m         the SOURCE frame the 2.5-D stack was emitted on. The
                        2.5-D pipeline solves per-cluster chambers -- 65 mm and
                        85 mm were both observed in the shipped jobs -- at
                        ng=160, dx ~ 0.5 mm.
      target_chamber_m  the VOXEL ARM's chamber, i.e. studio3d.runner.CHAMBER_M
                        (60 mm at n=64, dx 0.9375 mm). Defaults to it.

    These used to be ONE parameter, and the caller passed the source's value,
    so the target grid was built on the source chamber too. That does not
    translate frames, it RESCALES the part by target/source (85/60 = 1.4167x):
    every voxel sampled the stack at the wrong physical location. On the Tamper
    job it moved the in-part dopant level by 37.37 %, the 2 % gate correctly
    refused the map, and the chain fell through to the legacy inversion rung
    that then shipped a harmful correction. With the frames separated the same
    Tamper map measures 2.99 %.

    The 2 % gate itself is UNCHANGED; this fixes the resampling, not the
    threshold.

    Stack frame (stl_compensation_tool convention): sat[k, iy, ix] on a
    centered grid with pitch chamber/(ng-1); sat = 1.0 OUTSIDE the mask
    means unmodulated and must never be treated as data, so only in-mask
    values are used (extension supplies the rim). z_mm are layer centers
    relative to the part bottom; the voxel part is centered in the chamber.
    """
    if target_chamber_m is None:
        from studio3d.runner import CHAMBER_M
        target_chamber_m = CHAMBER_M
    nz_l, ng, _ = sat_stack.shape
    n = part.shape[0]
    h = target_chamber_m / n
    zc = (np.arange(n) + 0.5) * h - target_chamber_m / 2.0
    part_z = np.where(part.any(axis=(0, 1)))[0]
    if len(part_z) == 0:
        raise TransferError("target part is empty")
    z_bottom = zc[part_z[0]] - h / 2.0
    z_layers_phys = z_bottom + np.asarray(z_mm, float) * 1e-3

    mean0 = float(sat_stack[mask_stack].mean())
    if mean0 <= 0:
        raise TransferError("source stack has no in-mask dopant")

    # in-plane target sample coordinates (TARGET frame, metres) mapped into
    # SOURCE index space (source frame). Mixing the two frames here is exactly
    # the Tamper bug.
    c1 = (np.arange(n) + 0.5) * h - target_chamber_m / 2.0
    src_idx = (c1 + chamber_m / 2.0) * (ng - 1) / chamber_m
    X, Y = np.meshgrid(src_idx, src_idx, indexing="ij")   # [i,j] = (x_i, y_j)

    out = np.zeros((n, n, n))
    extended_cache: Dict[int, np.ndarray] = {}
    for k in part_z:
        li = int(np.argmin(np.abs(z_layers_phys - zc[k])))
        if li not in extended_cache:
            m2 = np.asarray(mask_stack[li], bool)
            if not m2.any():
                raise TransferError(f"source layer {li} has an empty mask")
            extended_cache[li] = _extend_into(
                np.where(m2, sat_stack[li], 0.0).astype(float), m2)
        plane = ndimage.map_coordinates(extended_cache[li], [Y, X],
                                        order=1, mode="nearest")
        out[:, :, k] = np.where(part[:, :, k], plane, 0.0)

    move_rel = abs(_mean_sat(out, part) - mean0) / mean0
    _gate(move_rel, "stack_to_voxel")
    return {"sat": out, "state": "measured_and_passed",
            "method": "per_layer_extend_before_interpolate_nearest_z",
            "dopant_mass_move_rel": float(move_rel),
            "source_chamber_m": float(chamber_m),
            "target_chamber_m": float(target_chamber_m),
            "gate": MASS_MOVE_GATE}


def dg0_to_voxel(centroids: np.ndarray, values: np.ndarray,
                 volumes: np.ndarray, part: np.ndarray,
                 chamber_m: float) -> Dict[str, Any]:
    """Transfer DG0 cell data (the solve3d artifact form) onto a voxel part.

    Volume-weighted binning of cell values into voxels, nearest-value
    extension into part voxels that received no cell, re-mask outside.
    """
    n = part.shape[0]
    h = chamber_m / n
    idx = np.floor((np.asarray(centroids) + chamber_m / 2.0) / h).astype(int)
    idx = np.clip(idx, 0, n - 1)
    flat = np.ravel_multi_index((idx[:, 0], idx[:, 1], idx[:, 2]),
                                (n, n, n))
    wsum = np.bincount(flat, weights=np.asarray(volumes, float),
                       minlength=n ** 3)
    vsum = np.bincount(flat, weights=np.asarray(values, float)
                       * np.asarray(volumes, float), minlength=n ** 3)
    got = wsum > 0
    binned = np.zeros(n ** 3)
    binned[got] = vsum[got] / wsum[got]
    binned = binned.reshape((n, n, n))
    support = got.reshape((n, n, n))

    mean0 = float(np.sum(np.asarray(values, float)
                         * np.asarray(volumes, float))
                  / np.sum(np.asarray(volumes, float)))
    if mean0 <= 0:
        raise TransferError("source DG0 field has no dopant")

    sat = _extend_into(binned, support)
    sat = np.where(part, sat, 0.0)
    move_rel = abs(_mean_sat(sat, part) - mean0) / mean0
    _gate(move_rel, "dg0_to_voxel")
    return {"sat": sat, "state": "measured_and_passed",
            "method": "volume_weighted_bin_extend_nearest_edt",
            "dopant_mass_move_rel": float(move_rel),
            "gate": MASS_MOVE_GATE}
