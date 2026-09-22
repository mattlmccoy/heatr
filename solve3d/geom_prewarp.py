"""Green-geometry pre-warp backsolve (Z densification compensation).

Pure per-column geometry logic + the outer-loop driver. The heavy heatr3d
densify march is injected as a callable so this module has NO solver dependency
and its tests run in the pure-numpy venv. Convention: arrays are (nx, ny, nz)
with z = axis 2 (heatr3d build axis); column heights are (nx, ny) in metres.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


def resample_column(src: np.ndarray, n_out: int) -> np.ndarray:
    """Linear, layer-centre-aligned 1-D resample of one column's values to
    n_out samples (output i samples input at (i+0.5)*n_in/n_out - 0.5)."""
    src = np.asarray(src, float)
    n_in = src.shape[0]
    if n_out <= 0:
        return np.zeros(0)
    if n_in == 0:
        return np.zeros(n_out)
    if n_out == n_in:
        return src.copy()
    zi = np.clip((np.arange(n_out) + 0.5) * n_in / n_out - 0.5, 0.0, n_in - 1.0)
    lo = np.floor(zi).astype(int)
    hi = np.minimum(lo + 1, n_in - 1)
    w = zi - lo
    return src[lo] * (1.0 - w) + src[hi] * w


def target_column_heights(mask0: np.ndarray, h: float) -> np.ndarray:
    """Per-(x,y) nominal (target dense) column height in metres = occupied
    voxel count along z (axis 2) times h."""
    return np.asarray(mask0, bool).sum(axis=2).astype(float) * float(h)


def column_height_update(H_green: np.ndarray, H_target: np.ndarray,
                         H_measured: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Multiplicative green-height update: H_green *= H_target / H_measured.
    Non-part columns (H_target == 0) stay 0. Robust form (matches the shrinkage
    compensation convention f = target/built)."""
    H_green = np.asarray(H_green, float)
    H_target = np.asarray(H_target, float)
    H_measured = np.asarray(H_measured, float)
    gain = H_target / np.maximum(H_measured, eps)
    return np.where(H_target > 0, H_green * gain, 0.0)


def max_rel_error(H_target: np.ndarray, H_measured: np.ndarray,
                  cols: np.ndarray | None = None) -> float:
    """Max over part columns of |H_target - H_measured| / H_target."""
    H_target = np.asarray(H_target, float)
    H_measured = np.asarray(H_measured, float)
    if cols is None:
        cols = H_target > 0
    if not np.any(cols):
        return 0.0
    return float(np.max(np.abs(H_target[cols] - H_measured[cols])
                        / np.maximum(H_target[cols], 1e-9)))


def build_green_volume(mask0: np.ndarray, dop0: np.ndarray,
                       H_green: np.ndarray, h: float):
    """Per-column pre-warped green volume. Each (x,y) column is occupied from
    z=0 to round(H_green/h) voxels; the nominal column's dopant (occupied voxels
    only) is resampled to that many green voxels. Returns (green_mask, green_dop),
    both (nx, ny, nz_out) with z = axis 2, base at k=0."""
    mask0 = np.asarray(mask0, bool)
    dop0 = np.asarray(dop0, float)
    nx, ny, _ = mask0.shape
    H_green = np.asarray(H_green, float)
    n_g = np.rint(H_green / float(h)).astype(int)
    n_g = np.where(H_green > 0, np.maximum(n_g, 1), 0)
    nz_out = int(n_g.max()) if n_g.max() > 0 else 1
    green_mask = np.zeros((nx, ny, nz_out), bool)
    green_dop = np.zeros((nx, ny, nz_out), float)
    for i in range(nx):
        for j in range(ny):
            ng = int(n_g[i, j])
            if ng <= 0:
                continue
            occ = mask0[i, j]
            src = dop0[i, j][occ]
            if src.size == 0:
                src = np.zeros(1)
            green_dop[i, j, :ng] = resample_column(src, ng)
            green_mask[i, j, :ng] = True
    return green_mask, green_dop


def prewarp_solve(mask0: np.ndarray, dop0: np.ndarray, h: float, forward_fn,
                  *, bulk_factor: float = 1.0, tol: float = 0.01,
                  k_max: int = 5, stall_patience: int = 2) -> dict:
    """Sequential outer-loop green-geometry backsolve.

    forward_fn(green_mask, green_dop) -> (H_measured(nx,ny), warp_std, aux).
    Warm start: green height = target * bulk_factor. Each iter builds the green
    volume, marches it (forward_fn), measures per-column dense height, applies
    the multiplicative update until max column error < tol or k_max reached.

    Stall-break: voxel rounding quantizes achievable heights, so a target finer than
    one voxel is unreachable and the update can enter a small limit cycle. If the
    best error does not improve for `stall_patience` consecutive iters, stop and
    return the best iterate. Always returns the best-error iterate seen.
    """
    H_target = target_column_heights(mask0, h)
    cols = H_target > 0
    H_green = H_target * float(bulk_factor)

    best = None
    err_history, warp_history = [], []
    converged = False
    iters = 0
    no_improve = 0
    for k in range(1, int(k_max) + 1):
        iters = k
        green_mask, green_dop = build_green_volume(mask0, dop0, H_green, h)
        H_measured, warp_std, aux = forward_fn(green_mask, green_dop)
        err = max_rel_error(H_target, H_measured, cols)
        err_history.append(err)
        warp_history.append(float(warp_std))
        if best is None or err < best["err"] - 1e-9:
            best = {"err": err, "H_green": H_green.copy(),
                    "green_mask": green_mask, "green_dop": green_dop,
                    "warp_std": float(warp_std)}
            no_improve = 0
        else:
            no_improve += 1
        if err < tol:
            converged = True
            break
        if no_improve >= stall_patience:
            break
        H_green = column_height_update(H_green, H_target, H_measured)

    return {"converged": converged, "iters": iters,
            "H_green": best["H_green"], "green_mask": best["green_mask"],
            "green_dop": best["green_dop"], "warp_std": best["warp_std"],
            "err_history": err_history, "warp_history": warp_history,
            "H_target": H_target}


def emit_prewarped_spec(green_mask: np.ndarray, green_dop: np.ndarray,
                        out_path, provenance: dict) -> None:
    """Write the pre-warped green volume as a staging spec npz. Fields match the
    spec that stage_3d consumes: SOLVE_cont (dopant), part_mask, proxy_field, plus
    a `prewarp` provenance dict. Dopant is zeroed outside the mask for staging."""
    green_mask = np.asarray(green_mask, bool)
    green_dop = np.where(green_mask, np.asarray(green_dop, float), 0.0)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        SOLVE_cont=green_dop.astype(np.float32),
        part_mask=green_mask,
        proxy_field="solve",
        prewarp=np.array(dict(provenance), dtype=object),
    )
