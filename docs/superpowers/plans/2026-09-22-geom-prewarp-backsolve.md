# Green-Geometry Pre-Warp Backsolve Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-OFF, revertible mechanism that pre-warps the green part geometry (per-column Z) so that after the densification collapse the part lands on the nominal CAD boundary, correcting the residual warp the dopant solve leaves behind.

**Architecture:** A sequential outer loop (march-in-the-loop, error-driven backsolve) wraps the *unchanged* dopant solve. Pure per-column geometry logic lives in `solve3d/geom_prewarp.py` (numpy only, fast tests, `.venv312`); the heavy `heatr3d` densify-march forward wrapper + CLI live in `solve3d/geom_prewarp_forward.py`. The loop optimizes a 2-D green-height field `H(x,y)` with a multiplicative update until each column's simulated dense height matches the target; it emits its own `*_prewarped_green_spec.npz` that the existing MetPrint staging consumes. Nothing in the dopant solve or staging changes behavior unless the pre-warped spec is explicitly staged.

**Tech Stack:** Python, numpy, `heatr3d` (scipy-based 3-D solver, runs in `.venv312`), pytest. Physics reused (not rebuilt): `heatr3d.shrinkage_factors`, `heatr3d.shrinkage_analysis`, `heatr3d.run(densify=True)`.

**Conventions locked for this module:** arrays are `(nx, ny, nz)` with **z = axis 2** (heatr3d's build axis, the axis `shrinkage_analysis` stacks along). `h` = voxel size (m). Column heights are `(nx, ny)` arrays in meters. `ρ_green = p.rho_rel = 0.55`, `xy_frac = 0.04`.

**Run all commands from:** `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`
**Python:** `./.venv312/bin/python` (call pytest as `./.venv312/bin/python -m pytest`).

---

## File Structure

- Create: `solve3d/geom_prewarp.py` — pure per-column geometry logic + the loop driver (forward injected as a callable). No heatr3d import.
- Create: `solve3d/geom_prewarp_forward.py` — the real `heatr3d` densify-march forward wrapper + the CLI that runs the loop on a saved densify baseline and emits the spec.
- Create: `solve3d/tests/test_geom_prewarp.py` — unit tests for all pure logic (no solver).
- Reference only (do not modify): `heatr3d.py` (`Grid`, `Params`, `run`, `shrinkage_analysis`, `shrinkage_factors`), `solve3d/results/densify_{cube,pyramid}/fields.npz` (baseline `part`, `sat`, `h`, `L`, `rho_final`).
- Staging (`software/meteor/tools/`) is **not** modified by the core plan; it already consumes a `proxy_field="solve"` spec. Task 8 is an OPTIONAL provenance stamp.

---

## Task 1: Per-column 1-D resample (`resample_column`)

**Files:**
- Create: `solve3d/geom_prewarp.py`
- Test: `solve3d/tests/test_geom_prewarp.py`

- [ ] **Step 1: Write the failing test**

```python
# solve3d/tests/test_geom_prewarp.py
import numpy as np
import pytest
from solve3d import geom_prewarp as gp


def test_resample_column_stretches_and_preserves_endpoints():
    src = np.linspace(0.0, 1.0, 4)          # 4 -> 9, monotone ramp
    out = gp.resample_column(src, 9)
    assert out.shape == (9,)
    assert abs(out[0] - 0.0) < 0.05 and abs(out[-1] - 1.0) < 0.05
    assert np.all(np.diff(out) >= -1e-9)


def test_resample_column_identity_and_empty():
    src = np.array([2.0, 5.0, 7.0])
    assert np.allclose(gp.resample_column(src, 3), src)
    assert gp.resample_column(np.zeros(0), 4).shape == (4,)   # empty src -> zeros
    assert gp.resample_column(src, 0).shape == (0,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'solve3d.geom_prewarp'`.

- [ ] **Step 3: Write minimal implementation**

```python
# solve3d/geom_prewarp.py
"""Green-geometry pre-warp backsolve (Z densification compensation).

Pure per-column geometry logic + the outer-loop driver. The heavy heatr3d
densify march is injected as a callable so this module has NO solver dependency
and its tests run in the pure-numpy venv. Convention: arrays are (nx, ny, nz)
with z = axis 2 (heatr3d build axis); column heights are (nx, ny) in metres.
"""
from __future__ import annotations

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add solve3d/geom_prewarp.py solve3d/tests/test_geom_prewarp.py
git commit -m "feat(geom-prewarp): per-column 1-D resample helper"
```

---

## Task 2: Target heights, multiplicative update, convergence metric

**Files:**
- Modify: `solve3d/geom_prewarp.py`
- Test: `solve3d/tests/test_geom_prewarp.py`

- [ ] **Step 1: Write the failing test**

```python
def test_target_column_heights_counts_occupied_times_h():
    mask = np.zeros((2, 2, 5), bool)
    mask[0, 0, :3] = True          # column (0,0): 3 voxels
    mask[1, 1, :5] = True          # column (1,1): 5 voxels
    H = gp.target_column_heights(mask, h=0.2)
    assert H.shape == (2, 2)
    assert np.isclose(H[0, 0], 0.6) and np.isclose(H[1, 1], 1.0)
    assert H[0, 1] == 0.0


def test_column_height_update_multiplicative_and_masks_empty():
    Hg = np.array([[1.0, 1.0]])
    Ht = np.array([[2.0, 0.0]])     # 2nd column not part
    Hm = np.array([[1.0, 0.0]])     # measured half of target -> gain 2x
    out = gp.column_height_update(Hg, Ht, Hm)
    assert np.isclose(out[0, 0], 2.0)   # 1.0 * 2.0/1.0
    assert out[0, 1] == 0.0             # non-part column stays 0


def test_max_rel_error_over_part_columns():
    Ht = np.array([[10.0, 0.0], [10.0, 10.0]])
    Hm = np.array([[9.5, 0.0], [10.0, 8.0]])   # errors 5%, -, 0%, 20%
    assert abs(gp.max_rel_error(Ht, Hm) - 0.20) < 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py -q`
Expected: FAIL — `AttributeError: module 'solve3d.geom_prewarp' has no attribute 'target_column_heights'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to solve3d/geom_prewarp.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add solve3d/geom_prewarp.py solve3d/tests/test_geom_prewarp.py
git commit -m "feat(geom-prewarp): target heights, multiplicative update, error metric"
```

---

## Task 3: Build the pre-warped green volume (`build_green_volume`)

**Files:**
- Modify: `solve3d/geom_prewarp.py`
- Test: `solve3d/tests/test_geom_prewarp.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_green_volume_stretches_columns_and_conserves_mask():
    # nominal: column A 3 voxels tall, column B 2 voxels; dopant ramps in z
    nx, ny, nz0 = 2, 1, 4
    mask0 = np.zeros((nx, ny, nz0), bool)
    dop0 = np.zeros((nx, ny, nz0))
    mask0[0, 0, :3] = True; dop0[0, 0, :3] = [0.2, 0.5, 0.8]
    mask0[1, 0, :2] = True; dop0[1, 0, :2] = [0.4, 0.6]
    h = 0.2
    # ask column A to be 6 voxels tall (double), B to be 2 (unchanged)
    H_green = np.array([[6 * h], [2 * h]])
    gm, gd = gp.build_green_volume(mask0, dop0, H_green, h)
    assert gm.shape == (nx, ny, 6) and gd.shape == (nx, ny, 6)
    # column A: 6 occupied, dopant nonzero within, zero above
    assert gm[0, 0].sum() == 6 and gm[0, 0, :6].all()
    assert np.all(gd[0, 0, :6] > 0)
    # column B: 2 occupied, rest empty
    assert gm[1, 0].sum() == 2 and not gm[1, 0, 2:].any()
    assert np.all(gd[1, 0, 2:] == 0.0)
    # dopant range preserved (resample stays within source min/max)
    assert 0.2 - 1e-9 <= gd[0, 0, :6].min() and gd[0, 0, :6].max() <= 0.8 + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py::test_build_green_volume_stretches_columns_and_conserves_mask -q`
Expected: FAIL — `AttributeError: ... has no attribute 'build_green_volume'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to solve3d/geom_prewarp.py

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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add solve3d/geom_prewarp.py solve3d/tests/test_geom_prewarp.py
git commit -m "feat(geom-prewarp): build per-column pre-warped green volume"
```

---

## Task 4: The outer loop with an injected forward (`prewarp_solve`)

**Files:**
- Modify: `solve3d/geom_prewarp.py`
- Test: `solve3d/tests/test_geom_prewarp.py`

**Why injected forward:** the loop logic must be testable without the heavy solver. `forward_fn(green_mask, green_dop) -> (H_measured, warp_std, aux)` is passed in; Task 6 supplies the real heatr3d one.

- [ ] **Step 1: Write the failing test**

```python
def test_prewarp_solve_converges_against_uniform_forward():
    # synthetic part: 3x3 footprint, nominal 4 voxels tall, uniform dopant
    nx, ny, nz0 = 3, 3, 8
    mask0 = np.zeros((nx, ny, nz0), bool); mask0[:, :, :4] = True
    dop0 = np.where(mask0, 0.5, 0.0)
    h = 0.2
    # LAM_Z=0.5 keeps the analytic green height an EXACT voxel multiple
    # (4 voxels / 0.5 = 8 voxels), so the loop can reach tol without fighting
    # voxel-rounding quantization (that quantization is exercised separately below).
    LAM_Z = 0.5

    def forward(green_mask, green_dop):
        # dense height = occupied green voxels * h * LAM_Z, per column
        occ = green_mask.sum(axis=2).astype(float)
        H_measured = occ * h * LAM_Z
        warp = 0.0                   # perfectly uniform forward
        return H_measured, warp, {}

    res = gp.prewarp_solve(mask0, dop0, h, forward,
                           bulk_factor=1.0, tol=0.01, k_max=8)
    assert res["converged"] is True
    # analytic green height = target / LAM_Z = (4*h) / 0.5 = 8 voxels
    Ht = gp.target_column_heights(mask0, h)
    expect = Ht / LAM_Z
    assert np.allclose(res["H_green"][Ht > 0], expect[Ht > 0], rtol=0.02)
    # converged under tol
    assert res["err_history"][-1] < 0.01
    assert res["iters"] >= 1


def test_prewarp_solve_stall_breaks_on_voxel_quantization():
    # LAM_Z=0.6 makes the analytic green height 6.67 voxels -> unreachable to 1%
    # by integer voxels (best is ~5%). The loop must NOT spin to k_max: it detects
    # the stall (no improvement over the best iterate for 2 iters) and returns best.
    nx, ny, nz0 = 3, 3, 8
    mask0 = np.zeros((nx, ny, nz0), bool); mask0[:, :, :4] = True
    dop0 = np.where(mask0, 0.5, 0.0)
    h = 0.2

    def forward(green_mask, green_dop):
        occ = green_mask.sum(axis=2).astype(float)
        return occ * h * 0.6, 0.0, {}

    res = gp.prewarp_solve(mask0, dop0, h, forward,
                           bulk_factor=1.0, tol=0.01, k_max=20)
    assert res["converged"] is False        # cannot reach 1% at this resolution
    assert res["iters"] < 8                  # stall-break stopped it early (not k_max)
    assert min(res["err_history"]) < 0.06    # best achievable ~5%
    assert res["warp_std"] is not None       # best iterate was captured/returned
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py::test_prewarp_solve_converges_against_uniform_forward -q`
Expected: FAIL — `AttributeError: ... has no attribute 'prewarp_solve'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to solve3d/geom_prewarp.py

def prewarp_solve(mask0: np.ndarray, dop0: np.ndarray, h: float, forward_fn,
                  *, bulk_factor: float = 1.0, tol: float = 0.01,
                  k_max: int = 5, stall_patience: int = 2) -> dict:
    """Sequential outer-loop green-geometry backsolve.

    forward_fn(green_mask, green_dop) -> (H_measured(nx,ny), warp_std, aux).
    Warm start: green height = target * bulk_factor. Each iter builds the green
    volume, marches it (forward_fn), measures per-column dense height, and applies
    the multiplicative update until max column error < tol or k_max reached.

    Stall-break: voxel rounding in build_green_volume quantizes achievable heights,
    so a target finer than one voxel is unreachable and the update can enter a small
    limit cycle. If the best error does not improve for `stall_patience` consecutive
    iters, stop and return the best iterate rather than spinning to k_max. Always
    returns the best-error iterate seen.
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
        if no_improve >= stall_patience:        # voxel-quantization limit cycle
            break
        H_green = column_height_update(H_green, H_target, H_measured)

    return {"converged": converged, "iters": iters,
            "H_green": best["H_green"], "green_mask": best["green_mask"],
            "green_dop": best["green_dop"], "warp_std": best["warp_std"],
            "err_history": err_history, "warp_history": warp_history,
            "H_target": H_target}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py -q`
Expected: PASS (7 passed).

- [ ] **Step 5: Commit**

```bash
git add solve3d/geom_prewarp.py solve3d/tests/test_geom_prewarp.py
git commit -m "feat(geom-prewarp): outer-loop backsolve with injected forward"
```

---

## Task 5: Emit the pre-warped green spec + provenance (`emit_prewarped_spec`)

**Files:**
- Modify: `solve3d/geom_prewarp.py`
- Test: `solve3d/tests/test_geom_prewarp.py`

- [ ] **Step 1: Write the failing test**

```python
def test_emit_prewarped_spec_roundtrips(tmp_path):
    gm = np.zeros((2, 2, 3), bool); gm[:, :, :2] = True
    gd = np.where(gm, 0.5, 0.0)
    prov = {"enabled": True, "iters": 3, "converged": True, "tol": 0.01,
            "warp_std_before": 9.5, "warp_std_after": 2.0,
            "bulk_factor": 1.58, "source_densify": "densify_pyramid"}
    out = tmp_path / "pyr_prewarped_green_spec.npz"
    gp.emit_prewarped_spec(gm, gd, out, prov)
    d = np.load(out, allow_pickle=True)
    assert str(d["proxy_field"]) == "solve"
    assert d["SOLVE_cont"].shape == gd.shape
    assert d["part_mask"].shape == gm.shape and d["part_mask"].dtype == bool
    rec = d["prewarp"].item()          # dict round-trips via object array
    assert rec["enabled"] is True and rec["converged"] is True
    # dopant zero outside the mask (staging validity)
    assert np.all(d["SOLVE_cont"][~d["part_mask"]] == 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py::test_emit_prewarped_spec_roundtrips -q`
Expected: FAIL — `AttributeError: ... has no attribute 'emit_prewarped_spec'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to solve3d/geom_prewarp.py
from pathlib import Path


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add solve3d/geom_prewarp.py solve3d/tests/test_geom_prewarp.py
git commit -m "feat(geom-prewarp): emit pre-warped green staging spec + provenance"
```

---

## Task 6: Real heatr3d forward wrapper (`march_dense_heights`)

**Files:**
- Create: `solve3d/geom_prewarp_forward.py`
- Test: `solve3d/tests/test_geom_prewarp_forward.py`

**Note on axes/placement:** the baseline `densify_{part}/fields.npz` already holds a correctly placed `part` (nx,ny,nz), `sat`, `h`, `L` at n=48 with build axis = z (axis 2). The forward wrapper embeds the green volume in the same `n=L/h` cubic grid, base of the part at the same z-start as the baseline, footprint centred as in the baseline.

- [ ] **Step 1: Write the failing test** (light: verifies embedding + the shrinkage read shape, mocking `heatr3d.run` so no heavy solve runs)

```python
# solve3d/tests/test_geom_prewarp_forward.py
import numpy as np
import types
import pytest
from solve3d import geom_prewarp_forward as gpf


def test_embed_in_cubic_grid_places_base_at_z0_centred_xy():
    green = np.zeros((3, 3, 4), bool); green[:, :, :4] = True
    gd = np.where(green, 0.5, 0.0)
    part, sat = gpf.embed_in_grid(green, gd, n=8, z0=1)
    assert part.shape == (8, 8, 8) and sat.shape == (8, 8, 8)
    # footprint centred in x,y (offset (8-3)//2 = 2), base at z0=1
    assert part[2:5, 2:5, 1:5].all()
    assert part.sum() == green.sum()
    assert np.all(sat[~part] == 0.0)


def test_march_dense_heights_reads_shrinkage(monkeypatch):
    # mock heatr3d.run -> a Result-like object; mock shrinkage_analysis
    import heatr3d as H
    fake_res = types.SimpleNamespace(rho_final=np.ones((8, 8, 8)) * 0.9,
                                     part=np.ones((8, 8, 8), bool))
    monkeypatch.setattr(H, "run", lambda *a, **k: fake_res)
    monkeypatch.setattr(H, "shrinkage_analysis",
                        lambda res, p, h, **k: {"_H_final": np.ones((8, 8)) * 0.5,
                                                "warp_std_pct": 3.0})
    grid = H.Grid(n=8, L=8 * 0.001)
    part = np.ones((8, 8, 8), bool); sat = np.full((8, 8, 8), 0.5)
    Hm, warp, res = gpf.march_dense_heights(part, sat, H.Params(), grid)
    assert Hm.shape == (8, 8) and np.isclose(warp, 3.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp_forward.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'solve3d.geom_prewarp_forward'`.

- [ ] **Step 3: Write minimal implementation**

```python
# solve3d/geom_prewarp_forward.py
"""Heavy heatr3d densify-march forward for the green-geometry pre-warp loop, plus
the CLI that runs the loop on a saved densify baseline and emits the spec. Imports
heatr3d (scipy); runs in .venv312. Kept separate from geom_prewarp.py so the pure
logic tests never import the solver."""
from __future__ import annotations

import numpy as np

import heatr3d as H
from solve3d import geom_prewarp as gp


def embed_in_grid(green_mask: np.ndarray, green_dop: np.ndarray, n: int,
                  z0: int = 0):
    """Place a (nx,ny,nz) green volume into a cubic (n,n,n) grid: footprint
    centred in x,y, base at z=z0. Returns (part, sat), sat zeroed outside part."""
    gx, gy, gz = green_mask.shape
    if gx > n or gy > n or z0 + gz > n:
        raise ValueError(f"green volume {green_mask.shape} + z0={z0} does not fit "
                         f"in a {n}^3 grid; raise n or scale the part down")
    part = np.zeros((n, n, n), bool)
    sat = np.zeros((n, n, n), float)
    ox, oy = (n - gx) // 2, (n - gy) // 2
    part[ox:ox + gx, oy:oy + gy, z0:z0 + gz] = green_mask
    sat[ox:ox + gx, oy:oy + gy, z0:z0 + gz] = np.where(green_mask, green_dop, 0.0)
    return part, sat


def march_dense_heights(part: np.ndarray, sat: np.ndarray, p: "H.Params",
                        grid: "H.Grid", max_time_s: float = 1200.0):
    """One densify march + shrinkage read. Returns (H_measured(nx,ny) in metres
    over the FULL grid footprint, warp_std_pct, Result)."""
    res = H.run(grid, part, p, sat=sat, max_time_s=max_time_s, densify=True)
    sh = H.shrinkage_analysis(res, p, grid.h)
    return np.asarray(sh["_H_final"], float), float(sh["warp_std_pct"]), res
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp_forward.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add solve3d/geom_prewarp_forward.py solve3d/tests/test_geom_prewarp_forward.py
git commit -m "feat(geom-prewarp): heatr3d densify-march forward wrapper"
```

---

## Task 7: CLI runner that wires the loop to the real march + emits the spec

**Files:**
- Modify: `solve3d/geom_prewarp_forward.py`
- Test: `solve3d/tests/test_geom_prewarp_forward.py`

The CLI loads a baseline `densify_{part}/fields.npz`, derives the nominal (target)
mask + dopant and `h`, builds a `forward_fn` closure that embeds each green volume
in the grid and marches it, runs `gp.prewarp_solve`, then emits the spec + a JSON
record. `forward_fn` must return `H_measured` cropped back to the part footprint so
it aligns with `target_column_heights(mask0)`.

- [ ] **Step 1: Write the failing test** (unit-tests the footprint crop; the full run is a manual command below)

```python
def test_crop_to_footprint_matches_nominal_shape():
    # a 3x3 footprint centred in an 8x8 grid; crop returns the 3x3 block
    full = np.zeros((8, 8)); full[2:5, 2:5] = np.arange(9).reshape(3, 3)
    crop = gpf.crop_to_footprint(full, nx=3, ny=3)
    assert crop.shape == (3, 3)
    assert np.allclose(crop, np.arange(9).reshape(3, 3))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp_forward.py::test_crop_to_footprint_matches_nominal_shape -q`
Expected: FAIL — `AttributeError: ... has no attribute 'crop_to_footprint'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to solve3d/geom_prewarp_forward.py
import argparse
import json
from pathlib import Path


def crop_to_footprint(full_xy: np.ndarray, nx: int, ny: int) -> np.ndarray:
    """Crop a full-grid (n,n) column-map back to the centred (nx,ny) footprint."""
    n = full_xy.shape[0]
    ox, oy = (n - nx) // 2, (n - ny) // 2
    return np.asarray(full_xy)[ox:ox + nx, oy:oy + ny]


def run_prewarp(fields_npz: str, out_spec: str, *, bulk_factor: float | None = None,
                tol: float = 0.01, k_max: int = 5, grid_n: int = 48,
                z0: int = 1, max_time_s: float = 1200.0) -> dict:
    """Load a densify baseline, run the green-geometry backsolve on the real
    march, emit the pre-warped spec + a JSON record next to it."""
    d = np.load(fields_npz, allow_pickle=True)
    mask0 = np.asarray(d["part"], bool)          # (nx,ny,nz) nominal/target
    dop0 = np.where(mask0, np.asarray(d["sat"], float), 0.0)
    h = float(d["h"])
    p = H.Params()
    grid = H.Grid(n=grid_n, L=grid_n * h)
    nx, ny, _ = mask0.shape
    if bulk_factor is None:            # warm start: bulk factor from the baseline march
        part0, sat0 = embed_in_grid(mask0, dop0, grid_n, z0)
        sh0 = H.shrinkage_analysis(
            H.run(grid, part0, p, sat=sat0, max_time_s=max_time_s, densify=True),
            p, h)
        bulk_factor = float(sh0["layer_multiplier"])

    def forward_fn(green_mask, green_dop):
        part, sat = embed_in_grid(green_mask, green_dop, grid_n, z0)
        Hm_full, warp, _res = march_dense_heights(part, sat, p, grid, max_time_s)
        gx, gy, _ = green_mask.shape
        return crop_to_footprint(Hm_full, gx, gy), warp, {}

    res = gp.prewarp_solve(mask0, dop0, h, forward_fn,
                           bulk_factor=bulk_factor, tol=tol, k_max=k_max)
    prov = {"enabled": True, "iters": res["iters"], "converged": res["converged"],
            "tol": tol, "bulk_factor": bulk_factor,
            "warp_std_history": res["warp_history"],
            "err_history": res["err_history"],
            "source_densify": Path(fields_npz).parent.name}
    gp.emit_prewarped_spec(res["green_mask"], res["green_dop"], out_spec, prov)
    Path(str(out_spec) + ".record.json").write_text(json.dumps(prov, indent=2))
    return prov


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="green-geometry pre-warp backsolve")
    ap.add_argument("fields_npz", help="densify baseline fields.npz (part/sat/h)")
    ap.add_argument("out_spec", help="output *_prewarped_green_spec.npz")
    ap.add_argument("--bulk-factor", type=float, default=None)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--k-max", type=int, default=5)
    ap.add_argument("--grid-n", type=int, default=48)
    args = ap.parse_args(argv)
    prov = run_prewarp(args.fields_npz, args.out_spec, bulk_factor=args.bulk_factor,
                       tol=args.tol, k_max=args.k_max, grid_n=args.grid_n)
    print(json.dumps(prov, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

Then delete the dead `types_stub`/`if False` warm-start stub and replace the
`bulk_factor is None` branch with the real baseline read:

```python
    if bulk_factor is None:
        part0, sat0 = embed_in_grid(mask0, dop0, grid_n, z0)
        sh0 = H.shrinkage_analysis(
            H.run(grid, part0, p, sat=sat0, max_time_s=max_time_s, densify=True),
            p, h)
        bulk_factor = float(sh0["layer_multiplier"])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv312/bin/python -m pytest solve3d/tests/test_geom_prewarp_forward.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add solve3d/geom_prewarp_forward.py solve3d/tests/test_geom_prewarp_forward.py
git commit -m "feat(geom-prewarp): CLI runner wiring the loop to the real march"
```

**Note:** Task 7 Step 3's `run_prewarp` warm-start reads `layer_multiplier` from one
baseline march when `--bulk-factor` is omitted; pass `--bulk-factor 1.707` (cube) or
`1.583` (pyramid) to skip that extra march since we already measured them.

---

## Task 8: Integration round-trip on the pyramid (acceptance gate — manual heavy run)

**Files:**
- Reference: `solve3d/results/densify_pyramid/fields.npz`
- Output: `solve3d/results/prewarp_pyramid/pyramid_prewarped_green_spec.npz` (+ `.record.json`)

This is the acceptance gate from the spec (§5): warp must drop vs the uncompensated
baseline. It runs real densify marches (heavy) — one at a time, checkpointed.

- [ ] **Step 1: Run the pyramid pre-warp (heavy)**

Run:
```bash
./.venv312/bin/python -m solve3d.geom_prewarp_forward \
  solve3d/results/densify_pyramid/fields.npz \
  solve3d/results/prewarp_pyramid/pyramid_prewarped_green_spec.npz \
  --tol 0.01 --k-max 5 --grid-n 48
```
Expected: JSON with `converged: true` (or best-effort at k_max), an `err_history`
that decreases, and a `warp_std_history`.

- [ ] **Step 2: Verify the acceptance gate**

Run:
```bash
./.venv312/bin/python -c "
import json
r = json.load(open('solve3d/results/prewarp_pyramid/pyramid_prewarped_green_spec.npz.record.json'))
before = 9.5   # uncompensated pyramid warp_std_pct (baseline densify)
after = r['warp_std_history'][r['err_history'].index(min(r['err_history']))]
print('iters', r['iters'], 'converged', r['converged'])
print('err_history (MAX rel col err)', [round(e,4) for e in r['err_history']])
print('best MAX rel col err', round(min(r['err_history']),4))
print('warp_std before/after', before, round(after,3))
print('ACCEPTANCE:', 'PASS' if after <= before else 'NO WARP IMPROVEMENT (investigate)')
"
```
**Interpreting the gate honestly:** `err_history` is the *max* per-column relative
height error. For a tapered part it is dominated by the near-apex slivers (1–3
voxels tall, where ±1 voxel is a huge relative error), so it will **plateau at the
apex voxel resolution, not reach 1–2%** — that is expected, not a failure, and the
stall-break is designed to stop there. The **primary acceptance signal is warp_std
dropping vs the 9.5% baseline**; the max-error plateau just confirms the loop
resolved to voxel resolution. (For the *cube*, whose columns are all full-height,
the max-error metric IS meaningful and should approach voxel resolution.) If warp
does not improve, record it as a real finding — do not force it. A future refinement
(not in this plan) is a height-weighted or apex-excluded convergence metric.

- [ ] **Step 3: Stage the pre-warped pyramid (prewarp ON) beside the nominal (OFF)**

Run (from the meteor tools dir; `<HF>` = the shared hot folder):
```bash
cd "../../software/meteor/tools" && ./stage_from_geo.sh   # if present, else:
"<geo>/.venv312/bin/python" stage_job.py \
  "<geo>/solve3d/results/prewarp_pyramid/pyramid_prewarped_green_spec.npz" \
  --3d --stl "<geo>/shape_library_3d/stl/pyramid.stl" \
  --hot-folder "<HF>" --chamber-mm 29.06 --layer-height 0.2 \
  --job-name pyramid_prewarped_3d
```
Expected: `all_pass: true`; a new job dir alongside the nominal `pyramid_graded_3d`.
Reverting = simply stage the nominal spec instead (prewarp OFF is the default path).

- [ ] **Step 4: Commit the artifacts + record**

```bash
cd "<geo>"
git add solve3d/results/prewarp_pyramid/pyramid_prewarped_green_spec.npz.record.json
git commit -m "test(geom-prewarp): pyramid round-trip record (warp before/after)"
```

---

## Task 9 (OPTIONAL): Stamp pre-warp provenance into `job_info.json`

**Files:**
- Modify: `software/meteor/tools/stage_lib.py` (`_emit_and_slice` manifest block)
- Test: `software/meteor/tools/test_stage.py`

Only if we want the staged job to carry the pre-warp provenance. Toggle-safe: when
the spec has no `prewarp` field, nothing changes.

- [ ] **Step 1: Write the failing test**

```python
def test_job_info_carries_prewarp_block_when_present(tmp_path):
    # a spec-less direct _emit_and_slice does NOT add prewarp; stage_3d passes it.
    # Here assert the helper copies a provided prewarp dict into job_info.
    import stage_lib as sl, json
    layers = [np.clip(np.ones((8, 8)) * 0.5, 0, 1) for _ in range(3)]
    ax = np.arange(8)
    tiffs = sl._emit_and_slice(layers, ax, ax, tmp_path / "j", "p",
                               720, 4, 8, 0.2, prewarp={"enabled": True, "iters": 3})
    info = json.load(open(tmp_path / "j" / "job_info.json"))
    assert info["prewarp"] == {"enabled": True, "iters": 3}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd "../../software/meteor/tools" && "<geo>/.venv312/bin/python" -m pytest test_stage.py::test_job_info_carries_prewarp_block_when_present -q`
Expected: FAIL — `_emit_and_slice() got an unexpected keyword argument 'prewarp'`.

- [ ] **Step 3: Write minimal implementation**

Add a keyword to `_emit_and_slice` and include it in the manifest:

```python
# stage_lib.py  _emit_and_slice signature: add  prewarp: dict | None = None
# in the `info = {...}` manifest dict, add:
        **({"prewarp": prewarp} if prewarp else {}),
```
Thread `prewarp` from `stage_3d`: read it from the spec (`spec.get("prewarp")`,
`.item()` if it is a 0-d object array) and pass it into `_emit_and_slice`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd "../../software/meteor/tools" && "<geo>/.venv312/bin/python" -m pytest test_stage.py -q`
Expected: PASS (all green).

- [ ] **Step 5: Commit** (meteor tools are non-git per repo convention — note the edit in the session log instead; do not `git add` outside the geo-prewarp repo).

---

## Self-Review

**Spec coverage:**
- §2 approach (sequential loop, per-column H, dopant fixed, multiplicative update, warm start) → Tasks 2, 4, 7. ✓
- §2 full per-column z-remap → Task 3 (`build_green_volume` resamples each column). ✓
- §3 toggle/revert/default-off → separate module + own spec file (Tasks 5,7); staging unchanged; revert = stage nominal; provenance Task 5/9. ✓
- §4 interfaces (inputs, forward, output spec) → Tasks 5,6,7. ✓
- §5 verification (unit update, z-remap mass/ordering, toggle, pyramid round-trip) → Tasks 1–5 units, Task 8 round-trip. ✓
- §6 compute (march per iter, k_max, tol, checkpoint) → Task 7 args, Task 8 run. ✓
- §7 assumptions (monotonic, dopant frozen, uncalibrated law) → encoded in module docstrings + provenance. ✓

**Placeholder scan:** Task 7 Step 3 deliberately shows a dead `types_stub`/`if False` stub then instructs replacing it with the real baseline-read block — the final code has no placeholder. No other TBD/TODO. ✓

**Type consistency:** `H_green/H_target/H_measured` are `(nx,ny)` metres throughout; `build_green_volume`/`prewarp_solve`/`emit_prewarped_spec` names and signatures match across tasks; `forward_fn` returns `(H_measured, warp_std, aux)` consistently (Tasks 4, 7); `_H_final`/`warp_std_pct` keys match `heatr3d.shrinkage_analysis`. ✓
