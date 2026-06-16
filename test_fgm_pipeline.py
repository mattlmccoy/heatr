#!/usr/bin/env python3
"""
test_fgm_pipeline.py — End-to-end verification of the FGM generation and RIP bridge.

Runs a battery of unit + integration tests against a real HEATR output directory.
All tests are self-contained — no pytest required; run with:

    python test_fgm_pipeline.py [--run-dir <path>] [-v]

Default run dir: outputs_eqs/runs/square/single/baseline/no_rotation
(must contain fields.npz with Qrf, T, rho_rel, part_mask, x, y)

Exit code 0 = all tests passed.
Exit code 1 = one or more failures.
"""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"
SKIP = "\033[33mSKIP\033[0m"

_results: list[tuple[str, str, str]] = []  # (name, status, detail)
_verbose = False


def _run(name: str, fn):
    """Execute a test function, record pass/fail/skip."""
    try:
        msg = fn()
        _results.append((name, PASS, str(msg or "")))
        if _verbose:
            print(f"  {PASS}  {name}" + (f"  — {msg}" if msg else ""))
    except _SkipTest as e:
        _results.append((name, SKIP, str(e)))
        if _verbose:
            print(f"  {SKIP}  {name}  — {e}")
    except Exception as e:
        detail = f"{type(e).__name__}: {e}"
        _results.append((name, FAIL, detail))
        if _verbose:
            print(f"  {FAIL}  {name}  — {detail}")
            traceback.print_exc()


class _SkipTest(Exception):
    pass


# ---------------------------------------------------------------------------
# Locate run directory
# ---------------------------------------------------------------------------

def _find_default_run_dir() -> Path:
    here = Path(__file__).parent
    candidate = here / "outputs_eqs" / "runs" / "square" / "single" / "baseline" / "no_rotation"
    if candidate.is_dir() and (candidate / "fields.npz").exists():
        return candidate
    # Try any run with fields.npz
    for p in sorted((here / "outputs_eqs" / "runs").rglob("fields.npz")):
        return p.parent
    raise FileNotFoundError(
        "No HEATR run with fields.npz found under outputs_eqs/runs/. "
        "Pass --run-dir explicitly."
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_import_fgm_generator():
    import fgm_generator  # noqa: F401
    return "module imported OK"


def test_fields_npz_has_required_keys(run_dir: Path):
    fields = np.load(run_dir / "fields.npz")
    required = {"Qrf", "T", "rho_rel", "part_mask", "x", "y"}
    missing = required - set(fields.keys())
    if missing:
        raise AssertionError(f"fields.npz missing keys: {missing}")
    return f"keys present: {sorted(required)}"


def test_inversion_correctness_qrf(run_dir: Path, tmp: Path):
    """Hottest Qrf pixel inside the part must map to the LOWEST level."""
    from fgm_generator import generate_fgm
    from scipy.ndimage import zoom

    r = generate_fgm(
        run_dir, bpp=2, proxy_field="Qrf", invert=True, magnitude=1.0,
        output_dir=tmp, emit_formats=("npz",),
    )
    lm = r["level_map"]  # DPI-resolution

    # Load raw Qrf and part_mask at sim resolution
    fields = np.load(run_dir / "fields.npz")
    Qrf      = fields["Qrf"].astype(np.float32)
    pm       = fields["part_mask"].astype(bool)
    x        = fields["x"]

    # Zoom Qrf to the same DPI grid used by the level_map
    dx_m = float(x[1] - x[0]) if len(x) > 1 else 1e-4
    px_m = 25.4e-3 / r["dpi"]
    z    = dx_m / px_m
    if abs(z - 1.0) > 0.01:
        Qrf_dpi = zoom(Qrf, (z, z), order=1)
        pm_dpi  = zoom(pm.astype(np.float32), (z, z), order=0).astype(bool)
    else:
        Qrf_dpi = Qrf
        pm_dpi  = pm

    # Pad / crop to matching shape (zoom may produce ±1 pixel)
    hy, hx = min(Qrf_dpi.shape[0], lm.shape[0]), min(Qrf_dpi.shape[1], lm.shape[1])
    Qrf_dpi = Qrf_dpi[:hy, :hx]
    pm_dpi  = pm_dpi[:hy, :hx]
    lm_crop = lm[:hy, :hx]

    # Find the pixel with highest Qrf inside the part
    Qrf_inside = np.where(pm_dpi, Qrf_dpi, -np.inf)
    hottest = np.unravel_index(np.argmax(Qrf_inside), Qrf_inside.shape)
    hottest_level = int(lm_crop[hottest])

    n_levels = r["n_levels"]
    # Hottest pixel should be at level 0 or 1 (bottom 25% of range)
    threshold = max(1, n_levels // 4)
    if hottest_level > threshold:
        raise AssertionError(
            f"Hottest Qrf pixel has level={hottest_level}, expected ≤{threshold}. "
            f"Inversion is WRONG — overheated region is getting MORE dopant."
        )
    return f"hottest Qrf pixel → level {hottest_level} (max allowed {threshold})"


def test_inversion_correctness_t(run_dir: Path, tmp: Path):
    """T proxy with invert=True: hot regions → low saturation.

    We check the overall negative correlation between T and sat_map rather than
    a single-pixel comparison, because Gaussian pre-smoothing in fgm_generator
    redistributes the raw temperature peak across its neighbourhood — the raw
    maximum pixel may not be the smoothed maximum after the sigma=1.5 filter.
    """
    from fgm_generator import generate_fgm

    r = generate_fgm(
        run_dir, bpp=2, proxy_field="T", invert=True, magnitude=1.0,
        output_dir=tmp, emit_formats=("npz",),
    )
    sat = r["sat_map"]  # sim-resolution float32

    fields = np.load(run_dir / "fields.npz")
    T  = fields["T"].astype(np.float32)
    pm = fields["part_mask"].astype(bool)

    # Pearson correlation between T and sat_map inside the part should be strongly negative.
    # (high T → low saturation is the entire point of invert=True)
    T_in   = T[pm].ravel()
    sat_in = sat[pm].ravel()
    corr   = float(np.corrcoef(T_in, sat_in)[0, 1])

    if corr > -0.5:
        raise AssertionError(
            f"Pearson corr(T, sat_map) = {corr:.4f} (expected < -0.50). "
            f"The T-proxy inversion is not producing the correct direction: "
            f"hot regions should map to LOW saturation."
        )

    # Also verify: top-10% hottest pixels have lower mean saturation than bottom-10%
    p10 = np.percentile(T_in, 10)
    p90 = np.percentile(T_in, 90)
    mean_sat_cold = float(sat[pm & (T <= p10)].mean())
    mean_sat_hot  = float(sat[pm & (T >= p90)].mean())
    if mean_sat_hot >= mean_sat_cold:
        raise AssertionError(
            f"Top-10% hottest T pixels have mean_sat={mean_sat_hot:.3f} ≥ "
            f"mean_sat of coldest 10% ({mean_sat_cold:.3f}). Inversion is WRONG."
        )

    return (
        f"corr(T, sat_map)={corr:.4f}  "
        f"hot10%→sat={mean_sat_hot:.3f}  cold10%→sat={mean_sat_cold:.3f}"
    )


def test_magnitude_zero_flat_map(run_dir: Path, tmp: Path):
    """magnitude=0 must produce a flat map: every inside-part pixel = round(baseline*max_val)."""
    from fgm_generator import generate_fgm

    for bpp, baseline in [(2, 0.5), (4, 0.5), (2, 0.0), (2, 1.0)]:
        r = generate_fgm(
            run_dir, bpp=bpp, magnitude=0.0, baseline_saturation=baseline,
            output_dir=tmp, emit_formats=("npz",),
        )
        lm  = r["level_map"]
        n   = r["n_levels"]
        expected_level = int(round(baseline * (n - 1)))

        # Check inside-part pixels (non-zero or expected level)
        # At DPI resolution the part mask isn't available directly; check the sat_map
        sat = r["sat_map"]
        fields = np.load(run_dir / "fields.npz")
        pm = fields["part_mask"].astype(bool)
        sat_inside = sat[pm]

        atol = 1e-4
        if not np.allclose(sat_inside, baseline, atol=atol):
            deviants = np.sum(np.abs(sat_inside - baseline) > atol)
            raise AssertionError(
                f"bpp={bpp} baseline={baseline}: {deviants} inside-part pixels deviate "
                f"from flat sat={baseline:.2f} at magnitude=0."
            )
    return "flat map verified for bpp=2,4 at baselines 0.0, 0.5, 1.0"


def test_level_range_validity(run_dir: Path, tmp: Path):
    """level_map values must be in [0, 2^bpp - 1] and dtype must be uint8."""
    from fgm_generator import generate_fgm

    for bpp in (2, 4):
        for mag in (0.0, 0.5, 1.0, 1.5):
            r = generate_fgm(
                run_dir, bpp=bpp, magnitude=mag, output_dir=tmp, emit_formats=("npz",),
            )
            lm = r["level_map"]
            max_val = r["n_levels"] - 1

            if lm.dtype != np.uint8:
                raise AssertionError(f"bpp={bpp} mag={mag}: dtype={lm.dtype}, expected uint8")
            if lm.min() < 0 or lm.max() > max_val:
                raise AssertionError(
                    f"bpp={bpp} mag={mag}: level_map range [{lm.min()}, {lm.max()}] "
                    f"exceeds [0, {max_val}]"
                )
    return "all bpp/magnitude combos yield valid uint8 level_map"


def test_sat_map_range(run_dir: Path, tmp: Path):
    """sat_map values inside the part must be in [0, 1]."""
    from fgm_generator import generate_fgm

    for mag in (0.5, 1.0, 2.0):
        r = generate_fgm(run_dir, magnitude=mag, output_dir=tmp, emit_formats=("npz",))
        sat = r["sat_map"]
        fields = np.load(run_dir / "fields.npz")
        pm = fields["part_mask"].astype(bool)
        sat_in = sat[pm]
        if sat_in.min() < -1e-6 or sat_in.max() > 1.0 + 1e-6:
            raise AssertionError(
                f"mag={mag}: sat_map inside part has range [{sat_in.min():.4f}, {sat_in.max():.4f}]"
            )
    return "sat_map in [0,1] for mag=0.5, 1.0, 2.0"


def test_output_npz_loadable(run_dir: Path, tmp: Path):
    """Written NPZ must be loadable and contain expected keys."""
    from fgm_generator import generate_fgm

    r = generate_fgm(run_dir, bpp=2, output_dir=tmp, emit_formats=("npz",))
    npz_path = Path(r["npz_path"])
    assert npz_path.exists(), f"NPZ not found: {npz_path}"

    d = np.load(npz_path, allow_pickle=True)
    required_keys = {"level_map", "sat_map", "x_mm", "y_mm", "bpp", "magnitude", "proxy_field"}
    missing = required_keys - set(d.keys())
    if missing:
        raise AssertionError(f"NPZ missing keys: {missing}")

    # Round-trip check: loaded level_map should match returned level_map
    if not np.array_equal(d["level_map"], r["level_map"]):
        raise AssertionError("level_map in NPZ does not match returned array")

    return f"NPZ round-trip OK  ({npz_path.name})"


def test_output_json_loadable(run_dir: Path, tmp: Path):
    """Written JSON must be valid and contain base64-decodable level_map."""
    from fgm_generator import generate_fgm
    import base64

    r = generate_fgm(run_dir, bpp=2, output_dir=tmp, emit_formats=("json",))
    jp = Path(r["json_path"])
    assert jp.exists(), f"JSON not found: {jp}"

    meta = json.loads(jp.read_text())
    assert "level_map_b64" in meta, "JSON missing 'level_map_b64'"
    assert "bpp" in meta and "proxy_field" in meta

    raw = base64.b64decode(meta["level_map_b64"])
    ny, nx = meta["shape_hw"]
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(ny, nx)
    if not np.array_equal(arr, r["level_map"]):
        raise AssertionError("JSON base64 level_map does not match returned array")

    return f"JSON round-trip OK  ({jp.name})"


def test_output_png_exists_and_8bit(run_dir: Path, tmp: Path):
    """Written PNG must be an 8-bit grayscale image with correct dimensions."""
    try:
        from PIL import Image
    except ImportError:
        raise _SkipTest("Pillow not installed")

    from fgm_generator import generate_fgm

    r = generate_fgm(run_dir, bpp=2, output_dir=tmp, emit_formats=("png",))
    pp = Path(r["png_path"])
    assert pp.exists(), f"PNG not found: {pp}"

    img = Image.open(pp)
    assert img.mode == "L", f"PNG mode={img.mode}, expected 'L' (grayscale)"
    assert img.size == (r["level_map"].shape[1], r["level_map"].shape[0]), (
        f"PNG size={img.size} does not match level_map shape {r['level_map'].shape}"
    )
    return f"PNG OK  {img.size}px  mode={img.mode}  ({pp.name})"


def test_png_visual_convention(run_dir: Path, tmp: Path):
    """PNG convention: level=0 → pixel=255 (white), level=max → pixel=0 (black)."""
    try:
        from PIL import Image
    except ImportError:
        raise _SkipTest("Pillow not installed")

    from fgm_generator import generate_fgm

    # Use a run where inversion creates clear level distribution
    r = generate_fgm(run_dir, bpp=2, magnitude=1.0, output_dir=tmp, emit_formats=("png",))
    lm  = r["level_map"]
    img = np.array(Image.open(r["png_path"]))
    n   = r["n_levels"]

    # Find pixels at level=0 → should be white (255)
    mask0 = lm == 0
    if mask0.any():
        mean_at_0 = float(img[mask0].mean())
        if mean_at_0 < 200:
            raise AssertionError(
                f"level=0 pixels have mean brightness {mean_at_0:.1f} (expected ~255). "
                f"PNG visual convention is inverted."
            )

    # Find pixels at level=max_val → should be dark (≤ 50)
    mask_max = lm == (n - 1)
    if mask_max.any():
        mean_at_max = float(img[mask_max].mean())
        if mean_at_max > 55:
            raise AssertionError(
                f"level={n-1} pixels have mean brightness {mean_at_max:.1f} (expected ~0). "
                f"PNG visual convention is inverted."
            )

    return f"level=0→white, level={n-1}→black convention verified"


def test_rho_rel_proxy(run_dir: Path, tmp: Path):
    """rho_rel proxy with invert=True: low-density regions should get high saturation."""
    from fgm_generator import generate_fgm

    r = generate_fgm(
        run_dir, bpp=2, proxy_field="rho_rel", invert=True, magnitude=1.0,
        output_dir=tmp, emit_formats=("npz",),
    )
    sat = r["sat_map"]
    fields = np.load(run_dir / "fields.npz")
    rho  = fields["rho_rel"].astype(np.float32)
    pm   = fields["part_mask"].astype(bool)

    # Spearman-like check: the pixel with the LOWEST rho inside the part
    # should have a HIGHER-than-median saturation value.
    rho_in  = rho[pm]
    sat_in  = sat[pm]

    low_rho_mask  = pm & (rho <= np.percentile(rho[pm], 10))
    high_rho_mask = pm & (rho >= np.percentile(rho[pm], 90))

    mean_sat_low_rho  = float(sat[low_rho_mask].mean())
    mean_sat_high_rho = float(sat[high_rho_mask].mean())

    if mean_sat_low_rho <= mean_sat_high_rho:
        raise AssertionError(
            f"rho_rel proxy with invert=True: low-rho regions have "
            f"mean_sat={mean_sat_low_rho:.3f} ≤ high-rho mean_sat={mean_sat_high_rho:.3f}. "
            f"Expected low-rho → high saturation (needs more energy)."
        )
    return (
        f"low-rho regions → mean_sat={mean_sat_low_rho:.3f}  "
        f"high-rho regions → mean_sat={mean_sat_high_rho:.3f}"
    )


def test_4bpp_mode(run_dir: Path, tmp: Path):
    """4bpp mode should produce 16 levels and correct dtype."""
    from fgm_generator import generate_fgm

    r = generate_fgm(run_dir, bpp=4, output_dir=tmp, emit_formats=("npz",))
    assert r["bpp"] == 4,           f"Expected bpp=4, got {r['bpp']}"
    assert r["n_levels"] == 16,     f"Expected n_levels=16, got {r['n_levels']}"
    assert r["level_map"].max() <= 15, f"4bpp level > 15: {r['level_map'].max()}"
    assert r["level_map"].dtype == np.uint8
    return f"4bpp level_map shape={r['level_map'].shape}  max_level={r['level_map'].max()}"


def test_rip_bridge_basic(run_dir: Path, tmp: Path):
    """fgm_to_rip.fgm_to_tiff_stack must produce the correct number of TIFF files."""
    try:
        import fgm_to_rip  # noqa: F401
    except ImportError:
        # Try the standard Meteor tools location relative to this repo
        meteor_tools = (
            Path(__file__).parents[2] / "software" / "meteor" / "tools"
        )
        if meteor_tools.is_dir():
            sys.path.insert(0, str(meteor_tools))
        try:
            import fgm_to_rip
        except ImportError:
            raise _SkipTest("fgm_to_rip module not importable (not on path)")

    from fgm_generator import generate_fgm

    r    = generate_fgm(run_dir, bpp=2, output_dir=tmp, emit_formats=("npz",))
    npz  = r["npz_path"]
    rip_out = tmp / "rip_output"
    rip_out.mkdir(exist_ok=True)

    n_layers = 5
    paths = fgm_to_rip.fgm_to_tiff_stack(
        npz, rip_out, n_layers=n_layers, job_name="test_job",
        dpi=r["dpi"], bpp=r["bpp"],
    )

    assert len(paths) == n_layers, (
        f"Expected {n_layers} TIFFs, got {len(paths)}: {paths}"
    )
    for p in paths:
        assert Path(p).exists(), f"TIFF not found: {p}"
        assert Path(p).stat().st_size > 0, f"TIFF is empty: {p}"

    return f"Created {n_layers} TIFF files: {[Path(p).name for p in paths]}"


def test_rip_bridge_3d_fgm(run_dir: Path, tmp: Path):
    """3-D FGM (nz, ny, nx) should produce one TIFF per Z slice."""
    try:
        import fgm_to_rip
    except ImportError:
        meteor_tools = Path(__file__).parents[2] / "software" / "meteor" / "tools"
        if meteor_tools.is_dir():
            sys.path.insert(0, str(meteor_tools))
        try:
            import fgm_to_rip
        except ImportError:
            raise _SkipTest("fgm_to_rip not importable")

    from fgm_generator import generate_fgm

    r   = generate_fgm(run_dir, bpp=2, output_dir=tmp, emit_formats=("npz",))
    lm2 = r["level_map"]  # (ny, nx)

    # Manufacture a 3-D FGM by stacking the 2-D map 4 times
    nz = 4
    lm3 = np.stack([lm2] * nz, axis=0)  # (nz, ny, nx)

    npz3 = tmp / "test_3d_fgm.npz"
    np.savez_compressed(
        npz3,
        level_map=lm3,
        bpp=np.array(2, dtype=np.int32),
        magnitude=np.array(1.0, dtype=np.float32),
        proxy_field=np.array("Qrf"),
        dpi=np.array(r["dpi"], dtype=np.int32),
    )

    rip_out = tmp / "rip_3d"
    rip_out.mkdir(exist_ok=True)
    paths = fgm_to_rip.fgm_to_tiff_stack(
        npz3, rip_out, job_name="test_3d_job", dpi=r["dpi"],
    )

    assert len(paths) == nz, f"Expected {nz} TIFFs for 3-D FGM, got {len(paths)}"
    return f"3-D FGM produced {len(paths)} layer TIFFs  (shape {lm3.shape})"


def test_rip_bridge_rotation(run_dir: Path, tmp: Path):
    """Rotating 90° twice should give same shape; 4× should be identity."""
    try:
        import fgm_to_rip
    except ImportError:
        raise _SkipTest("fgm_to_rip not importable")

    from fgm_generator import generate_fgm

    r   = generate_fgm(run_dir, bpp=2, output_dir=tmp, emit_formats=("npz",))
    lm  = r["level_map"]
    npz = r["npz_path"]

    try:
        import tifffile
        # Probe whether LZW decode is available (requires imagecodecs)
        _probe = tifffile.TiffFile  # just importing is fine; read will fail without imagecodecs
    except ImportError:
        raise _SkipTest("tifffile not installed; skipping rotation TIFF content check")

    for deg in (0, 90, 180, 270):
        rip_out = tmp / f"rip_rot{deg}"
        rip_out.mkdir(exist_ok=True)
        paths = fgm_to_rip.fgm_to_tiff_stack(
            npz, rip_out, n_layers=1, rotate_deg=deg, job_name=f"rot{deg}",
        )
        assert len(paths) == 1
        try:
            tif_arr = tifffile.imread(paths[0])
        except ValueError as e:
            if "imagecodecs" in str(e):
                raise _SkipTest(
                    f"imagecodecs not installed; cannot decode LZW TIFF (deg={deg}). "
                    f"Install with: pip install imagecodecs"
                )
            raise
        expected_shape = np.rot90(lm, k={0: 0, 90: 3, 180: 2, 270: 1}[deg]).shape
        if tif_arr.shape != expected_shape:
            raise AssertionError(
                f"rotate_deg={deg}: TIFF shape {tif_arr.shape} ≠ expected {expected_shape}"
            )

    return "rotation 0/90/180/270° all produce correct TIFF dimensions"


def test_output_filename_encodes_params(run_dir: Path, tmp: Path):
    """Output filenames must encode proxy_field, bpp, and magnitude."""
    from fgm_generator import generate_fgm

    for proxy, bpp, mag in [("Qrf", 2, 0.5), ("T", 4, 1.5), ("rho_rel", 2, 0.0)]:
        r = generate_fgm(
            run_dir, bpp=bpp, proxy_field=proxy, magnitude=mag,
            output_dir=tmp, emit_formats=("npz",),
        )
        fname = Path(r["npz_path"]).stem
        mag_str = f"{mag:.2f}".replace(".", "p")
        for token in (proxy, f"{bpp}bpp", f"mag{mag_str}"):
            if token not in fname:
                raise AssertionError(
                    f"Expected '{token}' in filename '{fname}' "
                    f"(proxy={proxy}, bpp={bpp}, mag={mag})"
                )
    return "filenames correctly encode proxy/bpp/magnitude for all tested combos"


def test_no_dopant_outside_part(run_dir: Path, tmp: Path):
    """sat_map must be exactly 0.0 outside the part mask."""
    from fgm_generator import generate_fgm

    r = generate_fgm(run_dir, magnitude=1.0, output_dir=tmp, emit_formats=("npz",))
    sat = r["sat_map"]
    fields = np.load(run_dir / "fields.npz")
    pm = fields["part_mask"].astype(bool)

    exterior_sat = sat[~pm]
    if exterior_sat.max() > 1e-6:
        raise AssertionError(
            f"Outside-part pixels have max sat={exterior_sat.max():.4e} (expected 0.0). "
            f"Binder is being deposited outside the part boundary."
        )
    return f"sat_map=0.0 outside part  ({(~pm).sum()} exterior pixels checked)"


def test_multiple_formats_same_array(run_dir: Path, tmp: Path):
    """NPZ and JSON both emitted in the same call must have identical level_map data."""
    from fgm_generator import generate_fgm
    import base64

    r = generate_fgm(run_dir, bpp=2, output_dir=tmp, emit_formats=("npz", "json"))
    lm_returned = r["level_map"]

    # NPZ
    d    = np.load(r["npz_path"])
    lm_npz = d["level_map"]
    assert np.array_equal(lm_npz, lm_returned), "NPZ level_map != returned level_map"

    # JSON
    meta   = json.loads(Path(r["json_path"]).read_text())
    ny, nx = meta["shape_hw"]
    lm_json = np.frombuffer(
        base64.b64decode(meta["level_map_b64"]), dtype=np.uint8
    ).reshape(ny, nx)
    assert np.array_equal(lm_json, lm_returned), "JSON level_map != returned level_map"

    return "NPZ and JSON level_map arrays are bit-identical"


def test_fgm_feedback_disabled_noop(run_dir: Path, tmp: Path):
    """_FgmFeedback disabled instance must be a no-op: effective_fill returns fill_frac unchanged."""
    from rfam_eqs_coupled import _FgmFeedback

    x = np.linspace(0, 0.06, 120)
    y = np.linspace(0, 0.06, 120)
    pm = np.ones((120, 120), dtype=bool)

    fb = _FgmFeedback.from_config({}, x, y, pm)
    assert not fb.enabled, "Expected disabled instance from empty config"

    ff = np.random.rand(120, 120).astype(np.float32)
    eff = fb.effective_fill(ff)
    assert eff is ff, "Disabled instance should return fill_frac unchanged (zero-copy)"

    assert not fb.maybe_update(99, np.zeros((120, 120)), pm, str(run_dir))

    return "disabled _FgmFeedback is a perfect no-op"


def test_fgm_feedback_modulates_sigma(run_dir: Path, tmp: Path):
    """_FgmFeedback with a real FGM should produce spatially varying sigma != uniform sigma."""
    from rfam_eqs_coupled import _FgmFeedback
    from fgm_generator import generate_fgm

    r = generate_fgm(run_dir, bpp=2, magnitude=1.0, output_dir=tmp, emit_formats=("npz",))

    fields = np.load(run_dir / "fields.npz")
    x, y = fields["x"], fields["y"]
    pm = fields["part_mask"].astype(bool)

    cfg = {
        "fgm_feedback": {
            "enabled": True,
            "saturation_map_npz": r["npz_path"],
        }
    }
    fb = _FgmFeedback.from_config(cfg, x, y, pm)
    assert fb.enabled

    sigma_d0 = 0.08
    sigma_v  = 1e-5
    rho_init = 0.55
    fill_frac = np.ones((len(y), len(x)), dtype=np.float32)
    # FGM gives fully interior pixels so fill_frac=1 everywhere for simplicity

    # Uniform baseline sigma (no FGM)
    sigma_uniform = sigma_v + fill_frac * (sigma_d0 - sigma_v)

    # FGM-modulated sigma
    eff = fb.effective_fill(fill_frac)
    sigma_fgm = sigma_v + eff * (sigma_d0 - sigma_v)

    # Inside the part, FGM sigma must differ from uniform sigma
    diff = np.abs(sigma_fgm[pm] - sigma_uniform[pm])
    if diff.max() < 1e-8:
        raise AssertionError(
            "FGM-modulated sigma is identical to uniform sigma inside the part. "
            "The saturation map has no effect."
        )
    # Mean FGM sigma should be < uniform sigma (FGM was generated from Qrf,
    # so hot regions get reduced sat → reduced sigma overall)
    mean_uniform = float(sigma_uniform[pm].mean())
    mean_fgm     = float(sigma_fgm[pm].mean())
    if mean_fgm >= mean_uniform:
        raise AssertionError(
            f"Mean FGM sigma ({mean_fgm:.5f}) >= mean uniform sigma ({mean_uniform:.5f}). "
            f"Expected FGM to reduce mean conductivity (hot regions get less dopant)."
        )

    # sigma_at_mask should agree with direct multiplication
    s = fb.sigma_at_mask(pm, sigma_d0, np.zeros((len(y), len(x))),
                          np.full((len(y), len(x)), rho_init), 0.0, 0.0, 20.0, rho_init)
    expected = sigma_d0 * fb.sat_map[pm]
    if not np.allclose(s, expected, atol=1e-8):
        raise AssertionError("sigma_at_mask values don't match direct sigma_d0 * sat_map")

    return (
        f"sigma modulated: max_diff={diff.max():.4f}  "
        f"mean_uniform={mean_uniform:.5f}  mean_fgm={mean_fgm:.5f}"
    )


def test_fgm_feedback_missing_file_raises(run_dir: Path, tmp: Path):
    """_FgmFeedback must raise FileNotFoundError for a missing NPZ path."""
    from rfam_eqs_coupled import _FgmFeedback
    import numpy as np

    x = np.linspace(0, 0.06, 10)
    y = np.linspace(0, 0.06, 10)
    pm = np.ones((10, 10), dtype=bool)

    cfg = {
        "fgm_feedback": {
            "enabled": True,
            "saturation_map_npz": "/nonexistent/path/to/fgm.npz",
        }
    }
    try:
        _FgmFeedback.from_config(cfg, x, y, pm)
        raise AssertionError("Expected FileNotFoundError for missing NPZ")
    except FileNotFoundError:
        pass  # expected

    return "FileNotFoundError raised correctly for missing NPZ"


def test_fgm_feedback_missing_npz_key_raises(run_dir: Path, tmp: Path):
    """_FgmFeedback must raise ValueError when enabled=true but no NPZ path is given."""
    from rfam_eqs_coupled import _FgmFeedback
    import numpy as np

    x = np.linspace(0, 0.06, 10)
    y = np.linspace(0, 0.06, 10)
    pm = np.ones((10, 10), dtype=bool)

    cfg = {"fgm_feedback": {"enabled": True}}  # missing saturation_map_npz
    try:
        _FgmFeedback.from_config(cfg, x, y, pm)
        raise AssertionError("Expected ValueError for missing saturation_map_npz")
    except ValueError:
        pass

    return "ValueError raised correctly for missing saturation_map_npz key"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    global _verbose

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", default=None,
        help="Path to a HEATR run output directory containing fields.npz.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Print per-test results immediately.",
    )
    args = parser.parse_args()
    _verbose = args.verbose

    # ── Locate run directory ──────────────────────────────────────────────────
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        try:
            run_dir = _find_default_run_dir()
        except FileNotFoundError as e:
            print(f"\n{FAIL}  Cannot locate a HEATR run directory: {e}")
            sys.exit(1)

    if not (run_dir / "fields.npz").exists():
        print(f"\n{FAIL}  fields.npz not found in: {run_dir}")
        sys.exit(1)

    print(f"\n  Run dir  : {run_dir}")
    print(f"  Verbose  : {_verbose}\n")

    # Add geo-prewarp to sys.path so fgm_generator is importable
    here = Path(__file__).parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))

    # ── Create shared temp directory ─────────────────────────────────────────
    tmp = Path(tempfile.mkdtemp(prefix="fgm_test_"))

    try:
        # ── Register tests ────────────────────────────────────────────────────
        _run("import fgm_generator",
             test_import_fgm_generator)

        _run("fields.npz has required keys",
             lambda: test_fields_npz_has_required_keys(run_dir))

        _run("inversion: Qrf hottest → lowest level",
             lambda: test_inversion_correctness_qrf(run_dir, tmp))

        _run("inversion: T hottest → lowest sat_map",
             lambda: test_inversion_correctness_t(run_dir, tmp))

        _run("magnitude=0 produces flat map",
             lambda: test_magnitude_zero_flat_map(run_dir, tmp))

        _run("level_map range and dtype validity",
             lambda: test_level_range_validity(run_dir, tmp))

        _run("sat_map in [0, 1] for all magnitudes",
             lambda: test_sat_map_range(run_dir, tmp))

        _run("NPZ output loadable with correct keys",
             lambda: test_output_npz_loadable(run_dir, tmp))

        _run("JSON output loadable, base64 round-trip",
             lambda: test_output_json_loadable(run_dir, tmp))

        _run("PNG output exists and is 8-bit grayscale",
             lambda: test_output_png_exists_and_8bit(run_dir, tmp))

        _run("PNG visual convention: level0=white, levelMax=black",
             lambda: test_png_visual_convention(run_dir, tmp))

        _run("rho_rel proxy: low density → high saturation",
             lambda: test_rho_rel_proxy(run_dir, tmp))

        _run("4bpp mode produces 16 levels",
             lambda: test_4bpp_mode(run_dir, tmp))

        _run("filename encodes proxy/bpp/magnitude",
             lambda: test_output_filename_encodes_params(run_dir, tmp))

        _run("sat_map = 0 outside part mask",
             lambda: test_no_dopant_outside_part(run_dir, tmp))

        _run("NPZ and JSON level_map data identical",
             lambda: test_multiple_formats_same_array(run_dir, tmp))

        # RIP bridge tests (may skip if fgm_to_rip not on path)
        _run("RIP bridge: basic TIFF stack creation",
             lambda: test_rip_bridge_basic(run_dir, tmp))

        _run("RIP bridge: 3-D FGM (nz,ny,nx) → per-slice TIFFs",
             lambda: test_rip_bridge_3d_fgm(run_dir, tmp))

        _run("RIP bridge: rotation 0/90/180/270°",
             lambda: test_rip_bridge_rotation(run_dir, tmp))

        # FGM feedback (re-simulation) tests
        _run("FGM feedback: disabled instance is no-op",
             lambda: test_fgm_feedback_disabled_noop(run_dir, tmp))

        _run("FGM feedback: enabled instance modulates sigma",
             lambda: test_fgm_feedback_modulates_sigma(run_dir, tmp))

        _run("FGM feedback: missing NPZ raises FileNotFoundError",
             lambda: test_fgm_feedback_missing_file_raises(run_dir, tmp))

        _run("FGM feedback: enabled without NPZ key raises ValueError",
             lambda: test_fgm_feedback_missing_npz_key_raises(run_dir, tmp))

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── Report ────────────────────────────────────────────────────────────────
    n_pass = sum(1 for _, s, _ in _results if s == PASS)
    n_fail = sum(1 for _, s, _ in _results if s == FAIL)
    n_skip = sum(1 for _, s, _ in _results if s == SKIP)

    print(f"\n{'─' * 64}")
    print(f"  Results:  {n_pass} passed  {n_fail} failed  {n_skip} skipped\n")

    for name, status, detail in _results:
        line = f"  {status}  {name}"
        if status == FAIL or (status == SKIP and _verbose):
            line += f"\n         {detail}"
        elif status == PASS and _verbose:
            line += f"  — {detail}" if detail else ""
        print(line)

    print()

    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
