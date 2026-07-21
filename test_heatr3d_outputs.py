#!/usr/bin/env python3
"""test_heatr3d_outputs.py — verify HEATR-3D job output writers (summary, fieldmeta, slices).
Self-contained (no pytest). Run with the job venv against a REAL run dir:
    .venv-heatr3d/bin/python test_heatr3d_outputs.py --run-dir outputs_eqs/_heatr3d/<id> -v
Exit 0 = all pass, 1 = any failure.
"""
from __future__ import annotations
import argparse, json, sys, tempfile, traceback
from pathlib import Path
import numpy as np
import heatr3d_job as J   # module under test (repo root)

def test_field_meta_only_reports_real_volumes():
    n = 4
    fields = {
        "sat": np.linspace(0.2, 0.8, n*n*n, dtype=np.float32).reshape(n, n, n),
        "rho_final": np.zeros((1,), np.float32),   # sentinel = absent
        "part": np.ones((n, n, n), dtype=bool),
    }
    meta = J._field_meta(fields, h=0.001875)
    assert meta["dims"] == [n, n, n], meta["dims"]
    assert "sat" in meta["fields"], "real volume must be reported"
    assert "rho_final" not in meta["fields"], "shape-(1,) sentinel must be treated as absent"
    assert "part" not in meta["fields"], "boolean mask is not a colorable field"
    fm = meta["fields"]["sat"]
    assert abs(fm["min"] - 0.2) < 1e-4 and abs(fm["max"] - 0.8) < 1e-4, fm
    assert fm["slices"] == n, "one slice per z index"
    assert abs(meta["h_mm"] - 1.875) < 1e-6, meta["h_mm"]

def test_write_summary_makes_run_collectible():
    results = {"sigma_T": 22.66, "T_max_C": 184.2, "dice": 0.94, "densify": True,
               "z_shrink_pct": 30.7, "fgm": "melt", "grid_n": 32}
    cfg = {"shape": "sphere", "n": 32, "fgm": "melt", "densify": True}
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        J._write_summary(out, results, cfg)
        sp = out / "summary.json"
        assert sp.exists(), "summary.json must be written (the Results-browser gate)"
        s = json.loads(sp.read_text())
        assert s.get("run_type") == "heatr3d", s.get("run_type")
        assert s.get("sigma_T") == 22.66 and s.get("dice") == 0.94, s
        assert s.get("shape") == "sphere", "config echoed for the run card"

def test_render_slices_writes_pngs_and_meta(run_dir: Path):
    z = np.load(run_dir / "fields.npz")
    fields = {k: z[k] for k in z.files}
    h = float(z["h"]) if z["h"].ndim == 0 else float(z["h"][0])
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        meta = J._field_meta(fields, h)
        J._render_slices(out, fields, meta)
        fm = json.loads((out / "fieldmeta.json").read_text())
        assert fm["fields"].keys() == meta["fields"].keys(), fm["fields"].keys()
        assert (out / "preview.png").exists(), "preview.png must exist"
        assert (out / "preview.png").stat().st_size > 0
        for name, info in meta["fields"].items():
            got = sorted((out / "slices").glob(f"{name}_z_*.png"))
            assert len(got) == info["slices"], f"{name}: {len(got)} != {info['slices']}"
            assert all(p.stat().st_size > 0 for p in got), f"{name}: empty PNG"

def _run(run_dir, verbose):
    plain = [
        ("field_meta_only_reports_real_volumes", test_field_meta_only_reports_real_volumes),
        ("write_summary_makes_run_collectible", test_write_summary_makes_run_collectible),
    ]
    needs_dir = [
        ("render_slices_writes_pngs_and_meta", test_render_slices_writes_pngs_and_meta),
    ]
    failures = 0
    for name, fn in plain:
        try: fn(); print(f"PASS {name}")
        except Exception:
            failures += 1; print(f"FAIL {name}")
            if verbose: traceback.print_exc()
    for name, fn in needs_dir:
        try: fn(run_dir); print(f"PASS {name}")
        except Exception:
            failures += 1; print(f"FAIL {name}")
            if verbose: traceback.print_exc()
    return failures

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs_eqs/_heatr3d")
    ap.add_argument("-v", dest="v", action="store_true")
    a = ap.parse_args()
    sys.exit(1 if _run(Path(a.run_dir), a.v) else 0)
