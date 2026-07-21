# HEATR-3D Phase 1 (Results visibility + per-layer slices) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make finished HEATR-3D runs appear in the Results browser (F1), and make each run's interior fields available to the UI as pre-rendered per-layer slice images plus metadata (F2).

**Architecture:** All heavy lifting happens in the job process (`heatr3d_job.py`), which already runs under a venv with numpy 2.1.3 **and matplotlib 3.10.9**. At run completion the job now also writes `summary.json` (so the run is picked up by `_collect_results`, which accepts any dir with a `summary.json` or a PNG — `rfam_gui_server.py:4189-4192`), a `preview.png`, per-layer slice PNGs, and a `fieldmeta.json`. The GUI server (system Python 3.14, buggy numpy) does **no** array work — it only serves these static files through two tiny new file-read routes. This keeps numpy entirely out of the server.

**Tech Stack:** Python 3.12 (job venv), numpy, matplotlib (Agg backend), stdlib `http.server` (GUI server). Tests follow the repo convention: self-contained `def test_*` scripts with a `__main__` runner (no pytest), run with the job venv against a real run directory. See `test_fgm_pipeline.py` for the pattern.

---

## Conventions (read once)

- **Job venv python** (used to run tests and jobs):
  `.venv-heatr3d/bin/python` (numpy 2.1.3 + matplotlib 3.10.9 confirmed present).
- **A real run directory** for tests (created during F0, keep it):
  `outputs_eqs/_heatr3d/1e10228aff1d/` — a densify+FGM sphere run whose `fields.npz` has
  `part,T_phi90,phi_final,Qrf,rho_final,sat,h`, all `(32,32,32)`, with `sat`/`rho_final` populated.
  If it has been cleaned up, regenerate one first:
  ```bash
  curl -s -X POST http://127.0.0.1:8796/api/heatr3d/run -H 'Content-Type: application/json' \
    -d '{"src":"parametric","shape":"sphere","diam":0.028,"zspan":0.030,"n":32,"fgm":"melt","magnitude":0.6,"densify":true,"exposure_s":600,"stop_mean_rho":0.85}'
  ```
  (`diam`/`zspan` are METERS — the frontend divides mm by 1000. Do not pass raw mm.)
- **fields.npz sentinel rule:** when a field is not computed, the job writes `np.zeros((1,), np.float32)`
  (a shape-`(1,)` sentinel), NOT a real zero volume. This is true for `rho_final` on non-densify runs
  and `sat` on non-FGM runs (`heatr3d_job.py:135-137`). All slice/meta code MUST treat any array whose
  shape != the grid `(n,n,n)` as ABSENT, never as data.
- **Run the whole test script**, not one function, per the existing harness:
  `.venv-heatr3d/bin/python test_heatr3d_outputs.py --run-dir outputs_eqs/_heatr3d/1e10228aff1d -v`

---

## File Structure

- **Modify** `heatr3d_job.py` — add three writer helpers and call them at the end of `main()`
  (after `results.json` is written, `heatr3d_job.py:140`):
  - `_write_summary(out, results, cfg)` → `summary.json`
  - `_field_meta(fields, h)` (pure) → dict of `{dims, h_mm, fields:{name:{min,max,slices}}}`
  - `_render_slices(out, fields, meta)` → `slices/<field>_z_<k>.png` + `preview.png`, writes `fieldmeta.json`
- **Modify** `rfam_gui_server.py` — add two static-file GET routes next to the existing
  `/api/heatr3d/*` handlers (router at `rfam_gui_server.py:5417`, helpers near `_h3d_status` ~5329):
  - `GET /api/heatr3d/fields?id=<jid>` → serve `fieldmeta.json`
  - `GET /api/heatr3d/slice?id=<jid>&field=<f>&k=<k>` → serve `slices/<f>_z_<k>.png`
- **Create** `test_heatr3d_outputs.py` (repo root, beside `test_fgm_pipeline.py`) — self-contained
  tests for the summary/meta/slice writers, run against a real run dir.

No frontend changes in Phase 1 (the slice viewer UI is F3, Phase 2). Phase 1 makes the data reachable
and the runs visible; it is verifiable entirely via files + HTTP.

---

## Task 1: `fieldmeta` — the pure slice-metadata function (F2 core, TDD)

`_field_meta` is the one piece of real logic; everything else is I/O around it. Build and test it first.

**Files:**
- Create: `test_heatr3d_outputs.py`
- Modify: `heatr3d_job.py` (add `_field_meta`)

- [ ] **Step 1: Write the failing test**

Create `test_heatr3d_outputs.py`:

```python
#!/usr/bin/env python3
"""test_heatr3d_outputs.py — verify HEATR-3D job output writers (summary, fieldmeta, slices).
Self-contained (no pytest). Run with the job venv against a REAL run dir:
    .venv-heatr3d/bin/python test_heatr3d_outputs.py --run-dir outputs_eqs/_heatr3d/<id> -v
Exit 0 = all pass, 1 = any failure.
"""
from __future__ import annotations
import argparse, sys, traceback
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
```

Add this runner block at the bottom (mirrors `test_fgm_pipeline.py`):

```python
def _run(verbose):
    tests = [("field_meta_only_reports_real_volumes", test_field_meta_only_reports_real_volumes)]
    # (later tasks append their (name, fn) or (name, fn_needing_run_dir) entries here)
    failures = 0
    for name, fn in tests:
        try:
            fn(); print(f"PASS {name}")
        except Exception:
            failures += 1; print(f"FAIL {name}")
            if verbose: traceback.print_exc()
    return failures

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs_eqs/_heatr3d")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    sys.exit(1 if _run(a.v if hasattr(a, "v") else a.__dict__.get("v", False)) else 0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-heatr3d/bin/python test_heatr3d_outputs.py -v`
Expected: FAIL `field_meta_only_reports_real_volumes` with `AttributeError: module 'heatr3d_job' has no attribute '_field_meta'`.

- [ ] **Step 3: Write the minimal implementation**

In `heatr3d_job.py`, add above `main()` (after the imports / existing helpers):

```python
# Fields worth coloring as slices (name -> human label). `part` is the mask, not a field.
_SLICE_FIELDS = {
    "sat": "FGM dopant fraction",
    "T_phi90": "Temperature at phi=0.90 (C)",
    "phi_final": "Melt fraction",
    "rho_final": "Relative density",
    "Qrf": "Absorbed RF power (W/m^3)",
}

def _field_meta(fields: dict, h: float) -> dict:
    """Metadata for the per-layer slice viewer. Only real (n,n,n) volumes are reported;
    shape-(1,) sentinels (absent fields) and the boolean `part` mask are skipped."""
    part = fields.get("part")
    dims = list(part.shape) if getattr(part, "ndim", 0) == 3 else None
    out_fields = {}
    for name in _SLICE_FIELDS:
        a = fields.get(name)
        if a is None or getattr(a, "ndim", 0) != 3:
            continue
        if dims is None:
            dims = list(a.shape)
        if list(a.shape) != dims:
            continue
        out_fields[name] = {
            "label": _SLICE_FIELDS[name],
            "min": float(a.min()),
            "max": float(a.max()),
            "slices": int(a.shape[2]),  # z-axis
        }
    return {"dims": dims or [0, 0, 0], "h_mm": float(h) * 1000.0, "axis": "z", "fields": out_fields}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv-heatr3d/bin/python test_heatr3d_outputs.py -v`
Expected: PASS `field_meta_only_reports_real_volumes`.

- [ ] **Step 5: Commit**

```bash
git add test_heatr3d_outputs.py heatr3d_job.py
git commit -m "feat(heatr3d): field-slice metadata helper (F2 core)"
```

---

## Task 2: `summary.json` writer (F1 — makes runs visible, TDD)

**Files:**
- Modify: `heatr3d_job.py` (add `_write_summary`, call it in `main()`)
- Modify: `test_heatr3d_outputs.py` (add a test)

- [ ] **Step 1: Write the failing test**

Add to `test_heatr3d_outputs.py`:

```python
import json, tempfile
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
```

Register it in `_run`'s `tests` list: `("write_summary_makes_run_collectible", test_write_summary_makes_run_collectible),`.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-heatr3d/bin/python test_heatr3d_outputs.py -v`
Expected: FAIL `write_summary_makes_run_collectible` with `AttributeError: ... has no attribute '_write_summary'`.

- [ ] **Step 3: Write the minimal implementation**

In `heatr3d_job.py`, add above `main()`:

```python
def _write_summary(out: Path, results: dict, cfg: dict) -> None:
    """Write summary.json so the run is picked up by the Results browser
    (rfam_gui_server _collect_results accepts any dir with summary.json). Carries a
    `run_type: heatr3d` tag plus the scalar metrics and the input config for the run card."""
    summary = {"run_type": "heatr3d"}
    summary.update(results)
    summary["config"] = {k: cfg.get(k) for k in ("shape", "n", "fgm", "magnitude", "densify",
                                                  "exposure_s", "stop_mean_rho", "diam", "zspan")
                         if cfg.get(k) is not None}
    # Also promote shape to top level so it shows on the run card without digging into config.
    if "shape" in cfg:
        summary["shape"] = cfg["shape"]
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv-heatr3d/bin/python test_heatr3d_outputs.py -v`
Expected: PASS both tests.

- [ ] **Step 5: Commit**

```bash
git add heatr3d_job.py test_heatr3d_outputs.py
git commit -m "feat(heatr3d): write summary.json so runs appear in Results browser (F1)"
```

---

## Task 3: slice + preview PNG writer (F1 preview + F2 slices, TDD against real data)

**Files:**
- Modify: `heatr3d_job.py` (add `_render_slices`)
- Modify: `test_heatr3d_outputs.py` (add a real-data test)

- [ ] **Step 1: Write the failing test**

Add to `test_heatr3d_outputs.py` (this one needs the real run dir, so it takes `run_dir`):

```python
def test_render_slices_writes_pngs_and_meta(run_dir: Path):
    z = np.load(run_dir / "fields.npz")
    fields = {k: z[k] for k in z.files}
    h = float(z["h"]) if z["h"].ndim == 0 else float(z["h"][0])
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        meta = J._field_meta(fields, h)
        J._render_slices(out, fields, meta)
        # fieldmeta.json persisted and matches the computed meta
        fm = json.loads((out / "fieldmeta.json").read_text())
        assert fm["fields"].keys() == meta["fields"].keys(), fm["fields"].keys()
        # a preview image exists (satisfies the Results-browser media gate on its own)
        assert (out / "preview.png").exists(), "preview.png must exist"
        assert (out / "preview.png").stat().st_size > 0
        # one PNG per z-slice for each real field
        for name, info in meta["fields"].items():
            got = sorted((out / "slices").glob(f"{name}_z_*.png"))
            assert len(got) == info["slices"], f"{name}: {len(got)} != {info['slices']}"
            assert all(p.stat().st_size > 0 for p in got), f"{name}: empty PNG"
```

Register it with a run-dir-needing marker. Update `_run` to pass `run_dir`:

```python
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
```

And update `__main__` to `sys.exit(1 if _run(Path(a.run_dir), a.v) else 0)` (rename the arg to `-v`/`a.v`
cleanly: `ap.add_argument("-v", dest="v", action="store_true")`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv-heatr3d/bin/python test_heatr3d_outputs.py --run-dir outputs_eqs/_heatr3d/1e10228aff1d -v`
Expected: FAIL `render_slices_writes_pngs_and_meta` with `AttributeError: ... '_render_slices'`.

- [ ] **Step 3: Write the minimal implementation**

In `heatr3d_job.py`, near the top add the matplotlib import guarded to the Agg backend (headless):

```python
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
```

Then add above `main()`:

```python
def _render_slices(out: Path, fields: dict, meta: dict) -> None:
    """Pre-render one PNG per z-slice for each real field (viridis, per-field global min/max so
    the colormap is stable across layers), plus a preview.png, plus fieldmeta.json. Done in-job
    because the GUI server's numpy is unreliable; the server only serves these static files."""
    (out / "fieldmeta.json").write_text(json.dumps(meta, indent=2))
    sl = out / "slices"; sl.mkdir(parents=True, exist_ok=True)
    part = fields.get("part")
    mask3d = part if getattr(part, "ndim", 0) == 3 else None
    preview_written = False
    for name, info in meta["fields"].items():
        a = fields[name]
        vmin, vmax = info["min"], info["max"]
        if vmax <= vmin:
            vmax = vmin + 1e-9
        nz = a.shape[2]
        for k in range(nz):
            img = a[:, :, k].astype(float)
            if mask3d is not None:
                img = np.where(mask3d[:, :, k], img, np.nan)  # outside-part = transparent, not 0
            fig = plt.figure(figsize=(2.2, 2.2), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.imshow(img.T, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
            fig.savefig(sl / f"{name}_z_{k:03d}.png", transparent=True)
            plt.close(fig)
        if not preview_written:
            _save_preview(out / "preview.png", a, mask3d, vmin, vmax, name)
            preview_written = True
    if not preview_written:  # no real fields (e.g. no-FGM, no-densify run): preview the geometry mask
        _save_preview(out / "preview.png", (mask3d.astype(float) if mask3d is not None
                      else np.zeros((1, 1, 1))), mask3d, 0.0, 1.0, "part")

def _save_preview(path: Path, a, mask3d, vmin, vmax, name):
    kmid = a.shape[2] // 2
    img = a[:, :, kmid].astype(float)
    if mask3d is not None and mask3d.shape == a.shape:
        img = np.where(mask3d[:, :, kmid], img, np.nan)
    fig = plt.figure(figsize=(3, 3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(img.T, origin="lower", cmap="viridis", vmin=vmin, vmax=(vmax if vmax > vmin else vmin + 1e-9),
              interpolation="nearest")
    fig.savefig(path)
    plt.close(fig)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv-heatr3d/bin/python test_heatr3d_outputs.py --run-dir outputs_eqs/_heatr3d/1e10228aff1d -v`
Expected: PASS all three tests. (`sat`, `T_phi90`, `phi_final`, `rho_final`, `Qrf` each get 32 slice PNGs; `preview.png` exists.)

- [ ] **Step 5: Commit**

```bash
git add heatr3d_job.py test_heatr3d_outputs.py
git commit -m "feat(heatr3d): pre-render per-layer slice PNGs + preview + fieldmeta (F1 preview, F2 slices)"
```

---

## Task 4: wire the writers into `main()` (F1+F2 integration, real run)

**Files:**
- Modify: `heatr3d_job.py:140` region (after `results.json` write)

- [ ] **Step 1: Add the calls**

In `heatr3d_job.py` `main()`, replace the tail (from the `results.json` write at line 140 through the
final prints) with:

```python
    (out / "results.json").write_text(json.dumps(results, indent=2))

    fields_for_view = {"part": part, "T_phi90": r.T_phi90, "phi_final": r.phi_final, "Qrf": r.Qrf,
                       "rho_final": (r.rho_final if r.rho_final is not None else np.zeros((1,), np.float32)),
                       "sat": (sat if sat is not None else np.zeros((1,), np.float32))}
    meta = _field_meta(fields_for_view, grid.h)
    _write_summary(out, results, cfg)
    _render_slices(out, fields_for_view, meta)

    print("PROGRESS 100")
    print("RESULTS " + json.dumps(results))
```

- [ ] **Step 2: Run a real job end-to-end**

```bash
curl -s -X POST http://127.0.0.1:8796/api/heatr3d/run -H 'Content-Type: application/json' \
  -d '{"src":"parametric","shape":"sphere","diam":0.028,"zspan":0.030,"n":32,"fgm":"melt","magnitude":0.6,"densify":true,"exposure_s":600,"stop_mean_rho":0.85}'
```
Poll `GET /api/heatr3d/status?id=<id>` until `done:true`. Then:
```bash
ls outputs_eqs/_heatr3d/<id>/            # expect: summary.json preview.png fieldmeta.json slices/ ...
ls outputs_eqs/_heatr3d/<id>/slices | head
```
Expected: `summary.json`, `preview.png`, `fieldmeta.json`, and `slices/` with `sat_z_000.png …`.

- [ ] **Step 3: Verify the run now appears in the Results browser**

```bash
curl -s http://127.0.0.1:8796/api/results | python3 -c "import sys,json; d=json.load(sys.stdin); print([r['name'] for r in (d if isinstance(d,list) else d.get('results',[])) if '_heatr3d' in r['name']][:3])"
```
Expected: the new `_heatr3d/<id>` run is listed (was empty before this task).

- [ ] **Step 4: Commit**

```bash
git add heatr3d_job.py
git commit -m "feat(heatr3d): emit summary/preview/slices on every run; runs now appear in Results (F1)"
```

---

## Task 5: server routes to serve fieldmeta + slice PNGs (F2 delivery, smoke test)

The server does only file reads here — no numpy.

**Files:**
- Modify: `rfam_gui_server.py` — a helper near `_h3d_status` (~5329) and two router lines (~5417)

- [ ] **Step 1: Add a path-safe run-dir resolver (module-level function, no `self`)**

Near the other `_h3d_*` helpers in `rfam_gui_server.py` (module scope, like `_h3d_write_config`), add:

```python
def _h3d_run_dir(jid: str) -> "Path | None":
    """Resolve a HEATR-3D run dir from a job id, rejecting path traversal."""
    if not jid or "/" in jid or ".." in jid:
        return None
    d = (_H3D_OUT / jid).resolve()
    try:
        d.relative_to(_H3D_OUT.resolve())
    except ValueError:
        return None
    return d if d.is_dir() else None
```

(The routes in Step 2 do the serving directly in the request handler, where `self._serve_file` /
`self._text` are in scope. `_serve_file` already sets `application/json` for `.json` and `image/png`
for `.png` — `rfam_gui_server.py:6790-6825`; `parse_qs`/`urlparse` are already imported at the top of the
file, and `_H3D_OUT` is defined at `rfam_gui_server.py:5275`.)

- [ ] **Step 2: Add the routes** in the GET router (beside `if path == "/api/heatr3d/status":`, ~5417):

```python
        if path == "/api/heatr3d/fields":
            jid = parse_qs(urlparse(self.path).query).get("id", [""])[0]
            d = _h3d_run_dir(jid)
            fm = (d / "fieldmeta.json") if d else None
            if fm and fm.exists():
                return self._serve_file(fm)
            return self._text("not found", status=404)
        if path == "/api/heatr3d/slice":
            q = parse_qs(urlparse(self.path).query)
            jid = q.get("id", [""])[0]; field = q.get("field", [""])[0]; k = q.get("k", [""])[0]
            d = _h3d_run_dir(jid)
            if d and field.isidentifier() and k.isdigit():
                png = d / "slices" / f"{field}_z_{int(k):03d}.png"
                if png.exists():
                    return self._serve_file(png)
            return self._text("not found", status=404)
```

- [ ] **Step 3: Restart the server and smoke-test**

Restart the HEATR server (kill + relaunch on 8796). Using the `<id>` from Task 4:
```bash
curl -s -o /dev/null -w "fields: %{http_code} %{content_type}\n" "http://127.0.0.1:8796/api/heatr3d/fields?id=<id>"
curl -s -o /dev/null -w "slice:  %{http_code} %{content_type}\n" "http://127.0.0.1:8796/api/heatr3d/slice?id=<id>&field=sat&k=16"
curl -s -o /dev/null -w "guard:  %{http_code}\n" "http://127.0.0.1:8796/api/heatr3d/slice?id=../../etc&field=sat&k=0"
```
Expected: `fields: 200 application/json`, `slice: 200 image/png`, `guard: 404` (path-traversal rejected).

- [ ] **Step 4: Commit**

```bash
git add rfam_gui_server.py
git commit -m "feat(heatr3d): serve fieldmeta + per-layer slice PNGs via /api/heatr3d/fields|slice (F2)"
```

---

## Task 6: end-to-end verification against real data (F1+F2 gate)

- [ ] **Step 1: Full unit suite green**

Run: `.venv-heatr3d/bin/python test_heatr3d_outputs.py --run-dir outputs_eqs/_heatr3d/<id> -v`
Expected: all PASS.

- [ ] **Step 2: Fieldmeta content is real (not invented)**

```bash
curl -s "http://127.0.0.1:8796/api/heatr3d/fields?id=<id>" | python3 -m json.tool | head -30
```
Expected: `dims:[32,32,32]`, `h_mm≈1.875`, `fields` includes `sat`,`T_phi90`,`phi_final`,`rho_final`,`Qrf`
each with real `min`/`max`/`slices:32`. On a **non-densify, non-FGM** run, `rho_final` and `sat` are ABSENT
(sentinel rule) — verify by running one and confirming they do not appear.

- [ ] **Step 3: Slice matches the raw array (contract check)**

```bash
.venv-heatr3d/bin/python - <<'PY'
import numpy as np
z = np.load("outputs_eqs/_heatr3d/<id>/fields.npz")
print("sat[:,:,16] range:", float(z["sat"][:,:,16].min()), float(z["sat"][:,:,16].max()))
PY
```
Confirm the served `sat_z_016.png` is a viridis image of that same slice (open it / spot-check visually).

- [ ] **Step 4: Results browser shows the run with its preview**

Open `http://127.0.0.1:8796/results` in the browser tool, confirm the `_heatr3d/<id>` run is listed and
its detail view shows `preview.png` and the summary metrics.

- [ ] **Step 5: Final commit (if any doc/notes)** — otherwise Phase 1 is complete.

---

## Self-review notes (author)

- **Spec coverage:** F1 = Tasks 2,4 (summary) + Task 3 (preview PNG media) + Task 6 (browser visibility).
  F2 = Task 1 (meta), Task 3 (slice PNGs), Task 5 (serve), Task 6 (contract check). F0 already passed.
- **Contract fidelity:** slice/meta code keys off the real `fields.npz` shapes and the `(1,)` sentinel
  rule (`heatr3d_job.py:135-137`); `summary.json` only needs to *exist* to pass the Results gate
  (`rfam_gui_server.py:4189-4192`), verified against `_collect_results`/`_result_detail`.
- **Server stays numpy-free:** all array work is in-job (venv has matplotlib); the server does file reads
  only — deliberate, given system Python 3.14's numpy bug.
- **Naming consistency:** `_field_meta`, `_write_summary`, `_render_slices`, `_save_preview`, `_h3d_run_dir`
  used identically across tasks; slice filename format `"{field}_z_{k:03d}.png"` is identical in the
  writer (Task 3) and the server route (Task 5).
- **Deferred to Phase 2/3 (not this plan):** the slice-viewer UI (F3), X/Y-axis slicing (only Z here),
  warped geometry (F4/F5), run history/compare (F6), robustness incl. the diam/zspan guard (F7).
