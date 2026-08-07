# Premix Baseline Dopant (2-D lane) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a continuous premix baseline dopant (`premix_frac` 0→1, `premix_budget` floor_added/budget_fixed) to the 2-D coupled model, mirroring the tested `heatr3d` semantics, with `premix_frac=0` bit-identical to current behavior, plus a 0→15 wt% study.

**Architecture:** A single pure helper `apply_premix()` computes `(sigma, eps_r)` from the blend fields + material endpoints + premix params. It replaces the two inline blend formulas in `rfam_eqs_coupled.py`. Premix is a *swept forward parameter*, not a design variable, so the existing adjoint (which differentiates only w.r.t. the printed map) is unchanged — no new gradient path. wt% is a display-only label via `premix_frac_from_wtpct()`.

**Tech Stack:** Python 3.12 (`.venv312`), NumPy, pytest. Spec: `docs/superpowers/specs/2026-08-07-premix-baseline-dopant-2d-design.md`.

---

## EXECUTION NOTE (2026-08-07) — scope discovery during Task 4/5

Wiring premix into `run_sim` was NOT a single material-formula swap. The coupled forward
enforces generator power and zeros the bed at **6 sites**, and its time-loop re-solve blocks
reset `sigma[:,:] = sigma_v` (wiping the premix bed). Resolution actually shipped:

- **Tasks 1–3 (premix.py):** done, 5/5 green. Unchanged from plan.
- **Task 4 (site 2 = `run_sim`):** done. Premix-aware at the **initial** and **final** re-solve
  blocks — the two that set the returned `Qrf` — via (i) `apply_premix` material, (ii) whole-domain
  power enforcement when `premix_on`, (iii) skip bed-zeroing when `premix_on`. Guards: bed-absorbs,
  total-conserves, multi-step-persist. Commit `68878e5`.
- **Task 5 (site 1 = antenna probe):** DEFERRED — jared uses no antennae; wiring an unexercised
  path would violate TDD. Revisit when an antennae config needs premix.
- **Loop re-solve blocks** (turntable / `update_interval` tick / FGM-iterate): NOT premix-aware
  (they reset `sigma=virgin`). **Follow-up**, out of scope for the jared study.
- **Task 7 (study config):** MUST set `update_interval: 0` so no in-run re-solve fires (valid here:
  jared has `sigma_temp_coeff=0`, `sigma_density_coeff=0`, so sigma is static). Otherwise premix is
  wiped mid-run.
- **Task 6 (fgm_generator):** re-scope pending — check whether the generator computes any total-dopant
  budget; if not, `budget_fixed` accounting is fully handled inside `apply_premix` (Task 6 → docstring note).

---

## File Structure

- **Create** `premix.py` — pure, dependency-light module: `apply_premix()`, `premix_frac_from_wtpct()`, `PREMIX_WTPCT_FULL`. One responsibility: the premix material law. Small (<80 lines). Kept separate so it is trivially unit-testable and importable by both the 2-D model and the study script.
- **Modify** `rfam_eqs_coupled.py` — call `apply_premix()` at the two assembly sites (`:1451-1452`, `:2801-2805`); read premix params from config.
- **Modify** `fgm_generator.py` — premix-aware total-dopant accounting for `budget_fixed` only (no-op for `floor_added`).
- **Create** `test_premix.py` — unit TDD ladder for `premix.py`.
- **Create** `test_premix_integration.py` — small-grid forward bit-identical + premix-on wiring guard.
- **Create** `study_premix_sweep.py` — Component D (run gated behind load-check + Matt's go).
- **Create** `configs/jared_exp1_40mm_premix.yaml` — jared config with a premix block (used by the study).

**Reference (bit-identical target — the current inline formulas):**
- `rfam_eqs_coupled.py:1451` `sigma = sigma_v + fill_frac * (sigma_d0 - sigma_v)`
- `rfam_eqs_coupled.py:1452` `eps_r  = eps_v  + fill_frac * (eps_d  - eps_v)`
- `rfam_eqs_coupled.py:2801` `sigma = sigma_v + _eff_fill * (sigma_d0 - sigma_v)`
- `rfam_eqs_coupled.py:2805` `eps_r  = eps_v  + _eff_fill_eps * (eps_d  - eps_v)`

**Semantics (mirrors `heatr3d.py:591-624`):**
- `sigma_premix = sigma_v + f*(sigma_d0 - sigma_v)`, `eps_premix = eps_v + f*(eps_d - eps_v)`.
- `floor_added`: span kept at `(sigma_d0 - sigma_v)`; result = current formula + uniform `f*(sigma_d0 - sigma_v)`.
- `budget_fixed`: span becomes `(sigma_d0 - sigma_premix)`; result = `sigma_premix + blend*(sigma_d0 - sigma_premix)`.
- `f == 0.0`: BOTH reduce to the reference inline formula, returned with identical float ops → bit-identical.

**Run tests with:** `python -m pytest <file> -v` (from repo root; `.venv312` active).

---

### Task 1: Pure premix helper — identity path (bit-identical off)

**Files:**
- Create: `premix.py`
- Test: `test_premix.py`

- [ ] **Step 1: Write the failing test**

```python
# test_premix.py
import numpy as np
from premix import apply_premix

# Material endpoints matching the jared config family (effective composite).
V = dict(sigma_v=1e-8, sigma_d0=0.04, eps_v=2.7, eps_d=20.0)

def test_premix_off_is_bit_identical_to_inline_formula():
    rng = np.random.default_rng(0)
    blend = rng.random((16, 16))            # fill_frac / _eff_fill in [0,1]
    ref_sigma = V["sigma_v"] + blend * (V["sigma_d0"] - V["sigma_v"])
    ref_eps   = V["eps_v"]   + blend * (V["eps_d"]   - V["eps_v"])
    sigma, eps_r = apply_premix(blend, blend, premix_frac=0.0, **V)
    # Bit-for-bit: identical float operations, identical order.
    assert np.array_equal(sigma, ref_sigma)
    assert np.array_equal(eps_r, ref_eps)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_premix.py::test_premix_off_is_bit_identical_to_inline_formula -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'premix'`.

- [ ] **Step 3: Write minimal implementation**

```python
# premix.py
"""Continuous premix baseline dopant material law for the 2-D RFAM model.

Mirrors heatr3d.build_gamma premix semantics (heatr3d.py:591-624). premix_frac=0.0
returns the reference inline blend BIT-FOR-BIT. premix is a swept forward parameter,
not a design variable (no new gradient path).
"""
from __future__ import annotations
import numpy as np

PREMIX_WTPCT_FULL = 25.0  # nominal label for the doped endpoint (rfam_eqs_coupled.py:333);
                          # NOT a measured value — see spec sec.1 / THEORY_REFERENCES.

def apply_premix(blend_sigma, blend_eps, *, sigma_v, sigma_d0, eps_v, eps_d,
                 premix_frac=0.0, premix_budget="floor_added"):
    """Return (sigma, eps_r) fields from blend fractions + endpoints + premix params.

    blend_sigma / blend_eps: array-like fill/saturation blend (part≈1, bed=0; may exceed 1).
    premix_frac=0.0 -> reference inline blend, bit-for-bit.
    """
    f = float(premix_frac)
    if f < 0.0:
        raise ValueError("premix_frac must be >= 0")
    bs = np.asarray(blend_sigma, dtype=float)
    be = np.asarray(blend_eps, dtype=float)
    if f == 0.0:
        # identity path: EXACT reference formula, same op order -> bit-identical
        sigma = sigma_v + bs * (sigma_d0 - sigma_v)
        eps_r = eps_v + be * (eps_d - eps_v)
        return sigma, eps_r
    raise NotImplementedError("premix_frac > 0 implemented in Task 2")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_premix.py::test_premix_off_is_bit_identical_to_inline_formula -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add premix.py test_premix.py
git commit -m "feat(premix): pure helper identity path (premix_frac=0 bit-identical)"
```

---

### Task 2: Premix ON — floor_added and budget_fixed

**Files:**
- Modify: `premix.py`
- Test: `test_premix.py`

- [ ] **Step 1: Write the failing tests**

```python
# test_premix.py  (append)
def test_floor_added_bed_and_part_values():
    # bed (blend=0) rises to sigma_premix; part (blend=1) = sigma_premix + full span.
    blend = np.array([[0.0, 1.0]])
    f = 0.5
    sigma, eps_r = apply_premix(blend, blend, premix_frac=f,
                                premix_budget="floor_added", **V)
    sigma_premix = V["sigma_v"] + f * (V["sigma_d0"] - V["sigma_v"])
    span = V["sigma_d0"] - V["sigma_v"]
    assert np.isclose(sigma[0, 0], sigma_premix)            # bed
    assert np.isclose(sigma[0, 1], sigma_premix + 1.0 * span)  # part boosted above doped
    eps_premix = V["eps_v"] + f * (V["eps_d"] - V["eps_v"])
    assert np.isclose(eps_r[0, 0], eps_premix)

def test_budget_fixed_part_pinned_to_doped():
    # budget_fixed: part (blend=1) pinned at sigma_d0; bed at sigma_premix.
    blend = np.array([[0.0, 1.0]])
    f = 0.5
    sigma, _ = apply_premix(blend, blend, premix_frac=f,
                            premix_budget="budget_fixed", **V)
    sigma_premix = V["sigma_v"] + f * (V["sigma_d0"] - V["sigma_v"])
    assert np.isclose(sigma[0, 0], sigma_premix)     # bed
    assert np.isclose(sigma[0, 1], V["sigma_d0"])    # part exactly doped (total ~const)

def test_unknown_budget_raises():
    import pytest
    with pytest.raises(ValueError):
        apply_premix(np.zeros((2, 2)), np.zeros((2, 2)),
                     premix_frac=0.5, premix_budget="bogus", **V)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest test_premix.py -v`
Expected: the three new tests FAIL with `NotImplementedError` (and the ValueError test fails to raise ValueError).

- [ ] **Step 3: Replace the NotImplementedError branch**

```python
    # premix.py — replace the `raise NotImplementedError(...)` line with:
    sigma_premix = sigma_v + f * (sigma_d0 - sigma_v)
    eps_premix = eps_v + f * (eps_d - eps_v)
    if premix_budget == "floor_added":
        s_span = sigma_d0 - sigma_v
        e_span = eps_d - eps_v
    elif premix_budget == "budget_fixed":
        s_span = sigma_d0 - sigma_premix
        e_span = eps_d - eps_premix
    else:
        raise ValueError(f"unknown premix_budget {premix_budget!r}")
    sigma = sigma_premix + bs * s_span
    eps_r = eps_premix + be * e_span
    return sigma, eps_r
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest test_premix.py -v`
Expected: all PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add premix.py test_premix.py
git commit -m "feat(premix): floor_added + budget_fixed material law"
```

---

### Task 3: wt% display bridge

**Files:**
- Modify: `premix.py`
- Test: `test_premix.py`

- [ ] **Step 1: Write the failing test**

```python
# test_premix.py  (append)
from premix import premix_frac_from_wtpct, PREMIX_WTPCT_FULL

def test_wtpct_bridge_is_linear_over_doped_label():
    assert premix_frac_from_wtpct(0.0) == 0.0
    assert np.isclose(premix_frac_from_wtpct(PREMIX_WTPCT_FULL), 1.0)
    assert np.isclose(premix_frac_from_wtpct(15.0), 15.0 / PREMIX_WTPCT_FULL)  # 0.6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_premix.py::test_wtpct_bridge_is_linear_over_doped_label -v`
Expected: FAIL with `ImportError: cannot import name 'premix_frac_from_wtpct'`.

- [ ] **Step 3: Add the function**

```python
# premix.py  (append)
def premix_frac_from_wtpct(wt_pct):
    """[ASSUMED linear, no percolation] map wt% dopant-to-nylon -> premix_frac.

    DISPLAY ONLY. The measured relationship (RFAM Paper v1.3 Fig 8a) shows a
    percolation toe below ~15 wt%; this linear map overstates sigma there and is
    used solely to label the premix_frac axis. See spec sec.3 and sec.8.
    """
    return float(wt_pct) / PREMIX_WTPCT_FULL
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_premix.py::test_wtpct_bridge_is_linear_over_doped_label -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add premix.py test_premix.py
git commit -m "feat(premix): [ASSUMED] wt% display bridge"
```

---

### Task 4: Wire helper into 2-D assembly site 1 (static builder)

**Files:**
- Modify: `rfam_eqs_coupled.py` (near `:1446-1452`)
- Test: `test_premix_integration.py`

- [ ] **Step 1: Write the failing integration test (bit-identical off-path)**

This test drives the actual builder function that contains site 1 and asserts that
adding `premix: {frac: 0.0}` to the config changes NOTHING. First identify the public
entry function containing line 1451 (grep for the enclosing `def`), call it `<ENTRY1>`.

```python
# test_premix_integration.py
import numpy as np, yaml, copy
import rfam_eqs_coupled as rc

def _load_jared():
    with open("configs/jared_exp1_40mm.yaml") as fh:
        return yaml.safe_load(fh)

def test_site1_premix_zero_matches_baseline():
    base = _load_jared()
    out_ref = rc.<ENTRY1>(copy.deepcopy(base))          # baseline (no premix key)
    cfg = copy.deepcopy(base); cfg["premix"] = {"frac": 0.0, "budget": "floor_added"}
    out_new = rc.<ENTRY1>(cfg)
    # Compare the Qrf field (or returned sigma/eps) bit-for-bit.
    assert np.array_equal(np.asarray(out_ref), np.asarray(out_new))
```

> During execution: replace `<ENTRY1>` with the real function name and adjust the
> return-value comparison to whatever the function returns (Qrf array or a record).
> If the entry runs a full solve that is too heavy, shrink `grid_nx/grid_ny` in `base`
> to ≤32 for the test.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_premix_integration.py::test_site1_premix_zero_matches_baseline -v`
Expected: FAIL — `premix` key is ignored today, so either an assertion error is impossible
(both equal) OR the function errors on the unknown key. If both sides are already equal
because premix is a no-op, this test is a GUARD; make it meaningful by first implementing
Step 3 with a deliberate bug (apply premix unconditionally) to see it fail, then fix.

- [ ] **Step 3: Read premix params and call the helper at site 1**

```python
# rfam_eqs_coupled.py — after line 1450 (endpoints resolved), replace lines 1451-1452:
from premix import apply_premix
_pm = base.get("premix", {}) or {}
_pm_frac = float(_pm.get("frac", 0.0))
_pm_budget = str(_pm.get("budget", "floor_added"))
sigma, eps_r = apply_premix(fill_frac, fill_frac,
                            sigma_v=sigma_v, sigma_d0=sigma_d0, eps_v=eps_v, eps_d=eps_d,
                            premix_frac=_pm_frac, premix_budget=_pm_budget)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_premix_integration.py::test_site1_premix_zero_matches_baseline -v`
Expected: PASS (premix_frac=0 → identity → bit-identical Qrf).

- [ ] **Step 5: Commit**

```bash
git add rfam_eqs_coupled.py test_premix_integration.py
git commit -m "feat(premix): wire premix into 2-D static builder (site 1), off-path bit-identical"
```

---

### Task 5: Wire helper into 2-D assembly site 2 (coupled/feedback builder)

**Files:**
- Modify: `rfam_eqs_coupled.py` (near `:2800-2805`)
- Test: `test_premix_integration.py`

- [ ] **Step 1: Write the failing test**

Identify the entry function containing line 2801 (`<ENTRY2>`; the main coupled `run`).

```python
# test_premix_integration.py  (append)
def test_site2_premix_zero_matches_baseline():
    base = _load_jared()
    # small grid + few steps so the coupled run is fast
    base["geometry"]["grid_nx"] = 24; base["geometry"]["grid_ny"] = 24
    base.setdefault("thermal", {})["n_steps"] = 2
    ref = rc.<ENTRY2>(copy.deepcopy(base))
    cfg = copy.deepcopy(base); cfg["premix"] = {"frac": 0.0, "budget": "floor_added"}
    new = rc.<ENTRY2>(cfg)
    assert np.array_equal(np.asarray(ref["Qrf"]), np.asarray(new["Qrf"]))
```

> Adjust config keys (`geometry`, `thermal`) and the result key (`"Qrf"`) to the real
> schema during execution.

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest test_premix_integration.py::test_site2_premix_zero_matches_baseline -v`
Expected: FAIL (as in Task 4 Step 2 — verify by temporarily forcing premix on).

- [ ] **Step 3: Call the helper at site 2**

```python
# rfam_eqs_coupled.py — replace lines 2801 and 2805 (keep _eff_fill / _eff_fill_eps as-is):
_pm = cfg.get("premix", {}) or {}
_pm_frac = float(_pm.get("frac", 0.0))
_pm_budget = str(_pm.get("budget", "floor_added"))
sigma, eps_r = apply_premix(_eff_fill, _eff_fill_eps,
                            sigma_v=sigma_v, sigma_d0=sigma_d0, eps_v=eps_v, eps_d=eps_d,
                            premix_frac=_pm_frac, premix_budget=_pm_budget)
```

(Ensure `from premix import apply_premix` is imported once at module top.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest test_premix_integration.py -v`
Expected: PASS (both site tests).

- [ ] **Step 5: Commit**

```bash
git add rfam_eqs_coupled.py test_premix_integration.py
git commit -m "feat(premix): wire premix into 2-D coupled builder (site 2), off-path bit-identical"
```

---

### Task 6: Premix-aware total-dopant accounting in fgm_generator (budget_fixed only)

**Files:**
- Modify: `fgm_generator.py`
- Test: `test_premix.py`

- [ ] **Step 1: Determine whether the generator needs any change**

Read `fgm_generator.py` for where it computes a total-dopant / mass budget. If the
generator produces only the printed sat map and never accounts for a baseline, then
`floor_added` needs no change (premix is added downstream in the material law) and
`budget_fixed` accounting is handled entirely by `apply_premix` (part pinned to doped).
**If so, this task reduces to a docstring note — record that finding and skip to commit.**

- [ ] **Step 2: If a budget is computed, write the failing test**

```python
# test_premix.py  (append) — only if fgm_generator exposes a total-dopant function
from fgm_generator import printed_dopant_fraction  # real name TBD during execution
def test_budget_fixed_conserves_total_dopant_vs_floor_added():
    # With budget_fixed, total dopant (premix + printed) should be ~independent of premix_frac.
    ...
```

- [ ] **Step 3: Implement minimal accounting** (only if Step 1 found a real budget).

- [ ] **Step 4: Run tests.** Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add fgm_generator.py test_premix.py
git commit -m "feat(premix): budget_fixed total-dopant accounting in fgm_generator"
```

---

### Task 7: Premix config for the study

**Files:**
- Create: `configs/jared_exp1_40mm_premix.yaml`

- [ ] **Step 1: Copy jared config and add a premix block**

```bash
cp configs/jared_exp1_40mm.yaml configs/jared_exp1_40mm_premix.yaml
```

Append (default OFF = identical to jared):
```yaml
premix:
  frac: 0.0            # swept by study_premix_sweep.py over {0,.15,.3,.45,.6}
  budget: floor_added
```

- [ ] **Step 2: Verify it loads and runs identically at frac=0**

Run: `python -m pytest test_premix_integration.py -v` (already covers frac=0 identity).

- [ ] **Step 3: Commit**

```bash
git add configs/jared_exp1_40mm_premix.yaml
git commit -m "feat(premix): jared_exp1_40mm premix study config (off by default)"
```

---

### Task 8: Study script — 0→15 wt% sweep (RUN GATED)

**Files:**
- Create: `study_premix_sweep.py`

- [ ] **Step 1: Write the sweep driver**

For each `premix_frac in [0.0, 0.15, 0.30, 0.45, 0.60]` (wt% = frac*25 via `premix_frac_from_wtpct` inverse):
1. re-solve the printed dopant map with this lane's cheap 2-D adjoint (reuse the existing
   adjoint entry — grep `fgm_solve_campaign/adjoint2d/` and the current cheap-adjoint runner);
2. record: printed map array, bulk RF absorption fraction (integral of Qrf over domain vs
   input power), drive-to-ceiling (bisect drive so 2-D end-state peak == 250 °C, consuming
   `solve3d/stage_a_phase2.py` ceiling value/method), dopant peak-relocation (Δpeak vs uniform map);
3. save a per-level record to `results/premix_sweep/level_<frac>.json` + `.npz`.

> This is the cheap 2-D adjoint × 5 levels — NOT a "heavy" solve. Still: run `uptime`
> first; if load ≥ 20 or ≥2 machine-wide heavy solves, defer and tell Matt (compute
> convention). Checkpoint each level so a crash resumes.

- [ ] **Step 2: Load-check, then run (Matt's go required)**

Run: `uptime` → confirm load acceptable → `python study_premix_sweep.py`
Expected: 5 level records under `results/premix_sweep/`.

- [ ] **Step 3: heatr3d 3-D cross-check at one level**

Re-run one premix level via `heatr3d.build_gamma(premix_frac=...)` and confirm the 2-D and
3-D absorption/peak-relocation agree in SIGN and rough magnitude. Record in the results dir.

- [ ] **Step 4: Commit**

```bash
git add study_premix_sweep.py results/premix_sweep/
git commit -m "study(premix): 0->15 wt% sweep on jared_exp1_40mm + heatr3d cross-check"
```

---

### Task 9: Composite figure

**Files:**
- Create: `make_premix_figure.py`, output `figures/fig_premix_sweep.png`

- [ ] **Step 1: Build the composite** (visualization-standard: DPI 180, bilinear, locked colormaps)

Top row: printed dopant maps at each premix level (shared colorbar).
Bottom row: absorption vs premix; drive-to-ceiling vs premix; peak-relocation vs premix.
Dual x-axis: premix_frac (bottom) + wt% [ASSUMED] (top). Caption carries the σ_d0=effective
and [ASSUMED]-wt% caveats from spec sec.8.

- [ ] **Step 2: View the PNG personally before delivering** (memory: view-figure-renders-personally).

- [ ] **Step 3: Commit**

```bash
git add make_premix_figure.py figures/fig_premix_sweep.png
git commit -m "figure(premix): composite 0->15 wt% premix sweep (model, +optional measured overlay)"
```

---

## Self-Review

**Spec coverage:** A (material knob) → Tasks 1-2,4-5. B (wt% bridge) → Task 3. C (config/generator)
→ Tasks 6-7. D (study) → Task 8. E (figure) → Task 9. Gates: bit-identical off-path → Tasks 1,4,5;
helper ladder → Tasks 1-2; FD no-new-gradient → stated in Architecture (premix is a swept param);
load-check → Task 8. Optional empirical overlay → Task 9 (flagged, not default). All covered.

**Placeholder scan:** Tasks 4-6 contain `<ENTRY1>`/`<ENTRY2>`/real-name-TBD markers — these are
UNAVOIDABLE because the exact enclosing function name and result schema must be read from the live
3000-line module at execution time. Each is accompanied by the exact grep to resolve it and the
concrete edit once resolved. Task 6 is explicitly conditional with a decision gate in Step 1. Not
lazy placeholders — bounded, instructed lookups.

**Type consistency:** `apply_premix(blend_sigma, blend_eps, *, sigma_v, sigma_d0, eps_v, eps_d,
premix_frac, premix_budget)` used identically in Tasks 1,2,4,5. `premix_frac_from_wtpct` and
`PREMIX_WTPCT_FULL` consistent in Tasks 1,3. Config key `premix: {frac, budget}` consistent in
Tasks 4,5,7,8.
