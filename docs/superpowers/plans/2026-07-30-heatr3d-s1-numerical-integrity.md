# heatr3d S1 Numerical Integrity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pass Gate S1 of the heatr3d graduation spec (docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md): root-cause and fix the melt-onset instability with a regression test, add a standing energy-conservation gate, and land analytic benchmarks for the thermal-phase and EQS cores.

**Architecture:** All changes go into `geo-prewarp/heatr3d.py` (the synced working copy; the canonical file in `dissertation_materials/analysis-3dfgm/` is re-synced only as a deliberate reviewed step at gate pass, with Matt's sign-off, updating `SYNCED_FROM_SHA256`). New behavior is opt-in via a `Params.phase_update` field defaulting to the legacy `"apparent_cp"` so all historical numbers are bit-for-bit reproducible; the S1 campaign runs with `"enthalpy"`. Tests live at repo root (`test_heatr3d_s1.py`), following the existing root-test convention (`test_heatr3d_outputs.py`). Expensive full-scale runs are marked `@pytest.mark.slow`.

**Tech Stack:** Python 3.12 (`./.venv312/bin/python`), numpy, scipy.sparse, pytest. Run all commands from the geo-prewarp repo root.

**Primary hypothesis (to verify, not assume):** the legacy phase treatment computes latent heat via pointwise `dphi = d(phi)/dT` at the current T inside `cp_eff = cp + latent * dphi`. With `max_dt_step_c = 10.0` exactly equal to the melt window `dt_pc_c = 10.0`, a strongly heated cell (sharp corner Q_rf at fine grids) can cross the entire window in one step where `dphi(T_current) ~ 0`, skipping the latent sink entirely: phi then snaps and the energy books show a huge surplus. This reproduces the documented symptom (phi 0.019 -> 0.953 in one step, residual ~1e5 x dose at grid >= 200). Secondary hypothesis: explicit conduction CFL violation (checked and expected to be innocent: alpha_max = k_liquid/(rho_liquid cp_liquid) ~ 7.8e-8 m2/s gives dt_max = h^2/(6 alpha) ~ 0.19 s > dt_s = 0.05 even at n = 200).

---

### Task 1: Energy audit instrumentation (measurement before surgery)

Add cumulative energy accounting to `run()` so the instability is quantified, and every future solve carries the standing gate the spec requires.

**Files:**
- Modify: `heatr3d.py` (class `Result` ~line 427, `run()` time loop ~lines 560-660 and the return path)
- Test: `test_heatr3d_s1.py` (create)

- [ ] **Step 1: Write the failing test**

Create `test_heatr3d_s1.py`:

```python
"""S1 numerical-integrity tests for heatr3d (spec: docs/superpowers/specs/
2026-07-30-heatr3d-graduation-design.md, Gate S1)."""
import dataclasses

import numpy as np
import pytest

from heatr3d import Grid, Params, make_geometry, run


def small_sphere_case(n=32):
    grid = Grid(n=n, L=0.060)
    part = make_geometry(grid, "sphere", diam=0.020)
    p = Params()
    return grid, part, p


def test_energy_audit_fields_present_and_small_on_benign_run():
    grid, part, p = small_sphere_case()
    res = run(grid, part, p, max_time_s=30.0, verbose=False)
    # new provenance fields
    assert hasattr(res, "energy_in_j")
    assert hasattr(res, "energy_stored_j")
    assert hasattr(res, "energy_loss_j")
    assert hasattr(res, "energy_residual_frac")
    assert res.energy_in_j > 0
    # benign pre-melt run must conserve energy to a few percent
    assert abs(res.energy_residual_frac) < 0.05
```

- [ ] **Step 2: Run test to verify it fails for the right reason**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_energy_audit_fields_present_and_small_on_benign_run -v`
Expected: FAIL with `AssertionError` on `hasattr(res, "energy_in_j")` (fields do not exist yet). If it fails on import instead, fix the import before proceeding.

- [ ] **Step 3: Implement the audit**

In `class Result`, add after `clamp_bound: bool = False`:

```python
    # S1 energy audit (Joules over the whole domain, cumulative over the run):
    # in = RF deposited; stored = sensible + latent actually banked in the
    # temperature/phase state; loss = convection + powder-loss + heatsink sinks.
    # residual_frac = (in - stored - loss) / max(in, 1e-30). The standing S1
    # gate: |residual_frac| must be small; a blow-up shows up here first.
    energy_in_j: float = 0.0
    energy_stored_j: float = 0.0
    energy_loss_j: float = 0.0
    energy_residual_frac: float = 0.0
```

In `run()`, before the `for it in range(nsteps):` loop add:

```python
    dV = grid.dV
    e_in = 0.0
    e_loss = 0.0
    T0 = T.copy()
    phi0, _ = phase_fraction(T0, p)
```

Inside the loop, immediately after `num -= q_conv` (and after the optional
`q_loss` / heatsink subtractions so every sink is counted), add:

```python
        # ---- S1 energy audit (per step, before the dT clamp) ----
        e_in += float((Qrf * (pmult if pmult != 1.0 else 1.0)).sum()) * dV * p.dt_s
        e_loss += float(q_conv.sum()) * dV * p.dt_s
        if h_eff_loss != 0.0:
            e_loss += float(q_loss.sum()) * dV * p.dt_s
        if heatsink_field is not None and heatsink_h != 0.0:
            e_loss += float((heatsink_h * np.asarray(heatsink_field, dtype=float)
                             * (T - p.ambient_c)).sum()) * dV * p.dt_s
```

(Place the block so it reads the same arrays the update uses; `q_loss` is only
defined when `h_eff_loss != 0.0`, matching the guard.)

After the loop ends (where the Result is assembled), compute stored energy from
the STATE CHANGE, which is what makes the audit catch latent-skip bugs:

```python
    phiN, _ = phase_fraction(T, p)
    rho_s_eff0 = p.rho_powder + rho_rel * (p.rho_solid - p.rho_powder)
    rho_map = np.full(part.shape, p.rho_powder)
    cp_map = np.full(part.shape, p.cp_powder)
    rho_map[part] = rho_s_eff0[part]
    cp_map[part] = p.cp_solid
    e_sensible = float((rho_map * cp_map * (T - T0)).sum()) * dV
    e_latent = float((rho_map[part] * p.latent_j_per_kg
                      * (phiN[part] - phi0[part])).sum()) * dV
    e_stored = e_sensible + e_latent
    e_resid_frac = (e_in - e_stored - e_loss) / max(e_in, 1e-30)
```

and populate the Result fields:

```python
        energy_in_j=e_in,
        energy_stored_j=e_stored,
        energy_loss_j=e_loss,
        energy_residual_frac=e_resid_frac,
```

Note: the stored-energy estimate deliberately uses initial-state properties
(a first-order sensible-heat model). It is an AUDIT, not a solver term; its
job is to be O(few %) on healthy runs and O(>>1) on latent-skip blow-ups.
Property-drift refinements are out of scope for this task.

- [ ] **Step 4: Run test to verify it passes**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_energy_audit_fields_present_and_small_on_benign_run -v`
Expected: PASS (~20-60 s; n=32, 600 steps).

- [ ] **Step 5: Verify bit-for-bit inertness on existing outputs**

Run: `./.venv312/bin/python -m pytest test_heatr3d_outputs.py -v`
Expected: all existing tests PASS (the audit only reads state).

- [ ] **Step 6: Commit**

```bash
git add heatr3d.py test_heatr3d_s1.py
git commit -m "feat(heatr3d): S1 cumulative energy audit on every run"
```

### Task 2: Mechanism-level instability reproduction (RED for the fix)

A cheap, deterministic repro of the melt-onset blow-up using the existing
`qrf_override` validation hook: a synthetic corner-spike heating field sized so
one Euler step crosses the whole melt window. This encodes the failure
mechanism, not the expensive n>=200 trigger.

**Files:**
- Test: `test_heatr3d_s1.py` (append)

- [ ] **Step 1: Write the reproduction test (asserts the LEGACY failure signature)**

Append:

```python
def spike_case(n=32, spike_mult=400.0):
    """Uniform mild heating plus one interior hot column whose raw per-step dT
    exceeds the full melt window (dt_pc_c), forcing a window-crossing step."""
    grid, part, p = small_sphere_case(n)
    q = np.zeros((n, n, n))
    q[part] = p.power_density_w_per_m3
    ii = np.argwhere(part)
    c = ii[len(ii) // 2]
    q[c[0], c[1], c[2]] *= spike_mult
    return grid, part, p, q


def test_legacy_phase_update_skips_latent_on_window_crossing():
    grid, part, p, q = spike_case()
    res = run(grid, part, p, qrf_override=q, max_time_s=120.0)
    # Legacy failure signature: the audit books a large energy surplus because
    # the spiked cell jumps the melt window without paying latent heat, and a
    # limiter binds (clamp_bound True). This test PINS the bug; it is inverted
    # into the regression test once the enthalpy update lands.
    assert res.clamp_bound is True
    assert res.energy_residual_frac > 0.10
```

- [ ] **Step 2: Run and calibrate the repro**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_legacy_phase_update_skips_latent_on_window_crossing -v`
Expected: PASS. If it does not blow up, raise `spike_mult` (try 1000.0) until
`clamp_bound` latches and the residual exceeds 0.10; record the final value in
the test. If no spike_mult up to 5000 reproduces it, STOP: the hypothesis is
wrong, write findings to `docs/superpowers/plans/s1-findings.md` and revisit
the diagnosis with the secondary (CFL) instrumentation from Task 3 before
touching any fix.

- [ ] **Step 3: Commit**

```bash
git add test_heatr3d_s1.py
git commit -m "test(heatr3d): pin melt-window latent-skip failure mechanism"
```

### Task 3: Diagnosis write-up (mechanism verified, not assumed)

**Files:**
- Create: `docs/superpowers/plans/s1-findings.md`
- Create: `scripts/analysis/s1_diagnose_instability.py`

- [ ] **Step 1: Write the diagnostic script**

Create `scripts/analysis/s1_diagnose_instability.py`:

```python
#!/usr/bin/env python3
"""S1 diagnosis: quantify per-step dT vs the melt window and the conduction
CFL number at the moment of blow-up, for the mechanism repro case.

Run: ./.venv312/bin/python scripts/analysis/s1_diagnose_instability.py
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from heatr3d import Grid, Params, make_geometry, phase_fraction  # noqa: E402


def main() -> None:
    n = 32
    grid = Grid(n=n, L=0.060)
    part = make_geometry(grid, "sphere", diam=0.020)
    p = Params()
    # conduction CFL bound (secondary hypothesis check)
    alpha_max = p.k_liquid / (p.rho_liquid * p.cp_liquid)
    dt_cfl = grid.h ** 2 / (6.0 * alpha_max)
    print(f"h={grid.h*1e3:.3f} mm  dt_s={p.dt_s}  dt_CFL={dt_cfl:.3f} s  "
          f"CFL ratio={p.dt_s/dt_cfl:.3f} (<1 means conduction-stable)")
    for nn in (48, 96, 200):
        hh = 0.060 / nn
        print(f"  n={nn}: dt_CFL={hh**2/(6*alpha_max):.3f} s "
              f"ratio={p.dt_s/(hh**2/(6*alpha_max)):.3f}")
    # source-term dT for a corner-spike voxel vs the melt window
    for mult in (1, 10, 100, 400, 1000):
        q = p.power_density_w_per_m3 * mult
        dT = q * p.dt_s / (p.rho_solid * p.cp_solid)
        print(f"  spike x{mult:5d}: raw source dT/step = {dT:8.3f} C "
              f"(melt window = {p.dt_pc_c} C, clamp = {p.max_dt_step_c} C)")
    # latent barrier a window-crossing step must pay
    e_latent_per_c = p.latent_j_per_kg / p.dt_pc_c
    print(f"latent per C inside window: {e_latent_per_c:.0f} J/(kg C) vs "
          f"cp_solid {p.cp_solid} J/(kg C) -> apparent cp in-window ~x"
          f"{1 + e_latent_per_c/p.cp_solid:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and write the findings**

Run: `./.venv312/bin/python scripts/analysis/s1_diagnose_instability.py`

Write `docs/superpowers/plans/s1-findings.md` recording, with the printed
numbers pasted in: (1) whether conduction CFL is innocent at n=200 (expected:
ratio < 1), (2) the spike multiplier at which one source step exceeds the melt
window, (3) the conclusion sentence naming the mechanism the Task 2 test
pinned, and (4) the grid-dependence argument (finer grid concentrates corner
Q_rf into fewer voxels, raising the per-voxel source dT past the window). If
the Task 2 repro required stopping (hypothesis wrong), this file instead
records what was observed and the revised hypothesis.

- [ ] **Step 3: Commit**

```bash
git add scripts/analysis/s1_diagnose_instability.py docs/superpowers/plans/s1-findings.md
git commit -m "docs(heatr3d): S1 instability diagnosis findings"
```

### Task 4: Enthalpy-conserving phase update (the fix), opt-in

Replace the pointwise apparent-cp step with an exact piecewise-linear enthalpy
inversion, guarded by `Params.phase_update` (default `"apparent_cp"` = legacy
bit-for-bit; `"enthalpy"` = fix).

**Files:**
- Modify: `heatr3d.py` (`Params` ~line 47, new module function near
  `phase_fraction` ~line 413, `run()` update step ~lines 636-660)
- Test: `test_heatr3d_s1.py` (append)

- [ ] **Step 1: Write the failing unit test for the enthalpy inversion**

Append to `test_heatr3d_s1.py`:

```python
def test_enthalpy_roundtrip_and_window_crossing():
    from heatr3d import enthalpy_from_T, T_from_enthalpy
    p = Params()
    rho_cp = p.rho_solid * p.cp_solid          # J/(m^3 K), sensible slope
    rho_L = p.rho_solid * p.latent_j_per_kg    # J/m^3, latent plateau
    Ts = np.array([25.0, 175.0, 180.0, 185.0, 190.0, 250.0])
    H = enthalpy_from_T(Ts, rho_cp, rho_L, p)
    Tb = T_from_enthalpy(H, rho_cp, rho_L, p)
    assert np.allclose(Tb, Ts, atol=1e-9)
    # depositing exactly the latent plateau plus 20 C sensible from the window
    # start lands 20 C above the window end, never skipping the latent barrier
    H0 = enthalpy_from_T(np.array([p.t_pc_c - p.dt_pc_c / 2]), rho_cp, rho_L, p)
    H1 = H0 + rho_L + rho_cp * (p.dt_pc_c + 20.0)
    T1 = T_from_enthalpy(H1, rho_cp, rho_L, p)
    assert np.allclose(T1, p.t_pc_c + p.dt_pc_c / 2 + 20.0, atol=1e-9)
```

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_enthalpy_roundtrip_and_window_crossing -v`
Expected: FAIL with `ImportError: cannot import name 'enthalpy_from_T'`.

- [ ] **Step 3: Implement the enthalpy pair + Params field**

In `Params` add (with the other numerics fields):

```python
    # S1: phase-update scheme. "apparent_cp" = legacy pointwise dphi in cp_eff
    # (bit-for-bit historical behavior; can skip the latent barrier when one
    # step crosses the melt window). "enthalpy" = exact piecewise-linear
    # enthalpy inversion (energy-conserving by construction; S1 fix).
    phase_update: str = "apparent_cp"
```

Add module functions after `phase_fraction`:

```python
def enthalpy_from_T(T, rho_cp, rho_L, p: Params):
    """Volumetric enthalpy H(T) [J/m^3], piecewise linear: sensible slope
    rho_cp everywhere plus the latent plateau rho_L ramped linearly across
    the melt window [t_pc - dt_pc/2, t_pc + dt_pc/2]."""
    T = np.asarray(T, dtype=np.float64)
    lo = p.t_pc_c - p.dt_pc_c / 2.0
    frac = np.clip((T - lo) / p.dt_pc_c, 0.0, 1.0)
    return rho_cp * T + rho_L * frac


def T_from_enthalpy(H, rho_cp, rho_L, p: Params):
    """Exact inverse of enthalpy_from_T for scalar-per-voxel rho_cp/rho_L."""
    H = np.asarray(H, dtype=np.float64)
    lo = p.t_pc_c - p.dt_pc_c / 2.0
    H_lo = rho_cp * lo
    H_hi = rho_cp * (lo + p.dt_pc_c) + rho_L
    T_below = H / rho_cp
    # in-window: H = rho_cp*T + rho_L*(T-lo)/dt_pc
    T_window = (H + rho_L * lo / p.dt_pc_c) / (rho_cp + rho_L / p.dt_pc_c)
    T_above = (H - rho_L) / rho_cp
    return np.where(H <= H_lo, T_below,
                    np.where(H >= H_hi, T_above, T_window))
```

- [ ] **Step 4: Run to verify the unit test passes**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_enthalpy_roundtrip_and_window_crossing -v`
Expected: PASS.

- [ ] **Step 5: Write the failing integration test (fix heals the repro)**

Append:

```python
def test_enthalpy_update_conserves_energy_on_window_crossing():
    # Task-3 finding: at spike_mult=400 x 120 s the spiked voxel saturates at
    # temp_max_c and RF into a clamped voxel dominates the residual, which the
    # phase fix cannot and should not hide. Isolate the phase mechanism with a
    # softer spike (still window-crossing: mult 200 > the ~145 crossing
    # threshold) and a shorter run, and assert no saturation occurred.
    grid, part, p, q = spike_case(spike_mult=200.0)
    p = dataclasses.replace(p, phase_update="enthalpy")   # Params is frozen
    res = run(grid, part, p, qrf_override=q, max_time_s=60.0)
    assert res.T_max_c < p.temp_max_c - 1.0     # no temp-clamp saturation
    assert abs(res.energy_residual_frac) < 0.05
    # the spiked cell still crosses the window; it pays the latent toll now
    assert res.phi_final.max() > 0.9


def test_legacy_default_is_unchanged():
    grid, part, p = small_sphere_case()
    r1 = run(grid, part, p, max_time_s=20.0)
    r2 = run(grid, part, p, max_time_s=20.0)
    assert np.array_equal(r1.T_phi90 if r1.T_phi90 is not None else np.zeros(1),
                          r2.T_phi90 if r2.T_phi90 is not None else np.zeros(1))
    assert Params().phase_update == "apparent_cp"
```

- [ ] **Step 6: Run to verify the integration test fails**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_enthalpy_update_conserves_energy_on_window_crossing -v`
Expected: FAIL (the flag exists but `run()` ignores it, so the legacy path
still skips latent and the residual assertion trips).

- [ ] **Step 7: Wire the enthalpy branch into run()**

In the `run()` update step, replace the block from `dTdt = num / ...` through
`T_cand = np.array(T, copy=True) + dT` with:

```python
        if p.phase_update == "enthalpy":
            # Energy-conserving update: deposit num*dt into volumetric
            # enthalpy and invert exactly. Latent uses the SOLID density
            # basis for a consistent H(T) (audit uses the same basis).
            rho_cp_map = rho * cp                     # sensible slope, J/(m^3 K)
            rho_L_map = np.zeros(part.shape)
            rho_L_map[part] = rho_s_eff[part] * p.latent_j_per_kg
            H = enthalpy_from_T(T, rho_cp_map, rho_L_map, p)
            H = H + p.dt_s * np.nan_to_num(num)
            T_new = T_from_enthalpy(H, rho_cp_map, rho_L_map, p)
            dT_raw = T_new - T
            dT = np.clip(dT_raw, -p.max_dt_step_c, p.max_dt_step_c)
        else:
            dTdt = num / np.maximum(rho * cp_eff, 1e-9)
            dT_raw = p.dt_s * np.nan_to_num(dTdt)
            dT = np.clip(dT_raw, -p.max_dt_step_c, p.max_dt_step_c)
```

(the THM-01/02 clamp diagnostics below continue to operate on `dT_raw`/`dT`
unchanged, so limiter binding stays visible in both schemes).

- [ ] **Step 8: Run the full S1 file**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py -v`
Expected: ALL PASS, including the Task 2 legacy-pin test (legacy path is
untouched) and `test_legacy_default_is_unchanged`.

- [ ] **Step 9: Run existing suites for regressions**

Run: `./.venv312/bin/python -m pytest test_heatr3d_outputs.py test_fgm_pipeline.py -v`
Expected: PASS (default behavior bit-for-bit).

- [ ] **Step 10: Commit**

```bash
git add heatr3d.py test_heatr3d_s1.py
git commit -m "feat(heatr3d): opt-in enthalpy-conserving phase update (S1 fix)"
```

### Task 4b: Audit property upgrade (Task-4 finding 2)

The Task-1 audit books stored energy with initial-state solid properties, so
it drifts to +0.38 on HEALTHY molten runs (solver blends to cp_liquid=3279,
rho_liquid=1010 while the audit holds cp_solid/rho_s). The standing gate must
be meaningful AT melt, which is where the instability lives. Fix: accumulate
stored energy per step with the SAME property maps the solver used that step.

**Files:**
- Modify: `heatr3d.py` (run() audit block)
- Test: `test_heatr3d_s1.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_audit_stays_tight_through_melt_both_schemes():
    # Healthy uniform molten run: residual must stay small ABOVE the window
    # (Task-4 finding: the v1 audit drifted to +0.38 here).
    grid, part, p, q = bulk_crossing_case()
    for scheme in ("apparent_cp", "enthalpy"):
        pp = dataclasses.replace(p, phase_update=scheme)
        res = run(grid, part, pp, qrf_override=q, max_time_s=3.0,
                  phi_target=2.0)
        assert abs(res.energy_residual_frac) < 0.02, scheme
```

(If bulk_crossing_case's legacy arm still books the latent-skip discrepancy
at 3.0 s, the legacy assertion may need its own looser bound; keep the
enthalpy bound at 0.02 and record the legacy value in the test docstring -
the AUDIT upgrade must not mask the SCHEME defect the Task-4 test pins.)

- [ ] **Step 2: Run to verify it fails**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_audit_stays_tight_through_melt_both_schemes -v`
Expected: FAIL on the residual bound (v1 audit drift).

- [ ] **Step 3: Implement per-step stored-energy accumulation**

In run(), replace the post-loop stored-energy computation with per-step
accumulation placed right after T is updated (dT applied):

```python
        # S1 audit v2: bank stored energy with THIS step's property maps
        e_stored_acc += float((rho * cp * dT).sum()) * dV
```

and latent banking from the actual phi change across the step:

```python
        phi_new, _ = phase_fraction(T, p)
        e_stored_acc += float((rho_s_eff[part] * p.latent_j_per_kg
                               * (phi_new[part] - phi[part])).sum()) * dV
```

(initialize `e_stored_acc = 0.0` before the loop; delete the old post-loop
sensible/latent estimate; keep the residual formula, using e_stored_acc).
Note phase_fraction(T) after the update is one extra evaluation per step;
if profiling shows it matters, reuse the next iteration's phi instead -
correctness first, measure before optimizing.

- [ ] **Step 4: Run the whole S1 file**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py -v`
Expected: ALL PASS including the Task 1 benign test (machine-precision
residual must survive the accumulation change) and Task 4's healing test.

- [ ] **Step 5: Commit**

```bash
git add heatr3d.py test_heatr3d_s1.py
git commit -m "fix(heatr3d): S1 audit v2 banks stored energy with per-step properties"
```

### Task 5: Analytic benchmark 1 - adiabatic uniform heating with latent plateau

**Files:**
- Test: `test_heatr3d_s1.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_adiabatic_uniform_heating_matches_analytic_plateau():
    """Whole domain = part, uniform q, conv off: T(t) is analytic including
    the latent plateau. Exact for the enthalpy scheme by construction; this
    pins the wiring (property maps, dt, source bookkeeping)."""
    n = 16
    grid = Grid(n=n, L=0.060)
    part = np.ones((n, n, n), dtype=bool)
    p = dataclasses.replace(Params(), phase_update="enthalpy", conv_h=0.0)
    q = np.full((n, n, n), 2.0e5)               # W/m^3, uniform
    t_end = 200.0
    res = run(grid, part, p, qrf_override=q, max_time_s=t_end,
              phi_target=2.0)                    # never stop early
    # analytic: uniform state, no gradients -> pure source integration
    rho_s = p.rho_powder + p.rho_rel * (p.rho_solid - p.rho_powder)
    rho_cp = rho_s * p.cp_solid
    rho_L = rho_s * p.latent_j_per_kg
    H_end = enthalpy_from_T(np.array([p.preheat_c]), rho_cp, rho_L, p) \
        + 2.0e5 * t_end
    from heatr3d import T_from_enthalpy
    T_exact = float(T_from_enthalpy(H_end, rho_cp, rho_L, p))
    T_num = float(res.T_phi90.mean()) if res.T_phi90 is not None else None
    # compare the FINAL temperature field (uniform), via phi_final/T bookkeeping:
    # run() must expose final T for this; see step 3 note.
    assert abs(float(res.T_final.mean()) - T_exact) < 0.5
    assert float(res.T_final.std()) < 1e-6
```

- [ ] **Step 2: Run to verify failure mode**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_adiabatic_uniform_heating_matches_analytic_plateau -v`
Expected: FAIL with `AttributeError: 'Result' object has no attribute 'T_final'`.

- [ ] **Step 3: Add `T_final` to Result and populate it**

In `class Result` add `T_final: np.ndarray | None = None`; in `run()`'s
result assembly pass `T_final=T.copy()`. Notes for the test: the enthalpy
scheme with liquid-fraction property blending is not exactly the fixed-slope
analytic H used above once phi > 0 (cp blends toward cp_liquid). Handle it in
the test, not the solver: choose q and t_end so the exact solution stays
BELOW the window top (mid-plateau), where H(T) inversion in-window is
governed by the same rho_cp used at window entry within 0.5 C tolerance; if
the 0.5 C tolerance still trips due to property blending, tighten the test by
building the Params with `dataclasses.replace(p, cp_liquid=p.cp_solid, k_liquid=p.k_solid, rho_liquid=rho_s)` inside this test (Params is a frozen dataclass; in-place mutation raises FrozenInstanceError) (a legitimate benchmark
configuration: constant properties are exactly what the analytic solution
assumes), and assert `< 0.05` instead.

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_adiabatic_uniform_heating_matches_analytic_plateau -v`
Expected: PASS (seconds; n=16).

- [ ] **Step 5: Commit**

```bash
git add heatr3d.py test_heatr3d_s1.py
git commit -m "test(heatr3d): analytic adiabatic latent-plateau benchmark"
```

### Task 6: Analytic benchmark 2 - pure conduction decay (no phase, no source)

**Files:**
- Test: `test_heatr3d_s1.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_conduction_decay_matches_fourier_mode():
    """No source, no convection, uniform powder medium, initial condition =
    lowest cosine Fourier mode compatible with Neumann walls. The mode decays
    as exp(-alpha k^2 t) exactly; second-order spatial accuracy expected."""
    n = 24
    grid = Grid(n=n, L=0.060)
    part = np.zeros((n, n, n), dtype=bool)     # all powder, no part
    p = dataclasses.replace(Params(), conv_h=0.0)
    q = np.zeros((n, n, n))
    # run() initializes T uniformly; to inject the mode this test uses the
    # T0_override hook added in step 3.
    kx = np.pi / grid.L
    x = grid.x.reshape(-1, 1, 1)
    T0 = p.preheat_c + 5.0 * np.cos(kx * (x + grid.L / 2.0)) * np.ones((n, n, n))
    t_end = 400.0
    res = run(grid, part, p, qrf_override=q, max_time_s=t_end, phi_target=2.0,
              T0_override=T0)
    alpha = p.k_powder / (p.rho_powder * p.cp_powder)
    decay = np.exp(-alpha * kx ** 2 * t_end)
    amp_num = float((res.T_final.max() - res.T_final.min()) / 2.0)
    amp_exact = 5.0 * decay
    assert abs(amp_num - amp_exact) / amp_exact < 0.05
```

- [ ] **Step 2: Run to verify failure**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_conduction_decay_matches_fourier_mode -v`
Expected: FAIL with `TypeError: run() got an unexpected keyword argument 'T0_override'`.

- [ ] **Step 3: Add the T0_override hook**

In `run()`'s signature add `T0_override: np.ndarray | None = None`, and
replace the initialization line with:

```python
    if T0_override is not None:
        T = np.array(T0_override, dtype=np.float64, copy=True)
        if T.shape != part.shape:
            raise ValueError("T0_override shape must match part.shape")
    else:
        T = np.full(part.shape, p.preheat_c, dtype=np.float64)
```

- [ ] **Step 4: Run to verify it passes**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_conduction_decay_matches_fourier_mode -v`
Expected: PASS. If the 5% tolerance trips, print the ratio and check the mode
wavenumber against the cell-centered Neumann grid (the discrete decay rate is
(2/h^2)(1-cos(k h)) alpha; using that discrete rate in `decay` is acceptable
and should agree to <1e-3 - document whichever form the test uses).

- [ ] **Step 5: Commit**

```bash
git add heatr3d.py test_heatr3d_s1.py
git commit -m "test(heatr3d): Fourier-mode conduction benchmark + T0 hook"
```

### Task 7: Analytic benchmark 3 - EQS parallel-plate field

**Files:**
- Test: `test_heatr3d_s1.py` (append)

- [ ] **Step 1: Write the failing test**

```python
def test_eqs_uniform_medium_is_parallel_plate():
    from heatr3d import build_gamma, solve_eqs_3d
    n = 24
    grid = Grid(n=n, L=0.060)
    part = np.zeros((n, n, n), dtype=bool)      # uniform virgin bed
    p = Params()
    gamma = build_gamma(part, p, None, h=grid.h)
    V = solve_eqs_3d(gamma, grid, p)
    Vr = np.real(V)
    # linear in y between the plates, uniform in x and z
    y_prof = Vr.mean(axis=(0, 2))
    y_lin = np.linspace(p.v_lo, p.v_hi, n)
    assert np.max(np.abs(y_prof - y_lin)) < 1e-6 * abs(p.v_lo)
    assert float(Vr.std(axis=(0, 2)).max()) < 1e-9 * abs(p.v_lo)
```

- [ ] **Step 2: Run**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py::test_eqs_uniform_medium_is_parallel_plate -v`
Expected: PASS immediately (this is a pure characterization test of existing
code; if it FAILS, that is an S1 finding of the first order - record the
discrepancy in s1-findings.md and investigate before proceeding). Check the
Dirichlet rows: `elec_lo` is y index 0 with `v_lo`, so `y_lin` runs from v_lo
to v_hi as written; if the profile is offset by half a cell, relax the linear
check to the interior points and document.

- [ ] **Step 3: Commit**

```bash
git add test_heatr3d_s1.py
git commit -m "test(heatr3d): EQS parallel-plate analytic characterization"
```

### Task 8: Full-scale regression (slow, opt-in) + gate wiring

**Files:**
- Test: `test_heatr3d_s1.py` (append)
- Modify: `heatr3d.py` (`run()` verbose print)

- [ ] **Step 1: Write the slow full-scale regression**

```python
@pytest.mark.slow
def test_full_scale_n200_melt_onset_clean_with_enthalpy():
    """The documented trigger (grid >= 200 melt-onset blow-up), run with the
    S1 fix: must stay clean. ~n=200^3 voxels: minutes to hours; run in the
    S1 campaign, not per-commit CI."""
    grid = Grid(n=200, L=0.060)
    part = make_geometry(grid, "sphere", diam=0.020)
    p = dataclasses.replace(Params(), phase_update="enthalpy")
    res = run(grid, part, p, max_time_s=900.0)
    assert res.reached is True
    assert abs(res.energy_residual_frac) < 0.05
    assert res.clamp_bound is False
```

- [ ] **Step 2: Register the marker**

Create or extend `pytest.ini` at repo root (check first: if a pytest config
already exists in `pyproject.toml` or `pytest.ini`, add the marker there
instead):

```ini
[pytest]
markers =
    slow: full-scale heatr3d runs (minutes+); excluded from default runs
addopts = -m "not slow"
```

- [ ] **Step 3: Verify default suite still green and fast**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py -v`
Expected: all fast tests PASS, slow test DESELECTED.

- [ ] **Step 4: Add the standing gate print**

In `run()`, at the end (inside `if verbose:` or add an always-on single
line), print:

```python
    print(f"  [s1-energy] in={e_in:.1f} J stored={e_stored:.1f} J "
          f"loss={e_loss:.1f} J residual_frac={e_resid_frac:+.4f}"
          f"{'  CLAMP-BOUND' if clamp_bound else ''}")
```

- [ ] **Step 5: Run the slow test once (the S1 campaign run)**

Run: `./.venv312/bin/python -m pytest test_heatr3d_s1.py -m slow -v`
Expected: PASS (budget hours; run overnight if needed). Paste the
`[s1-energy]` line and runtime into `docs/superpowers/plans/s1-findings.md`.
The audit here relies on Task 4b's per-step property accounting; if the residual assertion fails, first check whether the failure is audit drift (healthy fields, smooth phi history) vs a real blow-up (clamp_bound, phi discontinuity). If it FAILS for real: the mechanism fix is incomplete at scale - record the failure
signature in s1-findings.md, and add source-aware substepping as the next
fix (n_sub = ceil(max_source_dT / (0.5 * dt_pc_c)) applied to the enthalpy
deposit), then rerun. Do not weaken the assertions.

- [ ] **Step 6: Commit**

```bash
git add heatr3d.py test_heatr3d_s1.py pytest.ini docs/superpowers/plans/s1-findings.md
git commit -m "test(heatr3d): full-scale S1 regression + standing energy gate"
```

### Task 9: Gate S1 report + canonical-sync decision package

**Files:**
- Create: `docs/superpowers/specs/s1-gate-report.md`

- [ ] **Step 1: Write the gate report**

Summarize with numbers pasted from the runs (no placeholders): the pinned
mechanism and diagnosis, the fix, every benchmark result with tolerance, the
full-scale regression outcome, and the standing gate. End with the canonical
sync section: proposed diff summary for
`dissertation_materials/analysis-3dfgm/heatr3d.py`, the note that
`SYNCED_FROM_SHA256` must be updated, the statement that the default remains
`"apparent_cp"` (bit-for-bit historical), and an explicit sign-off line for
Matt. DO NOT touch the canonical file in this plan.

- [ ] **Step 2: Commit**

```bash
git add docs/superpowers/specs/s1-gate-report.md
git commit -m "docs(heatr3d): S1 gate report + canonical sync decision package"
```

- [ ] **Step 3: Present to Matt**

Post the report summary; S1 passes only with his sign-off. S2 (convergence)
planning starts after.

---

## Self-review notes

- Spec coverage: instability root-cause (Tasks 2-4), regression (Tasks 2, 8),
  analytic thermal benchmarks (Tasks 5-6), analytic EQS benchmark (Task 7),
  standing conservation gate on every solve (Tasks 1, 8), Stefan-style moving
  front: covered by the latent-plateau benchmark (Task 5) rather than a
  classical two-phase Stefan problem; if S2 reviewers want the classical
  Stefan front-position test it slots in as an S2 addition (documented choice,
  not an omission).
- Escape hatches are explicit (Task 2 step 2, Task 8 step 5): a wrong
  hypothesis stops work and reroutes through findings, never silently
  weakens an assertion.
- Type consistency: `enthalpy_from_T` / `T_from_enthalpy` signatures match
  between Tasks 4-5; `T_final` and `T0_override` introduced once each and
  reused; energy field names identical across Tasks 1, 4, 8.
- All commands run from geo-prewarp root with `./.venv312/bin/python`.
