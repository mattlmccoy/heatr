# Stage B3: Augmented Lagrangian with True-Peak Restoration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Checkbox steps.

**Goal:** Enforce the degradation ceiling on the TRUE arbiter peak by construction, via an augmented Lagrangian (reusing the FD-gated B1 density co-state) plus a restoration shift that re-targets KS to `ceiling - Delta` each outer iteration, so the shaped dopant at 0.40x lands is_shippable.

**Architecture:** Reuse `solve3d/density_adjoint.dks_peak_ds` (B1, FD-gated 3e-9) and `solve3d/stage_b.penalty_objective_and_grad` gradient assembly. B3 swaps the penalty term for the AL term `(1/2mu)[max(0, lambda + mu*g)^2 - lambda^2]` (gradient factor `max(0, lambda + mu*g)`), adds an outer loop (inner L-BFGS solve -> multiplier update -> restoration-shift update -> mu escalation), and the arbiter (`stage_a_phase2.ceiling_end_state_gate`) supplies the true peak for the shift and for is_shippable.

**Tech Stack:** Python, numpy, dolfinx (`heatr3d_d1_spike/env/bin/python`; `.venv312` lacks ufl). pytest. `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`.

**Spec:** `docs/superpowers/specs/2026-08-07-stage-b3-augmented-lagrangian-design.md` (APPROVED).

**Scope:** Tasks 1-5 are pure-logic + a light gradient gate (no heavy solve). Task 6 is the heavy AL solve — STOP before it for review. B4 (joint drive) is out of scope.

**Iron law:** the combined AL gradient is FD-gated before the solve; the TRUE arbiter peak (never KS) gates is_shippable; no threshold widening.

---

## File Structure
- **Create** `solve3d/stage_b3.py` — `al_gradient_factor`, `multiplier_update`, `restoration_shift`, `mu_escalation`, `honest_null_verdict` (fixed), `run_solve_al` (outer loop; heavy). ~250 lines.
- **Create** `solve3d/tests/test_stage_b3.py` — AL-gradient FD gate + pure-logic tests.
- **Reuse** `solve3d/stage_b.py` (penalty_objective_and_grad assembly), `solve3d/density_adjoint.py` (dks_peak_ds), `solve3d/stage_a_phase2.py` (ceiling_end_state_gate arbiter). No edits to the FD-gated B1 code.

---

## Task 1: AL multiplier + shift + mu pure-logic

**Files:** Create `solve3d/stage_b3.py`; Test `solve3d/tests/test_stage_b3.py`

- [ ] **Step 1: Write failing tests** (pure functions, no physics)

```python
# solve3d/tests/test_stage_b3.py
import numpy as np
from solve3d import stage_b3 as b3

def test_multiplier_update_inequality_kkt():
    # lambda_new = max(0, lambda + mu*g); stays >= 0, grows when violated (g>0), decays when satisfied (g<0)
    assert b3.multiplier_update(lam=0.0, mu=1e3, g=0.5) == 0.5e3
    assert b3.multiplier_update(lam=100.0, mu=1e3, g=-1.0) == 0.0     # clipped at 0
    assert b3.multiplier_update(lam=2000.0, mu=1e3, g=-1.0) == 1000.0

def test_restoration_shift_targets_true_peak():
    # T_target = ceiling - Delta, Delta EMA-damped from (true_arbiter - ks_solve)
    d = b3.restoration_shift(ceiling=250.0, true_peak=250.69, ks_peak=248.45,
                             prev_delta=0.0, ema=0.5)
    assert abs(d["delta"] - 0.5 * (250.69 - 248.45)) < 1e-9   # first EMA step from 0
    assert abs(d["t_target"] - (250.0 - d["delta"])) < 1e-9

def test_mu_escalation_when_violation_stalls():
    assert b3.mu_escalation(mu=1e3, viol_prev=2.0, viol_now=1.9, factor=5.0, shrink=0.5) == 5e3  # <50% drop -> grow
    assert b3.mu_escalation(mu=1e3, viol_prev=2.0, viol_now=0.5, factor=5.0, shrink=0.5) == 1e3  # good drop -> hold

def test_al_gradient_factor():
    # active branch factor = max(0, lambda + mu*g); zero when lambda+mu*g <= 0
    assert b3.al_gradient_factor(lam=0.0, mu=1e3, g=0.01) == 10.0
    assert b3.al_gradient_factor(lam=0.0, mu=1e3, g=-0.01) == 0.0

def test_honest_null_not_spurious_when_uniform_feasible():
    # a feasible uniform map (under ceiling) => NEVER honest-null, regardless of the shaped endpoint
    v = b3.honest_null_verdict(shaped_true_peak=250.69, uniform_true_peak=239.99, ceiling=250.0)
    assert v["verdict"] != "no_feasible_dopant_at_this_drive"
    v2 = b3.honest_null_verdict(shaped_true_peak=252.0, uniform_true_peak=251.0, ceiling=250.0)
    assert v2["verdict"] == "no_feasible_dopant_at_this_drive"   # only when even uniform is over
```

- [ ] **Step 2: Run, see fail.** `heatr3d_d1_spike/env/bin/python -m pytest solve3d/tests/test_stage_b3.py -x -q` → FAIL (no stage_b3).

- [ ] **Step 3: Implement the five pure functions** in `solve3d/stage_b3.py`: `multiplier_update` (max(0, lam+mu*g)), `restoration_shift` (Delta EMA from true-ks gap, t_target=ceiling-Delta), `mu_escalation` (grow by factor if violation drop < shrink fraction, else hold), `al_gradient_factor` (max(0, lam+mu*g)), `honest_null_verdict` (fires ONLY if uniform_true_peak > ceiling).

- [ ] **Step 4: Run, see pass.** Same command → all PASS.

- [ ] **Step 5: Commit**

```bash
git add solve3d/stage_b3.py solve3d/tests/test_stage_b3.py
git commit -m "feat(stage-b3): AL multiplier/shift/mu/honest-null pure logic (fixes B2 mislabel)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: AL combined-gradient FD gate

**Files:** Modify `solve3d/stage_b3.py` (add `al_objective_and_grad`); Test `solve3d/tests/test_stage_b3.py`

`al_objective_and_grad(case, v, lam, mu, t_target)` = J_shape + AL term; gradient = dJ_shape/ds + `al_gradient_factor(lam, mu, g)` * dks_peak_ds, with `g = KS_peak(v) - t_target`. Reuses stage_b.penalty_objective_and_grad's shape gradient and density_adjoint.dks_peak_ds.

- [ ] **Step 1: Write failing FD gate** (fixed lam, mu, t_target; hinge active)

```python
def test_al_combined_grad_matches_fd():
    from solve3d import stage_b3 as b3
    case = b3.build_al_coarse_case(lam=500.0, mu=1.0e4, t_target=247.8)  # active-hinge coarse case
    v = case.design_point()
    J, g = b3.al_objective_and_grad(case, v)
    assert np.all(np.isfinite(g))
    h = 1e-4
    for i in case.probe_indices():
        vp = v.copy(); vp[i] += h; vm = v.copy(); vm[i] -= h
        Jp, _ = b3.al_objective_and_grad(case, vp)
        Jm, _ = b3.al_objective_and_grad(case, vm)
        fd = (Jp - Jm) / (2 * h)
        assert abs(fd - g[i]) <= 1e-6 * max(1.0, abs(fd)) + 1e-9, (i, fd, g[i])
```

- [ ] **Step 2: Run, see fail.** → FAIL (no al_objective_and_grad).

- [ ] **Step 3: Implement `al_objective_and_grad`** — reuse the B2 shape gradient + B1 dks_peak_ds; apply the AL factor. `build_al_coarse_case`/`design_point`/`probe_indices` mirror stage_b.build_penalty_coarse_case.

- [ ] **Step 4: Run, see pass.** → PASS (combined AL gradient matches central FD; the density co-state is already gated, this confirms the AL composition).

- [ ] **Step 5: Commit**

```bash
git add solve3d/stage_b3.py solve3d/tests/test_stage_b3.py
git commit -m "feat(stage-b3): AL combined-gradient FD-gated (reuses B1 dks_peak_ds)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: the outer AL solve (heavy — STOP before running for review)

**Files:** Modify `solve3d/stage_b3.py` (add `run_solve_al`); Create `solve3d/STAGE_B3_REPORT.md`

- [ ] **Step 1: Implement `run_solve_al`** — the outer loop: initialize lam=0, mu=1e3, Delta=0, t_target=ceiling; repeat { inner L-BFGS min of `al_objective_and_grad` at (lam, mu, t_target) with frozen conventions; measure true arbiter peak via ceiling_end_state_gate; update Delta/t_target (restoration_shift), lam (multiplier_update), mu (mu_escalation) } until |true_peak - ceiling| < tol or max outer iters. Per-outer + per-eval checkpoint (resumable), detached under caffeinate -i. Refuse to start unless B1 launch_ok AND B2 fidelity agree (read the JSONs). is_shippable on the TRUE arbiter peak; honest_null_verdict with the uniform 239.99 feasibility. Log every (outer, lam, mu, Delta, t_target, ks_peak, true_peak).

- [ ] **Step 2: Pre-flight + slot check.** Confirm no heavy solve running (ps), load < 20, announce. (Compute convention.)

- [ ] **Step 3: STOP — report for review.** Report the AL-gradient FD-gate result (Task 2), the pure-logic test results (Task 1), and the run_solve_al design. Do NOT launch until reviewed. (Main session launches, as with B2.)

- [ ] **Step 4 (after go): launch detached, harvest.** On completion: true arbiter peak <= 250 (by construction), is_shippable, best shape; write `stage_b3_square.json` + `map_stage_b3_square.npz` + `STAGE_B3_REPORT.md`. If is_shippable, route (shaped map, 0.40x, chamber tag, rho_target) to the Studio is_sendable gate — the full loop.

- [ ] **Step 5: Commit** the result + report (after the solve).

---

## Self-Review

**Spec coverage:** AL formulation = Tasks 1-2 (factor + FD gate). Restoration shift = Task 1 `restoration_shift` + Task 3 outer loop. Multiplier/mu = Task 1. honest_null fix = Task 1 `honest_null_verdict`. By-construction true-peak = Task 3 outer loop + arbiter. Cross-engine routing = Task 3 Step 4. B1 gradient reuse (no re-derivation) = Task 2 consumes dks_peak_ds unchanged.

**Placeholder scan:** all test code complete; the outer-loop arithmetic in Task 3 reuses gated pieces (adjoint-development contract satisfied by Task 2's FD gate). Commands/env exact.

**Type consistency:** `multiplier_update`, `restoration_shift`, `mu_escalation`, `al_gradient_factor`, `honest_null_verdict` (Task 1) are consumed by `al_objective_and_grad` (Task 2) and `run_solve_al` (Task 3) with the same signatures.
