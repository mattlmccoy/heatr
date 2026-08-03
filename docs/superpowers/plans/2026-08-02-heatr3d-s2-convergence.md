# heatr3d S2: Convergence Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Published, measured convergence bands for corrected-default heatr3d (masked Q, enthalpy) that every downstream number quotes; the electrode-gauge decision; the densify=True coupled march (re-registered, exercises sigma_density_coeff for the first time); and mechanism explanations for the two known anomalies (L-shape outlier, cylinder null). S2 PASSES iff the pre-registered convergence criteria hold; anything that fails is reported failed.

**Architecture:** a campaign harness `heatr3d_s2/` (same isolation pattern as heatr3d_s4_flir/) driving the UNCHANGED heatr3d.py; no solver edits except a flag-gated electrode-gauge option if Task 2 motivates one (bit-identical when off, default flip only with Matt's sign-off - the EQS-02 precedent). Analysis code is TDD'd; the campaign itself is gated by pre-registered band rules and JSON artifacts.

**Tech Stack:** heatr3d.py (corrected defaults), numpy; OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1; runs within certified ceilings (n<=96 full physics, n<=128 EQS-only; NEVER >=200 for melt-state work - known instability).

**Authorship note:** same as Phases A-C - the executing agent knows the codebase; tasks specify files, red-first tests for harness logic, and numeric gates; campaign work uses pre-registered bands as the verification gate.

---

## Ground rules

- Pre-registration before any campaign run: shapes, grids, quantities, band rules, gauge arms, PASS criteria - JSON committed first.
- Quantity hierarchy per the recorded objective: SHAPE METRICS (melt-region IoU vs nominal at phi>=0.8/0.9, out-of-part melt fraction, front position) are the verdict-carrying convergence quantities; t90 co-primary; sigma_T reported as a diagnostic with its band but never verdict-carrying.
- Dual read states (heating-peak + melt-onset) on every run per the settled discipline.
- Band rule stated BEFORE numbers exist. Convergence claims need >= 3 grids per quantity; report the observed order; no Richardson extrapolation presented as truth - bands are measured spreads.
- No threshold widening; JSON-quoted reporting; atomic commits of heatr3d_s2/ + this plan only; no push; dissertation_materials READ-ONLY.
- The Phase A cross-family circle t90 offset is OUT OF SCOPE (S3's COMSOL anchor adjudicates it) - state this in the report.

## File map

- Create: `heatr3d_s2/{README.md,harness.py,gauge.py,bands.py,mechanisms.py,run_campaign.py}`, `heatr3d_s2/tests/`, `heatr3d_s2/results/*.json`, `heatr3d_s2/S2_GATE_REPORT.md`
- Modify (only if Task 2 motivates, flag-gated): `heatr3d.py` electrode-gauge option + `test_heatr3d_s1.py` bit-identity regression
- Reference: docs/superpowers/specs/2026-07-30-heatr3d-graduation-design.md (S2 definition), heatr3d_eqs02_rerank/RERANK_REPORT.md (L-shape outlier, cylinder null), heatr3d_s4_flir/ (march conventions for Task 4), solve3d/results/phase_a_shape_gate.json (the cross-family bands S2's same-engine bands will be compared against)

---

### Task 0: Pre-registration

- [x] `heatr3d_s2/results/s2_preregistration.json`, committed before any run: shapes = extruded circle, extruded square (the Phase A anchors), L-shape extrusion (the re-rank outlier); grids = 48/64/80/96 full physics (and 96/112/128 EQS-only for the EQS-only convergence check); quantities with hierarchy as in the ground rules; band rule (e.g. max successive-pair relative change over the finest pair, x1.5 - state and freeze it); gauge arms (Task 2); PASS criteria: per quantity, monotone or bounded oscillatory approach with the finest-pair spread below the frozen threshold per quantity class (state numbers now); densify-march registration (Task 4: case, coefficients incl. 0 control, validity bound |a| < 0.0044/K carried from S4, gates). Commit.

### Task 1: Harness + band machinery (TDD)

- [x] Failing tests: band computation on synthetic convergence sequences (monotone, oscillatory, diverging - diverging must FAIL loudly, never produce a band); gauge-switch plumbing returns bit-identical fields when the flag is off; dual-read extraction matches the S1 test conventions.
- [x] Implement `harness.py`/`bands.py`. Green. Commit. **19 passed (10 bands, 5 gauge, 4 harness).** The synthetic-sequence tests caught a real methodological bug: fitting log|successive change| vs log h inflates the observed order on a non-uniform ladder (read 2.98 on an exactly 2nd-order sequence); fitting the VALUES to q_inf + C h^p instead recovers p exactly.

### Task 2: Electrode-gauge decision ((n-1)h question)

- [ ] Failing test for `gauge.py`: the two gauge conventions (electrode spacing = n*h vs (n-1)*h at fixed physical chamber) produce measurably different chamber fields at coarse n and the harness extracts the grid-invariance observable (total absorbed power at fixed geometry and drive) for both.
- [ ] Run the circle at 3 grids under BOTH gauges. Decision rule (pre-registered): the gauge whose physical observable is grid-invariant (smallest drift across grids) is correct. Record the decision + numbers in `results/gauge_decision.json`.
- [ ] If the winner is not the current convention: implement flag-gated, bit-identical-off, present to Matt for the default flip (EQS-02 precedent) - do NOT flip the default yourself. Commit.

### Task 3: The convergence campaign

- [ ] Run the pre-registered grid ladder x 3 shapes, corrected defaults, coupling off, densify off (the baseline physics), dual read states. Emit per-run JSONs + `results/convergence_bands.json` with observed order and the frozen-rule bands per quantity.
- [ ] Verdict per quantity per shape against the pre-registered PASS criteria. Compare the same-engine bands against Phase A's cross-family bands (phase_a_shape_gate.json self-spreads) - consistency statement, not a gate. Commit.

### Task 4: densify=True coupled march (re-registered - Matt's assignment)

- [ ] Per the Task-0 registration: the S4-convention march with densify=True, exercising sigma_density_coeff for the FIRST time (it has been provably inert in every prior study). Arms: coefficient 0 control (must reproduce the densify-only march bit-for-bit at the tolerance class), plus the registered exploratory +/- values within validity bounds. Standing gates (energy audit, clamp, CFL) on every arm.
- [ ] Two questions, answered with numbers: (a) does density coupling move late-time surface topology in the FLIR-observed direction (S4 mechanism 3)? (b) do the Task-3 convergence bands hold with densification on (one shape, two grids - a spot check, labeled as such)? Emit `results/densify_coupled.json`. Commit.

### Task 5: Mechanism checks (analysis, not vibes)

- [ ] L-shape outlier: why does the corrected default degrade the L-shape most in the re-rank? Test the reentrant-corner field-concentration hypothesis quantitatively (corner-region Q share vs grid; does it converge or is it a singularity artifact?). Numbers + a one-paragraph mechanism statement in the report.
- [ ] Cylinder null mechanism: why is the corrected cylinder field already near-uniform (skin-depth-vs-radius argument, computed, vs the measured field flatness) - and reconcile with Phase C's result that a SOLVED map still finds 10.67%: state precisely what the inversion rule could not see that the solve could (the rim structure / bed-melt trade). Commit.

### Task 6: S2 gate report

- [ ] `heatr3d_s2/S2_GATE_REPORT.md`: verdict table per quantity/shape from the JSONs; gauge decision; densify-march findings; mechanism statements; revised ceilings if measured; honest deviations; not-covered list (COMSOL/2.5-D anchors = S3; the cross-family t90 offset = S3; no physical data). S2 verdict: PASS / PARTIAL / FAIL per the pre-registered criteria, never negotiated after the fact. Commit. Present to Matt with the canonical-sync question (heatr3d.py re-sync to dissertation_materials is gated on his sign-off).

## Out of scope

COMSOL 3-D anchor, 2.5-D extrusion anchor, densification-law literature consistency (all S3); any default flip without sign-off; Studio badge changes; dissertation edits; solve3d changes.
