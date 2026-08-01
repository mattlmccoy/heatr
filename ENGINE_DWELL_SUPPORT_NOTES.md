# Unequal-dwell execution in the production 2-D engine

**Date:** 2026-08-01. **Scope:** close the gap named in `DWELL_SCHEDULE_REPORT.md`
Section 4 and Section 12 item 3. The production engine could not execute an unequal dwell
program, so the asymmetric-dwell deliverables were verified only against the part-frame
march. They can now be executed by `rfam_eqs_coupled.run_sim` itself, and the result is
below.

**Acronyms, expanded on first use.** RFAM = radio-frequency additive manufacturing.
FGM = functionally graded material (a spatially varying dopant saturation map).
EQS = electro-quasi-static, the low-frequency Maxwell approximation the two-dimensional
solver uses. IoU = intersection over union. J = the whole-domain shape objective
`sum over the domain of (phi - chi_part)^2`. eps_r = relative permittivity.
phi = melt fraction. rho = relative density. TDD = test-driven development.

**Evidence tags.** PROVEN = unit tested or reproduced bit for bit against a stored run.
COMPUTED = measured from a run in this pass. ASSUMED = a modelling choice.

Nothing was committed. No dissertation file was touched. `.claude/worktrees/` and
`fgm_solve_campaign/adjoint2d/` were READ ONLY; not one byte was written in either.

---

## 1. Verdict, in one table

COMPUTED. Three programs, each executed twice: once by the part-frame march
(`fgm_solve_campaign/adjoint2d/dwell_march.program_forward`) and once by the production
engine in the new program mode. Same shape, same dopant map, same 1500-step horizon at
dt 0.5 s, same ordered hold list, metrics read at each arm's own J minimum.

| arm | holds | positions | dwell | engine as SHIPPED, dJ | engine with eps_r co-rotated, dJ | dIoU (eps co-rotated) |
|---|---|---|---|---|---|---|
| cross deliverable | 150 | 0, 90, 180, 270 | equal | **-0.19 %** | **-0.19 %** | **+0.0000** |
| T_shape control, 4 x 90 | 150 | 0, 90, 180, 270 | equal | -8.46 % | **-0.46 %** | +0.0039 |
| **T_shape deliverable** | **225** | **0, 45, 90, 135, 180, 270** | **UNEQUAL** | **-21.80 %** | **-1.16 %** | **+0.0042** |

`dJ = 100 * (J_march - J_engine) / J_engine`. Negative means the engine reads a HIGHER J,
that is a worse melt, than the part-frame march.

**The headline.** The engine now executes an arbitrary ordered (angle, duration) program,
and on the genuinely unequal 225-hold T_shape deliverable it agrees with the part-frame
march to **-1.16 percent in J and +0.0042 in IoU** once the relative-permittivity field is
also re-rasterized at each event. **That is one order of magnitude outside the +-0.1 percent
class the equal-dwell 90-degree checks reached, and the reason is now measured, not guessed**
(Section 5). With the engine exactly as shipped the same program disagrees by 21.8 percent,
so the permittivity term is not a detail.

---

## 2. What changed, by file and line

### 2.1 `rfam_eqs_coupled.py`, four edits, all opt in

| lines | what |
|---|---|
| **1629-1699** | `_program_segments_from_obj`, normalizes one program container into an ordered list of holds. Accepts a bare list or the dwell campaign's deliverable JSON object. |
| **1701-1785** | `parse_turntable_program(tt_cfg, dt_s, n_steps, base_dir=None)`, the pure compiler from an ordered program to the engine's `(event_steps, deltas)` queue. No engine state touched; unit tested on its own. |
| **1788-1821** | `corotate_sat_map(sat, cumulative_deg, part_mask_rot, outside)`, turns the printed dopant with the part. Delegates to `scripts/analysis/orientation_map_rotation.rotate_sat_map`, the convention already proved against the real engine, so the two can never drift apart. |
| **2996-3040** | turntable setup: the `if tt_program_mode:` branch, ahead of the legacy-phases and fixed-step branches. Captures the base saturation map for co-rotation and refuses (loudly) to co-rotate a map whose maximum exceeds 1. |
| **3050** | **bug fix**: the legacy `phases` branch printed `len(_n_evts)` on an `int` and raised `TypeError` on entry, so the only branch that accepted arbitrary angles was unusable. `_n_evts` is already the count. |
| **3219-3229** | at a rotation event, in PROGRAM MODE ONLY, re-rotate `_FgmFeedback.sat_map` to the cumulative angle before conductivity is recomputed. |
| **3230-3243** | at a rotation event, in PROGRAM MODE ONLY and opt in, re-rasterize `eps_r` from the current rotated fill fraction. Default off. |

### 2.2 New files

* `test_turntable_program.py` (16 tests, all red first).
* `tests_fixtures/make_turntable_baseline.py` and `tests_fixtures/turntable_baseline_pre.npz`,
  the backward-compatibility ground truth, captured from the engine BEFORE any edit.
* `scripts/analysis/run_engine_dwell_program_gate.py`, the real-data gate.
* Results: `fgm_solve_campaign/out_dwell/{cross_deliverable_moves,T_shape_deliverable_moves,T_shape_deliverable_control90}_engine_program_gate.json`.

Read, not modified: everything under `fgm_solve_campaign/adjoint2d/`,
`scripts/analysis/turntable_glue.py`, `scripts/analysis/orientation_map_rotation.py`.

---

## 3. The schema

```yaml
turntable:
  enabled: true

  # EITHER an inline ordered list of holds ...
  program:
    - {angle_deg: 0.0,   duration_s: 2.0}
    - {angle_deg: 45.0,  duration_s: 0.5}
    - {angle_deg: 90.0,  duration_s: 7.5}
    - {angle_deg: 135.0, duration_s: 0.5}

  # ... OR the dwell campaign's machine-readable deliverable, read straight off disk
  program_json: fgm_solve_campaign/out_dwell/T_shape_turntable_deliverable.json
  program_json_key: moves        # default; use "reduced_program" for the
                                 # half-turn-reduced program in the same file

  corotate_dopant: true          # default TRUE in program mode; turn the printed
                                 # dopant map with the part at every event
  corotate_eps_geometry: false   # default FALSE = the engine exactly as shipped;
                                 # true re-rasterizes eps_r at every event
```

Accepted hold keys, so the deliverable JSON needs no translation:
`angle_deg` or `position_deg`; `duration_s` or `dwell_s`; optional `move_at_s`
(when absent the start times accumulate from the durations).

**Semantics, stated rather than implied.**

1. Angles are ABSOLUTE turntable positions, added to `geometry.part.rotation_deg`. The
   engine is driven with incremental deltas, so `parse_turntable_program` differences them.
   Deltas are LITERAL, not shortest-path: a 270 to 0 cycle wrap is emitted as -270, which
   keeps the cumulative angle exactly equal to the commanded absolute position instead of
   letting it drift up by 360 degrees per cycle.
2. A hold at the angle the part is already at costs no event. The cross deliverable's 150
   holds therefore produce 149 rotation events.
3. Two holds landing inside the SAME outer step are MERGED, their deltas summed. This is
   not cosmetic: the time loop pops at most one event per outer step
   (`if (it + 1) == tt_event_steps[0]`), so a duplicated sentinel would stall the queue on
   a past value and silently freeze every later rotation. Unit tested.
4. Holds past the horizon are dropped. When the program ends before the horizon the part
   stays where the last hold left it. No wrap, no repeat.
5. A move is instantaneous and free, the same limit the part-frame march carries.
6. Co-rotation re-rotates from the ORIGINAL map at the cumulative angle every time, never
   by composing incremental rotations, so bilinear blur does not accumulate on the design.
7. `corotate_dopant` refuses a saturation map whose maximum exceeds 1 and says why: the
   shared rotation helper clips to [0, 1], so a two-sided per-node map above 1 would be
   silently truncated. It raises instead.

---

## 4. Backward compatibility, proved as bit identity

PROVEN. A fixed-step turntable run (grid 60, 260 outer steps, six 90-degree rotations at
an 18 s interval, from `configs/diamond_tt_15deg_48rot_nearcont.yaml`) was captured from
the engine BEFORE the first edit into `tests_fixtures/turntable_baseline_pre.npz`, then
re-run after all edits.

| field | max abs difference against the pre-edit engine |
|---|---|
| final temperature | **0.0** |
| final relative density | **0.0** |
| final melt fraction | **0.0** |
| final part mask | identical |
| rotation event steps | identical |
| doped-energy history, all 260 steps | **0.0** |

Exactly zero, not "small". This is `test_turntable_program.py::test_fixed_step_turntable_run_is_bit_identical_to_the_pre_edit_engine`
and it re-runs the whole simulation each time, so it is a live guard rather than a stored
claim.

Structurally, every change is gated on `tt_program_mode`, which is False unless
`turntable.program` or `turntable.program_json` is present. The single edit outside that
guard is the `len()` bug fix at :3050, in a branch that raised `TypeError` unconditionally
before, so it could not have had a working caller to regress. A test now drives that branch
end to end.

---

## 5. Engine against march, and the attribution

### 5.1 The three arms

COMPUTED, full numbers.

| arm | source | J | IoU | stop s | events | energy residual |
|---|---|---|---|---|---|---|
| cross deliverable | march | 34.04 | 0.9829 | 471.5 | | 0.43 % |
| | engine, shipped | 34.10 | 0.9829 | 471.0 | 149 | 0.43 % |
| | engine, eps co-rotated | 34.10 | 0.9829 | 471.0 | 149 | 0.43 % |
| T_shape 4 x 90 control | march | 507.41 | 0.6145 | 595.0 | | 2.12 % |
| | engine, shipped | 554.33 | 0.5653 | 395.0 | 149 | 1.77 % |
| | engine, eps co-rotated | 509.75 | 0.6106 | 585.0 | 149 | 2.06 % |
| **T_shape deliverable** | march | 427.38 | 0.6184 | 642.0 | | 0.99 % |
| | engine, shipped | 546.51 | 0.5242 | 402.5 | 224 | 1.10 % |
| | **engine, eps co-rotated** | **432.39** | **0.6142** | **662.5** | 224 | 1.60 % |

Every arm passes the standing 5 percent energy gate. No arm's stop sits on the horizon.

The march's cross number, **J 34.04 / IoU 0.9829 / stop 471.5 s**, reproduces
`DWELL_SCHEDULE_REPORT.md` Section 6 to every printed digit from an independent driver, so
the march side of this comparison is the same object the dwell report scored.

### 5.2 What the gap is made of

Three arms, chosen so each isolates one term.

| term | measured as | size in J |
|---|---|---|
| **the dielectric ghost**: `eps_r` is rasterized once at startup (`rfam_eqs_coupled.py:2532`) and never rebuilt, so a rotated part sits inside a stationary permittivity outline of itself at 0 degrees | shipped minus eps-co-rotated, same arm | **0.00 pp** on the cross, **8.00 pp** on the T_shape at 90 degrees, **20.64 pp** on the T_shape at 45 degrees |
| **the rotation remap of temperature, density and melt fraction**, exact pixel permutation at multiples of 90 | cross, eps co-rotated | **-0.19 %** (absolute J difference 0.06 on J = 34) |
| the same remap, plus shape asymmetry, still at multiples of 90 | T_shape control90, eps co-rotated | **-0.46 %** |
| the same, with 45-degree moves where the remap is BILINEAR, not a permutation, over 224 events | T_shape deliverable, eps co-rotated | **-1.16 %** |

Read the first row first. **The dielectric ghost is zero for a part that is invariant under
the commanded rotation and dominant for one that is not.** The cross is four-fold symmetric,
so its un-rotated permittivity outline coincides with its rotated part at every multiple of
90 degrees and the two engine arms are bit-for-bit indistinguishable in J, IoU and stop
time. The T_shape is not symmetric, and the ghost costs 8.0 percentage points at 90 degrees
and 20.6 at 45 degrees. This is the second gap `scripts/analysis/turntable_glue.py`
documented as `corotate_eps` and never measured. **It is now measured, and it is the largest
single error term in rotating RFAM simulation on an asymmetric part.**

The remaining -0.46 percent to -1.16 percent is the field remap. The 0.70 percentage-point
increment from the 90-degree control to the 45-degree deliverable is the bilinear
interpolation the exact permutation avoids, over 224 events, and it is the right order for
`CONTINUOUS_ROTATION_REPORT.md` Section 6.1's measured 0.007 to 0.010 percent dose loss per
non-permutation event (224 events gives 1.6 to 2.2 percent of dose).

### 5.3 Did we hit the target?

**No, and the honest answer is that the +-0.1 percent class was never available for these
programs.** The equal-dwell checks reached it because they combined a four-fold symmetric
shape (no dielectric ghost) with 90-degree moves (no interpolation). Remove either and the
gap grows in a way we can now name and size. On the actual asymmetric deliverable the
best-configured engine agrees to **-1.16 percent in J and +0.0042 in IoU**, and the
disagreement is a real physics difference between the two models, not a bug in either:
the part-frame march does not remap fields at all, and the engine has to.

Which of the two is closer to a real turntable is NOT settled here. The engine's remap loss
is a numerical artefact the machine does not have; the march's assumption that a part can
switch heating patterns with no field transport is exactly true only in the part frame,
which is where the physics lives. ASSUMED, and named as such.

---

## 6. Behavioural differences found

1. **The dielectric ghost is a first-order error on asymmetric parts** (Section 5.2). Any
   past rotating-turntable result on a shape that is not invariant under its own rotation
   increment carries it. It is off by default here so no existing number moves silently, but
   it should be switched ON for any new asymmetric turntable study, and the affected past
   results should be re-scored rather than trusted.
2. **The engine's stop time is far more sensitive to the ghost than to anything else.** On
   the T_shape deliverable the shipped engine stops at 402.5 s and the eps-co-rotated engine
   at 662.5 s, against the march's 642.0 s. A 260 s difference in recommended exposure from a
   permittivity field nobody re-rasterized.
3. **The legacy `phases` branch works again** and is now covered by a test. It still spaces
   its events EQUALLY, so it is not a dwell actuator; program mode is.
4. **Two holds inside one outer step used to be unrepresentable and would have been
   catastrophic**, not merely inaccurate: a duplicated event sentinel freezes the whole
   rotation queue. The parser merges them. This matters for real programs, since the
   T_shape deliverable already asks for 0.5 s holds at a 0.5 s control step.
5. **The march's own quasi-static step, re-measured on the refined map.** The T_shape
   deliverable's time-resolved march reads J 427.38 / IoU 0.6184 against the report's
   quasi-static 422.11 / 0.6231, a +1.25 percent J penalty for the finite 20 s cycle. The
   dwell report's cycle sweep measured -0.82 percent at 20 s, but on the SOLVE-STAGE map,
   which is a different map; the two are consistent in magnitude and are not the same
   measurement.

---

## 7. Tests, red first

PROVEN. 16 tests in `test_turntable_program.py`, each observed failing before the code
existed (the first was an `ImportError` on `parse_turntable_program`, which is the intended
red).

* Parsing (7): event sentinels and deltas from an inline program; a non-zero first hold
  emitting a move; the same-step MERGE; horizon truncation; rejection of an empty program
  and of a negative duration; the deliverable JSON read end to end with the cumulative
  angles reconstructed against the file's own `position_deg` list; selection of the reduced
  program.
* Execution order through the REAL engine (2): an instrumented angle-against-time trace
  from a four-hold program, asserting both the ORDER of the commanded angles and the outer
  step indices the moves land on; and the hold-past-the-end behaviour.
* Co-rotation (4): the 90-degree EXACT PIXEL PERMUTATION against `np.rot90(k=-1)`; exact
  agreement with the shared `rotate_sat_map` helper at a general angle; the map actually
  co-rotating inside a real engine run, checked against the rotated part mask the engine
  itself built; the off switch; and the SHIPPED fixed-step mode still leaving the map in
  the lab frame, which is the defaults-preserved check.
* Backward compatibility (1): the bit-identity regression of Section 4.
* Legacy branch (1): the `phases` config running to completion with the right angles.

Command: `./.venv312/bin/python -m pytest test_turntable_program.py -q` -> **16 passed** in 21 s.

**Pre-existing suite, the ones that actually drive this engine.**
`pytest test_turntable_glue.py test_orientation_map_rotation.py test_energy_balance.py`
-> **16 passed** in 5530 s (1 h 32 m; that cost is pre-existing, dominated by
`test_energy_balance.py::test_c_full_run_residual`, a full canonical run). Those three
files cover the rotation-event remap, the map-rotation convention against the real engine,
and the energy balance, so they are the ones a turntable edit could plausibly break.

---

## 8. Honest limits

1. **Two shapes.** The gate ran on the cross and the T_shape. The square and the L_shape
   were not run. The L_shape's deliverable is a single static hold, which program mode
   expresses as zero rotation events, so it is not a test of anything here.
2. **Grid 120, two dimensions, conductivity channel only**, exactly the dwell campaign's
   scope. `SOLVE_ROBUSTNESS_VALIDATION.md`'s warning that grid-120 fidelity does not
   transfer to grid 160 applies unchanged.
3. **The -1.16 percent residual is attributed, not eliminated.** The attribution rests on
   the three-arm bisect of Section 5.2 and on `CONTINUOUS_ROTATION_REPORT.md`'s
   independently measured per-event dose loss. It was not proved by refining the remap.
4. **`corotate_eps_geometry` is new and is gated by nothing but this pass's runs.** It has
   a unit test for the code path but no separate physics validation, and it changes the
   post-event EQS solve. It defaults OFF for that reason.
5. **Move duration and mechanical settling are still not modelled.** The T_shape program
   asks for 224 moves in 750 s.
6. **No dose matching.** Nothing in Section 5 is at equal delivered energy.
7. **The saturation-map co-rotation clips to [0, 1]** because the proved helper does. Maps
   above 1 raise rather than clip, so no two-sided per-node map can be co-rotated yet.
8. **Model, not hardware.**

---

## 9. The single most valuable next layer

**Re-score the asymmetric rotating results with `corotate_eps_geometry: true`, and decide
whether it becomes the default.** Section 5.2 shows it is worth 8 to 21 percentage points of
J on an asymmetric part, which is larger than every actuator effect the dwell campaign
measured. Until that decision is made, every rotating-turntable number on a shape that is
not invariant under its own rotation increment is carrying an uncontrolled error of that
size. The change is one config key and the runs are about 450 s each at grid 120.

Second: extend the program schema to a per-segment POWER level, which is the one field the
deliverable JSON already carries (`rf_program`) and the engine cannot yet vary in time.

---

## 10. Artifacts

Repository root:
`/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp`

* `rfam_eqs_coupled.py` lines 1629-1821, 2996-3040, 3050, 3219-3243
* `test_turntable_program.py`
* `tests_fixtures/make_turntable_baseline.py`, `tests_fixtures/turntable_baseline_pre.npz`
* `scripts/analysis/run_engine_dwell_program_gate.py`
* `fgm_solve_campaign/out_dwell/cross_deliverable_moves_engine_program_gate.json`
* `fgm_solve_campaign/out_dwell/T_shape_deliverable_moves_engine_program_gate.json`
* `fgm_solve_campaign/out_dwell/T_shape_deliverable_control90_engine_program_gate.json`

Wall time: three gate runs at 877, 995 and 886 s of process time, single-thread pinned
through `fgm_solve_campaign/env1.sh`; the test file runs in 21 s.
