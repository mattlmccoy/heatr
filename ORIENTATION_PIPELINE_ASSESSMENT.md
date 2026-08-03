# Orientation / Turntable Pipeline Assessment (for the shape-fidelity objective)

Date: 2026-07-31. Assessed against the requirement that the orientation actuator serve
the shape-fidelity objective J = sum over the whole domain of (phi - chi_part)^2, as
defined in the adjoint worktree report
(`.claude/worktrees/agent-a02efc1141ba69c58/SHAPE_LIBRARY_SOLVE_REPORT.md:25`).

## Verdict

**Good enough as the actuator, with two small glue upgrades and one real gap.**
The existing tooling already provides (1) a solver-native static-angle knob, (2) a
solver-native time-based turntable with arbitrary rotation schedules up to
near-continuous (15 degrees x 48 events proven in a stored run), and (3) working
per-angle sweep drivers on the trusted 2-D engine at the standard parameter set. Do
not build a new orientation pipeline. What is missing is only the objective, not the
actuator: nothing in the pipeline scores J, and only the final phi field is persisted,
so "J at the optimal stop time" cannot be read off a stored run. The one real physics
gap is that a functionally graded material (FGM) dopant saturation map does NOT
co-rotate with the part under the turntable, so combined turntable + graded-dopant
runs are physically wrong today (static-angle + graded-dopant runs are fine).

Upgrades needed, smallest first (Section 5):
1. J scorer on run outputs (proof of concept already written and verified, Section 4).
2. In-loop J(t) trace so the optimal stop is selectable per angle (small solver hook
   or a per-exposure re-run ladder; the ladder needs zero code).
3. Only if turntable + graded dopant is ever needed: rotate `sat_map` at rotation
   events (one remap call in an existing remap block).

Also two repair notes: `run_turntable_composite_ceiling.py` is import-broken (proven),
and the orientation optimizer's recommender treats the 250 C ceiling as a soft 10 %
score term and has recommended a 102 C violation (proven on a stored run).

Evidence labels used below: PROVEN = executed or read directly in this assessment;
COMPUTED = derived from stored artifacts; ASSUMED = stated belief, not verified.

## 1. Inventory: what exists

### 1.1 Solver-native capabilities (`rfam_eqs_coupled.py`, trusted 2-D / 2.5-D engine)

| Capability | Where | Status |
|---|---|---|
| Static angle | `geometry.part.rotation_deg`, re-applied per angle by sweep scripts (`scripts/analysis/run_L_orientation_sigmaT_sweep.py:26`) | PROVEN (smoke run, Section 2) |
| Time-based turntable | `rfam_eqs_coupled.py:2795-2837`: `turntable.rotation_deg` x `total_rotations`, evenly spaced by default, `rotation_interval_s` override; arbitrary (non-90-degree) angles | PROVEN via stored run (below) |
| Near-continuous schedule | `configs/diamond_tt_15deg_48rot_nearcont.yaml:96-99` (15 deg x 48 = 720 deg over the exposure) | COMPUTED (config read; a 15 deg x 48 L-shape variant ran to completion: stored `lshape_tt_20260304_12min/summary.json` reports `turntable_total_rotation_deg: 720.0`) |
| Physically correct rotation model | `rfam_eqs_coupled.py:1629-1726` `_build_rotated_part_mask`: electrodes fixed, part re-rasterized; `:2969-2977` rotates T, rho, phi (and crystallinity) with the part; the electro-quasi-static (EQS) field re-solved after each event (`:3001-3004`) | PROVEN (code read) |
| Orientation optimizer study | `rfam_eqs_coupled.py:5053-5231` `run_orientation_optimizer`: coarse angle x exposure grid, Pareto front, knee recommendation, +-10 deg refine pass, JSON/CSV/PNG reports | PROVEN (code read; stored run `outputs_eqs/runs/L_shape/orientation_optimizer/experimental/lshape_orient_20260304_6min`) |
| Optimizer objective | `rfam_eqs_coupled.py:4812-4831`: mean/min/std of relative density plus temperature and density-floor violations. `:4783-4795` knee score = 0.35 mean_rho + 0.35 min_rho + 0.20 (1-std_rho) + 0.10 (1-temp_violation). NOT sigma_T, NOT J | PROVEN (code read) |

### 1.2 Driver scripts (all loop the trusted 2-D engine)

| Script | What it does | State |
|---|---|---|
| `scripts/analysis/run_L_orientation_sigmaT_sweep.py` (95 lines) | Static-angle sweep 0-180 deg step 15; per-angle plain `run_sim` via the CLI; sigma_T = std(T_phi90 over part); merges into `sigmaT_vs_angle.json` | Working; outputs archived at `outputs_eqs/_archive/runs/L_shape/orientation_sigmaT/baseline/` (13 angles stored) |
| `scripts/analysis/run_L_orient_ceiling_sweep.py` (94 lines) | Same sweep but ceiling-respecting: fixed 550-step exposure, sigma_T on final T, reports maxT / phi-bar / energy residual / dT-clip per angle (`:26`, `:53-63`) | PROVEN working today (smoke run, Section 2) |
| `run_turntable_composite_ceiling.py` (280 lines) | Turntable-averaged Qrf composed with FGM masks, ceiling-respecting evaluation via the optimizer max_density snapshot. Note: this approximates rotation by angle-averaging a static Qrf stack injected through `qrf_file_npy`; it is not the true rotating transient | BROKEN: `:69` imports `run_turntable_composite_transient`, which was archived on 2026-06-12; `import run_turntable_composite_ceiling` raises ModuleNotFoundError (PROVEN by execution) |
| `run_diamond_turntable_ceiling_sweep.py` (200 lines) | Trajectory + drive-ladder diagnostic for the diamond turntable strategies; patches the archive path onto sys.path (`:40-41`) so its imports work | ASSUMED working (path patch read, not executed) |
| `launch_lshape_turntable.py` (218 lines) | Submits 4 L-shape turntable jobs to the graphical user interface server at `/api/run` (`:116-153`); requires the server to be running. The CLI + a `turntable:` config block is the equivalent headless path | Working only with the server up; not needed for scripted work |
| `backfill_orientation_reports.py` (418 lines) | Regenerates report figures for existing orientation runs | Not exercised here |

### 1.3 Configs and stored outputs

- `configs/diamond_tt_15deg_48rot_nearcont.yaml`: near-continuous turntable at the
  standard parameter set (grid 120, 27.12 MHz, 500 W at 2 % transfer efficiency,
  dt 0.5 s, 720 steps). PROVEN by read.
- `configs/_tmp_L_orient_singlebase.yaml`: the sweep base config, same standard
  parameter set. PROVEN by read.
- `configs/shape_L_shape_orientation_optimizer.yaml`, `shape_circle_6min_turntable.yaml`,
  `shape_H_turntable_4x.yaml`, plus four `rfam_*turntable*` configs: study-block configs.
- The `configs/vcross_*.yaml` family encodes only static `rotation_deg: 0.0`
  (grep across all twelve files); no rotation content there.
- Stored campaigns: `outputs_eqs/runs/{L_shape,square,circle,equilateral_triangle}/turntable/`,
  `outputs_eqs/runs/{L_shape,T_shape,square,star}/orientation_optimizer/`,
  `outputs_eqs/runs/diamond/turntable_composite/`, archived per-angle sweeps under
  `outputs_eqs/_archive/runs/L_shape/orientation_sigmaT{,_ceiling}/`, and
  `turntable_composite_ceiling_{L_shape,circle}.json` at repo root (5 strategies each:
  baseline, turntable without FGM, single static FGM, average-field composite,
  composite-of-masks).

### 1.4 Optimizer style

Everything is grid sweep plus one local refine pass. There is no gradient or smarter
search anywhere in the orientation tooling, which is appropriate: orientation is one
number (or a two-number rotation schedule), and each forward run is minutes.

## 2. Verification: it runs today

Command (exact):

```
./.venv312/bin/python scripts/analysis/run_L_orient_ceiling_sweep.py 45
```

Result (PROVEN, executed 2026-07-31): exit code 0, wall time 13 min 23 s (single
550-step run at grid 120 including full figure generation). Console record:

```
angle 45.0: sigma_T=24.73 C  maxT=315.7 C  phi_bar=0.737  resid=1.50%  dTclip=0
```

Outputs verified on disk: `outputs_eqs/runs/L_shape/orientation_sigmaT_ceiling/experimental/ang045p0/`
(summary.json, fields.npz, used_config.yaml, full figure set) and the merged
`sigmaT_vs_angle_ceiling.json` one level up. Health gates pass: energy-balance
residual 1.50 % of doped energy, dT-clip fraction 0.

Comparison to the stored reference
(`outputs_eqs/_archive/runs/L_shape/orientation_sigmaT_ceiling/experimental/sigmaT_vs_angle_ceiling.json`,
angle 45: sigma_T = 23.34 C, maxT = 250.0 C, phi_bar = 0.053): sigma_T agrees within
6 %, but maxT and phi_bar differ because the archived sweep used per-angle
250 C-crossing exposures (its own `note` field says so) while the current script fixes
`N_STEPS = 550` for all angles (`run_L_orient_ceiling_sweep.py:26`), and the working
tree carries uncommitted `rfam_eqs_coupled.py` modifications (branch
`feat/pernode-twosided-tuning`). Two consequences, both operational rather than
blocking: (1) the constant 550-step exposure is NOT ceiling-respecting at every angle
in the current tree (315.7 C at 45 deg), so ceiling-controlled comparisons need the
per-angle exposure ladder anyway; (2) an exact numerical reproduction of the archived
campaign was not attempted and is ASSUMED to require the committed solver state.

## 3. Fitness against the shape-fidelity requirements

### (a) Evaluate a shape at a static angle or rotation schedule under J: PARTIAL

- Actuation: MET. Static angle via `geometry.part.rotation_deg` (sweep scripts,
  `run_L_orientation_sigmaT_sweep.py:26`); rotation schedules via the `turntable`
  block (`rfam_eqs_coupled.py:2795-2837`). Both on the trusted engine.
- Scoring under J: MISSING as shipped. No file in the orientation tooling computes
  (phi - chi_part)^2; objectives are sigma_T (sweep scripts) or density Pareto
  (`rfam_eqs_coupled.py:4812-4831`). However J is directly computable from any run's
  `fields.npz` (`phi` and `part_mask` are both saved whole-domain,
  `rfam_eqs_coupled.py:4109-4110` block), PROVEN in Section 4.
- J at the OPTIMAL stop: MISSING. `fields.npz` stores only the final phi field. The
  in-run optimizer snapshots phi at mean-melt thresholds and at the last sub-ceiling
  step (`rfam_eqs_coupled.py:3310-3346`) but keeps them in memory (`opt_data["snapshots"]`,
  `:3561`); `save_outputs` persists only `T_phi90`, not snapshot phi fields (`:4106`).
  So the J-optimal stop is reachable today only through the Python interface
  (`run_sim` returns `opt_data`) or by an exposure ladder of separate runs.

### (b) Compose with a per-shape dopant map while rotating: PARTIAL

- Static angle + arbitrary saturation map: MET. `fgm_feedback.saturation_map_npz`
  (or `sat_map_npz_direct`, `rfam_eqs_coupled.py:323-345`) loads any grid-shaped map,
  and the part is rasterized at whatever `rotation_deg` the config sets. The map is a
  lab-frame array, so it composes correctly with any FIXED angle provided the map was
  generated at (or resampled to) that same angle.
- Under active turntable rotation: MISSING (physically wrong if attempted). Nothing
  blocks enabling `turntable` and `fgm_feedback` together (`:2796` only excludes the
  injected-Qrf mode, `fixed_qrf_mode`, defined at `:2284`). But at each rotation event
  the solver remaps T, rho, phi, and crystallinity into the rotated frame
  (`:2969-2985`) and then recomputes sigma with `_fgm_fb.sigma_at_mask`, which indexes
  the UNROTATED lab-frame `sat_map` (`:286`, `:443`). The dopant grading is printed
  into the part, so it must rotate with the part; it does not. Grep confirms `sat_map`
  is never remapped anywhere in the rotation-event block.
- The stored workaround, `run_turntable_composite_ceiling.py`, sidesteps this by
  angle-averaging part-frame Qrf stacks outside the solver, but it is import-broken
  today (Section 1.2) and is an approximation, not the rotating transient.

### (c) Trusted 2-D engine at the standard parameter set: MET

- All drivers call `rfam_eqs_coupled.py` (2-D / 2.5-D), never heatr3d.
- The sweep base config `configs/_tmp_L_orient_singlebase.yaml` and the near-continuous
  turntable config both use grid 120, 27.12 MHz, 860 V grounded, 500 W at 2 % transfer
  efficiency, dt 0.5 s, matching `HEATR_STANDARD_PARAMETERS.md:83` (grid <= 160, 120
  validated; never 240). The ceiling sweep additionally reports the standard health
  gates (energy residual, dT-clip fraction) per angle
  (`scripts/analysis/run_L_orient_ceiling_sweep.py:53-63`).

### Known weaknesses to respect (not blockers for the J workflow)

- The orientation optimizer's ceiling handling is soft: temperature violation is a
  10 % score term (`rfam_eqs_coupled.py:4791-4794`), and the stored L-shape run
  recommended 140 deg at 720 s with max T = 352.3 C, a 102 C violation
  (`outputs_eqs/runs/L_shape/orientation_optimizer/experimental/lshape_orient_20260304_6min/summary.json`,
  PROVEN by read). The ceiling sweep script was written specifically to correct this
  (`scripts/analysis/run_L_orient_ceiling_sweep.py:4-6`). For J-driven work this is
  moot if angle selection goes through the sweep scripts plus a J scorer rather than
  through the density knee.
- The stored 12-minute L-shape turntable run is over-driven (max T 325 C final, melt
  bleeding: 2905 melt cells vs 1079 part cells, J_final = 1749, COMPUTED in Section 4).
  Turntable runs for J must use ceiling- or J-controlled exposure, which the exposure
  ladder handles.

## 4. Proof-of-concept J scorer (implemented, verified)

`scripts/analysis/score_J_from_run.py` (new, 60 lines): computes J and melt
intersection-over-union from any run directory's `fields.npz`. Verification gates:
(1) hand-check, phi identical to the part mask gives J = 0 and IoU = 1, PASS;
(2) real-data run on three stored runs, PASS:

```
outputs_eqs/runs/L_shape/turntable/experimental/lshape_tt_20260304_12min
  J_final=1749.10  IoU_melt=0.3714  part_cells=1079  melt_cells=2905
outputs_eqs/_archive/runs/L_shape/orientation_sigmaT/baseline/ang000p0
  J_final=686.74   IoU_melt=0.4725  part_cells=871   melt_cells=1136
outputs_eqs/_archive/runs/L_shape/orientation_sigmaT/baseline/ang135p0
  J_final=400.30   IoU_melt=0.6625  part_cells=902   melt_cells=1211
```

The fresh smoke run also scores directly: ang045p0 (550-step exposure) gives
J_final = 332.90, IoU = 0.6548 (862 melt cells vs 902 part cells), PROVEN.

COMPUTED result worth noting: on the stored (fixed 6-minute exposure) L-shape sweep,
135 deg beats 0 deg under J by 42 % (400.3 vs 686.7), so the existing static-angle
sweep outputs are already J-rankable with zero re-simulation. These are end-of-exposure
J values, not optimal-stop J values; treat them as ranking evidence only.

## 5. Recommendation

**Keep the existing pipeline as the actuator. Do not rebuild.** Upgrades, smallest first:

1. **J scorer (done as proof of concept).** Adopt `scripts/analysis/score_J_from_run.py`
   or fold its 15 lines into the sweep scripts' per-angle record. Cost: minutes.
2. **Optimal-stop J, zero-code path.** For a candidate angle, run an exposure ladder
   (the sweep scripts already take `n_steps` style control; `run_L_orient_ceiling_sweep.py`
   fixes `N_STEPS` at `:26`, so parameterize it or copy the 94-line script) and take
   min J over the ladder. Legitimate because orientation and stop time are each one
   number.
3. **Optimal-stop J, small-hook path (better, still small).** Record J(t) in the time
   loop: chi_part is `part_mask`, phi is live, so one line per outer step appended to
   `hist` (near the existing per-step history appends), plus `argmin` in the summary.
   This makes every future run, turntable included, report its own J-optimal stop.
   This touches solver internals, so it belongs to the computational-solver engineer;
   specify it, do not self-implement.
4. **Only if combined turntable + graded dopant is needed:** remap `_fgm_fb.sat_map`
   with the same `_map_coords` call the solver already applies to T, rho, and phi at
   rotation events (`rfam_eqs_coupled.py:2969-2985`). One remap plus a regression test
   (rotating a graded circle 360 degrees must reproduce the static result). Until then,
   restrict dopant-composed runs to static angles, which requirement (b) mostly needs
   anyway since the adjoint work solves per-shape maps at fixed orientation.
5. **Repair or retire `run_turntable_composite_ceiling.py`:** either add the same
   archive-path insert used by `run_diamond_turntable_ceiling_sweep.py:40-41` or mark
   it superseded. Cost: two lines.

Not worth doing now: any smarter optimizer than sweep plus refine (orientation is
low-dimensional and runs are minutes), and any orientation work on heatr3d (the
trusted engine is the 2-D one, and all existing orientation tooling already targets it).

## Reproducibility record

- Smoke command: `./.venv312/bin/python scripts/analysis/run_L_orient_ceiling_sweep.py 45`
  from the repo root, using `.venv312`. Output run directory:
  `outputs_eqs/runs/L_shape/orientation_sigmaT_ceiling/experimental/ang045p0/`
  (config in `config_in.yaml`, resolved config in `used_config.yaml`).
- J scorer: `./.venv312/bin/python scripts/analysis/score_J_from_run.py <run_dir>`.
- New files created by this assessment: `ORIENTATION_PIPELINE_ASSESSMENT.md` (this
  file), `scripts/analysis/score_J_from_run.py`, and the smoke run directory above.
  Nothing committed; no solver internals modified.
