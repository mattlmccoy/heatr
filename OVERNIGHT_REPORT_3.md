# Overnight Report 3 (2026-08-01) - the full solve-workstream queue, completed

Every queued item ran to completion. Eleven commits on feat/pernode-twosided-tuning, local
only, nothing pushed. All figures were viewed personally before delivery. Three agents died
silently early in the night and were resumed with zero lost verified work.

## The one-paragraph verdict

Solving for the FGM (functionally graded material) map from the physics is now a working,
integrated, defensible method: it beats the actual historical masks 13/18 in the deployable
conductivity channel (17/18 with the permittivity channel, model-only pending a material
measurement), its claims survive grid transfer and blur robustness when the physical-length
filter is on, the cross reached IoU 0.9829 (grid 120) with a solved map plus an asymmetric
turntable dwell program, and the whole recipe is now a first-class HEATR mode (CLI + GUI).
The honest frontier: T_shape (best 0.644) and L_shape (0.666) resist every actuator tried;
the permittivity deployability question and the engine's inability to execute unequal dwells
are the two hardware-facing blockers.

## Per-item verdicts (chronological)

1. GUI P0 restoration (0d1f6a1): FGM form/presets regression (from recovery commit 8946538)
   restored + standard defaults fixed; verified live incl. a real job launch; includes the
   prior session's uncommitted results-tab perf work (hunks interleaved, both named).
2. Grid hold-out + rim robustness (1fb2f98): 3-D lane's skepticism VINDICATED - wins-over-
   uniform transfer 6/6 but the SOLVED class emptied at n=160; rim structure was grid-scale
   sculpture at ~1 cell. Drove the filter requirement into everything after.
3. Temporal power scheduling (436d939): helps ONLY the cross and there it is real structure
   (dose-matched control; 211 s OFF period; mechanism = selective RE-HEAT of the doped part,
   differential-cooling hypothesis refuted). Latent gradient bug found and fixed red-first.
4. Density-region objective (248ae6c): honest null - melt objective already delivers better
   density at the shape stop on 14/18; density lag is a scheduling problem, not an objective
   problem. Saturation-pathology guard held 18/18.
5. Multi-start (fd93e17): THE FILTER IS THE FIX (blur cost +153% -> +9.9%, square IoU 1.0000
   in-grid filter-only); budget-split multi-start costs depth; warm-start kept only where a
   strong historical mask exists.
6. Permittivity channel (890dc22): THE actuator confound - census 13/18 -> 17/18, rectangle
   "stall" was actuator-limited (J 97 -> 7). DEPLOYMENT GATE: rfam_eqs_coupled.py:290-292
   asserts real binder eps_r fixed at 20. Decide with a VNA dielectric sweep vs carbon-black
   loading. Until then eps = model-only everywhere (also flagged to the 3-D port lane).
7. Joint per-angle map re-solve (2f5af25): Matt's prediction CONFIRMED 2/4 - cross joint
   angle moved 30->45 (crossed IoU 0.80 deployable first time); star's fixed-map WORST angle
   became the joint BEST (sign flip, 3x reproducibility floor). T/L refuted (grading inert).
8. Continuous rotation (b1cb582): cross IoU 0.9866 via 90-deg INDEXING + solved 4-angle map;
   star 0.953 (rotation alone; map inert); square control confirmed annular-kernel physics;
   residual kernel anisotropy ranks all outcomes. 90-deg indexing beats continuous on 4/5 =
   the mechanically simpler mode wins. Engine finding: rotation-event bilinear remap loses
   5-19% energy at fast rotation (isolated by exact-permutation control).
9. TopOpt (b04e356): grid transfer achieved for the FIRST time (circle 0.9616 at held-out
   160), but the ablation credits the PHYSICAL 1.0 mm radius + grid-independent chi, NOT the
   Heaviside projection, which costs in-grid fidelity on 4/6 (partly a budget/optimizer
   artifact - a fair retest needs MMA or bigger budget). Production recipe: filter-only
   default, projection optional. FROZEN_CONVENTIONS_2D.md delivered to the 3-D port lane.
10. Asymmetric dwell (2cc1548): controls EXACT (cross rediscovered equal 90-deg dwells and
    zeroed the 45s; square exactly equal); cross reached in-grid SOLVED 0.9829 with a real
    20 s-cycle turntable program; T parked 73.5% at 90 deg = best T ever (0.644), honest
    partial; L collapsed to static 135 = rotational actuation exhausted. p(t) on top of
    dwells: +0.004 IoU - POWER SCHEDULING RETIRES (Matt's instinct confirmed). Machine-
    readable turntable programs emitted (out_dwell/*_turntable_*.json). DEPLOYMENT GAP:
    shipped engine cannot execute unequal dwells (rfam_eqs_coupled.py:2827-2836).
11. HEATR integration (4deff91): solve_fgm CLI + GUI mode, production recipe, map format
    contract-tested against the real injection path, verified live end to end.

## Judgment calls made on Matt's behalf (all reversible, all named)

- Split the first GUI commit's interleaved perf work rather than untangling hunks; named both.
- Treated the silent 3-agent death as an interruption; resumed rather than restarted; the
  resumed scheduling agent then found and fixed a latent gradient bug the dead one left.
- Enforced the physical-length filter on all post-robustness solves (fix the class).
- Ran the joint-angle and dwell studies conductivity-only as primary (deployables) with eps
  clearly labeled model-only.
- Demoted the Heaviside projection to optional after the ablation; did not spend budget on
  the MMA retest without Matt.
- One FD probe at 1.39e-05 vs the 1e-5 bar in the dwell pass: bisected to the read state,
  proceeded, named as a judgment call in that report.

## Open questions for Matt, ranked

1. EPS MATERIAL QUESTION (gates the 17/18 census and the rectangle fix): does 25 wt% carbon
   black move the composite's permittivity with loading, or is eps_r truly fixed ~20? A VNA
   dielectric sweep vs loading decides it. Highest-leverage physical measurement available.
2. Engine dwell support: small change to execute unequal dwell programs (needed for the real
   turntable controller and to re-verify the dwell arms on the shipped engine).
3. MMA + honest-budget retest of the projection (only path to printable near-binary maps
   that do not pay the in-grid tax).
4. T/L frontier: all single-part actuators exhausted (dopant, angle, rotation, dwell, power).
   Remaining ideas: auxiliary/sacrificial features (SRAF-class), multi-part loading, or state
   them as the method's boundary in the dissertation.
5. Bed-melt/part-growth metric still absent from every selection rule (visible in J's growth
   term but not gated); power-matched confirmation arm still on the backlog.
6. gt_logo remains unrun (cv2 missing from .venv312).

## Where everything lives

Reports (repo root): SOLVE_ROBUSTNESS_VALIDATION, TEMPORAL_SCHEDULING, DENSITY_OBJECTIVE_
LIBRARY, MULTISTART, EPS_CHANNEL, JOINT_ANGLE_MAP, CONTINUOUS_ROTATION, TOPOPT,
DWELL_SCHEDULE, FROZEN_CONVENTIONS_2D, SOLVE_MODE_USAGE, SOLVE_INTEGRATION_NOTES,
GUI_P0_RESTORATION_NOTES. Data/figures: fgm_solve_campaign/{out_*,figs_*}. Turntable
programs: fgm_solve_campaign/out_dwell/*_turntable_deliverable.json. The solve mode:
scripts/solve_fgm.py, GUI "FGM Solve (shape fidelity)". Cross-lane: conventions delivered
and acknowledged by the 3-D port lane (their spec commit 6847e0b).
