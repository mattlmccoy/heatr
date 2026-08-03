# Overnight Queue (2026-08-01) — Matt asleep, standing authorization to proceed

Authorization: Matt 2026-08-01, "work on all of those through the night... use your judgement
after knowing what my thoughts would be." Local commits allowed (never push). No dissertation
edits. Lanes stay separate (solve lane only; heatr3d tool lane untouched).

## Judgment guardrails (Matt's known preferences)
- View every figure personally before treating it as deliverable. Honest negatives stated with
  numbers; no silent cuts; state read/stop conventions; no em dashes; never "surrogate".
- Densification/oversinter checks: never judge by end-of-horizon rho; report stop-time window
  metrics and final mean rho; watch dose gaming.
- Quote census carefully (three baselines); dose-match limit stated.
- Commit deliverables as they verify (small conventional commits, explicit paths only —
  beware pre-staged files). GUI commit gated on browser verification evidence.

## Active (relaunched ~23:00 after silent mass-death of prior three)
A. Temporal scheduling (resume; worktree agent). On completion: commit canonical adjoint2d
   from worktree via snapshot-into-main refresh of fgm_solve_campaign/ + reports/figs.
B. Grid hold-out + rim robustness rerun (main tree). Commit report on completion.
C. GUI P0 restoration (resume). On verified completion: commit (split perf work from
   restoration if hunks separate; else one commit naming both).

## Solver batch (launch sequentially after A frees the module; use refreshed
## fgm_solve_campaign/adjoint2d in MAIN tree as the working copy from here on)
1. Density-region objective, FD-gated; 18-shape library re-run scored under BOTH objectives.
   Pin read state before saturation (lattice-carving pathology precedent).
2. Multi-start / warm-start (start from best historical mask + 2 seeds; fixes rectangle stall).
3. Permittivity channel dJ/d-eps_r (unpin via gated flag at rfam_eqs_coupled.py:342,
   defaults preserved); re-score census losses.
4. Joint per-angle map re-solve on T/L/cross/star (warm-start from rotated 0-deg map,
   ~15 solves/angle); test Matt's prediction that the joint angle differs from sweep angle.
5. Continuous-rotation averaged-kernel solve (two-level: solve vs rotationally-averaged
   kernel in part frame; verify winner on true rotating forward WITH sat_map co-rotation fix).
6. Topology-optimization prototype (filter + Heaviside projection + MMA + continuation);
   benchmark = library re-run on the movers.

## GUI track (after P0 commit)
7. Solve integration into HEATR 2-D (config-driven fgm_method: solve; production 4-bpp
   output; warm-start default). Then P1 promotion pass per memory item 5-GUI-P1 (three-intent
   chooser, per-mode/per-shape correct parameters, presets, dual-read cards, drive guidance,
   grid warning). Promote, never remove.

## Morning deliverable
OVERNIGHT_REPORT_3.md: verdict-first summary of everything completed, figures list,
commits made, failures/stalls honestly, and the decisions I made on Matt's behalf with
reasoning. Watchdog: periodic liveness checks (file mtimes + processes); relaunch dead
agents resume-aware; never assume a missing notification means still-running.
