# FGM Creation Method — Discussion + Inverse-Design Research Brief

**How to use:** open a fresh Claude Code session in a **separate git worktree** off the geo-prewarp
repo (`git worktree add ../geo-prewarp-fgm-solve -b explore/fgm-inverse-design`), and paste the
block under [Prompt to paste](#prompt-to-paste). This is a **discussion-and-analysis-first** task,
not a build-first task. Matt wants to interrogate how we make FGM dopant maps and whether we can do
better with actual physics solving.

- **Repo / worktree:** a NEW worktree off `.../binderjet/code/geo-prewarp` (do not work on the main tree)
- **Model:** a strong reasoning model; this is adjoint / inverse-design / numerical-correctness work
  (computational-solver-engineer territory)
- **Python:** `./.venv312/bin/python` (heatr3d); trusted 2-D/2.5-D via `rfam_eqs_coupled.py`

---

## The question (Matt, 2026-07-30)
> "How do we create our FGMs? Right now we invert and then we optimize from there. But are we doing
> any actual solving? I feel like there needs to be some physics solving that can figure out the ideal
> dopant mask based on how the simulation densified. How confident are we in our process, results, and
> compensation methods? They seem fine for now, but can they get better?"

## What our current FGM creation actually is (start by verifying this, do not assume)
1. **Invert:** build an initial dopant/saturation map by a proportional-inverse of the part's own heating
   / T_phi90 proxy field (`heatr3d.make_fgm(magnitude, baseline, bpp)` and the 2-D `make_fgm` in
   `rfam_eqs_coupled.py`). This is a heuristic map, NOT a solve.
2. **Optimize from there:** iterate per-node (the pernode two-sided adaptive-gain law; see
   `run_pernode_square.py`, `PERNODE_RESULTS.md`) to reduce sigma_T. This is local search on a scalar
   objective, still not a principled inverse solve.
READ the actual code before characterizing it: is any step solving an equation for the ideal dopant, or
is it invert + local descent? State honestly what it is.

## The hard prior history to LEARN FROM (do not repeat blindly)
- **A physics-based dopant adjoint was already tried** (`analysis-3dfgm/ilt_adjoint.py`, dissertation
  `sections/fgm.tex` sec:fgm_adjoint). Verdict: modest in-model gain over the heuristic
  (train -8.8%/-15.9%) but ~300x costlier AND it FLIPPED to +15.9% WORSE on an n=64 hold-out. So a naive
  adjoint did NOT reliably beat the heuristic. Understand WHY before proposing a new one.
- **Over-critical sigma branch** (`memory: fgm-overcritical-sigma-branch`): sigma0=0.04 > sigma*~0.0302,
  so the part is on the shielding branch and the naive inverse map ANTI-correlates with the COMSOL/Jared
  ground truth. The per-node two-sided law was the fix. Any "ideal dopant" solve must respect this branch.
- **Tool trust:** HEATR 2-D/2.5-D is validated; heatr3d is not (grid>=200 melt-onset instability). Weight
  2-D/2.5-D as authoritative; FD-gate every gradient.
- **sigma_T is ONE metric** = std(T_part) in deg C (ui_rms*(Tbar-23) == std(T), verified). State read-state
  (heating-peak vs melt-onset at phi_bar=0.90).

## What to deliver (analysis first, then a small principled prototype IF warranted)
1. **HONEST CHARACTERIZATION:** what our FGM creation is (invert + local search), where it is heuristic,
   where it can fail, and how confident we should be in the current compensation results (2-D and 3-D).
2. **THE PRINCIPLED-SOLVE QUESTION:** is there a well-posed inverse-design that computes the ideal dopant
   sigma(x) from the coupled EQS -> thermal -> densification response — e.g. an adjoint of the uniformity
   (or final-density) objective w.r.t. the dopant field, on the TRUSTED 2-D/2.5-D engine, that reliably
   beats invert+local-search? Diagnose why the prior adjoint didn't win (cost, grid-convergence, the
   over-critical branch, the objective choice) and whether those are fixable.
3. **A MINIMAL FD-GATED PROTOTYPE (only if the analysis says it can help):** one clean adjoint/sensitivity
   of a well-chosen objective on a small trusted 2-D case, finite-difference-gated, compared head-to-head
   with the current invert+optimize map at MATCHED cost and on a HOLD-OUT (not just training sigma_T).
4. **HONEST VERDICT:** can the compensation get meaningfully better, at what cost, and with what
   confidence — or is invert+local-search already near the achievable frontier for the validated tool?

## Rules
- Separate worktree; do NOT edit the dissertation. TDD any solver logic; FD-gate every gradient before
  trusting it. Distinguish proven / computed / assumed. No em dashes; never the word "surrogate".
- Report should read as a DISCUSSION Matt can reason from, with the confidence assessment front and center.

---

## Prompt to paste

```text
Read FGM_CREATION_METHOD_BRIEF.md in the repo root first and follow it exactly. This is a
DISCUSSION-AND-ANALYSIS-FIRST task, not a build-first task. The question: is the project's FGM
(functionally graded material) dopant-map creation a principled inverse solve, or invert + local
search, and can a proper physics inverse design on the trusted 2-D/2.5-D engine reliably beat it?

Do the following, in order:

1. VERIFY, do not assume, what the current FGM creation actually is. Read the real code:
   make_fgm in heatr3d.py, the 2-D make_fgm in rfam_eqs_coupled.py, the per-node two-sided
   adaptive-gain law in run_pernode_square.py (check scripts/analysis/), and PERNODE_RESULTS.md.
   State honestly whether any step solves an equation for the ideal dopant or whether it is a
   proportional-inverse heuristic plus local descent on sigma_T. Cite file:line for every claim.

2. Study the prior failed adjoint before proposing anything: locate ilt_adjoint.py (under
   dissertation_materials/analysis-3dfgm/ or nearby) and any in-repo summary of sec:fgm_adjoint.
   Known verdict: modest in-model train gain (-8.8%/-15.9%) at ~300x cost, FLIPPED to +15.9%
   WORSE on an n=64 hold-out. Diagnose WHY it lost (cost, grid convergence, the over-critical
   sigma branch sigma0=0.04 > sigma*~0.0302, objective choice, optimization pathology) and
   separate fixable causes from structural ones.

3. Answer the principled-solve question on the TRUSTED 2-D/2.5-D engine only
   (rfam_eqs_coupled.py; python at ./.venv312/bin/python). heatr3d is NOT trusted
   (grid>=200 melt-onset instability). Consider objective choice (sigma_T at heating peak vs
   melt-onset at phi_bar=0.90, or final-density uniformity), the two-sided actuation the
   over-critical branch requires, regularization/bounds, and whether the coupled
   EQS -> thermal -> densification response is differentiable enough for a clean adjoint.
   Small numerical probes on the 2-D engine are allowed and encouraged; report exactly what ran.

4. DELIVERABLE: write FGM_INVERSE_DESIGN_ASSESSMENT.md in the worktree root, leading with the
   confidence assessment: (a) what the current method is and how confident we should be in the
   existing compensation results, 2-D and 3-D separately; (b) why the prior adjoint failed and
   which causes are fixable; (c) whether a principled inverse solve is likely to reliably beat
   invert+local-search at matched cost on a hold-out, with reasoning; (d) ONLY if the analysis
   says it can help: design (do NOT build) one minimal FD-gated adjoint/sensitivity prototype on
   a small trusted 2-D case, head-to-head vs the current map at MATCHED cost, evaluated on a
   HOLD-OUT, not training sigma_T; (e) honest verdict: can compensation get meaningfully better,
   at what cost and confidence, or is invert+local-search already near the achievable frontier
   for the validated tool?

Rules: do NOT edit the dissertation or push anything. Distinguish proven / computed / assumed
throughout. No em dashes. Never the word "surrogate". Expand every acronym on first use.
sigma_T = std(T_part) in deg C; always state the read-state (heating-peak vs melt-onset at
phi_bar=0.90) for any number quoted. End with a concise verdict summary plus the absolute path
of FGM_INVERSE_DESIGN_ASSESSMENT.md.
```
