# heatr3d Re-Validation Brief — Can We Trust Its Outputs?

**How to use:** open a fresh Claude Code session (heatr-simulation-engineer territory; a worktree is
optional since most work is running sims and writing reports, not editing solver code) and paste the
block under [Prompt to paste](#prompt-to-paste). This is a VALIDATION campaign, not a feature build.

- **Context (Matt, 2026-07-30):** "We really don't trust heatr3d as much as HEATR 2/2.5D. heatr3d
  probably needs to be looked at again to make sure we can really trust its outputs."
- **Trust baseline:** HEATR 2-D/2.5-D (`rfam_eqs_coupled.py`) is COMSOL- and FLIR-anchored and is
  authoritative. heatr3d is NOT validated to that standard.
- **Python:** `./.venv312/bin/python` (geo-prewarp repo root)
- **Key sources:** `FGM_INVERSE_DESIGN_ASSESSMENT.md` (Sec 3.3, 9),
  `.../dissertation_materials/analysis-3dfgm/PREWARP_HANDOFF.md` (lines ~70-95),
  `.../dissertation_materials/analysis-3dfgm/notes-ilt-adjoint.md` (~48-113),
  `out_heatr3d_audit_dumbbell_cone/AUDIT_REPORT.md`, memory note `tool-trust-and-sigmaT-metric-split`.

## Known trust deficits (verified, cited — do not re-derive, do re-check paths)
1. **Mesh non-convergence under the shipped binary (one-cell) material boundary.** Uniform-baseline
   sigma_T falls 11.87 -> 2.88 C from n=32 to 64 with no plateau; Q_rf max/mean diverges roughly
   linearly in n (EQS corner singularity at the material boundary). Every published 3-D FGM number
   is single-grid n=48 under this boundary.
2. **Ranking sign-flips with grid under the binary boundary.** Heuristic map: -46.2% (n=48) ->
   +48.1% (n=64). Adjoint n32c0: -50.9% -> +15.9%. At smoothed width w=1e-03 rankings are
   grid-stable (Spearman 0.929, no sign changes) — but w=1e-03 is itself an assumption (38% of a
   10 mm part becomes "skin"; 12% of dose removed). Rankings at resolved width are stable;
   absolute values are not portable.
3. **Melt-onset instability at grid >= 200** (2026-07-29 finding, fgm-overcritical-sigma-branch).
4. **Partial failures in its own audit** (`out_heatr3d_audit_dumbbell_cone/AUDIT_REPORT.md`).
5. **Melt-onset read-state silent fallback:** if phi_bar never crosses 0.90 within the horizon the
   metric silently reads the final step instead (confirmed in the 2-D engine probe; UNCHECKED in
   heatr3d — verify whether the same fallback exists there).
6. **Stale run-state records:** the A6 `n48c0` continuation died 2026-07-26 (~16% under-trained vs
   best hold-out) while the handoff still says "STILL RUNNING."

## The campaign, in order of leverage
### P0 — the 8-run edge-width probe (~1 h) [BLOCKS honest phrasing of every 3-D FGM percentage]
Spec from `PREWARP_HANDOFF.md:78-83`, run exactly as specified:
- Arms: uniform AND the chapter's FGM map, for one SHARP shape (cone or cube) and one SMOOTH shape
  (sphere). n=48. edge width w=0 (shipped binary) vs w=1e-03. 2 maps x 2 shapes x 2 widths = 8 runs.
- Pre-registered predictions (state pass/fail against each): absolute sigma_T rises ~2.7x at
  w=1e-03; "% vs uniform" degrades sharply and may change sign; sharp corners move most, sphere
  least; rankings survive better than percentages.
- Read state: melt-onset at phi_bar=0.90; report heating-peak too. sigma_T = std(T_part) in deg C.
  If phi_bar=0.90 is never reached, SAY SO — do not let the silent fallback stand in for the metric.
- Deliverable: a table (shape x map x width -> sigma_T at both read states, % vs uniform), a
  pass/fail against each prediction, and the verdict: does the sign flip generalize off the slab?

### P0b — melt-onset fallback check in heatr3d (cheap, correctness-class)
Read the heatr3d read-state code: does it silently fall back to the final step when phi_bar never
crosses 0.90? If yes, make it loudly reported (flag in output, not changed behavior), and list which
existing published runs actually hit the fallback.

### P1 — cross-engine agreement check (the direct trust verdict)
One matched configuration (square slab; match physics, drive mode, material constants, read state)
run in heatr3d AND trusted 2.5-D. Compare: Q_rf spatial pattern (correlation), T field, sigma_T at
both read states, absorbed power. Pin the config completely (grid, dt, boundary width, drive).
Expected: patterns should correlate strongly; absolute sigma_T will differ (2-D section vs 3-D
volume) — the question is whether the DISAGREEMENT is explainable by dimensionality or signals a
heatr3d defect.

### P2 — converged-width mesh ladder on production quantities
n = {32, 48, 64} at w >= 1.5h for the production uniform and graded runs (not the adjoint arms):
which reported numbers are grid-stable? Rankings vs absolutes, per PREWARP_HANDOFF guidance
("report the width like dt and n; enforce w >= 1.5h; prefer rankings").

### P3 — external anchor re-check
COMSOL/Jared corner-X heating pattern vs heatr3d at converged width (pattern-level, not absolute).

## Rules
- Do NOT edit the dissertation. Do NOT change heatr3d solver behavior in this campaign; probes and
  instrumentation only (loud flags allowed, silent behavior changes not).
- Distinguish proven / computed / assumed. State the read-state for every number. No em dashes;
  never the word "surrogate"; expand acronyms on first use.
- Deliverable per phase: a short markdown report with the run table, config pins, and an honest
  verdict sentence. Lead with the verdict.
- Do not start P1+ until P0/P0b results are in; P0 may change what P1 should pin.

---

## Prompt to paste

```text
Read HEATR3D_REVALIDATION_BRIEF.md in the geo-prewarp repo root and follow it exactly. Goal:
determine whether heatr3d (the 3-D RF additive manufacturing solver) outputs can be trusted, and
with what qualifiers. HEATR 2-D/2.5-D (rfam_eqs_coupled.py) is the trusted reference.

Execute in order: P0 (the 8-run edge-width probe, exactly as specified in the brief, with the
pre-registered predictions scored pass/fail), then P0b (the melt-onset fallback check), then STOP
and report before P1. Python: ./.venv312/bin/python. sigma_T = std(T_part) in deg C; state the
read state (heating-peak vs melt-onset at phi_bar=0.90) for every number; if melt-onset is never
reached in a run, report that explicitly rather than letting any silent final-step fallback stand.

Rules: no dissertation edits; no silent solver-behavior changes (probes and loud instrumentation
only); distinguish proven / computed / assumed; no em dashes; never the word "surrogate"; expand
acronyms on first use. Deliverable: a markdown report per phase with the run table, full config
pins (grid, dt, edge width, drive mode), and a verdict-first summary. End with: do the published
3-D FGM percentages survive at a converged boundary width, and what qualifier must every quoted
3-D number carry?
```
