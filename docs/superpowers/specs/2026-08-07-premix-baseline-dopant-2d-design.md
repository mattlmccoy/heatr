# Design: Continuous premix baseline dopant in the 2-D lane + 0→15 wt% study

**Date:** 2026-08-07
**Author:** Claude (geo-prewarp 2-D lane), on a task handed over from the peer "fable 5 STL→FGM" session
**Status:** DRAFT — awaiting Matt's spec review before implementation
**Approved design decisions (this session):**
- **A. wt% → material law:** parameterize premix in `premix_frac` (0→1, fraction-of-doped) exactly like `heatr3d`; wt% is a labeled `[ASSUMED]` secondary display axis. (Matt, 2026-08-07)
- **B. Study scope:** re-solve the printed map on this lane's cheap bit-identical 2-D adjoint, one canonical shape, ~5 premix levels. (Matt, 2026-08-07)
- **Shape:** `jared_exp1_40mm` (validated Allison geometry). (Matt, 2026-08-07)
- **Gate:** written spec → Matt review → TDD implementation. (Matt, 2026-08-07)

---

## 1. Problem and scope correction

The peer task framed premix as a greenfield feature ("today the bed starts virgin; Matt wants a
continuous premix baseline 0→15 wt%"). **Recon shows premix is already implemented and tested in the
3-D solver, and absent from the 2-D model.** This lane owns the trusted 2-D solve (memory:
two-workstream-split), so the real, non-duplicative work is:

1. **Port** heatr3d's proven premix semantics into the 2-D coupled model so 2-D and 3-D share one
   convention and one honest knob.
2. **Study** how the optimal dopant map, RF absorption, and drive-to-ceiling evolve as premix ramps,
   on the trusted 2-D solver, cross-checked once against the existing 3-D premix.

### Evidence (code)
- `heatr3d.py:577` `build_gamma(..., premix_frac=0.0, premix_budget="floor_added"|"budget_fixed")` —
  complete, tested (`test_heatr3d_s1.py:917`). `premix_frac=0` is bit-for-bit the original path.
  - `floor_added` (primary): masked = `sigma_premix + full jetted increment`; premix boosts total dopant.
  - `budget_fixed` (uniformity sweep): jet brings premix up to `sigma_doped`; total ≈ const.
- `rfam_eqs_coupled.py` — **no premix.** Material law is a linear virgin↔doped blend
  `sigma = sigma_v + sat·fill·(sigma_d0 − sigma_v)` (docstring `:331`, live assembly `:1447-1461`
  static builder and `:2581-2908` main coupled/feedback builder). Bed outside the part stays virgin.
- Material endpoints come from `mats["virgin"|"doped"]` keys `sigma_s_per_m`, `eps_r`
  (`rfam_eqs_coupled.py:1447-1450`, `:2581-2584`). "25 wt%" (`:333`) is a **docstring label** for the
  doped endpoint, not a config value.

### Evidence (empirical — RFAM Paper Draft Ver 1.3.pdf, previous-work/)
- Real premixed powders were prepared and their conductivity measured. Table 1: Nylon-Graphite at
  **15/23/26/28/30/35 wt.%**; Nylon-Ink carbon **8.2/21/30 wt.%**; printed sample ≈ **17 wt.% carbon**.
- Virgin nylon σ ≈ **10⁻¹³ S/m** (p.128); graphite 10⁴–10⁵ S/m (p.131).
- **Fig 8(a)** = measured effective conductivity vs carbon concentration (the empirical σ(wt%) law;
  p.454). The heuristic maps effective-σ → carbon density via this relationship (p.78).
- **Percolation toe is empirical:** *"the carbon loading in the nylon powder should be at least 15 wt.%
  or even higher (30–40 wt.%) for rapid heating"* (p.105-106). Below ~15 wt% heats poorly.
- **Unit trap, quantified:** commercial ink is **5 wt.% carbon in-ink** (p.103,112); matrix target is
  **≥15 wt.% dopant-to-nylon**. The peer's "0–15 wt% in the matrix" is the matrix loading.
- Simulation operating point: initial σ = **0.04 S/m**, capped 0.0425 (p.542-545). The code's
  `sigma_doped` endpoint is an **effective composite** σ at a nominal label, not pure graphite
  (consistent with memory: σ₀=0.04 > σ*≈0.03).

**Consequence for the design:** the 0–15 wt% premix range is exactly the sub-threshold percolation
toe where a linear virgin↔doped map is least trustworthy. Decision A (premix_frac axis, wt% as a
labeled secondary) correctly sidesteps this. An *optional enrichment* (§6) can make the wt% display
axis empirically honest by mapping it through the measured Fig 8(a) curve.

---

## 2. Component A — premix material knob in the 2-D model

**Insertion strategy:** premix is applied at ≥2 assembly sites today (`:1447`, `:2801`). To avoid
scatter and keep it testable, extract ONE pure helper mirroring `heatr3d`:

```python
def apply_premix(sigma, eps_r, part_mask, *, premix_frac=0.0,
                 premix_budget="floor_added",
                 sigma_v, sigma_d0, eps_v, eps_d):
    """Raise the WHOLE domain to a premixed background and re-apply the jetted
    increment inside the part. premix_frac=0.0 returns (sigma, eps_r) UNCHANGED
    (bit-for-bit). Mirrors heatr3d.build_gamma premix semantics exactly."""
```

- `sigma_premix = sigma_v + f·(sigma_d0 − sigma_v)`, `eps_premix = eps_v + f·(eps_d − eps_v)`.
- `floor_added`: inside-part span = `sigma_d0 − sigma_v` on top of premix (total dopant rises).
- `budget_fixed`: inside-part span = `sigma_d0 − sigma_premix` (jet lifts premix to `sigma_d0`).
- Applied to BOTH sigma and eps_r fields; the bed (outside part) becomes `sigma_premix/eps_premix`,
  so the EQS field redistributes through the conductive bed (a genuine coupling, not a heat source).

**Off-path guarantee:** `premix_frac=0.0` → helper is identity → existing outputs bit-identical.

## 3. Component B — wt% bridge (display only)

```python
PREMIX_WTPCT_FULL = 25.0   # nominal label for the doped endpoint (docstring :333); NOT measured
def premix_frac_from_wtpct(wt):   # [ASSUMED linear, no percolation] — provenance: THEORY_REFERENCES
    return wt / PREMIX_WTPCT_FULL
```
wt% never enters physics; it labels the premix_frac axis in the figure. Tag `[ASSUMED]` per
THEORY_REFERENCES provenance discipline.

## 4. Component C — config plumbing

YAML (default OFF = bit-identical):
```yaml
premix:
  frac: 0.0            # 0.0 = off (current behavior). 0..1 fraction-of-doped.
  budget: floor_added  # or budget_fixed
```
`fgm_generator` needs premix-aware total-dopant accounting **only** for `budget_fixed`; `floor_added`
leaves the generator untouched.

## 5. Component D — the 0→15 wt% study

One script, shape = `jared_exp1_40mm`, premix_frac ∈ {0, .15, .3, .45, .6} (wt% ≈ {0, 3.75, 7.5, 11.25, 15}
via §3 — spans the full 0→15 wt% target, i.e. the sub-threshold toe through the ~15 wt% heating onset).
At each level, **re-solve the printed map with this lane's cheap bit-identical 2-D adjoint**, then record:
1. **Dopant map** (shape + magnitude) — shows the optimal map changing as the baseline conducts.
2. **Bulk RF absorption fraction** — nonzero bed σ raises absorption; field pattern shifts.
3. **Drive-to-ceiling** — the drive whose 2-D end-state peak = 250 °C ceiling
   (consume the ceiling VALUE + method from `solve3d/stage_a_phase2.py:323 ceiling_verdict` /
   `chosen_drive_power_density`, evaluated on the 2-D end-state peak — NOT a duplicate 3-D adjoint).
4. **Dopant peak-relocation** — the ±°C the printed map moves the peak (cf. Stage B +11–13 °C finding).

**One cross-check:** a single premix level re-run against heatr3d's existing 3-D premix
(`build_gamma(premix_frac=...)`) to confirm 2-D and 3-D agree in sign and rough magnitude.

## 6. Component E — figure (+ optional empirical overlay)

One composite (per figure-must-communicate memory): top row = dopant maps across premix levels;
bottom row = absorption / drive-to-ceiling / peak-relocation vs premix. Visualization-standard
settings (DPI 180, smoothing, colormaps).

**Optional enrichment (Matt decides at review):** overlay the group's measured premixed-composition
conductivities (Fig 8(a), Table 1 points) so the study reads as model-vs-measured. Requires digitizing
Fig 8(a) or locating its underlying data; flagged, not assumed.

---

## 7. Verification gates

- **Bit-identical off-path (RED first):** assert `premix_frac=0` reproduces current 2-D output
  bit-for-bit on the jared config (the off-path guarantee).
- **Helper TDD ladder:** RED `apply_premix(f=0)` == identity; RED bed rises by the linear law for f>0;
  RED `floor_added` vs `budget_fixed` masked values match hand-computed expectations; RED wt% bridge.
- **FD-gate:** premix here is a **swept forward parameter, not a design variable** → the adjoint still
  differentiates only w.r.t. the printed map (unchanged path) → no new gradient to gate. Stated
  explicitly. If premix later becomes a joint design variable, THAT gradient gets FD-gated then.
- **Compute:** feature+tests are CPU-light. The 5-level study is the cheap 2-D adjoint (not "heavy"),
  but load was 9.2 at spec time (solve3d Stage B). **Load-check + Matt's go before the study run**
  (compute-scheduling convention).

## 8. Honest provenance summary

- The σ/ε(wt%) mapping is `[ASSUMED linear]`. The **measured** relationship (Fig 8(a)) shows a
  percolation toe below ~15 wt% — the linear map overstates σ there. premix_frac-axis avoids baking
  the bad map into the physics; wt% labels carry the `[ASSUMED]` tag (or the empirical map if §6 taken).
- `sigma_doped` = effective composite σ (~0.04 S/m operating point), a nominal "25 wt%" label, not
  pure graphite. The figure caption will not assert a wt% number without this caveat.

## 9. Out of scope (YAGNI)

- Joint optimization of premix level (would need an FD-gated premix gradient — not now).
- Percolation-aware σ(wt%) law inside the physics (Decision A rejected it; the empirical Fig-8(a)
  map, if adopted, is display-only).
- Multi-shape / heavy sweeps (Decision B rejected; would need explicit go + load-check).
- Any change to the 3-D heatr3d premix (owned by the other session; we only consume it).

## 10. Files touched (planned)

- `rfam_eqs_coupled.py` — add `apply_premix` helper; call at assembly sites `:1447`, `:2801`.
- `fgm_generator.py` — premix-aware total-dopant accounting for `budget_fixed` only.
- `test_premix_2d.py` (new) — the TDD ladder in §7.
- `study_premix_sweep.py` (new) — Component D.
- config: `configs/jared_exp1_40mm_premix.yaml` (new, or a premix block toggle).
