# PA12 melt-energy floor vs delivered energy — "how much is too much"

**Date:** 2026-08-07 · **Scope:** materials/energy calculation, no solve.
**Reproducible:** every number below comes from `energy_floor.py` (TDD, `test_energy_floor.py`, 4/4).
Deepens `deck_figures_3d/fig5_energy_sls_vs_rfam.py` with the DSC-measured latent heat.

## Conclusion

The theoretical floor to fuse PA12 from a process preheat is **~120 J/cm³** (latent-dominated).
RFAM's *absorbed* energy (~2400 J/cm³) sits **~19× above** that floor; SLS's *system* energy
(1.1–3.6 ×10⁵ J/cm³) sits **~900–2900×** above it. The real operating band is narrow: the melt point
(171–186 °C) is the lower wall, the **250 °C degradation ceiling** is the upper wall, and there is only
**~160 J/cm³ of local thermal headroom** between "fused" and "burning." Energy delivered beyond what
keeps the *coldest* voxel fused while the *hottest* stays under the ceiling is waste — and that gap is
exactly what the FGM/premix uniformity work attacks.

## (a) Theoretical minimum melt energy (per cm³ of dense part)

`E = ρ·(cp_solid·ΔT + L)`, ρ ≈ ρ_liquid = 1010 kg/m³ (consolidated basis, comparable to fig5).

| Start temperature | Melt-energy floor | Note |
|---|---|---|
| **170 °C preheat** → full melt (186 °C) | **118–137 J/cm³** | sensible heat is tiny; **latent dominates** |
| **25 °C room temp** → full melt (186 °C) | **307–444 J/cm³** | cp-sensitive; the sensible term now leads |

Bands span latent **96.7–101.7 kJ/kg** `[MEASURED DSC 101.7 / LIT 96.7]` × cp_solid **1287–2100 J/kgK**
`[LIT; 1287 is room-value, ~2100 the effective average as cp rises toward melt]`.

**Assumptions stated:** (i) *full* melt to the DSC end (186 °C); onset (171 °C) is enough to *begin*
coalescence, a softer sub-floor. (ii) **Crystallinity X_c ≈ 42–49%** (DSC latent 101.7 kJ/kg ÷ 100%-
crystalline Δh_f 209–245 J/g `[LIT]`) — only that fraction absorbs latent, and the *measured* latent
already bakes X_c in, so no double-counting. (iii) Dense-part basis; per cm³ of *powder bed* (ρ≈490)
the mass and thus energy are ~½.

## (b) How far above the floor delivered energy sits

Positions vs the **preheat floor ≈ 124 J/cm³** (the realistic process start):

| Source | Energy (J/cm³) | × floor | Reading |
|---|---:|---:|---|
| SLS **laser dose** | 262 | **2.1×** | well-matched *at the point of delivery* |
| RFAM **absorbed (part)** | 2400 | **19.4×** | order-of-magnitude over floor |
| RFAM **system** @30% coupling | 8 000 | 65× | coupling band, unmeasured |
| RFAM **system** @5% coupling | 48 000 | 389× | pessimistic coupling |
| SLS **system** (low) | 1.08×10⁵ | 874× | chamber-dominated |
| SLS **system** (high) | 3.6×10⁵ | 2915× | chamber-dominated |

Even at pessimistic 5% RF coupling (389×), RFAM system energy stays **below** the SLS system band
(874×+) — consistent with the *rfam-vs-sls-energy* framing: compare **total system** energy, where
SLS's long hot-chamber serial build dominates. **RF coupling efficiency is the open, unmeasured factor
(the P-gate).**

## (c) "How much is too much" — the operating band

- **Lower wall (must fuse):** reach the melt window, 171 → 186 °C.
- **Upper wall (must not burn):** the **250 °C degradation ceiling** `[ASSUMED, thermal-ceiling workstream]`.
- **Headroom between them:** from a just-fused voxel to the ceiling is `ρ·cp_liq·ΔT` =
  **~162 J/cm³** of local energy. Beyond it, that voxel degrades.

So the honest operating band per voxel is roughly **[fuse ~120 J/cm³] … [+162 J/cm³ before burn]**.
RFAM's ~2400 J/cm³ *absorbed average* is ~19× the fuse floor **not** because any single voxel needs
that much, but because:
1. **Uniformity penalty** — the exposure must lift the *coldest* voxel over melt, so hotter voxels
   overshoot toward the ceiling. Narrowing that spread is the entire point of FGM grading and the
   premix baseline (peer TASK 1): a conductive baseline + graded map flattens absorption, so less
   average energy is needed to fuse the coldest region without cooking the hottest.
2. **Thermal loss** over the tens-of-minutes exposure (conduction to bed, convection, radiation).
3. **Sub-preheat heating** where the start is below 170 °C.

The gap between the ~120 J/cm³ floor and the ~2400 J/cm³ absorbed is the inefficiency budget the whole
compensation program is trying to close — and the 162 J/cm³ headroom is how little margin there is
before over-delivery becomes degradation.

## Provenance tags
- `[MEASURED]` latent 101.7 kJ/kg, melt window 171/180.8/186 °C — `configs/experimental_pa12_dsc_profile.yaml`
- `[LIT]` cp_solid 1287 / cp_liq 2500 J/kgK, ρ_solid 490 / ρ_liq 1010 kg/m³ — `RFAM_physics_from_literature.md:30-31`
- `[LIT]` 100%-crystalline Δh_f 209–245 J/g (crystallinity back-out) — general PA12 literature
- `[ASSUMED]` 250 °C degradation ceiling — thermal-ceiling (solve3d Stage A/B) workstream
- `[ESTIMATE]` SLS/RFAM delivered energies — `deck_figures_3d/fig5_energy_sls_vs_rfam.py` docstring
  (SLS laser 262, RFAM absorbed 2400, SLS system 1.08e5–3.6e5 J/cm³; RF coupling 5–30% unmeasured)

## Not done here (offered)
- A **deepened fig5** panel that draws the fuse floor *and* the 250 °C headroom band as the operating
  window (currently fig5 shows only the floor). Say the word and I'll add it (rendering only, no solve).
