# Polymer-AM Density and Thermal-Ceiling Literature Memo

Date: 2026-08-06
Author: Claude Scholar (literature survey), for Matt McCoy
Purpose: Supply two config values for the thermal-ceiling joint-solve spec
(`docs/superpowers/specs/2026-08-06-thermal-ceiling-joint-solve-design.md`,
sections 1 and 9): (1) the final-part density-target floor `rho_target`, and
(2) the PA12 degradation / thermal-ceiling temperature `T_ceiling` and its
margin. Both are per-material config in the spec; this memo gives cited
tables, a chosen value, and the selection rule for each.

Scope note on units: the spec's `rho_target` is FINAL-PART density normalized
so that 1.0 is a fully dense (zero-porosity) solid. Throughout this memo,
"relative density" means part density divided by the fully dense PA12 solid
density (~1.01 g/cm3, see Table 1a), and "porosity" = 1 - relative density.

Two separate values, do not conflate them (mirrors the spec's "two
temperatures" framing): the MELT onset (~176-185 C) is a completeness
requirement the part must exceed to fuse; the DEGRADATION ceiling (this
memo's Deliverable 2) is the upper bound the peak must stay under.

---

## Applicability caveat (read first)

Every number below is from powder-bed laser sintering (SLS / PBF-LS), HP Multi
Jet Fusion (MJF), injection molding, or bench thermal analysis (DSC/TGA) of
PA12 powder and stock. None is from RF / volumetric heating. RFAM differs in
three ways that plausibly move both values:

1. Heating mode. SLS deposits energy at the surface layer with a preheated
   powder bed near the melt onset; RFAM deposits energy volumetrically. Peak
   location and heating rate differ, which changes where and when the
   degradation peak occurs (exactly the end-state-peak problem in spec section
   2).
2. Atmosphere. SLS runs under inert or near-inert process gas; injection
   molding is a closed melt. If RFAM heats in air, the OXIDATIVE degradation
   pathway (lower onset) governs, not the inert-atmosphere pathway. This is
   the single biggest reason to prefer the lower ceiling number below.
3. Dwell and thermal history. SLS/MJF hold parts near 150-178 C for hours;
   the degradation literature is dominated by that long powder-bed residence.
   RFAM dwell/rotation schedules (spec section 4) have a different time-at-
   temperature profile.

Treat both chosen values as defensible STARTING POINTS pending direct RFAM
measurement: DSC + TGA (in air and inert) on the actual doped Nylon 12
feedstock, and Archimedes + micro-CT density on printed RFAM coupons. The
memo flags where a measurement would most change the answer.

---

## Deliverable 1: Achievable final-part density (rho_target floor + ideal)

### Table 1a. Reference: fully dense PA12 solid (the 1.0 normalization point)

| Quantity | Value | Source type | Citation |
|---|---|---|---|
| PA12 solid density (bulk stock, injection-molded/cast) | 1.01-1.02 g/cm3 | Datasheet/handbook | Xometry, "Nylon 12 (PA12): Properties, Density and Melting Point" (2023), https://www.xometry.com/resources/materials/nylon-12/ ; Wikipedia "Nylon 12", https://en.wikipedia.org/wiki/Nylon_12 |
| PA12 crystallinity (governs max attainable density) | 20-30% | Handbook | PatSnap Eureka, "Nylon 12: ... Molecular Structure, Processing" (2024), https://eureka.patsnap.com/materials/nylon-12-structure-processing |

PA12 has the lowest density of the commercial nylons because of its long
aliphatic segment between amide groups. Use ~1.01 g/cm3 as the "relative
density = 1.0" anchor. A powder-bed part cannot reach 1.0 in practice; the
useful question is how close, and where the mechanical knee sits.

### Table 1b. Measured PA12 powder-bed final-part density / porosity

All rows are MEASURED (Archimedes buoyancy and/or micro-CT), not datasheet,
unless labeled. "Rel. density" is computed as 1 - porosity for cross-study
comparability.

| Process / material | Porosity (measured) | Rel. density | Method | Conditions | Citation |
|---|---|---|---|---|---|
| SLS, EOS PA2200 (Stichel et al.) | 3.2-2.8% | 0.968-0.972 | micro-CT | ED 1.67-3.72 J/cm2 | in Morano & Pagnotta review [ref 25], Polymers 15(22):4446 (2023) |
| SLS, EOS PA2200 (Dewulf et al.) | 4.7-2.6% | 0.953-0.974 | micro-CT | ED 2.44-4.20 J/cm2 | in Morano & Pagnotta review [ref 36] (2023) |
| SLS, EOS PA2200 (Liebrich et al.) | 2.6-0.7% | 0.974-0.993 | micro-CT | thin-walled structures | in Morano & Pagnotta review [ref 43] (2023) |
| SLS, EOS PA2200 (Pavan et al.) | 4.8-3.6% | 0.952-0.964 | micro-CT | ED 2.00-5.00 J/cm2 | in Morano & Pagnotta review [ref 47] (2023) |
| SLS, DuraForm PA12 (Dupin et al.) | 16.1-4.3% | 0.839-0.957 | Archimedes + micro-CT | ED 1.07-2.67 J/cm2 | in Morano & Pagnotta review [ref 33] (2023) |
| SLS, DuraForm PA12 (chess pieces) | 4.4-6.3% (Archimedes); 8.8-9.2% (micro-CT) | 0.937-0.956 / 0.908-0.912 | Archimedes AND micro-CT | bed 166-170 C, 4.6 W, 3300 mm/s, 0.1 mm | Colucci, Fontana, Barberi, Vitale Brovarone, Messori, Polymers 16 (2024), PMC11678726, https://pmc.ncbi.nlm.nih.gov/articles/PMC11678726/ |
| SLS, PA12 + carbon fiber | 0.68% | 0.993 | (reported) | reinforced grade | ScienceDirect S2214860418306699 (Addit. Manuf.), via search summary |
| SLS, PA12 (typical range across many studies) | 0.7-16%, typ. 2.5-4.8% | 0.95-0.975 typical | micro-CT / Archimedes | review of ~10 studies | Morano & Pagnotta, Polymers 15(22):4446 (2023), doi:10.3390/polym15224446, https://pmc.ncbi.nlm.nih.gov/articles/PMC10675180/ |
| SLS, EOS PA2200 (vendor spec) | n/a | 0.90-0.95 g/cm3 => 0.89-0.94 rel. | EOS-Method, DATASHEET | "depends on exposure and x,y,z position" | EOS GmbH, "Product Information Feinpolyamide PA2200 for EOSINT P" (rev. 07.04), https://3dformtech.fi/wp-content/uploads/2019/11/Material-Data-PA2200.pdf |
| MJF (HP 3D HR PA12) | vendor claims near-full / isotropic | ~0.99-1.00 (vendor) | DATASHEET / vendor | voxel-level fusion | HP / Forge Labs, Proto3000 material pages (vendor), https://forgelabs.com/multi-jet-fusion-pa12/ |

Reading: well-processed SLS PA12 lands at 95-97.5% relative density (2.5-4.8%
porosity) as the working norm; optimized thin-wall or fiber-filled cases reach
99%+; poorly-processed or low-energy builds fall to 84-90%. The EOS datasheet's
own qualified-part density band is 0.90-0.95 g/cm3 (89-94% relative), and EOS
explicitly sells these for "qualified series production parts" - i.e. the
industry treats ~90%+ as a functional part.

Note the method gap: micro-CT reads 3-5 percentage points MORE porosity than
Archimedes on the same parts (Colucci 2024: 4.6% Archimedes vs 8.8% CT on the
same knight), because CT resolves surface-skin porosity Archimedes misses. If
RFAM density is later measured by Archimedes, expect it to read optimistic
relative to CT; pick one method and state it in the config.

### Table 1c. Context: other AM polymers

| Polymer / process | Achievable density / porosity | Source type | Citation |
|---|---|---|---|
| PA11, SLS/PBF-LS | comparable to PA12; modulus drops with porosity/defects on reuse | Measured | "Effect of PBF-LS ... on Reused Polyamide 11", Polymers 15(23):4602 (2023), doi:10.3390/polym15234602, https://pmc.ncbi.nlm.nih.gov/articles/PMC10708357/ |
| PP (polypropylene), PBF-LS | chemical resistance, ultra-low density; porosity governed by melt/packing | Review | Polym. review 18(5):622 (2025), doi:10.3390/polym18050622 |
| TPU, PBF-LS | flexible/elastomeric; intentionally more porous in many grades | Review | Same review (2025); Ma et al., Materials 14(5):1169 (2021), doi:10.3390/ma14051169 |
| PEEK, PBF-LS | apparent porosity rises sharply with spread speed; needs high-T system | Review | "L-PBF of Polymers: Quantitative Research Direction Indices", Materials 14(5):1169 (2021), https://pmc.ncbi.nlm.nih.gov/articles/PMC7958861/ |

Takeaway for context: PA12 is the best-densifying and most-studied of the
powder-bed polymers. PP/TPU/PEEK generally sit at equal or higher porosity, so
a floor tuned to PA12 is not conservative for those materials - each would need
its own value if RFAM moves to them.

### Does the literature accept <100% density for functional parts?

Yes, clearly. (a) EOS sells 0.90-0.95 g/cm3 (89-94% relative) PA12 as
qualified series-production material. (b) SLS PA12 tensile strength runs
~75-80% of injection-molded PA12 and is still the workhorse for functional
end-use parts (search summary of SLS-vs-IM comparisons; Morano & Pagnotta 2023
discuss the porosity/strength coupling directly). (c) Mechanical properties
rise with energy density / density and plateau: strength climbs steeply as
porosity falls through the 10-20% range, then flattens above ~90-92% relative
density. Below ~88-90% relative density, parts are "weak, porous and
anisotropic" (multiple SLS parameter studies, e.g. Materials 12(6):871 (2019),
doi:10.3390/ma12060871, https://pmc.ncbi.nlm.nih.gov/articles/PMC6471919/ ).
Full mechanical-vs-density curves for PA12 are sparse in open literature; the
consistent qualitative finding is a knee near 90% relative density.

### Chosen rho_target: floor and ideal

- Ideal (normalization reference): rho_target_ideal = 1.0 (fully dense PA12,
  ~1.01 g/cm3). Keep as the metric's 1.0 anchor per Matt. Flag honestly that
  powder-bed processes do not reach it: the realistic best is ~0.98-0.99, so a
  solve chasing 1.0 will always run against the ceiling. Consider a "practical
  ideal" of 0.98 as the value the solver actually targets.
- Floor: rho_target_floor = 0.90 (relative density; ~0.91 g/cm3 absolute).

Selection rule for the floor: set the floor at the mechanical knee - the
relative density below which tensile strength falls off its plateau. Two
independent anchors put that knee at ~0.90: (i) EOS's own qualified-part
density band bottoms at 0.90 g/cm3 (0.89-0.94 relative), and (ii) SLS
parameter studies describe sub-90%-density parts as weak/porous/anisotropic
while above-90% parts are functional. 0.90 is therefore the lowest density
that still buys a "good part," which is exactly what a flexible floor should
encode. Confirms Matt's provisional >= 0.90.

Recommended config band (flexible floor per spec section 1's "flexible
floor"):
- floor (hard): 0.90 - below this, refuse / warn (part is off the plateau).
- good (target band): 0.95 - the working norm for well-processed SLS PA12.
- practical ideal: 0.98 - realistic best; what the solver should chase.
- ideal (normalization): 1.0 - metric anchor only, not physically attainable.

Caveat: these are laser-sintering densification numbers. RFAM volumetric
heating may reach a different attainable ceiling; the FLOOR (mechanical knee)
is a material property and should transfer better than the CEILING (process-
limited best density). Measure printed RFAM coupon density before trusting
the 0.98 practical ideal.

---

## Deliverable 2: PA12 degradation / thermal-ceiling temperature

### Table 2a. Melt onset (completeness requirement, NOT the ceiling)

| Quantity | Value | Method | Citation |
|---|---|---|---|
| Tm onset (DSC) | 171-176 C | DSC, MEASURED | Vendittoli et al., Sci. Rep. (2025): virgin 171.4 C onset; Xometry (2023) 175-180 C |
| Tm peak (DSC) | 176-184 C | DSC, MEASURED / datasheet | Vendittoli et al. peak 179.8 C (virgin); EOS PA2200 datasheet 184 C at 20 C/min; general PA12 176-180 C |
| Crystallization temp | 138 C | DSC, datasheet | EOS PA2200 Product Information |
| Typical SLS bed / build temp | 140-178 C | process | Vendittoli et al. (2025): chamber 140 C, bed surface 178 C; Colucci et al. bed 166-170 C |

Matt's spec value ~185 C for melt onset is at the top of / just above the
measured DSC peak (176-184 C). It is a safe completeness threshold: exceeding
it guarantees the crystalline melt is complete. Keep it; if anything it is
slightly conservative (the true onset is ~171-176 C).

Citations: Vendittoli, Mascolo, Polini, Walter, Sorrentino, Sover,
"Degradation effects of reused PA12 powder in selective laser sintering...",
Scientific Reports (2025), https://pmc.ncbi.nlm.nih.gov/articles/PMC12485217/
(also nature.com/articles/s41598-025-20280-7). EOS GmbH PA2200 Product
Information (melting 184 C DSC, crystallization 138 C, melt enthalpy ~115 J/g),
https://3dformtech.fi/wp-content/uploads/2019/11/Material-Data-PA2200.pdf .

### Table 2b. Degradation ceiling (the constraint) - by atmosphere and method

| Quantity | Value | Method / atmosphere | Source type | Citation |
|---|---|---|---|---|
| TGA decomposition onset (5% mass loss), inert | ~350 C | TGA, nitrogen | Handbook/analysis | PatSnap Eureka PA12 analysis (2024), https://eureka.patsnap.com/materials/pa12-structure-processing |
| TGA max degradation rate, inert | 420-450 C | TGA, nitrogen | Handbook/analysis | same |
| Oxidative degradation onset, air | ~280 C | TGA, air (accelerated by Fe/Cu traces) | Handbook/analysis | same |
| Injection-molding do-not-exceed | ~260 C (melt 220-250 C) | processing practice | Handbook | same; general PA12 processing guides |
| SLS thermal-degradation relevance | degradation/chain effects appear from prolonged residence well below Tdecomp | TGA + MW, MEASURED | Journal | Vasquez et al., "MIE determination and thermal degradation study of PA12 ... laser sintering", J. Loss Prev. Process Ind. (2013), doi:10.1016/j.jlp.2013.08.001, https://www.sciencedirect.com/science/article/abs/pii/S0950423013001939 ; "Laser sintering of PA12 with limited thermal degradation", J. Manuf. Process. (2024), doi:10.1016/j.jmapro.2024.06.048 |
| Reuse-driven degradation signature | Tm rises ~1.2 C over 5 reuses (crystalline reorg from thermal-oxidative aging) | DSC, MEASURED | Journal | Vendittoli et al., Sci. Rep. (2025), PMC12485217 |

The atmosphere split is the crux. Under INERT gas, PA12 is stable to a 5%-loss
onset of ~350 C. In AIR, oxidation (catalyzed by trace metals) drops the onset
to ~280 C, and practical processing experience puts the "do not exceed"
discoloration/embrittlement bound around 250-260 C. PA12 also degrades by slow
thermal-oxidative chain scission and post-condensation at temperatures far
below any TGA onset given enough residence time (the entire reused-powder
degradation literature, e.g. Vendittoli 2025, is about aging at 140-178 C over
hours), so the fast-ramp TGA onset OVERSTATES the safe steady ceiling.

### Confirming / correcting Matt's ~250 C read

Matt's ~250 C is CORRECT as a practical degradation ceiling and is well-
justified, not arbitrary:

- It sits ~30 C below the ~280 C air/oxidative 5%-loss onset - a sensible
  margin against oxidation, which is the relevant pathway if RFAM heats in air.
- It matches the injection-molding "exceeding ~260 C risks thermal
  degradation" rule, with a few degrees of headroom.
- It is far below the inert-atmosphere ~350 C onset, correctly NOT trusting the
  nitrogen number for an in-air or partially-oxidative process.

The one correction: 250 C is not a hard burn point; it is a prudent practical
bound. The hard fast-ramp onsets are ~280 C (air) and ~350 C (inert). So 250 C
is the right CEILING to constrain against, but label it as the
degradation/discoloration bound, not "the temperature PA12 burns."

### Chosen T_ceiling and margin

- T_ceiling (constraint, in-air / conservative): 250 C. Enforce
  max_{t,x} T(x,t) <= 250 C as spec constraint g1. Confirms Matt.
- Underlying hard references (for config comments and any later relaxation):
  air oxidative onset ~280 C; inert 5%-loss onset ~350 C.
- Recommended solver margin below the ceiling: 10-20 C. Start a soft warning
  band at ~235 C so model error and the smoothed-peak / true-peak gap (spec
  section 3) cannot silently push the true peak over 250 C. Report the true
  max always; never let the smoothed peak proxy sit at 249 C while the true
  peak is at 255 C.
- If a later RFAM TGA in the ACTUAL process atmosphere shows the process is
  effectively inert (little oxygen at the hot zone), the ceiling could be
  raised toward 280-300 C - but only with that measurement in hand. Do not
  raise it on the inert-N2 350 C number alone.

Selection rule for the ceiling: take the lowest credible degradation onset for
the process's actual atmosphere, then subtract a margin for residence time and
model uncertainty. In air that is 280 C onset minus ~30 C = 250 C. This is
Matt's value, now with a citation chain and a stated rule.

---

## Summary of chosen config values

| Config key | Chosen value | Rule | Confidence / caveat |
|---|---|---|---|
| rho_target_floor | 0.90 (relative; ~0.91 g/cm3) | mechanical knee: lowest density that is still a functional part | Medium-high; material knee, transfers reasonably to RFAM |
| rho_target good-band | 0.95 | working norm for well-processed SLS PA12 | Medium; process-dependent |
| rho_target practical ideal | 0.98 | realistic powder-bed best | Medium; MEASURE RFAM coupon before trusting |
| rho_target ideal (anchor) | 1.0 | normalization reference only | Not physically attainable in powder bed |
| melt_onset (completeness) | 185 C (keep Matt's) | above DSC peak (176-184 C) => full melt | High; slightly conservative |
| T_ceiling (degradation) | 250 C | air oxidative onset (~280 C) minus ~30 C margin | Medium-high in air; MEASURE RFAM atmosphere |
| solver warning margin | 10-20 C (band from ~235 C) | absorb model error + smoothed-vs-true peak gap | Recommendation |

Single most valuable RFAM measurement to de-risk both values: a TGA of the
doped Nylon 12 feedstock in the actual RFAM process atmosphere (air vs inert
decides whether the ceiling is ~250 or ~300 C), paired with Archimedes +
micro-CT density on a printed coupon (decides whether 0.98 practical ideal is
reachable).

---

## Sources

Peer-reviewed / measured:
- Morano, C.; Pagnotta, L. "Additive Manufactured Parts Produced Using
  Selective Laser Sintering Technology: Comparison between Porosity of Pure and
  Blended Polymers." Polymers 15(22):4446 (2023). doi:10.3390/polym15224446.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC10675180/
- Colucci, G.; Fontana, L.; Barberi, J.; Vitale Brovarone, C.; Messori, M.
  "Chess-like Pieces Realized by Selective Laser Sintering of PA12 Powder: 3D
  Printing and Micro-Tomographic Assessment." Polymers 16 (2024).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC11678726/
- Vendittoli, V.; Mascolo, M.C.; Polini, W.; Walter, M.S.J.; Sorrentino, L.;
  Sover, A. "Degradation effects of reused PA12 powder in selective laser
  sintering on material characteristics, dimensional accuracy and mechanical
  strength." Scientific Reports (2025).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC12485217/ ;
  https://www.nature.com/articles/s41598-025-20280-7
- Vasquez, M.; et al. "MIE determination and thermal degradation study of PA12
  polymer powder used for laser sintering." J. Loss Prevention in the Process
  Industries (2013). doi:10.1016/j.jlp.2013.08.001.
  https://www.sciencedirect.com/science/article/abs/pii/S0950423013001939
- "Laser sintering of polyamide 12 with limited thermal degradation." Journal
  of Manufacturing Processes (2024). doi:10.1016/j.jmapro.2024.06.048.
  https://www.sciencedirect.com/science/article/pii/S1526612524006376
- "Effect of Powder Bed Fusion Laser Sintering on Dimensional Accuracy and
  Tensile Properties of Reused Polyamide 11." Polymers 15(23):4602 (2023).
  doi:10.3390/polym15234602. https://pmc.ncbi.nlm.nih.gov/articles/PMC10708357/
- "Influence of Manufacturing Parameters on Mechanical Properties of Porous
  Materials by Selective Laser Sintering." Materials 12(6):871 (2019).
  doi:10.3390/ma12060871. https://pmc.ncbi.nlm.nih.gov/articles/PMC6471919/
- Ma, and others. "Laser Powder Bed Fusion of Polymers: Quantitative Research
  Direction Indices." Materials 14(5):1169 (2021). doi:10.3390/ma14051169.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC7958861/
- "Polymeric Powders for Powder Bed Fusion..." Polymers 18(5):622 (2025).
  doi:10.3390/polym18050622.
- "Pore analysis and mechanical performance of selective laser sintered
  objects." Additive Manufacturing (2018).
  https://www.sciencedirect.com/science/article/pii/S2214860418306699

Datasheet / vendor / handbook (labeled as such in tables):
- EOS GmbH. "Product Information: Feinpolyamide PA2200 for EOSINT P" (rev.
  07.04). Part density 0.90-0.95 g/cm3 (EOS-Method); melting 184 C (DSC, 20
  C/min); crystallization 138 C; bulk powder density >0.43 g/cm3.
  https://3dformtech.fi/wp-content/uploads/2019/11/Material-Data-PA2200.pdf
- Xometry. "Nylon 12 (PA12): Properties, Density and Melting Point" (2023).
  https://www.xometry.com/resources/materials/nylon-12/
- Wikipedia. "Nylon 12." https://en.wikipedia.org/wiki/Nylon_12
- PatSnap Eureka. "Thermoplastic Polyamide PA12: ... Molecular Structure,
  Processing Technologies..." (2024).
  https://eureka.patsnap.com/materials/pa12-structure-processing
- HP / Forge Labs / Proto3000. MJF PA12 (HP 3D High Reusability PA12) vendor
  material pages. https://forgelabs.com/multi-jet-fusion-pa12/ ;
  https://proto3000.com/materials/hp-pa-12-nylon-12/
