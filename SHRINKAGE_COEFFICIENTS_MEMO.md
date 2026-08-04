# Shrinkage Coefficients Memo: PA12 Material Shrinkage for Level 0 Pre-Compensation

Date: 2026-08-04
Purpose: supply the Level 0 affine pre-compensation coefficients (s_xy, s_z_mat)
for the shrinkage-prewarp-v2 spec
(docs/superpowers/specs/2026-08-04-shrinkage-prewarp-v2-design.md, Section 1).
Scope: MATERIAL shrinkage of nylon 12 (PA12) only, i.e. the contraction of the
polymer itself on melt and recrystallization as published for SLS laser
sintering. Densification consolidation (powder-to-solid volume change) is
explicitly OUT of scope here; the model marches that density field and the
Level 0 pre-scale must not double-count it (named test in the spec).

Method: web literature survey (journal papers plus manufacturer and service
bureau documentation) and a pass over the local previous-work/ directory
(Jared Allison dissertation and the two Rapid Prototyping Journal papers) for
any RFAM-specific dimensional measurements. Every value below carries its
source, measurement condition, and directionality. Measured values are
distinguished from manufacturer or service-bureau defaults.

---

## 1. Published PA12 SLS shrinkage values

### 1.1 Measured values (parts or samples built with NO scaling factors, unless noted)

| # | Source | Value | Direction | Conditions | Type |
|---|--------|-------|-----------|------------|------|
| 1 | Wang et al., "Numerical Model and Experimental Validation for Laser Sinterable Semi-Crystalline Polymer: Shrinkage and Warping," Polymers 12(6):1373, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7361694 | 2.5% to 5% across the experimental design; single-parameter cooling study: 2.7% at 0.8 C/min, 3.2% at 25 C/min | In-plane (specimen length, 50 mm) | PA12 FS3300PA (Farsoon), EP-P3850 machine (E-Plus-3D), 50 x 10 x 1 mm specimens, shrinkage = (50 - L_i)/50; process parameters varied in an orthogonal design | Measured |
| 2 | Benedetti, Brule, Decreamer, Evans, Ghita, "Shrinkage behaviour of semi-crystalline polymers in laser sintering: PEKK and PA12," Materials and Design 180:107906, 2019. https://www.sciencedirect.com/science/article/pii/S0264127519303442 | Crystallization-attributed shrinkage 4.6%; crystallization is ~60% of the overall shrinkage observed on cooling (implying overall ~7 to 8% on their unscaled specimens); shrinkage slightly higher in Y than in X | Total measured on specimens spanning X, Y (10 mm each) and Z (10 to 100 mm); Y > X anisotropy reported | PA12 on EOS Formiga P100, all parts built with no scaling factors applied, specifically to expose material behavior | Measured |
| 3 | Soe, Eyers, Setchi, "Assessment of non-uniform shrinkage in the laser sintering of polymer materials," Int. J. Adv. Manuf. Technol. 68, 2013. https://link.springer.com/article/10.1007/s00170-012-4712-0 | Z-axis shrinkage is non-linear in all builds and linked to thermal inconsistencies in the build chamber; linearity improves for parts placed at the chamber center; per-axis factorial (single scale factor) adjustment judged inadequate | Z (build direction), with X, Y position dependence | PA12 on EOS P700 | Measured (behavioral finding; abstract does not publish a single percent number) |
| 4 | Shen et al., "Inhomogeneous Shrinkage of Polymer Materials in Selective Laser Sintering," SFF Symposium, Austin TX, 2000. https://repositories.lib.utexas.edu/bitstreams/4b728122-aea4-49a9-ac62-197091515cd7/download | Shrinkage is inhomogeneous, dependent on x-y position in the build room (bed temperature gradients) and on height; linear scaling of a geometry is not sufficient | X, Y, Z (spatially varying) | Polyamide SLS, hollow rectangular parallelepiped benchmarks (as summarized in ref [1] of Manetsberger 2001, same group) | Measured |
| 5 | Manetsberger, Shen, Muellers, "Compensation of Non-Linear Shrinkage of Polymer Materials in Selective Laser Sintering," SFF Symposium, Austin TX, 2001. https://repositories.lib.utexas.edu/bitstreams/5f3cc1c7-17b7-4eb4-9cc0-b06ad23cf663/download | Shrinkage depends on time, temperature, and pressure (powder weight above the part); over 50% of final shrinkage develops within the first 100 minutes of an isothermal hold, then saturates; final shrinkage is Arrhenius in temperature and linear in pressure | Height (Z) in dilatometer; conclusions applied to non-linear Z compensation | Dilatometer samples were laser-sintered PMMA cylinders (5 mm x 7 mm dia), 10 h holds at SLS-representative temperatures and pressures; the machine-benchmark shrinkage work in the same paper is polyamide SLS | Measured (note: dilatometry material is PMMA, mechanism-level evidence only) |
| 6 | Raghunath and Pandey, "Improving accuracy through shrinkage modelling by using Taguchi method in selective laser sintering," Int. J. Machine Tools and Manufacture 47(6), 2007. https://www.researchgate.net/publication/222571626 | Shrinkage is direction dependent: X shrinkage governed mainly by scan length and laser power; Y by scan speed and laser power; Z by bed temperature, scan speed, and scan spacing; empirical percent-shrinkage relations linear in scan length were fit per axis | X, Y, Z separately | Polyamide (DuraForm class) on an SLS machine, Taguchi design | Measured (per-axis empirical model; paywalled, per-axis percents not re-verified here) |
| 7 | Yang, Hwang, Lee, "A study on shrinkage compensation of the SLS process by using the Taguchi method," Int. J. Machine Tools and Manufacture 42(11), 2002. https://www.sciencedirect.com/science/article/abs/pii/S0890695502000706 | Shrinkage rates measured along X, Y, and Z; per-axis optimal scale factors derived via Taguchi method | X, Y, Z separately | DuraForm polyamide | Measured (paywalled, numbers not re-verified here) |

Notes on Table 1.1:
- Row 2 (Benedetti) is the cleanest attribution study: on unscaled parts the
  crystallization component alone is 4.6% and is ~60% of total. The remaining
  ~40% they attribute to powder bulk properties (low bulk density, particle
  morphology, porosity). That remainder overlaps what our model already
  captures as densification consolidation, so the Benedetti TOTAL must not be
  used as a material-shrinkage coefficient. This is the double-counting trap
  in concrete numeric form.
- Row 1 (Wang) is the best direct in-plane measurement range for the material
  effect at coupon scale: 2.5 to 5%, with ~3% typical at moderate cooling
  rates.
- Rows 3 to 5 establish that Z is non-linear and thermal-history dependent, so
  any single s_z number is a first-order convenience, not a law.

### 1.2 Manufacturer and service-bureau documented values (industry-calibrated defaults, not lab measurements)

| Source | Value | Direction | Notes |
|--------|-------|-----------|-------|
| 3DPRINTUK (SLS bureau, EOS machines, PA2200), "SLS PA12 PA2200 Nylon." https://www.3dprint-uk.co.uk/sls-pa12-pa2200-nylon/ | Standard nylon 12 shrinks 3 to 4% during the cooling phase; compensated by scale factors in the build file | Total (applied per axis in build prep) | Bureau default encoding EOS-calibrated compensation |
| SGD 3D, "SLS Design Guidelines." https://sgd3d.co.uk/3d-printing/sls-design-guidelines/ | Shrinkage typically 2 to 3% for PA12, compensated by machine scale factors | Total | Bureau design guideline |
| Formlabs, "X/Y Scaling (Fuse 1 generation printers)." https://formlabs.com/support/X-Y-Scaling-Fuse-1-generation-printers/ | Per-printer, per-powder X/Y scaling calibrated from a test print; varies unit to unit | XY | Manufacturer procedure: scale factors are calibrated, not fixed constants |
| EOS parameter documentation for PA2200 (e.g. "EOS PA2200 Parameter Overview," parameter sheets circulated per machine). https://www.scribd.com/document/431357563/EOS-PA2200-PrimeCast101 | Per-machine, per-material scaling factors in X, Y, Z plus beam offset | X, Y, Z separately | Manufacturer convention document; exact numbers are per machine and material batch and were not independently verifiable from the public copy |

---

## 2. Machine scale-factor conventions (cross-check)

The industry-standard compensation form matches our Level 0 design and gives an
independent sanity check on magnitudes:

- Per-axis linear scale factors. SLS build software applies independent X, Y,
  Z scale factors to the STL before slicing; the factors are calibrated by
  building and measuring test parts (Formlabs support doc above; Yang et al.
  2002; Raghunath and Pandey 2007). XY factors encode roughly the 3 to 4%
  class total contraction quoted by EOS-based bureaus (3DPRINTUK above).
- Z scale plus Z offset. Because Z shrinkage is non-linear in height and
  position (Soe et al. 2013; Shen et al. 2000; Manetsberger et al. 2001), EOS
  practice splits the Z correction into a height-proportional scale and a
  constant offset term, and the literature repeatedly concludes that a single
  linear Z factor is inadequate for tall parts (Soe et al. 2013: "current
  operational practices of factorial adjustment for each axis... are
  inadequate").
- Anisotropy direction. XY compensation is consistently LARGER than the
  Z material term in machine defaults, because much of the observed Z change
  is bed-collapse and consolidation that the machine handles through layer
  thickness and powder dosing rather than through the Z scale factor. This is
  the same partition our spec draws between material shrinkage (pre-scale) and
  densification consolidation (model).

Cross-check verdict: an affine per-axis pre-scale with XY around 3% and a
smaller Z material term is exactly the shape of the industry-calibrated
correction. Our Level 0 form is conventional; only the coefficient values are
uncertain for RFAM.

---

## 3. Chosen defaults for Level 0

Selection rule (stated, per spec open question 1): use only sources that
measured or compensate the MATERIAL effect at part scale; exclude any total
that includes powder-bed consolidation (the Benedetti total, and all
tall-part Z totals) to enforce the double-counting guard; then take the modal
overlap of (a) measured in-plane ranges on unscaled parts and (b) industry
XY compensation defaults, and carry the full literature spread as the band.
This is a midpoint-of-overlap rule, not a conservative-end rule; the band is
what carries the conservatism onto any dimensional claim.

- s_xy = 0.030 (3.0% in-plane material shrinkage)
  - Basis: Wang et al. 2020 measured 2.5 to 5% in-plane with ~2.7 to 3.2% at
    ordinary cooling rates; bureau defaults encode 3 to 4% (3DPRINTUK) and
    design guides 2 to 3% (SGD 3D). The overlap of all three is 3% and the
    modal measured value at moderate cooling is ~3%.
  - Band: 0.020 to 0.040. Covers the guideline low end and the bureau high
    end; excludes Wang's 5% corner (aggressive parameter corner) from the
    default but not from the band rationale.
- s_z_mat = 0.020 (2.0% material shrinkage in Z)
  - Basis: no clean PA12 material-only Z measurement exists in what was
    retrievable; observed Z totals are larger but contaminated by
    consolidation and bed thermal gradients (Benedetti 2019; Soe 2013).
    Machine conventions consistently use a Z material factor SMALLER than XY
    (Section 2). 2.0% takes the crystallization-dominated isotropic floor
    argument (crystallization is ~60% of total per Benedetti, and
    crystallization contraction is not strongly directional) and discounts it
    for the portion of Z change the densification model already owns.
  - Band: 0.010 to 0.030. This is the weakest number in the memo; it is a
    convention-anchored estimate, not a measured coefficient, and is flagged
    as the first target for P1 measurement.
- Anisotropy note: Benedetti reports Y slightly greater than X. We do not
  split s_x from s_y at Level 0; the X-Y difference in the literature is
  small relative to the band and is process-parameter dependent (Raghunath
  and Pandey 2007).

Config treatment (per spec): both coefficients config-driven, never
hardcoded, recorded in run provenance, and every dimensional claim carries
the band as a labeled uncertainty.

---

## 4. Applicability caveat (read this before trusting the numbers)

All values above are SLS LASER-SINTERING values. RFAM is volumetric RF
heating: the whole doped region melts in one exposure, there is no layer-wise
remelt, the thermal history (heating rate, peak temperature dwell, cooling
path through the crystallization window) is different, and the pressure state
of the surrounding powder differs from a progressively lowered SLS bed.
Every mechanism the literature identifies as controlling shrinkage magnitude
(cooling rate, temperature history, pressure, position in bed: Manetsberger
2001; Wang 2020; Soe 2013) therefore takes different values in RFAM. The
local previous work confirms the direction but not the coefficient: Allison,
Pearce, Beaman, Seepersad (Rapid Prototyping Journal 28(2):317-329, 2022,
previous-work/Computational_design_strategy... and the volumetric fusion
paper RPJ-09-2020-0218) report that RF-fused parts "appeared to show
shrinkage in all three coordinate axes," attributed to particle consolidation
on melting plus insufficient fusing in under-heated regions, with no
material-shrinkage coefficient extracted. The Allison dissertation
(previous-work/Jared_Allison_Dissertation_Final.pdf) cites the SLS shrinkage
literature (its refs [3] to [5]) but likewise reports no RFAM shrinkage
coefficient.

Plainly: these defaults are a literature-informed starting point. They are
the right ORDER and the right FORM, and they are the best available numbers
today, but the RFAM-specific coefficients are unmeasured. They stay
starting-point values pending the P1 measurement (print, measure, refit
s_xy and s_z_mat from our own parts).

---

## 5. The recrystallization-on-cooling question (feeds spec open question 2)

Question: does the literature attribute PA12 SLS shrinkage to cooling and
recrystallization, which would require the Level 2 read state to include a
cooldown segment?

Answer: yes, dominantly and with numbers.

- Benedetti et al. 2019: crystallization is responsible for ~60% of the
  overall shrinkage of PA12 laser-sintered parts, with the crystallization
  component alone measured at 4.6%. Crystallization happens on cooling below
  the crystallization temperature, after the melt.
- Wang et al. 2020: "the fused amorphous phase starts to form semi-crystalline
  structures during the cooling step," and measured shrinkage changes with
  cooling rate (2.7% at 0.8 C/min vs 3.2% at 25 C/min), which is only
  possible if the shrinkage is set during cooling, not at end of exposure.
- Manetsberger et al. 2001: dilatometry shows shrinkage developing over hours
  at temperature with more than half of the final value in the first 100
  minutes, saturating thereafter; final magnitude is Arrhenius in temperature.
  Shrinkage is a time-temperature integral, not an instantaneous event.
- Bureau documentation agrees: 3DPRINTUK states the 3 to 4% contraction
  happens "during the cooling phase."

Implication for the spec: the part measured after printing has passed through
the crystallization window; a Level 2 objective that reads final shape at
end-of-exposure misses the majority (order 60%) of the material shrinkage.
Two admissible resolutions, in increasing fidelity:
1. Keep the end-of-exposure read and apply the material shrinkage as the
   affine map (Level 0 coefficients) on top of the predicted shape. This is
   consistent because the affine map IS the cooldown, collapsed to a
   constant. Cheap, and correct to first order at part scale.
2. Add a cooldown segment to the Level 2 forward march and let shrinkage
   accrue with the local thermal history (the Manetsberger time-temperature
   form is a candidate law). Required only when spatially non-uniform cooling
   inside one part is believed to matter, which is also the regime where
   Level 3 mechanics starts to pay.
Recommendation: resolution 1 for Level 2 as specced; promote to resolution 2
only if P1 measurements show part-scale affine correction leaving structured
residuals.

---

## 6. Source list

Journal and symposium papers:
- Wang, Y. et al., Polymers 12(6):1373, 2020. https://pmc.ncbi.nlm.nih.gov/articles/PMC7361694 (open access, values verified against full text)
- Benedetti, L., Brule, B., Decreamer, N., Evans, K.E., Ghita, O., Materials and Design 180:107906, 2019. https://www.sciencedirect.com/science/article/pii/S0264127519303442 (values from abstract and indexed excerpts; full table paywalled)
- Soe, S.P., Eyers, D.R., Setchi, R., Int. J. Adv. Manuf. Technol. 68:111-125, 2013. https://link.springer.com/article/10.1007/s00170-012-4712-0
- Soe, S.P., J. Materials Processing Technology 212(11):2433-2442, 2012 (EOS P700 curling companion study). https://www.sciencedirect.com/science/article/abs/pii/S0924013612001914
- Shen, J. et al., SFF Symposium 2000. https://repositories.lib.utexas.edu/bitstreams/4b728122-aea4-49a9-ac62-197091515cd7/download (local copy read)
- Manetsberger, K., Shen, J., Muellers, J., SFF Symposium 2001. https://repositories.lib.utexas.edu/bitstreams/5f3cc1c7-17b7-4eb4-9cc0-b06ad23cf663/download (local copy read)
- Raghunath, N., Pandey, P.M., Int. J. Machine Tools and Manufacture 47(6):985-995, 2007. https://www.researchgate.net/publication/222571626
- Yang, H.J., Hwang, P.J., Lee, S.H., Int. J. Machine Tools and Manufacture 42(11):1203-1212, 2002. https://www.sciencedirect.com/science/article/abs/pii/S0890695502000706
- Allison, J., Pearce, J., Beaman, J., Seepersad, C., Rapid Prototyping Journal 28(2):317-329, 2022 (local PDF in previous-work/)
- Allison, J., PhD dissertation, UT Austin, 2023 (local PDF in previous-work/)

Manufacturer and bureau documentation:
- 3DPRINTUK: https://www.3dprint-uk.co.uk/sls-pa12-pa2200-nylon/
- SGD 3D SLS design guidelines: https://sgd3d.co.uk/3d-printing/sls-design-guidelines/
- Formlabs Fuse X/Y scaling procedure: https://formlabs.com/support/X-Y-Scaling-Fuse-1-generation-printers/
- EOS PA2200 parameter overview (per-machine scaling and beam offset convention): https://www.scribd.com/document/431357563/EOS-PA2200-PrimeCast101
