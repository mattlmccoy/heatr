# Graphite vs carbon black: does dopant loading move relative permittivity at radio frequency?

Date: 2026-08-01
Purpose: literature answer to the question that gates the permittivity-channel
census (EPS_CHANNEL_REPORT.md Section 11): is the engine assumption at
rfam_eqs_coupled.py:290-292, that the doped part's relative permittivity
(eps_r) is FIXED at 20 while only electrical conductivity (sigma) varies with
dose, physically plausible for a carbon-black-loaded Nylon 12 (polyamide 12)
matrix in the 1 to 30 MHz electro-quasi-static regime?
Method: web literature research only. No simulation, no measurement. Every
citation below was resolved against the publisher or an indexing service on
2026-08-01; none is from memory alone.

---

## 1. Verdict, first

**Likely yes, the permittivity channel is physically real: carbon black
loading moves eps_r wherever it usefully moves sigma, and the fixed-eps_r
assumption is very unlikely to hold over any loading range in which sigma
varies enough to grade.** Confidence: HIGH for the qualitative claim that
eps_r co-varies with loading (this is percolation physics with five decades of
theory and measurement behind it, verified in our frequency class, and it is
also what Jared Allison's own graphite data show). LOW-TO-NONE for any
specific quantitative law, including the co-varying blend the historical
campaign used AND the specific value eps_r = 20; both remain uncalibrated for
this ink until the planned impedance measurement (INK_DOPANT_CALIBRATION_BRIEF.md
M2) runs.

Three consequences, stated plainly:

1. **The pinned-eps_r mode (eps_geometry_only) is the physically implausible
   one, not the co-varying one.** A conductive filler in an insulating matrix
   cannot raise sigma by orders of magnitude across a loading window while
   holding eps_r constant. Below and near the percolation threshold the two
   properties rise TOGETHER, and eps_r in fact diverges at the threshold.
   This is established literature, not inference.
2. **That does not validate the solver's specific co-variation law.** The
   production hook blends eps_r linearly in fill fraction toward a doped
   value; the literature law near percolation is a power-law divergence,
   eps_r proportional to |p - pc| to the power -s with s near 1, which is far
   steeper than linear near the knee. The sign and existence of the channel
   are supported; the shape of the curve is not. Plausible inference, to be
   replaced by measurement.
3. **The eps_r = 20 number itself has no measurement provenance** (already
   flagged in INK_DOPANT_CALIBRATION_BRIEF.md Section 7: Allison's published
   doped value is 13.8 at 27.12 MHz, and the in-repo A/B study found 13.8
   fits better). The literature adds a second, independent reason to distrust
   it: a single fixed number cannot represent a dose-graded material at all.

So the census in EPS_CHANNEL_REPORT.md is NOT a model-only artifact on
physical-plausibility grounds. The permittivity channel survives the
literature test. What does not survive is treating the current actuator law
as calibrated. Deployability now hinges on the measured eps_r(s) slope across
the achievable 1 to 9 wt% window (decision rule D2), not on whether the
channel exists.

---

## 2. The established physics: why eps_r must move with loading

**Percolation theory ties the two properties to the same geometric object,
the filler network.** As conductive filler concentration p approaches the
percolation threshold pc from below, the direct-current conductivity of the
composite stays low but the relative permittivity DIVERGES as
eps_r proportional to (pc - p) to the power -s, with s approximately 1 in
three dimensions. Above pc, sigma rises as (p - pc) to the power t with t
near 2. This is the classical result of Efros and Shklovskii and has been
reviewed and confirmed across filler systems for decades:

- A. L. Efros and B. I. Shklovskii, "Critical behaviour of conductivity and
  dielectric constant near the metal-non-metal transition threshold,"
  Physica Status Solidi (b), vol. 76, pp. 475-485, 1976.
  DOI: 10.1002/pssb.2220760205 (resolves to Wiley Online Library).
- C.-W. Nan, Y. Shen, and J. Ma, "Physical properties of composites near
  percolation," Annual Review of Materials Research, vol. 40, pp. 131-151,
  2010. DOI: 10.1146/annurev-matsci-070909-104529 (resolves to Annual
  Reviews). Review confirming the giant-permittivity effect near percolation
  as a general property of conductor-insulator composites.

**The physical mechanism is interfacial polarization, also called
Maxwell-Wagner-Sillars (MWS) polarization.** Isolated and near-touching
conductive clusters act as internal electrode pairs separated by thin polymer
gaps; each gap is a micro-capacitor. As loading rises, clusters grow and gaps
shrink, capacitance per unit volume grows, and the effective eps_r climbs
steeply. The same cluster growth is what eventually creates the conductive
path, which is why eps_r and sigma cannot be decoupled by any choice of
filler amount: they are two readings of one microstructure.

**This is verified in carbon systems, at our frequencies, not just at GHz:**

- Y. Song, T. W. Noh, S.-I. Lee, and J. R. Gaines, "Experimental study of
  the three-dimensional ac conductivity and dielectric constant of a
  conductor-insulator composite near the percolation threshold," Physical
  Review B, vol. 33, pp. 904-908, 1986. DOI: 10.1103/PhysRevB.33.904.
  Amorphous carbon in polytetrafluoroethylene powder, measured 10 Hz to
  13 MHz, which overlaps the rig's operating class directly. Near pc both
  quantities follow weak power laws in frequency (sigma proportional to
  omega to the 0.86, eps proportional to omega to the -0.12), i.e. the giant
  permittivity persists into the tens-of-MHz range with only a mild decay.
- D. S. McLachlan and M. B. Heaney, "Complex ac conductivity of a carbon
  black composite as a function of frequency, composition, and temperature,"
  Physical Review B, vol. 60, pp. 12746-12751, 1999.
  DOI: 10.1103/PhysRevB.60.12746. Carbon black specifically, 10 mHz to
  1 MHz, composition-resolved through the percolation transition.
- H. Zois, L. Apekis, and M. Omastova, "Electrical properties of carbon
  black-filled polymer composites," Macromolecular Symposia, vol. 170,
  pp. 249-256, 2001.
  DOI: 10.1002/1521-3900(200106)170:1<249::AID-MASY249>3.0.CO;2-F.
  Carbon black in polypropylene: percolation threshold 6.2 wt%, and BOTH the
  dielectric constant and the conductivity follow the percolation power laws
  with exponents in agreement with theory. This is the single most on-point
  published result for the present question: in a carbon black composite,
  eps_r demonstrably obeys the diverging power law in loading.

**And it is what the project's own baseline data show for graphite.** Jared
Allison's dissertation Chapter 3 (HP 4194A impedance analyzer, 40 kHz to
40 MHz, graphite in Nylon 12 powder) measured eps_r rising from about 4 at
10 wt% to about 240 at 60 wt%, with the knee at 37.5 wt% (hand-digitized
curve in binderjet/code/RF_electrode_calculations.py:37-41). Established, and
in-house: the baseline filler system itself never had a fixed eps_r. The
engine comment "Allison's part eps_r = 20 is fixed" describes a modeling
convenience at one operating point, not a property of the material.

**What "fixed eps_r while sigma varies" would require, and where it can
happen.** Far ABOVE the percolation threshold, deep in the conductive
plateau, sigma still creeps up slowly with loading while eps_r behavior
becomes dominated by conduction (and can even turn negative past the
transition; see the percolation-triggered negative permittivity literature,
e.g. the nano carbon powder / polyvinylidene fluoride study, PMC11357240).
But that regime is useless for grading precisely because sigma sensitivity to
dose is smallest there, and it is unreachable anyway: the ink's achievable
window is 1 to 9 wt% in matrix (INK_DOPANT_CALIBRATION_BRIEF.md Section 3).
There is no loading range in the literature where sigma varies USEFULLY and
eps_r plateaus. That is the direct answer to the gating question.

---

## 3. Graphite vs carbon black: why Allison's 30-35 wt% does not transfer

The two fillers percolate in different loading regimes because percolation is
governed by particle geometry and aggregate structure, not by carbon
chemistry.

**Carbon black** is made of fused nanoscale primary particles forming
branched, high-surface-area aggregates. High-structure conductive grades
(the class sold for conductive composites, characterized by high oil
absorption number) percolate at very low loadings; low-structure and thermal
blacks percolate later:

- H. J. Choi, M. S. Kim, D. Ahn, S. Y. Yeo, and S. Lee, "Electrical
  percolation threshold of carbon black in a polymer matrix and its
  application to antistatic fibre," Scientific Reports, vol. 9, article
  6338, 2019. DOI: 10.1038/s41598-019-42495-1. Carbon black in polyethylene
  terephthalate percolated at 0.58 wt%.
- J.-C. Huang, "Carbon black filled conducting polymers and polymer blends,"
  Advances in Polymer Technology, vol. 21, no. 4, pp. 299-313, 2002.
  DOI: 10.1002/adv.10025. Review: the percolation threshold in carbon black
  composites is controlled by black structure (oil absorption), surface
  area, and matrix interaction, with conductive-grade blacks reaching
  percolation at low single-digit weight percent and low-structure blacks
  requiring substantially more.
- Zois et al. 2001 (full citation above): 6.2 wt% for a mid-grade black in
  polypropylene.

Literature-typical range for carbon black in thermoplastics: roughly 0.5 to
10 wt% depending on structure, dispersion, and matrix crystallinity.

**Graphite** used as micron-scale flakes (Allison's Loud Wolf 44 um powder is
this class) is a low-aspect-ratio, low-structure filler by comparison. Each
flake is dense and compact, so far more mass is needed before flake-to-flake
contact paths form:

- I. Krupa, I. Novak, and I. Chodak, "Electrically and thermally conductive
  polyethylene/graphite composites and their mechanical properties,"
  Synthetic Metals, vol. 145, no. 2-3, pp. 245-252, 2004.
  DOI: 10.1016/j.synthmet.2004.05.007 (resolves to ScienceDirect,
  PII S0379677904002188). Percolation at about 11 vol% graphite in both low
  and high density polyethylene. At graphite density near 2.26 g/cm3 against
  a polymer near 1 g/cm3, 11 vol% corresponds to roughly 20-22 wt%, an order
  of magnitude above conductive carbon black thresholds.
- M. Rahaman, P. Gupta, M. Hossain, et al., "Effect of carbons' structure
  and type on AC electrical properties of polymer composites: predicting the
  percolation threshold of permittivity through different models," Colloid
  and Polymer Science, vol. 301, pp. 1001-1019, 2023.
  DOI: 10.1007/s00396-023-05120-2. Recent confirmation that carbon type and
  structure set both the conductivity AND the permittivity percolation
  behavior, and that permittivity rises by orders of magnitude with loading
  in these systems.

Allison's own measured graphite percolation near 30 wt% in Nylon 12 powder
sits above the Krupa fully-dense-composite number, which is consistent: his
matrix is a partially packed powder bed (packing fraction near 0.44), so the
per-total-volume filler network is even more dilute at a given wt%.

### Comparison table

| Property | Graphite flake (Allison baseline) | Carbon black (current ink) |
|---|---|---|
| Particle geometry | Dense micron flakes (44 um) | Branched aggregates of fused nanoparticles |
| Percolation threshold, literature | ~11 vol% (~20 wt%) in polyethylene (Krupa 2004); Allison measured ~30 wt% in Nylon 12 powder | ~0.5 to 10 wt% depending on structure (Choi 2019: 0.58 wt%; Zois 2001: 6.2 wt%; Huang 2002 review) |
| Useful grading window | 30-35 wt% (Allison, impedance spectroscopy, sharp and peaky) | UNKNOWN for this ink; plausibly inside the achievable 1-9 wt% in-matrix window |
| eps_r behavior across that window | MEASURED to move: ~4 at 10 wt% to ~240 at 60 wt%, knee 37.5 wt% (Allison Ch. 3) | Not yet measured; literature says it must move (Zois 2001, Song 1986, Efros-Shklovskii 1976) |
| sigma-eps_r coupling | Both rise together through the knee | Same physics, same coupling, at 3 to 30 times lower loading |
| Implication for the fixed eps_r = 20 assumption | Contradicted by the baseline's own data | Contradicted by percolation theory and by every carbon black dataset found |

**Why the 30-35 wt% optimum is consistent with a different percolation
regime:** the optimum is not a magic number of carbon, it is "just past the
percolation knee of THAT filler in THAT matrix," where sigma is high enough
to absorb radio-frequency power but the network is not yet so dense that the
part is a reflective conductor. Carbon black reaches the equivalent knee at
far lower loading because its aggregate geometry builds a spanning network
with far less mass. The repo memory item "carbon black likely percolates far
below graphite loadings" is supported by every source found; the caveat that
it is untested FOR THIS INK stands, because threshold location within the
0.5 to 10 wt% band depends on the specific black's structure, the dispersant,
and how the jetted ink distributes through the powder bed, none of which any
paper measured.

---

## 4. Frequency caveats: 1 to 30 MHz vs the literature's frequency spread

1. **The relevant giant-permittivity literature is kHz-to-MHz, not GHz, so no
   extrapolation downward is needed.** Song 1986 spans 10 Hz to 13 MHz;
   McLachlan and Heaney 1999 spans 10 mHz to 1 MHz; Zois 2001 is an
   impedance-spectroscopy study in the same class; Allison's own instrument
   ran 40 kHz to 40 MHz. Our 1 to 30 MHz regime (27.12 MHz industrial band)
   sits inside or immediately adjacent to all of these. Established.
2. **Near percolation, eps_r decays only weakly with frequency.** Song 1986
   measured eps proportional to omega to the -0.12 near the threshold. A
   decade of frequency costs roughly 24 percent of eps_r, not an order of
   magnitude. So a permittivity enhancement seen at 1 MHz does not vanish by
   27 MHz. Established for that system; plausible inference for ours.
3. **The Maxwell-Wagner-Sillars relaxation moves with loading.** Interfacial
   polarization has a characteristic relaxation whose frequency scales with
   the cluster conductivity and geometry; as loading rises the dispersion
   region shifts and broadens. Practically: the eps_r(s) curve measured at
   1 MHz and at 27 MHz will have the same shape but different magnitudes,
   and the solver must consume the curve AT the operating frequency, not a
   band average. Established mechanism; the magnitude for this ink is
   unknown.
4. **GHz data would understate the effect.** Most microwave-band composite
   studies report smaller permittivity enhancements because interfacial
   polarization progressively freezes out above its relaxation. The census
   question lives at 27 MHz where the effect is near its largest, which cuts
   AGAINST the fixed-eps_r assumption, not for it. Plausible inference.
5. **Above-threshold samples read differently.** Once a direct-current path
   spans the electrodes, the measured apparent permittivity at low frequency
   is dominated by conduction (large imaginary part, possible negative real
   part), and instruments report it poorly. The measurement plan should
   expect clean eps_r(s) below and near the knee and a conduction-dominated
   regime past it. Established.

---

## 5. What the impedance / vector network analyzer measurement must discriminate

The planned M2 measurement (INK_DOPANT_CALIBRATION_BRIEF.md, coaxial-chamber
protocol per Allison Ch. 3, complex impedance 40 kHz to 40 MHz) settles the
question. It should be read against these specific discriminants:

1. **The D2 slope test, now with a literature prior.** The prior is that
   eps_r(s) RISES with dose across the achievable window, steeply if the
   window straddles the percolation knee. The null result the fixed-eps_r
   assumption requires is a flat eps_r(s) within rig uncertainty across all
   gray levels; the literature says that outcome would be anomalous and
   should itself be double-checked (bad contact, dose not actually varying,
   threshold far above 9 wt%).
2. **Knee coincidence.** Plot sigma(s) and eps_r(s) on the same dose axis.
   Percolation physics predicts the eps_r rise is steepest where the sigma
   rise is steepest (same pc). Coincident knees confirm the percolation
   picture and locate pc inside or outside the 1 to 9 wt% window (decision
   rule D1). Non-coincident knees would indicate a second mechanism
   (dispersant residue conduction, moisture) and would need diagnosis before
   either channel is trusted.
3. **The eps_r(s) functional form, for the actuator law D3.** Fit the
   below-threshold branch to A times (pc - p) to the power -s. If s comes
   out near 1, replace the solver's linear blend with the fitted law. If the
   window sits entirely below the knee and the curve is gentle, a linear
   blend may be an adequate local approximation; that is the ONLY outcome
   that partially rehabilitates the current co-varying law.
4. **Frequency dispersion across the band.** Record eps_r(s) at 1, 13.56,
   and 27.12 MHz at minimum. The solver needs the 27.12 MHz value; the
   dispersion between these points measures how far the Maxwell-Wagner
   relaxation intrudes on the operating band, and a strong dispersion at the
   highest doses is the expected signature of near-percolation operation.
5. **Loss tangent peak.** Allison saw tan-delta peak near 4.1 at 35 wt%
   graphite. A comparable peak versus dose for the carbon black system marks
   the most absorptive operating point and is the direct analog of his
   30-35 wt% optimum, expected at single-digit wt% here.
6. **Repeatability near the knee.** If the process operates near pc,
   dose-to-property sensitivity is maximal, which is good for grading
   authority and bad for repeatability. Replicate coupons at the same gray
   level bound the property noise the solver's robustness step must survive
   (the brief's section 6 D1 sensitivity check).

---

## 6. Evidence classification

Established literature findings:
- eps_r of a conductor-insulator composite diverges at the percolation
  threshold and co-varies with sigma below it (Efros-Shklovskii 1976; Nan,
  Shen, Ma 2010).
- This holds for carbon fillers at kHz-to-MHz frequencies specifically
  (Song 1986 to 13 MHz; McLachlan-Heaney 1999; Zois 2001 with measured
  critical exponents for BOTH properties).
- Carbon black percolates at roughly 0.5 to 10 wt% in thermoplastics
  depending on structure (Choi 2019; Zois 2001; Huang 2002).
- Micron flake graphite percolates near 11 vol%, roughly 20 wt%, in
  polyethylene (Krupa 2004), and Allison measured ~30 wt% in the Nylon 12
  powder system with eps_r moving 4 to 240 across 10 to 60 wt%.

Plausible inference (not yet measured for this ink):
- The current ink's in-matrix window (1 to 9 wt%) straddles or approaches its
  carbon black percolation threshold.
- The near-percolation permittivity enhancement survives essentially intact
  from 1 MHz to 27 MHz for this system.
- Allison's 30-35 wt% optimum maps to a single-digit wt% optimum for carbon
  black.

Unknown until M1/M2 run:
- The actual threshold location and sharpness for THIS black, dispersant,
  and jetted-into-powder-bed morphology.
- The eps_r(s) magnitude and slope at 27.12 MHz across the gray-level range,
  and therefore whether decision rule D2 promotes or retires the
  permittivity channel.
- Whether jetted deposition produces a percolation curve resembling
  melt-compounded literature composites at all; ink migration through the
  powder bed has no literature analog found in this search.

## 7. Full citation list

1. A. L. Efros and B. I. Shklovskii, "Critical behaviour of conductivity and
   dielectric constant near the metal-non-metal transition threshold,"
   Physica Status Solidi (b) 76, 475-485 (1976).
   DOI: 10.1002/pssb.2220760205.
2. C.-W. Nan, Y. Shen, and J. Ma, "Physical properties of composites near
   percolation," Annual Review of Materials Research 40, 131-151 (2010).
   DOI: 10.1146/annurev-matsci-070909-104529.
3. Y. Song, T. W. Noh, S.-I. Lee, and J. R. Gaines, "Experimental study of
   the three-dimensional ac conductivity and dielectric constant of a
   conductor-insulator composite near the percolation threshold," Physical
   Review B 33, 904-908 (1986). DOI: 10.1103/PhysRevB.33.904.
4. D. S. McLachlan and M. B. Heaney, "Complex ac conductivity of a carbon
   black composite as a function of frequency, composition, and
   temperature," Physical Review B 60, 12746-12751 (1999).
   DOI: 10.1103/PhysRevB.60.12746.
5. H. Zois, L. Apekis, and M. Omastova, "Electrical properties of carbon
   black-filled polymer composites," Macromolecular Symposia 170, 249-256
   (2001). DOI: 10.1002/1521-3900(200106)170:1<249::AID-MASY249>3.0.CO;2-F.
6. H. J. Choi, M. S. Kim, D. Ahn, S. Y. Yeo, and S. Lee, "Electrical
   percolation threshold of carbon black in a polymer matrix and its
   application to antistatic fibre," Scientific Reports 9, 6338 (2019).
   DOI: 10.1038/s41598-019-42495-1.
7. J.-C. Huang, "Carbon black filled conducting polymers and polymer
   blends," Advances in Polymer Technology 21(4), 299-313 (2002).
   DOI: 10.1002/adv.10025.
8. I. Krupa, I. Novak, and I. Chodak, "Electrically and thermally conductive
   polyethylene/graphite composites and their mechanical properties,"
   Synthetic Metals 145(2-3), 245-252 (2004).
   DOI: 10.1016/j.synthmet.2004.05.007.
9. M. Rahaman, P. Gupta, M. Hossain, et al., "Effect of carbons' structure
   and type on AC electrical properties of polymer composites: predicting
   the percolation threshold of permittivity through different models,"
   Colloid and Polymer Science 301, 1001-1019 (2023).
   DOI: 10.1007/s00396-023-05120-2.
10. In-repo baseline: Jared Allison dissertation Ch. 3 (previous-work/
    Jared_Allison_Dissertation_Final.pdf) and the journal versions cited in
    INK_DOPANT_CALIBRATION_BRIEF.md Section 7; digitized eps_r(wt%) curve at
    binderjet/code/RF_electrode_calculations.py:37-41.

Supplementary (consulted, not load-bearing): percolation-triggered negative
permittivity in nano carbon powder / polyvinylidene fluoride composites,
PMC11357240, for the above-threshold regime behavior.
