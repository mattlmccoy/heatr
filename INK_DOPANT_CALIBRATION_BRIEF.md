# Ink Dopant Calibration Brief: dose -> (sigma, eps_r) for the carbon-black ink

Date: 2026-08-01
Status: DRAFT for Matt's review; feeds the P1 experiment plan (rfam-experimentalist)
Decision it feeds: whether the permittivity design channel is deployable or
model-only, and what actuator law s -> (sigma, eps_r) both the 2-D and 3-D
solvers consume. Today both solvers ASSUME this law (eps_r fixed at 20,
sigma proportional to dose, rfam_eqs_coupled.py:290-292); nothing measures it
for the current ink.

## 1. Why the existing baseline does not transfer

All property assumptions trace to Jared Allison's graphite work, where loading
means WEIGHT PERCENT GRAPHITE IN THE NYLON MATRIX and impedance spectroscopy
put the useful percolation window at 30-35 wt% (sharp, "peaky", no shoulders).
The current ink is different in two ways:

1. Units: it is 25 wt% CARBON BLACK IN THE INK (balance IPA + dispersant),
   jetted onto the bed. The in-matrix loading after the IPA evaporates is set
   by a mass balance (section 3) and is far below 25 wt%.
2. Filler morphology: carbon black is not graphite. High-structure carbon
   black typically percolates at low single-digit wt% in polymer matrices,
   versus tens of wt% for graphite flakes. (Literature-typical claim, not yet
   verified for this ink; this brief exists to test it.) If true, the process
   may operate NEAR the percolation knee, where dose-to-sigma sensitivity is
   highest: good for grading authority, bad for repeatability.

Consequence: Jared's 30-35 wt% optimum and any property value keyed to it
cannot be assumed for the carbon-black ink. Reassessment is required before
solved dopant maps are treated as printable physics.

## 2. Known process inputs

| input | value | status |
|---|---|---|
| printhead | Xaar 2002 Aquinox, 720 dpi | confirmed |
| native drop volume | 12 pL (possibly 6 pL) | UNCONFIRMED - Matt to check datasheet; every predicted loading below scales linearly with this |
| ink | 25 wt% carbon black in IPA | confirmed (printing-constraints record) |
| ink density | ~0.92 g/cm3 | computed from 25 wt% CB (rho ~1.85) in IPA (0.786); replace with a measured value |
| layer height | 100 um typical, 100-200 um range | confirmed |
| bed packing fraction | ~0.44 assumed (poured apparent density of PA12 SLS powders, ~0.44-0.45 g/cm3 class, NOT tapped; solid rho 1.01 g/cm3) | literature-typical, replace with a measured apparent density of OUR powder |
| grayscale | 4 bpp; s>1 = double pass | confirmed |

## 3. Predicted in-matrix loading (mass balance, to be replaced by M1 data)

Per unit area, one full-coverage pass at 720x720 dpi:
- drops/m2 = (720/0.0254)^2 = 8.03e8
- CB deposited = drops/m2 x V_drop x rho_ink x 0.25
- nylon in layer = t_layer x phi_pack x 1010 kg/m3
- wt% in matrix = CB / (CB + nylon)

| V_drop | layer | passes | CB g/m2 | nylon g/m2 | predicted wt% |
|---|---|---|---|---|---|
| 12 pL | 100 um | 1 | 2.2 | 44.4 | 4.7% |
| 12 pL | 100 um | 2 | 4.4 | 44.4 | 9.1% |
| 6 pL | 100 um | 1 | 1.1 | 44.4 | 2.4% |
| 12 pL | 200 um | 1 | 2.2 | 88.9 | 2.4% |
| 6 pL | 200 um | 1 | 1.1 | 88.9 | 1.2% |

Every entry is an ORDER-OF-MAGNITUDE prediction (drop volume unconfirmed,
packing assumed). The point it makes survives the uncertainty: the achievable
window is roughly 1-9 wt%, an order of magnitude below the graphite optimum,
and plausibly straddling a carbon-black percolation threshold.

## 4. Coupon matrix

Substrate: nylon 12 powder layers at 100 um (the typical height), spread by
the normal recoater path so packing matches production, on a removable carrier.

| factor | levels |
|---|---|
| 4 bpp gray level | 0 (undoped control), 4, 8, 12, 15 (full) |
| passes | 1; plus double pass at level 15 |
| replicates | 3 per cell minimum (percolation-region cells 5 if budget allows) |

7 dose conditions x 3-5 replicates = 21-35 coupons, single layer each, plus a
short stack (5-10 layers) at level 15 for through-thickness measurements if
the single-layer geometry defeats the fixture.

## 5. Measurements

M1 - deposited-dose ground truth (kills the unit ambiguity permanently):
mass gain per coupon (microbalance) and/or TGA burn-off -> actual wt% CB in
matrix per gray level. Also fixes the drop-volume question independently of
the datasheet: measured CB mass / (drops x 0.25 x rho_ink) = V_drop.

M2 - impedance spectroscopy on the SAME coupons (same method class as the
graphite percolation study): complex impedance sweep spanning the rig's
operating frequency, reduced to sigma(s) and eps_r(s). Deliverables: the
percolation curve for carbon black in this matrix (threshold location and
sharpness vs Jared's peaky 30-35 wt% graphite result) and the eps_r(s) slope.

M3 (deferred until S2 motivates it): repeat M2 on consolidated (melted)
coupons for the densification shift of both properties.

## 6. Decision rules (pre-registered)

D1: if the measured percolation threshold lies INSIDE the achievable 1-9 wt%
window, the process operates on the knee: adopt the measured sigma(s) curve in
both solvers and add a dose-noise sensitivity check to the FGM verification
step. If the threshold lies ABOVE the achievable window, the ink cannot reach
useful conductivity at current dose limits; escalate (ink loading, passes, or
filler change) before any printed FGM validation.

D2: if eps_r(s) varies by more than the impedance rig's stated uncertainty
across the achievable window, the permittivity channel is promoted from
model-only to deployable and the solvers' fixed eps_r=20 assumption is
replaced by the measured curve. Otherwise conductivity-only stands and the
2-D lane's eps-channel results stay model-only permanently.

D3: the measured s -> (sigma, eps_r) table becomes the single actuator
contract consumed by rfam_eqs_coupled, heatr3d, and solve3d; no solver keeps
a private assumption.

## 7. Baseline sources (located 2026-08-01; PDF only, no raw data files exist)

- PRIMARY: Jared Allison dissertation, previous-work/Jared_Allison_Dissertation_Final.pdf,
  Ch. 3 "Permittivity Measurements of Graphite-Doped Nylon 12 Powders". HP 4194A
  impedance analyzer, coaxial chamber (a=1.59 cm, b=7.94 cm, Teflon base),
  40 kHz-40 MHz, parasitic de-embedding (R=3.06 ohm, L=119 nH, C=5.22 pF),
  10-60 wt% graphite (Arkema 42 um nylon 12, Loud Wolf 44 um graphite).
  Percolation ~30 wt%; eps_r knee at 37.5 wt%; tan-delta peak ~4.1 at 35 wt%;
  30 wt%: sigma_DC 0.0103 / sigma_eff(27.12 MHz) 0.0161 S/m.
  Journal versions: previous-work/Seepersad-volumetric_fusion...RPJ-09-2020-0218.pdf
  (Ch. 3) and Computational_design_strategy...RPJ-08-2021-0193.pdf (Ch. 4;
  Table 1: sigma_doped 0.04 S/m, eps_r doped 13.8, virgin eps_r 2, 27.12 MHz).
  This coaxial-chamber protocol is the fixture template for M2.
- Best transcription without opening PDFs:
  dissertation_materials/heatr_audit/findings/COMSOL_validation_basis.md.
- Only machine-readable eps_r(wt%) curve (hand-digitized from Fig. 3.6):
  binderjet/code/RF_electrode_calculations.py:37-41 -
  wt% [10,20,30,33,35,37.5,40,50,60] -> eps_r [4,5,10,12,13,16,40,140,240].
- PROVENANCE FLAG: the solvers' eps_r=20 (rfam_eqs_coupled.py:290-292) and the
  paper tex property table (rfam_prewarp_paper.tex:158,472) do NOT trace to any
  Allison measurement - published doped value is 13.8, and the in-repo A/B
  study (outputs_eqs/jared_exp1_match/EPS_SENSITIVITY.md, OVERNIGHT_REPORT.md:33-40)
  found 13.8 fits better. Also stale: impedance_calculations.py:12-14 uses 150
  (~50 wt% on the curve). MISSION_CONTROL.md Tier-3 VNA sweep is the open task
  this brief's M2 supersedes/implements for the CB ink.

## 8. Risks and limits

- Single-layer coupons are thin for contact impedance fixtures; the short
  stack in section 4 is the fallback geometry.
- IPA/dispersant residue may not fully leave the bed at room temperature;
  TGA distinguishes CB from residue, mass gain alone does not.
- Bed packing under the recoater may differ from poured apparent density;
  measure it (known mass over known layer volume) rather than assuming.
- Rig is down: nothing here needs the RF apparatus, only the printer,
  balance/TGA, and an impedance analyzer, so this can run pre-rig-up.
