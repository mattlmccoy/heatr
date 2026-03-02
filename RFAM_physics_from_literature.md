# RFAM Physics Baseline (from Prior Papers)

This note consolidates quantitative inputs and modeling guidance extracted from:

- `Computational_design_strategy_to_improve_RF_heating_uniformity_print.pdf`
- `Jared_Allison_Dissertation_Final.pdf`
- `Seepersad-volumetric_fusion_of_graphite_doped_nylon_10-1108_RPJ-09-2020-0218.pdf`

## Core physical picture

- RFAM heating is dominated by electric-field losses in graphite-doped nylon domains, while virgin nylon remains nearly transparent at RF frequencies.
- Non-spherical doped geometries distort the local electric field and produce hot spots (especially corners/edges) and cooler zones on some faces.
- Heating uniformity can be improved by:
  - functionally grading conductivity (dopant distribution), and/or
  - changing field orientation (multi-stage electrode configuration / part reorientation).
- Defense presentation emphasis: plate placement is first-order because field strength follows `E = ΔV/d` between parallel electrodes.

## Key parameters for 27.12 MHz modeling

- Frequency: `27.12 MHz`
- Reported bench generator settings in defense: approximately `±1200 Vrms`, `560 W`
- Electrical properties used in COMSOL studies:
  - Doped region conductivity: `~0.04 S/m` (nominal simulation value in tuning studies)
  - Virgin region conductivity: `~0 S/m`
  - Relative permittivity: virgin `~2`, doped `~13.8`
- Thermal/phase-change values used in modeling nylon 12:
  - Melt temperature: `~180 C`
  - Latent heat: `~96.7 kJ/kg`
  - Thermal conductivity: solid `~0.1 W/mK`, liquid `~0.26 W/mK`
  - Density: solid `~490 kg/m^3`, liquid `~1010 kg/m^3`
  - Heat capacity: solid `~1287 J/kgK`, liquid `~2500 J/kgK`
- Convection coefficient: `~25 W/m^2K`

## Clean equation set from RPJ 2022 (10-1108_RPJ-08-2021-0193)

The paper defines these governing equations (numbering preserved). Eq. (14) does not appear in the manuscript text.

Electrical model:

- (1) `n · J = 0` (insulating electrical boundary)
- (2) `div(J) = 0`
- (3) `E = -grad(V)`
- (4) `J = (sigma + j*omega*eps0*epsr) E`
- (5) `Qrh = Re{J · E}` (resistive RF volumetric heating source)

Thermal model:

- (6) `rho*Cp*(dT/dt) + div(q) = Qrh`
- (7) `q = -k*grad(T)`
- (8) `-n · q = 0` (thermal insulation)
- (9) `-n · q = h*(Text - T)` (convection boundary)

Apparent heat capacity / phase change:

- (10) `rho = theta*rhoS + (1 - theta)*rhoL`
- (11) `k = theta*kS + (1 - theta)*kL`
- (12) `alpha_m = 0.5 * ((1 - theta)*rhoL - theta*rhoS) / (theta*rhoS + (1 - theta)*rhoL)`
- (13) `Cp = (1/rho)*(theta*rhoS*Cp,S + (1 - theta)*rhoL*Cp,L) + L*(d(alpha_m)/dT)`

Heuristic conductivity tuning:

- (15) `sigma_i^(k+1) = sigma_i^(k) + Ki * (DeltaT_i / DeltaTmax)`
- (16) `DeltaT_i = Ttarget - Ti`
- (17) `DeltaTmax = max(Ttarget - Ti)`

Uniformity metric:

- (18) `UI = (1/(Tav - Tinitial)) * (1/V) * integral_V( sqrt((T - Tav)^2) dV )`
  - equivalent form: `UI = (1/(Tav - Tinitial)) * (1/V) * integral_V( |T - Tav| dV )`

## Dopant-composition trends (nylon 12 + graphite)

- Percolation onset is around `~30 wt%` graphite.
- Loss tangent peaks around `~35 wt%` graphite.
- Strong heating reported around `~32.5-35 wt%` mixtures.
- Reported effective conductivity example at 27.12 MHz for 30 wt%:
  - total `~0.0161 S/m`, with mixed conductive/polarization loss contributions.

## Heuristic tuning method used in prior RFAM work

- Iterative conductivity update per node:
  - `sigma(i+1) = sigma(i) + K(i) * DeltaT(i)/DeltaTmax`
  - `DeltaT(i) = T_target - T(i)`
- If a node crosses target temperature between iterations, reduce local gain `K(i)` (halving strategy) to stabilize convergence.
- Reported practical setup:
  - Start near conductivity that maximizes heating response (`~0.0425 S/m` in the reported simulation curve).
  - Typical convergence around ~15 FE calls depending on geometry.

## Uniformity metric

- Uniformity index (UI) used in studies:
  - normalized RMS temperature deviation over doped volume.
  - lower UI = better uniformity.

## Design guidance (from simulation + experiments)

- Favor rounded features over sharp corners.
- Align thin members/features parallel to field direction when possible.
- Reduce electrode spacing (within arcing constraints) to improve field strength and uniformity.
- Higher voltage can improve tuning effectiveness but increases arcing risk.
- For symmetric parts, multi-orientation heating can substantially improve uniformity without heavy grading.
- Best performance often from combining functional grading with multi-orientation heating.
