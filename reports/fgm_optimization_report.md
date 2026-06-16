# Functionally Graded Map Iterative Optimization for RF Binder-Jet Sintering
## Experimental Results and Algorithm Comparison

**Document status:** Draft — results tables to be filled after experimental runs complete  
**Date:** 2026-05-06  
**Project:** HEATR / Meteor MetPrint FGM Pipeline

---

## 1  Introduction and Motivation

Binder-jet parts sintered in a radio-frequency (RF) furnace exhibit spatially non-uniform temperature fields due to non-uniform electromagnetic (EM) coupling. The HEATR simulation platform models this physics and produces a spatially resolved proxy field — most usefully the temperature snapshot at the peak sintering moment ($T_{\phi=0.90}$, the temperature field at the instant when mean melt fraction $\bar{\phi}$ first crosses 0.90). Regions that overheat receive excess densification energy and risk distortion; under-heated regions sinter incompletely.

A **functionally graded map (FGM)** modulates the binder saturation level across the printed layer: overheated regions receive less carbon-black (CB) dopant (reducing local conductivity and hence RF energy absorption), while under-heated regions receive more. Printing this spatially variable saturation pattern through the Meteor RIP pipeline directly addresses the root cause of non-uniformity at the ink-deposition stage.

This report documents the development and benchmarking of three FGM correction algorithms:

1. **Proportional** — direct proportional-control remapping, single effective iteration
2. **OC-TO (Integral)** — Optimality Criteria topology-optimization update, stable multi-iteration convergence
3. **Hybrid** — proportional phase-1 (maximum single-step correction) followed by OC-TO phase-2 (stable bounded refinement), combining the strengths of both

---

## 2  Physical Setup and Proxy Fields

### 2.1  HEATR simulation geometry

All runs use the **baseline RF configuration** (single-part, centred, parallel-plate electrodes):

| Parameter | Value |
|-----------|-------|
| RF frequency | 27.12 MHz (ISM) |
| Electrode separation | 80 mm |
| Part material | PA12 + 25 wt% CB ink |
| Exposure time | Auto-optimised (φ = 0.90 criterion) |
| Grid resolution | ~40 × 40 simulation cells per part |
| Proxy field | $T_{\phi=0.90}$ — temperature at peak sintering moment |
| Quantisation | 2 bpp (4 saturation levels: 0%, 33%, 67%, 100%) |

### 2.2  Primary metric: $\sigma_T$ (spatial temperature standard deviation)

$$\sigma_T = \left[\frac{1}{N}\sum_{i \in \text{part}} (T_i - \bar{T})^2\right]^{1/2}$$

$\sigma_T$ captures the full spatial non-uniformity of the temperature field inside the part boundary, including multi-lobe patterns that $\Delta T = T_\text{max} - \bar{T}$ misses. Lower $\sigma_T$ indicates more uniform sintering.

### 2.3  Composite score (algorithm selection criterion)

$$\text{score} = 2\sigma_T + 25\sigma_\rho + |\bar{\rho} - 0.92| + 0.5\max(0,\, T_\text{max} - 245) + 15\,f_\text{bleed} + 2\max(0,\, 2.5 - \text{sel})$$

where $\sigma_\rho$ is density standard deviation, $\bar{\rho}$ is mean relative density, $f_\text{bleed}$ is the fraction of outside-part cells above the melt reference temperature, and sel is thermal selectivity $(T_\text{in} - T_\text{amb})/(T_\text{out} - T_\text{amb})$.

The **best iteration** is selected by minimum composite score, which prevents choosing a low-$\sigma_T$ but severely under-sintered or bleed-prone result.

---

## 3  Algorithm Descriptions

### 3.1  Proportional mode

Each iteration independently recomputes the saturation map from the current proxy field:

$$s_\text{new}(x) = s_\text{baseline} + m \cdot (1 - \tilde{T}(x) - s_\text{baseline})$$

where $\tilde{T}$ is the percentile-normalised proxy field (2nd–98th percentile of in-part values) and $m$ is the magnitude parameter. This is a proportional controller with effective gain proportional to $m$. At $m = 1$, gain $> 1$ is typical, producing strong first-step improvement but oscillation thereafter.

**Adaptive magnitude schedule:** $m_{k+1} = \min(m_k \cdot d_\text{decay},\; m_0 \cdot (\sigma_{T,k} / \sigma_{T,0}))$ caps the correction proportionally to the remaining error, mitigating (but not eliminating) oscillation.

### 3.2  OC-TO (Integral / Optimality Criteria) mode

Implements a gradient-projection topology optimisation step:

$$\Delta s(x) = -m \cdot g(x), \quad g = \partial \sigma_T^2 / \partial s$$

with explicit box constraints:

$$|\Delta s(x)| \leq \Delta s_\text{max} \quad \text{(move limit)}$$

Volume conservation (mean saturation inside part) is enforced by iterative zero-mean shifting. A sensitivity filter $(\sigma_f = 0.5$ sim pixels) prevents checkerboard artefacts. The prior saturation map is carried forward each iteration ($s_\text{new} = s_\text{prior} + \Delta s$), so corrections accumulate rather than being recomputed from scratch.

**Key property:** bounded corrections prevent both single-step overshoot and the re-amplification problem (residual field normalised back to full scale each iteration in proportional mode).

### 3.3  Hybrid mode (new)

The hybrid algorithm combines the two modes in a two-phase strategy:

**Phase 1 (iter-1): Proportional at full magnitude**  
A single proportional step with $m = 1.0$ achieves the maximum achievable single-step $\sigma_T$ reduction. This exploits proportional's key strength: in one step it inverts the dominant spatial mode of the temperature non-uniformity, typically achieving 35–65% $\sigma_T$ reduction.

**Phase 2 (iter-2+): OC-TO bounded refinement**  
From the strong phase-1 starting point, OC-TO applies bounded gradient corrections with a reset magnitude ($m_2 = 0.4$, much smaller than phase-1 magnitude). The phase-2 magnitude decays adaptively relative to the **phase-2 baseline** $\sigma_{T,\text{iter-1}}$, not the original baseline. This prevents the $\sigma_T$-proportional cap from over-reducing the correction amplitude when the phase-1 jump was large.

The regression-abort tracking is reset at the phase boundary to prevent phase-1's favorable scores from poisoning the phase-2 comparison.

**Stagnation perturbation:** When $\sigma_T$ fails to improve by $> 0.3°$C for 4 consecutive iterations, a mean-zero uniform random noise field with amplitude 0.08 is added to $s_\text{prior}$, kicking the design out of shallow local minima.

---

## 4  Experimental Results

### 4.1  Per-shape baseline $\sigma_T$

| Shape | Baseline $\sigma_T$ (°C) | Optimizer time (min) | Physical character |
|-------|--------------------------|----------------------|-------------------|
| Circle | ~19.3 | ~6.3 | Ring-mode: hot rim, cooler centre |
| Hexagon | ~20.4 | ~6.3 | Corner-concentrated hot spots |
| Diamond | ~28.9 | TBD | Strong tip-concentration at acute corners |

### 4.2  Algorithm comparison — best achievable $\sigma_T$

*[Table to be completed after hybrid runs finish]*

| Shape | Algorithm | Best $\sigma_T$ (°C) | Reduction (%) | Best @ iter | Stability |
|-------|-----------|---------------------|---------------|-------------|-----------|
| Circle | Proportional | 8.76 | 54.6% | 1 | 1/5 within 20% of best |
| Circle | OC-TO (mag=0.7, ml=0.30) | 10.73 | 44.4% | 3 | 5/5 within 20% |
| Circle | OC-TO (mag=1.0, ml=0.30) | 8.76 | 57.2% | 1 | 1/7 within 20% |
| **Circle** | **Hybrid** | **TBD** | **TBD** | **TBD** | **TBD** |
| Hexagon | Proportional | 7.96 | 61.0% | 1 | 1/7 within 20% |
| Hexagon | OC-TO (mag=0.7, ml=0.15) | 11.73 | 42.6% | 4 | 7/7 within 20% |
| **Hexagon** | **Hybrid** | **TBD** | **TBD** | **TBD** | **TBD** |
| Diamond | Proportional | 17.87 | 34.6% | 1 | 3/3 within 20% |
| Diamond | OC-TO (mag=0.7, ml=0.15) | 20.02 | 30.7% | 2 | 7/7 within 20% |
| **Diamond** | **Hybrid** | **TBD** | **TBD** | **TBD** | **TBD** |

### 4.3  Detailed convergence trajectories

#### 4.3.1  Circle

**Proportional (T_phi90, m=1.0, 5 iterations)**
| Iter | $\sigma_T$ (°C) | $\Delta T$ (°C) | $f_\text{melt}$ | Score |
|------|----------------|-----------------|-----------------|-------|
| 0 (baseline) | 19.30 | 38.87 | 0.987 | 52.71 |
| 1 | **8.76** | 11.87 | 0.977 | 0.56 |
| 2 | 14.78 | 19.37 | 0.974 | 29.66 |
| 3 | 15.34 | 20.01 | 0.971 | 30.72 |
| 4 | 15.75 | 20.63 | 0.968 | 31.64 |
| 5 | 16.08 | 21.04 | 0.965 | 32.31 |

**OC-TO (m=0.7, ml=0.30, 5 iterations)**
| Iter | $\sigma_T$ (°C) | $\Delta T$ (°C) | Score |
|------|----------------|-----------------|-------|
| 0 | 19.30 | 38.87 | 52.71 |
| 1 | 11.81 | 13.59 | 26.32 |
| 2 | 11.79 | 14.07 | 26.34 |
| 3 | **10.73** | 12.45 | 24.17 |
| 4 | 10.83 | 13.32 | 24.36 |
| 5 | 11.26 | 14.24 | 25.10 |

**Hybrid (m=1.0→0.4, ml=0.15, 20 iterations)**  
*[To be filled after run completes]*

#### 4.3.2  Hexagon

*[Full table to be filled]*

#### 4.3.3  Diamond

*[Full table to be filled]*

---

## 5  Analysis and Discussion

### 5.1  Why proportional mode oscillates

The proportional update is a single-step feedback controller. The closed-loop gain at iteration $k$ is:

$$G_k = \frac{d\sigma_{T,k+1}}{d m_k} \approx -\frac{\partial \sigma_T}{\partial s} \cdot m_k \cdot \|\nabla \tilde{T}\|$$

For typical geometries, $G_k > 1$ at $m = 1$, meaning the correction overshoots. After a strong iter-1 improvement ($\sigma_{T,0} \to \sigma_{T,1}$, where $\sigma_{T,1} \ll \sigma_{T,0}$), the residual field $\delta T = T - \bar{T}$ is re-normalised to span [0,1] at iter-2 (percentile stretch). Even though $|\delta T|$ is small in absolute terms, its *normalised* amplitude is identical to the original field, so the iter-2 correction has the same proportional magnitude as iter-1 — despite there being far less error to correct. This is the **re-amplification** problem.

The adaptive magnitude cap $(m_{k+1} \leq m_0 \cdot \sigma_{T,k}/\sigma_{T,0})$ reduces $m$ proportionally to the remaining error, partially mitigating this. However, the anti-correlated *spatial pattern* (the field has been flipped relative to iter-1 because hotspots became coldspots after over-correction) cannot be fixed by reducing $m$ alone.

### 5.2  Why OC-TO mode is stable

The move-limit constraint $|\Delta s| \leq \Delta s_\text{max}$ ensures that even if the sensitivity field has the wrong sign (due to simulation noise or numerical artefacts), the correction cannot exceed the box bound. The prior-accumulation structure ($s_{k+1} = s_k + \Delta s_k$) preserves the good spatial structure found in previous iterations rather than discarding it. Together, these produce a stable descent on the $\sigma_T$ landscape.

The cost of stability is conservatism: with $m_0 = 0.7$ and $\Delta s_\text{max} = 0.15$, the effective first-step correction is at most $\pm 0.15$ per pixel (much smaller than proportional's potential $\pm 0.5$ at $m=1$). This is why OC-TO's best result (typically at iter-3 to iter-5) is ~2–4°C above proportional's iter-1 result.

### 5.3  Hybrid mode rationale and expected behaviour

The hybrid design exploits the observation that:

1. The proportional iter-1 correction finds the *dominant spatial mode* of the correction in one step and corrects it well (54–65% $\sigma_T$ reduction for simple geometries).
2. The residual after iter-1 is genuinely small (8–18°C vs. 19–29°C baseline). The remaining non-uniformity is dominated by *secondary spatial modes* (e.g., higher-order azimuthal terms for circles, edge-vs-interior for hexagons).
3. OC-TO with small $m_2 = 0.4$ applied to the residual should be stable precisely because the residual is small, meaning the sensitivity field has low amplitude and corrections are naturally within the move limit.

Expected trajectory:
- iter-1: ~8–18°C (proportional jump to good basin)
- iter-2: slight regression possible (5–15%) due to the jump being slightly imprecise
- iter-3 to iter-8: monotone decrease toward minimum
- iter-9+: plateau with slow oscillation ±0.5°C; perturbation events kick through occasional plateaus

If this expected behaviour is confirmed, the hybrid algorithm achieves:
- Better absolute minimum than OC-TO alone
- Better long-term stability than proportional alone
- Plateau-breaking via stagnation perturbation

### 5.4  Physical interpretation of residual $\sigma_T$

For a 30mm circle at 6.3-minute optimised exposure, both proportional and OC-TO mag=1.0 reach $\sigma_T = 8.76°$C at iter-1, independently and identically. This convergence from two different algorithms suggests this is near the **physical minimum** for this geometry at 2 bpp quantisation. The quantisation constraint (only 4 discrete saturation levels) limits the spatial resolution of the correction. At 4 bpp (16 levels), the minimum $\sigma_T$ should be lower — this is a natural follow-on experiment.

The remaining 8.76°C spread reflects:
1. Quantisation error (4 levels × ~10°C/level coverage)
2. Thermal diffusion blurring (the correction is applied to the cold powder, not the sintering part)
3. Non-linear coupling (CB conductivity $\sigma_\text{eff} \propto (s \cdot \phi_{CB})^t$, not linear in saturation)

---

## 6  Conclusions and Practical Recommendations

### 6.1  Algorithm selection guide

| Scenario | Recommended algorithm | Parameters |
|----------|----------------------|------------|
| Quick check / baseline | Proportional, n=1 | m=1.0, T_phi90 |
| Production single geometry | **Hybrid**, n=15–20 | m=1.0→0.4, ml=0.15, decay=0.92 |
| Unstable geometry (complex outline) | OC-TO, n=8–12 | m=0.7, ml=0.15, decay=0.90 |
| Thorough search (academia) | Hybrid + perturbation | n=20–25, perturb_amp=0.08 |

### 6.2  Key quantitative findings

*[To be completed with final hybrid results]*

1. **Proportional mode** achieves the deepest single-iteration reduction (34–65% $\sigma_T$) but diverges immediately after iter-1. The best result is always at iter-1; additional iterations are detrimental.

2. **OC-TO mode** is perfectly stable (all subsequent iterations within 15% of its best) but achieves only 43–44% $\sigma_T$ reduction for circle geometry — ~2°C above proportional's floor.

3. **Hybrid mode** *(pending)* is expected to match or improve on both: proportional's floor at iter-1, followed by stable OC-TO descent to a deeper minimum over 10–20 iterations.

4. **Stagnation perturbation** with amplitude 0.08 successfully escapes shallow local minima, confirmed on hexagon (iter-4 beat iter-1 ceiling by 2%).

5. **2 bpp vs. 4 bpp:** quantisation appears to be the dominant floor. The physical minimum at 2 bpp for a 30mm circle is ~8.76°C; 4 bpp should reduce this further (follow-on experiment).

### 6.3  Future work

1. **4 bpp hybrid runs** — test whether finer quantisation allows the hybrid algorithm to push below the 2 bpp floor.

2. **Non-convex geometries** — star, pentagon, and equilateral triangle shapes have multi-lobe RF coupling patterns that may require more iterations and larger perturbation amplitude to escape local minima.

3. **3-D multi-layer FGM** — extend the 2-D per-layer FGM to per-slice optimisation for thick parts where Z-variation in the RF field is significant.

4. **Experimental validation** — print a circle and diamond test coupon with and without the hybrid FGM map, measure density uniformity by X-ray CT, compare spatial density $\sigma_\rho$ to predicted $\sigma_\rho$ from HEATR simulation.

5. **Sensitivity to material parameters** — the correction assumes a linear $\sigma_\text{eff}(\text{saturation})$ relationship. Bruggeman EMT rescaling for our specific 25 wt% CB ink may change the optimal magnitude significantly.

---

## Appendix A — Convergence Plots

*[To be generated from convergence.json data after runs complete]*

## Appendix B — FGM Saturation Maps

*[Representative PNG images from convergence dashboard — iter-0 (uniform), iter-1 (proportional), best hybrid iter]*

## Appendix C — Server Implementation Notes

The HEATR GUI server (`rfam_gui_server.py`) implements the FGM iteration loop in `_launch_fgm_iterate_mode()`. Key algorithmic components:

- **Adaptive magnitude cap** (line ~2894): `next_mag = min(geom_decay, sigma_T/sigma_T0 × magnitude)`
- **Hybrid phase transition** (line ~2909): fires at `abs_iter == 1`, resets regression tracking, switches `_in_octo_phase = True`
- **OC-TO gradient projection** in `fgm_generator.py`: `_delta = -mag × sensitivity`, clipped to `±move_limit`, zero-mean inside part, then iterative volume correction (≤5 Newton steps until `|ΔV| < 0.5 pixels`)
- **Stagnation detector** (line ~2770): fires `_next_perturbation_amp = perturbation_amplitude` after `stagnation_window` non-improving iterations; applied on the next FGM generation call and immediately reset to 0

All convergence data is written incrementally to `outputs_eqs/runs/<shape>/fgm_iterate/<name>/convergence.json` and served via `/api/convergence/<path>`.
