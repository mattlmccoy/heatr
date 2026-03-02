# HEATR Theory Notes

This document provides a compact but explicit theory reference for the HEATR solver stack.

## Governing Structure

HEATR advances a coupled state over time:

- electric potential and derived RF source terms
- temperature field
- melt fraction field
- relative density field

The simulation loop alternates between:

1. EQS update (periodic or event-driven)
2. thermal + phase + densification transient updates

## 1) Electro-Quasi-Static (EQS)

For harmonic excitation at angular frequency $\omega = 2\pi f$, HEATR solves:

$$
\nabla \cdot \left((\sigma + j\omega\varepsilon)\nabla V\right)=0
$$

with electrode boundary conditions prescribed by configuration.

Derived quantities:

$$
\mathbf{E}=-\nabla V, \quad |\mathbf{E}|=\sqrt{E_x^2+E_y^2}
$$

## 2) RF Volumetric Heating

Raw RF volumetric heating is taken as:

$$
Q_{\mathrm{rf,raw}}=\frac{1}{2}\sigma|\mathbf{E}|^2
$$

When generator-enforced scaling is enabled, HEATR rescales this field so integrated absorbed power over the target region matches configured generator assumptions (power and transfer efficiency). The spatial shape comes from EQS; only global magnitude is rescaled.

## 3) Thermal Transient

HEATR solves transient heat transfer with source loading and losses:

$$
\rho c_p\frac{\partial T}{\partial t}=\nabla\cdot(k\nabla T)+Q_{\mathrm{rf}}-Q_{\mathrm{loss}}
$$

with configurable convection on selected boundaries and bounded time integration for numerical stability.

## 4) Melt Fraction Model

A smooth phase transition model maps temperature to melt fraction $\phi\in[0,1]$.

Typical forms are parameterized by transition temperature and smoothing width, with latent heat represented in effective thermal response.

## 5) Densification Model

Relative density $\rho_{\mathrm{rel}}\in[0,1]$ evolves via temperature and melt-fraction dependent kinetics with per-step bounds.

HEATR supports dual-regime behavior (solid-state and liquid-assisted contributions), enabling practical process-window studies across exposure schedules.

## 6) Turntable Event Model

Turntable mode applies discrete geometry rotations at configured event times:

- `rotation_deg`
- `total_rotations`
- `rotation_interval_s` (optional)

At each event, HEATR:

1. rotates the part geometry mask
2. remaps state fields into the new orientation
3. re-solves EQS with fixed electrodes
4. continues transient advancement

Recent updates preserve thermal coherence across rotations and avoid non-physical state dilution at remap boundaries.

## Numerical Notes

- The solver is a 2D approximation with effective depth scaling assumptions.
- Stability controls include substepping, clipping, and bounded updates.
- Calibration sensitivity is highest for conductivity, coupling assumptions, and boundary losses.

## Interpretation Guidance

HEATR is intended for comparative process studies and operational optimization, not as a one-to-one replacement for full 3D multiphysics certification models.
