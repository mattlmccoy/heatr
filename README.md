# HEATR

**High-frequency Electrothermal Additive Thermal Resolver**

HEATR is a physics-based process modeling and control environment for RF-assisted additive manufacturing workflows. It couples electro-quasi-static (EQS) field solves with transient heat transfer, phase-change/melt progression, and densification kinetics to support:

- process planning
- exposure-time optimization
- turntable/rotation process studies
- structured result analysis and reporting
- browser-based operation of simulation runs

This repository is the documentation hub for HEATR. The solver and GUI implementation currently live in the sibling project at:

- `../` (geo-prewarp codebase)

## Why HEATR Exists

HEATR was designed to solve a practical RFAM problem: users need a fast, controllable surrogate for full multiphysics workflows (for example, COMSOL electrothermal studies), but with:

- explicit, editable assumptions
- repeatable batch execution
- automation-ready outputs
- lower overhead for process iteration

HEATR emphasizes *engineering usefulness*: reliable directional trends, interpretable outputs, and high-throughput comparative studies across geometry, time, and operating conditions.

## Scope

HEATR focuses on 2D cross-sectional process modeling with coupled modules:

1. EQS electric field solve
2. RF volumetric heating mapping
3. Transient thermal evolution
4. Melt-fraction progression
5. Densification progression

It supports multiple run modes:

- single exposure
- exposure sweep
- exposure optimizer
- turntable (time-scheduled rotation events)

Prewarp-specific flows are intentionally excluded from the browser GUI until finalized.

## Core Theory

### 1) Electro-Quasi-Static (EQS)

HEATR models high-frequency electric behavior using a quasi-static formulation over complex permittivity/conductivity response. In compact form:

$$
\nabla \cdot \left(\left(\sigma + j\omega\varepsilon\right) \nabla V\right) = 0
$$

where:

- $V$ is electric potential
- $\sigma$ is electrical conductivity
- $\varepsilon$ is permittivity
- $\omega = 2\pi f$ is angular frequency

Derived fields:

$$
\mathbf{E} = -\nabla V
$$

### 2) RF Heating Source

Local volumetric RF heating follows standard harmonic averaging form:

$$
Q_{\mathrm{rf}} = \frac{1}{2}\sigma\left|\mathbf{E}\right|^2
$$

In generator-enforced runs, HEATR scales the raw EQS power distribution so integrated absorbed power matches target generator-side assumptions (for example, generator power times coupling efficiency). This preserves spatial EQS shape while controlling global magnitude.

### 3) Transient Heat Transfer

Thermal evolution is solved in time with conduction, source loading, and boundary losses:

$$
\rho c_p \frac{\partial T}{\partial t} = \nabla \cdot (k\nabla T) + Q_{\mathrm{rf}} - Q_{\mathrm{loss}}
$$

Boundary treatment supports convection and configurable boundary sets. Model parameters include ambient temperature, convective coefficient, and timestep controls.

### 4) Phase/Melt Progression

HEATR uses smooth phase transition models (including COMSOL-style smoothed/Heaviside variants) to map temperature to melt fraction $\phi \in [0,1]$, with latent-heat handling in the thermal update.

### 5) Densification

Densification is represented by an effective relative density state $\rho_{\mathrm{rel}} \in [0,1]$ that evolves under thermal and melt-dependent kinetics. HEATR supports dual-regime behavior and bounded per-step changes for numerical stability.

## Turntable Physics Model

Turntable mode rotates part geometry relative to fixed electrodes using time-scheduled rotation events:

- `rotation_deg`: angular increment per event
- `total_rotations`: number of rotation events
- `rotation_interval_s`: optional explicit cadence (auto-spaced if omitted)

At each event, HEATR:

1. updates rotated part mask
2. remaps state to the new orientation
3. re-solves EQS with electrodes unchanged
4. continues transient evolution

Recent behavior includes thermal coherence improvements so rotated-state carryover does not create artificial hotspot persistence or non-physical density drops at event boundaries.

## Feature Set

### Simulation Modes

- **Single Exposure**: one configuration, one runtime horizon
- **Sweep**: batch over multiple exposure durations
- **Optimizer**: captures decision snapshots (melt and max-density ceilings)
- **Turntable**: staged rotations during one run

### GUI (Browser + Python Backend)

- shape and mode selection
- base-config matching and fallback generation
- manual base-config override
- advanced parameter override panel
- YAML inspector with highlighted important fields
- live job status and artifact updates
- results page and examples page
- settings: mode/accent/layout/background preferences

### Advanced Controls

Includes controls for electric, thermal, and materials domains, such as:

- frequency
- voltage and generator settings
- ambient and convection controls
- virgin/powder/doped material properties
- optional turntable timing and angle controls

### Outputs

Typical run directory contents:

- `used_config.yaml`
- `summary.json`
- `time_series.json`
- `fields.npz`
- static plots (`electric_fields.png`, `thermal_fields_final.png`, `time_series.png`, `paper_style_report.png`, `rf_summary_v5.png`)
- GIF artifacts (mode-dependent), for example:
  - `electric_field_evolution.gif`
  - `thermal_evolution.gif`
  - `density_evolution.gif`
  - `turntable_electric.gif`
  - `turntable_thermal.gif`
  - `turntable_density.gif`

## Configuration Philosophy

HEATR uses YAML configs as the contract between physics and operations.

Design goals:

- explicit defaults
- reproducible run records
- compare-by-diff across runs
- easy migration from GUI-generated configs to scripted batch runs

## Data and Metrics

Primary analysis dimensions:

- max/mean part temperature
- melt-fraction progression
- relative-density progression
- RF power/energy bookkeeping
- rotation schedule effects (turntable mode)

Outputs are designed for both quick visual triage and machine-readable downstream analysis.

## Known Modeling Assumptions and Limits

- 2D representation of a 3D process
- effective depth assumptions for power interpretation
- constitutive and densification simplifications relative to full multiphysics FEM
- calibration sensitivity to conductivity/coupling and boundary conditions

HEATR is best used as an engineering process tool and comparative simulator, not as a direct replacement for high-fidelity 3D multiphysics in final certification contexts.

## Repository Map (Current State)

This docs repository currently contains:

- `README.md` (this document)

Planned additions:

- theory note pages with derivations
- parameter reference tables
- runbook and troubleshooting guide
- validation/comparison studies

## Quick Start (Using Existing Implementation)

From the solver/GUI codebase (`../`):

```bash
# Launch browser GUI backend
python3 rfam_gui_server.py
```

Open:

- `http://127.0.0.1:8080` (operation page)
- `http://127.0.0.1:8080/results` (results browser)
- `http://127.0.0.1:8080/examples` (examples)
- `http://127.0.0.1:8080/theory` (in-app theory view)

## Intended Users

- process engineers exploring RFAM parameter sensitivity
- researchers benchmarking surrogate electrothermal pipelines
- operators needing repeatable scenario execution and visual result review

## Roadmap

Near-term:

- expanded docs and parameter reference coverage
- deeper validation notes (including COMSOL comparison methodology)
- scenario templates for standard study families (geometry, exposure, rotation)

Mid-term:

- tighter experiment-to-simulation traceability
- richer report packaging for publication and design reviews

---

If you use HEATR for studies or internal reports, cite the model assumptions and config snapshot (`used_config.yaml`) alongside key output metrics to preserve reproducibility.
