# HEATR

**High-frequency Electrothermal Additive Thermal Resolver**

HEATR is a browser-driven and scriptable electrothermal process-simulation tool for
RF additive manufacturing studies. It couples a 2D electro-quasi-static (EQS) solve
to transient heat transfer, phase transition, and relative-density evolution.

This repository contains:
- the production solver (`rfam_eqs_coupled.py`)
- the HEATR browser UI (`rfam_gui_server.py` + `webui/`)
- run/config plumbing for sweeps, optimizers, and turntable studies
- generated result datasets and examples

## What HEATR Is For

HEATR is intended for process-window exploration, comparative studies, and control
tuning where you need fast iteration with physically informed coupling:
- how shape changes electric-field concentration
- how heating evolves over exposure time
- how turntable motion changes thermal uniformity and densification
- how generator settings and material parameters shift outcomes

## Core Theory (EQS -> HT -> Phase -> Density)

### 1) Electro-Quasi-Static Field Solve

For harmonic drive frequency \(f\), with \(\omega = 2\pi f\), HEATR solves:

\[
\nabla \cdot \left((\sigma + j\omega\varepsilon)\nabla V\right) = 0
\]

where:
- \(V\): complex electric potential
- \(\sigma\): electrical conductivity
- \(\varepsilon\): permittivity
- \(\mathbf{E} = -\nabla V\): electric field

### 2) RF Volumetric Heating

The raw time-averaged Joule heating is:

\[
Q_{\mathrm{rf,raw}} = \frac{1}{2}\sigma |\mathbf{E}|^2
\]

If generator-enforcement is enabled, HEATR rescales the magnitude so total absorbed
power matches configured generator assumptions (e.g., generator power and coupling
efficiency). This does **not** change spatial EQS shape, only global scale.

### 3) Transient Heat Transfer

Temperature is advanced by:

\[
\rho c_p \frac{\partial T}{\partial t} =
\nabla \cdot (k \nabla T) + Q_{\mathrm{rf}} - Q_{\mathrm{loss}}
\]

including configurable boundary loss terms and numerically bounded stepping.

### 4) Melt Fraction

A smooth transition model maps temperature to melt fraction:

\[
\phi(T) \in [0,1]
\]

with configurable transition temperature and smoothing width (including latent-heat
style effects through effective response terms).

### 5) Relative Density Evolution

Relative density \(\rho_{\mathrm{rel}} \in [0,1]\) evolves with temperature and melt
state under bounded kinetics, allowing both solid-state and liquid-assisted regimes.

## Turntable Modeling

In turntable mode, geometry is rotated at event times set by:
- `rotation_deg`
- `total_rotations` (count of rotation events at `rotation_deg`, not count of full 360 deg revolutions)
- optional `rotation_interval_s`

At each rotation event HEATR remaps state fields, re-solves EQS with fixed electrodes,
and continues thermal advancement while preserving thermal coherence.

## HEATR GUI Features

The browser UI exposes:
- shape-first workflow with automatic base-config suggestion
- manual base-config override when suggested config is not desired
- run types: single exposure, time sweep, optimizer, turntable
- advanced parameter overrides (electrical, thermal, material, solver)
- YAML inspector with highlighted key values
- live run logs and live result updates while jobs execute
- dedicated results page with grouped browsing and in-page media viewing
- examples page sourced from existing sweeps/shapes outputs
- settings panel (accent/background/mode/layout preferences)

Prewarp controls are intentionally excluded from the main UI workflow at this stage.

## Quick Start

### Launch GUI

```bash
python3 rfam_gui_server.py
```

Open:
- `http://127.0.0.1:8080/`
- results: `http://127.0.0.1:8080/results`
- examples: `http://127.0.0.1:8080/examples`
- theory: `http://127.0.0.1:8080/theory`

### Run from CLI

```bash
python3 rfam_eqs_coupled.py \
  --config configs/rfam_eqs_xz_uniform_500w.yaml \
  --output-dir outputs_eqs/xz_uniform_6min_v2
```

## Repository Layout

```text
heatr/
  rfam_eqs_coupled.py          # main solver: EQS + thermal + melt + density + turntable
  rfam_gui_server.py           # HEATR backend, job runner, API routes, results indexing
  shapes.py                    # parametric shape generation utilities used by solver/UI
  rfam_spike_sweep.py          # spike-height/co-optimization sweep driver for geometry studies
  make_rf_summary_v5.py        # consolidated publication-style RF/thermal summary figure builder
  make_uniformity_comparison.py# compares runs on thermal/electric uniformity metrics
  make_H_comparison.py         # H-shape specific comparative plotting/analysis script
  extract_comsol_qrf_2d.py     # import/down-project COMSOL Q_rf maps for HEATR use
  rfam_prewarp.py              # prewarp pipeline (kept in repo; not exposed in primary GUI flow)
  rfam_prewarp_calibrate.py    # prewarp calibration/fit helper utilities
  rfam_prewarp_paper.tex       # manuscript/report source for prewarp/electrothermal writeup
  RFAM_physics_from_literature.md  # literature-grounded physics notes and assumptions
  README.md                    # project overview, theory, usage, architecture

  webui/
    static/index.html          # main HEATR control page
    static/app.js              # primary UI logic, run wiring, live status updates
    static/results.html        # dedicated results browser page
    static/results.js          # results grouping/rendering logic (images/gifs/metrics)
    static/examples.html       # examples gallery page
    static/examples.js         # examples data loading/rendering
    static/theory.html         # in-app theory page with rendered LaTeX
    static/settings.js         # theme/layout preferences and persistence
    static/styles.css          # base HEATR styling
    static/design-b.css        # selected redesign style system

  configs/
    rfam_eqs_xz_uniform_500w.yaml   # canonical baseline EQS/HT configuration
    shape_*.yaml                    # per-shape baseline configs
    _sweep_*.yaml                   # exposure-time sweep configs
    rfam_square_turntable_*.yaml    # turntable-oriented configs
    rfam_prewarp_*.yaml             # prewarp configs (for offline workflows)
    _gui_generated/                 # auto-generated configs from UI parameter combinations
    _archive_old/                   # legacy configs (intentionally not uploaded)

  outputs_eqs/
    <run_name>/                     # per-run artifacts: used_config, fields, summaries, plots, gifs
    shapes/                         # shape-library example outputs
    _logs/                          # run logs emitted by GUI/backend
    _archive_old/                   # legacy outputs (intentionally not uploaded)

  docs/
    index.md                   # docs entry point
    theory.md                  # detailed governing equations and model notes
    config-reference.md        # important parameters and their effects
    validation.md              # validation/verification guidance
    runbook.md                 # operational workflow and troubleshooting
```

## Documentation

- [Theory](docs/theory.md)
- [Configuration Reference](docs/config-reference.md)
- [Validation Guide](docs/validation.md)
- [Operational Runbook](docs/runbook.md)

## Notes on Data and Archives

Archive directories are intentionally excluded from repository uploads.
Active outputs, examples, and logs are organized under `outputs_eqs/`.
