# RFAM geo-prewarp — Canonical Electrothermal Forward Model

2D electro-quasi-static (EQS) + thermal solver for Radio Frequency Additive Manufacturing
(RFAM) of Nylon 12. Solves for RF heating, phase change, and densification to enable
geometry prewarp for shrinkage compensation.

## Project status: **CANONICAL MODEL LOCKED (Mar 2026)**

## Directory structure

```
geo-prewarp/
├── rfam_eqs_coupled.py          ← Main 2D EQS + thermal solver
├── extract_comsol_qrf_2d.py     ← Extract 2D Q_rf map from COMSOL export
├── RFAM_physics_from_literature.md  ← Physics reference notes
├── configs/
│   ├── rfam_eqs_xz_uniform_500w.yaml   ← CANONICAL base config (6 min)
│   ├── _sweep_7min.yaml                ← Time sweep: 7 min
│   ├── _sweep_8min.yaml                ← Time sweep: 8 min
│   ├── _sweep_9min.yaml                ← Time sweep: 9 min
│   ├── _sweep_10min.yaml               ← Time sweep: 10 min
│   ├── rfam_eqs_comsol_qrf_driven.yaml       ← COMSOL Q_rf injection (full res)
│   ├── rfam_eqs_comsol_qrf_driven_quick.yaml ← COMSOL Q_rf injection (quick test)
│   └── _archive_old/                   ← All deprecated configs
└── outputs_eqs/
    ├── xz_uniform_6min_v2/     ← CANONICAL 6-min run (fields.npz, rf_summary_v4.png)
    ├── _sweep_7min/             ← Time sweep results
    ├── _sweep_8min/
    ├── _sweep_9min/
    ├── _sweep_10min/
    ├── _sweep_time_sweep.png   ← Master time sweep figure (5-col × 2-row)
    ├── comsol_exports/         ← COMSOL Q_rf map (qrf_2d_map.npy)
    └── _archive_old/           ← All deprecated outputs
```

## Canonical model physics

**Uniform σ = 0.04 S/m** throughout the doped region. The EQS solve naturally
produces the correct RF heating distribution:
- **Corners**: Q_rf ≈ 37× mean (E-field concentration)
- **Top/bottom faces** (⊥ to E-field): ~11× mean (strong coupling)
- **Left/right faces** (∥ to E-field): ~1.6× mean (weak coupling)
- **Center**: ~1× mean (heated primarily by conduction)

**Key parameters:**
| Parameter | Value | Notes |
|-----------|-------|-------|
| Generator power | 500 W | Fixed RF generator |
| Coupling efficiency | 2% | Calibrated to 6-min melt |
| Chamber | 60×60 mm | 20mm powder gap on all sides |
| Part | 20×20 mm | Nylon 12, centered |
| Grid | 120×120 | ~0.5mm resolution |
| Electrode spacing | 60 mm | Top=0V, Bottom=860V |
| Phase change | T=180°C, L=96.7 kJ/kg | Heaviside, ΔT=10°C |

## Quick start

```bash
# Canonical 6-min run
python3 rfam_eqs_coupled.py \
  --config configs/rfam_eqs_xz_uniform_500w.yaml \
  --output-dir outputs_eqs/xz_uniform_6min_v2

# Time sweep 7–10 min (parallel)
for m in 7 8 9 10; do
  python3 rfam_eqs_coupled.py \
    --config configs/_sweep_${m}min.yaml \
    --output-dir outputs_eqs/_sweep_${m}min > /tmp/sweep_${m}.log 2>&1 &
done; wait
```

## Browser GUI (no prewarp controls)

Use the browser interface to run the forward model with:
- shape selection
- single exposure time runs
- exposure-time sweeps
- optimizer-enabled runs
- turntable runs
- automatic config matching: reuses an existing config when parameters match
- automatic config generation: creates a new `_gui_generated` config when no exact match exists
- advanced overrides (optional) for core numerical/electrical/thermal parameters
- live figure updates while jobs are running
- in-page image viewer (no new tab required)
- job logs + result browsing on a dedicated results page
- examples page for existing sweep/shape outputs

```bash
python3 rfam_gui_server.py
```

Then open `http://127.0.0.1:8080`.
Results page: `http://127.0.0.1:8080/results`
Examples page: `http://127.0.0.1:8080/examples`
Theory page: `http://127.0.0.1:8080/theory`

## Time-sweep results

| t [min] | T̄ [°C] | T_max [°C] | φ (melt) | ρ_rel |
|---------|---------|-----------|----------|-------|
| 6       | 181.2   | 197.6     | 82.5%    | 0.627 |
| 7       | 185.9   | 205.9     | 89.4%    | 0.692 |
| 8       | 195.2   | 214.1     | 96.4%    | 0.783 |
| 9       | 205.0   | 222.6     | 99.6%    | 0.868 |
| 10      | 214.9   | 231.3     | 100%     | 0.932 |

## Next steps
1. Compare Python temperature fields to COMSOL (export T from Tuned_Sigma.mph)
2. Use φ/ρ_rel field as forward model for ILT-inspired geometry prewarp
3. Higher-resolution run (240×240, 0.25mm) for publication quality
