# Experimental PA12 Physics Bucket

This project now supports an isolated experimental model family:

- `physics_model.family: experimental_pa12_hybrid`
- `physics_model.experimental_enabled: true`

Baseline behavior is unchanged unless the experimental family is explicitly selected.

## Config Blocks

When experimental mode is enabled, these blocks are expected:

- `physics_model`
- `phase_model` (must use `type: apparent_heat_capacity_dsc`)
- `dens_model` (must use `type: viscous_capillary_pa12`)
- `crystallization_model` (optional branch; default disabled)

Provenance validation uses:

- `configs/experimental_pa12_provenance.yaml`

## GUI

Use **Operation -> Experimental Physics Bucket (opt-in)** and set:

- model family
- experimental enabled
- provenance/calibration tags
- phase/densification/crystallization parameters

Experimental runs are routed to:

- `outputs_eqs/_experimental/<output_name>/`

## A/B Report Generation

Use the manifest-driven comparator:

```bash
python3 experimental_ab_compare.py \
  --manifest configs/experimental_ab_manifest.yaml \
  --outputs-root outputs_eqs
```

Artifacts are emitted to:

- `outputs_eqs/_experimental_ab/<timestamp>/`
  - `report.json`
  - `report.md`
  - `metrics.csv`
  - `plots/*.png`
  - `physics_review_checklist.md`

## Two-Stage Calibration Workflow

Stage A (phase+thermal first):

```bash
python3 run_experimental_benchmark.py --stage stage_a --bucket ab-stagea-001
```

Stage B (full hybrid model):

```bash
python3 run_experimental_benchmark.py --stage stage_b --bucket ab-stageb-001
```

Then generate A/B report with a manifest pointing baseline and experimental runs:

```bash
python3 experimental_ab_compare.py \
  --manifest configs/experimental_ab_manifest.yaml \
  --outputs-root outputs_eqs
```

New gate included in A/B:

- `exp_phi_dsc_mae_max` (DSC plausibility gate for melt-fraction consistency)
