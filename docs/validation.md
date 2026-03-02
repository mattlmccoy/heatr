# HEATR Validation and Comparison Guide

This document describes recommended validation workflows and current interpretation boundaries.

## Objectives

Validation should answer:

1. Are spatial heating patterns directionally correct?
2. Are thermal time scales calibrated for target materials and hardware assumptions?
3. Do rotation/turntable events preserve physically reasonable continuity?

## Recommended Workflow

## 1) Baseline Single Run

Run a canonical shape/config and archive:

- `used_config.yaml`
- `summary.json`
- `time_series.json`
- key plots and GIFs

Check for:

- stable monotonic trends where expected
- no discontinuities at non-event timesteps

## 2) Exposure Sweep

Run 6/7/8/9/10 min (or process-relevant range) and verify trend consistency in:

- mean/max temperature
- melt fraction
- relative density

## 3) Turntable Sanity

For turntable-enabled runs:

- inspect metrics around rotation event timestamps
- verify no non-physical drops in state due only to remapping
- inspect thermal/density GIF continuity across rotation boundaries

## 4) External Benchmarking

When benchmarking against COMSOL or lab reference:

- align geometry, boundary conditions, and effective depth assumptions
- compare both spatial distributions and aggregate trajectories
- report calibration knobs explicitly (conductivity, coupling, convection)

## Known Limits

- 2D cross-sectional abstraction of inherently 3D behavior
- effective-depth interpretation for power and energy
- empirical simplifications in kinetic submodels

## Reporting Checklist

For reproducible comparisons include:

- commit hash / code snapshot
- exact YAML used (`used_config.yaml`)
- run mode and output folder
- summary metrics and key plots
- note of any non-default advanced overrides
