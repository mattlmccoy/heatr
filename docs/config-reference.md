# HEATR Configuration Reference

This reference summarizes the most important YAML configuration domains used by HEATR.

## Top-Level Domains

- `geometry`
- `electrodes`
- `electric`
- `thermal`
- `densification`
- `materials`
- `optimizer` (optional)
- `turntable` (optional)
- `reporting`

## geometry

- `chamber_x`, `chamber_y`: chamber extents (m)
- `grid_nx`, `grid_ny`: grid resolution
- `part.*`: shape selection and dimensions

## electrodes

- boundary placement and mode
- high/low electrode assignment

## electric

- `frequency_hz`
- `voltage_v`
- `max_qrf_w_per_m3`
- `enforce_generator_power`
- `generator_power_w`
- `generator_transfer_efficiency`
- `effective_depth_m`

## thermal

- `ambient_c`
- `convection_h_w_per_m2k`
- `convective_boundaries`
- `dt_s`, `n_steps`
- `max_temp_c`
- `phase_change.*`

## densification

- `rho_rel_initial`
- kinetic model constants and exponents
- max per-step bounds

## materials

### virgin
- baseline dielectric/electrical properties outside doped region

### powder
- powder density, conductivity, heat capacity

### doped
- conductivity profile and thermal/electrical material values in active region

## optimizer (optional)

- `enabled`
- `phi_snapshots`
- `temp_ceiling_c`
- `highlight_phi`

## turntable (optional)

- `enabled`
- `rotation_deg` (degrees per event)
- `total_rotations` (number of events)
- `rotation_interval_s` (optional explicit spacing)

## reporting

- plot display controls
- stream/contour options
- GIF controls:
  - `gif_max_frames`
  - `gif_min_frames`
  - `gif_frame_duration_s`

## GUI Overrides

The browser GUI can override selected values and generate a run-specific config in `configs/_gui_generated/` when needed. This preserves base config immutability and traceability.
