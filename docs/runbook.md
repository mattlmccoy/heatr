# HEATR Operational Runbook

## Launch

From implementation repo (`../`):

```bash
python3 rfam_gui_server.py
```

Open `http://127.0.0.1:8080`.

## Typical Operator Flow

1. Select mode and shape
2. Review auto-matched base config
3. Optionally override advanced parameters
4. Launch run and monitor active jobs
5. Review results artifacts and logs

## Naming Convention

GUI prefills output name as:

- `<shape>_<mode>_<YYYYMMDD>`

Users can append custom suffix text.

## Turntable Inputs

- `rotation_deg`: rotation increment per event
- `total_rotations`: number of rotation events
- UI also shows equivalent full 360° turns

## Artifact Expectations

Non-turntable runs:

- static plots
- evolution GIFs (`electric_field_evolution.gif`, etc.)

Turntable runs:

- static plots
- turntable GIFs (`turntable_electric.gif`, etc.)

## Troubleshooting

## No artifacts shown

- verify run directory contains media files
- check job log in `outputs_eqs/_logs`

## GIFs missing

- ensure Pillow is installed (`PIL` import)
- verify run reached completion

## Unexpected metric jumps at rotation

- inspect `time_series.json` around `summary.turntable_rotations[*].event_time_s`
- confirm recent turntable remap/coherence updates are present in code snapshot
