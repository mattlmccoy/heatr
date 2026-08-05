# Metal / Apple-silicon GPU probes

Evidence behind the REFUSED Metal march port (SPEED_REPORT.md section 7,
`engine_speed/march_metal.py`). Recorded outputs live in
`engine_speed/metal_probe_results.json`.

Run order and what each answers:

| script | question | needs |
|---|---|---|
| `probe_metal_fp64.py` | does the live Metal shader compiler accept `double`? | `pyobjc-framework-Metal` |
| `probe_frameworks_fp64.py` | do torch-MPS / MLX carry float64 to the GPU? | `torch`, `mlx` |
| `probe_mlx_f64_gpu.py` | MLX exposes `mx.float64`: does it EXECUTE on the GPU, or silently fall back to CPU? | `mlx` |
| `probe_fp32_bridge.py` | is GPU float32 bit-identical to numpy float32, so numpy float32 is a faithful proxy for the Metal path's arithmetic? | `torch`, `mlx` |
| `probe_gpu_march_speed.py` | had float32 been acceptable, how much speed was on the table? | `mlx` |

`pyobjc-framework-Metal` is installed in the project venv, so
`probe_metal_fp64.py` and `python -m engine_speed.march_metal` run there
directly. The torch/MLX probes were run in a THROWAWAY venv, deliberately not in
`.venv312`: neither framework is a project dependency and neither is needed now
that the answer is recorded.

```sh
python3 -m venv /tmp/metalprobe
/tmp/metalprobe/bin/pip install torch mlx numpy pyobjc-framework-Metal
/tmp/metalprobe/bin/python engine_speed/metal_probes/probe_frameworks_fp64.py
```

Pinned versions for the recorded run: python 3.14.0, torch 2.13.0, mlx 0.32.0,
numpy 2.5.1, macOS 26.5.2, Apple M2 Pro.
