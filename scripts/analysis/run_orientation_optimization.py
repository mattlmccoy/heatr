#!/usr/bin/env python3
"""Orientation optimization for the NOT RESCUED shapes under the
shape-fidelity objective J.

    J(theta, arm, t_stop) = sum over the WHOLE domain of
                            (phi(x, t_stop) - chi_part_rotated(x))^2

chi_part_rotated is the part mask rasterized by the production engine at
geometry.part.rotation_deg = theta, so the nominal target co-rotates with the
part. Stop convention: t_stop = argmin of J over the arm's OWN trajectory,
1500-step horizon (dt 0.5 s, 750 s), early truncation 250 steps after the
running minimum (identical to the shape-library campaign).

Engine: the fgm_solve_campaign/adjoint2d forward (the engine of
SHAPE_LIBRARY_SOLVE_REPORT.md, bit-identity gated against stored production
runs). Geometry comes from the production rfam_eqs_coupled.make_domain, so
rotation_deg is the production rasterization.

Arms per angle:
  uniform   s = 1 everywhere
  graded    the solved single-pass 4-bpp map (out_lib/<shape>_maps.npz key
            A1_4bpp), ROTATED into the rotated part frame by the production
            rotation convention (see orientation_map_rotation.py) because the
            solver injects saturation maps in the LAB frame.

Usage:
    python scripts/analysis/run_orientation_optimization.py [shape ...]
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))
sys.path.insert(0, str(REPO / "scripts"))

from adjoint2d import energy_gate as eg          # noqa: E402
from adjoint2d import forward as fwd             # noqa: E402
from adjoint2d import shape_objective as so      # noqa: E402
from adjoint2d.pins import build_case, load_cfg  # noqa: E402

from analysis.orientation_map_rotation import rotate_sat_map  # noqa: E402

PATIENCE = 250   # shape-library convention (library_solve.PATIENCE)
N_STEPS = 1500   # full horizon, dt 0.5 s -> 750 s

CFG_DIR = REPO / "outputs_eqs/fgm_calibrated_control/configs"
MAPS_DIR = REPO / "fgm_solve_campaign/out_lib"
OUT_ROOT = REPO / "outputs_eqs/orientation_optimization"

# Symmetry-aware angle sets (degrees). Arguments recorded in the report:
#   T_shape / L_shape: one mirror plane composed with the exact x-mirror of
#     the domain gives J(theta) = J(-theta); the y-flip equivalence
#     (theta ~ theta+180) is ASSUMED up to the top-only convection term, the
#     same assumption the stored L-shape sweeps [0, 180] made.
#   cross: 4-fold rotation symmetry (period 90) plus mirrors -> [0, 45].
#   star (5-point): 5-fold symmetry (period 72) plus mirrors -> [0, 36].
ANGLES: dict[str, list[float]] = {
    "T_shape": [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0],
    "L_shape": [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0],
    "cross": [0.0, 15.0, 30.0, 45.0],
    "star": [0.0, 9.0, 18.0, 27.0, 36.0],
}

# Angle-0 reference values from SHAPE_LIBRARY_SOLVE_REPORT.md Section 7
# (uniform s=1 and solved [0,1] 4 bpp rows): the driver gate.
REF_AT_ZERO = {
    "T_shape": {"uniform": (614.26, 0.4574), "graded": (609.29, 0.4444)},
    "L_shape": {"uniform": (538.42, 0.5142), "graded": (521.20, 0.5209)},
    "cross": {"uniform": (471.82, 0.5465), "graded": (360.18, 0.6755)},
    "star": {"uniform": (192.58, 0.6517), "graded": (157.39, 0.7032)},
}


def shape_config(shape: str) -> Path:
    """First sorted calibrated config, the shape-library convention."""
    cands = sorted(CFG_DIR.glob(f"{shape}_m*.yaml"))
    if not cands:
        raise FileNotFoundError(f"no calibrated config for {shape!r}")
    return cands[0]


def score_arm(case, s: np.ndarray) -> tuple[dict, np.ndarray, np.ndarray]:
    """One forward, scored at its own J-stop. Returns (metrics, phi_stop, J_curve)."""
    tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=PATIENCE,
                     n_steps=N_STEPS)
    st = so.optimal_stop(tr, case)
    T_stop = tr.T_at_end(st.index)
    phi_stop, _ = so.phi_field(T_stop, case)
    m = so.full_metrics(tr, case)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["n_outer"] = tr.n_outer
    m["max_T_at_stop_c"] = float(np.max(T_stop))
    m["energy_gate"] = eg.gate_from_trajectory(tr, m["t_stop_index"])
    return m, phi_stop, st.J_curve


def run_shape(shape: str) -> dict:
    t0 = time.perf_counter()
    cfg_path = shape_config(shape)
    cfg0 = load_cfg(cfg_path)
    maps = np.load(MAPS_DIR / f"{shape}_maps.npz")
    sat0 = np.asarray(maps["A1_4bpp"], dtype=float)
    pm0 = np.asarray(maps["part_mask"], dtype=bool)

    out_dir = OUT_ROOT / shape
    (out_dir / "fields").mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for angle in ANGLES[shape]:
        cfg = copy.deepcopy(cfg0)
        cfg["geometry"]["part"]["rotation_deg"] = float(angle)
        case = build_case(cfg)
        if angle == 0.0 and int(np.sum(case.part_mask != pm0)) != 0:
            raise RuntimeError(f"{shape}: part mask at 0 deg does not match out_lib")
        arms = {
            "uniform": np.ones(case.part_mask.shape, dtype=float),
            "graded": rotate_sat_map(sat0, angle, part_mask_rot=case.part_mask),
        }
        for arm, s in arms.items():
            m, phi_stop, j_curve = score_arm(case, s)
            m.update(shape=shape, angle_deg=float(angle), arm=arm)
            tag = f"ang{angle:07.2f}_{arm}".replace(".", "p")
            np.savez_compressed(
                out_dir / "fields" / f"{tag}.npz",
                phi_stop=phi_stop.astype(np.float32),
                part_mask=case.part_mask,
                sat_map=s.astype(np.float32),
                J_curve=j_curve.astype(np.float32),
                angle_deg=float(angle),
            )
            rows.append(m)
            gate = m["energy_gate"]
            print(f"[{shape}] ang {angle:6.1f} {arm:7s} "
                  f"J {m['J']:8.2f}  IoU {m['IoU']:.4f}  "
                  f"stop {m['t_stop_s']:6.1f}s{' (H)' if m['t_stop_at_horizon'] else '    '}  "
                  f"under {m['part_under_melt_pct']:5.2f}%  bed {m['bed_melt_pct_of_part']:5.2f}%  "
                  f"Eres {gate['rel_residual_at_index'] * 100:.2f}%", flush=True)
            if angle == 0.0:
                ref_j, ref_iou = REF_AT_ZERO[shape][arm]
                dj = abs(m["J"] - ref_j) / ref_j
                print(f"    gate vs library report: J ref {ref_j} got {m['J']:.2f} "
                      f"(rel {dj:.2e}); IoU ref {ref_iou} got {m['IoU']:.4f}", flush=True)

    wall = time.perf_counter() - t0
    result = {
        "shape": shape,
        "config": str(cfg_path),
        "voltage_v": float(cfg0["electric"]["voltage_v"]),
        "graded_map": f"fgm_solve_campaign/out_lib/{shape}_maps.npz:A1_4bpp",
        "stop_convention": ("t_stop = argmin of J over the arm's own trajectory; "
                            "1500-step horizon (750 s), patience 250"),
        "angles_deg": ANGLES[shape],
        "rows": rows,
        "wall_s": wall,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(result, f, indent=1)
    print(f"[{shape}] done in {wall:.1f} s -> {out_dir/'results.json'}", flush=True)
    return result


def main() -> None:
    shapes = sys.argv[1:] or list(ANGLES.keys())
    for shape in shapes:
        run_shape(shape)


if __name__ == "__main__":
    main()
