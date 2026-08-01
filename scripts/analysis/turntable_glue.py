"""Run-script glue for the TRUE rotating forward on the production engine.

No line of `rfam_eqs_coupled.py` is edited. Everything here is a monkeypatch
applied by a context manager for the duration of one `run_sim` call, so the
engine's default behaviour is untouched for every other caller.

Three patches, each with its own reason and its own switch:

`corotate`  (default True, the physics fix)
    At a turntable rotation event the engine re-rasterizes the part and remaps
    temperature, relative density and melt fraction into the rotated frame
    (`rfam_eqs_coupled.py:2969-2985`), then recomputes conductivity through
    `_FgmFeedback.sigma_at_mask`, which indexes the UNROTATED lab-frame
    `sat_map` (`:286`, `:443`). The dopant is PRINTED INTO THE PART, so it has
    to rotate with the part. Hooking `_build_rotated_part_mask` gives both the
    cumulative angle and the new mask, which is exactly what the rotated map
    needs, and the hook fires once per event by construction.

`corotate_eps`  (default False, a SECOND and separate engine gap)
    The relative-permittivity field is built once at startup from the ORIGINAL
    fill fraction (`rfam_eqs_coupled.py:2532`) and is never re-rasterized at a
    rotation event; only conductivity is. With eps_r 20 inside the part against
    1 outside that leaves a stationary dielectric ghost of the un-rotated part
    in every post-event electro-quasi-static (EQS) solve. This switch overwrites
    eps_r in place from the CURRENT rotated fill fraction just before each EQS
    solve. It is off by default so the baseline arm is the engine as shipped,
    and its effect is measured rather than assumed.

`record`  (default True, an instrumentation hook, no physics)
    The engine persists only the final melt-fraction field, so J at the optimal
    stop is unreadable from a stored run. `part_energy_per_depth` is called
    exactly once per outer step (`rfam_eqs_coupled.py:3205`) with the live
    temperature, relative density and CURRENT rotated part mask, so wrapping it
    yields a per-outer-step shape-fidelity trace at zero physics risk. Melt
    fraction is re-derived from temperature by the engine's own phase-change
    formula; that derivation is PROVEN against the engine's own per-step mean in
    `test_turntable_glue.py`, not assumed.
"""
from __future__ import annotations

import contextlib
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.analysis.orientation_map_rotation import rotate_sat_map  # noqa: E402

MELT_LEVEL = 0.5


# ---------------------------------------------------------------------------
# pure logic
# ---------------------------------------------------------------------------

def phi_from_T(T: np.ndarray, t_pc_c: float, dt_pc_c: float) -> np.ndarray:
    """The engine's melt fraction, `comsol_heaviside` with a linear ramp."""
    return np.clip((np.asarray(T, dtype=float) - float(t_pc_c))
                   / max(float(dt_pc_c), 1e-9) + 0.5, 0.0, 1.0)


def step_metrics(T: np.ndarray, rho_rel: np.ndarray, part_mask: np.ndarray,
                 t_pc_c: float, dt_pc_c: float) -> dict:
    """Shape-fidelity metrics at one outer step, against the CURRENT part mask.

    The nominal target co-rotates with the part, which is the same convention
    the joint-angle campaign used at static angles: chi is the part mask the
    engine has rasterized at the orientation in force right now.
    """
    pm = np.asarray(part_mask, dtype=bool)
    phi = phi_from_T(T, t_pc_c, dt_pc_c)
    d = phi - pm.astype(float)
    melted = phi >= MELT_LEVEL
    n_part = int(pm.sum())
    inter = int(np.sum(melted & pm))
    union = int(np.sum(melted | pm))
    return {
        "J": float(np.sum(d * d)),
        "IoU": (inter / union) if union else float("nan"),
        "bed_melt_pct_of_part": 100.0 * int(np.sum(melted & ~pm)) / max(n_part, 1),
        "part_under_melt_pct": 100.0 * int(np.sum(pm & ~melted)) / max(n_part, 1),
        "mean_phi_part": float(np.mean(phi[pm])) if n_part else float("nan"),
        "mean_rho_rel_part": float(np.mean(np.asarray(rho_rel)[pm])) if n_part else float("nan"),
        "max_T_c": float(np.max(T)),
        "max_T_part_c": float(np.max(T[pm])) if n_part else float("nan"),
        "melted_cells": int(melted.sum()),
        "part_cells": n_part,
    }


def corotated_map(base_sat: np.ndarray, cumulative_deg: float,
                  part_mask_rot: np.ndarray, outside: float = 1.0) -> np.ndarray:
    """The injected map as it looks after the part has turned by `cumulative_deg`.

    Delegates to the already proven production-convention helper
    (`orientation_map_rotation.rotate_sat_map`, `rotation_deg = +90` equals
    `np.rot90(k=-1)` with zero mismatched cells) and re-masks so the value
    outside the rotated part is the nominal `outside`.
    """
    return rotate_sat_map(np.asarray(base_sat, dtype=float),
                          float(cumulative_deg),
                          part_mask_rot=np.asarray(part_mask_rot, dtype=bool),
                          outside=float(outside))


def optimal_stop_index(j_curve: Sequence[float]) -> tuple[int, bool]:
    """argmin of J and whether it sits on the last stored step.

    Same convention the whole solve campaign uses: a minimum on the last step
    means the run was truncated before the objective turned, so that arm's J is
    a BOUND and is flagged rather than quietly reported as a value.
    """
    jc = np.asarray(list(j_curve), dtype=float)
    if jc.size == 0:
        raise ValueError("optimal_stop_index needs a non-empty curve")
    i = int(np.argmin(jc))
    return i, bool(i == jc.size - 1)


# ---------------------------------------------------------------------------
# the patched run
# ---------------------------------------------------------------------------

@dataclass
class RotatingRunResult:
    hist: dict
    summary: dict
    steps: list[dict] = field(default_factory=list)
    rotation_events: list[int] = field(default_factory=list)
    cumulative_angles_deg: list[float] = field(default_factory=list)
    sat_map_snapshots: list[np.ndarray] = field(repr=False, default_factory=list)
    final_sat_map: np.ndarray = field(repr=False, default=None)
    final_part_mask: np.ndarray = field(repr=False, default=None)
    final_phi: np.ndarray = field(repr=False, default=None)
    final_T: np.ndarray = field(repr=False, default=None)
    # Fields at the RUNNING J minimum. The engine persists only the final
    # field, and every metric in this campaign is read at the arm's own J-stop,
    # so a figure drawn from the final field would not be the figure the
    # numbers describe.
    T_at_stop: np.ndarray = field(repr=False, default=None)
    rho_at_stop: np.ndarray = field(repr=False, default=None)
    mask_at_stop: np.ndarray = field(repr=False, default=None)
    phi_at_stop: np.ndarray = field(repr=False, default=None)
    wall_s: float = 0.0

    @property
    def J_curve(self) -> np.ndarray:
        return np.asarray([s["J"] for s in self.steps], dtype=float)

    def at_stop(self) -> dict:
        i, at_horizon = optimal_stop_index(self.J_curve)
        out = dict(self.steps[i])
        out["t_stop_index"] = i
        out["t_stop_at_horizon"] = at_horizon
        out["t_stop_s"] = float(self.hist["time_s"][i])
        e_in = float(self.hist["energy_doped_J_per_m"][i])
        resid = float(self.hist["energy_balance_residual_J_per_m"][i])
        out["energy_in_J_per_m"] = e_in
        out["energy_residual_rel"] = abs(resid) / max(e_in, 1e-12)
        out["energy_gate_PASS"] = bool(out["energy_residual_rel"] < 0.05)
        out["frac_cells_dT_clipped_max"] = float(
            np.max(self.hist["frac_cells_dT_clipped"][:i + 1]))
        return out


def _write_sat_npz(sat: np.ndarray) -> Path:
    f = tempfile.NamedTemporaryFile(suffix="_sat.npz", delete=False)
    f.close()
    np.savez_compressed(f.name, sat_map=np.asarray(sat, dtype=np.float32))
    return Path(f.name)


def run_rotating(cfg: dict, sat_map: np.ndarray | None = None,
                 corotate: bool = True, corotate_eps: bool = False,
                 record: bool = True, quiet: bool = False) -> RotatingRunResult:
    """One production `run_sim` with the glue in force.

    `sat_map` is a PART-FRAME map (the orientation the config rasterizes at
    zero extra rotation). It is injected through the two-sided per-node direct
    hook `fgm_feedback.sat_map_npz_direct`, which is the conductivity-only
    channel the solve campaign's forward reproduces (`eps_geometry_only` true,
    `rfam_eqs_coupled.py:342`). Passing None runs the uniform arm.
    """
    import copy
    import io
    import time
    from unittest import mock

    import rfam_eqs_coupled as rfam

    cfg = copy.deepcopy(cfg)
    tmp_path = None
    if sat_map is not None:
        tmp_path = _write_sat_npz(sat_map)
        cfg["fgm_feedback"] = {"enabled": True,
                               "sat_map_npz_direct": str(tmp_path),
                               "sat_max": float(max(1.0, np.max(sat_map)))}
    else:
        cfg.pop("fgm_feedback", None)

    therm = cfg["thermal"]
    t_pc = float(therm["phase_change"]["t_pc_c"])
    dt_pc = float(therm["phase_change"]["dt_pc_c"])

    res = RotatingRunResult(hist={}, summary={})
    state_box: dict[str, Any] = {"fb": None, "fill": None, "mask": None,
                                 "eps_v": float(cfg["materials"]["virgin"]["eps_r"]),
                                 "eps_d": float(cfg["materials"]["doped"]["eps_r"]),
                                 "step": -1, "j_min": float("inf")}
    base_sat = None if sat_map is None else np.asarray(sat_map, dtype=float).copy()

    orig_init = rfam._FgmFeedback.__init__
    orig_brpm = rfam._build_rotated_part_mask
    orig_pepd = rfam.part_energy_per_depth
    orig_ses = rfam.solve_electric_state

    def patched_init(self, sat, block):
        orig_init(self, sat, block)
        state_box["fb"] = self

    def patched_brpm(geom_cfg, extra_rot_deg, x, y):
        pm, dm, fill, pid = orig_brpm(geom_cfg, extra_rot_deg, x, y)
        state_box["fill"] = np.asarray(fill, dtype=float)
        state_box["mask"] = np.asarray(pm, dtype=bool)
        res.cumulative_angles_deg.append(float(extra_rot_deg))
        # The event fires at the TOP of outer step `it`, before that step's
        # substeps, and the recorder has so far seen steps 0 .. it-1. So the
        # index recorded here is the first outer step under the new
        # orientation, which is what the J trace has to be read against.
        res.rotation_events.append(int(state_box["step"]) + 1)
        fb = state_box["fb"]
        if corotate and base_sat is not None and fb is not None and fb.enabled:
            new = corotated_map(base_sat, float(extra_rot_deg), pm, outside=1.0)
            fb.sat_map = new.astype(np.float32)
            res.sat_map_snapshots.append(np.array(fb.sat_map, dtype=float))
        elif base_sat is not None and fb is not None and fb.enabled:
            res.sat_map_snapshots.append(np.array(fb.sat_map, dtype=float))
        return pm, dm, fill, pid

    def patched_pepd(T, rho_rel, part_mask, params):
        state_box["step"] += 1
        if record:
            m = step_metrics(T, rho_rel, part_mask, t_pc, dt_pc)
            res.steps.append(m)
            if m["J"] < state_box["j_min"]:
                state_box["j_min"] = m["J"]
                res.T_at_stop = np.array(T, dtype=float)
                res.rho_at_stop = np.array(rho_rel, dtype=float)
                res.mask_at_stop = np.array(part_mask, dtype=bool)
        return orig_pepd(T, rho_rel, part_mask, params)

    def patched_ses(sigma, eps_r, *a, **k):
        if corotate_eps and state_box["fill"] is not None:
            eps_r[:] = (state_box["eps_v"]
                        + state_box["fill"] * (state_box["eps_d"] - state_box["eps_v"]))
        return orig_ses(sigma, eps_r, *a, **k)

    t0 = time.perf_counter()
    sink = io.StringIO()
    try:
        with mock.patch.object(rfam._FgmFeedback, "__init__", patched_init), \
             mock.patch.object(rfam, "_build_rotated_part_mask", patched_brpm), \
             mock.patch.object(rfam, "part_energy_per_depth", patched_pepd), \
             mock.patch.object(rfam, "solve_electric_state", patched_ses):
            if quiet:
                # the engine writes its progress bar to stdout and a
                # machine-readable line to STDERR every step; both are silenced
                with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                    state, summary, hist, _tt_steps, _opt = rfam.run_sim(cfg)
            else:
                state, summary, hist, _tt_steps, _opt = rfam.run_sim(cfg)
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)
    res.wall_s = time.perf_counter() - t0
    res.hist = hist
    res.summary = summary
    res.final_T = np.asarray(state.T, dtype=float)
    res.final_phi = np.asarray(state.phi, dtype=float)
    res.final_part_mask = (np.asarray(state_box["mask"], dtype=bool)
                           if state_box["mask"] is not None
                           else np.asarray(state.part_mask, dtype=bool))
    fb = state_box["fb"]
    res.final_sat_map = (np.asarray(fb.sat_map, dtype=float)
                         if (fb is not None and fb.enabled) else None)
    if res.T_at_stop is not None:
        res.phi_at_stop = phi_from_T(res.T_at_stop, t_pc, dt_pc)
    return res
