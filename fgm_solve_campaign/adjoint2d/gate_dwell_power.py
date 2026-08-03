"""Finite-difference gate for the SECONDARY arm's power-schedule gradient.

Layer D4: dJ/dp_k with a dwell schedule already in force. Run at the production
horizon and segment count on the cross, which is the only shape the secondary
arm is run on. Probes: the maximum-sensitivity segment, a fixed pseudo-random
segment, a random unit direction over all segments, and the gradient direction.

Run:
  ./.venv312/bin/python -m adjoint2d.gate_dwell_power <shape> <out.json> [n_steps] [n_seg]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import design_filter as df, dwell, dwell_power as dp
from . import gradops, library_solve as lib, schedule as sch
from . import shape_objective as so, topopt
from .dwell_kernel import DwellKernel
from .gate_dwell import _sweep, _finish, _log, default_weights, CANDIDATE_ANGLES
from .gate_rho import default_v
from .pins import load_cfg


def main(shape: str, out_path: str, n_steps: int = 1500, n_seg: int = 12) -> dict:
    t0 = time.perf_counter()
    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    kern = DwellKernel.build(cfg, angles=CANDIDATE_ANGLES)
    case = kern.case0
    pm = case.part_mask
    ops = gradops.gradient_matrices(case.x, case.y)
    sigma_cells = topopt.sigma_cells_for(topopt.FILTER_RADIUS_M, case.dx)
    s0 = df.apply_filter(default_v(case), pm, sigma_cells)
    w0 = default_weights(len(CANDIDATE_ANGLES))
    kern.set_weights(w0)

    rng = np.random.default_rng(31)
    # a deliberately structured schedule: a ramp with one forced dip
    p0 = np.clip(1.2 - 0.06 * np.arange(n_seg) + 0.05 * rng.standard_normal(n_seg),
                 0.0, 1.5)
    p0[n_seg // 2] = 0.35

    def pf(pv):
        return sch.expand_full(pv, n_steps, n_steps, n_seg)

    tr_base = dp.scheduled_forward(kern, s0, pf(p0), n_steps=n_steps)
    st = so.optimal_stop(tr_base, case)
    read = int(st.index)
    # The read state MATTERS here, and not for the usual reason. MEASURED in
    # this pass: the densification solid-state driving term is
    # `(1 - phi)**0.8` (`forward.py:258`, `dens_phi_solid_exponent = 0.8`), and
    # an exponent below one is NOT Lipschitz where the base reaches zero. At a
    # read index deep into full melt a large cell population sits exactly
    # there, and the central-difference error then GROWS as epsilon shrinks:
    # the dwell gradient itself, already gated clean at 1e-10 at its own
    # production read state, shows relative errors of 3e-02 to 3e+00 when read
    # at the horizon of a 900-step full-power march. The gate is therefore run
    # at the arm's own argmin on the PRODUCTION horizon, which is the state the
    # optimizer actually reads, and `at_horizon` is reported rather than
    # assumed away.
    print(f"[{shape}] gate read index {read} of {tr_base.n_outer}, "
          f"at_horizon = {st.at_horizon}, J = {st.J:.4f}", flush=True)

    def J_of(pv):
        tr = dp.scheduled_forward(kern, s0, pf(pv), n_steps=n_steps)
        return so.shape_J_and_seed(tr.T_at_end(min(read, tr.n_outer - 1)), case)[0]

    tr0 = dp.scheduled_forward(kern, s0, pf(p0), n_steps=n_steps, keep_checkpoints=True)
    J0, seed = so.shape_J_and_seed(tr0.T_at_end(read), case)
    _gs, _gw, gp = dp.scheduled_gradients(kern, s0, tr0, {read: seed}, pf(p0),
                                          n_seg=n_seg, grad_ops=ops,
                                          n_window=n_steps)

    e_max = np.zeros(n_seg)
    e_max[int(np.argmax(np.abs(gp)))] = 1.0
    e_rnd = np.zeros(n_seg)
    e_rnd[int(np.random.default_rng(11).integers(n_seg))] = 1.0
    u = np.random.default_rng(5).standard_normal(n_seg)
    layer = {"layer": "D4_power_schedule_with_dwell", "variable": "power segments p",
             "J0": float(J0), "read_index": read, "n_seg": int(n_seg),
             "n_steps": int(n_steps), "p0": [float(x) for x in p0],
             "dJ_dp": [float(x) for x in gp],
             "grad_norm": float(np.linalg.norm(gp)),
             "read_at_horizon": bool(st.at_horizon),
             "weights": [float(x) for x in w0], "probes": {}}
    for name, d in (("max_sensitivity_segment", e_max), ("random_segment", e_rnd),
                    ("random_direction", u / np.linalg.norm(u)),
                    ("gradient_direction", gp / max(np.linalg.norm(gp), 1e-30))):
        layer["probes"][name] = _sweep(J_of, p0, d, float(np.dot(gp, d)))
    _finish(layer)
    _log(layer)

    res = {"shape": shape, "config": str(cfg_path), "layers": [layer],
           "dwell_weights": [float(x) for x in w0],
           "candidate_angles_deg": [float(a) for a in CANDIDATE_ANGLES],
           "ALL_GATES_PASS": layer["PASS"],
           "ALL_GATES_PASS_SUBGRADIENT": layer["PASS_subgradient"],
           "wall_s": time.perf_counter() - t0}
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] D4 PASS at 1e-6 = {res['ALL_GATES_PASS']}, at 1e-5 = "
          f"{res['ALL_GATES_PASS_SUBGRADIENT']}, wall {res['wall_s']:.1f} s", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         int(sys.argv[3]) if len(sys.argv) > 3 else 1500,
         int(sys.argv[4]) if len(sys.argv) > 4 else 12)
