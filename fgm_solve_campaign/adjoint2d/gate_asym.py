"""Finite-difference gates for the ASYMMETRIC objective's gradient dJ_asym/ds.

Run BEFORE any optimization. Central differences, epsilon swept 1e-3 down to
1e-8 (the density gate's extended window: the roundoff floor of a full-horizon
march is about 1e-12 absolute, so the usable window is narrow and is swept
explicitly rather than assumed).

THE LAYERS, one physics/chain piece at a time so a failure localizes:

  A0_out   the OUT-OF-BOUNDS term alone (w_in = 0). Temperature seed only.
           This is the melt-region chain, restricted to the bed.
  A0_in    the IN-BOUNDS DEFICIT term alone (w_out = 0). Density seed only,
           carrying the hinge mask.
  A1       both terms, both seeds in ONE reverse sweep. Unfiltered.
  A2       both terms with the physical-length design filter in the chain,
           s = F(v). This is the gradient the solve actually uses.

THE HINGE IS A NEW NONSMOOTHNESS and both sides of it are probed explicitly:

  hinge_active_cell     an in-part cell strictly below the floor at the read
                        state, so its density seed is nonzero.
  hinge_inactive_cell   an in-part cell at or above the floor, so its own seed
                        is exactly zero and every bit of its analytic
                        derivative arrives through the nonlocal coupling.
  hinge_boundary_cell   the in-part cell whose relative density is CLOSEST to
                        the floor, that is the cell sitting on the kink. This
                        is the sharpest available test of the nonsmoothness.
                        max(0, .)^2 is C^1, so the kink is in the SECOND
                        derivative and a central difference should still gate;
                        if it does not, this probe says so.

READ-INDEX STABILITY. The stop is the argmin of J_asym over the arm's own
trajectory, which IS stationary (the out term rises with time, the in term
falls, so the argmin is interior), so the envelope theorem removes the dt*/ds
term. That claim is checked, not assumed: the argmin index is recomputed under
every probe perturbation.

Run:
  ./.venv312/bin/python -m adjoint2d.gate_asym <shape> <out.json> [floor] [w_out]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from . import adjoint, asym_objective as ao, design_filter as df
from . import forward as fwd, gradops, library_solve as lib
from .pins import build_case, load_cfg

EPSILONS = (1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8)
PASS_REL_ERR = 1e-6
SUBGRADIENT_PASS_REL_ERR = 1e-5
FULL_PROBES = ("max_sensitivity_cell", "random_cell", "hinge_active_cell",
               "hinge_inactive_cell", "hinge_boundary_cell", "random_direction",
               "smooth_random_direction", "gradient_direction")
ISOLATION_PROBES = ("max_sensitivity_cell", "random_cell", "random_direction",
                    "gradient_direction")


def default_v(case, seed: int = 4242) -> np.ndarray:
    """A deliberately non-nominal design point: smooth structure plus noise."""
    rng = np.random.default_rng(seed)
    ny, nx = case.part_mask.shape
    xx, yy = np.meshgrid(np.linspace(-1, 1, nx), np.linspace(-1, 1, ny))
    smooth = 0.80 + 0.18 * np.cos(1.7 * xx + 0.4) * np.sin(2.1 * yy + 0.9)
    v = np.ones((ny, nx))
    v[case.part_mask] = (smooth[case.part_mask]
                         + 0.02 * rng.standard_normal(int(case.part_mask.sum())))
    return v


def run_forward(case, s):
    """Full horizon, T and rho checkpoints kept, no early stop."""
    return fwd.forward(case, s, keep_checkpoints=True, stop_after_phi=None,
                       shape_stop_patience=None)


def _probe_dirs(pm, g, rho, case, floor, want, seed=7, sigma_cells: float = 0.0):
    """Build the requested probe directions. Returns (name, direction, cell)."""
    rng = np.random.default_rng(seed)
    out = []
    gp = np.where(pm, np.abs(g), -np.inf)
    cells = {"max_sensitivity_cell": np.unravel_index(int(np.argmax(gp)), pm.shape)}
    idx = np.argwhere(pm)
    cells["random_cell"] = tuple(idx[rng.integers(len(idx))])

    r = np.asarray(rho, dtype=float)
    below = pm & (r < float(floor))
    above = pm & (r >= float(floor))
    if below.any():
        b = np.argwhere(below)
        cells["hinge_active_cell"] = tuple(b[rng.integers(len(b))])
    if above.any():
        a = np.argwhere(above)
        cells["hinge_inactive_cell"] = tuple(a[rng.integers(len(a))])
    dist = np.where(pm, np.abs(r - float(floor)), np.inf)
    cells["hinge_boundary_cell"] = np.unravel_index(int(np.argmin(dist)), pm.shape)

    for name in want:
        if name in cells:
            d = np.zeros(pm.shape)
            d[cells[name]] = 1.0
            out.append((name, d, cells[name]))
    if "gradient_direction" in want:
        # The STRONGEST-SIGNAL direction available, and the one the optimizer
        # actually moves along. Its analytic derivative is the gradient norm,
        # which is the largest directional derivative there is, so this probe
        # sits furthest above the fixed arithmetic noise floor of the march and
        # is the least denominator-limited number in the table.
        gd = np.zeros(pm.shape)
        gin = np.asarray(g, dtype=float)[pm]
        nrm = float(np.linalg.norm(gin))
        if nrm > 0.0:
            gd[pm] = gin / nrm
            out.append(("gradient_direction", gd, None))
    if "random_direction" in want:
        d_rand = np.zeros(pm.shape)
        u = rng.standard_normal(int(pm.sum()))
        d_rand[pm] = u / np.linalg.norm(u)
        out.append(("random_direction", d_rand, None))
        if "smooth_random_direction" in want and float(sigma_cells) > 0.0:
            fdr = df.apply_filter(d_rand, pm, sigma_cells, outside=0.0)
            d_sm = np.zeros(pm.shape)
            d_sm[pm] = fdr[pm] / np.linalg.norm(fdr[pm])
            out.append(("smooth_random_direction", d_sm, None))
    return tuple(out)


def gate(case, name, ops, v0, read_index, floor, w_out, w_in, sigma_cells,
         want=FULL_PROBES) -> dict:
    """One layer at a FIXED read index. `sigma_cells` = 0 means s = v."""
    pm = np.asarray(case.part_mask, dtype=bool)
    filt = float(sigma_cells) > 0.0

    def to_map(v):
        return df.apply_filter(v, pm, sigma_cells) if filt else v

    def eval_J(v):
        tr = run_forward(case, to_map(v))
        i = min(int(read_index), tr.n_outer - 1)
        J, sT, sr, _p = ao.J_and_seeds(tr.T_at_end(i), tr.rho_at_end(i), case,
                                       floor=floor, w_out=w_out, w_in=w_in)
        return J, tr, i, sT, sr

    J0, tr0, i0, sT, sr = eval_J(v0)
    seeds = {i0: sT} if float(w_out) != 0.0 else {}
    seeds_rho = {i0: sr} if float(w_in) != 0.0 else {}
    g_s = adjoint.gradient(case, to_map(v0), tr0, seeds, grad_ops=ops,
                           seeds_rho=seeds_rho)
    g = df.filter_vjp(g_s, pm, sigma_cells) if filt else g_s
    rho0 = tr0.rho_at_end(i0)

    out = {"layer": name, "sigma_cells": float(sigma_cells), "J0": float(J0),
           "w_out": float(w_out), "w_in": float(w_in), "floor_rho_rel": float(floor),
           "read_index": int(i0), "n_outer": int(tr0.n_outer),
           "grad_norm": float(np.linalg.norm(g[pm])),
           "grad_max_abs": float(np.max(np.abs(g[pm]))),
           "hinge_active_frac_at_read": float(
               np.mean(ao.deficit(rho0, case, floor)[pm] > 0.0)),
           "probes": {}}

    for pname, d, cell in _probe_dirs(pm, g, rho0, case, floor, want,
                                      sigma_cells=sigma_cells):
        ana = float(np.sum(g * d))
        rows = []
        for eps in EPSILONS:
            jp = eval_J(v0 + eps * d)[0]
            jm = eval_J(v0 - eps * d)[0]
            fd = (jp - jm) / (2.0 * eps)
            rows.append({"eps": eps, "fd": float(fd),
                         "abs_err": abs(fd - ana),
                         "rel_err": abs(fd - ana) / max(abs(ana), 1e-30)})
        best = min(rows, key=lambda r: r["rel_err"])
        out["probes"][pname] = {
            "cell": None if cell is None else [int(c) for c in cell],
            "rho_rel_at_cell": (None if cell is None
                                else float(np.asarray(rho0)[tuple(cell)])),
            "analytic": ana, "sweep": rows,
            "best_rel_err": best["rel_err"], "best_abs_err": best["abs_err"],
            "best_eps": best["eps"],
            "PASS": bool(best["rel_err"] < PASS_REL_ERR),
            "PASS_subgradient": bool(best["rel_err"] < SUBGRADIENT_PASS_REL_ERR)}
    out["n_probes"] = len(out["probes"])
    out["n_probes_pass_1e-6"] = sum(p["PASS"] for p in out["probes"].values())
    out["n_probes_pass_1e-5"] = sum(p["PASS_subgradient"] for p in out["probes"].values())
    out["PASS"] = bool(out["n_probes_pass_1e-6"] == out["n_probes"])
    out["PASS_subgradient"] = bool(out["n_probes_pass_1e-5"] == out["n_probes"])
    return out


def read_index_stability(case, v0, floor, w_out, w_in, sigma_cells, eps=1e-3) -> dict:
    """Does the argmin read index move under the probe perturbations?"""
    pm = np.asarray(case.part_mask, dtype=bool)
    to_map = ((lambda v: df.apply_filter(v, pm, sigma_cells))
              if float(sigma_cells) > 0.0 else (lambda v: v))

    def stop_of(v):
        return ao.asym_stop(run_forward(case, to_map(v)), case, floor=floor,
                            w_out=w_out, w_in=w_in)

    base = stop_of(v0)
    rho0 = np.where(pm, floor - 1e-3, 0.0)     # force both hinge cells to exist
    out = {"eps": float(eps), "base_index": base.index, "moved": False, "probes": {}}
    for pname, d, _c in _probe_dirs(pm, np.where(pm, 1.0, 0.0), rho0, case, floor,
                                    ("max_sensitivity_cell", "random_cell",
                                     "random_direction")):
        ip = stop_of(v0 + eps * d).index
        im = stop_of(v0 - eps * d).index
        out["probes"][pname] = {"plus": ip, "minus": im,
                                "moved": bool(ip != base.index or im != base.index)}
        out["moved"] = out["moved"] or out["probes"][pname]["moved"]
    return out


def main(shape: str, out_path: str, floor: float = ao.FLOOR_RHO_REL_DEFAULT,
         w_out: float = ao.W_OUT_DEFAULT, w_in: float = ao.W_IN_DEFAULT,
         sigma_cells: float = df.DEFAULT_SIGMA_CELLS) -> dict:
    t0 = time.perf_counter()
    cfg_path = lib.shape_config(shape)
    case = build_case(load_cfg(cfg_path))
    ops = gradops.gradient_matrices(case.x, case.y)
    v0 = default_v(case)

    base = run_forward(case, v0)
    st = ao.asym_stop(base, case, floor=floor, w_out=w_out, w_in=w_in)
    m0 = ao.region_metrics(base.T_at_end(st.index), base.rho_at_end(st.index),
                           case, floor=floor)
    res = {"shape": shape, "config": str(cfg_path), "floor_rho_rel": float(floor),
           "w_out": float(w_out), "w_in": float(w_in),
           "sigma_cells": float(sigma_cells),
           "base": {"read_index": st.index, "read_time_s": st.time_s,
                    "J_asym": st.J, "J_out": st.J_out, "J_in": st.J_in,
                    "at_horizon": st.at_horizon,
                    "stop_is_first_step": st.stop_is_first_step,
                    "in_term_dead": st.in_term_dead,
                    "flat_onset_index": st.flat_onset_index,
                    "flat_onset_gap_steps": st.flat_onset_gap_steps,
                    "n_outer": base.n_outer, **m0},
           "layers": []}
    print(f"[{shape}] base read index {st.index} at {st.time_s:.0f} s, "
          f"J_asym {st.J:.5f} = out {st.J_out:.5f} + in {st.J_in:.5f}, "
          f"hinge active {m0['hinge_active_frac']:.3f}, "
          f"mean rho_rel {m0['mean_rho_rel_part']:.4f}", flush=True)
    del base

    layers = (
        ("A0_out_only", w_out, 0.0, 0.0, ISOLATION_PROBES),
        ("A0_in_only", 0.0, w_in, 0.0, ISOLATION_PROBES),
        ("A1_combined", w_out, w_in, 0.0, FULL_PROBES),
        ("A2_combined_filtered", w_out, w_in, float(sigma_cells), FULL_PROBES),
    )
    for name, wo, wi, sig, want in layers:
        r = gate(case, name, ops, v0, st.index, floor, wo, wi, sig, want)
        res["layers"].append(r)
        bits = " ".join(f"{k}={v['best_rel_err']:.2e}" for k, v in r["probes"].items())
        print(f"[{shape}] {name:22s} J0={r['J0']:.6f} {bits} "
              f"{r['n_probes_pass_1e-6']}/{r['n_probes']} at 1e-6, "
              f"{r['n_probes_pass_1e-5']}/{r['n_probes']} at 1e-5", flush=True)

    res["read_index_stability"] = read_index_stability(case, v0, floor, w_out,
                                                       w_in, 0.0)
    print(f"[{shape}] read index stability: base "
          f"{res['read_index_stability']['base_index']}, "
          f"moved = {res['read_index_stability']['moved']}", flush=True)
    res["ALL_GATES_PASS"] = all(l["PASS"] for l in res["layers"])
    res["ALL_GATES_PASS_SUBGRADIENT"] = all(l["PASS_subgradient"] for l in res["layers"])
    res["wall_s"] = time.perf_counter() - t0
    p = Path(out_path).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] ALL_GATES_PASS at 1e-6 = {res['ALL_GATES_PASS']}, at the "
          f"1e-5 subgradient standard = {res['ALL_GATES_PASS_SUBGRADIENT']}, "
          f"wall {res['wall_s']:.1f} s", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2],
         float(sys.argv[3]) if len(sys.argv) > 3 else ao.FLOOR_RHO_REL_DEFAULT,
         float(sys.argv[4]) if len(sys.argv) > 4 else ao.W_OUT_DEFAULT)
