"""Phase E conventions, applied to a real arbitrary user part.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.phase_e.run_tamper --budget 40

WHAT THIS IS. The Phase E opener ran on two library primitives constructed
analytically in OCC. This runs the SAME conventions on "Part Studio 1 -
Tamper.stl", the STL from Grade-and-Print job feb850ec -- the part
TAMPER_DIAGNOSIS.md showed had never been near an optimizer, because the
chamber-embedded mesh it needs did not build (TRANCHE1_REPORT.md's named
blocker, fixed in solve3d/stl_mesh.py).

WHAT IS UNCHANGED FROM PHASE E, deliberately, so the two are comparable:
asymmetric objective at the pre-registered w and phi_floor, the envelope stop
(the objective is read at its own argmin over the trajectory, not at a fixed
time), the 1/|g0| first-step rescale, per-evaluation checkpointing, and a
uniform-doping baseline scored at ITS OWN matched read step.

WHAT IS DIFFERENT, and it is only the geometry route: the mesh comes from
stl_mesh.build_mesh_from_stl(with_chamber=True) instead of
phase_e.geometry.build_mesh, and the part indicator comes from the gmsh
physical group rather than an analytic predicate. Both are gated in
solve3d/tests/test_stl_chamber.py against the OCC route and against an
independent ray cast.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import (adjoint, design_chain as dc, forward as fwd, gates,
                     objective as obj, stl_mesh)

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
REPO = HERE.parents[1]
TAMPER_STL = (REPO.parents[1] / "software" / "meteor" / "tools" / "uploads"
              / "feb850ec" / "Part Studio 1 - Tamper.stl")

PART = "tamper"
# DEVIATION FROM PHASE E, named. Phase E used 0.9375e-3 (the Phase A coarse
# size). This part cannot afford it: the Tamper's OWN tessellation carries a
# 0.105 mm minimum edge, so any conforming mesh inherits ~0.092 mm elements and
# the explicit march's CFL substep follows -- independent of lc_part. Phase E's
# primitives have no such feature (pyramid minimum edge 23.2 mm). Refining
# lc_part would add cells without buying back a single time step.
LC_PART_M = 2.5e-3
# Longer than Phase E's 650 s: at the adaptive chamber this part is heading to
# a ~365 C plateau and reaches only 110.8 C mean by 150 s, so a 650 s horizon
# would stop it before melt and hand the envelope a degenerate argmin.
MAX_TIME_S = 900.0
CHECKPOINT_INTERVAL = 25
# BOUNDED TRAJECTORY RECORDING. Storing every substep state cost 51.15 GB on
# this part (18000 sample steps x 36 CFL substeps x 9866 nodes x 8 B) and the
# OS SIGKILLed the run three times, silently, ~25-30 min in. The reverse sweep
# only ever replays from anchors every CHECKPOINT_INTERVAL steps, so the
# forward stores exactly those and records J(t) as scalars instead. Must
# DIVIDE CHECKPOINT_INTERVAL or the reader would ask for an anchor that was
# never kept. At 25 this is 2.05 GB.
RECORD_STRIDE = 25
FILTER_RADIUS_M = 1.0e-3         # the Phase E design-chain filter radius

_L0_DEFAULT = object()


# --------------------------------------------------------------------------- #
# Case
# --------------------------------------------------------------------------- #
def build_case(lc_part: float = LC_PART_M, max_time_s: float = MAX_TIME_S,
               p: fwd.ForwardParams | None = None,
               precomp_coeffs=_L0_DEFAULT, stl_path: Path = TAMPER_STL):
    """A Phase B TransientCase on a chamber-embedded ARBITRARY STL.

    LEVEL 0 IS APPLIED BY DEFAULT, exactly as phase_e/run.build_case does. The
    same pre-compensated solid is meshed AND tagged, so the solve domain and
    the solve target are the same object; there is no second predicate that
    could disagree with the mesh.
    """
    from solve3d import precomp as _pc
    if precomp_coeffs is _L0_DEFAULT:
        precomp_coeffs = _pc.load_defaults()
    p = p or fwd.ForwardParams()
    # ADAPTIVE CHAMBER (Matt, 2026-08-05). L=None lets solve3d/chamber.py size
    # the box; at the frozen 60 mm this part cannot melt AND the forward is not
    # physical (clamp latched, energy residual 3.07e-03). At the adaptive
    # 85 mm both are fixed: residual 5.40e-13, clamp False.
    # Reproduction of a frozen-chamber run needs L=0.060 passed explicitly.
    msh, info = stl_mesh.build_mesh_from_stl(
        stl_path, lc_part=lc_part, with_chamber=True, L=None,
        precomp_coeffs=precomp_coeffs)
    mats = fwd.build_materials(msh, stl_mesh.part_mask_predicate(info), p)
    # the SAME L the mesh was built with reaches the electrodes, the
    # convective facet and the march; a frame mismatch here is a zero RHS
    L = float(info.L_chamber_m)
    eqs = adjoint.SteadyEqs(msh, mats, p, L=L)
    tc = adjoint.TransientCase(msh, mats, p, eqs, info, 50.0, max_time_s,
                               L=L, record_stride=RECORD_STRIDE)
    return tc, info


def _chamber_run_id(info) -> str:
    """Chamber-tagged, so a ch060 and a ch085 record can never be confused."""
    from solve3d import chamber as _ch
    return _ch.run_id(PART, info.L_chamber_m)


def mesh_record(tc, info, chain) -> dict:
    return {
        "part": PART, "run_id": _chamber_run_id(info), "stl": str(info.path), "stl_facets": info.n_facets_stl,
        "lc_part_m": float(info.lc_part), "lc_bed_m": float(info.lc_bed),
        "L_chamber_m": float(info.L_chamber_m),
        "chamber": info.chamber, "chamber_tag": info.chamber_tag,
        "feature_angle_deg": float(info.feature_angle_deg),
        "n_cells": int(info.n_cells_total),
        "n_part_cells": int(info.n_part_cells),
        "n_bed_cells": int(info.n_bed_cells),
        "n_nodes": int(info.n_nodes_total),
        "n_design_cells": int(chain.n_design),
        "stl_volume_m3": float(info.stl_volume_m3),
        "part_volume_m3": float(info.part_volume_m3),
        "part_volume_rel_err_vs_stl": float(info.part_volume_rel_err_vs_stl),
        "precomp": info.precomp,
        "kernel": chain.kernel_report(),
    }


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def score_arm(tc, s_map: np.ndarray, name: str, extra: dict | None = None,
              save_fields: bool = True) -> dict:
    """The Phase E score, minus the raster IoU/SSD block.

    NOT COVERED, and stated rather than quietly dropped: Phase E's per-plane
    IoU / surface-distance metrics need an analytic nominal cross-section on
    the shared evaluation grid. For an arbitrary STL that mask would have to
    come from a ray cast over 200x200 points per plane, which the current
    `points_inside` (a Python loop over points) cannot deliver at a sensible
    cost. Every metric below is FEM-native and needs no raster, so the verdict
    for this part rests on the objective and its split, not on IoU.
    """
    t0 = time.perf_counter()
    tr = tc.forward(tc.design_to_sigma(s_map))
    wall = time.perf_counter() - t0
    tc.set_objective("symmetric")
    Js = tc.J_trajectory(tr)
    ks = int(np.argmin(Js))
    tc.set_objective("asymmetric")
    Ja = tc.J_trajectory(tr)
    ka = int(np.argmin(Ja))                    # the ENVELOPE stop
    T_read = tc.state_at(tr, ka)
    phi = fwd.phase_fraction(T_read, tc.p)[0]
    vol, chi = tc.vol_nodal, tc.m_nodal
    split = obj.split_asymmetric(phi, chi, vol)
    split3 = obj.split_asymmetric(phi, chi, vol, w_ratio=obj.W_SENSITIVITY)
    part_w = vol * chi
    rec = {
        "arm": name, "part": PART, "wall_forward_s": wall,
        "n_steps": tr.n_steps, "n_eqs_solves": len(tr.events),
        "J_symmetric": float(Js[ks]), "argmin_symmetric": ks,
        "J_asymmetric": float(Ja[ka]), "argmin_asymmetric": ka,
        "t_stop_s": step_to_time_s(tr, ka, tc.p),
        "n_sub": int(getattr(tr, "n_sub", 1)),
        "at_horizon_asymmetric": bool(ka >= tr.n_steps),
        "J_asym_w3_at_same_read": split3["J_asym"],
        "J_out_of_bounds": split["J_out_of_bounds"],
        "J_in_bounds_deficit": split["J_in_bounds_deficit"],
        "out_of_part_melt_fraction_of_part":
            split["out_of_bounds_melt_fraction_of_part"],
        "in_bounds_below_floor_fraction":
            split["in_bounds_below_floor_fraction"],
        "part_mean_phi": float(np.dot(phi, part_w) / part_w.sum()),
        "part_max_T_c": float(T_read.max()),
        "sigma_T_diagnostic_c": float(np.sqrt(
            np.dot((T_read - np.dot(T_read, part_w) / part_w.sum()) ** 2,
                   part_w) / part_w.sum())),
        "map_stats": {
            "mean": float(np.average(s_map, weights=tc.eqs.vol[tc.eqs.part])),
            "min": float(s_map.min()), "max": float(s_map.max())},
        "gates": {"energy_residual_frac": float(tr.out["energy_residual_frac"]),
                  "clamp_bound": bool(tr.out["clamp_bound"]),
                  "cfl_violated": bool(tr.out["cfl_violated"])},
        # the shared cross-lane shape; `peak_T_c` is the TRUE trajectory
        # maximum, not the read-step or end-state value (studio3d ab08872)
        "standing_gates": gates.standing_gates(
            tr.out or {}, peak_T_c=float(max(
                float(np.max(tc.state_at(tr, j)))
                for j in (ka, int(tr.n_steps))))),
        "objective_prereg": {"phi_floor": obj.PHI_FLOOR,
                             "w_out_over_w_in": obj.W_OUT_OVER_W_IN},
    }
    if extra:
        rec.update(extra)
    if save_fields:
        RESULTS.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(RESULTS / f"field_{PART}_{name}.npz",
                            s_map=s_map, T_read=T_read, phi=phi,
                            chi=chi, vol=vol,
                            centroids=_part_centroids(tc))
    return rec


def step_to_time_s(tr, k: int, p) -> float:
    """Simulated time at trajectory step `k`, in seconds.

    The trajectory is indexed by CFL SUBSTEP, not by sample step, so the time
    is `k * dt_s / n_sub`. Using `k * dt_s` is off by n_sub -- on this part
    (n_sub 33) it reported t_stop_s = 8729.65 s for a run whose horizon was
    900 s. Inherited from phase_e/run.py, where n_sub is 1 and the error is
    invisible; it becomes a confidently impossible number on a fine mesh.
    """
    return float(k) * float(p.dt_s) / float(max(1, int(getattr(tr, "n_sub", 1))))


def _part_centroids(tc) -> np.ndarray:
    import dolfinx
    return np.asarray(dolfinx.mesh.compute_midpoints(
        tc.msh, tc.msh.topology.dim,
        np.arange(tc.ncells, dtype=np.int32)))[tc.eqs.part]


# --------------------------------------------------------------------------- #
# Arms
# --------------------------------------------------------------------------- #
def run_solve_arm(tc, chain, name: str, budget_evals: int) -> dict:
    """L-BFGS-B on the Phase B gradient, checkpointed after EVERY evaluation.

    The 1/|g0| rescale is the standing convention, not a deviation. The
    checkpoint is what makes a multi-hour run survivable; see
    solve3d/phase_e/checkpoint.py for what a resume does and does not restore.
    """
    from solve3d.phase_e import checkpoint as ck
    tc.set_objective("asymmetric")
    ckpt = RESULTS / f"ckpt_{PART}_{name}.npz"
    t0 = time.perf_counter()

    def fg(v):
        s = chain.design_to_map(v, 0.0)
        tr = tc.forward(tc.design_to_sigma(s))
        Jt = tc.J_trajectory(tr)
        k = int(np.argmin(Jt))
        J = float(Jt[k])
        g_s, _ = tc.gradient_design(s, tr=tr, read_step=k,
                                    checkpoint_interval=CHECKPOINT_INTERVAL)
        g_v = chain.design_vjp(v, g_s, beta=0.0)
        fg.last = {"argmin_step": k, "t_stop_s": step_to_time_s(tr, k, tc.p),
                   "at_horizon": bool(k >= tr.n_steps),
                   "wall_s": time.perf_counter() - t0}
        return J, g_v

    def on_eval(v, J, g):
        info = dict(getattr(fg, "last", {}))
        print(f"  [{name}] eval J={J:.6e} t_stop={info.get('t_stop_s')}s "
              f"|g|={np.linalg.norm(g):.3e} wall={info.get('wall_s'):.0f}s",
              flush=True)
        return info

    res = ck.run_with_checkpoint(fg, np.ones(chain.n_design),
                                 budget=budget_evals, path=ckpt,
                                 bounds=(0.0, 1.0), scale_first_step=True,
                                 on_eval=on_eval)
    v_best = np.asarray(res["best_v"], float)
    s_best = chain.design_to_map(v_best, 0.0)
    rec = score_arm(tc, s_best, name, extra={
        "status": res.get("status", "resumed_complete"),
        "objective_optimized": "asymmetric",
        "budget_gradient_evaluations": budget_evals,
        "gradient_evaluations_used": res["evals_used"],
        "resumed_from_eval": res["resumed_from_eval"],
        "J_first_eval": res["hist"][0]["J"] if res["hist"] else None,
        "trajectory": res["hist"], "scale_first_step": True,
        "objective_scale_applied": res["scale"],
        "checkpoint": ckpt.name,
        "wall_total_s": time.perf_counter() - t0})
    np.savez_compressed(RESULTS / f"map_{PART}_{name}.npz", v_raw=v_best,
                        s_map=s_best, centroids=chain.centroids,
                        volumes=chain.volumes)
    return rec


def _doc_path() -> Path:
    return RESULTS / f"phase_e_{PART}.json"


def _merge(key: str, rec) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = _doc_path()
    doc = json.loads(p.read_text()) if p.exists() else {"part": PART,
                                                        "arms": {}}
    doc.setdefault("arms", {})[key] = rec
    p.write_text(json.dumps(doc, indent=1, default=float))


def _have(key: str) -> bool:
    p = _doc_path()
    return p.exists() and key in json.loads(p.read_text()).get("arms", {})


def run(budget_evals: int = 40, stage: str = "all",
        lc_part: float = LC_PART_M) -> dict:
    """Arms are SKIPPED if already recorded, so a resume costs nothing."""
    print(f"[tamper] building case, lc_part={lc_part:.6f} m", flush=True)
    t0 = time.perf_counter()
    tc, info = build_case(lc_part=lc_part)
    chain = dc.DesignChain(_part_centroids(tc), tc.eqs.vol[tc.eqs.part],
                           FILTER_RADIUS_M, [0.0])
    print(f"[tamper] mesh: {info.n_cells_total} cells, "
          f"{info.n_part_cells} in part, {chain.n_design} design cells, "
          f"{time.perf_counter() - t0:.0f}s", flush=True)
    if not _have("_mesh"):
        _merge("_mesh", mesh_record(tc, info, chain))
    n = chain.n_design
    if stage in ("cheap", "all") and not _have("uniform_baseline"):
        print("[tamper] uniform baseline", flush=True)
        _merge("uniform_baseline",
               score_arm(tc, np.ones(n), "uniform_baseline"))
        print("[tamper] uniform baseline done", flush=True)
    if stage in ("solve", "all") and not _have("solve_filter_only"):
        print(f"[tamper] filter-only solve, budget {budget_evals}", flush=True)
        _merge("solve_filter_only",
               run_solve_arm(tc, chain, "solve_filter_only", budget_evals))
    return json.loads(_doc_path().read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=40)
    ap.add_argument("--stage", default="all",
                    choices=["cheap", "solve", "all"])
    ap.add_argument("--lc", type=float, default=LC_PART_M)
    a = ap.parse_args()
    run(budget_evals=a.budget, stage=a.stage, lc_part=a.lc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
