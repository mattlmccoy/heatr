"""On-demand DIRECT SOLVE service for the RFAM Print Studio (spec section 7e).

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.studio_solve part.npz \
        --out-dir DIR [--budget 40] [--warm-start sat.npz]

WHAT THIS IS. The Phase C solve, wrapped so the Studio can run it on an
IMPORTED part instead of the Phase A anchor circle. Every piece of physics,
every convention and both acceptance gates are the Phase C ones, reused:

  * forward / adjoint / objective  -- solve3d.forward, .adjoint, .objective
  * design chain                   -- solve3d.design_chain (filter-only,
                                      1.0 mm PHYSICAL radius, from the Phase C
                                      pre-registration)
  * solve                          -- L-BFGS-B, jac=True, box [0, 1],
                                      ftol/gtol 1e-16, budget in
                                      forward-equivalents, scaled first step
                                      (the recorded Phase C deviation, a pure
                                      1/|g0| reparameterization)
  * acceptance                     -- mesh hold-out with the pre-registered
                                      bands + 0.5 mm sub-filter smoothing at a
                                      10 % tolerance
  * cross-mesh transfer            -- phase_c_run.transfer_map_across_meshes

WHAT IS NEW HERE, and therefore what can be wrong here: the geometry front end
(solve3d.studio_geom, unit-tested) and the extruded-polygon mesher below.

NAMED DEVIATIONS FROM PHASE C (all recorded into studio_solve_results.json):
 D1 Mesh resolution is set by a NODE DENSITY carried over from the Phase C
    solve/score meshes (23040 and 77952 in-part nodes on the 1.885e-5 m^3
    anchor circle) rather than by those absolute node counts, because a Studio
    part has a different volume. The in-part ELEMENT SIZE is what is held
    fixed, which is the quantity the filter radius is quoted against.
 D2 Shape metrics (the hold-out's banded quantities) are read on a grid built
    around THIS part's bounding box with z stations inside its own z extent.
    Phase C used a fixed +-15 mm grid with z stations at +-20 mm, which only
    makes sense for a full-height 20 mm anchor. Metric DEFINITIONS are
    unchanged (solve3d.shape_metrics, gates.phase_fraction_phi).
 D3 The budget conversion uses Phase C's MEASURED 3.328 forward-equivalents per
    gradient evaluation, which was measured on the anchor circle at the Phase C
    solve mesh. This run's OWN measured cost is reported beside it; when they
    disagree the measured one is the truth about this part.
 D4 --warm-start is a recorded deviation from the frozen single-cold-start
    convention. Cold start stays the reference; a warm-started result is
    labelled `warm_start: true` and is NOT a same-budget comparison unless the
    caller ran the cold arm too.
 D5 chi: the objective's target indicator is the conforming-mesh nodal part
    fraction (Phase C's chi), and the shared fill contract's volume fill is run
    against the mesh as a CHECK (`chi_check` in the results), reporting whether
    every cell was exactly in or out. That is exactly what Phase C measured.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from solve3d import (adjoint, design_chain as dc, forward as fwd,
                     objective as obj, shape_metrics as sm, studio_geom as sg)
from solve3d import gates as G
from solve3d import stage_a as stage_a

RESULTS = Path(__file__).resolve().parent / "results"
L_DOMAIN = sg.L_DOMAIN

# --------------------------------------------------------------------------- #
# Ceiling-feasible drive selection (the Grade-and-Print PRODUCER, Matt path A).
#
# baseline power density: a = 1.0 is exactly the Studio's hardcoded 1.5915e6
# (studio3d/package.py:193). The recommended drive is a * baseline; carried as
# an ABSOLUTE W/m3 in studio_solve_results.json for the Studio to print at.
# --------------------------------------------------------------------------- #
DRIVE_BASELINE_W_PER_M3 = float(
    stage_a.recommended_power_settings(1.0)["power_density_w_per_m3"])
# candidate drive multipliers, low -> high; the HIGHEST feasible one is picked
# (most part throughput while staying under the ceiling). A ladder around the
# B4 square result (0.34x) widened so an arbitrary part can find feasibility.
CEILING_DRIVE_CANDIDATES = (0.26, 0.30, 0.34, 0.38, 0.42)

# Phase C anchor: circle d=20 mm, full height -> in-part node density.
_ANCHOR_VOLUME_M3 = float(np.pi * 0.010 ** 2 * 0.060)
SOLVE_NODE_DENSITY = 23040 / _ANCHOR_VOLUME_M3      # phase_a_coarse
SCORE_NODE_DENSITY = 77952 / _ANCHOR_VOLUME_M3      # phase_a_mid
SOLVE_LC0 = 0.0009375
SCORE_LC0 = 0.000625
MAX_TIME_S = 500.0
SAMPLE_DT_S = 50.0
CHECKPOINT_INTERVAL = 25
NODE_MATCH_TOL = 0.20


def prereg() -> dict:
    return json.loads((RESULTS / "phase_c_preregistration.json").read_text())


# --------------------------------------------------------------------------- #
# Ceiling-feasible drive selection: pure pick, physics probe, wiring, emission
# --------------------------------------------------------------------------- #
def select_recommended_drive(peaks_by_drive: dict, *, baseline: float,
                             ceiling_c: float, chamber_tag: str,
                             thermal_config_path: str, rho_target: float) -> dict:
    """Pick the ceiling-feasible drive from MEASURED uniform end-state peaks.

    `peaks_by_drive` maps a drive multiplier `a` to a dict with at least
    `true_peak_c`, `reached_rho` and `achieved_rho` (as measured by a uniform
    densify on THIS part's mesh at power a * baseline). A drive is FEASIBLE iff
    it both densifies (`reached_rho`) AND keeps the true end-state peak at or
    under the real ceiling. The HIGHEST feasible drive is recommended (most part
    throughput). Pure logic; no physics.

    HONEST-NULL (the false-green refusal): if no drive is feasible the part is
    drive-limited -- `recommended_power_density_w_per_m3` and
    `recommended_drive_frac` are BOTH None and `recommended_drive_reason` says
    why. A drive-limited part NEVER emits a cooking power number; the Studio
    consumer then falls back to nominal + drive_recommended=false + the
    heatr3d ceiling gate backstop.
    """
    ceiling_c = float(ceiling_c)
    baseline = float(baseline)
    rho_target = float(rho_target)
    candidates = []
    feasible = []
    for a in sorted(peaks_by_drive):
        rec = peaks_by_drive[a]
        peak = float(rec["true_peak_c"])
        reached = bool(rec["reached_rho"])
        under = bool(peak <= ceiling_c)
        is_feasible = bool(reached and under)
        candidates.append({
            "drive_a": float(a),
            "power_density_w_per_m3": float(a) * baseline,
            "true_peak_c": peak,
            "reached_rho": reached,
            "achieved_rho": float(rec.get("achieved_rho", float("nan"))),
            "under_ceiling": under,
            "feasible": is_feasible,
        })
        if is_feasible:
            feasible.append(float(a))

    out = {
        "chamber_tag": str(chamber_tag),
        "ceiling_c": ceiling_c,
        "thermal_config_path": str(thermal_config_path),
        "baseline_power_density_w_per_m3": baseline,
        "rho_target": rho_target,
        "candidates": candidates,
    }
    if feasible:
        a = max(feasible)                        # highest feasible drive
        peak = float(peaks_by_drive[a]["true_peak_c"])
        out["recommended_drive_frac"] = float(a)
        out["recommended_power_density_w_per_m3"] = float(a) * baseline
        out["recommended_drive_reason"] = (
            "ceiling_feasible: uniform end-state peak %.2f C <= %.1f C at %.3fx "
            "(highest drive that densifies to rho>=%.2f and stays under the "
            "ceiling in %s)" % (peak, ceiling_c, a, rho_target, chamber_tag))
        return out

    # honest-null: distinguish "cooks" from "never densifies" in the reason
    n_over = sum(1 for c in candidates if c["reached_rho"] and not c["under_ceiling"])
    n_cold = sum(1 for c in candidates if not c["reached_rho"])
    if n_over and not n_cold:
        why = ("every drive that densifies to rho>=%.2f exceeds the ceiling"
               % rho_target)
    elif n_cold and not n_over:
        why = "no drive reaches rho>=%.2f within the horizon" % rho_target
    else:
        why = ("no drive both densifies to rho>=%.2f and stays under the ceiling"
               % rho_target)
    out["recommended_drive_frac"] = None
    out["recommended_power_density_w_per_m3"] = None
    out["recommended_drive_reason"] = (
        "drive_limited: no feasible drive reaches rho_target under %.0fC (%s)"
        % (ceiling_c, why))
    return out


def _uniform_end_state_peak(msh, rings: list, z_lo: float, z_hi: float,
                            drive_a: float, *, baseline: float,
                            rho_target: float, max_time_s: float,
                            sample_dt_s: float = 20.0) -> dict:
    """Measure the UNIFORM (fully doped) densify end-state peak on THIS part's
    mesh at power a * baseline. Mirrors stage_a_phase2.ceiling_end_state_gate's
    forward read (solve_eqs -> qrf_dg0 -> march_densify), but on the Studio
    part's own conforming mesh -- an arbitrary extruded part has no
    pre-registered hold-out geometry, so the drive must be judged on the part
    that will actually print. A handful of densify forwards, not a heavy solve.
    """
    from solve3d import densify_forward as df
    import dataclasses

    pw = float(drive_a) * float(baseline)
    p = dataclasses.replace(fwd.ForwardParams(), power_density_w_per_m3=pw)
    mats = fwd.build_materials(msh, in_part_predicate(rings, z_lo, z_hi), p)
    Vr, Vi = fwd.solve_eqs(msh, mats, p)
    drive = fwd.qrf_dg0(msh, Vr, Vi, mats, p)
    march = df.march_densify(msh, p, stop_mean_rho=float(rho_target), mats=mats,
                             q_dg0=drive["q"], max_time_s=float(max_time_s),
                             sample_dt_s=float(sample_dt_s))
    return {
        "drive_a": float(drive_a),
        "power_density_w_per_m3": pw,
        "true_peak_c": float(march["true_peak_T_c"]),
        "reached_rho": bool(march["reached_rho"]),
        "achieved_rho": float(march["part_mean_rho"]),
    }


def recommended_drive_for_part(msh, rings: list, z_lo: float, z_hi: float, *,
                               candidates: tuple = CEILING_DRIVE_CANDIDATES,
                               baseline: float = DRIVE_BASELINE_W_PER_M3,
                               ceiling_c: float | None = None,
                               chamber_tag: str | None = None,
                               thermal_config_path: str = "solve3d/thermal_config.json",
                               rho_target: float | None = None,
                               max_time_s: float = 3000.0,
                               sample_dt_s: float = 20.0,
                               peak_probe=None) -> dict:
    """Run the ceiling-coupled drive selection for the uploaded part.

    Measures the uniform end-state peak at each candidate drive (via
    `peak_probe`, defaulting to the real densify probe on `msh`), then picks the
    highest feasible drive or honest-nulls. `peak_probe(drive_a, msh=..,
    rings=.., z_lo=.., z_hi=.., baseline=.., rho_target=.., max_time_s=..,
    sample_dt_s=..)` is injectable so the contract is testable without physics.
    """
    from solve3d import chamber as chamber_mod
    tcfg = stage_a.thermal_config()
    if ceiling_c is None:
        ceiling_c = float(tcfg["T_ceiling_C"])
    if rho_target is None:
        rho_target = float(tcfg["rho_target"]["practical_ideal"])
    if chamber_tag is None:
        chamber_tag = chamber_mod.chamber_tag(L_DOMAIN)
    probe = peak_probe if peak_probe is not None else _uniform_end_state_peak

    peaks = {}
    for a in candidates:
        rec = probe(float(a), msh=msh, rings=rings, z_lo=z_lo, z_hi=z_hi,
                    baseline=baseline, rho_target=rho_target,
                    max_time_s=max_time_s, sample_dt_s=sample_dt_s)
        peaks[float(a)] = rec
        print(f"[studio_solve drive={a:.3f}x pw={float(a) * baseline:.1f}] "
              f"uniform peak={rec['true_peak_c']:.2f}C "
              f"reached_rho={rec['reached_rho']} "
              f"rho={rec['achieved_rho']:.3f}", flush=True)
    return select_recommended_drive(
        peaks, baseline=baseline, ceiling_c=ceiling_c, chamber_tag=chamber_tag,
        thermal_config_path=thermal_config_path, rho_target=rho_target)


_RECOMMENDED_DRIVE_FIELDS = (
    "recommended_power_density_w_per_m3", "recommended_drive_frac",
    "chamber_tag", "ceiling_c", "thermal_config_path", "recommended_drive_reason")


def _merge_recommended_drive(doc: dict, rec: dict) -> dict:
    """Fold the PINNED contract fields into the studio_solve_results.json doc at
    the top level (the fields the Studio consumer reads), and carry the full
    selection record nested under `recommended_drive` for provenance."""
    for k in _RECOMMENDED_DRIVE_FIELDS:
        doc[k] = rec.get(k)
    doc["recommended_drive"] = rec
    return doc


# --------------------------------------------------------------------------- #
# The extruded-polygon conforming mesh (generalizes heatr3d_d1_spike/mesh_gmsh)
# --------------------------------------------------------------------------- #
def build_extruded_mesh(rings: list, z_lo: float, z_hi: float, lc_part: float,
                        lc_bed_factor: float = 4.0, L: float = L_DOMAIN,
                        seed: int = 1):
    """One conformal mesh of the chamber box with the extruded polygon as a
    separate volume (physical tag 1 = part, 2 = bed), the same construction
    mesh_gmsh.build uses for the cylinder / prism anchors.

    The part volume is identified from gmsh's fragment OUT-MAP, not by picking
    the smallest volume: a part with a bore splits the bed into pieces, and the
    smallest piece is then the BORE, not the part.
    """
    import gmsh
    from dolfinx.io import gmsh as dgmsh
    from mpi4py import MPI

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.clear()
        gmsh.model.add("studio_solve")
        occ = gmsh.model.occ
        box = occ.addBox(-L / 2, -L / 2, -L / 2, L, L, L)
        loops = []
        for i, ring in enumerate(rings):
            # OCC subtracts an inner wire only when it is oriented the SAME way
            # as the outer one; handed the CW hole ring straight from
            # studio_geom it ADDS the bore instead (measured: face area came out
            # outer+inner, 5.168e-4 instead of 3.516e-4 on the tube). Holes are
            # therefore reversed here, and the resulting solid volume is
            # asserted against the polygon volume below so this cannot regress
            # silently.
            r = np.asarray(ring, float)
            if i > 0:
                r = r[::-1]
            pts = [occ.addPoint(float(x), float(y), float(z_lo))
                   for x, y in r]
            lines = [occ.addLine(pts[i], pts[(i + 1) % len(pts)])
                     for i in range(len(pts))]
            loops.append(occ.addCurveLoop(lines))
        surf = occ.addPlaneSurface(loops)
        ext = occ.extrude([(2, surf)], 0.0, 0.0, float(z_hi - z_lo))
        part_in = [t for (d, t) in ext if d == 3]
        out, outmap = occ.fragment([(3, box)], [(3, t) for t in part_in])
        occ.synchronize()
        all_vols = [t for (d, t) in out if d == 3]
        part_vols = sorted({t for grp in outmap[1:] for (d, t) in grp if d == 3})
        bed_vols = [t for t in all_vols if t not in part_vols]
        if not part_vols or not bed_vols:
            raise RuntimeError("fragment did not separate part from bed "
                               f"(part={part_vols}, bed={bed_vols})")
        gmsh.model.addPhysicalGroup(3, part_vols, 1)
        gmsh.model.addPhysicalGroup(3, bed_vols, 2)

        # region size field over the part bounding box (the mesh_gmsh 'square'
        # branch, generalized to the polygon bbox)
        allp = np.vstack([np.asarray(r, float) for r in rings])
        f = gmsh.model.mesh.field
        t = f.add("Box")
        pad = 2.0 * lc_part
        for k, v in (("XMin", allp[:, 0].min() - pad), ("XMax", allp[:, 0].max() + pad),
                     ("YMin", allp[:, 1].min() - pad), ("YMax", allp[:, 1].max() + pad),
                     ("ZMin", z_lo - pad), ("ZMax", z_hi + pad)):
            f.setNumber(t, k, float(v))
        f.setNumber(t, "VIn", float(lc_part))
        f.setNumber(t, "VOut", float(lc_bed_factor * lc_part))
        f.setNumber(t, "Thickness", 3.0 * float(lc_part))
        mn = f.add("Min")
        f.setNumbers(mn, "FieldsList", [float(t)])
        f.setAsBackgroundMesh(mn)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)          # Delaunay
        gmsh.option.setNumber("Mesh.RandomSeed", seed)
        gmsh.model.mesh.generate(3)

        vol_m3 = float(sum(occ.getMass(3, t) for t in part_vols))
        v_expect = abs(sg.rings_area(rings)) * float(z_hi - z_lo)
        if abs(vol_m3 - v_expect) > 1e-9 * v_expect:
            raise RuntimeError(
                f"meshed part volume {vol_m3:.9e} m^3 does not match the "
                f"outline prism {v_expect:.9e} m^3 -- the hole handling or the "
                "fragment out-map is wrong; refusing to solve on it")
        tags = np.concatenate([gmsh.model.mesh.getNodes(3, t,
                                                        includeBoundary=True)[0]
                               for t in part_vols])
        n_part_nodes = int(np.unique(tags).size)
        md = dgmsh.model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
        msh = md.mesh
    finally:
        gmsh.finalize()
    tdim = msh.topology.dim
    info = {"lc_part_m": float(lc_part), "lc_bed_m": float(lc_bed_factor * lc_part),
            "n_nodes_in_part": n_part_nodes,
            "n_nodes_total": int(msh.geometry.index_map().size_global),
            "n_cells_total": int(msh.topology.index_map(tdim).size_global),
            "part_volume_m3_gmsh": vol_m3,
            "z_lo_m": float(z_lo), "z_hi_m": float(z_hi)}
    return msh, info


def match_lc_extruded(rings: list, z_lo: float, z_hi: float,
                      target_nodes_in_part: int, lc0: float,
                      tol: float = NODE_MATCH_TOL, max_iter: int = 5):
    """mesh_gmsh.match_lc's node-count matching, for the extruded polygon."""
    lc, history, msh, info = float(lc0), [], None, None
    for _ in range(max_iter):
        msh, info = build_extruded_mesh(rings, z_lo, z_hi, lc)
        ratio = info["n_nodes_in_part"] / float(target_nodes_in_part)
        history.append({"lc_part_m": lc,
                        "n_nodes_in_part": info["n_nodes_in_part"],
                        "ratio_vs_target": ratio})
        print(f"  [mesh] lc={lc:.6g} m -> {info['n_nodes_in_part']} in-part "
              f"nodes (target {target_nodes_in_part}, ratio {ratio:.3f})",
              flush=True)
        if abs(ratio - 1.0) <= tol:
            break
        lc = lc * ratio ** (1.0 / 3.0)
    info["match_history"] = history
    info["target_nodes_in_part"] = int(target_nodes_in_part)
    return msh, info


def in_part_predicate(rings: list, z_lo: float, z_hi: float):
    """Midpoint predicate for the extruded polygon; `mp` is (3, ncell)."""
    def pred(mp):
        mp = np.asarray(mp, float)
        return (sg.points_in_rings(rings, mp[0], mp[1])
                & (mp[2] >= z_lo) & (mp[2] <= z_hi))
    return pred


def build_case(rings: list, z_lo: float, z_hi: float, node_density: float,
               lc0: float, p: fwd.ForwardParams, max_time_s: float,
               sample_dt_s: float):
    v_part = abs(sg.rings_area(rings)) * (z_hi - z_lo)
    target = max(int(round(node_density * v_part)), 200)
    msh, info = match_lc_extruded(rings, z_lo, z_hi, target, lc0)
    mats = fwd.build_materials(msh, in_part_predicate(rings, z_lo, z_hi), p)
    eqs = adjoint.SteadyEqs(msh, mats, p)
    tc = adjoint.TransientCase(msh, mats, p, eqs, info, sample_dt_s, max_time_s)
    info["part_volume_m3_polygon"] = float(v_part)
    info["n_design_cells"] = int(eqs.part.size)
    info["part_volume_m3_cells"] = float(eqs.vol[eqs.part].sum())
    return tc, info


# --------------------------------------------------------------------------- #
# chi check via the shared fill contract (Phase C's measurement, not a claim)
# --------------------------------------------------------------------------- #
def chi_check(tc, rings: list, z_lo: float, z_hi: float,
              max_cells: int = 20000) -> dict:
    """Run the fill contract's volume fill against the mesh and report whether
    every cell came out exactly 0 or 1 (i.e. the mesh conformed), plus the max
    deviation from the DG0 indicator the objective actually uses."""
    from solve3d import fill
    import dolfinx
    ncell = tc.ncells
    idx = np.arange(ncell, dtype=np.int32)
    if ncell > max_cells:                     # sample, and say so
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(ncell, size=max_cells,
                                 replace=False)).astype(np.int32)
    geo = tc.msh.geometry
    dofmap = getattr(geo, "dofmaps", None)
    dofs = (dofmap(0) if callable(dofmap) else geo.dofmap)[idx]
    verts = geo.x[dofs][:, :4, :]
    bary = fill._tet_lattice(6)
    w = np.concatenate([1.0 - bary.sum(axis=1, keepdims=True), bary], axis=1)
    pts = np.einsum("qk,ckd->cqd", w, verts).reshape(-1, 3)
    ins = sg.points_in_rings(rings, pts[:, 0], pts[:, 1])
    ins &= (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi)
    frac = ins.reshape(idx.size, -1).mean(axis=1)
    ind = np.real(tc.mats.doped.x.array).astype(float)[idx]
    return {"n_cells_checked": int(idx.size), "n_cells_total": int(ncell),
            "sampled": bool(idx.size < ncell),
            "mesh_conformed": bool(np.all((frac == 0.0) | (frac == 1.0))),
            "n_partial_cells": int(np.count_nonzero((frac > 0.0) & (frac < 1.0))),
            "max_abs_diff_vs_dg0_indicator": float(np.max(np.abs(frac - ind))),
            "quadrature": "symmetric barycentric lattice (solve3d.fill, the "
                          "documented tetrahedral deviation from the shared "
                          "contract's rectangular sub-cell grid)",
            "rings": "even-odd across outer ring and holes"}


# --------------------------------------------------------------------------- #
# Scoring (Phase C's score_arm, with the D2 part-adapted read grid)
# --------------------------------------------------------------------------- #
def eval_grid(rings: list, z_lo: float, z_hi: float, n_xy: int = 200,
              n_z: int = 5, pad_m: float = 0.005):
    allp = np.vstack([np.asarray(r, float) for r in rings])
    half = float(np.max(np.abs(allp))) + pad_m
    h = 2.0 * half / n_xy
    c = (np.arange(n_xy) + 0.5) * h - half
    zs = z_lo + (np.arange(n_z) + 0.5) / n_z * (z_hi - z_lo)
    Z, X, Y = np.meshgrid(zs, c, c, indexing="ij")
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    XX, YY = np.meshgrid(c, c, indexing="ij")
    mask = sg.points_in_rings(rings, XX.ravel(), YY.ravel()).reshape(XX.shape)
    return pts, (zs.size, n_xy, n_xy), mask, h


def score_arm(tc, s_map: np.ndarray, name: str, grid_cache: dict) -> dict:
    """One delivered map: both objective weightings at their own argmin, the
    banded shape metrics, and sigma_T as a diagnostic (never optimized)."""
    t0 = time.perf_counter()
    tr = tc.forward(tc.design_to_sigma(s_map))
    wall = time.perf_counter() - t0
    tc.set_objective("symmetric")
    Js = tc.J_trajectory(tr)
    k = int(np.argmin(Js))
    tc.set_objective("asymmetric")
    Ja = tc.J_trajectory(tr)
    ka = int(np.argmin(Ja))
    T_read = tc.state_at(tr, ka)
    # STANDING GATES, emitted with every arm so the Studio's scheduler can
    # tell "the acceptance gates did not pass" from "the forward was not
    # physical". Those are different verdicts and only one of them means the
    # number must not be shown. See solve3d/gates.standing_gates.
    stand = G.standing_gates(tr.out or {})
    phi = fwd.phase_fraction(T_read, tc.p)[0]
    vol, chi = tc.vol_nodal, tc.m_nodal
    split = obj.split_asymmetric(phi, chi, vol)
    split3 = obj.split_asymmetric(phi, chi, vol, w_ratio=obj.W_SENSITIVITY)

    pts, shp, part2d, _ = grid_cache["grid"]
    W = fwd.functionspace(tc.msh, ("Lagrange", 1))
    Tf = fwd.fem.Function(W)
    Tf.x.array[:] = T_read.astype(fwd.dolfinx.default_scalar_type)
    Te, missed = fwd.eval_at(Tf, tc.msh, pts)
    Te = np.nan_to_num(Te, nan=tc.p.preheat_c).reshape(shp)
    per = []
    for i in range(Te.shape[0]):
        p2 = G.phase_fraction_phi(Te[i])
        per.append({"in08": sm.in_part_melt_fraction(p2 >= 0.8, part2d),
                    "in09": sm.in_part_melt_fraction(p2 >= 0.9, part2d),
                    "bed08": sm.out_of_part_fraction(p2 >= 0.8, part2d),
                    "bed09": sm.out_of_part_fraction(p2 >= 0.9, part2d)})
    agg = {k2: float(np.mean([r[k2] for r in per])) for k2 in per[0]}
    part_w = vol * chi
    mu = float(np.dot(T_read, part_w) / part_w.sum())
    return {
        "arm": name, "wall_forward_s": wall, "n_steps": tr.n_steps,
        "J_symmetric": float(Js[k]), "argmin_symmetric": k,
        "J_asymmetric": float(Ja[ka]), "argmin_asymmetric": ka,
        "t_stop_s": float(ka * tc.p.dt_s),
        "at_horizon_asymmetric": bool(ka >= tr.n_steps),
        "J_asym_w3_at_same_read": split3["J_asym"],
        "J_out_of_bounds": split["J_out_of_bounds"],
        "J_in_bounds_deficit": split["J_in_bounds_deficit"],
        "out_of_part_melt_fraction_of_part":
            split["out_of_bounds_melt_fraction_of_part"],
        "in_bounds_below_floor_fraction": split["in_bounds_below_floor_fraction"],
        "part_mean_phi": float(np.dot(phi, part_w) / part_w.sum()),
        "sigma_T_diagnostic_c": float(np.sqrt(
            np.dot((T_read - mu) ** 2, part_w) / part_w.sum())),
        "in_part_melt_frac_phi08": agg["in08"],
        "in_part_melt_frac_phi09": agg["in09"],
        "bed_melt_frac_phi08": agg["bed08"], "bed_melt_frac_phi09": agg["bed09"],
        "eval_missed": int(missed),
        "map_stats": {"mean": float(np.average(
            s_map, weights=tc.eqs.vol[tc.eqs.part])),
            "min": float(np.min(s_map)), "max": float(np.max(s_map))},
        "standing_gates": stand,
        "forward_physical": stand["forward_physical"],
        "peak_T_c": stand["peak_T_c"],
        "peak_over_ceiling": stand["peak_over_ceiling"],
    }


# --------------------------------------------------------------------------- #
# The solve (Phase C run_solve_arm, with STUDIO_SOLVE_PROGRESS lines)
# --------------------------------------------------------------------------- #
class _BudgetExhausted(Exception):
    pass


def warm_start_vector(sat_path, tc, chain) -> dict:
    """Sample a voxel sat volume at the design centroids (nearest voxel).

    RECORDED DEVIATION D4. Nearest-voxel rather than trilinear on purpose: the
    Phase C inversion arm was dropped because interpolating a voxel map whose
    support is the STAIRCASE part pulls in zeros at the rim and thins the map
    (PHASE_C_REPORT section 4). Nearest sampling cannot do that; cells that
    still land on a zero (outside the source support) are counted and filled
    with the map mean rather than silently zeroed."""
    with np.load(sat_path) as d:
        sat = np.asarray(d["sat"], float)
    n = sat.shape[0]
    h = L_DOMAIN / n
    c = chain.centroids
    idx = np.clip(np.floor((c + L_DOMAIN / 2.0) / h).astype(int), 0, n - 1)
    v = sat[idx[:, 0], idx[:, 1], idx[:, 2]]
    inside = v > 0.0
    n_fallback = int(np.count_nonzero(~inside))
    fill_value = float(v[inside].mean()) if inside.any() else 1.0
    v = np.where(inside, v, fill_value)
    return {"v0": np.clip(v, 0.0, 1.0), "source": str(sat_path),
            "source_grid_n": int(n), "n_cells_outside_source_support": n_fallback,
            "fallback_value": fill_value, "method": "nearest_voxel",
            "note": "the raw design v is initialized to the transferred "
                    "saturation; filter(v0) is not exactly v0, so the first "
                    "iterate is a smoothed version of the warm start"}


def run_solve(tc, chain, budget_evals: int, v0: np.ndarray,
              objective_name: str = "asymmetric", beta: float = 0.0) -> dict:
    from scipy.optimize import minimize
    tc.set_objective(objective_name)
    n = chain.n_design
    st = {"n": 0, "best_J": np.inf, "best_v": np.asarray(v0, float).copy(),
          "hist": [], "t0": time.perf_counter(), "scale": 1.0,
          "wall_fwd": None, "wall_grad": None}

    def fg(v):
        if st["n"] >= budget_evals:
            raise _BudgetExhausted()
        s = chain.design_to_map(v, beta)
        t0 = time.perf_counter()
        tr = tc.forward(tc.design_to_sigma(s))
        t_fwd = time.perf_counter() - t0
        Jt = tc.J_trajectory(tr)
        k = int(np.argmin(Jt))
        J = float(Jt[k])
        t0 = time.perf_counter()
        g_s, _ = tc.gradient_design(s, tr=tr, read_step=k,
                                    checkpoint_interval=CHECKPOINT_INTERVAL)
        t_grad = time.perf_counter() - t0
        g_v = chain.design_vjp(v, g_s, beta=beta)
        st["n"] += 1
        if st["n"] == 1:
            st["wall_fwd"], st["wall_grad"] = t_fwd, t_grad
            gn = float(np.linalg.norm(g_v))
            st["scale"] = (1.0 / gn) if gn > 0 else 1.0
        if J < st["best_J"]:
            st["best_J"], st["best_v"] = J, np.asarray(v, float).copy()
        wall = time.perf_counter() - st["t0"]
        st["hist"].append({
            "eval": st["n"], "J": J, "argmin_step": k,
            "t_stop_s": float(k * tc.p.dt_s), "at_horizon": bool(k >= tr.n_steps),
            "grad_norm": float(np.linalg.norm(g_v)),
            "map_mean": float(np.average(s, weights=tc.eqs.vol[tc.eqs.part])),
            "wall_s": wall})
        print(f"STUDIO_SOLVE_PROGRESS eval={st['n']}/{budget_evals} "
              f"J={J:.6e} wall_s={wall:.1f}", flush=True)
        return J * st["scale"], g_v * st["scale"]

    status = "budget_exhausted"
    try:
        res = minimize(fg, np.asarray(v0, float), jac=True, method="L-BFGS-B",
                       bounds=[(0.0, 1.0)] * n,
                       options={"ftol": 1e-16, "gtol": 1e-16, "maxiter": 10000})
        status = f"converged:{res.message}"
    except _BudgetExhausted:
        pass
    # A read at step 0 means NOTHING melted inside the horizon: phi is 0
    # everywhere for the whole march, so J is constant in time, the envelope
    # argmin lands on the initial state and the reverse sweep has no steps to
    # walk -- the "solve" then moves nothing while looking like it ran. Measured
    # on a 20 s shakedown horizon (gradient wall 8e-4 s). Flagged, never hidden.
    degenerate = bool(st["hist"] and all(r["argmin_step"] == 0
                                         for r in st["hist"]))
    return {"status": status, "v_best": st["best_v"], "J_best": st["best_J"],
            "degenerate_read_at_step_zero": degenerate,
            "degenerate_note": (
                "every evaluation's envelope argmin was step 0: no melt occurred "
                "within max_time_s, so J is constant in time and the gradient is "
                "empty. The horizon is too short for this part; the solve result "
                "is NOT usable." if degenerate else None),
            "trajectory": st["hist"], "evals_used": st["n"],
            "objective_scale_applied": st["scale"],
            "wall_total_s": time.perf_counter() - st["t0"],
            "measured_wall_forward_s": st["wall_fwd"],
            "measured_wall_gradient_s": st["wall_grad"],
            "measured_forward_equivalents_per_eval":
                (1.0 + st["wall_grad"] / st["wall_fwd"])
                if st["wall_fwd"] else None}


# --------------------------------------------------------------------------- #
# Acceptance gates (Phase C's, verbatim protocol)
# --------------------------------------------------------------------------- #
def run_gates(tc, chain, s_best: np.ndarray, solved_rec: dict,
              uniform_rec: dict, rings: list, z_lo: float, z_hi: float,
              p: fwd.ForwardParams, grid_cache: dict) -> dict:
    from solve3d.phase_c_run import transfer_map_across_meshes
    pr = prereg()["acceptance"]
    r_filter = float(prereg()["design_chain"]["filter_radius_m"])
    src_c, src_v = chain.centroids, chain.volumes

    # -------- mesh hold-out -------- #
    import dolfinx
    tc_f, info_f = build_case(rings, z_lo, z_hi, SCORE_NODE_DENSITY, SCORE_LC0,
                              p, tc.max_time_s, tc.sample_dt_s)
    grid_f = {"grid": grid_cache["grid"]}
    mp_f = np.asarray(dolfinx.mesh.compute_midpoints(
        tc_f.msh, tc_f.msh.topology.dim,
        np.arange(tc_f.ncells, dtype=np.int32)))[tc_f.eqs.part]
    tr_map = transfer_map_across_meshes(src_c, src_v, s_best, mp_f, r_filter)
    s_fine = tr_map.pop("map")
    dop_src = float(np.dot(s_best, src_v))
    dop_dst = float(np.dot(s_fine, tc_f.eqs.vol[tc_f.eqs.part]))
    tr_map.update({"total_dopant_src_m3": dop_src, "total_dopant_dst_m3": dop_dst,
                   "total_dopant_rel_move": abs(dop_dst - dop_src) / dop_src})
    uni_f = score_arm(tc_f, np.ones(tc_f.eqs.part.size),
                      "uniform_at_score_mesh", grid_f)
    sol_f = score_arm(tc_f, s_fine, "solved_at_score_mesh", grid_f)

    def _J(rec):
        return rec["J_asymmetric"]

    j_band = 1.5 * abs(_J(uni_f) - _J(uniform_rec)) / abs(_J(uniform_rec))
    j_move = abs(_J(sol_f) - _J(solved_rec)) / abs(_J(solved_rec))
    checks = {"J_rel": {"measured": j_move, "band": j_band,
                        "pass": bool(j_move <= j_band),
                        "band_rule": pr["mesh_holdout"]["bands_deferred"]["J_rel"]["rule"],
                        "uniform_own_move":
                            abs(_J(uni_f) - _J(uniform_rec)) / abs(_J(uniform_rec))}}
    for key, mine in (("in_part_absdiff_phi0p8", "in_part_melt_frac_phi08"),
                      ("in_part_absdiff_phi0p9", "in_part_melt_frac_phi09"),
                      ("bed_melt_absdiff_phi0p8", "bed_melt_frac_phi08"),
                      ("bed_melt_absdiff_phi0p9", "bed_melt_frac_phi09")):
        band = float(pr["mesh_holdout"]["bands"][key])
        moved = abs(sol_f[mine] - solved_rec[mine])
        checks[key] = {"measured": moved, "band": band,
                       "pass": bool(moved <= band),
                       "uniform_own_move": abs(uni_f[mine] - uniform_rec[mine])}
    holdout = {"solve_mesh_nodes_in_part": chain.n_design,
               "score_mesh": info_f, "transfer": tr_map, "checks": checks,
               "pass": all(c["pass"] for c in checks.values()),
               "solved_at_solve_mesh": _J(solved_rec),
               "solved_at_score_mesh": _J(sol_f),
               "uniform_at_solve_mesh": _J(uniform_rec),
               "uniform_at_score_mesh": _J(uni_f),
               "scores": {"uniform_at_score_mesh": uni_f,
                          "solved_at_score_mesh": sol_f}}

    # -------- smoothing robustness -------- #
    r_sub = float(pr["smoothing_robustness"]["perturbation_radius_m"])
    blur = dc.DesignChain(src_c, src_v, r_sub, [0.0])
    rec_blur = score_arm(tc, blur.filter_apply(s_best),
                         f"blurred_{r_sub * 1e3:g}mm", grid_cache)
    dJ = abs(_J(rec_blur) - _J(solved_rec)) / abs(_J(solved_rec))
    tol = float(pr["smoothing_robustness"]["tolerance_rel_J"])
    smooth = {"perturbation_radius_m": r_sub, "filter_radius_m": r_filter,
              "J_unblurred": _J(solved_rec), "J_blurred": _J(rec_blur),
              "rel_change": dJ, "tolerance": tol, "pass": bool(dJ <= tol),
              "scores": {"blurred": rec_blur}}
    return {"mesh_holdout": holdout, "smoothing_robustness": smooth}


# --------------------------------------------------------------------------- #
# The service
# --------------------------------------------------------------------------- #
def solve_extruded(part_npz, out_dir, budget_fwd_equiv: float = 40.0,
                   warm_start_sat=None, max_time_s: float = MAX_TIME_S,
                   sample_dt_s: float = SAMPLE_DT_S,
                   ceiling_drive: bool = False,
                   drive_candidates: tuple = CEILING_DRIVE_CANDIDATES,
                   drive_max_time_s: float = 3000.0,
                   _peak_probe=None) -> dict:
    """Solve one imported EXTRUDED part; write the Studio's artifact pair.

    ceiling_drive (Matt path A: the ceiling-coupled solve is the standard path):
    when True, ALSO run the ceiling-feasible drive selection for THIS part and
    emit the PINNED contract fields (recommended_power_density_w_per_m3 etc.)
    into studio_solve_results.json. Default False preserves the exact legacy
    output (no new fields, no extra forwards); the CLI turns it on by default.
    `_peak_probe` injects a stubbed drive-selection probe for wiring tests.
    """
    t_start = time.perf_counter()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(part_npz) as d:
        part = np.asarray(d["part"], bool)
        h = float(d["h"]) if "h" in d.files else L_DOMAIN / part.shape[0]
        n = int(d["n"]) if "n" in d.files else part.shape[0]

    # 1. extrusion detection -- DETECTED, no shape heuristic
    det = sg.detect_extrusion(part)
    print(f"[studio_solve] extrusion detection: {det['is_extruded']}", flush=True)
    if not det["is_extruded"]:
        raise sg.NotExtrudedError(det["refusal"])

    # 2. outline + extruded conforming mesh
    z_lo, z_hi = sg.z_extent(part, h)
    rings = sg.outline_rings(sg.mid_slice(part), h)
    print(f"[studio_solve] outline: {len(rings)} ring(s), "
          f"{sum(len(r) for r in rings)} vertices, net area "
          f"{sg.rings_area(rings):.6e} m^2, z {z_lo:.4f}..{z_hi:.4f} m",
          flush=True)
    p = fwd.ForwardParams()
    tc, info = build_case(rings, z_lo, z_hi, SOLVE_NODE_DENSITY, SOLVE_LC0,
                          p, max_time_s, sample_dt_s)
    chain = dc.DesignChain.build(tc)
    grid_cache = {"grid": eval_grid(rings, z_lo, z_hi)}

    # 3. chi (shared fill contract, as a measurement)
    chi = chi_check(tc, rings, z_lo, z_hi)
    print(f"[studio_solve] chi check: conformed={chi['mesh_conformed']} "
          f"partial={chi['n_partial_cells']} "
          f"maxdiff={chi['max_abs_diff_vs_dg0_indicator']:.3e}", flush=True)

    # 4. the solve
    per_eval = float(prereg()["budget"]["measured_at_solve_scale"]
                     ["per_gradient_eval_forward_equivalents"])
    budget_evals = max(1, int(round(float(budget_fwd_equiv) / per_eval)))
    warm = None
    if warm_start_sat:
        warm = warm_start_vector(warm_start_sat, tc, chain)
        v0 = warm["v0"]
    else:
        v0 = np.ones(chain.n_design)
    uniform_rec = score_arm(tc, np.ones(chain.n_design), "uniform_baseline",
                            grid_cache)
    print(f"[studio_solve] uniform baseline J_asym={uniform_rec['J_asymmetric']:.6e}",
          flush=True)
    solve = run_solve(tc, chain, budget_evals, v0)
    s_best = chain.design_to_map(solve["v_best"], 0.0)
    solved_rec = score_arm(tc, s_best, "solved", grid_cache)

    # 5. both acceptance gates
    gate = run_gates(tc, chain, s_best, solved_rec, uniform_rec, rings,
                     z_lo, z_hi, p, grid_cache)
    beats = bool(solved_rec["J_asymmetric"] < uniform_rec["J_asymmetric"])
    improvement = ((uniform_rec["J_asymmetric"] - solved_rec["J_asymmetric"])
                   / abs(uniform_rec["J_asymmetric"]))
    solved_label = bool(beats and gate["mesh_holdout"]["pass"]
                        and gate["smoothing_robustness"]["pass"]
                        and not solve["degenerate_read_at_step_zero"])

    # 6. artifacts
    sg.write_map_npz(out / "studio_solve_map.npz", chain.centroids, s_best,
                     chain.volumes, solve["v_best"])
    doc = {
        "what": "solve3d direct solve for the RFAM Print Studio (spec 7e): "
                "extruded part -> conforming mesh -> Phase C filter-only "
                "scaled-first-step asymmetric solve -> both Phase C gates",
        "part": {"npz": str(part_npz), "grid_n": n, "h_m": h,
                 "n_voxels": int(part.sum()),
                 "voxel_volume_m3": float(part.sum() * h ** 3)},
        "extrusion_detection": det,
        "outline": {"n_rings": len(rings),
                    "n_vertices": [int(len(r)) for r in rings],
                    "ring_areas_m2": [sg.ring_area(r) for r in rings],
                    "net_area_m2": sg.rings_area(rings),
                    "z_lo_m": z_lo, "z_hi_m": z_hi,
                    "method": "sub-cell marching squares at the 0.5 level of "
                              "the voxel mask (NOT the staircase); inside-left "
                              "orientation, saddles resolved inside-connected"},
        "mesh": info, "chi_check": chi,
        "design_chain": {**chain.kernel_report(), "beta": 0.0,
                         "form": "filter-only (Phase C primary)"},
        "objective": {"primary": "asymmetric", "w_out_over_w_in": obj.W_OUT_OVER_W_IN,
                      "phi_floor": obj.PHI_FLOOR,
                      "read": "argmin over the arm's own trajectory (envelope)"},
        "budget": {"forward_equivalents_requested": float(budget_fwd_equiv),
                   "per_gradient_eval_forward_equivalents_phase_c": per_eval,
                   "gradient_evaluations": budget_evals,
                   "gradient_evaluations_used": solve["evals_used"],
                   "forward_equivalents_spent": solve["evals_used"] * per_eval,
                   "measured_forward_equivalents_per_eval":
                       solve["measured_forward_equivalents_per_eval"],
                   "measured_wall_forward_s": solve["measured_wall_forward_s"],
                   "measured_wall_gradient_s": solve["measured_wall_gradient_s"]},
        "case": {"max_time_s": max_time_s, "dt_s": p.dt_s, "coupling": "off",
                 "densification": "none (rho fixed, as in Phase C)"},
        "solve": {k: v for k, v in solve.items() if k != "v_best"},
        "J_trajectory": [r["J"] for r in solve["trajectory"]],
        "uniform_baseline": uniform_rec,
        "solved": solved_rec,
        "improvement_pct": 100.0 * improvement,
        "beats_uniform_in_grid": beats,
        "gates": {"mesh_holdout_pass": gate["mesh_holdout"]["pass"],
                  "smoothing_pass": gate["smoothing_robustness"]["pass"],
                  "detail": gate},
        # SURFACED AT THE TOP LEVEL on the Studio lane's request: their rung
        # needs to separate "acceptance gates not passed" (a real solve that
        # did not help) from "forward not physical" (a number that must not be
        # shown at all) WITHOUT having to infer it from solved_label.
        "standing_gates": {"uniform_baseline": uniform_rec["standing_gates"],
                           "solved": solved_rec["standing_gates"]},
        "forward_physical": bool(uniform_rec["forward_physical"]
                                 and solved_rec["forward_physical"]),
        "peak_over_ceiling": bool(uniform_rec["peak_over_ceiling"]
                                  or solved_rec["peak_over_ceiling"]),
        "solved_label": solved_label,
        "solved_label_rule": prereg()["acceptance"]["solved_label_rule"],
        "warm_start": bool(warm_start_sat),
        "warm_start_detail": ({k: v for k, v in warm.items() if k != "v0"}
                              if warm else None),
        "deviations": [
            "D1 mesh resolution set by the Phase C in-part NODE DENSITY (the "
            "element size is what is held fixed), because a Studio part has a "
            "different volume than the anchor circle",
            "D2 shape metrics read on a grid built around THIS part's bbox "
            "with z stations inside its own z extent; metric definitions "
            "unchanged",
            "D3 budget converted with Phase C's MEASURED 3.328 fwd-eq per "
            "gradient evaluation (measured on the anchor circle); this run's "
            "own measured cost is reported beside it",
            "D4 warm start (only when --warm-start is given) deviates from the "
            "frozen single-cold-start convention; cold start stays the "
            "reference" if warm_start_sat else
            "D4 not exercised: cold start v=1, the frozen convention",
            "D5 chi for the objective is the conforming-mesh nodal part "
            "fraction; the shared fill contract is run as a CHECK (chi_check)",
            "scaled first step (1/|g0|): the recorded Phase C deviation, a "
            "pure reparameterization that cannot move the minimizer",
        ],
        "not_covered": [
            "non-extruded parts (refused; needs Phase E STL tet meshing)",
            "densification, eps_r channel, drive reconciliation, coupling",
            "physics trust is unchanged: this certifies the DESIGN METHOD on "
            "the dolfinx forward, sim-only",
        ],
        "wall_seconds": time.perf_counter() - t_start,
    }

    # PRODUCER (path A): fold in the ceiling-feasible recommended drive. Off by
    # default so the legacy artifact is byte-identical; the CLI/Studio turns it
    # on. The drive is measured on THIS part's own solve mesh (an arbitrary
    # extruded part has no pre-registered hold-out) and is valid ONLY in the
    # chamber it was solved in -- carried as chamber_tag so the Studio verifies
    # like-for-like.
    if ceiling_drive:
        rec = recommended_drive_for_part(
            tc.msh, rings, z_lo, z_hi, candidates=drive_candidates,
            max_time_s=drive_max_time_s, peak_probe=_peak_probe)
        rec["drive_probe_mesh"] = {"which": "solve_mesh",
                                   "n_nodes_in_part": info.get("n_nodes_in_part")}
        _merge_recommended_drive(doc, rec)
        print(f"[studio_solve] recommended drive: "
              f"frac={doc['recommended_drive_frac']} "
              f"pw={doc['recommended_power_density_w_per_m3']} "
              f"reason={doc['recommended_drive_reason']}", flush=True)

    (out / "studio_solve_results.json").write_text(
        json.dumps(doc, indent=1, default=G._jsonable))
    print(f"[studio_solve] DONE solved_label={solved_label} "
          f"improvement={100.0 * improvement:+.2f}% "
          f"wall={doc['wall_seconds']:.0f}s", flush=True)
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description="solve3d direct solve for the Studio")
    ap.add_argument("part_npz", nargs="?",
                    help="npz with part (n,n,n) bool, h, n")
    ap.add_argument("--out-dir")
    ap.add_argument("--budget", type=float, default=40.0,
                    help="budget in FORWARD-EQUIVALENTS (Phase C counting)")
    ap.add_argument("--warm-start", default=None,
                    help="npz with a (n,n,n) sat volume (recorded deviation)")
    ap.add_argument("--max-time-s", type=float, default=MAX_TIME_S)
    ap.add_argument("--ceiling-drive", action=argparse.BooleanOptionalAction,
                    default=True,
                    help="run the ceiling-feasible drive selection and emit the "
                         "recommended_power_density_w_per_m3 contract (path A, "
                         "default on); --no-ceiling-drive for the legacy output")
    ap.add_argument("--drive-max-time-s", type=float, default=3000.0,
                    help="densify horizon for the uniform drive-probe forwards")
    ap.add_argument("--make-tube", default=None, metavar="OUT_NPZ",
                    help="write the validation tube part and exit")
    ap.add_argument("--n", type=int, default=32)
    args = ap.parse_args()
    if args.make_tube:
        part = sg.make_tube_part(n=args.n)
        np.savez_compressed(args.make_tube, part=part, h=L_DOMAIN / args.n,
                            n=args.n)
        print(f"wrote {args.make_tube}: {int(part.sum())} voxels")
        return 0
    if not args.part_npz or not args.out_dir:
        ap.error("part_npz and --out-dir are required")
    solve_extruded(args.part_npz, args.out_dir, budget_fwd_equiv=args.budget,
                   warm_start_sat=args.warm_start, max_time_s=args.max_time_s,
                   ceiling_drive=args.ceiling_drive,
                   drive_max_time_s=args.drive_max_time_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
