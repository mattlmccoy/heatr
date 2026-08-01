"""The two acceptance gates for the topology-optimization maps.

Both are FORWARD RUNS ONLY. Nothing is re-solved. A map does not get a SOLVED
label unless both pass, which is the rule the three-dimensional port lane
adopted and this pass inherits.

GATE A, GRID HOLD-OUT. Solve at grid 120, score at grid 160 with the drive
voltage RECALIBRATED at 160 so the uniform arm absorbs 500 W per metre of depth
there (`robust.recalibrated_voltage`, one exact rescale because the
electro-quasi-static solve is quadratic in the drive). Two transfers are run and
they answer different questions:

  map transfer      the CONTINUOUS saturation map is resampled 120 -> 160 in the
                    production map-injection convention (bilinear, then clipped)
                    and re-quantized. This is what the printer pipeline actually
                    does and it is the transfer every earlier report measured.
  design transfer   the DESIGN VARIABLE v is resampled 120 -> 160 and the filter
                    and projection are re-applied AT 160 with the same PHYSICAL
                    radius. This is the transfer the parameterization itself
                    claims: the design is a field of 1.0 mm features, and a
                    1.0 mm feature is representable on both grids.

The target chi is rebuilt at 160 from the geometry, so unlike every earlier
grid hold-out the two grids are scored against the SAME physical target. That
removes one of the two confounds the earlier passes could not separate; the
other, forward discretization convergence, is still present and is measured by
the uniform arm's own move between grids.

GATE B, SUB-FILTER-RADIUS PERTURBATION. The solved map is blurred by a
part-masked normalized-convolution Gaussian at radii BELOW the filter radius
and re-scored. If the design really carries no structure below the radius, the
objective must be insensitive. Threshold: less than 10 percent change in J at
every radius strictly below the filter radius. The radius AT the filter length
is also reported, as context and not as part of the gate.

Run:
  ./.venv312/bin/python -m adjoint2d.topopt_robust <shape> <outdir>
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from . import chi_area, library_solve as lib
from . import printability as pq
from . import robust as rb
from . import topopt
from . import topopt_solve as tos
from .pins import build_case, load_cfg

OUT_TOPOPT = Path(__file__).resolve().parents[1] / "out_topopt"
OUT_ROBUST = Path(__file__).resolve().parents[1] / "out_robust"
OUT_MS = Path(__file__).resolve().parents[1] / "out_ms"
GRID_HOLDOUT = 160
P_TARGET_W_PER_M = 500.0
SUB_RADII_CELLS = (0.5, 1.0, 1.5)     # all strictly below 1.983 cells at grid 120
CONTEXT_RADII_CELLS = (2.0,)          # at the filter radius, reported not gated
GATE_B_TOL = 0.10


def _log(shape, name, m):
    print(f"[{shape}] {name:26s} J {m['J']:9.2f}  IoU {m['IoU']:.4f}  "
          f"IoU_area {m['IoU_area']:.4f}  grow {m['bed_melt_pct_of_part']:6.2f}  "
          f"under {m['part_under_melt_pct']:6.2f}  stop {m['t_stop_s']:6.1f} s"
          f"{' HORIZON' if m['t_stop_at_horizon'] else ''}  "
          f"P {m['P_abs_W_per_m']:6.1f}  {m['wall_s']:.0f} s", flush=True)


def resample_design(v: np.ndarray, part_mask_hi: np.ndarray, ny: int, nx: int
                    ) -> np.ndarray:
    """Bilinear transfer of the design variable, then the box, then the mask."""
    a = np.asarray(v, dtype=float)
    zy = float(ny) / a.shape[0]
    zx = float(nx) / a.shape[1]
    if abs(zy - 1.0) > rb.ZOOM_DEADBAND or abs(zx - 1.0) > rb.ZOOM_DEADBAND:
        a = zoom(a, (zy, zx), order=1)
    return np.where(part_mask_hi, np.clip(a, topopt.BOX[0], topopt.BOX[1]), 1.0)


def main(shape: str, outdir: str, arm: str = "") -> dict:
    """`arm='control_filteronly'` gates the beta = 0 control instead.

    Running BOTH arms through the same two gates is what attributes the result:
    the control changes the radius and the target but not the projection, so if
    it passes as well, the projection is not what bought the transfer.
    """
    if arm not in ("", "control_filteronly"):
        raise ValueError(f"unknown arm {arm!r}")
    stem = f"{shape}_{arm}" if arm else shape
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    src = json.loads((OUT_TOPOPT / f"{stem}.json").read_text())
    with np.load(OUT_TOPOPT / f"{stem}_maps.npz") as d:
        s_cont_120 = np.asarray(d["TO_cont"], dtype=float)
        v_120 = np.asarray(d["TO_v"], dtype=float)

    cfg_path = lib.shape_config(shape)
    cfg = load_cfg(cfg_path)
    case = build_case(cfg)
    pm = case.part_mask
    chi120, _ = chi_area.chi_from_cfg(cfg, case.x, case.y)
    beta_final = float(src["arms"]["TO_cont"]["beta"])

    res: dict = {"shape": shape, "arm": arm or "topopt_continuation",
                 "config": str(cfg_path),
                 "source": str(OUT_TOPOPT / f"{stem}.json"),
                 "filter_radius_m": topopt.FILTER_RADIUS_M,
                 "sigma_cells_at_120": src["sigma_cells"],
                 "beta_final": beta_final,
                 "baseline_120": {k: src["arms"][k] for k in
                                  ("U_uniform", "TO_cont", "TO_4bpp")},
                 "gate_B_sub_radius": {}, "gate_A_grid": {}}

    # ---------------- Gate B: sub-filter-radius perturbation ----------------
    base = src["arms"]["TO_4bpp"]
    for r in tuple(SUB_RADII_CELLS) + tuple(CONTEXT_RADII_CELLS):
        sm = rb.smooth_in_part(s_cont_120, pm, float(r))
        sq = pq.quantize_in_part(sm, pm, bpp=4, sat_max=1.0)
        m = tos.score(case, sq, chi120)
        m["arm"] = f"TO_4bpp_blur{r:.1f}"
        m["gaussian_sigma_cells"] = float(r)
        m["gaussian_sigma_mm"] = float(r) * case.dx * 1e3
        m["below_filter_radius"] = bool(float(r) < float(src["sigma_cells"]))
        m["rms_change_vs_r0"] = float(np.sqrt(np.mean((sm[pm] - s_cont_120[pm]) ** 2)))
        m["dJ_rel"] = (m["J"] - base["J"]) / max(abs(base["J"]), 1e-30)
        m["dIoU"] = m["IoU"] - base["IoU"]
        res["gate_B_sub_radius"][f"r{r:.1f}"] = m
        _log(shape, f"blur r={r:.1f} cells", m)
    sub = [m for m in res["gate_B_sub_radius"].values() if m["below_filter_radius"]]
    res["gate_B_verdict"] = {
        "tolerance": GATE_B_TOL,
        "radii_cells_gated": [m["gaussian_sigma_cells"] for m in sub],
        "max_abs_dJ_rel": max(abs(m["dJ_rel"]) for m in sub),
        "PASS": bool(all(abs(m["dJ_rel"]) < GATE_B_TOL for m in sub))}
    print(f"[{shape}] GATE B sub-filter-radius: max |dJ| "
          f"{res['gate_B_verdict']['max_abs_dJ_rel']*100:.2f} percent, PASS = "
          f"{res['gate_B_verdict']['PASS']}", flush=True)

    # ------- Gate B2: the same perturbation applied in DESIGN space ---------
    # MEASURED reason this second form exists. Gate B blurs the DELIVERED map,
    # which after a beta-16 projection is nearly binary, so part of what it
    # measures is how far a blur moves the 0.5 level set of a crisp map rather
    # than whether the design carries sub-radius structure. Blurring the DESIGN
    # variable below the filter radius and re-applying the same filter and
    # projection tests the intended claim directly: a perturbation the filter
    # was going to remove anyway must not change the answer.
    for r in SUB_RADII_CELLS:
        vb = rb.smooth_in_part(v_120, pm, float(r), outside=1.0)
        sb = topopt.design_to_map(vb, pm, dx=case.dx,
                                  radius_m=topopt.FILTER_RADIUS_M, beta=beta_final)
        qb = pq.quantize_in_part(sb, pm, bpp=4, sat_max=1.0)
        m = tos.score(case, qb, chi120)
        m["arm"] = f"TO_4bpp_designblur{r:.1f}"
        m["gaussian_sigma_cells"] = float(r)
        m["gaussian_sigma_mm"] = float(r) * case.dx * 1e3
        m["dJ_rel"] = (m["J"] - base["J"]) / max(abs(base["J"]), 1e-30)
        m["dIoU"] = m["IoU"] - base["IoU"]
        res.setdefault("gate_B2_design_blur", {})[f"r{r:.1f}"] = m
        _log(shape, f"design blur r={r:.1f} cells", m)
    b2 = res["gate_B2_design_blur"].values()
    res["gate_B2_verdict"] = {
        "tolerance": GATE_B_TOL,
        "max_abs_dJ_rel": max(abs(m["dJ_rel"]) for m in b2),
        "PASS": bool(all(abs(m["dJ_rel"]) < GATE_B_TOL for m in b2))}
    print(f"[{shape}] GATE B2 design-space sub-radius: max |dJ| "
          f"{res['gate_B2_verdict']['max_abs_dJ_rel']*100:.2f} percent, PASS = "
          f"{res['gate_B2_verdict']['PASS']}", flush=True)

    # ---------------- Gate A: grid hold-out ---------------------------------
    cfg160 = json.loads(json.dumps(cfg))
    cfg160["geometry"]["grid_nx"] = GRID_HOLDOUT
    cfg160["geometry"]["grid_ny"] = GRID_HOLDOUT
    case160 = build_case(cfg160)
    pm160 = case160.part_mask
    chi160, chi160_info = chi_area.chi_from_cfg(cfg160, case160.x, case160.y)
    res["chi_160"] = chi160_info
    res["chi_area_grid_consistency"] = {
        "area_120_m2": float(np.sum(chi120)) * case.dA,
        "area_160_m2": float(np.sum(chi160)) * case160.dA,
        "raster_area_120_m2": float(pm.sum()) * case.dA,
        "raster_area_160_m2": float(pm160.sum()) * case160.dA}
    c = res["chi_area_grid_consistency"]
    c["chi_rel_move"] = (c["area_160_m2"] - c["area_120_m2"]) / c["area_120_m2"]
    c["raster_rel_move"] = ((c["raster_area_160_m2"] - c["raster_area_120_m2"])
                            / c["raster_area_120_m2"])
    print(f"[{shape}] target area across grids: chi moves "
          f"{c['chi_rel_move']*100:+.3f} percent, the binary raster moves "
          f"{c['raster_rel_move']*100:+.3f} percent", flush=True)

    # uniform at 160, pinned voltage; this both calibrates and is the reference
    m_u160 = tos.score(case160, np.ones(pm160.shape), chi160)
    m_u160["arm"] = "U_uniform_160"
    res["gate_A_grid"]["U_uniform_160"] = m_u160
    _log(shape, "U_uniform_160", m_u160)

    v_recal = rb.recalibrated_voltage(float(cfg["electric"]["voltage_v"]),
                                      float(m_u160["P_abs_W_per_m"]),
                                      P_TARGET_W_PER_M)
    cfg_re = json.loads(json.dumps(cfg160))
    cfg_re["electric"]["voltage_v"] = v_recal
    case_re = build_case(cfg_re)
    res["recal"] = {"voltage_v_pinned": float(cfg["electric"]["voltage_v"]),
                    "voltage_v_recalibrated": v_recal,
                    "P_uniform_160_at_pinned_W_per_m": m_u160["P_abs_W_per_m"],
                    "P_target_W_per_m": P_TARGET_W_PER_M}
    prev = OUT_ROBUST / f"{shape}_grid.json"
    if prev.exists():
        pj = json.loads(prev.read_text())
        res["recal"]["voltage_v_recalibrated_earlier_pass"] = float(
            pj["recal"]["voltage_v_recalibrated"])
        res["reference_160_earlier_pass"] = {
            k: pj["arms"][k] for k in pj.get("arms", {})}
        res["reference_160_earlier_pass"]["_recal_arms"] = pj["recal"].get("arms")
        res["reference_160_earlier_pass"]["_source"] = str(prev)
    ms_rb = OUT_MS / f"{shape}_robust.json"
    if ms_rb.exists():
        mj = json.loads(ms_rb.read_text())
        res["reference_160_multistart"] = {"grid": mj.get("grid"),
                                           "_source": str(ms_rb)}

    m_u160r = tos.score(case_re, np.ones(pm160.shape), chi160)
    m_u160r["arm"] = "U_uniform_160_recal"
    m_u160r["voltage_v"] = v_recal
    res["gate_A_grid"]["U_uniform_160_recal"] = m_u160r
    _log(shape, "U_uniform_160_recal", m_u160r)

    # transfer 1: the map, production convention
    s_map = np.where(pm160, rb.resample_map(s_cont_120, GRID_HOLDOUT, GRID_HOLDOUT), 1.0)
    q_map = pq.quantize_in_part(s_map, pm160, bpp=4, sat_max=1.0)
    # transfer 2: the design, filter and projection re-applied at 160
    v_hi = resample_design(v_120, pm160, GRID_HOLDOUT, GRID_HOLDOUT)
    s_des = topopt.design_to_map(v_hi, pm160, dx=case160.dx,
                                 radius_m=topopt.FILTER_RADIUS_M, beta=beta_final)
    q_des = pq.quantize_in_part(s_des, pm160, bpp=4, sat_max=1.0)
    res["transfer_maps_differ"] = {
        "rms_map_minus_design": float(np.sqrt(np.mean((s_map[pm160] - s_des[pm160]) ** 2))),
        "max_abs": float(np.max(np.abs(s_map[pm160] - s_des[pm160])))}

    for tag, smap, kase, volts in (
            ("TO_4bpp_160_maptransfer", q_map, case160, None),
            ("TO_4bpp_160_maptransfer_recal", q_map, case_re, v_recal),
            ("TO_4bpp_160_designtransfer", q_des, case160, None),
            ("TO_4bpp_160_designtransfer_recal", q_des, case_re, v_recal)):
        m = tos.score(kase, smap, chi160)
        m["arm"] = tag
        if volts is not None:
            m["voltage_v"] = volts
        res["gate_A_grid"][tag] = m
        _log(shape, tag, m)

    b120 = src["arms"]["TO_4bpp"]
    best160 = min(("TO_4bpp_160_maptransfer_recal", "TO_4bpp_160_designtransfer_recal"),
                  key=lambda k: res["gate_A_grid"][k]["J"])
    res["gate_A_verdict"] = {
        "IoU_at_120": b120["IoU"], "IoU_area_at_120": b120["IoU_area"],
        "IoU_at_160_maptransfer_recal":
            res["gate_A_grid"]["TO_4bpp_160_maptransfer_recal"]["IoU"],
        "IoU_at_160_designtransfer_recal":
            res["gate_A_grid"]["TO_4bpp_160_designtransfer_recal"]["IoU"],
        "best_transfer": best160,
        "IoU_drop_best": b120["IoU"] - res["gate_A_grid"][best160]["IoU"],
        "uniform_IoU_move_120_to_160": (m_u160r["IoU"]
                                        - src["arms"]["U_uniform"]["IoU"]),
        "beats_uniform_at_160_recal": bool(
            res["gate_A_grid"][best160]["J"] < m_u160r["J"]),
        "reaches_SOLVED_at_160": bool(
            res["gate_A_grid"][best160]["IoU"] >= lib.SOLVED_IOU),
        "note": "The uniform arm's own move between grids bounds how much of any "
                "drop is forward discretization rather than map transfer. The "
                "TARGET no longer moves: chi is rebuilt from the geometry at each "
                "grid."}
    print(f"[{shape}] GATE A grid hold-out: IoU {b120['IoU']:.4f} at 120 -> "
          f"{res['gate_A_grid'][best160]['IoU']:.4f} at 160 ({best160}), drop "
          f"{res['gate_A_verdict']['IoU_drop_best']:+.4f}; uniform moved "
          f"{res['gate_A_verdict']['uniform_IoU_move_120_to_160']:+.4f}", flush=True)

    res["energy_gate_violations"] = [
        a for grp in ("gate_A_grid", "gate_B_sub_radius")
        for a, m in res[grp].items() if not m["energy_gate"]["PASS"]]
    res["energy_gate_violations"] += [
        a for a, m in res.get("gate_B2_design_blur", {}).items()
        if not m["energy_gate"]["PASS"]]
    res["ACCEPTANCE_BOTH_GATES_PASS"] = bool(
        res["gate_B_verdict"]["PASS"] and res["gate_A_verdict"]["reaches_SOLVED_at_160"])
    res["wall_s"] = time.perf_counter() - t0
    np.savez_compressed(out / f"{stem}_robust_maps.npz",
                        TO_cont_120=s_cont_120, TO_v_120=v_120,
                        s_map_160=s_map, s_design_160=s_des,
                        q_map_160=q_map, q_design_160=q_des,
                        chi_160=chi160.astype(np.float32),
                        part_mask_160=pm160.astype(np.uint8))
    (out / f"{stem}_robust.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"[{shape}] robust done, wall {res['wall_s']:.0f} s, energy gate violations "
          f"{res['energy_gate_violations'] or 'none'}, BOTH GATES PASS = "
          f"{res['ACCEPTANCE_BOTH_GATES_PASS']}", flush=True)
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else str(OUT_TOPOPT),
         sys.argv[3] if len(sys.argv) > 3 else "")
