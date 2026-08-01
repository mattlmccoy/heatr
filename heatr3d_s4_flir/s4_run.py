#!/usr/bin/env python3
"""Gate S4 driver: heatr3d forward prediction vs Allison's FLIR sequences.

Protocol is pre-registered in README.md. Nothing here tunes anything except the
single scalar absorbed power (one per case, fitted to the measured early-time
mean heating rate).

Usage:
  ./.venv312/bin/python heatr3d_s4_flir/s4_run.py [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from heatr3d import Grid, Params, run  # noqa: E402
import s4_flir_lib as L  # noqa: E402

logger = logging.getLogger("s4")

# --- geometry (README section 3; inferred from Allison Square/FE.m) ---------- #
NX = NZ = 96
NY = 31
CHAMBER_L = 0.0624            # m, in-plane chamber edge (mirrored quarter model)
PART_W_CELLS = 62             # 40.3 mm at h = 0.65 mm
PART_H_CELLS = 15             # 9.75 mm
DT_S = 0.15                   # <= CFL_SAFETY * dt_stable_thermal (0.169 s at this h)
CACHE = HERE / "cache" / "qrf_square_n96.npz"

T_FIT_S = 60.0                # early-time (pre-melt) power-fit window
MATCH_FRACS = (0.60, 0.95)    # matched thermal states for pattern scoring
NG = 64                       # part-relative scoring grid
N_CHECKPOINTS = 40

CASES = [
    dict(key="A", seq="010320_exp1", cls="untuned", role="primary anchor"),
    dict(key="B", seq="010320_exp5", cls="untuned", role="independent repeat"),
    dict(key="C", seq="031920_exp2", cls="tuned", role="discrimination control"),
]


# --------------------------------------------------------------------------- #
def build_part() -> Tuple[Grid, np.ndarray]:
    g = Grid(n=NX, L=CHAMBER_L)
    part = np.zeros((NX, NY, NZ), bool)
    i0 = (NX - PART_W_CELLS) // 2
    part[i0:i0 + PART_W_CELLS, 0:PART_H_CELLS, i0:i0 + PART_W_CELLS] = True
    return g, part


def load_qrf() -> Tuple[np.ndarray, np.ndarray, float]:
    """The single EQS solve behind every case (~26 s at the working grid).

    Q_rf enters the march only through `qrf_override`, and the per-case power fit
    is an exact linear rescale of this field (compute_qrf_3d renormalizes to a
    total-power target), so one solve serves all cases and both fit variants.
    """
    if not CACHE.exists():
        from heatr3d import build_gamma, compute_qrf_3d, solve_eqs_3d
        grid, part = build_part()
        p0 = Params()
        logger.info("no EQS cache; solving %s (this is the only EQS solve)", part.shape)
        gamma = build_gamma(part, p0)
        V = solve_eqs_3d(gamma, grid, p0)
        Q = compute_qrf_3d(V, gamma, grid, p0, part)      # qrf_gradient="masked"
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, Q=Q, part=part, h=grid.h, L=grid.L)
    d = np.load(CACHE)
    return d["Q"], d["part"], float(d["h"])


# --------------------------------------------------------------------------- #
# measured side
# --------------------------------------------------------------------------- #
def measured_case(seq: str, n_samples: int) -> Dict:
    out: Dict = {}
    for eps_tag, eps in (("e095", None), ("e085", 0.85)):
        d = L.read_seq(seq, n_samples=n_samples, emissivity=eps)
        t = d["t_s"]
        frames = d["frames"]
        amb = float(np.nanmean(frames[0]))
        gmax = np.array([float(np.nanmax(f)) for f in frames])
        rise = gmax - amb
        total = float(rise.max())
        ref_i = int(np.argmax(rise >= 0.6 * total))
        roi = L.largest_component_roi(frames[ref_i], amb, frac=0.5)
        rect = L.min_area_rect(roi)
        roi_mean = np.array([float(np.nanmean(f[roi])) for f in frames])
        roi_max = np.array([float(np.nanmax(f[roi])) for f in frames])
        rec = dict(
            seq=seq, path=d["path"], n_frames=int(d["n_frames"]),
            n_sampled=int(len(t)), duration_s=float(t[-1]),
            emissivity_used=(d["emissivity"] if eps is None else eps),
            emissivity_recorded=d["emissivity"], distance_m=d["distance_m"],
            t_refl_c=d["t_refl_c"], ambient_c=amb,
            optical_mm_per_px=L.optical_mm_per_px(d["distance_m"]),
            ref_frame_time_s=float(t[ref_i]), roi_px=int(roi.sum()),
            rect=dataclasses.asdict(rect),
            rect_side_mm_optical=[rect.side_u * L.optical_mm_per_px(d["distance_m"]),
                                  rect.side_v * L.optical_mm_per_px(d["distance_m"])],
            fitted_mm_per_px=L.PART_MM / (0.5 * (rect.side_u + rect.side_v)),
            global_tmax_peak_c=float(gmax.max()),
            global_tmax_end_c=float(gmax[-1]),
            t185_global_s=L.time_at_value(t, gmax, 185.0),
            roi_mean_end_c=float(roi_mean[-1]), roi_mean_peak_c=float(roi_mean.max()),
        )
        theta = L.normalized_rise_curve(t, roi_mean, amb)
        rec["t_s"] = t.tolist()
        rec["roi_mean_c"] = roi_mean.tolist()
        rec["roi_max_c"] = roi_max.tolist()
        rec["theta"] = theta.tolist()
        rec["rise_at_fit_c"] = float(np.interp(T_FIT_S, t, roi_mean) - amb)
        # registered fields at the matched thermal states
        fields = {}
        for fr in MATCH_FRACS:
            tt = L.time_at_fraction(t, theta, fr)
            j = int(np.argmin(np.abs(t - tt)))
            fields[f"{fr:.2f}"] = dict(
                t_s=float(t[j]),
                field=L.resample_unit_square(frames[j], rect, ng=NG).tolist(),
            )
        rec["matched"] = fields
        out[eps_tag] = rec
    return out


# --------------------------------------------------------------------------- #
# prediction side
# --------------------------------------------------------------------------- #
def march(grid: Grid, part: np.ndarray, p: Params, qrf: np.ndarray,
          duration_s: float, n_ckpt: int) -> Dict:
    """Chained segments (verified bit-identical to one long march by
    test_chained_T0_override_segments_equal_one_long_march)."""
    ny = part.shape[1]
    y_free = int(np.nonzero(part.any(axis=(0, 2)))[0].max())     # part's free face
    seg = duration_s / n_ckpt
    T0 = None
    t_list, planeP, planeS = [], [], []
    part_mean, part_max, p_mean, p_max = [], [], [], []
    e_in = e_st = e_loss = 0.0
    clamp = False
    cfl = False
    nsub = 1
    for k in range(n_ckpt):
        res = run(grid, part, p, max_time_s=seg, phi_target=1.5,
                  qrf_override=qrf, T0_override=T0)
        T0 = res.T_final
        e_in += res.energy_in_j
        e_st += res.energy_stored_j
        e_loss += res.energy_loss_j
        clamp |= bool(res.clamp_bound)
        cfl |= bool(res.cfl_violated)
        nsub = max(nsub, int(res.n_substeps_used))
        T = res.T_final
        t_list.append((k + 1) * seg)
        planeP.append(T[:, y_free, :].copy())
        planeS.append(T[:, ny - 1, :].copy())
        part_mean.append(float(T[part].mean()))
        part_max.append(float(T[part].max()))
        fp = T[:, y_free, :]
        pm = part[:, y_free, :]
        p_mean.append(float(fp[pm].mean()))
        p_max.append(float(fp[pm].max()))
    resid = (e_in - e_st - e_loss) / max(e_in, 1e-30)
    return dict(t_s=np.array(t_list), planeP=np.array(planeP), planeS=np.array(planeS),
                part_mean=np.array(part_mean), part_max=np.array(part_max),
                face_mean=np.array(p_mean), face_max=np.array(p_max),
                energy_in_j=e_in, energy_stored_j=e_st, energy_loss_j=e_loss,
                energy_residual_frac=resid, clamp_bound=clamp, cfl_violated=cfl,
                n_substeps_used=nsub, y_free=y_free)


def fit_power(grid: Grid, part: np.ndarray, p: Params, qrf_unit: np.ndarray,
              target_rise_c: float, t_fit_s: float) -> Tuple[float, float, List]:
    """Fit ONE scalar: the multiplier on the absorbed power such that the
    predicted part free-face mean RISE at t_fit_s equals the measured value.
    Secant on an almost exactly linear relation (pre-melt)."""
    y_free = int(np.nonzero(part.any(axis=(0, 2)))[0].max())
    hist = []

    def rise_at(scale: float) -> float:
        res = run(grid, part, p, max_time_s=t_fit_s, phi_target=1.5,
                  qrf_override=qrf_unit * scale)
        fp = res.T_final[:, y_free, :]
        pm = part[:, y_free, :]
        r = float(fp[pm].mean()) - p.preheat_c
        hist.append(dict(scale=scale, rise_c=r))
        return r

    s0 = 1.0
    r0 = rise_at(s0)
    s1 = s0 * target_rise_c / max(r0, 1e-9)
    r1 = rise_at(s1)
    for _ in range(3):
        if abs(r1 - target_rise_c) / max(target_rise_c, 1e-9) < 0.005:
            break
        denom = (r1 - r0)
        if abs(denom) < 1e-12:
            break
        s2 = s1 + (target_rise_c - r1) * (s1 - s0) / denom
        s0, r0, s1 = s1, r1, max(s2, 1e-6)
        r1 = rise_at(s1)
    return s1, r1, hist


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def score_pair(meas_field: np.ndarray, pred_field: np.ndarray,
               amb_m: float, amb_p: float) -> Dict:
    a = L.normalize_rise(meas_field, amb_m)
    b = L.normalize_rise(pred_field, amb_p)
    return dict(
        r=L.pattern_correlation(a, b),
        chamfer_mm=L.chamfer_mm(a, b),
        argmax_offset_mm=L.argmax_offset_mm(a, b),
        cX_measured=L.corner_edge_contrast(meas_field),
        cX_predicted=L.corner_edge_contrast(pred_field),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=180)
    ap.add_argument("--ckpt", type=int, default=N_CHECKPOINTS)
    ap.add_argument("--out", type=Path, default=HERE / "results.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    grid, part = build_part()
    qrf, part_c, h = load_qrf()
    assert np.array_equal(part, part_c) and abs(h - grid.h) < 1e-12
    p_ref = Params()
    p_unit_w = float(qrf.sum() * grid.dV)
    logger.info("grid h=%.4f mm  part %s  cached EQS power %.3f W",
                grid.h * 1e3, part.sum(), p_unit_w)

    out: Dict = dict(
        meta=dict(
            grid=dict(n_xz=NX, n_y=NY, h_m=grid.h, chamber_m=[CHAMBER_L, NY * grid.h, CHAMBER_L],
                      part_cells=[PART_W_CELLS, PART_H_CELLS, PART_W_CELLS],
                      part_mm=[PART_W_CELLS * grid.h * 1e3, PART_H_CELLS * grid.h * 1e3,
                               PART_W_CELLS * grid.h * 1e3]),
            qrf_gradient="masked", phase_update="enthalpy",
            eqs_reference_power_w=p_unit_w,
            span=L.SPAN, ng=NG, match_fracs=list(MATCH_FRACS), t_fit_s=T_FIT_S,
        ),
        cases={}, gates={},
    )

    measured = {}
    for c in CASES:
        t0 = time.time()
        measured[c["key"]] = measured_case(c["seq"], args.samples)
        logger.info("measured %s (%s) decoded in %.1f s", c["key"], c["seq"], time.time() - t0)

    preds = {}
    for c in CASES:
        m = measured[c["key"]]["e095"]
        amb = m["ambient_c"]
        p = dataclasses.replace(
            p_ref, phase_update="enthalpy", t_pc_c=185.0, dt_s=DT_S,
            ambient_c=amb, preheat_c=amb,
        )
        scale, r_fit, hist = fit_power(grid, part, p, qrf, m["rise_at_fit_c"], T_FIT_S)
        logger.info("case %s power scale %.4g -> %.3f W absorbed (target rise %.2f C, got %.2f C)",
                    c["key"], scale, p_unit_w * scale, m["rise_at_fit_c"], r_fit)
        pr = march(grid, part, p, qrf * scale, m["duration_s"], args.ckpt)
        pr["power_w"] = p_unit_w * scale
        pr["power_scale"] = scale
        pr["fit_hist"] = hist
        pr["ambient_c"] = amb
        preds[c["key"]] = pr
        logger.info("case %s march done: face max %.1f C, part max %.1f C, resid %.2e, "
                    "clamp %s, cfl %s", c["key"], pr["face_max"][-1], pr["part_max"][-1],
                    pr["energy_residual_frac"], pr["clamp_bound"], pr["cfl_violated"])

    # ---- registration + scoring -------------------------------------------- #
    pred_fields = {}
    for c in CASES:
        k = c["key"]
        pr = preds[k]
        amb = pr["ambient_c"]
        theta_p = L.normalized_rise_curve(pr["t_s"], pr["face_mean"], amb)
        entry = dict(theta=theta_p.tolist(), t_s=pr["t_s"].tolist(),
                     face_mean=pr["face_mean"].tolist(), face_max=pr["face_max"].tolist(),
                     part_mean=pr["part_mean"].tolist(), part_max=pr["part_max"].tolist())
        for plane in ("planeP", "planeS"):
            imgs = pr[plane]
            # reference frame + rect by the SAME operator as the measured side
            gmax = imgs.max(axis=(1, 2))
            rise = gmax - amb
            ref_i = int(np.argmax(rise >= 0.6 * rise.max()))
            roi = L.largest_component_roi(imgs[ref_i], amb, frac=0.5)
            rect = L.min_area_rect(roi)
            fields = {}
            for fr in MATCH_FRACS:
                tt = L.time_at_fraction(pr["t_s"], theta_p, fr)
                j = int(np.argmin(np.abs(pr["t_s"] - tt)))
                fields[f"{fr:.2f}"] = dict(
                    t_s=float(pr["t_s"][j]),
                    field=L.resample_unit_square(imgs[j], rect, ng=NG),
                )
            entry[plane] = dict(rect=dataclasses.asdict(rect), ref_t_s=float(pr["t_s"][ref_i]),
                                roi_px=int(roi.sum()),
                                matched={k2: dict(t_s=v["t_s"]) for k2, v in fields.items()})
            pred_fields[(k, plane)] = fields
        out["cases"][k] = dict(case=c, measured={
            tag: {kk: vv for kk, vv in measured[k][tag].items() if kk != "matched"}
            for tag in ("e095", "e085")}, predicted=entry)

    for c in CASES:
        k = c["key"]
        amb_m = measured[k]["e095"]["ambient_c"]
        amb_p = preds[k]["ambient_c"]
        scores = {}
        for plane in ("planeP", "planeS"):
            for fr in MATCH_FRACS:
                key = f"{fr:.2f}"
                mf = np.array(measured[k]["e095"]["matched"][key]["field"])
                pf = pred_fields[(k, plane)][key]["field"]
                scores[f"{plane}_{key}"] = dict(
                    **score_pair(mf, pf, amb_m, amb_p),
                    t_meas_s=measured[k]["e095"]["matched"][key]["t_s"],
                    t_pred_s=pred_fields[(k, plane)][key]["t_s"],
                )
        # heating-curve shape (M6) and timing (M7)
        tm = np.array(measured[k]["e095"]["t_s"])
        th_m = np.array(measured[k]["e095"]["theta"])
        th_p = np.interp(tm, preds[k]["t_s"], L.normalized_rise_curve(
            preds[k]["t_s"], preds[k]["face_mean"], amb_p))
        scores["curve_rms"] = float(np.sqrt(np.mean((th_m - th_p) ** 2)))
        scores["t50_meas_s"] = L.time_at_fraction(tm, th_m, 0.5)
        scores["t50_pred_s"] = L.time_at_fraction(preds[k]["t_s"], L.normalized_rise_curve(
            preds[k]["t_s"], preds[k]["face_mean"], amb_p), 0.5)
        scores["t185_meas_roi_max_s"] = L.time_at_value(
            tm, np.array(measured[k]["e095"]["roi_max_c"]), 185.0)
        scores["t185_pred_face_max_s"] = L.time_at_value(
            preds[k]["t_s"], preds[k]["face_max"], 185.0)
        scores["t185_pred_part_max_s"] = L.time_at_value(
            preds[k]["t_s"], preds[k]["part_max"], 185.0)
        out["cases"][k]["scores"] = scores
        out["gates"][k] = dict(
            energy_residual_frac=preds[k]["energy_residual_frac"],
            energy_gate_pass=bool(abs(preds[k]["energy_residual_frac"]) <= 1e-2),
            clamp_bound=preds[k]["clamp_bound"], cfl_violated=preds[k]["cfl_violated"],
            n_substeps_used=preds[k]["n_substeps_used"],
            power_w=preds[k]["power_w"],
        )

    # discrimination margin M2 (planeP, 0.95 state)
    for k in ("A", "B"):
        r_un = out["cases"][k]["scores"]["planeP_0.95"]["r"]
        r_tuned_pred = None
        # score the SAME prediction (case k's field) against case C's measured field
        mf_c = np.array(measured["C"]["e095"]["matched"]["0.95"]["field"])
        pf = pred_fields[(k, "planeP")]["0.95"]["field"]
        r_tuned_pred = L.pattern_correlation(
            L.normalize_rise(mf_c, measured["C"]["e095"]["ambient_c"]),
            L.normalize_rise(pf, preds[k]["ambient_c"]))
        out["cases"][k]["scores"]["M2_r_vs_tuned"] = r_tuned_pred
        out["cases"][k]["scores"]["M2_margin"] = r_un - r_tuned_pred

    # save registered fields for the figures
    np.savez_compressed(
        HERE / "fields.npz",
        **{f"meas_{k}_{fr:.2f}": np.array(measured[k]["e095"]["matched"][f"{fr:.2f}"]["field"])
           for k in ("A", "B", "C") for fr in MATCH_FRACS},
        **{f"pred_{k}_{plane}_{fr:.2f}": pred_fields[(k, plane)][f"{fr:.2f}"]["field"]
           for k in ("A", "B", "C") for plane in ("planeP", "planeS") for fr in MATCH_FRACS},
    )
    args.out.write_text(json.dumps(out, indent=1, default=float))
    logger.info("wrote %s", args.out)


if __name__ == "__main__":
    main()
