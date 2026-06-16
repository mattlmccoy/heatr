#!/usr/bin/env python3
"""
run_perlayer_fgm_pilot.py
=========================
PILOT for dissertation editing note 31: PER-LAYER FGM mask for the turntable
workflow.

WHY
---
The dissertation's existing turntable+FGM result reused a SINGLE static FGM
dopant mask across the whole rotating build. That is physically inconsistent
with a turntable: as the part rotates, each rotation state sees a DIFFERENT
RF power-deposition field Qrf (the electrodes are fixed; rotating the part by
90 deg swaps which faces couple). A single mask, computed for one orientation,
corrects hot spots that rotation has already averaged away -> it can be
counterproductive (the "turntable + static FGM" bar in fig:turntable_diamond
is WORSE than turntable-alone).

A PHYSICALLY CORRECT turntable FGM must account for every orientation the part
visits. This pilot implements and compares two corrected approaches against the
flawed single-mask one, plus the no-FGM references.

WHAT IS COMPARED (all on the rotation-AVERAGED Qrf the part carries)
--------------------------------------------------------------------
  A. baseline           : single orientation, no FGM, no rotation.
  B. turntable, no FGM  : N orientations, average Qrf in the PART frame.
  C. single-mask TT+FGM : ONE FGM mask (from orientation-0 field) printed on the
                          part, rotating WITH it (the corrected-but-still-single
                          version of the dissertation approach).
  D. per-layer-mask TT  : a DISTINCT FGM mask per rotation state, each computed
                          to flatten the field THAT state sees, combined in the
                          part frame into one printed dopant pattern.

UNIFORMITY METRIC
-----------------
sigma_Q = std of the rotation-averaged Qrf over part-interior cells, reported
both absolute (W/m^3) and as a coefficient of variation CoV = sigma/mean (%).
Qrf is the FGM grading proxy used in the dissertation (T_phi90 tracks Qrf;
see fgm_generator._PROXY_HIGH_MEANS_HOT). This is an EQS-field uniformity pilot;
it does NOT run the full transient thermal march (that is the heavy server
path) -- the relative ranking of the four approaches is what note 31 needs.

PHYSICS REUSE (faithful, not re-derived)
----------------------------------------
make_domain, _build_rotated_part_mask, solve_electric_state,
enforce_generator_power, and the FGM grading rule (normalize->invert->magnitude)
are taken DIRECTLY from rfam_eqs_coupled.py / fgm_generator.py. No new
gradient/adjoint code is introduced, so no finite-difference gate is required;
the FGM map is a closed-form algebraic transform of Qrf.

Run under the project venv:
  .../analysis-3dfgm/.venv312/bin/python run_perlayer_fgm_pilot.py --shape diamond
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates, rotate as ndrotate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rfam_eqs_coupled as R  # noqa: E402


# ---------------------------------------------------------------------------
# FGM grading rule -- copied semantics from fgm_generator.generate_fgm
# (normalize on part-interior percentiles -> invert -> magnitude-scale).
# Returns a sat_map in [0,1] on the simulation grid, 0 outside the part.
# ---------------------------------------------------------------------------
def fgm_sat_from_proxy(
    proxy: np.ndarray,
    part_mask: np.ndarray,
    *,
    invert: bool = True,
    magnitude: float = 1.0,
    baseline: float = 0.5,
    smoothing_sigma: float = 1.5,
    clip_pct: tuple[float, float] = (2.0, 98.0),
) -> np.ndarray:
    raw = gaussian_filter(proxy.astype(np.float64), sigma=smoothing_sigma)
    inside = raw[part_mask]
    lo = float(np.percentile(inside, clip_pct[0]))
    hi = float(np.percentile(inside, clip_pct[1]))
    span = max(hi - lo, 1e-12)
    norm = np.clip((raw - lo) / span, 0.0, 1.0)
    sat_raw = (1.0 - norm) if invert else norm
    sat = baseline + float(magnitude) * (sat_raw - baseline)
    sat = np.clip(sat, 0.0, 1.0).astype(np.float64)
    sat[~part_mask] = 0.0
    return sat


# ---------------------------------------------------------------------------
# Rotate a part-frame field by `deg` about the grid centre (lab<->part frame).
# Used to carry a printed dopant pattern WITH the part as it turns, and to pull
# each orientation's lab-frame Qrf back into the common part frame.
# ---------------------------------------------------------------------------
def rotate_field(field: np.ndarray, deg: float) -> np.ndarray:
    if abs(deg % 360.0) < 1e-9:
        return field.astype(np.float64).copy()
    # 90-deg multiples: use exact array rotation (no interpolation blur).
    n90 = deg / 90.0
    if abs(n90 - round(n90)) < 1e-9:
        return np.rot90(field, k=int(round(n90)) % 4).astype(np.float64).copy()
    out = ndrotate(field.astype(np.float64), angle=deg, reshape=False, order=1, mode="constant", cval=0.0)
    return out


def thermal_proxy(qrf: np.ndarray, part_mask: np.ndarray, diff_sigma: float) -> np.ndarray:
    """Cheap stand-in for the steady T field: diffuse Qrf with a Gaussian.

    The dissertation's sigma_T is measured on T_phi90, which is the heat-diffused
    image of Qrf -- far smoother than raw Qrf (thermal conduction averages out the
    electrode-contact corner singularities).  A Gaussian blur of width diff_sigma
    (in cells) is the linear-diffusion Green's-function proxy.  Reported alongside
    the raw-Qrf metric so the singularity-dominated vs thermal-relevant uniformity
    are both visible.  diff_sigma=0 -> raw Qrf.
    """
    if diff_sigma <= 0:
        return qrf.astype(np.float64)
    # mask-normalized blur so powder (Qrf=0) does not bleed the part field down.
    m = part_mask.astype(np.float64)
    num = gaussian_filter(np.where(part_mask, qrf, 0.0).astype(np.float64), diff_sigma)
    den = gaussian_filter(m, diff_sigma)
    out = np.zeros_like(num)
    good = den > 1e-9
    out[good] = num[good] / den[good]
    return out


def sigma_q_metrics(qrf_avg: np.ndarray, part_mask: np.ndarray,
                    diff_sigma: float = 0.0) -> dict:
    vals = qrf_avg[part_mask]
    mean = float(vals.mean())
    std = float(vals.std())
    cov = 100.0 * std / max(mean, 1e-30)
    # thermal-proxy (diffused) uniformity -- the dissertation-relevant one
    Tp = thermal_proxy(qrf_avg, part_mask, diff_sigma)
    tvals = Tp[part_mask]
    tmean = float(tvals.mean())
    tstd = float(tvals.std())
    tcov = 100.0 * tstd / max(tmean, 1e-30)
    return {"sigma_Q_w_per_m3": std, "mean_Q_w_per_m3": mean, "CoV_pct": cov,
            "sigma_Tproxy": tstd, "mean_Tproxy": tmean, "CoV_Tproxy_pct": tcov}


# ---------------------------------------------------------------------------
# Core: solve the EQS field for the part at a given cumulative rotation, with an
# OPTIONAL printed dopant (sat) pattern carried in the PART frame.
# Returns Qrf in the LAB frame (electrodes fixed) and the part_mask at that
# rotation.  sat_part is expressed in the part frame at rotation-0; we rotate it
# into the lab frame to multiply the local sigma.
# ---------------------------------------------------------------------------
def solve_orientation(
    cfg: dict,
    x: np.ndarray,
    y: np.ndarray,
    cum_deg: float,
    *,
    sigma_d0: float,
    sigma_v: float,
    eps_d: float,
    eps_v: float,
    omega: float,
    elec_hi: np.ndarray,
    elec_lo: np.ndarray,
    v_hi: float,
    v_lo: float,
    dx: float,
    dy: float,
    dA: float,
    elec_cfg: dict,
    power_factor: float,
    max_qrf: float,
    target_power: float | None,
    sat_part0: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    pmask, dmask, _ff, _idm = R._build_rotated_part_mask(cfg["geometry"], cum_deg, x, y)

    sigma = np.full(pmask.shape, sigma_v, dtype=np.float64)
    eps_r = np.full(pmask.shape, eps_v, dtype=np.float64)
    sigma[pmask] = sigma_d0
    eps_r[pmask] = eps_d

    if sat_part0 is not None:
        # Carry the printed pattern WITH the part: rotate the part-frame map into
        # the current lab orientation, then modulate doped-cell sigma by it.
        sat_lab = rotate_field(sat_part0, cum_deg)
        # Renormalize the multiplicative dopant scaling to be volume-neutral so
        # the comparison isolates PATTERN, not total dopant mass (matches the
        # mean-0.5 baseline of the FGM rule).
        scale = np.ones_like(sigma)
        m = pmask & (sat_lab > 1e-6)
        if np.any(m):
            sat_in = sat_lab[m]
            # scale relative to baseline 0.5 -> factor in [0,2]; clip like engine
            scale[m] = np.clip(sat_in / 0.5, 1e-3, 5.0)
        sigma[pmask] = sigma_d0 * scale[pmask]

    _g, _V, _Ex, _Ey, _Em, Qrf = R.solve_electric_state(
        sigma, eps_r, omega, elec_hi, elec_lo, v_hi, v_lo,
        dx, dy, elec_cfg, x, y, power_factor, max_qrf,
    )
    Qrf, _p, _s = R.enforce_generator_power(Qrf, dmask, dA, target_power, max_qrf)
    return Qrf, pmask


def run_pilot(shape: str, angles: list[float], magnitude: float, out_json: Path,
             diff_sigma: float = 3.0) -> dict:
    # Build a minimal config from the diamond template but override the shape.
    cfg = R.load_config(HERE / "configs" / "shape_diamond_6min.yaml")
    cfg["geometry"]["part"]["shape"] = shape

    x, y, _poly, p0_mask, d0_mask, hi, lo, _ff, _idm, _polys = R.make_domain(cfg)
    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dA = dx * dy

    elec_cfg = cfg["electric"]
    omega = 2.0 * np.pi * float(elec_cfg["frequency_hz"])
    power_factor = float(elec_cfg.get("power_factor", 1.0))
    max_qrf = float(elec_cfg.get("max_qrf_w_per_m3", 1e11))
    v_hi = float(elec_cfg.get("voltage_v", 860.0))
    v_lo = 0.0

    mat = cfg["materials"]
    sigma_d0 = float(mat["doped"]["sigma_s_per_m"])
    sigma_v = float(mat["virgin"]["sigma_s_per_m"]) + 1e-6  # ersatz nonzero for invertibility
    eps_d = float(mat["doped"]["eps_r"])
    eps_v = float(mat["virgin"]["eps_r"])

    # generator power target [W per metre of depth] (2.5D): P / depth
    gen_w = float(elec_cfg.get("generator_power_w", 500.0))
    eff = float(elec_cfg.get("generator_transfer_efficiency", 0.02))
    depth = float(elec_cfg.get("effective_depth_m", 0.02))
    target_power = gen_w * eff / depth  # W/m

    base_kw = dict(
        x=x, y=y, sigma_d0=sigma_d0, sigma_v=sigma_v, eps_d=eps_d, eps_v=eps_v,
        omega=omega, elec_hi=hi, elec_lo=lo, v_hi=v_hi, v_lo=v_lo,
        dx=dx, dy=dy, dA=dA, elec_cfg=elec_cfg, power_factor=power_factor,
        max_qrf=max_qrf, target_power=target_power,
    )

    # ── A. baseline: single orientation, no FGM ────────────────────────────────
    Qrf0, pm0 = solve_orientation(cfg, cum_deg=0.0, sat_part0=None, **base_kw)
    metr_base = sigma_q_metrics(Qrf0, pm0, diff_sigma)

    # ── Per-orientation no-FGM fields, pulled into the PART frame ───────────────
    # part frame = rotation-0 frame.  Pull each lab Qrf back by -cum_deg.
    qrf_part_noFGM = []
    pm_part = pm0  # part mask is the same in the part frame (rotation-0 raster)
    for ang in angles:
        Qrf_lab, pm_lab = solve_orientation(cfg, cum_deg=ang, sat_part0=None, **base_kw)
        q_part = rotate_field(Qrf_lab, -ang)
        qrf_part_noFGM.append(q_part)
    qrf_part_noFGM = np.array(qrf_part_noFGM)
    qrf_avg_noFGM = qrf_part_noFGM.mean(axis=0)
    metr_tt = sigma_q_metrics(qrf_avg_noFGM, pm_part, diff_sigma)

    # The FGM grades dopant against the THERMAL proxy (the dissertation grades on
    # T_phi90, the diffused field) -- so smooth each orientation's part-frame Qrf
    # before deriving the saturation map.
    Tproxy_part = np.array([thermal_proxy(q, pm_part, diff_sigma) for q in qrf_part_noFGM])

    # ── C. single-mask TT+FGM (the dissertation approach, corrected to rotate) ──
    # ONE mask from orientation-0's field, in the PART frame.  Print it; it
    # rotates with the part; average the resulting Qrf.
    sat_single = fgm_sat_from_proxy(
        Tproxy_part[0], pm_part, magnitude=magnitude,
    )
    qrf_part_single = []
    for ang in angles:
        Qrf_lab, _pm = solve_orientation(cfg, cum_deg=ang, sat_part0=sat_single, **base_kw)
        qrf_part_single.append(rotate_field(Qrf_lab, -ang))
    qrf_avg_single = np.array(qrf_part_single).mean(axis=0)
    metr_single = sigma_q_metrics(qrf_avg_single, pm_part, diff_sigma)

    # ── D. per-layer-mask TT+FGM ───────────────────────────────────────────────
    # Compute a DISTINCT mask per rotation state from THAT state's part-frame
    # field, then combine (mean) into one printed pattern.  This is the dopant
    # pattern that flattens the rotation-AVERAGED field the part carries.
    per_layer_sats = []
    for k, ang in enumerate(angles):
        sat_k = fgm_sat_from_proxy(Tproxy_part[k], pm_part, magnitude=magnitude)
        per_layer_sats.append(sat_k)
    per_layer_sats = np.array(per_layer_sats)
    sat_perlayer = per_layer_sats.mean(axis=0)
    sat_perlayer[~pm_part] = 0.0

    qrf_part_perlayer = []
    for ang in angles:
        Qrf_lab, _pm = solve_orientation(cfg, cum_deg=ang, sat_part0=sat_perlayer, **base_kw)
        qrf_part_perlayer.append(rotate_field(Qrf_lab, -ang))
    qrf_avg_perlayer = np.array(qrf_part_perlayer).mean(axis=0)
    metr_perlayer = sigma_q_metrics(qrf_avg_perlayer, pm_part, diff_sigma)

    # ── E. averaged-field mask (principled single pattern) ──────────────────────
    # Grade ONE dopant pattern against the rotation-AVERAGED no-FGM thermal proxy
    # (the field the part actually experiences over a full revolution).  This is
    # the most defensible single-pattern competitor to the per-layer-mean.
    Tproxy_avg = thermal_proxy(qrf_avg_noFGM, pm_part, diff_sigma)
    sat_avgfield = fgm_sat_from_proxy(Tproxy_avg, pm_part, magnitude=magnitude)
    qrf_part_avgfield = []
    for ang in angles:
        Qrf_lab, _pm = solve_orientation(cfg, cum_deg=ang, sat_part0=sat_avgfield, **base_kw)
        qrf_part_avgfield.append(rotate_field(Qrf_lab, -ang))
    qrf_avg_avgfield = np.array(qrf_part_avgfield).mean(axis=0)
    metr_avgfield = sigma_q_metrics(qrf_avg_avgfield, pm_part, diff_sigma)

    result = {
        "shape": shape,
        "angles_deg": angles,
        "n_orientations": len(angles),
        "magnitude": magnitude,
        "metric": "sigma of rotation-averaged Qrf over part interior (CoV %)",
        "A_baseline_no_tt_no_fgm": metr_base,
        "B_turntable_no_fgm": metr_tt,
        "C_single_mask_tt_fgm": metr_single,
        "D_perlayer_mask_tt_fgm": metr_perlayer,
        "E_avgfield_mask_tt_fgm": metr_avgfield,
    }

    np.savez_compressed(
        out_json.with_suffix(".npz"),
        qrf0=Qrf0, qrf_avg_noFGM=qrf_avg_noFGM, qrf_avg_single=qrf_avg_single,
        qrf_avg_perlayer=qrf_avg_perlayer, qrf_avg_avgfield=qrf_avg_avgfield,
        sat_single=sat_single, sat_perlayer=sat_perlayer, sat_avgfield=sat_avgfield,
        per_layer_sats=per_layer_sats, part_mask=pm_part, x=x, y=y,
    )
    out_json.write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", default="diamond")
    ap.add_argument("--angles", default="0,90,180,270",
                    help="comma-separated cumulative turntable angles (deg)")
    ap.add_argument("--magnitude", type=float, default=1.0)
    ap.add_argument("--diff-sigma", type=float, default=3.0,
                    help="thermal-proxy diffusion width in cells (0=raw Qrf)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    angles = [float(a) for a in args.angles.split(",")]
    out = Path(args.out) if args.out else HERE / f"perlayer_fgm_pilot_{args.shape}.json"

    res = run_pilot(args.shape, angles, args.magnitude, out, diff_sigma=args.diff_sigma)

    def line(tag: str, m: dict) -> str:
        return (f"  {tag:<26} CoV_Tproxy={m['CoV_Tproxy_pct']:7.2f}%   "
                f"(raw Qrf CoV={m['CoV_pct']:7.2f}%)")

    print(f"\n=== PER-LAYER FGM PILOT — shape={args.shape}  orientations={angles}  "
          f"magnitude={args.magnitude}  diff_sigma={args.diff_sigma} ===")
    print("  metric = CoV of rotation-averaged DIFFUSED Qrf (thermal proxy for sigma_T)")
    print(line("A baseline (no tt,no fgm)", res["A_baseline_no_tt_no_fgm"]))
    print(line("B turntable (no fgm)",      res["B_turntable_no_fgm"]))
    print(line("C single-mask tt+fgm",      res["C_single_mask_tt_fgm"]))
    print(line("D per-layer-mask tt+fgm",   res["D_perlayer_mask_tt_fgm"]))
    print(line("E avg-field-mask tt+fgm",   res["E_avgfield_mask_tt_fgm"]))
    cov = lambda k: res[k]["CoV_Tproxy_pct"]
    base = cov("B_turntable_no_fgm")
    print(f"\n  single-mask vs turntable-alone : {base:.2f}% -> {cov('C_single_mask_tt_fgm'):.2f}%  "
          f"({(cov('C_single_mask_tt_fgm')-base)/base*100:+.1f}%)")
    print(f"  per-layer  vs turntable-alone : {base:.2f}% -> {cov('D_perlayer_mask_tt_fgm'):.2f}%  "
          f"({(cov('D_perlayer_mask_tt_fgm')-base)/base*100:+.1f}%)")
    print(f"  per-layer  vs single-mask     : {cov('C_single_mask_tt_fgm'):.2f}% -> "
          f"{cov('D_perlayer_mask_tt_fgm'):.2f}%  "
          f"({(cov('D_perlayer_mask_tt_fgm')-cov('C_single_mask_tt_fgm'))/cov('C_single_mask_tt_fgm')*100:+.1f}%)")
    print(f"  wrote {out.name} and {out.with_suffix('.npz').name}")


if __name__ == "__main__":
    main()
