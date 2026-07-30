#!/usr/bin/env python3
"""Outer-loop driver: Allison's per-node adaptive-gain absolute-target tuning
law with TWO-SIDED actuation, applied to the voltage-driven square.

Law (pernode_tuning.pernode_sigma_update):
    sigma_i(k+1) = clip( sigma_i(k) + K1_i*(Tt - T_i)/maxDiff, 0, sigma_max )
    K1_i halved on per-node error sign flip; converged when maxDiff <= eps.

Setup (faithful to Allison + DOSECHECK matched-melt convention):
  * Base config: outputs_eqs/fgm_dosecheck/configs/square_baseline_voltage.yaml
    (enforce_generator_power=false, voltage 2428.2 V, fixed exposure 845 steps).
    Voltage drive lets total absorbed dose FLOAT as sigma evolves (Allison's
    unconstrained plant) — this is what makes the absolute target Tt meaningful.
  * Per-cell sigma is injected via the two-sided direct hook (fgm_feedback.
    sat_map_npz_direct + sat_max), which allows sigma to rise ABOVE baseline
    0.04 up to sigma_max while keeping part eps_r fixed at 20.
  * Tt = baseline (iter-0) part-mean temperature at matched melt (~186 C).
  * The per-node UPDATE reads the final per-cell T field (the exposure ends
    at phi_mean~0.90 for the baseline; FGM arms overshoot, and the absolute
    target then also regulates total dose back toward Tt — faithful to
    Allison's min(T_part)>Ttarg stop logic on his converged field).
  * Headline uniformity metric sigma_T is read at the phi_mean=0.90 crossing
    (dissertation matched-melt criterion) for comparability with baseline 4.92.

Two variants:
  * Smax0425 : sigma_max = 0.0425 S/m  (Allison's printable Smax)
  * Smax0060 : sigma_max = 0.06 S/m    (BEYOND printable; head-room probe only)

Run: ./.venv312/bin/python run_pernode_square.py [--max-iters 12] [--variant both]
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]  # repo root (moved to scripts/analysis/)
sys.path.insert(0, str(REPO))  # pernode_tuning lives at the repo root

from pernode_tuning import pernode_sigma_update  # noqa: E402

BASE_CFG = REPO / "outputs_eqs/fgm_dosecheck/configs/square_baseline_voltage.yaml"
OUT_ROOT = REPO / "outputs_eqs/pernode_square"
PY = str(REPO / ".venv312/bin/python")

SIGMA_D0 = 0.04
SIGMA_MIN = 0.0
K1_0 = 0.006          # initial per-node gain (S/m); adaptive halving refines it
EPS_CONV = 5.0        # convergence tolerance on final-field maxDiff (deg C)
AMBIENT_C = 23.0

VARIANTS = {
    "Smax0425": 0.0425,   # Allison's printable clamp
    "Smax0060": 0.06,     # beyond printable — head-room probe
}


def _sigma_T_matched(ts: dict) -> tuple[float, int, dict]:
    """sigma_T = ui_rms*(Tmean-23) at first phi_mean>=0.90 crossing (else final)."""
    phi = np.asarray(ts["mean_phi_part"], float)
    ui = np.asarray(ts["ui_rms_part"], float)
    Tm = np.asarray(ts["mean_T_part_c"], float)
    tt = np.asarray(ts["time_s"], float)
    if np.any(phi >= 0.90):
        idx = int(np.argmax(phi >= 0.90))
    else:
        idx = len(phi) - 1
    sigT = float(ui[idx] * (Tm[idx] - AMBIENT_C))
    info = {"idx": idx, "time_s": float(tt[idx]), "phi": float(phi[idx]),
            "Tmean_c": float(Tm[idx]), "ui_rms": float(ui[idx])}
    return sigT, idx, info


def _run_forward(cfg: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / "used_config_in.yaml"
    with open(cfg_path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=True)
    log = out_dir / "run.log"
    with open(log, "w") as lf:
        rc = subprocess.run(
            [PY, str(REPO / "rfam_eqs_coupled.py"),
             "--config", str(cfg_path), "--output-dir", str(out_dir)],
            stdout=lf, stderr=subprocess.STDOUT, cwd=str(REPO),
        ).returncode
    if rc != 0:
        raise RuntimeError(f"forward solve failed (rc={rc}); see {log}")


def _write_sat_npz(path: Path, sat_grid: np.ndarray) -> None:
    np.savez_compressed(path, sat_map=sat_grid.astype(np.float32))


def run_variant(name: str, sigma_max: float, max_iters: int) -> dict:
    base = yaml.safe_load(BASE_CFG.read_text())
    vout = OUT_ROOT / name
    vout.mkdir(parents=True, exist_ok=True)
    sat_max = sigma_max / SIGMA_D0

    part_mask = None
    sigma_part = None            # per-cell sigma over part cells (applied at iter k)
    K1 = None
    sign_prev = None
    Tt = None
    rows = []

    for k in range(max_iters + 1):
        it_dir = vout / f"iter_{k}"
        cfg = copy.deepcopy(base)

        if k == 0:
            cfg["fgm_feedback"]["enabled"] = False
            applied_sigma_part = None      # uniform 0.04
        else:
            # Build sat grid at sim resolution using the part_mask from iter 0.
            sat_grid = np.ones(part_mask.shape, dtype=np.float32)
            sat_grid[part_mask] = (sigma_part / SIGMA_D0).astype(np.float32)
            sat_path = it_dir
            sat_path.mkdir(parents=True, exist_ok=True)
            npz = it_dir / "sigma_direct.npz"
            _write_sat_npz(npz, sat_grid)
            fb = cfg["fgm_feedback"]
            fb["enabled"] = True
            fb["iterate"] = False
            fb["sat_map_npz_direct"] = str(npz)
            fb["sat_max"] = float(sat_max)
            applied_sigma_part = sigma_part.copy()

        _run_forward(cfg, it_dir)

        # ── Load results ──────────────────────────────────────────────────────
        fields = np.load(it_dir / "fields.npz")
        ts = json.loads((it_dir / "time_series.json").read_text())
        summ = json.loads((it_dir / "summary.json").read_text())
        pm = fields["part_mask"].astype(bool)
        T_final = fields["T"][pm].astype(float)

        if k == 0:
            part_mask = pm
            sigma_part = np.full(int(pm.sum()), SIGMA_D0, dtype=float)
            K1 = np.full(int(pm.sum()), K1_0, dtype=float)
            sign_prev = np.zeros(int(pm.sum()), dtype=float)
            Tt = float(summ["mean_T_part_final_c"])   # matched-melt part-mean
            applied_sigma_part = sigma_part.copy()

        sigT_matched, midx, minfo = _sigma_T_matched(ts)
        sigT_final = float(ts["ui_rms_part"][-1] * (ts["mean_T_part_c"][-1] - AMBIENT_C))
        P_abs = float(summ["integrated_power_doped_W_per_m"])

        # frac of nodes at clamp in the APPLIED map for this iteration
        at_hi = np.isclose(applied_sigma_part, sigma_max, atol=1e-6)
        at_lo = np.isclose(applied_sigma_part, SIGMA_MIN, atol=1e-6)
        frac_clamp = float((at_hi | at_lo).mean())

        # ── Per-node update (drives next iteration's map) ─────────────────────
        res = pernode_sigma_update(
            sigma_part, T_final, Tt, K1, sign_prev, sigma_max,
            sigma_min=SIGMA_MIN, eps=EPS_CONV,
        )

        row = {
            "iter": k, "Tt_c": Tt,
            "sigma_T_matched_c": round(sigT_matched, 3),
            "sigma_T_final_c": round(sigT_final, 3),
            "maxDiff_c": round(res.max_diff, 3),
            "P_abs_W_per_m": round(P_abs, 1),
            "frac_nodes_at_clamp": round(frac_clamp, 4),
            "T_final_mean_c": round(float(T_final.mean()), 2),
            "T_final_min_c": round(float(T_final.min()), 2),
            "T_final_max_c": round(float(T_final.max()), 2),
            "applied_sigma_min": round(float(applied_sigma_part.min()), 5),
            "applied_sigma_max": round(float(applied_sigma_part.max()), 5),
            "matched_melt": minfo,
            "converged": bool(res.converged),
        }
        rows.append(row)
        last_applied_part = applied_sigma_part.copy()   # last actually-run map
        print(f"[{name}] iter {k}: sigT_matched={sigT_matched:.2f} "
              f"maxDiff={res.max_diff:.2f} P={P_abs:.0f} clamp={frac_clamp:.2f} "
              f"Tmean={T_final.mean():.1f}", flush=True)

        if res.converged:
            print(f"[{name}] CONVERGED at iter {k} (maxDiff {res.max_diff:.2f} <= {EPS_CONV})")
            break

        sigma_part = res.sigma
        K1 = res.K1
        sign_prev = res.sign

    # persist converged sigma field (full grid) for map comparison vs Jared
    final_sigma_grid = np.zeros(part_mask.shape, dtype=np.float32)
    final_sigma_grid[part_mask] = last_applied_part.astype(np.float32)
    np.savez_compressed(vout / "converged_sigma_map.npz",
                        sigma_map=final_sigma_grid, part_mask=part_mask,
                        x=np.load(vout / "iter_0/fields.npz")["x"],
                        y=np.load(vout / "iter_0/fields.npz")["y"])
    conv = {"variant": name, "sigma_max": sigma_max, "sat_max": sat_max,
            "Tt_c": Tt, "K1_0": K1_0, "eps_conv_c": EPS_CONV,
            "sigma_d0": SIGMA_D0, "iterations": rows}
    (vout / "convergence.json").write_text(json.dumps(conv, indent=2))
    print(f"[{name}] wrote {vout/'convergence.json'}")
    return conv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-iters", type=int, default=12)
    ap.add_argument("--variant", choices=["Smax0425", "Smax0060", "both"],
                    default="both")
    args = ap.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    names = list(VARIANTS) if args.variant == "both" else [args.variant]
    results = {}
    for nm in names:
        results[nm] = run_variant(nm, VARIANTS[nm], args.max_iters)
    (OUT_ROOT / "summary_all.json").write_text(json.dumps(results, indent=2))
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
