"""Premix vs graded, head-to-head in run_sim at the 250 C ceiling.

Three dopant configurations, each driven (bisected power) to the same 250 C part
peak, compared on part densification and uniformity:

  uniform   s = 1 inside the part, no premix         (the printed baseline)
  premix    uniform bed floor at 15 wt%, no grading  (parasitic, from the sweep)
  graded    the adjoint's optimal graded s-map        (adjoint2d on jared, voltage)
            injected via fgm_feedback.sat_map_npz_direct (two-sided per-node)

The graded map is produced by `adjoint2d.shape_solve` on jared's own geometry, so
it lands on the run_sim grid directly (no resampling). Answers: does putting the
dopant WHERE THE PART NEEDS IT (grading) beat a uniform bed floor (premix)?
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import yaml

import rfam_eqs_coupled as rc
from premix_study_metrics import peak_to_mean

BASE = "configs/jared_exp1_40mm_premix.yaml"
ADJ_MAPS = Path("results/premix_vs_graded/jared_rect_maps.npz")
GRADED_SAT = Path("results/premix_vs_graded/graded_sat_map.npz")
OUTDIR = Path("results/premix_vs_graded")
CEILING_C = 250.0


def _base():
    return yaml.safe_load(open(BASE))


def _cfg_uniform(gen_power_w):
    c = _base(); c["premix"] = {"frac": 0.0, "budget": "floor_added"}
    c["electric"]["generator_power_w"] = gen_power_w
    c["electric"]["target_power_w"] = gen_power_w
    return c


def _cfg_premix(gen_power_w):
    c = _cfg_uniform(gen_power_w); c["premix"] = {"frac": 0.6, "budget": "floor_added"}
    return c


def _cfg_graded(gen_power_w, sat_max):
    c = _cfg_uniform(gen_power_w)
    c["fgm_feedback"] = {"enabled": True,
                         "sat_map_npz_direct": str(GRADED_SAT.resolve()),
                         "sat_max": float(sat_max)}
    return c


def _run(cfg):
    return rc.run_sim(copy.deepcopy(cfg))[0]


def _peak(cfg, part_mask):
    return float(_run(cfg).T[part_mask].max())


def drive_to_ceiling(cfg_fn, part_mask, lo=300.0, hi=8000.0, iters=8):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if _peak(cfg_fn(mid), part_mask) < CEILING_C:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _write_graded_sat(arm_key: str):
    """Extract the chosen adjoint arm's s-map and write it as a run_sim sat_map."""
    d = np.load(ADJ_MAPS, allow_pickle=True)
    s = np.asarray(d[arm_key], dtype=np.float32)  # sat = sigma/sigma_d0, full grid
    np.savez(GRADED_SAT, sat_map=s)
    return float(s[d["part_mask"].astype(bool)].min()), float(s[d["part_mask"].astype(bool)].max())


def main(graded_arm: str = "A15"):
    OUTDIR.mkdir(parents=True, exist_ok=True)
    base = _base()
    part_mask = rc.make_domain(base)[3]
    lo_g, hi_g = _write_graded_sat(graded_arm)
    sat_max = max(1.0, hi_g)
    arms = {
        "uniform": (_cfg_uniform, 1.0),
        "premix_15wt": (_cfg_premix, 1.0),
        "graded": (lambda p: _cfg_graded(p, sat_max), 1.0),
    }
    results = {}
    for name, (cfg_fn, _) in arms.items():
        drive = drive_to_ceiling(cfg_fn, part_mask)
        st = _run(cfg_fn(drive))
        rec = dict(drive_to_ceiling_W=drive,
                   part_peak_T_C=float(st.T[part_mask].max()),
                   part_mean_phi=float(st.phi[part_mask].mean()),
                   part_peak_to_mean=peak_to_mean(st.T, part_mask),
                   part_mean_T_C=float(st.T[part_mask].mean()))
        results[name] = rec
        print(f"[{name:12s}] drive={drive:6.0f}W peakT={rec['part_peak_T_C']:.1f} "
              f"phi={rec['part_mean_phi']:.3f} peak/mean={rec['part_peak_to_mean']:.3f}")
    results["_meta"] = dict(graded_arm=graded_arm, graded_sat_range=[lo_g, hi_g], sat_max=sat_max)
    (OUTDIR / "premix_vs_graded.json").write_text(json.dumps(results, indent=2))
    print(f"wrote {OUTDIR/'premix_vs_graded.json'}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "A15")
