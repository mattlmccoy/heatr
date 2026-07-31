"""Dense scan of the control's gain curve.

The step-2 campaign seeded its line search with the stored four-point magnitude
grid {0.30, 0.50, 0.70, 0.85}. If the fit metric is bimodal in the gain, that
seeding can trap the search in the wrong basin. This scans the whole
pre-registered domain on a dense grid so the shape of the curve is measured
rather than inferred.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import control, forward as fwd, objective as obj
from .control_eps import evaluate
from .pins import build_case, load_cfg


def run(cfg_path: str, shape: str, outdir: str, outside: float, eps_covary: bool,
        n: int = 26) -> dict:
    cfg = load_cfg(Path(cfg_path).resolve())
    case = build_case(cfg)
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    s_u = np.ones(case.part_mask.shape)
    tr_u = fwd.forward(case, s_u, stop_after_phi=0.90, stop_margin_steps=2)
    rs = obj.read_states(tr_u)
    proxy = tr_u.T_at_end(rs.melt_onset_index)

    gains = np.linspace(0.05, 2.50, n)
    rows = []
    for g in gains:
        sat = control.proportional_inverse_map(proxy, case.part_mask, magnitude=float(g),
                                               outside=outside)
        met = evaluate(case, sat, eps_covary)
        met["magnitude"] = float(g)
        rows.append(met)
        h = met["holdout"]
        hs = "NOT_REACHED" if h is None else f"{h:9.4f}"
        print(f"m={g:6.4f} fit={met['fit']:9.4f} melt={hs:>11s} "
              f"phi={met['final_phi_bar']:.4f} P={met['P_abs_W_per_m']:7.1f}", flush=True)
    tag = f"{shape}_gaincurve_outside{outside:g}_eps{int(eps_covary)}"
    res = {"shape": shape, "outside": outside, "eps_covary": eps_covary,
           "uniform": evaluate(case, s_u, eps_covary), "rows": rows}
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), bool(int(sys.argv[5])))
