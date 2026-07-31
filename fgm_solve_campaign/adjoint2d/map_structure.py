"""How far is the ideal dopant map from the proportional-inverse family?

The heuristic family is, by construction, a monotone affine function of one
smoothed proxy field. So the question "could the heuristic have found this map"
has a quantitative answer: regress the ideal map on the proxy field over the
part cells and report the coefficient of determination. A low value means the
ideal map carries structure no gain, and no monotone rescaling of the proxy,
can express.

Also reports a rim-versus-core contrast, since that is the structure the
figures show: mean saturation in the outer two cell rings of the part against
the mean in the eroded core.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion

from . import forward as fwd, objective as obj, shape_objective as so
from .pins import build_case, load_cfg


def rim_core(s: np.ndarray, part_mask: np.ndarray, rings: int = 2) -> dict:
    core = binary_erosion(part_mask, iterations=rings)
    rim = part_mask & ~core
    return {"rim_mean": float(np.mean(s[rim])), "core_mean": float(np.mean(s[core])),
            "rim_minus_core": float(np.mean(s[rim]) - np.mean(s[core])),
            "n_rim": int(rim.sum()), "n_core": int(core.sum())}


def r2_against_proxy(s: np.ndarray, proxy: np.ndarray, part_mask: np.ndarray) -> float:
    y = s[part_mask]
    x = proxy[part_mask]
    A = np.column_stack([x, np.ones_like(x)])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - float(np.sum(resid ** 2)) / max(ss_tot, 1e-30)


def main(cfg_path: str, shape: str, outdir: str) -> dict:
    case = build_case(load_cfg(Path(cfg_path).resolve()))
    out = Path(outdir).resolve()
    data = np.load(out / f"{shape}_maps.npz")
    pm = case.part_mask

    tr_u = fwd.forward(case, np.ones(pm.shape), stop_after_phi=None)
    rs = obj.read_states(tr_u)
    idx = rs.melt_onset_index if rs.melt_onset_index is not None else \
        so.optimal_stop(tr_u, case).index
    proxy = tr_u.T_at_end(idx)

    res = {"shape": shape, "proxy_step": int(idx), "arms": {}}
    for arm in ("heuristic", "A1", "A15"):
        if arm not in data:
            continue
        s = data[arm]
        e = {"mean_in_part": float(np.mean(s[pm])),
             "r2_against_proxy": r2_against_proxy(s, proxy, pm)}
        e.update(rim_core(s, pm))
        res["arms"][arm] = e
        print(f"{shape:9s} {arm:10s} mean {e['mean_in_part']:.3f}  "
              f"R2 vs proxy {e['r2_against_proxy']:+.4f}  "
              f"rim {e['rim_mean']:.3f} core {e['core_mean']:.3f} "
              f"rim-core {e['rim_minus_core']:+.3f}", flush=True)
    (out / f"{shape}_mapstructure.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
