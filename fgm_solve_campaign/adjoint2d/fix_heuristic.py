"""Recompute the heuristic control arms with the full-horizon proxy field.

The first pass built the control's proportional-inverse map from a march that
the shape-fidelity early stop had truncated, so on the slow shapes the proxy
was not the melt-onset temperature field the published gains were selected
against. This recomputes H_sig and H_eps only and patches the stored artifacts.
The adjoint arms are untouched (they never used the proxy).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import shape_solve as ss
from .pins import build_case, load_cfg


def run(cfg_path: str, shape: str, gain: float, outdir: str) -> dict:
    out = Path(outdir).resolve()
    case = build_case(load_cfg(Path(cfg_path).resolve()))
    res = json.loads((out / f"{shape}.json").read_text())

    sat_h, proxy_info = ss.heuristic_map(case, gain)
    m_hs = ss._score(case, ss._forward(case, sat_h))
    m_hs.update({"arm": "H_sig", "magnitude": gain})
    m_he = ss._score(case, ss._forward(case, sat_h, eps_covary=True))
    m_he.update({"arm": "H_eps", "magnitude": gain})

    res["H_sig"], res["H_eps"], res["proxy_info"] = m_hs, m_he, proxy_info
    res["heuristic_recomputed_with_full_horizon_proxy"] = True
    (out / f"{shape}.json").write_text(json.dumps(res, indent=2, default=float))

    npz = dict(np.load(out / f"{shape}_maps.npz"))
    npz["heuristic"] = sat_h
    np.savez_compressed(out / f"{shape}_maps.npz", **npz)
    print(f"{shape:9s} proxy={proxy_info['proxy']:32s} "
          f"H_sig J={m_hs['J']:9.2f} IoU={m_hs['IoU']:.4f}  "
          f"H_eps J={m_he['J']:9.2f} IoU={m_he['IoU']:.4f}", flush=True)
    return res


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4])
