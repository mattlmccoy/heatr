"""Task 2 and 3: printability of the solved dopant maps.

The printer rasterizes binder saturation at 2 or 4 bits per pixel. Nothing in
hardware is continuous, so the continuous solved map is a bound, not a product.
This module puts each solved map through the production quantizer
(`adjoint2d.printability`, contract-tested against the stored artifacts), re-runs
the real forward on the quantized map, re-optimizes the stop time, and re-scores.

Arms per shape, all scored under the SAME J at their OWN optimal stop
(t_stop = argmin over the trajectory of J):

  A1_cont        continuous solved map, box [0, 1]      single printing pass
  A1_4bpp        the same map on the 16-level grid       single pass
  A1_2bpp        the same map on the 4-level grid        single pass
  A1_4bpp_dpi    the same map through the full printer-resolution round trip
  A15_*          the same four, box [0, 1.5], which needs a SECOND pass
                 wherever saturation exceeds 1.0

Convention, stated because it changes the numbers: quantization is applied
INSIDE the part only. Outside the part the saturation is held at its nominal
value 1 so that an arm changes the dopant map and nothing else; quantizing the
outside would change the sub-pixel geometry fill of boundary cells, which is a
geometry change, not a printing effect.

Run: ./.venv312/bin/python -m adjoint2d.verify_print <out.json>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from . import forward as fwd, printability as pq
from . import shape_objective as so
from .pins import build_case, load_cfg
from .verify_hist import CFGD, WINNERS

WT = Path(__file__).resolve().parent.parent
OUT_SHAPE = WT / "out_shape"
PATIENCE = 250
BOX_MAX = {"A1": 1.0, "A15": 1.5}


def score(case, s: np.ndarray) -> dict:
    tr = fwd.forward(case, s, stop_after_phi=None, shape_stop_patience=PATIENCE,
                     eps_covary=False)
    m = so.full_metrics(tr, case)
    m["P_abs_W_per_m"] = tr.P_abs_B
    m["frac_dT_clipped_max"] = tr.frac_dT_clipped_max
    m["frac_temp_cap_max"] = tr.frac_temp_cap_max
    m["frac_qrf_cap"] = tr.frac_qrf_cap
    m["sat_mean_in_part"] = float(np.mean(s[case.part_mask]))
    m["sat_max_in_part"] = float(np.max(s[case.part_mask]))
    return m


def run_shape(shape: str, outdir: Path) -> dict:
    cfg_name, _gain, _npz = WINNERS[shape]
    cfg = load_cfg(CFGD / f"{cfg_name}.yaml")
    case = build_case(cfg)
    pm = case.part_mask
    maps = np.load(OUT_SHAPE / f"{shape}_maps.npz")

    res = {"shape": shape, "config": str(CFGD / f"{cfg_name}.yaml"),
           "n_part_cells": case.n_part, "arms": {}, "maps": {}}
    store: dict[str, np.ndarray] = {}

    for tag in ("A1", "A15"):
        if tag not in maps.files:
            continue
        s_cont = np.asarray(maps[tag], dtype=float)
        smax = BOX_MAX[tag]
        variants = {
            f"{tag}_cont": s_cont,
            f"{tag}_4bpp": pq.quantize_in_part(s_cont, pm, bpp=4, sat_max=smax),
            f"{tag}_2bpp": pq.quantize_in_part(s_cont, pm, bpp=2, sat_max=smax),
        }
        # Full printer round trip is defined on [0, 1] by the production
        # pipeline (`level_map` is uint8 in [0, max_val]); the double-pass arm
        # is round-tripped on its scaled map and rescaled back so the level
        # quantum stays 1/max_val of a single pass.
        rt = pq.printer_round_trip(np.where(pm, np.clip(s_cont / smax, 0, 1), 1.0),
                                   case.x, case.y, bpp=4) * smax
        variants[f"{tag}_4bpp_dpi"] = np.where(pm, rt, 1.0)

        for name, s in variants.items():
            m = score(case, s)
            m["arm"] = name
            m.update({f"census_{k}": v for k, v in
                      pq.level_census(s, pm, bpp=2 if "2bpp" in name else 4).items()})
            res["arms"][name] = m
            store[name] = s
            print(f"  {name:14s} J {m['J']:8.2f}  IoU {m['IoU']:.4f}  "
                  f"grow {m['bed_melt_pct_of_part']:6.2f}  under {m['part_under_melt_pct']:6.2f}  "
                  f"stop {m['t_stop_index']:4d} ({m['t_stop_s']:6.1f} s)  "
                  f"P_abs {m['P_abs_W_per_m']:6.1f}  "
                  f"levels {m['census_n_levels_used']:3d}")

    np.savez_compressed(outdir / f"{shape}_quantized_maps.npz",
                        part_mask=pm, x=case.x, y=case.y, **store)
    return res


def main(out_path: str) -> dict:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    res = {}
    for shape in WINNERS:
        print(f"=== {shape}")
        res[shape] = run_shape(shape, out.parent)
    out.write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
