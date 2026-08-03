"""Why does the prototype's re-implementation of the step-2 control not
reproduce the step-2 harmful class?

Bisects the map-construction pipeline one stage at a time, on the triangle,
against the stored step-2 artifact:

  s2_stored      the exact stored step-2 map, loaded by the PRODUCTION loader
  proto_cont     the prototype map at the same gain, continuous, no printer
                 round trip
  proto_4bpp     the prototype map at the same gain, quantized to 4 bits per
                 pixel at simulation resolution
  proto_dpi      the prototype map pushed through the printer-resolution round
                 trip and quantized, the way `fgm_generator.generate_fgm` does

All four are injected through the SAME permittivity-co-varying hook and scored
at the same read states.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom

from . import control, forward as fwd, objective as obj
from .control_eps import evaluate, reproduce_step2_map
from .pins import build_case, load_cfg


def printer_round_trip(sat: np.ndarray, x: np.ndarray, y: np.ndarray,
                       dpi: int = 720, bpp: int = 4) -> np.ndarray:
    """Reproduce the resolution round trip of `fgm_generator.generate_fgm`.

    The map is upsampled to printer dots per inch, quantized, written, then
    resampled back to the simulation grid by `_FgmFeedback.from_config`.
    """
    ny, nx = sat.shape
    span_x = float(x[-1] - x[0])
    span_y = float(y[-1] - y[0])
    nx_dpi = max(1, int(round(span_x / 0.0254 * dpi)))
    ny_dpi = max(1, int(round(span_y / 0.0254 * dpi)))
    up = zoom(sat, (ny_dpi / ny, nx_dpi / nx), order=1)
    max_val = float((1 << bpp) - 1)
    lm = np.clip(np.round(up * max_val), 0, max_val).astype(np.uint8)
    back = zoom(lm.astype(np.float32) / np.float32(max_val), (ny / ny_dpi, nx / nx_dpi), order=1)
    return np.clip(back, 0.0, 1.0).astype(np.float64)


def run(cfg_path: str, shape: str, stored_npz: str, stored_mag: float,
        gain: float, outdir: str) -> dict:
    cfg = load_cfg(Path(cfg_path).resolve())
    case = build_case(cfg)
    out = Path(outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    s_u = np.ones(case.part_mask.shape)
    tr_u = fwd.forward(case, s_u, stop_after_phi=0.90, stop_margin_steps=2)
    rs = obj.read_states(tr_u)
    proxy = tr_u.T_at_end(rs.melt_onset_index)

    res = {"shape": shape, "gain": gain, "uniform": evaluate(case, s_u, True), "arms": {}}
    res["arms"]["s2_stored"] = reproduce_step2_map(case, stored_npz, stored_mag, cfg)

    cont = control.proportional_inverse_map(proxy, case.part_mask, magnitude=gain)
    res["arms"]["proto_cont"] = evaluate(case, cont, True)

    q4 = np.where(case.part_mask, control.quantize(cont, bpp=4), 1.0)
    res["arms"]["proto_4bpp"] = evaluate(case, q4, True)

    dpi = printer_round_trip(np.where(case.part_mask, cont, 0.0), case.x, case.y)
    res["arms"]["proto_dpi"] = evaluate(case, dpi, True)

    res["arms"]["proto_dpi_outside_one"] = evaluate(
        case, np.where(case.part_mask, dpi, 1.0), True)

    (out / f"{shape}_bisect.json").write_text(json.dumps(res, indent=2, default=float))
    return res


if __name__ == "__main__":
    r = run(sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]),
            float(sys.argv[5]), sys.argv[6])
    u = r["uniform"]["holdout"]
    print(f"{'arm':26s} {'peak':>10s} {'melt':>10s} {'vs uniform':>12s} {'P_abs':>9s}  status")
    print(f"{'uniform':26s} {r['uniform']['fit']:10.4f} {u:10.4f} {'':>12s} "
          f"{r['uniform']['P_abs_W_per_m']:9.1f}")
    for k, v in r["arms"].items():
        h = v["holdout"]
        pct = "n/a" if h is None else f"{100 * (h - u) / u:+11.1f}%"
        print(f"{k:26s} {v['fit']:10.4f} {-1 if h is None else h:10.4f} {pct:>12s} "
              f"{v['P_abs_W_per_m']:9.1f}  {v['status']}")
