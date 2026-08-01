"""Compute stage for gif_rotation_kernels: per-angle part-frame heating fields
of the cross at uniform saturation, 24 angles at 15 degrees.

Memory-light and resumable: one angle case is built at a time, each angle's
part-frame field is saved to a per-angle npy in the cache directory, and
already-saved angles are skipped on restart. The final npz bundles them.

From these the render stage builds every mode as a partial average:
  static              = the angle-0 field alone
  continuous rotation = running average over all 24 angles
  90 degree indexing  = average over {0, 90, 180, 270}

Read-only reuse of adjoint2d. Displayed intersection-over-union numbers come
from the stored campaign JSONs (out_rot/cross_headline.json, cross_index90.json).
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "fgm_solve_campaign"))

from adjoint2d import forward as fwd                 # noqa: E402
from adjoint2d.pins import build_case, load_cfg      # noqa: E402
from adjoint2d.rot_frame import rotation_operator    # noqa: E402

OUT_ROT = REPO / "fgm_solve_campaign/out_rot"
CACHE_DIR = REPO / "deck_gifs/cache/c3_parts"
CACHE = REPO / "deck_gifs/cache/c3_kernels_cross.npz"


def angle_Q(cfg: dict, theta: float, shape: tuple[int, int]) -> np.ndarray:
    """One angle's part-frame state-B heating at uniform saturation."""
    c = copy.deepcopy(cfg)
    c["geometry"]["part"]["rotation_deg"] = float(theta)
    case = build_case(c)
    s_lab = np.ones(case.part_mask.shape)
    eps = fwd.eps_field(case)
    sig_b, _inrange = fwd.sigma_state_b(case, s_lab)
    st_b = fwd.solve_electric(case, sig_b, eps)
    R_to_part = rotation_operator(shape, -float(theta))
    return R_to_part.apply(st_b.Qrf, outside=0.0)


def main() -> None:
    t0 = time.perf_counter()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    meta = json.loads((OUT_ROT / "cross_rotavg.json").read_text())
    cfg = load_cfg(meta["config"])
    angles = np.asarray(meta["angles_deg"], dtype=float)

    cfg0 = copy.deepcopy(cfg)
    cfg0["geometry"]["part"]["rotation_deg"] = 0.0
    case0 = build_case(cfg0)
    shape = case0.part_mask.shape

    Qk = []
    for k, th in enumerate(angles):
        part = CACHE_DIR / f"Q_{k:02d}.npy"
        if part.exists():
            Qk.append(np.load(part))
            continue
        q = angle_Q(cfg, th, shape).astype(np.float32)
        np.save(part, q)
        Qk.append(q)
        print(f"angle {th:5.1f} deg done ({k + 1}/{len(angles)}) "
              f"wall {time.perf_counter() - t0:.0f} s", flush=True)

    heads = json.loads((OUT_ROT / "cross_headline.json").read_text())["rows"]
    idx90 = json.loads((OUT_ROT / "cross_index90.json").read_text())["rows"]
    iou = {r["arm"]: r["IoU"] for r in heads}
    iou.update({r["arm"]: r["IoU"] for r in idx90})

    np.savez_compressed(
        CACHE, Qk=np.asarray(Qk, dtype=np.float32), angles_deg=angles,
        part_mask=case0.part_mask, x=case0.x, y=case0.y,
        iou_static=iou["S_uniform"], iou_cont=iou["R_uniform"],
        iou_index90=iou["I90_uniform"],
    )
    print(f"wrote {CACHE}  wall {time.perf_counter() - t0:.1f} s")
    print(f"IoU static {iou['S_uniform']:.4f}  continuous {iou['R_uniform']:.4f} "
          f"index90 {iou['I90_uniform']:.4f}")


if __name__ == "__main__":
    main()
