"""Solve-at-the-production-price runner: one shape at one out-of-bounds price.

Pure argument forwarding into `adjoint2d.asym_solve.main`, which owns every
piece of logic and is unit tested in `adjoint2d/tests/test_asym_solve_logic.py`.
The only thing this file adds over `python -m adjoint2d.asym_solve` is the
ability to select the optimizer list, because the v2.1.0 optimizer policy sends
the constrained (hinge) objective class to the method of moving asymptotes and
running the L-BFGS-B arm here would double the cost for an arm the policy would
not ship.

Each invocation produces the three arms the question needs at ONE price:

  U_uniform   the uniform-saturation control, read at its own J_asym argmin at
              THIS price.
  PHI4_prev   the stored melt-region-solved 4-bits-per-pixel map, RE-READ at
              this price at its own J_asym argmin. This is the current
              production recipe (solve on the melt objective, stop on J_asym).
  ASYM_mma_*  the map SOLVED at this price, continuous and quantized to 4 bits
              per pixel inside the part. This is the arm that has never been
              run.

Run:
  ./.venv312/bin/python run_wout_solve.py <shape> <w_out> [budget] [floor]
"""
from __future__ import annotations

import sys

from adjoint2d import asym_solve


def main(shape: str, w_out: float, budget: float = 40.0,
         floor: float = 0.85, outdir: str = "out_wout") -> dict:
    tag = f"_w{str(w_out).replace('.', 'p')}"
    return asym_solve.main(shape, outdir, budget=budget, floor=floor,
                           w_out=w_out, tag=tag, optimizers=("mma",))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]),
         float(sys.argv[3]) if len(sys.argv) > 3 else 40.0,
         float(sys.argv[4]) if len(sys.argv) > 4 else 0.85)
