"""Does a heatr3d verification survive moving to the grown chamber frame?

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    heatr3d_d1_spike/env/bin/python -m solve3d.make_chamber_frame_gate

Writes solve3d/results/chamber_frame_gate.json.

THE QUESTION. The solve lane now sizes the chamber adaptively, and the heatr3d
job wiring can follow it. But a solved map is only useful if the VERIFICATION
agrees, so: run the same part, at matched CELL SIZE, in the frozen 60 mm frame
and in its adaptive frame, and report what actually moves.

THE PART IS THE LIBRARY PYRAMID, not the Tamper, and that is deliberate: the
pyramid MELTS in both frames (378.5 C and 417.2 C plateau estimates, both well
over the 180 C onset), so the comparison isolates the FRAME. The Tamper melts
in only one of them, which would confound "the frame changed" with "the part
finally melted".

MATCHED CELL SIZE, NOT MATCHED n. Holding n while growing L coarsens every
cell, so the part would silently lose resolution and the comparison would be
frame-plus-resolution. `heatr3d_job.n_for_h` holds the cell instead. This is
the S2 commensurability lesson applied rather than restated.

WHAT THIS IS NOT. It is NOT a claim that the two frames agree. The EQS sweep
already measured the in-part Q pattern moving well beyond the refinement band
across chamber sizes, so a shift is EXPECTED here. The gate exists to say how
big it is in the quantities a verification actually reports, so that "re-run
rather than reuse" is a measured instruction and not a slogan.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import heatr3d as H
import heatr3d_job as J
from solve3d import chamber as ch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "solve3d" / "results" / "chamber_frame_gate.json"

N_FROZEN = 64                     # cells/axis in the 60 mm frame
PYR_SIDE_M = 23.248947030192525e-3


def _run(L_m: float, n: int) -> dict:
    grid = H.Grid(n=n, L=L_m)
    part = H.make_geometry(grid, "cone", diam=PYR_SIDE_M, zspan=PYR_SIDE_M)
    p = H.Params()
    t0 = time.time()
    r = H.run(grid, part, p, verbose=False)
    return {
        "L_m": float(L_m), "tag": ch.chamber_tag(L_m), "n": int(n),
        "cell_h_mm": float(grid.h * 1e3),
        "n_part_voxels": int(part.sum()),
        "part_volume_mm3": float(part.sum() * grid.dV * 1e9),
        "sigma_T": float(r.sigma_T),
        "t_phi90_s": float(r.t_phi90_s),
        "reached_phi90": bool(r.reached),
        "T_max_c": float(r.T_max_c),
        "wall_s": time.time() - t0,
    }


def main() -> int:
    frozen = _run(0.060, N_FROZEN)
    h_target = 0.060 / N_FROZEN
    L_adapt = ch.chamber_for_bbox(PYR_SIDE_M, PYR_SIDE_M, PYR_SIDE_M)
    n_adapt = J.n_for_h(L_adapt, h_target)
    grown = _run(L_adapt, n_adapt)

    def rel(k):
        a, b = grown[k], frozen[k]
        return float(a / b - 1.0) if b else float("nan")

    doc = {
        "what": ("the SAME part verified in the frozen 60 mm frame and in its "
                 "adaptive frame, at MATCHED CELL SIZE"),
        "part": "library pyramid (heatr3d cone), side/height 23.2489 mm",
        "part_choice_rationale": (
            "the pyramid MELTS in both frames, so this isolates the frame. "
            "The Tamper melts in only one, which would confound 'the frame "
            "changed' with 'the part finally melted'."),
        "matching_rule": (
            "cell size held, not n: heatr3d_job.n_for_h. Holding n while "
            "growing L coarsens every cell and the part would silently lose "
            "resolution (S2 commensurability)."),
        "frozen": frozen,
        "grown": grown,
        "cell_size_match_rel": float(
            grown["cell_h_mm"] / frozen["cell_h_mm"] - 1.0),
        "part_volume_match_rel": rel("part_volume_mm3"),
        "shifts": {"sigma_T_rel": rel("sigma_T"),
                   "t_phi90_rel": rel("t_phi90_s"),
                   "T_max_rel": rel("T_max_c")},
        "expectation": (
            "a shift is EXPECTED and is not a bug: the EQS sweep already "
            "measured the in-part Q pattern moving 0.175-0.208 rel-L2 across "
            "chamber sizes against a 0.037 remeshing noise floor "
            "(solve3d/results/chamber_field_check.json)"),
        "caveats": {
            "part_volume_not_matched": (
                "heatr3d voxelises the part on each grid and the two grids "
                "are not commensurate, so the discretised part differs "
                "between frames. Any shift below is FRAME PLUS VOXELISATION "
                "and is an upper bound on the frame effect alone."),
            "melt_onset_fallback": (
                "if reached_phi90 is False the sigma_T / T_max reads are "
                "FINAL-TIMESTEP, not melt-onset (heatr3d's loud "
                "MELT_ONSET_FALLBACK flag). Compared like-for-like, but not "
                "the melt-onset quantity Phase E reports.")},
        "instruction": (
            "frozen-frame heatr3d artifacts must be RE-RUN in the grown frame "
            "rather than reused or compared across it; the chamber tag is the "
            "guard that makes accidental reuse visible"),
        "preregistration": ch.preregistration(),
        "hardware_note": ("modelling result only; electrode gap and matching "
                          "network are P-gate territory"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1, default=float))
    print(json.dumps({"frozen": frozen, "grown": grown,
                      "shifts": doc["shifts"]}, indent=1, default=float))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
