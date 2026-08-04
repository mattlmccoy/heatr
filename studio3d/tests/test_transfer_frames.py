"""RED-first: stack_to_voxel must consume TWO frames, not one.

TAMPER_DIAGNOSIS.md section 3(f): the 2.5-D reps are solved on per-cluster
chambers (65 / 85 mm, ng=160) while the voxel arm is a 60 mm chamber at n=64.
stack_to_voxel took a single ``chamber_m`` and used it for BOTH the source
index mapping and the target voxel grid, so the part was resampled at the wrong
physical scale (85/60 = 1.4167x) and the transfer gate measured a 37.37 % move.
"""
from __future__ import annotations

import numpy as np
import pytest

from studio3d.transfer import TransferError, stack_to_voxel


def _synthetic(src_chamber=0.085, ng=160, nz=11, n=64, tgt_chamber=0.060):
    """A source stack and a target part that describe the SAME physical part
    in two different chamber frames.

    The in-mask map carries a radial GRADIENT on purpose: a uniform map
    cannot detect the rescaling bug (extension supplies the same value
    everywhere, so the mean cannot move) - the first version of this file
    proved exactly that by failing to fail."""
    half_mm = 10.0
    yy, xx = np.meshgrid(np.arange(ng), np.arange(ng), indexing="ij")
    pitch_mm = src_chamber * 1e3 / (ng - 1)
    xs = (xx - (ng - 1) / 2) * pitch_mm
    ys = (yy - (ng - 1) / 2) * pitch_mm
    m2 = (np.abs(xs) <= half_mm) & (np.abs(ys) <= half_mm)
    mask = np.repeat(m2[None], nz, 0)
    r = np.sqrt(xs ** 2 + ys ** 2)
    grad = np.clip(0.8 - 0.06 * r, 0.2, 0.8)     # 0.8 center -> 0.2 rim
    sat = np.where(mask, grad, 1.0).astype(float)
    z_mm = (np.arange(nz) + 0.5) * (2 * half_mm / nz)
    h = tgt_chamber / n
    c = (np.arange(n) + 0.5) * h - tgt_chamber / 2
    X, Y, Z = np.meshgrid(c, c, c, indexing="ij")
    part = ((np.abs(X) <= half_mm * 1e-3) & (np.abs(Y) <= half_mm * 1e-3)
            & (np.abs(Z) <= half_mm * 1e-3))
    return sat, mask, z_mm, part


def test_two_frames_are_accepted_and_preserve_physical_scale():
    sat, mask, z_mm, part = _synthetic()
    rec = stack_to_voxel(sat, mask, z_mm, part, chamber_m=0.085,
                         target_chamber_m=0.060)
    assert rec["dopant_mass_move_rel"] < 0.02
    assert rec["source_chamber_m"] == 0.085
    assert rec["target_chamber_m"] == 0.060


def test_single_frame_mismatch_is_what_broke_the_tamper():
    """Passing the SOURCE chamber as the target (the old behaviour) rescales
    the part and blows the gate. Kept as an executable statement of the bug."""
    sat, mask, z_mm, part = _synthetic()
    with pytest.raises(TransferError):
        stack_to_voxel(sat, mask, z_mm, part, chamber_m=0.085,
                       target_chamber_m=0.085)


def test_target_frame_defaults_to_the_voxel_arm_chamber():
    """Callers that omit it get the runner's chamber, not the source's."""
    from studio3d.runner import CHAMBER_M
    sat, mask, z_mm, part = _synthetic()
    rec = stack_to_voxel(sat, mask, z_mm, part, chamber_m=0.085)
    assert rec["target_chamber_m"] == CHAMBER_M == 0.060
    assert rec["dopant_mass_move_rel"] < 0.02


def test_matched_frames_still_work():
    """A source already emitted on the voxel arm's own chamber is unaffected."""
    sat, mask, z_mm, part = _synthetic(src_chamber=0.060, ng=120)
    rec = stack_to_voxel(sat, mask, z_mm, part, chamber_m=0.060,
                         target_chamber_m=0.060)
    assert rec["dopant_mass_move_rel"] < 0.02


def test_the_two_percent_gate_is_unchanged():
    from studio3d.transfer import MASS_MOVE_GATE
    assert MASS_MOVE_GATE == 0.02


def test_real_tamper_artifact_before_and_after_the_fix():
    """The captured feb850ec inputs (data-contract fixture): the mixed-frame
    bug measured 37.37 percent; the separated frames measure ~2.99 percent.
    Still OVER the 2 percent gate, so the chain still refuses the 2.5-D map
    on this part - the fix removes the frame error, not the verdict."""
    from pathlib import Path
    fx = Path(__file__).parent / "fixtures" / "tamper" / \
        "feb850ec_transfer_inputs.npz"
    with np.load(fx) as d:
        sat, mask = d["sat"], d["part_mask"].astype(bool)
        z_mm, part = d["z_mm"], d["part"].astype(bool)
        src_c = float(d["source_chamber_m"])
        tgt_c = float(d["target_chamber_m"])

    def move(**kw):
        try:
            return stack_to_voxel(sat, mask, z_mm, part, **kw)[
                "dopant_mass_move_rel"]
        except TransferError as e:
            import re
            return float(re.search(r"by ([\d.]+) %", str(e)).group(1)) / 100

    buggy = move(chamber_m=src_c, target_chamber_m=src_c)   # old behaviour
    fixed = move(chamber_m=src_c, target_chamber_m=tgt_c)
    assert buggy == pytest.approx(0.3737, abs=0.01)
    assert fixed == pytest.approx(0.0299, abs=0.01)
    assert fixed > 0.02, "still refused by the unchanged gate on this part"
