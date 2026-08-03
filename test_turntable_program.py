"""Red-first tests for UNEQUAL-DWELL turntable execution in the production engine.

Gap being closed (documented in DWELL_SCHEDULE_REPORT.md Section 4 and
Section 12 item 3): the shipped turntable block only executes a FIXED rotation
increment at a FIXED interval (`rfam_eqs_coupled.py:2827-2836`), so an unequal
dwell program cannot be expressed. Its legacy `phases` branch raises
`TypeError` on `len(_n_evts)` at `rfam_eqs_coupled.py:2819`.

Three groups of tests here:

1. Pure parsing of an explicit ordered program, including the deliverable JSON
   the dwell campaign emits (`fgm_solve_campaign/out_dwell/*_turntable_deliverable.json`).
2. Execution order through the REAL engine, read from an instrumented
   angle-against-time trace.
3. Backward compatibility, proved as bit identity against a run captured from
   the engine BEFORE the program-mode edit
   (`tests_fixtures/turntable_baseline_pre.npz`).

Acronyms: FGM = functionally graded material (a spatially varying dopant
saturation map). EQS = electro-quasi-static.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DELIVERABLE = REPO / "fgm_solve_campaign/out_dwell/cross_turntable_deliverable.json"
BASELINE = REPO / "tests_fixtures/turntable_baseline_pre.npz"


# ---------------------------------------------------------------------------
# 1. parsing
# ---------------------------------------------------------------------------

def test_parse_simple_program_gives_event_steps_and_deltas():
    """Three ordered segments at dt 0.5 s become two rotation events.

    Segment 0 is at 0 degrees, which is where the part already is, so it costs
    no event. The moves to 90 and to 45 land at t = 4 s and t = 6 s, i.e. at
    outer step indices 8 and 12, and the engine fires an event at the top of
    outer step k when its sentinel equals k + 1.
    """
    from rfam_eqs_coupled import parse_turntable_program

    prog = [
        {"angle_deg": 0.0, "duration_s": 4.0},
        {"angle_deg": 90.0, "duration_s": 2.0},
        {"angle_deg": 45.0, "duration_s": 4.0},
    ]
    ev, deltas, segs = parse_turntable_program({"program": prog}, dt_s=0.5, n_steps=40)
    assert ev == [9, 13]
    assert deltas == pytest.approx([90.0, -45.0])
    assert [s["angle_deg"] for s in segs] == [0.0, 90.0, 45.0]
    assert [s["start_s"] for s in segs] == pytest.approx([0.0, 4.0, 6.0])


def test_parse_emits_an_event_for_a_nonzero_first_segment():
    """A program whose first hold is not at the base orientation must move."""
    from rfam_eqs_coupled import parse_turntable_program

    prog = [{"angle_deg": 135.0, "duration_s": 10.0}]
    ev, deltas, _segs = parse_turntable_program({"program": prog}, dt_s=0.5, n_steps=40)
    assert ev == [1]
    assert deltas == pytest.approx([135.0])


def test_parse_merges_segments_that_land_on_the_same_outer_step():
    """Two moves inside one outer step must merge into ONE event.

    The engine pops at most one event per outer step
    (`if (it + 1) == tt_event_steps[0]`), so a duplicated sentinel would stall
    the whole queue on a past value and silently freeze every later rotation.
    The merged event carries the SUM of the deltas, which lands the part at the
    later of the two commanded positions.
    """
    from rfam_eqs_coupled import parse_turntable_program

    prog = [
        {"angle_deg": 0.0, "duration_s": 1.0},
        {"angle_deg": 45.0, "duration_s": 0.2},   # under one control step
        {"angle_deg": 90.0, "duration_s": 5.0},
    ]
    ev, deltas, _segs = parse_turntable_program({"program": prog}, dt_s=0.5, n_steps=40)
    assert ev == [3]
    assert deltas == pytest.approx([90.0])
    assert len(ev) == len(set(ev))


def test_parse_drops_events_past_the_horizon():
    from rfam_eqs_coupled import parse_turntable_program

    prog = [{"angle_deg": 90.0 * i, "duration_s": 5.0} for i in range(1, 6)]
    ev, deltas, _segs = parse_turntable_program({"program": prog}, dt_s=0.5, n_steps=20)
    # horizon is 10 s: moves at 0, 5 s survive; 10, 15, 20 s do not.
    assert ev == [1, 11]
    assert len(deltas) == 2


def test_parse_rejects_a_non_monotone_or_empty_program():
    from rfam_eqs_coupled import parse_turntable_program

    with pytest.raises(ValueError):
        parse_turntable_program({"program": []}, dt_s=0.5, n_steps=10)
    with pytest.raises(ValueError):
        parse_turntable_program(
            {"program": [{"angle_deg": 0.0, "duration_s": -1.0}]}, dt_s=0.5, n_steps=10)


def test_parse_reads_the_dwell_campaign_deliverable_json():
    """The exact format `fgm_solve_campaign` emits: `moves` of
    {position_deg, dwell_s, move_at_s}."""
    from rfam_eqs_coupled import parse_turntable_program

    raw = json.loads(DELIVERABLE.read_text())
    ev, deltas, segs = parse_turntable_program(
        {"program_json": str(DELIVERABLE)}, dt_s=0.5, n_steps=1500)
    assert len(segs) == len(raw["moves"])
    assert ev == sorted(set(ev))
    # cross literal program: 0, 90, 180, 270 repeating, 5 s each at dt 0.5 s
    assert ev[:4] == [11, 21, 31, 41]
    # deltas are LITERAL, not shortest-path: the cycle wrap 270 -> 0 is -270,
    # which keeps the cumulative angle exactly equal to the program's absolute
    # position instead of letting it drift up by 360 per cycle.
    assert deltas[:4] == pytest.approx([90.0, 90.0, 90.0, -270.0])
    cum = np.cumsum(deltas)
    want = [m["position_deg"] for m in raw["moves"][1:]]
    assert cum == pytest.approx(want)
    # first segment is at 0 degrees so it costs no event
    assert len(ev) == len(raw["moves"]) - 1


def test_parse_can_select_the_reduced_program():
    from rfam_eqs_coupled import parse_turntable_program

    raw = json.loads(DELIVERABLE.read_text())
    _ev, _d, segs = parse_turntable_program(
        {"program_json": str(DELIVERABLE), "program_json_key": "reduced_program"},
        dt_s=0.5, n_steps=1500)
    assert len(segs) == len(raw["reduced_program"]["moves"])
    assert {s["angle_deg"] for s in segs} == {0.0, 90.0}


# ---------------------------------------------------------------------------
# 2. execution order through the real engine
# ---------------------------------------------------------------------------

def _tiny_cfg(program: list[dict], corotate: bool | None = None) -> dict:
    import copy

    import yaml
    cfg = copy.deepcopy(yaml.safe_load(
        (REPO / "configs/diamond_tt_15deg_48rot_nearcont.yaml").read_text()))
    cfg["geometry"]["grid_nx"] = 40
    cfg["geometry"]["grid_ny"] = 40
    cfg["thermal"]["n_steps"] = 40
    cfg["turntable"] = {"enabled": True, "program": program}
    if corotate is not None:
        cfg["turntable"]["corotate_dopant"] = bool(corotate)
    return cfg


def _run_with_angle_trace(cfg: dict):
    """Run the engine and record (outer step index, commanded angle) per event.

    `_build_rotated_part_mask` is called exactly once per rotation event with
    the CUMULATIVE angle, so wrapping it is a zero-physics instrumentation
    hook, the same one `scripts/analysis/turntable_glue.py` uses.
    """
    from unittest import mock

    import rfam_eqs_coupled as rfam

    trace: list[float] = []
    box: dict = {"sat": [], "mask": []}
    orig = rfam._build_rotated_part_mask
    orig_fb_init = rfam._FgmFeedback.__init__

    def patched(geom_cfg, extra_rot_deg, x, y):
        out = orig(geom_cfg, extra_rot_deg, x, y)
        trace.append(float(extra_rot_deg))
        box["mask"].append(np.asarray(out[0], dtype=bool))
        fb = box.get("fb")
        if fb is not None and fb.enabled:
            box["sat"].append(np.array(fb.sat_map, dtype=float))
        return out

    def patched_init(self, sat, block):
        orig_fb_init(self, sat, block)
        box["fb"] = self

    sink = io.StringIO()
    with mock.patch.object(rfam, "_build_rotated_part_mask", patched), \
         mock.patch.object(rfam._FgmFeedback, "__init__", patched_init), \
         contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        state, summary, hist, tt_steps, _opt = rfam.run_sim(cfg)
    fb = box.get("fb")
    box["final_sat"] = (np.array(fb.sat_map, dtype=float)
                        if (fb is not None and fb.enabled) else None)
    return trace, list(tt_steps), state, box


def test_program_segments_execute_in_order_for_their_durations():
    """Instrumented angle-against-time trace of the real engine.

    dt is 0.5 s, so the commanded moves at t = 3, 5 and 9 s must land at outer
    step indices 6, 10 and 18 and the cumulative angles must appear in the
    programmed order. The engine records `it`, which is the sentinel minus 1.
    """
    program = [
        {"angle_deg": 0.0, "duration_s": 3.0},
        {"angle_deg": 30.0, "duration_s": 2.0},
        {"angle_deg": 120.0, "duration_s": 4.0},
        {"angle_deg": 45.0, "duration_s": 11.0},
    ]
    trace, tt_steps, _state, _box = _run_with_angle_trace(_tiny_cfg(program))
    assert trace == pytest.approx([30.0, 120.0, 45.0])
    assert tt_steps == [6, 10, 18]


def test_program_holds_the_last_position_past_the_end_of_the_program():
    program = [
        {"angle_deg": 0.0, "duration_s": 2.0},
        {"angle_deg": 90.0, "duration_s": 1.0},
    ]
    trace, tt_steps, _state, _box = _run_with_angle_trace(_tiny_cfg(program))
    assert trace == pytest.approx([90.0])
    assert tt_steps == [4]


# ---------------------------------------------------------------------------
# 3. dopant co-rotation, program mode only
# ---------------------------------------------------------------------------

def test_corotated_sat_map_at_90_is_the_exact_pixel_permutation():
    """The engine-internal co-rotation must reproduce the production
    convention: rotation_deg = +90 is np.rot90(k=-1) in array coordinates."""
    from rfam_eqs_coupled import corotate_sat_map

    rng = np.random.default_rng(0)
    sat = rng.random((40, 40))
    mask = np.zeros((40, 40), dtype=bool)
    mask[8:32, 8:32] = True
    out = corotate_sat_map(sat, 90.0, mask, outside=1.0)
    expect = np.where(mask, np.rot90(sat, k=-1), 1.0)
    assert np.max(np.abs(out - expect)) < 1e-9


def test_corotated_sat_map_agrees_with_the_shared_helper():
    from rfam_eqs_coupled import corotate_sat_map
    from scripts.analysis.orientation_map_rotation import rotate_sat_map

    rng = np.random.default_rng(1)
    sat = rng.random((40, 40))
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 6:34] = True
    got = corotate_sat_map(sat, 37.0, mask, outside=1.0)
    want = rotate_sat_map(sat, 37.0, part_mask_rot=mask, outside=1.0)
    assert np.max(np.abs(got - want)) == 0.0


def _sat_cfg(cfg: dict, sat: np.ndarray, tmp_path: Path) -> dict:
    p = tmp_path / "sat.npz"
    np.savez_compressed(p, sat_map=np.asarray(sat, dtype=np.float32))
    cfg["fgm_feedback"] = {"enabled": True, "sat_map_npz_direct": str(p),
                           "sat_max": 1.0}
    return cfg


def test_dopant_map_corotates_at_each_program_event(tmp_path):
    """The dopant is PRINTED INTO the part, so it has to turn with the part.

    The engine as shipped remaps temperature, relative density and melt
    fraction at a rotation event but leaves `_FgmFeedback.sat_map` in the lab
    frame (`rfam_eqs_coupled.py:286, :443`). In program mode the map must be
    re-rotated to the cumulative angle before conductivity is recomputed.
    """
    from rfam_eqs_coupled import corotate_sat_map

    rng = np.random.default_rng(3)
    sat = 0.2 + 0.8 * rng.random((40, 40))
    program = [{"angle_deg": 0.0, "duration_s": 2.0},
               {"angle_deg": 90.0, "duration_s": 8.0}]
    cfg = _sat_cfg(_tiny_cfg(program), sat, tmp_path)
    trace, _tt, _state, box = _run_with_angle_trace(cfg)
    assert trace == pytest.approx([90.0])
    got = box["final_sat"]
    mask90 = box["mask"][-1]
    want = corotate_sat_map(sat, 90.0, mask90, outside=1.0).astype(np.float32)
    assert np.max(np.abs(got - want)) == 0.0
    # and it is a real 90-degree pixel permutation inside the part
    assert np.max(np.abs(got - np.rot90(sat, k=-1).astype(np.float32))[mask90]) < 1e-6
    # the decisive check: the map is NO LONGER the lab-frame original
    assert np.max(np.abs(got - sat.astype(np.float32))) > 1e-3


def test_corotation_can_be_switched_off_in_program_mode(tmp_path):
    rng = np.random.default_rng(4)
    sat = 0.2 + 0.8 * rng.random((40, 40))
    program = [{"angle_deg": 0.0, "duration_s": 2.0},
               {"angle_deg": 90.0, "duration_s": 8.0}]
    cfg = _sat_cfg(_tiny_cfg(program, corotate=False), sat, tmp_path)
    _trace, _tt, _state, box = _run_with_angle_trace(cfg)
    assert np.max(np.abs(box["final_sat"] - sat.astype(np.float32))) == 0.0


def test_fixed_step_mode_still_leaves_the_map_in_the_lab_frame(tmp_path):
    """Defaults preserved: the documented lab-frame behaviour of the SHIPPED
    fixed-step mode is untouched by this change."""
    import copy

    import yaml
    rng = np.random.default_rng(5)
    sat = 0.2 + 0.8 * rng.random((40, 40))
    cfg = copy.deepcopy(yaml.safe_load(
        (REPO / "configs/diamond_tt_15deg_48rot_nearcont.yaml").read_text()))
    cfg["geometry"]["grid_nx"] = 40
    cfg["geometry"]["grid_ny"] = 40
    cfg["thermal"]["n_steps"] = 20
    cfg["turntable"] = {"enabled": True, "rotation_deg": 90.0,
                        "total_rotations": 2, "rotation_interval_s": 3.0}
    cfg = _sat_cfg(cfg, sat, tmp_path)
    _trace, _tt, _state, box = _run_with_angle_trace(cfg)
    assert len(box["mask"]) >= 1
    assert np.max(np.abs(box["final_sat"] - sat.astype(np.float32))) == 0.0


# ---------------------------------------------------------------------------
# 4. backward compatibility, bit identity
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not BASELINE.exists(), reason="baseline fixture not captured")
def test_fixed_step_turntable_run_is_bit_identical_to_the_pre_edit_engine():
    sys.path.insert(0, str(REPO / "tests_fixtures"))
    from make_turntable_baseline import run_baseline

    ref = np.load(BASELINE)
    got = run_baseline()
    for key in ("T", "rho_rel", "phi"):
        d = float(np.max(np.abs(got[key] - ref[key])))
        assert d == 0.0, f"{key} drifted by {d!r} against the pre-edit engine"
    assert np.array_equal(got["part_mask"], ref["part_mask"])
    assert np.array_equal(got["tt_rotation_steps"], ref["tt_rotation_steps"])
    de = float(np.max(np.abs(got["energy_doped_J_per_m"]
                             - ref["energy_doped_J_per_m"])))
    assert de == 0.0, f"doped energy history drifted by {de!r}"


# ---------------------------------------------------------------------------
# 5. the legacy phases branch
# ---------------------------------------------------------------------------

def test_legacy_phases_branch_no_longer_raises_typeerror():
    """`rfam_eqs_coupled.py:2819` called `len()` on an int, so the only branch
    that accepted arbitrary angles crashed on entry."""
    import copy

    import yaml
    cfg = copy.deepcopy(yaml.safe_load(
        (REPO / "configs/diamond_tt_15deg_48rot_nearcont.yaml").read_text()))
    cfg["geometry"]["grid_nx"] = 40
    cfg["geometry"]["grid_ny"] = 40
    cfg["thermal"]["n_steps"] = 20
    cfg["turntable"] = {"enabled": True,
                        "phases": [{"angle_deg": 0.0}, {"angle_deg": 30.0},
                                   {"angle_deg": 75.0}]}
    trace, _tt, _state, _box = _run_with_angle_trace(cfg)
    assert trace == pytest.approx([30.0, 75.0])
