"""Unit / wiring tests for the producer ceiling-restoration mode (spec 3a,
docs/superpowers/specs/2026-08-19-producer-ceiling-restoration-design.md).

NO heavy solve. The AL restoration solve and the mesh front end are STUBBED so
these are pure logic, mirroring how test_studio_solve_drive stubs the drive
probe. Two surfaces are pinned:

  1. The DRY helper stage_b3.build_al_case_from_tc and that
     run_tamper_rescue.build_tamper_al_case is now a THIN WRAPPER over it that
     builds a byte-identical ALCase (implicit default True, LAM0/MU0 threaded).
  2. The studio_solve ceiling_restore ORCHESTRATION: drive selected BEFORE the
     AL solve; a drive-limited part or a fine mesh honest-nulls WITHOUT running
     the AL (never a cooking map); a coarse feasible part emits the graded map +
     recommended drive + the map's standing peak + sendable. And that the flag
     is opt-in: ceiling_restore=False takes the exact legacy path.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import stage_b3 as b3
from solve3d import studio_solve as ss
from solve3d.phase_e import run_tamper_rescue as rrescue


# --------------------------------------------------------------------------- #
# 1. DRY helper + tamper thin-wrapper byte-identity
# --------------------------------------------------------------------------- #
class _FakeEqs:
    vol = np.arange(10, dtype=float) + 1.0
    part = np.array([0, 1, 2, 3, 4], dtype=np.int64)


class _FakeTC:
    eqs = _FakeEqs()


class _FakeChain:
    def __init__(self, n_design: int = 5):
        self.n_design = int(n_design)


def _alcase_fields(case):
    dc = case.da_case
    return (case.lam, case.mu, case.t_target, dc.dt, dc.n_steps, dc.implicit,
            tuple(np.asarray(dc._v0).tolist()))


def test_build_al_case_from_tc_fields():
    tc = _FakeTC()
    chain = _FakeChain(5)
    case = b3.build_al_case_from_tc(
        tc, chain, power_density=None, t_target=200.0, dt=0.5, n_steps=2800,
        lam=500.0, mu=1.0e4)
    assert case.lam == 500.0
    assert case.mu == 1.0e4
    assert case.t_target == 200.0
    assert case.da_case.dt == 0.5
    assert case.da_case.n_steps == 2800
    assert case.da_case.tc is tc
    assert case.da_case.chain is chain
    assert case.da_case.implicit is True           # tamper/fine default
    assert np.array_equal(case.da_case._v0, np.ones(5))


def test_build_al_case_from_tc_implicit_override():
    """The coarse producer passes implicit=False to keep the certified explicit
    adjoint; the field must thread through."""
    case = b3.build_al_case_from_tc(
        _FakeTC(), _FakeChain(3), power_density=1.0, t_target=235.0, dt=0.5,
        n_steps=100, implicit=False)
    assert case.da_case.implicit is False


def test_tamper_wrapper_is_byte_identical_to_helper(monkeypatch):
    """run_tamper_rescue.build_tamper_al_case must build the SAME ALCase as a
    direct helper call on the same tc/chain -- one code path, no drift."""
    tc = _FakeTC()
    chain = _FakeChain(5)
    monkeypatch.setattr(rrescue.rt, "build_case", lambda **kw: (tc, {}))
    monkeypatch.setattr(rrescue.rt, "_part_centroids",
                        lambda _tc: np.zeros((5, 3)))
    monkeypatch.setattr(rrescue.dc, "DesignChain", lambda *a, **k: chain)

    wrapped = rrescue.build_tamper_al_case(
        lam=500.0, mu=1.0e4, t_target=200.0, power_density=None, dt=0.5,
        n_steps=2800, envelope_max_time_s=1800.0)
    direct = b3.build_al_case_from_tc(
        tc, chain, power_density=None, t_target=200.0, dt=0.5, n_steps=2800,
        lam=500.0, mu=1.0e4)
    assert _alcase_fields(wrapped) == _alcase_fields(direct)
    assert wrapped.da_case.tc is tc
    assert wrapped.da_case.chain is chain
    assert wrapped.da_case.implicit is True


# --------------------------------------------------------------------------- #
# 2. Orchestration: order, honest-null, fine-mesh guard, graded emission
# --------------------------------------------------------------------------- #
def _feasible_drive_rec():
    return {
        "recommended_power_density_w_per_m3": 0.58 * 1.5915e6,
        "recommended_drive_frac": 0.58,
        "t_eff_c": 235.0,
        "ceiling_c": 250.0,
        "chamber_tag": "ch060",
        "thermal_config_path": "solve3d/thermal_config.json",
        "recommended_drive_reason": "densification_bracket_under_ceiling ...",
    }


def _drive_limited_rec():
    return {
        "recommended_power_density_w_per_m3": None,
        "recommended_drive_frac": None,
        "t_eff_c": 235.0,
        "ceiling_c": 250.0,
        "chamber_tag": "ch060",
        "thermal_config_path": "solve3d/thermal_config.json",
        "recommended_drive_reason": "drive_limited: no feasible drive ...",
    }


def _run(select_rec, n_sub, *, al_return=None, score_return=None):
    calls = []

    def select_drive():
        calls.append("drive")
        return select_rec

    def probe_n_sub():
        calls.append("guard")
        return (n_sub, 1.0e-3)

    def run_al(power_density, t_target):
        calls.append("al")
        run_al.seen = {"power_density": power_density, "t_target": t_target}
        return al_return or {"s_best": np.array([0.1, 0.2, 0.3]),
                             "v_best": np.array([0.1, 0.2, 0.3]), "outer": []}

    run_al.seen = None

    def score(restore):
        calls.append("score")
        return score_return or {
            "scored_rec": {"map_stats": {"mean": 0.2}},
            "peak_T_c": 234.5, "peak_over_ceiling": False,
            "symmetry_gate": {"group": "d4", "sendable": True},
            "sendable": True}

    doc = ss.ceiling_restore_solve(
        select_drive=select_drive, probe_n_sub=probe_n_sub, run_al=run_al,
        score=score)
    return doc, calls, run_al


def test_orchestrator_selects_drive_before_solving():
    doc, calls, run_al = _run(_feasible_drive_rec(), n_sub=1)
    assert calls == ["drive", "guard", "al", "score"]
    assert calls.index("drive") < calls.index("al")
    # the AL targets the drive: it is handed the recommended power + T_eff
    assert run_al.seen["power_density"] == pytest.approx(0.58 * 1.5915e6)
    assert run_al.seen["t_target"] == pytest.approx(235.0)


def test_orchestrator_drive_limited_honest_nulls_without_al():
    doc, calls, run_al = _run(_drive_limited_rec(), n_sub=1)
    assert "al" not in calls                        # never cook a map
    assert run_al.seen is None
    assert doc["ceiling_restored_map"] is None
    assert doc["sendable"] is False
    assert doc["recommended_power_density_w_per_m3"] is None
    assert "drive_limited" in doc["recommended_drive_reason"]


def test_orchestrator_fine_mesh_guard_honest_nulls_without_al():
    doc, calls, run_al = _run(_feasible_drive_rec(), n_sub=3)
    assert "al" not in calls                        # guard fired before the solve
    assert run_al.seen is None
    assert doc["ceiling_restored_map"] is None
    assert doc["sendable"] is False
    assert doc["fine_mesh_guard"]["n_sub"] == 3
    reason = (doc["fine_mesh_guard"]["reason"] + doc.get(
        "ceiling_restore_reason", "")).lower()
    assert "fine mesh" in reason
    assert "n_sub" in reason


def test_orchestrator_graded_emits_map_drive_peak_sendable():
    s_best = np.array([0.11, 0.22, 0.33, 0.44])
    al_return = {"s_best": s_best, "v_best": s_best, "outer": [{"k": 0}]}
    score_return = {
        "scored_rec": {"map_stats": {"mean": 0.275}, "peak_T_c": 236.1},
        "peak_T_c": 236.1, "peak_over_ceiling": False,
        "symmetry_gate": {"group": "d4", "sendable": True}, "sendable": True}
    doc, calls, run_al = _run(_feasible_drive_rec(), n_sub=1,
                              al_return=al_return, score_return=score_return)
    assert calls == ["drive", "guard", "al", "score"]
    # the graded map rides in the doc
    assert np.array_equal(np.asarray(doc["ceiling_restored_map"]), s_best)
    # the recommended drive rides in the doc (the Studio consumer's pinned field)
    assert doc["recommended_drive_frac"] == 0.58
    assert doc["recommended_power_density_w_per_m3"] == pytest.approx(0.58 * 1.5915e6)
    # the map's OWN standing peak + sendable
    assert doc["standing_peak_c"] == pytest.approx(236.1)
    assert doc["sendable"] is True
    assert doc["symmetry_gate"]["sendable"] is True
    assert doc["t_eff_target_c"] == pytest.approx(235.0)


# --------------------------------------------------------------------------- #
# 2b. Over-driven fallback: run the AL where grading is exactly the point
# --------------------------------------------------------------------------- #
def _over_driven_fallback_rec():
    """drive_limited (no feasible uniform drive) BUT a densifying-yet-busting
    drive exists -> the ceiling_restore path should run the AL there, not null."""
    r = _drive_limited_rec()
    r["over_driven_fallback"] = {
        "drive_frac": 0.34,
        "power_density_w_per_m3": 0.34 * 1.5915e6,
        "true_peak_c": 255.0,
        "reason": "lowest densifying drive; uniform busts, AL must pull under",
    }
    return r


def test_orchestrator_runs_al_on_over_driven_fallback():
    doc, calls, run_al = _run(_over_driven_fallback_rec(), n_sub=1)
    assert "al" in calls                             # grading IS the point here
    assert run_al.seen["power_density"] == pytest.approx(0.34 * 1.5915e6)
    assert run_al.seen["t_target"] == pytest.approx(235.0)
    assert doc["ceiling_restore_stage"] == "graded"
    assert doc["drive_source"] == "over_driven_fallback"


def test_orchestrator_over_driven_fallback_still_guarded_by_fine_mesh():
    """The fine-mesh guard still fires before the AL on the fallback path (never
    a days-long fine solve)."""
    doc, calls, run_al = _run(_over_driven_fallback_rec(), n_sub=4)
    assert "al" not in calls
    assert run_al.seen is None
    assert doc["ceiling_restored_map"] is None
    assert doc["sendable"] is False


def test_orchestrator_fallback_al_that_busts_is_not_sendable():
    """Safety net: if the AL runs on the fallback but cannot bring the peak under
    250, score() -> sendable False and the map is NOT shipped as sendable."""
    score_return = {
        "scored_rec": {}, "peak_T_c": 258.0, "peak_over_ceiling": True,
        "symmetry_gate": {"sendable": False}, "sendable": False}
    doc, calls, run_al = _run(_over_driven_fallback_rec(), n_sub=1,
                              score_return=score_return)
    assert "al" in calls                             # it TRIED (grading's job)
    assert doc["sendable"] is False                  # but the gate refused it
    assert doc["peak_over_ceiling"] is True


def test_orchestrator_drive_source_is_recommended_on_feasible():
    """The provenance field distinguishes a normal feasible pick from a
    fallback."""
    doc, calls, run_al = _run(_feasible_drive_rec(), n_sub=1)
    assert doc["drive_source"] == "recommended"


def test_fallback_graded_doc_exposes_actual_solve_power():
    """CONTRACT: a graded map from the over-driven fallback must ship WITH the
    power it was solved at -- else the Studio consumer sees sendable=True but a
    null recommended power (drive-limited leftover) and cannot drive the print.
    The pinned recommended power is populated with the fallback power, and a
    solved_power_density_w_per_m3 field always states the actual AL power."""
    doc, calls, run_al = _run(_over_driven_fallback_rec(), n_sub=1)
    fb_pw = 0.34 * 1.5915e6
    assert doc["solved_power_density_w_per_m3"] == pytest.approx(fb_pw)
    assert doc["recommended_power_density_w_per_m3"] == pytest.approx(fb_pw)
    assert doc["recommended_drive_frac"] == pytest.approx(0.34)
    # provenance stays honest about how the drive was chosen
    assert doc["drive_source"] == "over_driven_fallback"


def test_feasible_graded_doc_also_states_solve_power():
    """The solved-power field is present on the normal feasible path too (always
    states what the shipped map was solved at)."""
    doc, calls, run_al = _run(_feasible_drive_rec(), n_sub=1)
    fpw = 0.58 * 1.5915e6
    assert doc["solved_power_density_w_per_m3"] == pytest.approx(fpw)
    assert doc["recommended_power_density_w_per_m3"] == pytest.approx(fpw)


def test_orchestrator_nulls_when_drive_limited_and_no_fallback():
    """drive_limited with fallback None (cold part) stays honest-null -- the AL
    cannot help a part no drive densifies."""
    rec = _drive_limited_rec()
    rec["over_driven_fallback"] = None
    doc, calls, run_al = _run(rec, n_sub=1)
    assert "al" not in calls
    assert run_al.seen is None
    assert doc["ceiling_restored_map"] is None
    assert doc["sendable"] is False


# --------------------------------------------------------------------------- #
# 3. Doc assembler content per stage (pure)
# --------------------------------------------------------------------------- #
def test_assemble_doc_marks_mode_and_stage():
    doc = ss._assemble_ceiling_restore_doc(
        "drive_limited", drive_rec=_drive_limited_rec())
    assert doc["mode"] == "ceiling_restore"
    assert doc["ceiling_restore_stage"] == "drive_limited"


def test_assemble_doc_graded_carries_provenance():
    s_best = np.array([0.5, 0.6])
    doc = ss._assemble_ceiling_restore_doc(
        "graded", drive_rec=_feasible_drive_rec(),
        restore={"s_best": s_best, "outer": [{"k": 0}]},
        scored={"scored_rec": {"map_stats": {"mean": 0.55}},
                "peak_T_c": 233.0, "peak_over_ceiling": False,
                "symmetry_gate": {"sendable": True}, "sendable": True})
    assert doc["ceiling_restore_stage"] == "graded"
    assert np.array_equal(np.asarray(doc["ceiling_restored_map"]), s_best)
    assert doc["standing_peak_c"] == pytest.approx(233.0)
    assert doc["peak_over_ceiling"] is False
    assert doc["sendable"] is True


# --------------------------------------------------------------------------- #
# 4. solve_extruded dispatch: opt-in, legacy path untouched
# --------------------------------------------------------------------------- #
class _LegacySentinel(Exception):
    pass


def test_ceiling_restore_true_dispatches_before_legacy(monkeypatch, tmp_path):
    """ceiling_restore=True routes to the new path BEFORE the legacy np.load /
    mesh build even begins."""
    seen = {}

    def fake_restore(part_npz, out, t_start, **kw):
        seen["hit"] = True
        return {"mode": "ceiling_restore", "ok": True}

    monkeypatch.setattr(ss, "_solve_extruded_ceiling_restore", fake_restore)
    # if the legacy body were entered it would np.load this missing file
    doc = ss.solve_extruded(str(tmp_path / "missing.npz"), str(tmp_path),
                            ceiling_restore=True)
    assert seen.get("hit") is True
    assert doc["ok"] is True


def test_ceiling_restore_false_takes_legacy_path(monkeypatch, tmp_path):
    """Opt-in: with the flag OFF the legacy body runs (reaches np.load) and the
    ceiling-restore path is never touched -- no drive-first reorder."""
    def boom(*a, **k):
        raise _LegacySentinel()

    monkeypatch.setattr(ss, "ceiling_restore_solve",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("ceiling path entered with flag off")))
    monkeypatch.setattr(ss, "_solve_extruded_ceiling_restore",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("ceiling path entered with flag off")))
    monkeypatch.setattr(ss.np, "load", boom)
    with pytest.raises(_LegacySentinel):
        ss.solve_extruded(str(tmp_path / "any.npz"), str(tmp_path),
                          ceiling_restore=False)


# --------------------------------------------------------------------------- #
# 5. grade-mesh knob: coarsen the AL/grading mesh so the full march is tractable
# --------------------------------------------------------------------------- #
class _BuildSentinel(Exception):
    pass


def _record_build_case(recorded):
    def _stub(rings, z_lo, z_hi, node_density, lc0, p, max_time_s, sample_dt_s):
        recorded.append({"node_density": node_density, "lc0": lc0})
        raise _BuildSentinel()
    return _stub


def test_al_loop_threads_coarse_grade_mesh(monkeypatch, tmp_path):
    """The AL/grading build_case uses the COARSE grade density/lc0 when set (the
    old code hardcoded SOLVE_NODE_DENSITY -> 942 s/eval, infeasible)."""
    rec = []
    monkeypatch.setattr(ss, "build_case", _record_build_case(rec))
    coarse = ss.SOLVE_NODE_DENSITY / 28.0
    lc = ss.SOLVE_LC0 * 3.0
    with pytest.raises(_BuildSentinel):
        ss._ceiling_restore_al_loop(
            [np.zeros((4, 2))], 0.0, 0.004, 50.0, power_density=1.0,
            t_target=235.0, envelope_max_time_s=1800.0, march_time_s=3000.0,
            outer_max=1, inner_budget=1, ckpt_dir=tmp_path,
            grade_node_density=coarse, grade_lc0=lc)
    assert rec[0]["node_density"] == pytest.approx(coarse)
    assert rec[0]["lc0"] == pytest.approx(lc)


def test_al_loop_default_grade_mesh_preserves_solve_density(monkeypatch, tmp_path):
    """Default (no knob) preserves the Phase C SOLVE mesh density -- byte-identical
    behavior for a caller that does not coarsen."""
    rec = []
    monkeypatch.setattr(ss, "build_case", _record_build_case(rec))
    with pytest.raises(_BuildSentinel):
        ss._ceiling_restore_al_loop(
            [np.zeros((4, 2))], 0.0, 0.004, 50.0, power_density=1.0,
            t_target=235.0, envelope_max_time_s=1800.0, march_time_s=3000.0,
            outer_max=1, inner_budget=1, ckpt_dir=tmp_path)
    assert rec[0]["node_density"] == pytest.approx(ss.SOLVE_NODE_DENSITY)
    assert rec[0]["lc0"] == pytest.approx(ss.SOLVE_LC0)


def test_restore_front_end_threads_grade_mesh(monkeypatch, tmp_path):
    """The front-end tc (drive probe + fine-mesh guard read it) is built on the
    SAME grading mesh -- so the guard/score numbers are self-consistent with the
    mesh the AL and the shipped map live on."""
    rec = []

    class _FakeNpz:
        files = ["part", "h", "n"]

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def __getitem__(self, k):
            return {"part": np.ones((4, 4, 4), bool), "h": 0.001, "n": 4}[k]

    monkeypatch.setattr(ss.np, "load", lambda p: _FakeNpz())
    monkeypatch.setattr(ss.sg, "detect_extrusion",
                        lambda part: {"is_extruded": True, "refusal": None})
    monkeypatch.setattr(ss.sg, "z_extent", lambda part, h: (0.0, 0.004))
    monkeypatch.setattr(ss.sg, "mid_slice", lambda part: part)
    monkeypatch.setattr(ss.sg, "outline_rings", lambda sl, h: [np.zeros((4, 2))])
    monkeypatch.setattr(ss, "build_case", _record_build_case(rec))
    coarse = ss.SOLVE_NODE_DENSITY / 28.0
    with pytest.raises(_BuildSentinel):
        ss._solve_extruded_ceiling_restore(
            "x.npz", tmp_path, 0.0, max_time_s=500.0, sample_dt_s=50.0,
            drive_candidates=(0.3,), drive_max_time_s=1.0, adaptive_drive=False,
            drive_max_evals=4, grade_node_density=coarse,
            grade_lc0=ss.SOLVE_LC0 * 3.0)
    assert rec[0]["node_density"] == pytest.approx(coarse)
    assert rec[0]["lc0"] == pytest.approx(ss.SOLVE_LC0 * 3.0)


def test_solve_extruded_passes_grade_params_to_restore(monkeypatch, tmp_path):
    """solve_extruded plumbs the grade knobs through the ceiling_restore
    dispatch."""
    seen = {}

    def rec(part_npz, out, t_start, **kw):
        seen.update(kw)
        return {"ok": True}

    monkeypatch.setattr(ss, "_solve_extruded_ceiling_restore", rec)
    coarse = ss.SOLVE_NODE_DENSITY / 28.0
    ss.solve_extruded("x.npz", str(tmp_path), ceiling_restore=True,
                      grade_node_density=coarse, grade_lc0=ss.SOLVE_LC0 * 3.0)
    assert seen["grade_node_density"] == pytest.approx(coarse)
    assert seen["grade_lc0"] == pytest.approx(ss.SOLVE_LC0 * 3.0)


def test_solve_extruded_default_grade_params_are_solve_constants(monkeypatch,
                                                                 tmp_path):
    seen = {}

    def rec(part_npz, out, t_start, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(ss, "_solve_extruded_ceiling_restore", rec)
    ss.solve_extruded("x.npz", str(tmp_path), ceiling_restore=True)
    assert seen["grade_node_density"] == pytest.approx(ss.SOLVE_NODE_DENSITY)
    assert seen["grade_lc0"] == pytest.approx(ss.SOLVE_LC0)


# --------------------------------------------------------------------------- #
# 6. restore-budget knobs: bound the AL march/envelope + outer/inner evals so
#    the ceiling solve runs in ~minutes-hours instead of never (the second speed
#    lever; grade-mesh in section 5 is the first). _solve_extruded_ceiling_restore
#    already accepts these -- solve_extruded and the CLI must thread them.
# --------------------------------------------------------------------------- #
def test_solve_extruded_passes_restore_march_time_to_restore(monkeypatch,
                                                             tmp_path):
    """The bounded restore-march knob threads through the dispatch. Without it the
    AL march defaults to drive_max_time_s (3000 s -> 60000 steps), the dominant
    per-eval cost."""
    seen = {}

    def rec(part_npz, out, t_start, **kw):
        seen.update(kw)
        return {"ok": True}

    monkeypatch.setattr(ss, "_solve_extruded_ceiling_restore", rec)
    ss.solve_extruded("x.npz", str(tmp_path), ceiling_restore=True,
                      restore_march_time_s=500.0)
    assert seen["restore_march_time_s"] == pytest.approx(500.0)


def test_solve_extruded_passes_restore_budget_knobs_to_restore(monkeypatch,
                                                              tmp_path):
    """The envelope horizon and the outer/inner eval budget thread through too --
    outer x inner AL evaluations are the multiplier on the per-eval march cost."""
    seen = {}

    def rec(part_npz, out, t_start, **kw):
        seen.update(kw)
        return {"ok": True}

    monkeypatch.setattr(ss, "_solve_extruded_ceiling_restore", rec)
    ss.solve_extruded("x.npz", str(tmp_path), ceiling_restore=True,
                      restore_envelope_max_time_s=600.0,
                      restore_outer_max=2, restore_inner_budget=4)
    assert seen["restore_envelope_max_time_s"] == pytest.approx(600.0)
    assert seen["restore_outer_max"] == 2
    assert seen["restore_inner_budget"] == 4


def test_solve_extruded_default_restore_budget_knobs_preserve_behavior(
        monkeypatch, tmp_path):
    """Defaults must equal _solve_extruded_ceiling_restore's own current defaults
    (march None -> drive_max_time_s fallback, envelope 1800 s, outer/inner None ->
    stage_b3 defaults), so a caller that does not set them is byte-identical to
    the pre-knob behavior."""
    seen = {}

    def rec(part_npz, out, t_start, **kw):
        seen.update(kw)
        return {}

    monkeypatch.setattr(ss, "_solve_extruded_ceiling_restore", rec)
    ss.solve_extruded("x.npz", str(tmp_path), ceiling_restore=True)
    assert seen["restore_march_time_s"] is None
    assert seen["restore_envelope_max_time_s"] == pytest.approx(1800.0)
    assert seen["restore_outer_max"] is None
    assert seen["restore_inner_budget"] is None
