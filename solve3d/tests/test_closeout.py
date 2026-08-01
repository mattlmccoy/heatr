"""Phase A close-out gates (STEP 1 cross-family band, STEP 2 shape parity).

RUNS IN THE geo-prewarp VENV (shape_gate needs scipy):
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    ./.venv312/bin/python -m pytest solve3d/tests/test_closeout.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

RESULTS = Path(__file__).resolve().parents[1] / "results"


def _need(name: str) -> dict:
    p = RESULTS / name
    if not p.exists():
        pytest.skip(f"{name} missing: run the close-out compute first")
    return json.loads(p.read_text())


def test_close_out_is_additive_and_the_task4_record_stands():
    """The close-out must not rewrite history: the frozen Task-1 tolerances and
    the recorded Task-4 verdict stay exactly as they were."""
    tol = _need("parity_tolerances.json")
    gate = _need("phase_a_gate.json")
    cf = _need("parity_tolerances_crossfamily.json")
    # Task-1 tolerances still the measured 1.5x same-method values
    assert tol["safety_factor"] == 1.5
    for q in ("t90_rel", "curve_rel_l2", "sigma_T_rel"):
        assert cf["band"][q]["task1_same_method_tolerance"] == tol["tolerances"][q]
    # the original Task-4 verdict is untouched and still fails
    assert gate["verdict"]["gate_ok"] is False
    assert gate["tolerances"] == tol["tolerances"]


def test_cross_family_band_is_a_bound_not_a_fudge():
    """The band must be the declared triangle-inequality SUM x 1.5, and it must
    be built from a dolfinx spread that was actually measured (non-zero)."""
    cf = _need("parity_tolerances_crossfamily.json")
    assert cf["combination_rule"] == "sum"
    assert cf["safety_factor"] == 1.5
    for q, b in cf["band_strict"].items():
        assert b["dolfinx_spread"] > 0.0, q
        assert b["heatr3d_spread"] > 0.0, q
        assert b["combined_spread"] == pytest.approx(
            b["heatr3d_spread"] + b["dolfinx_spread"])
        assert b["tolerance"] == pytest.approx(1.5 * b["combined_spread"])
        assert b["tolerance"] > b["task1_same_method_tolerance"], q


def test_exported_fields_belong_to_the_recorded_task4_runs():
    """The shape gate scores fields from a re-run; those re-runs must reproduce
    the recorded Task-4 numbers, or the shape verdict describes a different
    experiment than the parity verdict."""
    ex = _need("field_export.json")
    assert len(ex["arms"]) == 4
    assert ex["all_reproduce"] is True, {
        k: (v["t90_rel_vs_recorded"], v["sigma_T_rel_vs_recorded"])
        for k, v in ex["arms"].items()}


def test_t90_passes_the_pre_declared_cross_family_band_on_all_arms():
    """t90 against the band built by the rule declared BEFORE computing
    (dolfinx spread = MAX over its refinement pairs)."""
    cf = _need("parity_tolerances_crossfamily.json")
    fails = {k: v["t90_rel"]["margin_declared_max_rule"]
             for k, v in cf["arms"].items()
             if not v["t90_rel"]["pass_declared_max_rule"]}
    assert not fails, fails


def test_t90_strict_band_residual_is_pinned_where_it_was_measured():
    """CHARACTERIZATION, not a pass. Under the stricter band (dolfinx spread
    from mid-vs-fine, the local convergence estimate at the mesh Task 4 ran)
    the two CIRCLE arms sit just outside, and the square arms inside.

    This is a real converged offset, not noise: dolfinx t90 goes 325.00 -> 324.35
    s from mid to fine while heatr3d goes 323.35 -> 322.75 s from n=64 to n=96,
    so both engines have essentially stopped moving ~1.6 s apart. Pinned so the
    residual cannot silently grow, and reported as the one open item."""
    cf = _need("parity_tolerances_crossfamily.json")
    strict_fail = sorted(k for k, v in cf["arms"].items()
                         if not v["t90_rel"]["pass"])
    assert strict_fail == ["circle_coupled", "circle_off"], strict_fail
    for k in strict_fail:
        assert cf["arms"][k]["t90_rel"]["margin"] < 1.30, (
            k, cf["arms"][k]["t90_rel"]["margin"])
    for k in ("square_off", "square_coupled"):
        assert cf["arms"][k]["t90_rel"]["pass"]


def test_eqs_resolve_census_still_exact_on_every_arm():
    cf = _need("parity_tolerances_crossfamily.json")
    assert all(v["n_eqs_solves_ok"] for v in cf["arms"].values())


def test_shape_parity_gate():
    """STEP 2, the verdict-carrying gate: melt-region IoU, melt-front position
    and bed-melt agreement, each against its MEASURED cross-family band."""
    sg = _need("phase_a_shape_gate.json")
    assert len(sg["arms"]) == 4
    failures = []
    for name, a in sg["arms"].items():
        for k, c in a["checks"].items():
            if isinstance(c, dict) and not c["pass"]:
                failures.append(f"{name}.{k} = {c['measured']:.6g} > "
                                f"tol {c['tolerance']:.6g} "
                                f"(margin {c['margin']:.2f}x)")
    assert not failures, "\n".join(failures)


def test_circle_shape_gate_survives_the_strictest_available_band():
    """Robustness of the STEP-2 verdict to the band choice: rebuild the circle
    band from dolfinx's mid-vs-fine pair (its spreads there are ~10x smaller
    than coarse-vs-mid) and the circle arms must still pass. If the shape
    verdict depended on the looser pair it would not be worth much."""
    from solve3d import gates
    sg = _need("phase_a_shape_gate.json")
    sd = sg["self_spread_detail"]["circle"]
    for arm in ("circle_off", "circle_coupled"):
        checks = sg["arms"][arm]["checks"]
        for k, c in checks.items():
            if not isinstance(c, dict):
                continue
            tol = gates.combine_spreads(sd["heatr3d_n64_vs_n96"][k],
                                        sd["dolfinx_mid_vs_fine"][k])
            meas = c["measured"]
            ok = (meas <= tol) if tol > 0 else (meas == 0.0)
            assert ok, (arm, k, meas, tol)


def test_bed_melt_band_is_measured_per_shape_not_borrowed():
    """The circle never spills, so its bed-melt band is identically zero and
    cannot bound the square (which spills ~4% of its volume in BOTH engines).
    The square's own band must therefore be non-degenerate and measured."""
    sg = _need("phase_a_shape_gate.json")
    cb = sg["band_by_shape"]["circle"]
    sb = sg["band_by_shape"]["square"]
    assert cb["bed_melt_absdiff_phi0p9"]["degenerate_zero_width"] is True
    assert sb["bed_melt_absdiff_phi0p9"]["degenerate_zero_width"] is False
    assert sb["bed_melt_absdiff_phi0p9"]["tolerance"] > 0.0
    # and both engines really do agree that the square spills
    for arm in ("square_off", "square_coupled"):
        m = sg["arms"][arm]["metrics"]["phi0p9"]
        assert m["out_of_part_frac_a"] > 0.03
        assert m["out_of_part_frac_b"] > 0.03


def test_shape_metrics_are_z_invariant_enough_to_justify_planar_reduction():
    """Both anchors are full-height extrusions. The planar reduction is only
    valid if the metrics do not vary between z-planes -- measured, not assumed."""
    sg = _need("phase_a_shape_gate.json")
    for name, a in sg["arms"].items():
        m = a["metrics"]
        assert m["phi0p9"]["iou__plane_spread"] < 0.02, (name, m["phi0p9"])
        assert m["front_ssd_mm_phi0p9__plane_spread"] < 0.1, name
