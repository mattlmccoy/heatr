"""RED-first: the refusal and the benefit gate must actually change what ships.

A refusal that only logs is not a fix. These tests check the two end-to-end
behaviours the user sees:
  * the inversion rung REFUSES a degenerate proxy and the build emits an
    explicit UNIFORM correction carrying a no_correction_applied banner;
  * a REJECTED benefit verdict reverts the ACTIVE packaging artifacts to
    uniform and records the verdict in provenance.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import heatr3d as H
from studio3d.correction import build_correction
from studio3d.correction_gate import evaluate_correction

FIX = Path(__file__).resolve().parent / "fixtures" / "tamper"


@pytest.fixture()
def box20_stl(tmp_path) -> Path:
    import trimesh
    p = tmp_path / "box20.stl"
    trimesh.creation.box(extents=(20.0, 20.0, 20.0)).export(p)
    return p


def _write_before_arm(grade_dir: Path, part: np.ndarray, *, saturated: bool,
                      rho_mean: float = 0.98, stop: float | None = 0.98,
                      flat_T: bool = False):
    """Write a BEFORE arm artifact pair with a controlled proxy character.

    saturated=True pins a large fraction of in-part rho_final at the maximum,
    reproducing the real Tamper condition. flat_T=True makes T_phi90 flat
    inside one melt window, which is the residual case the spread floor covers
    once rule 2(a) has switched the proxy away from rho_final."""
    out = grade_dir / "heatr3d" / "uncorrected"
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    rho = np.zeros(part.shape)
    T = np.zeros(part.shape)
    n_in = int(part.sum())
    if saturated:
        v = np.ones(n_in)
        v[: n_in // 10] = rng.uniform(0.55, 0.9, n_in // 10)
    else:
        v = rng.uniform(0.55, 0.95, n_in)
    rho[part] = v
    T[part] = (rng.uniform(179.5, 180.5, n_in) if flat_T
               else rng.uniform(150.0, 280.0, n_in))
    phi = np.zeros(part.shape)
    np.savez_compressed(out / "fields.npz", part=part, rho_final=rho,
                        T_phi90=T, phi_final=phi, Qrf=np.zeros(part.shape),
                        sat=np.zeros((1,), np.float32), h=0.060 / part.shape[0])
    (out / "results.json").write_text(json.dumps({
        "arm": "uncorrected", "grid_n": int(part.shape[0]),
        "stop_mean_rho": stop, "rho_final_mean": rho_mean,
        "sim_time_s": 100.0, "max_time_s": 1500.0, "sigma_T": 20.0,
        "rho_final_std": 0.05, "warp_std_pct": 5.0,
        "out_of_part_melt_frac": 0.01,
        "gates": {"T_max_C": 200.0, "T_ceiling_C": 250.0,
                  "T_ceiling_ok": True, "clamp_bound": False,
                  "energy_residual_ok": True, "reached_phi90": True},
    }, indent=2))


# --------------------------------------------------------------------------- #
# refusal -> uniform + banner
# --------------------------------------------------------------------------- #
def test_saturated_rho_proxy_yields_uniform_with_a_banner(box20_stl, tmp_path):
    """Refusal branch reached via a SATURATED rho_final on an arm that did not
    reach its density stop (so rule 2(a) leaves rho_final as the proxy and the
    degeneracy floor is what has to catch it)."""
    grade = tmp_path / "grade"
    from studio3d.runner import voxelize_stl
    part = voxelize_stl(str(box20_stl), 16)
    _write_before_arm(grade, part, saturated=True, rho_mean=0.61, stop=0.98)
    prov = build_correction(grade, str(box20_stl), 16)
    assert prov.get("no_correction_applied") is True, json.dumps(prov, indent=2)
    assert prov["engine"] == "uniform_no_correction"
    assert "refus" in json.dumps(prov).lower()
    with np.load(grade / "heatr3d" / "correction_sat.npz") as d:
        sat = np.asarray(d["sat"], float)
    assert np.allclose(sat[part], 1.0), "the shipped map must be UNIFORM"


def test_flat_T_proxy_on_a_density_stopped_arm_also_refuses(box20_stl, tmp_path):
    """The residual case the SPREAD floor covers: rule 2(a) switches the proxy
    to T_phi90, but this part's T_phi90 sits inside a single melt window, so it
    carries no rankable structure either."""
    grade = tmp_path / "grade"
    from studio3d.runner import voxelize_stl
    part = voxelize_stl(str(box20_stl), 16)
    _write_before_arm(grade, part, saturated=True, rho_mean=0.98, stop=0.98,
                      flat_T=True)
    prov = build_correction(grade, str(box20_stl), 16)
    assert prov.get("no_correction_applied") is True, json.dumps(prov, indent=2)
    assert "flat" in json.dumps(prov).lower()


def test_healthy_before_arm_still_produces_a_modulated_map(box20_stl, tmp_path):
    """The rung is KEPT, not retired: a non-degenerate arm still grades."""
    grade = tmp_path / "grade"
    from studio3d.runner import voxelize_stl
    part = voxelize_stl(str(box20_stl), 16)
    # did not reach the stop -> rho_final is a legitimate proxy, and it is
    # non-degenerate here
    _write_before_arm(grade, part, saturated=False, rho_mean=0.61)
    prov = build_correction(grade, str(box20_stl), 16)
    assert prov.get("no_correction_applied") is not True
    assert prov["engine"] == "heatr3d_native_inversion"
    with np.load(grade / "heatr3d" / "correction_sat.npz") as d:
        sat = np.asarray(d["sat"], float)
    assert not np.allclose(sat[part], 1.0), "expected a modulated map"


def test_density_stopped_arm_records_the_proxy_switch(box20_stl, tmp_path):
    """Even when it proceeds, provenance must say which proxy was used and
    why -- the Tamper shipped with proxy 'rho_final' and nobody could tell
    from the record that this was the wrong choice."""
    grade = tmp_path / "grade"
    from studio3d.runner import voxelize_stl
    part = voxelize_stl(str(box20_stl), 16)
    _write_before_arm(grade, part, saturated=False, rho_mean=0.98, stop=0.98)
    prov = build_correction(grade, str(box20_stl), 16)
    if prov.get("no_correction_applied"):
        pytest.skip("refused; proxy-switch recording covered by the unit test")
    assert prov["proxy"] == "T_phi90"
    assert "density stop" in prov["proxy_reason"]


# --------------------------------------------------------------------------- #
# benefit gate -> revert to uniform
# --------------------------------------------------------------------------- #
def test_rejected_verdict_reverts_active_artifacts_to_uniform(box20_stl, tmp_path):
    from studio3d.correction_gate import apply_verdict
    grade = tmp_path / "grade"
    from studio3d.runner import voxelize_stl
    part = voxelize_stl(str(box20_stl), 16)
    _write_before_arm(grade, part, saturated=False, rho_mean=0.61)
    build_correction(grade, str(box20_stl), 16)
    with np.load(grade / "heatr3d" / "correction_sat.npz") as d:
        assert not np.allclose(np.asarray(d["sat"], float)[part], 1.0)

    before = json.loads((FIX / "feb850ec_uncorrected_results.json").read_text())
    after = json.loads((FIX / "feb850ec_corrected_results.json").read_text())
    before["out_of_part_melt_frac"] = 0.03209
    after["out_of_part_melt_frac"] = 0.17537
    verdict = evaluate_correction(before, after)
    assert verdict["verdict"] == "REJECTED"

    prov = apply_verdict(grade, verdict)
    assert prov["no_correction_applied"] is True
    assert prov["correction_gate"]["verdict"] == "REJECTED"
    with np.load(grade / "heatr3d" / "correction_sat.npz") as d:
        assert np.allclose(np.asarray(d["sat"], float)[part], 1.0), \
            "ACTIVE packaging map was not reverted to uniform"
    with np.load(grade / "heatr3d" / "correction_stack.npz") as d:
        assert np.allclose(np.asarray(d["sat"], float), 1.0)
    disk = json.loads(
        (grade / "heatr3d" / "correction_provenance.json").read_text())
    assert disk["no_correction_applied"] is True
    assert disk["correction_gate"]["failed_guards"]


def test_accepted_verdict_leaves_the_map_alone(box20_stl, tmp_path):
    from studio3d.correction_gate import apply_verdict
    grade = tmp_path / "grade"
    from studio3d.runner import voxelize_stl
    part = voxelize_stl(str(box20_stl), 16)
    _write_before_arm(grade, part, saturated=False, rho_mean=0.61)
    build_correction(grade, str(box20_stl), 16)
    with np.load(grade / "heatr3d" / "correction_sat.npz") as d:
        original = np.asarray(d["sat"], float).copy()
    before = json.loads((FIX / "feb850ec_uncorrected_results.json").read_text())
    before["out_of_part_melt_frac"] = 0.03209
    verdict = evaluate_correction(before, dict(before))
    assert verdict["verdict"] == "ACCEPTED"
    prov = apply_verdict(grade, verdict)
    assert prov.get("no_correction_applied") is not True
    with np.load(grade / "heatr3d" / "correction_sat.npz") as d:
        assert np.array_equal(np.asarray(d["sat"], float), original)
