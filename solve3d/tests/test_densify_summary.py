import json
from pathlib import Path

import numpy as np
import pytest

from solve3d import densify_summary as ds

REPO = Path(__file__).resolve().parents[2]


def _fields(tmp_path, rho=1.0, n=10, h=1e-3):
    part = np.zeros((n, n, n), bool)
    part[2:8, 2:8, 2:8] = True
    p = tmp_path / "fields.npz"
    np.savez(p, part=part, rho_final=np.where(part, rho, 0.55), h=h)
    return p


def test_uniform_full_density_matches_the_shrink_law(tmp_path):
    s = ds.summarize_densify(_fields(tmp_path, rho=1.0))
    S = 0.55 / 1.0
    lam_xy = S ** 0.04
    lam_z = S / lam_xy ** 2
    assert s["layer_multiplier"] == pytest.approx(1.0 / lam_z, abs=1e-3)
    assert s["warp_std_pct"] == pytest.approx(0.0, abs=1e-6)
    assert s["kind"] == "densify_summary" and s["schema"] == 1
    assert s["rho_green"] == pytest.approx(0.55)


def test_missing_key_is_refused(tmp_path):
    p = tmp_path / "bad.npz"
    np.savez(p, part=np.ones((2, 2, 2), bool))
    with pytest.raises(ValueError, match="rho_final"):
        ds.summarize_densify(p)


def test_write_summary_defaults_next_to_fields(tmp_path):
    out = ds.write_summary(_fields(tmp_path))
    assert out == tmp_path / "densify_summary.json"
    assert json.loads(out.read_text())["kind"] == "densify_summary"


@pytest.mark.parametrize("part,expect", [("cube", 1.707), ("pyramid", 1.583)])
def test_real_densify_runs_reproduce_measured_factors(part, expect):
    f = REPO / "solve3d" / "results" / f"densify_{part}" / "fields.npz"
    if not f.exists():
        pytest.skip(f"{f} not present")
    assert ds.summarize_densify(f)["layer_multiplier"] == pytest.approx(expect, abs=2e-3)
