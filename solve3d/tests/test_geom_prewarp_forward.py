import numpy as np
import types
import pytest
from solve3d import geom_prewarp_forward as gpf


def test_embed_in_cubic_grid_places_base_at_z0_centred_xy():
    green = np.zeros((3, 3, 4), bool); green[:, :, :4] = True
    gd = np.where(green, 0.5, 0.0)
    part, sat = gpf.embed_in_grid(green, gd, n=8, z0=1)
    assert part.shape == (8, 8, 8) and sat.shape == (8, 8, 8)
    # footprint centred in x,y (offset (8-3)//2 = 2), base at z0=1
    assert part[2:5, 2:5, 1:5].all()
    assert part.sum() == green.sum()
    assert np.all(sat[~part] == 0.0)


def test_march_dense_heights_reads_shrinkage(monkeypatch):
    # mock heatr3d.run -> a Result-like object; mock shrinkage_analysis
    import heatr3d as H
    fake_res = types.SimpleNamespace(rho_final=np.ones((8, 8, 8)) * 0.9,
                                     part=np.ones((8, 8, 8), bool))
    monkeypatch.setattr(H, "run", lambda *a, **k: fake_res)
    monkeypatch.setattr(H, "shrinkage_analysis",
                        lambda res, p, h, **k: {"_H_final": np.ones((8, 8)) * 0.5,
                                                "warp_std_pct": 3.0})
    grid = H.Grid(n=8, L=8 * 0.001)
    part = np.ones((8, 8, 8), bool); sat = np.full((8, 8, 8), 0.5)
    Hm, warp, res = gpf.march_dense_heights(part, sat, H.Params(), grid)
    assert Hm.shape == (8, 8) and np.isclose(warp, 3.0)


def test_crop_to_footprint_matches_nominal_shape():
    # a 3x3 footprint centred in an 8x8 grid; crop returns the 3x3 block
    full = np.zeros((8, 8)); full[2:5, 2:5] = np.arange(9).reshape(3, 3)
    crop = gpf.crop_to_footprint(full, nx=3, ny=3)
    assert crop.shape == (3, 3)
    assert np.allclose(crop, np.arange(9).reshape(3, 3))


def test_embed_in_grid_centers_z_by_default():
    green = np.ones((2, 2, 4), bool); gd = np.where(green, 0.5, 0.0)
    part, sat = gpf.embed_in_grid(green, gd, n=10)      # z0 default -> (10-4)//2 = 3
    zocc = np.where(part.any(axis=(0, 1)))[0]
    assert zocc.min() == 3 and zocc.max() == 6


def test_embed_in_grid_raises_when_too_tall():
    green = np.ones((2, 2, 12), bool)
    with pytest.raises(ValueError):
        gpf.embed_in_grid(green, np.where(green, 0.5, 0.0), n=8)


def test_run_prewarp_glue_writes_spec_and_record(tmp_path, monkeypatch):
    import types
    import heatr3d as H
    part = np.zeros((3, 3, 4), bool); part[:, :, :4] = True
    sat = np.where(part, 0.5, 0.0)
    fp = tmp_path / "fields.npz"
    np.savez(fp, part=part, sat=sat, h=0.2, L=0.6)

    def fake_run(grid, prt, p, **kw):
        return types.SimpleNamespace(rho_final=np.ones(prt.shape) * 0.9, part=prt)

    def fake_sh(res, p, h, **kw):
        occ = res.part.sum(axis=2).astype(float)
        return {"_H_final": occ * h * 0.5, "warp_std_pct": 1.0, "layer_multiplier": 2.0}

    monkeypatch.setattr(H, "run", fake_run)
    monkeypatch.setattr(H, "shrinkage_analysis", fake_sh)
    out = tmp_path / "pw_spec.npz"
    prov = gpf.run_prewarp(str(fp), str(out), bulk_factor=2.0, tol=0.01,
                           k_max=6, grid_n=8)
    assert out.exists() and (tmp_path / "pw_spec.npz.record.json").exists()
    assert prov["enabled"] is True and "grid_n" in prov and "z0" in prov
    d = np.load(out, allow_pickle=True)
    assert str(d["proxy_field"]) == "solve"
