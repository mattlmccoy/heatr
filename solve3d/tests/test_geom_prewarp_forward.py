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
