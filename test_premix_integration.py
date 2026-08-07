"""Integration guards for premix wiring into the 2-D coupled model (run_sim).

Two behaviors, both driven through the real forward path on a tiny jared grid:
  1. premix.frac=0 is bit-identical to no-premix (off-path guarantee, config threading).
  2. premix.frac>0 makes the premixed BED absorb RF (nonzero Qrf outside the part) and
     redistributes a FIXED total power -- mirroring heatr3d.compute_qrf_3d(premix=True).
     This forces the full 3-point wiring: material law + skip bed-zeroing + whole-domain
     power enforcement.
"""
import copy

import numpy as np
import yaml

import rfam_eqs_coupled as rc


def _tiny_jared():
    cfg = yaml.safe_load(open("configs/jared_exp1_40mm.yaml"))
    cfg["geometry"]["grid_nx"] = 24
    cfg["geometry"]["grid_ny"] = 24
    cfg["thermal"]["n_steps"] = 1
    return cfg


def _doped_mask(cfg):
    # make_domain returns: x, y, _, part_mask, doped_mask, elec_hi, elec_lo, fill_frac, ...
    return rc.make_domain(cfg)[4]


def test_premix_zero_is_bit_identical_qrf():
    base = _tiny_jared()
    ref = rc.run_sim(copy.deepcopy(base))[0].Qrf
    cfg = copy.deepcopy(base)
    cfg["premix"] = {"frac": 0.0, "budget": "floor_added"}
    new = rc.run_sim(cfg)[0].Qrf
    assert np.array_equal(ref, new)


def test_premix_positive_makes_bed_absorb_and_conserves_total():
    base = _tiny_jared()
    doped = _doped_mask(base)
    bed = ~doped
    ref_state = rc.run_sim(copy.deepcopy(base))[0]
    cfg = copy.deepcopy(base)
    cfg["premix"] = {"frac": 0.3, "budget": "floor_added"}
    pm_state = rc.run_sim(cfg)[0]
    # Bed absorbs under premix (it does NOT without premix -> zero_qrf_outside_doped).
    assert ref_state.Qrf[bed].sum() == 0.0
    assert pm_state.Qrf[bed].sum() > 0.0
    # Fixed total power: premix redistributes, it does not invent energy. Both runs
    # enforce the SAME generator target (ref over the doped region, premix over the
    # whole domain), so the total absorbed Qrf matches within tolerance.
    assert np.isclose(pm_state.Qrf.sum(), ref_state.Qrf.sum(), rtol=0.02)


def test_premix_persists_across_multistep_run_with_updates_off():
    # The study runs many steps; the loop re-solve blocks (update_interval ticks) are
    # NOT premix-aware and would reset the bed to virgin. With update_interval=0 (valid
    # here: sigma has no T/rho coupling) no in-run re-solve fires, so premix persists
    # to the returned Qrf. This guards the actual study path.
    base = _tiny_jared()
    base["thermal"]["n_steps"] = 40
    base["electric"]["update_interval"] = 0
    doped = _doped_mask(base)
    cfg = copy.deepcopy(base)
    cfg["premix"] = {"frac": 0.3, "budget": "floor_added"}
    st = rc.run_sim(cfg)[0]
    assert st.Qrf[~doped].sum() > 0.0
