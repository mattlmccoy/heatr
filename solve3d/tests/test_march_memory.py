"""The forward's trajectory storage must be BOUNDED, not proportional to substeps.

THE BUG THIS PINS. `march_enthalpy(record=...)` appended `T.copy()` inside the
SUBSTEP loop, so storage was nsteps * n_sub nodal states. On the Tamper that is

    nsteps 18000 (900 s / 0.05 s) x n_sub 36 = 648000 states
    648000 x 9866 nodes x 8 B                = 51.15 GB

and the run was silently SIGKILLed by the OS about 25-30 minutes in, three
times, with no traceback. Phase E's pyramid survived only because its coarse
mesh gave n_sub = 1: 13000 x 18172 x 8 = 1.89 GB.

n_sub = 36 is the Tamper's 0.105 mm tessellation edge again, now spending
MEMORY rather than wall time: dt_stable is 1.576e-3 s against a 0.05 s sample
step.

THE FIX IS NOT A SMALLER HORIZON. The reverse sweep never needed every state:
`_checkpoint_reader` already replays from ANCHORS every `checkpoint_interval`
steps. The forward was storing 51 GB purely to build 2 GB of anchors it then
threw away. So the forward stores the anchors directly, plus the scalar
objective series the envelope read needs -- J(t) is scalars, not fields.

DEFAULTS ARE UNCHANGED. `record_stride=1` and `record_scalar_fn=None` keep the
old store-everything behaviour bit-identical, so the 19 adjoint tests are
untouched by construction and only the Tamper driver opts in.
"""
from __future__ import annotations

import numpy as np
import pytest

from solve3d import forward as fwd


def _tiny_case(n: int = 5):
    msh = fwd.box_mesh(n)
    p = fwd.ForwardParams()
    mats = fwd.build_materials(msh, fwd.in_part_predicate("circle"), p)
    return msh, p, mats


# --------------------------------------------------------------------------- #
# The default path is untouched
# --------------------------------------------------------------------------- #
def test_default_recording_still_stores_every_step():
    msh, p, mats = _tiny_case()
    rec: dict = {}
    fwd.march_enthalpy(msh, p, mats=mats, q_uniform=1.0e6, max_time_s=0.2,
                       phi_target=2.0, record=rec)
    assert len(rec["T_steps"]) == rec["n_recorded_of"]
    assert rec["record_stride"] == 1


# --------------------------------------------------------------------------- #
# Strided anchors: the bound, and that the anchors are the RIGHT states
# --------------------------------------------------------------------------- #
def test_a_stride_stores_only_anchor_states():
    msh, p, mats = _tiny_case()
    full: dict = {}
    fwd.march_enthalpy(msh, p, mats=mats, q_uniform=1.0e6, max_time_s=0.2,
                       phi_target=2.0, record=full)
    strided: dict = {}
    fwd.march_enthalpy(msh, p, mats=mats, q_uniform=1.0e6, max_time_s=0.2,
                       phi_target=2.0, record=strided, record_stride=4)
    total = full["n_recorded_of"]
    assert strided["n_recorded_of"] == total
    assert len(strided["T_steps"]) == len(range(0, total, 4))
    assert strided["T_step_index"] == list(range(0, total, 4))
    assert len(strided["T_steps"]) < len(full["T_steps"])


def test_the_stored_anchors_are_bit_identical_to_the_full_run():
    """Striding must DROP states, never perturb the march. If the anchors
    differed at all, the reverse sweep would replay from a wrong state."""
    msh, p, mats = _tiny_case()
    full: dict = {}
    fwd.march_enthalpy(msh, p, mats=mats, q_uniform=1.0e6, max_time_s=0.2,
                       phi_target=2.0, record=full)
    strided: dict = {}
    fwd.march_enthalpy(msh, p, mats=mats, q_uniform=1.0e6, max_time_s=0.2,
                       phi_target=2.0, record=strided, record_stride=4)
    for j, idx in enumerate(strided["T_step_index"]):
        assert strided["T_steps"][j].tobytes() == full["T_steps"][idx].tobytes()


# --------------------------------------------------------------------------- #
# The scalar series: J(t) without fields
# --------------------------------------------------------------------------- #
def test_a_scalar_callback_records_one_number_per_step():
    """The envelope argmin needs J(t), which is scalars. Storing fields to get
    them is what cost 51 GB."""
    msh, p, mats = _tiny_case()
    rec: dict = {}
    fwd.march_enthalpy(msh, p, mats=mats, q_uniform=1.0e6, max_time_s=0.2,
                       phi_target=2.0, record=rec, record_stride=4,
                       record_scalar_fn=lambda T: (float(T.mean()),
                                                   float(T.max())))
    assert len(rec["scalars"]) == rec["n_recorded_of"]
    assert all(len(row) == 2 for row in rec["scalars"])


def test_the_scalar_series_matches_the_full_field_series():
    """Guards the scalars against being computed at the wrong point in the
    step: they must equal the same function applied to every stored state."""
    msh, p, mats = _tiny_case()
    full: dict = {}
    fwd.march_enthalpy(msh, p, mats=mats, q_uniform=1.0e6, max_time_s=0.2,
                       phi_target=2.0, record=full)
    strided: dict = {}
    fwd.march_enthalpy(msh, p, mats=mats, q_uniform=1.0e6, max_time_s=0.2,
                       phi_target=2.0, record=strided, record_stride=4,
                       record_scalar_fn=lambda T: (float(T.mean()),))
    want = [float(T.mean()) for T in full["T_steps"]]
    got = [row[0] for row in strided["scalars"]]
    assert got == pytest.approx(want, rel=0, abs=0)


# --------------------------------------------------------------------------- #
# THE MEMORY BOUND, stated as a cap
# --------------------------------------------------------------------------- #
TAMPER_STEPS = 18000 * 36           # nsteps * n_sub, measured
TAMPER_NODES = 9866                 # measured
CAP_BYTES = 4.0e9                   # 4 GB on a 32 GB machine, stated not fitted


def predicted_bytes(n_steps: int, n_nodes: int, stride: int) -> int:
    return len(range(0, n_steps, stride)) * n_nodes * 8


def test_the_unstrided_tamper_case_would_blow_the_cap():
    """The regression guard: without a stride this is 51 GB."""
    b = predicted_bytes(TAMPER_STEPS, TAMPER_NODES, 1)
    assert b > 50e9
    assert b > CAP_BYTES


def test_the_strided_tamper_case_is_under_the_stated_cap():
    """At the gradient checkpoint interval the forward stores the anchors the
    reverse sweep was going to build anyway."""
    from solve3d.phase_e import run_tamper as rt
    b = predicted_bytes(TAMPER_STEPS, TAMPER_NODES, rt.CHECKPOINT_INTERVAL)
    assert b < CAP_BYTES, f"{b / 1e9:.2f} GB exceeds the {CAP_BYTES / 1e9} GB cap"


def test_the_driver_opts_into_a_stride_that_meets_the_cap():
    """Not just possible -- actually configured. A cap nothing uses is decor."""
    from solve3d.phase_e import run_tamper as rt
    assert rt.RECORD_STRIDE >= 1
    b = predicted_bytes(TAMPER_STEPS, TAMPER_NODES, rt.RECORD_STRIDE)
    assert b < CAP_BYTES


def test_the_stride_divides_the_gradient_checkpoint_interval():
    """The reverse sweep replays from anchors every `checkpoint_interval`
    steps, so every one of those must BE a stored anchor. If the stride did
    not divide it, the reader would ask for a state that was never kept."""
    from solve3d.phase_e import run_tamper as rt
    assert rt.CHECKPOINT_INTERVAL % rt.RECORD_STRIDE == 0


# --------------------------------------------------------------------------- #
# EQUIVALENCE: bounded recording must not change a single answer
# --------------------------------------------------------------------------- #
def _small_case(record_stride: int):
    """A deliberately small TransientCase, built twice with different striding
    so the two are otherwise identical (same mesh, same seed, same params)."""
    import numpy as np
    from solve3d import adjoint, forward as fwd

    p = fwd.ForwardParams(dt_s=0.5, conv_h=5.0)
    msh = fwd.box_mesh(8)
    mats = fwd.build_materials(msh, fwd.in_part_predicate("circle"), p)
    eqs = adjoint.SteadyEqs(msh, mats, p)
    return adjoint.TransientCase(msh, mats, p, eqs, None, 0.5, 12.0,
                                 record_stride=record_stride)


@pytest.mark.slow
def test_strided_recording_reproduces_the_unstrided_J_trajectory():
    """THE correctness gate. The whole fix rests on J(t) recorded as scalars
    being the SAME J(t) the old path got by re-reading a stored field."""
    import numpy as np

    a, b = _small_case(1), _small_case(4)
    s0 = np.full(a.eqs.part.size, a.p.sigma_doped)
    tr_a, tr_b = a.forward(s0), b.forward(s0)
    assert tr_a.n_steps == tr_b.n_steps
    assert len(tr_b.T_steps) < len(tr_a.T_steps)
    for name in ("symmetric", "asymmetric"):
        a.set_objective(name)
        b.set_objective(name)
        Ja, Jb = a.J_trajectory(tr_a), b.J_trajectory(tr_b)
        assert Ja.shape == Jb.shape
        assert Jb == pytest.approx(Ja, rel=1e-12, abs=1e-18), name
        assert int(np.argmin(Ja)) == int(np.argmin(Jb)), name


@pytest.mark.slow
def test_strided_state_at_reproduces_the_unstrided_read_state():
    """The read state is REPLAYED from an anchor. If the replay drifted, the
    scored temperature field would be subtly wrong while J looked fine."""
    import numpy as np

    a, b = _small_case(1), _small_case(4)
    s0 = np.full(a.eqs.part.size, a.p.sigma_doped)
    tr_a, tr_b = a.forward(s0), b.forward(s0)
    for k in (0, 1, 5, 7, tr_a.n_steps - 1, tr_a.n_steps):
        Ta, Tb = a.state_at(tr_a, k), b.state_at(tr_b, k)
        assert Tb == pytest.approx(Ta, rel=1e-12, abs=1e-12), k


@pytest.mark.slow
def test_a_checkpoint_interval_that_is_not_a_multiple_of_the_stride_is_refused():
    """Fail loudly rather than hand back an anchor that was never stored."""
    import numpy as np

    b = _small_case(4)
    s0 = np.full(b.eqs.part.size, b.p.sigma_doped)
    tr = b.forward(s0)
    b.set_objective("asymmetric")
    with pytest.raises(ValueError, match="not a multiple"):
        b.gradient(s0, tr=tr, read_step=4, checkpoint_interval=6)
