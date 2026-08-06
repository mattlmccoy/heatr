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


# --------------------------------------------------------------------------- #
# Step index -> TIME. The substep trap.
# --------------------------------------------------------------------------- #
def test_the_trajectory_records_its_substep_count():
    """Without it, a step index cannot be converted to a time."""
    import numpy as np
    b = _small_case(1)
    s0 = np.full(b.eqs.part.size, b.p.sigma_doped)
    tr = b.forward(s0)
    assert tr.n_sub >= 1
    assert tr.n_steps % tr.n_sub == 0


def test_step_to_time_uses_the_substep_not_the_sample_step():
    """THE BUG THIS PINS. The trajectory is indexed by SUBSTEP, so
    `k * p.dt_s` overstates the time by n_sub. On the Tamper (n_sub 33) the
    uniform baseline reported t_stop_s = 8729.65 s for a run whose horizon was
    900 s -- a number that is not merely wrong but impossible, and it was
    inherited from phase_e/run.py where n_sub is 1 and the bug is invisible.
    """
    import numpy as np
    from solve3d.phase_e import run_tamper as rt

    b = _small_case(1)
    s0 = np.full(b.eqs.part.size, b.p.sigma_doped)
    tr = b.forward(s0)
    t_end = rt.step_to_time_s(tr, tr.n_steps, b.p)
    assert t_end == pytest.approx(b.max_time_s, rel=1e-9)
    assert rt.step_to_time_s(tr, 0, b.p) == 0.0
    # and it must never exceed the horizon for any valid index
    for k in (0, 1, tr.n_steps // 2, tr.n_steps):
        assert 0.0 <= rt.step_to_time_s(tr, k, b.p) <= b.max_time_s + 1e-12


def test_the_tamper_substep_case_would_have_been_reported_33x_too_long():
    """Regression guard with the real numbers."""
    from solve3d.phase_e import run_tamper as rt

    class _T:
        n_sub = 33
    class _P:
        dt_s = 0.05
    naive = 174593 * 0.05
    fixed = rt.step_to_time_s(_T(), 174593, _P())
    assert naive == pytest.approx(8729.65, abs=0.01)
    assert fixed == pytest.approx(264.535, abs=0.01)
    assert fixed < 900.0


# --------------------------------------------------------------------------- #
# THE SUBSTEP ADJOINT BUG: the reverse sweep must step at dt_sub, not dt_s
# --------------------------------------------------------------------------- #
def _substep_case(stride: int = 1, n: int = 20, dt_s: float = 30.0,
                  max_t: float = 1200.0):
    """A case with n_sub > 1. Every prior solve3d campaign ran at n_sub = 1
    (Phase A/B/C/E anchors, pyramid, cube), which is why this went unnoticed."""
    import numpy as np
    from solve3d import adjoint, forward as fwd

    p = fwd.ForwardParams(dt_s=dt_s, conv_h=5.0)
    msh = fwd.box_mesh(n)
    mats = fwd.build_materials(msh, fwd.in_part_predicate("circle"), p)
    eqs = adjoint.SteadyEqs(msh, mats, p)
    return adjoint.TransientCase(msh, mats, p, eqs, None, dt_s, max_t,
                                 record_stride=stride)


@pytest.mark.slow
def test_the_substep_case_actually_substeps():
    """Guards the fixture: without n_sub > 1 the rest of this section is vacuous."""
    import numpy as np
    tc = _substep_case()
    s0 = np.full(tc.eqs.part.size, tc.p.sigma_doped)
    tr = tc.forward(s0)
    assert tr.n_sub > 1


@pytest.mark.slow
def test_the_gradient_is_finite_and_sane_when_the_march_substeps():
    """THE BUG. `_step_cache` and the reverse sweep used p.dt_s while the
    forward marched at dt_sub = p.dt_s / n_sub, so the adjoint stepped n_sub
    times too far and went explicitly unstable.

    Measured before the fix, same case, same read step:
        stride 1, ci None   |g| = 3.87e+42
        stride 1, ci 10     |g| = 5.29e+13
        stride 5, ci 10     |g| = 2.58e+12
    On the Tamper (n_sub 33, 594000 steps) it overflowed to NaN, and the solve
    burned ten hours on eight evaluations whose gradient was NaN every time.
    """
    import numpy as np
    tc = _substep_case()
    s0 = np.full(tc.eqs.part.size, tc.p.sigma_doped)
    tr = tc.forward(s0)
    tc.set_objective("asymmetric")
    k = int(np.argmin(tc.J_trajectory(tr)))
    g, _ = tc.gradient(s0, tr=tr, read_step=k, checkpoint_interval=None)
    assert np.isfinite(g).all()
    # a sane objective gradient here is O(1e-3) or smaller; 1e42 is not "large"
    assert np.linalg.norm(g) < 1.0e3, f"|g| = {np.linalg.norm(g):.3e}"


@pytest.mark.slow
def test_the_substepped_gradient_is_independent_of_the_checkpoint_interval():
    """Checkpointing only changes WHERE states come from, never the answer.
    Before the fix these differed by thirty orders of magnitude, which is what
    made the instability unmistakable rather than merely suspicious."""
    import numpy as np
    tc = _substep_case()
    s0 = np.full(tc.eqs.part.size, tc.p.sigma_doped)
    tr = tc.forward(s0)
    tc.set_objective("asymmetric")
    k = int(np.argmin(tc.J_trajectory(tr)))
    ref, _ = tc.gradient(s0, tr=tr, read_step=k, checkpoint_interval=None)
    got, _ = tc.gradient(s0, tr=tr, read_step=k, checkpoint_interval=10)
    assert got == pytest.approx(ref, rel=1e-9, abs=1e-30)


@pytest.mark.slow
def test_the_substepped_gradient_is_independent_of_the_recording_stride():
    """And the anchor path must agree with store-everything."""
    import numpy as np
    a, b = _substep_case(stride=1), _substep_case(stride=5)
    s0 = np.full(a.eqs.part.size, a.p.sigma_doped)
    tr_a, tr_b = a.forward(s0), b.forward(s0)
    for tc in (a, b):
        tc.set_objective("asymmetric")
    k = int(np.argmin(a.J_trajectory(tr_a)))
    ga, _ = a.gradient(s0, tr=tr_a, read_step=k, checkpoint_interval=10)
    gb, _ = b.gradient(s0, tr=tr_b, read_step=k, checkpoint_interval=10)
    assert gb == pytest.approx(ga, rel=1e-9, abs=1e-30)


@pytest.mark.slow
def test_the_substepped_gradient_passes_a_finite_difference_check():
    """FINITE and CONSISTENT is not the same as CORRECT. The dt_step fix
    changes the adjoint's step, so the gradient has to be re-gated against
    finite differences on a substepped case -- the configuration no existing FD
    gate covers (they all run at n_sub = 1).

    Central differences on a directional derivative at a FIXED read step, so
    the envelope argmin cannot move between the perturbed evaluations and
    contaminate the comparison.
    """
    import numpy as np
    tc = _substep_case()
    s0 = np.full(tc.eqs.part.size, tc.p.sigma_doped)
    tr = tc.forward(s0)
    tc.set_objective("asymmetric")
    k = int(np.argmin(tc.J_trajectory(tr)))
    g, _ = tc.gradient(s0, tr=tr, read_step=k, checkpoint_interval=None)

    rng = np.random.default_rng(0)
    d = rng.standard_normal(s0.size)
    d /= np.linalg.norm(d)
    analytic = float(np.dot(g, d))

    def J_at(s):
        t = tc.forward(s)
        return float(tc.J_trajectory(t)[k])

    # h = 1e-5 relative, NOT larger. This case runs with the per-step
    # temperature cap ACTIVE (clamp_bound True), so the objective has a kink
    # and a coarse central difference straddles it: the same check reads
    # 2.2e-01 at h=1e-2, 1.2e-01 at 1e-3, 5.0e-02 at 1e-4 and 8.9e-08 at 1e-5.
    # That is the documented subgradient regime, not gradient error -- the
    # convergence AS h SHRINKS is the evidence, which is why the sweep is
    # recorded here rather than a single lucky step size.
    h = 1e-5 * float(np.abs(s0).mean())
    fd = (J_at(s0 + h * d) - J_at(s0 - h * d)) / (2.0 * h)
    rel = abs(analytic - fd) / max(abs(fd), 1e-30)
    assert rel < 1e-6, (
        f"substepped gradient fails FD: analytic {analytic:.6e} vs "
        f"fd {fd:.6e}, rel {rel:.3e}")


@pytest.mark.slow
def test_the_gradient_depends_on_dt_sub_not_on_the_sample_step():
    """THE CLEANEST STATEMENT OF THE BUG, and the sharpest regression guard.

    dt_s=30 with n_sub=6 and dt_s=15 with n_sub=3 both march at dt_sub = 5.0 s.
    The physics is identical, so the gradient must be too. Before the fix the
    adjoint used p.dt_s, so these two differed by a factor of two in the
    adjoint's step and gave different answers; now they agree exactly.
    """
    import numpy as np

    def grad_for(dt_s):
        tc = _substep_case(dt_s=dt_s)
        s0 = np.full(tc.eqs.part.size, tc.p.sigma_doped)
        tr = tc.forward(s0)
        assert tr.n_sub > 1
        assert abs(tc.dt_step - 5.0) < 1e-12, tc.dt_step
        tc.set_objective("asymmetric")
        k = int(np.argmin(tc.J_trajectory(tr)))
        g, _ = tc.gradient(s0, tr=tr, read_step=k, checkpoint_interval=None)
        return g

    a, b = grad_for(30.0), grad_for(15.0)
    assert b == pytest.approx(a, rel=1e-12, abs=1e-30)


# --------------------------------------------------------------------------- #
# The guard: ten hours of NaN must be structurally impossible
# --------------------------------------------------------------------------- #
def test_the_guard_refuses_a_nan_gradient():
    import numpy as np
    from solve3d.phase_e import run_tamper as rt

    g = np.ones(5)
    g[2] = np.nan
    with pytest.raises(rt.NonFiniteGradientError) as e:
        rt.assert_finite_first_eval(1.0e-6, g)
    assert "NaN" in str(e.value)
    assert "Refusing to run the optimizer" in str(e.value)


def test_the_guard_refuses_an_inf_gradient_and_a_nan_objective():
    import numpy as np
    from solve3d.phase_e import run_tamper as rt

    with pytest.raises(rt.NonFiniteGradientError):
        rt.assert_finite_first_eval(1.0e-6, np.array([1.0, np.inf, 2.0]))
    with pytest.raises(rt.NonFiniteGradientError):
        rt.assert_finite_first_eval(float("nan"), np.ones(3))


def test_the_guard_passes_a_healthy_gradient():
    """Mutation check: a guard that refuses everything is not a guard."""
    import numpy as np
    from solve3d.phase_e import run_tamper as rt

    rt.assert_finite_first_eval(1.0e-6, np.array([1e-3, -2e-3, 5e-4]))


def test_the_guard_names_what_was_wrong():
    """The message must say WHICH quantity failed, so the next person does not
    have to re-derive it from a checkpoint."""
    import numpy as np
    from solve3d.phase_e import run_tamper as rt

    with pytest.raises(rt.NonFiniteGradientError) as e:
        rt.assert_finite_first_eval(1.0e-6, np.full(4, np.nan))
    assert "4 NaN" in str(e.value)
