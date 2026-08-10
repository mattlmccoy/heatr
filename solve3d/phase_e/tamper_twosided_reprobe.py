"""LIGHT re-probe: can two-sided grading (cap max_sat) flatten the Tamper's ~3x
radial power gradient enough that a single drive densifies the core while the rim
stays under the ceiling?

Method (EQS only, no march): fixed-point flatten the deposited power toward its
mean under a per-node saturation cap -- sat_i *= clip(qbar/q_i), re-solve the EQS
(|E|^2 shifts as sigma shifts), iterate. Compare the achievable rim/core Q ratio
and peak/mean for one-sided (cap 1.0) vs two-sided (cap 2.0) vs the uniform start.

FEASIBILITY THRESHOLD. Fusing the core needs a rise of ~133 K (50->183.5 C floor);
the rim must stay under ~200 K rise (50->250 ceiling). To first order T-rise
tracks local deposited power, so a single drive is feasible only if the
rim/core power ratio < 200/133 ~= 1.5. This is the crisp number the flattening
must beat. (The TRUE peak is a transient march quantity; this EQS ratio is the
steady actuator-authority proxy the heavy solve confirms.)
"""
import numpy as np
from solve3d.phase_e import run_tamper as rt

CORE_R_MM, RIM_R_MM = 10.0, 12.0
FEAS_RATIO = 200.0 / 133.5


def flatten(tc, part, vol, max_sat, iters=6):
    """Fixed-point de-noise of Q toward its mean under sat in [0, max_sat]."""
    n = part.size
    sat = np.ones(n)
    for _ in range(iters):
        st = tc.eqs.forward(tc.design_to_sigma(sat))
        q = st.q[part]
        qbar = np.average(q, weights=vol)
        sat = np.clip(sat * (qbar / np.maximum(q, qbar * 1e-3)) ** 0.5, 0.0, max_sat)
    st = tc.eqs.forward(tc.design_to_sigma(sat))
    return st.q[part], sat


def ratios(tag, q, vol, r):
    qbar = np.average(q, weights=vol)
    core = r < CORE_R_MM
    rim = r >= RIM_R_MM
    qc = np.average(q[core], weights=vol[core]) / qbar
    qr = np.average(q[rim], weights=vol[rim]) / qbar
    peak = q.max() / qbar
    print(f"  {tag:16s} core={qc:.2f}x  rim={qr:.2f}x  rim/core={qr/qc:5.2f}  "
          f"peak/mean={peak:5.1f}x  {'FEASIBLE' if qr/qc < FEAS_RATIO else 'infeasible'}"
          f" (need rim/core < {FEAS_RATIO:.2f})")
    return qr / qc, peak


def main():
    tc, info = rt.build_case()
    part = tc.eqs.part
    vol = tc.eqs.vol[part]
    cent = rt._part_centroids(tc)
    r = np.hypot(cent[:, 0], cent[:, 1]) * 1e3

    print(f"Tamper two-sided re-probe (EQS deposited-power ratio proxy)\n"
          f"n_part={part.size}  feasibility needs rim/core < {FEAS_RATIO:.2f}\n")

    qU = tc.eqs.forward(tc.design_to_sigma(np.ones(part.size))).q[part]
    print("actuator authority (rim/core power ratio):")
    ratios("uniform s=1", qU, vol, r)

    q1, sat1 = flatten(tc, part, vol, max_sat=1.0)
    r1, _ = ratios("one-sided cap1.0", q1, vol, r)

    q2, sat2 = flatten(tc, part, vol, max_sat=2.0)
    r2, p2 = ratios("two-sided cap2.0", q2, vol, r)

    q3, sat3 = flatten(tc, part, vol, max_sat=3.0)
    ratios("two-sided cap3.0", q3, vol, r)

    print(f"\ntwo-sided(2.0) sat map: min {sat2.min():.2f} mean "
          f"{np.average(sat2,weights=vol):.2f} max {sat2.max():.2f}; "
          f"core mean sat {np.average(sat2[r<CORE_R_MM],weights=vol[r<CORE_R_MM]):.2f} "
          f"(boosted), rim mean sat {np.average(sat2[r>=RIM_R_MM],weights=vol[r>=RIM_R_MM]):.2f} (pulled)")

    # provisional drive: linear peak model with the flattened peak/mean
    # one-sided reaches rim/core ~ r1; two-sided ~ r2. Report the ratio drop.
    print(f"\nrim/core ratio: uniform {ratios.__defaults__ if False else ''}"
          f" 3.0 -> one-sided {r1:.2f} -> two-sided(2.0) {r2:.2f}")
    print(f"feasibility threshold rim/core < {FEAS_RATIO:.2f}: "
          f"{'OPENS a window' if r2 < FEAS_RATIO else 'still NOT met by two-sided alone (needs Stage C dwell)'}")


if __name__ == "__main__":
    main()
