"""FIRST-ORDER drive-feasibility estimate from the saved end-state field.

Model (pre-phase-change linear scaling of temperature rise with drive):
    T_node(a) = Tamb + a*(T_read_node - Tamb)
    phi_node(a) = clip((T_node - 175)/10, 0, 1)     # melt band 175-185 C
    peak(a)     = Tamb + a*(peak1 - Tamb)            # trajectory rim peak
below-floor = vol-wt fraction of in-part nodes with phi < 0.85.

CAVEATS (why this is a PRIOR, not the verdict):
  - ignores latent heat -> real fusion is SLOWER -> real needs MORE drive
    (this estimate is OPTIMISTIC about feasibility).
  - ignores stop-time re-optimization and conduction nonlinearity.
  - peak1=282 is a single-cell trajectory max on sub-mm slivers; if that is a
    mesh/faceting artifact the true peak(a) is lower (feasibility better).
The real answer is the transient drive sweep (queued heavy).
"""
import numpy as np

TAMB, TPC_LO, FLOOR = 50.0, 175.0, 0.85
PEAK1 = 281.88

d = np.load("solve3d/phase_e/results/field_tamper_solve_filter_only.npz", allow_pickle=True)
T = d["T_read"]; chi = d["chi"]; vol = d["vol"]
part = chi > 0.5
Tp = T[part]; wp = (vol * chi)[part]

print(f"{'drive':>6} {'qbar/pd':>8} {'peak_est':>9} {'over250':>8} "
      f"{'belowfloor':>11} {'meanphi':>8}")
for a in [0.4, 0.6, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]:
    Ta = TAMB + a * (Tp - TAMB)
    phi = np.clip((Ta - TPC_LO) / 10.0, 0.0, 1.0)
    below = wp[phi < FLOOR].sum() / wp.sum()
    peak = TAMB + a * (PEAK1 - TAMB)
    mphi = np.average(phi, weights=wp)
    print(f"{a:6.2f} {a:8.2f} {peak:8.1f}C {str(peak>250):>8} "
          f"{below:11.3f} {mphi:8.3f}")

# smallest drive that gets below-floor under 15%, and its peak
print("\nfeasibility scan:")
for target in (0.15, 0.10, 0.05):
    a_ok = None
    for a in np.linspace(0.4, 4.0, 361):
        Ta = TAMB + a * (Tp - TAMB)
        phi = np.clip((Ta - TPC_LO) / 10.0, 0.0, 1.0)
        if wp[phi < FLOOR].sum() / wp.sum() <= target:
            a_ok = a; break
    if a_ok:
        peak = TAMB + a_ok * (PEAK1 - TAMB)
        print(f"  below-floor<={target:.2f} needs drive>={a_ok:.2f}x -> rim peak_est {peak:.0f}C "
              f"({'OVER' if peak>250 else 'under'} 250)")
    else:
        print(f"  below-floor<={target:.2f}: not reached even at 4x drive")
