# Stage B (B2): ceiling-coupled dopant solve at fixed 0.40x

Penalty solve J = J_shape + mu*(KS_peak - 250 C)_+^2 with the rho+T density co-state gradient (B1, FD-gated 3.0e-9; mutation bites). The TRUE end-state peak on a mesh HOLD-OUT arbitrates is_shippable.

- drive: 0.4x (636619.8 W/m^3), FIXED
- ceiling: 250.0 C (thermal_config.json, shared with the Studio lane)
- solve mesh: 11850 design cells; solve-mesh uniform peak ~245 C vs 9868-node arbiter ~240 C (peak is a single-cell MAX, mesh-sensitive); the hold-out is the truth for is_shippable
- solved-map KS peak (solve mesh): 248.45 C
- TRUE hold-out peak (arbiter, 9868 in-part nodes): 250.69 C  (margin -0.69 C, feasible=False)
- is_shippable: False  (reason: over_ceiling_true_peak, over by 0.69 C)
- honest-null verdict: EMITTED BUT SPURIOUS -- the code stamped
  `no_feasible_dopant_at_this_drive`, which is FALSE: the uniform map at 0.40x is
  239.99 C (feasible, Task 4). A feasible dopant DOES exist; the penalty solve
  traded toward shape and landed 0.69 C over. Correct read: shaped map marginally
  over ceiling via the KS-from-below gap; feasible region exists. The honest_null
  logic is fixed in Stage B3 (it must fire only if the peak-minimizing/uniform map
  is over ceiling, which at 0.40x it is not).

## mu-continuation
- mu=1e+02: 12 evals, best_J=1.1372e-06, status=budget_exhausted
- mu=1e+03: 12 evals, best_J=8.8699e-07, status=budget_exhausted
- mu=1e+04: 12 evals, best_J=7.6441e-07, status=budget_exhausted

## reading
is_shippable = FD-gated gradient (B1 launch_ok) AND _march-vs-production fidelity AND true hold-out peak <= ceiling. The KS aggregate is the smooth gradient proxy ONLY; it never decides shippability. At fixed 0.40x, forcing the peak under 250 C costs shape vs the phase-2 unconstrained (infeasible, 251.15 C) map -- the honest price of feasibility. More shape at full feasibility needs a drive backoff (B4).

wall_total_s = 1312.8
