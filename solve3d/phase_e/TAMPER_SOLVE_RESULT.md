# Tamper.stl Phase-E Solve: Result

Date: 2026-08-07
Part: Tamper.stl (real Grade-and-Print upload, feb850ec), 2436 facets.
Chamber: grown to L_chamber = 0.086 m (ch086) so the part is meltable (Matt's
grow-the-chamber decision). Mesh: 54495 cells, 20106 in-part, 10221 nodes.
Solve: solve3d Phase-E asymmetric dense-iff-in-bounds adjoint, budget 12 gradient
evaluations, n_sub = 33, wall ~17 h (per-eval grew 3747 -> 4963 s in later evals).

## Conclusion

The solved dopant map BEATS the uniform dopant decisively, reversing the earlier
"Tamper performed horribly" Grade-and-Print failure. That failure was the legacy
heatr3d_native_inversion rung inverting a saturated rho_final (TAMPER_DIAGNOSIS.md,
b028958); this is the real solve3d Phase-E adjoint, and it wins on BOTH terms of
the dense-iff-in-bounds objective at once.

## Solved vs uniform (arm reads at optimized stop time)

| Metric | uniform_baseline | solve_filter_only | Change |
|---|---|---|---|
| Objective J (asymmetric) | 4.5549e-6 | 2.5029e-6 | **-45.05%** |
| Out-of-bounds melt penalty J_oob | 5.111e-7 | 0.929e-7 | -81.8% (5.5x lower) |
| Out-of-part melt fraction of part | 0.01434 (1.43%) | 0.00487 (0.49%) | 2.94x less spill |
| In-bounds under-density deficit J | 4.0438e-6 | 2.4099e-6 | -40.4% |
| In-bounds below-floor fraction | 0.9396 (94.0%) | 0.7225 (72.2%) | much more reaches floor |
| Mean melt fraction (part) | 0.0975 | 0.3384 | 3.47x more densification |
| Peak T | 201.5 C | 208.4 C | both under 250 ceiling |
| sigma_T diagnostic | 27.94 C | 27.50 C | marginal (not the story) |
| Optimized stop time | 264.5 s | 359.4 s | longer bake |

The win is exactly the objective Matt defined ("every part fully dense IF AND ONLY
IF within nominal shape bounds"): the solved map cut out-of-bounds melt ~3x AND
raised in-bounds densification ~3.5x. sigma_T barely moved and is not the metric
of merit here.

## Honest scope

The solved map is a large improvement but NOT a fully dense part: 72% of the
in-bounds region is still below the density floor (mean melt fraction 34%). The
dopant SHAPE relocates and improves heating but cannot, at this drive, fully
densify the part - the DRIVE/exposure lever is needed for that (the conservation
argument, and exactly what thermal-ceiling Stage A supplies). So Tamper motivates
the joint (dopant + drive) solve: the dopant shapes within bounds, the drive
carries the part to full density under the ceiling.

## Provenance

Result: solve3d/phase_e/results/phase_e_tamper.json (arms: _mesh,
uniform_baseline, solve_filter_only). Checkpoint:
ckpt_tamper_solve_filter_only.npz (eval 12/12, resumable). Solve driver:
solve3d/phase_e/run_tamper.py --budget 12 --stage solve. J trajectory:
4.5549e-6 (eval 1) -> 2.5029e-6 (eval 12), monotone descending, gradients finite
throughout (the n_sub=33 adjoint bug fixed earlier; FD-gated).
