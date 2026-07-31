# EQS-02 shape re-ranking campaign

**Question.** heatr3d's `compute_qrf_3d` default flipped to `qrf_gradient="masked"` in
commit `76eb22e` (EQS-02). `heatr3d_d1_spike/EQS02_IMPACT.md` proved the sigma_T change
is **geometry-dependent in sign** (-23.1 % extruded circle, +10.9 % extruded square at
n = 64; -3.7 % extruded circle at n = 96). A sign that depends on geometry means
**shape rankings cannot be assumed to survive**. This campaign measures which ones do.

Written **before** the runs. Nothing here is a dissertation claim; it is a measurement
of what moved.

---

## 1. Inventory: which heatr3d-derived shape comparisons exist

Searched `geo-prewarp/` and (read-only) `dissertation_materials/analysis-3dfgm/`.

| artifact | tool | shapes | comparative quantity | in scope? |
|---|---|---|---|---|
| `heatr3d_d1_spike/results.json["eqs02_impact"]`, `EQS02_IMPACT.md` | heatr3d 3-D | cylinder, square (+cylinder n=96) | sigma_T, t90, T_max, surface/interior split, surface power fraction | **yes** — this campaign's anchor |
| `analysis-3dfgm/cone_dumbbell_fgm_results_fixed.json` | heatr3d 3-D, n=64 | cone, dumbbell | baseline sigma_T 32.648 / 20.127 C -> *dumbbell ranks more uniform than cone*; FGM deltas -31.9 % / -44.2 % | **yes** |
| `analysis-3dfgm/schedule_summary.csv` + `run_schedule_study.py` | heatr3d 3-D | sphere only | baseline sigma_T 38.32 C, t90 572 s (`constant x1.0`) | **yes** (sphere) |
| `HEATR3D_P0_EDGEWIDTH_REPORT.md` (n=48, 8 runs) | heatr3d 3-D | cone, sphere | sigma_T 32.035 (cone) vs 38.319 (sphere) -> *cone ranks more uniform than sphere*; FGM % vs uniform | **yes** (cone, sphere) |
| `analysis-3dfgm/layerC_shape_gate_metrics.csv` | 2-D adjoint gate | square, lshape, cross | FD-gate rel_err, J — establishes lshape/cross as a studied shape family | shapes carried over; numbers not comparable |
| `perlayer_fgm_pilot_summary.csv` / `perlayer_fgm_pilot_*.json` | **2-D `rfam_eqs_coupled`**, not heatr3d | circle, square, diamond, L_shape | CoV of rotation-averaged Qrf T-proxy (%) — a ranking: circle 66.7 < square 73.9 < diamond 109.2 (A baseline) | **out of scope for re-run** (different tool, Q-proxy not a thermal march) — but it names `diamond` and `L_shape` as ranked shapes, so both are carried into the 3-D set |
| `analysis-3dfgm/method_selection_*` , `run_melt_mtf_doseopt.py`, dose attribution | 2-D / 2.5-D `ui_rms` basis | 18 shapes | `ui_rms*(T_bar-23)` | **out of scope** — 2-D tool, unaffected by EQS-02 (that fix is in `heatr3d.py` only) |

**Consequence for scope.** Only `heatr3d.py`-derived numbers are exposed to EQS-02. The
2-D/2.5-D campaigns (`rfam_eqs_coupled.py`) are **not** touched by commit `76eb22e`.

`diamond` is **not** a `heatr3d.make_geometry` shape. It is constructed here as a local
boolean mask `(|x|+|y| <= diam/2) & full-height`, i.e. the 2-D published diamond
cross-section extruded, and passed in as an array. **`heatr3d.py` is not modified.**

---

## 2. Shape set (the defensible core), one grid

n = 64, L = 60 mm, `Params()` defaults except `phase_update="enthalpy"`.

**Geometry is pinned to the published runs**, so the `legacy` arm is a reproduction
anchor rather than a re-parameterisation. `n_voxels_in_part` was verified to match the
published counts exactly for all five shapes that have one.

| shape | diam / zspan | provenance | part voxels @ n=64 (published) | why it is in the set |
|---|---|---|---|---|
| `cylinder` | 20 mm, full height | `run_3d_study.py` + EQS02_IMPACT | 23040 (23040 ✓) | the published 4-shape row; the "extruded circle" |
| `sphere` | 28 mm | `run_3d_study.py` | 13992 (13992 ✓) | published 4-shape row; schedule study; P0 report |
| `cone` | 24 mm / 30 mm | `run_3d_study.py` | 5492 (5492 ✓) | published 4-shape row; cone/dumbbell FGM; P0 report |
| `dumbbell` | 18 mm / 34 mm | `run_3d_study.py` | 8328 (8328 ✓) | published 4-shape row; cone/dumbbell FGM |
| `square` | 20 mm, full height | EQS02_IMPACT.md | 30976 (30976 ✓) | EQS02_IMPACT anchor |
| `diamond` | 20 mm, full height | local mask (above) | 14080 | named in the 2-D published ranking; sharp-corner counterpart to `square` |
| `lshape` | 20 mm, full height | layerC prism family | 20160 | re-entrant corner |
| `cross` | 20 mm, full height | layerC prism family | 21760 | 4 re-entrant corners |

**The published table this re-ranks** is `analysis-3dfgm/study_summary.csv`
(n=64, baseline rows): sigma_T `dumbbell 20.13 < cylinder 26.34 < cone 32.65 <
sphere 32.92`; t90 `cylinder 436.1 < sphere 541.8 < dumbbell 856.2 < cone 1276.4`.
Caveat on the anchor: that study ran `Params()` defaults, i.e.
`phase_update="apparent_cp"`; this campaign uses `"enthalpy"` (the S1 energy-conserving
scheme) on **both** arms, so the legacy arm is expected to sit near, not on, the
published values (EQS02_IMPACT's enthalpy cylinder reads 26.13 C vs the published
26.34 C). The re-ranking claim is legacy-vs-masked **within this campaign**.

Two classes on purpose: **full-height prisms** (z-invariant V — the EQS02_IMPACT
regime) and **z-varying solids** (sphere/cone/dumbbell — where the artifact also acts
on the top/bottom caps, untested until now). They are not the same physical size;
`power_density_w_per_m3` is a per-volume reference so heating rate per volume is
size-independent, but cross-class size differences remain a stated limit on the
cross-shape ranking.

## 3. Method — both arms from ONE EQS solve

Per shape:

1. `gamma = build_gamma(part, p)`; `V = solve_eqs_3d(gamma, grid, p)` — **once**.
2. `q_legacy = compute_qrf_3d(V, gamma, grid, p, doped=part, qrf_gradient="legacy")`
   `q_masked = compute_qrf_3d(V, gamma, grid, p, doped=part, qrf_gradient="masked")`
   Same `V`, same `gamma`, same renormalization to
   `power_density_w_per_m3 * doped_volume`. **Identical total absorbed power** in both
   arms by construction; recorded as `power_identity.rel_diff` and required < 1e-12.
3. Two thermal marches, `run(grid, part, p_th, qrf_override=Q, max_time_s=1500,
   phi_target=0.90)` with `p_th = dataclasses.replace(p, phase_update="enthalpy")`.
   `qrf_override` **skips the EQS solve**, which is exactly why one solve serves both
   arms. The override is used verbatim (no re-renormalization), so the arms differ
   only in the spatial distribution of an identical total power.

`heatr3d.py` is **imported read-only**. No file in `dissertation_materials/` is written.

### Metrics per shape/arm
`sigma_T` (**3-D**: `std(T_phi90)` over part voxels, `heatr3d.Result.sigma_T` — never
compared to any 2-D `ui_rms*(T_bar-23)` number), `t_phi90_s`, `T_max_c`, `T_mean_c`,
interior/surface mean and std T, `surface_minus_interior_mean_c`, and on the Q field
`power_fraction_in_surface_band`, `max_over_mean`, `cv`.
"Surface" = within 1.5 voxels of the part boundary (the EQS02_IMPACT/Task-2 rule).

### Standing gates, reported on every march
`energy_residual_frac`, `clamp_bound`, `cfl_violated`, `n_substeps_used`, `reached`
(phi_bar = 0.90 within 1500 s). A march that fails a gate is reported as failed, not
dropped.

### Ranking analysis
For each metric, rank the 8 shapes under each arm, report Spearman rho and the explicit
list of **pair inversions** (shape pairs whose order swaps). A ranking "flips" if at
least one adjacent pair inverts.

## 4. Runtime estimate (stated before running)

Measured basis, `results.json["eqs02_impact"]` on this machine, n = 64: EQS solve
307 s (square) / 570 s (cylinder); thermal wall = **0.27 s per simulated second**
(108.7 s / 397.0 s and 87.5 s / 327.7 s — the per-step cost is set by the n^3 domain,
not the part volume). Peak RSS 0.64 GB at n=64, 1.86 GB at n=96.

| item | estimate |
|---|---|
| 8 EQS solves @ ~450 s | 3600 s |
| 16 thermal marches, 0.27 x t90 (cone/dumbbell/sphere are the long ones) | ~2240 s |
| **serial total, uncontended** | **~1.6 h** |
| machine is currently ~10/12 cores busy with another agent's `adjoint2d` gates -> assume 2x | ~3.2 h serial |
| run 3 shapes in parallel, `OMP_NUM_THREADS=1` | **~1.0-1.4 h wall** |
| n = 96 spot check, top-2 ranking-flip shapes only (~1000 s each, contended ~2000 s) | +~1.1 h |
| **total** | **~2.1-2.5 h** — under the ~4 h budget, so the full 8-shape set is kept |

If the wall clock exceeds the estimate materially, the n = 96 spot check is the part
that gets cut, and that cut will be stated in the report.

## 5. Files

- `rank_utils.py` + `test_rank_utils.py` — pure ranking/flip logic (TDD'd).
- `run_rerank.py` — one shape, both arms -> `shards/<shape>_n<N>.json`.
- `build_report.py` — merge shards -> `rerank_results.json` + `RERANK_REPORT.md`.
