# engine_speed — fast thermal march for heatr3d

**Status: bit-identity gate PASSED (8/8 cases, exact). Speed target NOT met —
5.9x at n=48 and 6.3x at n=96 against a 10x bar.**

PROTOTYPE for the graduation lane. `heatr3d.py` is not modified, not
monkeypatched, and not imported from `heatr3d_s2/`. Nothing here is wired into
the Studio.

All numbers below are quoted from recorded JSON
(`gate_results_n32.json`, `bench_results.json`), not transcribed by hand.

---

## 1. What was built

| File | Role |
|---|---|
| `march_fast.py` | drop-in `heatr3d.run` replacement over a proven envelope; calls heatr3d's OWN `build_gamma`/`solve_eqs_3d`/`compute_qrf_3d` for the drive |
| `kernels.py` | five single-threaded `@njit(cache=True)` kernels: `props_kernel`, `faces_kernel`, `step_kernel`, `compress_part`, `densify_kernel` |
| `cases.py` | the 8 gate cases, including two guard-TRIGGERING ones |
| `gate.py` | parity harness; writes `gate_results_n32.json` |
| `bench.py` | march-only benchmark; writes `bench_results.json` |
| `profile_march.py` | per-stage profile |
| `tests/test_parity_gate.py` | red-first gate test (12 tests, all passing) |

Supported envelope (anything else raises `UnsupportedConfig`, it does not
silently run different physics): `power_schedule=None`, no heatsink field, no
powder-loss BC, no eps perturbation, `premix_frac=0`, `edge_width_m=0`,
`eqs_update_interval_s=0`, `phase_update` in `{enthalpy, apparent_cp}`.

Reproduced exactly: THM-03 powder-bed CFL substepping, THM-01 per-step dT cap,
THM-02 temp_min/temp_max clamp and the `clamp_bound` latch, the S1 energy audit,
`T0_override`, `qrf_override`, and the melt-onset / `stop_mean_rho` break
semantics including which substep appends to `phi_hist`.

---

## 2. Accuracy gate — PASSED, exact bit-identity

`gate_results_n32.json`, n=32, numba 0.66.0, numpy 2.2.6, floor_rtol 1e-16.
Compared per case: `T_final`, `T_phi90`, `phi_final`, `Qrf`, `rho_final`,
`part`, `phi_hist`, `sigma_T`, `T_max_c`, `t_phi90_s`, `exposure_s`, the four
energy-audit scalars, `reached`, `clamp_bound`, `n_substeps_used`,
`cfl_violated`, `n_eqs_solves`, `n_eqs_resolves_skipped`.

| case | steps | bit-identical | max rel dev | guard fired | energy_residual_frac (both) |
|---|---|---|---|---|---|
| uniform_cube | 1200 | yes | 0.0e+00 | — | +1.4985037078256556e-14 |
| graded_sat_cube | 1200 | yes | 0.0e+00 | — | +1.2846506400718772e-14 |
| tube (hollow cylinder) | 1200 | yes | 0.0e+00 | — | −1.776967620100925e-14 |
| **cfl_substep** | 20 | yes | 0.0e+00 | **yes, n_sub 3 vs 3** | −1.4051857431013536e-15 |
| **clamp_hot** | 285 | yes | 0.0e+00 | **yes, clamp_bound True vs True** | +0.24600319577399585 |
| melt_onset_break | 226 | yes | 0.0e+00 | — | +5.423896269342888e-17 |
| apparent_cp | 1200 | yes | 0.0e+00 | — | +1.60682689620542e-14 |
| hot_top_convection | 1200 | yes | 0.0e+00 | — | +1.5667672021780775e-14 |

`all_pass: true`, `all_bit_identical: true`. **Every max_rel_dev is exactly
0.0** — the S1 1e-16 fallback floor was never needed, and `FLOOR_RTOL` was never
widened. The energy audit matches to the same standard (identical bits, so the
`clamp_hot` +0.246 residual — real energy destroyed by the temp_max clamp — is
reproduced rather than hidden).

### Guard-triggering cases (per the adoption gate)

* **CFL / THM-03**: `dt_s = 3.0 s` at n=32 exceeds `CFL_SAFETY *
  dt_stable_thermal = 1.406 s`, so `cfl_substeps` returns 3. Asserted to fire
  in BOTH engines (`test_cfl_guard_actually_fires`); `n_substeps_used` is 3 in
  both and the fields still match exactly.
* **Clamp / THM-01+02**: 100x absorbed-power density drives the field into the
  600 C ceiling; `clamp_bound` is True in both and 2.47 % of cells clip per step
  at the peak. Asserted to fire in BOTH (`test_temperature_clamp_actually_fires`).

### Branch coverage added after the first pass

The first 5-case suite passed while leaving four paths dormant, so four cases
were added: `melt_onset_break` (the `densify=False` melt-onset read, `t_phi90_s`,
append-then-break exit), `apparent_cp` (the other phase-update branch),
`hot_top_convection` (a `T0_override` hot band on the y_max plane, which raises
`energy_loss_j` from ~5e-4 J to 207.3 J so the convection sink is actually
exercised), and the hollow `tube` (an internal mask boundary). All bit-identical.

### One named behavioural deviation

`max |dT_raw|` — used ONLY in the THM-01 warning text, never in a `Result`
field — is a running scalar max in the kernel. If `dT_raw` were ever NaN,
`np.max` in heatr3d would report NaN while the kernel reports the largest
non-NaN magnitude. No gated field depends on it.

---

## 3. Benchmark — march loop only

`bench_results.json`. Both engines driven with the SAME frozen `Q_rf` via
`qrf_override`, so no EQS solve occurs on either side and the number is the
march loop and nothing else.

**Threads: 1.** `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1`,
and `numba.config.NUMBA_NUM_THREADS == 1` is recorded in the JSON. The kernels
contain no `prange` and no `parallel=True`, so they are single-threaded
regardless of the environment; there is no parallel mode and therefore nothing
that needs a bigger scheduler slot.

| case | n | cells | steps | Q_rf | heatr3d ms/step | march_fast ms/step | speedup | T & rho bit-identical |
|---|---|---|---|---|---|---|---|---|
| n48 | 48 | 110 592 | 1200 | real EQS | 9.6926 | 1.6406 | **5.91x** | yes |
| n96 | 96 | 884 736 | 100 | synthetic | 84.6638 | 13.4599 | **6.29x** | yes |
| n48 (synthetic Q_rf) | 48 | 110 592 | 400 | synthetic | 9.6434 | 1.5899 | 6.07x | yes |

The speedup is **not** a small-grid artifact: it is 5.9x at n=48 and 6.3x at
n=96, i.e. it holds (slightly improves) at 8x the cell count.

*Q_rf substitution is evidenced, not assumed.* The synthetic uniform in-part
drive (renormalised to the same absorbed-power target) changes march ms/step at
n=48 by **3.09 %** versus the real EQS field, so using it at n=96 — where a real
EQS solve is ~322 s — does not distort the comparison.

**Projected n=96, 30 000-step march:** heatr3d **2540 s** → march_fast **404 s**.

numba compile cost is one-time and excluded from the timed region:
`numba_warmup_s` = 0.11 s on a cold `cache=True` load, 0.0003 s warm.

**Measurement honesty.** The machine carried one competing `solve3d.phase_e.run`
at ~285 % CPU throughout (load average 9.8–11.6, within the <20 rule; 1 heavy
solve, so the <2 rule held). Both engines were timed back-to-back under the same
conditions, so the RATIO is sound, but the absolute ms/step is inflated —
heatr3d measured 9.69 ms/step here against the ~8 ms/step in the quoted clean
profile. A quiet-machine re-run would likely move both numbers down together.

---

## 4. Where the time goes, and why 10x was not reached

`profile_march.py`, n=48, single-threaded, per substep:

| stage | ms/step |
|---|---|
| props_kernel | 0.13 |
| faces_kernel | 0.33 |
| **step_kernel** | **1.10** |
| qconv.sum + esens.sum | 0.04 |
| compress_part + part reductions | 0.02 |
| densify_kernel | 0.10 |
| total | ~1.7 |

`step_kernel` is ~65 % of the march. Three measurements localise the ceiling:

1. **It is not memory-bound.** A kernel that touches exactly the same arrays and
   does trivial arithmetic runs in **0.219 ms**; single-core copy bandwidth
   measured **43 GB/s**. `step_kernel` is 5x above its own memory floor.
2. **It is not division-bound in the way first assumed.** Divisions are real
   cost, but the block-by-block bisection shows the expense is spread: faces
   0.182, +convection/source 0.156, +enthalpy 0.372, +clamp counters 0.423,
   +esens/phi 0.225.
3. **LLVM does not vectorise it at all.** Disassembly of the generated code
   (`inspect_asm`, arm64) counts **0 NEON `*.2d` double-precision FP ops** and
   66 scalar `d`-register FP ops per cell in `step_kernel` (15 in
   `faces_kernel`). At ~9.4 ns/cell ≈ 33 cycles for 66 scalar FP ops, the kernel
   is already running at ~2 FP ops/cycle — near the scalar superscalar limit.
   The remaining 4x is locked behind SIMD.

Four attempts to unlock it, all measured and all rejected:

| attempt | result |
|---|---|
| branchless top-face mask (`qc * tmask`, bit-identical) | 1.17 ms vs 1.04 ms branchy — **slower** |
| clamp counters as selects instead of branches | 1.12 ms — **slower** |
| clamp counters replaced by min/max reductions | 1.53 ms — **much slower** |
| `fastmath={'nnan','ninf','nsz'}` (no reassoc, no FMA contraction) | 1.11 ms — **no gain**, and it would silently disable the `_nan_to_num` guards |

So the honest ceiling for a **bit-identical, single-threaded** march on this
machine is roughly the 6x measured. 10x needs one of three things, each of which
breaks a stated constraint:

* **threads** — `prange` over the i-planes would give near-linear scaling, but
  violates the single-thread compute convention (and would need a bigger
  scheduler slot);
* **SIMD** — would require hand-written intrinsics or a restructure LLVM will
  actually vectorise; the four cheap routes above are exhausted;
* **larger dt / implicit stepping** — explicitly OUT OF SCOPE for this lane and
  not benchmarked here even for comparison.

Two bit-identity-preserving optimisations WERE banked and are in the code:

* **Halo padding.** T is carried in an `(n+2)³` buffer with a permanently-zero
  halo and face conductivities that are permanently 0.0 on domain-boundary
  faces, so a boundary face contributes `(0.0*(0.0−T))/h² = −0.0` and
  `x + (−0.0) == x`. This removes six boundary branches per cell.
* **Face-conductivity caching.** `_harmonic` is exactly symmetric — `a+b` and
  `0.5*(a+b)` obviously, and `2.0*a*b/den` because `2.0*a` is an exact
  power-of-two scaling so `(2a)*b` and `(2b)*a` round the same real number
  identically — so each internal face is evaluated once instead of twice.
* **No-latent fast path.** Where `rho_L == 0` (~96 % of the domain, everything
  outside the part) all three `np.where` enthalpy candidates provably collapse
  to `H/rho_cp`, dropping two of three candidate divides plus the `frac` divide.
  This is an exact algebraic identity, not an approximation; the signed-zero
  corner is unreachable because `H_lo = rho_cp * 175 > 0`.

---

## 5. What was NOT achieved / not done

* **The 10x target was missed** (5.9x / 6.3x). Reported, not papered over. The
  accuracy tolerance was never widened to compensate — `FLOOR_RTOL` is still
  1e-16 and every field matches at exactly 0.0.
* **The float32 stretch (item 4) was NOT attempted.** It was gated on (1)–(3)
  landing, and (3) did not. It would also be a poor lever here: since the kernel
  does not vectorise, float32 buys only memory traffic, and the kernel is 5x
  above its memory floor. It would additionally forfeit bit-identity by
  construction, which is the module's entire value proposition.
* **The EQS solve is untouched.** The 6.9 s LU factorisation and 2.4 s of
  back-solves in the quoted n=48 profile are not addressed and not claimed. On
  a 1200-step n=48 run the march is now 2.0 s against ~10 s of EQS, so the EQS
  is the new dominant cost and is the obvious next target.
* **Rolling-plane fusion of `props_kernel` + `faces_kernel` into `step_kernel`**
  (keeping k / rho_cp / rho_L / kf in L1 plane buffers) was scoped but not
  implemented; it would reclaim at most ~0.3 ms of the 1.7, i.e. ~7x not 10x.
* **Unsupported run() options** (power schedules, heatsink lattices, powder-loss
  BC, premix, edge regularisation, S4 in-march EQS re-solve) raise rather than
  approximate. Those runs must still use `heatr3d.run`.

## 6. How to re-run

```bash
cd <repo root>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
.venv312/bin/python -m pytest engine_speed/tests -q      # the gate (12 tests, ~5 min)
.venv312/bin/python -m engine_speed.gate 32              # -> gate_results_n32.json
.venv312/bin/python -m engine_speed.bench                # -> bench_results.json
.venv312/bin/python -m engine_speed.profile_march 48     # per-stage profile
```

Pinned: **numba 0.66.0, llvmlite 0.48.0** (installed into `.venv312`),
numpy 2.2.6, scipy 1.13.1, Python 3.12.7, arm64 macOS.
