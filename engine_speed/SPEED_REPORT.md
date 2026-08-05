# engine_speed — fast thermal march for heatr3d

Two work items, both under the same bit-identity discipline. `heatr3d.py` is not
modified, not monkeypatched, and not imported from `heatr3d_s2/`.

| item | status |
|---|---|
| **§2–5 numba thermal march** | bit-identity gate PASSED (8/8 exact). Speed **5.9x** at n=48 / **6.3x** at n=96, against a 10x bar — **target NOT met**. Blessed by the engine lane as an opt-in; wired into studio3d behind `fast_march=True`. |
| **§6 EQS cache** | gate PASSED (29 tests). Saves **8.26 s** per hit at n=48 and **195.50 s** at n=96; the corrected (AFTER) arm correctly MISSES. The renorm shortcut is **rejected with measurements**. |
| **§6.6-6.7 disk store + recorded acceleration** | gate PASSED. A **fresh process** hits the per-job store: 8.280 s -> 0.0064 s, bit-identical, zero solves. Corruption (truncation, bit flip, key mismatch) MISSES loudly. Every run records `eqs_cache` hits/misses in `results.json`. |
| **§7 Metal / Apple-silicon GPU march** | **REFUSED, with probe evidence.** Metal Shading Language has no `double`: the live shader compiler rejects it, torch-MPS raises on float64, MLX float64 is CPU-only. float32 would miss the 1e-16 parity floor by ~1e10 and buys only **1.30x** at n=48 / **2.28x** at n=96 over the numba path already shipped. Recommendation: wait for the CUDA bifurcation. |

All numbers below are quoted from recorded JSON (`gate_results_n32.json`,
`bench_results.json`, `eqs_cache_bench.json`, `fresh_process_demo.json`), not
transcribed by hand.

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
| `eqs_cache.py` | content-addressed EQS solution/factorization cache + `DiskSolutionStore` |
| `bench_eqs_cache.py` | Express-scenario cache benchmark; writes `eqs_cache_bench.json` |
| `demo_fresh_process.py` | two-subprocess cross-process store demo; writes `fresh_process_demo.json` |
| `tests/test_parity_gate.py` | red-first march gate (12 tests) |
| `tests/test_eqs_cache.py` | red-first cache gate (29 tests) |

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
* **The EQS solve is not made faster**, only *avoidable* when it repeats — see
  section 6. A cold solve costs exactly what it did before.
* **Rolling-plane fusion of `props_kernel` + `faces_kernel` into `step_kernel`**
  (keeping k / rho_cp / rho_L / kf in L1 plane buffers) was scoped but not
  implemented; it would reclaim at most ~0.3 ms of the 1.7, i.e. ~7x not 10x.
* **Unsupported run() options** (power schedules, heatsink lattices, powder-loss
  BC, premix, edge regularisation, S4 in-march EQS re-solve) raise rather than
  approximate. Those runs must still use `heatr3d.run`.

## 6. EQS factorization / solution caching (`eqs_cache.py`)

Second work item, assigned after the march prototype was blessed. Same
bit-identity discipline. `heatr3d.py` still untouched.

### 6.1 What actually enters the assembly (read-only enumeration)

From `heatr3d.solve_eqs_3d`, the matrix **A** is a function of exactly:

* `gamma` — via `_harmonic(gamma, roll(gamma))` on all six faces,
* `grid.h` — via `h2 = grid.h**2`,

plus the hard-coded electrode geometry (Dirichlet on the y_min / y_max planes)
and the fixed `+ 1e-18*I` regularisation. The RHS **b** additionally consumes
`p.v_lo` and `p.v_hi`.

Nothing else reaches the linear system. The part mask, `sat`, sigma/eps,
frequency, `edge_width_m` and premix all reach it **only through `gamma`**, so
the key hashes `gamma` itself — strictly safer than hashing its inputs, because
it cannot forget one. Hashing the mask+sat instead would have missed a change of
`L` at fixed `n` (different `h`, same everything else); that case is pinned by
`test_changed_grid_spacing_at_same_n_misses`.

Key = blake2b(gamma bytes, shape+dtype, `h` bits, `v_lo`/`v_hi` bits, resolved
solver path, **fingerprint of heatr3d's own EQS source**). The last term hashes
`inspect.getsource(solve_eqs_3d)` + `_harmonic` + `EQS_DIRECT_MAX_UNKNOWNS`, so
a future edit to heatr3d's EQS invalidates every entry instead of silently
serving fields computed by the old code
(`test_solver_source_change_invalidates_the_cache`).

Keys are **byte**-exact, which is stronger than value equality (+0.0 and −0.0
hash differently and would MISS). That direction wastes a solve; it can never
return the wrong field.

### 6.2 Which reuse routes are bit-identical (probed before building)

| route | bit-identical vs heatr3d? |
|---|---|
| ported assembly + `spsolve` | **yes**, exact |
| cached ILU + BiCGSTAB (iterative path, N > 50 000) | **yes**, exact — `spilu` is deterministic |
| `splu(A).solve(b)` instead of `spsolve(A,b)` (direct path) | **NO** — 5.0e-15 rel |

So the **iterative path is factorization-cached** (that is where the cost is:
n ≥ 37, and the ILU build is the dominant term), and the **direct path is
solution-cached only** — heatr3d calls `spsolve`, `splu` does not reproduce it
bitwise, so substituting it would break the gate. The direct path is the
small-grid path, so nothing valuable is lost.

Two cache levels, both gated: SOLUTION (key = matrix key + voltages; a hit does
no linear algebra at all) and FACTORIZATION (key = matrix key alone; serves the
same-geometry-different-voltage case).

**Memory warning, in code:** an ILU at n=48 with `fill_factor=12` is order 1e7
complex nonzeros (~250 MB); heatr3d's own note records 2.21 GB at n=96.
`max_factorizations` therefore defaults to **1**.

### 6.3 THE RENORM QUESTION — answered NO, with numbers

**The exact-arithmetic argument is valid.** Under `gamma → c·gamma` (real
c > 0): every harmonic face conductance scales by c, since
`harmonic(ca,cb) = 2(ca)(cb)/(ca+cb) = c·harmonic(a,b)`; Dirichlet rows are
untouched (entry 1, RHS `v_lo`/`v_hi`); every interior row has **both** its
matrix entries and its RHS contribution scaled by c, so **c cancels and V is
unchanged**. `Qrf = 0.5·Re(gamma|E|²)` then scales by exactly c, and the
fixed-power renormalisation in `compute_qrf_3d` divides that constant straight
back out. So in exact arithmetic the renormalised drive *is* invariant.

**In floating point it does not hold.** The `+1e-18·I` regularisation does not
scale with c, and every product and sum rounds differently. Recorded survey
(`eqs_cache_bench.json → renorm_survey`), max relative deviation of the
renormalised Qrf:

| n | c=2 | c=10 | c=1000 |
|---|---|---|---|
| 16 | 0.00e+00 (exact) | 5.36e-14 | 6.11e-14 |
| 20 | 0.00e+00 (exact) | 1.18e-13 | 9.08e-14 |
| 24 | 3.64e-16 | 1.02e-13 | 1.38e-13 |
| 28 | 0.00e+00 (exact) | 2.47e-13 | 9.24e-14 |

At c=10 and c=1000 the deviation is **1e-13 to 2.5e-13 — three orders of
magnitude above the 1e-16 S1 floor.** Rejected.

Worse than merely failing: it is **exact at some (n,c) and not others** (c=2 is
bit-identical at n=16/20 but not n=24). A shortcut that passes a cheap
small-grid gate and then drifts in production is the most dangerous kind, so
this is pinned by an executable test
(`test_scale_invariance_sometimes_IS_exact_which_is_why_it_is_a_trap`).

**Separately, S4 re-solves are not scalar rescales anyway.**
`apply_sigma_coupling` multiplies **only `Re(gamma)`**, by the *spatially
varying* factor `(1 + a(T−T_ref))(1 + b(ρ−ρ_ref))`. The displacement part is
left alone (verified: `np.array_equal(np.imag(gamma), np.imag(gc))`), so even a
spatially uniform coupling does not produce `c·gamma`. Therefore:

* **a = b = 0** (the engine lane's named reference case): `apply_sigma_coupling`
  returns gamma **bit-for-bit**, so every scheduled re-solve is an **exact cache
  HIT with no trick needed**
  (`test_s4_zero_coefficient_resolve_is_an_exact_cache_hit`).
* **a or b ≠ 0**: the field genuinely changes and **must** be re-solved. The
  cache correctly misses.

Conclusion as instructed: **cache exact matches only.**

### 6.4 Benchmark — the Express scenario

`eqs_cache_bench.json`. Single-threaded; load average 7.7–9.1 with one
competing heavy solve (within the schedule rule).

| | n=48 (N=110 592) | n=96 (N=884 736) |
|---|---|---|
| BEFORE arm (cold miss) | 8.75 s | 180.57 s |
| AFTER arm, sat differs (**must miss**) | 8.21 s — **missed** | 194.20 s — **missed** |
| package-verify, sat identical (**must hit**) | **0.0026 s** | **0.0271 s** |
| same solve uncached | 8.26 s | 195.52 s |
| **saved per hit** | **8.26 s** | **195.50 s** |
| hit bit-identical to the original | **yes** | **yes** |
| AFTER field differs from BEFORE | yes | yes |

The n=48 saving (8.26 s) is above the ~7 s the engine lane estimated. Both
sizes take the iterative ILU path.

**Voltage sweep (factorization reuse), n=32, forced iterative**, `v_lo` =
860 / 1800 / 3600 V at fixed geometry — matrix unchanged, RHS changed, so the
solution cache must miss and the ILU must be reused:

* first solve 1.298 s → reused solves 0.364 s mean (**3.6x**), 2 factorization
  hits, 0 solution hits;
* all three fields **bit-identical** to `heatr3d.solve_eqs_3d`.

### 6.5 Gate results

`engine_speed/tests/test_eqs_cache.py` — **20 passed**. Coverage weighted
toward the catastrophic failure mode (a false hit):

* assembly port reproduces heatr3d's solution exactly;
* hits are bit-identical on both the direct and iterative paths, and return an
  independent copy so a caller cannot poison the cache;
* **misses**: changed `sat`, changed `n`, changed `h` at fixed `n`, changed
  `v_lo`, changed `v_hi`;
* **poisoned keys**: a **one-ULP** mutation of `gamma` — separately in the real
  (sigma) and imaginary (eps_r) parts — must miss; a simulated change of
  heatr3d's EQS source must miss;
* factorization reuse happens for a voltage sweep and does **not** happen for
  changed gamma;
* the renorm findings above;
* end-to-end: `march_fast` with the cache is bit-identical to `heatr3d.run` on
  every field including `phi_hist` and `energy_residual_frac`, on both the miss
  and the hit.

### 6.6 Per-job disk solution store (`DiskSolutionStore`)

Approved location: `<grade_dir>/heatr3d/eqs_store/`. **Off by default**
everywhere; only reachable via `fast_march=True`, the only path that takes an
`EqsCache`.

**Solutions only.** SuperLU/ILU objects are not picklable, so the factorization
cache stays in memory and a fresh process still rebuilds the ILU on a genuine
miss. Only exact repeats are free.

One pair of files per key: `<key>.npy` (raw complex128 V) + `<key>.json`
(format, key, payload blake2b, shape, dtype, engine fingerprint).

**Why `.npy` and not `.npz`.** A `.npy` payload has no checksum of its own,
which is exactly what makes the sidecar hash load-bearing: a flipped bit in the
data region loads without complaint and would otherwise be marched with. A
`.npz` would hide that behind the zip CRC and the corruption gate would never
actually be exercised — an untested corruption gate is not a corruption gate.

**Corruption policy — never load garbage, and say so.** Missing sidecar,
missing payload, key mismatch, shape/dtype mismatch, payload-hash mismatch, or
an unreadable file are each logged at **ERROR**, counted in `disk_corrupt`, and
treated as a **MISS** that falls back to a real solve. Writes are atomic
(tmp + `os.replace`) with the payload written *before* the sidecar, so a torn
write leaves a sidecar-less payload, which misses. A read-only or full disk
degrades to memory-only with a warning rather than killing a run that would
otherwise have succeeded.

**Campaign-level shared store: deliberately NOT built** (deferred by the engine
lane). A campaign that wants one points `store_dir` at a shared directory
explicitly.

#### Fresh-process demonstration

`demo_fresh_process.py` → `fresh_process_demo.json`: two separate interpreters,
one shared store. Process 2 must report a disk hit, perform **zero** solves,
and return a bit-identical field. The subprocess test
(`test_fresh_process_hits_the_disk_store`) asserts all three, not just the
timing.

Recorded at **n=48**, single-threaded, load-checked (load 17.8, one competing
heavy solve):

| | |
|---|---|
| process 1 (cold solve) | **8.280 s** |
| process 2 (fresh interpreter, disk hit) | **0.0064 s** |
| **saved** | **8.274 s (1291x)** |
| process 2 disk hit / solves performed | yes / **0** |
| bit-identical across processes | **yes** |
| store size on disk | 1.8 MB |

This is the package-verify claim demonstrated end to end: a fresh process no
longer pays for the EQS solve when gamma is identical.

### 6.7 Recorded acceleration

Engine-lane requirement: silent acceleration is fine, **unrecorded acceleration
is not**. `studio3d.runner.run_densify` now always writes an `eqs_cache` block
into `results.json` next to `engine_march` / `env_provenance`:

```json
"eqs_cache": {"enabled": true, "store": "<grade_dir>/heatr3d/eqs_store",
              "hits": 1, "misses": 0,
              "memory_hits": 0, "disk_hits": 1, "disk_corrupt": 0}
```

A default (reference-march) run records `{"enabled": false}` **explicitly** —
"no cache" must never be confusable with "nobody recorded it", which is what a
missing key would mean. `hits` and `misses` always sum to the number of EQS
solves requested, because the lookup increments exactly one counter per call.
`disk_corrupt` surfaces any refused entry, so a silently rotting store shows up
in the results rather than only in a log nobody reads.

`eqs_store_dir` passed *without* `fast_march=True` is ignored with an explicit
warning and still records `enabled: false`, rather than pretending to
accelerate. `package_verify` carries `engine_march`, `env_provenance` and
`eqs_cache` into its verify record too.

Note a first package-verify normally **misses**: the emitted rasters are
re-quantized, so its `sat` differs from the corrected arm's. A hit there would
mean the cache had ignored a real change to the dopant map.

### 6.8 What the cache does NOT do

* **No direct-path factorization reuse** (see 6.2 — `splu` ≠ `spsolve` bitwise).
* **No cross-process factorization reuse.** Only solutions persist; a fresh
  process rebuilds the ILU on a genuine miss.
* **No CG/AMG.** That is the other lane's; not prototyped here.
* **It does not make S4 coupled re-solves cheap** when the coefficients are
  nonzero. That is a genuine physics change, not a cache miss to be optimised
  away.
* **It does not make a cold solve faster.** The win is entirely in avoiding
  repeats; a first solve costs exactly what it did before.

## 7. Metal / Apple-silicon GPU march port: REFUSED, with measurements

Compute item (2) of the shrinkage-prewarp v2 spec (section 3) asked for a
Metal/MPS port of the thermal march, built as a parallel module in the
`march_fast.py` pattern but gated at a **measured tolerance floor** rather than
bit-identity, since GPU floats forfeit bit-identity.

**The port was not written.** The deciding question was asked first, and it
closes the door. This section is the negative report.

Files: `march_metal.py` (live probe, hard refusal, provenance),
`fp32_cost.py` (the measured cost of the only precision the GPU offers),
`metal_probes/` (the probe scripts), `metal_probe_results.json` and
`fp32_cost_results.json` (recorded output; every number below is quoted from
them).

### 7.1 The deciding question: float64 on this GPU

**Metal Shading Language has no `double` type.** Not as a buffer element and not
as a local scalar. Asked of the live runtime shader compiler on this machine
(`metal_probes/probe_metal_fp64.py`, device `Apple M2 Pro`, families Apple7 and
Apple8 true, Apple9 false):

| MSL source | compiles |
|---|---|
| `device float* a` kernel (control) | **yes** |
| `device double* a` kernel | **no**. `program_source:4:22: error: 'double' is not supported in Metal` |
| `double x = (double)a[i];` local scalar | **no**. Same error, twice |

This is the API and hardware layer, not a binding gap, so it propagates upward.
Both candidate frameworks were still checked directly rather than inferred
(pinned: torch 2.13.0, mlx 0.32.0, python 3.14.0, numpy 2.5.1, macOS 26.5.2):

| framework | float64 on the GPU |
|---|---|
| torch MPS (`mps_available: true`) | `TypeError: Cannot convert a MPS Tensor to float64 dtype as the MPS framework doesn't support float64.` |
| MLX | `mx.float64` **exists** and works on the CPU stream (0.0 deviation vs numpy). On the GPU stream: `ValueError: float64 is not supported on the GPU`. |

MLX carries a trap worth recording: `mx.array(<float64 numpy array>)` returns
**float32**, silently. A port written against MLX without an explicit dtype
assertion would have downgraded the physics with no error and no log line.

Timing corroborates that MLX's "float64" is CPU-only: on a 1 Mi-element vector,
`gpu float32` 0.458 ms, `cpu float32` 0.468 ms, `cpu float64` 0.905 ms, and
`gpu float64` raises.

### 7.2 What float32 would cost, measured against heatr3d itself

Shipping the march in float32 is a numerics change this project does not accept
silently, so the cost was measured rather than argued.

`fp32_cost.py` runs a REDUCED march (densification, powder loss, heat sinks,
power scheduling and in-march EQS re-solves off; fixed `qrf_override` drive)
written in the same association order as `heatr3d.run` and parameterised on
dtype. It is not trusted as a stand-in. It is **gated against the real
engine**, and on all six configurations the float64 arm is **bit-identical to
`heatr3d.run`** (`max_rel_dev_T = 0.0`, `bit_identical_T = true`, including both
guard-firing cases). The stand-in is the kernel.

The float32 arm is a faithful proxy for what a Metal kernel would compute:
`metal_probes/probe_fp32_bridge.py` measured torch-MPS float32 and MLX-GPU
float32 as **bit-identical to numpy float32** on the harmonic-face expression:
**0 ULP** difference on every element, both backends. If anything it flatters
the GPU, which would additionally be free to contract multiply-add pairs.

Suite (n=32, mirroring the shape of `cases.py`, where two cases exist only to
make the guards fire):

| case | max rel dev T | max abs dev T | phi cells differing | enthalpy branch flips | dT-cap hits f64/f32 | temp-clamp hits f64/f32 | melt-onset substep f64/f32 |
|---|---|---|---|---|---|---|---|
| benign | 7.046e-07 | 2.554e-05 C | 0 | 0 | 0/0 | 0/0 | never/never |
| melt_window | 7.705e-07 | 1.387e-04 C | 992 | 0 | 0/0 | 0/0 | never/never |
| melt_crossed | 1.648e-06 | 3.183e-04 C | 104 | 0 | 0/0 | 0/0 | 171/171 |
| cfl_substep (n_sub=3) | 1.803e-07 | 1.033e-04 C | 72 | 0 | 15000/15000 | 0/0 | 18/18 |
| clamp_temp | 8.226e-07 | 7.300e-05 C | 144 | 0 | 0/0 | 183672/183672 | 31/31 |
| clamp_dt | 3.231e-07 | 1.954e-05 C | 0 | 0 | 60000/60000 | 3000/3000 | 17/17 |

Read this honestly, in both directions:

* **float32 misses the gate by ten orders of magnitude.** `gate.py` holds
  `FLOOR_RTOL = 1e-16` and the file says in as many words that it is never
  widened to make a case pass. The float32 deviations sit at 1.8e-07 to 1.6e-06.
  Certifying a Metal march would mean widening the floor by ~1e10, a different
  standard of evidence from the one the numba port passed (8/8 EXACT), on the
  same engine, in the same report.
* **The discrete guards did not flip here.** Enthalpy branch selection, the
  THM-01 dT-cap count, the THM-02 clamp count and the melt-onset substep index
  came out identical in every case, including the two guard-firing ones. That is
  a measurement, not a guarantee: 992 cells already disagree on `phi` in
  `melt_window`, and the melt-onset read is a threshold crossing, so a longer
  march or a more marginal configuration can move it by a whole substep.
  Nothing here licenses "float32 is fine."

### 7.3 How much speed was on the table anyway

Had float32 been acceptable, the prize is smaller than the framing assumed.
`metal_probes/probe_gpu_march_speed.py` times one reduced substep, MLX float32
on the M2 Pro GPU against the same formulation in numpy float64, single process:

| n | cells | numpy f64 (same formulation) | MLX f32 GPU | MLX f32 CPU | numba f64 (§3) |
|---|---|---|---|---|---|
| 48 | 110 592 | 4.3509 ms | **1.2605 ms** | 2.9252 ms | 1.6406 ms |
| 96 | 884 736 | 38.1914 ms | **5.9098 ms** | 21.3259 ms | 13.4599 ms |

* Against the same unfused formulation: **3.45x** at n=48, **6.46x** at n=96.
* Against the fused numba kernel already shipped and already bit-identical:
  **1.30x** at n=48 and **2.28x** at n=96.

Caveat stated rather than buried: the MLX arm is a framework-level
implementation (roll-based neighbours, mask multiplies, many temporaries), so a
hand-fused Metal kernel would beat it. By how much is unmeasured, because MSL
has no `double` and the kernel was therefore never written. The comparison that
is apples-to-apples is the 3.45x/6.46x column, and even the optimistic reading
leaves the working grid size (n=48) close to what the CPU already does.

### 7.4 Recommendation

**Do not pursue a Metal march.** Wait for the CUDA bifurcation named in spec
section 3 item (3), where float64 is native and the existing gate suite can run
unchanged at the 1e-16 floor instead of at a widened one.

If GPU throughput is wanted on this machine before that hardware exists, the
honest options are not this port:

1. **Compute item (1), the CG/AMG iterative EQS on CPU**, is untouched and is
   the larger prize on the profile (the EQS cache section already measures
   195.50 s saved per n=96 hit; the march is 13.5 ms/step).
2. **A mixed-precision march**, float32 GPU stencil with a float64 CPU
   correction, is arithmetically possible but would need its own gate design
   and its own error-growth proof over a full exposure. It is not a smaller
   piece of work than the CUDA path; it is a larger one with a weaker result.

### 7.5 What ships instead

* `march_metal.py`. Default OFF by construction: nothing calls it, and
  `march_metal(...)` raises `MetalUnavailable` naming float64 and quoting the
  MSL rejection. There is deliberately no `allow_float32` escape hatch:
  passing one is a `TypeError`.
* `metal_fp64_supported()` returns True **only** on a positively confirmed
  float64 shader compile. A probe that could not run (binding missing, no
  device) reports `probe_error` and reads as NOT supported. Unknown never
  renders as healthy.
* `metal_provenance()` extends the recorded-acceleration fields:
  `acceleration: "metal_refused"`, `metal_fp64_supported: false`,
  `metal_device`, `metal_refusal_reason`, and the full probe record.
* 12 tests (`tests/test_march_metal.py`, `tests/test_fp32_cost.py`), written
  red-first, including the live probe against this machine's Metal compiler and
  the float64-vs-`heatr3d.run` fidelity gate.

`heatr3d.py` was not modified, not imported from a fork, and not monkeypatched.
The only environment change is `pyobjc-framework-Metal` added to `.venv312` so
the probe is re-runnable there; torch and MLX were kept out of the project venv
and run in a throwaway one.

Load disclosure: the machine carried load average ~4.2 with one competing python
process during the GPU benchmark capture, the same caveat §3 records for the
numba numbers. Ratios are sound; absolute ms/step is inflated for every arm.

## 8. How to re-run

```bash
cd <repo root>
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
.venv312/bin/python -m pytest engine_speed/tests -q      # 41 tests (~6 min)
.venv312/bin/python -m engine_speed.gate 32              # -> gate_results_n32.json
.venv312/bin/python -m engine_speed.bench                # -> bench_results.json
.venv312/bin/python -m engine_speed.profile_march 48     # per-stage profile
# EQS cache bench: n=96 takes ~6 min of real solves. --no-n96 to skip.
.venv312/bin/python -m engine_speed.bench_eqs_cache      # -> eqs_cache_bench.json
# cross-process disk-store demo (two subprocesses, one shared store)
.venv312/bin/python -m engine_speed.demo_fresh_process 48  # -> fresh_process_demo.json
# Metal probe + float32 cost (§7); seconds, not minutes
.venv312/bin/python -m engine_speed.march_metal            # live fp64 shader probe
.venv312/bin/python -m engine_speed.fp32_cost              # -> fp32_cost_results.json
# torch/MLX probes need a throwaway venv: see engine_speed/metal_probes/README.md
```

Pinned: **numba 0.66.0, llvmlite 0.48.0** (installed into `.venv312`),
numpy 2.2.6, scipy 1.13.1, Python 3.12.7, arm64 macOS.
