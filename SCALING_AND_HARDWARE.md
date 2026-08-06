# Scaling and Hardware Brief: the 3-D direct solver

Status: planning brief, 2026-08-06. Owners: Studio lane (this repo's
`studio3d/`) and the solve3d graduation lane. Nothing here is a committed
build; it is the shared map for sequencing the speed work and any hardware
purchase. Numbers are cited to their source; where a figure is a recorded
direction rather than a re-measured value, it says so.

## TL;DR

- One direct solve costs `(_wall time of one forward march_) x (~130
  forward-equivalents)`. The forward march is dominated by the EQS sparse
  linear solve. That solve is currently single-threaded, by design.
- A Linux server is worth buying **now** for throughput, persistence, and
  storage. It does **not** make a single part solve faster until the solver
  is parallelized, because the code cannot yet use many cores on one solve.
- GPU is a **later, at-scale, mixed-precision** lever. What you would give
  up is verification strength (bit-identity) and clean reproducibility, not
  physical accuracy. The physics does not need float64; the project's
  _methodology_ does.
- Recommended order: CPU iterative solver (CG/AMG) first, proven against the
  direct reference with the existing gates, then GPU as a backend swap under
  a tolerance-band equivalence gate. Do not couple new-solver, new-precision,
  and new-hardware into one change.

## 1. Where the time goes (the cost model)

A solve is L-BFGS-B over the full dopant field. Each iteration is one
forward march (transient EQS + thermal + densification on the tet mesh,
producing the objective J) and one adjoint march (the same physics in
reverse, producing dJ/ds at **every** cell at once). The whole-field
gradient for the price of roughly one extra forward is the entire reason
this is tractable: a finite-difference gradient would cost one forward per
cell (tens of centuries at ~30k cells).

Cost structure, grounded:

- Budget is ~40 gradient evaluations (`--budget 40`,
  `solve3d/studio_solve.py`) at a **measured 3.328 forward-equivalents per
  gradient evaluation** (D3, Phase C measured on the anchor circle). That is
  ~130 forward-equivalents per solve.
- The forward march is EQS-solve-dominated. Hard evidence: the EQS solution
  cache saves **195.50 s per hit at n=96** and **8.26 s at n=48**
  (`engine_speed/SPEED_REPORT.md` section 6), i.e. the linear solve is a
  large majority of a step's wall time. The recorded engine direction names
  the EQS solve as ~83% of the run after the fast-march landed (project
  memory; the cache saving above is the re-measured proxy).
- The numba fast-march is **bit-identical, 5.91x at n=48 / 6.31x at n=96**,
  single-threaded (`engine_speed/SPEED_REPORT.md`; kernels carry no `prange`,
  no `parallel=True`, measured zero NEON). It speeds up the thermal march,
  not the EQS solve, which is why EQS is now the dominant term.

Consequence: the lever that matters most is the EQS linear solve, and the
single biggest structural change is moving it from direct factorization to an
iterative method (CG + algebraic multigrid). That is the solve3d lane's
post-S2 item ("No CG/AMG. That is the other lane's",
`engine_speed/SPEED_REPORT.md`; "the CG/AMG iterative EQS on CPU is
untouched", `solve3d/PHASE_C_REPORT.md`).

## 2. The three scaling axes

### Complexity (harder shapes)
More cells **and** a harder optimization landscape (more iterations). Hardware
helps throughput; warm starts and multigrid help single-solve latency.

### Size (bigger parts)
This is the axis to **verify before spending on it**. If the Buckingham-Pi
nondimensionalization holds, cost is roughly flat in physical size, because
the same nondimensional problem is the same number of cells. The adaptive
chamber result is consistent with a partial version: element size is held, so
**in-part cell count stays ~constant as the part grows (19,421 -> 19,462),
only the bed grows** (`solve3d` chamber sweep, `chamber_field_check.json`).
That makes cost sublinear in size, not linear.

OPEN, and it decides the whole size axis: **does the Pi-theorem result
actually hold end to end?** If yes, size mostly stops being a cost problem and
you do not need hardware to brute-force it. Owner: solve3d lane. Flagged to
that session.

### Resolution (finer detail)
Decouple two grids that are easy to conflate:

- **Dopant / FGM resolution = the tet mesh.** Element size is the knob held
  fixed (D1). Raising mesh density raises solve cost.
- **Densified-form resolution = the voxel grid (n <= 96,
  `studio3d/runner.py`).** This is the densify march and viewer
  representation, NOT the solve. Raising voxel n costs the densify march and
  the viewer; it does not touch the direct solve.

Matt's read is correct: the resolution you _see_ in the rendered densified
part is limited by the voxel grid, and that can be pushed independently of the
expensive solve.

## 3. Hardware

### What it buys now vs later

**Now, even with single-threaded solve code:**
- **Throughput.** Independent jobs, and the cold-vs-warm and coarse-vs-fine
  arms, are embarrassingly parallel. A 16-32 core box runs many at once.
- **Persistence.** An always-on headless server means jobs stop dying when the
  app or laptop restarts, which has bitten this project repeatedly (server
  sweeps killing live marches). The compute-schedule rules
  (`studio3d/solve_scheduling.py`) become a real queue instead of ad-hoc load
  checks.
- **Storage** for the growing solved-map library and job artifacts, with fast
  NVMe for the per-job disk solution store.

**Later, once the CG/AMG solver lands:**
- **Single-solve latency**, the thing you actually care about for one part.
  It needs the threaded/GPU sparse solve first. The machine is what that work
  parallelizes onto.

### Spec, driven by the bottleneck

Sparse solves are **memory-bandwidth bound**, not FLOP bound. Spec follows:

- **CPU:** high core count AND many memory channels. AMD Threadripper Pro or
  EPYC (8 memory channels) beats a higher-GHz consumer part here, because
  AMG/CG are bandwidth-bound. 16-32 cores is the sweet spot.
- **RAM:** 128 GB. The transient adjoint must store or recompute the forward
  trajectory for the backward pass (checkpointing is a memory/compute
  tradeoff), and tet meshes plus bed cells are memory-hungry.
- **Storage:** NVMe for the working set, plus a large SSD/HDD for the artifact
  and solved-map archive.
- **GPU:** not day one; see section 4. Buy a board with a spare PCIe slot and
  PSU headroom so a CUDA card can be added if the solver goes there.
- **OS:** Ubuntu LTS, headless, SSH. Matches the existing `.venv312` / dolfinx
  stack.

Rough bands (technical purchasing guidance, not a financial recommendation):
a Threadripper/EPYC workstation in this spec is ~$3-6k depending on cores and
RAM; a consumer Ryzen 9 + 128 GB is a ~$2-3k entry point that gives throughput
and persistence today but fewer memory channels for the eventual multigrid
solve. The bottleneck is bandwidth, so the many-memory-channel option is the
one that keeps paying off.

## 4. GPU tradeoffs

The honest reframe: you would not be trading away _physical_ precision. Model
error (simplified physics, unmeasured Nylon-12 parameters, the quasi-static
assumption) dwarfs float32 roundoff by ~6 orders of magnitude. The real trade
is **verification strength and reproducibility for speed, and the speed only
appears at problem sizes bigger than today's**.

### Where float32 actually bites in THIS solver
The physics tolerates it; three diagnostically load-bearing places do not,
without care:

1. **Linear-solve residual.** FEM stiffness matrices are ill-conditioned
   (condition numbers 1e6+). A float32 CG/AMG solve can stagnate above the
   residual a float64 solve reaches. The standing energy gate wants
   `residual_frac < 1e-2`, so it likely still clears, with less headroom.
2. **Accumulation over thousands of substeps.** The energy audit sums stored
   energy step by step; the residual currently reads ~1e-13. Pure float32
   degrades that to ~1e-5..1e-6, still inside the 1e-2 gate but losing the
   sharp diagnostic that catches a melt-onset blow-up early.
3. **The adjoint gradient.** A float32 backward march yields a noisier
   gradient, forcing the finite-difference gate (the core gradient-correctness
   check) to loosen. That weakens the solver lane's whole methodology.

Mitigation for all three: **mixed precision.** float32 only in the sparse
matrix-vector inner loop where roundoff is harmless; float64 for accumulation,
residuals, and the gradient checks. Keeps most of the speed and most of the
diagnostic strength.

### Two things that bite harder than roundoff
- **Consumer cards cripple float64.** Gaming NVIDIA cards run FP64 at
  1/32..1/64 of FP32; only datacenter cards (A100/H100, ~10x price) have
  strong FP64. On an affordable card, "GPU speed" IS float32 speed, which is
  the crux of the precision question: the speed is bought by dropping to
  float32, not incidental to the GPU.
- **GPU non-determinism breaks bit-identity.** Parallel atomic reductions
  change summation order run to run, so results typically vary at ~1e-6
  between identical runs unless deterministic kernels are forced (slower).
  This project's speedup culture is built on `array_equal` against a reference
  (bit-identical numba fast-march; splu-vs-spsolve treated as a real 5e-15
  distinction). GPU moves that to "statistically equivalent within a band,"
  and also dents the run-to-run reproducibility the dissertation prizes.

### What actually speeds up (and what does not)
- **Iterative sparse solvers, yes.** cuSPARSE / AMGX / PETSc-GPU do algebraic
  multigrid well; the sparse matrix-vector product is bandwidth-bound where a
  GPU has ~5-10x the CPU's bandwidth. This is the real win, and it is the
  dominant term.
- **Direct factorization, no.** The triangular back-substitution is inherently
  sequential and maps poorly to a GPU. Going GPU essentially FORCES the move
  to iterative solvers, which is the same restructuring the CPU multigrid work
  already requires. The two efforts converge on one code shape.
- **Small-problem break-even risk.** ~19-30k cells is small for a GPU. PCIe
  transfer and kernel-launch overhead can eat the gains, and a problem that
  fits in CPU cache can beat a GPU stuck on overhead. GPU pays off at the
  bigger, higher-resolution parts, which is exactly the axis of concern, so
  the alignment is good, but at today's sizes a GPU could be slower.

## 5. Recommended sequencing

Do not couple three risky changes (new solver + new precision regime + new
hardware) at once. Clean path:

1. **Buy the server** for throughput, persistence, and storage. Stand up the
   always-on job runner and the real queue. Immediate operational win,
   independent of solver speed.
2. **Verify the Pi-theorem size result** with the solve3d lane. If it holds,
   the size axis mostly closes and de-risks the whole scaling story.
3. **Land the CPU CG/AMG iterative EQS solve** (solve3d lane, post-S2) and
   prove it against the direct-factorization reference with the existing
   gates. This is also the GPU-shaped structure.
4. **Only then** swap in a GPU iterative backend, with a tolerance-band
   equivalence gate replacing bit-identity and mixed precision keeping float64
   where it is diagnostically load-bearing. The GPU becomes a verified backend
   swap, not a rewrite-and-revalidate-and-buy gamble.

## 6. Open questions and owners

- **Pi-theorem size invariance: does it hold end to end?** Decides the size
  axis. Owner: solve3d lane.
- **CG/AMG on CPU:** the dominant-term speedup. Owner: solve3d lane, post-S2.
- **Server purchase and provisioning** (queue, storage layout, always-on job
  runner). Owner: Studio lane + Matt.
- **heatr3d frozen 60 mm chamber vs adaptive solve chamber:** parts over
  60 mm can solve but cannot be densify-verified, so intake still refuses
  them. Unfreezing `Grid` L is graduation-lane territory and is the unlock for
  large parts (see `studio3d` chamber-tagging, commit cf1a203).
