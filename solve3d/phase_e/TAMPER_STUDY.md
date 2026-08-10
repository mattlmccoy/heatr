# Tamper Study — why the shape solve lands over-ceiling AND under-dense, and can it close?

**Part:** `Part Studio 1 - Tamper.stl` (Grade-and-Print job feb850ec), a real
complex user part. Extent 45.3 x 45.5 x 11.2 mm: a broad flat top plate
(z ≈ 3.3–5.6 mm holds most of the mass, ~45 mm wide) over a narrower lower
core/stem (z ≈ −5.5–3 mm, few cells per slab).

**Known failure (solve3d/phase_e/results/phase_e_tamper.json, `solve_filter_only`):**
trajectory peak `peak_T_c = 281.9 C` (over the 250 C degradation ceiling),
end-state in-part max 208.4 C, `in_bounds_below_floor_fraction = 0.72`,
`part_mean_phi = 0.34`. The uniform arm is the same story (peak 279.8, end 201.5).
This solve predates the B1–B4 ceiling coupling; it is a pure dense-iff-in-bounds
shape solve with no ceiling term.

**Verdict up front:** the Tamper is **infeasible at any single static uniform
drive** — the peak crosses 250 C at ~0.86x nominal drive, but densifying the
bulk needs ≥1.56x (rim then at ~410 C). The root cause is an **intrinsic ~3x
radial power-deposition gradient** (hot outer rim of the top plate, cold lower
core) that no scalar drive can flatten. A scalar knob cannot fix a spatial
problem. The rescue path is **spatial**: two-sided per-node dopant grading (the
`feat/pernode-twosided-tuning` actuator) to flatten the gradient, backed by a
Stage C dwell schedule for the transient/azimuthal component and a modest drive
backoff. Whether that path *closes* the part needs the queued heavy solves.

Figure: `results/fig_tamper_study.png`.

---

## Data contract (verified against the code, not assumed)

- `peak_T_c` (the ceiling quantity) is the **TRUE trajectory maximum of the
  in-part temperature over (t, x)**, tracked substep-by-substep
  (`solve3d/forward.py:889–894`, mask `part_peak_mask = m_nodal > 0.5`). It is a
  **single-cell** maximum. The end-state in-part max is a *different* number
  (208 C) — so the 282 is a **transient overshoot**, not the end state.
- `phi` in the score is the **melt fraction of the END-STATE temperature**:
  `phi = clip((T_read − 175)/10, 0, 1)`, melt band 175–185 C
  (`forward.phase_fraction`, `phase_e/run.py:165`). It is **not** the accumulated
  relative density `rho` that governs the march stop. So
  "72% below the density floor" is an **end-state melt-fraction** metric; the
  ceiling is a **trajectory-peak** metric. Different temporal references.
- Drive 1.0 = `power_density_w_per_m3 = 1.5915e6`; the EQS solve is fixed-power
  normalized, so `qbar` equals the drive exactly (verified: probe `qbar = 1.592e6`).
- Ambient / start 50 C. Fusion floor (phi ≥ 0.85) ⇒ end-state T ≥ 183.5 C.
- Saved-map centroids reproduce the rebuilt mesh centroids to 0.0e0 m (order matches).

---

## Q1 (Matt's first question): is the 282 C peak even real, or a mesh artifact?

**Premise tested:** localized hot spots are over-predicted on coarse meshes
(pyramid apex, square `peak_mesh_offset`). Is the 282 a single-cell coarse-mesh
over-prediction?

**Cheap evidence (EQS-only probe, one linear solve, no march —
`tamper_qrf_probe.py`, `tamper_qrf_profiles.py`):**

- The mesh is **NOT** uniformly coarse. `lc_part = 2.5 mm` is nominal only; the
  Tamper's own tessellation carries a 0.105 mm minimum edge, so the conforming
  mesh inherits fine elements. **Actual cell size: min 0.054 mm, median 0.321 mm,
  max 1.883 mm** — 84% of cells are sub-1 mm. The naive "coarse 2.5 mm single
  cell" picture is already false; the median cell is ~8x finer than nominal.
- The peak-Q cells **are sub-mm slivers at a geometric edge**: the hottest cell
  is 0.25–0.35 mm (uniform 9.4x mean Q, solved 11.6x mean Q), sitting on the
  outer rim (r ≈ 12–22 mm) of the top plate. This is a **field concentration at
  a sharp edge** (|E|² rises at edges), on cells that are already fine.
- Because the hot cells are *fine, not coarse*, the classic coarse-cell
  over-prediction (conduction can't spread the deposit) is **weaker** here than
  on the pyramid apex. The open sub-question is whether the edge is a genuine
  geometric feature (peak converges near 282) or an **STL-faceting singularity**
  (flat facets meeting at artificial angles; a filleted or finer-tessellated STL
  would relax it).

**Honest limit:** the exact convergence of the *transient* 282 needs a refined
forward (queued heavy — see run A below). **BUT the feasibility verdict does not
depend on it**: even deflating the peak 30% leaves static-uniform infeasible
(§Q2). The mesh test refines *how much* drive-backoff/headroom a grading
solution needs; it cannot rescue static-uniform feasibility.

---

## Step 1 — where and when is the peak?

- **WHERE (space):** hottest 1% of cells at **r ≈ 13 mm (rim), z ≈ 3 mm (upper
  plate)**, i.e. the outer edge of the broad top plate — **not** the bulk, not
  the stem. Azimuthally it is a **two-lobe (bilateral) pattern** occupying 8/12
  30°-bins in two opposing groups (histogram `[0 17 36 39 19 0 0 12 30 33 16 0]`),
  not a full ring and not a single spot. The all-cell field is nearly
  axisymmetric; only the *hottest* cells break into two opposing lobes.
- **WHEN (time):** **transient.** In-part end-state max is 208 C but the
  trajectory max is 282 C — a ~74 C overshoot mid-heating. The rim (high Q, low
  local mass) spikes early, then conducts/relaxes toward ~208 C by the stop while
  the bulk slowly catches up in density. (Exact peak substep is not saved; it
  needs a forward with `true_peak_step_index` exported — run A.)
- **The under-dense region is the complement:** coldest 25% of cells at
  **r ≈ 8 mm, z ≈ −1 mm** — the inner/lower **core**, ~48% of part volume at
  <0.6x mean Q. That is what stays below the fusion floor.

So the failure is a **radial + axial power gradient**: RF couples into the outer
upper rim (edge fringing) and starves the inner lower core.

---

## Step 2 — density-vs-ceiling feasibility (the scissors)

**Deposited-power gradient (fixed by geometry + field, EQS probe):** radial Q
runs from ~0.6x mean in the core (r < 10 mm) to ~1.85x mean at the rim
(r ≈ 15 mm) — a **~3x core-to-rim ratio** — plus a transient sliver overshoot on
top. This ratio is **independent of drive** (fixed-power normalization scales all
cells together).

**First-order drive-feasibility map** (linear pre-phase-change scaling of the
saved end-state field; `tamper_feasibility.py`, gated against
`forward.phase_fraction` in `tests/test_tamper_feasibility.py`):

| drive x | rim peak (est.) | below-floor frac | mean phi |
|--------:|----------------:|-----------------:|---------:|
| 0.80    | 236 C  (under)  | 1.00             | 0.00 |
| **0.86**| **250 C (ceiling)** | **0.98**     | ~0.00 |
| 1.00    | 282 C  (over)   | 0.65             | 0.44 |
| 1.20    | 328 C  (over)   | 0.33             | 0.70 |
| **1.56**| **412 C**       | **0.15**         | 0.85 |
| 1.90    | 491 C           | 0.05             | 0.94 |

- At the drive where the rim peak just reaches the 250 C ceiling (~0.86x),
  **~98–100% of the part is below the density floor** (nothing fuses).
- To get below-floor under 15%, drive must reach **≥1.56x → rim at ~412 C**;
  under 5% needs ~1.9x → ~491 C.
- The peak crosses 250 C **long before** the bulk densifies, at **every** drive.

**This is an honest INFEASIBLE-at-static-uniform-drive result** — a genuine
method-envelope finding: a part that cannot be both dense and under-ceiling at
any single static drive. It is corroborated three independent ways:
1. the ~3x geometric Q gradient (a scalar cannot flatten a spatial ratio);
2. the code's own note that the Tamper heads to a **~365 C plateau**
   (`run_tamper.py:53`) — to fuse the core you must approach that plateau;
3. the low-drive feasibility run (`solve3d/results/tamper_feasibility_ch085_short.json`):
   peak 173 C (under melt), **part_mean_phi 0.0, mean 110 C after 1200 s** — a
   drive gentle enough to stay under ceiling fuses **nothing**.

**Caveat (direction of the error):** the linear estimate **ignores latent heat**
(real fusion is slower → real needs *more* drive → the true picture is *worse*,
not better) and ignores stop-time re-optimization. It is a conservative *prior*;
the quantitative peak(drive)/below-floor(drive) curves want the transient sweep
(run B). Mesh deflation of the 282 (Q1) shifts the red curve down modestly but
does not create an overlap window.

---

## Step 3 — lever ranking (tied to the diagnosis)

The failure is a **fixed spatial power non-uniformity** (3x radial gradient) with
a **transient, bilateral** rim overshoot on top. Rank:

1. **Two-sided per-node dopant grading — STRONGEST, attacks the root cause.**
   The gradient is spatial, so the fix must be spatial. Pull dopant **out of the
   rim** (lower sigma → lower local Q) and **add dopant into the cold core**
   (raise sigma → raise local Q), flattening the 3x ratio. The current solve is
   **one-sided** — `map_stats` max = 1.0, mean 0.76, it only *pulls* dopant
   0→1 and **never boosts the starved core above baseline** (probe: 14% of volume
   pulled, 0% added). It is structurally unable to feed the core. The
   `feat/pernode-twosided-tuning` branch is named for exactly this actuator
   (sigma range doped/virgin ~4e6 gives ample authority to flatten).

2. **Stage C dwell / rotation — addresses the transient + azimuthal part only.**
   The rim overshoot is **bilateral** (two opposing lobes), so an indexed
   rotation (match indexing to the 2-fold symmetry: 90°/180° dwell) spreads the
   lobe dwell-time and lowers the transient peak. But rotation **cannot fix the
   radial gradient** — after azimuthal averaging the rim is still hot at all
   angles and the core still starves. Complement to grading, not a substitute.

3. **Stage B exposure / lower-drive-longer-bake — helps at the margin.** A gentler
   drive shrinks the transient overshoot (peak nearer steady state) and lets the
   core accumulate rho over time. But the *steady* rim-vs-core gradient persists;
   if the rim steady state at the drive that fuses the core is >250 (it is, ~365 C
   plateau), a longer bake alone does not close it.

4. **Drive backoff + 15 C headroom (B4 on Tamper) — necessary, not sufficient.**
   Only works if a feasible window exists; §Q2 shows none exists for static
   uniform. It becomes the *finishing* lever **after** grading opens a window.

5. **Orientation — plausible but unproven, separate campaign.** RF couples to the
   broad flat plate faces; reorienting (e.g. plate edge-on) changes which faces
   fringe and could cut the rim concentration. Unpredictable; a reorientation
   sweep is its own heavy study.

**Recommendation:** the single most promising lever is **two-sided per-node
grading** to flatten the radial Q gradient, then a **Stage C 2-fold dwell** plus a
**modest drive backoff + headroom** to hold the residual transient rim peak. Do
the mesh-convergence check first so the backoff/headroom is sized to the *true*
peak, not an inflated single-cell one.

---

## Step 4 — concrete next solves (all HEAVY; queue behind a free slot)

At the time of writing both heavy slots are full (pid 27889 adaptive probe, pid
27538 Studio acceptance; load ~11). **No heavy solve was launched.** Priority order:

- **A. Mesh-convergence forward (answers Q1).** Re-run the *uniform* Tamper
  forward with the rim edge refined (locally lc ≈ 1.0–1.25 mm, or a
  finer/filleted STL retessellation), export `true_peak_T_c` and
  `true_peak_step_index`, and report peak-vs-resolution like the pyramid
  n64/80/96 study. Converges <250 ⇒ Tamper is closer than it looks; converges
  near 282 ⇒ real hot edge. ~1 forward (~30–40 min).
  Entry: adapt `solve3d/phase_e/run_tamper.py::build_case(lc_part=…)` +
  `tc.forward(tc.design_to_sigma(np.ones(n)))`, read `standing_gates`.

- **B. Transient drive-feasibility sweep (confirms §Q2 quantitatively).** 4–5
  uniform forwards at drive {0.6, 0.86, 1.0, 1.3, 1.6}x, plot true peak(drive)
  and below-floor(drive). Confirms the scissors with the real transient + latent
  heat. Cheaper if capped in time. (`solve3d/stage_a.py` already has a drive-sweep
  harness — `[drive-sweep] a=… peak=…`.)

- **C. Two-sided ceiling-coupled grading solve (the rescue).** The B1–B4 stack
  (density co-state adjoint + augmented-Lagrangian ceiling term) on Tamper with a
  **two-sided** per-node dopant law (s free above 1.0, not clamped). Multi-hour.
  Do NOT launch without reporting and a confirmed free slot. If it opens a
  dense-and-under-250 window, layer Stage C dwell + B4 backoff to ship it; if it
  does not, the honest dissertation finding is that the Tamper geometry is outside
  the single-drive method envelope and needs a process lever (Stage C / orientation)
  or a geometry change.

---

## Artifacts

- `results/fig_tamper_study.png` — (a) radial Q gradient, (b) feasibility scissors.
- `tamper_qrf_probe.py`, `tamper_qrf_profiles.py` — EQS-only hot-spot probes (light).
- `tamper_feasibility.py` — tested pure core of the drive estimate.
- `tests/test_tamper_feasibility.py` — gates the phi law vs `forward.phase_fraction` (3 pass).
- `tamper_feasibility_estimate.py`, `fig_tamper_study.py` — estimate table + figure drivers.
