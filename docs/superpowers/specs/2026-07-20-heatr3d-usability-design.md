# HEATR-3D usability — design spec

Date: 2026-07-20
Status: approved for planning (awaiting spec review)
Scope: make the HEATR-3D tab a genuinely usable design instrument, then add the
presentation polish that tells the FGM story. Functional-first, then demo-facing.

---

## 1. Goal

Today a user can configure a 3D geometry, launch a coupled EQS → thermal → densification
solve, and orbit a voxel shell. What they **cannot** do: find the run afterward, see how the
part was functionally graded through its interior, or see what the part sintered into. This
spec closes those gaps.

Primary user (in order): (1) Matt, running real 3D FGM cases for the dissertation — depth and
correctness first; (2) committee/collaborators — a clear visual narrative second.

Success = a user can run a 3D FGM case, **see the grading and the sintered result**, save it,
find it again, and compare runs — without reading the code or the filesystem.

---

## 2. Current state (verified, with citations)

All paths relative to `geo-prewarp/`. Line numbers as of this spec.

**Architecture.** The GUI server does not run the solver in-process. `POST /api/heatr3d/run`
writes `config.json` and spawns `heatr3d_job.py` under a dedicated venv as a detached
subprocess (`rfam_gui_server.py:5318-5326`); job state lives in the in-memory dict `_H3D_JOBS`
(`rfam_gui_server.py:5276`) and on disk under `outputs_eqs/_heatr3d/<12-hex-id>/`
(`rfam_gui_server.py:5275`). The solver `heatr3d.py` is a SHA-checked synced copy of the
canonical dissertation solver (`heatr3d.py:1-22, 843`).

**Endpoints.** `POST /api/heatr3d/preview` (synchronous, returns `geometry.json`, no solve),
`POST /api/heatr3d/run` (spawns, returns `{id}`), `GET /api/heatr3d/status?id=` (returns
`{progress, done, geometry, results, error}`) — `rfam_gui_server.py:5307-5368, 5417-5426,
5806-5827`. Frontend polls status every 1.5 s (`heatr3d.js:122-141`).

**Renderer.** three.js v0.160 (`heatr3d.html:181-186`). `renderSurface()` draws one
`InstancedMesh` of cubes, one per **surface** voxel, from `geom.surface_xyz_mm`
(`heatr3d.js:48-74`). "Color surface by" has exactly two options: uniform blue (`geom`) and a
hand-rolled viridis over `geom.surface_sat` (`sat`) (`heatr3d.js:40-46, 54-61`,
`heatr3d.html:159-164`). If `sat` is requested but absent it silently renders blue
(`heatr3d.js:54-55`) — a false-quiet failure.

**In-page results grid.** `showResults()` **is** wired: on `s.done` it fills `#resultsGrid`
from `s.results` over `RESULT_KEYS` (`heatr3d.js:137, 143-160`). Keys include `sigma_T`,
`T_max_C`, `sintered_frac`, `dice`, `rho_final_mean/std`, `z_shrink_pct`, `xy_shrink_pct`,
`warp_std_pct`, `layer_multiplier`, `solve_s`. Rendered as raw `key → toFixed(3)` — no units,
no labels, no grouping.

**Solver outputs.** `Result` (`heatr3d.py:427-443`) carries four physically distinct 3D
arrays at grid resolution (default 48³): `T_phi90` (temperature °C), `Qrf` (absorbed RF power
W/m³), `phi_final` (melt fraction), and `rho_final` (relative density, densify runs only). The
FGM dopant field `sat` is a full 3D `(n,n,n)` array from `make_fgm()` (`heatr3d.py:697-713`).
The job writer persists all of these to **`fields.npz`** (`heatr3d_job.py:131-138`): keys
`part, T_phi90, phi_final, Qrf, rho_final, sat, h`.

**Deformation.** `shrinkage_analysis()` (`heatr3d.py:749-810`) computes a per-column final
height map `_H_final` and a full 3D per-voxel through-thickness shrink field `_lam_z`, plus
in-plane `lam_xy`. The job writer strips every underscore-prefixed key before saving
(`heatr3d_job.py:128`), so only scalar shrink/warp numbers survive; the deformation fields are
discarded.

---

## 3. Root causes of the three complaints

1. **"Results don't populate anywhere."** Two mechanisms. (a) On the HEATR-3D page the grid
   does populate on completion — so if it appears empty, runs may be failing in the local venv
   (see F0). (b) The Results browser never lists HEATR-3D runs: `_collect_results()` only
   accepts a run dir with a `summary.json` or a PNG/GIF/SVG (`rfam_gui_server.py:4189-4192`),
   and HEATR-3D writes neither. Not an exclusion — a schema mismatch.
2. **No layerwise slice view.** `sat` (and `T_phi90`, `phi_final`, `rho_final`, `Qrf`) already
   exist as full 3D arrays in `fields.npz`. No endpoint serves them; the renderer only draws
   the surface shell. **UI + one endpoint. No solver work.**
3. **No post-sinter warped geometry.** The deformation field is computed then discarded
   (`heatr3d_job.py:128`). Reconstructing the warped shape is a closed-form integration of
   `_lam_z`/`lam_xy` — **post-processing + serialization, not new physics** (no PDE, adjoint,
   or FD gate).

---

## 4. Feature set

Each feature lists: purpose, where the work is, the data contract, and how it is verified.
Effort: S ≈ hours, M ≈ a day-ish, L ≈ multi-day. TDD applies to pure logic (slicing,
deformation math, results collection); UI/rendering uses a concrete browser-verification gate.

### Phase 1 — foundation (kill "results don't populate")

**F0 — Prove a run completes end-to-end (verification gate, blocks all). — PASSED 2026-07-20.**
Purpose: confirm the solver venv spawns and a real run writes `results.json` + `fields.npz` in
this environment. If runs silently error, that is itself part of complaint #1.
Result: a real run (sphere, `diam=0.028 m`, `n=32`, `fgm=melt`, `densify=true`) completed in
~19 s (run id `1e10228aff1d`). `fields.npz` written with all six 3D arrays at correct shape/dtype
(`part, T_phi90, phi_final, Qrf, rho_final, sat, h`, each `(32,32,32)`). `sat` grades in space
(0.33–0.67) and layer-by-layer along z (0.667→0.554→0.667 — hotter middle, less dopant), so the
slice/warp views will have meaningful data. Metrics real: σ_T=22.7, dice=0.94, z_shrink=30.7%.
The finished run was confirmed **absent** from `/api/results` (0 of 352 listed) because its dir
has no `summary.json`/media — validating F1's premise. Solver is healthy; "no results" is a
visibility gap, not a solver break.
Incidental (folded into F7): `diam`/`zspan` are meters (the frontend divides mm by 1000); the
API silently accepts absurd magnitudes — passing `28` yields a grid-filling "part" with no
warning.

**F1 — HEATR-3D runs appear in the Results browser.**
Purpose: every finished run shows up where all other runs live.
Work (backend, `heatr3d_job.py` writer + possibly `_collect_results`): on run completion write
(a) a `summary.json` matching the shape `_collect_results()` expects, and (b) one preview PNG
(e.g. the surface shell colored by `sat`, or a mid-plane `sat` slice) so the content gate at
`rfam_gui_server.py:4189-4192` accepts the dir. Prefer emitting the artifacts the existing
browser already recognizes over widening the gate, so no other run type is affected.
Data contract: reuse the existing `summary.json` schema — capture its real keys from an
existing accepted run before writing (do not invent the schema).
Effort: S–M. Verify: the run appears in `/api/results`; opening it shows metrics + preview.

### Phase 2 — the science views

**F2 — Serve volumetric fields.**
Purpose: expose the interior arrays the slice/warp views need.
Work (backend): new endpoint, `GET /api/heatr3d/field?id=<jid>&name=<field>&axis=<z|y|x>&k=<index>`
returning one 2D slice (JSON array or PNG) from `fields.npz`, plus `GET
/api/heatr3d/fieldmeta?id=<jid>` returning `{fields:[...], dims, h_mm, ranges:{field:[min,max]}}`.
Reads `fields.npz` keys `part, T_phi90, phi_final, Qrf, rho_final, sat, h` (`heatr3d_job.py:131-138`).
Decide slice transport (per-slice PNG vs JSON) during planning; PNG is lighter for 48² and
avoids large JSON. Effort: S. Verify (TDD on the slice extraction; API smoke): requesting
`sat` slice `k` returns the same values as `np.load(fields.npz)['sat'][:,:,k]`.

**F3 — Layerwise slice viewer.**
Purpose: scrub through the part and see how it was graded and how it responded.
Work (frontend): a layer slider (Z by default, axis selectable), a field selector (`sat`/T/φ/ρ/Qrf),
a 2D heatmap panel (viridis, shared with the legend), min/max readout, and the current slice
plane indicated in the 3D view. Masked (outside-part) voxels rendered distinctly, not as 0.
Depends on F2. Effort: M. Verify (browser gate): slider changes the rendered slice; `sat`
grading is visibly non-uniform corner-to-center; the "not-computed" case (e.g. `rho_final` on a
non-densify run) shows an explicit empty state, never a false-quiet blank.

**F4 — Emit warped (post-sinter) geometry.**
Purpose: produce the deformed part shape from fields the solver already computes.
Work (solver post-proc + job writer): a new pure function that integrates `h·lam_z` along each
z-column (and applies `lam_xy` in-plane) to produce displaced voxel coordinates; persist the
displacement field / warped surface to the run dir (extend `fields.npz` or a new
`warp.npz`/`warped_geometry.json`). Stop stripping the underscore arrays needed for this
(`heatr3d_job.py:128`) or compute displacement explicitly before the strip. **No new physics.**
Effort: S–M. Verify (TDD): zero-shrink input → zero displacement; uniform shrink S → uniform
linear scale; monotonic `lam_z` → monotonic column compaction. One real-data run: displaced
top-surface height matches `_H_final` (`heatr3d.py:770`) within tolerance.

**F5 — Warped-geometry view.**
Purpose: see what the part warped into.
Work (frontend): render the warped surface as a second shell; pre/post toggle and/or
side-by-side; optional color-by-displacement-magnitude. Depends on F4. Effort: M. Verify
(browser gate): toggling shows nominal vs sintered shape; displacement coloring matches the
scalar `z_shrink_pct`/`warp_std_pct` in the results grid in sign and rough magnitude.

### Phase 3 — polish and demo

**F6 — Run history + compare on the HEATR-3D page.**
Purpose: reload past runs; compare σ_T and shrink across FGM modes. Work: backend list endpoint
over `outputs_eqs/_heatr3d/`, frontend history list + a small compare table. Depends on F1.
Effort: M.

**F7 — Robustness + presentation.**
Purpose: the sharp edges. Includes: surface run errors in the UI (not swallowed), remove the
silent `sat→blue` fallback (`heatr3d.js:54-55`) in favor of an explicit state, verify the STL
import path and the `cone`/`cylinder`/`dumbbell` shapes actually solve (frontend claims them;
backend support unverified), give the results grid real labels, units, and grouping, and add a
**units/range guard on `diam`/`zspan`** — the API accepts meters but silently accepts absurd
magnitudes (e.g. `28` → a grid-filling part), so validate against the chamber size and reject or
warn. Effort: M.

---

## 5. Data contracts to capture before building (iron law: no invented shapes)

- `summary.json` — capture the real schema from an existing **accepted** run dir before writing
  F1's summary; do not infer it from `_collect_results()` alone.
- `fields.npz` keys and dtypes — confirmed `part(bool), T_phi90, phi_final, Qrf, rho_final, sat,
  h` (`heatr3d_job.py:131-138`); `rho_final` is `zeros(1)` sentinel on non-densify runs, `sat`
  likewise when no FGM — the slice API must distinguish sentinel-absent from real-zero.
- `_lam_z` / `lam_xy` / `_H_final` semantics — re-read `shrinkage_analysis` (`heatr3d.py:749-810`)
  at F4 time and unit-check against `_H_final` before trusting the integration.

---

## 6. Non-goals (YAGNI)

- No new sintering/thermal physics, no adjoint, no solver accuracy changes. This spec only
  surfaces and visualizes what the solver already computes.
- No merge of the HEATR-3D job system into the main run queue — F1 makes runs *visible* in the
  Results browser via the existing artifact contract, nothing deeper.
- No triangulated isosurface/marching-cubes meshing in v1 — the voxel-shell + 2D slice panel
  is the visual language. Revisit only if the shell proves inadequate.
- No STL export of the warped part in v1 (candidate follow-up, not a blocker).

---

## 7. Risks

- **F0 is a real gate.** If the solver venv is missing/broken in this environment, runs fail
  silently and F1 alone won't fix the perceived "no results." Resolve F0 before committing to
  the rest.
- **Slice transport size.** Serving many JSON slices for larger grids (64³) could be heavy;
  PNG-per-slice or on-demand fetching mitigates. Decide in planning.
- **Shared solver copy.** `heatr3d.py` is a synced copy of the dissertation solver. F4 changes
  must land in a way that respects that sync boundary (edit the canonical source + re-sync, or
  keep displacement in the job wrapper) — decide in planning, do not silently fork.

---

## 8. Build order

Phase 1 (F0 → F1 → F2) → Phase 2 (F3, then F4 → F5) → Phase 3 (F6, F7). Each feature is
independently shippable behind the existing page; nothing here requires a big-bang merge.
