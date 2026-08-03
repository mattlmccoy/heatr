# Deck GIFs: the FGM solve story

Presentation GIFs for the slide deck, produced 2026-08-01. FGM = functionally
graded material. All animation is driven by re-running the campaign's own
physics (read-only imports of `fgm_solve_campaign/adjoint2d`); every number
printed on a GIF is either a stored campaign value (cited below) or a fresh
value from the re-run and is labeled as such here.

Style: near-black background (#0e1013), Space Mono for all text (registered
from `~/Library/Fonts/SpaceMono-*.ttf`; it is NOT in the matplotlib system
list, the render scripts add it explicitly), inferno for temperature, magma
for relative density, viridis for dopant saturation, thin cyan nominal
outline, white dashed melt front. 12 frames per second.

## gif_dwell_cross.gif (2.9 MB, 11.2 s)

The turntable story. The cross marches in the part frame under its solved
asymmetric dwell program: left temperature, right relative density (masked to
the part), turntable dial showing the commanded position, and the program
timeline with a moving cursor. The loop ends AT the recommended stop, on the
crisp cross; it never runs into oversinter.

- Program and stop: `fgm_solve_campaign/out_dwell/cross_turntable_deliverable.json`
  (150 moves, positions 0/90/180/270 deg, 5 s holds, 20 s cycle,
  `recommended_stop_s` 471.5).
- Dopant map: key `sat_D_refined_4bpp_discovered_lib` in
  `out_dwell/cross_dwell_maps.npz`.
- Caption numbers J_phi 34.2 and IoU 0.985: arm `D_refined_4bpp_discovered_lib`
  in `out_dwell/cross_dwell.json`.
- VERIFICATION: the re-run march's per-step J_phi curve matches the stored
  campaign curve `J_curve_D_refined_timeresolved_discovered_lib` with max abs
  difference 1.6e-4 (J at the stop 34.0380 vs stored 34.0380). Printed by
  `src/c1_dwell_march.py`, saved in `cache/c1_dwell_cross.npz`.

## gif_solve_vs_inversion.gif (2.3 MB, 12.4 s)

The thesis beat, on the square. Left INVERT: the best stored historical
proportional-inverse 4-bits-per-pixel mask, one static guess, whose melt front
then runs to its own stop. Right SOLVE: the dopant map evolving across 15
adjoint optimizer evaluations (best-so-far map shown; the sidebar objective
panel shows raw evaluations as dots and the best-so-far line), then its melt
front runs to its own stop.

- Invert arm: `HIST_best` in `fgm_solve_campaign/out_lib/square.json`, mask
  `outputs_eqs/fgm_calibrated_control/runs/square/map_m0p5477/fgm_baseline_T_phi90_4bpp_mag0p55.npz`,
  re-run in its stored permittivity-co-varying channel. The re-run reproduces
  the stored stop and objective exactly (stop index 778, J 12.77).
- Solve arm: FRESH small filtered adjoint solve (design filter at the
  campaign's 1.0 mm radius, box [0, 1], budget 15 gradient evaluations),
  `src/c2_solve_vs_invert.py`. Its end numbers J_phi 26.0, IoU 0.996 are from
  this fresh run, not from the stored library campaign (which used a deeper
  unfiltered solve).
- Honesty note: on the square the historical mask is itself already
  essentially solved (stored J 12.8, IoU 0.998), and it edges the fresh
  15-evaluation solve on J. The GIF's message is the METHOD contrast (a fixed
  guess against the physics choosing the map, with the objective visibly
  dropping); it makes no head-to-head win claim, and each side's caption
  quotes only its own numbers.

## gif_rotation_kernels.gif (0.6 MB, 14.5 s)

The symmetry story. The part-frame radio-frequency heating kernel of the
cross at uniform saturation, morphing across rotation modes: static
(electrode-axis bands, the vertical limbs heat and the horizontal limbs
starve), continuous rotation built up angle by angle into the annular blur,
then 90 degree indexing accumulated position by position until the four-fold
pattern is restored.

- Kernels: recomputed per angle (24 angles at 15 deg, state B electrical
  fields) by `src/c3_kernels.py` from the campaign config cited in
  `fgm_solve_campaign/out_rot/cross_rotavg.json`; cached in
  `cache/c3_kernels_cross.npz`.
- IoU captions 0.546 / 0.852 / 0.870: stored campaign values, arms
  `S_uniform` and `R_uniform` in `out_rot/cross_headline.json` and
  `I90_uniform` in `out_rot/cross_index90.json` (uniform dopant map, each
  mode at its own stop).
- Display uses a gamma 0.4 power scale (stated on the colorbar) so the bulk
  pattern reads; the edge singularities saturate deliberately.

## gif_sequential_L.gif (2.5 MB, 13.8 s)

The path-dependence story, added after the sequential-dwell campaign landed.
The L_shape marches in the part frame under its DELIVERABLE sequential
program: hold 90 degrees for 465 s (phase 1, the horizontal foot melts while
the vertical upright stays cold), a quarter-turn to 0 degrees flashed on the
turntable dial, then phase 2 grows the upright while the melted foot holds.
Same layout language as gif_dwell_cross (temperature left, relative density
right, dial, two-band program timeline). The loop ends at the recommended
594.5 s stop.

- Program: `fgm_solve_campaign/out_seq/L_shape_turntable_DELIVERABLE.json`
  (arm `S_seq_cosolved_4bpp_interior_switch`). The march uses the arm's
  interior-refined switch time 465.103 s from `out_seq/L_shape_arms.json`
  plus `L_shape_kink.json`, which is what the stored curve was scored at; the
  machine-facing DELIVERABLE program snaps it to 465.0 s.
- Dopant map: key `sat_S_seq_cosolved_4bpp_interior_switch` in
  `out_seq/L_shape_seq_maps.npz`.
- Caption numbers J_phi 276.4 and IoU 0.7161 at the stop: the stored arm in
  the merged arms/kink JSONs, quoted with the campaign's honest framing (best
  on record for the L at grid 120, NOT solved; the prior best on record was
  J 396.2, IoU 0.648).
- Limb story checked against the stored snapshots: at the switch the wide
  (foot) limb is 60.6 percent melted and the narrow (upright) limb 0.7
  percent; at the stop 81.2 against 79.2 percent.
- VERIFICATION: the re-run (campaign's own `seq_dwell_march.sequential_forward`,
  read-only import) matches the stored curve
  `J_curve_S_seq_cosolved_4bpp_interior_switch` with max abs difference
  6.1e-5, and J at the stop 276.3893 equals the stored arm J 276.3893.
  Printed by `src/c4_seq_L.py`, saved in `cache/c4_seq_L.npz`.

## Reproduction

Compute stages (cache npz, resumable): `src/c1_dwell_march.py`,
`src/c2_solve_vs_invert.py`, `src/c3_kernels.py`, `src/c4_seq_L.py`.
Render stages: `src/r1_dwell_cross.py`, `src/r2_solve_vs_inversion.py`,
`src/r3_rotation_kernels.py`, `src/r4_sequential_L.py`.
Shared style: `src/style.py`.
Frame extraction for review: `src/extract_frames.py` (writes `frames/`).
Interpreter: `./.venv312/bin/python` from the repo root. Nothing in
`fgm_solve_campaign/adjoint2d` was modified.

Verification performed: extracted 6 evenly spaced frames per GIF and reviewed
them visually (composition, Space Mono rendering, no clipped labels, colorbar
sanity), plus the numeric gates above (GIF 1 J-curve match to the stored
campaign curve; GIF 2 invert re-run reproducing the stored stop and J).

## fig_solve_loop_schematic.png (static, 2026-08-01)

One-slide conceptual schematic of the filter-only adjoint solve loop, five
stages plus two exits. Rendered by `src/r4_loop_schematic.py` from STORED
artifacts only; no new forward solves were run for this figure. The Heaviside
projection stage is deliberately absent: it is retired from the production
recipe (MMA_RETEST_REPORT.md Section 1 verdict; CHANGELOG_ENGINE.md v2.0.0,
"no Heaviside projection").

Thumbnail sources, all on the square:

- Stage 1 nominal outline, stage 2 mid-march temperature, stage 3
  melt-minus-nominal residual, stage 5 filtered map, and the J sparkline:
  `cache/c2_square.npz` (the solve-arm capture documented under
  gif_solve_vs_inversion.gif above: filtered adjoint solve, 1.0 mm physical
  filter radius, conductivity-only channel, 15 gradient evaluations; its J
  values are from that fresh run, stored in the cache). Stage 2 shows
  `T_solve` at the middle snapshot; stage 3 is the melt fraction computed
  from the last `T_solve` snapshot (phi = clip((T - t_pc_c)/dt_pc_c + 0.5))
  minus the part mask, drawn in the red/blue melt-minus-target style of
  `fgm_solve_campaign/figs_topopt/fig_topopt_maps.png` row 3.
- Exit thumbnail "4 bpp production raster": key `TO_4bpp` in
  `fgm_solve_campaign/out_topopt/square_control_filteronly_maps.npz`, the
  stored production filter-only deliverable map.
- Stage 4 adjoint backward sweep is a labeled arrow, NOT a field: no stored
  dJ/ds field exists anywhere on disk (gate JSONs store scalar
  finite-difference probes only), so none is drawn.
- The numbers printed on the figure are the frozen conventions 1.0 mm
  (`adjoint2d/topopt.py` `FILTER_RADIUS_M`) and 4 bpp (the deliverable
  quantization convention), plus the qualitative cost remark "about one
  forward run," whose measured basis is 0.44 to 2.16 forward-solve
  equivalents per gradient (ADJOINT_PROTOTYPE_REPORT.md summary, line 527);
  the schematic defers the real magnitude to that report. No fidelity
  numbers are printed. (Concept-figure review 2026-08-01: PASS with this
  sources correction.)

Verification: rendered PNG viewed directly over three iterations
(composition, Space Mono rendering, no clipped or overlapping text, colormap
sanity, no em dashes).

## gif_solve_vs_inversion_hexagon.gif (3.1 MB, 13.2 s, round 2, 2026-08-01)

The head-to-head beat, on the HEXAGON, where the solve genuinely wins. Same
layout language as gif_solve_vs_inversion.gif. Left INVERT: the best stored
historical proportional-inverse 4-bits-per-pixel mask, one static guess, whose
melt front runs to its own stop. Right SOLVE: phase A sweeps the iterates of a
fresh filtered adjoint solve (budget 15 gradient evaluations, cold start,
1.0 mm filter radius) purely as a visualization of the process; phase B then
runs the STORED solved single-pass 4 bpp deliverable map to its own stop.
Because the hexagon is a genuine win, the closing frame makes the head-to-head
claim; each side's caption quotes only its own stored numbers, melt-region
framing, grid-120 qualifier in the footer.

- Invert arm: `HIST_best` in `fgm_solve_campaign/out_lib/hexagon.json`
  (source arm `hist_cal_map_m0p8327_mag0p83_outside1_eps`, mask
  `outputs_eqs/fgm_calibrated_control/runs/hexagon/map_m0p8327/fgm_baseline_T_phi90_4bpp_mag0p83.npz`,
  outside1 convention, permittivity-co-varying channel). Stored J 69.63,
  IoU 0.9211, stop index 403 (202.0 s).
- Solve arm melt segment and end numbers: stored arm `A1_4bpp` in
  `out_lib/hexagon.json`, map key `A1_4bpp` in `out_lib/hexagon_maps.npz`,
  conductivity-only channel. Stored J 22.15, IoU 0.9767, stop index 896
  (448.5 s). Head-to-head claim dJ +68.2 percent is the stored
  `verdict.dJ_rel` (0.6819).
- Display solve (phase A only): fresh 15-evaluation filtered solve by
  `src/c2h_hexagon.py`; its per-iterate J values (best reached 23.6) are shown
  in the sidebar objective panel but are NEVER captioned as end numbers.
- NUMERIC GATE: the re-run of each stored arm reproduced its stored numbers
  EXACTLY (delta 0 in all fields): HIST_best stop 403/403, J 69.6350/69.6350,
  IoU 0.921053/0.921053; A1_4bpp stop 896/896, J 22.1502/22.1502,
  IoU 0.976654/0.976654; U_uniform stop 652/652, J 251.2749/251.2749,
  IoU 0.764901/0.764901. Printed by `src/c2h_hexagon.py`, cached in
  `cache/c2h_arms.npz` / `cache/c2h_hexagon.npz`.
- VERIFICATION: 8 frames extracted and reviewed visually (phase A iterates,
  melt phase both sides, closing verdict frame, Space Mono, no clipped text).

## fig_hexagon_ungraded_vs_graded.png (static 16:9, round 2, 2026-08-01)

Two-sample comparison replacing a deck triptych: LEFT the hexagon with
UNIFORM (ungraded) dopant, RIGHT with the SOLVED graded 4 bpp map; each side
shows the end-state temperature at that arm's own optimal stop (melt front
white dashed, nominal outline cyan) plus a small dopant-map inset. Message at
a glance: same part, same power, the graded map turns a hot-cored blob into
the hexagon. Rendered by `src/r6_hexagon_static.py` from `cache/c2h_hexagon.npz`.

- Caption numbers are the STORED arms in `out_lib/hexagon.json`: U_uniform
  J 251.3 / IoU 0.7649 at stop 326.5 s; A1_4bpp J 22.2 / IoU 0.9767 at stop
  448.5 s. Fields are the gated re-runs above (delta 0 against stored).
- HONESTY NOTE / source deviation: the brief pointed at
  `outputs_eqs/geometry_dual_readstate/runs/hexagon/baseline/fields.npz` for
  the uniform side, but that snapshot does NOT correspond to the stored
  U_uniform arm (its melt region gives IoU 0.389 vs the stored 0.7649; it is a
  different, longer-cooked run). The uniform field was therefore re-run with
  the campaign's own forward to the stored stop 652 and gated (delta 0).
- VERIFICATION: PNG viewed directly over two iterations (inset size/placement
  fixed after the first look; composition, fonts, no em dashes).

## fig_solve_census_wide.png (static 16:9, round 2, 2026-08-01)

The 18-shape library census re-rendered as a denser 16:9 slide in the deck
style (IoU dumbbell left, J-change bars right, bigger labels). All numbers
read fresh from `fgm_solve_campaign/out_lib/<shape>.json`; nothing altered.
The verdict header (J: 13 of 18, IoU: 13 of 18, IoU >= 0.95: 7 of 18) is
recomputed from the JSONs and ASSERTED in `src/r5_census_wide.py` to equal
the stored verdict; the render fails if any count changes. Grid-120 qualifier
and melt-region framing in the footer. Layout content matches
`fgm_solve_campaign/figs/fig_lib_census.png`.

- VERIFICATION: PNG viewed directly over three iterations (clipped panel-B
  title and footer overlap fixed; final render clean).
