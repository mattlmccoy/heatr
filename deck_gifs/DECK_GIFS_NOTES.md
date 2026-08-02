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
