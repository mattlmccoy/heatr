# Deck figures: true 3-D solve visuals

Conference-deck figures rendered ONLY from committed artifacts of the Phase A
solve3d close-out and the D1 spike. No physics was run to produce anything in
this directory; every script is a deterministic re-render of saved fields plus
the shape gate's own resampling rule. Style follows the deck's solve-visual
language (near-black background, Space Mono, inferno, cyan nominal outlines;
same palette as deck_gifs/src/style.py). All PNGs are 200 DPI.

Rebuild any figure with:

    deck_figures_3d/.venv/bin/python deck_figures_3d/figN_*.py

(the local .venv is gitignored; recreate with python3 -m venv and
`pip install numpy scipy matplotlib scikit-image`).

---

## fig1_dense_inside_bounds.png

Deck caption: "The 3-D solve melts the part where it should: the circle spills
nothing, the square leaks 4 percent through its faces."

Data provenance: `solve3d/results/anchor_heatr3d_circle_n96.npz` and
`anchor_heatr3d_square_n96.npz` (heatr3d melt-onset T_phi90 + part mask,
n=96 anchors of the Phase A Task-4 reference runs). Spill agreement numbers
(3.81 vs 3.85 percent) from `solve3d/results/phase_a_shape_gate.json`
(square_off bed melt fractions, both engines, shared gate grid).

Honest scoping: quarter cutaway of a 20 mm mid slab of the full-height
extrusion (the melt-onset field is z-invariant; plane spread 0.09 to 0.13 C).
The spill fraction printed on the figure (4.3 percent) is the voxel-grid count
on this anchor; the shared-gate-grid values are 3.81 / 3.85 percent, quoted in
the footer. MEASURED CORRECTION to the task brief: the square's out-of-part
melt is NOT at the corners; it is a 1-2 voxel skin outside the two x-normal
faces (x = +-10.3 to +-10.9 mm, |y| < 5.4 mm). The figure says "face spill"
for that reason. heatr3d fields only; the dolfinx arm agrees on the amount
(gate) but is not drawn here.

## fig2_eqs02_correction.png

Deck caption: "One stencil fix moved the RF power from a false surface skin
into the part interior, at identical total power."

Data provenance: `heatr3d_d1_spike/d1_circle_coarse.npz` (q_heatr3d = shipped
legacy Q_rf, q_heatr3d_maskgrad = EQS-02 corrected Q_rf, 812 mid-plane
evaluation points, n=64-matched; the exact pair `EQS02_IMPACT.md` reports:
max/mean 12.25 vs 1.86). "Skin steals 74 percent" and "interior power up
2.59x" are the recorded circle values in `heatr3d_d1_spike/EQS02_IMPACT.md`
(power fraction in surface band 0.737; interior absolute mean rises 2.59x).

Honest scoping: one z plane of the extruded circle (field is z-invariant,
max|Ez|/mean|E| = 6.5e-7), displayed as a 3-D relief on a refined
triangulation of the 812 saved points; color is clipped at 3x the mean so the
corrected panel's topology is visible, heights are unclipped. The corrected
DRIVE is near-uniform (not "interior-hot"; interior-hot is the resulting
temperature topology). Circle only; the square's correction is smaller
(19.15x to 2.20x) and its sigma_T moves the OTHER way (+10.9 percent), so
this figure should not be used to claim sigma_T always improves.

## fig3_two_engines_one_front.png

Deck caption: "An independent FEM engine reproduces the voxel engine's melt
front to 0.07-0.13 mm, sub-pixel on a 20 mm part."

Data provenance: dolfinx fronts from `solve3d/results/eval_dolfinx_circle_off.npz`
and `eval_dolfinx_square_off.npz` (melt-onset T on the shared 0.15 mm
evaluation grid, mid z plane); heatr3d fronts from the n=96 anchor npz files,
trilinearly resampled onto the same grid exactly as `solve3d/shape_gate.py`
does; front symmetric-surface-distance numbers printed on the figure from
`solve3d/results/phase_a_shape_gate.json`
(arms.*.metrics.front_ssd_mm_phi0p9: 0.129, 0.123, 0.082, 0.067 mm).

Honest scoping: coupling-off arms drawn; the coupled arms' numbers are inside
the quoted 0.07-0.13 mm range. The gate averages over five z planes; the
drawing shows the mid plane (plane-to-plane front spread 0.002-0.008 mm).
Front agreement is the SHAPE gate that passed; sigma_T and the heating-curve
diagnostics still fail their cross-family bands (PHASE_A_REPORT C5) and are
not shown here.

## fig4_layer_stack.png

Deck caption: "One 3-D simulation read layer by layer: the temperature field
through the build, with the melt front and nominal bounds on every plane.
Every printed layer is a slice of one simulated field."

REVISED 2026-08-02 (Matt's feedback + honesty fix): color field is now
TEMPERATURE (70-235 C, melt window 175-185 marked on the colorbar) - the phi
version was saturated at 1.0 across the interior and showed no gradient.
Title says SIMULATION, not solve (no 3-D solve exists yet; footer carries
"the 3-D dopant solve is in progress"). The phi = 0.9 front stays as the
dashed overlay.

Data provenance: `solve3d/results/eval_dolfinx_square_off.npz` (T on the five
exported z planes at z = -20, -10, 0, +10, +20 mm, 200x200 at 0.15 mm,
committed Phase A close-out field export). Melt front via the shared
`phase_fraction_phi` (same conversion as `solve3d/gates.py`).

Honest scoping: the five planes are the close-out's exported sample of a
40 mm full-height extrusion, not a 0.1 mm print-layer stack; the vertical
spacing is exploded for display (real planes are 10 mm apart) and the figure
says so. The visible melt beyond the cyan bounds on the faces is the same
real ~3.85 percent face spill as fig1, seen layerwise.
