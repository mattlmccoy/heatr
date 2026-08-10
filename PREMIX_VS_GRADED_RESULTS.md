# Premix vs graded: grading beats a uniform bed floor, decisively

**Date:** 2026-08-09 · **Head-to-head in `run_sim` at the 250 °C ceiling** (each arm bisected to a
~250 °C part peak). Graded map = `adjoint2d.shape_solve` run on **jared's own geometry** (voltage
drive, shape-fidelity objective), injected into `run_sim` via `fgm_feedback.sat_map_npz_direct`.
**Reproduce:** `python -m adjoint2d.shape_solve configs/jared_exp1_40mm_adjoint.yaml jared_rect 1.2 results/premix_vs_graded`
(from `fgm_solve_campaign/`), then `python premix_vs_graded.py A15 && python make_premix_vs_graded_figure.py`.

## Conclusion

Putting the dopant **where the part needs it** (graded) beats both a uniform printed map and a uniform
premix bed floor — on fusion **and** uniformity, at the same ceiling. Premix is the worst of the three.
This closes the premix investigation: a uniform baseline is the wrong lever; **grading is the right one.**

| arm | drive→ceiling | part mean φ (fusion) | part peak/mean (uniformity) | part mean T |
|---|---:|---:|---:|---:|
| uniform (s=1, printed) | 736 W | 0.999 | 1.147 | 220.9 °C |
| **premix** (15 wt% bed floor) | 796 W | **0.632** | **1.251** (worst) | 197.8 °C |
| **graded** (adjoint optimal) | 796 W | **1.000** | **1.035** (near-perfect) | 239.2 °C |

- **Graded wins both axes:** full fusion (φ=1.000) *and* the flattest part (peak/mean 1.035, almost
  perfectly uniform), with the whole part sitting hot (mean 239 °C) just under the ceiling.
- **Premix loses both:** it under-fuses (φ=0.632) and is the least uniform (1.251) — the parasitic bed
  starves the part (see `PREMIX_SWEEP_RESULTS.md`).
- **Uniform printed** fuses (φ=0.999) but leaves a residual hot spot (1.147).

## Cross-engine agreement

The result holds in **both** forwards, independently:
- **Adjoint's own forward** (voltage, shape-fidelity J): grading drops **J 804 → 25** (A15@40), IoU
  0.712 → 0.991, part under-melt 10.2% → 0.3% vs uniform.
- **`run_sim`** (power, matched ceiling): grading gives φ 1.000, peak/mean 1.035 vs uniform's 0.999 / 1.147.

Two different solvers, two different objectives, same verdict: grading beats uniform beats premix.

## Mechanism

Grading removes conductivity from the would-be hot spots and adds it to the cold regions (adjoint map:
part sat 0.107 → 1.5, two-sided), flattening the temperature so the *whole* part reaches fusion at the
ceiling instead of one spot burning while the bulk stays cold. Premix does the opposite of useful — it
adds conductivity to the large surrounding **bed**, which parasitically absorbs the power.

## The grading win is PRINTABLE — it survives 2/4-bpp quantization (2026-08-10)

RFAM prints 2/4 bits-per-pixel, so the continuous adjoint map must be quantized. Quantizing the A15
map (`quantize_sat.py`, TDD 4/4) and re-scoring in `run_sim` at the ceiling
(`premix_graded_quantization.py`, figure `fig_graded_quantization.png`):

| graded variant | levels in part | part mean φ | part peak/mean |
|---|---:|---:|---:|
| continuous | ∞ | 1.000 | 1.035 |
| **4-bpp** | 15 | 1.000 | **1.035** (identical) |
| **2-bpp** | 4 | 1.000 | 1.072 |

- **4-bpp is indistinguishable from continuous** (1.035) — no erosion.
- **2-bpp** (only 4 levels) erodes uniformity slightly (1.035 → 1.072) but still **retains ~⅔ of the
  win** vs uniform's 1.147 and crushes premix's 1.251; fusion stays φ=1.000.
- So the grading benefit is not a continuous-map artifact — it is realizable on the actual 2/4-bpp
  printer. (A double pass realizes the s>1 cells; see `printing-constraints-and-ink`.)

## Caveats / honest scope
- The graded map is **continuous** and uses **two-sided** actuation (sat up to 1.5 = σ up to 1.5·σ_d0).
  A printed realization quantizes to 2/4 bpp and needs a double pass for s>1 (`printing-constraints-and-ink`);
  quantization erodes some of the 1.035 uniformity. The campaign's 4-bpp study covers that erosion.
- Graded map optimized for **shape-fidelity J** (adjoint), evaluated on **φ/uniformity** (run_sim) — the
  agreement across both metrics is the point, not a single tuned number.
- Comparison drive mode is power (matched ceiling); the part outcome at matched ceiling is drive-mode-
  independent (`PREMIX_SWEEP_RESULTS.md` voltage section).
- One shape (jared rectangle). The campaign's 18-shape study shows ~10–12 shapes benefit from grading
  (`search-the-chapter-before-proposing-work`).
