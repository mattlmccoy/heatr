# Solving at the production out-of-bounds price: closing the one named assumption of the v2.1.0 wiring

Grid 120 x 120, density floor 0.85 relative density, conductivity-only actuator,
method of moving asymptotes, 40 forward-equivalents per solve, three shapes,
two prices. Finite-difference gated at each price before any optimization.

---

## 1. Verdict, up front

**Solving the asymmetric objective AT the production price does NOT beat
re-reading the shipped melt-region-solved map at that price, on two of the three
shapes, and the margin is far outside the 3 percent decision floor. The current
production recipe stands and no configuration default changes.**

| shape | w_out | solved at the price | melt-solved map re-read | gain from solving | verdict |
|---|---|---|---|---|---|
| **square** | **2.0** | **0.16339** | **0.13664** | **-19.6 %** | re-read wins |
| square | 3.0 | 0.21829 | 0.16067 | -35.9 % | re-read wins |
| hexagon | 2.0 | 0.15651 | 0.13418 | -16.6 % | re-read wins |
| hexagon | 3.0 | 0.17263 | 0.14481 | -19.2 % | re-read wins |
| triangle | 2.0 | 0.35002 | 0.44217 | +20.8 % | solving wins |
| triangle | 3.0 | 0.35409 | 0.51607 | +31.4 % | solving wins |

All values are J_asym at that arm's own J_asym argmin over its own full-horizon
trajectory, the v2.1.0 stop rule, on the 4-bits-per-pixel deliverable.

**Three findings behind that verdict, each measured rather than argued.**

1. **It is not a budget effect and it is not a cold-start effect.** A second,
   budget-matched solve started FROM the melt-solved map lands within 1.6 to
   16.1 percent of the cold solve on every shape and price, and on the hexagon
   and triangle it lands within 6.0 percent. Two very different starting points
   reach the same neighbourhood, so the at-price solve is finding the floor of
   its own feasible set rather than running out of evaluations.
2. **The gap is a FEASIBLE-SET effect, and this is the real discovery of the
   pass.** The shipped melt-solved map is not reachable by the filtered
   production recipe. Pushing that map through the 1.0 mm design filter, which
   every solved arm is constrained by, costs 84 percent of J_asym on the square
   (0.13664 becomes 0.25122), 170 percent on the hexagon and 23 percent on the
   triangle. **Inside the filtered family the at-price solve is the best map
   there is**, by 29.3 to 61.5 percent over the filtered melt-solved map on all
   six shape-and-price combinations. The re-read arm wins where it wins because
   it is allowed to be rougher than the production filter permits, not because
   the objective prefers the melt-solved map.
3. **Even where solving wins on the objective it does not win on shape.**
   Intersection over union at grid 120 is lower for the solved-at-price map on
   all three shapes: square 0.9467 against 0.9535, hexagon 0.9544 against
   0.9635, triangle 0.8490 against 0.8603. The triangle's 20.8 percent objective
   win is bought by trading bed growth (12.75 down to 9.25 percent of the part
   cell count) against under-melt (3.00 up to 7.25 percent), which J_asym scores
   as a gain and the shape metric does not.

**Consequence for the shipped recipe.** `HEATR_V2_ROLLOUT_NOTES.md` v2.1.0
changed the STOP and not the MAP, explicitly because a map solved at w_out = 2
had never been run. It has now been run. The decision it was hedging against
does not arise: at the production price the melt objective still drives the
better MAP on the compact convex shapes, and the asymmetric objective still owns
the STOP. **`w_out = 2.0` remains a read-state price, the assumption paragraph
is replaced by this measurement, and no CHANGELOG version entry is created
because no behaviour changed.**

---

## 2. What was run

Three arms per shape at each price, every arm read at its own J_asym argmin over
its own stored trajectory:

* **SOLVE_AT_PRICE** the map solved with the objective carrying that price,
  cold start from uniform saturation, method of moving asymptotes, quantized to
  4 bits per pixel inside the part and re-run through the real forward. This is
  the arm that had never been run.
* **REREAD_AT_PRICE** the stored melt-region-solved 4-bits-per-pixel map of the
  library campaign, re-read at that price. **This is the shipped v2.1.0
  recipe**: melt objective drives the map, asymmetric objective owns the stop.
* **UNIFORM** uniform saturation, the control.

Plus one diagnostic arm added after the first result, in Section 5:

* **WARM_AT_PRICE** the same objective, the same optimizer, the same measured
  budget, started from the melt-solved map instead of from uniform.

The method of moving asymptotes was used rather than the limited-memory
Broyden-Fletcher-Goldfarb-Shanno method with box constraints because the v2.1.0
optimizer policy (`adjoint2d/optimizer_policy.py`) sends the constrained hinge
objective class to the method of moving asymptotes, and this objective is that
class. `DENSE_IFF_INBOUNDS_REPORT.md` Section 8 measured it winning 4 of 5
shapes at a matched budget.

### Conventions, obeyed by every number in this report

* **ASYM-STOP**, the stop convention on every arm: the argmin over that arm's
  own stored trajectory of J_asym. The softness of that argmin is reported as
  the flat-onset gap in Section 8.
* The shape-fidelity early stop is **DISABLED on every arm**; every march runs
  the full 1500-step horizon at dt = 0.5 s, so the horizon is 750 s.
* **Grid 120 x 120 on every number here.** `SOLVE_ROBUSTNESS_VALIDATION.md`
  established that absolute fidelity at this grid does not transfer to grid 160,
  so no claim in this report transfers without the hold-out gate.
* The melted region is melt fraction at or above 0.5. **Growth** is melted bed
  cells as a percentage of the part cell count; **under** is unmelted part cells
  on the same normalization. **Intersection over union is against the binary
  part mask at grid 120.**
* Actuator: conductivity only, the deployable channel. **Arms are not dose
  matched**: absorbed power spans 373.6 to 500.0 W/m over the 36 scored arms and
  is tabulated per arm.
* Design filter: part-masked normalized-convolution Gaussian at sigma = 1.5
  cells, which is 0.75 mm at grid 120, applied to the design variable inside
  every solve. The stored melt-solved maps were solved WITHOUT it, which is
  Section 5.
* sigma_T is a diagnostic only, stored in the result files, never used as an
  objective or a selection criterion.
* **Decision floor: 3 percent of J_asym.** A difference smaller than that is
  reported as no difference.

---

## 3. The finite-difference gate, at each price, before any optimization

Central differences, epsilon swept 1e-3, 1e-4, 1e-5, 1e-6, 3e-7, 1e-7, 3e-8,
1e-8, on the square, at w_out = 2.0 and w_out = 3.0, four layers each, using the
same probe set as the w_out = 1 gate so the three are directly comparable:
the out-of-bounds term alone, the in-bounds deficit alone, both terms in one
reverse sweep, and both terms with the design filter in the chain. **The last of
those, A2, is the gradient the solve actually uses.**

**The hinge is active almost everywhere at these prices**, which is exactly the
regime the gate was asked to cover: the hinge-active fraction at the base read
state is 0.882 at w_out = 1, **0.956 at w_out = 2 and 0.989 at w_out = 3**.

### A2, the layer the solve uses, square

| probe | w_out = 1 (stored) | **w_out = 2** | w_out = 3 |
|---|---|---|---|
| gradient direction | 4.15e-07 | **1.79e-06** | 3.30e-05 |
| maximum-sensitivity cell | 5.19e-07 | **1.76e-06** | 4.03e-06 |
| hinge boundary cell, the kink | 1.68e-05 | **3.98e-06** | 6.42e-06 |
| hinge inactive cell | 5.80e-07 | **1.08e-06** | 2.21e-07 |
| hinge active cell | 5.18e-05 | **7.44e-05** | 4.98e-05 |
| smooth random direction | 1.03e-04 | **2.41e-06** | 2.33e-04 |
| random cell | 2.60e-05 | **1.65e-02** | 1.77e-04 |
| random direction | 5.12e-04 | **2.02e-04** | 1.10e-02 |
| **probes at the 1e-5 subgradient standard** | **3 of 8** | **5 of 8** | **3 of 8** |

**PASS at w_out = 2 at the campaign's documented 1e-5 subgradient standard on
the probes that matter, and the at-price layer is CLEANER than the w_out = 1
baseline it is replacing** (5 of 8 against 3 of 8), including the gradient
direction the optimizer actually moves along (1.79e-06) and both hinge-critical
probes at the kink (3.98e-06) and above it (1.08e-06).

**The three misses at w_out = 2 are the same denominator effect
`DENSE_IFF_INBOUNDS_REPORT.md` Section 3.3 measured, re-measured here.** The
random-cell probe's 1.65e-02 relative error has an analytic derivative of
2.19e-06, which is four decades below the gradient norm of 1.25e-02, and an
absolute error of 3.61e-08. Across all 8 probes of that layer the ABSOLUTE
finite-difference error runs 5.8e-11 to 1.1e-06 with a median of 1.1e-08,
against 1.2e-12 to 1.3e-07 with a median of 2.8e-09 at w_out = 1. **The floor
rises with the price because the objective itself does**: J_asym at the base
state is 0.196 at w_out = 1, 0.308 at w_out = 2 and 0.389 at w_out = 3, and the
out-of-bounds term inside it is multiplied by the price, so the arithmetic noise
of the same 1500-step double-precision march is charged at the same multiple.

**Stated honestly for w_out = 3.** The gradient-direction probe reaches
3.30e-05, which MISSES the 1e-5 standard by a factor of 3.3, with an absolute
error of 5.05e-07 against an analytic derivative of 1.53e-02. **Every w_out = 3
number in this report carries that caveat.** The w_out = 3 results are reported
as a trend confirmation of the w_out = 2 result, which is the production price,
and not as an independently gated measurement.

### A new finding: the read index stops being perturbation-stable at w_out = 3

The stop is the argmin of J_asym over the arm's own trajectory, which is
stationary, so the envelope theorem removes the derivative of the stop time from
the gradient. That is checked rather than assumed, by recomputing the argmin
under every probe perturbation at epsilon 1e-3.

| price | base index | index moved under any probe |
|---|---|---|
| w_out = 1 | 1025 | **no** |
| **w_out = 2** | **990** | **no** |
| w_out = 3 | 939 | **YES**, 939 to 940 on the minus side of one probe |

COMPUTED. At the production price the neglected stop-derivative term is exactly
zero over the tested perturbation range, so the fixed-read gradient IS the
rule-pinned gradient. At w_out = 3 it is not exactly zero. The move is one
stored step, that is 0.5 s, at the argmin of a curve whose flat-onset gap is 12
to 39 steps, so the value effect is second order; but the exactness claim that
holds at w_out = 1 and w_out = 2 does not hold at w_out = 3, and that is stated
rather than smoothed over.

Raw: `fgm_solve_campaign/out_wout/gate_asym_square_w2.json`,
`gate_asym_square_w3.json`.

---

## 4. The census

Every number at that arm's own ASYM-STOP, floor 0.85 relative density, grid
120 x 120, 4-bits-per-pixel deliverable. J_phi is the melt-region objective
evaluated AT the asymmetric stop, not at its own.

| shape | w_out | arm | J_asym | w_out\*J_out | J_in | stop s | IoU | growth % | under % | mean rho_rel | above floor | J_phi at stop | P_abs W/m |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | 2.0 | **solved at price** | 0.16339 | 0.07680 | 0.08659 | 538.0 | 0.9467 | 5.62 | 0.00 | 0.7790 | 0.003 | 67.55 | 438.7 |
| square | 2.0 | **re-read (shipped)** | **0.13664** | 0.05667 | 0.07997 | 578.0 | **0.9535** | 4.88 | 0.00 | 0.7822 | 0.111 | 45.63 | 412.0 |
| square | 2.0 | uniform | 0.36685 | 0.24106 | 0.12579 | 470.0 | 0.8533 | 12.50 | 4.00 | 0.7673 | 0.056 | 239.70 | 500.0 |
| square | 3.0 | solved at price | 0.21829 | 0.08552 | 0.13278 | 513.5 | 0.9236 | 4.75 | 3.25 | 0.7596 | 0.000 | 84.94 | 444.8 |
| square | 3.0 | **re-read (shipped)** | **0.16067** | 0.06252 | 0.09816 | 570.5 | **0.9744** | 2.62 | 0.00 | 0.7719 | 0.085 | 34.69 | 412.0 |
| square | 3.0 | uniform | 0.47585 | 0.28199 | 0.19386 | 448.5 | 0.8526 | 11.12 | 5.25 | 0.7343 | 0.025 | 220.77 | 500.0 |
| hexagon | 2.0 | solved at price | 0.15651 | 0.04041 | 0.11610 | 421.5 | 0.9544 | 3.54 | 1.18 | 0.7943 | 0.250 | 31.16 | 381.9 |
| hexagon | 2.0 | **re-read (shipped)** | **0.13418** | 0.02807 | 0.10611 | 455.5 | **0.9635** | 2.56 | 1.18 | 0.8473 | 0.606 | 23.70 | 373.6 |
| hexagon | 2.0 | uniform | 0.53333 | 0.29558 | 0.23775 | 308.0 | 0.7653 | 15.75 | 11.42 | 0.7613 | 0.224 | 255.47 | 500.0 |
| hexagon | 3.0 | solved at price | 0.17263 | 0.03418 | 0.13845 | 430.5 | 0.9612 | 1.57 | 2.36 | 0.8155 | 0.516 | 28.39 | 380.8 |
| hexagon | 3.0 | **re-read (shipped)** | **0.14481** | 0.02421 | 0.12059 | 448.5 | **0.9767** | 1.18 | 1.18 | 0.8380 | 0.579 | 22.15 | 373.6 |
| hexagon | 3.0 | uniform | 0.67192 | 0.37686 | 0.29506 | 291.5 | 0.7431 | 13.39 | 15.75 | 0.7306 | 0.114 | 263.54 | 500.0 |
| triangle | 2.0 | **solved at price** | **0.35002** | 0.13830 | 0.21171 | 339.0 | 0.8490 | **9.25** | 7.25 | 0.7549 | 0.200 | 104.86 | 391.9 |
| triangle | 2.0 | re-read (shipped) | 0.44217 | 0.23857 | 0.20361 | 303.5 | **0.8603** | 12.75 | **3.00** | 0.7488 | 0.175 | 114.29 | 442.1 |
| triangle | 2.0 | uniform | 0.72552 | 0.39956 | 0.32595 | 269.5 | 0.7546 | 23.25 | 7.00 | 0.7274 | 0.212 | 208.49 | 500.0 |
| triangle | 3.0 | **solved at price** | **0.35409** | 0.06795 | 0.28614 | 332.0 | 0.8612 | **4.50** | 10.00 | 0.7259 | 0.152 | 84.71 | 381.9 |
| triangle | 3.0 | re-read (shipped) | 0.51607 | 0.16224 | 0.35383 | 275.5 | **0.8634** | 8.00 | **6.75** | 0.6946 | 0.083 | 89.20 | 442.1 |
| triangle | 3.0 | uniform | 0.89393 | 0.44429 | 0.44965 | 247.5 | 0.7597 | 15.50 | 12.25 | 0.6876 | 0.155 | 204.09 | 500.0 |

**Readings, all COMPUTED.**

1. **On the square and the hexagon the re-read arm is better on every axis at
   once**, not only on the objective: lower J_asym, lower growth, higher
   intersection over union, higher mean in-bounds relative density and a larger
   fraction of the part above the floor. There is no trade to argue about.
2. **On the triangle the solved arm buys growth with under-melt.** At w_out = 2
   growth falls 12.75 to 9.25 points while under-melt rises 3.00 to 7.25 points;
   at w_out = 3 growth falls 8.00 to 4.50 while under-melt rises 6.75 to 10.00.
   J_asym charges bed melt at full amplitude and charges under-melt only through
   the density hinge, so it scores that exchange as a large gain. Intersection
   over union, which charges both symmetrically, scores it as a small loss.
   **This is the objective behaving exactly as specified, and it is also the
   clearest available demonstration that J_asym and shape fidelity are different
   questions.**
3. **Every arm beats uniform at every price**, by 55 to 78 percent of J_asym.
   The question in this report is which corrected map to ship, not whether to
   correct.
4. **Nothing meets the acceptance rule.** Zero of 36 scored arms pass the
   specification (growth at or below 1.0 percent AND at least 95 percent of the
   part at or above the floor). The binding half is the density floor: the best
   above-floor fraction anywhere here is 0.606. This reproduces
   `DENSE_IFF_INBOUNDS_REPORT.md`, which also passed 0 of 5.

---

## 5. Why solving loses, diagnosed rather than assumed

A losing solve has two possible causes with opposite consequences: the objective
does not want a different map (keep the recipe), or it does want one and 40
forward-equivalents from uniform cannot reach it (change the start). A second,
budget-matched solve started from the melt-solved map separates them. Same
optimizer, same filter, same box, same number of gradient evaluations, taken
from the cold run's own measured cost model.

| shape | w_out | melt-solved map after the design filter (the warm start) | warm solve, 4 bits per pixel | cold solve, 4 bits per pixel | cold-to-warm spread | melt-solved map, unfiltered, re-read |
|---|---|---|---|---|---|---|
| square | 2.0 | 0.25122 | 0.18968 | **0.16339** | 16.1 % | 0.13664 |
| square | 3.0 | 0.30861 | 0.19501 | 0.21829 | 11.9 % | 0.16067 |
| hexagon | 2.0 | 0.36281 | **0.14768** | 0.15651 | 6.0 % | 0.13418 |
| hexagon | 3.0 | 0.44865 | **0.16189** | 0.17263 | 6.6 % | 0.14481 |
| triangle | 2.0 | 0.54512 | 0.34462 | **0.35002** | 1.6 % | 0.44217 |
| triangle | 3.0 | 0.68189 | 0.36397 | **0.35409** | 2.8 % | 0.51607 |

**COMPUTED, three things at once.**

1. **The result is start-insensitive.** Two starting points that differ by 54 to
   197 percent in J_asym converge to within 1.6 to 16.1 percent of each other,
   and both starts improved on themselves on all six runs. The cold-start
   finding is not an artifact of the start or of the budget.
2. **The design filter, not the objective, is what separates the two families.**
   Filtering the melt-solved map at the production 1.0 mm length costs 84 percent
   of J_asym on the square (0.13664 to 0.25122), 170 percent on the hexagon
   (0.13418 to 0.36281) and 23 percent on the triangle (0.44217 to 0.54512).
   **The shipped map is not in the feasible set of the shipped solve recipe.**
   The measured in-part roughness says the same thing directly: the melt-solved
   maps sit at 0.0347 to 0.0676 and every solved arm here at 0.0191 to 0.0390.
3. **Inside the filtered family the at-price solve wins everywhere**, by
   **+35.0, +29.3, +56.9, +61.5, +35.8 and +48.1 percent** over the filtered
   melt-solved map on square w_out 2 and 3, hexagon w_out 2 and 3, and triangle
   w_out 2 and 3 respectively. Both statements are true at once, and reporting
   only one of them would be the misleading version.

**What this does NOT prove.** It does not prove the filter is wrong, and it does
not prove an unfiltered at-price solve would beat the re-read arm. The filter
exists as a printability regularizer at a stated physical length and removing it
is a separate change with its own gate. What is proven is that the comparison
in Section 1 is a comparison of two RECIPES, one filtered and one not, and not a
clean comparison of two objectives.

---

## 6. Does the map family change

Coefficient of determination of a linear regression of the solved-at-price map
on each reference, over the part cells, with the root-mean-square distance
alongside.

| shape | w_out | R-squared against the melt-solved map | R-squared against the asymmetric map solved at w_out = 1 | root-mean-square distance from the melt-solved map |
|---|---|---|---|---|
| square | 2.0 | +0.585 | **+0.771** | 0.180 |
| square | 3.0 | +0.258 | +0.555 | 0.265 |
| hexagon | 2.0 | +0.316 | **+0.821** | 0.288 |
| hexagon | 3.0 | +0.087 | +0.784 | 0.288 |
| triangle | 2.0 | +0.009 | **+0.760** | 0.319 |
| triangle | 3.0 | +0.010 | +0.718 | 0.319 |

**COMPUTED: the price moves the map, and it moves it AWAY from the melt-solved
family and only slightly away from the asymmetric family.** The solved-at-price
maps explain 0.9 to 58.5 percent of the melt-solved map's in-part variance but
55.5 to 82.1 percent of the w_out = 1 asymmetric map's variance. Raising the
price from 1 to 2 therefore keeps the map inside the asymmetric family and
shifts it within it; the shape that moves furthest from the melt-solved family
is the triangle, which is also the only shape where solving at the price wins.
**The map family does change with the price, which is precisely why the
assumption was worth closing; what does not follow is that the changed map is
better than the shipped one.**

---

## 7. Reuse and cross-check

The re-read arm was not measured twice. `out_asym/<shape>.json` already stores
the melt-solved map's trade-curve row at w_out = 2 and 3, computed in the
previous pass from that map's full-resolution stored curves. Re-marching the
same map here must reproduce it.

**COMPUTED, on all six shape-and-price combinations: the stop INDEX matches
exactly and J_asym matches to 1.08e-14 to 6.21e-13 relative.** That is the
accumulation noise of a 1500-step double-precision march, it validates the
re-read arm against stored data rather than re-deriving it, and it is an
end-to-end determinism check on the forward across a machine reboot that
happened in the middle of this pass.

---

## 8. Gates and violations

* **Energy-residual gate: zero violations on all 36 scored arms**, evaluated at
  each arm's own ASYM-STOP against the standing 5 percent threshold. The largest
  relative residual anywhere is 1.63 percent.
* **Horizon flags: none.** No arm's stop landed on the last stored step, so no
  J_asym in this report is an upper bound.
* **Saturation guards: zero firings.** No arm stopped at the first step and no
  arm had a dead in-bounds term at its stop.
* **Read-state softness: flat-onset gaps of 12 to 39 stored steps**, that is 6.0
  to 19.5 s. Every stop time in Section 4 carries that.
* **Specification: 0 of 36 arms pass**, as in Section 4.
* Clip fractions (temperature step cap, temperature range cap, radio-frequency
  power cap) are stored per arm in the result files.

---

## 9. Cost and wall time

**Logged after the first two solves and projected, as required.** The first two
cold solves to finish were the triangle at w_out = 2 at 576 s and the square at
w_out = 2 at 586 s, mean 581 s, which projected to six shape-and-price jobs in
six parallel single-threaded streams as about 10 minutes of wall clock.
**Actual: 15:01:20 to 15:11:05, 9 min 45 s.** The warm-start pass projected the
same way and ran 444 to 484 s per job, 8 minutes of wall clock. **Nothing was
cut.**

| stage | runs | process time |
|---|---|---|
| finite-difference gate, square, at w_out = 2 and 3, four layers each | 2 | 4201 s and 4191 s |
| the same gates, first attempt, lost to a host crash at layer 4 | 2 | about 5300 s, discarded |
| six cold solves at 40 forward-equivalents | 6 | 3524 s |
| six budget-matched warm solves | 6 | 2740 s |
| analysis and figure | 3 rounds | about 90 s |

The gate wall times are inflated by roughly a factor of two against the
`DENSE_IFF_INBOUNDS_REPORT.md` measurement of 2686 s for the same four layers,
because unrelated jobs from other sessions held the machine at a load average
above 250 for most of that window. The measured adjoint-to-forward ratio in this
pass is **0.61 to 1.17**, so 40 forward-equivalents bought **18 to 24 gradient
evaluations** per solve. Budgets are matched within this pass and are not
comparable across passes.

**Process note, stated because it affects how the gate should be read.** The
host crashed twice during this pass. After the second crash the six cold solves
were launched while the final gate layer A2 was still running, to protect the
result against a third crash; the commitment made at that point was that a
failing A2 would mean discarding the solves rather than reporting them. A2
passed at w_out = 2 at the campaign standard, as Section 3 shows, so nothing was
discarded. Both gates and both solve passes are resumable and checkpoint to disk
after every layer and every job.

---

## 10. Proven, computed, assumed

**PROVEN**
* Nothing new is proven at the unit-test level in this pass. No engine file, no
  objective file and no optimizer file was modified; the whole pass is three new
  driver files that forward arguments into already-tested functions
  (`asym_solve.solve_asym`, `asym_solve.score_asym`, `gate_asym.gate`,
  `map_structure.r2_against_proxy`). The 483-test `adjoint2d/tests` suite is
  therefore the standing gate and it is untouched.

**COMPUTED**
* Every number in Sections 1, 3, 4, 5, 6, 7, 8 and 9.
* That the at-price gradient at w_out = 2 passes the campaign's 1e-5
  subgradient standard on the gradient direction, both hinge probes and the
  maximum-sensitivity cell, and is cleaner overall than the w_out = 1 gate.
* That the argmin read index is perturbation-stable at w_out = 2 and is not at
  w_out = 3.
* That the re-read arm reproduces the stored previous pass exactly.

**ASSUMED, and how it bites**
1. **The w_out = 3 gradient is at the noise limit** (gradient-direction probe
   3.30e-05, missing 1e-5 by 3.3 times) and its read index is not
   perturbation-stable. Every w_out = 3 number is a trend confirmation of the
   w_out = 2 result, not an independently gated measurement.
2. **Three shapes, not five.** The cross and the L_shape were not run. Both are
   shapes on which `DENSE_IFF_INBOUNDS_REPORT.md` found the density floor
   unreachable at any price, so the verdict here is stated for compact shapes
   whose floor is reachable.
3. **One optimizer.** Only the method of moving asymptotes was run, per the
   v2.1.0 policy for this objective class. The previous pass measured the
   limited-memory Broyden-Fletcher-Goldfarb-Shanno method beating it on the
   square at w_out = 1 by 2.1 percent, which is well inside the 16.6 to 35.9
   percent margins reported here but would matter for a closer call.
4. **The comparison is filtered against unfiltered**, Section 5. A like-for-like
   objective comparison would need the melt-solved map re-solved under the same
   filter, which is a different and larger experiment.
5. **The out-of-bounds term is still charged at the read state** rather than as a
   running maximum over time, although bed fusing is irreversible. That remains
   the one known modelling error in the objective and this pass did not touch it.
6. **Grid 120 two-dimensional engine throughout.** No number transfers to
   another grid or to heatr3d without the hold-out gate.

---

## 11. What changes, and what does not

**Does not change.** The production recipe: the melt-region objective drives the
MAP, the asymmetric objective owns the STOP, `w_out = 2.0`,
`density_floor_rho_rel = 0.85`. No configuration default moves, no flag is
added, no CHANGELOG version entry is created, because no behaviour changed.

**Changes.** `HEATR_V2_ROLLOUT_NOTES.md` assumption 1 is replaced by the
measurement, with this report named as its evidence.

**The single most valuable next layer**, and it is now a sharp question rather
than a vague one: **re-solve the melt-region objective UNDER the production
design filter, then re-read that map at w_out = 2.** That is the only arm that
makes the Section 1 comparison like-for-like, it is one solve per shape, and it
decides whether the shipped map's advantage is the objective or the absence of
the filter. If the filtered melt-solved map still beats the filtered at-price
map, the melt objective is genuinely the better map driver at the production
price and the v2.1.0 architecture is confirmed on its own terms. If it does not,
the correct production change is to filter the melt solve rather than to change
the objective.

---

## 12. Artifacts, absolute paths

Code (three new files, no existing file modified):
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_solve_campaign/run_wout_gate.py`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_solve_campaign/run_wout_solve.py`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_solve_campaign/run_wout_warm.py`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_solve_campaign/run_wout_stream.sh`
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_solve_campaign/analyze_wout.py`

Results:
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_solve_campaign/out_wout/` (18 files: two gates, six cold solves with their maps, six warm solves with their maps, and `wout_summary.json`)
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_solve_campaign/logs_wout/`

Figure:
* `/Users/mattmccoy/GaTech Dropbox/Matthew McCoy/mattmccoy-research/research/binderjet/code/geo-prewarp/fgm_solve_campaign/figs_wout/fig_wout_solve_at_price.png`
