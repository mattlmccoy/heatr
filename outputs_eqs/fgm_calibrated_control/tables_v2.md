### Table 5. Seeded (v1) versus unseeded (v2) calibration, melt-onset sigma_T (HOLD-OUT read state), deg C

| shape | (a) uniform | (b) best-of-four | (c1) v1 seeded | (c2) v2 unseeded | c2 vs a | c2 vs c1 | m (v1) | m (v2) | new solves (v2) |
|---|---|---|---|---|---|---|---|---|
| triangle | 23.043 | 45.334 | 41.423 | 21.401 | -7.1 % | -48.3 % | 0.1090 | 6.0000 | 6 |
| H_shape | 6.562 | 11.077 | 10.996 | 10.996 | +67.6 % | +0.0 % | 0.1090 | 0.1095 | 9 |
| trapezoid | 10.842 | 16.735 | 16.124 | 16.124 | +48.7 % | +0.0 % | 0.1090 | 0.1093 | 9 |
| T_shape | 60.149 | 71.039 | 71.039 | 69.106 | +14.9 % | -2.7 % | 0.3000 | 0.1110 | 10 |
| L_shape | 58.729 | NOT_REACHED | NOT_REACHED | 74.108 | +26.2 % | n/a | NOT_REACHED | 0.1110 | 10 |
| square | 4.931 | 3.768 | 3.669 | 3.647 | -26.0 % | -0.6 % | 0.8775 | 0.9033 | 2 |
| circle | 17.428 | 7.134 | 7.117 | 7.159 | -58.9 % | +0.6 % | 0.6644 | 0.6402 | 10 |
| hexagon | 18.083 | 6.788 | 6.726 | 6.773 | -62.5 % | +0.7 % | 0.8327 | 0.8217 | 10 |
| cross | 43.771 | 31.401 | 13.858 | 14.519 | -66.8 % | +4.8 % | 1.0927 | 1.2164 | 10 |
| diamond | 32.234 | 26.637 | 15.948 | 19.005 | -41.0 % | +19.2 % | 2.1208 | 1.2164 | 10 |
| ellipse | 22.046 | 8.684 | 8.603 | 8.485 | -61.5 % | -1.4 % | 0.7429 | 0.6713 | 10 |
| equilateral_triangle | 41.480 | 43.933 | 23.310 | 23.346 | -43.7 % | +0.2 % | 1.4854 | 1.6219 | 5 |
| octagon | 18.248 | 7.045 | 6.959 | 7.002 | -61.6 % | +0.6 % | 0.4513 | 0.5052 | 8 |
| pentagon | 24.036 | 14.617 | 14.880 | 14.787 | -38.5 % | -0.6 % | 0.3000 | 0.3424 | 9 |
| rectangle | 6.971 | 2.356 | 3.888 | 3.075 | -55.9 % | -20.9 % | 0.3000 | 1.2164 | 9 |
| rounded_rect | 7.589 | 4.612 | 4.748 | 4.771 | -37.1 % | +0.5 % | 0.5877 | 0.5756 | 10 |
| star | 24.858 | 24.168 | 24.168 | 27.204 | +9.4 % | +12.6 % | 0.8500 | 0.5477 | 10 |
| star6 | 15.927 | 8.272 | 6.945 | 7.233 | -54.6 % | +4.1 % | 0.9375 | 0.9028 | 10 |

Over 18 shapes: v2 beats best-of-four on the hold-out in **12**; harmful versus uniform at melt-onset, **4** for v1 and **5** for v2; **1** shapes rescued by going unseeded.

### Table 5b. Union arm (same hold-out rule, all gains already solved), melt-onset sigma_T, deg C

| shape | (a) uniform | (c1) v1 seeded | (c2) v2 unseeded | (c3) union | c3 vs a | m (union) | evaluations in union |
|---|---|---|---|---|---|---|
| triangle | 23.043 | 41.423 | 21.401 | 21.401 | -7.1 % | 6.0000 | 14 |
| H_shape | 6.562 | 10.996 | 10.996 | 10.996 | +67.6 % | 0.1090 | 17 |
| trapezoid | 10.842 | 16.124 | 16.124 | 16.124 | +48.7 % | 0.1090 | 17 |
| T_shape | 60.149 | 71.039 | 69.106 | 69.106 | +14.9 % | 0.1110 | 18 |
| L_shape | 58.729 | NOT_REACHED | 74.108 | 74.108 | +26.2 % | 0.1110 | 18 |
| square | 4.931 | 3.669 | 3.647 | 3.647 | -26.0 % | 0.9033 | 18 |
| circle | 17.428 | 7.117 | 7.159 | 7.117 | -59.2 % | 0.6644 | 18 |
| hexagon | 18.083 | 6.726 | 6.773 | 6.773 | -62.5 % | 0.8217 | 18 |
| cross | 43.771 | 13.858 | 14.519 | 13.858 | -68.3 % | 1.0927 | 18 |
| diamond | 32.234 | 15.948 | 19.005 | 15.948 | -50.5 % | 2.1208 | 18 |
| ellipse | 22.046 | 8.603 | 8.485 | 8.485 | -61.5 % | 0.6713 | 18 |
| equilateral_triangle | 41.480 | 23.310 | 23.346 | 23.310 | -43.8 % | 1.4854 | 18 |
| octagon | 18.248 | 6.959 | 7.002 | 6.959 | -61.9 % | 0.4513 | 18 |
| pentagon | 24.036 | 14.880 | 14.787 | 14.880 | -38.1 % | 0.3000 | 17 |
| rectangle | 6.971 | 3.888 | 3.075 | 3.075 | -55.9 % | 1.2164 | 17 |
| rounded_rect | 7.589 | 4.748 | 4.771 | 4.771 | -37.1 % | 0.5756 | 18 |
| star | 24.858 | 24.168 | 27.204 | 24.168 | -2.8 % | 0.8500 | 18 |
| star6 | 15.927 | 6.945 | 7.233 | 6.945 | -56.4 % | 0.9375 | 18 |

Union arm over 18 shapes: harmful versus uniform at melt-onset in **4**; beats best-of-four on the hold-out in **13**.

### Table 6. v2 search diagnostics and gates

| shape | selected m | evaluations (new / cached) | stop reason | status | infeasible gains | P_abs at melt, W/m (uniform / v2) | residual frac (v2) | dT clip frac (v2) |
|---|---|---|---|---|---|---|---|---|
| triangle | 6.0000 | 6 / 1 | domain_edge_reached | OK | none | 500.0 / 353.7 | 0.013704 | 0.0000 |
| H_shape | 0.1095 | 9 / 1 | budget_exhausted | OK | none | 500.0 / 570.7 | 0.014133 | 0.0000 |
| trapezoid | 0.1093 | 9 / 1 | budget_exhausted | OK | none | 500.0 / 593.6 | 0.005518 | 0.0000 |
| T_shape | 0.1110 | 10 / 0 | budget_exhausted | OK | 7 of 10 | 500.0 / 484.8 | 0.051698 | 0.0000 |
| L_shape | 0.1110 | 10 / 0 | budget_exhausted | OK | 8 of 10 | 500.0 / 518.4 | 0.061154 | 0.0000 |
| square | 0.9033 | 2 / 8 | budget_exhausted | OK | none | 500.0 / 563.9 | 0.000154 | 0.0000 |
| circle | 0.6402 | 10 / 0 | budget_exhausted | OK | none | 500.0 / 763.5 | 0.002961 | 0.0000 |
| hexagon | 0.8217 | 10 / 0 | budget_exhausted | OK | none | 500.0 / 745.4 | 0.004823 | 0.0000 |
| cross | 1.2164 | 10 / 0 | budget_exhausted | OK | 4 of 10 | 500.0 / 325.3 | 0.021542 | 0.0000 |
| diamond | 1.2164 | 10 / 0 | budget_exhausted | OK | 5 of 10 | 500.0 / 466.6 | 0.012300 | 0.0000 |
| ellipse | 0.6713 | 10 / 0 | budget_exhausted | OK | none | 500.0 / 708.6 | 0.006986 | 0.0000 |
| equilateral_triangle | 1.6219 | 5 / 5 | budget_exhausted | OK | none | 500.0 / 504.9 | 0.014182 | 0.0000 |
| octagon | 0.5052 | 8 / 2 | budget_exhausted | OK | none | 500.0 / 1067.9 | 0.005766 | 0.0000 |
| pentagon | 0.3424 | 9 / 1 | budget_exhausted | OK | none | 500.0 / 672.7 | 0.009214 | 0.0000 |
| rectangle | 1.2164 | 9 / 1 | budget_exhausted | OK | none | 500.0 / 458.8 | 0.003129 | 0.0000 |
| rounded_rect | 0.5756 | 10 / 0 | budget_exhausted | OK | none | 500.0 / 641.8 | 0.000258 | 0.0000 |
| star | 0.5477 | 10 / 0 | budget_exhausted | OK | 5 of 10 | 500.0 / 590.7 | 0.035783 | 0.0000 |
| star6 | 0.9028 | 10 / 0 | budget_exhausted | OK | none | 500.0 / 608.3 | 0.009056 | 0.0000 |

### Table 7. v2 heating-peak sigma_T (the FIT read state), deg C

| shape | (a) uniform | (c1) v1 seeded | (c2) v2 unseeded | c2 vs a |
|---|---|---|---|---|
| triangle | 24.831 | 41.360 | 22.444 | -9.6 % |
| H_shape | 14.869 | 16.064 | 16.064 | +8.0 % |
| trapezoid | 14.867 | 19.521 | 19.521 | +31.3 % |
| T_shape | 60.127 | 71.020 | 69.087 | +14.9 % |
| L_shape | 58.707 | NOT_REACHED | 74.085 | +26.2 % |
| square | 11.049 | 6.923 | 6.918 | -37.4 % |
| circle | 19.961 | 11.991 | 12.010 | -39.8 % |
| hexagon | 19.934 | 11.853 | 11.804 | -40.8 % |
| cross | 43.757 | 15.562 | 18.019 | -58.8 % |
| diamond | 32.226 | 19.996 | 23.106 | -28.3 % |
| ellipse | 22.018 | 12.891 | 12.756 | -42.1 % |
| equilateral_triangle | 41.431 | 24.704 | 24.795 | -40.2 % |
| octagon | 20.787 | 11.370 | 11.627 | -44.1 % |
| pentagon | 24.098 | 19.264 | 19.265 | -20.1 % |
| rectangle | 14.943 | 8.881 | 8.537 | -42.9 % |
| rounded_rect | 10.817 | 7.867 | 7.854 | -27.4 % |
| star | 26.967 | 25.156 | 28.629 | +6.2 % |
| star6 | 18.346 | 9.378 | 10.107 | -44.9 % |
