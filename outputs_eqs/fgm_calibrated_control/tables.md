### Table 1. Melt-onset sigma_T (hold-out read state), deg C

| shape | (a) uniform | (b) best-of-four in-sample | (c) calibrated gain | c vs a | c vs b | b gain m | c gain m | new solves |
|---|---|---|---|---|---|---|---|---|
| square | 4.931 | 3.768 | 3.669 | -25.6 % | -2.6 % | 0.85 | 0.8775 | 4 |
| circle | 17.428 | 7.134 | 7.117 | -59.2 % | -0.2 % | 0.70 | 0.6644 | 4 |
| hexagon | 18.083 | 6.788 | 6.726 | -62.8 % | -0.9 % | 0.85 | 0.8327 | 4 |
| triangle | 23.043 | 45.334 | 41.423 | +79.8 % | -8.6 % | 0.30 | 0.1090 | 4 |
| L_shape | 58.729 | NOT_REACHED | NOT_REACHED | n/a | n/a | none | NOT_REACHED | 4 |
| H_shape | 6.562 | 11.077 | 10.996 | +67.6 % | -0.7 % | 0.30 | 0.1090 | 4 |
| T_shape | 60.149 | 71.039 | 71.039 | +18.1 % | +0.0 % | 0.30 | 0.3000 | 4 |
| cross | 43.771 | 31.401 | 13.858 | -68.3 % | -55.9 % | 0.85 | 1.0927 | 4 |
| diamond | 32.234 | 26.637 | 15.948 | -50.5 % | -40.1 % | 0.85 | 2.1208 | 4 |
| ellipse | 22.046 | 8.684 | 8.603 | -61.0 % | -0.9 % | 0.70 | 0.7429 | 4 |
| equilateral_triangle | 41.480 | 43.933 | 23.310 | -43.8 % | -46.9 % | 0.85 | 1.4854 | 4 |
| octagon | 18.248 | 7.045 | 6.959 | -61.9 % | -1.2 % | 0.50 | 0.4513 | 4 |
| pentagon | 24.036 | 14.617 | 14.880 | -38.1 % | +1.8 % | 0.50 | 0.3000 | 4 |
| rectangle | 6.971 | 2.356 | 3.888 | -44.2 % | +65.0 % | 0.85 | 0.3000 | 4 |
| rounded_rect | 7.589 | 4.612 | 4.748 | -37.4 % | +2.9 % | 0.70 | 0.5877 | 4 |
| star | 24.858 | 24.168 | 24.168 | -2.8 % | +0.0 % | 0.85 | 0.8500 | 4 |
| star6 | 15.927 | 8.272 | 6.945 | -56.4 % | -16.0 % | 0.85 | 0.9375 | 4 |
| trapezoid | 10.842 | 16.735 | 16.124 | +48.7 % | -3.7 % | 0.30 | 0.1090 | 4 |

### Table 2. Heating-peak sigma_T (fit read state), deg C

| shape | (a) uniform | (b) best-of-four in-sample | (c) calibrated gain | c vs a | c vs b |
|---|---|---|---|---|---|
| square | 11.049 | 6.993 | 6.923 | -37.3 % | -1.0 % |
| circle | 19.961 | 12.136 | 11.991 | -39.9 % | -1.2 % |
| hexagon | 19.934 | 12.083 | 11.853 | -40.5 % | -1.9 % |
| triangle | 24.831 | 45.269 | 41.360 | +66.6 % | -8.6 % |
| L_shape | 58.707 | NOT_REACHED | NOT_REACHED | n/a | n/a |
| H_shape | 14.869 | 16.595 | 16.064 | +8.0 % | -3.2 % |
| T_shape | 60.127 | 71.020 | 71.020 | +18.1 % | +0.0 % |
| cross | 43.757 | 31.679 | 15.562 | -64.4 % | -50.9 % |
| diamond | 32.226 | 29.299 | 19.996 | -38.0 % | -31.8 % |
| ellipse | 22.018 | 12.982 | 12.891 | -41.5 % | -0.7 % |
| equilateral_triangle | 41.431 | 43.808 | 24.704 | -40.4 % | -43.6 % |
| octagon | 20.787 | 11.693 | 11.370 | -45.3 % | -2.8 % |
| pentagon | 24.098 | 19.435 | 19.264 | -20.1 % | -0.9 % |
| rectangle | 14.943 | 9.689 | 8.881 | -40.6 % | -8.3 % |
| rounded_rect | 10.817 | 8.524 | 7.867 | -27.3 % | -7.7 % |
| star | 26.967 | 25.156 | 25.156 | -6.7 % | +0.0 % |
| star6 | 18.346 | 11.315 | 9.378 | -48.9 % | -17.1 % |
| trapezoid | 14.867 | 20.463 | 19.521 | +31.3 % | -4.6 % |

### Table 3. Search diagnostics and gates

| shape | gains evaluated (new) | stop reason | status | infeasible gains | P_abs at melt, W/m (a / c) | energy residual frac (c) | dT clip frac (c) | wall s (new solves) |
|---|---|---|---|---|---|---|---|---|
| square | 1.0927, 0.8775, 0.8970, 0.8807 | budget_exhausted | OK | none | 500.0 / 564.8 | 0.000149 | 0.0000 | 364 |
| circle | 0.6235, 0.6644, 0.6629, 0.6801 | budget_exhausted | OK | none | 500.0 / 764.7 | 0.002936 | 0.0000 | 256 |
| hexagon | 1.0927, 0.8327, 0.7934, 0.8084 | budget_exhausted | OK | none | 500.0 / 746.7 | 0.004835 | 0.0000 | 312 |
| triangle | 0.0500, 0.2045, 0.1455, 0.1090 | budget_exhausted | OK | none | 500.0 / 604.0 | 0.025510 | 0.0000 | 183 |
| L_shape | 1.0927, 1.4854, 2.1208, 1.7632 | budget_exhausted | NOT_REACHED | 0.3000, 0.5000, 0.7000, 0.8500, 1.0927, 1.4854, 1.7632, 2.1208 | 500.0 / NOT_REACHED | n/a | n/a | 508 |
| H_shape | 0.0500, 0.2045, 0.1455, 0.1090 | budget_exhausted | OK | none | 500.0 / 570.7 | 0.014133 | 0.0000 | 185 |
| T_shape | 1.0927, 1.4854, 2.1208, 2.5000 | budget_exhausted | OK | 0.5000, 0.7000, 0.8500, 1.0927, 1.4854, 2.1208, 2.5000 | 500.0 / 474.4 | 0.051990 | 0.0000 | 487 |
| cross | 1.0927, 1.4854, 1.1855, 1.1184 | budget_exhausted | OK | 1.4854 | 500.0 / 397.9 | 0.018349 | 0.0000 | 967 |
| diamond | 1.0927, 1.4854, 2.1208, 2.5000 | budget_exhausted | OK | 2.5000 | 500.0 / 376.6 | 0.011773 | 0.0000 | 60 |
| ellipse | 0.7429, 0.7624, 0.7164, 0.7472 | budget_exhausted | OK | none | 500.0 / 717.8 | 0.007200 | 0.0000 | 251 |
| equilateral_triangle | 1.0927, 1.4854, 2.1208, 1.6243 | budget_exhausted | OK | none | 500.0 / 518.1 | 0.013633 | 0.0000 | 249 |
| octagon | 0.4591, 0.4407, 0.4513, 0.4502 | budget_exhausted | OK | none | 500.0 / 1053.9 | 0.005479 | 0.0000 | 437 |
| pentagon | 0.0500, 0.3277, 0.2478, 0.2935 | budget_exhausted | OK | none | 500.0 / 669.8 | 0.009199 | 0.0000 | 407 |
| rectangle | 0.0500, 0.2703, 0.3630, 0.3131 | budget_exhausted | OK | none | 500.0 / 491.2 | 0.000366 | 0.0000 | 903 |
| rounded_rect | 0.6028, 0.6047, 0.5877, 0.5569 | budget_exhausted | OK | none | 500.0 / 643.4 | 0.000262 | 0.0000 | 254 |
| star | 1.0927, 0.8558, 0.8370, 0.8419 | budget_exhausted | OK | none | 500.0 / 535.3 | 0.029054 | 0.0000 | 329 |
| star6 | 1.0927, 0.9089, 0.9375, 0.9644 | budget_exhausted | OK | none | 500.0 / 594.3 | 0.007907 | 0.0000 | 492 |
| trapezoid | 0.0500, 0.2045, 0.1455, 0.1090 | budget_exhausted | OK | none | 500.0 / 593.6 | 0.005518 | 0.0000 | 186 |

### Table 4. Verdict per shape (pre-registered criteria)

| shape | (b) harmful vs uniform? | (c) harmful vs uniform? | calibration beats (b) on hold-out? | rescued? |
|---|---|---|---|---|
| square | no | no | YES | no |
| circle | no | no | YES | no |
| hexagon | no | no | YES | no |
| triangle | YES | YES | YES | no |
| L_shape | n/a (no arm melts) | NOT_REACHED | n/a | n/a |
| H_shape | YES | YES | YES | no |
| T_shape | YES | YES | no | no |
| cross | no | no | YES | no |
| diamond | no | no | YES | no |
| ellipse | no | no | YES | no |
| equilateral_triangle | YES | no | YES | YES |
| octagon | no | no | YES | no |
| pentagon | no | no | no | no |
| rectangle | no | no | no | no |
| rounded_rect | no | no | no | no |
| star | no | no | no | no |
| star6 | no | no | YES | no |
| trapezoid | YES | YES | YES | no |

Totals over 18 shapes: calibration beats best-of-four on the hold-out in **12**; shapes harmful versus uniform at melt-onset: **5** for (b), **4** for (c); rescued: **1**.
