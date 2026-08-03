## 1. Per-shape census, grid 120, each arm at its own J-stop

| shape | win | J_eps_best | J_eps_cold | J_hist | J_ms | J_ctl | J_lib | J_unif | IoU_eps_best | IoU_eps_cold | IoU_hist | IoU_ms | grow % | under % | rho at stop | P_abs W/m | lvl | class |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | warm | 8.39 | 9.91 | 12.77 | 35.56 | 18.51 | 25.66 | 210.19 | 1.0000 | 1.0000 | 0.9975 | 0.9681 | 0.00 | 0.00 | 0.7740 | 540.8 | 7 | SOLVED |
| circle | warm | 5.93 | 7.24 | 52.18 | 15.55 | 12.33 | 14.68 | 275.02 | 1.0000 | 1.0000 | 0.9492 | 0.9872 | 0.00 | 0.00 | 0.7307 | 716.1 | 10 | SOLVED |
| hexagon | warm | 7.47 | 42.57 | 69.63 | 24.57 | 24.24 | 22.15 | 251.27 | 0.9941 | 0.9502 | 0.9211 | 0.9650 | 0.00 | 0.59 | 0.6427 | 847.1 | 14 | SOLVED |
| triangle | cold | 44.30 | 44.30 | 171.27 | 82.29 | 82.29 | 87.99 | 202.82 | 0.9327 | 0.9327 | 0.7768 | 0.8783 | 4.00 | 3.00 | 0.6715 | 609.2 | 15 | IMPROVED |
| equilateral_triangle | cold | 63.30 | 63.30 | 145.55 | 100.06 | 97.89 | 119.00 | 320.80 | 0.9000 | 0.9000 | 0.8058 | 0.8451 | 3.86 | 6.53 | 0.6973 | 681.3 | 16 | IMPROVED |
| L_shape | warm | 337.77 | 355.84 | 597.86 | 530.14 | 528.35 | 521.20 | 538.42 | 0.6838 | 0.6670 | 0.4413 | 0.5109 | 2.87 | 29.66 | 0.6750 | 488.2 | 16 | IMPROVED |
| H_shape | warm | 81.51 | 101.41 | 219.11 | 149.76 | 120.07 | 150.86 | 214.68 | 0.9238 | 0.8777 | 0.7631 | 0.8499 | 4.86 | 3.12 | 0.6799 | 613.4 | 16 | IMPROVED |
| T_shape | warm | 457.74 | 476.96 | 676.39 | 612.03 | 612.13 | 609.29 | 614.26 | 0.5829 | 0.5653 | 0.3983 | 0.4583 | 1.63 | 40.76 | 0.6443 | 480.5 | 16 | IMPROVED |
| cross | warm | 106.50 | 213.78 | 331.36 | 385.07 | 380.93 | 360.18 | 471.82 | 0.9079 | 0.7980 | 0.7052 | 0.6508 | 6.95 | 2.90 | 0.7181 | 365.2 | 16 | IMPROVED |
| diamond | warm | 51.99 | 60.13 | 355.07 | 272.11 | 227.78 | 211.16 | 768.11 | 0.9647 | 0.9611 | 0.7793 | 0.8161 | 1.11 | 2.46 | 0.6754 | 719.7 | 16 | SOLVED |
| ellipse | warm | 2.62 | 2.64 | 74.70 | 28.84 | 21.46 | 13.98 | 241.72 | 1.0000 | 1.0000 | 0.8854 | 0.9524 | 0.00 | 0.00 | 0.6543 | 775.2 | 13 | SOLVED |
| octagon | warm | 4.20 | 6.43 | 16.22 | 10.58 | 11.96 | 10.66 | 42.96 | 0.9963 | 0.9963 | 0.9888 | 0.9963 | 0.00 | 0.37 | 0.8147 | 862.5 | 14 | SOLVED |
| pentagon | warm | 8.95 | 24.05 | 134.64 | 57.66 | 49.65 | 44.22 | 300.33 | 0.9914 | 0.9640 | 0.8446 | 0.9253 | 0.43 | 0.43 | 0.6925 | 734.5 | 13 | SOLVED |
| rectangle | warm | 7.00 | 9.51 | 10.69 | 97.19 | 187.57 | 196.94 | 203.07 | 1.0000 | 1.0000 | 1.0000 | 0.8842 | 0.00 | 0.00 | 0.6988 | 483.5 | 9 | SOLVED |
| rounded_rect | warm | 5.65 | 8.25 | 8.97 | 31.65 | 16.06 | 18.21 | 266.38 | 1.0000 | 1.0000 | 0.9974 | 0.9795 | 0.00 | 0.00 | 0.7314 | 612.0 | 11 | SOLVED |
| star | cold | 114.06 | 114.06 | 172.11 | 144.45 | 163.22 | 157.39 | 192.58 | 0.7862 | 0.7862 | 0.6713 | 0.7138 | 7.01 | 15.87 | 0.6126 | 583.7 | 16 | IMPROVED |
| star6 | cold | 84.55 | 84.55 | 70.57 | 72.42 | 82.03 | 84.69 | 190.62 | 0.8395 | 0.8395 | 0.8782 | 0.8742 | 9.46 | 8.11 | 0.6277 | 611.9 | 15 | NOT RESCUED |
| trapezoid | warm | 9.38 | 11.13 | 122.22 | 29.27 | 24.12 | 27.36 | 171.32 | 0.9966 | 0.9950 | 0.8688 | 0.9693 | 0.25 | 0.08 | 0.7381 | 567.6 | 13 | SOLVED |


## 2. Census counts against the three baselines

| arm | beats HIST on J_phi | beats HIST on IoU | beats UNIFORM on J_phi | IoU >= 0.95 at grid 120 | summed J_phi |
|---|---|---|---|---|---|
| EPS best of two full-depth starts (80 fwd-equiv) | 17 of 18 | 16 of 18 | 18 of 18 | 10 of 18 | 1401 |
| EPS cold full depth (40 fwd-equiv, budget matched) | 17 of 18 | 16 of 18 | 18 of 18 | 10 of 18 | 1636 |
| EPS warm full depth (40 fwd-equiv) | 17 of 18 | 16 of 18 | 18 of 18 | 10 of 18 | 1552 |
| conductivity-only multi-start (out_ms) | 13 of 18 | 13 of 18 | 18 of 18 | 7 of 18 | 2679 |
| conductivity-only filtered cold (out_ms control) | 13 of 18 | 13 of 18 | 18 of 18 | 7 of 18 | 2661 |
| conductivity-only single start (out_lib) | 13 of 18 | 13 of 18 | 18 of 18 | 7 of 18 | 2676 |


## 3. The four shapes the conductivity-only solve lost

| shape | J_hist | J_ms (sigma only) | J_eps_cold | J_eps_best | dJ vs hist % | IoU_hist | IoU_eps_best | verdict |
|---|---|---|---|---|---|---|---|---|
| square | 12.77 | 35.56 | 9.91 | 8.39 | +34.3 | 0.9975 | 1.0000 | BEATS HIST |
| rectangle | 10.69 | 97.19 | 9.51 | 7.00 | +34.5 | 1.0000 | 1.0000 | BEATS HIST |
| rounded_rect | 8.97 | 31.65 | 8.25 | 5.65 | +37.0 | 0.9974 | 1.0000 | BEATS HIST |
| star6 | 70.57 | 72.42 | 84.55 | 84.55 | -19.8 | 0.8782 | 0.8395 | still loses |


## 4. What the permittivity channel bought at matched budget

| shape | J_ctl (sigma only, 40) | J_eps_cold (eps, 40) | dJ % | IoU_ctl | IoU_eps_cold | dIoU |
|---|---|---|---|---|---|---|
| square | 18.51 | 9.91 | +46.5 | 1.0000 | 1.0000 | +0.0000 |
| circle | 12.33 | 7.24 | +41.2 | 0.9936 | 1.0000 | +0.0064 |
| hexagon | 24.24 | 42.57 | -75.6 | 0.9746 | 0.9502 | -0.0244 |
| triangle | 82.29 | 44.30 | +46.2 | 0.8783 | 0.9327 | +0.0544 |
| equilateral_triangle | 97.89 | 63.30 | +35.3 | 0.8370 | 0.9000 | +0.0630 |
| L_shape | 528.35 | 355.84 | +32.7 | 0.5135 | 0.6670 | +0.1534 |
| H_shape | 120.07 | 101.41 | +15.5 | 0.8599 | 0.8777 | +0.0178 |
| T_shape | 612.13 | 476.96 | +22.1 | 0.4583 | 0.5653 | +0.1070 |
| cross | 380.93 | 213.78 | +43.9 | 0.6589 | 0.7980 | +0.1391 |
| diamond | 227.78 | 60.13 | +73.6 | 0.8432 | 0.9611 | +0.1179 |
| ellipse | 21.46 | 2.64 | +87.7 | 0.9681 | 1.0000 | +0.0319 |
| octagon | 11.96 | 6.43 | +46.2 | 0.9852 | 0.9963 | +0.0110 |
| pentagon | 49.65 | 24.05 | +51.6 | 0.9289 | 0.9640 | +0.0351 |
| rectangle | 187.57 | 9.51 | +94.9 | 0.8580 | 1.0000 | +0.1420 |
| rounded_rect | 16.06 | 8.25 | +48.6 | 0.9974 | 1.0000 | +0.0026 |
| star | 163.22 | 114.06 | +30.1 | 0.6809 | 0.7862 | +0.1054 |
| star6 | 82.03 | 84.55 | -3.1 | 0.8408 | 0.8395 | -0.0013 |
| trapezoid | 24.12 | 11.13 | +53.8 | 0.9792 | 0.9950 | +0.0158 |


## 5. Gates and flags

| shape | energy gate violations | horizon arms | uniform channel invariance abs diff in J | wall s |
|---|---|---|---|---|
| square | none | none | 0.000e+00 | 371 |
| circle | none | none | 0.000e+00 | 258 |
| hexagon | none | none | 0.000e+00 | 198 |
| triangle | none | none | 0.000e+00 | 282 |
| equilateral_triangle | none | none | 0.000e+00 | 192 |
| L_shape | none | none | 0.000e+00 | 343 |
| H_shape | none | none | 0.000e+00 | 259 |
| T_shape | none | none | 0.000e+00 | 301 |
| cross | none | none | 0.000e+00 | 405 |
| diamond | none | none | 0.000e+00 | 282 |
| ellipse | none | none | 0.000e+00 | 188 |
| octagon | none | none | 0.000e+00 | 255 |
| pentagon | none | none | 0.000e+00 | 227 |
| rectangle | none | none | 0.000e+00 | 298 |
| rounded_rect | none | none | 0.000e+00 | 364 |
| star | none | none | 0.000e+00 | 192 |
| star6 | none | none | 0.000e+00 | 202 |
| trapezoid | none | none | 0.000e+00 | 310 |
