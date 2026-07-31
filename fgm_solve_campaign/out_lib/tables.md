## census

| shape | best stored mask, J | its IoU | solved 4 bpp, J | its IoU | J change | IoU change | beats on J | beats on IoU | class |
|---|---|---|---|---|---|---|---|---|---|
| square | 12.77 | 0.9975 | 25.66 | 0.9816 | -100.9 % | -0.0159 | no | no | SOLVED |
| circle | 52.18 | 0.9492 | 14.68 | 0.9904 | +71.9 % | +0.0412 | YES | YES | SOLVED |
| hexagon | 69.63 | 0.9211 | 22.15 | 0.9767 | +68.2 % | +0.0556 | YES | YES | SOLVED |
| triangle | 171.27 | 0.7768 | 87.99 | 0.8578 | +48.6 % | +0.0810 | YES | YES | IMPROVED |
| equilateral_triangle | 145.55 | 0.8058 | 119.00 | 0.8128 | +18.2 % | +0.0071 | YES | YES | IMPROVED |
| L_shape | 597.86 (H) | 0.4413 | 521.20 | 0.5209 | +12.8 % | +0.0796 | YES | YES | IMPROVED |
| H_shape | 219.11 | 0.7631 | 150.86 | 0.8423 | +31.1 % | +0.0792 | YES | YES | IMPROVED |
| T_shape | 676.39 | 0.3983 | 609.29 | 0.4444 | +9.9 % | +0.0462 | YES | YES | IMPROVED |
| cross | 331.36 | 0.7052 | 360.18 (H) | 0.6755 | -8.7 % | -0.0297 | no | no | NOT RESCUED |
| diamond | 355.07 | 0.7793 | 211.16 (H) | 0.8520 | +40.5 % | +0.0727 | YES | YES | IMPROVED |
| ellipse | 74.70 | 0.8854 | 13.98 | 0.9787 | +81.3 % | +0.0933 | YES | YES | SOLVED |
| octagon | 16.22 | 0.9888 | 10.66 | 0.9944 | +34.3 % | +0.0056 | YES | YES | SOLVED |
| pentagon | 134.64 | 0.8446 | 44.22 | 0.9387 | +67.2 % | +0.0941 | YES | YES | IMPROVED |
| rectangle | 10.69 | 1.0000 | 196.94 | 0.8424 | -1742.6 % | -0.1576 | no | no | NOT RESCUED |
| rounded_rect | 8.97 | 0.9974 | 18.21 | 0.9974 | -103.1 % | +0.0000 | no | no | SOLVED |
| star | 172.11 | 0.6713 | 157.39 | 0.7032 | +8.6 % | +0.0319 | YES | YES | IMPROVED |
| star6 | 70.57 | 0.8782 | 84.69 | 0.8408 | -20.0 % | -0.0374 | no | no | NOT RESCUED |
| trapezoid | 122.22 | 0.8688 | 27.36 | 0.9718 | +77.6 % | +0.1030 | YES | YES | SOLVED |

## historical winners

| shape | stored masks scanned | winning mask | campaign | convention | J | IoU |
|---|---|---|---|---|---|---|
| square | 19 | `cal_map_m0p5477_mag0p55` | calibration campaign | asstored | 12.77 | 0.9975 |
| circle | 18 | `oldgrid_map_m0p70_mag0p70` | old {0.30 .. 0.85} grid | outside1 | 52.18 | 0.9492 |
| hexagon | 18 | `cal_map_m0p8327_mag0p83` | calibration campaign | outside1 | 69.63 | 0.9211 |
| triangle | 14 | `cal_map_m2p7016_mag2p70` | calibration campaign | outside1 | 171.27 | 0.7768 |
| equilateral_triangle | 18 | `cal_map_m1p2164_mag1p22` | calibration campaign | asstored | 145.55 | 0.8058 |
| L_shape | 18 | `cal_map_m1p0927_mag1p09` | calibration campaign | outside1 | 597.86 | 0.4413 |
| H_shape | 17 | `cal_map_m0p1090_mag0p11` | calibration campaign | asstored | 219.11 | 0.7631 |
| T_shape | 18 | `cal_map_m1p0927_mag1p09` | calibration campaign | asstored | 676.39 | 0.3983 |
| cross | 18 | `cal_map_m1p0927_mag1p09` | calibration campaign | asstored | 331.36 | 0.7052 |
| diamond | 18 | `cal_map_m1p4854_mag1p49` | calibration campaign | asstored | 355.07 | 0.7793 |
| ellipse | 18 | `oldgrid_map_m0p85_mag0p85` | old {0.30 .. 0.85} grid | outside1 | 74.70 | 0.8854 |
| octagon | 18 | `cal_map_m0p1110_mag0p11` | calibration campaign | outside1 | 16.22 | 0.9888 |
| pentagon | 17 | `cal_map_m0p1110_mag0p11` | calibration campaign | outside1 | 134.64 | 0.8446 |
| rectangle | 17 | `oldgrid_map_m0p50_mag0p50` | old {0.30 .. 0.85} grid | asstored | 10.69 | 1.0000 |
| rounded_rect | 18 | `cal_map_m0p5569_mag0p56` | calibration campaign | asstored | 8.97 | 0.9974 |
| star | 18 | `oldgrid_map_m0p50_mag0p50` | old {0.30 .. 0.85} grid | outside1 | 172.11 | 0.6713 |
| star6 | 18 | `cal_map_m0p9644_mag0p96` | calibration campaign | asstored | 70.57 | 0.8782 |
| trapezoid | 17 | `cal_map_m0p1110_mag0p11` | calibration campaign | outside1 | 122.22 | 0.8688 |

## full table

| shape | arm | J | J per part cell | IoU | growth % | under % | stop idx | stop s | horizon | phi_bar | P_abs W/m | energy residual % of dose | gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| square | uniform s = 1 | 210.19 | 0.1314 | 0.8508 | 7.25 | 8.75 | 832 | 416.5 | no | 0.884 | 500.0 | 0.70 | PASS |
| square | best stored historical mask | 12.77 | 0.0080 | 0.9975 | 0.00 | 0.25 | 778 | 389.5 | no | 0.982 | 559.8 | 0.16 | PASS |
| square | solved [0, 1] continuous | 25.56 | 0.0160 | 0.9804 | 2.00 | 0.00 | 1107 | 554.0 | no | 0.968 | 412.0 | 0.25 | PASS |
| square | **solved [0, 1] 4 bpp** | 25.66 | 0.0160 | 0.9816 | 1.88 | 0.00 | 1107 | 554.0 | no | 0.967 | 412.0 | 0.24 | PASS |
| square | solved [0, 1] 2 bpp | 25.52 | 0.0160 | 0.9901 | 1.00 | 0.00 | 1117 | 559.0 | no | 0.963 | 409.5 | 0.25 | PASS |
| circle | uniform s = 1 | 275.02 | 0.2218 | 0.7881 | 14.19 | 10.00 | 755 | 378.0 | no | 0.898 | 500.0 | 1.17 | PASS |
| circle | best stored historical mask | 52.18 | 0.0421 | 0.9492 | 1.61 | 3.55 | 452 | 226.5 | no | 0.959 | 689.3 | 0.22 | PASS |
| circle | solved [0, 1] continuous | 14.56 | 0.0117 | 0.9904 | 0.65 | 0.32 | 1136 | 568.5 | no | 0.983 | 365.7 | 0.14 | PASS |
| circle | **solved [0, 1] 4 bpp** | 14.68 | 0.0118 | 0.9904 | 0.65 | 0.32 | 1132 | 566.5 | no | 0.983 | 366.4 | 0.14 | PASS |
| circle | solved [0, 1] 2 bpp | 26.63 | 0.0215 | 0.9667 | 1.77 | 1.61 | 1186 | 593.5 | no | 0.977 | 354.2 | 0.17 | PASS |
| hexagon | uniform s = 1 | 251.27 | 0.2473 | 0.7649 | 18.90 | 9.06 | 652 | 326.5 | no | 0.912 | 500.0 | 1.43 | PASS |
| hexagon | best stored historical mask | 69.63 | 0.0685 | 0.9211 | 4.72 | 3.54 | 403 | 202.0 | no | 0.962 | 663.2 | 0.49 | PASS |
| hexagon | solved [0, 1] continuous | 22.73 | 0.0224 | 0.9785 | 0.59 | 1.57 | 885 | 443.0 | no | 0.973 | 376.2 | 0.17 | PASS |
| hexagon | **solved [0, 1] 4 bpp** | 22.15 | 0.0218 | 0.9767 | 1.18 | 1.18 | 896 | 448.5 | no | 0.974 | 373.6 | 0.18 | PASS |
| hexagon | solved [0, 1] 2 bpp | 23.52 | 0.0231 | 0.9728 | 1.18 | 1.57 | 911 | 456.0 | no | 0.975 | 370.7 | 0.18 | PASS |
| triangle | uniform s = 1 | 202.82 | 0.2535 | 0.7687 | 16.75 | 10.25 | 506 | 253.5 | no | 0.850 | 500.0 | 1.40 | PASS |
| triangle | best stored historical mask | 171.27 | 0.2141 | 0.7768 | 12.00 | 13.00 | 736 | 368.5 | no | 0.855 | 354.4 | 0.86 | PASS |
| triangle | solved [0, 1] continuous | 88.06 | 0.1101 | 0.8578 | 7.25 | 8.00 | 539 | 270.0 | no | 0.892 | 442.3 | 0.52 | PASS |
| triangle | **solved [0, 1] 4 bpp** | 87.99 | 0.1100 | 0.8578 | 7.25 | 8.00 | 539 | 270.0 | no | 0.892 | 442.1 | 0.52 | PASS |
| triangle | solved [0, 1] 2 bpp | 90.65 | 0.1133 | 0.8538 | 7.75 | 8.00 | 536 | 268.5 | no | 0.890 | 444.6 | 0.53 | PASS |
| equilateral_triangle | uniform s = 1 | 320.80 | 0.4760 | 0.6067 | 32.05 | 19.88 | 479 | 240.0 | no | 0.797 | 500.0 | 2.21 | PASS |
| equilateral_triangle | best stored historical mask | 145.55 | 0.2159 | 0.8058 | 13.06 | 8.90 | 348 | 174.5 | no | 0.871 | 571.3 | 1.26 | PASS |
| equilateral_triangle | solved [0, 1] continuous | 118.81 | 0.1763 | 0.8128 | 10.98 | 9.79 | 589 | 295.0 | no | 0.895 | 395.4 | 0.78 | PASS |
| equilateral_triangle | **solved [0, 1] 4 bpp** | 119.00 | 0.1766 | 0.8128 | 10.98 | 9.79 | 589 | 295.0 | no | 0.896 | 395.6 | 0.79 | PASS |
| equilateral_triangle | solved [0, 1] 2 bpp | 118.64 | 0.1760 | 0.8155 | 10.98 | 9.50 | 590 | 295.5 | no | 0.896 | 395.1 | 0.78 | PASS |
| L_shape | uniform s = 1 | 538.42 | 0.4990 | 0.5142 | 7.78 | 44.58 | 526 | 263.5 | no | 0.550 | 500.0 | 0.85 | PASS |
| L_shape | best stored historical mask | 597.86 | 0.5541 | 0.4413 | 5.00 | 53.66 | 1499 | 750.0 | YES | 0.443 | 228.4 | 0.41 | PASS |
| L_shape | solved [0, 1] continuous | 521.16 | 0.4830 | 0.5209 | 6.39 | 44.58 | 525 | 263.0 | no | 0.547 | 493.6 | 0.68 | PASS |
| L_shape | **solved [0, 1] 4 bpp** | 521.20 | 0.4830 | 0.5209 | 6.39 | 44.58 | 525 | 263.0 | no | 0.547 | 493.7 | 0.68 | PASS |
| L_shape | solved [0, 1] 2 bpp | 521.42 | 0.4832 | 0.5200 | 6.39 | 44.67 | 525 | 263.0 | no | 0.547 | 493.5 | 0.68 | PASS |
| L_shape | solved [0, 1.5] continuous | 394.59 | 0.3657 | 0.6272 | 6.39 | 33.27 | 503 | 252.0 | no | 0.656 | 552.5 | 0.64 | PASS |
| L_shape | solved [0, 1.5] 4 bpp | 402.55 | 0.3731 | 0.6225 | 6.30 | 33.83 | 505 | 253.0 | no | 0.651 | 549.3 | 0.65 | PASS |
| L_shape | solved [0, 1.5] 2 bpp | 440.63 | 0.4084 | 0.5795 | 7.78 | 37.53 | 517 | 259.0 | no | 0.623 | 533.1 | 0.73 | PASS |
| H_shape | uniform s = 1 | 214.68 | 0.1864 | 0.7563 | 9.72 | 17.01 | 694 | 347.5 | no | 0.832 | 500.0 | 0.92 | PASS |
| H_shape | best stored historical mask | 219.11 | 0.1902 | 0.7631 | 12.85 | 13.89 | 589 | 295.0 | no | 0.867 | 570.7 | 1.16 | PASS |
| H_shape | solved [0, 1] continuous | 150.78 | 0.1309 | 0.8412 | 10.42 | 7.12 | 827 | 414.0 | no | 0.904 | 450.0 | 0.94 | PASS |
| H_shape | **solved [0, 1] 4 bpp** | 150.86 | 0.1310 | 0.8423 | 10.07 | 7.29 | 825 | 413.0 | no | 0.903 | 450.3 | 0.93 | PASS |
| H_shape | solved [0, 1] 2 bpp | 156.38 | 0.1357 | 0.8331 | 10.24 | 8.16 | 826 | 413.5 | no | 0.900 | 450.5 | 0.95 | PASS |
| T_shape | uniform s = 1 | 614.26 | 0.5564 | 0.4574 | 4.17 | 52.36 | 460 | 230.5 | no | 0.462 | 500.0 | 0.62 | PASS |
| T_shape | best stored historical mask | 676.39 | 0.6127 | 0.3983 | 5.07 | 58.15 | 1497 | 749.0 | no | 0.397 | 227.9 | 0.45 | PASS |
| T_shape | solved [0, 1] continuous | 609.31 | 0.5519 | 0.4444 | 2.72 | 54.35 | 433 | 217.0 | no | 0.438 | 500.3 | 0.33 | PASS |
| T_shape | **solved [0, 1] 4 bpp** | 609.29 | 0.5519 | 0.4444 | 2.72 | 54.35 | 433 | 217.0 | no | 0.438 | 500.3 | 0.33 | PASS |
| T_shape | solved [0, 1] 2 bpp | 610.08 | 0.5526 | 0.4444 | 2.72 | 54.35 | 433 | 217.0 | no | 0.438 | 500.8 | 0.34 | PASS |
| T_shape | solved [0, 1.5] continuous | 551.60 | 0.4996 | 0.4937 | 1.27 | 50.00 | 410 | 205.5 | no | 0.493 | 557.0 | 0.28 | PASS |
| T_shape | solved [0, 1.5] 4 bpp | 553.99 | 0.5018 | 0.4875 | 1.45 | 50.54 | 410 | 205.5 | no | 0.491 | 555.3 | 0.28 | PASS |
| T_shape | solved [0, 1.5] 2 bpp | 566.58 | 0.5132 | 0.4734 | 2.17 | 51.63 | 412 | 206.5 | no | 0.477 | 544.1 | 0.28 | PASS |
| cross | uniform s = 1 | 471.82 | 0.4554 | 0.5465 | 3.86 | 43.24 | 439 | 220.0 | no | 0.563 | 500.0 | 0.59 | PASS |
| cross | best stored historical mask | 331.36 | 0.3198 | 0.7052 | 21.81 | 14.09 | 859 | 430.0 | no | 0.857 | 397.9 | 1.64 | PASS |
| cross | solved [0, 1] continuous | 360.07 | 0.3476 | 0.6777 | 16.80 | 20.85 | 1498 | 749.5 | no | 0.790 | 263.0 | 1.10 | PASS |
| cross | **solved [0, 1] 4 bpp** | 360.18 | 0.3477 | 0.6755 | 16.60 | 21.24 | 1499 | 750.0 | YES | 0.789 | 262.6 | 1.10 | PASS |
| cross | solved [0, 1] 2 bpp | 359.55 | 0.3471 | 0.6833 | 15.83 | 20.85 | 1499 | 750.0 | YES | 0.788 | 261.8 | 1.08 | PASS |
| cross | solved [0, 1.5] continuous | 367.28 | 0.3545 | 0.6542 | 13.90 | 25.48 | 1122 | 561.5 | no | 0.750 | 309.0 | 1.00 | PASS |
| cross | solved [0, 1.5] 4 bpp | 368.13 | 0.3553 | 0.6610 | 13.32 | 25.10 | 1129 | 565.0 | no | 0.750 | 307.8 | 1.01 | PASS |
| cross | solved [0, 1.5] 2 bpp | 370.39 | 0.3575 | 0.6570 | 13.71 | 25.29 | 1150 | 575.5 | no | 0.748 | 302.8 | 1.00 | PASS |
| diamond | uniform s = 1 | 768.11 | 0.4730 | 0.5903 | 24.75 | 26.35 | 946 | 473.5 | no | 0.738 | 500.0 | 2.07 | PASS |
| diamond | best stored historical mask | 355.07 | 0.2186 | 0.7793 | 9.36 | 14.78 | 1073 | 537.0 | no | 0.852 | 433.1 | 0.83 | PASS |
| diamond | solved [0, 1] continuous | 211.79 | 0.1304 | 0.8535 | 2.59 | 12.44 | 1499 | 750.0 | YES | 0.869 | 319.4 | 0.27 | PASS |
| diamond | **solved [0, 1] 4 bpp** | 211.16 | 0.1300 | 0.8520 | 2.34 | 12.81 | 1499 | 750.0 | YES | 0.867 | 318.5 | 0.25 | PASS |
| diamond | solved [0, 1] 2 bpp | 226.67 | 0.1396 | 0.8406 | 1.97 | 14.29 | 1499 | 750.0 | YES | 0.837 | 313.0 | 0.20 | PASS |
| ellipse | uniform s = 1 | 241.72 | 0.3249 | 0.6909 | 18.28 | 18.28 | 426 | 213.5 | no | 0.814 | 500.0 | 1.56 | PASS |
| ellipse | best stored historical mask | 74.70 | 0.1004 | 0.8854 | 3.23 | 8.60 | 306 | 153.5 | no | 0.909 | 608.9 | 0.37 | PASS |
| ellipse | solved [0, 1] continuous | 13.62 | 0.0183 | 0.9787 | 1.08 | 1.08 | 1314 | 657.5 | no | 0.974 | 232.9 | 0.16 | PASS |
| ellipse | **solved [0, 1] 4 bpp** | 13.98 | 0.0188 | 0.9787 | 1.08 | 1.08 | 1283 | 642.0 | no | 0.974 | 236.2 | 0.16 | PASS |
| ellipse | solved [0, 1] 2 bpp | 22.05 | 0.0296 | 0.9577 | 1.61 | 2.69 | 1433 | 717.0 | no | 0.968 | 221.3 | 0.18 | PASS |
| octagon | uniform s = 1 | 42.96 | 0.0399 | 0.9489 | 1.86 | 3.35 | 656 | 328.5 | no | 0.961 | 500.0 | 0.22 | PASS |
| octagon | best stored historical mask | 16.22 | 0.0151 | 0.9888 | 0.00 | 1.12 | 390 | 195.5 | no | 0.982 | 792.4 | 0.16 | PASS |
| octagon | solved [0, 1] continuous | 10.60 | 0.0099 | 0.9944 | 0.00 | 0.56 | 758 | 379.5 | no | 0.984 | 437.6 | 0.14 | PASS |
| octagon | **solved [0, 1] 4 bpp** | 10.66 | 0.0099 | 0.9944 | 0.00 | 0.56 | 758 | 379.5 | no | 0.984 | 437.4 | 0.14 | PASS |
| octagon | solved [0, 1] 2 bpp | 10.60 | 0.0099 | 0.9907 | 0.00 | 0.93 | 760 | 380.5 | no | 0.984 | 440.0 | 0.14 | PASS |
| pentagon | uniform s = 1 | 300.33 | 0.3229 | 0.7130 | 20.65 | 13.98 | 579 | 290.0 | no | 0.857 | 500.0 | 1.66 | PASS |
| pentagon | best stored historical mask | 134.64 | 0.1448 | 0.8446 | 5.16 | 11.18 | 411 | 206.0 | no | 0.885 | 602.1 | 0.52 | PASS |
| pentagon | solved [0, 1] continuous | 43.89 | 0.0472 | 0.9387 | 1.72 | 4.52 | 867 | 434.0 | no | 0.952 | 352.0 | 0.21 | PASS |
| pentagon | **solved [0, 1] 4 bpp** | 44.22 | 0.0475 | 0.9387 | 1.72 | 4.52 | 864 | 432.5 | no | 0.952 | 352.4 | 0.21 | PASS |
| pentagon | solved [0, 1] 2 bpp | 67.11 | 0.0722 | 0.9083 | 3.23 | 6.24 | 887 | 444.0 | no | 0.935 | 344.4 | 0.27 | PASS |
| rectangle | uniform s = 1 | 203.07 | 0.1763 | 0.8528 | 13.19 | 3.47 | 642 | 321.5 | no | 0.839 | 500.0 | 1.16 | PASS |
| rectangle | best stored historical mask | 10.69 | 0.0093 | 1.0000 | 0.00 | 0.00 | 661 | 331.0 | no | 0.972 | 483.7 | 0.19 | PASS |
| rectangle | solved [0, 1] continuous | 196.87 | 0.1709 | 0.8450 | 14.24 | 3.47 | 642 | 321.5 | no | 0.863 | 503.2 | 1.23 | PASS |
| rectangle | **solved [0, 1] 4 bpp** | 196.94 | 0.1710 | 0.8424 | 14.58 | 3.47 | 642 | 321.5 | no | 0.863 | 503.3 | 1.23 | PASS |
| rectangle | solved [0, 1] 2 bpp | 197.68 | 0.1716 | 0.8450 | 14.24 | 3.47 | 643 | 322.0 | no | 0.864 | 502.8 | 1.23 | PASS |
| rounded_rect | uniform s = 1 | 266.38 | 0.1734 | 0.8190 | 12.24 | 8.07 | 853 | 427.0 | no | 0.912 | 500.0 | 1.04 | PASS |
| rounded_rect | best stored historical mask | 8.97 | 0.0058 | 0.9974 | 0.00 | 0.26 | 616 | 308.5 | no | 0.987 | 639.9 | 0.13 | PASS |
| rounded_rect | solved [0, 1] continuous | 18.21 | 0.0119 | 0.9948 | 0.13 | 0.39 | 1282 | 641.5 | no | 0.979 | 376.6 | 0.16 | PASS |
| rounded_rect | **solved [0, 1] 4 bpp** | 18.21 | 0.0119 | 0.9974 | 0.00 | 0.26 | 1282 | 641.5 | no | 0.979 | 376.7 | 0.16 | PASS |
| rounded_rect | solved [0, 1] 2 bpp | 21.22 | 0.0138 | 0.9935 | 0.00 | 0.65 | 1225 | 613.0 | no | 0.978 | 386.4 | 0.17 | PASS |
| star | uniform s = 1 | 192.58 | 0.3553 | 0.6517 | 7.01 | 30.26 | 284 | 142.5 | no | 0.664 | 500.0 | 0.79 | PASS |
| star | best stored historical mask | 172.11 | 0.3176 | 0.6713 | 5.54 | 29.15 | 242 | 121.5 | no | 0.703 | 551.5 | 0.65 | PASS |
| star | solved [0, 1] continuous | 157.36 | 0.2903 | 0.6996 | 4.43 | 26.94 | 326 | 163.5 | no | 0.707 | 449.0 | 0.48 | PASS |
| star | **solved [0, 1] 4 bpp** | 157.39 | 0.2904 | 0.7032 | 4.43 | 26.57 | 327 | 164.0 | no | 0.708 | 448.5 | 0.49 | PASS |
| star | solved [0, 1] 2 bpp | 160.38 | 0.2959 | 0.6926 | 4.43 | 27.68 | 320 | 160.5 | no | 0.701 | 454.8 | 0.50 | PASS |
| star | solved [0, 1.5] continuous | 149.41 | 0.2757 | 0.7077 | 4.80 | 25.83 | 310 | 155.5 | no | 0.715 | 473.1 | 0.54 | PASS |
| star | solved [0, 1.5] 4 bpp | 149.79 | 0.2764 | 0.7092 | 4.06 | 26.20 | 310 | 155.5 | no | 0.712 | 472.0 | 0.53 | PASS |
| star | solved [0, 1.5] 2 bpp | 152.02 | 0.2805 | 0.7092 | 4.06 | 26.20 | 312 | 156.5 | no | 0.703 | 467.6 | 0.50 | PASS |
| star6 | uniform s = 1 | 190.62 | 0.3220 | 0.6758 | 22.97 | 16.89 | 391 | 196.0 | no | 0.850 | 500.0 | 1.82 | PASS |
| star6 | best stored historical mask | 70.57 | 0.1192 | 0.8782 | 5.41 | 7.43 | 273 | 137.0 | no | 0.894 | 585.9 | 0.73 | PASS |
| star6 | solved [0, 1] continuous | 84.80 | 0.1432 | 0.8408 | 6.08 | 10.81 | 460 | 230.5 | no | 0.874 | 410.6 | 0.64 | PASS |
| star6 | **solved [0, 1] 4 bpp** | 84.69 | 0.1431 | 0.8408 | 6.08 | 10.81 | 461 | 231.0 | no | 0.876 | 410.5 | 0.65 | PASS |
| star6 | solved [0, 1] 2 bpp | 82.88 | 0.1400 | 0.8408 | 6.08 | 10.81 | 460 | 230.5 | no | 0.873 | 409.0 | 0.61 | PASS |
| trapezoid | uniform s = 1 | 171.32 | 0.1441 | 0.8387 | 9.50 | 8.16 | 682 | 341.5 | no | 0.909 | 500.0 | 0.80 | PASS |
| trapezoid | best stored historical mask | 122.22 | 0.1028 | 0.8688 | 5.13 | 8.66 | 592 | 296.5 | no | 0.912 | 571.3 | 0.50 | PASS |
| trapezoid | solved [0, 1] continuous | 27.24 | 0.0229 | 0.9718 | 1.35 | 1.51 | 869 | 435.0 | no | 0.970 | 422.5 | 0.23 | PASS |
| trapezoid | **solved [0, 1] 4 bpp** | 27.36 | 0.0230 | 0.9718 | 1.35 | 1.51 | 867 | 434.0 | no | 0.969 | 422.9 | 0.23 | PASS |
| trapezoid | solved [0, 1] 2 bpp | 28.95 | 0.0243 | 0.9693 | 1.26 | 1.85 | 855 | 428.0 | no | 0.969 | 426.2 | 0.24 | PASS |

## cost

| shape | part cells | forward s | adjoint s | ratio | gradient evaluations inside 40 forward-equivalents | double pass run | wall s |
|---|---|---|---|---|---|---|---|
| square | 1600 | 9.87 | 17.59 | 1.782 | 14 | no | 530 |
| circle | 1240 | 14.53 | 24.54 | 1.690 | 14 | no | 861 |
| hexagon | 1016 | 12.38 | 19.48 | 1.574 | 15 | no | 753 |
| triangle | 800 | 10.19 | 14.99 | 1.472 | 16 | no | 582 |
| equilateral_triangle | 674 | 9.79 | 13.85 | 1.415 | 16 | no | 631 |
| L_shape | 1079 | 7.52 | 8.83 | 1.174 | 18 | yes | 679 |
| H_shape | 1152 | 13.75 | 20.77 | 1.510 | 15 | no | 786 |
| T_shape | 1104 | 9.72 | 12.64 | 1.301 | 17 | yes | 1019 |
| cross | 1036 | 6.99 | 10.16 | 1.454 | 16 | yes | 1720 |
| diamond | 1624 | 13.76 | 21.03 | 1.528 | 15 | no | 1349 |
| ellipse | 744 | 7.09 | 8.82 | 1.244 | 17 | no | 963 |
| octagon | 1076 | 10.08 | 14.77 | 1.465 | 16 | no | 785 |
| pentagon | 930 | 8.91 | 12.99 | 1.457 | 16 | no | 862 |
| rectangle | 1152 | 11.67 | 18.37 | 1.574 | 15 | no | 859 |
| rounded_rect | 1536 | 14.00 | 22.77 | 1.626 | 15 | no | 864 |
| star | 542 | 5.23 | 5.82 | 1.113 | 18 | yes | 661 |
| star6 | 592 | 6.23 | 8.10 | 1.299 | 17 | no | 504 |
| trapezoid | 1189 | 11.25 | 17.15 | 1.525 | 15 | no | 590 |

## summary

{
  "n": 18,
  "beats_on_J": [
    "circle",
    "hexagon",
    "triangle",
    "equilateral_triangle",
    "L_shape",
    "H_shape",
    "T_shape",
    "diamond",
    "ellipse",
    "octagon",
    "pentagon",
    "star",
    "trapezoid"
  ],
  "beats_on_IoU": [
    "circle",
    "hexagon",
    "triangle",
    "equilateral_triangle",
    "L_shape",
    "H_shape",
    "T_shape",
    "diamond",
    "ellipse",
    "octagon",
    "pentagon",
    "star",
    "trapezoid"
  ],
  "solved": [
    "square",
    "circle",
    "hexagon",
    "ellipse",
    "octagon",
    "rounded_rect",
    "trapezoid"
  ],
  "energy_gate_violations": {},
  "horizon_flags": {
    "L_shape": [
      "HIST_best"
    ],
    "cross": [
      "A1_4bpp",
      "A1_2bpp"
    ],
    "diamond": [
      "A1_cont",
      "A1_4bpp",
      "A1_2bpp"
    ]
  },
  "classes": {
    "square": "SOLVED",
    "circle": "SOLVED",
    "hexagon": "SOLVED",
    "triangle": "IMPROVED",
    "equilateral_triangle": "IMPROVED",
    "L_shape": "IMPROVED",
    "H_shape": "IMPROVED",
    "T_shape": "IMPROVED",
    "cross": "NOT RESCUED",
    "diamond": "IMPROVED",
    "ellipse": "SOLVED",
    "octagon": "SOLVED",
    "pentagon": "IMPROVED",
    "rectangle": "NOT RESCUED",
    "rounded_rect": "SOLVED",
    "star": "IMPROVED",
    "star6": "NOT RESCUED",
    "trapezoid": "SOLVED"
  },
  "max_energy_residual": 0.022142437927782525,
  "max_clip": 0.0
}
