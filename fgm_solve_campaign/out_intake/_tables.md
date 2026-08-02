# Geometry intake, all tables

Grid 120 throughout. Anisotropy is the occupancy-weighted azimuthal
anisotropy of `adjoint2d.geometry_actuator`, measured against the
24-angle reference rotation group, uniform dopant map, electrical
state B.


## Table A. Intake fidelity against the stored library configurations

| shape | part mask matches | cells differing | voltage calibrated (V) | voltage stored (V) | relative error | part cells | raster minus area-fill area |
|---|---|---|---|---|---|---|---|
| square | True | 0 | 2428.17 | 2428.17 | 0.000000 pct | 1600 | +1.58 pct |
| circle | True | 0 | 3399.63 | 3399.63 | 0.000000 pct | 1240 | +0.34 pct |
| hexagon | True | 0 | 3688.21 | 3688.21 | 0.000000 pct | 1016 | -0.63 pct |
| triangle | True | 0 | 3005.54 | 3005.54 | 0.000000 pct | 800 | +1.58 pct |
| equilateral_triangle | True | 0 | 3891.44 | 3891.44 | 0.000000 pct | 674 | -1.14 pct |
| L_shape | True | 0 | 1804.22 | 1804.22 | 0.000000 pct | 1079 | -0.57 pct |
| H_shape | True | 0 | 2329.59 | 2329.59 | 0.000000 pct | 1152 | +1.63 pct |
| T_shape | True | 0 | 1718.98 | 1718.98 | 0.000000 pct | 1104 | +1.71 pct |
| cross | True | 0 | 2815.38 | 2815.38 | 0.000000 pct | 1036 | -2.13 pct |
| diamond | True | 0 | 2475.40 | 2475.40 | 0.000000 pct | 1624 | +3.23 pct |
| ellipse | True | 0 | 4900.01 | 4900.01 | 0.000000 pct | 744 | +0.34 pct |
| octagon | True | 0 | 4233.54 | 4233.54 | 0.000000 pct | 1076 | -3.26 pct |
| pentagon | True | 0 | 3669.74 | 3669.74 | 0.000000 pct | 930 | -0.53 pct |
| rectangle | True | 0 | 3316.22 | 3316.22 | 0.000000 pct | 1152 | +1.58 pct |
| rounded_rect | True | 0 | 2876.69 | 2876.69 | 0.000000 pct | 1536 | +1.63 pct |
| star | True | 0 | 3223.45 | 3223.45 | 0.000000 pct | 542 | +1.38 pct |
| star6 | True | 0 | 3468.09 | 3468.09 | 0.000000 pct | 592 | +0.33 pct |
| trapezoid | True | 0 | 2635.83 | 2635.83 | 0.000000 pct | 1189 | +0.70 pct |

## Table B. Symmetry detection

| shape | detected order | point group | mirror axes | order the library defines | agrees |
|---|---|---|---|---|---|
| square | 4 | C4v | 4 | 4 | yes |
| circle | 12 | O(2) | 1 | - | - |
| hexagon | 6 | C6v | 6 | 6 | yes |
| triangle | 1 | C1v | 1 | 1 | yes |
| equilateral_triangle | 3 | C3v | 3 | 3 | yes |
| L_shape | 1 | C1 | 0 | 1 | yes |
| H_shape | 2 | C2v | 2 | 2 | yes |
| T_shape | 1 | C1v | 1 | 1 | yes |
| cross | 4 | C4v | 4 | 4 | yes |
| diamond | 4 | C4v | 4 | 4 | yes |
| ellipse | 2 | C2v | 2 | 2 | yes |
| octagon | 10 | C10v | 8 | 8 | NO |
| pentagon | 5 | C5v | 5 | 5 | yes |
| rectangle | 2 | C2v | 2 | 2 | yes |
| rounded_rect | 2 | C2v | 2 | - | - |
| star | 5 | C5v | 5 | 5 | yes |
| star6 | 6 | C6v | 6 | 6 | yes |
| trapezoid | 1 | C1v | 1 | 1 | yes |

## Table C. Anisotropy spectrum and actuator recommendation

| shape | static | continuous | best indexing | recommended mode | residual | reduction | class | measured rotation outcome | static-solve class (out_lib) |
|---|---|---|---|---|---|---|---|---|---|
| square | 0.5634 | 0.2766 | index4 0.4503 | continuous | 0.2766 | 2.04 | MODE_SUFFICES | rotation_wins | SOLVED |
| circle | 0.3167 | 0.0621 | index12 0.0707 | continuous | 0.0621 | 5.10 | MODE_SUFFICES | - | SOLVED |
| hexagon | 0.3156 | 0.1517 | index6 0.1625 | continuous | 0.1517 | 2.08 | MODE_SUFFICES | - | SOLVED |
| triangle | 0.6691 | 0.6241 | - | static | 0.6691 | 1.07 | MAP_PLUS_MODE | - | IMPROVED |
| equilateral_triangle | 0.6118 | 0.5656 | index3 0.5615 | static | 0.6118 | 1.09 | MAP_PLUS_MODE | - | IMPROVED |
| L_shape | 1.2560 | 1.0977 | - | static | 1.2560 | 1.14 | PHYSICAL_LIMIT | rotation_fails | IMPROVED |
| H_shape | 0.8941 | 0.6170 | index2 0.8941 | continuous | 0.6170 | 1.45 | MAP_PLUS_MODE | - | IMPROVED |
| T_shape | 0.8041 | 0.8366 | - | static | 0.8041 | 1.00 | PHYSICAL_LIMIT | rotation_fails | IMPROVED |
| cross | 0.8223 | 0.4190 | index4 0.5251 | continuous | 0.4190 | 1.96 | MODE_SUFFICES | rotation_wins | NOT RESCUED |
| diamond | 0.5161 | 0.2785 | index4 0.3756 | continuous | 0.2785 | 1.85 | MODE_SUFFICES | - | IMPROVED |
| ellipse | 0.7918 | 0.3287 | index2 0.7918 | continuous | 0.3287 | 2.41 | MODE_SUFFICES | - | SOLVED |
| octagon | 0.1029 | 0.0700 | index5 0.0849 | continuous | 0.0700 | 1.47 | MODE_SUFFICES | - | SOLVED |
| pentagon | 0.4451 | 0.1925 | index5 0.2587 | continuous | 0.1925 | 2.31 | MODE_SUFFICES | - | IMPROVED |
| rectangle | 1.0169 | 0.4664 | index2 1.0169 | continuous | 0.4664 | 2.18 | MODE_SUFFICES | - | NOT RESCUED |
| rounded_rect | 0.6692 | 0.3007 | index2 0.6692 | continuous | 0.3007 | 2.23 | MODE_SUFFICES | - | SOLVED |
| star | 0.7041 | 0.4618 | index5 0.4631 | continuous | 0.4618 | 1.52 | MODE_SUFFICES | rotation_wins | IMPROVED |
| star6 | 0.5790 | 0.3914 | index3 0.3997 | continuous | 0.3914 | 1.48 | MODE_SUFFICES | - | NOT RESCUED |
| trapezoid | 0.5149 | 0.3334 | - | continuous | 0.3334 | 1.54 | MODE_SUFFICES | - | SOLVED |

## Table D. Classifier against the five MEASURED rotation outcomes

| shape | residual | predicted class | predicted rotation helps | measured outcome | agrees |
|---|---|---|---|---|---|
| square | 0.2766 | MODE_SUFFICES | True | rotation_wins | yes |
| L_shape | 1.2560 | PHYSICAL_LIMIT | False | rotation_fails | yes |
| T_shape | 0.8041 | PHYSICAL_LIMIT | False | rotation_fails | yes |
| cross | 0.4190 | MODE_SUFFICES | True | rotation_wins | yes |
| star | 0.4618 | MODE_SUFFICES | True | rotation_wins | yes |

Agreement: **5 of 5**. These five are the points the bands were CALIBRATED on, so this is threshold reproduction, not out-of-sample validation.


## Table E. Novel geometries, not in the library

| shape | vertices | part cells | calibrated V | order | mirrors | static | best mode | residual | class |
|---|---|---|---|---|---|---|---|---|---|
| arrow | 7 | 890 | 3830.1 | 1 | 1 | 1.1266 | continuous | 0.5312 | MAP_PLUS_MODE |
| gear8 | 72 | 1144 | 3098.7 | 8 | 8 | 0.4440 | continuous | 0.2280 | MODE_SUFFICES |
| keyhole | 98 | 1030 | 2630.3 | 1 | 1 | 0.6110 | continuous | 0.5156 | MAP_PLUS_MODE |

## Table F. Solve arms, gear8 (grid 120, area-fill chi)

| arm | J | J vs raster chi | IoU | IoU_area | growth pct | under-melt pct | stop s | horizon | P_abs W/m | max T C | energy residual pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U_uniform_static | 207.89 | 222.85 | 0.8042 | 0.7908 | 8.92 | 12.41 | 320.0 | - | 500.0 | 214.0 | 0.81 |
| A_static_cont | 131.72 | 149.83 | 0.8437 | 0.8417 | 8.48 | 8.48 | 412.5 | - | 409.0 | 208.3 | 0.69 |
| A_static_4bpp | 132.36 | 150.56 | 0.8458 | 0.8412 | 8.30 | 8.39 | 412.5 | - | 409.0 | 208.4 | 0.69 |
| U_uniform_continuous | 138.59 | 164.72 | 0.8273 | 0.8393 | 12.85 | 6.64 | 332.5 | - | 460.6 | 218.6 | 1.00 |
| A_continuous_cont | 138.59 | 164.72 | 0.8273 | 0.8393 | 12.85 | 6.64 | 332.5 | - | 460.6 | 218.6 | 1.00 |
| A_continuous_4bpp | 138.59 | 164.72 | 0.8273 | 0.8393 | 12.85 | 6.64 | 332.5 | - | 460.6 | 218.6 | 1.00 |
| U_uniform_index8 | 148.63 | 175.78 | 0.8215 | 0.8342 | 13.64 | 6.64 | 353.0 | - | 465.2 | 223.5 | 1.02 |
| A_index8_cont | 148.63 | 175.78 | 0.8215 | 0.8342 | 13.64 | 6.64 | 353.0 | - | 465.2 | 223.5 | 1.02 |
| A_index8_4bpp | 148.63 | 175.78 | 0.8215 | 0.8342 | 13.64 | 6.64 | 353.0 | - | 465.2 | 223.5 | 1.02 |

## Table F. Solve arms, keyhole (grid 120, area-fill chi)

| arm | J | J vs raster chi | IoU | IoU_area | growth pct | under-melt pct | stop s | horizon | P_abs W/m | max T C | energy residual pct |
|---|---|---|---|---|---|---|---|---|---|---|---|
| U_uniform_static | 509.96 | 524.99 | 0.5750 | 0.5659 | 30.68 | 24.85 | 336.0 | - | 500.0 | 290.5 | 2.27 |
| A_static_cont | 168.25 | 191.67 | 0.8143 | 0.8158 | 13.98 | 7.18 | 500.5 | - | 366.7 | 252.8 | 0.97 |
| A_static_4bpp | 168.35 | 191.73 | 0.8163 | 0.8162 | 14.17 | 6.80 | 501.5 | - | 367.2 | 253.1 | 0.98 |
| U_uniform_continuous | 176.35 | 196.11 | 0.7973 | 0.7949 | 13.01 | 9.90 | 573.0 | - | 301.3 | 223.9 | 0.89 |
| A_continuous_cont | 8.15 | 23.54 | 0.9753 | 0.9653 | 2.14 | 0.39 | 748.0 | - | 245.5 | 212.4 | 0.19 |
| A_continuous_4bpp | 8.08 | 23.32 | 0.9753 | 0.9656 | 2.14 | 0.39 | 747.5 | - | 245.7 | 212.4 | 0.20 |
