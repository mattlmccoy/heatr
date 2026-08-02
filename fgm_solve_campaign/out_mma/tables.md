
**J against the grid-independent area-fill target, grid 120**

| shape | a_filteronly_lbfgsb_40 | e_projection_lbfgsb_40 | b_projection_mma_40 | c_projection_mma_80 | d_projection_lbfgsbcarry_80 |
|---|---|---|---|---|---|
| square |     12.30 |    154.91 |    117.91 |     43.26 |     44.35 |
| circle |      2.50 |     45.13 |    240.15 |     41.57 |     51.47 |
| trapezoid |     15.82 |     61.35 |    141.02 |     34.18 |     48.27 |
| triangle |     69.93 |     81.92 |     80.25 |     49.26 |     77.02 |
| diamond |    187.53 |    243.96 |    232.13 |    217.52 |    234.43 |
| rectangle |    182.40 |    189.50 |    184.42 |    180.31 |    184.80 |

**J_raster_chi, the same map under the old binary target, grid 120**

| shape | a_filteronly_lbfgsb_40 | e_projection_lbfgsb_40 | b_projection_mma_40 | c_projection_mma_80 | d_projection_lbfgsbcarry_80 |
|---|---|---|---|---|---|
| square |     18.88 |    165.72 |    126.99 |     50.84 |     54.13 |
| circle |     13.07 |     69.05 |    258.49 |     56.68 |     70.14 |
| trapezoid |     26.08 |     72.16 |    153.41 |     43.85 |     65.01 |
| triangle |     81.79 |     94.45 |     93.09 |     58.79 |     89.62 |
| diamond |    214.84 |    282.01 |    266.83 |    245.91 |    274.83 |
| rectangle |    188.67 |    194.86 |    189.89 |    186.32 |    190.54 |

**IoU against the binary part mask, grid 120**

| shape | a_filteronly_lbfgsb_40 | e_projection_lbfgsb_40 | b_projection_mma_40 | c_projection_mma_80 | d_projection_lbfgsbcarry_80 |
|---|---|---|---|---|---|
| square |  1.0000 |  0.8734 |  0.8989 |  0.9574 |  0.9572 |
| circle |  0.9968 |  0.9190 |  0.7983 |  0.9375 |  0.9279 |
| trapezoid |  0.9735 |  0.9308 |  0.8465 |  0.9495 |  0.9309 |
| triangle |  0.8815 |  0.8538 |  0.8483 |  0.8907 |  0.8541 |
| diamond |  0.8501 |  0.8115 |  0.8145 |  0.8341 |  0.8127 |
| rectangle |  0.8580 |  0.8558 |  0.8585 |  0.8580 |  0.8558 |

**growth: melted bed as a percent of part area, grid 120**

| shape | a_filteronly_lbfgsb_40 | e_projection_lbfgsb_40 | b_projection_mma_40 | c_projection_mma_80 | d_projection_lbfgsbcarry_80 |
|---|---|---|---|---|---|
| square |    0.00 |    7.62 |    2.62 |    2.62 |    2.12 |
| circle |    0.32 |    3.55 |   13.55 |    3.23 |    2.90 |
| trapezoid |    1.51 |    3.28 |    8.49 |    3.28 |    3.45 |
| triangle |    5.50 |    6.00 |    5.50 |    5.25 |    6.25 |
| diamond |    2.71 |    2.59 |    3.57 |    4.68 |    2.59 |
| rectangle |   12.50 |   13.19 |   12.85 |   12.50 |   13.19 |

**under: unmelted part as a percent of part area, grid 120**

| shape | a_filteronly_lbfgsb_40 | e_projection_lbfgsb_40 | b_projection_mma_40 | c_projection_mma_80 | d_projection_lbfgsbcarry_80 |
|---|---|---|---|---|---|
| square |    0.00 |    6.00 |    7.75 |    1.75 |    2.25 |
| circle |    0.00 |    4.84 |    9.35 |    3.23 |    4.52 |
| trapezoid |    1.18 |    3.87 |    8.16 |    1.93 |    3.70 |
| triangle |    7.00 |    9.50 |   10.50 |    6.25 |    9.25 |
| diamond |   12.68 |   16.75 |   15.64 |   12.68 |   16.63 |
| rectangle |    3.47 |    3.12 |    3.12 |    3.47 |    3.12 |

**non-discreteness M_nd = mean 4 s (1 - s) over the part**

| shape | a_filteronly_lbfgsb_40 | e_projection_lbfgsb_40 | b_projection_mma_40 | c_projection_mma_80 | d_projection_lbfgsbcarry_80 |
|---|---|---|---|---|---|
| square |  0.651 |  0.151 |  0.216 |  0.398 |  0.170 |
| circle |  0.690 |  0.026 |  0.114 |  0.428 |  0.179 |
| trapezoid |  0.549 |  0.056 |  0.089 |  0.369 |  0.064 |
| triangle |  0.479 |  0.093 |  0.169 |  0.302 |  0.122 |
| diamond |  0.465 |  0.087 |  0.233 |  0.156 |  0.103 |
| rectangle |  0.323 |  0.068 |  0.135 |  0.270 |  0.104 |

**gradient evaluations actually used**

| shape | a_filteronly_lbfgsb_40 | e_projection_lbfgsb_40 | b_projection_mma_40 | c_projection_mma_80 | d_projection_lbfgsbcarry_80 |
|---|---|---|---|---|---|
| square |  14 |  14 |  14 |  28 |  28 |
| circle |  14 |  14 |  14 |  29 |  29 |
| trapezoid |  15 |  15 |  15 |  31 |  31 |
| triangle |  16 |  16 |  16 |  32 |  32 |
| diamond |  15 |  15 |  15 |  31 |  31 |
| rectangle |  15 |  15 |  15 |  31 |  31 |