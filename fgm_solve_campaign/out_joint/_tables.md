# Joint per-angle re-solve, tables

## Headline: did the best angle move?

| shape | actuator | fixed-map best deg | joint best deg | move deg | J fixed-map best | J joint best | J improvement % | IoU joint best | crosses 0.80 | crosses 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| T_shape | sigma | 90 | 90 | +0.0 | 552.91 | 530.27 | +4.1 | 0.5339 | no | no |
| L_shape | sigma | 135 | 135 | +0.0 | 380.42 | 378.98 | +0.4 | 0.6669 | no | no |
| cross | sigma | 30 | 45 | +15.0 | 257.77 | 206.16 | +20.0 | 0.8048 | YES | no |
| star | sigma | 0 | 18 | +18.0 | 157.39 | 126.74 | +19.5 | 0.7448 | no | no |
| cross | eps | 30 | 45 | +15.0 | 257.77 | 125.61 | +51.3 | 0.8737 | YES | no |

## Depth check: the same angles re-solved at 40 forward-equivalents per start

| shape | actuator | angles refined | scan best | refined best | argmin survives depth | J refined best | IoU refined best |
|---|---|---|---|---|---|---|---|
| T_shape | sigma | 90, 135 | 90 | 90 | YES | 522.11 | 0.5364 |
| L_shape | sigma | 135, 180 | 135 | 135 | YES | 376.75 | 0.6693 |
| cross | sigma | 30, 45 | 45 | 45 | YES | 146.80 | 0.8497 |
| star | sigma | 0, 18, 27 | 18 | 18 | YES | 106.27 | 0.7852 |
| cross | eps | 0, 15, 30, 45 | 45 | 0 | NO | 100.22 | 0.8919 |

Per-angle refined values:

| shape | actuator | angle | J 4 bpp | IoU | start | max T C |
|---|---|---|---|---|---|---|
| T_shape | sigma | 90 | 522.11 | 0.5364 | cold | 201.6 |
| T_shape | sigma | 135 | 523.40 | 0.5282 | cold | 207.7 |
| L_shape | sigma | 135 | 376.75 | 0.6693 | warm | 241.9 |
| L_shape | sigma | 180 | 527.55 | 0.5139 | warm | 222.7 |
| cross | sigma | 30 | 198.55 | 0.8066 | cold | 233.1 |
| cross | sigma | 45 | 146.80 | 0.8497 | warm | 203.7 |
| star | sigma | 0 | 140.10 | 0.7270 | warm | 197.9 |
| star | sigma | 18 | 106.27 | 0.7852 | cold | 200.5 |
| star | sigma | 27 | 138.61 | 0.7189 | warm | 196.1 |
| cross | eps | 0 | 100.22 | 0.8919 | warm | 212.0 |
| cross | eps | 15 | 185.22 | 0.8267 | warm | 232.5 |
| cross | eps | 30 | 141.14 | 0.8638 | cold | 239.8 |
| cross | eps | 45 | 109.09 | 0.8931 | warm | 220.6 |

## The campaign's own noise floor: symmetry-equivalent angle pairs

The forward physics is identical at these angle pairs, so any disagreement is the solve's angle-to-angle reproducibility, not a physical effect. The uniform column is the fixed-map sweep's uniform arm at the same pair and is the reference for how exactly the equivalence holds in the forward.

| shape | actuator | pair deg | J joint a | J joint b | relative gap % | uniform relative gap % |
|---|---|---|---|---|---|---|
| T_shape | sigma | 0 and 180 | 610.49 | 610.08 | 0.07 | 0.007 |
| T_shape | sigma | 22.5 and 157.5 | 669.21 | 669.23 | 0.00 | 0.004 |
| T_shape | sigma | 45 and 135 | 538.74 | 537.57 | 0.22 | 0.164 |
| T_shape | sigma | 67.5 and 112.5 | 658.86 | 656.75 | 0.32 | 0.722 |
| L_shape | sigma | 0 and 180 | 530.47 | 530.06 | 0.08 | 0.008 |
| star | sigma | 0 and 36 | 145.23 | 156.26 | 7.59 | 0.000 |
| star | sigma | 9 and 27 | 140.69 | 139.27 | 1.01 | 0.000 |

## Budget actually spent

| shape | actuator | gradient evaluations per start | starts per angle | forward-equivalents per start | forward-equivalents per angle | angles | wall s |
|---|---|---|---|---|---|---|---|
| T_shape | sigma | 6 | 2 | 13.8 | 29.6 (plus 2 scoring forwards) | 9 | 1363 |
| L_shape | sigma | 6 | 2 | 13.0 | 28.1 (plus 2 scoring forwards) | 9 | 1399 |
| cross | sigma | 6 | 2 | 14.7 | 31.5 (plus 2 scoring forwards) | 4 | 581 |
| star | sigma | 7 | 2 | 14.8 | 31.6 (plus 2 scoring forwards) | 5 | 464 |
| cross | eps | 6 | 2 | 14.7 | 31.5 (plus 2 scoring forwards) | 4 | 468 |

## Per-shape angle tables

### T_shape, actuator sigma

| angle deg | J joint 4 bpp | J fixed map | J uniform | IoU joint | IoU fixed | grow % | under % | rho at stop | P_abs W/m | stop s | max T C | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 610.49 | 609.29 | 614.26 | 0.4591 | 0.4444 | 4.17 | 52.17 | 0.6242 | 493.6 | 234.5 | 219.0 | cold |
| 22.5 | 669.21 | 674.66 | 674.74 | 0.3978 | 0.4015 | 7.18 | 57.37 | 0.6358 | 406.7 | 290.0 | 220.6 | cold |
| 45 | 538.74 | 564.21 | 565.36 | 0.5240 | 0.5126 | 10.40 | 42.15 | 0.6949 | 323.1 | 496.5 | 214.1 | cold |
| 67.5 | 658.86 | 698.15 | 698.33 | 0.4298 | 0.3980 | 13.54 | 51.20 | 0.7034 | 243.4 | 750.0 H | 229.2 | warm |
| **90** | **530.27** | 552.91 | 552.32 | 0.5339 | 0.4939 | 4.35 | 44.29 | 0.6865 | 263.1 | 627.5 | 206.5 | warm |
| 112.5 | 656.75 | 693.14 | 693.33 | 0.4297 | 0.4028 | 13.35 | 51.29 | 0.7033 | 243.9 | 750.0 H | 229.3 | warm |
| 135 | 537.57 | 565.14 | 566.28 | 0.5187 | 0.5090 | 9.58 | 43.16 | 0.6918 | 320.4 | 498.5 | 213.0 | warm |
| 157.5 | 669.23 | 674.68 | 674.77 | 0.3978 | 0.4015 | 7.18 | 57.37 | 0.6360 | 406.4 | 290.5 | 220.7 | cold |
| 180 | 610.08 | 609.27 | 614.21 | 0.4591 | 0.4444 | 4.17 | 52.17 | 0.6244 | 492.6 | 235.0 | 219.0 | cold |

### L_shape, actuator sigma

| angle deg | J joint 4 bpp | J fixed map | J uniform | IoU joint | IoU fixed | grow % | under % | rho at stop | P_abs W/m | stop s | max T C | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 530.47 | 521.20 | 538.42 | 0.5100 | 0.5209 | 6.86 | 45.51 | 0.6493 | 493.6 | 261.0 | 222.8 | warm |
| 22.5 | 601.93 | 605.52 | 607.66 | 0.4494 | 0.4491 | 6.65 | 52.08 | 0.6687 | 313.5 | 442.0 | 216.7 | cold |
| 45 | 684.91 | 704.06 | 693.48 | 0.3441 | 0.3226 | 0.63 | 65.38 | 0.6059 | 218.8 | 750.0 H | 195.4 | cold |
| 67.5 | 997.01 | 1048.62 | 1008.10 | 0.0452 | 0.0000 | 0.00 | 95.48 | 0.5550 | 163.2 | 750.0 H | 181.8 | cold |
| 90 | 649.14 | 669.82 | 677.95 | 0.4369 | 0.3891 | 13.07 | 50.60 | 0.7027 | 240.4 | 728.0 | 222.9 | cold |
| 112.5 | 543.39 | 541.44 | 551.76 | 0.5513 | 0.5491 | 19.76 | 33.98 | 0.7460 | 361.7 | 492.0 | 249.0 | cold |
| **135** | **378.98** | 380.42 | 395.13 | 0.6669 | 0.6664 | 14.21 | 23.83 | 0.7217 | 529.5 | 312.0 | 242.9 | warm |
| 157.5 | 539.49 | 545.00 | 553.70 | 0.5469 | 0.5418 | 18.19 | 35.36 | 0.7199 | 529.5 | 292.5 | 249.8 | cold |
| 180 | 530.06 | 521.18 | 538.38 | 0.5109 | 0.5209 | 6.67 | 45.51 | 0.6491 | 493.4 | 261.0 | 222.7 | warm |

### cross, actuator sigma

| angle deg | J joint 4 bpp | J fixed map | J uniform | IoU joint | IoU fixed | grow % | under % | rho at stop | P_abs W/m | stop s | max T C | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 391.88 | 360.18 | 471.82 | 0.6394 | 0.6755 | 10.81 | 29.15 | 0.7680 | 366.2 | 422.0 | 235.5 | warm |
| 15 | 349.64 | 333.16 | 408.15 | 0.6794 | 0.6938 | 11.93 | 23.96 | 0.7778 | 388.3 | 420.5 | 236.2 | warm |
| 30 | 222.12 | 257.77 | 301.70 | 0.7834 | 0.7492 | 10.61 | 13.35 | 0.7586 | 416.2 | 399.0 | 231.8 | warm |
| **45** | **206.16** | 270.93 | 283.65 | 0.8048 | 0.7462 | 11.13 | 10.57 | 0.7875 | 365.9 | 478.5 | 237.3 | warm |

### star, actuator sigma

| angle deg | J joint 4 bpp | J fixed map | J uniform | IoU joint | IoU fixed | grow % | under % | rho at stop | P_abs W/m | stop s | max T C | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 145.23 | 157.39 | 192.58 | 0.7172 | 0.7032 | 7.01 | 23.25 | 0.6691 | 291.6 | 334.0 | 198.9 | warm |
| 9 | 140.69 | 212.91 | 237.92 | 0.7104 | 0.6129 | 4.32 | 25.89 | 0.6255 | 382.7 | 215.0 | 197.1 | warm |
| **18** | **126.74** | 220.55 | 245.74 | 0.7448 | 0.5898 | 5.54 | 21.40 | 0.6391 | 378.9 | 222.5 | 199.9 | warm |
| 27 | 139.27 | 207.42 | 237.92 | 0.7176 | 0.6246 | 4.32 | 25.14 | 0.6221 | 384.9 | 213.5 | 195.9 | warm |
| 36 | 156.26 | 183.45 | 192.58 | 0.6991 | 0.6535 | 4.24 | 27.12 | 0.6174 | 391.3 | 203.0 | 201.6 | warm |

### cross, actuator eps

| angle deg | J joint 4 bpp | J fixed map | J uniform | IoU joint | IoU fixed | grow % | under % | rho at stop | P_abs W/m | stop s | max T C | start |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 226.30 | 360.18 | 471.82 | 0.7068 | 0.6755 | 13.90 | 19.50 | 0.7149 | 355.0 | 489.5 | 212.2 | warm |
| 15 | 188.78 | 333.16 | 408.15 | 0.8258 | 0.6938 | 13.64 | 6.16 | 0.7527 | 495.2 | 329.5 | 240.8 | warm |
| 30 | 176.83 | 257.77 | 301.70 | 0.8357 | 0.7492 | 11.84 | 6.53 | 0.7528 | 596.5 | 265.0 | 252.1 CEILING | cold |
| **45** | **125.61** | 270.93 | 283.65 | 0.8737 | 0.7462 | 7.55 | 6.04 | 0.7249 | 535.0 | 283.0 | 217.9 | cold |
