tasks=445 counted cells=2670

| harness | run 1 | run 2 | mean pass rate |
|---|---|---|---|
| CC | 368/445 | 374/445 | 83.4% |
| mini | 368/445 | 360/445 | 81.8% |
| OC | 369/445 | 375/445 | 83.6% |

| A - B | run 1: diff [90% Wald], A-only / B-only, McNemar p | run 2: same | pooled W / L | pooled diff [90% t] |
|---|---|---|---|---|
| CC - mini | +0.0 [-2.1, +2.1], 16 / 16, p=1.000 | +3.1 [+0.9, +5.4], 26 / 12, p=0.034 | 42 / 28 | +1.6 [-0.0, +3.2] |
| CC - OC | -0.2 [-2.4, +2.0], 17 / 18, p=1.000 | -0.2 [-2.5, +2.0], 18 / 19, p=1.000 | 35 / 37 | -0.2 [-1.8, +1.4] |
| mini - OC | -0.2 [-2.3, +1.8], 15 / 16, p=1.000 | -3.4 [-5.3, -1.4], 7 / 22, p=0.008 | 22 / 38 | -1.8 [-3.2, -0.4] |

Rerun flips (same harness, run 1 vs run 2):

| harness | pass->fail | fail->pass | flips | flip rate |
|---|---|---|---|---|
| CC | 11 | 17 | 28 | 6.3% |
| mini | 21 | 13 | 34 | 7.6% |
| OC | 13 | 19 | 32 | 7.2% |

Harness-swap flips, averaged over the four run pairings.

| A vs B | swap flips (11, 22, 12, 21) | swap flip rate |
|---|---|---|
| CC vs mini | 32, 38, 40, 28 | 7.8% |
| CC vs OC | 35, 37, 29, 35 | 7.6% |
| mini vs OC | 31, 29, 31, 31 | 6.9% |
