
## C1: SpreadsheetBench Verified, 400 tasks x 2 runs

| harness | pass r1 | pass r2 |
|---|---|---|
| CC | 353/400 | 354/400 |
| mini | 355/400 | 348/400 |
| OC | 352/400 | 355/400 |

Paired harness differences (A minus B, percentage points). Per run: A-only / B-only, paired difference with 90% Wald interval, exact McNemar p. Pooled: W/L summed over runs; per task the mean over runs, then mean over tasks, 90% t interval.

| A - B | r1: diff [90% CI], A-only / B-only, p | r2: diff [90% CI], A-only / B-only, p | pooled W / L | pooled diff [90% CI t] |
|---|---|---|---|---|
| CC - mini | -0.5 [-2.4, +1.4], 10 / 12, p=0.832 | +1.5 [-0.3, +3.3], 13 / 7, p=0.263 | 23 / 19 | +0.5 [-0.9, +1.9] |
| CC - OC | +0.2 [-1.2, +1.7], 7 / 6, p=1.000 | -0.2 [-2.0, +1.5], 9 / 10, p=1.000 | 16 / 16 | +0.0 [-1.2, +1.2] |
| mini - OC | +0.8 [-1.1, +2.6], 12 / 9, p=0.664 | -1.8 [-3.7, +0.2], 8 / 15, p=0.210 | 20 / 24 | -0.5 [-1.9, +0.9] |

Rerun flips (same harness, run 1 vs run 2):

| harness | pass->fail | fail->pass | flips | flip rate |
|---|---|---|---|---|
| CC | 7 | 8 | 15 | 3.8% |
| mini | 12 | 5 | 17 | 4.2% |
| OC | 10 | 13 | 23 | 5.8% |

Harness-swap flips (share of tasks whose outcome differs when only the harness changes), averaged over the four run pairings (A r1-B r1, A r2-B r2, A r1-B r2, A r2-B r1).

| A vs B | swap flips per pairing (r1r1, r2r2, r1r2, r2r1) | swap flip rate |
|---|---|---|
| CC vs mini | 22, 20, 21, 17 | 5.0% |
| CC vs OC | 13, 19, 22, 20 | 4.6% |
| mini vs OC | 21, 23, 20, 22 | 5.4% |

## C2: DABstep hard, fixed random 150, 150 tasks x 2 runs

| harness | pass r1 | pass r2 |
|---|---|---|
| CC | 111/150 | 112/150 |
| mini | 108/150 | 114/150 |
| OC | 107/150 | 113/150 |

Paired harness differences (A minus B, percentage points). Per run: A-only / B-only, paired difference with 90% Wald interval, exact McNemar p. Pooled: W/L summed over runs; per task the mean over runs, then mean over tasks, 90% t interval.

| A - B | r1: diff [90% CI], A-only / B-only, p | r2: diff [90% CI], A-only / B-only, p | pooled W / L | pooled diff [90% CI t] |
|---|---|---|---|---|
| CC - mini | +2.0 [-3.7, +7.7], 15 / 12, p=0.701 | -1.3 [-6.5, +3.8], 10 / 12, p=0.832 | 25 / 24 | +0.3 [-3.8, +4.4] |
| CC - OC | +2.7 [-3.5, +8.9], 18 / 14, p=0.597 | -0.7 [-5.9, +4.6], 11 / 12, p=1.000 | 29 / 26 | +1.0 [-3.4, +5.4] |
| mini - OC | +0.7 [-5.4, +6.8], 16 / 15, p=1.000 | +0.7 [-4.8, +6.1], 13 / 12, p=1.000 | 29 / 27 | +0.7 [-3.5, +4.8] |

Rerun flips (same harness, run 1 vs run 2):

| harness | pass->fail | fail->pass | flips | flip rate |
|---|---|---|---|---|
| CC | 10 | 11 | 21 | 14.0% |
| mini | 10 | 16 | 26 | 17.3% |
| OC | 10 | 16 | 26 | 17.3% |

Harness-swap flips (share of tasks whose outcome differs when only the harness changes), averaged over the four run pairings (A r1-B r1, A r2-B r2, A r1-B r2, A r2-B r1).

| A vs B | swap flips per pairing (r1r1, r2r2, r1r2, r2r1) | swap flip rate |
|---|---|---|
| CC vs mini | 27, 22, 25, 28 | 17.0% |
| CC vs OC | 32, 23, 28, 29 | 18.7% |
| mini vs OC | 31, 25, 27, 25 | 18.0% |
