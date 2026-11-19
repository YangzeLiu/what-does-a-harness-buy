# DeepSeek-V4.1-Flash, pool447: Holm-adjusted paired tests

Scoring as in `ds_pool447_tables.md`: 445 tasks (two scikit-learn tasks unscored), pass = reward >= 1. Script: `analysis/round2/ds_pool447_holm.py`.

Per run (exact McNemar, Holm within the run):

| pair | run 1 A-only / B-only | p | Holm | run 2 A-only / B-only | p | Holm |
|---|---|---|---|---|---|---|
| CC - mini | 16 / 16 | 1.000 | 1.000 | 26 / 12 | 0.034 | 0.067 |
| CC - OC | 17 / 18 | 1.000 | 1.000 | 18 / 19 | 1.000 | 1.000 |
| mini - OC | 15 / 16 | 1.000 | 1.000 | 7 / 22 | 0.008 | 0.024 |

Pooled over both runs (task x run pairs, 890 per pair; exact McNemar on the pooled discordant pairs, Holm over the three pairs):

| pair | pooled W / L | McNemar p | Holm |
|---|---|---|---|
| CC - mini | 42 / 28 | 0.120 | 0.239 |
| CC - OC | 35 / 37 | 0.906 | 0.906 |
| mini - OC | 22 / 38 | 0.052 | 0.156 |
