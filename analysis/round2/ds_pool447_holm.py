"""DeepSeek-V4.1-Flash on pool447: exact McNemar tests with Holm adjustment over the three harness pairs, per run
and pooled over both runs.
Usage: python3 -I ds_pool447_holm.py > ../../paper/round2/ds_pool447_holm.md
Input: data/round2/trials_round2.csv, experiment swe_dsv41flash_pool447.
Same scoring as ds_pool447_tables.py: pass = reward >= 1, the two scikit-learn tasks unscored."""
import csv, math, os
from itertools import combinations

CSVP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "round2", "trials_round2.csv")
HS = {"claude-code": "CC", "mini-swe-agent": "mini", "opencode": "OC"}
UN = {"scikit-learn__scikit-learn-25232", "scikit-learn__scikit-learn-14710"}
H = ("CC", "mini", "OC")
rows = [r for r in csv.DictReader(open(CSVP)) if r["experiment"] == "swe_dsv41flash_pool447"]
O = {}
for r in rows:
    t = r["task"]
    if t in UN:
        continue
    k = (HS[r["harness"]], int(r["run"]), t)
    O[k] = 1 if r["reward"] and float(r["reward"]) >= 1 else 0
T = sorted({t for _, _, t in O})


def mcn(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def holm(ps):
    o = sorted(range(len(ps)), key=lambda i: ps[i]); adj = [0] * len(ps); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def pooled_wl(a, b, tasks):
    W = sum(1 for p in (1, 2) for t in tasks if O[(a, p, t)] > O[(b, p, t)])
    L = sum(1 for p in (1, 2) for t in tasks if O[(a, p, t)] < O[(b, p, t)])
    return W, L


out = ["# DeepSeek-V4.1-Flash, pool447: Holm-adjusted paired tests", "",
       "Scoring as in `ds_pool447_tables.md`: 445 tasks (two scikit-learn tasks unscored), pass = reward >= 1. Script: `analysis/round2/ds_pool447_holm.py`.", ""]

P = list(combinations(H, 2))
out += ["Per run (exact McNemar, Holm within the run):", "", "| pair | run 1 A-only / B-only | p | Holm | run 2 A-only / B-only | p | Holm |", "|---|---|---|---|---|---|---|"]
per = {}
for p in (1, 2):
    bc = [(sum(1 for t in T if O[(a, p, t)] > O[(b, p, t)]), sum(1 for t in T if O[(a, p, t)] < O[(b, p, t)])) for a, b in P]
    ps = [mcn(*x) for x in bc]; per[p] = (bc, ps, holm(ps))
for i, (a, b) in enumerate(P):
    (b1, c1), (b2, c2) = per[1][0][i], per[2][0][i]
    out.append(f"| {a} - {b} | {b1} / {c1} | {per[1][1][i]:.3f} | {per[1][2][i]:.3f} | {b2} / {c2} | {per[2][1][i]:.3f} | {per[2][2][i]:.3f} |")
out += ["", "Pooled over both runs (task x run pairs, 890 per pair; exact McNemar on the pooled discordant pairs, Holm over the three pairs):", "",
        "| pair | pooled W / L | McNemar p | Holm |", "|---|---|---|---|"]
res = [pooled_wl(a, b, T) for a, b in P]
pm = [mcn(W, L) for W, L in res]
hm = holm(pm)
for i, (a, b) in enumerate(P):
    W, L = res[i]
    out.append(f"| {a} - {b} | {W} / {L} | {pm[i]:.3f} | {hm[i]:.3f} |")

print("\n".join(out))
