"""DeepSeek-V4.1-Flash on pool447 (3 harnesses x 2 runs): pass counts, paired differences and flip rates, same
statistics as bench2_report.py.
Usage: python3 -I ds_pool447_tables.py > ../../paper/round2/ds_pool447_tables.md
Input: data/round2/trials_round2.csv, experiment swe_dsv41flash_pool447.

Rules (fixed before the run): pass = reward >= 1; scikit-learn-25232 and scikit-learn-14710 are unscored for every
harness, so 445 tasks enter every table.
  * paired differences: per run A-only / B-only, diff with 90% Wald interval, exact McNemar p; pooled: per task
    the mean over runs, mean over tasks, 90% t interval.
  * rerun flips: same harness, run 1 vs run 2.
  * swap flips: tasks whose outcome differs when only the harness changes, averaged over the four run pairings
    (A1-B1, A2-B2, A1-B2, A2-B1)."""
import csv, math, os
from itertools import combinations

CSVP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "round2", "trials_round2.csv")
HS = {"claude-code": "CC", "mini-swe-agent": "mini", "opencode": "OC"}
UN = {"scikit-learn__scikit-learn-25232", "scikit-learn__scikit-learn-14710"}
H = ("CC", "mini", "OC")
Z90 = 1.6448536
rows = [r for r in csv.DictReader(open(CSVP)) if r["experiment"] == "swe_dsv41flash_pool447"]
O = {}
for r in rows:
    t = r["task"]
    if t in UN:
        continue
    k = (HS[r["harness"]], int(r["run"]), t)
    assert k not in O, k
    O[k] = 1 if r["reward"] and float(r["reward"]) >= 1 else 0
T = sorted({t for _, _, t in O})
assert all((h, p, t) in O for h in H for p in (1, 2) for t in T), "missing cell"
print(f"tasks={len(T)} counted cells={len(O)}")


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


print("\n| harness | run 1 | run 2 | mean pass rate |\n|---|---|---|---|")
for h in H:
    k1, k2 = (sum(O[(h, p, t)] for t in T) for p in (1, 2))
    print(f"| {h} | {k1}/{len(T)} | {k2}/{len(T)} | {100 * (k1 + k2) / (2 * len(T)):.1f}% |")

print("\n| A - B | run 1: diff [90% Wald], A-only / B-only, McNemar p | run 2: same | pooled W / L | pooled diff [90% t] |")
print("|---|---|---|---|---|")
for a, b in combinations(H, 2):
    cells, W, L = [], 0, 0
    for p in (1, 2):
        d = [O[(a, p, t)] - O[(b, p, t)] for t in T]
        bb, cc = d.count(1), d.count(-1); W += bb; L += cc
        m = sum(d) / len(d); se = math.sqrt((sum(x * x for x in d) / len(d) - m * m) / len(d))
        cells.append(f"{100 * m:+.1f} [{100 * (m - Z90 * se):+.1f}, {100 * (m + Z90 * se):+.1f}], {bb} / {cc}, p={mcnemar(bb, cc):.3f}")
    d = [(O[(a, 1, t)] + O[(a, 2, t)] - O[(b, 1, t)] - O[(b, 2, t)]) / 2 for t in T]
    m = sum(d) / len(d); sd = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1)); se = sd / math.sqrt(len(d))
    print(f"| {a} - {b} | {cells[0]} | {cells[1]} | {W} / {L} | {100 * m:+.1f} [{100 * (m - Z90 * se):+.1f}, {100 * (m + Z90 * se):+.1f}] |")

print("\nRerun flips (same harness, run 1 vs run 2):\n\n| harness | pass->fail | fail->pass | flips | flip rate |\n|---|---|---|---|---|")
for h in H:
    pf = sum(1 for t in T if O[(h, 1, t)] > O[(h, 2, t)]); fp = sum(1 for t in T if O[(h, 1, t)] < O[(h, 2, t)])
    print(f"| {h} | {pf} | {fp} | {pf + fp} | {100 * (pf + fp) / len(T):.1f}% |")

print("\nHarness-swap flips, averaged over the four run pairings.\n")
print("| A vs B | swap flips (11, 22, 12, 21) | swap flip rate |\n|---|---|---|")
for a, b in combinations(H, 2):
    f = [sum(1 for t in T if O[(a, pa, t)] != O[(b, pb, t)]) for pa, pb in ((1, 1), (2, 2), (1, 2), (2, 1))]
    print(f"| {a} vs {b} | {', '.join(map(str, f))} | {100 * sum(f) / 4 / len(T):.1f}% |")
