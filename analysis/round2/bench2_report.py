"""SpreadsheetBench Verified (400 tasks) and DABstep (fixed random 150 hard tasks), DeepSeek-V4.1-Flash, 3 harnesses x
2 runs: pass counts, paired harness differences, rerun flips and harness-swap flips.
Input: data/round2/trials_round2.csv, experiments bench2_ssb and bench2_dabstep (one final trial per task, harness
and run).
Usage: python3 -I bench2_report.py   (writes paper/round2/bench2_report.md)

Rules (fixed before looking at outcomes): pass = reward >= 1; a final trial with no reward counts as a fail.
"""
import csv, math, os
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
CSVP = os.path.join(HERE, "..", "..", "data", "round2", "trials_round2.csv")
OUT = os.path.join(HERE, "..", "..", "paper", "round2")
os.makedirs(OUT, exist_ok=True)
HS = ["CC", "mini", "OC"]
HMAP = {"claude-code": "CC", "mini-swe-agent": "mini", "opencode": "OC"}
Z90 = 1.6448536
DS = {"bench2_ssb": "ssb", "bench2_dabstep": "dabstep"}
PAIRS = [("CC", "mini"), ("CC", "OC"), ("mini", "OC")]
RUNS = ["r1", "r2"]

cell = defaultdict(dict)
for r in csv.DictReader(open(CSVP, newline="")):
    if r["experiment"] not in DS:
        continue
    k = (DS[r["experiment"]], "r" + r["run"], HMAP[r["harness"]])
    assert r["task"] not in cell[k], ("duplicate", k, r["task"])
    cell[k][r["task"]] = int(r["passed"])


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def pct(x):
    return "%.1f" % (100 * x)


L = []
P = L.append


def block(ds, title, N):
    tasks = sorted(set.intersection(*[set(cell[(ds, r, h)]) for r in RUNS for h in HS]))
    assert len(tasks) == N and all(len(cell[(ds, r, h)]) == N for r in RUNS for h in HS)
    P(f"\n## {title}, {N} tasks x {len(RUNS)} runs\n")
    P("| harness | " + " | ".join(f"pass {r}" for r in RUNS) + " |")
    P("|---|" + "---|" * len(RUNS))
    for h in HS:
        P(f"| {h} | " + " | ".join(f"{sum(cell[(ds, r, h)][t] for t in tasks)}/{len(tasks)}" for r in RUNS) + " |")
    P("\nPaired harness differences (A minus B, percentage points). Per run: A-only / B-only, paired difference with 90% "
      "Wald interval, exact McNemar p. Pooled: W/L summed over runs; per task the mean over runs, then mean over tasks, "
      "90% t interval.\n")
    P("| A - B | " + " | ".join(f"{r}: diff [90% CI], A-only / B-only, p" for r in RUNS) + " | pooled W / L | pooled diff [90% CI t] |")
    P("|---|" + "---|" * len(RUNS) + "---|---|")
    for a, b in PAIRS:
        out, W_, L_ = [], 0, 0
        n = len(tasks)
        for run in RUNS:
            X, Y = cell[(ds, run, a)], cell[(ds, run, b)]
            bb = sum(1 for t in tasks if X[t] and not Y[t])
            cc = sum(1 for t in tasks if Y[t] and not X[t])
            W_ += bb; L_ += cc
            dlt = (bb - cc) / n; se = math.sqrt(max(bb + cc - (bb - cc) ** 2 / n, 0)) / n
            out.append(f"{100*dlt:+.1f} [{100*(dlt-Z90*se):+.1f}, {100*(dlt+Z90*se):+.1f}], {bb} / {cc}, p={mcnemar(bb, cc):.3f}")
        d = [sum(cell[(ds, r, a)][t] - cell[(ds, r, b)][t] for r in RUNS) / len(RUNS) for t in tasks]
        m = sum(d) / n; se = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) / math.sqrt(n)
        P(f"| {a} - {b} | " + " | ".join(out) + f" | {W_} / {L_} | {100*m:+.1f} [{100*(m-1.6487*se):+.1f}, {100*(m+1.6487*se):+.1f}] |")
    P("\nRerun flips (same harness, run 1 vs run 2):\n")
    P("| harness | pass->fail | fail->pass | flips | flip rate |")
    P("|---|---|---|---|---|")
    for h in HS:
        X, Y = cell[(ds, "r1", h)], cell[(ds, "r2", h)]
        pf = sum(1 for t in tasks if X[t] and not Y[t])
        fp = sum(1 for t in tasks if Y[t] and not X[t])
        P(f"| {h} | {pf} | {fp} | {pf+fp} | {pct((pf + fp) / len(tasks))}% |")
    P("\nHarness-swap flips (share of tasks whose outcome differs when only the harness changes), averaged over the four "
      "run pairings (A r1-B r1, A r2-B r2, A r1-B r2, A r2-B r1).\n")
    P("| A vs B | swap flips per pairing (r1r1, r2r2, r1r2, r2r1) | swap flip rate |")
    P("|---|---|---|")
    for a, b in PAIRS:
        ks = [sum(1 for t in tasks if cell[(ds, ra, a)][t] != cell[(ds, rb, b)][t])
              for ra, rb in (("r1", "r1"), ("r2", "r2"), ("r1", "r2"), ("r2", "r1"))]
        P(f"| {a} vs {b} | {', '.join(map(str, ks))} | {pct(sum(ks) / (4 * len(tasks)))}% |")


block("ssb", "C1: SpreadsheetBench Verified", 400)
block("dabstep", "C2: DABstep hard, fixed random 150", 150)
open(os.path.join(OUT, "bench2_report.md"), "w").write("\n".join(L) + "\n")
