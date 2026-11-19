"""Claude Opus 5.5 on pool447: the 391 tasks that all three harnesses completed (paper, Appendix "Second round of
runs", Table "pool447" Opus row, Figure "Claude Opus 5.5 on pool447, task by task").

Input: data/round2/trials_round2.csv, experiment swe_opus55_pool447: Claude Code's canonical trial for every
task (all 447 scored), and the usable canonical trials of mini-SWE-agent and OpenCode (the tasks ran in a
pre-drawn random order and these two harnesses did not reach every task).
Paired comparison on the 391 tasks: Newcombe paired 90% interval (paper_numbers.newcombe_paired), exact McNemar
(main_table.mcnemar_exact_p), Holm over the three pairs.
Also writes paper/round2/opus55_pool447_391.json: per-task outcomes on the 391 tasks, sorted by task id (the
input of the figure "Claude Opus 5.5 on pool447, task by task").
Run: python3 -I opus55_pool447.py  (writes paper/round2/opus55_pool447.txt)"""
import json
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def holm(ps):
    idx = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for k, i in enumerate(idx):
        run = max(run, min(1.0, (len(ps) - k) * ps[i]))
        adj[i] = run
    return adj


P = C.Tee(os.path.join(C.OUT, "opus55_pool447.txt"))
rows = C.r2rows("swe_opus55_pool447")
U = {h: {} for h in C.H3}
for r in rows:
    if r["usable"] == "1":
        h = C.H_FROM[r["harness"]]
        assert r["task"] not in U[h]
        U[h][r["task"]] = int((C.fnum(r["reward"]) or 0) >= 1)

P("Claude Opus 5.5, pool447 (canonical usable trials)")
P(f"  Claude Code: {len(U['CC'])} tasks, passed {sum(U['CC'].values())}")
PAIRS = [("CC", "mini"), ("CC", "OC"), ("mini", "OC")]


def paired(x, y, ts_):
    a_ = sum(x[t] and y[t] for t in ts_)
    b_ = sum(x[t] and not y[t] for t in ts_)
    c_ = sum(y[t] and not x[t] for t in ts_)
    d_ = len(ts_) - a_ - b_ - c_
    n = len(ts_)
    lo95, hi95 = C.pn.newcombe_paired(a_, b_, c_, d_)
    lo90, hi90 = C.pn.newcombe_paired(a_, b_, c_, d_, z=1.6448536269514722)
    return dict(n=n, kA=a_ + b_, kB=a_ + c_, b=b_, c=c_, diff=100 * (b_ - c_) / n, ci95=(100 * lo95, 100 * hi95),
                ci90=(100 * lo90, 100 * hi90), p=C.mt.mcnemar_exact_p(b_, c_))


common3 = sorted(set(U["CC"]) & set(U["mini"]) & set(U["OC"]))
P(f"usable: CC {len(U['CC'])}, mini {len(U['mini'])}, OC {len(U['OC'])}; all three {len(common3)}; "
  f"passes on 3-way set: " + ", ".join(f"{h} {sum(U[h][t] for t in common3)}" for h in C.H3))
outc = [[U[h][t] for h in C.H3] for t in common3]
P(f"  on the 3-way set: passed by all three {sum(o == [1, 1, 1] for o in outc)}, by none "
  f"{sum(o == [0, 0, 0] for o in outc)}, outcome differs {sum(len(set(o)) > 1 for o in outc)}")
for label, sets in (("3-way common set", {p: common3 for p in PAIRS}),):
    res_ = {p: paired(U[p[0]], U[p[1]], sets[p]) for p in PAIRS}
    hp = holm([res_[p]["p"] if not math.isnan(res_[p]["p"]) else 1.0 for p in PAIRS])
    P(f"-- {label}")
    for p, h in zip(PAIRS, hp):
        e = res_[p]
        P(f"   {p[0]}-{p[1]}: n={e['n']} {p[0]} {e['kA']} {p[1]} {e['kB']} b={e['b']} c={e['c']} "
          f"diff {e['diff']:+.1f}pp "
          f"90% [{e['ci90'][0]:+.1f}, {e['ci90'][1]:+.1f}] McNemar p={e['p']:.3f} Holm p={h:.3f}")
P.close()
json.dump({"source": "data/round2/trials_round2.csv, experiment swe_opus55_pool447, usable trials, 391 tasks "
                     "completed by all three harnesses, sorted by task id",
           "order": C.H3, "outcomes": outc}, open(os.path.join(C.OUT, "opus55_pool447_391.json"), "w"))
