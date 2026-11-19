"""T4: cost per solved task and mean cost per trial, per (model, harness, run-set).
Cost basis = the paper_numbers method: tokens of each usable canonical
trial priced at the vendor list price in paper_numbers.PRICES (CNY/1M: cache-miss input, cache-hit
input, output).  DeepSeek V4.1-Flash is priced at the DeepSeek-V4-Flash list price (the API name and the
price list did not change).  For Claude Opus 5.5 the cost is the per-trial cost in USD recorded at run time at
Anthropic's list prices (column cost_usd), because the token counts of two harnesses (mini-SWE-agent, OpenCode) do
not separate cache writes.  Each model stays in its own currency.  Locally served Qwen: not costed.
Cost per solved = sum(cost of usable canonical trials) / number solved; 95% interval = task bootstrap
(B=10000, seed 20261110; all runs of a resampled task move together).
Run: python3 -I t4_cost.py  (writes paper/round2/t4_cost.txt)"""
import os
import random
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

P = C.Tee(os.path.join(C.OUT, "t4_cost.txt"))
cells = C.load_all()
PR = C.pn.PRICES


def list_cny(r, p):
    return ((r["pt"] - r["cr"]) * p["miss"] + r["cr"] * p["hit"] + r["ot"] * p["out"]) / 1e6


def cost_of(m, r):
    if m == "opus55":
        return r["cost_usd"]
    p = PR["ds-v4-flash" if m == "ds-v41-flash" else m]
    return list_cny(r, p)


def stats(m, s, h, runs):
    by_task = {}
    for rn in runs:
        for t, r in C.usable(cells[(m, s, h, rn)]).items():
            c = cost_of(m, r)
            by_task.setdefault(t, []).append((c, r["passed"]))
    tasks = sorted(by_task)
    tot = sum(c for t in tasks for c, _ in by_task[t])
    k = sum(p for t in tasks for _, p in by_task[t])
    n = sum(len(by_task[t]) for t in tasks)
    rng = random.Random(20261110)
    bs = []
    for _ in range(10000):
        cs = ks = 0
        for _i in range(len(tasks)):
            for c, p in by_task[tasks[rng.randrange(len(tasks))]]:
                cs += c
                ks += p
        if ks:
            bs.append(cs / ks)
    bs.sort()
    return {"n": n, "k": k, "per_trial": tot / n, "per_solved": tot / k, "ci": (C.pct(bs, .025), C.pct(bs, .975)), "total": tot}


GROUPS = [  # (model, subset, list of run-sets to report, currency)
    ("ds-v4-flash", "hard45", None, "CNY"),
    ("glm-5.3-flash", "hard45", None, "CNY"),
    ("hy4-preview", "hard45", None, "CNY"),
    ("ds-v41-flash", "hard45", None, "CNY"),
    ("ds-v41-flash", "pool447", None, "CNY"),
    ("opus55", "hard45", None, "USD"),
]
P("T4  cost per solved task and mean cost per trial (usable canonical trials), list price from tokens;")
P("    Claude Opus 5.5: per-trial USD cost recorded at run time at Anthropic's list prices (cost_usd)")
P("    [ ] = 95% task-bootstrap interval of cost per solved")
ratios = []
for m, s, _, cur in GROUPS:
    allruns = sorted({k[3] for k in cells if k[0] == m and k[1] == s})
    runsets = [[r] for r in allruns]
    if len(allruns) > 1:
        runsets.append(allruns)
    P()
    P(f"--- {C.LABEL[m]} / {s}  ({cur})")
    for rs in runsets:
        tag = "runs " + "+".join(map(str, rs))
        res = {}
        for h in C.H3:
            if not all((m, s, h, r) in cells for r in rs):
                continue
            st = stats(m, s, h, rs)
            res[h] = st
            P(f"  {tag:10s} {C.HFULL[h]:15s} n={st['n']:4d} solved={st['k']:4d}  per trial {st['per_trial']:.3f}  "
              f"per solved {st['per_solved']:.3f} [{st['ci'][0]:.3f}, {st['ci'][1]:.3f}]")
        if len(res) == 3:
            pt = {h: res[h]["per_trial"] for h in res}
            ps = {h: res[h]["per_solved"] for h in res}
            hi_t, lo_t = max(pt, key=pt.get), min(pt, key=pt.get)
            hi_s, lo_s = max(ps, key=ps.get), min(ps, key=ps.get)
            line = (f"  {tag:10s} ratio max/min  per trial {pt[hi_t]/pt[lo_t]:.2f}x ({hi_t}/{lo_t})   "
                    f"per solved {ps[hi_s]/ps[lo_s]:.2f}x ({hi_s}/{lo_s})")
            P(line)
            ratios.append(f"{C.LABEL[m]} {s} {tag}: per trial {pt[hi_t]/pt[lo_t]:.2f}x ({hi_t}/{lo_t}), per solved {ps[hi_s]/ps[lo_s]:.2f}x ({hi_s}/{lo_s})")
P()
P("Summary of ratios (most expensive / cheapest harness)")
for x in ratios:
    P("  " + x)
P.close()
