"""T3: pool447 paired harness differences, four models (paper Table "pool447").
Models: Qwen3.6-35B-A3B and Qwen3.8-27B (first round, canonical run 1 only), DeepSeek-V4.1-Flash (2 runs per
harness, per-task mean over usable runs), Claude Opus 5.5 (one run; for each pair, the tasks both harnesses
completed).  Per model: paired difference on tasks usable in both cells; pooled = equal-weight mean over the
four models with a task-cluster bootstrap (10,000 resamples, seed 20261110, percentile 90% interval).
Heterogeneity: Cochran's Q with inverse-variance weights from the per-model bootstrap SEs (nominal: the models
share tasks).  The pooled average is printed only for pairs where Q does not reject at 0.05.
For the two Qwen models the per-model interval of the paper (Newcombe paired, 90%) is printed with its counts.
Run: python3 -I t3_pool447_pooled.py  (writes paper/round2/t3_pool447_pooled.txt)"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

P = C.Tee(os.path.join(C.OUT, "t3_pool447_pooled.txt"))
cells = C.load_all()
PAIRS = [("CC", "mini"), ("CC", "OC"), ("mini", "OC")]
f = lambda x: f"{100*x:+.1f}"          # noqa: E731
ci = lambda c: f"[{100*c[0]:+.1f}, {100*c[1]:+.1f}]"  # noqa: E731
OPUS = "Claude Opus 5.5 (tasks both harnesses completed)"

# per-model task means
means = {}
for m in ["qwen3.6-35b-a3b", "qwen3.8-27b"]:
    means[C.LABEL[m]] = {h: C.task_mean(cells, m, "pool447", h, runs=[1]) for h in C.H3}
means[C.LABEL["ds-v41-flash"]] = {h: C.task_mean(cells, "ds-v41-flash", "pool447", h) for h in C.H3}
acc = {}
for r in C.r2rows("swe_opus55_pool447"):
    if r["usable"] == "1":
        acc.setdefault(C.H_FROM[r["harness"]], {}).setdefault(r["task"], []).append(int(r["passed"] == "1"))
means[OPUS] = {h: {t: sum(v) / len(v) for t, v in acc.get(h, {}).items()} for h in C.H3}

P("T3  pool447 paired differences (pp), four models; equal-weight pooled average with task-cluster bootstrap, B=10000 seed=20261110")
P("    Qwen models: canonical run 1 (OpenCode = OpenCode (rg)); DeepSeek-V4.1-Flash: mean over 2 runs")
for A, B in PAIRS:
    diffs = {lab: C.paired_diffs(mh[A], mh[B]) for lab, mh in means.items()}
    r = C.pooled_boot(diffs)
    pr = r["pooled"]
    pm = [(r["models"][lab]["est"], r["models"][lab]["se"]) for lab in diffs]
    Q, df, p, I2, mu = C.cochran_q(pm)
    P()
    P(f"{A}-{B}  do the models disagree? Cochran's Q = {Q:.2f} on {df} df, p = {p:.3f}")
    if p >= 0.05:
        P(f"     POOLED ({len(diffs)} models) {f(pr['est'])}  90% {ci(pr['ci90'])}  (task clusters {r['n_tasks_union']})")
    else:
        P("     POOLED: not averaged (the models disagree)")
    for lab in diffs:
        e = r["models"][lab]
        P(f"     {lab:50s} n={e['n']:3d}  {f(e['est'])}  se {100*e['se']:.1f}")
    for m in ["qwen3.6-35b-a3b", "qwen3.8-27b"]:
        ua = C.usable(cells[(m, "pool447", A, 1)])
        ub = C.usable(cells[(m, "pool447", B, 1)])
        a = b = c = d = 0
        for t in set(ua) & set(ub):
            x, y = ua[t]["passed"], ub[t]["passed"]
            a += x and y
            b += x and not y
            c += y and not x
            d += (not x) and (not y)
        lo90, hi90 = C.pn.newcombe_paired(a, b, c, d, z=1.6448536269514722)
        n = a + b + c + d
        P(f"       {C.LABEL[m]} Newcombe paired: n={n} b={b} c={c} {100*(b-c)/n:+.1f} 90% [{100*lo90:+.1f}, {100*hi90:+.1f}]")
P()
P("Per-arm DeepSeek-V4.1-Flash usable tasks (any run): " + ", ".join(f"{h} {len(means[C.LABEL['ds-v41-flash']][h])}" for h in C.H3))
P.close()
