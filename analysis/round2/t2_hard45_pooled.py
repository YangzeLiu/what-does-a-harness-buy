"""T2: hard45 pooled paired harness differences.
Per (model, task): mean reward over that cell's usable runs; paired diff A-B per (model, task) on
tasks usable in both cells; pooled = equal-weight mean over models; task-cluster bootstrap
(10,000 resamples, seed 20261110, percentile 95% and 90%), all seven models.
Heterogeneity: Cochran's Q with inverse-variance weights from the per-model bootstrap SEs
(nominal: the models share the same 45 tasks, so the per-model estimates are not independent).
Run: python3 -I t2_hard45_pooled.py  (writes paper/round2/t2_hard45_pooled.txt)"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

P = C.Tee(os.path.join(C.OUT, "t2_hard45_pooled.txt"))
cells = C.load_all()
PAIRS = [("CC", "mini"), ("CC", "OC"), ("mini", "OC")]
f = lambda x: f"{100*x:+.1f}"          # noqa: E731
ci = lambda c: f"[{100*c[0]:+.1f}, {100*c[1]:+.1f}]"  # noqa: E731

P("T2  hard45 pooled paired differences (percentage points), models weighted equally, task-cluster bootstrap B=10000 seed=20261110")
for setname, models in [("all seven models", C.ALL7)]:
    P()
    P("=" * 110)
    P(setname + ": " + ", ".join(C.LABEL[m] for m in models))
    P("=" * 110)
    for A, B in PAIRS:
        diffs = {}
        for m in models:
            diffs[m] = C.paired_diffs(C.task_mean(cells, m, "hard45", A), C.task_mean(cells, m, "hard45", B))
        r = C.pooled_boot(diffs)
        pr = r["pooled"]
        P(f"{A}-{B}  POOLED {f(pr['est'])}  95% {ci(pr['ci95'])}  90% {ci(pr['ci90'])}   "
          f"(model-task pairs {sum(len(d) for d in diffs.values())}, task clusters {r['n_tasks_union']})")
        pm = []
        for m in models:
            e = r["models"][m]
            runsA = len(C.runs_of(cells, m, "hard45", A))
            runsB = len(C.runs_of(cells, m, "hard45", B))
            P(f"     {C.LABEL[m]:30s} n={e['n']:2d} runs {runsA}v{runsB}  {f(e['est'])}  95% {ci(e['ci95'])}  90% {ci(e['ci90'])}")
            pm.append((e["est"], e["se"]))
        Q, df, p, I2, mu = C.cochran_q(pm)
        lo = min(e for e, _ in pm)
        hi = max(e for e, _ in pm)
        P(f"     heterogeneity: Cochran's Q = {Q:.2f} on {df} df, p = {p:.3f}; per-model range {f(lo)} .. {f(hi)}")
P()
P("Note: Q uses bootstrap SEs and treats models as independent; they share tasks, so p is nominal.")
P.close()
