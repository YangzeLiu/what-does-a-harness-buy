#!/usr/bin/env python3
"""Does a harness effect replicate at the task level?
For every model with two runs of both harnesses of a pair (hard45), take d_r(t) = pass_A - pass_B in run r.
If the harness had a task-specific effect, tasks that split the pair in run 1 would split the same way in run 2.
Under no task-specific effect the two signs are independent, so among tasks split in both runs
P(same direction) = 1/2.  Reports per pair and pooled, with an exact binomial test against 1/2."""
import csv, os, sys, math, json, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_table as mt
import paper_numbers as pn
from itertools import combinations

rows = list(csv.DictReader(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "trials.csv"))))
cells = pn.canonical_by_seed(rows)
H = pn.HARNESSES
SHORT = {"claude-code": "CC", "mini-swe-agent": "mini", "opencode": "OC"}

def run(cell, sd):
    d = mt.usable(cell.get(sd, {}))
    return {t: int(pn.is_pass(r)) for t, r in d.items()}

def binom_two_sided(k, n):
    p = 0.5
    pk = lambda i: math.comb(n, i) * p ** n
    obs = pk(k)
    return min(1.0, sum(pk(i) for i in range(n + 1) if pk(i) <= obs + 1e-12))

out = []
ev_h, ev_m = [], []
tot_same = tot_opp = 0
for m in pn.MODELS:
    for a, b in combinations(H, 2):
        la, ca = pn.pick_arm(cells, m, "hard45", a)
        lb, cb = pn.pick_arm(cells, m, "hard45", b)
        if ca is None or cb is None or 2 not in ca or 2 not in cb:
            continue
        r = {}
        for sd in (1, 2):
            pa, pb = run(ca, sd), run(cb, sd)
            r[sd] = {t: pa[t] - pb[t] for t in pa if t in pb}
        tasks = [t for t in r[1] if t in r[2]]
        split1 = [t for t in tasks if r[1][t] != 0]
        split2 = [t for t in tasks if r[2][t] != 0]
        both = [t for t in split1 if r[2][t] != 0]
        same = sum(1 for t in both if r[1][t] == r[2][t])
        opp = len(both) - same
        back0 = sum(1 for t in split1 if r[2][t] == 0)
        gap1 = sum(r[1].values()); gap2 = sum(r[2].values())
        tot_same += same; tot_opp += opp
        for t in split1:
            ev_h.append(dict(task=t, model=m, pair=f"{SHORT[a]}-{SHORT[b]}",
                             e=0 if r[2][t] == 0 else (1 if r[1][t] == r[2][t] else -1)))
        out.append(dict(model=m, pair=f"{SHORT[a]}-{SHORT[b]}", n=len(tasks), split1=len(split1), split2=len(split2),
                        gap1=gap1, gap2=gap2, both=len(both), same=same, opp=opp, back0=back0))
        print(f"{m[:8]:8s} {SHORT[a]}-{SHORT[b]:5s} n={len(tasks)} gap run1={gap1:+d} run2={gap2:+d}  split run1={len(split1)} run2={len(split2)}  "
              f"split in both={len(both)} same={same} opp={opp}  run1-split tasks tied in run2={back0}")
n = tot_same + tot_opp
p = binom_two_sided(tot_same, n)
print(f"pooled: split in both runs {n}, same direction {tot_same}, opposite {tot_opp}, binomial p vs 1/2 = {p:.2f}")
tot_split1 = sum(o["split1"] for o in out)
print(f"tasks that split a pair in run 1: {tot_split1}; still split the same way in run 2: {tot_same} ({100*tot_same/tot_split1:.0f}%)")

# positive control: the same statistic for a model swap (Qwen3.8 vs Qwen3.6, same harness), a real effect
pc = []
ps_same = ps_opp = ps_split1 = 0
for h in H:
    l6, c6 = pn.pick_arm(cells, "qwen3.6-35b-a3b", "hard45", h)
    l8, c8 = pn.pick_arm(cells, "qwen3.8-27b", "hard45", h)
    if c6 is None or c8 is None or 2 not in c6 or 2 not in c8:
        continue
    r = {}
    for sd in (1, 2):
        p6, p8 = run(c6, sd), run(c8, sd)
        r[sd] = {t: p8[t] - p6[t] for t in p8 if t in p6}
    tasks = [t for t in r[1] if t in r[2]]
    split1 = [t for t in tasks if r[1][t] != 0]
    both = [t for t in split1 if r[2][t] != 0]
    same = sum(1 for t in both if r[1][t] == r[2][t]); opp = len(both) - same
    ps_same += same; ps_opp += opp; ps_split1 += len(split1)
    for t in split1:
        ev_m.append(dict(task=t, model="swap", pair=SHORT[h],
                         e=0 if r[2][t] == 0 else (1 if r[1][t] == r[2][t] else -1)))
    pc.append(dict(harness=SHORT[h], n=len(tasks), split1=len(split1), both=len(both), same=same, opp=opp,
                   gap1=sum(r[1].values()), gap2=sum(r[2].values())))
    print(f"model swap under {SHORT[h]:5s} n={len(tasks)} gap run1={sum(r[1].values()):+d} run2={sum(r[2].values()):+d} split run1={len(split1)} both={len(both)} same={same} opp={opp}")
pn_ = binom_two_sided(ps_same, ps_same + ps_opp)
print(f"model-swap pooled: split in both {ps_same+ps_opp}, same {ps_same}, opp {ps_opp}, p={pn_:.3f}; run1-split {ps_split1}, same in run2 {ps_same} ({100*ps_same/ps_split1:.0f}%)")


# ---------------------------------------------------------------- clustered permutation test
# Events that share a task (and a harness arm) are not independent: the 41 harness events come
# from far fewer distinct tasks.  Cluster = task; under the null the run-2 direction of a task is
# exchangeable, so flip the sign of every event of a task together (one coin per task).
NPERM, SEED = 20000, 20260919


def clustered(events, name):
    same = sum(1 for e in events if e["e"] == 1)
    opp = sum(1 for e in events if e["e"] == -1)
    tied = sum(1 for e in events if e["e"] == 0)
    by_task = {}
    for e in events:
        by_task.setdefault(e["task"], []).append(e)
    tasks = sorted(by_task)
    units = {(e["model"], e["task"]) for e in events}
    tsum = [sum(e["e"] for e in by_task[t]) for t in tasks]
    obs = sum(tsum)
    rnd = random.Random(SEED)
    ge = 0
    for _ in range(NPERM):
        st = sum(v if rnd.random() < 0.5 else -v for v in tsum)
        if abs(st) >= abs(obs) - 1e-9:
            ge += 1
    p = (ge + 1) / (NPERM + 1)
    print(f"\n[{name}] events={len(events)} same={same} opp={opp} tied={tied} "
          f"unique tasks={len(tasks)} unique (model,task) units={len(units)}")
    print(f"[{name}] observed same-minus-opposite = {obs:+d}; clustered permutation "
          f"({NPERM} flips, seed {SEED}) two-sided p = {p:.4f}")
    print(f"[{name}] per-task breakdown (task: same/opp/tied, cluster sum):")
    for t, v in sorted(zip(tasks, tsum), key=lambda kv: (-len(by_task[kv[0]]), kv[0])):
        ee = [e["e"] for e in by_task[t]]
        who = ",".join(sorted(f"{e['model'][:8]}:{e['pair']}" for e in by_task[t]))
        print(f"    {t:32s} n={len(ee):2d} same={ee.count(1)} opp={ee.count(-1)} "
              f"tied={ee.count(0)} sum={v:+d}  [{who}]")
    return dict(events=len(events), same=same, opp=opp, tied=tied, tasks=len(tasks),
                units=len(units), obs=obs, p=p)


ch = clustered(ev_h, "harness pairs")
cm = clustered(ev_m, "model swap")


def fp(x):
    return f"{x:.3f}".lstrip("0") if x >= .001 else "<.001"


macros = {
 "rep.harness.split1": tot_split1, "rep.harness.same": tot_same, "rep.harness.opp": tot_opp,
 "rep.harness.both": n, "rep.harness.samepct": round(100 * tot_same / tot_split1),
 "rep.harness.p": f"{p:.2f}".lstrip("0"),
 "rep.model.split1": ps_split1, "rep.model.same": ps_same, "rep.model.opp": ps_opp,
 "rep.model.both": ps_same + ps_opp, "rep.model.samepct": round(100 * ps_same / ps_split1),
 "rep.model.p": (f"{pn_:.3f}".lstrip("0") if pn_ >= .001 else "<.001"),
 "rep.harness.tasks": ch["tasks"], "rep.harness.units": ch["units"], "rep.harness.pclust": fp(ch["p"]),
 "rep.model.tasks": cm["tasks"], "rep.model.units": cm["units"], "rep.model.pclust": fp(cm["p"]),
}
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "latex", "numbers_rep.tex"), "w") as f:
    f.write("% generated by analysis/replicability.py\n")
    for k, v in sorted(macros.items()):
        f.write(f"\\expandafter\\def\\csname pn@{k}\\endcsname{{{v}}}\n")
print("macros:", len(macros))

json.dump(dict(pairs=out, control=pc, pooled=dict(both=n, same=tot_same, opp=tot_opp, p=p, split1=tot_split1),
               clustered=dict(harness=ch, model=cm, nperm=NPERM, seed=SEED),
               events=dict(harness=ev_h, model=ev_m)),
          open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "paper", "replicability.json"), "w"), indent=1)
