#!/usr/bin/env python3
"""E4a (Qwen3.8-27B x OpenCode (rg) x hard45, wall clock 36,000 s, two runs) against the first-round OpenCode (rg) arm.

Inputs, all in data/:
  round2/trials_round2.csv, experiment e4a_wallclock: one row per (task, run); usable as in the paper rule
  round2/trials_round2.csv, experiment e4a_original_oc: end class of each trial of the first-round OpenCode (rg) job
                      (read from the trajectories with the same reader)
  trials.csv          the first-round table source; per-run canonical rows come from analysis/main_table.py canonical()
Conventions (main_table.py): usable = scored and not an infrastructure exception; pass rate = passes / usable;
exact McNemar on discordant pairs.
Run: python3 -I e4a_summary.py > ../../paper/round2/e4a_summary.txt
"""
import csv, os, sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
R2 = os.path.join(HERE, "..", "..", "data", "round2", "trials_round2.csv")
sys.path.insert(0, os.path.join(HERE, ".."))
import main_table as M  # noqa: E402

R = list(csv.DictReader(open(R2)))
slots = {(r["task"], int(r["run"])): r for r in R if r["experiment"] == "e4a_wallclock"}
O_end = {r["trial"].rsplit("__", 1)[1]: r for r in R if r["experiment"] == "e4a_original_oc"}
tasks = sorted({t for t, _ in slots})


def e_ok(r):
    return r["usable"] == "1"


def e_pass(r):
    return float(r["reward"] or 0) >= 1


T = M.load(M.CSV_PATH)
if isinstance(T, tuple):
    T = T[0]
arm = {}
for sd in (1, 2):
    M.MAIN_SEED = sd
    best, _, _ = M.canonical(T)
    arm[sd] = best[("OC-rg", "qwen3.8-27b", "hard45")]


def p_pass(r):
    return (M.num(r.get("reward")) or 0) >= 1


out = []
P = out.append
P(f"E4a: {len(slots)} trials, {len(tasks)} tasks, two runs")

P("\n== pass rate (usable trials) ==")
for sd in (1, 2, None):
    eu = [r for k, r in slots.items() if (sd is None or k[1] == sd) and e_ok(r)]
    rr = [r for s in ((sd,) if sd else (1, 2)) for r in arm[s].values() if M.ok(r)]
    P(f"run {sd or '1+2'}:  E4a {sum(map(e_pass, eu))}/{len(eu)} | first-round OpenCode {sum(map(p_pass, rr))}/{len(rr)}")
P(f"not scored: E4a {sum(not e_ok(r) for r in slots.values())}, first-round OpenCode "
  f"{sum(not M.ok(r) for s in (1, 2) for r in arm[s].values())}")

P("\n== end class (all trials) ==")
ec = Counter(r["end_class"] for r in slots.values())
oc_rows = {sd: [(r, O_end[r["trial_id"]]) for r in arm[sd].values() if r["trial_id"] in O_end] for sd in (1, 2)}
P(f"E4a                  {dict(ec)}  ended by the wall clock = {ec['timeout']}")
P(f"first-round OpenCode {dict(Counter(o['end_class'] for sd in (1, 2) for _, o in oc_rows[sd]))}  "
  f"(n={sum(len(v) for v in oc_rows.values())})")
for sd in (1, 2):
    to = [r for r, o in oc_rows[sd] if o["end_class"] == "timeout"]
    P(f"first-round OpenCode run {sd}: ended by the wall clock {len(to)}, of which failed {sum(not p_pass(r) for r in to)}")

P("\n== paired, same task and run ==")
for sd in (1, 2):
    common = [t for t in tasks if e_ok(slots[(t, sd)]) and t in arm[sd] and M.ok(arm[sd][t])]
    a = [e_pass(slots[(t, sd)]) for t in common]
    b = [p_pass(arm[sd][t]) for t in common]
    bb = sum(x and not y for x, y in zip(a, b))
    cc = sum(y and not x for x, y in zip(a, b))
    P(f"E4a vs first-round OpenCode run {sd}: n={len(common)}  {sum(a)} vs {sum(b)}  b={bb} c={cc}  exact p={M.mcnemar_exact_p(bb, cc):.4f}")
print("\n".join(out))
