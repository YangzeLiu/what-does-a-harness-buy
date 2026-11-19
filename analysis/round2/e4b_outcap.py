"""E4b: OpenCode with Qwen3.8-27B on the 73 pool447 tasks where an OpenCode trial of the first round had ended on its
output cap; caps of 32k (as shipped) and 131k output tokens, two runs each, 36,000 s of wall clock.
One row per (cap, task, run); a trial that could not be scored was rerun and the rerun takes its place (job name
ending in -r1old / -r1new).  Per task, the cap with more passes over its two runs "wins"; two-sided exact sign test
over the tasks that differ.
Input: data/round2/trials_round2.csv, experiment e4b_outcap.
Run: python3 -I e4b_outcap.py  (writes paper/round2/e4b_outcap.txt)"""
import math
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

P = C.Tee(os.path.join(C.OUT, "e4b_outcap.txt"))
ARM = {"outcap-32k": "OLD", "outcap-131k": "NEW"}
final = {}
nrerun = 0
for r in C.r2rows("e4b_outcap"):
    final.setdefault((ARM[r["arm"]], r["task"]), {})[int(r["run"])] = r
    nrerun += r["job"] != "oc-qwen-e4b-outcap"
tasks = sorted({t for _, t in final})
assert all(sorted(final[(a, t)]) == [1, 2] for a in ARM.values() for t in tasks)
P(f"E4b  OpenCode + Qwen3.8-27B, {len(tasks)} tasks x 2 caps x 2 runs; trials {sum(len(v) for v in final.values())}, "
  f"of which {nrerun} are reruns of a trial that could not be scored")
for a in ("OLD", "NEW"):
    u = [x for t in tasks for x in final[(a, t)].values() if x["usable"] == "1"]
    P(f"  {a} ({'32k' if a == 'OLD' else '131k'} output cap): passed {sum(x['passed'] == '1' for x in u)}/{len(u)} usable "
      f"(trials {2 * len(tasks)})")
win = {"NEW": 0, "OLD": 0, "tie": 0}
for t in tasks:
    k = {a: sum(x["passed"] == "1" for x in final[(a, t)].values() if x["usable"] == "1") for a in ("OLD", "NEW")}
    win["NEW" if k["NEW"] > k["OLD"] else "OLD" if k["OLD"] > k["NEW"] else "tie"] += 1
n = win["NEW"] + win["OLD"]
p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(win["NEW"], win["OLD"]) + 1)) / 2 ** n)
P(f"  per task: NEW passes more often on {win['NEW']}, OLD on {win['OLD']}, ties {win['tie']}; "
  f"two-sided exact sign test p = {p:.4f}")
P.close()
