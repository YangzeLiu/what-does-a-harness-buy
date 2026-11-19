#!/usr/bin/env python3
"""Appendix figure (app:equalbudget): pass rate on pool447 for DeepSeek-V4.1-Flash, run 1, when every trial
whose output tokens exceed a common budget is counted as a failure (finished trials truncated after the fact,
not a rerun under a cap).  Check value: at the pooled median budget the paper gives Claude Code 53.5, mini-SWE-agent 33.7, OpenCode 49.2.  Read-only on inputs."""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402
sys.path.insert(0, os.path.join(C.ROOT, "analysis"))
import figstyle as fs  # noqa: E402

fs.FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "round2", "figures")
fs.setup()
cells = C.load_all()
M, S, RUN = "ds-v41-flash", "pool447", 1
data = {h: C.usable(cells[(M, S, h, RUN)]) for h in C.H3}
pooled = np.array([x["ot"] for h in C.H3 for x in data[h].values()])
med = float(np.median(pooled))
print(f"pooled median output tokens = {med:.0f}  (n trials {len(pooled)})")
for h in C.H3:
    u = data[h]
    n = len(u)
    full = 100 * sum(x["passed"] for x in u.values()) / n
    at = 100 * sum(x["passed"] and x["ot"] <= med for x in u.values()) / n
    print(f"  {C.HFULL[h]:15s} n={n} no budget {full:.1f}%  at median budget {at:.1f}%  "
          f"median own output {np.median([x['ot'] for x in u.values()]):.0f}")
if "--print-only" in sys.argv:
    sys.exit(0)

COL = {"CC": "#D55E00", "mini": "#0072B2", "OC": "#009E73"}
LS = {"CC": "-", "mini": "--", "OC": "-."}
xs = np.logspace(np.log10(1500), np.log10(max(pooled) * 1.05), 300)
fig, ax = plt.subplots(figsize=(3.6, 2.4))
for h in C.H3:
    u = list(data[h].values())
    ys = [100 * sum(x["passed"] and x["ot"] <= b for x in u) / len(u) for b in xs]
    ax.plot(xs / 1e3, ys, LS[h], color=COL[h], lw=1.4, label=C.HFULL[h])
ax.axvline(med / 1e3, color=fs.MUTED, lw=0.8, ls=":")
ax.text(med / 1e3 * 1.06, 4, "pooled median", fontsize=6.5, color=fs.MUTED, ha="left")
ax.set_xscale("log")
ax.set_xticks([2, 5, 10, 20, 50, 100])
ax.set_xticklabels(["2k", "5k", "10k", "20k", "50k", "100k"])
ax.set_xlabel("budget: output tokens per trial")
ax.set_ylabel("pass rate if trials over\nbudget count as failures (%)")
ax.set_ylim(0, 100)
fs.despine(ax)
fs.hgrid(ax)
ax.legend(loc="upper left", fontsize=6.8, frameon=False)
fs.save(fig, "figA_budget_rev")
