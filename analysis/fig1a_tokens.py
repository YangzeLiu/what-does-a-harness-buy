#!/usr/bin/env python3
"""Fig 1a: input tokens per task against pass rate, hard45, every model x harness (canonical run 1).
Reads data/trials.csv through paper_numbers' canonical rule.  Output paper/figures/fig1a_tokens.pdf/.png"""
import csv, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import main_table as mt
import paper_numbers as pn

fs.setup()
rows = list(csv.DictReader(open(os.path.join(fs.HERE, "..", "data", "trials.csv"))))
cells = pn.canonical_by_seed(rows)

pts = {}   # (model, harness) -> (mean input Mtok, rate, lo, hi, k, n, n_tok)
for m in fs.MODEL_ORDER:
    for h in fs.HARNESS_ORDER:
        lab, cell = pn.pick_arm(cells, m, "hard45", h)
        if cell is None:
            continue
        d = mt.usable(cell[mt.MAIN_SEED])
        k, n = mt.passes(d), len(d)
        lo, hi = mt.wilson(k, n)
        toks = [pn.num(r.get("prompt_tokens")) for r in d.values()]
        toks = [t for t in toks if t is not None and t == t]
        pts[(m, h)] = (np.mean(toks) / 1e6, 100 * k / n, 100 * lo, 100 * hi, k, n, len(toks))

MARK = {"claude-code": "o", "mini-swe-agent": "s", "opencode": "D"}
MCOL = {"qwen3.6-35b-a3b": "#8C8C8C", "qwen3.8-27b": "#56B4E9", "ds-v4-flash": "#D55E00",
        "glm-5.3-flash": "#CC79A7", "hy4-preview": "#009E73", "opus5": "#111111"}
fig, ax = plt.subplots(figsize=(2.7, 3.2))
for m in fs.MODEL_ORDER:
    xy = sorted((pts[(m, h)][0], pts[(m, h)][1]) for h in fs.HARNESS_ORDER if (m, h) in pts)
    if len(xy) > 1:
        ax.plot(*zip(*xy), color=MCOL[m], lw=0.9, alpha=0.6, zorder=1)
for (m, h), (x, y, lo, hi, k, n, nt) in pts.items():
    ax.scatter(x, y, marker=MARK[h], s=26, color=MCOL[m], edgecolor="white", lw=0.5, zorder=3)
# the paired gap 45 tasks catch half the time (exact McNemar, 50% power), as a scale bar
import json
mde = json.load(open(os.path.join(fs.HERE, "..", "paper", "numbers.json")))["mde"]["hard45"]["mde_pp_exact50"]
xb, yb = 0.55, 3
ax.plot([xb, xb], [yb, yb + mde], color=fs.INK, lw=1.0, zorder=3)
for yy in (yb, yb + mde):
    ax.plot([xb - 0.18, xb + 0.18], [yy, yy], color=fs.INK, lw=1.0, zorder=3)
ax.text(xb + 0.35, yb + mde / 2, f"{mde:.0f} pp: the gap 45 tasks\ncatch half the time", ha="left", va="center", fontsize=6.3, color=fs.INK)
ax.set_xlabel("input tokens per task (millions)")
ax.set_ylabel("pass rate on the 45 hard tasks (%)")
ax.set_xlim(0, 12.5); ax.set_ylim(0, 100)
ax.set_xticks([0, 3, 6, 9, 12])
fs.despine(ax); fs.hgrid(ax)
from matplotlib.lines import Line2D
l1 = ax.legend([Line2D([], [], marker=MARK[h], color="#555555", ls="none", ms=4.5) for h in fs.HARNESS_ORDER],
               [fs.HARNESS_LONG[h] for h in fs.HARNESS_ORDER], loc="upper right", fontsize=6.6, frameon=False,
               handletextpad=0.3, borderaxespad=0.2, title="harness", title_fontsize=6.6)
ax.add_artist(l1)
ax.legend([Line2D([], [], marker="o", color=MCOL[m], ls="-", lw=0.9, ms=4) for m in fs.MODEL_ORDER],
          [fs.MODEL_SHORT[m] for m in fs.MODEL_ORDER], loc="lower right", fontsize=6.6,
          frameon=False, handletextpad=0.3, borderaxespad=0.2, title="model", title_fontsize=6.6)
fs.save(fig, "fig1a_tokens")
for (m, h), v in pts.items():
    print(f"{fs.MODEL_SHORT[m]:8s} {fs.HARNESS_SHORT[h]:5s} tok={v[0]:5.2f}M pass={v[4]}/{v[5]} ({v[1]:.1f}%) ntok={v[6]}")
