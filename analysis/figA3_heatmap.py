#!/usr/bin/env python3
"""Appendix A3: per-task outcome heatmap, hard45, every model x harness (canonical run 1).
Cells: pass / fail / void.  Tasks sorted by how many arms solve them."""
import csv, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs
import main_table as mt
from paper_numbers import canonical_by_seed, pick_arm, HARNESSES, is_pass

fs.setup()
rows, _ = mt.load(mt.CSV_PATH)
cells = canonical_by_seed(rows)
cols, mats = [], {}
for m in fs.MODEL_ORDER:
    for h in HARNESSES:
        lab, c = pick_arm(cells, m, "hard45", h)
        if c is None:
            continue
        cols.append((m, h))
        mats[(m, h)] = c[mt.MAIN_SEED]
tasks = sorted({t for d in mats.values() for t in d})
V = np.full((len(tasks), len(cols)), np.nan)
for j, key in enumerate(cols):
    for i, t in enumerate(tasks):
        r = mats[key].get(t)
        if r is None:
            continue
        V[i, j] = 1.0 if (mt.ok(r) and is_pass(r)) else (0.0 if mt.ok(r) else 0.5)
order = np.argsort(-np.nansum(V == 1, axis=1), kind="stable")
V = V[order]; tasks = [tasks[i] for i in order]
from matplotlib.colors import ListedColormap
cmap = ListedColormap([fs.OUTCOME_COLOR["fail_patch"], fs.OUTCOME_COLOR["void"], fs.OUTCOME_COLOR["pass"]])
fig, ax = plt.subplots(figsize=(4.15, 0.115 * len(tasks) + 1.1))
ax.imshow(V, cmap=cmap, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
ax.set_xticks(range(len(cols)))
ax.set_xticklabels([fs.HARNESS_SHORT[h] for _, h in cols], fontsize=6.8)
ax.set_yticks(range(len(tasks)))
ax.set_yticklabels([t.replace("__", " / ") for t in tasks], fontsize=5.6)
# model group brackets on top
j = 0
while j < len(cols):
    m = cols[j][0]; k = j
    while k < len(cols) and cols[k][0] == m:
        k += 1
    ax.text((j + k - 1) / 2, -1.4, fs.MODEL_SHORT[m], ha="center", va="bottom", fontsize=6.6, fontweight="bold")
    ax.plot([j - 0.4, k - 0.6], [-0.85, -0.85], color=fs.INK, lw=0.7, clip_on=False)
    if k < len(cols):
        ax.axvline(k - 0.5, color="white", lw=2.0)
    j = k
ax.set_ylim(len(tasks) - 0.5, -0.5)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
ax.set_xticks(np.arange(-0.5, len(cols), 1), minor=True); ax.set_yticks(np.arange(-0.5, len(tasks), 1), minor=True)
ax.grid(which="minor", color="white", lw=0.5); ax.tick_params(which="minor", length=0)
# legend above the model brackets, left; the summary line sits to its right
leg = ax.legend([Patch(color=cmap(1.0)), Patch(color=cmap(0.0)), Patch(facecolor=cmap(0.5), edgecolor="#BBBBBB", lw=0.4)],
                ["pass", "fail", "infra void"], loc="lower left", bbox_to_anchor=(0.0, 1.045), ncol=3, fontsize=6.5,
                handlelength=1.0, columnspacing=1.0, handletextpad=0.4, frameon=False, borderaxespad=0)
solved_by = (V == 1).sum(axis=1)
ax.text(1.0, 1.06, f"{int((solved_by == 0).sum())} tasks solved by no arm · {int((solved_by == len(cols)).sum())} by every arm",
        transform=ax.transAxes, ha="right", va="bottom", fontsize=6.2, color=fs.MUTED)
fs.save(fig, "figA3_heatmap")
print("ok", V.shape)
