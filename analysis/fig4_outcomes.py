#!/usr/bin/env python3
"""Fig 4: how trials end.  One stacked bar per model x harness, hard45 and pool447 panels."""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

N = json.load(open(os.path.join(fs.HERE, "..", "paper", "numbers.json")))
fs.setup()
CATS = ["pass", "fail", "cutoff", "overflow", "cap", "void"]
COL = {"pass": fs.OUTCOME_COLOR["pass"], "fail": fs.OUTCOME_COLOR["fail_patch"],
       "cutoff": fs.OUTCOME_COLOR["early_stop"], "overflow": fs.OUTCOME_COLOR["overflow"],
       "cap": fs.OUTCOME_COLOR["step_cap"], "void": fs.OUTCOME_COLOR["void"]}
LAB = {"pass": "pass", "fail": "fail (model stopped)", "cutoff": "output-token cutoff (OC)",
       "overflow": "context overflow", "cap": "step / time cap, crash", "void": "infra void"}

panels = []
for s in ("hard45", "pool447"):
    O = N["outcomes"].get(s, {})
    bars = []
    for m in fs.MODEL_ORDER:
        for h in fs.HARNESS_ORDER + fs.EXTRA_HARNESSES:
            if m in O and h in O[m]:
                bars.append((m, h, O[m][h]))
    if bars:
        panels.append((s, bars))

# pool447 bars carry three-digit labels: give that panel ~1.7x the per-bar width
wr = [(len(b) + 0.8 * (len({m for m, _, _ in b}) - 1)) * (1.25 if s == "pool447" else 1.0) for s, b in panels]
fig, axes = plt.subplots(1, len(panels), figsize=(6.6, 2.35), gridspec_kw={"width_ratios": wr, "wspace": 0.12})  # 6.4 in, scaled to \textwidth (5.5 in) in LaTeX; fonts sized for that 86% scale
axes = np.atleast_1d(axes)
for ax, (s, bars) in zip(axes, panels):
    x = 0.0
    xs, xlab, gx = [], [], {}
    prev_m = None
    for m, h, cnt in bars:
        if prev_m is not None and m != prev_m:
            x += 0.8
        tot = sum(cnt.values())
        base = 0.0
        for c in CATS:
            v = 100 * cnt.get(c, 0) / tot
            if v:
                ax.bar(x, v, bottom=base, width=(0.84 if s == "pool447" else 0.78), color=COL[c], lw=0.3, edgecolor="#999999" if c == "void" else "white", hatch="////" if c == "void" else None, zorder=3)
                if c == "pass":
                    ax.text(x, v / 2, f"{cnt.get(c,0)}", ha="center", va="center", fontsize=7.2, color="white", fontweight="bold")
                base += v
        xs.append(x); xlab.append(fs.HARNESS_SHORT[h]); gx.setdefault(m, []).append(x)
        prev_m = m; x += 1.0
    ax.set_xticks(xs); ax.set_xticklabels(xlab, fontsize=6.4, rotation=90)
    for m, xx in gx.items():
        ax.annotate(fs.MODEL_SHORT[m], xy=(np.mean(xx), 0), xytext=(0, -24), textcoords="offset points", ha="center", va="top", fontsize=7.8, color=fs.INK,
                    annotation_clip=False)
    ax.set_ylim(0, 100); ax.set_xlim(-0.6, x - 0.4)
    ax.set_title(f"{s}  (n = {'45' if s=='hard45' else '447'} tasks per bar)", pad=3)
    fs.despine(ax); fs.hgrid(ax)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("share of tasks (%)")
for ax in axes[1:]:
    ax.tick_params(labelleft=False)
from matplotlib.patches import Patch
fig.legend([Patch(facecolor=COL[c], edgecolor="#999999" if c == "void" else "none", hatch="////" if c == "void" else None, lw=0.4) for c in CATS],
           [LAB[c] for c in CATS], loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=6, fontsize=6.5, handlelength=1.0, frameon=False, columnspacing=1.2)
fs.save(fig, "fig4_outcomes")
print("ok")
