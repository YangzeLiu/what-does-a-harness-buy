#!/usr/bin/env python3
"""Revised Figure 1 (two panels on one canvas, saved as fig1_rev.pdf).

(a) input tokens per task against pass rate on hard45, every model that ran in
    all three harnesses (seven) ; each point is the mean over that configuration's runs.  No connecting
    lines, legends outside the plot.
(b) paired pass-rate difference for each pair of harnesses on hard45, one dot per
    model plus the equal-weight average over the seven models with its 95% interval.

Read-only on every input: uses analysis/round2/common.py loaders (paper rule for usable trials).
Also prints the plotted values so they can be checked against t1/t2 outputs.
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.ROOT, "analysis"))
import figstyle as fs  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "round2", "figures")
fs.FIG_DIR = OUTDIR
fs.setup()

MODELS = C.ALL7
MNAME = {"qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "qwen3.8-27b": "Qwen3.8-27B",
         "ds-v4-flash": "DeepSeek-V4-Flash", "glm-5.3-flash": "GLM-5.3-Flash",
         "hy4-preview": "HY4-Preview", "ds-v41-flash": "DeepSeek-V4.1-Flash",
         "opus55": "Claude Opus 5.5"}
MCOL = {"qwen3.6-35b-a3b": "#8C8C8C", "qwen3.8-27b": "#56B4E9", "ds-v4-flash": "#D55E00",
        "glm-5.3-flash": "#CC79A7", "hy4-preview": "#009E73", "ds-v41-flash": "#E69F00",
        "opus55": "#111111"}
HMARK = {"CC": "o", "mini": "s", "OC": "D"}

cells = C.load_all()

# ------------------------------------------------------------------ panel (a)
pts = {}
for m in MODELS:
    for h in C.H3:
        runs = C.runs_of(cells, m, "hard45", h)
        if not runs:
            continue
        k = n = 0
        toks = []
        for r in runs:
            u = C.usable(cells[(m, "hard45", h, r)])
            k += sum(x["passed"] for x in u.values())
            n += len(u)
            # total input tokens of a trial = uncached prompt + cache read + cache write when the
            # loader keeps them apart; paper rows carry the total in pt already (cr is a subset).
            toks += [x["pt"] for x in u.values() if x["pt"]]
        pts[(m, h)] = (np.mean(toks) / 1e6 if toks else float("nan"), 100.0 * k / n, k, n, len(runs), len(toks))

print("panel (a): model harness runs  mean input Mtok  pass%  (k/n)  n_tok")
for (m, h), v in pts.items():
    print(f"  {MNAME[m]:22s} {h:5s} runs={v[4]}  {v[0]:6.2f}M  {v[1]:5.1f}%  ({v[2]}/{v[3]})  ntok={v[5]}")

if "--print-only" in sys.argv:
    sys.exit(0)

# ------------------------------------------------------------------ panel (b) data
PAIRS = [("CC", "mini"), ("CC", "OC"), ("mini", "OC")]
print("\npanel (b): pair  pooled diff [95% CI]  ; per-model diffs")
rows = []
for a, b in PAIRS:
    diffs = {}
    for m in MODELS:
        ma = C.task_mean(cells, m, "hard45", a)
        mb = C.task_mean(cells, m, "hard45", b)
        if ma and mb:
            diffs[m] = C.paired_diffs(ma, mb)
    res = C.pooled_boot(diffs)
    rows.append((a, b, res, diffs))

# ------------------------------------------------------------------ one figure, two panels
# One canvas at text width, so both panels share font size and baseline.
FW, FH = fs.WIDTH["full"], 2.62
fig = plt.figure(figsize=(FW, FH))
axa = fig.add_axes([0.070, 0.335, 0.365, 0.605])
axb = fig.add_axes([0.640, 0.335, 0.345, 0.605])
LAB = dict(fontsize=7.5)

for (m, h), v in pts.items():
    axa.scatter(v[0], v[1], marker=HMARK[h], s=22, color=MCOL[m], edgecolor="white", lw=0.5, zorder=3,
                clip_on=False)
axa.set_xlabel("input tokens per task (millions)", **LAB)
axa.set_ylabel("pass rate on hard45 (%)", **LAB)
axa.set_ylim(0, 100)
axa.set_xlim(0, 12)
axa.set_xticks([0, 3, 6, 9, 12])
axa.set_yticks([0, 25, 50, 75, 100])
axa.tick_params(labelsize=7)
fs.despine(axa)
fs.hgrid(axa)
axa.text(-0.19, 1.045, "(a)", transform=axa.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")

# (b) one row per pair: the seven models on a line, the average and its interval just below.
axb.axvline(0, color=fs.ZERO, lw=0.7, zorder=2)
for i, (a, b, res, diffs) in enumerate(rows):
    pooled = 100 * res["pooled"]["est"]
    lo, hi = (100 * x for x in res["pooled"]["ci95"])
    per = {m: 100 * v["est"] for m, v in res["models"].items()}
    axb.axhspan(i - 0.40, i + 0.40, color="#F6F6F6", lw=0, zorder=0)
    # all models on one line; two models with the same difference share one two-coloured point
    groups = {}
    for m in per:
        groups.setdefault(round(per[m], 1), []).append(m)
    for x, ms in sorted(groups.items()):
        assert len(ms) <= 2, ms
        if len(ms) == 1:
            axb.plot(x, i - 0.13, marker="o", ms=4.0, ls="none", mfc=MCOL[ms[0]], mec="white", mew=0.4, zorder=3)
        else:
            axb.plot(x, i - 0.13, marker="o", ms=4.6, ls="none", fillstyle="left", mfc=MCOL[ms[0]],
                     markerfacecoloralt=MCOL[ms[1]], mec="white", mew=0.4, zorder=3)
    axb.plot([lo, hi], [i + 0.21, i + 0.21], color=fs.INK, lw=1.3, solid_capstyle="butt", zorder=4)
    axb.scatter(pooled, i + 0.21, marker="D", s=20, color=fs.INK, edgecolor="white", lw=0.5, zorder=5)
    print(f"  {a}-{b}: pooled {pooled:+.1f} [{lo:+.1f}, {hi:+.1f}]  " +
          "  ".join(f"{MNAME[m]}:{per[m]:+.1f}" for m in MODELS if m in per))
axb.set_ylim(2.48, -0.48)
axb.set_yticks([0, 1, 2])
axb.set_yticklabels([f"{C.HFULL[a]}\n$-$ {C.HFULL[b]}" for a, b, _, _ in rows], fontsize=7, linespacing=1.15)
axb.tick_params(axis="y", length=0, pad=4)
axb.set_xlim(-17, 17)
axb.set_xticks([-15, -10, -5, 0, 5, 10, 15])
axb.tick_params(axis="x", labelsize=7)
axb.set_xlabel("difference in pass rate on hard45 (points)", **LAB)
fs.despine(axb, left=False)
axb.text(-0.9, -0.56, "second harness higher", ha="right", va="bottom", fontsize=6.3, color=fs.MUTED)
axb.text(0.9, -0.56, "first harness higher", ha="left", va="bottom", fontsize=6.3, color=fs.MUTED)
axb.text(-0.60, 1.045, "(b)", transform=axb.transAxes, fontsize=8, fontweight="bold", ha="left", va="bottom")

# legends: one row of what the marks are, then the model colours shared by both panels
LG = dict(fontsize=6.6, frameon=False, handletextpad=0.25, columnspacing=1.1, borderaxespad=0)
h1 = [Line2D([], [], marker=HMARK[h], color="#666666", ls="none", ms=4.2) for h in C.H3]
fig.legend(h1, [C.HFULL[h] for h in C.H3], loc="upper center", bbox_to_anchor=(0.2525, 0.175), ncol=3, **LG)
hb = [Line2D([], [], marker="o", color="#888888", ls="none", ms=3.8),
      Line2D([], [], marker="D", color=fs.INK, ls="-", lw=1.3, ms=4.0)]
fig.legend(hb, ["one model", "average of the seven, 95% interval"], loc="upper center",
           bbox_to_anchor=(0.8125, 0.175), ncol=2, **LG)
h2 = [Line2D([], [], marker="o", color=MCOL[m], ls="none", ms=4.2) for m in MODELS]
fig.legend(h2, [MNAME[m] for m in MODELS], loc="upper center", bbox_to_anchor=(0.5, 0.098), ncol=7,
           fontsize=6.6, frameon=False, handletextpad=0.1, columnspacing=0.75, borderaxespad=0)
fs.save(fig, "fig1_rev")
