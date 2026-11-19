#!/usr/bin/env python3
"""Revised Figure 2: flip rate under a rerun, a harness swap and a model swap, with the runs added in
the second round.  Same definition as the first version (analysis/paper_numbers.py, flip rates, reproduced by
analysis/round2/t7_flips.py):
  rerun   : every pair of runs of one (model, harness); flips / tasks usable in both
  harness : run 1, every harness pair within a model
  model   : run 1, every model pair within a harness
hard45 : Qwen3.6, Qwen3.8, DeepSeek-V4-Flash, GLM-5.3-Flash, HY4-Preview, DeepSeek-V4.1-Flash,
         Claude Opus 5.5 (all three harnesses) and Claude Opus 5 (Claude Code only, reruns and model swaps).
pool447: Qwen3.6, Qwen3.8, DeepSeek-V4.1-Flash (complete runs only).
Read-only on inputs.  Prints every plotted group so the text can be checked against it.
"""
import os
import statistics as st
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

sys.path.insert(0, os.path.join(C.ROOT, "analysis"))
import figstyle as fs  # noqa: E402

fs.FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "round2", "figures")
fs.setup()

cells = C.load_all()
PAIRS = [("CC", "mini"), ("CC", "OC"), ("mini", "OC")]
MODELS = {"hard45": ["qwen3.6-35b-a3b", "qwen3.8-27b", "ds-v4-flash", "glm-5.3-flash", "hy4-preview",
                     "opus5", "ds-v41-flash", "opus55"],
          "pool447": ["qwen3.6-35b-a3b", "qwen3.8-27b", "ds-v41-flash"]}


def flips(taskset, models):
    pm = {(m, h, r): {t: x["passed"] for t, x in C.usable(d).items()}
          for (m, s, h, r), d in cells.items() if s == taskset and m in models}
    F = {"rerun": [], "harness": [], "model": []}

    def add(g, a, b, tag):
        sh = set(pm[a]) & set(pm[b])
        if len(sh) >= 20:
            k = sum(pm[a][t] != pm[b][t] for t in sh)
            F[g].append({"tag": tag, "flips": k, "n": len(sh), "rate": k / len(sh)})

    for m in models:
        for h in C.H3:
            rs = sorted(k[2] for k in pm if k[0] == m and k[1] == h)
            for i in range(len(rs)):
                for j in range(i + 1, len(rs)):
                    add("rerun", (m, h, rs[i]), (m, h, rs[j]), f"{m} {h} r{rs[i]}v{rs[j]}")
        for h1, h2 in PAIRS:
            if (m, h1, 1) in pm and (m, h2, 1) in pm:
                add("harness", (m, h1, 1), (m, h2, 1), f"{m} {h1}-{h2}")
    for h in C.H3:
        ms = [m for m in models if (m, h, 1) in pm]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                add("model", (ms[i], h, 1), (ms[j], h, 1), f"{h} {ms[i]}|{ms[j]}")
    return F


F = {s: flips(s, MODELS[s]) for s in MODELS}
for s in F:
    for g, v in F[s].items():
        r = [e["rate"] for e in v]
        print(f"{s:8s} {g:8s} pairs {len(v):3d}  median {100*st.median(r):5.1f}%  "
              f"range {100*min(r):.1f}-{100*max(r):.1f}%")
        if s == "pool447" or g != "model":
            for e in v:
                print(f"      {e['tag']:45s} {e['flips']:3d}/{e['n']} = {100*e['rate']:.1f}%")
if "--print-only" in sys.argv:
    sys.exit(0)

MCOL = {"qwen3.6-35b-a3b": "#8C8C8C", "qwen3.8-27b": "#56B4E9", "ds-v41-flash": "#E69F00"}
MNAME = {"qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "qwen3.8-27b": "Qwen3.8-27B", "ds-v41-flash": "DeepSeek-V4.1-Flash"}
fig, a = plt.subplots(figsize=(fs.WIDTH["half"], 2.3))
cats = [("rerun", "rerun"), ("harness", "harness swap"), ("model", "model swap")]
DX = 0.058      # horizontal step between dots that share a value
XC = -0.10      # centre of the hard45 column inside a category
XP = 0.33       # centre of the pool447 column


def stacked(values, step, width):
    """Dot histogram: equal (binned) values sit side by side, centred, so no two dots overlap."""
    out, groups = [], {}
    for v in values:
        groups.setdefault(round(v / width), []).append(v)
    for g in groups.values():
        for j, v in enumerate(sorted(g)):
            out.append(((j - (len(g) - 1) / 2) * step, v))
    return out


for i, (k, lab) in enumerate(cats):
    ys = [100 * e["rate"] for e in F["hard45"][k]]
    med = float(np.median(ys))
    a.plot([i + XC - 0.27, i + XC + 0.27], [med, med], color=fs.INK, lw=0.9, zorder=2)
    for dx, y in stacked(ys, DX, 2.2):
        a.plot(i + XC + dx, y, "o", ms=2.5, mfc="#4A4A4A", mec="white", mew=0.25, zorder=3)
    a.text(i + XC, 1.015, f"median {med:.0f}%", transform=a.get_xaxis_transform(), va="bottom", ha="center",
           fontsize=6.6, color=fs.INK)
    if k != "model":  # a model swap has no single model to colour by
        pp = sorted(F["pool447"][k], key=lambda e: e["rate"])
        vals = [100 * e["rate"] for e in pp]
        for (dx, y), e in zip(sorted(stacked(vals, 0.07, 1.6), key=lambda t: t[1]), pp):
            a.plot(i + XP + dx, 100 * e["rate"], "D", ms=2.7, mfc=MCOL[e["tag"].split()[0]], mec="white",
                   mew=0.25, zorder=3)
a.set_xticks([i + XC for i in range(len(cats))])
a.set_xticklabels([l for _, l in cats])
a.tick_params(axis="x", length=0, pad=3)
a.set_xlim(-0.52, len(cats) - 0.62)
a.set_ylim(0, 70)
a.set_yticks([0, 10, 20, 30, 40, 50, 60, 70])
a.set_ylabel("tasks whose outcome flips (%)")
fs.despine(a)
fs.hgrid(a)
hs = [Line2D([], [], marker="o", ls="none", mfc="#4A4A4A", mec="white", ms=3.6)] + \
     [Line2D([], [], marker="D", ls="none", mfc=MCOL[m], mec="white", ms=3.8) for m in MCOL]
fig.legend(hs, ["hard45, all models"] + [f"pool447, {MNAME[m]}" for m in MCOL], loc="upper center",
           bbox_to_anchor=(0.55, 0.0), ncol=2, fontsize=6.5, frameon=False, handletextpad=0.2,
           columnspacing=0.9)
fs.save(fig, "fig2_noise_rev")
