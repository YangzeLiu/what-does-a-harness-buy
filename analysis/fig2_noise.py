#!/usr/bin/env python3
"""Figure 2: per-task flip rate, rerun vs harness swap vs model swap (single panel).
The MDE-vs-n curve moved to the appendix: analysis/figA_mde.py."""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

N = json.load(open(os.path.join(fs.HERE, "..", "paper", "numbers.json")))
fs.setup()
fig, a = plt.subplots(figsize=(fs.WIDTH["half"], 2.25))

# ---- (a) strip plot
cats = [("rerun", "same harness,\nrerun"), ("harness", "swap\nharness"), ("model", "swap\nmodel")]
rng = np.random.default_rng(3)
for i, (k, lab) in enumerate(cats):
    for s, mk, fill in (("hard45", "o", fs.INK), ("pool447", "D", "white")):
        pts = N["flips"].get(s, {}).get(k, [])
        if not pts:
            continue
        xs = i + rng.uniform(-0.18, 0.18, len(pts))
        ys = [100 * e["rate"] for e in pts]
        a.plot(xs, ys, mk, ms=3.6 if mk == "o" else 3.2, mfc=fill, mec=fs.INK, mew=0.7, ls="none", zorder=3, alpha=0.9)
    allr = [100 * e["rate"] for s in ("hard45", "pool447") for e in N["flips"].get(s, {}).get(k, [])]
    if allr:
        med = float(np.median(allr))
        a.plot([i - 0.3, i + 0.3], [med, med], color=fs.MUTED, lw=1.2, zorder=2)
        a.text(i + 0.33, med, f"{med:.0f}%", va="center", ha="left", fontsize=6.5, color=fs.MUTED)
a.set_xticks(range(len(cats)))
a.set_xticklabels([l for _, l in cats])
a.set_xlim(-0.55, len(cats) - 0.3)
a.set_ylim(0, None)
a.set_ylabel("tasks whose outcome flips (%)")
fs.despine(a)
fs.hgrid(a)
from matplotlib.lines import Line2D
a.set_ylim(0, 62)
a.legend([Line2D([], [], marker="o", ls="none", mfc=fs.INK, mec=fs.INK, ms=3.6),
          Line2D([], [], marker="D", ls="none", mfc="white", mec=fs.INK, ms=3.2)],
         ["hard45 (n=45)", "pool447 (n≈445)"], loc="upper left", fontsize=6.5, handletextpad=0.3)

fs.save(fig, "fig2_noise")
print("ok")
