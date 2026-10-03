#!/usr/bin/env python3
"""Fig 1b: paired harness effects (pp) with Newcombe 95% CI, every model x subset rung.
Reads paper/numbers.json.  Output paper/figures/fig1b_forest.pdf/.png"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

N = json.load(open(os.path.join(fs.HERE, "..", "paper", "numbers.json")))
fs.setup()

GROUPS = [(m, s) for m in fs.MODEL_ORDER for s in ("pool447", "hard45")]
PAIR_ORDER = ["CC−mini", "CC−OC", "mini−OC"]
PAIR_MARK = {"CC−mini": "o", "CC−OC": "s", "mini−OC": "D"}
SIG = "#B5485D"

rows = []           # (y, entry)
labels = []         # (y, text, kind)
y = 0
for m, s in GROUPS:
    ents = {e["pair"]: e for e in N["pairs"] if e["model"] == m and e["subset"] == s}
    if not ents:
        continue
    labels.append((y, f"{fs.MODEL_SHORT[m]} · {s}", "group"))
    y += 1
    for p in PAIR_ORDER:
        if p in ents:
            rows.append((y, ents[p]))
            labels.append((y, p.replace("−", " − "), "pair"))
            y += 1
    y += 0.5   # group gap

H = 0.115 * y + 0.55
fig, ax = plt.subplots(figsize=(fs.WIDTH["twothird"], H))
ax.axvspan(-5, 5, color="#F1F1F1", zorder=0, lw=0)
ax.axvline(0, color=fs.ZERO, lw=1.1, zorder=2)
for yy, e in rows:
    sig = e["p_mcnemar"] < 0.05
    col = SIG if sig else fs.INK
    lo, hi = e["ci95_pp"]
    ax.plot([lo, hi], [yy, yy], color=col, lw=0.9, solid_capstyle="butt", zorder=3)
    for c in e.get("seed_combos", []):
        ax.plot([c["diff_pp"]] * 2, [yy - 0.22, yy + 0.22], color=fs.MUTED, lw=0.7, zorder=3)
    ax.plot(e["diff_pp"], yy, marker=PAIR_MARK[e["pair"]], ms=4.2, mfc=col if sig else "white",
            mec=col, mew=0.9, zorder=4)
    if sig:
        ax.text(hi + 1.2, yy, ("p=" + f"{e['p_mcnemar']:.2f}".lstrip("0")) if e["p_mcnemar"] >= 0.001 else "p<.001",
                va="center", ha="left", fontsize=6.3, color=SIG)
for yy, t, kind in labels:
    if kind == "group":
        ax.text(-0.02, yy, t, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=7, fontweight="bold", color=fs.INK)
    else:
        ax.text(-0.02, yy, t, transform=ax.get_yaxis_transform(), ha="right", va="center",
                fontsize=6.8, color=fs.INK)
ax.set_ylim(y - 0.5, -1.0)
ax.set_yticks([])
ax.set_xlim(-30, 30)
ax.set_xticks([-30, -20, -10, -5, 0, 5, 10, 20, 30])
ax.set_xticklabels(["−30", "−20", "−10", "", "0", "", "10", "20", "30"])
ax.set_xlabel("difference in pass rate, first − second harness (pp)")
fs.despine(ax, left=False)
ax.tick_params(axis="y", length=0)
fig.text(0.5, -0.01, "band ±5 pp · line 95% CI (Newcombe, paired) · ticks other run pairings · filled McNemar p<.05",
         ha="center", va="top", fontsize=6.2, color=fs.MUTED)
ax.text(-5.5, -0.55, "← second harness better", ha="right", va="center", fontsize=6.2, color=fs.MUTED)
ax.text(5.5, -0.55, "first better →", ha="left", va="center", fontsize=6.2, color=fs.MUTED)
fs.save(fig, "fig1b_forest")
print("ok", len(rows), "rows")
