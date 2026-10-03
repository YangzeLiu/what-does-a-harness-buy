#!/usr/bin/env python3
"""Fig 3: where the bill comes from.  Per-trial mean tokens (top) and CNY (bottom), three
segments: input cache miss / input cache hit / output incl. reasoning; DS Flash vs GLM Flash."""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

N = json.load(open(os.path.join(fs.HERE, "..", "paper", "numbers.json")))
fs.setup()
MODELS = ["ds-v4-flash", "glm-5.3-flash"]
SEG = ["input_miss", "input_hit", "output"]

fig, axes = plt.subplots(2, 2, figsize=(fs.WIDTH["full"], 3.4), sharex="col",
                         gridspec_kw={"hspace": 0.12, "wspace": 0.25, "height_ratios": [1, 1]})
tok_max = max(sum(e[s] for s in SEG) for m in MODELS for e in N["cost"].get(m, {}).values()) / 1e6
cny_max = max(e["cny_per_trial"] for m in MODELS for e in N["cost"].get(m, {}).values())

for j, m in enumerate(MODELS):
    C = N["cost"].get(m, {})
    p = N["prices"][m]
    xs = np.arange(len(fs.HARNESS_ORDER))
    for i, (ax, key, ymax, unit) in enumerate(((axes[0, j], "tok", tok_max, "M tokens / trial"),
                                               (axes[1, j], "cny", cny_max, "CNY / trial"))):
        for x, h in zip(xs, fs.HARNESS_ORDER):
            e = C.get(h)
            if e is None:
                ax.text(x, 0.03 * ymax, "running", ha="center", va="bottom", fontsize=6.5, color=fs.MUTED, rotation=90)
                continue
            base = 0.0
            for s in SEG:
                v = e[s] / 1e6 if key == "tok" else e["cny_split"][s]
                ax.bar(x, v, bottom=base, width=0.62, color=fs.TOKEN_COLOR[s], lw=0, zorder=3)
                base += v
            if key == "cny":
                ax.text(x, base + 0.02 * ymax, f"¥{e['cny_per_trial']:.2f}", ha="center", va="bottom", fontsize=7, color=fs.INK, fontweight="bold")
            else:
                ax.text(x, base + 0.02 * ymax, f"{base:.1f}M", ha="center", va="bottom", fontsize=6.5, color=fs.MUTED)
        ax.set_ylim(0, ymax * 1.62)
        ax.set_xlim(-0.6, len(xs) - 0.4)
        fs.despine(ax)
        fs.hgrid(ax)
        if j == 0:
            ax.set_ylabel(unit)
        else:
            ax.tick_params(labelleft=False)
    axes[0, j].set_title(fs.MODEL_LABEL[m], pad=4)
    axes[1, j].set_xticks(xs)
    axes[1, j].set_xticklabels(["Claude\nCode", "mini-\nSWE-agent", "OpenCode"], fontsize=7)
    # price list in the whitespace of the CNY panel
    fs.note(axes[1, j], f"{p['vendor']} list, CNY / M tokens: hit {p['hit']} · miss {p['miss']} · output {p['out']}\n"
            f"a cache hit costs {p['hit']/p['miss']:.2g}× a miss",
            xy=(0.02, 0.97), size=6.0)
    # steps / cache-hit share in the token panel
    C2 = N["cost"].get(m, {})
    if C2:
        hitpct = " / ".join(f"{100*C2[h]['cache_hit_frac']:.0f}%" for h in fs.HARNESS_ORDER if h in C2)
        stp = " / ".join(f"{C2[h]['steps']:.0f}" for h in fs.HARNESS_ORDER if h in C2)
        ips = " / ".join(f"{(C2[h]['input_miss'] + C2[h]['input_hit']) / C2[h]['steps'] / 1e3:.0f}k" for h in fs.HARNESS_ORDER if h in C2)
        fs.note(axes[0, j], f"cache-hit share of input: {hitpct}\nsteps per trial: {stp}\ninput tokens per step: {ips}",
                xy=(0.02, 0.97), size=6.0)

from matplotlib.patches import Patch
fig.legend([Patch(color=fs.TOKEN_COLOR[s]) for s in SEG], [fs.TOKEN_LABEL[s] for s in SEG],
           loc="lower center", bbox_to_anchor=(0.5, -0.04), ncol=3, fontsize=6.8, handlelength=1.1, frameon=False)
fs.save(fig, "fig3_cost")
print("ok")
