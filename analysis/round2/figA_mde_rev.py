#!/usr/bin/env python3
"""Appendix: minimum detectable paired difference vs n at the observed discordance (was Fig 2b).

Main line = the exact two-sided McNemar test we actually report (unconditional power over
M ~ Binomial(n, discordance)); the asymptotic inversion is kept as a thin dashed reference.
The across-pairs band is drawn exactly as well (two more curves, ~1 s each).
"""
import json, os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import figstyle as fs
from paper_numbers import mde_pp, mde_pp_exact

N = json.load(open(os.path.join(fs.HERE, "..", "paper", "numbers.json")))
fs.FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "paper", "round2", "figures")
fs.setup()
fig, b = plt.subplots(figsize=(fs.WIDTH["half"], 2.25))

# ---- MDE curve
ns = [int(round(x)) for x in np.logspace(np.log10(20), np.log10(1000), 46)]
d_h = N["mde"]["hard45"]
d_p = N["mde"]["pool447"]
disc = d_h["disc_pooled"]
lo_d, hi_d = min(d_h["disc_min"], d_p["disc_min"]), max(d_h["disc_max"], d_p["disc_max"])
ex_lo = [mde_pp_exact(n, lo_d) for n in ns]
ex_hi = [mde_pp_exact(n, hi_d) for n in ns]
ex = [mde_pp_exact(n, disc) for n in ns]
b.fill_between(ns, ex_lo, ex_hi, color="#E6E6E6", lw=0, zorder=1)
b.plot(ns, [mde_pp(n, disc) for n in ns], color=fs.MUTED, lw=0.8, ls=(0, (3, 2)), zorder=2)
b.plot(ns, ex, color=fs.INK, lw=1.3, zorder=3)
# n = 45: no 80%-power effect size exists at all, so mark the ceiling instead of a point
b.plot([45, 45], [0, 25], color=fs.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
b.text(47, 24.2, f"hard45 (45 tasks): no difference is\ndetected with 80% power (at most {100*d_h['power_max_at_n']:.0f}% for any size);\n"
                 f"a {d_h['mde_pp_exact50']:.0f}-point difference is detected half the time", fontsize=6.0, ha="left", va="top", color=fs.INK)
n0, dd = 447, d_p["mde_pp_exact80"]
b.plot([n0, n0], [0, dd], color=fs.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
b.plot([20, n0], [dd, dd], color=fs.MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
b.plot(n0, dd, "o", ms=4, mfc="white", mec=fs.INK, mew=0.9, zorder=4)
b.text(n0 * 1.12, dd + 1.3, f"pool447 ({n0} tasks):\nabout {dd:.1f} points", fontsize=6.5, ha="left", va="bottom", color=fs.INK)
b.set_xscale("log")
b.set_xlim(20, 1000)
b.set_xticks([20, 45, 100, 447, 1000])
b.set_xticklabels(["20", "45", "100", "447", "1000"])
b.set_ylim(0, 25)
# the exact curve stops existing below n_min: there the discordant set is too small for the
# conditional binomial test to reach α even when every discordance runs one way (M < 6 at .05).
n_line = d_h["n_min_for_80"]
b.axvline(n_line, color=fs.MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
b.text(21, 4.7, f"below about {n_line} tasks no difference reaches\n80% power: fewer than 6 tasks that disagree\ncannot give a significant test",
       fontsize=5.6, color=fs.MUTED, va="top")
b.set_xlabel("number of tasks")
b.set_ylabel("smallest detectable difference (points)")
fs.despine(b)
fs.hgrid(b)
fs.save(fig, "figA_mde_rev")
print("ok")
