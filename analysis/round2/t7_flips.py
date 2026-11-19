"""T7: hard45 flip rates, paper definition (analysis/paper_numbers.py, flip rates, Figure 2):
  passmap[(model, harness, seed)] = {task: pass} over usable canonical trials (OpenCode = OC-rg arm)
  rerun   : every pair of seeds of the same (model, harness); flips / tasks usable in both (needs >= 20)
  harness : seed 1 (MAIN_SEED), every harness pair within a model
  model   : seed 1, every model pair within a harness
  rate = flips / n.
Printed: the rerun flips of every hard45 cell with two or more runs (first-round models with GLM run 2, HY4 run 2
and DeepSeek-V4.1-Flash), and the largest change in pass count between two runs of the same cell.
Run: python3 -I t7_flips.py  (writes paper/round2/t7_flips.txt)"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

P = C.Tee(os.path.join(C.OUT, "t7_flips.txt"))
cells = C.load_all()
PAPER_M = ["qwen3.6-35b-a3b", "qwen3.8-27b", "ds-v4-flash", "glm-5.3-flash", "hy4-preview", "opus5"]
PAIRS = [("CC", "mini"), ("CC", "OC"), ("mini", "OC")]


def passmap(models, runs_ok):
    pm = {}
    for (m, s, h, r), d in cells.items():
        if s == "hard45" and m in models and runs_ok(m, r):
            pm[(m, h, r)] = {t: x["passed"] for t, x in C.usable(d).items()}
    return pm


def flips(pm, models):
    F = {"rerun": [], "harness": [], "model": []}
    for m in models:
        for h in C.H3:
            seeds = sorted(k[2] for k in pm if k[0] == m and k[1] == h)
            for i in range(len(seeds)):
                for j in range(i + 1, len(seeds)):
                    A, B = pm[(m, h, seeds[i])], pm[(m, h, seeds[j])]
                    sh = set(A) & set(B)
                    if len(sh) >= 20:
                        F["rerun"].append({"model": m, "harness": h, "seeds": (seeds[i], seeds[j]),
                                           "flips": sum(A[t] != B[t] for t in sh), "n": len(sh)})
        for h1, h2 in PAIRS:
            a, b = (m, h1, 1), (m, h2, 1)
            if a in pm and b in pm:
                sh = set(pm[a]) & set(pm[b])
                if len(sh) >= 20:
                    F["harness"].append({"model": m, "pair": f"{h1}-{h2}",
                                         "flips": sum(pm[a][t] != pm[b][t] for t in sh), "n": len(sh)})
    for h in C.H3:
        ms = [m for m in models if (m, h, 1) in pm]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = (ms[i], h, 1), (ms[j], h, 1)
                sh = set(pm[a]) & set(pm[b])
                if len(sh) >= 20:
                    F["model"].append({"harness": h, "models": (ms[i], ms[j]),
                                       "flips": sum(pm[a][t] != pm[b][t] for t in sh), "n": len(sh)})
    for g in F.values():
        for e in g:
            e["rate"] = e["flips"] / e["n"]
    return F



AFTER_M = PAPER_M + ["ds-v41-flash"]
pm1 = passmap(set(AFTER_M), lambda m, r: True)
F1 = flips(pm1, AFTER_M)
P("T7  hard45 flip rates (paper definition, analysis/paper_numbers.py)")
P()
P("Per-cell rerun flips (tasks usable in both runs), every cell with >= 2 runs")
for e in F1["rerun"]:
    P(f"  {C.LABEL[e['model']]:36s} {C.HFULL[e['harness']]:15s} runs {e['seeds'][0]}v{e['seeds'][1]}: "
      f"{e['flips']:2d}/{e['n']} = {100*e['rate']:4.1f}%")
P()
P("Largest change in pass count between two runs of the same cell (hard45)")
worst = []
for m in AFTER_M:
    for h in C.H3:
        rs = C.runs_of(cells, m, "hard45", h)
        if len(rs) < 2:
            continue
        kn = {r: C.kn(cells[(m, "hard45", h, r)]) for r in rs}
        best = max(((abs(kn[x][0] - kn[y][0]), x, y) for i, x in enumerate(rs) for y in rs[i + 1:]))
        d, x, y = best
        dpp = max(abs(100 * kn[p][0] / kn[p][1] - 100 * kn[q][0] / kn[q][1]) for i, p in enumerate(rs) for q in rs[i + 1:])
        worst.append(d)
        P(f"  {C.LABEL[m]:36s} {C.HFULL[h]:15s} " + "  ".join(f"r{r} {kn[r][0]}/{kn[r][1]}" for r in rs)
          + f"   max |delta passes| = {d}  (max |delta rate| = {dpp:.1f} pts)")
P(f"  overall max |delta passes| = {max(worst)}  (paper: 'rerunning one cell moves it by up to three')")
P.close()
