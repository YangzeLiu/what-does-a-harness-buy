"""T1: hard45, every model x harness x run, canonical passed/usable; sums over runs;
cross-check of run 1 of the first-round models against the first-round Table 1; best harness per model.
Run: python3 -I t1_hard45_table.py   (writes paper/round2/t1_hard45_table.txt)"""
import os
import re
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

P = C.Tee(os.path.join(C.OUT, "t1_hard45_table.txt"))
cells = C.load_all()
MODELS = C.ALL7

# paper Table 1 (first run) parsed from the tex
tex = open(os.path.join(C.REF, "tables/table1_hard45.tex")).read()
TEXNAME = {"Qwen3.6-35B-A3B": "qwen3.6-35b-a3b", "Qwen3.8-27B": "qwen3.8-27b",
           "DeepSeek-V4-Flash": "ds-v4-flash", "GLM-5.3-Flash": "glm-5.3-flash", "HY4-Preview": "hy4-preview"}
paper_t1 = {}
for line in tex.splitlines():
    name = line.split("&")[0].strip()
    if name in TEXNAME:
        fr = re.findall(r"(\d+)/(\d+)", line)
        for h, (k, n) in zip(C.H3, fr):
            paper_t1[(TEXNAME[name], h)] = (int(k), int(n))

P("T1  hard45: passed/usable per run (canonical, paper rule); OpenCode = OpenCode (rg) arm for paper models")
P("source: data/trials.csv via paper_numbers.canonical_by_seed + pick_arm (paper models, run1 GLM/HY4);")
P("        data/round2/trials_round2.csv: swe_glm53flash_run2 (GLM run 2); swe_hy4preview_run2 (HY4 run 2);")
P("        swe_dsv41flash_hard45 (DS V4.1); swe_opus55_hard45 (Opus 5.5)")
P()
P(f"{'model':30s} {'harness':15s} {'runs':40s} {'sum':>10s} {'rate':>6s}")
summ = {}
for m in MODELS:
    for h in C.H3:
        rs = C.runs_of(cells, m, "hard45", h)
        per = [C.kn(cells[(m, "hard45", h, r)]) for r in rs]
        K = sum(k for k, _ in per)
        N = sum(n for _, n in per)
        summ[(m, h)] = (K, N)
        runs = "  ".join(f"r{r}:{k}/{n}" for r, (k, n) in zip(rs, per))
        P(f"{C.LABEL[m]:30s} {C.HFULL[h]:15s} {runs:40s} {f'{K}/{N}':>10s} {100*K/N:6.1f}")
    P()

P("Cross-check of run 1 against the first-round Table 1 -- only mismatches are listed below the OK count")
okc = 0
bad = []
for (m, h), ref in list(paper_t1.items()):
    got = C.kn(cells[(m, "hard45", h, 1)])
    if got == ref:
        okc += 1
    else:
        bad.append(f"  table1_hard45.tex {C.LABEL[m]} {h}: tex {ref[0]}/{ref[1]} vs recomputed {got[0]}/{got[1]}")
P(f"  {okc} cells match; {len(bad)} mismatches")
for b in bad:
    P(b)
P()

P("Best harness (pass rate summed over runs)")
for m in MODELS:
    rates = {h: 100 * summ[(m, h)][0] / summ[(m, h)][1] for h in C.H3}
    top = max(rates.values())
    best = [C.HFULL[h] for h in C.H3 if abs(rates[h] - top) < 1e-9]
    P(f"  {C.LABEL[m]:30s} best: {' = '.join(best):35s} "
      + "  ".join(f"{h} {rates[h]:5.1f}" for h in C.H3))
P.close()
