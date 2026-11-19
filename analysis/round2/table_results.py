"""Every number of the paper's main results table (tables/rev_results.tex), in one place.  New script for this
release; the table itself was assembled by hand from t1_hard45_table.txt, ds_pool447_tables.md,
opus55_pool447.txt and bench2_report.md, which this script cross-checks.
  hard45  : tasks passed per run (canonical usable trials; n shown when it is not 45)
  pool447 : pass rate per run (%), first-round Qwen models and DeepSeek-V4.1-Flash; Claude Opus 5.5 on the 391
            tasks that all three harnesses completed
  SpreadsheetBench Verified / DABstep: pass rate per run (%), DeepSeek-V4.1-Flash
Percentages are rounded half up, as in the paper.
Run: python3 -I table_results.py  (writes paper/round2/table_results.txt)"""
import os
import sys
from decimal import Decimal, ROUND_HALF_UP
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

def pct1(k, n):
    """percentage with one decimal, halves rounded up (353/400 = 88.25 -> 88.3), as printed in the paper"""
    return str((Decimal(100 * k) / Decimal(n)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


P = C.Tee(os.path.join(C.OUT, "table_results.txt"))
cells = C.load_all()
ROWS = [("Qwen3.6-35B-A3B", "qwen3.6-35b-a3b"), ("Qwen3.8-27B", "qwen3.8-27b"), ("DeepSeek-V4-Flash", "ds-v4-flash"),
        ("GLM-5.3-Flash", "glm-5.3-flash"), ("HY4-Preview", "hy4-preview"), ("DeepSeek-V4.1-Flash", "ds-v41-flash"),
        ("Claude Opus 5.5", "opus55")]
P(f"{'':30s} " + "  ".join(f"{C.HFULL[h]:22s}" for h in C.H3))
P("hard45: tasks passed out of 45 (per run)")
for name, m in ROWS + [("Claude Opus 5", "opus5")]:
    cols = []
    for h in C.H3:
        rs = C.runs_of(cells, m, "hard45", h)
        per = [C.kn(cells[(m, "hard45", h, r)]) for r in rs]
        cols.append(", ".join(f"{k}" + ("" if n == 45 else f"/{n}") for k, n in per) or "n/a")
    P(f"  {name:28s} " + "  ".join(f"{c:22s}" for c in cols))
P("pool447: pass rate (%) per run")
for name, m in ROWS[:2] + ROWS[5:6]:
    cols = []
    for h in C.H3:
        per = [C.kn(cells[(m, "pool447", h, r)]) for r in C.runs_of(cells, m, "pool447", h)]
        cols.append(", ".join(pct1(k, n) for k, n in per))
    P(f"  {name:28s} " + "  ".join(f"{c:22s}" for c in cols))
U = {h: {} for h in C.H3}
for r in C.r2rows("swe_opus55_pool447"):
    if r["usable"] == "1":
        U[C.H_FROM[r["harness"]]][r["task"]] = int(r["passed"])
c3 = set(U["CC"]) & set(U["mini"]) & set(U["OC"])
P(f"  {'Claude Opus 5.5 (%d tasks)' % len(c3):28s} " + "  ".join(
    f"{pct1(sum(U[h][t] for t in c3), len(c3)):22s}" for h in C.H3)
  + f"   (Claude Code on all {len(U['CC'])}: passed {sum(U['CC'].values())})")
P("Two other benchmarks, DeepSeek-V4.1-Flash: pass rate (%) per run")
for name, ex in [("SpreadsheetBench Verified", "bench2_ssb"), ("DABstep", "bench2_dabstep")]:
    rows = C.r2rows(ex)
    cols = []
    for h in C.H3:
        per = []
        for run in ("1", "2"):
            rs = [r for r in rows if C.H_FROM[r["harness"]] == h and r["run"] == run]
            per.append(pct1(sum(r['passed'] == '1' for r in rs), len(rs)))
        cols.append(", ".join(per))
    P(f"  {name:28s} " + "  ".join(f"{c:22s}" for c in cols))
P.close()
