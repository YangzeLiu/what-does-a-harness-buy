#!/usr/bin/env python3
"""Paper main tables (no-web harness swap) from data/trials.csv.

Scope: models opus5 / ds-v4-flash / qwen3.8-27b / qwen3.6-35b-a3b,
harnesses mini-swe-agent / opencode / claude-code, subsets pool447 (main) and hard45.
Outside the main tables: fable5 / fable51, ds-v4-pro, the 'dsh' harness, every
web-enabled arm and every ablation arm (nothink, midsys, c2-*, ctx*, cc-v1.0.100-*).
The appendix scripts report dsh and some of the ablation arms.  The OpenCode ripgrep pre-placement arm (control-nw-rg) is the
canonical OpenCode column where it exists; the un-fixed control-nw arm is reported next
to it so the ripgrep confound stays visible.

Conventions (shared with analysis/agg.py and analysis/paper_numbers.py):
  * exclusion      analysis/extract_trials.py 'exclude' flag (crashed/smoke batches).
  * usable trial   ok(r) = scored and exception_type not in INFRA, the 14-type set shared
                   with analysis/agg.py.
                   Infra failures are rerun in -r jobs and never count as a task outcome;
                   every other scored trial counts, budget exhaustion included = fail.
  * canonical row  one row per (arm_key, task): usable rows beat unusable, ties go to the
                   latest finished_at (the agg.py rule); see canonical() for the one k=2
                   exception.
  * seed           the job-level 'repeat', except that the Qwen3.8 hard45 jobs pack k=2
                   attempts of every task into one repeat=1 job; there the extractor's within-job
                   'attempt' index is the seed (same split as agg.best_by_seed, done here with
                   the 'attempt' column instead of re-deriving it from started_at).  The main
                   table is seed 1 (MAIN_SEED); extra seeds are listed in the notes.
  * pass rate      passes / usable tasks, with a Wilson 95% interval (wilson()).
  * McNemar        two-sided exact binomial on the discordant pairs (mcnemar_exact_p());
                   scipy is used when available purely to cross-check the same quantity.

Run: python3 analysis/main_table.py
Read-only; stdlib plus an optional scipy cross-check.
"""
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

try:                                     # optional, cross-check only
    from scipy import stats as _st
    HAVE_SCIPY = True
except Exception:                        # noqa: BLE001
    HAVE_SCIPY = False

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "trials.csv")

# keep identical to analysis/agg.py INFRA (14 types)
INFRA = {"NetworkConnectionError", "AgentSetupTimeoutError", "RuntimeError", "CancelledError",
         "OpenCodeStartupHang", "OpenCodeStartupError", "AgentVerifierOverlap",
         "VerifierTimeoutError", "verifier_incomplete", "VerifierUvMissing",
         "RewardFileNotFoundError", "ApiQuotaOrAuthError", "AgentIncomplete", "cc_api_error"}

MAIN_SEED = 1
MODEL_ORDER = ["opus5", "hy4-preview", "glm-5.3-flash", "ds-v4-flash", "qwen3.8-27b", "qwen3.6-35b-a3b"]
MODEL_LABEL = {"opus5": "Opus 5", "hy4-preview": "HY4-Preview",
               "glm-5.3-flash": "GLM-5.3-Flash", "ds-v4-flash": "DS-V4-Flash",
               "qwen3.8-27b": "Qwen3.8-27B", "qwen3.6-35b-a3b": "Qwen3.6-35B"}
SUBSET_ORDER = ["pool447", "hard45"]
# column label -> (harness, arm).  OC-rg first: it is the canonical OpenCode arm.
COLS = [("OC-rg", "opencode", "control-nw-rg"),
        ("OC", "opencode", "control-nw"),
        ("CC", "claude-code", "control-nw"),
        ("mini", "mini-swe-agent", "control-nw")]
COL_LABELS = [c[0] for c in COLS]
CANON_PREF = [("OC-rg", "OC"), ("CC",), ("mini",)]   # one column per harness for the paired block
DASH = "—"


# ----------------------------------------------------------------------------- stats
def wilson(k, n):
    """Wilson 95% interval."""
    if not n:
        return (float("nan"), float("nan"))
    z = 1.959963984540054
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def mcnemar_exact_p(b, c):
    """Two-sided exact binomial test on the discordant pairs.

    scipy, when present, only cross-checks the pure-python value; the reported number is
    the same closed form either way, so the table does not depend on scipy being installed.
    """
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    if HAVE_SCIPY:
        try:
            q = float(_st.binomtest(k, n, 0.5).pvalue) if hasattr(_st, "binomtest") \
                else float(_st.binom_test(k, n, 0.5))
            if not math.isnan(q):
                p = min(1.0, q)
        except Exception:                # noqa: BLE001
            pass
    return p


def median(vals):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return statistics.median(vals) if vals else None


# ----------------------------------------------------------------------------- data
def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load(path):
    if not os.path.exists(path):
        print(f"trials.csv not found at {path}", file=sys.stderr)
        return [], []
    with open(path, newline="") as fh:
        rd = csv.DictReader(fh)
        return list(rd), list(rd.fieldnames or [])


def ok(r):
    return bool(r) and r.get("scored") == "1" and (r.get("exception_type") or "") not in INFRA


def _fin(r):
    return r.get("finished_at") or ""


def canonical(rows):
    """-> best[(col_label, model, subset)] = {task: canonical seed-MAIN_SEED row}, plus meta.

    Seeds.  Normally the job-level 'repeat' is the seed.  The Qwen3.8 hard45 runs instead pack
    k=2 attempts of every task into a single repeat=1 job (qwen-hard45-nw, cc-/mini-qwen-hard45-nw,
    oc-qwen-hard45-nw-rg); there the extractor's within-job 'attempt' index is the seed, the same
    split analysis/agg.py makes by re-sorting on started_at.

    Repairs.  Retry / rerun jobs (-r, -r2, the AgentVerifierOverlap rerun) carry attempt=1 in
    their own job and belong to no slot.  Inside a k=2 arm they are used only where the base
    attempt of that seed is unusable, so one repair can refill both seeds and can never displace
    a usable base attempt.  Outside a k=2 arm every row of the arm competes directly and the
    canonical pick is the agg.py rule: usable beats unusable, ties go to the
    latest finished_at.
    """
    col_of = {(h, a): lab for lab, h, a in COLS}
    buckets = defaultdict(lambda: defaultdict(list))   # (col, model, subset, repeat) -> task -> rows
    max_att = defaultdict(int)                         # (bucket, job) -> max attempt index
    for r in rows:
        if r.get("exclude") == "1":
            continue
        lab = col_of.get((r.get("harness"), r.get("arm")))
        if lab is None or r.get("model") not in MODEL_ORDER or r.get("subset") not in SUBSET_ORDER:
            continue
        bk = (lab, r["model"], r["subset"], int(num(r.get("repeat")) or 1))
        buckets[bk][r["task"]].append(r)
        max_att[(bk, r["job"])] = max(max_att[(bk, r["job"])], int(num(r.get("attempt")) or 1))

    best = defaultdict(dict)
    seeds_seen = defaultdict(set)
    k2_jobs = defaultdict(set)
    for bk, tasks in buckets.items():
        lab, model, subset, rp = bk
        k2 = {job for (b, job), mx in max_att.items() if b == bk and mx >= 2}
        cell = (lab, model, subset)
        if k2:
            k2_jobs[cell] |= k2
        for task, lst in tasks.items():
            base = {}
            for r in lst:
                if r["job"] in k2:
                    base[int(num(r.get("attempt")) or 1)] = r
            if base:
                repairs = [r for r in lst if r["job"] not in k2 and ok(r)]
                repair = max(repairs, key=_fin) if repairs else None
                for sd, br in sorted(base.items()):
                    row = br if (ok(br) or repair is None) else repair
                    seeds_seen[cell].add(sd)
                    if sd == MAIN_SEED:
                        best[cell][task] = row
                if MAIN_SEED not in base and repair is not None:
                    best[cell][task] = repair       # task only ever run by a repair job
            else:
                row = max(lst, key=lambda r: (ok(r), _fin(r)))
                seeds_seen[cell].add(rp)
                if rp == MAIN_SEED:
                    best[cell][task] = row
    return best, seeds_seen, dict(k2_jobs)


def usable(d):
    return {t: r for t, r in d.items() if ok(r)}


def passes(d):
    return sum(1 for r in d.values() if (num(r.get("reward")) or 0) >= 1)


# ----------------------------------------------------------------------------- rendering
def fmt_cell(d, width=26):
    """'66.6 (297/446) [62.1,70.9]' or an em dash when the arm has no usable trial."""
    if not d:
        return DASH.center(width)
    u = usable(d)
    n = len(u)
    if n == 0:
        return f"{DASH} (0/{len(d)})".center(width)
    k = passes(u)
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f} ({k}/{n}) [{100 * lo:.1f},{100 * hi:.1f}]".center(width)


def rule(char, widths):
    return "+".join(char * w for w in widths)


def hsep(widths):
    return "+" + rule("-", widths) + "+"


def row(cells, widths):
    return "|" + "|".join(c.center(w) if len(c) < w else c[:w] for c, w in zip(cells, widths)) + "|"


def section(title):
    print()
    print("=" * 100)
    print(title)
    print("=" * 100)


# ----------------------------------------------------------------------------- tables
def main_table(best):
    section(f"TABLE 1 — MAIN: pass rate % (passes/usable tasks) [Wilson 95% CI], no-web, seed {MAIN_SEED}")
    w0, wc = 24, 28
    widths = [w0] + [wc] * len(COLS)
    print(hsep(widths))
    print(row(["model / subset"] + COL_LABELS, widths))
    print(hsep(widths))
    for m in MODEL_ORDER:
        any_row = False
        for s in SUBSET_ORDER:
            cells = [best.get((lab, m, s), {}) for lab in COL_LABELS]
            if not any(cells):
                continue
            any_row = True
            print(row([f"{MODEL_LABEL[m]} / {s}"] + [fmt_cell(d, wc) for d in cells], widths))
        if any_row:
            print(hsep(widths))
    print("OC-rg = OpenCode with ripgrep pre-placed (canonical); OC = the un-fixed no-web OpenCode arm.")
    print("Denominator = tasks whose canonical trial is usable (scored, non-infra); %s = no data." % DASH)


def canon_cols(best, m, s):
    """One column per harness for the paired block: OC-rg if it exists, else OC."""
    out = []
    for group in CANON_PREF:
        for lab in group:
            if usable(best.get((lab, m, s), {})):
                out.append(lab)
                break
    return out


def paired_table(best):
    section("TABLE 2 — PAIRED: common usable task set + pairwise exact McNemar")
    for m in MODEL_ORDER:
        for s in SUBSET_ORDER:
            present = [lab for lab in COL_LABELS if usable(best.get((lab, m, s), {}))]
            if not present:
                continue
            print()
            print(f"--- {MODEL_LABEL[m]} / {s} ---")
            canon = canon_cols(best, m, s)
            U = {lab: usable(best[(lab, m, s)]) for lab in present}
            if len(canon) >= 2:
                common = sorted(set.intersection(*[set(U[lab]) for lab in canon]))
                print(f"  common usable tasks across {len(canon)} harnesses ({', '.join(canon)}): n={len(common)}")
                if common:
                    for lab in canon:
                        k = sum(1 for t in common if (num(U[lab][t].get("reward")) or 0) >= 1)
                        lo, hi = wilson(k, len(common))
                        print(f"    {lab:6s} {100 * k / len(common):5.1f}% ({k}/{len(common)}) "
                              f"[{100 * lo:.1f},{100 * hi:.1f}]")
                else:
                    print(f"    {DASH} (no task usable in every harness)")
            else:
                print(f"  common usable tasks: {DASH} (only {len(canon)} harness with data)")
            print("  pairwise (each pair on its own common usable set):")
            if len(present) < 2:
                print(f"    {DASH}")
                continue
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = present[i], present[j]
                    ts = sorted(set(U[a]) & set(U[b]))
                    if not ts:
                        print(f"    {a:6s} vs {b:6s} {DASH} (no shared usable task)")
                        continue
                    pa = [(num(U[a][t].get("reward")) or 0) >= 1 for t in ts]
                    pb = [(num(U[b][t].get("reward")) or 0) >= 1 for t in ts]
                    only_a = sum(1 for x, y in zip(pa, pb) if x and not y)
                    only_b = sum(1 for x, y in zip(pa, pb) if y and not x)
                    ka, kb = sum(pa), sum(pb)
                    p = mcnemar_exact_p(only_a, only_b)
                    print(f"    {a:6s} vs {b:6s} n={len(ts):3d}  {ka:3d} ({100 * ka / len(ts):4.1f}%) vs "
                          f"{kb:3d} ({100 * kb / len(ts):4.1f}%)  b(only-{a})={only_a:3d} c(only-{b})={only_b:3d}  "
                          f"exact p={p:.4f}  flip={(only_a + only_b) / len(ts):.3f}")


def cost_table(best):
    section("TABLE 3 — COST / EFFORT: medians over the canonical usable trials")
    widths = [24, 10, 8, 8, 9, 12, 13]
    print(hsep(widths))
    print(row(["model / subset", "harness", "n", "steps", "wall_s", "ctok", "total $"], widths))
    print(hsep(widths))
    for m in MODEL_ORDER:
        printed = False
        for s in SUBSET_ORDER:
            for lab in COL_LABELS:
                u = usable(best.get((lab, m, s), {}))
                if not u:
                    continue
                rs = list(u.values())
                st = median([num(r.get("steps")) for r in rs])
                ws = median([num(r.get("wall_seconds")) for r in rs])
                ct = median([num(r.get("completion_tokens")) for r in rs])
                cost = sum(num(r.get("cost_usd")) or 0.0 for r in rs)
                print(row([f"{MODEL_LABEL[m]} / {s}", lab, str(len(rs)),
                           DASH if st is None else f"{st:.0f}",
                           DASH if ws is None else f"{ws:.0f}",
                           DASH if ct is None else f"{ct:.0f}",
                           DASH if cost == 0 else f"{cost:.2f}"], widths))
                printed = True
        if printed:
            print(hsep(widths))
    print("steps/wall_s/ctok are per-trial medians; total $ is the sum of the harness-reported cost_usd.")
    print("Caveat: Claude Code reports Anthropic list prices even when proxied to a local vLLM, so the")
    print("Qwen rows' dollars are notional accounting, not spend; mini/OpenCode on local Qwen report 0.")


def notes(rows, best, seeds_seen, k2_jobs, fields):
    section("NOTES — coverage, missing cells, seeds, data hygiene")
    missing = []
    for m in MODEL_ORDER:
        for s in SUBSET_ORDER:
            if not any(best.get((lab, m, s)) for lab in COL_LABELS):
                continue            # the whole model x subset block was never planned
            for lab in COL_LABELS:
                d = best.get((lab, m, s), {})
                if not d:
                    missing.append(f"{MODEL_LABEL[m]}/{s}/{lab}: no trial in trials.csv")
                elif not usable(d):
                    missing.append(f"{MODEL_LABEL[m]}/{s}/{lab}: {len(d)} trials, none usable")
    print("cells with no data:")
    for line in missing or [f"  {DASH} (every planned cell has data)"]:
        print("  " + line if not line.startswith("  ") else line)
    print()
    print("partial coverage (usable < tasks attempted in the arm):")
    part = []
    for (lab, m, s), d in sorted(best.items()):
        u = usable(d)
        if d and len(u) < len(d):
            bad = defaultdict(int)
            for t, r in d.items():
                if not ok(r):
                    bad[(r.get("exception_type") or r.get("status") or "unscored")] += 1
            part.append(f"  {MODEL_LABEL[m]}/{s}/{lab}: {len(u)}/{len(d)} usable; "
                        + ", ".join(f"{k} x{v}" for k, v in sorted(bad.items())))
    for line in part or [f"  {DASH}"]:
        print(line)
    print()
    print("arms whose seed index comes from the within-job 'attempt' column (k=2 attempts packed")
    print("into one repeat=1 job; retry/rerun jobs of these arms only refill unusable slots):")
    for k in sorted(k2_jobs):
        lab, m, s = k
        print(f"  {MODEL_LABEL[m]}/{s}/{lab}: {', '.join(sorted(k2_jobs[k]))}")
    if not k2_jobs:
        print(f"  {DASH}")
    print("  (agg.best_by_seed splits the same jobs with a slot heuristic on started_at, so its")
    print("   seed-1 counts for Qwen3.8/hard45 can differ from this table by one or two tasks.)")
    print()
    print(f"extra seeds present in trials.csv but outside this table (seed != {MAIN_SEED}):")
    extra = {k: sorted(v - {MAIN_SEED}) for k, v in seeds_seen.items() if v - {MAIN_SEED}}
    for k in sorted(extra):
        lab, m, s = k
        print(f"  {MODEL_LABEL[m]}/{s}/{lab}: seeds {extra[k]}")
    if not extra:
        print(f"  {DASH}")
    print()
    n_run = sum(1 for r in rows if r.get("status") == "running")
    print(f"trials.csv: {len(rows)} rows, {len(fields)} columns, {n_run} still running "
          f"(running rows are unscored and therefore never canonical).")
    for col in ("reward", "scored", "exclude", "steps", "wall_seconds", "completion_tokens", "cost_usd"):
        if col not in fields:
            print(f"  WARNING: column '{col}' absent; the affected cells print {DASH}.")
    print("scipy: " + ("available (McNemar cross-checked)" if HAVE_SCIPY else "absent (pure-python exact binomial)"))


def main():
    rows, fields = load(CSV_PATH)
    if not rows:
        print("no trials to aggregate.")
        return
    best, seeds_seen, k2_jobs = canonical(rows)
    print("MAIN RESULT TABLES — agent-harness (no-web), generated from data/trials.csv")
    print("models: " + ", ".join(MODEL_LABEL[m] for m in MODEL_ORDER)
          + "  |  subsets: " + ", ".join(SUBSET_ORDER)
          + "  |  arms: control-nw, control-nw-rg  |  seed: %d" % MAIN_SEED)
    main_table(best)
    paired_table(best)
    cost_table(best)
    notes(rows, best, seeds_seen, k2_jobs, fields)


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
    except BrokenPipeError:          # e.g. `main_table.py | head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
