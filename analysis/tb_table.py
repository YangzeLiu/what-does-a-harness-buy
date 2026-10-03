#!/usr/bin/env python3
"""Appendix table: Terminal-Bench 2.1 (tb21) harness swap, from data/trials.csv.

Scope: the 173 rows of trials.csv whose subset is 'tb21' — one model (Qwen3.6-35B-A3B),
one seed, three harnesses (claude-code / mini-swe-agent / opencode).  Each harness ran a
main job plus continuation jobs (-c, -c2), one retry job (-r) and one or two smoke jobs;
all jobs of a harness are one arm and are merged before the canonical attempt is picked.

Nothing statistical is defined here.  This script imports analysis/main_table.py and reuses
its definitions verbatim so the appendix cannot contradict the paper:
  * INFRA             the 14-type infrastructure-exception set,
  * ok()              usable trial = scored and exception_type not in INFRA,
  * canonical()       one row per (arm, task): usable beats unusable, ties to latest
                      finished_at (no tb21 job packs k=2 attempts, so that branch is inert),
  * usable()/passes(), wilson(), mcnemar_exact_p(), and the hsep()/row() rendering.
main_table.canonical() reads MODEL_ORDER / SUBSET_ORDER / COLS as module globals, so the
tb21 column spec below is installed into that module before it is called; that is the only
thing this script changes about main_table.

Subset.  Two nested task lists are defined: offline-55 is
scripts/tb21_off_tasks.txt (the 89 upstream tasks minus the 34 whose reference solution needs
the network during the agent phase), and offline-53 — the REPORTED subset — is those 55 minus
qemu-alpine-ssh and qemu-startup, whose agent setup fails deterministically on a Debian 11 base
whose bullseye-security mirror is gone (an environment defect, not a harness one; only OpenCode
ever attempted them).  Every rate in A1 and A2 is computed on offline-53, read from that file
rather than hardcoded and asserted to still be 55 lines containing both qemu names.  A3 is the
full accounting and shows how the 56 distinct tasks in trials.csv reduce to the reported 53.

Smoke rows.  trials.csv marks every smoke row exclude=1 (8 rows, exactly the five *smoke*
jobs), and main_table drops exclude=1.  Terminal-Bench smoke jobs ran the same configuration,
so they are included here (INCLUDE_SMOKE) and the counts block reports whether including
them moves any cell.

Run: python3 analysis/tb_table.py > analysis/tb_table.txt   (reproduce.sh checks it)
Read-only; stdlib plus whatever main_table.py itself imports.
"""
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import main_table as mt          # noqa: E402  (path fix must precede the import)

SUBSET = "tb21"
MODEL = "qwen3.6-35b-a3b"
MODEL_LABEL = "Qwen3.6-35B"
# column label -> (harness, arm).  OpenCode only ever ran the ripgrep-fixed arm on tb21.
TB_COLS = [("CC", "claude-code", "control-nw"),
           ("mini", "mini-swe-agent", "control-nw"),
           ("OC-rg", "opencode", "control-nw-rg")]
TB_LABELS = [c[0] for c in TB_COLS]
HARNESS_OF = {lab: h for lab, h, _ in TB_COLS}
INCLUDE_SMOKE = True
DASH = mt.DASH

# offline-55 -> offline-53 (see the docstring).
OFF_TASKS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              "scripts", "tb21_off_tasks.txt")
QEMU_DROP = ("qemu-alpine-ssh", "qemu-startup")
QEMU_REASON = ("agent setup fails deterministically on the Debian 11 base "
               "(bullseye-security gone from the mirrors -> apt 404 -> exit 100)")


def load_offline_subsets():
    """-> (offline_55, offline_53) from scripts/tb21_off_tasks.txt, loudly checked.

    The list is read from the file, never hardcoded; if it is ever regenerated with a
    different length or without the two qemu tasks, this raises instead of silently
    reporting a different denominator than the appendix promises.
    """
    with open(OFF_TASKS_PATH) as fh:
        off55 = [ln.strip() for ln in fh if ln.strip()]
    if len(off55) != 55 or len(set(off55)) != 55:
        raise AssertionError(f"{OFF_TASKS_PATH}: expected 55 distinct tasks (offline-55), "
                             f"got {len(off55)} lines / {len(set(off55))} distinct")
    missing = [t for t in QEMU_DROP if t not in off55]
    if missing:
        raise AssertionError(f"{OFF_TASKS_PATH}: offline-53 is offline-55 minus "
                             f"{', '.join(QEMU_DROP)}, but {', '.join(missing)} is not in the file")
    off53 = set(off55) - set(QEMU_DROP)
    assert len(off53) == 53, len(off53)
    return set(off55), off53


def restrict(best, tasks):
    """The canonical dicts cut down to a task subset (offline-53 for A1/A2)."""
    return {lab: {t: r for t, r in d.items() if t in tasks} for lab, d in best.items()}


def configure_main_table():
    """Point main_table's canonical() at the tb21 cell (it reads these as globals)."""
    mt.MODEL_ORDER = [MODEL]
    mt.MODEL_LABEL = {MODEL: MODEL_LABEL}
    mt.SUBSET_ORDER = [SUBSET]
    mt.COLS = TB_COLS
    mt.COL_LABELS = TB_LABELS


def tb_rows(rows, include_smoke):
    """The tb21 rows, with smoke rows un-excluded when include_smoke (they carry exclude=1)."""
    out = []
    for r in rows:
        if r.get("subset") != SUBSET:
            continue
        if include_smoke and "smoke" in (r.get("job") or "") and r.get("exclude") == "1":
            r = dict(r, exclude="0")
        out.append(r)
    return out


def best_of(rows, include_smoke):
    best, _seeds, _k2 = mt.canonical(tb_rows(rows, include_smoke))
    return {lab: best.get((lab, MODEL, SUBSET), {}) for lab in TB_LABELS}


# ----------------------------------------------------------------------------- blocks
def rate_table(best):
    mt.section("TABLE A1 — TERMINAL-BENCH 2.1 (offline-53): pass rate % (passes/usable tasks) "
               "[Wilson 95%], no-web, seed 1")
    widths = [10, 15, 9, 10, 9, 20]
    print(mt.hsep(widths))
    print(mt.row(["harness", "arm", "passes", "usable n", "pass %", "Wilson 95% CI"], widths))
    print(mt.hsep(widths))
    for lab, h, arm in TB_COLS:
        d = best[lab]
        u = mt.usable(d)
        n = len(u)
        if n == 0:
            print(mt.row([lab, arm, DASH, "0", DASH, DASH], widths))
            continue
        k = mt.passes(u)
        lo, hi = mt.wilson(k, n)
        print(mt.row([lab, arm, str(k), str(n), f"{100 * k / n:.1f}",
                      f"[{100 * lo:.1f},{100 * hi:.1f}]"], widths))
    print(mt.hsep(widths))
    print(f"model {MODEL_LABEL} ({MODEL}); OC-rg = OpenCode with ripgrep pre-placed (the only")
    print("OpenCode arm run on Terminal-Bench).  Task universe = offline-53 (scripts/tb21_off_tasks.txt")
    print("minus the two qemu tasks); denominator = the offline-53 tasks whose")
    print("canonical trial is usable (scored, exception_type not in the 14-type INFRA set).  Each")
    print("harness's main, continuation, retry and smoke jobs are merged into one arm first.")


def paired_block(best):
    mt.section("TABLE A2 — TERMINAL-BENCH 2.1 PAIRED (offline-53): tasks usable in all three harnesses")
    U = {lab: mt.usable(best[lab]) for lab in TB_LABELS}
    common = sorted(set.intersection(*[set(U[lab]) for lab in TB_LABELS]))
    print()
    print(f"  common usable offline-53 tasks across 3 harnesses ({', '.join(TB_LABELS)}): "
          f"n={len(common)}")
    if not common:
        print(f"    {DASH} (no task usable in every harness)")
        return common
    for lab in TB_LABELS:
        k = sum(1 for t in common if (mt.num(U[lab][t].get("reward")) or 0) >= 1)
        lo, hi = mt.wilson(k, len(common))
        print(f"    {lab:6s} {100 * k / len(common):5.1f}% ({k}/{len(common)}) "
              f"[{100 * lo:.1f},{100 * hi:.1f}]")
    print("  pairwise exact McNemar on the common set:")
    for i in range(len(TB_LABELS)):
        for j in range(i + 1, len(TB_LABELS)):
            a, b = TB_LABELS[i], TB_LABELS[j]
            pa = [(mt.num(U[a][t].get("reward")) or 0) >= 1 for t in common]
            pb = [(mt.num(U[b][t].get("reward")) or 0) >= 1 for t in common]
            only_a = sum(1 for x, y in zip(pa, pb) if x and not y)
            only_b = sum(1 for x, y in zip(pa, pb) if y and not x)
            ka, kb = sum(pa), sum(pb)
            p = mt.mcnemar_exact_p(only_a, only_b)
            print(f"    {a:6s} vs {b:6s} n={len(common):3d}  {ka:3d} ({100 * ka / len(common):4.1f}%) vs "
                  f"{kb:3d} ({100 * kb / len(common):4.1f}%)  b(only-{a})={only_a:3d} c(only-{b})={only_b:3d}  "
                  f"exact p={p:.4f}  flip={(only_a + only_b) / len(common):.3f}")
    return common


def counts_block(rows, best, best53, best_nosmoke, off55, off53, common):
    """Full accounting: how trials.csv's 56 distinct tasks reduce to the reported 53, then 50."""
    mt.section("TABLE A3 — TERMINAL-BENCH 2.1 COUNTS: subset nesting, coverage, dropped rows")
    tb = tb_rows(rows, INCLUDE_SMOKE)
    all_tasks = sorted({r["task"] for r in tb})
    per_h = {lab: {r["task"] for r in tb if r.get("harness") == HARNESS_OF[lab]} for lab in TB_LABELS}
    in53 = {lab: per_h[lab] & off53 for lab in TB_LABELS}
    U53 = {lab: mt.usable(best53[lab]) for lab in TB_LABELS}
    smoke_only = sorted(set(all_tasks) - off55)
    attempted_all = set.intersection(*in53.values())

    print()
    print("  subset nesting (each step names what it loses and why):")
    print(f"    {len(all_tasks):3d}  distinct tasks in trials.csv, subset=tb21 "
          f"({len(tb)} rows in {len({r['job'] for r in tb})} jobs)")
    print(f"     -{len(smoke_only):<2d} {', '.join(smoke_only) if smoke_only else DASH}: "
          "run only by a smoke job, not in scripts/tb21_off_tasks.txt")
    print(f"    {len(off55):3d}  offline-55 = scripts/tb21_off_tasks.txt "
          "(89 upstream tasks minus the 34 needing the network in the agent phase)")
    print(f"     -{len(QEMU_DROP):<2d} {', '.join(QEMU_DROP)}: {QEMU_REASON};")
    print("         an environment defect, not a harness one, and only OC-rg ever attempted them")
    print(f"    {len(off53):3d}  offline-53 = the REPORTED subset; every rate in A1 and A2 uses it")
    lost_att = sorted(off53 - attempted_all)
    print(f"     -{len(lost_att):<2d} " + (", ".join(lost_att) if lost_att
                                           else "(none: all three harnesses attempted all 53)"))
    print(f"    {len(attempted_all):3d}  attempted by all three harnesses")
    lost_use = sorted(attempted_all - set(common))
    print(f"     -{len(lost_use):<2d} " + (", ".join(lost_use) if lost_use else "(none)")
          + ": not usable in every harness (see the per-task reasons below)")
    print(f"    {len(common):3d}  common usable set (the paired n of A2)")
    print()
    print("  offline-53 per harness (A1 denominators):")
    widths = [10, 10, 11, 9, 9, 10]
    print(mt.hsep(widths))
    print(mt.row(["harness", "attempted", "not run", "usable", "passes", "dropped"], widths))
    print(mt.hsep(widths))
    for lab in TB_LABELS:
        d, u = best53[lab], U53[lab]
        print(mt.row([lab, str(len(d)), str(len(off53) - len(in53[lab])),
                      str(len(u)), str(mt.passes(u)), str(len(d) - len(u))], widths))
    print(mt.hsep(widths))
    print()
    print("  full tb21 footprint per harness (offline-53 plus everything outside it):")
    widths2 = [10, 9, 9, 11, 12, 14]
    print(mt.hsep(widths2))
    print(mt.row(["harness", "rows", "jobs", "all tasks", "of which 53", "outside 53"], widths2))
    print(mt.hsep(widths2))
    for lab in TB_LABELS:
        rs = [r for r in tb if r.get("harness") == HARNESS_OF[lab]]
        out = sorted(per_h[lab] - off53)
        print(mt.row([lab, str(len(rs)), str(len({r["job"] for r in rs})),
                      str(len(per_h[lab])), str(len(in53[lab])), str(len(out))], widths2))
    print(mt.hsep(widths2))
    for lab in TB_LABELS:
        out = sorted(per_h[lab] - off53)
        print(f"    {lab:6s} outside offline-53: " + (", ".join(out) if out else DASH))
    print()
    print("  offline-53 tasks a harness never ran:")
    miss = False
    for lab in TB_LABELS:
        gone = sorted(off53 - in53[lab])
        if gone:
            miss = True
            print(f"    {lab:6s} missing {len(gone)}: {', '.join(gone)}")
    if not miss:
        print(f"    {DASH} (all three harnesses ran all {len(off53)} offline-53 tasks)")
    print()
    print("  offline-53 canonical trials dropped as not usable (task counted once, by its canonical row):")
    for lab in TB_LABELS:
        d = best53[lab]
        bad = defaultdict(int)
        for t, r in d.items():
            if not mt.ok(r):
                exc = r.get("exception_type") or "no exception"
                # ok() has two gates; say which one rejected the row.
                bad[exc if r.get("scored") == "1" else f"unscored/{exc}"] += 1
        tot = sum(bad.values())
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(bad.items())) if bad else DASH
        print(f"    {lab:6s} {tot:2d} of {len(d)}: {detail}")
    print()
    print("  why each offline-53 task falls out of the common usable set:")
    for t in sorted(attempted_all - set(common)):
        why = []
        for lab in TB_LABELS:
            r = best53[lab].get(t)
            if r is None:
                why.append(f"{lab}=not run")
            elif not mt.ok(r):
                exc = r.get("exception_type") or "no exception"
                why.append(f"{lab}=" + (exc if r.get("scored") == "1" else f"unscored/{exc}"))
        print(f"    {t}: " + "; ".join(why))
    print()
    print("  raw tb21 rows dropped as INFRA (all rows of the harness, before the canonical pick):")
    for lab in TB_LABELS:
        bad = defaultdict(int)
        for r in tb:
            if r.get("harness") == HARNESS_OF[lab] and (r.get("exception_type") or "") in mt.INFRA:
                bad[r["exception_type"]] += 1
        tot = sum(bad.values())
        detail = ", ".join(f"{k} x{v}" for k, v in sorted(bad.items())) if bad else DASH
        print(f"    {lab:6s} {tot:2d}: {detail}")
    print()
    print("  non-INFRA exceptions kept as ordinary failures (budget/agent errors count as fail):")
    keep = defaultdict(int)
    for r in tb:
        e = r.get("exception_type") or ""
        if e and e not in mt.INFRA:
            keep[e] += 1
    print("    " + (", ".join(f"{k} x{v}" for k, v in sorted(keep.items())) if keep else DASH))
    print()
    print("  smoke jobs (exclude=1 in trials.csv, the rule main_table.py applies) are merged")
    print("  into their harness's arm here.  Effect of dropping them instead:")
    cell_diff, task_diff, attempted = [], [], []
    for lab in TB_LABELS:
        a, b = best[lab], best_nosmoke[lab]
        ua, ub = mt.usable(a), mt.usable(b)
        if (len(ua), mt.passes(ua)) != (len(ub), mt.passes(ub)):
            cell_diff.append(f"    {lab:6s} with smoke {mt.passes(ua)}/{len(ua)}; "
                             f"without {mt.passes(ub)}/{len(ub)}")
        for t in sorted(set(a) | set(b)):
            ra, rb = a.get(t), b.get(t)
            if ra is None or rb is None:
                attempted.append(f"    {lab:6s} task {t} ("
                                 + ("in offline-53" if t in off53 else "outside offline-53") + "): "
                                 + ("smoke-only" if rb is None else "smoke row not canonical"))
            elif (mt.ok(ra), mt.num(ra.get("reward"))) != (mt.ok(rb), mt.num(rb.get("reward"))):
                task_diff.append(f"    {lab:6s} task {t}: canonical outcome differs")
    for line in cell_diff or ["    no cell moves anywhere: passes and usable n are identical either way."]:
        print(line)
    for line in task_diff or ["    no task's canonical outcome changes."]:
        print(line)
    print("    only the attempted-task count outside offline-53 moves:")
    for line in attempted or [f"    {DASH}"]:
        print(line)


def main():
    rows, fields = mt.load(mt.CSV_PATH)
    if not rows:
        print("no trials to aggregate.")
        return
    configure_main_table()
    off55, off53 = load_offline_subsets()
    best = best_of(rows, INCLUDE_SMOKE)                 # every tb21 task, for the A3 accounting
    best53 = restrict(best, off53)                      # the reported subset, for A1 and A2
    best_nosmoke = best_of(rows, False)
    print("APPENDIX TABLE — Terminal-Bench 2.1 harness swap, generated from data/trials.csv")
    print(f"model: {MODEL_LABEL}  |  subset: {SUBSET} / offline-53  |  harnesses: "
          + ", ".join(f"{lab} ({h}/{a})" for lab, h, a in TB_COLS)
          + f"  |  seed: {mt.MAIN_SEED}")
    print("rates are computed on offline-53 = scripts/tb21_off_tasks.txt (offline-55) minus "
          + ", ".join(QEMU_DROP) + ";")
    print("statistics, usability rule and canonical-attempt rule are imported from analysis/main_table.py")
    rate_table(best53)
    common = paired_block(best53)
    counts_block(rows, best, best53, best_nosmoke, off55, off53, common)
    print()
    print(f"trials.csv: {len(rows)} rows, {len(fields)} columns.")
    print("scipy: " + ("available (McNemar cross-checked)" if mt.HAVE_SCIPY
                       else "absent (pure-python exact binomial)"))


if __name__ == "__main__":
    try:
        main()
        sys.stdout.flush()
    except BrokenPipeError:              # e.g. `tb_table.py | head`
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
