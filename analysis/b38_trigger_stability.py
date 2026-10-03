#!/usr/bin/env python3
"""Is a long think a property of the task, or of the run?

B38's 41 tasks were selected because round 1 hit the auto-continue trigger in the
parent run (`oc-qwen-pool447-nw-rg`).  On rerun the trigger mostly does not fire
again, which dilutes the ITT contrast.  This script asks why, by comparing the
parent run and the B38 stock rerun task by task on measures of how much the model
thought.

The obvious measure -- max per-step reasoning tokens -- is useless here: in the
parent run every cut trial is censored at exactly 32000, so most x-values are
tied and a rank correlation on it means nothing.  The measures below are not
censored.

Usage: b38_trigger_stability.py <jobs_root> [--parent JOB] [--rerun JOB]
                                [--tasks scripts/B38_tasks.txt] [--tokenizer PATH]
"""
import argparse, glob, json, math, os, statistics, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thinking_budget import scan, load_tokenizer, tok_len, DEFAULT_TOK  # noqa: E402


def spearman(x, y):
    """Rank correlation with tie-averaged ranks."""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    a, b = rank(x), rank(y)
    n = len(x)
    m1, m2 = statistics.mean(a), statistics.mean(b)
    num = sum((a[i] - m1) * (b[i] - m2) for i in range(n))
    den = math.sqrt(sum((a[i] - m1) ** 2 for i in range(n)) * sum((b[i] - m2) ** 2 for i in range(n)))
    return num / den if den else 0.0


def features(tk, root, job):
    """First attempt per task, CHRONOLOGICALLY (result.json started_at).

    Trial directories are <task>__<random suffix>, so iterating them in sorted order and
    keeping the first picks the alphabetically smallest random suffix, not the first
    attempt -- on the two-attempt B38 arms that disagrees with the real attempt order on
    23 of 41 tasks, and it moved the rank correlations below by up to 0.2.
    """
    by_task = {}
    for d in sorted(glob.glob(os.path.join(root, job, "*"))):
        f = os.path.join(d, "agent", "opencode.txt")
        if not os.path.exists(f):
            continue
        task = os.path.basename(d).rsplit("__", 1)[0]
        started = ""
        rj = os.path.join(d, "result.json")
        if os.path.exists(rj):
            try:
                started = json.load(open(rj)).get("started_at") or ""
            except Exception:
                started = ""
        by_task.setdefault(task, []).append((started, d, f))
    out = {}
    for task, attempts in by_task.items():
        attempts.sort(key=lambda x: (x[0] == "", x[0], x[1]))   # missing timestamps last
        f = attempts[0][2]
        rs, total, cut = [], 0, False
        for m in scan(f):
            r = tok_len(tk, m["reason"])
            rs.append(r)
            total += r
            if m["reason_fin"] == "length":
                cut = True
        if rs:
            out[task] = dict(total=total, med=statistics.median(rs), mx=max(rs),
                             steps=len(rs), cut=cut, attempts=len(attempts))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_root")
    ap.add_argument("--parent", default="oc-qwen-pool447-nw-rg")
    ap.add_argument("--rerun", default="oc-qwen38-B-stock")
    ap.add_argument("--tasks", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "B38_tasks.txt"))
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    a = ap.parse_args()
    tk = load_tokenizer(a.tokenizer)

    P = features(tk, a.jobs_root, a.parent)
    R = features(tk, a.jobs_root, a.rerun)
    tasks = [t.strip() for t in open(a.tasks)
             if t.strip() and not t.startswith("#")]
    common = [t for t in tasks if t in P and t in R]

    print("parent = %s   rerun = %s" % (a.parent, a.rerun))
    print("one trial per task in each run: the CHRONOLOGICALLY first attempt (started_at)")
    print("B38 tasks with a first-attempt log in both: %d of %d" % (len(common), len(tasks)))
    print()
    print("does the selection variable itself come back?")
    pc = sum(1 for t in common if P[t]["cut"])
    rc = sum(1 for t in common if R[t]["cut"])
    print("   length cut: parent %d/%d (%.0f%%)   rerun %d/%d (%.0f%%)"
          % (pc, len(common), 100.0 * pc / len(common), rc, len(common), 100.0 * rc / len(common)))
    print()
    print("cross-run rank correlation on measures that are NOT censored at the 32000 cut:")
    for key, label in (("total", "total reasoning tokens"), ("med", "median per-step reasoning"),
                       ("steps", "steps per trial")):
        x = [P[t][key] for t in common]
        y = [R[t][key] for t in common]
        print("   %-26s parent median %7.0f   rerun median %7.0f   rho = %+.3f"
              % (label, statistics.median(x), statistics.median(y), spearman(x, y)))
    print("   (max per-step reasoning is deliberately omitted: censored at 32000 in the parent)")
    print()
    print("what the cut costs, visible in the step counts above: a cut ends the OpenCode")
    print("loop, so the parent trials stop early while the same tasks run to length on rerun.")


if __name__ == "__main__":
    main()
