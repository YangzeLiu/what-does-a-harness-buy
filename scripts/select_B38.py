#!/usr/bin/env python3
"""Define the Qwen3.8 termination-ablation set B from a finished OpenCode job.

B = failed trials that ended either (a) with the final step cut off by the output cap
(step_finish reason=length) or (b) without touching the repository (no edit/write/patch tool call).
Those are exactly the failures CC's cutoff-recovery prompt is hypothesised to rescue.

Usage: select_B38.py <jobs_root> <job_name> [--out FILE]
Prints one task id per line to --out, and a breakdown to stderr.
"""
import json, os, sys, collections

EDIT_TOOLS = {"edit", "write", "patch", "multiedit", "apply_patch"}

def scan(job_dir):
    rows = []
    for name in sorted(os.listdir(job_dir)):
        d = os.path.join(job_dir, name)
        rj, oc = os.path.join(d, "result.json"), os.path.join(d, "agent", "opencode.txt")
        if not os.path.isfile(rj):
            continue
        task = name.rsplit("__", 1)[0]
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        vr = r.get("verifier_result") or {}
        rew = (vr.get("rewards") or {}).get("reward")
        exc = (r.get("exception_info") or {}).get("exception_type")
        last_reason, n_len, edits, calls = None, 0, 0, 0
        if os.path.isfile(oc):
            for line in open(oc, errors="replace"):
                if '"step_finish"' in line:
                    try:
                        p = json.loads(line)["part"]
                    except Exception:
                        continue
                    rs = p.get("reason")
                    if rs:
                        last_reason = rs
                        n_len += rs == "length"
                elif '"tool_use"' in line:
                    try:
                        p = json.loads(line)["part"]
                    except Exception:
                        continue
                    calls += 1
                    if (p.get("tool") or "").lower() in EDIT_TOOLS:
                        edits += 1
        rows.append(dict(task=task, trial=name, reward=rew, exc=exc,
                         last_reason=last_reason, n_length=n_len, edits=edits, calls=calls))
    return rows

def main():
    jobs_root, job = sys.argv[1], sys.argv[2]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else None
    rows = scan(os.path.join(jobs_root, job))
    fails = [r for r in rows if r["reward"] == 0.0 and not r["exc"]]
    cut = [r for r in fails if r["last_reason"] == "length"]
    zero = [r for r in fails if r["last_reason"] != "length" and r["edits"] == 0]
    B = sorted({r["task"] for r in cut} | {r["task"] for r in zero})
    e = sys.stderr
    print(f"trials with result.json : {len(rows)}", file=e)
    print(f"  pass                  : {sum(1 for r in rows if r['reward'] == 1.0)}", file=e)
    print(f"  fail (no exception)   : {len(fails)}", file=e)
    print(f"  exceptions            : {dict(collections.Counter(r['exc'] for r in rows if r['exc']))}", file=e)
    print(f"B = {len(B)} tasks", file=e)
    print(f"  (a) final step length-cut : {len(cut)}", file=e)
    print(f"  (b) zero-edit, not cut    : {len(zero)}", file=e)
    print(f"  any length step anywhere  : {sum(1 for r in rows if r['n_length'])}", file=e)
    fh = open(out, "w") if out else sys.stdout
    for t in B:
        print(t, file=fh)
    if out:
        fh.close()
        print(f"wrote {out}", file=e)

main()
