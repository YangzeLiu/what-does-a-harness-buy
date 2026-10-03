#!/usr/bin/env python3
"""Recompute the cutoff-audit numbers (paper appendix "The output cutoff and a recovery
transplant") from the job directories.

Why this exists: the numbers in the cutoff-audit table were first computed ad hoc, while two of the
arms were still running (Claude Code at 248/447, OpenCode-rg at 347/447).  Nothing recorded how
they were counted, so they could not be refreshed when the arms completed.  This script is the
provenance: give it job directories and it prints the table.

A "cut" is one output-token cutoff *event*, not a trial:

  OpenCode     a ``step_finish`` part whose ``reason`` is ``"length"``.  The provider config
               carries ``limit.output: 32768`` and the cuts land at output = 32000.  OpenCode
               has no recovery for this, so the cut is terminal whenever it lands on the final
               step of the log.
  Claude Code  the client detects the truncation and injects its own resume prompt,
               "Output token limit hit. Resume directly - no apology, no recap of what you were
               doing."  The cap can also come back as a fatal API error ("exceeded the 32000
               output token maximum"), which ends the run with OutputTokenExceededError; and it
               can hit the *compaction* request instead, which Claude Code logs as
               ``compact_result: "failed"`` and survives.  All three are the same 32000 ceiling.

               All three are detected STRUCTURALLY, on the event shape, never by a bare
               substring search.  An earlier version counted any line containing the resume
               string and over-counted the arm by 53 % (100 events instead of 66), because the
               string also occurs in two other places that are not cut events:
                 * Claude Code's compaction summaries ("This session is being continued from a
                   previous conversation that ran out of context ...") quote the earlier resume
                   prompt inside the recap, and re-quote it at every later compaction;
                 * the model sometimes reproduces the prompt verbatim inside its own
                   ``thinking`` block.
               A genuine injection is a *whole user turn* whose content is nothing but the
               prompt, which is what RESUME_EVENT below requires.

Pass rates use the project's usable-trial rule: the exception type is not in the INFRA set and
the trial carries a reward.

    python3 analysis/cutoff_audit.py <jobs_root> <job> [<job> ...]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thinking_budget import scan  # noqa: E402  (OpenCode event-log reader)

INFRA = {"NetworkConnectionError", "AgentSetupTimeoutError", "RuntimeError", "CancelledError",
         "OpenCodeStartupHang", "OpenCodeStartupError", "AgentVerifierOverlap",
         "VerifierTimeoutError", "verifier_incomplete", "VerifierUvMissing",
         "RewardFileNotFoundError", "ApiQuotaOrAuthError", "AgentIncomplete", "cc_api_error"}

CC_RESUME = "Output token limit hit"          # prefilter only; see is_resume_injection()
CC_CAP_ERR = "output token maximum"           # prefilter only; see cc_cuts()
CC_API_ERR = "max_output_tokens"              # Claude Code's own error code for the cap
CC_COMPACT = '"compact_result"'


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def oc_cuts(path):
    """-> (n_cut_events, terminal) for an OpenCode trial."""
    msgs = scan(path)
    cuts = [i for i, m in enumerate(msgs) if m["reason_fin"] == "length"]
    return len(cuts), bool(cuts) and cuts[-1] == len(msgs) - 1


def is_resume_injection(o):
    """True iff this event is Claude Code injecting its resume prompt as a user turn.

    Shape: {"type":"user", "parent_tool_use_id":null,
            "message":{"role":"user","content":[{"type":"text","text":"Output token limit hit...."}]}}
    Requiring a SINGLE text part that STARTS with the prompt is what makes the two
    false-positive classes impossible to count: a compaction summary is one text part but
    starts with "This session is being continued", and a model quotation lives in an
    ``assistant`` event (and inside a ``thinking`` part, never as the whole user turn).
    A tool result is also a "user" event, but carries parent_tool_use_id and a
    tool_result part rather than a lone text part.
    """
    if o.get("type") != "user" or o.get("parent_tool_use_id") is not None:
        return False
    msg = o.get("message") or {}
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list) or len(content) != 1:
        return False
    part = content[0]
    if not isinstance(part, dict) or part.get("type") != "text":
        return False
    return (part.get("text") or "").lstrip().startswith(CC_RESUME)


def is_fatal_cap_error(o):
    """The cap returning as a fatal API error: Claude Code emits a <synthetic> assistant
    message flagged is_api_error_message with api_error == "max_output_tokens", then exits."""
    return (o.get("type") == "assistant"
            and bool(o.get("is_api_error_message"))
            and (o.get("api_error") == CC_API_ERR or o.get("error") == CC_API_ERR))


def is_compact_cap_failure(o):
    """The cap hitting the compaction request instead of an assistant turn.  Claude Code
    logs a system:status event with compact_result "failed"; the cap text is in
    compact_error.  Successful compactions carry compact_result too, hence both tests."""
    return (o.get("type") == "system"
            and o.get("compact_result") == "failed"
            and CC_CAP_ERR in (o.get("compact_error") or ""))


def cc_cuts(path):
    """-> (n_recovered, n_cap_errors, n_compact_failures) for a Claude Code trial.

    Streamed as text: these logs reach 30 MB, almost all of it ``system:thinking_tokens``
    progress events, and json.loads on every line costs minutes per arm for nothing.  So a
    cheap substring prefilter picks the candidate lines and only those are parsed; the
    decision itself is always structural.  ``result`` events are excluded by type, which is
    what the old '"total_cost_usd" in line' skip was reaching for.
    """
    rec = err = comp = 0
    for line in open(path, errors="replace"):
        if not (CC_RESUME in line or CC_API_ERR in line or CC_COMPACT in line):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") == "result":
            continue  # the final result event echoes the last error text; not a separate event
        if is_resume_injection(o):
            rec += 1
        elif is_fatal_cap_error(o):
            err += 1
        elif is_compact_cap_failure(o):
            err += 1
            comp += 1
    return rec, err, comp


def audit(root, job):
    jd = os.path.join(root, job)
    rows = []
    for name in sorted(os.listdir(jd)):
        rp = os.path.join(jd, name, "result.json")
        if not os.path.exists(rp):
            continue
        try:
            d = json.load(open(rp))
        except Exception:
            continue
        exc = (d.get("exception_info") or {}).get("exception_type")
        rew = ((d.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        ag = os.path.join(jd, name, "agent")
        oc_log = os.path.join(ag, "opencode.txt")
        cc_log = os.path.join(ag, "claude-code.txt")
        cuts = terminal = 0
        extra = ""
        if os.path.exists(oc_log):
            n, term = oc_cuts(oc_log)
            cuts, terminal = n, int(term)
        elif os.path.exists(cc_log):
            rec, err, comp = cc_cuts(cc_log)
            cuts = rec + err
            # a cut is terminal for Claude Code only when the cap came back as the fatal error
            terminal = int(exc == "OutputTokenExceededError")
            extra = "rec=%d err=%d compact_fail=%d" % (rec, err, comp)
        else:
            continue
        rows.append(dict(task=name, cuts=cuts, terminal=terminal, exc=exc, rew=rew, extra=extra))
    return rows


def report(job, rows):
    n = len(rows)
    cut = [r for r in rows if r["cuts"]]
    term = [r for r in cut if r["terminal"]]
    usable = [r for r in rows if r["exc"] not in INFRA and r["rew"] is not None]
    with_cut = [r for r in usable if r["cuts"]]
    no_cut = [r for r in usable if not r["cuts"]]

    def rate(lst):
        if not lst:
            return "n/a"
        k = sum(1 for r in lst if r["rew"] >= 1.0)
        lo, hi = wilson(k, len(lst))
        return "%.1f %% [%.1f, %.1f] (%d/%d)" % (100 * k / len(lst), lo, hi, k, len(lst))

    print("== %s" % job)
    print("   trials scanned          : %d" % n)
    print("   trials with >=1 cut     : %d  (%.1f %%)" % (len(cut), 100 * len(cut) / n if n else 0))
    print("   cut events              : %d" % sum(r["cuts"] for r in cut))
    print("   trials ending on the cut: %d" % len(term))
    print("   pass rate with a cut    : %s" % rate(with_cut))
    print("   pass rate without       : %s" % rate(no_cut))
    if any(r["extra"] for r in cut):
        rec = sum(int(r["extra"].split()[0].split("=")[1]) for r in cut if r["extra"])
        err = sum(int(r["extra"].split()[1].split("=")[1]) for r in cut if r["extra"])
        comp = sum(int(r["extra"].split()[2].split("=")[1]) for r in cut if r["extra"])
        print("   claude-code breakdown   : %d resume injections, %d cap errors "
              "(%d of them on the compaction request)" % (rec, err, comp))
    if term:
        print("   terminal trials         : %s" % ", ".join(r["task"] for r in term))
    print("   INFRA-excluded          : %d" % (n - len(usable)))


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)
    root, jobs = sys.argv[1], sys.argv[2:]
    for job in jobs:
        report(job, audit(root, job))
        print()


if __name__ == "__main__":
    main()
