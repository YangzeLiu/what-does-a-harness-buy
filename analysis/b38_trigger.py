#!/usr/bin/env python3
"""Reconstruct the B38 round-1 auto-continue trigger state from agent logs.

Why this exists.  B38's 41 tasks were selected because round 1 triggered the
auto-continue rule in the *parent* oc-qwen-pool447-nw-rg run.  On rerun the
failure mode does not reproduce on most of them, so the ITT contrast over all
41 tasks is diluted.  To quantify the dilution we need the round-1 trigger
state in BOTH arms -- but the stock arm runs with HARBOR_OC_AUTOCONTINUE unset
and therefore writes no autocontinue.txt at all.  It has to come from the logs.

How.  agent/opencode.txt is one JSON event per line.  The relevant events:
  {"type":"step_finish", "part":{"reason":"length"|"stop"|"tool-calls", ...}}
  {"type":"tool_use",    "part":{"tool":"edit"|"bash"|..., "state":{"status":...}}}

  rule (a)  final step of the round finished with reason == "length"
  rule (b)  the round left the repository untouched

Rule (b) is approximated by "no completed edit/write/patch call and no bash
command that writes into /testbed".  Validated below against the ac arm's own
autocontinue.txt on every trial where no continue round ran (rounds_used=0),
because there the whole log file IS round 1.  Trials that did continue span two
rounds in one file with no round marker, so they are excluded from validation
and their trigger state is read from the log instead of reconstructed.

Usage:  b38_trigger.py <jobs_root> [--stock JOB] [--ac JOB]
"""
import argparse, glob, json, os, re, sys

EDIT_TOOLS = {"edit", "write", "patch", "multiedit"}
RE_REASON = re.compile(r'"reason":"([a-z-]+)"')
RE_TOOL = re.compile(r'"tool":"([a-z_]+)"')

# ---------------------------------------------------------------------------
# Does a bash command write into the repository at /testbed?
#
# This is deliberately narrow: read-only probing (git log/show/diff/status, pytest,
# python -c, ls) must not count as a modification.  It is also deliberately ANCHORED:
# an earlier one-line regex counted `sed -i` anywhere and `git -C /testbed stash` with
# any suffix, and so booked two read-only trials as repository writes --
#   cd /tmp/opencode && sed -i 's/.../.../' repro_view.py     (writes /tmp, not the repo)
#   git -C /testbed stash list                                (a listing)
# -- which cost the cutoff audit two of its "32 of 34 terminal cuts left the repo unmodified".
#
# The command is split into segments on ; && || | and newlines, `cd` is tracked so a
# relative write after `cd /testbed` still counts, and every write test requires either
# a /testbed-rooted argument or a /testbed cwd.
# ---------------------------------------------------------------------------
REPO = "/testbed"
GIT_WRITE = {"apply", "checkout", "restore", "revert", "reset", "stash", "am",
             "cherry-pick", "clean", "mv", "rm", "commit", "merge", "rebase"}
# bare `git stash` == `git stash push`, so it IS a write; only these verbs are read-only.
GIT_STASH_READONLY = {"list", "show"}
FILE_WRITERS = {"cp", "mv", "rm", "touch", "mkdir", "install", "ln", "truncate", "dd", "rsync"}
SEP_OPS = {";", "|", "||", "&&", "&", "\n"}


def _segments(cmd):
    """Lex a shell command into (words, redirect_targets) per simple command.

    Quote- and heredoc-aware, which matters: an agent's `cd /testbed && python -c "...
    if M.rows >= r ..."` must not be read as a redirection into /testbed just because the
    payload contains '>='.  Everything inside quotes, and every heredoc body, is opaque.
    """
    words, redirs = [], []
    i, n = 0, len(cmd)
    pending = None                      # '>' or '>>' waiting for its target
    while i < n:
        c = cmd[i]
        if c == "\\" and i + 1 < n:     # line continuation / escaped char
            i += 2
            continue
        if c in " \t":
            i += 1
            continue
        if c in ";\n" or (c == "&" and cmd[i:i + 2] != "&&"):
            if words or redirs:
                yield words, redirs
            words, redirs, pending = [], [], None
            i += 1
            continue
        if cmd[i:i + 2] in ("&&", "||"):
            if words or redirs:
                yield words, redirs
            words, redirs, pending = [], [], None
            i += 2
            continue
        if c == "|":
            if words or redirs:
                yield words, redirs
            words, redirs, pending = [], [], None
            i += 1
            continue
        if cmd[i:i + 2] == "<<":         # heredoc: skip the whole body
            i += 3 if cmd[i:i + 3] == "<<-" else 2
            while i < n and cmd[i] in " \t":
                i += 1
            j = i
            while j < n and cmd[j] not in " \t\n;|&":
                j += 1
            delim = cmd[i:j].strip("'\"")
            i = j
            nl = cmd.find("\n", i)
            if nl == -1 or not delim:
                continue
            k = nl + 1
            while k < n:
                e = cmd.find("\n", k)
                line = cmd[k:(e if e != -1 else n)]
                if line.strip() == delim:
                    i = (e + 1) if e != -1 else n
                    break
                if e == -1:
                    i = n
                    break
                k = e + 1
            else:
                i = n
            continue
        if c == "<":
            i += 1
            continue
        if cmd[i:i + 2] == ">>":
            pending = ">>"
            i += 2
            continue
        if c == ">":
            # '>' immediately followed by '=' is a comparison, not a redirection
            if cmd[i:i + 2] == ">=":
                i += 2
                continue
            pending = ">"
            i += 1
            continue
        # --- a word, with quotes kept opaque ---
        buf, q = [], None
        while i < n:
            ch = cmd[i]
            if q:
                buf.append(ch)
                if ch == q:
                    q = None
                elif ch == "\\" and q == '"' and i + 1 < n:
                    i += 1
                    buf.append(cmd[i])
                i += 1
                continue
            if ch in "'\"":
                q = ch
                buf.append(ch)
                i += 1
                continue
            if ch in " \t\n;|&<>" or cmd[i:i + 2] in ("&&", "||"):
                break
            if ch == "\\" and i + 1 < n:
                i += 1
                buf.append(cmd[i])
                i += 1
                continue
            buf.append(ch)
            i += 1
        w = "".join(buf)
        if not w:
            i += 1
            continue
        if pending:
            redirs.append(w)
            pending = None
        else:
            words.append(w)
    if words or redirs:
        yield words, redirs


def _under_repo(path):
    p = path.strip("'\"")
    return p == REPO or p.startswith(REPO + "/")


def _is_relative_file(path):
    """A bare relative target -- meaningful only together with a /testbed cwd."""
    p = path.strip("'\"")
    return bool(p) and not p.startswith(("/", "-", "&", "$", "(", "=")) and p != "/dev/null"


def bash_writes_repo(cmd):
    """True iff this shell command writes somewhere under /testbed."""
    if not cmd:
        return False
    cwd = None                                   # unknown until a `cd` says otherwise
    for toks, redirs in _segments(cmd):
        # --- track cwd -----------------------------------------------------
        if toks and toks[0] == "cd":
            cwd = toks[1].strip("'\"") if len(toks) > 1 else None
            continue
        in_repo = cwd is not None and _under_repo(cwd)
        # --- redirections into the repo ------------------------------------
        for target in redirs:
            if _under_repo(target) or (in_repo and _is_relative_file(target)):
                return True
        if not toks:
            continue
        # env-var prefixes (FOO=bar cmd ...) are not the command itself
        i = 0
        while i < len(toks) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", toks[i]):
            i += 1
        if i >= len(toks):
            continue
        argv = toks[i:]
        prog = os.path.basename(argv[0].strip("'\""))
        args = argv[1:]
        # --- git, by subcommand --------------------------------------------
        if prog == "git":
            repo_dir = cwd
            sub = None
            j = 0
            while j < len(args):
                a = args[j].strip("'\"")
                if a == "-C" and j + 1 < len(args):
                    repo_dir = args[j + 1].strip("'\"")
                    j += 2
                    continue
                if a.startswith("-"):
                    j += 1
                    continue
                sub = a
                rest = [x.strip("'\"") for x in args[j + 1:]]
                break
            if sub in GIT_WRITE and repo_dir is not None and _under_repo(repo_dir):
                verbs = [r for r in rest if not r.startswith("-")]
                if sub == "stash" and verbs and verbs[0] in GIT_STASH_READONLY:
                    continue          # `git stash list` / `git stash show`
                return True
            continue
        # --- in-place editors ----------------------------------------------
        if prog in ("sed", "perl") and any(a.startswith("-i") for a in args):
            targets = [a for a in args if not a.startswith("-")]
            # for sed the first non-flag arg is the script, the rest are files
            files = targets[1:] if prog == "sed" and len(targets) > 1 else targets
            if any(_under_repo(f) for f in files) or (in_repo and any(_is_relative_file(f) for f in files)):
                return True
            continue
        if prog == "patch":
            if any(_under_repo(a) for a in args) or in_repo:
                return True
            continue
        if prog == "tee":
            files = [a for a in args if not a.startswith("-")]
            if any(_under_repo(f) for f in files) or (in_repo and any(_is_relative_file(f) for f in files)):
                return True
            continue
        if prog in FILE_WRITERS:
            files = [a for a in args if not a.startswith("-")]
            if any(_under_repo(f) for f in files) or (in_repo and any(_is_relative_file(f) for f in files)):
                return True
            continue
    return False


def scan_log(path):
    """Round-level features of one opencode.txt.  Single-round files only for
    the trigger reconstruction; multi-round files still give useful totals."""
    last_reason = None
    reasons = []
    edits = bash_writes = steps = 0
    with open(path, errors="replace") as fh:
        for line in fh:
            if '"type":"step_finish"' in line:
                steps += 1
                m = RE_REASON.search(line)
                if m:
                    last_reason = m.group(1)
                    reasons.append(m.group(1))
            elif '"type":"tool_use"' in line:
                m = RE_TOOL.search(line)
                if not m or '"status":"completed"' not in line:
                    continue
                if m.group(1) in EDIT_TOOLS or m.group(1) == "bash":
                    # parse for real: the command has to be JSON-unescaped before it can
                    # be tokenised, and the cheap regex cannot tell part.tool from a
                    # "tool" key echoed inside a tool's own output.
                    try:
                        p = json.loads(line)["part"]
                    except Exception:
                        continue
                    st = p.get("state") or {}
                    if st.get("status") != "completed":
                        continue
                    tool = (p.get("tool") or "").lower()
                    if tool in EDIT_TOOLS:
                        edits += 1
                    elif tool == "bash":
                        cmd = (st.get("input") or {}).get("command")
                        if isinstance(cmd, str) and bash_writes_repo(cmd):
                            bash_writes += 1
    return dict(steps=steps, last_reason=last_reason, n_length=reasons.count("length"),
                edits=edits, bash_writes=bash_writes)


def read_autocontinue(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"round=1 modified_before=(\d+) cut_before=(\d+) rc[_a-z]*=(\d+)(.*)", line)
            if m:
                out.update(modified_before=int(m.group(1)), cut_before=int(m.group(2)),
                           skipped="skipped" in m.group(4))
            m = re.match(r"rounds_used=(\d+) modified_final=(\d+) cut_final=(\d+)", line)
            if m:
                out.update(rounds_used=int(m.group(1)), modified_final=int(m.group(2)),
                           cut_final=int(m.group(3)))
    return out


def trial_task(name):
    return name.rsplit("__", 1)[0]


def collect(root, job):
    rows = []
    for d in sorted(glob.glob(os.path.join(root, job, "*"))):
        rj = os.path.join(d, "result.json")
        if not os.path.isdir(d) or not os.path.exists(rj):
            continue
        r = json.load(open(rj))
        log = os.path.join(d, "agent", "opencode.txt")
        feat = scan_log(log) if os.path.exists(log) else None
        ac = os.path.join(d, "agent", "autocontinue.txt")
        row = dict(
            trial=os.path.basename(d), task=trial_task(os.path.basename(d)),
            reward=((r.get("verifier_result") or {}).get("rewards") or {}).get("reward"),
            exc=(r.get("exception_info") or {}).get("exception_type"),
            feat=feat, logged=read_autocontinue(ac) if os.path.exists(ac) else None,
        )
        rows.append(row)
    return rows


def reconstruct(feat):
    """Round-1 trigger state from a SINGLE-round log.  Returns (triggered, rule)."""
    if feat is None:
        return None, None
    if feat["last_reason"] == "length":
        return True, "a"
    if feat["edits"] == 0 and feat["bash_writes"] == 0:
        return True, "b"
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_root")
    ap.add_argument("--stock", default="oc-qwen38-B-stock")
    ap.add_argument("--ac", default="oc-qwen38-B-ac")
    a = ap.parse_args()

    stock = collect(a.jobs_root, a.stock)
    acs = collect(a.jobs_root, a.ac)

    # ---- validation: ac trials that never continued are single-round files ----
    ok = bad = 0
    misses = []
    for r in acs:
        L = r["logged"]
        if not L or L.get("rounds_used") != 0 or r["feat"] is None:
            continue
        got, rule = reconstruct(r["feat"])
        want = bool(L.get("cut_before")) or L.get("modified_before") == 0
        if got == want:
            ok += 1
        else:
            bad += 1
            misses.append((r["trial"], got, rule, L, r["feat"]))
    print("== validation on ac trials with rounds_used=0 (whole log == round 1)")
    print("   agree %d / %d" % (ok, ok + bad))
    for m in misses:
        print("   MISS %-40s recon=%s/%s logged=%s feat=%s" % m)
    print()

    # ---- trigger rate per arm ----
    # The round-1 trigger state is a property of the LOG, not of the score: it is what the
    # wrapper saw before it decided whether to continue.  So every trial that has a log
    # counts, including the ones carrying an exception.  (An earlier version skipped any
    # trial with an exception, which silently dropped AgentTimeoutError and
    # NonZeroAgentExitCodeError -- both of which the protocol keeps SCORED -- and
    # disagreed with b38_paired.py on the same quantity.)  Unscored trials are still
    # counted here; their reward is reported as None in the per-task table below.
    print("== round-1 trigger state  (all trials with a log; exceptions are NOT dropped)")
    for label, rows, src in (("ac   (logged)", acs, "log"), ("stock(recon) ", stock, "recon")):
        trig = tot = 0
        by_rule = {"a": 0, "b": 0}
        for r in rows:
            if src == "log":
                L = r["logged"]
                if not L:
                    continue
                t = bool(L.get("cut_before")) or L.get("modified_before") == 0
                rule = "a" if L.get("cut_before") else ("b" if L.get("modified_before") == 0 else None)
            else:
                t, rule = reconstruct(r["feat"])
                if t is None:
                    continue
            tot += 1
            if t:
                trig += 1
                by_rule[rule] += 1
        pct = 100.0 * trig / tot if tot else 0.0
        print("   %s  %3d/%3d = %5.1f%%   rule(a)=%d rule(b)=%d" % (label, trig, tot, pct, by_rule["a"], by_rule["b"]))
    print()

    # ---- per-task table ----
    print("== per-task (attempt-level; task, arm, reward, trigger)")
    tasks = sorted({r["task"] for r in stock + acs})
    for t in tasks:
        s = [r for r in stock if r["task"] == t]
        c = [r for r in acs if r["task"] == t]
        def fmt(rows, src):
            out = []
            for r in rows:
                if src == "log" and r["logged"]:
                    L = r["logged"]
                    tg = "a" if L.get("cut_before") else ("b" if L.get("modified_before") == 0 else "-")
                else:
                    _, rule = reconstruct(r["feat"])
                    tg = rule or "-"
                # reward is None when the trial is unscored (no reward file); say so
                # rather than printing it as a 0, and mark any exception with a '!'.
                rw = "?" if r["reward"] is None else ("1" if r["reward"] else "0")
                out.append("%s%s/%s" % (rw, "!" if r["exc"] else "", tg))
            return ",".join(out) or "-"
        print("   %-34s stock=%-12s ac=%-12s" % (t, fmt(s, "recon"), fmt(c, "log")))


if __name__ == "__main__":
    main()
