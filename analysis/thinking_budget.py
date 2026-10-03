#!/usr/bin/env python3
"""How much of Qwen3.8's output budget goes to reasoning?

OpenCode's agent/opencode.txt reports per-step usage with a `reasoning` field,
but that field is always 0: the provider is @ai-sdk/openai-compatible talking to
vLLM, and vLLM's usage block counts all generated tokens in `completion_tokens`
without separating the think block.  The reasoning TEXT is preserved though --
one {"type":"reasoning"} event per assistant message -- so the split can be
recovered by tokenising that text with the model's own tokenizer and dividing by
the step's exact `tokens.output`.

  reasoning_share(step) = tok(reasoning_text) / tokens.output

This slightly UNDERSTATES the share: the raw generation also contains the
<think></think> delimiters and the tool-call scaffolding, which are counted in
tokens.output but are not in any part's text.  The residual check below reports
how much of tokens.output the three attributable parts (reasoning, assistant
text, tool-call arguments) fail to account for; that gap is the scaffolding.

Tool-call argument tokens are approximate -- they are re-serialised from the
parsed input dict, so key order and escaping need not match what the model
emitted.  They are used only for the residual check, never for the headline.

Usage:  thinking_budget.py <jobs_root> <job> [<job> ...] [--tokenizer PATH]
"""
import argparse, glob, json, math, os, statistics, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b38_paired import t_crit  # noqa: E402  (t(n-1) quantile; the paper uses t, not z)

# Tokeniser.  Pass --tokenizer PATH or set $HARNESS_QWEN_TOKENIZER to a Qwen tokenizer.json.
# The serving model's own tokenizer (Qwen3.8-27B) and a Qwen3.5-9B tokenizer of the same
# family agree to the token on this corpus (the 39 length-cut steps come back at 1247998 of
# 1248000 either way, and every share in thinking_budget.txt is unchanged).
TOKENIZER_CANDIDATES = [
    os.environ.get("HARNESS_QWEN_TOKENIZER"),
]
DEFAULT_TOK = None          # resolved lazily by resolve_tokenizer()
CUT_OUTPUT = 32000  # limit.output is 32768; OpenCode cuts at 32000


def resolve_tokenizer(path=None):
    """First readable candidate among --tokenizer and $HARNESS_QWEN_TOKENIZER."""
    for c in ([path] if path else []) + TOKENIZER_CANDIDATES:
        if c and os.path.exists(c):
            return c
    raise SystemExit(
        "no tokenizer found; tried:\n  " +
        "\n  ".join(c for c in ([path] if path else []) + TOKENIZER_CANDIDATES if c) +
        "\nset --tokenizer PATH or $HARNESS_QWEN_TOKENIZER")


def load_tokenizer(path=None):
    from tokenizers import Tokenizer
    return Tokenizer.from_file(resolve_tokenizer(path))


def scan(path):
    """-> per-message dicts with char counts and exact output tokens."""
    msgs = defaultdict(lambda: dict(reason="", text="", tool_args="", tool_calls=0, out_tok=None, reason_fin=None))
    order = []
    for line in open(path, errors="replace"):
        try:
            d = json.loads(line)
        except Exception:
            continue
        t = d.get("type")
        p = d.get("part") or {}
        mid = p.get("messageID")
        if mid is None:
            continue
        if mid not in msgs:
            order.append(mid)
        m = msgs[mid]
        if t == "reasoning":
            m["reason"] += p.get("text") or ""
        elif t == "text":
            m["text"] += p.get("text") or ""
        elif t == "tool_use":
            st = p.get("state") or {}
            m["tool_args"] += json.dumps(st.get("input") or {}, ensure_ascii=False)
            m["tool_calls"] += 1
        elif t == "step_finish":
            m["out_tok"] = (p.get("tokens") or {}).get("output")
            m["reason_fin"] = p.get("reason")
    return [msgs[m] for m in order]


def scan_cc(path):
    """Claude Code's agent/claude-code.txt: one JSON event per line.  Unlike
    OpenCode it emits ONE assistant event per content block, with
    usage.output_tokens = 0 and stop_reason = null on every one -- only the final
    `result` event carries real totals.  So CC supports an exact TRIAL-level
    reasoning share but no per-step distribution.  (Its
    usage.output_tokens_details.thinking_tokens is 0 for the same reason
    OpenCode's reasoning field is: vLLM does not split the think block out.)
    Returns a single aggregated pseudo-row for the whole trial."""
    agg = dict(reason="", text="", tool_args="", tool_calls=0, out_tok=None, reason_fin=None)
    total = None
    for line in open(path, errors="replace"):
        try:
            o = json.loads(line)
        except Exception:
            continue
        if o.get("type") == "result":
            total = (o.get("usage") or {}).get("output_tokens")
            continue
        if o.get("type") != "assistant":
            continue
        for part in (o.get("message") or {}).get("content") or []:
            t = part.get("type")
            if t == "thinking":
                agg["reason"] += part.get("thinking") or ""
            elif t == "text":
                agg["text"] += part.get("text") or ""
            elif t == "tool_use":
                agg["tool_args"] += json.dumps(part.get("input") or {}, ensure_ascii=False)
                agg["tool_calls"] += 1
    agg["out_tok"] = total
    return [agg] if total else []


def tok_len(tk, s):
    if not s:
        return 0
    return len(tk.encode(s, add_special_tokens=False).ids)


def analyse_trial(tk, d):
    oc = os.path.join(d, "agent", "opencode.txt")
    cc = os.path.join(d, "agent", "claude-code.txt")
    per_step = True
    if os.path.exists(oc):
        msgs = scan(oc)
    elif os.path.exists(cc):
        msgs = scan_cc(cc)
        per_step = False  # CC has no usable per-message denominator
    else:
        return None
    if not msgs:
        return None
    rows = []
    for m in msgs:
        if m["out_tok"] is None:
            continue
        rows.append(dict(
            out=m["out_tok"],
            rtok=tok_len(tk, m["reason"]),
            ttok=tok_len(tk, m["text"]),
            atok=tok_len(tk, m["tool_args"]),
            calls=m["tool_calls"],
            fin=m["reason_fin"],
        ))
    if not rows:
        return None
    rj = os.path.join(d, "result.json")
    rew = exc = None
    if os.path.exists(rj):
        try:
            r = json.load(open(rj))
            rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
            exc = (r.get("exception_info") or {}).get("exception_type")
        except Exception:
            pass
    return dict(trial=os.path.basename(d), rows=rows, reward=rew, exc=exc, per_step=per_step)


def pct(x, n):
    return 100.0 * x / n if n else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_root")
    ap.add_argument("jobs", nargs="+")
    ap.add_argument("--tokenizer", default=DEFAULT_TOK)
    ap.add_argument("--per-trial", action="store_true")
    ap.add_argument("--pair", nargs=2, metavar=("JOB_A", "JOB_B"),
                    help="paired per-task comparison of two jobs over their common tasks")
    a = ap.parse_args()
    tk = load_tokenizer(a.tokenizer)

    if a.pair:
        paired(tk, a.jobs_root, a.pair[0], a.pair[1])

    for job in a.jobs:
        trials = []
        for d in sorted(glob.glob(os.path.join(a.jobs_root, job, "*"))):
            if not os.path.isdir(d):
                continue
            t = analyse_trial(tk, d)
            if t:
                trials.append(t)
        if not trials:
            print("== %s: no parsable trials" % job)
            continue

        allrows = [r for t in trials for r in t["rows"]]
        out = sum(r["out"] for r in allrows)
        rt = sum(r["rtok"] for r in allrows)
        tt = sum(r["ttok"] for r in allrows)
        at = sum(r["atok"] for r in allrows)

        print("== %s   trials %d   assistant steps %d" % (job, len(trials), len(allrows)))
        print("   output tokens (exact, from step usage): %d" % out)
        print("   reasoning      %9d  %5.1f%%   <- the thinking budget" % (rt, pct(rt, out)))
        print("   assistant text %9d  %5.1f%%" % (tt, pct(tt, out)))
        print("   tool arguments %9d  %5.1f%%  (approx, re-serialised)" % (at, pct(at, out)))
        print("   unattributed   %9d  %5.1f%%  (think tags + tool-call scaffolding)"
              % (out - rt - tt - at, pct(out - rt - tt - at, out)))

        # per-trial share, so one runaway trial cannot carry the mean
        shares = [pct(sum(r["rtok"] for r in t["rows"]), sum(r["out"] for r in t["rows"]))
                  for t in trials if sum(r["out"] for r in t["rows"])]
        shares.sort()
        print("   per-trial reasoning share: median %.1f%%  IQR [%.1f, %.1f]  min %.1f  max %.1f"
              % (statistics.median(shares), shares[len(shares) // 4], shares[3 * len(shares) // 4],
                 shares[0], shares[-1]))

        per_step = all(t["per_step"] for t in trials)
        if not per_step:
            print("   (per-step distribution unavailable: this harness reports no per-message")
            print("    output-token count; the share above is trial-level and exact)")
            print()
            continue

        # per-step reasoning length: this is what actually eats the 32k window
        rs = sorted(r["rtok"] for r in allrows)
        n = len(rs)
        print("   per-step reasoning tokens: median %d  p75 %d  p90 %d  p99 %d  max %d"
              % (rs[n // 2], rs[int(n * .75)], rs[int(n * .90)], rs[int(n * .99)], rs[-1]))
        nothink = sum(1 for x in rs if x == 0)
        print("   steps with zero reasoning: %d (%.1f%%)" % (nothink, pct(nothink, n)))

        # concentration: is it thinking a lot every step, or falling into rare
        # very long thinks?  The second is what a budget cap would actually fix.
        print("   reasoning-token concentration (share of ALL reasoning tokens):")
        for thr in (2000, 8000, 16000):
            hi = [r for r in allrows if r["rtok"] > thr]
            share = pct(sum(r["rtok"] for r in hi), rt)
            print("      steps with >%5d reasoning tok: %5d (%4.1f%% of steps) carry %5.1f%% of reasoning"
                  % (thr, len(hi), pct(len(hi), len(allrows)), share))
        atrisk = sum(1 for t in trials if any(r["rtok"] > 8000 for r in t["rows"]))
        print("      trials with at least one >8000-token think: %d/%d (%.1f%%)"
              % (atrisk, len(trials), pct(atrisk, len(trials))))

        # steps that ended on a length cut -- where did the 32k go?
        cut = [r for r in allrows if r["fin"] in ("length", "max_tokens")]
        if cut:
            co = sum(r["out"] for r in cut); cr = sum(r["rtok"] for r in cut)
            print("   length-cut steps: %d   output %d   reasoning %d = %.1f%% of the cut budget"
                  % (len(cut), co, cr, pct(cr, co)))

        # does thinking correlate with success?
        for label, sel in (("pass", lambda t: t["reward"]), ("fail", lambda t: not t["reward"])):
            sub = [t for t in trials if t["exc"] is None and sel(t)]
            if not sub:
                continue
            so = sum(r["out"] for t in sub for r in t["rows"])
            sr = sum(r["rtok"] for t in sub for r in t["rows"])
            steps = sum(len(t["rows"]) for t in sub)
            print("   %-4s trials %3d: %5.1f%% reasoning, %6.0f output tok/step, %5.1f steps/trial"
                  % (label, len(sub), pct(sr, so), so / steps if steps else 0, steps / len(sub)))

        if a.per_trial:
            print("   --- per trial")
            for t in sorted(trials, key=lambda t: -pct(sum(r["rtok"] for r in t["rows"]),
                                                       max(1, sum(r["out"] for r in t["rows"])))):
                o = sum(r["out"] for r in t["rows"]); r_ = sum(r["rtok"] for r in t["rows"])
                print("   %-46s %5.1f%%  out %7d  steps %3d  rew %s"
                      % (t["trial"][:46], pct(r_, o), o, len(t["rows"]), t["reward"]))
        print()


def paired(tk, root, ja, jb):
    """Same model, same tasks, different harness: is the reasoning share a model
    property or a harness property?  Volume and share are reported separately."""
    def collect(job):
        out = {}
        for d in sorted(glob.glob(os.path.join(root, job, "*"))):
            t = analyse_trial(tk, d)
            if not t:
                continue
            o = sum(r["out"] for r in t["rows"])
            if o:
                out[os.path.basename(d).rsplit("__", 1)[0]] = (o, sum(r["rtok"] for r in t["rows"]))
        return out

    A, B = collect(ja), collect(jb)
    common = sorted(set(A) & set(B))
    print("== paired: %s  vs  %s   common tasks %d" % (ja, jb, len(common)))
    for name, D in ((ja, A), (jb, B)):
        o = sum(D[t][0] for t in common)
        r = sum(D[t][1] for t in common)
        sh = sorted(100.0 * D[t][1] / D[t][0] for t in common)
        print("   %-24s output %10d (%6.0f/task)  reasoning %5.1f%%  per-trial median share %4.1f%%"
              % (name, o, o / len(common), pct(r, o), statistics.median(sh)))
    # The paper uses t(n-1) on the mean log-ratio for every efficiency interval; these two
    # were once computed with z = 1.96.  At n = 446 it is a third-decimal change, but t(n-1)
    # is the stated rule.
    for label, lg in (("output-token volume",
                       [math.log(B[t][0] / A[t][0]) for t in common if A[t][0] and B[t][0]]),
                      ("reasoning SHARE",
                       [math.log((B[t][1] / B[t][0]) / (A[t][1] / A[t][0]))
                        for t in common if A[t][1] and B[t][1]])):
        m, se = statistics.mean(lg), statistics.stdev(lg) / math.sqrt(len(lg))
        tc = t_crit(len(lg) - 1)
        print("   %-20s B/A = %.3fx  [%.3f, %.3f]  n=%d"
              % (label, math.exp(m), math.exp(m - tc * se), math.exp(m + tc * se), len(lg)))
    print()


if __name__ == "__main__":
    main()
