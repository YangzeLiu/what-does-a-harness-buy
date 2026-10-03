#!/usr/bin/env python3
"""Appendix Table A4: where the per-step input budget goes.

Replaces the dropped Listing 1.  The point of the table: with the model held fixed
(GLM-5.3-Flash, hard45, 45 tasks per arm), a single tool return is the same size in all
three harnesses -- Claude Code's 2x larger per-step input is fixed prompt cost, a carried
transcript (its own reasoning is re-sent with every request) and more steps.

  (a) tool-return size per call, in characters, over every tool call of the arm
  (b) the per-step input budget in tokens: fixed cost, what one run appends to the
      conversation, steps, compactions, and the reconstruction against provider usage.

Data sources (read-only; nothing here is re-derived from data/trials.csv)
  * ATIF trajectory   results/raw/jobs/<job>/<task>__<id>/agent/trajectory.json
      - faithful for mini-SWE-agent and OpenCode (verified against their raw logs)
      - for Claude Code the converter APPENDS the SDK's `tool_use_result` metadata blob to
        every observation (it re-serialises the whole file / the Edit old+new string), which
        roughly doubles the stored observation.  That blob was never sent to the model, so
        CC observation sizes come from the raw stream log instead.
  * raw agent log     agent/claude-code.txt (CC, stream-json: the ground truth for what went
        into the API request, and the only place the auto-compaction records live).
  Token counts always come from the trajectory `metrics`, i.e. usage reported by the provider.

Run: python3 analysis/tab_a4_budget.py
  -> paper/latex/tables/tableA4_budget.tex
  -> paper/latex/numbers_a4.tex        (\\pn macros, keys budget.<arm>.*)
Stdlib + analysis/figstyle.py (labels only).
"""
import glob
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import figstyle as fs                      # noqa: E402  harness order / labels only

ROOT = os.path.normpath(os.path.join(HERE, ".."))
JOBS = os.environ.get("HARNESS_JOBS_ROOT", os.path.join(ROOT, "results", "raw", "jobs"))
TDIR = os.path.join(ROOT, "paper", "latex", "tables")

# arm -> (harness key in figstyle, job).  Order is the paper's: CC / mini / OC.
ARMS = [("CC", "claude-code", "cc-glm53f-hard45-nw"),
        ("mini", "mini-swe-agent", "mini-glm53f-hard45b-nw"),
        ("OC", "opencode", "oc-glm53f-hard45-nw-rg")]
ISSUE_CPT = 3.7          # chars per token for the issue text, used only for the fixed-cost split
OC_COMPACT_TOK = 98000 - 20000   # OpenCode compacts at limit.input minus its 20k reserve
TOP_TOOL_MIN = 0.05      # a tool is named in panel (a) if it carries >=5% of the arm's obs chars
TOP_TOOL_MAX = 3


# ----------------------------------------------------------------------------- formatting
def num(x, d=0):
    """1237 -> '1{,}237' (LaTeX-safe thousands separator)."""
    s = f"{x:,.{d}f}"
    return s.replace(",", "{,}")


def sgn(v, d=1):
    return f"{v:+.{d}f}".replace("-", "$-$")


def kt(x):
    """tokens in thousands, one decimal."""
    return f"{x/1e3:.1f}"


def pctl(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((len(xs) - 1) * p)))]


# ----------------------------------------------------------------------------- trajectory helpers
def steps(t):
    return t.get("steps") or t.get("trajectory") or []


def tool_calls(step):
    out = []
    for tc in step.get("tool_calls") or []:
        fn = tc.get("function") or {}
        name = tc.get("function_name") or fn.get("name") or tc.get("name") or ""
        args = tc.get("arguments") or fn.get("arguments") or {}
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {"raw": args}
        out.append((tc.get("tool_call_id") or tc.get("id") or "", name, args))
    return out


def obs_results(step):
    o = step.get("observation") or {}
    res = []
    for r in o.get("results") or []:
        c = r.get("content")
        if isinstance(c, list):
            c = "\n".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
        res.append((r.get("source_call_id") or "", c or ""))
    return res


def template_const(msgs):
    """Chars of the harness's own wrapper around the issue statement in the first user turn.

    mini-SWE-agent renders its `instance_template` (a constant text with the issue substituted
    into it) as its one user turn, so the constant part is exactly what every rendered turn has
    in common: the longest common prefix plus the longest common suffix over the arm's trials.
    Claude Code and OpenCode pass the issue verbatim and this returns ~0.
    """
    if len(msgs) < 2:
        return 0
    a = msgs[0]
    pre = min(len(os.path.commonprefix([a, m])) for m in msgs[1:])
    suf = min(len(os.path.commonprefix([a[::-1], m[::-1]])) for m in msgs[1:])
    return pre + suf


def trial_dirs(job):
    return sorted(glob.glob(os.path.join(JOBS, job, "*__*")))


def load_traj(d):
    f = os.path.join(d, "agent", "trajectory.json")
    if not os.path.exists(f):
        return None
    try:
        return json.load(open(f))
    except Exception as e:
        print("SKIP", d, e, file=sys.stderr)
        return None


# ----------------------------------------------------------------------------- CC raw stream log
def cc_raw(path):
    """Ground-truth decomposition of one Claude Code run from its stream-json log.

    Returns the observations actually sent back to the model (tool_result blocks), the
    assistant's own text and thinking, the tool-call arguments, and the auto-compaction
    boundaries the SDK records.
    """
    tool_by_id = {}
    obs = []                       # (tool name, content chars)
    text = think = args_chars = 0
    compacts = []
    for line in open(path, errors="replace"):
        if '"thinking_tokens"' in line:     # ~88k progress lines per run, no payload
            continue
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            o = json.loads(line)
        except Exception:
            continue
        ty = o.get("type")
        if ty == "system":
            if o.get("subtype") == "compact_boundary":
                cm = o.get("compact_metadata") or {}
                compacts.append((cm.get("trigger"), cm.get("pre_tokens"), cm.get("post_tokens")))
            continue
        if ty == "assistant":
            for c in (o.get("message") or {}).get("content") or []:
                t = c.get("type")
                if t == "text":
                    text += len(c.get("text") or "")
                elif t == "thinking":
                    think += len(c.get("thinking") or "")
                elif t == "tool_use":
                    tool_by_id[c.get("id")] = c.get("name")
                    args_chars += len(json.dumps(c.get("input"), ensure_ascii=False))
        elif ty == "user":
            for c in (o.get("message") or {}).get("content") or []:
                if c.get("type") == "tool_result":
                    txt = c.get("content")
                    if isinstance(txt, list):
                        txt = "".join(x.get("text", "") for x in txt if isinstance(x, dict))
                    obs.append((tool_by_id.get(c.get("tool_use_id"), "?"), len(txt or "")))
    return dict(obs=obs, text=text, think=think, args=args_chars, compacts=compacts)


# ----------------------------------------------------------------------------- per-arm analysis
def analyse(arm, job):
    """One arm: observation sizes, per-run transcript, per-step tokens, compactions."""
    obs_all = []                  # (tool, chars) over every trial
    per_traj = []                 # per trial: obs / text+think / args chars, steps
    cum_means = []                # per trial: mean over steps of the accumulated transcript
    prompt_tok = []               # provider-reported input tokens, every step
    first_prompt_tok = []
    first_user_msgs = []          # the FIRST user turn of every trial (the issue statement)
    compact_pre = []              # prompt tokens at each auto-compaction (CC only)
    d_tok = d_ch = 0              # step-to-step deltas for the empirical chars-per-token fit
    compact_events = compact_runs = 0

    for d in trial_dirs(job):
        t = load_traj(d)
        if t is None:
            continue
        S = steps(t)
        ac = ((t.get("agent") or {}).get("extra") or {}).get("agent_config") or {}
        sysc = len(ac.get("system_template") or "") + len(ac.get("instance_template") or "") if ac else 0
        pre_chars = sum(len(s.get("message") or "") for s in S if s.get("source") in ("system", "user"))
        # the issue statement is the FIRST user turn only: later user turns are the resume
        # messages Claude Code injects after an auto-compaction, not part of the task text
        fu = [s.get("message") or "" for s in S if s.get("source") == "user"]
        if fu:
            first_user_msgs.append(fu[0])

        raw = raw_sizes = None
        if arm == "CC":
            lg = os.path.join(d, "agent", "claude-code.txt")
            if os.path.exists(lg):
                raw = cc_raw(lg)
                raw_sizes = [n for _, n in raw["obs"]]
                obs_all.extend(raw["obs"])
                compact_events += len(raw["compacts"])
                compact_runs += 1 if raw["compacts"] else 0
                compact_pre += [pre for _, pre, _ in raw["compacts"] if pre]

        tot_obs = tot_txt = tot_args = 0
        if raw is not None:
            tot_obs = sum(raw_sizes)
            tot_txt = raw["text"] + raw["think"]
            tot_args = raw["args"]

        # step walk over the trajectory: step count and tokens always come from here
        idx = rp = 0
        cum = sysc + pre_chars
        cums, pairs = [], []
        for i, s in enumerate(S):
            if s.get("source") != "agent":
                continue
            idx += 1
            cums.append(cum)
            txt = (s.get("message") or "") + (s.get("reasoning_content") or "")
            calls = tool_calls(s)
            ac_chars = sum(len(json.dumps(a, ensure_ascii=False)) for _, _, a in calls)
            res = obs_results(s)
            if not res and i + 1 < len(S):
                res = obs_results(S[i + 1])
            byid = {cid: nm for cid, nm, _ in calls}
            ostep = 0
            for j, (scid, content) in enumerate(res):
                nm = byid.get(scid) or (calls[j][1] if j < len(calls) else "(unmatched)")
                if raw_sizes is not None:               # CC: true size from the stream log
                    ostep += raw_sizes[rp] if rp < len(raw_sizes) else len(content)
                    rp += 1
                else:
                    n = len(content)
                    ostep += n
                    obs_all.append((nm, n))
            if raw is None:
                tot_obs += ostep
                tot_txt += len(txt)
                tot_args += ac_chars
            cum += len(txt) + ac_chars + ostep
            pt = (s.get("metrics") or {}).get("prompt_tokens")
            if pt:
                prompt_tok.append(pt)
                pairs.append((cums[-1], pt))
        agent_steps = [s for s in S if s.get("source") == "agent"]
        if agent_steps:
            m0 = agent_steps[0].get("metrics") or {}
            if m0.get("prompt_tokens"):
                first_prompt_tok.append(m0["prompt_tokens"])
        if cums:
            cum_means.append(st.mean(cums))
        for (c0, t0), (c1, t1) in zip(pairs, pairs[1:]):
            # only forward growth, and skip the jumps a context reset produces
            if t1 > t0 and c1 > c0 and (t1 - t0) < 40000:
                d_tok += t1 - t0
                d_ch += c1 - c0
        per_traj.append(dict(obs=tot_obs, txt=tot_txt, args=tot_args, steps=idx))

    R = dict(arm=arm, job=job, trials=len(per_traj), obs_all=obs_all)
    ocs = [n for _, n in obs_all]
    R["obs_n"] = len(ocs)
    R["obs_mean"] = st.mean(ocs)
    R["obs_median"] = st.median(ocs)
    R["obs_p90"] = pctl(ocs, .9)
    R["obs_total"] = sum(ocs)
    R["steps"] = st.mean([x["steps"] for x in per_traj])
    R["obs_run"] = st.mean([x["obs"] for x in per_traj])
    R["txt_run"] = st.mean([x["txt"] for x in per_traj])
    R["args_run"] = st.mean([x["args"] for x in per_traj])
    R["cpt"] = d_ch / d_tok if d_tok else ISSUE_CPT
    R["first_tok"] = st.mean(first_prompt_tok) if first_prompt_tok else 0
    # issue statement: the first user turn, minus the harness's own constant wrapper if it has
    # one (mini-SWE-agent's instance template; > 200 chars of shared text is the test).  The
    # remainder is the same issue text in all three arms and is checked to be so.
    const = template_const(first_user_msgs)
    R["tmpl_const"] = const if const > 200 else 0
    R["first_user_mean"] = st.mean(len(m) for m in first_user_msgs) if first_user_msgs else 0
    R["issue_chars"] = R["first_user_mean"] - R["tmpl_const"]
    R["measured"] = st.mean(prompt_tok)
    R["recon"] = R["first_tok"] + (st.mean(cum_means) if cum_means else 0) / R["cpt"]
    R["compact_events"] = compact_events
    R["compact_runs"] = compact_runs
    R["compact_pre_median"] = st.median(compact_pre) if compact_pre else 0
    return R


def top_tools(R):
    """'Bash 68\\%, Read 30\\%' -- the tools carrying most of the observation chars."""
    by = {}
    for nm, n in R["obs_all"]:
        by[nm] = by.get(nm, 0) + n
    tot = R["obs_total"] or 1
    out = []
    for nm, s in sorted(by.items(), key=lambda kv: -kv[1])[:TOP_TOOL_MAX]:
        if s / tot < TOP_TOOL_MIN and out:
            break
        out.append((nm, s / tot))
    return out


# ----------------------------------------------------------------------------- the table
def build(res):
    macros = {}
    # The issue statement is the same text in every column (OpenCode and Claude Code receive it
    # verbatim, mini-SWE-agent inside its template), so one estimate is subtracted from all three
    # fixed costs; the median over the arms guards against a single arm's measurement.
    issue_chars = st.median([R["issue_chars"] for R in res])
    issue_tok = issue_chars / ISSUE_CPT
    for R in res:
        R["issue_tok"] = issue_tok
        R["fixed_harness"] = max(R["first_tok"] - issue_tok, 0)
    macros["budget.issue.chars"] = num(issue_chars)
    macros["budget.issue.tok"] = num(issue_tok)
    # the wrapper mini-SWE-agent puts around the issue, and where each harness compacts
    mini_const = max(R["first_user_mean"] - issue_chars for R in res)
    cc_pre_median = max(R["compact_pre_median"] for R in res)
    macros["budget.mini.template.chars"] = num(mini_const)
    macros["budget.cc.compact.pre.k"] = num(cc_pre_median / 1e3, 0)
    macros["budget.oc.compact.limit.k"] = num(OC_COMPACT_TOK / 1e3, 0)
    # both panels sit in a full-width minipage so panel (a) lines up with the wider panel (b)
    L = [r"{\footnotesize\setlength{\tabcolsep}{4pt}",
         r"\begin{minipage}{\linewidth}",
         r"\begin{tabular}{@{}lrrrrl@{}}",
         r"\multicolumn{6}{@{}l}{\emph{(a) Tool-return size per call (chars)}} \\",
         r"\toprule",
         r"Harness & calls & mean & median & p90 & Largest sources \\",
         r"\midrule"]
    for R in res:
        tops = top_tools(R)
        src = ", ".join(rf"\texttt{{{nm}}} {100*f:.0f}\%" for nm, f in tops)
        L.append(f"{fs.HARNESS_LONG[R['hkey']]} & {num(R['obs_n'])} & {num(R['obs_mean'])} & "
                 f"{num(R['obs_median'])} & {num(R['obs_p90'])} & {src} \\\\")
        a = R["key"]
        for kk, vv in (("obs.n", num(R["obs_n"])), ("obs.mean", num(R["obs_mean"])),
                       ("obs.median", num(R["obs_median"])), ("obs.p90", num(R["obs_p90"])),
                       ("obs.totalM", f"{R['obs_total']/1e6:.1f}"),
                       ("obs.top", tops[0][0]), ("obs.toppct", f"{100*tops[0][1]:.0f}")):
            macros[f"budget.{a}.{kk}"] = vv
    L += [r"\bottomrule", r"\end{tabular}", "", r"\vspace{5pt}", "",
          r"\begin{tabular}{@{}lrrr@{}}",
          r"\multicolumn{4}{@{}l}{\emph{(b) Per-step input budget (tokens, hard45 means over 45 runs)}} \\",
          r"\toprule",
          " & ".join([""] + [fs.HARNESS_LONG[R["hkey"]] for R in res]) + r" \\",
          r"\midrule"]

    def row(label, vals, rule=""):
        L.append(f"{label} & " + " & ".join(vals) + r" \\" + rule)

    row(r"Fixed cost per call$^{a}$", [num(R["first_tok"]) for R in res])
    row(r"\quad of which the issue statement", [num(R["issue_tok"]) for R in res])
    row(r"\quad of which harness preamble", [num(R["fixed_harness"]) for R in res])
    L.append(r"\addlinespace")
    row(r"Model text $+$ reasoning per run$^{b}$", [num(R["txt_run"] / R["cpt"]) for R in res])
    row(r"Tool returns (observations) per run$^{b,c}$", [num(R["obs_run"] / R["cpt"]) for R in res])
    row(r"Tool-call arguments per run$^{b}$", [num(R["args_run"] / R["cpt"]) for R in res])
    row(r"\quad chars per token (empirical)$^{b}$", [f"{R['cpt']:.2f}" for R in res])
    L.append(r"\addlinespace")
    row(r"Steps per run", [f"{R['steps']:.1f}" for R in res])
    row(r"Runs with an auto-compaction (of 45)$^{d}$", [str(R["compact_runs"]) for R in res])
    L.append(r"\addlinespace")
    row(r"Reconstructed input per step", [num(R["recon"]) for R in res])
    row(r"\textbf{Measured input per step}$^{e}$", [r"\textbf{" + num(R["measured"]) + "}" for R in res])
    for R in res:
        a = R["key"]
        for kk, vv in (("fixed", num(R["first_tok"])), ("fixed.harness", num(R["fixed_harness"])),
                       ("cpt", f"{R['cpt']:.2f}"), ("steps", f"{R['steps']:.1f}"),
                       ("text.run", num(R["txt_run"] / R["cpt"])), ("obs.run", num(R["obs_run"] / R["cpt"])),
                       ("args.run", num(R["args_run"] / R["cpt"])), ("compact", str(R["compact_runs"])),
                       ("perstep.recon", num(R["recon"])), ("perstep.measured", num(R["measured"])),
                       ("perstep.recon.k", kt(R["recon"])), ("perstep.measured.k", kt(R["measured"])),
                       ("perstep.gap", sgn(100 * (R["recon"] - R["measured"]) / R["measured"], 0) + r"\%"),
                       ("trials", str(R["trials"])), ("job", r"\texttt{" + R["job"].replace("-", r"-\allowbreak ") + "}")):
            macros[f"budget.{a}.{kk}"] = vv
    ratio = res[0]["measured"] / min(R["measured"] for R in res[1:])
    macros["budget.ratio.ccvsmin"] = f"{ratio:.1f}"
    macros["budget.obs.spread"] = f"{max(R['obs_mean'] for R in res)/min(R['obs_mean'] for R in res):.2f}"

    L += [r"\bottomrule",
          r"\multicolumn{4}{@{}p{\linewidth}@{}}{\footnotesize "
          r"GLM-5.3-Flash on hard45, 45 runs per harness "
          r"(\texttt{cc-\allowbreak glm53f-\allowbreak hard45-\allowbreak nw}, "
          r"\texttt{mini-\allowbreak glm53f-\allowbreak hard45b-\allowbreak nw}, "
          r"\texttt{oc-\allowbreak glm53f-\allowbreak hard45-\allowbreak nw-\allowbreak rg}); "
          r"the model, its settings and the task set are identical across the three columns, so "
          r"every difference here belongs to the harness.  "
          r"$^{a}$~the first call's prompt tokens, i.e.\ what every later call also pays before any "
          r"transcript: the harness's system prompt and tool schemas, its own user-message wrapper, "
          r"and the issue statement.  The split subtracts the issue statement at "
          + f"{ISSUE_CPT:.1f}" + r"~chars/token, measured on the first user turn, which OpenCode and "
          r"Claude Code receive verbatim and identically task by task (" + num(issue_chars) +
          r"~chars $=$ " + num(issue_tok) + r"~tokens on average); mini-SWE-agent's user turn "
          r"additionally carries its instance template (a constant " + num(mini_const) +
          r"~chars), which is counted as harness preamble.  The harnesses' own system prompts are "
          r"in neither the trajectory nor the agent log, so the preamble is a residual.  "
          r"$^{b}$~characters appended to the conversation over one run, converted at the arm's own "
          r"empirical rate (fitted on step-to-step $\Delta$prompt\_tokens vs.\ $\Delta$chars): Claude "
          r"Code and OpenCode carry prose at $\approx$3.8~chars/token, mini-SWE-agent's raw shell "
          r"output is token-dense at 2.2.  Everything a step appends is re-sent with every later "
          r"request; the fit only closes for Claude Code if its reasoning blocks are re-sent too, "
          r"which we infer from the fit rather than observe directly.  "
          r"$^{c}$~Claude Code's observation sizes are read from the raw stream log "
          r"(\texttt{agent/claude-code.txt}), because the trajectory converter appends the SDK's "
          r"\texttt{tool\_use\_result} metadata to every observation and roughly doubles it; that "
          r"blob was never sent to the model.  mini-SWE-agent and OpenCode trajectories match their "
          r"raw logs and are used as stored.  "
          r"$^{d}$~auto-compaction boundaries recorded by the Claude Code SDK; mini-SWE-agent and "
          r"OpenCode write no such record and never reset in these runs, so Claude Code's "
          r"reconstruction is an upper bound and the other two are exact.  On this arm Claude Code "
          r"runs with its own default auto-compaction window --- the 110k override of "
          r"Table~\ref{tab:controls} is set on the Qwen arms only --- and compacts at a median of "
          + num(cc_pre_median / 1e3, 0) + r"k prompt tokens, where OpenCode compacts at "
          r"\texttt{limit.input} minus its 20k reserve, i.e.\ at " + num(OC_COMPACT_TOK / 1e3, 0) +
          r"k, which is why the compaction counts differ.  "
          r"$^{e}$~input tokens reported by the provider, averaged over all steps of all 45 runs.} \\",
          r"\end{tabular}", r"\end{minipage}", r"}"]
    return L, macros


def main():
    res = []
    for arm, hkey, job in ARMS:
        R = analyse(arm, job)
        R["hkey"] = hkey
        R["key"] = arm.lower()
        res.append(R)
        print(f"{arm:5} trials={R['trials']:3d} obs={R['obs_n']:5d} mean={R['obs_mean']:.0f} "
              f"steps={R['steps']:.1f} recon={R['recon']:.0f} measured={R['measured']:.0f}", file=sys.stderr)

    L, macros = build(res)
    os.makedirs(TDIR, exist_ok=True)
    path = os.path.join(TDIR, "tableA4_budget.tex")
    with open(path, "w") as fh:
        fh.write("% generated by analysis/tab_a4_budget.py -- do not edit\n")
        fh.write("\n".join(L) + "\n")
    print("wrote", path)

    hdr = [r"% generated by analysis/tab_a4_budget.py -- do not edit",
           r"% usage: \pn{budget.cc.perstep.measured}   (Table A4; keys listed below)",
           r"% *.k keys are thousands of tokens; every other token key is a plain count",
           r"\makeatletter",
           r"\DeclareRobustCommand{\pn}[1]{\csname pn@#1\endcsname}"]
    body = [f"\\expandafter\\def\\csname pn@{k}\\endcsname{{{v}}}" for k, v in sorted(macros.items())]
    npath = os.path.join(ROOT, "paper", "latex", "numbers_a4.tex")
    open(npath, "w").write("\n".join(hdr + body + [r"\makeatother"]) + "\n")
    print("wrote", npath, "--", len(macros), "macros")


if __name__ == "__main__":
    main()
