#!/usr/bin/env python3
"""Appendix Tables A6 / A7 and their \\pn macros, from data/trials.csv.

  A6  tableA6_opus.tex       Claude Opus 5 inside Claude Code, four settings, hard45:
                             three control runs, bash-only (two runs), no-shell, and the
                             one-year-old Claude Code 1.0.100 release.
  A7  tableA7_thinking.tex   thinking on vs off, Qwen3.6-35B-A3B, three harnesses, hard45,
                             two runs per side, paired over tasks with a sign test.
  numbers_a67.tex            the \\pn macros for both, same mechanism as numbers_a4.tex.

Nothing statistical is reinvented here.  Reused verbatim from the project:
  * analysis/main_table.py   load(), num(), ok() + INFRA (14 infra exception types),
                             usable(), passes(), _fin(), wilson(), mcnemar_exact_p(),
                             MAIN_SEED.  The canonical-trial rule for an explicit list of
                             jobs -- usable beats unusable, ties go to the latest
                             finished_at -- is main_table.canonical()'s non-k=2 branch,
                             copied into pick() below because the arms in these two tables
                             (c2-bashonly-nw, c2-noshell-nw, cc-v1.0.100-nw,
                             control-nw-nothink) sit outside main_table.COLS and therefore
                             cannot be obtained by calling canonical() itself.
  * analysis/paper_numbers.py  canonical_by_seed() + pick_arm() for the Qwen3.8 pool447
                             CC vs OC-rg thinking-share comparison (those arms ARE in the
                             main grid, so the helpers are called, not copied).
  * the A7 method          per-task mean pass over the two runs of a side, paired over
                             the tasks both sides scored, sign test on the tasks whose mean
                             changed.  An earlier version broke ties on started_at instead
                             of finished_at; both tiebreaks give byte-identical A7 numbers
                             on this data (checked), so the canonical finished_at rule is
                             used.
  * analysis/tab_appendix.py   table style (booktabs, \\footnotesize, a footnote row in a
                             \\multicolumn p{\\linewidth}); no \\caption / \\label -- main.tex
                             wraps these two tables itself.
  * analysis/tab_a4_budget.py  the numbers_a4.tex macro mechanism (\\csname pn@<key>\\endcsname).
                             numbers_a4.tex declares \\pn unconditionally; this file guards the
                             declaration with \\@ifundefined so it never redefines the \\pn that
                             main.tex already got from numbers.tex.

One number cannot come from trials.csv: the reasoning share of Qwen3.8's output.  vLLM reports
every generated token in completion_tokens and leaves reasoning_tokens at 0 for the locally
served models, so the share is read from analysis/thinking_budget.txt, which recovers it by
tokenising the reasoning text out of the raw agent logs.  See THINKING_TXT below.

Run: python3 analysis/tab_a67.py
  -> paper/latex/tables/tableA6_opus.tex
  -> paper/latex/tables/tableA7_thinking.tex
  -> paper/latex/numbers_a67.tex
Read-only w.r.t. the data; stdlib only (plus the optional scipy main_table already uses).
"""
import os
import re
import statistics as st
import sys
from collections import defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main_table as mt            # noqa: E402  canonical rule, INFRA/usable, wilson, mcnemar
import paper_numbers as pn         # noqa: E402  canonical_by_seed, pick_arm

ROOT = os.path.normpath(os.path.join(HERE, ".."))
TDIR = os.path.join(ROOT, "paper", "latex", "tables")
NPATH = os.path.join(ROOT, "paper", "latex", "numbers_a67.tex")
THINKING_TXT = os.path.join(HERE, "thinking_budget.txt")
DASH = "---"
ARROW = r"\,$\to$\,"

# ----------------------------------------------------------------------------- A6 arms
# key -> (row label, jobs).  The jobs of one setting: the base job plus its repair jobs
# ("-r", "-r2", "-r3"), which rerun trials the base job lost to an infra exception.
OPUS_ROWS = [
    ("ctrl1", "Control, run 1",
     ["cc-opus-hard45-nw", "cc-opus-hard45-nw-r", "cc-opus-hard45-nw-r2", "cc-opus-hard45-nw-r3"]),
    ("ctrl2", "Control, run 2",
     ["cc-opus-hard45-nw-k2", "cc-opus-hard45-nw-k2-r", "cc-opus-hard45-nw-k2-r2"]),
    ("ctrl3", "Control, run 3",
     ["cc-opus-hard45-nw-k3", "cc-opus-hard45-nw-k3-r"]),
    ("bashonly1", "Bash-only, run 1",
     ["cc-opus-hard45-nw-c2", "cc-opus-hard45-nw-c2-r"]),
    ("bashonly2", "Bash-only, run 2",
     ["cc-opus-hard45-nw-c2-s2"]),
    ("noshell", "No shell",
     ["cc-opus-hard45-nw-c2ns"]),
    ("v1100", "Claude Code 1.0.100",
     ["cc-opus-hard45-nw-v1100"]),
]
CONTROLS = ["ctrl1", "ctrl2", "ctrl3"]

# ----------------------------------------------------------------------------- A7 arms
# harness -> thinking-on runs (base job + repairs) and the thinking-off job (2 attempts in one job).
# OpenCode is the arm WITHOUT ripgrep pre-placed: the nothink OpenCode arm had no rg either, so
# the -rg jobs would not be a like-for-like comparator.
THINK_ROWS = [
    ("cc", "claude-code", "Claude Code",
     [["cc-qwen36-hard45-nw"], ["cc-qwen36-hard45-nw-k2"]],
     "cc-qwen36-hard45-nw-nothink"),
    ("mini", "mini-swe-agent", "mini-SWE-agent",
     [["mini-qwen36-hard45-nw", "mini-qwen36-hard45-nw-r", "mini-qwen36-hard45-nw-r2"],
      ["mini-qwen36-hard45-nw-k2"]],
     "mini-qwen36-hard45-nw-nothink"),
    ("oc", "opencode", "OpenCode",
     [["oc-qwen36-hard45-nw", "oc-qwen36-hard45-nw-r", "oc-qwen36-hard45-nw-r2",
       "oc-qwen36-hard45-nw-r3"], ["oc-qwen36-hard45-nw-k2"]],
     "oc-qwen36-hard45-nw-nothink"),
]
# A7, second block: HY4-Preview inside Claude Code, ONE run per side.  Thinking is switched off
# through the API's thinking field (the endpoint defaults a missing field to on), not at the
# serving layer as on Qwen3.6, and only Claude Code was run this way.  With one run per side the
# per-task mean of a task is 0 or 1, so the up/down tasks are exactly the discordant pairs and the
# two-sided sign test on them is McNemar's exact test; both are computed and checked against
# each other.
HY4_ROW = ("hy4", "claude-code", "Claude Code", "hy4-preview",
           ["cc-hy4-hard45-nw-r"], "cc-hy4-hard45-nw-nothink")


# ----------------------------------------------------------------------------- formatting
def num(x, d=0):
    """1237 -> '1{,}237' (tab_a4_budget.num, LaTeX-safe thousands separator)."""
    return f"{x:,.{d}f}".replace(",", "{,}")


def smart(x):
    """Integer when the value is integral, one decimal otherwise (medians of even n)."""
    if x is None:
        return DASH
    return num(x, 0) if float(x).is_integer() else num(x, 1)


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


def pval(p, d=2):
    if p is None or p != p:
        return DASH
    return f"{p:.{d}f}"


def dotp(p):
    """'.77' / '.04' -- a p value without its leading zero (two decimals)."""
    if p is None or p != p:
        return DASH
    s = f"{p:.2f}"
    return s[1:] if s.startswith("0") else s


def sgn(v, d=1):
    return f"{v:+.{d}f}".replace("-", "$-$")


def tt(s):
    """Job / flag names in \\texttt with a break opportunity after every hyphen."""
    return r"\texttt{" + s.replace("-", r"-\allowbreak ").replace("_", r"\_") + "}"


def med(vals):
    vals = [v for v in vals if v is not None]
    return st.median(vals) if vals else None


def is_pass(r):
    return (mt.num(r.get("reward")) or 0) >= 1


# ----------------------------------------------------------------------------- canonical pick
def pick(rows_by_job, jobs, attempt=None):
    """{task: canonical row} over an explicit list of jobs.

    This is main_table.canonical()'s non-k=2 branch: excluded rows never enter, every row of
    the arm competes directly, usable beats unusable and ties go to the latest finished_at.
    `attempt` selects one within-job attempt index (the nothink jobs pack both runs of a side
    into a single repeat=1 job, exactly as the Qwen3.8 k=2 jobs do).
    """
    d = {}
    for job in jobs:
        for r in rows_by_job.get(job, []):
            if attempt is not None and int(mt.num(r.get("attempt")) or 1) != attempt:
                continue
            t = r["task"]
            cur = d.get(t)
            if cur is None or (mt.ok(r), mt._fin(r)) > (mt.ok(cur), mt._fin(cur)):
                d[t] = r
    return d


def flips(a, b):
    """(disagreeing tasks, tasks usable in both) between two runs."""
    ts = [t for t in a if t in b]
    return sum(1 for t in ts if is_pass(a[t]) != is_pass(b[t])), len(ts)


def mcnemar(a, b):
    """(only-a, only-b, exact two-sided p) -- main_table.mcnemar_exact_p on the discordants."""
    ts = [t for t in a if t in b]
    x = sum(1 for t in ts if is_pass(a[t]) and not is_pass(b[t]))
    y = sum(1 for t in ts if is_pass(b[t]) and not is_pass(a[t]))
    return x, y, mt.mcnemar_exact_p(x, y)


def sign_test(up, dn):
    """Two-sided sign test on the changed pairs (nothink_vs_think.py, stdlib math.comb)."""
    n = up + dn
    if n == 0:
        return float("nan")
    k = min(up, dn)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


# ============================================================================= A6
def build_a6(rows_by_job):
    runs, meta = {}, {}
    for key, label, jobs in OPUS_ROWS:
        d = pick(rows_by_job, jobs)
        u = mt.usable(d)
        vers = sorted({(r.get("agent_version") or "?") for r in u.values()})
        meta[key] = dict(
            label=label, jobs=jobs, n=len(d), u=len(u), k=mt.passes(u),
            version=", ".join(vers) if vers else DASH,
            out=med([mt.num(r.get("output_tokens_total")) for r in u.values()]),
            reas=med([mt.num(r.get("reasoning_tokens")) for r in u.values()]),
            steps=med([mt.num(r.get("steps")) for r in u.values()]),
            cost_mean=st.mean([mt.num(r.get("cost_usd")) or 0.0 for r in u.values()]) if u else None,
            cost_med=med([mt.num(r.get("cost_usd")) for r in u.values()]),
            cost_sum=sum(mt.num(r.get("cost_usd")) or 0.0 for r in u.values()),
        )
        runs[key] = u

    # flips against the three control runs pooled, and McNemar vs control run 1
    for key in meta:
        f = n = 0
        for c in CONTROLS:
            if c == key:
                continue
            ff, nn = flips(runs[key], runs[c])
            f += ff
            n += nn
        meta[key]["flips"], meta[key]["flipn"] = f, n
        if key == "ctrl1":
            meta[key]["mc"] = (None, None, float("nan"))
        else:
            meta[key]["mc"] = mcnemar(runs[key], runs["ctrl1"])
    floor_f = sum(flips(runs[a], runs[b])[0]
                  for a, b in (("ctrl1", "ctrl2"), ("ctrl1", "ctrl3"), ("ctrl2", "ctrl3")))
    floor_n = sum(flips(runs[a], runs[b])[1]
                  for a, b in (("ctrl1", "ctrl2"), ("ctrl1", "ctrl3"), ("ctrl2", "ctrl3")))

    # pooled control reference for the two ratio macros
    ctrl_trials = [r for c in CONTROLS for r in runs[c].values()]
    ctrl_out = med([mt.num(r.get("output_tokens_total")) for r in ctrl_trials])
    ctrl_cost_mean = st.mean([mt.num(r.get("cost_usd")) or 0.0 for r in ctrl_trials])
    ctrl_cost_med = med([mt.num(r.get("cost_usd")) for r in ctrl_trials])

    # ---- table
    allfull = all(m["u"] == 45 for m in meta.values())
    cover = (r"and all 45~tasks are usable in every row" if allfull
             else r"and the usable-task count is the denominator of the \emph{Solved} column")
    L = [r"\begin{tabular}{@{}llrcrrrrrr@{}}", r"\toprule",
         r"Setting & CC version & \multicolumn{2}{c}{Solved / 45} & flips & McNemar & "
         r"output & reas. & steps & USD/trial \\",
         r" &  & $k$ & 95\% CI (\%) & /135 & $p$ vs.\ s1 & tok. & tok. & & (mean) \\",
         r"\midrule"]
    for key, label, _jobs in OPUS_ROWS:
        m = meta[key]
        lo, hi = mt.wilson(m["k"], m["u"])
        fl = DASH if key in CONTROLS else f"{m['flips']}/{m['flipn']}"
        mp = DASH if key == "ctrl1" else pval(m["mc"][2])
        L.append(f"{label} & {m['version']} & {m['k']} & [{pct(lo)}, {pct(hi)}] & {fl} & {mp} & "
                 f"{smart(m['out'])} & {smart(m['reas'])} & {smart(m['steps'])} & "
                 f"{m['cost_mean']:.2f} \\\\")
        if key == "ctrl3":
            L.append(r"\addlinespace")
    L += [r"\bottomrule",
          r"\multicolumn{10}{@{}p{\linewidth}@{}}{\footnotesize Claude Opus~5 through Claude "
          r"Code on hard45, 45 tasks per row, no web access; every row is one run, "
          + cover + r".  \emph{Control} is the arm of the main tables "
          r"(three independent runs).  \emph{Bash-only} removes the file-editing tools with "
          r"\texttt{--disallowedTools} "
          r"(\texttt{Edit,\allowbreak Write,\allowbreak MultiEdit,\allowbreak "
          r"NotebookEdit,\allowbreak WebSearch}), so the model can only "
          r"edit through the shell; it was run twice.  \emph{No shell} removes execution instead "
          r"(\texttt{--disallowedTools WebSearch,Bash}), leaving the editing tools.  "
          r"\emph{Claude Code 1.0.100} is the npm release of 2025-09-01, one year older than the "
          r"2.1.263 the other six rows run, with the job YAML otherwise identical.  "
          r"\emph{Solved} is tasks with reward~1 out of the usable trials, with a Wilson 95\% "
          r"interval.  \emph{flips} counts tasks whose outcome differs from a control run, "
          r"pooled over all three runs ($3\times45$); the within-control floor --- the same "
          r"count between the three control runs themselves --- is " + f"{floor_f}/{floor_n}" +
          r" $=$ " + f"{100 * floor_f / floor_n:.1f}" + r"\%, so no setting here separates "
          r"itself from rerun noise by more than about a factor of two.  \emph{McNemar} is the "
          r"two-sided exact binomial on the discordant pairs against control run~1.  "
          r"\emph{output tok.} and \emph{reas. tok.} are per-trial medians of the provider's "
          r"output-token and reasoning-token counts; 1.0.100 predates extended thinking being on "
          r"by default, which is where its 4$\times$ smaller output and its zero reasoning come "
          r"from.  \emph{USD/trial} is the mean of the \texttt{cost\_usd} Claude Code reports for "
          r"itself, i.e.\ Anthropic list price for the tokens used; these runs were served by a "
          r"Claude Code subscription and were not billed per token, so the column is a "
          r"token-weighted effort index, not spend.} \\",
          r"\end{tabular}"]

    # ---- macros
    M = {}
    for key, _label, jobs in OPUS_ROWS:
        m = meta[key]
        lo, hi = mt.wilson(m["k"], m["u"])
        for kk, vv in (("k", str(m["k"])), ("n", str(m["u"])), ("rows", str(m["n"])),
                       ("pct", pct(m["k"] / m["u"])), ("ci", f"[{pct(lo)}, {pct(hi)}]"),
                       ("cilo", pct(lo)), ("cihi", pct(hi)),
                       ("version", m["version"]),
                       ("flips", DASH if key in CONTROLS else f"{m['flips']}/{m['flipn']}"),
                       ("flippct", DASH if key in CONTROLS
                        else f"{100 * m['flips'] / m['flipn']:.1f}"),
                       ("mcnemar", DASH if key == "ctrl1" else pval(m["mc"][2])),
                       ("out", smart(m["out"])), ("reas", smart(m["reas"])),
                       ("steps", smart(m["steps"])),
                       ("cost", f"{m['cost_mean']:.2f}"), ("costmed", f"{m['cost_med']:.2f}"),
                       ("costsum", f"{m['cost_sum']:.1f}"),
                       ("job", tt(jobs[0]))):
            M[f"opus.{key}.{kk}"] = vv
    M["opus.ctrl.k1"] = str(meta["ctrl1"]["k"])
    M["opus.ctrl.k2"] = str(meta["ctrl2"]["k"])
    M["opus.ctrl.k3"] = str(meta["ctrl3"]["k"])
    M["opus.bashonly.k1"] = str(meta["bashonly1"]["k"])
    M["opus.bashonly.k2"] = str(meta["bashonly2"]["k"])
    M["opus.noshell.k"] = str(meta["noshell"]["k"])
    M["opus.v1100.k"] = str(meta["v1100"]["k"])
    M["opus.v1100.costratio"] = f"{meta['v1100']['cost_mean'] / ctrl_cost_mean:.2f}"
    M["opus.v1100.costratio.med"] = f"{meta['v1100']['cost_med'] / ctrl_cost_med:.2f}"
    M["opus.v1100.outratio"] = f"{meta['v1100']['out'] / ctrl_out:.2f}"
    M["opus.ctrl.floor"] = f"{floor_f}/{floor_n}"
    M["opus.ctrl.floorpct"] = f"{100 * floor_f / floor_n:.1f}"
    M["opus.ctrl.out"] = smart(ctrl_out)
    M["opus.ctrl.cost"] = f"{ctrl_cost_mean:.2f}"
    M["opus.ctrl.costmed"] = f"{ctrl_cost_med:.2f}"
    return L, M, meta


# ============================================================================= A7
def build_a7(rows_by_job):
    res = {}
    for key, harness, label, on_runs, off_job in THINK_ROWS:
        sides = {"on": [pick(rows_by_job, jobs) for jobs in on_runs],
                 "off": [pick(rows_by_job, [off_job], attempt=a) for a in (1, 2)]}
        stat = {}
        for side, runs in sides.items():
            per_task = defaultdict(list)
            k = n = rows_n = 0
            steps, ctok = [], []
            for d in runs:
                u = mt.usable(d)
                rows_n += len(d)
                n += len(u)
                k += mt.passes(u)
                for t, r in u.items():
                    per_task[t].append(1.0 if is_pass(r) else 0.0)
                steps += [mt.num(r.get("steps")) for r in u.values()]
                ctok += [mt.num(r.get("completion_tokens")) for r in u.values()]
            stat[side] = dict(per_task={t: sum(v) / len(v) for t, v in per_task.items()},
                              k=k, n=n, rows=rows_n, steps=med(steps), ctok=med(ctok))
        a, b = stat["on"]["per_task"], stat["off"]["per_task"]
        common = sorted(set(a) & set(b))
        diffs = [b[t] - a[t] for t in common]
        up = sum(1 for x in diffs if x > 0)
        dn = sum(1 for x in diffs if x < 0)
        res[key] = dict(label=label, harness=harness, on=stat["on"], off=stat["off"],
                        common=len(common), up=up, dn=dn, same=len(common) - up - dn,
                        mean_on=sum(a[t] for t in common) / len(common),
                        mean_off=sum(b[t] for t in common) / len(common),
                        delta=sum(diffs) / len(common), p=sign_test(up, dn),
                        off_job=off_job)

    void_on = sum(2 * 45 - res[k]["on"]["n"] for k in res)
    void_off = sum(2 * 45 - res[k]["off"]["n"] for k in res)
    # ---- second block: HY4-Preview x Claude Code, one run per side
    hkey, _hh, hlabel, hmodel, hon_jobs, hoff_job = HY4_ROW
    hsides = {"on": mt.usable(pick(rows_by_job, hon_jobs)),
              "off": mt.usable(pick(rows_by_job, [hoff_job]))}
    hstat = {}
    for side, u in hsides.items():
        ent = pn.cost_entry(list(u.values()), pn.PRICES[hmodel]) if u else None
        hstat[side] = dict(
            per_task={t: (1.0 if is_pass(r) else 0.0) for t, r in u.items()},
            k=mt.passes(u), n=len(u),
            steps=med([mt.num(r.get("steps")) for r in u.values()]),
            ctok=med([mt.num(r.get("completion_tokens")) for r in u.values()]),
            outmean=st.mean([mt.num(r.get("output_tokens_total")) or 0.0
                             for r in u.values()]) if u else None,
            cny=ent["cny_per_trial"] if ent else None)
    ha, hb = hstat["on"]["per_task"], hstat["off"]["per_task"]
    hcommon = sorted(set(ha) & set(hb))
    hup = sum(1 for t in hcommon if hb[t] > ha[t])     # gained with thinking off
    hdn = sum(1 for t in hcommon if hb[t] < ha[t])     # lost with thinking off
    hres = dict(label=hlabel, model=hmodel, on=hstat["on"], off=hstat["off"],
                common=len(hcommon), up=hup, dn=hdn, same=len(hcommon) - hup - hdn,
                mean_on=sum(ha[t] for t in hcommon) / len(hcommon),
                mean_off=sum(hb[t] for t in hcommon) / len(hcommon),
                delta=sum(hb[t] - ha[t] for t in hcommon) / len(hcommon),
                p=sign_test(hup, hdn), mcp=mt.mcnemar_exact_p(hdn, hup),
                on_job=hon_jobs[0], off_job=hoff_job)
    assert abs(hres["p"] - hres["mcp"]) < 1e-9      # one run per side: sign test == McNemar exact
    res[hkey] = hres

    L = [r"\begin{tabular}{@{}lrrcrrrcc@{}}", r"\toprule",
         r"Harness & \multicolumn{2}{c}{solved / attempts} & per-task mean & $\Delta$ & "
         r"up / & sign & med.\ steps & med.\ output \\",
         r" & on & off & on " + ARROW + r" off & (pp) & down & $p$ & on " + ARROW + r" off & "
         r"on " + ARROW + r" off \\",
         r"\midrule"]
    for key, _h, _lab, _on, _off in THINK_ROWS:
        R = res[key]
        L.append(
            f"{R['label']} & {R['on']['k']}/{R['on']['n']} & {R['off']['k']}/{R['off']['n']} & "
            f"{R['mean_on']:.3f}{ARROW}{R['mean_off']:.3f} & {sgn(100 * R['delta'])} & "
            f"{R['up']}\\,/\\,{R['dn']} & {pval(R['p'])} & "
            f"{smart(R['on']['steps'])}{ARROW}{smart(R['off']['steps'])} & "
            f"{smart(R['on']['ctok'])}{ARROW}{smart(R['off']['ctok'])} \\\\")
    L += [r"\addlinespace",
          r"\multicolumn{9}{@{}l@{}}{\emph{HY4-Preview, Claude Code only, one run per side, "
          r"thinking switched at the API}} \\",
          f"{hres['label']} & {hres['on']['k']}/{hres['on']['n']} & "
          f"{hres['off']['k']}/{hres['off']['n']} & "
          f"{hres['mean_on']:.3f}{ARROW}{hres['mean_off']:.3f} & {sgn(100 * hres['delta'])} & "
          f"{hres['up']}\\,/\\,{hres['dn']} & {pval(hres['p'])} & "
          f"{smart(hres['on']['steps'])}{ARROW}{smart(hres['off']['steps'])} & "
          f"{smart(hres['on']['ctok'])}{ARROW}{smart(hres['off']['ctok'])} \\\\"]
    L += [r"\bottomrule",
          r"\multicolumn{9}{@{}p{\linewidth}@{}}{\footnotesize Qwen3.6-35B-A3B on hard45, no web "
          r"access, two independent runs per side and per harness (90 attempts), thinking "
          r"switched off at the server, not in the harness: a chat template that closes the "
          r"think block before generation, no reasoning parser, and the "
          r"\texttt{<think>} token banned by a logits processor, so all three harnesses see the "
          r"same model.  \emph{solved} counts attempts with reward~1 over the attempts that are "
          r"usable (scored, no infra exception); " + f"{void_off}" + r" thinking-off and "
          + f"{void_on}" + r" thinking-on attempts were lost to verifier faults "
          r"(\texttt{verifier\_incomplete}, one \texttt{VerifierTimeoutError}) and were not "
          r"rerun, which is why some denominators are below 90 and why the OpenCode pairing is "
          r"over 44 tasks: \texttt{django-13449} voided in both thinking-off attempts.  "
          r"\emph{per-task mean} is the mean outcome of a task "
          r"over the usable attempts of that side, and the comparison is paired over the tasks "
          r"both sides scored ($n$ $=$ 45, 45 and 44); $\Delta$ is off $-$ on in percentage "
          r"points.  \emph{up / down} counts the tasks whose per-task mean rose / fell; the sign "
          r"test is the two-sided exact binomial on those tasks, the remaining tasks being "
          r"ties.  This is one $k{=}2$ observation on 45 tasks.  "
          r"Three caveats.  Both OpenCode arms here run \emph{without} the pre-placed ripgrep, "
          r"because the thinking-off arm had none either; they are therefore not the OC-rg arm "
          r"of the main tables.  The Claude Code arms carry reasoning effort \texttt{xhigh}, its "
          r"setting throughout this paper.  And mini-SWE-agent's "
          r"\texttt{reasoning\_chars} are visible narration duplicated from the message text, "
          r"not hidden thinking, so its reasoning length is not comparable with the other two "
          r"and is omitted here.  The second block is a different experiment: HY4-Preview "
          r"through the vendor's API under Claude Code, one run per side on the same 45 tasks, "
          r"with thinking switched off by setting the API's thinking field to disabled --- the "
          r"endpoint treats a missing field as on --- instead of at the serving layer.  With one "
          r"run per side a per-task mean is 0 or 1, so \emph{up / down} are the discordant tasks "
          r"and the two-sided sign test on them is McNemar's exact test.  At Tencent's list "
          r"price the trial costs " + f"{hres['on']['cny']:.2f}" + r" CNY with thinking on and "
          + f"{hres['off']['cny']:.2f}" + r" with it off, from mean output of "
          + f"{hres['on']['outmean'] / 1e3:.1f}" + r"k and "
          + f"{hres['off']['outmean'] / 1e3:.1f}" + r"k tokens per trial.} \\",
          r"\end{tabular}"]

    M = {}
    for key, _h, _lab, _on, _off in THINK_ROWS:
        R = res[key]
        for kk, vv in (("on.k", str(R["on"]["k"])), ("on.n", str(R["on"]["n"])),
                       ("off.k", str(R["off"]["k"])), ("off.n", str(R["off"]["n"])),
                       ("on.pct", f"{100 * R['mean_on']:.0f}"),
                       ("off.pct", f"{100 * R['mean_off']:.0f}"),
                       ("on.mean", f"{R['mean_on']:.3f}"), ("off.mean", f"{R['mean_off']:.3f}"),
                       ("delta.pp", sgn(100 * R["delta"])),
                       ("up", str(R["up"])), ("down", str(R["dn"])), ("same", str(R["same"])),
                       ("n", str(R["common"])), ("p", dotp(R["p"])), ("pfull", pval(R["p"], 3)),
                       ("steps.on", smart(R["on"]["steps"])),
                       ("steps.off", smart(R["off"]["steps"])),
                       ("ctok.on", smart(R["on"]["ctok"])),
                       ("ctok.off", smart(R["off"]["ctok"])),
                       ("job.off", tt(R["off_job"]))):
            M[f"think.{key}.{kk}"] = vv
    M["think.mini.drop.pp"] = f"{-100 * res['mini']['delta']:.0f}"
    R = hres
    for kk, vv in (("on.k", str(R["on"]["k"])), ("on.n", str(R["on"]["n"])),
                   ("off.k", str(R["off"]["k"])), ("off.n", str(R["off"]["n"])),
                   ("on.pct", f"{100 * R['mean_on']:.0f}"),
                   ("off.pct", f"{100 * R['mean_off']:.0f}"),
                   ("on.mean", f"{R['mean_on']:.3f}"), ("off.mean", f"{R['mean_off']:.3f}"),
                   ("delta.pp", sgn(100 * R["delta"])),
                   ("drop.pp", f"{-100 * R['delta']:.0f}"),
                   ("up", str(R["up"])), ("down", str(R["dn"])), ("same", str(R["same"])),
                   ("n", str(R["common"])), ("p", dotp(R["p"])), ("pfull", pval(R["p"], 3)),
                   ("mcnemar", dotp(R["mcp"])),
                   ("steps.on", smart(R["on"]["steps"])),
                   ("steps.off", smart(R["off"]["steps"])),
                   ("ctok.on", smart(R["on"]["ctok"])),
                   ("ctok.off", smart(R["off"]["ctok"])),
                   ("outk.on", num(R["on"]["outmean"] / 1e3, 1)),
                   ("outk.off", num(R["off"]["outmean"] / 1e3, 1)),
                   ("outratio", f"{R['off']['outmean'] / R['on']['outmean']:.2f}"),
                   ("cny.on", f"{R['on']['cny']:.2f}"), ("cny.off", f"{R['off']['cny']:.2f}"),
                   ("job.on", tt(R["on_job"])), ("job.off", tt(R["off_job"]))):
        M[f"think.hy4.{kk}"] = vv
    return L, M, res


# ============================================================================= thinking share
def parse_thinking_txt(path):
    """The paired cross-harness block of analysis/thinking_budget.txt.

    That script recovers the reasoning share by tokenising the reasoning TEXT out of the raw
    agent logs, because vLLM reports every generated token in completion_tokens and leaves
    reasoning_tokens at 0 -- the share is therefore not derivable from trials.csv.
    """
    if not os.path.exists(path):
        return {}
    txt = open(path).read()
    blk = txt.split("== paired:", 1)
    if len(blk) < 2:
        return {}
    blk = blk[1].split("###", 1)[0]
    out = {}
    for jb, o, per, sh in re.findall(
            r"^\s+(\S+)\s+output\s+(\d+)\s+\(\s*(\d+)/task\)\s+reasoning\s+([\d.]+)%",
            blk, re.M):
        key = "cc" if jb.startswith("cc-") else "oc"
        out[key] = dict(job=jb, output=int(o), per_task=int(per), share=float(sh))
    m = re.search(r"output-token volume\s+B/A = ([\d.]+)x.*?n=(\d+)", blk)
    if m:
        out["ratio_geo"] = float(m.group(1))
        out["ratio_n"] = int(m.group(2))
    m = re.search(r"common tasks (\d+)", blk)
    if m:
        out["common"] = int(m.group(1))
    return out


def build_share(rows):
    """Qwen3.8-27B on pool447, Claude Code vs OpenCode-rg, paired over tasks both arms scored."""
    cells = pn.canonical_by_seed(rows)
    M, info = {}, {}
    arms = {}
    for key, harness in (("cc", "claude-code"), ("oc", "opencode")):
        lab, cell = pn.pick_arm(cells, "qwen3.8-27b", "pool447", harness)
        arms[key] = (lab, mt.usable((cell or {}).get(mt.MAIN_SEED, {})))
    common = sorted(set(arms["cc"][1]) & set(arms["oc"][1]))
    per_trial = {}
    for key in ("cc", "oc"):
        u = arms[key][1]
        outs = [mt.num(u[t].get("output_tokens_total")) or 0.0 for t in common]
        reas = sum(mt.num(u[t].get("reasoning_tokens")) or 0.0 for t in common)
        per_trial[key] = st.mean(outs) if outs else 0.0
        info[key] = dict(lab=arms[key][0], out=sum(outs), reas=reas, mean=per_trial[key],
                         share_csv=(100 * reas / sum(outs)) if sum(outs) else 0.0)
    ratio_means = per_trial["cc"] / per_trial["oc"] if per_trial["oc"] else float("nan")
    lr = [(mt.num(arms["cc"][1][t].get("output_tokens_total")) or 0.0)
          / (mt.num(arms["oc"][1][t].get("output_tokens_total")) or 1.0) for t in common]
    lr = [x for x in lr if x > 0]
    ratio_geo = st.geometric_mean(lr) if lr else float("nan")

    tb = parse_thinking_txt(THINKING_TXT)
    for key in ("cc", "oc"):
        if key in tb:
            M[f"think.share.{key}"] = f"{tb[key]['share']:.1f}"
            info[key]["share_txt"] = tb[key]["share"]
            info[key]["job_txt"] = tb[key]["job"]
    # thinking_budget.txt's "output-token volume B/A" is the paired GEOMETRIC mean of the
    # per-task CC/OC ratio; trials.csv reproduces it to three decimals.  The ratio of the two
    # arms' mean output per trial is a different, slightly smaller number and is kept separately.
    M["think.outratio"] = f"{ratio_geo:.2f}"
    M["think.outratio.means"] = f"{ratio_means:.2f}"
    M["think.share.n"] = str(len(common))
    M["think.share.cc.outk"] = num(per_trial["cc"] / 1e3, 1)
    M["think.share.oc.outk"] = num(per_trial["oc"] / 1e3, 1)
    info["common"] = len(common)
    info["ratio_means"] = ratio_means
    info["ratio_geo"] = ratio_geo
    info["txt"] = tb
    return M, info


# ============================================================================= output
def write_table(name, lines):
    os.makedirs(TDIR, exist_ok=True)
    path = os.path.join(TDIR, name)
    with open(path, "w") as fh:
        fh.write("% generated by analysis/tab_a67.py -- do not edit\n")
        fh.write("{\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n")
        fh.write("\n".join(lines) + "\n}\n")
    print("wrote", path)
    return path


def write_macros(macros):
    hdr = [r"% generated by analysis/tab_a67.py -- do not edit",
           r"% usage: \pn{opus.v1100.costratio}, \pn{think.mini.p}   (Tables A6/A7)",
           r"% \pn itself is defined by numbers.tex; the guard below only covers the case where",
           r"% this file is \input on its own (e.g. a scratch preview), never redefining it.",
           r"\makeatletter",
           r"\@ifundefined{pn}{\DeclareRobustCommand{\pn}[1]{\csname pn@#1\endcsname}}{}"]
    body = [f"\\expandafter\\def\\csname pn@{k}\\endcsname{{{v}}}" for k, v in sorted(macros.items())]
    with open(NPATH, "w") as fh:
        fh.write("\n".join(hdr + body + [r"\makeatother"]) + "\n")
    print("wrote", NPATH, "--", len(macros), "macros")
    return NPATH


def main():
    rows, _fields = mt.load(mt.CSV_PATH)
    if not rows:
        print("no trials to aggregate.", file=sys.stderr)
        return 1
    rows_by_job = defaultdict(list)
    for r in rows:
        if r.get("exclude") != "1":
            rows_by_job[r["job"]].append(r)

    l6, m6, meta6 = build_a6(rows_by_job)
    l7, m7, res7 = build_a7(rows_by_job)
    ms, infos = build_share(rows)

    write_table("tableA6_opus.tex", l6)
    write_table("tableA7_thinking.tex", l7)
    macros = {}
    macros.update(m6)
    macros.update(m7)
    macros.update(ms)
    write_macros(macros)

    # ------------------------------------------------------------------ stderr summary
    e = sys.stderr
    print("\nA6  Opus 5 x Claude Code, hard45", file=e)
    for key, label, _j in OPUS_ROWS:
        m = meta6[key]
        b, c, p = m["mc"]
        mcs = "     -" if key == "ctrl1" else f"{b}:{c} p={p:.3f}"
        print(f"  {label:20s} {m['k']:2d}/{m['u']:2d}  ver {m['version']:8s} "
              f"flips {m['flips']:3d}/{m['flipn']:3d}  mcnemar {mcs:12s}"
              f"  out {smart(m['out']):>8s}  reas {smart(m['reas']):>7s}  "
              f"steps {smart(m['steps']):>5s}  cost mean {m['cost_mean']:.3f} med {m['cost_med']:.3f}",
              file=e)
    print(f"  ratios: cost mean {macros['opus.v1100.costratio']} "
          f"(median-based {macros['opus.v1100.costratio.med']}), "
          f"output tokens {macros['opus.v1100.outratio']}", file=e)
    print("\nA7  Qwen3.6 thinking on vs off, hard45", file=e)
    for key, _h, _l, _o, _f in THINK_ROWS:
        R = res7[key]
        print(f"  {R['label']:15s} on {R['on']['k']:2d}/{R['on']['n']:2d} "
              f"off {R['off']['k']:2d}/{R['off']['n']:2d}  "
              f"mean {R['mean_on']:.3f} -> {R['mean_off']:.3f} ({100 * R['delta']:+.1f} pp) "
              f"up/dn {R['up']}/{R['dn']} p {R['p']:.3f} n={R['common']}", file=e)
    R = res7["hy4"]
    print(f"  {'HY4 x CC':15s} on {R['on']['k']:2d}/{R['on']['n']:2d} "
          f"off {R['off']['k']:2d}/{R['off']['n']:2d}  "
          f"mean {R['mean_on']:.3f} -> {R['mean_off']:.3f} ({100 * R['delta']:+.1f} pp) "
          f"up/dn {R['up']}/{R['dn']} p {R['p']:.3f} (McNemar {R['mcp']:.3f}) n={R['common']}  "
          f"steps {R['on']['steps']:.0f}->{R['off']['steps']:.0f}  "
          f"out {R['on']['outmean']/1e3:.1f}k->{R['off']['outmean']/1e3:.1f}k  "
          f"CNY {R['on']['cny']:.2f}->{R['off']['cny']:.2f}", file=e)
    print("\nthinking share, Qwen3.8 pool447 (CC vs OC-rg), paired n=%d" % infos["common"], file=e)
    for key in ("cc", "oc"):
        i = infos[key]
        print(f"  {key.upper():3s} {i['lab']:5s} share(trials.csv reasoning_tokens) "
              f"{i['share_csv']:.1f}%  share(thinking_budget.txt) "
              f"{i.get('share_txt', float('nan')):.1f}%  mean output/trial {i['mean']:.0f}", file=e)
    print(f"  output ratio CC/OC: geometric (paired) {infos['ratio_geo']:.4f}, "
          f"ratio of means {infos['ratio_means']:.4f}; thinking_budget.txt B/A "
          f"{infos['txt'].get('ratio_geo')}", file=e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
