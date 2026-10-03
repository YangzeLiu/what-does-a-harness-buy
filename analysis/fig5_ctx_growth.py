#!/usr/bin/env python3
"""Figure 5 + Table-A4 companion: what the model reads at every step.

The same model costs ~3x more under Claude Code than under mini-SWE-agent because of the input
it re-reads at each step, and that input is a line: an intercept (the harness preamble that is
re-sent verbatim on every call) plus a slope (what one step appends to the transcript).  Claude
Code re-sends the model's own reasoning blocks in the carried history; mini-SWE-agent and
OpenCode talk to OpenAI-compatible endpoints, where reasoning is not carried back.

  panel (a) GLM-5.3-Flash, panel (b) DeepSeek-V4-Flash, hard45, canonical run 1
  x = main-thread agent step index (0 = the first model call), y = input tokens of that call
  faint line per trial, bold median over the trials still running at that step (>= 10 alive)

Per harness x model we also fit a robust (Theil-Sen) line ctx_k = a + b*k on the steps before the
first compaction drop and write the medians of a and b to paper/latex/numbers_ctx.tex.

Data (read-only): the ATIF trajectory of each canonical trial,
results/raw/jobs/<job>/<task>__<trial_id>/agent/trajectory.json, main thread only (sidechains
excluded, same rule as extract_trials.peak_ctx).  Per-step usage is the provider's own
prompt_tokens; for Claude Code the stream log agent/claude-code.txt carries usage only on the
Anthropic endpoint (GLM/DeepSeek trials report input_tokens=0 there), so the trajectory metrics
are the only per-step source on these two models -- exactly what extract_trials.py used to fill
peak_ctx / n_compaction, and every series below is checked against those two csv columns.

Run: python3 analysis/fig5_ctx_growth.py
  -> paper/figures/fig5_ctx_growth.pdf/.png
  -> paper/latex/numbers_ctx.tex   (\\pn macros, keys ctx.<glmflash|dsflash>.<CC|mini|OC>.*)
  -> paper/ctx_growth.json         (per-trial fits)
"""
import csv
import json
import os
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import extract_trials as ex   # noqa: E402  trajectory reading + the compaction-drop rule
import main_table as mt       # noqa: E402  canonical rule (usable / MAIN_SEED)
import paper_numbers as pn    # noqa: E402  canonical_by_seed / pick_arm
import figstyle as fs         # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.normpath(os.path.join(HERE, ".."))
JOBS = os.environ.get("HARNESS_JOBS_ROOT", os.path.join(ROOT, "results", "raw", "jobs"))
TEX = os.path.join(ROOT, "paper", "latex", "numbers_ctx.tex")
JSON_OUT = os.path.join(ROOT, "paper", "ctx_growth.json")

MODELS = [("glm-5.3-flash", "glmflash"), ("ds-v4-flash", "dsflash")]
HARNESSES = ["claude-code", "mini-swe-agent", "opencode"]
KEY = {"claude-code": "CC", "mini-swe-agent": "mini", "opencode": "OC"}
SUBSET = "hard45"

DROP_ABS = 20_000     # a fall of more than this many prompt tokens vs the previous step = compaction
MIN_FIT = 5           # steps needed before the first drop to fit a line
MIN_ALIVE = 23        # draw the median only while this many trials are still running
CC_CPT = 4.0          # chars per token for CC reasoning_content (no usage split on these endpoints)


# ----------------------------------------------------------------------------- per-trial series
def series(row):
    """(ctx, reason) for one trial: input tokens and reasoning tokens of every main-thread step.

    Reasoning is the provider's own count where it exists (OpenCode extra.reasoning_tokens,
    mini/litellm completion_tokens_details.reasoning_tokens, CC output_tokens_details.
    thinking_tokens).  Claude Code on a non-Anthropic endpoint reports none, so its reasoning is
    the length of the ATIF reasoning_content divided by CC_CPT; on OpenCode, where both exist,
    chars/4 reproduces the reported reasoning tokens to within a few per cent (see stdout).
    """
    d = os.path.join(JOBS, row["job"], f"{row['task']}__{row['trial_id']}")
    traj = ex.read_json(os.path.join(d, "agent", "trajectory.json"))
    if not traj:
        return None, None, None
    agent = [s for s in (traj.get("steps") or []) if s.get("source") == "agent"]
    main = [s for s in agent if not (s.get("extra") or {}).get("is_sidechain")]
    ctx, reason, rchars = [], [], []
    for s in main:
        m = s.get("metrics") or {}
        if m.get("prompt_tokens") is None:
            continue
        e = m.get("extra") or {}
        ctx.append(m["prompt_tokens"])
        reason.append((e.get("reasoning_tokens")
                       or (e.get("output_tokens_details") or {}).get("thinking_tokens")
                       or (e.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0))
        rchars.append(len(s.get("reasoning_content") or ""))
    return ctx, reason, rchars


def first_drop(ctx):
    """Index of the first compaction drop, or len(ctx) if the transcript never dropped."""
    for k in range(1, len(ctx)):
        if ctx[k - 1] - ctx[k] > DROP_ABS:
            return k
    return len(ctx)


def theilsen(y):
    """Robust fit y_k = a + b*k over k = 0..n-1; returns (a, b)."""
    n = len(y)
    slopes = [(y[j] - y[i]) / (j - i) for i in range(n) for j in range(i + 1, n)]
    b = st.median(slopes)
    a = st.median(y[k] - b * k for k in range(n))
    return a, b


def med(v):
    return st.median(v) if v else float("nan")


# ----------------------------------------------------------------------------- collect
def collect():
    rows = list(csv.DictReader(open(os.path.join(HERE, "..", "data", "trials.csv"))))
    cells = pn.canonical_by_seed(rows)
    data = {}      # (model, harness) -> list of per-trial dicts
    bad = []
    for model, _mk in MODELS:
        for h in HARNESSES:
            lab, cell = pn.pick_arm(cells, model, SUBSET, h)
            if cell is None:
                bad.append(f"no arm for {model}/{h}")
                continue
            trials = mt.usable(cell[mt.MAIN_SEED])
            out = []
            for task, r in sorted(trials.items()):
                ctx, reason, rchars = series(r)
                if not ctx:
                    bad.append(f"no trajectory: {r['job']}/{task}")
                    continue
                # the series must reproduce the two csv columns derived from it
                pk, nc = mt.num(r.get("peak_ctx")), mt.num(r.get("n_compaction"))
                if pk is not None and int(pk) != max(ctx):
                    bad.append(f"peak_ctx mismatch {r['job']}/{task}: csv {pk:.0f} vs {max(ctx)}")
                if nc is not None and int(nc) != ex.count_drops(ctx):
                    bad.append(f"n_compaction mismatch {r['job']}/{task}: csv {nc:.0f} vs "
                               f"{ex.count_drops(ctx)}")
                k0 = first_drop(ctx)
                fit = theilsen(ctx[:k0]) if k0 >= MIN_FIT else (None, None)
                rtok = [x for x in reason]
                rch = [c / CC_CPT for c in rchars]
                out.append(dict(job=r["job"], task=task, trial_id=r["trial_id"], lab=lab,
                                n_steps=len(ctx), first_drop=k0, n_drop=ex.count_drops(ctx),
                                peak=max(ctx), first_ctx=ctx[0],
                                a=fit[0], b=fit[1],
                                reason_tok=med(rtok), reason_char4=med(rch),
                                ctx=ctx))
            data[(model, h)] = out
    return data, bad


# ----------------------------------------------------------------------------- figure
def survivor_median(trials, min_alive=MIN_ALIVE):
    """(xs, ys) of the median input at step k over the trials still running at k."""
    xs, ys = [], []
    kmax = max((t["n_steps"] for t in trials), default=0)
    for k in range(kmax):
        alive = [t["ctx"][k] for t in trials if t["n_steps"] > k]
        if len(alive) >= min_alive:
            xs.append(k)
            ys.append(st.median(alive))
    return xs, ys


def figure(data):
    fs.setup()
    fig, axes = plt.subplots(1, 2, figsize=(fs.WIDTH["full"], 2.4), sharey=True)
    for ax, (model, _mk), letter in zip(axes, MODELS, "ab"):
        xlim = 0
        for h in HARNESSES:
            trials = data.get((model, h)) or []
            col = fs.HARNESS_COLOR[h]
            for t in trials:
                ax.plot(range(t["n_steps"]), [c / 1e3 for c in t["ctx"]], color=col,
                        lw=0.5, alpha=0.15, solid_capstyle="round", zorder=2)
            xs, ys = survivor_median(trials)
            if xs:
                ax.plot(xs, [y / 1e3 for y in ys], color=col, lw=1.6,
                        label=fs.HARNESS_LONG[h], zorder=4)
                xlim = max(xlim, xs[-1])
            # the faint lines may run much further than the median; keep a little of that tail
            x3, _ = survivor_median(trials, 3)
            if x3:
                xlim = max(xlim, min(x3[-1], (xs[-1] if xs else 0) + 40))
        fs.despine(ax)
        fs.hgrid(ax)
        ax.set_xlim(0, xlim + 2)
        ax.set_title(fs.MODEL_LABEL[model])
        ax.set_xlabel("agent step (main thread)")
        fs.panel_label(ax, letter)
    axes[0].set_ylabel("input tokens of that call (k)")
    axes[0].set_ylim(0, None)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.01, 1.0))
    t = fs.note(axes[1], "bold line: median over the trials\nstill running at that step\n"
                "(drawn while $\\geq$%d are alive)" % MIN_ALIVE,
                xy=(0.98, 0.04), ha="right", va="bottom")
    t.set_bbox(dict(facecolor="white", edgecolor="none", alpha=0.75, pad=1.5))
    fig.subplots_adjust(wspace=0.08)
    return fs.save(fig, "fig5_ctx_growth")


# ----------------------------------------------------------------------------- output
def fmt(x, d=0):
    s = f"{x:,.{d}f}"
    return s.replace(",", "{,}")


def main():
    data, bad = collect()
    summary = []
    macros = {}
    for model, mk in MODELS:
        for h in HARNESSES:
            trials = data.get((model, h)) or []
            fits = [t for t in trials if t["a"] is not None]
            a = med([t["a"] for t in fits])
            b = med([t["b"] for t in fits])
            # reasoning per step: provider count where it exists, else CC's reasoning chars / 4
            rt = med([t["reason_tok"] for t in trials])
            rc = med([t["reason_char4"] for t in trials])
            r = rc if (rt == 0 or rt != rt) else rt
            comp = sum(1 for t in trials if t["n_drop"]) / len(trials) if trials else float("nan")
            summary.append(dict(model=model, harness=h, n=len(trials), n_fit=len(fits),
                                intercept=a, slope=b, reason=r, reason_tok=rt, reason_char4=rc,
                                comp_share=comp,
                                first_ctx=med([t["first_ctx"] for t in trials]),
                                first_ctx_mean=st.mean([t["first_ctx"] for t in trials]),
                                steps=med([t["n_steps"] for t in trials]),
                                perstep=med([st.mean(t["ctx"]) for t in trials]),
                                # pooled over every step of every trial: the same statistic as
                                # Table A4's budget.<arm>.perstep.measured
                                perstep_pooled=st.mean([c for t in trials for c in t["ctx"]])))
            k = f"ctx.{mk}.{KEY[h]}"
            macros[k + ".intercept.k"] = f"{a / 1e3:.1f}"
            macros[k + ".slope"] = fmt(round(b))
            macros[k + ".reason.step"] = fmt(round(r))
            macros[k + ".n"] = str(len(trials))
            # the measured first call, i.e. preamble + issue before any tool return: the same
            # quantity Table A4 reports as budget.<arm>.fixed (mean over the arm's trials)
            macros[k + ".first.k"] = f"{st.mean([t['first_ctx'] for t in trials]) / 1e3:.1f}"

    # ---- stdout table
    print(f"per-step input growth, {SUBSET}, canonical run {mt.MAIN_SEED}; fit on the steps before "
          f"the first drop of >{DROP_ABS // 1000}k tokens (Theil-Sen, >={MIN_FIT} steps)")
    hdr = ("model", "harness", "n", "fit", "steps", "1st call", "intercept", "slope",
           "reason/step", "compaction", "input/step")
    print("%-10s %-5s %3s %3s %6s %9s %10s %7s %11s %11s %16s" % hdr)
    for s in summary:
        print("%-10s %-5s %3d %3d %6.0f %9s %10s %7s %11s %10.0f%% %16s"
              % (fs.MODEL_SHORT[s["model"]], KEY[s["harness"]], s["n"], s["n_fit"], s["steps"],
                 f"{s['first_ctx_mean']:,.0f}", f"{s['intercept']:,.0f}", f"{s['slope']:,.0f}",
                 f"{s['reason']:,.0f}", 100 * s["comp_share"],
                 f"{s['perstep_pooled']:,.0f} ({s['perstep']:,.0f})"))
    print("  input/step is pooled over every step of every trial (Table A4's statistic); the "
          "median of the per-trial means is in brackets")

    # slope ratios against the paper's per-step input ratio (Table A4, GLM)
    for model, mk in MODELS:
        g = {s["harness"]: s for s in summary if s["model"] == model}
        base = g["mini-swe-agent"]
        print(f"  {fs.MODEL_SHORT[model]}: slope CC:mini:OC = "
              + ":".join(f"{g[h]['slope'] / base['slope']:.2f}" for h in HARNESSES)
              + "   measured input/step = "
              + ":".join(f"{g[h]['perstep_pooled'] / base['perstep_pooled']:.2f}" for h in HARNESSES)
              + "   intercept = "
              + ":".join(f"{g[h]['intercept'] / base['intercept']:.2f}" for h in HARNESSES))

    # reasoning sanity: OpenCode reports both, so chars/4 can be checked against the usage count
    for model, _mk in MODELS:
        s = [x for x in summary if x["model"] == model and x["harness"] == "opencode"][0]
        print(f"  OC {fs.MODEL_SHORT[model]}: reasoning tokens/step {s['reason_tok']:.0f} vs "
              f"reasoning chars/4 {s['reason_char4']:.0f} (the CC proxy, checked where both exist)")

    for b in bad:
        print("  ! " + b)

    # ---- macros + per-trial dump
    with open(TEX, "w") as fh:
        fh.write("% generated by analysis/fig5_ctx_growth.py -- do not edit\n")
        fh.write("% usage: \\pn{ctx.glmflash.CC.slope}   (Fig 5; keys listed below)\n")
        fh.write("% *.intercept.k is thousands of tokens; slope and reason.step are tokens per step\n")
        fh.write("\\makeatletter\n")
        fh.write("\\@ifundefined{pn}{\\DeclareRobustCommand{\\pn}[1]{\\csname pn@#1\\endcsname}}{}\n")
        for k, v in sorted(macros.items()):
            fh.write(f"\\expandafter\\def\\csname pn@{k}\\endcsname{{{v}}}\n")
        fh.write("\\makeatother\n")
    dump = {f"{m}/{h}": [{k: v for k, v in t.items() if k != "ctx"} for t in ts]
            for (m, h), ts in data.items()}
    json.dump(dict(rule=dict(drop_abs=DROP_ABS, min_fit=MIN_FIT, min_alive=MIN_ALIVE,
                             cc_chars_per_token=CC_CPT, subset=SUBSET, seed=mt.MAIN_SEED),
                   summary=summary, trials=dump), open(JSON_OUT, "w"), indent=1)
    path = figure(data)
    print(f"macros: {len(macros)} -> {os.path.relpath(TEX, ROOT)};  "
          f"fits -> {os.path.relpath(JSON_OUT, ROOT)};  figure -> {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
