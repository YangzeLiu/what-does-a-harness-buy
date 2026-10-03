#!/usr/bin/env python3
"""Tables 1, 2 and the number macros from paper/numbers.json.
Outputs paper/latex/tables/table1_hard45.tex, table2_pool447.tex, paper/latex/numbers.tex.
Macros: \\pn{key} with keys like qwen36.hard45.CC.k  (see numbers.tex header)."""
import json, os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import figstyle as fs

HERE = fs.HERE
N = json.load(open(os.path.join(HERE, "..", "paper", "numbers.json")))
TDIR = os.path.normpath(os.path.join(HERE, "..", "paper", "latex", "tables"))
os.makedirs(TDIR, exist_ok=True)
KEY = {"qwen3.6-35b-a3b": "qwen36", "qwen3.8-27b": "qwen38", "ds-v4-flash": "dsflash", "glm-5.3-flash": "glmflash",
       "hy4-preview": "hy4", "opus5": "opus"}
def sgn(v):
    return f"{v:+.1f}".replace("-", "$-$")

HK = {"claude-code": "CC", "mini-swe-agent": "mini", "opencode": "OC", "dsh": "dsh"}
TEXNAME = {"qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "qwen3.8-27b": "Qwen3.8-27B", "ds-v4-flash": "DeepSeek-V4-Flash",
           "glm-5.3-flash": "GLM-5.3-Flash", "hy4-preview": "HY4-Preview", "opus5": "Claude Opus 5"}
macros = {}


def pct(x):
    return f"{100*x:.0f}"


def fmt_p(p):
    return "$<$.001" if p < 0.001 else f"{p:.2f}".lstrip("0") if p >= 0.01 else f"{p:.3f}".lstrip("0")


# ------------------------------------------------------------------ Table 1 (hard45)
lines = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"Model & Claude Code & mini-SWE-agent & OpenCode (rg) \\", r"\midrule"]
for m in fs.MODEL_ORDER:
    t = N["table1"].get(m, {}).get("hard45")
    if not t:
        continue
    cells = []
    for h in fs.HARNESS_ORDER:
        e = t.get(h)
        if not e:
            cells.append("---")
            continue
        k, n = e["k"], e["n"]
        lo, hi = e["wilson"]
        ns = len(e["seeds"])
        sup = f"$^{{\\,\\times{ns}}}$"   # every cell carries its run count so rows align
        star = "" if e["arm"] != "OC" else r"$^{\dagger}$"
        cells.append(f"{k}/{n}{sup}{star} \;{pct(k/n)}\\% [{pct(lo)},\\,{pct(hi)}]")
        for kk, vv in (("k", k), ("n", n), ("pct", pct(k / n)), ("lo", pct(lo)), ("hi", pct(hi)), ("seeds", ns)):
            macros[f"{KEY[m]}.hard45.{HK[h]}.{kk}"] = vv
    lines.append(f"{TEXNAME[m]} & " + " & ".join(cells) + r" \\")
    # arms outside the three-harness grid (e.g. dsh): macros only, cited in the table footnote
    for h, e in t.items():
        if h in fs.HARNESS_ORDER or not e:
            continue
        lo, hi = e["wilson"]
        for kk, vv in (("k", e["k"]), ("n", e["n"]), ("pct", pct(e["k"] / e["n"])), ("lo", pct(lo)), ("hi", pct(hi))):
            macros[f"{KEY[m]}.hard45.{HK[h]}.{kk}"] = vv
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TDIR, "table1_hard45.tex"), "w").write("\n".join(lines) + "\n")

# ------------------------------------------------------------------ Table 2 (pool447 paired)
lines = [r"\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{@{}llrrrrrrcc@{}}", r"\toprule",
         r"Model & Pair $A-B$ & $n$ & $A$ & $B$ & $b$ & $c$ & $\Delta$ (pp) & McNemar $p$ & Verdict \\", r"\midrule"]
for m in ["qwen3.6-35b-a3b", "qwen3.8-27b"]:
    first = True
    for e in [e for e in N["pairs"] if e["model"] == m and e["subset"] == "pool447"]:
        pair = e["pair"].replace("−", "$-$")
        lo9, hi9 = e["ci90_pp"]
        verdict = "equivalent" if e["tost5"] else ("inferior $B$" if e["p_mcnemar"] < 0.05 and e["diff_pp"] > 0 else
                                                 "inferior $A$" if e["p_mcnemar"] < 0.05 else "inconclusive")
        lines.append(f"{TEXNAME[m] if first else ''} & {pair} & {e['n']} & {e['k1']} & {e['k2']} & {e['b']} & {e['c']} & "
                     f"{sgn(e['diff_pp'])} [{sgn(e['ci95_pp'][0])}, {sgn(e['ci95_pp'][1])}] & {fmt_p(e['p_mcnemar'])} & {verdict} \\\\")
        first = False
        key = f"{KEY[m]}.pool447.{e['pair'].replace('−', 'v')}"
        for kk, vv in (("n", e["n"]), ("b", e["b"]), ("c", e["c"]), ("diff", f"{e['diff_pp']:+.1f}"),
                       ("lo", f"{e['ci95_pp'][0]:+.1f}"), ("hi", f"{e['ci95_pp'][1]:+.1f}"), ("p", fmt_p(e["p_mcnemar"])),
                       ("lonine", f"{lo9:+.1f}"), ("hinine", f"{hi9:+.1f}"), ("tost", "yes" if e["tost5"] else "no")):
            macros[f"{key}.{kk}"] = vv
    lines.append(r"\addlinespace")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(TDIR, "table2_pool447.tex"), "w").write("\n".join(lines) + "\n")

# hard45 pair macros too (used in text)
for e in [e for e in N["pairs"] if e["subset"] == "hard45"]:
    key = f"{KEY[e['model']]}.hard45.{e['pair'].replace('−', 'v')}"
    for kk, vv in (("n", e["n"]), ("b", e["b"]), ("c", e["c"]), ("diff", f"{e['diff_pp']:+.1f}"),
                   ("lo", f"{e['ci95_pp'][0]:+.1f}"), ("hi", f"{e['ci95_pp'][1]:+.1f}"), ("p", fmt_p(e["p_mcnemar"]))):
        macros[f"{key}.{kk}"] = vv

# noise floor / mde / cost / outcome macros
for s in ("hard45", "pool447"):
    F = N["flips"][s]
    # rerun_ladder = the rerun pairs of the four open-weight / vendor-API models (no Opus 5)
    for grp in ("rerun", "rerun_ladder", "harness", "model"):
        if grp not in F:
            continue
        if grp == "rerun_ladder" and s != "hard45":
            continue
        gk = grp.replace("_", ".")
        rs = [e["rate"] for e in F[grp]]
        if rs:
            rs.sort()
            macros[f"flip.{s}.{gk}.n"] = len(rs)
            macros[f"flip.{s}.{gk}.median"] = pct(rs[len(rs) // 2] if len(rs) % 2 else (rs[len(rs)//2 - 1] + rs[len(rs)//2]) / 2)
            macros[f"flip.{s}.{gk}.min"] = pct(rs[0])
            macros[f"flip.{s}.{gk}.max"] = pct(rs[-1])
    if s in N["mde"]:
        M = N["mde"][s]
        macros[f"mde.{s}.disc"] = pct(M["disc_pooled"])
        macros[f"mde.{s}.pp"] = f"{M['mde_pp_at_n']:.0f}"          # asymptotic inversion (kept)
        # exact two-sided McNemar with unconditional power: at n=45 no 80%-power MDE exists,
        # so the hard45 statement moves to a 50% MDE plus the power ceiling.
        macros[f"mde.{s}.pp50"] = f"{M['mde_pp_exact50']:.1f}"
        macros[f"mde.{s}.powermax"] = f"{100 * M['power_max_at_n']:.0f}"
        macros[f"mde.{s}.nmin80"] = f"{M['n_min_for_80']:.0f}"
        e80 = M["mde_pp_exact80"]
        macros[f"mde.{s}.ppexact"] = "--" if e80 != e80 else f"{e80:.1f}"
for m, C in N["cost"].items():
    for h, e in C.items():
        k = f"cost.{KEY[m]}.{HK[h]}"
        macros[f"{k}.cny"] = f"{e['cny_per_trial']:.2f}"
        macros[f"{k}.tokM"] = f"{(e['input_miss'] + e['input_hit'] + e['output'])/1e6:.1f}"
        macros[f"{k}.hitpct"] = pct(e["cache_hit_frac"])
        macros[f"{k}.steps"] = f"{e['steps']:.0f}"
        macros[f"{k}.outk"] = f"{e['output']/1e3:.0f}"
# the vendor's own harness on the same list price (Appendix; kept out of Fig 3's axes)
for m, C in N.get("cost_extra", {}).items():
    for h, e in C.items():
        macros[f"{KEY[m]}.hard45.{HK[h]}.cny"] = f"{e['cny_per_trial']:.2f}"
        macros[f"{KEY[m]}.hard45.{HK[h]}.steps"] = f"{e['steps']:.0f}"
# re-pricing: every hard45 token profile against both vendors' list prices
for m, R in N.get("reprice", {}).items():
    for tag, e in R.items():
        for h, v in e["cny"].items():
            macros[f"reprice.{KEY[m]}.{tag}.{h}.cny"] = f"{v:.2f}"
        for kk in ("CCvmini", "CCvOC"):
            if kk in e:
                macros[f"reprice.{KEY[m]}.{tag}.{kk}"] = f"{e[kk]:.2f}"
# model calls per trial (agent steps) for every canonical arm
for m, SS in N.get("steps", {}).items():
    for s, HH in SS.items():
        for h, e in HH.items():
            macros[f"steps.{KEY[m]}.{s}.{HK[h]}.mean"] = f"{e['mean']:.1f}"
            macros[f"steps.{KEY[m]}.{s}.{HK[h]}.median"] = f"{e['median']:.0f}"
for m, p in N["prices"].items():
    macros[f"price.{KEY[m]}.hit"] = p["hit"]; macros[f"price.{KEY[m]}.miss"] = p["miss"]; macros[f"price.{KEY[m]}.out"] = p["out"]
    macros[f"price.{KEY[m]}.ratio"] = f"{p['hit']/p['miss']:.3f}"
for s in N["outcomes"]:
    for m in N["outcomes"][s]:
        for h, c in N["outcomes"][s][m].items():
            for cat, v in c.items():
                macros[f"out.{KEY[m]}.{s}.{HK[h]}.{cat}"] = v

hdr = [r"% generated by analysis/tab_tables.py from paper/numbers.json -- do not edit",
       r"% usage: \pn{qwen36.hard45.CC.k}   (keys listed below)",
       r"\makeatletter", r"\DeclareRobustCommand{\pn}[1]{\csname pn@#1\endcsname}"]
body = [f"\\expandafter\\def\\csname pn@{k}\\endcsname{{{v}}}" for k, v in sorted(macros.items())]
open(os.path.join(HERE, "..", "paper", "latex", "numbers.tex"), "w").write("\n".join(hdr + body + [r"\makeatother"]) + "\n")
print("tables written;", len(macros), "macros")
