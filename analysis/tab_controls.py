#!/usr/bin/env python3
"""Table 3: control variables per harness.  Every cell was verified read-only on the serving
host and against data/trials.csv on 2026-09-17.
Writes paper/latex/tables/table3_controls.tex."""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.normpath(os.path.join(HERE, "..", "paper", "latex", "tables", "table3_controls.tex"))

ROWS = [
    ("Version (pinned per arm)$^{a}$",
     r"2.1.259--2.1.273 (2.1.263 on GLM and Opus)", r"2.4.6", r"1.18.30--1.18.31 (1.18.30 on GLM)"),
    ("Sampling",
     r"none sent (no flag)", r"none sent", r"none sent"),
    ("Thinking / effort",
     r"GLM: effort \texttt{high} (CLI flag); Qwen: effort \texttt{xhigh} via env (vLLM compat., a no-op)$^{b}$; DS: provider default",
     r"GLM: \texttt{reasoning\_effort} = \texttt{high} (request body); Qwen/DS: default",
     r"GLM: \texttt{reasoningEffort} = \texttt{high} (model config); Qwen/DS: default"),
    ("Tools",
     r"built-in set (Bash, Read, Edit, Write, Grep\ldots); \texttt{WebSearch} disallowed",
     r"\texttt{bash} only",
     r"bash, read, edit, write, glob, grep, webfetch, task; ripgrep pre-placed (rg arms)"),
    ("Context management",
     r"auto-compaction: 110k window (env override) on the Qwen arms, client default elsewhere$^{c}$",
     r"none; per-observation head/tail trim (10k chars); hard stop at 131k",
     r"prune (40k tool-output tail) + auto-compaction at \texttt{limit.input} minus a 20k reserve (78k); input limit 98k, output limit 32k"),
    ("Termination",
     r"\texttt{--max-turns 300}; soft timeout (exit 124)",
     r"\texttt{step\_limit=300}; no soft timeout",
     r"\texttt{build.steps=300}; soft timeout (exit 124)"),
    ("Wall clock (safety net)$^{d}$",
     r"Qwen arms: 12$\times$ the task's agent timeout; vendor-API arms: the batch's single soft timeout",
     r"Qwen arms: 6$\times$ the task's agent timeout; vendor-API arms: the same soft timeout",
     r"Qwen arms: 6$\times$ the task's agent timeout; vendor-API arms: the same soft timeout"),
    ("Network (agent phase)",
     r"allowlist: model API host + 172.17.0.1", r"same", r"same"),
    ("Runner",
     r"Harbor 0.22.0, one container per trial, verifier after the agent exits", r"same", r"same"),
]

lines = [r"\footnotesize\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}>{\raggedright\arraybackslash}p{2.2cm}>{\raggedright\arraybackslash}p{3.9cm}>{\raggedright\arraybackslash}p{3.3cm}>{\raggedright\arraybackslash}p{3.7cm}@{}}", r"\toprule",
         r" & Claude Code & mini-SWE-agent & OpenCode \\", r"\midrule"]
for name, a, b, c in ROWS:
    lines.append(f"{name} & {a} & {b} & {c} \\\\")
lines += [r"\bottomrule",
          r"\multicolumn{4}{@{}p{\linewidth}@{}}{\footnotesize "
          r"$^{a}$~the versions of the arms scored in Tables~\ref{tab:hard45} and \ref{tab:pool447}, "
          r"read off every scored trial; OpenCode 1.18.25--1.18.29 appear only in the as-shipped "
          r"OpenCode arm of Table~\ref{tab:sensitivity} and in the ablations.  "
          r"$^{b}$~a compatibility setting, not a treatment: \texttt{xhigh} is the Qwen chat "
          r"template's own default, and vLLM's Anthropic-compatible \texttt{/v1/messages} endpoint "
          r"ignores \texttt{effort} altogether (verified by a single-prompt probe against the same serving stack); "
          r"Claude Code's reasoning per step is within 3--6\% of OpenCode's on the same model, and "
          r"OpenCode sends no effort at all, so all three harnesses see the same thinking budget on "
          r"the Qwen arms.  "
          r"$^{c}$~\texttt{CLAUDE\_CODE\_AUTO\_COMPACT\_WINDOW}${=}$110000 is set on the Qwen arms "
          r"only, where it fires at a measured median of 78k tokens and so matches OpenCode by "
          r"construction; on the vendor-API arms Claude Code keeps its own default window and "
          r"compacts much later (on GLM-5.3-Flash at a median of \pn{budget.cc.compact.pre.k}k "
          r"prompt tokens, Table~\ref{tab:budget}).  "
          r"$^{d}$~on the locally served Qwen arms the limit is a multiple of the task's agent "
          r"timeout, six times for mini-SWE-agent and OpenCode and twelve for Claude Code; on the "
          r"vendor-API arms (DeepSeek-V4-Flash, GLM-5.3-Flash, Opus~5) one soft timeout applies to "
          r"all three harnesses, 2{,}760\,s on DeepSeek and Opus and 5{,}760\,s on GLM.} \\",
          r"\end{tabular}"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
open(OUT, "w").write("\n".join(lines) + "\n")
print("wrote", OUT)
