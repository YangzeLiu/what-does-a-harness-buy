#!/usr/bin/env bash
# Regenerate every number, table and figure of the paper that can be derived from the
# shipped trial tables (data/trials.csv, first round; data/round2/trials_round2.csv, second
# round).  Writes into paper/ (created here) and compares the results with reference_outputs/
# (second round: reference_outputs/round2/).  Runs in about two minutes on a laptop CPU.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p paper/latex/tables paper/figures
cd analysis

python3 paper_numbers.py            # -> paper/numbers.json
python3 main_table.py > /dev/null   # main tables + canonical-trial rule
python3 tab_tables.py               # -> Table 1 (pool447), Table 2 (hard45), numbers.tex
python3 tab_controls.py             # -> Table A1 (what was held fixed)
python3 tab_appendix.py             # -> Tables A2, A3, A7
python3 tab_a67.py                  # -> Tables A5, A6, numbers_a67.tex
python3 fig1a_tokens.py; python3 fig1b_forest.py; python3 fig2_noise.py
python3 fig3_cost.py; python3 fig4_outcomes.py; python3 figA3_heatmap.py; python3 figA_mde.py
python3 overflow_bound.py; python3 replicability.py; python3 coverage_audit.py
python3 tb_table.py > ../paper/tb_table.txt   # Terminal-Bench 2.1 appendix

# Not run here, because they need the raw Harbor job trees (to be released separately):
#   extract_trials.py, tab_a4_budget.py, fig5_ctx_growth.py, cutoff_audit.py, b38_*.py,
#   thinking_budget.py, web_leakage_audit.py, noweb_leak_scan_*.py
# Their outputs are shipped in reference_outputs/, analysis/*.txt and analysis/web_leakage.csv.
# Output file names predate the paper's final numbering. README.md maps them to the paper.

# ---- second round (arXiv v2): analysis/round2/, outputs in paper/round2/
cd round2
python3 t1_hard45_table.py > /dev/null      # hard45 pass counts per run, best harness
python3 t2_hard45_pooled.py > /dev/null     # hard45 paired differences, pooled over models (Table rev_pairs)
python3 t3_pool447_pooled.py > /dev/null    # pool447 paired differences, four models (Table rev_pool447)
python3 opus55_pool447.py > /dev/null       # Claude Opus 5.5 on pool447 (391 common tasks)
python3 t4_cost.py > /dev/null              # cost per trial / per solved task (Table rev_cost)
python3 t7_flips.py > /dev/null             # hard45 flip rates with the second round
python3 table_results.py > /dev/null        # every number of Table rev_results
python3 e4b_outcap.py > /dev/null           # larger output cap for OpenCode (Table rev_conditions)
python3 e4a_summary.py > ../../paper/round2/e4a_summary.txt     # equal wall clock for OpenCode
python3 e5_equiv.py > ../../paper/round2/e5_equiv.txt           # equivalence bounds, Qwen rows of Table rev_pool447
python3 ds_pool447_tables.py > ../../paper/round2/ds_pool447_tables.md   # DeepSeek-V4.1-Flash pool447: runs, intervals, flips
python3 ds_pool447_holm.py > ../../paper/round2/ds_pool447_holm.md     # ... Holm-adjusted tests
python3 bench2_report.py                    # SpreadsheetBench Verified and DABstep
python3 fig1_rev.py > ../../paper/round2/fig1_rev.txt           # Figure 1 (+ plotted values)
python3 fig2_rev.py > ../../paper/round2/fig2_rev.txt           # Figure 2 (+ plotted values)
python3 figA_budget_rev.py > ../../paper/round2/figA_budget_rev.txt   # equal-budget figure (+ values)
python3 figA3_heatmap_rev.py; python3 figA_mde_rev.py; python3 figA_outcomes_rev.py
cd ..
# Not regenerated here: the edited-file similarity (needs raw trajectories; no output shipped),
# figA_opus391.pdf (drawn with headless Chromium from opus55_pool447_391.json), and the
# hand-assembled tables reference_outputs/round2/tables/rev_*.tex.

cd ..
echo "--- comparison with reference_outputs/ ---"
status=0
python3 - <<'EOF' || status=1
import json, sys
a = json.load(open("paper/numbers.json")); b = json.load(open("reference_outputs/numbers.json"))
for d in (a, b):
    d.get("meta", {}).pop("generated", None)
ok = a == b
print(("identical  " if ok else "DIFFERS    ") + "numbers.json (ignoring meta.generated)")
sys.exit(0 if ok else 1)
EOF
for f in numbers numbers_a67 numbers_ovf numbers_rep; do
  cmp -s "paper/latex/$f.tex" "reference_outputs/$f.tex" && echo "identical  $f.tex" || { echo "DIFFERS    $f.tex"; status=1; }
done
for f in reference_outputs/tables/*.tex; do
  b=$(basename "$f")
  if [ -f "paper/latex/tables/$b" ]; then
    cmp -s "paper/latex/tables/$b" "$f" && echo "identical  tables/$b" || { echo "DIFFERS    tables/$b"; status=1; }
  else echo "not regenerated (needs raw trajectories)  tables/$b"; fi
done
cmp -s paper/tb_table.txt analysis/tb_table.txt && echo "identical  analysis/tb_table.txt" || { echo "DIFFERS    analysis/tb_table.txt"; status=1; }
python3 - <<'EOF' || status=1
# Figure PDFs embed their creation time, so they are compared with /CreationDate removed.
import os, re, sys
bad = 0
for b in sorted(os.listdir("reference_outputs/figures")):
    new = os.path.join("paper/figures", b)
    if not os.path.exists(new):
        print("not regenerated (needs raw trajectories)  figures/" + b)
        continue
    a, r = (re.sub(rb"/CreationDate \([^)]*\)", b"", open(p, "rb").read())
            for p in (new, os.path.join("reference_outputs/figures", b)))
    print(("identical  " if a == r else "DIFFERS    ") + "figures/" + b + " (ignoring /CreationDate)")
    bad |= a != r
sys.exit(bad)
EOF
echo "--- second round: comparison with reference_outputs/round2/ ---"
for f in reference_outputs/round2/*.txt reference_outputs/round2/*.md reference_outputs/round2/*.json; do
  b=$(basename "$f")
  if [ -f "paper/round2/$b" ]; then
    cmp -s "paper/round2/$b" "$f" && echo "identical  round2/$b" || { echo "DIFFERS    round2/$b"; status=1; }
  else echo "not regenerated  round2/$b"; status=1; fi
done
for f in reference_outputs/round2/tables/*.tex; do echo "not regenerated (assembled by hand from the outputs above)  round2/tables/$(basename "$f")"; done
python3 - <<'EOF' || status=1
import os, re, sys
bad = 0
for b in sorted(os.listdir("reference_outputs/round2/figures")):
    new = os.path.join("paper/round2/figures", b)
    if not os.path.exists(new):
        why = "drawn with headless Chromium" if b == "figA_opus391.pdf" else "missing"
        print(f"not regenerated ({why})  round2/figures/" + b)
        bad |= why == "missing"
        continue
    a, r = (re.sub(rb"/CreationDate \([^)]*\)", b"", open(p, "rb").read())
            for p in (new, os.path.join("reference_outputs/round2/figures", b)))
    print(("identical  " if a == r else "DIFFERS    ") + "round2/figures/" + b + " (ignoring /CreationDate)")
    bad |= a != r
sys.exit(bad)
EOF
exit $status
