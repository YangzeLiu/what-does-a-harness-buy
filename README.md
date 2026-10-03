# What Does a Harness Buy? Tokens, Mostly.

Code and data for the paper "What Does a Harness Buy? Tokens, Mostly." by Yangze Liu and Zhongyi Han (Shandong University). The arXiv identifier will be added here once it is assigned.

## Summary

A coding agent is a language model wrapped in a harness: the system prompt, the tool set, and the context management that let a chat model work inside a repository. The paper measures how much the harness moves the score when the model is held fixed. It runs five models through three production harnesses, Claude Code, mini-SWE-agent and OpenCode, on SWE-bench Verified, and reruns the same configurations to calibrate how much a score moves when nothing changes but the run.

On 447 tasks and the two models run there, Claude Code and mini-SWE-agent are equivalent within five points. On a 45-task hard subset and five models, swapping the harness flips as many tasks as rerunning the same harness, 13% in both cases. The one harness effect that clears the noise is a loss: OpenCode trails by up to 9 points on the large pool, and on one model half of that gap sits in runs its output cap cut short. What the harness does decide is the bill. With the same model, the same tasks and one price list, cost per task differs by up to 3x across harnesses. The gap is set by the preamble of system prompt and tool schemas that each harness sends with every step, scaled by the number of steps. The rerun data also give the resolution a harness comparison needs: at the observed discordance, 45 tasks catch a 13-point gap only half the time, and 447 tasks resolve 5 points.

Every number, table and figure in the paper is computed from one table of trials, `data/trials.csv`, by the scripts in `analysis/`.

## Quickstart

```
git clone https://github.com/YangzeLiu/what-does-a-harness-buy.git
cd what-does-a-harness-buy
pip install numpy matplotlib scipy
bash reproduce.sh
```

`reproduce.sh` needs no GPU and no network and finishes in under a minute on a laptop CPU. It writes into `paper/` (created on the first run) and compares every regenerated file with `reference_outputs/`, which holds the files the paper was built from. Each line of the comparison reads `identical`, `DIFFERS`, or `not regenerated (needs raw trajectories)`, and the script exits non-zero if anything differs. Figure PDFs are compared with their embedded creation date removed.

The release was checked with Python 3.10.12, numpy 1.26.4, matplotlib 3.9.0 and scipy 1.14.0. With these versions every regenerated `numbers*.tex`, table and figure is identical to `reference_outputs/`, and those files match the arXiv sources apart from the header comment lines that name the generating script. scipy is optional and only cross-checks the exact McNemar test. Without it, every `.tex` file and figure is still identical, while `paper/numbers.json` differs in the 17th significant digit of one p-value and `tb_table.txt` differs in the line that reports the scipy status.

## Repository layout

| path | contents |
|---|---|
| `data/trials.csv`, `data/trials.json` | the trial table, 8,032 trials by 50 columns, one row per trial |
| `data/remote/` | trial rows extracted on other hosts, merged into the table by `extract_trials.py --merge-json` (see below) |
| `analysis/` | the scripts that compute every number, table and figure, and the shipped outputs of the scripts that need raw trajectories |
| `jobs/` | the Harbor job configuration of each of the 150 jobs, with credentials replaced by `<REDACTED>` |
| `agents/dsh_agent.py` | Harbor adapter for DeepSeek's own harness (Appendix B) |
| `benchmarks/` | task pools and the subset-selection script |
| `configs/` | chat templates used to serve the local Qwen models with thinking off |
| `scripts/` | patches applied to the Harbor task directories and harness wrappers, task lists, and job generators |
| `reference_outputs/` | the generated `\pn{}` macro files, tables, figures and JSON summaries that the paper was built from |
| `reproduce.sh` | regenerates everything that can be derived from `data/trials.csv` and compares it with `reference_outputs/` |

### The trial table

`analysis/extract_trials.py` builds `data/trials.csv` from the raw Harbor job directories. Its docstring defines every column. The main ones are `job`, `harness`, `model`, `subset`, `arm`, `repeat` and `attempt`, which identify a trial, and `status`, `scored`, `reward` and `exception_type`, which give its outcome, followed by token, step, tool-call and timing counts. The table also contains trials of arms the paper does not report, such as pilot runs and arms stopped for quota or infrastructure reasons, so that the coverage audit (`analysis/coverage_audit.py`) accounts for every trial. The analysis scripts select the reported arms.

Each table cell uses one trial per task and run. A trial lost to one of 14 infrastructure exception types is void and leaves the denominator, and a trial that ends at a step or time limit is scored as a failure. Where a task was retried within a run, a usable trial beats a void one and ties go to the latest finished trial. `analysis/main_table.py` (`canonical()`) implements this rule and `analysis/agg.py` shares the infrastructure set.

`data/remote/host-b_trials.json` and `data/remote/host-b_b36_trials.json` are the rows of jobs that ran on a second host and were extracted there. `data/remote/local_lost_jobs_20260911.json` holds the rows of eight Claude Opus 5 jobs whose raw job directories were not preserved after extraction, so these rows cannot be re-extracted. All three files are already merged into `data/trials.csv`.

### Task pools

| file | tasks | definition |
|---|---|---|
| `benchmarks/pool492.txt` | 492 | SWE-bench Verified without the 8 `psf/requests` tasks |
| `benchmarks/hard45.txt` | 45 | the pool tasks in the difficulty bands "1-4 hours" (42) and ">4 hours" (3) |
| `benchmarks/pool447_rest.txt` | 447 | `pool492.txt` without `hard45.txt` |

`benchmarks/select_subset.py` derives the pool from the SWE-bench Verified metadata on the Hugging Face hub (`princeton-nlp/SWE-bench_Verified`), and `benchmarks/selection_protocol.md` records its rules. `main150.json` and `pilot30.json` come from an earlier stratified design and are kept because the pilot runs in `data/trials.csv` used them.

### Job files

The job files in `jobs/` are the configurations Harbor 0.22.0 ran, with credentials redacted and machine-specific paths made relative. Task paths point to `tasks_pool/<task>` and `tasks_noweb/<task>`, and `jobs_dir` is `results/raw/jobs`. `tasks_pool/` holds the Harbor SWE-bench task directories after the patches in `scripts/patch_*.py`. `tasks_noweb/` holds copies of the 45 hard45 task directories and of one smoke-test task, whose `task.toml` adds an agent-phase network allowlist, the same edit that `scripts/tb21_patch_noweb.py` makes for Terminal-Bench 2.1. Ten Claude Code job files on Claude Opus 5 attach a Docker Compose override, `noweb_compose.yaml`, which is not included. The addresses `172.17.0.1` and `172.30.0.1` are Docker bridge gateways through which the containers reached the model server, a package mirror and the egress proxy on the run host. The label `host-a` in job and provider names and `host-b` in `data/remote/` stand for the machines the runs were split across.

## Reproduction map

Output file names predate the paper's final numbering, so the map gives both. Paths under "output" are relative to `paper/latex/` for `.tex` files and `paper/figures/` for PDFs. Scripts marked "raw" need the raw trajectories, which will be released separately. Their outputs are shipped in `reference_outputs/` or `analysis/`.

| paper item | script(s) in `analysis/` | output | input |
|---|---|---|---|
| in-text numbers (`\pn{}` macros) | `paper_numbers.py`, `tab_tables.py` | `paper/numbers.json`, `numbers.tex` | trials.csv |
| Table 1, paired comparison on pool447 | `tab_tables.py` (rule from `main_table.py`) | `tables/table2_pool447.tex` | trials.csv |
| Table 2, pass rate on hard45 | `tab_tables.py` | `tables/table1_hard45.tex` | trials.csv |
| Figure 1a, 1b | `fig1a_tokens.py`, `fig1b_forest.py` | `fig1a_tokens.pdf`, `fig1b_forest.pdf` | trials.csv |
| Figure 2, flip rates | `fig2_noise.py` | `fig2_noise.pdf` | trials.csv |
| Figure 3, how trials end | `fig4_outcomes.py` | `fig4_outcomes.pdf` | trials.csv |
| Figure 4, input tokens per step | `fig5_ctx_growth.py` | `fig5_ctx_growth.pdf`, `numbers_ctx.tex`, `ctx_growth.json` | raw |
| output-cap share of the OpenCode gap (Results, Appendix K) | `overflow_bound.py` | `numbers_ovf.tex`, `overflow_bound.json` | trials.csv |
| Table A1, what was held fixed (Appendix A) | `tab_controls.py` | `tables/table3_controls.tex` | trials.csv |
| content-level audit of the offline arms (Appendix A) | `noweb_leak_scan_cc.py`, `noweb_leak_scan_oc_mini.py` | stdout | raw |
| DeepSeek's own harness (Appendix B) | `paper_numbers.py`, `tab_tables.py` | `numbers.tex` | trials.csv |
| Tables A2 and A3, denominators and coverage (Appendix C) | `tab_appendix.py`, `coverage_audit.py` | `tables/tableA2_denominators.tex`, `tables/tableA1_coverage.tex` | trials.csv |
| leakage audit of the web-enabled runs (Appendix D) | `web_leakage_audit.py` | `analysis/web_leakage.csv` | raw |
| Figure A1, per-task outcomes (Appendix E) | `figA3_heatmap.py` | `figA3_heatmap.pdf` | trials.csv |
| Figure A2 and rerun statistics (Results, Appendix F) | `figA_mde.py`, `replicability.py` | `figA_mde.pdf`, `numbers_rep.tex`, `replicability.json` | trials.csv |
| Figure A3, where the bill comes from (Appendix G) | `fig3_cost.py` | `fig3_cost.pdf` | trials.csv |
| Table A4, per-step input budget (Appendix G) | `tab_a4_budget.py` | `tables/tableA4_budget.tex`, `numbers_a4.tex` | raw |
| output cutoff counts (Appendix H) | `cutoff_audit.py` | `analysis/cutoff_audit.txt` | raw |
| recovery transplant (Appendix H) | `b38_paired.py`, `b38_trigger.py`, `b38_trigger_stability.py` | `analysis/b38_paired.txt`, `analysis/b38_trigger_stability.txt` | raw |
| Table A5, Claude Code settings on Claude Opus 5 (Appendix I) | `tab_a67.py` | `tables/tableA6_opus.tex`, `numbers_a67.tex` | trials.csv |
| Table A6, thinking on and off (Appendix J) | `tab_a67.py`, which reads `thinking_budget.txt` from `thinking_budget.py` | `tables/tableA7_thinking.tex` | trials.csv, raw |
| Table A7, sensitivity (Appendix K) | `tab_appendix.py` | `tables/tableA10_sensitivity.tex` | trials.csv |
| Terminal-Bench 2.1 transfer check (Appendix L) | `tb_table.py` | `analysis/tb_table.txt` | trials.csv |

`main_table.py` prints the main tables and the canonical-trial bookkeeping to stdout, `agg.py` holds shared aggregation helpers, and `figstyle.py` the figure style. `extract_trials.py` (raw) rebuilds `data/trials.csv`.

## Running the scripts that need raw trajectories

These scripts read Harbor job directories (`<jobs_root>/<job>/<task>__<trial>/`). Set the paths through environment variables or arguments.

| variable | used by | meaning |
|---|---|---|
| `HARNESS_JOBS_ROOT` | `extract_trials.py`, `tab_a4_budget.py`, `fig5_ctx_growth.py`, `web_leakage_audit.py`, `b38_paired.py`, `scripts/gen_tb21_job.py` | Harbor jobs directory, default `results/raw/jobs` |
| `HARNESS_QWEN_TOKENIZER` | `thinking_budget.py`, `b38_trigger_stability.py` | path to a Qwen `tokenizer.json` (or pass `--tokenizer`) |
| `TRIALS` | `agg.py` | trial table in JSON, default `data/trials.json` |
| `HF_ENDPOINT`, `HF_DATASETS_OFFLINE` | `benchmarks/select_subset.py` | Hugging Face mirror, or `HF_DATASETS_OFFLINE=1` to read a local cache |

`cutoff_audit.py`, `b38_trigger.py`, `b38_trigger_stability.py`, `thinking_budget.py` and the two `noweb_leak_scan_*.py` scripts take the jobs root as their first argument. Each script's docstring gives its usage.

## What is not included

The full trajectories of the mini-SWE-agent and OpenCode runs, and the tool-call sequences and results of the Claude Code runs, will be released separately. The raw Harbor job trees they come from are needed by the scripts marked "raw" above. The task directories (`tasks_pool/`, `tasks_noweb/`), the Docker images, the egress proxy and `noweb_compose.yaml` are not included. Harbor generates the task directories from SWE-bench Verified, and `scripts/` contains the patches applied to them. Model weights for Qwen3.6-35B-A3B and Qwen3.8-27B are available from their publishers, and the paper's Setup section and Appendix A give the serving configuration.

Rerunning the agents needs Harbor 0.22.0, Docker, the harness versions given in the paper (Claude Code 2.1.259 to 2.1.273, mini-SWE-agent 2.4.6, OpenCode 1.18.30 and 1.18.31), model API access, and vLLM 0.27.1 in bfloat16 for the local models.

## Random seeds

Task subset selection uses numpy seed 20260901 (`benchmarks/select_subset.py`). The permutation tests in `analysis/replicability.py` use seed 20260919, and the jitter in `analysis/fig2_noise.py` uses seed 3. Agent runs are numbered by the `repeat` column, or, for the Qwen3.8-27B hard45 runs that pack two attempts per task into one job, by the `attempt` column.

## License

The code in this repository (`*.py`, `*.sh`) is released under the MIT License, see `LICENSE`. The chat templates in `configs/` are derived from the templates shipped with the Qwen model weights and remain under the license of those releases. The data files (`data/`, `jobs/`, `benchmarks/*.txt`, `benchmarks/*.json`, `benchmarks/selection_protocol.md`, `reference_outputs/`, `analysis/*.txt`, `analysis/web_leakage.csv`, `scripts/*.txt`) are released under the Creative Commons Attribution 4.0 International License (CC BY 4.0). The SWE-bench Verified task ids come from SWE-bench, which is released under the MIT License.

## Citation

```bibtex
@misc{liu2026harness,
  title  = {What Does a Harness Buy? {T}okens, Mostly.},
  author = {Liu, Yangze and Han, Zhongyi},
  year   = {2026},
  note   = {arXiv preprint, identifier to be added}
}
```

`CITATION.cff` carries the same entry for GitHub's citation widget.
