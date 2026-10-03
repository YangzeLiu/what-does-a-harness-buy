#!/usr/bin/env python3
"""
Walk Harbor job directories and emit one row per trial (data/trials.csv + .json).

Layout discovered 2026-09-02 (Harbor 0.x, agents opencode / claude-code):

  jobs/<job>/config.json, result.json (job-level stats only, no per-trial reward)
  jobs/<job>/<task>__<trialid>/
      result.json             per-trial; reward at verifier_result.rewards.reward,
                              exception_info when the trial errored, timestamps for
                              agent_execution/verifier.  ABSENT while the trial runs.
      verifier/reward.txt     same reward as a bare number (fallback)
      agent/trajectory.json   ATIF-v1.7; written only when the agent finished
      agent/opencode.txt      OpenCode JSONL events (step_finish carries tokens)
      agent/claude-code.txt   Claude Code --output-format=stream-json JSONL

Metric definitions:
  steps             ATIF final_metrics.total_steps (includes the initial user step)
  prompt_tokens     sum of per-step prompt_tokens (CC: input+cache_creation+cache_read)
  completion_tokens sum of per-step completion_tokens (OpenCode: EXCLUDES reasoning
                    tokens; CC output_tokens INCLUDES thinking) -> reasoning_tokens column
  peak_ctx          max per-step prompt_tokens over main-thread agent steps
  n_compaction      context-drop heuristic: consecutive agent steps with prev > 20k and
                    cur < 0.6*prev (harness-agnostic; the only rule available on OpenCode
                    ATIF).  n_compaction_log counts explicit log markers: OpenCode
                    synthetic part with metadata.compaction_continue=true; Claude Code
                    system event whose subtype contains "compact" (compact_boundary).
  touch_<tool>      trial made >=1 successful call of that tool (case-insensitive;
                    OpenCode 'invalid' calls, i.e. hallucinated unavailable tools, are
                    counted separately in n_invalid_tool and tool_counts as invalid:<name>)
  hard_failure      agent-side crash: result.json exception (other than the verifier's
                    RewardFileNotFoundError), CC result event is_error, context-overflow
                    string in the log, or a finished trial without trajectory.json.

Running trials (no result.json) are parsed from the raw log so partial steps/tokens/tool
counts are available; status='running', scored=False.  CC raw-log completion_tokens are
left empty because stream-json assistant events only carry a first-block snapshot.

Usage: python3 analysis/extract_trials.py [--jobs-root DIR] [--out-dir data]
           [--merge-json data/remote/*.json]
The jobs root is the Harbor jobs directory (default $HARNESS_JOBS_ROOT, else
results/raw/jobs relative to the current directory).  It needs the raw trajectories,
which are not part of this repository.  Jobs that ran on a second host and were not
mirrored locally enter through --merge-json (their extracted rows are in data/remote/).
"""
import argparse
import collections
import csv
import fnmatch
import glob
import json
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone

DEFAULT_JOBS_ROOT = os.path.join("results", "raw", "jobs")  # relative to the current directory

# job_name -> arm descriptor.  Jobs that share a descriptor are retries/remainders of the
# same arm and are merged per task by the analysis scripts (main_table.canonical()).
# 'exclude' marks jobs whose trials are all infrastructure failures or smoke tests; they
# are kept in trials.csv but skipped by the analysis.
def _a(harness, model, subset, arm, repeat=1, exclude=False, note=""):
    return dict(harness=harness, model=model, subset=subset, arm=arm, repeat=repeat,
                exclude=exclude, note=note)

OC, CC, MINI, DSH = "opencode", "claude-code", "mini-swe-agent", "dsh"
DS, OPUS, F5, F51, QWEN = "ds-v4-flash", "opus5", "fable5", "fable51", "qwen3.8-27b"
QWEN36, DSPRO, GLMF = "qwen3.6-35b-a3b", "ds-v4-pro", "glm-5.3-flash"
HY4 = "hy4-preview"

JOB_ARMS = {
    # DS-V4-Flash x OpenCode, pilot30 control (+ retries of infra failures)
    "ds-flash-pilot-b1":        _a(OC, DS, "pilot30", "control"),
    "ds-flash-pilot-b1-retry":  _a(OC, DS, "pilot30", "control", note="retry of infra failures"),
    "ds-flash-pilot-b1-retry2": _a(OC, DS, "pilot30", "control", note="retry of infra failures"),
    # DS-V4-Flash x OpenCode, hard45 control (-r2 = remainder after the 01:25 kill)
    "ds-flash-hard45":          _a(OC, DS, "hard45", "control"),
    "ds-flash-hard45-r2":       _a(OC, DS, "hard45", "control", note="remainder batch"),
    # DS-V4-Flash x OpenCode config-only ablation arms on the Fable-15 subset
    "abl-c1off":                _a(OC, DS, "fable15", "c1off"),
    "abl-c2-bashonly":          _a(OC, DS, "fable15", "c2-bashonly"),
    "abl-c2-r2":                _a(OC, DS, "fable15", "c2-bashonly", note="retry"),
    "abl-c2-r3":                _a(OC, DS, "fable15", "c2-bashonly", note="retry"),
    "abl-c3-notask":            _a(OC, DS, "fable15", "c3-notask"),
    "abl-c7-notodo":            _a(OC, DS, "fable15", "c7-notodo"),
    "abl-c7-retry":             _a(OC, DS, "fable15", "c7-notodo", note="retry"),
    "abl-ctx40k":               _a(OC, DS, "fable15", "ctx40k"),
    "abl-ctx40k.crashed1":      _a(OC, DS, "fable15", "ctx40k", exclude=True,
                                   note="config validation crash (missing limit.output)"),
    # Opus 5 x Claude Code
    "cc-opus-pilot-b1":         _a(CC, OPUS, "pilot30", "control"),
    "cc-opus-pilot-b2":         _a(CC, OPUS, "pilot30", "control", repeat=2),
    "cc-opus-pilot-b2r":        _a(CC, OPUS, "pilot30", "control", repeat=2, note="remainder"),
    "cc-opus-pilot-b2r2":       _a(CC, OPUS, "pilot30", "control", repeat=2, note="remainder"),
    "cc-opus-pilot-b3":         _a(CC, OPUS, "pilot30", "control", repeat=3),
    "cc-opus-hard45":           _a(CC, OPUS, "hard45", "control"),
    "cc-opus-hard45-r2":        _a(CC, OPUS, "hard45", "control", note="remainder batch"),
    "cc-opus-hard45.tok401":    _a(CC, OPUS, "hard45", "control", exclude=True,
                                   note="OAuth 401 batch"),
    # Fable 5 x Claude Code (15-task stratified subsample)
    "cc-fable-pilot-b1":        _a(CC, F5, "fable15", "control"),
    "cc-fable-retry":           _a(CC, F5, "fable15", "control", note="retry of 401 failures"),
    # Fable 5.1 x Claude Code
    "cc-fable51-pilot30":       _a(CC, F51, "pilot30", "control"),
    "cc-fable51-hard45":        _a(CC, F51, "hard45", "control"),
    "cc-fable51-hard45-c2":     _a(CC, F51, "hard45", "c2-bashonly",
                                   note="--disallowedTools Edit,Write,MultiEdit,NotebookEdit"),
    "cc-fable51-hard45-ctx50k": _a(CC, F51, "hard45", "ctx50k",
                                   note="AUTO_COMPACT_WINDOW=100000 + AUTOCOMPACT_PCT=50"),
    "cc-fable51-hard45-r":      _a(CC, F51, "hard45", "control", note="retry of 429 session-limit failures"),
    "cc-fable51-hard45-c2-r":   _a(CC, F51, "hard45", "c2-bashonly", note="retry of 429 failures"),
    "cc-fable51-hard45-ctx50k-r": _a(CC, F51, "hard45", "ctx50k", note="retry of 429 failures"),
    "cc-fable51-pilot30-r":     _a(CC, F51, "pilot30", "control", note="retry of 429 failures"),
    # Qwen local (OpenCode, vLLM on host-a/host-b)
    "qwen-smoke":               _a(OC, QWEN, "smoke", "control", note="2-task smoke"),
    "qwen-pilot30":             _a(OC, QWEN, "pilot30", "control"),
    "qwen-pilot30-r":           _a(OC, QWEN, "pilot30", "control", note="retry of ContextOverflow exit-1 trials"),
    "qwen-p30-ctx40k":          _a(OC, QWEN, "pilot30", "ctx40k",
                                   note="limit.input=40000 + compaction.reserved=8000 (usable 32k), output kept 32768"),
    "qwen-p30-ctx40k.thrash-config": _a(OC, QWEN, "pilot30", "ctx40k", exclude=True,
                                   note="killed: context 40000/output 32768 gave usable 8k"),
    "qwen-p30-c2":              _a(OC, QWEN, "pilot30", "c2-bashonly", note="permission edit/write deny"),
    "qwen-hard45":              _a(OC, QWEN, "hard45", "control", note="k=2 attempts, conc 8, host-a 8002 TP=4"),
    "qwen-hard45-r":            _a(OC, QWEN, "hard45", "control", note="host-a retry of hard-failed attempts (timeout/overflow/OOM-137), 1 attempt per task"),
    "qwen-hard45-r2x":          _a(OC, QWEN, "hard45", "control", note="host-a retry, tasks with both attempts hard-failed, 2 attempts"),
    # same-model harness ladder on hard45: mini-swe-agent -> OpenCode -> Claude Code, all DS-V4-Flash
    # (CC + mini reach DS via api.deepseek.com/anthropic resp. litellm deepseek/; key from the environment)
    "cc-ds-hard45":             _a(CC, DS, "hard45", "control", note="conc 3, ANTHROPIC_BASE_URL=api.deepseek.com/anthropic, adaptive thinking on"),
    "mini-ds-hard45":           _a(MINI, DS, "hard45", "control", note="conc 3, litellm deepseek/deepseek-v4-flash, DS default reasoning"),
    "mini-ds-hard45-r":         _a(MINI, DS, "hard45", "control", note="retry of 2 infra failures (13837 verifier collision, 14631 setup timeout)"),
    # no-web reruns on host-a (task.toml [agent] network_mode=allowlist -> only the model API + docker gateway;
    # setup/verifier phases stay public). Supersede the web-contaminated DS/Qwen controls above.
    "qwen-hard45-nw":           _a(OC, QWEN, "hard45", "control-nw", note="k=2 attempts, conc 20, limit.input 98000, agent_timeout x2, host-a 8002 TP=4"),
    "mini-ds-hard45-nw":        _a(MINI, DS, "hard45", "control-nw", note="conc 6, host-a, allowlist egress"),
    "oc-ds-hard45-nw":          _a(OC, DS, "hard45", "control-nw", note="conc 6, host-a, allowlist egress; OpenCode-DS control rerun"),
    "qwen-hard45-nw-r":         _a(OC, QWEN, "hard45", "control-nw", note="retry of the one OOM-137 (exit 137) attempt, django-13128, 1 attempt"),
    "qwen-nw-smoke":            _a(OC, QWEN, "hard45", "control-nw", exclude=True, note="host-a no-web smoke"),
    "mini-ds-nw-smoke":         _a(MINI, DS, "hard45", "control-nw", exclude=True, note="host-a no-web smoke"),
    "oc-ds-nw-smoke":           _a(OC, DS, "hard45", "control-nw", exclude=True, note="host-a no-web smoke"),
    "cc-ds-hard45-nw":          _a(CC, DS, "hard45", "control-nw", note="conc 6, host-a, allowlist egress; CC-DS control rerun (ladder top rung)"),
    "cc-ds-nw-smoke":           _a(CC, DS, "hard45", "control-nw", exclude=True, note="host-a no-web smoke"),
    "mini-ds-hard45-nw-r":      _a(MINI, DS, "hard45", "control-nw", note="retry of 6 AgentSetupTimeoutError trials (uv install hang 1080s) from mini-ds-hard45-nw; setup timeout x6, 1 attempt"),
    "cc-ds-hard45-nw.leak":     _a(CC, DS, "hard45", "control-nw", exclude=True, note="09-03 18:02-18:51 first launch, stopped at 10% ckpt: CC WebSearch (server-side tool via api.deepseek.com, allowlisted) returned live results incl. upstream fix diffs; 3 complete/9 started; kept as leak evidence"),
    "cc-ds-nw-smoke2":          _a(CC, DS, "hard45", "control-nw", exclude=True, note="host-a no-web smoke #2 on django-11400 with --disallowedTools WebSearch"),
    # step-budget protocol (2026-09-05): uniform 300-step cap (mini agent.step_limit / OpenCode agent.build.steps /
    # CC --max-turns), wall-clock only as a x6 safety net (5 h). Retries below replace the wall-clock-timeout fails.
    "cc-ds-hard45-nw-r":        _a(CC, DS, "hard45", "control-nw", note="step-budget retry of the 2 AgentTimeoutError fails of cc-ds-hard45-nw (django-10554, 16263): --max-turns 300, wall x6, 1 attempt"),
    "qwen-hard45-nw-r2":        _a(OC, QWEN, "hard45", "control-nw", note="step-budget retry of the 6 failed AgentTimeoutError attempts of qwen-hard45-nw: agent.build.steps 300, wall x6, 1 attempt each"),
    "oc-qwen-hard45-overlap-nw": _a(OC, QWEN, "hard45", "control-nw", note="host-b rerun of the 6 qwen-hard45-nw attempts voided as AgentVerifierOverlap (11885/13344/15957/16263/xarray-6992/sympy-13878), 1 attempt each, original config (x2 timeout, no steps cap) + soft timeout 5760; queued after the host-b pool"),
    # Qwen x {mini, CC} no-web hard45 (same vLLM 8002 TP=4, 131072 max-model-len as the Qwen OpenCode arm)
    "mini-qwen-hard45-nw":      _a(MINI, QWEN, "hard45", "control-nw", note="k=2, conc 8, litellm openai/ -> host TLS front https://172.17.0.1:8443 -> vLLM (Harbor egress gost kills plain-HTTP non-streaming responses >15s; TLS relayed byte-wise), SSL_VERIFY=false (no compaction in mini -> 131k hard ceiling), step_limit 300, wall x6"),
    "mini-qwen-hard45-nw-r":      _a(MINI, QWEN, "hard45", "control-nw", note="redo of the ONE unscored mini-qwen attempt (django-15503 attempt 1: agent overflow-exit, then verifier hung 3000 s and the container was OOM-killed by orphaned parallel Django test workers); k=1, conc 1, same mini config / TLS front / step_limit 300 / wall x6; merge as per-attempt replacement"),
    "cc-qwen-hard45-nw":        _a(CC, QWEN, "hard45", "control-nw", note="k=2, conc 8, vLLM native /v1/messages, CLAUDE_CODE_EFFORT_LEVEL=xhigh (Qwen3.8 template default; CC default effort high is rejected -> 500), CLAUDE_CODE_AUTO_COMPACT_WINDOW=110000 (~78k compaction = OpenCode limit.input 98000 x 0.8 parity), --max-turns 300, wall x12 (relaunched 09-05 after mini finished: first launch shared vLLM with mini -> ~2 min/step, 5 h net would bind before 300 steps; ran alone at conc 8), WebSearch disallowed"),
    "cc-qwen-hard45-nw-r":      _a(CC, QWEN, "hard45", "control-nw", note="redo of the unscored cc-qwen attempts lost to AgentSetupTimeoutError (django-15957 and astropy-13579 attempt 1: node tarball curl from cdn.npmmirror.com hung past the 1080 s setup budget -- infra, not agent); k=1, conc 1, same CC config / wall x12; merge as per-attempt replacement"),
    "mini-qwen-nw-smoke":       _a(MINI, QWEN, "hard45", "control-nw", exclude=True, note="host-a no-web smoke, django-11133"),
    "cc-qwen-nw-smoke":         _a(CC, QWEN, "hard45", "control-nw", exclude=True, note="host-a no-web smoke, django-11133"),
    "cc-opus-nw-smoke":         _a(CC, OPUS, "hard45", "control-nw", exclude=True, note="local no-web smoke, django-11133 (3rd attempt passed; first two were verifier pypi/raw.githubusercontent denials)"),
    "cc-opus-nw-smoke.try2":    _a(CC, OPUS, "hard45", "control-nw", exclude=True, note="local no-web smoke attempt 2 (verifier denied -> false 0), kept as proxy-design evidence"),
    "cc-opus-hard45-nw":        _a(CC, OPUS, "hard45", "control-nw", note="k=1, local WSL2 (subscription CC only on this machine), conc 8 -> 4 (host anon memory >24 GB swapped at 8); no-web = internal docker net harbor-noweb + phase-aware proxy proxy_noweb.py (agent phase: api.anthropic.com only, decided per request from claude-process liveness; denials in proxy_noweb_denied.log); CC 2.1.263, --max-turns 300, WebSearch disallowed, CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1; verifier env lacked UV_HTTP_TIMEOUT (xarray-6992 false 0 -> rerun)"),
    "cc-opus-hard45-nw-r":      _a(CC, OPUS, "hard45", "control-nw", note="rerun of the infra-failed cc-opus-hard45-nw trials: 2 proxy-v2 false zeros (django-13212, 14631: verifier denied after per-IP state reset), 2 AgentSetupTimeoutError (16560, 16631: node tarball via VPN stalled), 2 NonZeroAgentExitCodeError (sklearn-25102, sympy-12489: 97 MB CC native binary via VPN at 0.4 MB/s -> npm skipped the optional dep), xarray-6992 verifier uv timeout; fixes: npmmirror direct tunnel + npm --registry npmmirror + UV_HTTP_TIMEOUT=300; same agent config, conc 4, 1 attempt; merge as per-task replacement"),
    "cc-opus-hard45-nw-r2":     _a(CC, OPUS, "hard45", "control-nw", note="1-task retry of django-13837 (RewardFileNotFoundError: agent created tests/utils_tests/test_module/__main__.py, test patch cannot apply); collided again -> the known verifier collision, task excluded from hard45 pass@1 (n=44)"),
    "cc-opus-nw-smoke-k2":      _a(CC, OPUS, "hard45", "control-nw", exclude=True, note="1-task smoke (django-11133) before the seed-2 run, 2026-09-08 10:33 (first attempt 10:19 without HARBOR_AGENT_NETWORK_MODE=public failed at trial start and was discarded); pass"),
    "cc-opus-hard45-nw-k2":     _a(CC, OPUS, "hard45", "control-nw", repeat=2, note="second independent seed of cc-opus-hard45-nw: identical YAML (conc 8, noweb proxy, CC 2.1.263, max_turns 300), launched 2026-09-08 10:38 with HARBOR_AGENT_NETWORK_MODE=public (same as seed 1; enforcement = noweb compose + proxy, Harbor egress sidecar cannot run on WSL2); hard45 grid moves to k=2 uniformly"),
    "cc-opus-hard45-nw-k3":     _a(CC, OPUS, "hard45", "control-nw", repeat=3, note="third independent seed, identical YAML to k2 (conc 8, noweb proxy, CC 2.1.263, max_turns 300, HARBOR_AGENT_SOFT_TIMEOUT_SEC=2760); launched 2026-09-08 21:05 to firm up the seed-variance trend (Opus 1/45 flip vs Qwen3.8 12%, trend p=0.061 with 2 seeds)"),
    "cc-opus-hard45-nw-k3-r":   _a(CC, OPUS, "hard45", "control-nw", repeat=3, note="seed-3 rerun of sympy-13878 (verifier parser env: aiohttp download from pypi.org timed out -> no report.json) and sympy-16597 (agent cut off by 401 OAuth token revoked at 22:55 after the local credential refresh at 22:50; 30 steps); conc 2, same YAML otherwise; local 2026-09-08 23:1x"),
    "cc-opus-hard45-nw-c2":     _a(CC, OPUS, "hard45", "c2-bashonly-nw", note="C2 tool-granularity removal on Opus: --disallowedTools WebSearch,Edit,Write,MultiEdit,NotebookEdit (schema-level, same as cc-fable51-hard45-c2); otherwise identical to the k3 YAML (noweb proxy, conc 8, CC 2.1.263, max_turns 300, soft timeout 2760); queued after seed 3, 2026-09-08 22:20"),
    "cc-opus-hard45-nw-c2-r":   _a(CC, OPUS, "hard45", "c2-bashonly-nw", note="rerun of the 11 c2 tasks lost to the 5-hour subscription window (reset 00:40 CST): 5 CancelledError (operator pause at 00:4x), 3 cc_api_error (rate_limit rejected mid-run), 3 verifier_incomplete; same YAML, conc 8; local 2026-09-09 00:5x"),
    "cc-opus-hard45-nw-c2-s2": _a(CC, OPUS, "hard45", "c2-bashonly-nw", repeat=2, note="second independent seed of the C2 bash-only arm, same YAML as cc-opus-hard45-nw-c2 except n_concurrent_trials 8 -> 16 (local RAM headroom checked). Seeds stay separate, never pooled into pass@2."),
    "cc-opus-hard45-nw-c2ns":  _a(CC, OPUS, "hard45", "c2-noshell-nw", note="C2 tool-granularity, opposite level: --disallowedTools WebSearch,Bash (schema-level; agent keeps Read/Glob/Grep/Edit/Write/MultiEdit, cannot execute anything, tests only run in the verifier); otherwise identical to the c2 YAML (noweb proxy, conc 8, CC 2.1.263, max_turns 300, soft timeout 2760). Manipulation metric = refused Bash attempts. Local 2026-09-09 01:4x"),
    "cc-opus-hard45-nw-k2-r":   _a(CC, OPUS, "hard45", "control-nw", repeat=2, note="1-task rerun of seed-2 sympy-16597 (AgentVerifierOverlap: CC still active 30.5 s after the 3000 s timeout), same YAML, conc 1, HARBOR_AGENT_SOFT_TIMEOUT_SEC=2760 (venv patch), launched 2026-09-08 12:3x local"),
    "cc-opus-hard45-nw-r3":     _a(CC, OPUS, "hard45", "control-nw", note="seed-1 rerun of django-13837 after the grader fix (test.sh now deletes files the gold test patch adds before applying it; the 3 earlier Opus attempts had no grade because Opus wrote tests/utils_tests/test_module/__main__.py itself), conc 1, soft timeout 2760, local 2026-09-08 13:2x"),
    "cc-opus-hard45-nw-k2-r2":  _a(CC, OPUS, "hard45", "control-nw", repeat=2, note="seed-2 rerun of django-13837 after the grader fix (see cc-opus-hard45-nw-r3), conc 1, soft timeout 2760, local 2026-09-08 13:2x"),
    "cc-opus-v1100-smoke":      _a(CC, OPUS, "hard45", "cc-v1.0.100-nw", exclude=True, note="1-task smoke for the harness-version-drift experiment (fixed model, vary CC version): Claude Code 1.0.100 (npm, 2025-09) x Opus x django-12708 through the noweb proxy, conc 1, local 2026-09-09 11:38-11:43, passed; init message lists the 1.0-era tool set (MultiEdit, TodoWrite, BashOutput, KillBash, no Skill/WebSearch)"),
    "cc-opus-hard45-nw-v1100":  _a(CC, OPUS, "hard45", "cc-v1.0.100-nw", note="harness-version-drift arm: Claude Code 1.0.100 (npm 2025-09-01, last 1.0.x month before 2.0; one year before the control's 2.1.263 of 2026-09-06) x Opus x hard45; YAML identical to cc-opus-hard45-nw except job_name and version (conc 8, noweb proxy, disallowed WebSearch, max_turns 300, HARBOR_AGENT_SOFT_TIMEOUT_SEC=2760); local 2026-09-09 12:1x"),
    # host-a 1-task smokes (excluded from analysis)
    "cc-ds-smoke-host-a":         _a(CC, DS, "hard45", "control", exclude=True, note="host-a smoke"),
    "mini-ds-smoke-host-a":       _a(MINI, DS, "hard45", "control", exclude=True, note="host-a smoke"),
    "qwen-smoke-host-a":          _a(OC, QWEN, "hard45", "control", exclude=True, note="host-a smoke"),
    "cc-ds-smoke":              _a(CC, DS, "smoke", "smoke", exclude=True),
    "mini-ds-smoke":            _a(MINI, DS, "smoke", "smoke", exclude=True),
    "mini-ds-smoke.badkey1":    _a(MINI, DS, "smoke", "smoke", exclude=True, note="auth fail: stale key with \\r"),
    "mini-ds-smoke.badkey2":    _a(MINI, DS, "smoke", "smoke", exclude=True, note="auth fail: mini 2.4.6 ignores MSWEA_API_KEY; needs DEEPSEEK_API_KEY passthrough"),
    "cc-ds-smoke.badkey1":      _a(CC, DS, "smoke", "smoke", exclude=True, note="auth fail: stale key with \\r"),
    # --- host-a Qwen3.6-35B-A3B (vLLM GPU 4-7 :8002 TP=4, tlsfront 172.17.0.1:8443), no-web, 09-07/08.
    # hard45 = same protocol as the Qwen3.8 arms (300-step cap, k=1); pool447 = the full SWE-bench Verified
    # pool packaged in tasks_pool/ (mini/OC conc 16, CC conc 16 midsys template). "-nw-c" = the mini pool
    # relaunch after the docker migration (UV_DEFAULT_INDEX=tuna, setup mult 6.0); "-r"/"-r2" = infra reruns.
    "mini-qwen36-hard45-nw":    _a(MINI, QWEN36, "hard45", "control-nw", note="host-a, conc 8, litellm openai/ via tlsfront"),
    "mini-qwen36-hard45-nw-r":  _a(MINI, QWEN36, "hard45", "control-nw", note="infra rerun (setup timeouts) of mini-qwen36-hard45-nw"),
    "mini-qwen36-hard45-nw-r2": _a(MINI, QWEN36, "hard45", "control-nw", note="2nd infra rerun of mini-qwen36-hard45-nw"),
    "oc-qwen36-hard45-nw":      _a(OC, QWEN36, "hard45", "control-nw", note="host-a, conc 8, OpenCode openai-compatible provider via tlsfront"),
    "oc-qwen36-hard45-nw-r":    _a(OC, QWEN36, "hard45", "control-nw", note="infra rerun of oc-qwen36-hard45-nw (node download / setup failures after the docker migration)"),
    "oc-qwen36-hard45-nw-r2":   _a(OC, QWEN36, "hard45", "control-nw", note="2nd infra rerun of oc-qwen36-hard45-nw"),
    "oc-qwen36-hard45-nw-r3":   _a(OC, QWEN36, "hard45", "control-nw", note="rerun of django-16560 (OpenCodeStartupError: first event 401 from the vLLM key window, 0 tool calls, had been scored fail), host-a 2026-09-08 12:06, HARBOR_AGENT_SOFT_TIMEOUT_SEC=17760 (x6 timeout)"),
    "cc-qwen36-hard45-nw":      _a(CC, QWEN36, "hard45", "control-nw", note="host-a, conc 8, vLLM native /v1/messages, midsys template"),
    "mini-qwen36-hard45-nw-k2": _a(MINI, QWEN36, "hard45", "control-nw", repeat=2, note="second seed of mini-qwen36-hard45-nw, identical YAML (conc 8, x6 timeout, step_limit 300); queued after the host-a pool 2026-09-08"),
    "oc-qwen36-hard45-nw-k2":   _a(OC, QWEN36, "hard45", "control-nw", repeat=2, note="second seed of oc-qwen36-hard45-nw, identical YAML (conc 8, x6 timeout, steps 300) + HARBOR_AGENT_SOFT_TIMEOUT_SEC=17760; queued after the host-a pool 2026-09-08"),
    "cc-qwen36-hard45-nw-k2":   _a(CC, QWEN36, "hard45", "control-nw", repeat=2, note="second seed of cc-qwen36-hard45-nw, identical YAML (conc 8, x12 timeout, max_turns 300, midsys template) + HARBOR_AGENT_SOFT_TIMEOUT_SEC=35760; queued after the host-a pool 2026-09-08"),
    "cc-qwen36-hard45-nw-nothink":   _a(CC, QWEN36, "hard45", "control-nw-nothink", note="thinking OFF, hard and harness-independent: vLLM --chat-template configs/qwen36_nothink.jinja (the stock Qwen3.6 template with the generation prompt fixed to <think>\\n\\n</think>\\n\\n regardless of enable_thinking), NO --reasoning-parser, plus a logits processor banning the <think> token 248068 only (scripts/nothink_lp.py, NOTHINK_BAN_IDS). Why: vLLM 0.27.1 turns any request reasoning_effort into chat_template_kwargs enable_thinking=True (chat protocol build_chat_params), and its Anthropic adapter maps Claude Code output_config.effort (CLAUDE_CODE_EFFORT_LEVEL xhigh in every CC YAML) onto reasoning_effort -- so under the server-default switch (--default-chat-template-kwargs) every CC request was rendered thinking-ON while OC/mini (no effort field) were thinking-off; the earlier CC leak (38 blocks/9.5k chars, 18:57 soft attempt) was this, not model leakage; the token bans on top of it (19:45 <think>-only: 22.9k reasoning chars; 20:16 both tokens: 4-5-step trials, all output labelled thinking until EOS, 0/11 pass) were symptoms of the parser starting in REASONING state. Probe 20:38: same prompt with output_config.effort -> thinking block until max_tokens, without -> plain text. Fourth start 2026-09-10 ~20:55 on host-b after the switch; the three aborted attempts are archived on host-b tmp/aborted_nothink_{soft,ban1,ban2}. YAML identical to cc-qwen36-hard45-nw-k2 except n_attempts 2 (k=2 in one job); soft timeout 35760"),
    "mini-qwen36-hard45-nw-nothink": _a(MINI, QWEN36, "hard45", "control-nw-nothink", note="thinking OFF, hard (fixed nothink chat template, no reasoning parser, <think> token ban; see cc-qwen36-hard45-nw-nothink); YAML = mini-qwen36-hard45-nw-k2 with n_attempts 2; host-b, vLLM launch_vllm36_nothink.sh, queued after CC/OC nothink 2026-09-10"),
    "oc-qwen36-hard45-nw-nothink":   _a(OC, QWEN36, "hard45", "control-nw-nothink", note="thinking OFF, hard (fixed nothink chat template, no reasoning parser, <think> token ban; see cc-qwen36-hard45-nw-nothink); YAML = oc-qwen36-hard45-nw-k2 with n_attempts 2, host-b, soft timeout 17760, NO rg pre-placement (HARBOR_OC_PREPLACE_RG unset) so it differs from oc-qwen36-hard45-nw-k2 (42/45 trials there show ripgrep failures, i.e. the k2 arm ran before the rg patch) only in thinking; relaunched 2026-09-10 ~20:55 (fourth start; soft switch with rg 18:57-19:41, <think>-only ban 19:45-20:13 and two-token ban 20:16-20:38 all aborted, dirs archived on host-b tmp/aborted_nothink_*)"),
    "mini-qwen36-pool447-nw":   _a(MINI, QWEN36, "pool447", "control-nw", note="first launch, 42 results before the 21:05 docker migration stopped it; kept"),
    "mini-qwen36-pool447-nw-c": _a(MINI, QWEN36, "pool447", "control-nw", note="continuation: the 405 unrun + 19 infra-failed tasks, conc 16, UV_DEFAULT_INDEX=tuna"),
    "oc-qwen36-pool447-nw":     _a(OC, QWEN36, "pool447", "control-nw", note="conc 16"),
    "oc-qwen36-pool447-nw-r":   _a(OC, QWEN36, "pool447", "control-nw", note="remaining 411 tasks after the startup-hang stop (conc 16->8, hang watchdog)"),
    "oc-qwen36-pool447-nw-r2":  _a(OC, QWEN36, "pool447", "control-nw", note="392 tasks re-opened 09-08 06:04 with the in-container relaunch loop (opencode.py patch) + 5-min watchdog + swap guard"),
    "cc-qwen36-pool447-nw":     _a(CC, QWEN36, "pool447", "control-nw", note="conc 16, midsys template, launched 09-08 02:21"),
    "mini-qwen36-nw-smoke":     _a(MINI, QWEN36, "hard45", "control-nw", exclude=True, note="host-a smoke"),
    "oc-qwen36-rgsmoke":        _a(OC, QWEN36, "hard45", "control-nw-rg", exclude=True, note="1-task smoke (django-12708) of the ripgrep pre-placement patch (harbor opencode.py install(), backup opencode.py.orig-20260909-rg): rg 15.1.0 fetched from the 8090 mirror into ~/.cache/opencode/bin during agent setup; host-a 2026-09-09 11:46"),
    "oc-qwen36-pool447-nw-rg":  _a(OC, QWEN36, "pool447", "control-nw-rg", note="whole-arm sensitivity rerun of OC x Qwen3.6 x pool447 with ripgrep pre-placed (OC grep/glob had failed offline in 361/392 trials of oc-qwen36-pool447-nw-r2: 'ripgrep execution failed'); full 447 list, YAML identical to -r2 otherwise (conc 8, x6 timeouts, steps 300, relaunch loop); vLLM Qwen3.6 TP=2 on GPU 6/7 (GPU 4/5 shared with another user); launched 2026-09-09. control-nw remains the label of the pre-fix data"),
    "cc-qwen36-pool447-nw-r":     _a(CC, QWEN36, "pool447", "control-nw", note="host-b rerun of the 53 unusable cc-qwen36-pool447-nw tasks (48 VerifierUvMissing, 3 VerifierTimeout, 2 NetworkConnectionError, 1 verifier_incomplete; list cc_qwen36_pool_gap.txt), thinking ON (launch_vllm36.sh), conc 8, x6/x12; queued after the nothink arms 2026-09-10. Outcome 2026-09-11: 44/53 usable; unusable = 11276/11451/13028 verifier_incomplete (swebench parser GitHub fetch), 12419 NetworkConnectionError (CC node tarball from cdn.npmmirror.com), 13109/13112/13195/pytest-7571 AgentSetupTimeout/CancelledError (host iowait 36-47% while django-12273 thrashed), 12273 stopped by me: the agent ran the full django suite (112 workers) inside the 8 GiB cgroup, its 120 s Bash call hung from 10:29, job SIGTERMed 11:29 -> all 9 rerun in -r2"),
    "mini-qwen36-pool447-nw-r":   _a(MINI, QWEN36, "pool447", "control-nw", note="host-b rerun of the 8 unusable mini-qwen36-pool447-nw(-c) tasks (mini_qwen36_pool_gap.txt), thinking ON, conc 8, UV_DEFAULT_INDEX tuna; ran 2026-09-11 08:47-09:07: 2 sympy scored, the other 6 (the same 5 tasks that failed in -c plus xarray-7393) hit NetworkConnectionError in the mini install step again"),
    "cc-qwen36-pool447-nw-r2":    _a(CC, QWEN36, "pool447", "control-nw", note="host-b rerun of the 9 unusable -r tasks (11276 11451 12419 13028 13109 13112 13195 pytest-7571 12273), same YAML (conc 4, thinking ON), launched 2026-09-11 11:55 after the -r stop; prefer this row over the -r row for these tasks. Outcome: 9/9 usable, 6 pass (12273/13112/13195 fail)"),
    "mini-qwen36-pool447-nw-r2":  _a(MINI, QWEN36, "pool447", "control-nw", note="host-b rerun of the 6 NetworkConnectionError tasks of -r, same YAML (conc 3), launched 2026-09-11 09:05 after the mini installer patch: Harbor execs with bash -c, so the image uv (/root/.local/bin) is off PATH and the mini installer re-downloads uv 0.12.13 via ghfast.top every trial; that DNS lookup fails intermittently inside the trial network. Patch (mini_swe_agent.py, backup .pre-uvmirror-20260911) fetches the identical uv 0.12.13 from the host mirror 172.17.0.1:8090/uv-0.12.13 first, old path as fallback; agent/uv/mini versions unchanged"),
    "mini-qwen36-pool447-nw-k2-c2": _a(MINI, QWEN36, "pool447", "control-nw", repeat=2, note="host-b continuation of -k2-c after the 2026-09-11 16:36 host-b DNS outage (gateway resolver <lan-gateway> stopped answering; every new trial failed the mini install; -k2-c SIGTERMed at 16:40 with 204 usable). 211 tasks = every -k2-c task without verifier/report.json (unrun + cancelled + 6 verifier_incomplete + 4 VerifierTimeout fetch hangs + 1 install timeout), same YAML (conc 12); launched 17:05 after host-b DNS was switched to 119.29.29.29/8.8.8.8 and with the ghfix verifier stub in place. Outcome 17:05-20:59: 211/211 done, 209 usable, 145 pass; 5 ContextWindowExceeded (scored) + 2 VerifierTimeoutError = sklearn-14710 / sklearn-25232, the same two tasks that time out in every seed-1 arm (task-level), so no -k2-c3 round"),
    "mini-qwen36-pool447-nw-k2-c": _a(MINI, QWEN36, "pool447", "control-nw", repeat=2, note="host-b continuation of the seed-2 mini pool (415 tasks not usable in mini-qwen36-pool447-nw-k2 after its host-a stop), same YAML settings (x6, step_limit 300, tuna), conc 12; queued after the nothink arms 2026-09-10"),
    "oc-qwen36-pool447-nw-rg-r2": _a(OC, QWEN36, "pool447", "control-nw-rg", note="host-b rerun of the 4 verifier_incomplete tasks of the rg arm (oc_qwen36_pool_rg_gap.txt), HARBOR_OC_PREPLACE_RG=1, thinking ON, conc 4; queued after the nothink arms 2026-09-10; ran 2026-09-11 08:47-09:17: 3 scored (1 pass), pylint-4604 verifier_incomplete again — the swebench parser (make_test_spec) fetches requirements_test.txt from raw.githubusercontent.com and GitHub IPv4 was transiently unreachable from host-b (the logged Errno 101 is the IPv6 fallback)"),

    "oc-qwen36-pool447-nw-rg-r3": _a(OC, QWEN36, "pool447", "control-nw-rg", note="host-b rerun of pylint-4604 after its transient verifier network failure in -rg-r2, same YAML (HARBOR_OC_PREPLACE_RG=1, thinking ON), launched 2026-09-11 09:25"),
    "oc-qwen36-pool447-nw-rg-r":  _a(OC, QWEN36, "pool447", "control-nw-rg", note="rerun of the 6 infra failures of oc-qwen36-pool447-nw-rg (django-13315 exit 137, matplotlib-25960 npm EBADPLATFORM, sphinx-8120 npm ECONNRESET, matplotlib-26291 + xarray-6744 verifier pip timeout, sklearn-25232 verifier timeout; sklearn-14710 not rerun: times out in every arm), conc 4, same YAML; host-a 2026-09-10 00:47"),
    "oc-qwen36-hard45-nw-rg":   _a(OC, QWEN36, "hard45", "control-nw-rg", note="hard45 rerun with the ripgrep fix, YAML identical to oc-qwen36-hard45-nw otherwise (conc 8); host-a 2026-09-09"),
    "oc-ds-rgsmoke":            _a(OC, DS, "hard45", "control-nw-rg", exclude=True, note="1-task smoke (django-12708) of the ripgrep pre-placement patch with DeepSeek V4 Flash (API alias deepseek/deepseek-v4-flash, docs map it to V4-Flash-0731; direct-call system_fingerprint a26a7955944dc5c60445bff77fac9c8e recorded 09-09 18:04); reward 1.0, 0 rg failures; host-a 2026-09-09 18:06-18:26"),
    "oc-ds-hard45-nw-rg":       _a(OC, DS, "hard45", "control-nw-rg", note="hard45 sensitivity rerun of OC x DS-Flash with ripgrep pre-placed (old arm oc-ds-hard45-nw: 82 grep + 17 glob calls, all 'ripgrep execution failed', vs 3242 bash); YAML identical to oc-ds-hard45-nw otherwise except conc 6 + agent_setup_timeout_multiplier 3.0; HARBOR_AGENT_SOFT_TIMEOUT_SEC=2760; host-a off-peak 2026-09-09 18:27 launch"),
    "oc-ds-hard45-nw-rg-r":     _a(OC, DS, "hard45", "control-nw-rg", note="rerun of the 2 exit-137 trials of oc-ds-hard45-nw-rg (django-15128, django-16263: opencode SIGKILLed mid-run at ~100 steps during large runtests batches, verifier still ran; same signature as django-13315 in the Qwen3.6 pool rg arm), conc 2, same YAML; host-a off-peak 2026-09-09 21:49"),
    "oc-ds-hard45-nw-rg-r2":    _a(OC, DS, "hard45", "control-nw-rg", note="rerun of the 3 verifier-only failures of oc-ds-hard45-nw-rg (pylint-4551, pytest-10356, sphinx-11510: verifier uv sync ReadTimeout on numpy/distlib, agent finished normally), conc 3, same YAML; host-a off-peak 2026-09-09 22:31"),
    "oc-qwen36-hard45-nw-rg-r": _a(OC, QWEN36, "hard45", "control-nw-rg", note="rerun of astropy-13398 (npm install network error in the parent job); conc 1; host-a 2026-09-09"),
    "oc-qwen36-hard45-nw-rg-k2": _a(OC, QWEN36, "hard45", "control-nw-rg", repeat=2, note="second seed of OC x Qwen3.6 x hard45 with ripgrep pre-placed (HARBOR_OC_PREPLACE_RG=1); YAML = host-b rg template + agent.build.steps 300, x3/x6, conc 8, 1 attempt, thinking ON; host-b Qwen3.6 vLLM; launched 2026-09-12 17:24"),
    # Terminal-Bench 2.1 (89 tasks, upstream harbor-framework/terminal-bench-2 @ 53ff2b87 "Various task fixes for
    # TB2.1", prebuilt images alexgshaw/<task>:20251031 pulled via docker.1panel.live; host-b tasks_tb21 with the
    # agent-phase allowlist added to every task.toml by scripts/tb21_patch_noweb.py; Qwen3.6 vLLM; 2026-09-12).
    # --- B36: termination ablation on Qwen3.6 (host-b, 149 tasks = every failure of the
    # control-nw-rg pool447 arm, unfiltered).  Arms differ ONLY in the auto-continue prompt.
    # The 2026-09-14 single-disk I/O incident killed trials in both arms; they were rerun in
    # -stock2 / -stock3 / -ac2, and within an arm a clean trial supersedes an exception trial
    # for the same task.  B36 is not reported in the paper.
    "oc-qwen36-B-stock":  _a(OC, QWEN36, "B36", "B-stock", note="B36 stock arm (no HARBOR_OC_AUTOCONTINUE), host-b, conc 10, launched 2026-09-14; 149 trials"),
    "oc-qwen36-B-stock2": _a(OC, QWEN36, "B36", "B-stock", note="rerun of 4 B36 stock trials killed by the 2026-09-14 I/O incident; all 4 came back CancelledError and were re-run again in -stock3"),
    "oc-qwen36-B-stock3": _a(OC, QWEN36, "B36", "B-stock", note="second rerun of the 5 remaining B36 stock infra failures, conc 4, host-b 2026-09-15 01:08; all 5 scored clean"),
    "oc-qwen36-B-ac":     _a(OC, QWEN36, "B36", "B-ac", note="B36 auto-continue arm (HARBOR_OC_AUTOCONTINUE=2), host-b, conc 10, launched 2026-09-14; stopped by the I/O incident at 54 trials"),
    "oc-qwen36-B-ac2":    _a(OC, QWEN36, "B36", "B-ac", note="continuation of the B36 auto-continue arm over the remaining 102 tasks, conc 10, host-b 2026-09-14"),

    "oc-qwen-B-smoke-ac": _a(OC, QWEN, "B38", "B-ac", exclude=True, note="B38 launch smoke (auto-continue arm, 2 tasks); excluded -- declared explicitly so a jobs glob cannot pull it into the B38 arms by accident"),
    "oc-qwen38-B-stock":  _a(OC, QWEN, "B38", "B-stock", note="B38 stock arm (no HARBOR_OC_AUTOCONTINUE, HARBOR_OC_PREPLACE_RG=1), host-a, conc 4, 2 attempts/task packed into one job, so repeat stays 1 and the within-job attempt index is the seed (same convention as the Qwen3.8 hard45 arms); launched 2026-09-15 13:45 together with -ac (pre-registration rule 4, time-interleaved); 41 tasks x 2 attempts = 82 trials"),
    "oc-qwen38-B-ac":     _a(OC, QWEN, "B38", "B-ac", note="B38 auto-continue arm (HARBOR_OC_AUTOCONTINUE=2, HARBOR_OC_PREPLACE_RG=1), host-a, conc 4, 2 attempts/task packed into one job, so repeat stays 1 and the within-job attempt index is the seed; launched 2026-09-15 13:45 together with -stock; 41 tasks x 2 attempts = 82 trials"),
    "oc-qwen36-tb21-nw-smoke":   _a(OC, QWEN36, "tb21", "control-nw-rg", exclude=True, note="TB2.1 smoke: adaptive-rejection-sampler + bn-fit-modify, conc 2, HARBOR_OC_PREPLACE_RG=1; host-b 2026-09-12 19:14"),
    "oc-qwen36-tb21-nw-smoke2":  _a(OC, QWEN36, "tb21", "control-nw-rg", exclude=True, note="TB2.1 smoke rerun of bn-fit-modify with verifier env INSTALLER_DOWNLOAD_URL=8090 mirror (first smoke: uv download from GitHub failed with curl 56 in both OC verifiers -> reward 0); host-b 2026-09-12 20:16"),
    "cc-qwen36-tb21-nw-smoke":   _a(CC, QWEN36, "tb21", "control-nw", exclude=True, note="TB2.1 smoke (same 2 tasks), conc 2; host-b 2026-09-12 19:14"),
    "mini-qwen36-tb21-nw-smoke": _a(MINI, QWEN36, "tb21", "control-nw", exclude=True, note="TB2.1 smoke (same 2 tasks), conc 2; host-b 2026-09-12 19:14"),
    "mini-qwen36-tb21-nw-smoke3": _a(MINI, QWEN36, "tb21", "control-nw", exclude=True, note="TB2.1 smoke of the full verifier mirror route (uv installer + CPython via 8090) on bn-fit-modify; host-b 2026-09-12 ~21:00"),
    # Full TB2.1 runs use the 55-task offline-solvable subset (fixed 2026-09-12 20:45): the 34 tasks whose oracle
    # solution installs/clones/downloads in the agent phase are dropped so the agent-phase allowlist stays identical to
    # the SWE-bench arms (list: scripts/tb21_off_tasks.txt). Step cap 300 is the budget; wall clock x30 is a safety net
    # only, because host-b's vLLM is shared with another project.
    "oc-qwen36-tb21-nw-rg":      _a(OC, QWEN36, "tb21", "control-nw-rg", note="OpenCode x Qwen3.6 x TB2.1 offline-55 (same first 44 tasks as mini), rg pre-placed (HARBOR_OC_PREPLACE_RG=1 verified in the harbor environ), steps 300, x6/x30, conc 8; host-b 2026-09-13 00:07, launched once mini stood at 16/30 pass (>=15% floor met) path-tracing-reverse: Harbor label ApiUsageLimitError is spurious (r'Quota exceeded.' hit 'Disk quota exceeded' in a strerror listing the agent printed); the trial hit ContextOverflowError at step ~115, OpenCode kept going 99 more steps (compaction), ended with a stop after 214 steps, exit 1, verifier 0 -> ContextWindowExceeded (2nd OC overflow on TB, non-terminal) mailman: OpenCode stalled at 01:45:04 after the 44th vLLM response (17 KB, closed normally per the egress sidecar log): no tool part was ever logged, no child process, no vLLM connection, process idle at ~2% CPU for 8+ h; the daemonised mailman/postfix masters do NOT hold its pipes (unlike the mini case) -> OpenCode-internal hang after a response; left to the 15 h agent cap (~16:38) -> expect AgentTimeoutError with 20 steps."),
    "oc-qwen36-tb21-nw-rg-c":    _a(OC, QWEN36, "tb21", "control-nw-rg", note="continuation: 9 of the remaining 11 offline-55 tasks whose images arrived (custom-memory-heap-crash fix-ocaml-gc git-multibranch mteb-retrieve polyglot-rust-c qemu-alpine-ssh qemu-startup reshard-c4-data write-compressor), conc 4, rg pre-placed; host-b 2026-09-13 02:38; pytorch-model-recovery + hf-model-inference (5.8 GB images) went to -c2; qemu-alpine-ssh + qemu-startup failed agent setup (Debian 11 bullseye images: bullseye-security gone from security.debian.org/tuna after LTS EOL, apt 404 -> exit 100; environment, not harness); reshard-c4-data exit 137 = opencode process OOM-killed by the task's 2G memory cap (anon-rss 1.99 GB), rerun in -c2"),
    "oc-qwen36-tb21-nw-rg-c2":   _a(OC, QWEN36, "tb21", "control-nw-rg", note="continuation 2: pytorch-model-recovery + hf-model-inference (images arrived 03:00/03:20) + reshard-c4-data rerun after its exit-137 in -c; conc 3, rg pre-placed; host-b 2026-09-13 03:22"),
    "cc-qwen36-tb21-nw":         _a(CC, QWEN36, "tb21", "control-nw", note="Claude Code x Qwen3.6 x TB2.1 offline-55 (same first 44 tasks), max_turns 300, x6/x30, conc 8; host-b 2026-09-13 03:24, launched while mini still had 3 long trials running (mailman stalled since 23:06: mini _run timeout handler blocked on the stdout pipe held by the daemonized mailman master) 05:2x: feal-differential-cryptanalysis + regex-chess = OutputTokenExceededError (CC's per-response 32000 output-token cap, 'API Error: Claude's response exceeded the 32000 output token maximum', is_error result after 5-6 turns / ~128k output tokens incl. thinking; verifier 0 -> scored fail, harness-attributable; first seen on pool447 on 2026-09-16, once, in cc-qwen-pool447-nw/sympy-21612 -- see A.6.2). git-leak-recovery = NetworkConnectionError at agent setup (CC node tarball from cdn.npmmirror.com, curl (7) for 88 s at 05:22, host and later trials fine) -> rerun in a -r job."),
    "cc-qwen36-tb21-nw-c":       _a(CC, QWEN36, "tb21", "control-nw", note="continuation: the 9 non-qemu remaining offline-55 tasks (same list as mini -c), conc 3, x6/x30; host-b 2026-09-13 03:58 Closed 09-13 13:34 CST 9/9: 3 pass; fix-ocaml-gc StepCapReached (301 turns); 1 OutputTokenExceededError; write-compressor AgentTimeoutError = wall cap (task agent timeout 900 s x30 = 7.5 h, 226 assistant msgs, 132 MB log, no result line) -> first CC x TB wall-cap hit, soft-timeout rule applies."),
    "cc-qwen36-tb21-nw-r":       _a(CC, QWEN36, "tb21", "control-nw", note="rerun of the 3 offline-55 tasks whose CC agent setup failed with a transient cdn.npmmirror.com connect error in the 44-task job (git-leak-recovery, polyglot-c-py, fix-code-vulnerability); conc 3, x6/x30, same template; host-b 2026-09-13 12:43 -> 14:48. All 3 setups succeeded (confirms transient network); git-leak-recovery pass (13 turns), fix-code-vulnerability fail (26 turns, hidden test), polyglot-c-py OutputTokenExceededError after 2 h (47 assistant msgs, 283k output tokens) -> 7th CC x TB output-cap case."),
    "mini-qwen36-tb21-nw":       _a(MINI, QWEN36, "tb21", "control-nw", note="mini-swe-agent x Qwen3.6 x TB2.1 offline-55 (first 44 tasks whose images were prewarmed), step_limit 300, x6/x30, conc 8; host-b 2026-09-12 21:43 feal-differential-cryptanalysis (04:40): ContextWindowExceeded after 5 h, then the verifier (which runs the agent's attack script) timed out at 5400 s = 1800 x3 -> no reward: VerifierTimeoutError, unscored under the pool447 convention (TB's own semantics would score it 0; decision pending). mailman (stalled since 23:06): will end on the 15 h agent timeout ~14:02; root cause = mini _run timeout handler blocks on communicate() while the daemonized mailman master holds the stdout pipe. Closed 09-13 14:03 CST 44/44: 19 pass; mailman ended as predicted: AgentTimeoutError at the 15 h cap (15:03Z->06:03Z), reward 0, no agent activity since 23:06 CST (daemon-pipe hang) -> record as harness stall, soft-timeout rule."),
    "mini-qwen36-tb21-nw-c":     _a(MINI, QWEN36, "tb21", "control-nw", note="continuation: the 9 non-qemu remaining offline-55 tasks (custom-memory-heap-crash fix-ocaml-gc git-multibranch mteb-retrieve polyglot-rust-c reshard-c4-data write-compressor pytorch-model-recovery hf-model-inference), conc 2; same protocol; host-b 2026-09-13 03:25. mteb-retrieve: NonZeroAgentExitCodeError at step 0 = mini harness install defect (image python3 is 3.10.19 -> uv tool env on 3.10; litellm 1.98-1.100.1 declare >=3.10 but import typing.NotRequired (3.11+) in context_management/editors/compact.py -> ImportError before any model call; would fail online too); image survey tmp/tb21_py_survey.txt: only mteb-retrieve (3.10) and the two qemu tasks (3.9) are below 3.11"),
    # --- host-a Qwen3.8-27B pool447 fill (2026-09-12): OC with rg (the host-b oc-qwen-pool447-nw arm had grep/glob dead in 420/447 trials) + the never-run CC cell.
    # vLLM 0.27.1 TP=4 on GPU 4-7 :8002 (layout test 2026-09-12: 2xTP=2 was slower per request, same aggregate), Qwen3.8 hard45 templates (OC limit.input 98000 / steps 300; CC compact 110000 / max_turns 300 / effort xhigh), setup x6, 1 attempt.
    "oc-qwen-pool447-nw-rg-smoke": _a(OC, QWEN, "pool447", "control-nw-rg", exclude=True, note="2-task smoke (astropy-12907/13033) of the host-a Qwen3.8 OC-rg pool YAML"),
    "cc-qwen-pool447-nw-smoke":    _a(CC, QWEN, "pool447", "control-nw", exclude=True, note="2-task smoke (astropy-12907/13033) of the host-a Qwen3.8 CC pool YAML"),
    "oc-qwen-pool447-nw-rg":       _a(OC, QWEN, "pool447", "control-nw-rg", note="OC x Qwen3.8 x pool447 with ripgrep pre-placed (HARBOR_OC_PREPLACE_RG=1), agent wall x6, conc 8, run concurrently with cc-qwen-pool447-nw on the same TP=4 server; host-a 2026-09-12. Monitor RIPGREP_FAILURES counter (7 by 12:02 09-13, all single grep calls inside otherwise-working trials) is NOT rg infra: every failing grep targets a non-existent path the model guessed (tests/q_tests/test_q.py, tests/querysets/tests.py, tests/management_tests/..., verified with ls + rg rc=2 in the django-14017 container); OpenCode reports rg exit 2 as the generic 'ripgrep execution failed'. Same holds for the 5 sporadic errors in oc-qwen36-pool447-nw-rg (bad regex, glob as path, invalid args, 64 KB JSON record) -> sporadic = model tool-input errors; only whole-trial failures (binary missing) are the harness effect."),
    "cc-qwen-pool447-nw":          _a(CC, QWEN, "pool447", "control-nw", note="CC x Qwen3.8 x pool447 (stock template, NOT midsys), vLLM /v1/messages, agent wall x12, conc 8; host-a 2026-09-12"),
    "mini-qwen36-pool447-nw-k2": _a(MINI, QWEN36, "pool447", "control-nw", repeat=2, note="second independent seed of mini x Qwen3.6 x pool447 (all 447 tasks, conc 8, YAML identical to the -c continuation: x6 timeouts, step_limit 300, UV_DEFAULT_INDEX tuna); purpose: seed floor for pool447, which had no replicate in any cell; host-a 2026-09-09 11:49-13:22, stopped by Ctrl-C after 41 trials (33 scored + 8 CancelledError = infra) because the TP=2 vLLM could not feed 24 agents; the remaining tasks continue on host-b as mini-qwen36-pool447-nw-k2-c once its Qwen3.6 server is up"),
    "oc-qwen36-nw-smoke":       _a(OC, QWEN36, "hard45", "control-nw", exclude=True, note="host-a smoke"),
    "cc-qwen36-nw-smoke":       _a(CC, QWEN36, "hard45", "control-nw", exclude=True, note="host-a smoke"),
    "mini-ds-nw-smoke.0902":    _a(MINI, DS, "hard45", "control-nw", exclude=True, note="09-02 no-web smoke (superseded)"),
    "oc-ds-nw-smoke.0902":      _a(OC, DS, "hard45", "control-nw", exclude=True, note="09-02 no-web smoke (superseded)"),
    # --- host-b Qwen3.8-27B (4x L40, vLLM :8002), no-web, pool447 + CC midsys template.
    # midsys = CC system-prompt layout that keeps the task prompt out of the cached prefix (prefix-cache hit 83% vs 42%).
    "cc-qwen-ab10-midsys":      _a(CC, QWEN, "ab10", "control-nw-midsys", note="host-b A/B of the midsys template on 10 hard45 tasks"),
    "cc-qwen-ab10-midsys-r18199": _a(CC, QWEN, "ab10", "control-nw-midsys", note="1-task rerun (lock check refused resume after the Dockerfile uv-mirror edit)"),
    "cc-qwen-hard45rest-midsys-nw": _a(CC, QWEN, "hard45", "control-nw-midsys", note="host-b: the 34 hard45 tasks not in ab10, midsys template, conc 8"),
    "cc-qwen-hard45rest-midsys-nw-r": _a(CC, QWEN, "hard45", "control-nw-midsys", note="remaining 23 hard45 tasks after the 09-08 04:00 KV-thrash stop (conc 8->3)"),
    "mini-qwen-pool447-nw":     _a(MINI, QWEN, "pool447", "control-nw", note="host-b, conc 8"),
    "oc-qwen-pool447-nw":       _a(OC, QWEN, "pool447", "control-nw", note="host-b, conc 8"),
    "mini-qwen-pool447-nw-r":   _a(MINI, QWEN, "pool447", "control-nw", note="host-b rerun of the 40 unscored mini-qwen-pool447-nw trials (18 matplotlib RuntimeError = image absent at the time, 11 verifier_incomplete, 5 NetworkConnectionError, 3 VerifierTimeout, 1 each overlap/NonZero/RewardFileNotFound; list = extract_trials scored=False on host-b), YAML identical to the pool otherwise (conc 8, x6 timeouts, steps 300); queued after the pools 2026-09-09 evening"),
    "oc-qwen-pool447-nw-r":     _a(OC, QWEN, "pool447", "control-nw", note="host-b rerun of the 81 unscored oc-qwen-pool447-nw trials (43 VerifierUvMissing from before the 09-08 06:39 test.sh uv fix, 18 matplotlib RuntimeError = image absent, 10 AgentSetupTimeout, 5 verifier_incomplete, 2 VerifierTimeout, 2 NonZero, 1 overlap), YAML identical to the pool otherwise (conc 8, x6 timeouts, steps 300); pre-fix ripgrep behaviour kept (host-b opencode.py rg patch is env-gated, HARBOR_OC_PREPLACE_RG unset); queued after the pools 2026-09-09 evening"),
    "mini-qwen-pool447-nw-r2":  _a(MINI, QWEN, "pool447", "control-nw", note="host-b rerun of the mini-qwen-pool447-nw-r verifier-parser hang (matplotlib-24970: tests finished, parser never printed, VerifierTimeout 3000 s); conc 1, YAML otherwise identical; launched 2026-09-10 07:2x alongside the rg arm"),
    "oc-qwen-pool447-nw-r2":    _a(OC, QWEN, "pool447", "control-nw", note="host-b rerun of the two oc-qwen-pool447-nw-r verifier-parser hangs (django-11095, django-11211); conc 2, pre-fix ripgrep kept (HARBOR_OC_PREPLACE_RG unset); launched 2026-09-10 07:2x alongside the rg arm"),
    "mini-qwen-pool447-nw-r3":  _a(MINI, QWEN, "pool447", "control-nw", note="host-b coverage-gap rerun (2026-09-10 08:2x, alongside the rg arm): the 3 mini x Qwen3.8 pool447 tasks left without a usable row (django-14534/15278/16901: verifier_incomplete = trial cut when the pool job stopped); conc 3, YAML otherwise identical"),
    "oc-qwen-pool447-nw-r3":    _a(OC, QWEN, "pool447", "control-nw", note="host-b coverage-gap rerun (2026-09-10 08:2x): django-11433, the one OC x Qwen3.8 pool447 task still VerifierUvMissing after -r; conc 1, pre-fix ripgrep kept"),
    "oc-qwen-hard45-nw-rg":     _a(OC, QWEN, "hard45", "control-nw-rg", note="hard45 sensitivity rerun of OC x Qwen3.8 with ripgrep pre-placed (host-b opencode.py env-gated patch HARBOR_OC_PREPLACE_RG=1, backup opencode.py.orig-20260909-rg, rg 15.1.0 from the host-b 8090 mirror); k=2 attempts like qwen-hard45-nw, conc 8, x2 agent timeout, setup x3, no step cap, limit.input 98000, HARBOR_AGENT_SOFT_TIMEOUT_SEC=5760 (same as the overlap rerun); host-b vLLM Qwen3.8-27B TP=4; queued after the pool reruns 2026-09-09/10"),
    "mini-qwen-nw-smoke-host-b": _a(MINI, QWEN, "hard45", "control-nw", exclude=True, note="host-b smoke"),
    "oc-qwen-nw-smoke-host-b":  _a(OC, QWEN, "hard45", "control-nw", exclude=True, note="host-b smoke"),
    "cc-qwen-nw-smoke-host-b":  _a(CC, QWEN, "hard45", "control-nw", exclude=True, note="host-b smoke"),
    # --- DeepSeek V4 Pro probe (CC no-web template, model deepseek-v4-pro): 10 Opus-only-solved hard45 tasks + 3 controls.
    "cc-dspro-smoke1-nw":       _a(CC, DSPRO, "probe13", "control-nw", exclude=True, note="host-a smoke, django-12708"),
    "cc-dspro-probe13-nw":      _a(CC, DSPRO, "probe13", "control-nw", note="host-a, conc 6; 4/13 pass (3 controls + sympy-14248)"),
    # --- DeepSeek Harness (dsh, npm @deepseek-ai/dsh 0.1.2-rc.1, headless profile) x DS-V4-Flash, no-web hard45.
    # Adapter agents/dsh_agent.py; web tools disabled by patch overlay; 46 min soft timeout (SIGINT) + 2 min kill.
    "dsh-ds-smoke1-nw":         _a(DSH, DS, "hard45", "control-nw", exclude=True, note="host-a smoke, django-12708"),
    "dsh-ds-hard45-nw":         _a(DSH, DS, "hard45", "control-nw", note="host-a, conc 6, agent_setup_timeout_multiplier 6.0, launched 09-08 02:05 off-peak"),
    "dsh-ds-hard45-nw-t2x":     _a(DSH, DS, "hard45", "control-nw", exclude=True, note="sensitivity: the 3 soft-timeout fails (django-12325/16263, xarray-6992) rerun with soft_timeout_sec 5520 (2x) and agent_timeout_multiplier 2.0; reported separately, not merged"),
    "mini-ds-nw-smoke-k2":      _a(MINI, DS, "hard45", "control-nw", exclude=True, note="smoke before the Flash seed-2 run (django-11133), 2026-09-08 off-peak"),
    "cc-ds-nw-smoke2-k2":       _a(CC, DS, "hard45", "control-nw", exclude=True, note="smoke before the Flash seed-2 run (django-11400), 2026-09-08 off-peak"),
    "oc-ds-nw-smoke-k2":        _a(OC, DS, "hard45", "control-nw", exclude=True, note="smoke before the Flash seed-2 run (django-11133), 2026-09-08 off-peak"),
    "mini-ds-hard45-nw-k2":     _a(MINI, DS, "hard45", "control-nw", repeat=2, note="second independent seed of mini-ds-hard45-nw, YAML identical except job_name (conc 6, host-a), off-peak 2026-09-08 evening"),
    "cc-ds-hard45-nw-k2":       _a(CC, DS, "hard45", "control-nw", repeat=2, note="agent wrapped in a 2760 s soft timeout (HARBOR_AGENT_SOFT_TIMEOUT_SEC, venv patch 2026-09-08) so it exits before the verifier; second independent seed of cc-ds-hard45-nw, YAML identical except job_name (conc 6, host-a, --disallowedTools WebSearch), off-peak 2026-09-08 evening"),
    "cc-ds-hard45-nw-k2-r":     _a(CC, DS, "hard45", "control-nw", repeat=2, note="rerun of the cc-ds-hard45-nw-k2 setup-phase network failures (astropy-13579, django-11885: node tarball fetch from cdn.npmmirror.com failed before the agent started), same YAML/soft timeout, off-peak 2026-09-08 night"),
    "oc-ds-hard45-nw-k2":       _a(OC, DS, "hard45", "control-nw", repeat=2, note="agent wrapped in a 2760 s soft timeout (HARBOR_AGENT_SOFT_TIMEOUT_SEC, venv patch 2026-09-08) so it exits before the verifier; second independent seed of oc-ds-hard45-nw, YAML identical except job_name (conc 6, host-a), off-peak 2026-09-08 evening"),
    "oc-ds-hard45-nw-r2":       _a(OC, DS, "hard45", "control-nw", note="rerun of oc-ds-hard45-nw django-13128 (AgentVerifierOverlap: OpenCode kept editing 5.5 min into verification) under HARBOR_AGENT_SOFT_TIMEOUT_SEC=2760, off-peak 2026-09-08 evening"),
    # GLM-5.3-Flash x 3 harnesses x hard45 (host-a, 2026-09-17; all traffic via the glmfront proxy
    # 172.17.0.1:8011 with retry + in-flight cap; reasoning_effort pinned to "high" in every harness)
    "cc-glm53f-hard45-nw":      _a(CC, GLMF, "hard45", "control-nw", note="conc 2, CC --effort high via Harbor kwarg, anthropic-route through glmfront; HARBOR_AGENT_SOFT_TIMEOUT_SEC; launched 2026-09-17 02:20"),
    "mini-glm53f-hard45-nw":    _a(MINI, GLMF, "hard45", "control-nw", exclude=True, note="first mini launch 02:20, dropped: Harbor egress gost cut plain-HTTP connections whose headers took >15 s (glmfront non-stream POSTs); superseded by mini-glm53f-hard45b-nw"),
    "mini-glm53f-hard45b-nw":   _a(MINI, GLMF, "hard45", "control-nw", note="relaunch 2026-09-17 03:00 after the glmfront early-header fix; conc 2, extra_body.reasoning_effort=high, UV_INDEX_URL PyPI cache 8091"),
    "oc-glm53f-hard45-nw-rg":   _a(OC, GLMF, "hard45", "control-nw-rg", note="conc 2, ripgrep pre-placed (HARBOR_OC_PREPLACE_RG=1), model options.reasoningEffort=high; launched 2026-09-17 02:20"),
    "oc-glm53f-r1-nw-rg":       _a(OC, GLMF, "hard45", "control-nw-rg", note="rerun of the one infra-failed trial of oc-glm53f-hard45-nw-rg (verifier pipmirror patch applied), conc 1, same YAML; 2026-09-17"),
    # HY4 preview (Tencent tokenhub) x 3 harnesses x hard45 (host-a, 2026-09-19/20; reasoning
    # effort pinned high in every harness; OpenAI-compatible traffic via the hyfront proxy
    # 172.17.0.1:8013, Anthropic-route CC via its own hyfront instances 8016/8017)
    "mini-hy4-hard45-nw":       _a(MINI, HY4, "hard45", "control-nw", note="conc 8, mini-swe-agent 2.4.6, step_limit 300, thinking ON (vendor default), litellm openai-compatible route via hyfront 172.17.0.1:8013; launched 2026-09-19 17:55, done 00:45"),
    "cc-hy4-hard45-nw-r":       _a(CC, HY4, "hard45", "control-nw", note="conc 8, CC 2.1.263 --effort high, --max-turns 300, thinking ON, anthropic route via hyfront 172.17.0.1:8016 with --fold-mid-system (the proxy folds CC mid-conversation system messages into the preceding user message so tokenhub's prefix cache hits; without it cache hit stalled at system+tools); launched 2026-09-20 01:52, done 08:27"),
    "cc-hy4-hard45-nw-nothink": _a(CC, HY4, "hard45", "control-nw-nothink", note="thinking OFF ablation of cc-hy4-hard45-nw-r: identical YAML plus CLAUDE_CODE_DISABLE_THINKING=1 and hyfront 172.17.0.1:8017 with --fold-mid-system --default-thinking-disabled (tokenhub defaults a missing thinking field to ON, so the proxy sets thinking:{type:disabled} explicitly); conc 8, CC 2.1.263 --effort high, --max-turns 300; launched 2026-09-20 08:41, done 14:12"),
    "oc-hy4-hard45-nw-rg":      _a(OC, HY4, "hard45", "control-nw-rg", note="conc 8, OpenCode 1.18.30, model options.reasoningEffort=high, agent.build.steps 300, ripgrep pre-placed (HARBOR_OC_PREPLACE_RG=1), openai-compatible route via hyfront 172.17.0.1:8013; launched 2026-09-20 15:35, done 23:37; django-11138 and django-14007 exit 137 and sklearn-25102 AgentTimeoutError all carry reward 1.0 and stay scored per protocol A.8.2"),
    "oc-hy4-hard45-nw-rg-r":    _a(OC, HY4, "hard45", "control-nw-rg", note="rerun of the one AgentVerifierOverlap trial of oc-hy4-hard45-nw-rg (sklearn-25102: OpenCode kept writing 1083 s past agent_execution.finished_at while the verifier ran, A.8.2 rule 3 -> voided), conc 1, same YAML; host-a 2026-09-20 23:51 -> 01:47, clean exit, reward 0"),
    # infra smoke / pipeline tests
    "smoke1": _a(OC, DS, "smoke", "smoke", exclude=True),
    "e2e1":   _a(OC, DS, "smoke", "smoke", exclude=True),
    "base1":  _a(OC, DS, "smoke", "smoke", exclude=True),
    "abl1":   _a(OC, DS, "smoke", "smoke", exclude=True),
    "conc2":  _a(OC, DS, "smoke", "smoke", exclude=True),
    "pin1":   _a(OC, DS, "smoke", "smoke", exclude=True),
}
for _i in range(1, 8):
    JOB_ARMS[f"cc-opus-smoke{_i}"] = _a(CC, OPUS, "smoke", "smoke", exclude=True)

TOUCH_TOOLS = ["task", "todowrite", "edit", "write", "bash", "webfetch"]

COMPACT_DROP_MIN_PREV = 20_000   # previous step must exceed this
COMPACT_DROP_FRAC = 0.40         # and the next step must drop by more than this fraction

OVERFLOW_PATTERNS = ("prompt is too long", "context_length_exceeded", "maximum context length",
                     "exceeds the context window", "context window exceeded",
                     "input length and `max_tokens` exceed")

# Per-trial exclusions (job, trial_id) -> reason. Operator interventions that invalidate one trial
# without invalidating its job; the analysis drops these like job-level excludes.
EXCLUDE_TRIALS = {
    ("cc-qwen36-pool447-nw", "CbuTiyf"):
        "2026-09-08 06:14 host-a: agent backgrounded django runtests with 128 workers, container filled "
        "host swap; operator ended the test workers -> trajectory tail not genuine, rerun",
}

# task-level exclusions, (subset, task): django-13837's verifier test_patch collides deterministically
# with the agent's edit on every grid cell (Opus and DeepSeek alike: RewardFileNotFoundError, no
# report), so the task carried no information while that collision lasted.
EXCLUDE_TASKS = set()  # django-13837 restored 2026-09-08 after the test.sh new-file fix; Opus rerun in both seeds

COLUMNS = [
    "job", "harness", "model", "subset", "arm", "repeat", "attempt", "arm_key", "exclude", "task",
    "trial_id", "status", "scored", "reward", "exception_type", "harbor_exception",
    "hard_failure", "verifier_ok",
    "context_overflow", "source", "steps", "agent_steps", "prompt_tokens",
    "completion_tokens", "reasoning_tokens", "reasoning_chars", "output_tokens_total", "cache_read_tokens", "cache_write_tokens",
    "peak_ctx", "n_compaction", "n_compaction_log", "n_sidechain_steps", "wall_seconds",
    "agent_seconds", "cost_usd", "n_tool_calls", "n_invalid_tool", "tool_counts",
] + [f"touch_{t}" for t in TOUCH_TOOLS] + ["post_timeout_sec", "started_at", "finished_at", "agent_version",
                                            "model_name"]


# ----------------------------------------------------------------------------- helpers
def read_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def iter_jsonl(path):
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("{"):
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    continue
    except OSError:
        return


def parse_ts(s):
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        return None


def count_drops(series):
    n = 0
    for prev, cur in zip(series, series[1:]):
        if prev > COMPACT_DROP_MIN_PREV and cur < (1 - COMPACT_DROP_FRAC) * prev:
            n += 1
    return n


def file_has_overflow(path):
    try:
        with open(path, errors="replace") as fh:
            for line in fh:
                low = line.lower()
                if any(p in low for p in OVERFLOW_PATTERNS):
                    return True
    except OSError:
        pass
    return False


def tool_summary(counter):
    """counter of function names -> (json string, n_calls, n_invalid, touch dict)."""
    touch = {t: False for t in TOUCH_TOOLS}
    n_calls = 0
    n_invalid = 0
    for name, k in counter.items():
        if name.startswith("invalid"):
            n_invalid += k
            continue
        n_calls += k
        low = name.lower()
        if low in touch:
            touch[low] = True
    return json.dumps(dict(sorted(counter.items()))), n_calls, n_invalid, touch


# ----------------------------------------------------------------------------- ATIF
def metrics_from_atif(traj):
    steps = traj.get("steps") or []
    agent = [s for s in steps if s.get("source") == "agent"]
    main = [s for s in agent if not (s.get("extra") or {}).get("is_sidechain")]
    fm = traj.get("final_metrics") or {}
    tools = Counter()
    ctx = []
    ptok = ctok = rtok = cread = cwrite = 0
    rchars = 0
    for s in agent:
        rchars += len(s.get("reasoning_content") or "")
        m = s.get("metrics") or {}
        ptok += m.get("prompt_tokens") or 0
        ctok += m.get("completion_tokens") or 0
        cread += m.get("cached_tokens") or 0
        ex = m.get("extra") or {}
        # OpenCode: extra.reasoning_tokens, NOT part of completion_tokens (raw step_finish keeps
        # output and reasoning apart); mini/litellm: completion_tokens_details.reasoning_tokens, a
        # subset of completion_tokens; CC: output_tokens_details.thinking_tokens (subset, usually 0).
        rtok += (ex.get("reasoning_tokens")
                 or (ex.get("output_tokens_details") or {}).get("thinking_tokens")
                 or (ex.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
        cwrite += ex.get("cache_creation_input_tokens") or 0
        for tc in s.get("tool_calls") or []:
            name = tc.get("function_name") or "?"
            if name == "invalid":
                attempted = (tc.get("arguments") or {}).get("tool") or "?"
                name = f"invalid:{attempted}"
            tools[name] += 1
    for s in main:
        m = s.get("metrics") or {}
        if m.get("prompt_tokens") is not None:
            ctx.append(m["prompt_tokens"])
    tc_json, n_calls, n_invalid, touch = tool_summary(tools)
    out = dict(
        source="atif",
        steps=fm.get("total_steps", len(steps)),
        agent_steps=len(agent),
        prompt_tokens=fm.get("total_prompt_tokens", ptok),
        completion_tokens=fm.get("total_completion_tokens", ctok),
        reasoning_tokens=rtok,
        reasoning_chars=rchars,   # ATIF reasoning_content length; the only thinking measure for CC on non-Claude endpoints (usage has no split)
        cache_read_tokens=fm.get("total_cached_tokens", cread),
        cache_write_tokens=cwrite or None,
        peak_ctx=max(ctx) if ctx else None,
        n_compaction=count_drops(ctx),
        n_sidechain_steps=len(agent) - len(main),
        cost_usd=fm.get("total_cost_usd"),
        n_tool_calls=n_calls, n_invalid_tool=n_invalid, tool_counts=tc_json,
        agent_version=(traj.get("agent") or {}).get("version"),
        model_name=(traj.get("agent") or {}).get("model_name"),
    )
    for t, v in touch.items():
        out[f"touch_{t}"] = v
    return out


# ----------------------------------------------------------------------------- raw logs
def opencode_startup_error(path):
    """Name:status of a top-level OpenCode error event that precedes every session/tool event
    (401/403/429/5xx from the endpoint at startup), else None."""
    for e in iter_jsonl(path):
        if e.get("type") == "error":
            err = e.get("error") or {}
            data = err.get("data") or {}
            return "%s:%s" % (err.get("name") or "error", data.get("statusCode") or "")
        return None
    return None


def last_agent_activity(agent_dir, harness):
    """Latest agent-side activity (epoch seconds): OpenCode event timestamps, CC session-jsonl
    timestamps, plus the mtime of the agent's own stdout log (written by tee inside the container,
    preserved by rsync -a). None when nothing is dated."""
    cands = []
    if harness == OC:
        for e in iter_jsonl(os.path.join(agent_dir, "opencode.txt")):
            ts = e.get("timestamp")
            if isinstance(ts, (int, float)):
                cands.append(ts / 1000.0)
        log = os.path.join(agent_dir, "opencode.txt")
    elif harness == CC:
        for p in glob.glob(os.path.join(agent_dir, "sessions", "projects", "**", "*.jsonl"), recursive=True):
            for e in iter_jsonl(p):
                t = parse_ts(e.get("timestamp")) if isinstance(e.get("timestamp"), str) else None
                if t:
                    cands.append(t)
        log = os.path.join(agent_dir, "claude-code.txt")
    else:
        # mini / dsh: the harness log's mtime is the last agent activity (2026-09-08 review, F5)
        log = os.path.join(agent_dir, {MINI: "mini-swe-agent.txt", DSH: "dsh.txt"}.get(harness, "-"))
        if not os.path.exists(log):
            return None
    if os.path.exists(log):
        cands.append(os.path.getmtime(log))
    return max(cands) if cands else None


def scan_opencode_log(path):
    """Returns (metrics dict or None, n_compaction_log, first_ts, last_ts, version)."""
    ctx, tools = [], Counter()
    ptok = ctok = rtok = cread = cwrite = 0
    n_log = 0
    first = last = None
    n_steps = 0
    for e in iter_jsonl(path):
        ts = e.get("timestamp")
        if isinstance(ts, (int, float)):
            ts = ts / 1000.0
            first = ts if first is None else first
            last = ts
        p = e.get("part") or {}
        pt = p.get("type")
        if pt == "step-finish":
            tk = p.get("tokens") or {}
            cache = tk.get("cache") or {}
            prompt = (tk.get("input") or 0) + (cache.get("read") or 0) + (cache.get("write") or 0)
            ctx.append(prompt)
            ptok += prompt
            ctok += tk.get("output") or 0
            rtok += tk.get("reasoning") or 0
            cread += cache.get("read") or 0
            cwrite += cache.get("write") or 0
            n_steps += 1
        elif pt == "tool":
            name = p.get("tool") or "?"
            if name == "invalid":
                name = "invalid:" + str(((p.get("state") or {}).get("input") or {}).get("tool") or "?")
            tools[name] += 1
        elif pt == "text" and (p.get("metadata") or {}).get("compaction_continue"):
            n_log += 1
    if n_steps == 0:
        return None, n_log, first, last
    tc_json, n_calls, n_invalid, touch = tool_summary(tools)
    out = dict(source="rawlog", steps=n_steps + 1, agent_steps=n_steps, prompt_tokens=ptok,
               completion_tokens=ctok, reasoning_tokens=rtok, cache_read_tokens=cread,
               cache_write_tokens=cwrite, peak_ctx=max(ctx), n_compaction=count_drops(ctx),
               n_sidechain_steps=0, n_tool_calls=n_calls, n_invalid_tool=n_invalid,
               tool_counts=tc_json)
    for t, v in touch.items():
        out[f"touch_{t}"] = v
    return out, n_log, first, last


DSH_TOUCH_ALIAS = {"todo_write": "todowrite", "subagent": "task", "str_replace_editor": "edit",
                   "web_fetch": "webfetch"}


def _dsh_time(v):
    if isinstance(v, (int, float)):
        return v / 1000.0 if v > 1e12 else float(v)
    return parse_ts(v) if isinstance(v, str) else None


def scan_dsh_sessions(sessions_dir):
    """dsh keeps one session.jsonl per (sub)agent under agent/dsh/home/sessions/.  The main
    session is the file whose first event is earliest; the others are subagent sidechains.
    Returns (metrics dict or None, first_ts, last_ts, end_reason_kind)."""
    files = sorted(glob.glob(os.path.join(sessions_dir, "**", "*.jsonl"), recursive=True))
    per = []
    for f in files:
        msgs, ctx, tools = 0, [], Counter()
        ptok = ctok = rtok = cread = cwrite = 0
        t0 = t1 = None
        end = None
        for e in iter_jsonl(f):
            ts = _dsh_time(e.get("time"))
            if ts is not None:
                t0 = ts if t0 is None else t0
                t1 = ts
            t = e.get("type")
            d = e.get("data") or {}
            if t == "assistant/message":
                u = d.get("usage") or {}
                if not isinstance(u, dict):
                    continue
                msgs += 1
                prompt = (u.get("inputTokens") or 0) + (u.get("cacheReadTokens") or 0)
                ctx.append(prompt)
                ptok += prompt
                ctok += u.get("outputTokens") or 0
                rtok += u.get("reasoningTokens") or 0
                cread += u.get("cacheReadTokens") or 0
                cwrite += u.get("cacheWriteTokens") or 0
            elif t == "tool/call":
                tools[d.get("name") or "?"] += 1
            elif t == "turn/end":
                r = d.get("reason")
                end = r.get("kind") if isinstance(r, dict) else r
        per.append(dict(msgs=msgs, ctx=ctx, tools=tools, ptok=ptok, ctok=ctok, rtok=rtok,
                        cread=cread, cwrite=cwrite, t0=t0, t1=t1, end=end))
    per = [p for p in per if p["msgs"]]
    if not per:
        return None, None, None, None
    per.sort(key=lambda p: p["t0"] if p["t0"] is not None else float("inf"))
    main, side = per[0], per[1:]
    tools = Counter()
    for p in per:
        tools.update(p["tools"])
    tc_json, n_calls, n_invalid, touch = tool_summary(tools)
    for raw, canon in DSH_TOUCH_ALIAS.items():
        if tools.get(raw):
            touch[canon] = True
    agent_steps = sum(p["msgs"] for p in per)
    first = min((p["t0"] for p in per if p["t0"] is not None), default=None)
    last = max((p["t1"] for p in per if p["t1"] is not None), default=None)
    out = dict(source="dsh-session", steps=agent_steps + 1, agent_steps=agent_steps,
               prompt_tokens=sum(p["ptok"] for p in per),
               completion_tokens=sum(p["ctok"] for p in per),
               reasoning_tokens=sum(p["rtok"] for p in per),
               cache_read_tokens=sum(p["cread"] for p in per),
               cache_write_tokens=sum(p["cwrite"] for p in per) or None,
               peak_ctx=max(main["ctx"]) if main["ctx"] else None,
               n_compaction=count_drops(main["ctx"]),
               n_sidechain_steps=sum(p["msgs"] for p in side),
               n_tool_calls=n_calls, n_invalid_tool=n_invalid, tool_counts=tc_json)
    for t, v in touch.items():
        out[f"touch_{t}"] = v
    return out, first, last, main["end"]


def scan_claude_code_log(path):
    """Returns (metrics dict or None, n_compaction_log, first_ts, last_ts, info dict)."""
    usage_by_msg = {}   # message id -> usage (main thread)
    order = []
    side_ids = set()
    tools = Counter()
    seen_tool_use = set()
    n_log = 0
    first = last = None
    info = {"is_error": False, "model": None, "version": None, "result_usage": None,
            "duration_ms": None, "result_text": None}
    for e in iter_jsonl(path):
        t = e.get("type")
        ts = parse_ts(e.get("timestamp"))
        if ts:
            first = ts if first is None else first
            last = ts
        if t == "system":
            st = e.get("subtype") or ""
            if st == "init":
                info["model"] = e.get("model")
                info["version"] = e.get("claude_code_version")
            elif "compact" in st:
                n_log += 1
        elif t == "assistant":
            m = e.get("message") or {}
            mid = m.get("id")
            if mid is None:
                continue
            if e.get("parent_tool_use_id"):
                side_ids.add(mid)
            if mid not in usage_by_msg:
                order.append(mid)
            usage_by_msg[mid] = m.get("usage") or {}
            for c in m.get("content") or []:
                if c.get("type") == "tool_use":
                    key = c.get("id") or (mid, c.get("name"))
                    if key in seen_tool_use:
                        continue
                    seen_tool_use.add(key)
                    tools[c.get("name") or "?"] += 1
        elif t == "result":
            info["is_error"] = bool(e.get("is_error")) or e.get("terminal_reason") == "api_error"
            info["result_usage"] = e.get("usage")
            info["duration_ms"] = e.get("duration_ms")
            info["result_text"] = e.get("result")
            info["result_subtype"] = e.get("subtype")
    if not order:
        return None, n_log, first, last, info

    def prompt_of(u):
        return (u.get("input_tokens") or 0) + (u.get("cache_creation_input_tokens") or 0) + \
               (u.get("cache_read_input_tokens") or 0)
    ctx = [prompt_of(usage_by_msg[m]) for m in order if m not in side_ids]
    ptok = sum(prompt_of(usage_by_msg[m]) for m in order)
    cread = sum(usage_by_msg[m].get("cache_read_input_tokens") or 0 for m in order)
    cwrite = sum(usage_by_msg[m].get("cache_creation_input_tokens") or 0 for m in order)
    ru = info["result_usage"] or {}
    ctok = ru.get("output_tokens") if ru else None
    rtok = (ru.get("output_tokens_details") or {}).get("thinking_tokens") if ru else None
    tc_json, n_calls, n_invalid, touch = tool_summary(tools)
    out = dict(source="rawlog", steps=len(order) + 1, agent_steps=len(order), prompt_tokens=ptok,
               completion_tokens=ctok, reasoning_tokens=rtok, cache_read_tokens=cread,
               cache_write_tokens=cwrite, peak_ctx=max(ctx) if ctx else None,
               n_compaction=count_drops(ctx), n_sidechain_steps=len(side_ids),
               n_tool_calls=n_calls, n_invalid_tool=n_invalid, tool_counts=tc_json,
               agent_version=info["version"], model_name=info["model"])
    for t, v in touch.items():
        out[f"touch_{t}"] = v
    return out, n_log, first, last, info


# ----------------------------------------------------------------------------- per trial
def extract_trial(job, trial_dir):
    name = os.path.basename(trial_dir)
    task, _, trial_id = name.rpartition("__")
    arm = JOB_ARMS.get(job)
    row = {c: None for c in COLUMNS}
    row.update(job=job, task=task, trial_id=trial_id)
    if arm:
        row.update(harness=arm["harness"], model=arm["model"], subset=arm["subset"],
                   arm=arm["arm"], repeat=arm["repeat"], exclude=arm["exclude"])
    else:
        cfg = read_json(os.path.join(trial_dir, "config.json")) or {}
        a = cfg.get("agent") or {}
        row.update(harness=(a.get("name") or "UNKNOWN").replace("_", "-"),
                   model=a.get("model_name") or "UNKNOWN", subset="UNKNOWN", arm="UNKNOWN",
                   repeat=1, exclude=False)
    if (job, trial_id) in EXCLUDE_TRIALS or (row["subset"], task) in EXCLUDE_TASKS:
        row["exclude"] = True
    row["arm_key"] = "/".join(str(row[k]) for k in ("harness", "model", "subset", "arm", "repeat"))

    res = read_json(os.path.join(trial_dir, "result.json"))
    agent_dir = os.path.join(trial_dir, "agent")
    traj_path = os.path.join(agent_dir, "trajectory.json")
    oc_log = os.path.join(agent_dir, "opencode.txt")
    cc_log = os.path.join(agent_dir, "claude-code.txt")
    dsh_log = os.path.join(agent_dir, "dsh.txt")
    dsh_sessions = os.path.join(agent_dir, "dsh", "home", "sessions")

    reward = None
    exc_type = None
    if res is not None:
        row["status"] = "complete"
        vr = (res.get("verifier_result") or {}).get("rewards") or {}
        reward = vr.get("reward")
        ei = res.get("exception_info") or {}
        exc_type = ei.get("exception_type")
        row["started_at"] = res.get("started_at")
        row["finished_at"] = res.get("finished_at")
        st, ft = parse_ts(res.get("started_at")), parse_ts(res.get("finished_at"))
        if st and ft:
            row["wall_seconds"] = round(ft - st, 1)
        ae = res.get("agent_execution") or {}
        st, ft = parse_ts(ae.get("started_at")), parse_ts(ae.get("finished_at"))
        if st and ft:
            row["agent_seconds"] = round(ft - st, 1)
        ar = res.get("agent_result") or {}
        if ar.get("cost_usd") is not None:
            row["cost_usd"] = ar.get("cost_usd")
        ai = res.get("agent_info") or {}
        row["agent_version"] = ai.get("version")
        row["model_name"] = (ai.get("model_info") or {}).get("name")
    else:
        row["status"] = "running"
        rt = os.path.join(trial_dir, "verifier", "reward.txt")
        if os.path.exists(rt):
            try:
                reward = float(open(rt).read().strip())
            except ValueError:
                reward = None
    if reward is None and res is not None:
        rt = os.path.join(trial_dir, "verifier", "reward.txt")
        if os.path.exists(rt):
            try:
                reward = float(open(rt).read().strip())
            except ValueError:
                pass
    # verifier integrity: tests/test.sh runs `uv run parser.py` and writes reward=0 whenever the
    # parser exits non-zero -- including when uv could not fetch its deps (pypi timeouts). Such a
    # zero is an infra failure, not a fail: parser.py prints the marker only when it actually ran.
    vdir = os.path.join(trial_dir, "verifier")
    v_out = os.path.join(vdir, "test-stdout.txt")
    marker = False
    uv_missing = False
    if os.path.exists(v_out):
        with open(v_out, "rb") as fh:
            v_bytes = fh.read()
        marker = b"SWEBench results starts here" in v_bytes
        # 2026-09-08: task images built while the host uv mirror was down have no uv at all, so
        # test.sh died at `uv run parser.py` (CC/OpenCode only; mini reinstalls uv itself). Infra.
        uv_missing = b"uv: command not found" in v_bytes
    # TB2.1 tasks ship their own pytest verifier (tests/run-tests.sh -> pytest --json-ctrf), which
    # writes verifier/ctrf.json and no SWE-bench report.json/marker (2026-09-13).  Integrity there =
    # the CTRF report exists and pytest actually collected at least one test; a verifier that died
    # before collection leaves no ctrf.json and its reward.txt zero stays void, as on SWE-bench.
    if row["subset"] == "tb21":
        ctrf = read_json(os.path.join(vdir, "ctrf.json"))
        summary = ((ctrf or {}).get("results") or {}).get("summary") or {}
        row["verifier_ok"] = bool(ctrf is not None and (summary.get("tests") or 0) >= 1)
    else:
        row["verifier_ok"] = bool(marker and os.path.exists(os.path.join(vdir, "report.json")))
    if reward is not None and not row["verifier_ok"]:   # any reward without a parser run is void
        reward = None
        exc_type = exc_type or ("VerifierUvMissing" if uv_missing else "verifier_incomplete")
    row["reward"] = float(reward) if reward is not None else float("nan")
    row["scored"] = reward is not None
    row["exception_type"] = exc_type

    # metrics: ATIF first, raw log as fallback (running trials / crashed before ATIF)
    metrics = None
    traj = read_json(traj_path) if os.path.exists(traj_path) else None
    if traj is not None:
        try:
            metrics = metrics_from_atif(traj)
        except (KeyError, TypeError, ValueError):
            metrics = None
    n_log = 0
    cc_info = {"is_error": False}
    first = last = None
    if os.path.exists(cc_log):
        m2, n_log, first, last, cc_info = scan_claude_code_log(cc_log)
        if metrics is None:
            metrics = m2
        elif m2 is not None:
            for k in ("agent_version", "model_name"):
                metrics.setdefault(k, m2.get(k))
        row["context_overflow"] = file_has_overflow(cc_log)
        if row["agent_seconds"] is None and cc_info.get("duration_ms"):
            row["agent_seconds"] = round(cc_info["duration_ms"] / 1000.0, 1)
    elif os.path.exists(oc_log):
        m2, n_log, first, last = scan_opencode_log(oc_log)
        if metrics is None:
            metrics = m2
        row["context_overflow"] = file_has_overflow(oc_log)
        # host-a 2026-09-08: under heavy host load `opencode run` sometimes never leaves startup
        # (git child <defunct>, Bun stuck in futex): 0-byte opencode.txt, no trajectory, trial ends
        # only by agent timeout / watchdog TERM / cancel. Infra, not a model outcome -> not scored.
        if (res is not None and traj is None and os.path.getsize(oc_log) == 0
                and exc_type not in ("NetworkConnectionError", "AgentSetupTimeoutError")):
            row["exception_type"] = exc_type = "OpenCodeStartupHang"
            row["scored"] = False
            row["reward"] = float("nan")
        # first event a top-level API error and not a single tool call afterwards (401 from the
        # vLLM key window, host-b 2026-09-07; 429/5xx likewise): infra, not a model outcome.
        if (res is not None and exc_type != "OpenCodeStartupHang"
                and not (metrics or {}).get("n_tool_calls") and opencode_startup_error(oc_log)):
            row["exception_type"] = exc_type = "OpenCodeStartupError"
            row["scored"] = False
            row["reward"] = float("nan")
    elif os.path.exists(dsh_log) or os.path.isdir(dsh_sessions):
        m2, first, last, dsh_end = scan_dsh_sessions(dsh_sessions)
        if metrics is None:
            metrics = m2
        row["context_overflow"] = file_has_overflow(dsh_log) if os.path.exists(dsh_log) else False
        # a finished dsh trial whose main turn did not end with kind=completed (aborted/error/
        # max-tokens, or no turn/end at all = SIGINT soft timeout) is an agent-side hard failure
        if res is not None and dsh_end != "completed":
            cc_info["is_error"] = True
            cc_info["dsh_end"] = dsh_end
    else:
        mini_log = os.path.join(agent_dir, "mini-swe-agent.txt")
        row["context_overflow"] = file_has_overflow(mini_log) if os.path.exists(mini_log) else False
    if metrics:
        row.update({k: v for k, v in metrics.items() if k in row})
    else:
        row["source"] = "none"
        row["tool_counts"] = "{}"
        for t in TOUCH_TOOLS:
            row[f"touch_{t}"] = False
    row["n_compaction_log"] = n_log
    if row["agent_seconds"] is None and first and last and last > first:
        row["agent_seconds"] = round(last - first, 1)
    if row["started_at"] is None and first:
        row["started_at"] = datetime.fromtimestamp(first, timezone.utc).isoformat()

    if row.get("completion_tokens") is not None:
        # one output figure per trial: OpenCode reports reasoning outside completion, the others inside
        row["output_tokens_total"] = row["completion_tokens"] + (
            (row.get("reasoning_tokens") or 0) if row["harness"] == OC else 0)
    # Harbor's AgentTimeoutError only cancels the client side of the exec: the agent keeps running
    # inside the container while the verifier executes (2026-09-08 audit: up to 26 min, two passes
    # among the hard45 cases). Activity more than 30 s past agent_execution.finished_at means the
    # verifier ran on a tree the agent was still editing -> infra; rerun under the soft timeout.
    if exc_type == "AgentTimeoutError" and res is not None:
        ae_end = parse_ts((res.get("agent_execution") or {}).get("finished_at"))
        last_act = last_agent_activity(agent_dir, row["harness"])
        if ae_end and last_act:
            row["post_timeout_sec"] = round(last_act - ae_end, 1)
            if last_act - ae_end > 30:
                row["exception_type"] = exc_type = "AgentVerifierOverlap"
                row["scored"] = False
                row["reward"] = float("nan")
    # HARBOR_AGENT_SOFT_TIMEOUT_SEC (our venv patch of the OpenCode/CC agents, 2026-09-08): the
    # agent binary is wrapped in coreutils timeout and leaves a marker; budget exhaustion = scored
    # fail, same convention as dsh's 46-min soft timeout.
    # ApiRateLimitError is included because Harbor's claude_code agent labels any non-zero exit as a
    # rate limit once the stream contains CC's routine `rate_limit_event` lines (status "allowed");
    # the marker file is the authoritative signal (cc-opus-hard45-nw-k2-r sympy-16597, 2026-09-08).
    # Harbor classifies a failed exec by *output patterns*: a non-zero agent exit whose stream carries
    # a `curl: (N)` line (the agent's own bash output) is labeled NetworkConnectionError, the same
    # name as a real setup-stage network failure (2026-09-08: cc-ds-hard45-nw-k2 django-10554 = soft
    # timeout, mini-qwen36-pool447-nw-c sympy-19040 = context overflow; verifier ran in both). Once
    # agent_execution has started the label only means "agent exited non-zero"; the rules below decide.
    # ApiUsageLimitError joins the list for the same reason: Harbor's base ERROR_PATTERNS include
    # r"Quota exceeded." (case-insensitive) and it matched "Disk quota exceeded" in an errno/strerror
    # listing the agent printed with bash (oc-qwen36-tb21-nw-rg path-tracing-reverse, 2026-09-13;
    # the real story of that trial is a mid-run ContextOverflowError, relabeled below). Subscription
    # quota hits on CC carry "usage limit" in the result event and stay ApiUsageLimitError.
    _rtext0 = str(cc_info.get("result_text") or "")
    if (exc_type in ("NetworkConnectionError", "ApiUsageLimitError") and res is not None
            and (res.get("agent_execution") or {}).get("started_at")
            and not (exc_type == "ApiUsageLimitError"
                     and any(p in _rtext0 for p in ("session limit", "usage limit")))):
        row["harbor_exception"] = exc_type
        row["exception_type"] = exc_type = "NonZeroAgentExitCodeError"
    if (res is not None and os.path.exists(os.path.join(agent_dir, "soft-timeout.txt"))
            and exc_type in (None, "NonZeroAgentExitCodeError", "ApiRateLimitError")):
        row["exception_type"] = exc_type = "SoftTimeout"
    # Subscription quota / OAuth revocation: the CC result event says "You've hit your session limit"
    # or "OAuth access token has been revoked". Harbor labels them ApiRateLimitError / UnknownApiError,
    # but the agent median is 1 step / 5 s and the verifier graded an untouched tree -> infra, rerun
    # (2026-09-08 review F1: 286 rows, all in web-arm / Fable / pilot jobs; the soft-timeout marker
    # rule above takes precedence). A real 429 from a local vLLM keeps its label and its reward.
    _rtext = str(cc_info.get("result_text") or "")
    if exc_type in ("ApiRateLimitError", "UnknownApiError") and any(
            p in _rtext for p in ("session limit", "usage limit",
                                  "OAuth access token has been revoked", "Failed to authenticate")):
        row["harbor_exception"] = exc_type
        row["exception_type"] = exc_type = "ApiQuotaOrAuthError"
        row["scored"] = False
        row["reward"] = float("nan")
    # A trial without result.json was cut off by a job stop; its reward.txt is only trustworthy if the
    # agent actually produced steps (2026-09-07 21:05 docker migration, 6 rows; review F2).
    if row["status"] == "running" and row["scored"] and not (row.get("agent_steps") or 0):
        row["exception_type"] = exc_type = "AgentIncomplete"
        row["scored"] = False
        row["reward"] = float("nan")
    # CC's own turn cap: max_turns 300 -> result subtype error_max_turns (is_error true); the CC
    # build in use exits non-zero (Harbor: NonZeroAgentExitCodeError), older ones exited 0 (no
    # label). Budget exhaustion = scored fail, the SoftTimeout / ContextWindowExceeded convention
    # (cc-qwen36-tb21-nw-c fix-ocaml-gc 2026-09-13: 301 turns, exit 1, verifier 0).
    if (cc_info.get("result_subtype") == "error_max_turns" and row["scored"]
            and exc_type in (None, "NonZeroAgentExitCodeError")):
        row["harbor_exception"] = exc_type or ""
        row["exception_type"] = exc_type = "StepCapReached"
    if (exc_type in ("NetworkConnectionError", "NonZeroAgentExitCodeError", "ApiRateLimitError")
            and row["context_overflow"] and row["scored"]):
        # host-b 2026-09-08: mini-swe-agent died with litellm ContextWindowExceededError after hours of
        # work; Harbor wrapped the exit-1 as NetworkConnectionError (or, since 08:50, as
        # NonZeroAgentExitCodeError). Verifier still ran. Budget exhaustion = scored fail.
        row["exception_type"] = exc_type = "ContextWindowExceeded"
    verifier_only_exc = exc_type in ("RewardFileNotFoundError", "verifier_incomplete", "VerifierUvMissing",
                                     "VerifierTimeoutError")
    row["hard_failure"] = bool(
        (exc_type and not verifier_only_exc)
        or cc_info.get("is_error")
        # CC compacts and can recover from an overflow mid-run; only a terminal error counts there
        or (row["context_overflow"] and (row["harness"] != CC or cc_info.get("is_error") or bool(exc_type)))
        or (res is not None and traj is None and not verifier_only_exc
            and row["harness"] != DSH)          # dsh writes no ATIF; its sessions are parsed above
    )
    if row["hard_failure"] and not exc_type and cc_info.get("is_error"):
        row["exception_type"] = ("dsh_turn_" + str(cc_info.get("dsh_end") or "no_end")
                                 if "dsh_end" in cc_info else "cc_api_error")
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs-root", default=os.environ.get("HARNESS_JOBS_ROOT", DEFAULT_JOBS_ROOT))
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data"))
    ap.add_argument("--jobs", default="*", help="glob over job names (default all)")
    ap.add_argument("--merge-json", nargs="*", default=[],
                    help="trials.json files produced by this extractor on a second host (data/remote/; those "
                         "jobs are not mirrored locally); a remote job replaces the local rows of the same job "
                         "whenever it holds more trials")
    args = ap.parse_args()

    rows = []
    unknown = set()
    for job_dir in sorted(glob.glob(os.path.join(args.jobs_root, "*"))):
        job = os.path.basename(job_dir)
        if not os.path.isdir(job_dir) or not fnmatch.fnmatch(job, args.jobs):
            continue
        if job not in JOB_ARMS:
            unknown.add(job)
        for td in sorted(glob.glob(os.path.join(job_dir, "*__*"))):
            if os.path.isdir(td):
                try:
                    rows.append(extract_trial(job, td))
                except Exception as ex:  # never let one broken trial kill the sweep
                    print(f"WARN {job}/{os.path.basename(td)}: {type(ex).__name__}: {ex}",
                          file=sys.stderr)

    for mj in args.merge_json:
        remote = json.load(open(mj))
        local_n = collections.Counter(r["job"] for r in rows)
        remote_n = collections.Counter(r["job"] for r in remote)
        take = {j for j, n in remote_n.items() if n > local_n.get(j, 0)}
        rows = [r for r in rows if r["job"] not in take] + [r for r in remote if r["job"] in take]
        print(f"merged {mj}: jobs {sorted(take)}", file=sys.stderr)
        unknown |= {j for j in take if j not in JOB_ARMS}

    # attempt index within a job: some jobs hold k=2 attempts of a task under repeat=1
    # (qwen-hard45{,-nw}, mini-/cc-qwen-hard45-nw); consumers can average per attempt instead of
    # keeping only the latest (2026-09-08 review F3).
    by_task = {}
    for r in rows:
        by_task.setdefault((r["job"], r["task"]), []).append(r)
    for grp in by_task.values():
        grp.sort(key=lambda r: (r.get("started_at") or "", r.get("trial_id") or ""))
        for i, r in enumerate(grp, 1):
            r["attempt"] = i
    for r in rows:
        r.setdefault("harbor_exception", None)

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "trials.csv")
    json_path = os.path.join(args.out_dir, "trials.json")
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            out = {}
            for k in COLUMNS:
                v = r.get(k)
                if isinstance(v, float) and math.isnan(v):
                    v = "NaN"
                elif isinstance(v, bool):
                    v = int(v)
                out[k] = v
            w.writerow(out)
    with open(json_path, "w") as fh:
        json.dump([{k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in r.items()} for r in rows], fh, indent=1)

    n_run = sum(r["status"] == "running" for r in rows)
    n_scored = sum(bool(r["scored"]) for r in rows)
    n_hard = sum(bool(r["hard_failure"]) for r in rows)
    print(f"wrote {len(rows)} trials -> {csv_path} ({n_scored} scored, {n_run} running, "
          f"{n_hard} hard failures)")
    if unknown:
        print("jobs not in JOB_ARMS (arm=UNKNOWN):", ", ".join(sorted(unknown)))


if __name__ == "__main__":
    main()
