"""Shared aggregation: the canonical-trial rule of main_table.py, extended so the Qwen3.8
within-job k=2 attempts become separate seeds.  Read-only."""
import json, os
from collections import defaultdict

# data/trials.json written by analysis/extract_trials.py; override with env TRIALS=...
TRIALS = os.environ.get("TRIALS", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                               "..", "data", "trials.json"))

INFRA = {"NetworkConnectionError", "AgentSetupTimeoutError", "RuntimeError", "CancelledError",
         "OpenCodeStartupHang", "OpenCodeStartupError", "AgentVerifierOverlap",
         "VerifierTimeoutError", "verifier_incomplete", "VerifierUvMissing",
         "RewardFileNotFoundError", "ApiQuotaOrAuthError", "AgentIncomplete",
         "cc_api_error"}   # keep identical to analysis/main_table.py INFRA (14 types)

# Qwen3.8 base jobs carry both seeds inside one job (90 rows); seed = per-task attempt index
# ordered by started_at.  Their retry jobs replace one attempt slot (notes in extract_trials.py).
QWEN38_BASE = {"mini-qwen-hard45-nw", "cc-qwen-hard45-nw", "qwen-hard45-nw"}
QWEN38_RETRY = {"mini-qwen-hard45-nw-r": "mini-qwen-hard45-nw",
                "cc-qwen-hard45-nw-r": "cc-qwen-hard45-nw",
                "qwen-hard45-nw-r": "qwen-hard45-nw",
                "qwen-hard45-nw-r2": "qwen-hard45-nw"}

def ok(r):
    return bool(r and r["scored"] and r["exception_type"] not in INFRA)

def load(subset="hard45", arm="control-nw"):
    rows = [r for r in json.load(open(TRIALS))
            if r["subset"] == subset and r["arm"] == arm and not r["exclude"]]
    return rows

def _slot(base_atts, retry):
    """Which attempt slot a retry row replaces: the unique unusable slot, else the slot whose
    base attempt carried the agent-side exception the retry note names."""
    bad = [i for i, r in enumerate(base_atts) if not ok(r)]
    if len(bad) == 1:
        return bad[0]
    exc = [i for i, r in enumerate(base_atts) if r["exception_type"]]
    if len(exc) == 1:
        return exc[0]
    return len(base_atts) - 1

def best_by_seed(rows):
    """(harness, model, seed) -> task -> chosen row."""
    # 1. Qwen3.8: split base jobs into attempt slots
    slots = defaultdict(dict)     # (job, task) -> {slot_idx: row}
    for job in QWEN38_BASE:
        per = defaultdict(list)
        for r in rows:
            if r["job"] == job:
                per[r["task"]].append(r)
        for t, atts in per.items():
            atts.sort(key=lambda x: x["started_at"] or "")
            slots[(job, t)] = {i: r for i, r in enumerate(atts)}
    for rjob, base in QWEN38_RETRY.items():
        for r in rows:
            if r["job"] != rjob:
                continue
            d = slots.get((base, r["task"]))
            if d is None:
                continue
            atts = [d[i] for i in sorted(d)]
            d[_slot(atts, r)] = r
    best = defaultdict(dict)
    handled = set()
    for (job, t), d in slots.items():
        h = m = None
        for i in sorted(d):
            r = d[i]
            h, m = r["harness"], r["model"]
            best[(h, m, i + 1)][t] = r
            handled.add(r["trial_id"])
    for r in rows:
        if r["trial_id"] in handled or r["job"] in QWEN38_BASE or r["job"] in QWEN38_RETRY:
            continue
        key = (r["harness"], r["model"], r["repeat"] or 1)
        cur = best[key].get(r["task"])
        if cur is None or (ok(r), r.get("finished_at") or "") > (ok(cur), cur.get("finished_at") or ""):
            best[key][r["task"]] = r
    return best

MODEL_ORDER = ["opus5", "ds-v4-flash", "qwen3.8-27b", "qwen3.6-35b-a3b"]
MODEL_LABEL = {"opus5": "Opus 5", "ds-v4-flash": "DeepSeek-V4-Flash",
               "qwen3.8-27b": "Qwen3.8-27B", "qwen3.6-35b-a3b": "Qwen3.6-35B-A3B"}
HARNESS_ORDER = ["mini-swe-agent", "opencode", "claude-code", "dsh"]
HLAB = {"mini-swe-agent": "mini", "opencode": "OC", "claude-code": "CC", "dsh": "dsh"}
