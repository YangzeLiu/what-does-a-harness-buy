#!/usr/bin/env python3
"""Freeze the stratified SWE-bench Verified subsets for the agent-harness ablation.

Produces (all under the directory holding this script):

    pilot30.json            n=30  pilot list   (subset of main150)
    main150.json            n=150 main list
    selection_protocol.md   protocol + repo x difficulty count tables + proxy summary

The paper uses the sampling pool this script defines (SWE-bench Verified minus
the excluded repo, listed in pool492.txt) and two sets derived from it:
hard45 (every pool task in the "1-4 hours" and ">4 hours" bands) and
pool447_rest.txt (the remaining 447).  main150/pilot30 are an earlier
stratified design.  They are kept because the pilot30 trials in
data/trials.csv were run on them.

Stratification
--------------
1. repo                 (12 repos in Verified, 11 after the EXCLUDED_REPOS filter)
2. difficulty           (dataset field literally named ``difficulty``; values
                         "<15 min fix" / "15 min - 1 hour" / "1-4 hours" / ">4 hours")
3. expected-context proxy, used *within* each repo x difficulty cell to force a
   high/low mix so that long-context tasks are represented.

Sampling is deterministic and re-running reproduces byte-identical artifacts.
``pilot30 ⊂ main150`` holds by construction: main150 is drawn from the pool, then
pilot30 is drawn from main150 with the same procedure.

REPRODUCIBILITY NOTE.  The default ``--mode incremental`` treats the existing
main150.json / pilot30.json as a carry-over baseline and only patches them onto
the current target, so the committed JSON files are *part of the input*, not just
output.  Re-running with them present is a no-op.  Deleting them and re-running
falls back to a from-scratch draw, which is a DIFFERENT (equally valid) sample --
so recover them from git rather than regenerating if they ever go missing.

Requires ``datasets``, ``pandas``, ``numpy`` (no scipy).

Usage
-----
    python select_subset.py                      # write all three artifacts
    python select_subset.py --report             # print tables, write nothing

If HF_ENDPOINT points at a Hugging Face mirror, its host is added to NO_PROXY
so that a local http(s)_proxy / ALL_PROXY does not intercept it (see
load_frame()).  With the dataset already cached, HF_DATASETS_OFFLINE=1 works.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants -- the frozen protocol.  Changing any of these changes the subsets.
# --------------------------------------------------------------------------

PROTOCOL_VERSION = "1.1"

DATASET = "princeton-nlp/SWE-bench_Verified"
SPLIT = "test"
SEED = 20260901

N_MAIN = 150
N_PILOT = 30

# --- repo-level exclusions from the sampling pool -------------------------
# Keyed by repo -> (reason, evidence, date).  A repo listed here is dropped
# from the pool *before* any quota is computed, so its slots flow back through
# the ordinary two-stage apportionment rather than being patched in by hand.
# Exclusions are protocol, not preprocessing: every entry needs a reason and a
# reproducible piece of evidence, because dropping a repo changes what the
# resolved-rate estimates generalise over (see selection_protocol.md section 8).
EXCLUDED_REPOS = {
    "psf/requests": (
        "verifier test suite reaches httpbin.org; unreachable from CN, the "
        "verifier blocks indefinitely instead of failing fast",
        "Harbor smoke test on psf__requests-2317 hung to the 3000s timeout",
        "2026-09-01",
    ),
}

# difficulty stratum order (easy -> hard), used for stable sorting / tables
DIFFICULTY_ORDER = [
    "<15 min fix",
    "15 min - 1 hour",
    "1-4 hours",
    ">4 hours",
]

# ">4 hours" is the "weak models wipe out" stratum: low discriminative value for
# component ablation, so it is capped rather than allowed to float upward.
HARD_STRATUM = ">4 hours"
HARD_CAP_FRAC = 0.10

# every repo present in the frame must land at least this many slots in main150
MIN_PER_REPO_MAIN = 1
# pilot30 must touch at least this many distinct repos (sanity requirement)
MIN_REPO_COVERAGE_PILOT = 6

# expected-context proxy: mean of the within-500 percentile ranks of
# (problem_statement word count) and (gold patch line count).  The two raw
# quantities are almost uncorrelated on Verified (Spearman ~0.07), so both carry
# independent information; equal weighting is the neutral default.
PROXY_WEIGHTS = {"ps_words": 0.5, "patch_lines": 0.5}

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Frame construction
# --------------------------------------------------------------------------


def load_frame() -> pd.DataFrame:
    """Load SWE-bench Verified and attach the derived stratification columns."""
    # A local VPN/SOCKS proxy in the environment can break access to a
    # Hugging Face mirror (datasets>=5 routes through httpx, which then needs
    # socksio).  If a mirror is configured, bypass the proxy for its host.
    if os.environ.get("HF_ENDPOINT"):
        host = os.environ["HF_ENDPOINT"].split("://")[-1].split("/")[0]
        for var in ("no_proxy", "NO_PROXY"):
            cur = os.environ.get(var, "")
            if host not in cur:
                os.environ[var] = f"{cur},{host}" if cur else host

    from datasets import load_dataset  # imported late: heavy, and only needed here

    ds = load_dataset(DATASET, split=SPLIT)
    df = ds.to_pandas()

    missing = {"instance_id", "repo", "difficulty", "problem_statement", "patch"} - set(
        df.columns
    )
    if missing:
        raise RuntimeError(f"dataset schema changed, missing columns: {sorted(missing)}")

    df["ps_words"] = df["problem_statement"].str.split().str.len().astype(int)
    df["ps_chars"] = df["problem_statement"].str.len().astype(int)
    # gold patch size: total diff lines, plus added/removed for downstream use
    df["patch_lines"] = df["patch"].str.count("\n").astype(int) + 1
    df["patch_added"] = (
        df["patch"].str.count(r"(?m)^\+(?!\+\+)").astype(int)
    )
    df["patch_removed"] = (
        df["patch"].str.count(r"(?m)^-(?!--)").astype(int)
    )

    df["excluded"] = df["repo"].isin(EXCLUDED_REPOS)

    # Percentile ranks and context bands are defined over the *sampling pool*,
    # not the raw 500: the pool is the population the subsets are drawn from and
    # that the estimates generalise over, so the margins the sampler is held to
    # must be the pool's margins.  Excluded rows keep NaN here.
    pool = ~df["excluded"]
    ps_rank = df.loc[pool, "ps_words"].rank(pct=True, method="average")
    patch_rank = df.loc[pool, "patch_lines"].rank(pct=True, method="average")
    df["ctx_proxy"] = (
        PROXY_WEIGHTS["ps_words"] * ps_rank + PROXY_WEIGHTS["patch_lines"] * patch_rank
    )
    df["ctx_band"] = ""
    df.loc[pool, "ctx_band"] = pd.qcut(
        df.loc[pool, "ctx_proxy"].rank(method="first"),
        4,
        labels=["Q1", "Q2", "Q3", "Q4"],
    ).astype(str)

    df["cell"] = list(zip(df["repo"], df["difficulty"]))
    df = df.sort_values("instance_id").reset_index(drop=True)
    return df


def frame_fingerprint(df: pd.DataFrame) -> str:
    """Hash of the sorted instance_id list: detects silent dataset drift."""
    blob = "\n".join(sorted(df["instance_id"])).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


# --------------------------------------------------------------------------
# Apportionment: proportional allocation with largest-remainder rounding,
# then the two protocol constraints (hard-stratum cap, per-repo floor).
# --------------------------------------------------------------------------


def _cell_key(cell):
    """Deterministic, human-meaningful ordering key for a (repo, difficulty) cell."""
    repo, diff = cell
    return (repo, DIFFICULTY_ORDER.index(diff))


def apportion(counts: dict, n_target: int, keyfn=None) -> tuple[dict, dict]:
    """Largest-remainder (Hamilton) apportionment of n_target over cells.

    ``counts`` maps cell -> population size in the frame.  No cell may receive
    more slots than it has members.  Ties break on cell size (descending) then
    on ``keyfn`` so the result is fully deterministic.
    """
    keyfn = keyfn or (lambda c: c)
    pop = sum(counts.values())
    if n_target > pop:
        raise ValueError(f"cannot draw {n_target} from a frame of {pop}")
    quota = {c: counts[c] * n_target / pop for c in counts}
    alloc = {c: min(int(math.floor(quota[c])), counts[c]) for c in counts}

    # round 1: hand out leftovers by descending fractional remainder
    order = sorted(
        counts,
        key=lambda c: (-(quota[c] - math.floor(quota[c])), -counts[c], keyfn(c)),
    )
    for c in order:
        if sum(alloc.values()) >= n_target:
            break
        if alloc[c] < counts[c]:
            alloc[c] += 1

    # round 2 (only if capacity constraints bit): fill by largest quota deficit
    while sum(alloc.values()) < n_target:
        cands = [c for c in counts if alloc[c] < counts[c]]
        if not cands:
            raise RuntimeError("apportionment stuck: no capacity left")
        c = min(cands, key=lambda c: (-(quota[c] - alloc[c]), -counts[c], keyfn(c)))
        alloc[c] += 1
    return alloc, quota


def allocate(df: pd.DataFrame, n_target: int, min_per_repo: int) -> tuple[dict, list]:
    """Two-stage allocation: difficulty margin first, then repos inside it.

    Allocating directly over the ~32 non-empty repo x difficulty cells is
    fragile: a thin stratum spread over several repos (``>4 hours``: 3 instances
    across 3 repos, each with quota 0.3) loses every largest-remainder contest
    and disappears from the sample entirely.  Doing difficulty first pins that
    margin to its proportional value (0.9 -> 1 slot) and only then splits each
    stratum across repos, so no stratum can be rounded out of existence.
    """
    log = []
    pop = len(df)

    # ---- stage 1: difficulty margin ---------------------------------------
    dcounts = df.groupby("difficulty").size().to_dict()
    dalloc, dquota = apportion(dcounts, n_target, keyfn=DIFFICULTY_ORDER.index)

    cap = int(math.floor(HARD_CAP_FRAC * n_target))
    hard = dalloc.get(HARD_STRATUM, 0)
    if hard > cap:
        log.append(
            f"'{HARD_STRATUM}' cap BINDING: proportional gave {hard}, cap {cap}"
        )
        while dalloc[HARD_STRATUM] > cap:
            dalloc[HARD_STRATUM] -= 1
            r = min(
                (d for d in dalloc if d != HARD_STRATUM and dalloc[d] < dcounts[d]),
                key=lambda d: (-(dquota[d] - dalloc[d]), -dcounts[d],
                               DIFFICULTY_ORDER.index(d)),
            )
            dalloc[r] += 1
            log.append(f"  moved 1 slot '{HARD_STRATUM}' -> '{r}'")
    else:
        log.append(
            f"'{HARD_STRATUM}' cap NOT binding: proportional gave {hard} "
            f"(quota {dquota.get(HARD_STRATUM, 0):.2f}) <= cap {cap}"
        )
    log.append(
        "difficulty margin: "
        + ", ".join(
            f"{d}={dalloc.get(d,0)} (quota {dquota.get(d,0):.2f})"
            for d in DIFFICULTY_ORDER
            if d in dcounts
        )
    )

    # ---- stage 2: repos inside each difficulty stratum ---------------------
    alloc = {}
    for diff, k in dalloc.items():
        rc = df[df.difficulty == diff].groupby("repo").size().to_dict()
        ra, _ = apportion(rc, k)
        for repo, v in ra.items():
            alloc[(repo, diff)] = v
    counts = df.groupby("cell").size().to_dict()
    quota = {c: counts[c] * n_target / pop for c in counts}

    # ---- constraint: per-repo floor, repaired inside the same stratum ------
    # (swapping within a stratum keeps the stage-1 difficulty margin exact)
    if min_per_repo > 0:
        repo_tot = {}
        for (repo, _), v in alloc.items():
            repo_tot[repo] = repo_tot.get(repo, 0) + v
        for repo in sorted(repo_tot):
            while repo_tot[repo] < min_per_repo:
                recv = [c for c in alloc if c[0] == repo and alloc[c] < counts[c]]
                if not recv:
                    raise RuntimeError(f"repo {repo} cannot reach floor")
                r = max(recv, key=lambda c: (counts[c], _cell_key(c)))
                donors = [
                    c
                    for c in alloc
                    if c[1] == r[1]  # same difficulty stratum
                    and c[0] != repo
                    and alloc[c] > 0
                    and repo_tot[c[0]] - 1 >= min_per_repo
                ]
                if not donors:
                    raise RuntimeError(f"no donor for repo floor of {repo}")
                d = min(
                    donors,
                    key=lambda c: (-(alloc[c] - quota[c]), -counts[c], _cell_key(c)),
                )
                alloc[r] += 1
                alloc[d] -= 1
                repo_tot[repo] += 1
                repo_tot[d[0]] -= 1
                log.append(
                    f"repo floor: {repo} below {min_per_repo}; moved 1 slot "
                    f"{d} -> {r}"
                )
    if sum(alloc.values()) != n_target:
        raise RuntimeError("allocation total drifted")
    return alloc, log


# --------------------------------------------------------------------------
# Within-cell draw: high/low context mixing
# --------------------------------------------------------------------------


def _systematic(sorted_ids, k, rng):
    """Systematic sample of k from a proxy-sorted list, random start.

    Splits the cell into k equal-width proxy bands and takes exactly one member
    from each: guarantees the draw spans the cell's full context range instead
    of clumping at one end (the high/low context mix requirement).
    """
    n = len(sorted_ids)
    u = float(rng.random())
    step = n / k
    idx = [min(n - 1, int(math.floor((i + u) * step))) for i in range(k)]
    if len(set(idx)) != k:  # defensive; cannot happen for k <= n
        idx = sorted(rng.choice(n, size=k, replace=False).tolist())
    return [sorted_ids[i] for i in idx]


def draw(df: pd.DataFrame, alloc: dict, rng) -> list:
    """Draw the allocated slots cell by cell, mixing high/low context proxy.

    Cells with k>=2 use systematic sampling over the proxy-sorted members.
    Cells with k==1 admit no within-cell mixing, so they are balanced *across*
    cells: the singleton cells are shuffled and then alternately assigned to the
    upper / lower proxy half of their own cell.  Expected long-share stays ~50%
    (no systematic inflation of long tasks) while no stratum can end up
    accidentally all-short.
    """
    picked = []
    singleton_cells = []
    for cell in sorted(alloc, key=_cell_key):
        k = alloc[cell]
        if k == 0:
            continue
        sub = df[df["cell"] == cell].sort_values(
            ["ctx_proxy", "instance_id"]
        )
        ids = sub["instance_id"].tolist()
        if k == 1:
            singleton_cells.append((cell, ids))
        else:
            picked.extend(_systematic(ids, k, rng))

    # balanced high/low assignment for the k==1 cells
    order = rng.permutation(len(singleton_cells))
    for rank, j in enumerate(order):
        cell, ids = singleton_cells[j]
        n = len(ids)
        if n == 1:
            picked.append(ids[0])
            continue
        half = n // 2
        band = ids[half:] if rank % 2 == 0 else ids[:half]  # upper / lower half
        picked.append(band[int(rng.integers(len(band)))])

    if len(set(picked)) != len(picked):
        raise RuntimeError("duplicate draw")
    return picked


def topup(df: pd.DataFrame, alloc: dict, carry: list, log: list) -> list:
    """Minimal-perturbation refresh: keep an earlier list, fix it to a new target.

    Drawing a fresh sample after a pool change is statistically clean but
    operationally destructive: excluding one repo shifts every quota denominator
    and every percentile rank, so a clean-slate redraw churned 72 of 150
    instances -- 72 new container images to mirror, and every pilot rollout
    already spent thrown away, to fix an 8-instance problem.

    So instead: carry the previous list forward, drop what is no longer eligible
    or no longer fits the new per-cell target, and top the deficit cells back up.
    The result satisfies the new allocation *exactly* -- it is the same stratified
    design, just realised with maximal reuse of the units already drawn.  Both
    drop and add pick the member that best moves the ctx_band margin toward
    balance, so the context margin is preserved as the list is patched.

    Fully deterministic and RNG-free: ties break on instance_id.  That also means
    the carried-over instances stay put regardless of how the RNG stream shifted.
    """
    pool_ids = set(df["instance_id"])
    info = df.set_index("instance_id")
    keep = [i for i in carry if i in pool_ids]
    dropped_ineligible = [i for i in carry if i not in pool_ids]
    if dropped_ineligible:
        log.append(
            f"  carry-over: {len(dropped_ineligible)} instance(s) no longer in the "
            f"pool -> {sorted(dropped_ineligible)}"
        )

    bands = ["Q1", "Q2", "Q3", "Q4"]

    def band_counts(sel):
        c = {b: 0 for b in bands}
        for i in sel:
            c[info.loc[i, "ctx_band"]] += 1
        return c

    n_target = sum(alloc.values())
    band_target, _ = apportion(
        {
            b: int((df["ctx_band"] == b).sum())
            for b in bands
            if int((df["ctx_band"] == b).sum())
        },
        n_target,
        keyfn=bands.index,
    )

    # ---- drop where the carried list now exceeds its cell target -----------
    by_cell = {}
    for i in keep:
        by_cell.setdefault(info.loc[i, "cell"], []).append(i)
    for cell in sorted(by_cell, key=_cell_key):
        want = alloc.get(cell, 0)
        while len(by_cell[cell]) > want:
            cur = band_counts([x for v in by_cell.values() for x in v])
            # drop from the most over-represented band in this cell
            victim = min(
                sorted(by_cell[cell]),
                key=lambda i: (
                    -(cur[info.loc[i, "ctx_band"]] - band_target.get(info.loc[i, "ctx_band"], 0)),
                    i,
                ),
            )
            by_cell[cell].remove(victim)
            log.append(f"  carry-over drop (cell over target) {cell}: {victim}")

    # ---- add where the carried list falls short ---------------------------
    for cell in sorted(alloc, key=_cell_key):
        have = by_cell.get(cell, [])
        want = alloc.get(cell, 0)
        while len(have) < want:
            taken = {x for v in by_cell.values() for x in v}
            cur = band_counts(taken)
            cands = sorted(
                df[(df["cell"] == cell) & (~df["instance_id"].isin(taken))][
                    "instance_id"
                ].tolist()
            )
            if not cands:
                raise RuntimeError(f"cell {cell} cannot reach target {want}")
            # add from the most under-represented band
            pick = min(
                cands,
                key=lambda i: (
                    -(band_target.get(info.loc[i, "ctx_band"], 0) - cur[info.loc[i, "ctx_band"]]),
                    i,
                ),
            )
            have.append(pick)
            by_cell[cell] = have
            log.append(
                f"  carry-over add  (cell under target) {cell}: {pick} "
                f"({info.loc[pick, 'ctx_band']})"
            )

    picked = [x for v in by_cell.values() for x in v]
    if len(picked) != n_target or len(set(picked)) != len(picked):
        raise RuntimeError("top-up produced a malformed list")
    log.append(
        f"  carry-over result: kept {len(set(picked) & set(carry))} of {len(carry)}, "
        f"replaced {len(set(carry) - set(picked))}"
    )
    return picked


def repair_proxy_margin(df: pd.DataFrame, picked: list, n_target: int, log: list):
    """Nudge the context-proxy quartile margin onto its proportional target.

    The cell-by-cell draw fixes the repo x difficulty margin but leaves the
    context-length margin to chance, and at n=30 that chance is expensive: a
    pilot that happens to skew short would under-measure how often long-context
    behaviour (such as compaction) is triggered.

    Repair by swapping instances *within the same repo x difficulty cell*, so
    the primary allocation is preserved exactly while the proxy-band counts move
    toward target.  Deterministic: candidates are visited in instance_id order.
    """
    bands = ["Q1", "Q2", "Q3", "Q4"]
    band_pop = df.groupby("ctx_band").size().to_dict()
    target, _ = apportion(
        {b: band_pop.get(b, 0) for b in bands if band_pop.get(b, 0)},
        n_target,
        keyfn=bands.index,
    )
    info = df.set_index("instance_id")
    picked = list(picked)

    def counts_now(sel):
        c = {b: 0 for b in bands}
        for i in sel:
            c[info.loc[i, "ctx_band"]] += 1
        return c

    before = counts_now(picked)
    for _ in range(4 * n_target):
        cur = counts_now(picked)
        over = [b for b in bands if cur[b] > target.get(b, 0)]
        under = [b for b in bands if cur[b] < target.get(b, 0)]
        if not over or not under:
            break
        swapped = False
        # prefer the largest imbalance first, deterministic tie-break
        over.sort(key=lambda b: (-(cur[b] - target.get(b, 0)), bands.index(b)))
        under.sort(key=lambda b: (-(target.get(b, 0) - cur[b]), bands.index(b)))
        sel = set(picked)
        for ob in over:
            for ub in under:
                cands = sorted(
                    i for i in picked if info.loc[i, "ctx_band"] == ob
                )
                for out_id in cands:
                    cell = info.loc[out_id, "cell"]
                    pool = sorted(
                        df[
                            (df["cell"] == cell)
                            & (df["ctx_band"] == ub)
                            & (~df["instance_id"].isin(sel))
                        ]["instance_id"].tolist()
                    )
                    if pool:
                        in_id = pool[0]
                        picked[picked.index(out_id)] = in_id
                        log.append(
                            f"  proxy-margin swap in cell {cell}: "
                            f"{out_id} ({ob}) -> {in_id} ({ub})"
                        )
                        swapped = True
                        break
                if swapped:
                    break
            if swapped:
                break
        if not swapped:
            log.append("  proxy-margin repair: no feasible swap left, stopping")
            break
    after = counts_now(picked)
    log.append(
        "  ctx_proxy quartile margin  target="
        + str([target.get(b, 0) for b in bands])
        + "  before="
        + str([before[b] for b in bands])
        + "  after="
        + str([after[b] for b in bands])
    )
    return picked


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------


def records(df: pd.DataFrame, ids: list) -> list:
    sub = df[df["instance_id"].isin(ids)].copy()
    sub["_d"] = sub["difficulty"].map(DIFFICULTY_ORDER.index)
    sub = sub.sort_values(["repo", "_d", "instance_id"])
    out = []
    for _, r in sub.iterrows():
        out.append(
            {
                "instance_id": r["instance_id"],
                "repo": r["repo"],
                "base_commit": r["base_commit"],
                "difficulty": r["difficulty"],
                "problem_statement_words": int(r["ps_words"]),
                "problem_statement_chars": int(r["ps_chars"]),
                "patch_lines": int(r["patch_lines"]),
                "patch_added_lines": int(r["patch_added"]),
                "patch_removed_lines": int(r["patch_removed"]),
                "ctx_proxy_pct": round(float(r["ctx_proxy"]), 4),
                "ctx_band": r["ctx_band"],
            }
        )
    return out


def three_column_table(df_all, main_ids, pilot_ids) -> str:
    """repo x difficulty counts: pilot30 / main150 / pool / raw 500.

    Excluded repos are kept as rows (flagged) rather than hidden, so the table
    shows exactly what left the pool and what it cost.
    """
    pool = df_all[~df_all["excluded"]]
    lines = [
        f"| {'repo':<26} | {'difficulty':<16} | n30 | n150 | pool | n500 | |",
        "|" + "-" * 28 + "|" + "-" * 18 + "|----:|-----:|-----:|-----:|--|",
    ]
    full = df_all.groupby(["repo", "difficulty"]).size().to_dict()
    pl = pool.groupby(["repo", "difficulty"]).size().to_dict()
    m = df_all[df_all.instance_id.isin(main_ids)].groupby(["repo", "difficulty"]).size().to_dict()
    p = df_all[df_all.instance_id.isin(pilot_ids)].groupby(["repo", "difficulty"]).size().to_dict()
    for cell in sorted(full, key=_cell_key):
        repo, diff = cell
        flag = "**EXCL**" if repo in EXCLUDED_REPOS else ""
        lines.append(
            f"| {repo:<26} | {diff:<16} | {p.get(cell,0):>3} | "
            f"{m.get(cell,0):>4} | {pl.get(cell,0):>4} | {full[cell]:>4} | {flag} |"
        )
    lines.append(
        f"| {'TOTAL':<26} | {'':<16} | {len(pilot_ids):>3} | "
        f"{len(main_ids):>4} | {len(pool):>4} | {len(df_all):>4} | |"
    )
    return "\n".join(lines)


def marginal_table(df_all, main_ids, pilot_ids, col) -> str:
    pool = df_all[~df_all["excluded"]]
    order = DIFFICULTY_ORDER if col == "difficulty" else None
    full = df_all.groupby(col).size()
    poolc = pool.groupby(col).size()
    m = df_all[df_all.instance_id.isin(main_ids)].groupby(col).size()
    p = df_all[df_all.instance_id.isin(pilot_ids)].groupby(col).size()
    keys = order if order else full.sort_values(ascending=False).index.tolist()
    keys = [k for k in keys if int(full.get(k, 0))]
    w = max(len(str(k)) for k in keys)
    lines = [
        f"| {col:<{w}} | n30 | %30 | n150 | %150 | pool | %pool | n500 |",
        "|" + "-" * (w + 2) + "|----:|----:|-----:|-----:|-----:|------:|-----:|",
    ]
    for k in keys:
        fv, cv = int(full.get(k, 0)), int(poolc.get(k, 0))
        mv, pv = int(m.get(k, 0)), int(p.get(k, 0))
        lines.append(
            f"| {str(k):<{w}} | {pv:>3} | {100*pv/len(pilot_ids):>4.1f} | "
            f"{mv:>4} | {100*mv/len(main_ids):>5.1f} | {cv:>4} | "
            f"{100*cv/len(pool):>5.1f} | {fv:>4} |"
        )
    return "\n".join(lines)


def proxy_summary(df_all, main_ids, pilot_ids) -> str:
    # everything here is relative to the pool, which is what the subsets are
    # drawn from and what the estimates generalise over
    df = df_all[~df_all["excluded"]]
    sets = [
        (f"pool{len(df)}", df),
        ("main150", df[df.instance_id.isin(main_ids)]),
        ("pilot30", df[df.instance_id.isin(pilot_ids)]),
    ]
    q_ps = df["ps_words"].quantile([0.25, 0.5, 0.75]).tolist()
    q_pl = df["patch_lines"].quantile([0.25, 0.5, 0.75]).tolist()
    lines = [
        "| set      | metric                    |  min |  p25 |  p50 |  p75 |  p90 |   max |  mean |",
        "|----------|---------------------------|-----:|-----:|-----:|-----:|-----:|------:|------:|",
    ]
    for name, s in sets:
        for metric, col in [
            ("problem_statement words", "ps_words"),
            ("gold patch lines", "patch_lines"),
        ]:
            v = s[col]
            lines.append(
                f"| {name:<8} | {metric:<25} | {v.min():>4.0f} | {v.quantile(.25):>4.0f} | "
                f"{v.median():>4.0f} | {v.quantile(.75):>4.0f} | {v.quantile(.90):>4.0f} | "
                f"{v.max():>5.0f} | {v.mean():>5.0f} |"
            )
    lines.append("")
    lines.append(
        "Share of each list at or above the pool p75 (the long tail that drives compaction):"
    )
    lines.append("")
    lines.append("| set      | ps_words >= p75 | patch_lines >= p75 |")
    lines.append("|----------|----------------:|-------------------:|")
    for name, s in sets:
        a = 100 * (s["ps_words"] >= q_ps[2]).mean()
        b = 100 * (s["patch_lines"] >= q_pl[2]).mean()
        lines.append(f"| {name:<8} | {a:>14.1f}% | {b:>17.1f}% |")
    lines.append("")
    lines.append(
        "`ctx_band` is the quartile of `ctx_proxy` over the sampling pool (Q4 = longest). "
        "`ctx_proxy` is itself a mean of two percentiles and concentrates around 0.5, "
        "so its absolute value should not be compared with 0.75. The long tail is the Q4 "
        "column below. The draw is constrained to a balanced margin here (section 2.4):"
    )
    lines.append("")
    lines.append("| set      |  Q1 |  Q2 |  Q3 |  Q4 | Q4 share |")
    lines.append("|----------|----:|----:|----:|----:|---------:|")
    for name, s in sets:
        bc = s["ctx_band"].value_counts().to_dict()
        g = [bc.get(b, 0) for b in ["Q1", "Q2", "Q3", "Q4"]]
        lines.append(
            f"| {name:<8} | {g[0]:>3} | {g[1]:>3} | {g[2]:>3} | {g[3]:>3} | "
            f"{100*g[3]/len(s):>7.1f}% |"
        )
    return "\n".join(lines)


REVISION_LOG = """### v1.1, 2026-09-01: exclude `psf/requests`

Trigger. A Harbor smoke test found that the `psf/requests` verifier test suite
depends on `httpbin.org`. That host was unreachable from our network, and the
verifier blocks indefinitely instead of failing fast: `psf__requests-2317` hung
until the 3000 s timeout. This is an external dependency of the evaluation
setup, not a model-capability signal. Keeping the repo in the pool would turn
its rollouts into random timeouts and spend a full timeout budget per task.

Action. A repo-level exclusion constant `EXCLUDED_REPOS` (top of the script,
with reason, evidence and date) was added. An excluded repo leaves the sampling
pool before any quota is computed, so the freed slots flow back through the
two-stage rule of section 2.1 rather than being patched in by hand. The seed is
unchanged (20260901).

Cost. The pool shrinks from 500 to 492 (-8) and the repo count from 12 to 11.
In v1.0 `psf/requests` held 3 slots of main150 and 1 of pilot30. Estimates
therefore generalise over the 11-repo pool rather than all of Verified. The
exclusion is an availability filter caused by the network environment, not a
difficulty or quality filter, so it does not bias resolved rates upward. Code of
the "thin library with heavy network I/O" kind is absent from the sample.

Why not a full redraw. Excluding a repo changes (1) every quota denominator
(500 to 492), (2) every percentile rank and `ctx_band` (recomputed on 492) and
(3) the order in which the RNG is consumed. A clean redraw (`--mode fresh`)
churned 72 of 150 instances, which would mean pulling 72 new container images
and discarding pilot rollouts already run, to fix an 8-instance problem. The
default is therefore `--mode incremental` (see `topup()`). The old list is
carried forward, instances that are no longer eligible or exceed the new cell
quota are dropped, and deficit cells are filled. The final lists satisfy the
v1.1 stratified quotas exactly, realised with maximal reuse. The fill rule is
deterministic and RNG-free (greedy on the `ctx_band` margin deficit, ties broken
on `instance_id`), so retained instances do not move because of RNG drift.

Actual churn: 4 instances in main150 and 3 in pilot30.

main150, removed (4)
```
psf__requests-1766        psf/requests       <15 min fix        (repo excluded)
psf__requests-1921        psf/requests       <15 min fix        (repo excluded)
psf__requests-2931        psf/requests       15 min - 1 hour    (repo excluded)
pylint-dev__pylint-6386   pylint-dev/pylint  15 min - 1 hour    (cell over new quota)
```

main150, added (4)
```
astropy__astropy-12907    astropy/astropy    15 min - 1 hour    Q1
django__django-11087      django/django      15 min - 1 hour    Q4
pydata__xarray-4695       pydata/xarray      15 min - 1 hour    Q3
sympy__sympy-17655        sympy/sympy        <15 min fix        Q2
```

pilot30, removed (3)
```
psf__requests-1921        psf/requests       <15 min fix        (repo excluded)
pylint-dev__pylint-7080   pylint-dev/pylint  15 min - 1 hour    (cell over new quota)
pytest-dev__pytest-5809   pytest-dev/pytest  <15 min fix        (cell over new quota)
```

pilot30, added (3)
```
django__django-11820      django/django      <15 min fix        Q1   (already in v1.0 main150)
pytest-dev__pytest-5840   pytest-dev/pytest  15 min - 1 hour    Q3   (already in v1.0 main150)
sympy__sympy-17655        sympy/sympy        <15 min fix        Q2   (new in main150)
```

Net new container images: the 4 main150 additions. The other two pilot30
additions were already in v1.0 main150.

Known side effects, both within the sanity thresholds:

1. pilot30 repo coverage falls from 10 to 8 (psf/requests and pylint are
   lost), still above the floor of 6.
2. The pilot30 problem-statement long tail thins (`ps_words >= p75` falls from
   16.7% to 13.3%). The fill balances the composite `ctx_band` margin, so the
   pilot's patch-size long tail instead rises to 26.7%, and it keeps the largest
   gold patch in the dataset (523 lines). Patch size, which drives the number of
   turns and the amount of code read, is a stronger driver of compaction than
   problem-statement length, so no further constraint was added. One would only
   raise churn.

### v1.0, 2026-09-01: initial freeze

Two-stage repo x difficulty stratification plus a context-proxy margin
constraint, pool = all 500.
"""


PROTOCOL_TEMPLATE = """# SWE-bench Verified stratified subsets: sampling protocol

> This file is generated by `select_subset.py`. Do not edit it by hand. Change
> the script and rerun it: `python select_subset.py` (add
> `HF_DATASETS_OFFLINE=1` when the dataset is already cached).
> The paper uses the sampling pool defined here (`pool492.txt`) and two sets
> derived from it, `hard45.txt` (every pool task in the `1-4 hours` and
> `>4 hours` bands) and `pool447_rest.txt` (the other 447). main150 and pilot30
> are an earlier stratified design, kept because the pilot30 trials in
> `data/trials.csv` were run on them.

## 0. Frozen parameters

| item | value |
|---|---|
| protocol version | v{version} (revision log in section 9) |
| dataset | `{dataset}` (split=`{split}`, N={n_full}) |
| dataset fingerprint (sha256[:16] of sorted instance_id) | `{fingerprint}` |
| sampling pool | N={n_pool} (= {n_full} minus excluded repos, see section 8) |
| pool fingerprint | `{pool_fp}` |
| numpy seed | `{seed}` |
| pilot / main size | {n_pilot} / {n_main} |
| `>4 hours` stratum cap | {hard_cap_frac:.0%} of n |
| main per-repo floor | {min_per_repo} |
| pilot repo-coverage floor | {min_cov} |

The difficulty annotation is the top-level field `difficulty`, with values
{difficulty_values}. No join with another source is needed. Differences from
what we expected are listed in section 7.

## 1. Stratification dimensions

Dimension 1, repo. The sampling pool covers {n_repos} repos and is highly
unbalanced (django holds {django_share:.1f}%, pallets/flask has a single task).
Inside each difficulty stratum, slots are apportioned to repos in proportion to
their counts (section 2.1), with a floor of {min_per_repo} task per repo so that
small repos do not vanish from the 150. Effects can differ across code bases,
and that heterogeneity should stay measurable rather than collapse onto django.

Dimension 2, difficulty. These are the four human-annotated bands from OpenAI.
main150 is allocated in proportion to the pool, with a cap of
{hard_cap_frac:.0%} on `>4 hours`. Weak models fail that stratum almost
entirely, so its resolved variance tends to zero and it carries little
information for comparing components. The cap does not bind (see the audit in
section 3), because the pool has only {n_hard} `>4 hours` tasks
({hard_share:.1f}%) and proportional allocation gives them {main_hard} slot.

Dimension 3, expected-context proxy. It sets the high/low mix within cells and
is not a separate stratification key. `ctx_proxy` = 0.5 x (percentile of the
problem_statement word count within the pool) + 0.5 x (percentile of the gold
patch line count within the pool). The two raw quantities are nearly
uncorrelated on the pool (Spearman rho={spearman:.3f}), so each carries
independent information. Problem-statement length sets the size of the first
prompt, and gold patch size stands in for the amount of code the agent must
read and edit, and through that for the number of turns and the context growth.
Equal weights are the neutral default.

This dimension is not promoted to a third stratification key. The repo x
difficulty grid of the pool already has 32 non-empty cells, and splitting each in three
would leave mostly cells of size 1. It serves instead as the within-cell draw
order (section 2.3) and as a post-hoc margin constraint (section 2.4), so that
long tasks, where compaction triggers, are represented. The context-quartile
margins of all three lists sit near 25% each (section 5).

## 2. Sampling algorithm (deterministic, seed={seed})

### 2.1 Quotas: two-stage proportional allocation (difficulty first, then repo)

A single largest-remainder pass over the 32 non-empty repo x difficulty cells
of the pool does not work. The `>4 hours` stratum has only 3 tasks spread over 3 repos, each
of those cells gets a quota of about 0.3, and such cells never win a
largest-remainder contest, so the whole stratum is rounded out and main150 gets
none of it. The allocation therefore has two stages.

Stage 1, the difficulty margin. q_d = N_d x n / N_frame with largest-remainder
rounding. The `>4 hours` quota of about 0.9 rounds up to 1 here, which pins the
margin.

Stage 2, repos within a stratum. Inside each difficulty stratum, its slots are
apportioned over repos by count, again with largest remainders.

The cost of two stages is that the repo margin is proportional only within
strata. Repo is a source of heterogeneity rather than a main analysis axis,
while difficulty needs to match exactly, so the trade-off is deliberate.

### 2.2 Two constraint repairs

1. If `>4 hours` exceeds floor({hard_cap_frac:.0%} x n), slots are moved from
   that stratum to the stratum with the largest quota deficit. This repair did
   not trigger (section 3).
2. If a repo has fewer than {min_per_repo} slot in main150, one slot is moved to
   it from the most over-allocated repo in the same difficulty stratum.
   Restricting the move to one stratum keeps the stage-1 difficulty margin
   intact.

Both repairs are written to the audit log (section 3), so every changed cell can
be checked afterwards.

### 2.3 Within-cell draw: high/low context mix

A cell with k >= 2 slots is sorted by `ctx_proxy`, and a systematic sample with a
random start is taken from it (k equal-width proxy bands, one task from each).
The draw then spans the cell's full context range instead of clumping at one
end.

A cell with k = 1 admits no mix within the cell, so these cells are balanced
across cells instead. All k = 1 cells are shuffled and alternately assigned to
the upper or lower proxy half of their own cell. No stratum ends up all short by
accident, and the expected share of long tasks stays near 50%.

### 2.4 Context-margin repair (swap within cell)

Section 2.3 mixes context within cells but does not fix the global context
margin. At n=30 that luck is expensive, because a pilot that happens to skew
short under-measures how often compaction triggers.

A deterministic repair therefore runs after the draw. `ctx_proxy` is cut into
quartiles Q1 to Q4 over the sampling pool, the target is a balanced margin (n/4
per band), and swaps are allowed only within the same repo x difficulty cell.
The main allocation (section 2.1) is untouched while the context margin returns
to target. Every swap is logged.

### 2.5 Nesting: pilot inside main

main150 is drawn from the pool first. pilot30 is then drawn with the same
algorithm (including the section 2.4 repair) with main150 as its frame. Nesting
holds by construction, and the pilot margins track those of main and the pool.

### 2.6 Incremental maintenance after freezing (`--mode incremental`, the default)

When the protocol changes, for example through a new repo exclusion in section
8, the lists are not redrawn. The default mode treats the `main150.json` and
`pilot30.json` on disk as a carry-over baseline and only drops and fills against
the new quotas, in three steps.

1. Drop instances that are no longer in the pool.
2. Drop instances whose cell now exceeds its new quota, choosing the one that
   best balances the `ctx_band` margin.
3. Fill deficit cells, choosing the instance that best balances the `ctx_band`
   margin.

The fill rule is deterministic and consumes no RNG (greedy on the margin
deficit, ties broken on `instance_id`), so retained instances do not move
because the RNG stream shifted. The final lists satisfy the new quotas exactly,
realised with maximal reuse. Section 9 gives the reason: a clean redraw changes
every denominator and percentile rank and churned 72 of 150 instances.

Reproducibility. The two JSON files are therefore both outputs and inputs.
Rerunning with them present is idempotent (all three artifacts come out
byte-identical), while deleting them and rerunning falls back to the different
lists of `--mode fresh`. Recover them from git if they go missing rather than
regenerating them.

## 3. Constraint audit log

```
{audit}
```

## 4. Count tables

### 4.1 repo x difficulty (pilot30 / main150 / pool / all 500)

{cell_table}

### 4.2 Difficulty margin

{diff_table}

### 4.3 Repo margin

{repo_table}

## 5. Context-proxy summary

{proxy_table}

## 6. Sanity checks

```
{sanity}
```

## 7. Dataset fields versus expectations

The difficulty field is named `difficulty` and is a top-level column of
`princeton-nlp/SWE-bench_Verified`, so no separate annotation file is needed.
Its four values are {difficulty_values}. The easiest band is written
`<15 min fix`, including the word "fix", not `<15 min`.

The `>4 hours` stratum has only {n_hard} tasks in the pool ({hard_share:.1f}%).
The {hard_cap_frac:.0%} cap therefore never binds, and proportional allocation
gives main150 {main_hard} and pilot30 {pilot_hard}. Giving the hardest stratum
more weight would require oversampling it (up to all {n_hard} tasks), which
would be a protocol deviation to be recorded separately.

The dataset has a single split, `test` (500 tasks), and no train or dev split.
`hints_text` exists but is mostly empty, and this protocol does not use it. The
two raw proxy quantities are nearly uncorrelated (rho={spearman:.3f}), which is
why the proxy is the equal-weight mean of their percentiles rather than either
one alone.

## 8. Sampling-pool exclusions

Exclusions are part of the protocol, not preprocessing. Dropping a repo changes
the population the results generalise over, so every exclusion carries a
reason, reproducible evidence and a date, and is recorded in the table below and
in the revision log of section 9. Exclusions take effect before any quota is
computed (`EXCLUDED_REPOS`, top of the script). Quota targets are recomputed on
the reduced pool per section 2.1, and the freed slots flow back without manual
patching. Lists that were already frozen are maintained incrementally to the new
target per section 2.6 rather than redrawn.

{excl_table}

After exclusion the pool has {n_pool} tasks in {n_repos} repos. The count tables
of section 4 keep the excluded repo's rows (flagged `EXCL`), so what was
removed, and how much, stays visible.

Scope. The population of every resolved rate and token effect is this
{n_pool}-task pool, not the full Verified 500. The exclusion follows from
verifier network availability and is unrelated to task difficulty, so it does
not bias resolved rates upward.

## 9. Revision log

{revision_log}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-main", type=int, default=N_MAIN)
    ap.add_argument("--n-pilot", type=int, default=N_PILOT)
    ap.add_argument("--outdir", type=Path, default=HERE)
    ap.add_argument(
        "--report", action="store_true", help="print tables only, write nothing"
    )
    ap.add_argument(
        "--mode",
        choices=["incremental", "fresh"],
        default="incremental",
        help=(
            "incremental (default): carry the existing main150/pilot30 forward and "
            "patch them onto the current target, minimising churn. "
            "fresh: ignore any existing list and draw from scratch."
        ),
    )
    args = ap.parse_args()

    df_all = load_frame()
    fp = frame_fingerprint(df_all)
    # the sampling pool: full dataset minus repo-level exclusions
    df = df_all[~df_all["excluded"]].reset_index(drop=True)

    audit = [f"raw dataset: N={len(df_all)} fingerprint={fp}"]
    for repo, (reason, evidence, date) in sorted(EXCLUDED_REPOS.items()):
        n = int((df_all["repo"] == repo).sum())
        audit.append(
            f"EXCLUDED repo '{repo}' ({n} instances) [{date}]: {reason}"
            f" | evidence: {evidence}"
        )
    audit.append(f"sampling pool: N={len(df)} fingerprint={frame_fingerprint(df)}")
    audit.append("")
    # transition detail (churn vs whatever was on disk) is run-specific, so it is
    # printed but never embedded in the protocol md -- that keeps the md byte-stable
    # across re-runs.  The authoritative record of a change is the revision log.
    runlog = []

    def carry_over(name):
        """Previous frozen list, if one is on disk and we are in incremental mode."""
        p = args.outdir / f"{name}.json"
        if args.mode != "incremental" or not p.exists():
            return None
        prev = json.loads(p.read_text())
        return [r["instance_id"] for r in prev["instances"]]

    # ---- main150 from the pool --------------------------------------------
    rng = np.random.default_rng(args.seed)
    audit.append(f"--- main{args.n_main} allocation (frame = pool {len(df)}) ---")
    alloc, log = allocate(df, args.n_main, MIN_PER_REPO_MAIN)
    audit.extend(log)
    prev_main = carry_over("main150")
    if prev_main:
        runlog.append(f"mode=incremental: carrying {len(prev_main)} instances forward")
        main_ids = topup(df, alloc, prev_main, runlog)
    else:
        runlog.append("mode=fresh: drawing main from scratch")
        main_ids = draw(df, alloc, rng)
    audit.append("main context-proxy margin repair:")
    main_ids = repair_proxy_margin(df, main_ids, args.n_main, audit)

    # ---- pilot30 from main150 (guarantees nesting) ------------------------
    dmain = df[df.instance_id.isin(main_ids)].reset_index(drop=True)
    audit.append("")
    audit.append(f"--- pilot{args.n_pilot} allocation (frame = main{args.n_main}) ---")
    alloc_p, log_p = allocate(dmain, args.n_pilot, 0)
    audit.extend(log_p)
    prev_pilot = carry_over("pilot30")
    if prev_pilot:
        runlog.append(f"mode=incremental: carrying {len(prev_pilot)} instances forward")
        pilot_ids = topup(dmain, alloc_p, prev_pilot, runlog)
    else:
        runlog.append("mode=fresh: drawing pilot from scratch")
        pilot_ids = draw(dmain, alloc_p, rng)
    audit.append("pilot context-proxy margin repair:")
    pilot_ids = repair_proxy_margin(dmain, pilot_ids, args.n_pilot, audit)

    # churn accounting against whatever was on disk before this run
    churn = {}
    for nm, prev, cur in [
        ("main150", prev_main, main_ids),
        ("pilot30", prev_pilot, pilot_ids),
    ]:
        if prev:
            churn[nm] = (sorted(set(prev) - set(cur)), sorted(set(cur) - set(prev)))
            runlog.append(
                f"churn {nm}: kept {len(set(prev) & set(cur))}, "
                f"out {len(churn[nm][0])}, in {len(churn[nm][1])}"
            )
            for i in churn[nm][0]:
                runlog.append(f"  OUT {nm}: {i}")
            for i in churn[nm][1]:
                runlog.append(f"  IN  {nm}: {i}")

    # pilot repo-coverage repair (documented; expected to be a no-op)
    cov = dmain[dmain.instance_id.isin(pilot_ids)]["repo"].nunique()
    if cov < MIN_REPO_COVERAGE_PILOT:
        audit.append(
            f"pilot repo coverage {cov} < {MIN_REPO_COVERAGE_PILOT}: repairing"
        )
        have = set(dmain[dmain.instance_id.isin(pilot_ids)]["repo"])
        missing = [r for r in sorted(set(dmain["repo"])) if r not in have]
        for repo in missing[: MIN_REPO_COVERAGE_PILOT - cov]:
            cand = dmain[dmain.repo == repo].sort_values("instance_id")
            add = cand["instance_id"].iloc[int(rng.integers(len(cand)))]
            over = (
                dmain[dmain.instance_id.isin(pilot_ids)]["repo"]
                .value_counts()
                .idxmax()
            )
            drop = [
                i
                for i in pilot_ids
                if dmain.set_index("instance_id").loc[i, "repo"] == over
            ][0]
            pilot_ids = [i for i in pilot_ids if i != drop] + [add]
            audit.append(f"  swapped {drop} ({over}) -> {add} ({repo})")
    else:
        audit.append(
            f"pilot repo coverage = {cov} >= {MIN_REPO_COVERAGE_PILOT}: no repair"
        )

    # ---- sanity ------------------------------------------------------------
    sanity = []

    def chk(name, ok, detail=""):
        sanity.append(f"[{'PASS' if ok else 'FAIL'}] {name}{(' :: ' + detail) if detail else ''}")
        return ok

    all_ok = True
    all_ok &= chk("main size == %d" % args.n_main, len(main_ids) == args.n_main, str(len(main_ids)))
    all_ok &= chk("pilot size == %d" % args.n_pilot, len(pilot_ids) == args.n_pilot, str(len(pilot_ids)))
    all_ok &= chk("no duplicate in main", len(set(main_ids)) == len(main_ids))
    all_ok &= chk("no duplicate in pilot", len(set(pilot_ids)) == len(pilot_ids))
    all_ok &= chk("pilot subset of main", set(pilot_ids) <= set(main_ids))
    all_ok &= chk("all ids exist in Verified", set(main_ids) <= set(df_all.instance_id))
    all_ok &= chk(
        "all ids inside the sampling pool", set(main_ids) <= set(df.instance_id)
    )
    hit = sorted(
        set(df_all[df_all["excluded"]]["instance_id"]) & (set(main_ids) | set(pilot_ids))
    )
    all_ok &= chk(
        "no excluded repo leaks into either list",
        not hit,
        f"excluded repos = {sorted(EXCLUDED_REPOS)}; leaks = {hit}",
    )
    nrepo_p = df[df.instance_id.isin(pilot_ids)]["repo"].nunique()
    all_ok &= chk(
        "pilot repo coverage >= %d" % MIN_REPO_COVERAGE_PILOT,
        nrepo_p >= MIN_REPO_COVERAGE_PILOT,
        f"{nrepo_p} repos",
    )
    nrepo_m = df[df.instance_id.isin(main_ids)]["repo"].nunique()
    all_ok &= chk(
        "main covers all %d pool repos" % df.repo.nunique(),
        nrepo_m == df.repo.nunique(),
        f"{nrepo_m} repos",
    )
    hard_m = int((df[df.instance_id.isin(main_ids)]["difficulty"] == HARD_STRATUM).sum())
    all_ok &= chk(
        f"main '{HARD_STRATUM}' <= {HARD_CAP_FRAC:.0%}",
        hard_m <= math.floor(HARD_CAP_FRAC * args.n_main),
        f"{hard_m}/{args.n_main}",
    )
    dm = df[df.instance_id.isin(main_ids)]
    all_ok &= chk(
        "main covers every difficulty stratum present in the pool",
        dm["difficulty"].nunique() == df["difficulty"].nunique(),
        f"{dm['difficulty'].nunique()}/{df['difficulty'].nunique()}",
    )
    # difficulty margin must stay within one instance of proportional
    worst = max(
        abs(int((dm["difficulty"] == d).sum()) - (df["difficulty"] == d).sum() * args.n_main / len(df))
        for d in df["difficulty"].unique()
    )
    all_ok &= chk(
        "main difficulty margin within 1 of proportional",
        worst < 1.0,
        f"max |dev| = {worst:.2f}",
    )
    dp = df[df.instance_id.isin(pilot_ids)]
    worstp = max(
        abs(int((dp["difficulty"] == d).sum()) - (dm["difficulty"] == d).sum() * args.n_pilot / args.n_main)
        for d in dm["difficulty"].unique()
    )
    all_ok &= chk(
        "pilot difficulty margin within 1 of main-proportional",
        worstp < 1.0,
        f"max |dev| = {worstp:.2f}",
    )
    for nm, s, ntot in [("main", dm, args.n_main), ("pilot", dp, args.n_pilot)]:
        bc = s["ctx_band"].value_counts().to_dict()
        got = [bc.get(b, 0) for b in ["Q1", "Q2", "Q3", "Q4"]]
        all_ok &= chk(
            f"{nm} ctx_proxy quartile margin balanced (target {ntot/4:.1f} each)",
            max(abs(g - ntot / 4) for g in got) <= 1.0,
            str(got),
        )
    sanity.append("")
    sanity.append("OVERALL: " + ("ALL CHECKS PASS" if all_ok else "*** FAILURE ***"))

    if runlog:
        print("--- run log (not embedded in the protocol md) ---")
        print("\n".join(runlog))
        print()
    audit_s = "\n".join(audit)
    sanity_s = "\n".join(sanity)
    cell_t = three_column_table(df_all, main_ids, pilot_ids)
    diff_t = marginal_table(df_all, main_ids, pilot_ids, "difficulty")
    repo_t = marginal_table(df_all, main_ids, pilot_ids, "repo")
    proxy_t = proxy_summary(df_all, main_ids, pilot_ids)

    print(audit_s)
    print()
    print(cell_t)
    print()
    print(diff_t)
    print()
    print(repo_t)
    print()
    print(proxy_t)
    print()
    print(sanity_s)

    if args.report:
        return 0 if all_ok else 1
    if not all_ok:
        print("\nsanity failed -> refusing to write artifacts", file=sys.stderr)
        return 1

    args.outdir.mkdir(parents=True, exist_ok=True)
    for name, ids in [("main150", main_ids), ("pilot30", pilot_ids)]:
        payload = {
            "name": name,
            "protocol_version": PROTOCOL_VERSION,
            "dataset": DATASET,
            "split": SPLIT,
            "dataset_size": len(df_all),
            "dataset_fingerprint_sha256_16": fp,
            "pool_size": len(df),
            "pool_fingerprint_sha256_16": frame_fingerprint(df),
            "excluded_repos": {
                r: {"reason": rs, "evidence": ev, "date": dt}
                for r, (rs, ev, dt) in sorted(EXCLUDED_REPOS.items())
            },
            "n": len(ids),
            "seed": args.seed,
            "generated_by": "select_subset.py",
            "stratification": ["repo", "difficulty", "ctx_proxy (within-cell)"],
            "ctx_proxy_definition": (
                "0.5*pctrank(problem_statement word count) + "
                "0.5*pctrank(gold patch line count), ranks over the sampling pool"
            ),
            "nested_in": None if name == "main150" else "main150",
            "instances": records(df, ids),
        }
        p = args.outdir / f"{name}.json"
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {p}")

    # Spearman = Pearson on ranks; done by hand so scipy is not a dependency
    spearman = df["ps_words"].rank().corr(df["patch_lines"].rank())
    md = PROTOCOL_TEMPLATE.format(
        dataset=DATASET,
        split=SPLIT,
        version=PROTOCOL_VERSION,
        n_full=len(df_all),
        n_pool=len(df),
        pool_fp=frame_fingerprint(df),
        excl_table="\n".join(
            [
                "| repo | tasks | reason | evidence | date |",
                "|---|---:|---|---|---|",
            ]
            + [
                f"| `{r}` | {int((df_all.repo == r).sum())} | {rs} | {ev} | {dt} |"
                for r, (rs, ev, dt) in sorted(EXCLUDED_REPOS.items())
            ]
        )
        if EXCLUDED_REPOS
        else "(none)",
        revision_log=REVISION_LOG,
        fingerprint=fp,
        seed=args.seed,
        n_pilot=args.n_pilot,
        n_main=args.n_main,
        hard_cap_frac=HARD_CAP_FRAC,
        min_per_repo=MIN_PER_REPO_MAIN,
        min_cov=MIN_REPO_COVERAGE_PILOT,
        difficulty_values=" / ".join(f"`{d}`" for d in DIFFICULTY_ORDER),
        n_repos=df.repo.nunique(),
        django_share=100 * (df.repo == "django/django").mean(),
        n_hard=int((df.difficulty == HARD_STRATUM).sum()),
        hard_share=100 * (df.difficulty == HARD_STRATUM).mean(),
        main_hard=hard_m,
        pilot_hard=int(
            (df[df.instance_id.isin(pilot_ids)]["difficulty"] == HARD_STRATUM).sum()
        ),
        spearman=spearman,
        audit=audit_s,
        cell_table=cell_t,
        diff_table=diff_t,
        repo_table=repo_t,
        proxy_table=proxy_t,
        sanity=sanity_s,
    )
    p = args.outdir / "selection_protocol.md"
    p.write_text(md)
    print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
