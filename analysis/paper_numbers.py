#!/usr/bin/env python3
"""Single number source for every paper figure and table.

Reads data/trials.csv, applies the canonical-trial rule (analysis/main_table.py:
usable beats unusable, latest finished_at wins, Qwen3.8 k=2 attempts split into seeds, repair jobs
refill only unusable slots) and writes paper/numbers.json.  Figures (analysis/fig_*.py) and tables
(analysis/tab_*.py) read the JSON and never touch trials.csv themselves.

Conventions: seed 1 canonical, OC-rg canonical where it exists (else OC, flagged), overflow kept
and scored, exclude rows dropped, hard45 denominator = usable tasks (45 unless noted), pool447 = 447.

Run: python3 analysis/paper_numbers.py   -> paper/numbers.json (+ a short stdout summary)
"""
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import main_table as mt  # noqa: E402  (canonical rule, INFRA set, wilson, mcnemar)

OUT = os.path.normpath(os.path.join(HERE, "..", "paper", "numbers.json"))
CUTOFF_TXT = os.path.join(HERE, "cutoff_audit.txt")

MODELS = ["qwen3.6-35b-a3b", "qwen3.8-27b", "ds-v4-flash", "glm-5.3-flash", "hy4-preview", "opus5"]
SUBSETS = ["hard45", "pool447"]
HARNESSES = ["claude-code", "mini-swe-agent", "opencode"]
H_SHORT = {"claude-code": "CC", "mini-swe-agent": "mini", "opencode": "OC", "dsh": "dsh"}
# (harness, model, subset, job) arms outside the 3-harness grid that Fig 4 / numbers.tex still show
EXTRA_ARMS = [("dsh", "ds-v4-flash", "hard45", "dsh-ds-hard45-nw")]
# column label in main_table -> harness; OC-rg preferred, OC fallback
LAB2H = {"CC": "claude-code", "mini": "mini-swe-agent", "OC-rg": "opencode", "OC": "opencode"}
PAIRS = [("claude-code", "mini-swe-agent"), ("claude-code", "opencode"), ("mini-swe-agent", "opencode")]

# vendor list prices, CNY per 1M tokens: (input cache hit, input cache miss, output)
# DeepSeek: official list price
# Zhipu GLM-5.3-Flash: 0.23 / 0.8 / 2.8
# Tencent HY4 preview (tokenhub list price, 2026-09): 0.3 / 6 / 18
PRICES = {"ds-v4-flash": {"hit": 0.05, "miss": 1.5, "out": 4.5, "vendor": "DeepSeek"},
          "glm-5.3-flash": {"hit": 0.23, "miss": 0.8, "out": 2.8, "vendor": "Zhipu"},
          "hy4-preview": {"hit": 0.3, "miss": 6.0, "out": 18.0, "vendor": "Tencent"}}

CAP_EXC = {"StepCapReached", "AgentTimeoutError", "SoftTimeout", "NonZeroAgentExitCodeError",
           "OutputTokenExceededError"}


def num(v):
    return mt.num(v)


def is_pass(r):
    return (num(r.get("reward")) or 0) >= 1


# ----------------------------------------------------------------------------- canonical by seed
def canonical_by_seed(rows):
    """cells[(lab, model, subset)][seed] = {task: row}; same rule as main_table.canonical but
    keeping every seed instead of MAIN_SEED only (verified to reproduce seed 1 exactly)."""
    col_of = {(h, a): lab for lab, h, a in mt.COLS}
    buckets = defaultdict(lambda: defaultdict(list))
    max_att = defaultdict(int)
    for r in rows:
        if r.get("exclude") == "1":
            continue
        lab = col_of.get((r.get("harness"), r.get("arm")))
        if lab is None or r.get("model") not in MODELS or r.get("subset") not in SUBSETS:
            continue
        bk = (lab, r["model"], r["subset"], int(num(r.get("repeat")) or 1))
        buckets[bk][r["task"]].append(r)
        max_att[(bk, r["job"])] = max(max_att[(bk, r["job"])], int(num(r.get("attempt")) or 1))
    cells = defaultdict(lambda: defaultdict(dict))
    for bk, tasks in buckets.items():
        lab, model, subset, rp = bk
        k2 = {job for (b, job), mx in max_att.items() if b == bk and mx >= 2}
        cell = (lab, model, subset)
        for task, lst in tasks.items():
            base = {}
            for r in lst:
                if r["job"] in k2:
                    base[int(num(r.get("attempt")) or 1)] = r
            if base:
                repairs = [r for r in lst if r["job"] not in k2 and mt.ok(r)]
                repair = max(repairs, key=mt._fin) if repairs else None
                for sd, br in sorted(base.items()):
                    cells[cell][sd][task] = br if (mt.ok(br) or repair is None) else repair
                if mt.MAIN_SEED not in base and repair is not None:
                    cells[cell][mt.MAIN_SEED][task] = repair
            else:
                cells[cell][rp][task] = max(lst, key=lambda r: (mt.ok(r), mt._fin(r)))
    return cells


def pick_arm(cells, model, subset, harness):
    """Canonical column for a harness: OC-rg if it has usable trials, else OC; CC; mini."""
    labs = ["OC-rg", "OC"] if harness == "opencode" else [H_SHORT[harness]]
    for lab in labs:
        c = cells.get((lab, model, subset))
        if c and mt.usable(c.get(mt.MAIN_SEED, {})):
            return lab, c
    return None, None


# ----------------------------------------------------------------------------- stats
def newcombe_paired(a, b, c, d, z=1.959963984540054):
    """Newcombe (1998) method 10 CI for p1-p2 with paired data.
    a = both pass, b = only first, c = only second, d = neither."""
    n = a + b + c + d
    if n == 0:
        return (float("nan"), float("nan"))
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = mt.wilson(a + b, n) if z == 1.959963984540054 else _wilson_z(a + b, n, z)
    l2, u2 = mt.wilson(a + c, n) if z == 1.959963984540054 else _wilson_z(a + c, n, z)
    A = (a + b) * (c + d) * (a + c) * (b + d)
    if A <= 0:
        phi = 0.0
    else:
        t = a * d - b * c
        if t > n / 2:
            phi = (t - n / 2) / math.sqrt(A)
        elif t >= 0:
            phi = 0.0
        else:
            phi = t / math.sqrt(A)
    D = p1 - p2
    lo = D - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = D + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return (lo, hi)


def _wilson_z(k, n, z):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired(ua, ub):
    """ua/ub: task -> row (usable).  Returns the 2x2 and derived quantities on the shared tasks."""
    ts = sorted(set(ua) & set(ub))
    a = b = c = d = 0
    for t in ts:
        x, y = is_pass(ua[t]), is_pass(ub[t])
        if x and y:
            a += 1
        elif x:
            b += 1
        elif y:
            c += 1
        else:
            d += 1
    n = len(ts)
    out = {"n": n, "a": a, "b": b, "c": c, "d": d, "k1": a + b, "k2": a + c}
    if n:
        out["diff_pp"] = 100 * (b - c) / n
        lo, hi = newcombe_paired(a, b, c, d)
        out["ci95_pp"] = [100 * lo, 100 * hi]
        lo9, hi9 = newcombe_paired(a, b, c, d, z=1.6448536269514722)
        out["ci90_pp"] = [100 * lo9, 100 * hi9]
        out["p_mcnemar"] = mt.mcnemar_exact_p(b, c)
        out["flip_rate"] = (b + c) / n
        out["tost5"] = bool(lo9 > -0.05 and hi9 < 0.05)          # equivalence at δ = 5 pp (90% CI inside)
    return out


def mde_pp(n, disc, alpha_z=1.959963984540054, beta_z=0.8416212335729143):
    """Smallest |p1-p2| (pp) a paired McNemar test detects with power .8 at discordance rate `disc`.
    Inverts the normal-approximation sample size n = (za + zb*sqrt(1-psi^2))^2 / (disc*psi^2)."""
    lo, hi = 1e-6, 1.0
    if ((alpha_z + beta_z * 0.0) ** 2) / (disc * 1.0) > n:      # even psi = 1 is out of reach
        return float("nan")
    for _ in range(60):
        psi = (lo + hi) / 2
        need = ((alpha_z + beta_z * math.sqrt(1 - psi ** 2)) ** 2) / (disc * psi ** 2)
        if need > n:
            lo = psi
        else:
            hi = psi
    return 100 * disc * hi


_REJECT_K = {}


def _reject_k(m, alpha=0.05):
    """Largest k with min(b, m-b) <= k rejecting the two-sided exact binomial test on m
    discordant pairs at `alpha`; -1 when no split of m pairs can reach alpha (m < 6 at .05)."""
    key = (m, alpha)
    if key not in _REJECT_K:
        k, tail = -1, 0
        for i in range(m // 2 + 1):
            tail += math.comb(m, i)
            if min(1.0, 2 * tail / 2 ** m) <= alpha:
                k = i
            else:
                break
        _REJECT_K[key] = k
    return _REJECT_K[key]


def exact_power(n, disc, delta_pp, alpha=0.05):
    """Unconditional power of the two-sided *exact* McNemar test we actually report.

    M ~ Binomial(n, disc) discordant pairs; given M the one-way count is Binomial(M, q) with
    q = (1 + delta/disc)/2; reject when the conditional two-sided binomial test hits `alpha`.
    Unlike mde_pp's asymptotic inversion this is capped: M < 6 can never reject at .05, so the
    power at n=45, disc=.138 tops out near .60 even at delta = disc."""
    delta = delta_pp / 100.0
    if delta > disc * (1 + 1e-9):                               # |p1-p2| <= discordance
        return float("nan")
    q = min(1.0, (1 + delta / disc) / 2)
    tot = 0.0
    for m in range(n + 1):
        pm = math.comb(n, m) * disc ** m * (1 - disc) ** (n - m)
        if pm < 1e-15:
            continue
        k = _reject_k(m, alpha)
        if k < 0:
            continue
        pr = sum(math.comb(m, b) * (q ** b * (1 - q) ** (m - b) + (1 - q) ** b * q ** (m - b))
                 for b in range(k + 1))
        tot += pm * pr
    return tot


def mde_pp_exact(n, disc, power=0.8, alpha=0.05):
    """Smallest |p1-p2| (pp) the exact test detects with `power`; nan when unattainable.
    Power rises with delta, so bisect between 0 and the ceiling delta = disc."""
    if not exact_power(n, disc, 100 * disc, alpha) >= power:
        return float("nan")
    lo, hi = 0.0, 100 * disc
    for _ in range(50):
        if hi - lo < 1e-3:                                      # 0.001 pp is far below what we quote
            break
        mid = (lo + hi) / 2
        if exact_power(n, disc, mid, alpha) >= power:
            hi = mid
        else:
            lo = mid
    return hi


def n_min_for_power(disc, power=0.8, alpha=0.05, cap=4000):
    """Smallest n whose ceiling power (delta = disc, i.e. every discordance one way) reaches
    `power`; that ceiling is P(M >= 6) at .05 and rises monotonically with n."""
    for n in range(2, cap + 1):
        if exact_power(n, disc, 100 * disc, alpha) >= power:
            return n
    return None


# ----------------------------------------------------------------------------- cost / steps
def cost_entry(rs, p):
    """Per-trial mean token profile of a set of usable trials, priced with list `p`.
    steps are AGENT steps = one model call each (trials.csv agent_steps, no bookkeeping step)."""
    n = len(rs)
    pt = sum(num(r["prompt_tokens"]) for r in rs) / n
    cr = sum(num(r.get("cache_read_tokens")) or 0 for r in rs) / n
    ot = sum(num(r.get("output_tokens_total")) or num(r.get("completion_tokens")) or 0 for r in rs) / n
    rt = sum(num(r.get("reasoning_tokens")) or 0 for r in rs) / n
    st = sum(num(r.get("agent_steps")) or 0 for r in rs) / n
    cny = ((pt - cr) * p["miss"] + cr * p["hit"] + ot * p["out"]) / 1e6
    return {"n": n, "input_miss": pt - cr, "input_hit": cr, "output": ot,
            "reasoning_in_output": rt, "steps": st, "cny_per_trial": cny,
            "cny_split": {"input_miss": (pt - cr) * p["miss"] / 1e6,
                          "input_hit": cr * p["hit"] / 1e6,
                          "output": ot * p["out"] / 1e6},
            "cache_hit_frac": (cr / pt if pt else None)}


def step_stats(cell):
    """Mean / median model calls per trial over the usable trials of one canonical arm."""
    vs = sorted(v for v in (num(r.get("agent_steps")) for r in mt.usable(cell).values()) if v is not None)
    if not vs:
        return None
    md = vs[len(vs) // 2] if len(vs) % 2 else (vs[len(vs) // 2 - 1] + vs[len(vs) // 2]) / 2
    return {"n": len(vs), "mean": sum(vs) / len(vs), "median": md}


# ----------------------------------------------------------------------------- outcome classes
def load_cutoff_terminals(path):
    """job -> set(trial_id) of OpenCode trials that ENDED on an output-token cut (cutoff_audit.txt)."""
    out = defaultdict(set)
    if not os.path.exists(path):
        return out
    job = None
    for line in open(path):
        m = re.match(r"== (\S+)", line)
        if m:
            job = m.group(1)
        elif job and "terminal trials" in line:
            ids = re.findall(r"__([A-Za-z0-9]{7})\b", line.split(":", 1)[1])
            out[job] |= set(ids)
    return out


# ----------------------------------------------------------------------------- outcome classes
def outcome(r, cut_ids):
    if not mt.ok(r):
        return "void"
    if is_pass(r):
        return "pass"
    exc = r.get("exception_type") or ""
    if r.get("context_overflow") in ("1", "True", "true") or exc == "ContextWindowExceeded":
        return "overflow"
    if exc in CAP_EXC:
        return "cap"
    if r.get("trial_id") in cut_ids.get(r.get("job"), set()):
        return "cutoff"
    return "fail"


# ----------------------------------------------------------------------------- main
def main():
    rows, fields = mt.load(mt.CSV_PATH)
    cells = canonical_by_seed(rows)
    cut_ids = load_cutoff_terminals(CUTOFF_TXT)
    N = {"meta": {"generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                  "csv_rows": len(rows), "seed": mt.MAIN_SEED, "rule": "analysis/main_table.py canonical()",
                  "models": MODELS, "subsets": SUBSETS, "harnesses": HARNESSES},
         "prices": PRICES, "table1": {}, "pairs": [], "flips": {}, "mde": {}, "cost": {},
         "cost_extra": {}, "reprice": {}, "steps": {}, "outcomes": {}, "coverage": []}

    # ---- Table 1 + per-seed pass maps
    passmap = {}     # (model, subset, harness, seed) -> {task: 0/1}  (usable only)
    rowmap = {}      # (model, subset, harness, seed) -> {task: row}  (all canonical incl. void)
    for m in MODELS:
        for s in SUBSETS:
            for h in HARNESSES:
                lab, c = pick_arm(cells, m, s, h)
                if c is None:
                    continue
                ent = {"arm": lab, "seeds": sorted(c), "per_seed": {}}
                for sd in sorted(c):
                    u = mt.usable(c[sd])
                    passmap[(m, s, h, sd)] = {t: int(is_pass(r)) for t, r in u.items()}
                    rowmap[(m, s, h, sd)] = c[sd]
                    k, n = mt.passes(u), len(u)
                    lo, hi = mt.wilson(k, n) if n else (float("nan"),) * 2
                    ent["per_seed"][str(sd)] = {"k": k, "n": n, "n_all": len(c[sd]), "rate": (k / n if n else None),
                                                "wilson": [lo, hi]}
                    N["coverage"].append({"model": m, "subset": s, "harness": h, "arm": lab, "seed": sd,
                                          "n_tasks": len(c[sd]), "n_usable": n, "k": k})
                main = ent["per_seed"].get(str(mt.MAIN_SEED))
                if main:
                    ent.update(main)
                N["table1"].setdefault(m, {}).setdefault(s, {})[h] = ent

    # ---- extra single-job arms shown alongside the three harnesses (Fig 4 / Table 1 footnote):
    # DeepSeek's own harness (dsh) on DS-V4-Flash, hard45, one seed, no reruns.
    for h, m, s, job in EXTRA_ARMS:
        cell = {r["task"]: r for r in rows if r.get("job") == job and r.get("exclude") != "1"}
        if not cell:
            continue
        u = mt.usable(cell)
        k, n = mt.passes(u), len(u)
        lo, hi = mt.wilson(k, n) if n else (float("nan"),) * 2
        ent = {"arm": h, "seeds": [1], "per_seed": {"1": {"k": k, "n": n, "n_all": len(cell), "rate": (k / n if n else None),
                                                          "wilson": [lo, hi]}}}
        ent.update(ent["per_seed"]["1"])
        N["table1"].setdefault(m, {}).setdefault(s, {})[h] = ent
        rowmap[(m, s, h, mt.MAIN_SEED)] = cell
        N["coverage"].append({"model": m, "subset": s, "harness": h, "arm": job, "seed": 1,
                              "n_tasks": len(cell), "n_usable": n, "k": k})

    # ---- paired effects (Fig 1b, Table 2), all seed combinations for the range ticks
    for m in MODELS:
        for s in SUBSETS:
            for h1, h2 in PAIRS:
                k1 = [k for k in rowmap if k[:3] == (m, s, h1)]
                k2 = [k for k in rowmap if k[:3] == (m, s, h2)]
                if not k1 or not k2:
                    continue
                u1 = mt.usable(rowmap[(m, s, h1, mt.MAIN_SEED)]) if (m, s, h1, mt.MAIN_SEED) in rowmap else None
                u2 = mt.usable(rowmap[(m, s, h2, mt.MAIN_SEED)]) if (m, s, h2, mt.MAIN_SEED) in rowmap else None
                if not u1 or not u2:
                    continue
                ent = paired(u1, u2)
                ent.update({"model": m, "subset": s, "h1": h1, "h2": h2,
                            "pair": f"{H_SHORT[h1]}−{H_SHORT[h2]}",
                            "arm1": N["table1"][m][s][h1]["arm"], "arm2": N["table1"][m][s][h2]["arm"]})
                combos = []
                for a in k1:
                    for b in k2:
                        if (a[3], b[3]) == (mt.MAIN_SEED, mt.MAIN_SEED):
                            continue
                        pp = paired(mt.usable(rowmap[a]), mt.usable(rowmap[b]))
                        if pp["n"] >= 20:
                            combos.append({"seed1": a[3], "seed2": b[3], "diff_pp": pp["diff_pp"], "n": pp["n"]})
                ent["seed_combos"] = combos
                N["pairs"].append(ent)

    # ---- flip rates (Figure 2): rerun / harness swap / model swap, seed-level maps
    for s in SUBSETS:
        F = {"rerun": [], "harness": [], "model": []}
        keys = [k for k in passmap if k[1] == s]
        for m in MODELS:
            for h in HARNESSES:
                seeds = sorted(k[3] for k in keys if k[0] == m and k[2] == h)
                for i in range(len(seeds)):
                    for j in range(i + 1, len(seeds)):
                        A, B = passmap[(m, s, h, seeds[i])], passmap[(m, s, h, seeds[j])]
                        sh = set(A) & set(B)
                        if len(sh) >= 20:
                            F["rerun"].append({"model": m, "harness": h, "seeds": [seeds[i], seeds[j]],
                                               "flips": sum(A[t] != B[t] for t in sh), "n": len(sh)})
        for m in MODELS:
            for h1, h2 in PAIRS:
                a, b = (m, s, h1, mt.MAIN_SEED), (m, s, h2, mt.MAIN_SEED)
                if a in passmap and b in passmap:
                    sh = set(passmap[a]) & set(passmap[b])
                    if len(sh) >= 20:
                        F["harness"].append({"model": m, "pair": f"{H_SHORT[h1]}−{H_SHORT[h2]}",
                                             "flips": sum(passmap[a][t] != passmap[b][t] for t in sh), "n": len(sh)})
        for h in HARNESSES:
            ms = [m for m in MODELS if (m, s, h, mt.MAIN_SEED) in passmap]
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a, b = (ms[i], s, h, mt.MAIN_SEED), (ms[j], s, h, mt.MAIN_SEED)
                    sh = set(passmap[a]) & set(passmap[b])
                    if len(sh) >= 20:
                        F["model"].append({"harness": h, "models": [ms[i], ms[j]],
                                           "flips": sum(passmap[a][t] != passmap[b][t] for t in sh), "n": len(sh)})
        for grp in F.values():
            for e in grp:
                e["rate"] = e["flips"] / e["n"]
        # the same rerun pairs without Opus 5, whose rerun noise is much lower
        F["rerun_ladder"] = [e for e in F["rerun"] if e["model"] != "opus5"]
        N["flips"][s] = F

    # ---- MDE (Fig 2b): discordance rate = observed harness-swap flip rate, pooled per subset
    for s in SUBSETS:
        hs = N["flips"][s]["harness"]
        if hs:
            disc = sum(e["flips"] for e in hs) / sum(e["n"] for e in hs)
            rates = sorted(e["rate"] for e in hs)
            n_s = 45 if s == "hard45" else 447
            N["mde"][s] = {"disc_pooled": disc, "disc_min": rates[0], "disc_max": rates[-1],
                           "n_typical": n_s,
                           "mde_pp_at_n": mde_pp(n_s, disc),
                           # exact two-sided McNemar, unconditional power (the asymptotic
                           # inversion overstates what n=45 can resolve)
                           "mde_pp_exact80": mde_pp_exact(n_s, disc, 0.8),
                           "power_max_at_n": exact_power(n_s, disc, 100 * disc),
                           "mde_pp_exact50": mde_pp_exact(n_s, disc, 0.5),
                           "n_min_for_80": n_min_for_power(disc, 0.8)}

    # ---- cost (Fig 3): per-trial mean tokens over canonical seed-1 usable trials, hard45
    for m in PRICES:
        for h in HARNESSES:
            key = (m, "hard45", h, mt.MAIN_SEED)
            if key not in rowmap:
                continue
            rs = [r for r in mt.usable(rowmap[key]).values()]
            rs = [r for r in rs if num(r.get("prompt_tokens")) is not None]
            if not rs:
                continue
            ent = cost_entry(rs, PRICES[m])
            ent["arm"] = N["table1"][m]["hard45"][h]["arm"]
            N["cost"].setdefault(m, {})[h] = ent

    # ---- the same profile for the vendor's own harness (dsh), kept out of Fig 3's shared axes
    for h, m, s, job in EXTRA_ARMS:
        key = (m, s, h, mt.MAIN_SEED)
        if m not in PRICES or key not in rowmap:
            continue
        rs = [r for r in mt.usable(rowmap[key]).values() if num(r.get("prompt_tokens")) is not None]
        if not rs:
            continue
        ent = cost_entry(rs, PRICES[m])
        ent["arm"] = job
        N["cost_extra"].setdefault(m, {})[h] = ent

    # ---- re-pricing: each hard45 token profile against BOTH vendors' list prices
    for m in PRICES:                                   # m = the token profile (the arms of Fig 3)
        prof = N["cost"].get(m, {})
        if not prof:
            continue
        for tag, pm in (("ds", "ds-v4-flash"), ("zhipu", "glm-5.3-flash")):
            p = PRICES[pm]
            cny = {H_SHORT[h]: (e["input_miss"] * p["miss"] + e["input_hit"] * p["hit"]
                                + e["output"] * p["out"]) / 1e6 for h, e in prof.items()}
            ent = {"vendor": p["vendor"], "cny": cny,
                   "order": [k for k, _ in sorted(cny.items(), key=lambda kv: kv[1])]}
            for other in ("mini", "OC"):
                if "CC" in cny and other in cny and cny[other]:
                    ent[f"CCv{other}"] = cny["CC"] / cny[other]
            N["reprice"].setdefault(m, {})[tag] = ent

    # ---- model calls per trial (agent steps) for every canonical arm, both subsets
    for (m, s, h, sd), cell in rowmap.items():
        if sd != mt.MAIN_SEED:
            continue
        ss = step_stats(cell)
        if ss:
            N["steps"].setdefault(m, {}).setdefault(s, {})[h] = ss

    # ---- outcomes (Fig 4): canonical seed-1 trials incl. void, by class
    for s in SUBSETS:
        for m in MODELS:
            for h in HARNESSES + [e[0] for e in EXTRA_ARMS]:
                key = (m, s, h, mt.MAIN_SEED)
                if key not in rowmap:
                    continue
                cnt = defaultdict(int)
                for r in rowmap[key].values():
                    cnt[outcome(r, cut_ids)] += 1
                N["outcomes"].setdefault(s, {}).setdefault(m, {})[h] = dict(cnt)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(N, fh, indent=1, ensure_ascii=False)

    # ---- stdout summary
    print(f"wrote {OUT}")
    for m in MODELS:
        for s in SUBSETS:
            t = N["table1"].get(m, {}).get(s)
            if not t:
                continue
            print(f"{m:16s} {s:8s} " + "  ".join(
                f"{H_SHORT[h]}:{t[h]['k']}/{t[h]['n']}({t[h]['arm']},seeds={t[h]['seeds']})" for h in HARNESSES if h in t))
    for e in N["pairs"]:
        print(f"  pair {e['model']:16s} {e['subset']:8s} {e['pair']:9s} n={e['n']:3d} diff={e['diff_pp']:+5.1f}pp "
              f"CI95=[{e['ci95_pp'][0]:+5.1f},{e['ci95_pp'][1]:+5.1f}] p={e['p_mcnemar']:.3f} b={e['b']} c={e['c']} "
              f"tost5={e['tost5']} combos={len(e['seed_combos'])}")
    for s in SUBSETS:
        F = N["flips"][s]
        print(f"  flips {s}: rerun n={len(F['rerun'])} harness n={len(F['harness'])} model n={len(F['model'])}  "
              f"mde={N['mde'].get(s)}")
    for m in N["cost"]:
        for h, e in N["cost"][m].items():
            print(f"  cost {m} {H_SHORT[h]:5s} n={e['n']} miss={e['input_miss']/1e3:.0f}k hit={e['input_hit']/1e3:.0f}k "
                  f"out={e['output']/1e3:.0f}k steps={e['steps']:.0f} CNY={e['cny_per_trial']:.2f}")
    for m in N["cost_extra"]:
        for h, e in N["cost_extra"][m].items():
            print(f"  cost {m} {H_SHORT[h]:5s} n={e['n']} steps={e['steps']:.0f} CNY={e['cny_per_trial']:.2f} (extra arm)")
    for m in N["reprice"]:
        for tag, e in N["reprice"][m].items():
            print(f"  reprice {m:14s} on {e['vendor']:8s} " +
                  "  ".join(f"{k}={v:.3f}" for k, v in e["cny"].items()) +
                  f"  CC/mini={e.get('CCvmini', float('nan')):.2f} CC/OC={e.get('CCvOC', float('nan')):.2f}")
    for m in N["steps"]:
        for s in N["steps"][m]:
            print(f"  steps {m:16s} {s:8s} " + "  ".join(
                f"{H_SHORT[h]}:{e['mean']:.1f}/{e['median']:.0f}" for h, e in N["steps"][m][s].items()))
    for s in N["outcomes"]:
        for m in N["outcomes"][s]:
            for h, c in N["outcomes"][s][m].items():
                print(f"  outcome {s} {m:16s} {H_SHORT[h]:5s} {c}")


if __name__ == "__main__":
    main()
