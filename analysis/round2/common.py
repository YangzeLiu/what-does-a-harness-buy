"""Shared loaders for the second-round tables (read-only on every input).

Every loader returns cells[(model, subset, harness, run)] = {task: rec}, where rec is a dict
    task, usable (bool), passed (0/1), pt, cr, cw, ot (tokens), steps, secs,
    cost_usd (second-round Claude Opus 5.5 hard45 only), src.
harness is one of 'CC', 'mini', 'OC'.  'run' is an int (1, 2, ...).

Counting rule = paper rule (analysis/main_table.py): usable = scored and exception not in the 14-class
INFRA set; one canonical row per (model, harness, run, task): usable beats unusable, ties go to the latest
finished_at (the second-round table holds one row per task and run).  For the first-round models the paper's paper_numbers.canonical_by_seed() / pick_arm() are
imported (OpenCode = 'OC-rg' arm, as in the first-round tables).

Inputs: data/trials.csv (first round, via analysis/main_table.py) and data/round2/trials_round2.csv
(second round; column 'experiment' says which run set a row belongs to).
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
R2CSV = os.path.join(ROOT, "data", "round2", "trials_round2.csv")
REF = os.path.join(ROOT, "reference_outputs")          # first-round tables and numbers.json, used as cross-checks
OUT = os.path.join(ROOT, "paper", "round2")             # generated outputs (compared by reproduce.sh)
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import main_table as mt          # noqa: E402
import paper_numbers as pn       # noqa: E402

INFRA = mt.INFRA
H3 = ["CC", "mini", "OC"]
HFULL = {"CC": "Claude Code", "mini": "mini-SWE-agent", "OC": "OpenCode"}
H_FROM = {"claude-code": "CC", "mini-swe-agent": "mini", "opencode": "OC",
          "CC": "CC", "mini": "mini", "OC": "OC", "cc": "CC", "oc": "OC"}

LABEL = {"qwen3.6-35b-a3b": "Qwen3.6-35B-A3B", "qwen3.8-27b": "Qwen3.8-27B",
         "ds-v4-flash": "DeepSeek-V4-Flash", "glm-5.3-flash": "GLM-5.3-Flash",
         "hy4-preview": "HY4-Preview", "ds-v41-flash": "DeepSeek-V4.1-Flash",
         "opus55": "Claude Opus 5.5", "opus5": "Claude Opus 5"}
PAPER5 = ["qwen3.6-35b-a3b", "qwen3.8-27b", "ds-v4-flash", "glm-5.3-flash", "hy4-preview"]
ALL7 = PAPER5 + ["ds-v41-flash", "opus55"]

# trials_round2.csv experiment -> internal model key used by the scripts
EXP_MODEL = {"swe_glm53flash_run2": "glm-5.3-flash", "swe_hy4preview_run2": "hy4-preview",
             "swe_opus55_hard45": "opus55",
             "swe_dsv41flash_hard45": "ds-v41-flash", "swe_dsv41flash_pool447": "ds-v41-flash"}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def ntask(t):
    return t.split("/", 1)[1] if "/" in t else t


_R2 = None


def r2rows(experiment=None):
    """rows of data/round2/trials_round2.csv, optionally of one experiment (list of dicts, file order)"""
    global _R2
    if _R2 is None:
        _R2 = list(csv.DictReader(open(R2CSV, newline="")))
    return [r for r in _R2 if experiment is None or r["experiment"] == experiment]


# ------------------------------------------------------------------ first-round rows
def rec_from_paper_row(r, src):
    ok = r.get("scored") == "1" and (r.get("exception_type") or "") not in INFRA
    ot = fnum(r.get("output_tokens_total"))
    if ot is None:
        ot = fnum(r.get("completion_tokens"))
    return {"task": r["task"], "usable": ok, "passed": int((fnum(r.get("reward")) or 0) >= 1),
            "pt": fnum(r.get("prompt_tokens")) or 0, "cr": fnum(r.get("cache_read_tokens")) or 0,
            "cw": fnum(r.get("cache_write_tokens")) or 0, "ot": ot or 0,
            "steps": fnum(r.get("agent_steps")), "secs": fnum(r.get("agent_seconds")),
            "exc": r.get("exception_type") or "", "job": r.get("job"),
            "trial": r.get("trial_id"), "src": src}


def load_paper():
    """First-round models from data/trials.csv, every seed, via the paper's own canonical rule."""
    rows, _ = mt.load(mt.CSV_PATH)
    cells = pn.canonical_by_seed(rows)
    out = {}
    for m in PAPER5 + ["opus5"]:
        for s in ["hard45", "pool447"]:
            for h in pn.HARNESSES:
                lab, c = pn.pick_arm(cells, m, s, h)
                if c is None:
                    continue
                for sd, d in c.items():
                    out[(m, s, H_FROM[h], int(sd))] = {
                        t: dict(rec_from_paper_row(r, "data/trials.csv"), arm=lab)
                        for t, r in d.items()}
    return out


# ------------------------------------------------------------------ second-round rows
def rec_from_r2(r):
    return {"task": r["task"], "usable": r["usable"] == "1", "passed": int(r["passed"] == "1"),
            "pt": fnum(r["prompt_tokens"]) or 0, "cr": fnum(r["cache_read_tokens"]) or 0,
            "cw": fnum(r["cache_write_tokens"]) or 0, "ot": fnum(r["output_tokens"]) or 0,
            "steps": fnum(r["steps"]), "secs": fnum(r["agent_seconds"]), "cost_usd": fnum(r["cost_usd"]),
            "exc": r["exception_type"], "job": r["job"], "trial": r["trial"], "src": "data/round2/trials_round2.csv"}


def load_r2(experiment):
    """one row per (harness, run, task): the canonical trial (already selected in the table)."""
    m = EXP_MODEL[experiment]
    by = {}
    for r in r2rows(experiment):
        by.setdefault((r["subset"], H_FROM[r["harness"]], int(r["run"])), {}).setdefault(r["task"], []).append(r)
    out = {}
    for (s, h, run), d in by.items():
        cell = {}
        for t, lst in d.items():
            assert len(lst) == 1, (experiment, h, run, t)
            cell[t] = rec_from_r2(lst[0])
        out[(m, s, h, run)] = cell
    return out


def load_glm_r2():
    return load_r2("swe_glm53flash_run2")


def load_hy4_r2():
    return load_r2("swe_hy4preview_run2")


def load_opus55_hard45():
    """mini + OC: one run each; CC: three runs."""
    return load_r2("swe_opus55_hard45")


def load_all():
    cells = {}
    cells.update(load_paper())
    cells.update(load_glm_r2())
    cells.update(load_hy4_r2())
    cells.update(load_opus55_hard45())
    cells.update(load_r2("swe_dsv41flash_hard45"))
    cells.update(load_r2("swe_dsv41flash_pool447"))
    return cells


def runs_of(cells, m, s, h):
    return sorted(k[3] for k in cells if k[0] == m and k[1] == s and k[2] == h)


def usable(d):
    return {t: r for t, r in d.items() if r["usable"]}


def kn(d):
    u = usable(d)
    return sum(r["passed"] for r in u.values()), len(u)


def task_mean(cells, m, s, h, runs=None):
    """task -> mean pass over that cell's usable runs (tasks with no usable run are absent)."""
    acc = {}
    for rn in (runs or runs_of(cells, m, s, h)):
        for t, r in usable(cells[(m, s, h, rn)]).items():
            acc.setdefault(t, []).append(r["passed"])
    return {t: sum(v) / len(v) for t, v in acc.items()}


# ------------------------------------------------------------------ stats helpers
def pct(sorted_vals, q):
    """linear-interpolated percentile of an already sorted list"""
    n = len(sorted_vals)
    x = q * (n - 1)
    lo = int(x)
    hi = min(lo + 1, n - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (x - lo)


def gammaincc(a, x):
    """regularised upper incomplete gamma Q(a, x) (Numerical Recipes series / continued fraction)."""
    import math
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:
        ap, s, d = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1
            d *= x / ap
            s += d
            if abs(d) < abs(s) * 1e-14:
                break
        return 1.0 - s * math.exp(-x + a * math.log(x) - gln)
    b = x + 1 - a
    c = 1e300
    d = 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1e-300 if abs(d) < 1e-300 else d
        c = b + an / c
        c = 1e-300 if abs(c) < 1e-300 else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-14:
            break
    return math.exp(-x + a * math.log(x) - gln) * h


def chi2_sf(x, df):
    return gammaincc(df / 2.0, x / 2.0)


class Tee:
    """print to stdout and to a file"""
    def __init__(self, path):
        self.f = open(path, "w")

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        print(s)
        self.f.write(s + "\n")

    def close(self):
        self.f.close()


# ------------------------------------------------------------------ pooled paired differences
def paired_diffs(meanA, meanB):
    return {t: meanA[t] - meanB[t] for t in set(meanA) & set(meanB)}


def pooled_boot(diffs, B=10000, seed=20261110):
    """diffs: model -> {task: diff}.  Models weighted equally; task-cluster bootstrap over the
    union of tasks (the same resampled task set is applied to every model).
    Returns dict with point estimates, per-model and pooled 95/90 percentile intervals, boot SEs."""
    import random
    rng = random.Random(seed)
    models = list(diffs)
    tasks = sorted(set().union(*[set(d) for d in diffs.values()]))
    T = len(tasks)
    vec = {m: [diffs[m].get(t) for t in tasks] for m in models}
    est = {m: sum(diffs[m].values()) / len(diffs[m]) for m in models}
    pooled = sum(est.values()) / len(models)
    bs = {m: [] for m in models}
    bp = []
    for _ in range(B):
        idx = [rng.randrange(T) for _ in range(T)]
        vals = []
        for m in models:
            v = vec[m]
            s = 0.0
            c = 0
            for i in idx:
                x = v[i]
                if x is not None:
                    s += x
                    c += 1
            mm = s / c if c else est[m]
            bs[m].append(mm)
            vals.append(mm)
        bp.append(sum(vals) / len(vals))

    def summ(arr, point):
        a = sorted(arr)
        mu = sum(a) / len(a)
        sd = (sum((x - mu) ** 2 for x in a) / (len(a) - 1)) ** 0.5
        return {"est": point, "ci95": (pct(a, .025), pct(a, .975)), "ci90": (pct(a, .05), pct(a, .95)), "se": sd}
    out = {"pooled": summ(bp, pooled), "n_tasks_union": T, "models": {}}
    for m in models:
        out["models"][m] = dict(summ(bs[m], est[m]), n=len(diffs[m]))
    return out


def cochran_q(per_model):
    """per_model: list of (est, se).  Inverse-variance Cochran's Q, df = k-1, I^2."""
    w = [1 / se ** 2 for _, se in per_model]
    mu = sum(wi * e for wi, (e, _) in zip(w, per_model)) / sum(w)
    Q = sum(wi * (e - mu) ** 2 for wi, (e, _) in zip(w, per_model))
    df = len(per_model) - 1
    I2 = max(0.0, (Q - df) / Q) if Q > 0 else 0.0
    return Q, df, chi2_sf(Q, df), I2, mu
