#!/usr/bin/env python3
"""Paired analysis of the Qwen3.8 termination ablation (B38).

Design (pre-registered; paper appendix "The output cutoff and a recovery transplant"):
  41 tasks x 2 arms x 2 attempts = 164 trials, both arms started together and
  time-interleaved on the same host so the cost columns are comparable.
    stock = oc-qwen38-B-stock   (HARBOR_OC_AUTOCONTINUE unset)
    ac    = oc-qwen38-B-ac      (HARBOR_OC_AUTOCONTINUE=2)

Reporting order, fixed before the data were seen:
  (i)   ITT over all 41 tasks -- Wilcoxon signed-rank on the per-task mean
        reward difference.  This is the primary result: what turning the switch
        on actually buys on a subset selected for the failure mode.
  (ii)  the measured round-1 trigger rate, which converts (i) to a per-protocol
        scale (ITT ~= r * per-protocol effect).
  (iii) a secondary, explicitly NON-RANDOMISED per-protocol contrast restricted
        to tasks whose round 1 triggered.  Conditioning on a post-randomisation
        variable; reported as a mechanism check, not a causal estimate.
Also printed: the canonical-attempt McNemar, for comparability with the rest of
the paper, and the cost ratios (legitimate here -- the arms are interleaved).

Two attempts on one task are NOT independent and must never be pooled as 82
McNemar pairs.

Usage: b38_paired.py [jobs_root] [--stock JOB] [--ac JOB]
"""
import argparse, glob, json, math, os, statistics, sys
from functools import lru_cache
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from b38_trigger import scan_log, read_autocontinue, reconstruct  # noqa: E402

INFRA = {"NetworkConnectionError", "AgentSetupTimeoutError", "RuntimeError", "CancelledError",
         "OpenCodeStartupHang", "OpenCodeStartupError", "AgentVerifierOverlap",
         "VerifierTimeoutError", "verifier_incomplete", "VerifierUvMissing",
         "RewardFileNotFoundError", "ApiQuotaOrAuthError", "AgentIncomplete", "cc_api_error"}


def mcnemar_exact_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / (2.0 ** n))


EXACT_MAX = 200         # the exact null is built by DP, not by enumeration, so this is cheap


def _signed_ranks(diffs):
    """Drop zeros, return (values, tie-averaged ranks of |value|)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return d, ranks


@lru_cache(maxsize=None)
def _null_dist(scaled_ranks):
    """Exact null distribution of 2*W+ over all 2**m sign assignments, by DP.

    Enumerating the 2**m subsets is the definition but is exponential; convolving the
    ranks one at a time gives the identical distribution in O(m * sum(ranks)).  Ranks are
    tie-averaged so they are always multiples of 0.5 -> scale by 2 to stay in integers.
    Cached on the rank multiset, which repeats constantly inside the power simulation.
    """
    total = sum(scaled_ranks)
    dist = [0] * (total + 1)
    dist[0] = 1
    for r in scaled_ranks:
        for s in range(total, r - 1, -1):
            c = dist[s - r]
            if c:
                dist[s] += c
    return tuple(dist)


def wilcoxon_exact(diffs):
    """EXACT two-sided signed-rank p.

    The null is that each non-zero difference is equally likely to have come out + or -,
    so the reference distribution of W+ is the distribution of the sum of a random subset
    of the (tie-averaged) ranks.  This is the right test here: all of B38's non-zero
    per-task mean differences have the same magnitude (0.5, one attempt flipping out of
    two), so the ranks are completely tied and the signed-rank statistic degenerates to a
    sign test.  The normal approximation the earlier version used -- even with the tie and
    continuity corrections -- is anti-conservative in exactly that regime (0.3506 against
    an exact 0.5078 on the ITT contrast), and at the per-protocol subset's m = 3 it has no
    basis at all.

    -> (p, m, method).  Falls back to the old approximation above EXACT_MAX.
    """
    d, ranks = _signed_ranks(diffs)
    m = len(d)
    if m == 0:
        return 1.0, 0, "exact"
    if m > EXACT_MAX:
        return wilcoxon_approx(diffs)[0], m, "normal approx (m > %d)" % EXACT_MAX
    scaled = tuple(sorted(int(round(r * 2)) for r in ranks))
    dist = _null_dist(scaled)
    total = sum(scaled)
    mu = total / 2.0
    w_obs = sum(r for r, x in zip(ranks, d) if x > 0) * 2
    dev = abs(w_obs - mu)
    hits = sum(c for s, c in enumerate(dist) if abs(s - mu) >= dev - 1e-9)
    return hits / float(1 << m), m, "exact"


def wilcoxon_approx(diffs):
    """The previous statistic, kept so the change in reporting is visible: two-sided
    signed-rank, tie-averaged ranks, normal approximation with tie AND continuity
    correction.  Reported alongside the exact p, never used for a claim."""
    d, ranks = _signed_ranks(diffs)
    n = len(d)
    if n == 0:
        return 1.0, 0
    wp = sum(r for r, x in zip(ranks, d) if x > 0)
    wm = sum(r for r, x in zip(ranks, d) if x < 0)
    w = min(wp, wm)
    mu = n * (n + 1) / 4.0
    from collections import Counter
    tc = sum(t ** 3 - t for t in Counter(abs(x) for x in d).values())
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tc / 48.0
    if var <= 0:
        return 1.0, n
    z = (abs(w - mu) - 0.5) / math.sqrt(var)
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
    return min(1.0, p), n


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    ph = k / n
    den = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / den
    h = z * math.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def dur(r):
    """Wall-clock of the AGENT EXECUTION phase only (agent_execution.started_at ->
    .finished_at).  Not the whole trial: environment build, agent setup and the verifier
    are treatment-independent legs that only dilute the ratio toward 1 (the whole-trial
    reading gives 1.087x where the agent phase gives 1.092x)."""
    from datetime import datetime
    ae = r.get("agent_execution") or {}
    try:
        a = datetime.fromisoformat(ae["started_at"].replace("Z", "+00:00"))
        b = datetime.fromisoformat(ae["finished_at"].replace("Z", "+00:00"))
        return (b - a).total_seconds()
    except Exception:
        return None


def t_crit(df):
    """Two-sided 0.975 Student-t quantile, stdlib only (no scipy dependency here)."""
    if df <= 0:
        return float("nan")
    lo, hi = 0.0, 1000.0
    for _ in range(200):                       # bisect the Student-t CDF
        mid = (lo + hi) / 2.0
        if _t_cdf(mid, df) < 0.975:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def _t_cdf(t, df):
    x = df / (df + t * t)
    ib = _betainc(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t > 0 else 0.5 * ib


def _betainc(a, b, x):
    """Regularised incomplete beta I_x(a,b) by continued fraction."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1.0 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1.0 - c * d) < 1e-12:
            break
    return front * (f - 1.0)


def load_labels(csv_path):
    """(job, trial_dir) -> {exc, scored, reward} from data/trials.csv.

    The CSV is the authority on the exception label, not result.json, because the
    extractor RECLASSIFIES some trials after the fact.  The case that matters here is
    protocol rule 3: when the agent is still writing more than 30 s after
    agent_execution.finished_at the trial is relabelled AgentVerifierOverlap and voided
    as infrastructure.  oc-qwen38-B-stock/sympy__sympy-23413__3TnxUWa is exactly that --
    post_timeout_sec = 41.6 -- and its result.json still says AgentTimeoutError with
    reward 1.0, so reading result.json directly counts a voided trial as a scored pass.
    """
    if not csv_path or not os.path.exists(csv_path):
        return None
    import csv as _csv
    out = {}
    with open(csv_path, newline="") as fh:
        for row in _csv.DictReader(fh):
            try:
                rew = float(row.get("reward", ""))
                if rew != rew:                      # NaN
                    rew = None
            except (TypeError, ValueError):
                rew = None
            out[(row["job"], "%s__%s" % (row["task"], row["trial_id"]))] = dict(
                exc=(row.get("exception_type") or "") or None,
                scored=str(row.get("scored", "")).strip() in ("1", "True", "true"),
                rew=rew,
            )
    return out


def load_arm(root, job, logged_trigger, labels=None):
    """-> {task: [attempt dicts]}, newest-first order preserved by trial name sort."""
    out = {}
    for d in sorted(glob.glob(os.path.join(root, job, "*"))):
        rj = os.path.join(d, "result.json")
        if not os.path.isdir(d) or not os.path.exists(rj):
            continue
        try:
            r = json.load(open(rj))
        except Exception:
            continue
        exc = (r.get("exception_info") or {}).get("exception_type")
        rew = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        ar = r.get("agent_result") or {}
        log = os.path.join(d, "agent", "opencode.txt")
        feat = scan_log(log) if os.path.exists(log) else None
        acf = os.path.join(d, "agent", "autocontinue.txt")
        L = read_autocontinue(acf) if os.path.exists(acf) else None
        if logged_trigger:
            if L:
                trig = bool(L.get("cut_before")) or L.get("modified_before") == 0
                rule = "a" if L.get("cut_before") else ("b" if L.get("modified_before") == 0 else None)
            else:
                trig, rule = None, None
        else:
            trig, rule = reconstruct(feat)
        task = os.path.basename(d).rsplit("__", 1)[0]
        name = os.path.basename(d)
        lab = (labels or {}).get((job, name))
        scored = rew is not None
        if lab is not None:                         # trials.csv overrides result.json
            exc, rew, scored = lab["exc"], lab["rew"], lab["scored"]
        out.setdefault(task, []).append(dict(
            trial=name, exc=exc, rew=rew, scored=scored, trig=trig, rule=rule,
            rounds=(L or {}).get("rounds_used"),
            steps=(feat or {}).get("steps"), secs=dur(r),
            ptok=ar.get("n_input_tokens"), ctok=ar.get("n_output_tokens"),
            started=r.get("started_at") or "",
        ))
    # canonical attempt = the one that started first
    for v in out.values():
        v.sort(key=lambda x: x["started"])
    return out


def usable(a):
    """Scored AND not infrastructure.  INFRA contains AgentVerifierOverlap, so rule 3's
    void is applied here; AgentTimeoutError is not in INFRA and stays scored (rule 2)."""
    return a["exc"] not in INFRA and a["scored"] and a["rew"] is not None


def gmean_ratio(pairs):
    """pairs of (ac_value, stock_value) -> geometric mean ratio + 95% CI on the mean
    log-ratio using t(n-1), which is the convention the paper uses for every efficiency
    interval in the paper.  (This script previously used z = 1.96 while the appendix
    said t; at n = 41 that is ~3 % too narrow.)"""
    lg = [math.log(x / y) for x, y in pairs if x and y and x > 0 and y > 0]
    if len(lg) < 2:
        return None
    n = len(lg)
    m = statistics.mean(lg)
    se = statistics.stdev(lg) / math.sqrt(n)
    tc = t_crit(n - 1)
    return math.exp(m), math.exp(m - tc * se), math.exp(m + tc * se), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_root", nargs="?",
                    default=os.environ.get("HARNESS_JOBS_ROOT", os.path.join("results", "raw", "jobs")))
    ap.add_argument("--stock", default="oc-qwen38-B-stock")
    ap.add_argument("--ac", default="oc-qwen38-B-ac")
    ap.add_argument("--parent", default="oc-qwen-pool447-nw-rg",
                    help="parent pool job, for the unconditional trigger rate")
    ap.add_argument("--trials-csv", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "data", "trials.csv"),
        help="exception/scored labels; the CSV overrides result.json where they disagree")
    a = ap.parse_args()

    labels = load_labels(a.trials_csv)
    S = load_arm(a.jobs_root, a.stock, logged_trigger=False, labels=labels)
    A = load_arm(a.jobs_root, a.ac, logged_trigger=True, labels=labels)
    tasks = sorted(set(S) & set(A))

    print("B38  stock=%s  ac=%s  tasks in both arms=%d" % (a.stock, a.ac, len(tasks)))
    ns = sum(len(v) for v in S.values()); na = sum(len(v) for v in A.values())
    print("trials: stock %d, ac %d (target 82 each)" % (ns, na))
    print("wall-clock = the agent_execution phase only (started_at -> finished_at), not the")
    print("whole trial; Wilcoxon p-values are EXACT (2^m sign enumeration, tie-averaged ranks);")
    print("cost intervals are t(n-1) on the mean log-ratio.")
    print("labels: %s" % ("data/trials.csv (overrides result.json)" if labels
                           else "result.json only -- trials.csv not found"))
    print()

    # ---- protocol rule 3: per-arm exception counts, reported ALONGSIDE the paired table ----
    # "because that void is correlated with arm length its per-arm counts are reported
    #  alongside the paired table rather than folded silently into the exclusion total".
    print("== exceptions per arm (protocol rule 3)")
    from collections import Counter
    for arm, D in (("stock", S), ("ac", A)):
        allx = [x for v in D.values() for x in v]
        exc = Counter(x["exc"] for x in allx if x["exc"])
        voided = [x for x in allx if x["exc"] in INFRA]
        unscored = [x for x in allx if x["exc"] not in INFRA and not (x["scored"] and x["rew"] is not None)]
        print("   %-5s trials %d   usable %d   VOIDED(INFRA) %d   unscored-but-not-INFRA %d"
              % (arm, len(allx), sum(1 for x in allx if usable(x)), len(voided), len(unscored)))
        print("         AgentVerifierOverlap: %d   other exception types: %s"
              % (exc.get("AgentVerifierOverlap", 0),
                 ", ".join("%s=%d" % kv for kv in sorted(exc.items())
                           if kv[0] != "AgentVerifierOverlap") or "none"))
        for x in voided:
            print("         VOID  %-34s %s" % (x["trial"], x["exc"]))
        for x in unscored:
            print("         UNSC  %-34s %s (scored=%s reward=%s)"
                  % (x["trial"], x["exc"] or "-", x["scored"], x["rew"]))
    print()

    # ---------- (i) ITT: per-task mean reward, Wilcoxon ----------
    diffs, kept, per_task = [], [], []
    for t in tasks:
        su = [x for x in S[t] if usable(x)]
        au = [x for x in A[t] if usable(x)]
        if not su or not au:
            continue
        ms = statistics.mean(float(bool(x["rew"])) for x in su)
        ma = statistics.mean(float(bool(x["rew"])) for x in au)
        diffs.append(ma - ms)
        kept.append(t)
        per_task.append((t, ms, ma, len(su), len(au)))
    p, npos, how = wilcoxon_exact(diffs)
    p_approx, _ = wilcoxon_approx(diffs)
    mean_s = statistics.mean(m for _, m, _, _, _ in per_task) if per_task else 0.0
    mean_a = statistics.mean(m for _, _, m, _, _ in per_task) if per_task else 0.0
    print("== (i) ITT over all tasks with >=1 usable attempt in both arms  (PRIMARY)")
    print("   tasks n=%d   mean per-task reward: stock %.4f  ac %.4f   delta %+.4f" %
          (len(kept), mean_s, mean_a, mean_a - mean_s))
    print("   Wilcoxon signed-rank: non-zero diffs %d, two-sided p = %.4f  [%s]" % (npos, p, how))
    print("   (for the record, the normal approximation this used to report: p = %.4f)" % p_approx)
    print("   direction: ac better on %d, stock better on %d, tied %d" %
          (sum(1 for d in diffs if d > 0), sum(1 for d in diffs if d < 0), sum(1 for d in diffs if d == 0)))
    print()

    # canonical-attempt McNemar, for comparability with the rest of the paper
    b = c = 0
    for t in kept:
        su = [x for x in S[t] if usable(x)]; au = [x for x in A[t] if usable(x)]
        s0 = bool(su[0]["rew"]); a0 = bool(au[0]["rew"])
        if a0 and not s0:
            b += 1
        elif s0 and not a0:
            c += 1
    print("   [comparability] canonical-attempt McNemar: b(ac-only)=%d c(stock-only)=%d p=%.4f  flip=%.3f" %
          (b, c, mcnemar_exact_p(b, c), (b + c) / len(kept) if kept else 0.0))
    print()

    # ---------- (ii) trigger rate ----------
    print("== (ii) round-1 trigger rate")
    trg_all, tot_all = [], []
    for label, D, src in (("ac    (logged)", A, "log"), ("stock (reconstructed)", S, "rec")):
        tot = trg = ra = rb = 0
        for t in tasks:
            for x in D[t]:
                if not usable(x) or x["trig"] is None:
                    continue
                tot += 1
                if x["trig"]:
                    trg += 1
                    if x["rule"] == "a":
                        ra += 1
                    else:
                        rb += 1
        trg_all.append(trg); tot_all.append(tot)
        lo, hi = wilson(trg, tot)
        print("   %-22s %3d/%3d = %5.1f%% [%.1f, %.1f]  rule(a)=%d rule(b)=%d" %
              (label, trg, tot, 100.0 * trg / tot if tot else 0, 100 * lo, 100 * hi, ra, rb))
        # Denominator disclosure: this table counts USABLE trials, because the rate is used
        # to put the ITT contrast on a per-protocol scale and the ITT runs on usable trials.
        # analysis/b38_trigger.py counts every trial that HAS A LOG, because there the round-1
        # trigger state is a property of the log and does not depend on the score.  The two
        # denominators differ by exactly the voided/unscored attempts of each arm.
        log_trg = log_tot = 0
        for t in tasks:
            for x in D[t]:
                if x["trig"] is None:
                    continue
                log_tot += 1
                log_trg += bool(x["trig"])
        if (log_trg, log_tot) != (trg, tot):
            print("   %-22s %3d/%3d = %5.1f%%   <- same trials as analysis/b38_trigger.py "
                  "(all with a log; +%d voided/unscored)"
                  % ("  same, all-with-log:", log_trg, log_tot,
                     100.0 * log_trg / log_tot if log_tot else 0, log_tot - tot))
    # The parent rate used to be a hardcoded string.  It is computed now, with the same
    # reconstruction the stock arm uses, so it cannot drift from analysis/b38_trigger.py.
    ptrig = ptot = 0
    for d in sorted(glob.glob(os.path.join(a.jobs_root, a.parent, "*"))):
        log = os.path.join(d, "agent", "opencode.txt")
        if not os.path.isdir(d) or not os.path.exists(log):
            continue
        t, _ = reconstruct(scan_log(log))
        if t is None:
            continue
        ptot += 1
        ptrig += bool(t)
    if ptot:
        print("   parent job %s unconditional rate: %d/%d = %.1f%%  (all trials with a log)"
              % (a.parent, ptrig, ptot, 100.0 * ptrig / ptot))
        print("   -> selection enriches the trigger rate by %.2fx" %
              ((100.0 * (trg_all[0] + trg_all[1]) / (tot_all[0] + tot_all[1])) / (100.0 * ptrig / ptot)))
    print()

    # ---------- (iii) per-protocol, non-randomised ----------
    pp = [i for i, t in enumerate(kept)
          if any(x["trig"] for x in A[t] if usable(x) and x["trig"] is not None)]
    if pp:
        d2 = [diffs[i] for i in pp]
        p2, n2, how2 = wilcoxon_exact(d2)
        p2a, _ = wilcoxon_approx(d2)
        print("== (iii) per-protocol, NON-RANDOMISED (tasks where the ac arm triggered in >=1 attempt)")
        print("   tasks n=%d   mean delta %+.4f   Wilcoxon non-zero %d, p = %.4f  [%s]" %
              (len(pp), statistics.mean(d2), n2, p2, how2))
        print("   (for the record, the normal approximation this used to report: p = %.4f)" % p2a)
        print("   NOTE: conditions on a post-randomisation variable; mechanism check, not a causal estimate.")
    else:
        print("== (iii) per-protocol: no ac attempt triggered -- nothing to report")
    print()

    # ---------- cost (arms are interleaved, so ratios are legitimate) ----------
    COST_KEYS = (("secs", "wall-clock"), ("steps", "steps"),
                 ("ctok", "output tokens"), ("ptok", "input tokens"))

    def cost_block(both_pass):
        for key, name in COST_KEYS:
            pr = []
            for t in kept:
                sx = [x for x in S[t] if usable(x)]
                ax = [x for x in A[t] if usable(x)]
                if both_pass:
                    sx = [x for x in sx if x["rew"]]
                    ax = [x for x in ax if x["rew"]]
                sv = [x[key] for x in sx if x[key]]
                av = [x[key] for x in ax if x[key]]
                if sv and av:
                    pr.append((statistics.mean(av), statistics.mean(sv)))
            g = gmean_ratio(pr)
            if g:
                print("   %-14s %.3fx  [%.3f, %.3f]  n=%d" % (name, g[0], g[1], g[2], g[3]))

    print("== cost (a) PRIMARY -- ITT, paired per task over ALL usable attempts")
    print("   what turning the switch on costs, on the same task set as the ITT contrast.")
    cost_block(both_pass=False)
    print()
    print("== cost (b) SENSITIVITY -- both-pass pairs only (the paper's efficiency convention)")
    print("   restricts to tasks both arms solved, so it conditions on the outcome; reported")
    print("   for comparability with the rest of the paper, not as the ablation's cost.")
    cost_block(both_pass=True)
    print()
    print("   geometric mean ac/stock; 95% CI on the mean log-ratio from t(n-1).")
    print("   A/A floor under unmatched host load is 1.10-1.16x; these arms ARE interleaved.")
    print()

    print("== per task: stock_mean ac_mean (attempts), ac trigger rules")
    for t, ms, ma, ks, ka in per_task:
        rules = "".join(sorted({x["rule"] for x in A[t] if x["trig"] and x["rule"]})) or "-"
        print("   %-34s %.2f -> %.2f  (%d/%d)  trig=%s" % (t, ms, ma, ks, ka, rules))


if __name__ == "__main__":
    main()
