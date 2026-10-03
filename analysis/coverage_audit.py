#!/usr/bin/env python3
"""Per-arm coverage and void accounting for the hard45 grid.

Written after noticing that fable51's three component arms look large by trial count
(82/80/85) but collapse to 15/17/9 usable tasks once infra voids are removed, because
~83% of them died on ApiQuotaOrAuthError partway through a fixed dispatch order.  A
trial count is therefore not a sample size, and an arm that lost most of its trials to
quota keeps a dispatch-order PREFIX rather than a random subset.

Two things this separates that are easy to conflate:
  * still-running trials (status == "running") are not failures.  Counting them as
    voids overstates the void rate of whatever arm is executing right now.
  * a void rate and a coverage are different numbers.  An arm can void 20% of its
    trials and still cover 46/46 tasks if the retries landed.

Read-only.  Usage: python3 analysis/coverage_audit.py [--subset hard45]
"""
import argparse, collections, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import agg  # noqa: E402  -- shared ok()/INFRA, so voids match the main tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default=os.path.join(HERE, "..", "data", "trials.json"))
    ap.add_argument("--subset", default="hard45")
    ap.add_argument("--full-coverage", type=int, default=46)
    args = ap.parse_args()

    T = [r for r in json.load(open(args.trials)) if r.get("subset") == args.subset]
    N = args.full_coverage

    running = [r for r in T if r.get("status") == "running"]
    if running:
        print("in flight at extraction (excluded from void rates): %d trials" % len(running))
        for ak, n in collections.Counter(r["arm_key"] for r in running).most_common():
            print("   %-52s %d" % (ak, n))
        print()

    print("%-52s %7s %5s %6s %6s  %s" % ("arm_key", "settled", "ok", "void%", "cov", "dominant void"))
    short, quota_hit = [], []
    for ak in sorted({r["arm_key"] for r in T}):
        rs = [r for r in T if r["arm_key"] == ak and r.get("status") != "running"]
        if not rs:
            continue
        ok = [r for r in rs if agg.ok(r)]
        cov = len({r["task"] for r in ok})
        ex = collections.Counter(r.get("exception_type") for r in rs if not agg.ok(r))
        top = ex.most_common(1)[0] if ex else ("-", 0)
        void = 100 * (1 - len(ok) / len(rs))
        flag = ""
        if cov < N - 2:
            short.append((ak, cov))
            flag = "  <-- under-covered"
        if top[0] == "ApiQuotaOrAuthError":
            quota_hit.append(ak)
        print("%-52s %7d %5d %5.1f%% %3d/%d  %-26s%s"
              % (ak, len(rs), len(ok), void, cov, N, "%s x%d" % top, flag))

    if short:
        print("\nUnder-covered arms (a paired contrast using one of these is limited to its tasks):")
        for ak, cov in short:
            print("   %-52s %2d/%d" % (ak, cov, N))
    if quota_hit:
        print("\nArms whose dominant void is API quota -- survivors are a dispatch-order prefix,")
        print("not a random subset, so their marginal rates do not estimate the full subset:")
        for ak in quota_hit:
            print("   %s" % ak)

    bad = [r for r in T if r.get("status") != "running" and not agg.ok(r)]
    spend = collections.Counter()
    count = collections.Counter()
    for r in bad:
        spend[r.get("exception_type")] += r.get("cost_usd") or 0
        count[r.get("exception_type")] += 1
    print("\nSpend on voided trials (loud failures are cheap; the costly ones run to completion first):")
    for k, v in sorted(spend.items(), key=lambda x: -x[1]):
        print("   %-28s %3d trials  $%8.2f  ($%.2f each)" % (k, count[k], v, v / count[k]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
