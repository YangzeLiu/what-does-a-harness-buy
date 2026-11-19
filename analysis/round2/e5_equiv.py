"""Equivalence bounds (TOST at 2, 3, 4, 5 points) for the pool447 paired differences of the two Qwen models
(first-round data; feeds the Qwen rows and the 'within' column of the pool447 table).  Per model and pair:
n, difference, 90% Newcombe paired interval, exact McNemar p, which bounds the interval fits (Y/.), and the
smallest symmetric bound it fits.
Run: python3 -I e5_equiv.py > ../../paper/round2/e5_equiv.txt"""
import os, sys, csv
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import paper_numbers as pn, main_table as mt
rows = list(csv.DictReader(open(os.path.join(ROOT, 'data', 'trials.csv'))))
cells = pn.canonical_by_seed(rows)
S = mt.MAIN_SEED
def arm(m, s, h):
    lab, c = pn.pick_arm(cells, m, s, h)
    return mt.usable(c[S]) if c else None
DELTAS = [2, 3, 4, 5]
for s, models in [('pool447', ['qwen3.6-35b-a3b', 'qwen3.8-27b'])]:
    print(f'== {s}')
    for h1, h2 in pn.PAIRS:
        for m in models:
            a, b = arm(m, s, h1), arm(m, s, h2)
            if not a or not b: continue
            e = pn.paired(a, b); lo, hi = e['ci90_pp']
            v = ''.join('Y' if (lo > -d and hi < d) else '.' for d in DELTAS)
            print(f"  {pn.H_SHORT[h1]}-{pn.H_SHORT[h2]:4} {m:16} n={e['n']:3} diff={e['diff_pp']:+5.1f} 90%[{lo:+5.1f},{hi:+5.1f}] p={e['p_mcnemar']:.3f} TOST±2345={v} minδ={max(-lo,hi):.1f}")
