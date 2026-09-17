#!/usr/bin/env python3
"""Turn the results of run_main.py into the table of the paper.

Rho is the share of the gap between a randomly drawn detector and the best one
that a row closes, and the last column is a Wilcoxon signed-rank test of that
row against OneShot on the same series.

    python scripts/summarize.py out/main.json
"""
import argparse
import glob
import json

import numpy as np
from scipy import stats

from select_temperature import TEMPERATURES, leave_one_out

ROWS = [('rand', 'Random detector'), ('avg_all', 'Score average'),
        ('rank_all', 'Rank average'), ('argmax', 'Best on labeled window'),
        ('top2', 'Top-2 average'), ('top3', 'Top-3 average'),
        ('top5', 'Top-5 average'), ('top10', 'Top-10 average'),
        ('linear', 'Linear weighting'), ('__ours', 'OneShot'),
        ('oracle', 'Oracle (best for each)')]


def load(patterns):
    rows, seen = [], set()
    for pattern in patterns:
        for path in sorted(glob.glob(pattern)):
            for r in json.load(open(path)):
                if r['series'] not in seen:
                    seen.add(r['series'])
                    rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', nargs='+', help='JSON files or globs')
    args = ap.parse_args()

    rows = [r for r in load(args.results)
            if all(np.isfinite(r.get(f'soft{t}', np.nan)) for t in TEMPERATURES)]
    if len(rows) < 8:
        raise SystemExit('not enough series')

    ours, picks = leave_one_out(rows)
    rand = np.mean([r['rand'] for r in rows])
    oracle = np.mean([r['oracle'] for r in rows])

    print(f'\n{len(rows)} series, temperature {sorted(set(picks))}\n')
    print(f'{"Method":26s}{"VUS-PR":>9s}{"rho":>9s}{"p vs ours":>11s}')
    print('-' * 55)
    for key, name in ROWS:
        v = ours if key == '__ours' else np.array(
            [r[key] for r in rows if key in r], float)
        if len(v) != len(rows):
            continue
        rho = (v.mean() - rand) / (oracle - rand) * 100
        p = float('nan') if key == '__ours' else stats.wilcoxon(ours - v).pvalue
        print(f'{name:26s}{v.mean():9.3f}{rho:8.1f}%{p:11.3g}')


if __name__ == '__main__':
    main()
