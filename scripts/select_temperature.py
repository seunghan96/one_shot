#!/usr/bin/env python3
"""Choose the temperature without ever reading the series it scores.

The temperature is the only quantity OneShot does not read from the confirmed
event, and it therefore needs a rule of its own. We hold out one series at a
time, compare the candidates on all the others, and apply the winner to the
series we held out, so no series takes part in the choice that scores it.

    python scripts/select_temperature.py out/main.json
"""
import argparse
import json
import sys

import numpy as np

TEMPERATURES = (0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.3)


def leave_one_out(rows, temperatures=TEMPERATURES):
    """Return the per-series accuracy under the held-out choice, and the picks."""
    M = np.array([[r[f'soft{t}'] for t in temperatures] for r in rows], float)
    total = M.sum(0)
    value, picks = np.empty(len(rows)), []
    for i in range(len(rows)):
        j = int(np.argmax((total - M[i]) / max(len(rows) - 1, 1)))
        picks.append(temperatures[j])
        value[i] = M[i, j]
    return value, picks


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('results', help='JSON written by run_main.py')
    args = ap.parse_args()

    rows = [r for r in json.load(open(args.results))
            if all(np.isfinite(r.get(f'soft{t}', np.nan)) for t in TEMPERATURES)]
    if len(rows) < 8:
        sys.exit('not enough series to choose a temperature')

    value, picks = leave_one_out(rows)
    chosen = sorted(set(picks))
    print(f'{len(rows)} series, {len(chosen)} distinct choice(s): '
          + ', '.join(str(t) for t in chosen))
    print(f'held-out accuracy {value.mean():.4f}')
    for t in TEMPERATURES:
        fixed = np.mean([r[f'soft{t}'] for r in rows])
        mark = ' <- chosen' if t in chosen else ''
        print(f'  fixed at {t:<6} {fixed:.4f}{mark}')


if __name__ == '__main__':
    main()
