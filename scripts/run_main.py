#!/usr/bin/env python3
"""The main comparison of the paper (Table 3).

Every row uses the same confirmed event on the same series and is scored on
the same hidden part, so the only thing that changes across rows is what a
method does with what it sees.

    no label     rand      the detector a coin flip would give
                 avg_all   plain average of every score
                 rank_all  average after a rank transform
    one event    argmax    the best detector on the confirmed window
                 top{K}    average of the window's top K
                 linear    weights proportional to window accuracy
                 soft{T}   OneShot at temperature T
    upper bound  oracle    the best detector on the hidden part

Results are written as JSON, one record per series, and ``summarize.py``
turns a directory of them into the table.

    python scripts/run_main.py --scores scores_u --out out/main.json
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    # One series is processed in order, so extra BLAS threads only add
    # contention when several shards share a host.
    os.environ.setdefault(_v, '1')

import argparse          # noqa: E402
import json              # noqa: E402
import sys               # noqa: E402
import zlib              # noqa: E402

import numpy as np       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oneshot.core import combine, observed_window, soft_weights   # noqa: E402
from oneshot.data import events, load_series, rank_transform, series_names  # noqa: E402
from oneshot.metrics import vus_pr                                # noqa: E402

TEMPERATURES = (0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.3)
TOP_KS = (2, 3, 5, 10)


def one_series(labels, scores, k, reps, rng, earliest, metric):
    """Average every method over ``reps`` draws of the confirmed event."""
    T = len(labels)
    ev = events(labels)
    ranks = rank_transform(scores)
    acc = {}

    for _ in range(reps):
        seen = observed_window(labels, ev, k=k, rng=rng, earliest=earliest)
        if labels[seen].max() == 0 or labels[seen].min() == 1:
            continue
        if labels[~seen].max() == 0 or labels[~seen].min() == 1:
            continue

        hidden = np.array([metric(labels[~seen], s[~seen]) for s in scores], float)
        window = np.array([metric(labels[seen], s[seen]) for s in scores], float)
        if not np.isfinite(hidden).any() or not np.isfinite(window).any():
            continue
        window = np.nan_to_num(window, nan=float(np.nanmin(window)))

        def mix(weights, matrix=None):
            m = scores if matrix is None else matrix
            w = np.clip(np.asarray(weights, float), 0, None)
            if w.sum() <= 0:
                return float('nan')
            return float(metric(labels[~seen], combine(m[:, ~seen], w / w.sum())))

        row = {
            'rand': float(np.nanmean(hidden)),
            'avg_all': mix(np.ones(len(scores))),
            'rank_all': mix(np.ones(len(scores)), ranks),
            'argmax': float(hidden[int(np.argmax(window))]),
            'linear': mix(window - window.min()),
            'oracle': float(np.nanmax(hidden)),
        }
        order = np.argsort(-window)
        for K in TOP_KS:
            if K <= len(scores):
                w = np.zeros(len(scores))
                w[order[:K]] = 1.0
                row[f'top{K}'] = mix(w)
        for t in TEMPERATURES:
            row[f'soft{t}'] = mix(soft_weights(window, t))

        for key, value in row.items():
            if np.isfinite(value):
                acc.setdefault(key, []).append(value)
    return acc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--scores', required=True, help='directory of score .npy files')
    ap.add_argument('--out', required=True, help='output .json')
    ap.add_argument('--k', type=int, default=1, help='confirmed events per series')
    ap.add_argument('--reps', type=int, default=8, help='draws of the confirmed event')
    ap.add_argument('--min-det', type=int, default=10)
    ap.add_argument('--earliest', action='store_true',
                    help='confirm the earliest events instead of drawing them')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    rows = json.load(open(args.out)) if os.path.exists(args.out) else []
    done = {r['series'] for r in rows}

    names = series_names(args.scores)[args.shard::args.nshard]
    for name in names:
        if name in done:
            continue
        labels, scores, _ = load_series(name, args.scores)
        if scores is None or len(scores) < args.min_det:
            continue
        if labels.max() == 0 or labels.min() == 1:
            continue
        ev = events(labels)
        if len(ev) < args.k + 1:
            continue

        # VUS-PR costs time in proportion to the length of the series, so a
        # very long one is read fewer times. The repetitions average over
        # which event is confirmed, and three already hold the direction.
        reps = args.reps if len(labels) <= 20000 else max(3, int(args.reps * 20000 / len(labels)))
        # The seed comes from the series name, never from hash(), which
        # changes between processes and would make a rerun a different trial.
        rng = np.random.default_rng(zlib.crc32(name.encode()))

        acc = one_series(labels, scores, args.k, reps, rng, args.earliest, vus_pr)
        if 'argmax' not in acc:
            continue
        rows.append(dict(series=name, n_det=len(scores), n_ev=len(ev), reps=reps,
                         **{k: float(np.mean(v)) for k, v in acc.items()}))
        json.dump(rows, open(args.out, 'w'))       # keep partial work on a crash
        print(f'{name:50s} n_det={len(scores):3d} ours={rows[-1]["soft0.02"]:.3f}', flush=True)

    json.dump(rows, open(args.out, 'w'))
    print(f'\n{len(rows)} series -> {args.out}')


if __name__ == '__main__':
    main()
