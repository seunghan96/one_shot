#!/usr/bin/env python3
"""Write the detector pool OneShot reads, using TSB-AD.

Every detector scores every series once, before any label arrives, and no
detector is ever retrained. The output is one file per detector per series,

    <out>/<series>__<detector>.npy
    <out>/<series>__label.npy

which is the layout ``oneshot.data.load_series`` expects. A deployment that
already runs its own detectors can skip this script and write that layout
directly.

    python scripts/prepare_scores.py --detector IForest --out scores_u
"""
import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '4')

import argparse   # noqa: E402
import json       # noqa: E402
import sys        # noqa: E402

import numpy as np    # noqa: E402
import pandas as pd   # noqa: E402


def _patch_numpy_aliases():
    """TSB-AD still uses names numpy 2.0 removed, e.g. np.Inf."""
    for old, new in (('Inf', 'inf'), ('NaN', 'nan'), ('NAN', 'nan'),
                     ('PINF', 'inf'), ('float_', 'float64'), ('int_', 'int64'),
                     ('bool8', 'bool_'), ('infty', 'inf')):
        if not hasattr(np, old):
            setattr(np, old, getattr(np, new))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--tsb', required=True, help='path to a TSB-AD checkout')
    ap.add_argument('--detector', required=True, help='detector name in the TSB-AD pool')
    ap.add_argument('--out', required=True, help='output directory')
    ap.add_argument('--dataset', default='U', choices=['U', 'M'])
    ap.add_argument('--split', default='Eva', choices=['Eva', 'Tuning'])
    ap.add_argument('--hp', default='', help='JSON overriding the tuned hyperparameters')
    ap.add_argument('--tag', default='', help='suffix, e.g. to keep several window lengths apart')
    ap.add_argument('--seed', type=int, default=2024)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshard', type=int, default=1)
    args = ap.parse_args()

    sys.path.insert(0, args.tsb)
    _patch_numpy_aliases()

    import random
    import torch
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    from TSB_AD.model_wrapper import (Semisupervise_AD_Pool, Unsupervise_AD_Pool,
                                      run_Semisupervise_AD, run_Unsupervise_AD)
    if args.dataset == 'U':
        from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict as HP
        data_dir = f'{args.tsb}/Datasets/TSB-AD-U/'
        file_list = f'{args.tsb}/Datasets/File_List/TSB-AD-U-{args.split}.csv'
    else:
        from TSB_AD.HP_list import Optimal_Multi_algo_HP_dict as HP
        data_dir = f'{args.tsb}/Datasets/TSB-AD-M/'
        file_list = f'{args.tsb}/Datasets/File_List/TSB-AD-M-{args.split}.csv'

    os.makedirs(args.out, exist_ok=True)
    files = pd.read_csv(file_list)['file_name'].values[args.shard::args.nshard]
    hp = dict(HP.get(args.detector, {}))
    if args.hp:
        hp.update(json.loads(args.hp))
    name = args.detector + (f'-{args.tag}' if args.tag else '')
    print(f'{name} on {len(files)} series, hyperparameters {hp}', flush=True)

    ok = failed = 0
    for fn in files:
        base = fn.split('.')[0]
        out = os.path.join(args.out, f'{base}__{name}.npy')
        if os.path.exists(out):
            continue
        try:
            df = pd.read_csv(os.path.join(data_dir, fn)).dropna()
            data = df.iloc[:, 0:-1].values.astype(float)
            label = df.iloc[:, -1].values.astype(int)
            # tr_<n> in the file name is the length of the training part, which
            # is all a semi-supervised detector may look at.
            train_end = int(base.split('_')[-3])
            if args.detector in Semisupervise_AD_Pool:
                score = run_Semisupervise_AD(args.detector, data[:train_end, :], data, **hp)
            elif args.detector in Unsupervise_AD_Pool:
                score = run_Unsupervise_AD(args.detector, data, **hp)
            else:
                raise SystemExit(f'{args.detector} is not in the TSB-AD pool')
            if isinstance(score, np.ndarray) and score.shape[0] == data.shape[0]:
                np.save(out, score.astype(np.float32))
                lab = os.path.join(args.out, f'{base}__label.npy')
                if not os.path.exists(lab):
                    np.save(lab, label.astype(np.int8))
                ok += 1
            else:
                failed += 1
        except Exception as exc:                      # a detector may fail on a series
            failed += 1
            print(f'  skip {base}: {exc}', flush=True)
    print(f'done, {ok} written and {failed} skipped')


if __name__ == '__main__':
    main()
