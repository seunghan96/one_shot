"""Ranking measures.

Every comparison in the paper is read under a ranking measure, i.e., one that
asks only whether anomalous points score above normal ones. VUS-PR comes from
the TSB-AD implementation when the benchmark is importable, and falls back to
average precision otherwise, which keeps this package usable on its own.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score

SLIDING_WINDOW = 100


def _usable(labels, score):
    score = np.asarray(score, float)
    labels = np.asarray(labels).astype(int)
    if not np.isfinite(score).all():
        return None, None
    if labels.max() == 0 or labels.min() == 1 or len(labels) < 20:
        return None, None
    return labels, score


def vus_pr(labels, score, sliding_window=SLIDING_WINDOW):
    """VUS-PR, the measure TSB-AD reports as the most reliable one.

    Returns NaN when the input cannot carry a ranking measure, e.g. a window
    that holds no normal point.
    """
    labels, score = _usable(labels, score)
    if labels is None:
        return float('nan')
    try:
        from TSB_AD.evaluation.metrics import get_metrics
        return float(get_metrics(score, labels, slidingWindow=sliding_window)['VUS-PR'])
    except Exception:
        return float(average_precision_score(labels, score))


def auc_pr(labels, score):
    labels, score = _usable(labels, score)
    if labels is None:
        return float('nan')
    return float(average_precision_score(labels, score))


def f1_at(labels, score, threshold):
    """Point-wise F1 of a thresholded decision."""
    pred = (np.asarray(score, float) > threshold).astype(int)
    return float(f1_score(np.asarray(labels).astype(int), pred, zero_division=0))
