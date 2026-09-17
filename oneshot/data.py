"""Reading a detector pool off disk.

A deployment keeps one score vector per detector per series. We expect the
same layout, i.e. one directory holding

    <series>__label.npy          the ground truth, 0 or 1 per time point
    <series>__<detector>.npy     that detector's score, one per time point

``scripts/prepare_scores.py`` writes this layout from TSB-AD.
"""
from __future__ import annotations

import glob
import os

import numpy as np


def series_names(score_dir):
    """Every series that has a label file in ``score_dir``."""
    labs = sorted(glob.glob(os.path.join(score_dir, '*__label.npy')))
    return [os.path.basename(p)[:-len('__label.npy')] for p in labs]


def load_series(name, score_dir, min_unique=10):
    """Return ``(labels, scores, detector_names)`` for one series.

    Scores are standardized per detector, which lets detectors of different
    scales be summed. Two kinds of detector are dropped here.

    * A score with too many non-finite values, since some deep detectors emit
      inf and a single one poisons every mean the pipeline takes afterwards.
    * A constant or near-constant score, since it carries no ranking and only
      inflates the oracle a comparison is measured against.
    """
    labels = np.load(os.path.join(score_dir, f'{name}__label.npy'))
    scores, names = [], []
    for path in sorted(glob.glob(os.path.join(score_dir, f'{name}__*.npy'))):
        det = os.path.basename(path)[len(name) + 2:-4]
        if det == 'label':
            continue
        s = np.load(path).astype(np.float64)
        if s.shape[0] != labels.shape[0] or s.size == 0:
            continue
        finite = np.isfinite(s)
        if finite.sum() < 0.5 * s.size:
            continue
        if not finite.all():
            s = np.where(finite, s, np.median(s[finite]))
        if s.std() < 1e-9 or len(np.unique(s)) < max(min_unique, 0.001 * len(s)):
            continue
        z = (s - s.mean()) / s.std()
        if not np.isfinite(z).all():
            continue
        scores.append(z.astype(np.float32))
        names.append(det)
    return labels, (np.stack(scores) if scores else None), names


def events(labels):
    """Contiguous anomaly events as ``[(start, end), ...]``, end exclusive."""
    d = np.diff(np.r_[0, np.asarray(labels).astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def rank_transform(scores):
    """Each detector's score mapped to [0, 1] ranks, for the rank-average baseline."""
    out = np.empty_like(scores)
    for i, s in enumerate(scores):
        out[i] = np.argsort(np.argsort(s)) / max(1, len(s) - 1)
    return out


def eligible(name, score_dir, min_detectors=10, min_events=2, min_event_len=6):
    """Whether the protocol can run on a series at all.

    A series needs a second event to evaluate on once the first is confirmed,
    a label that is not constant, and a confirmed event long enough to score.
    """
    labels, scores, _ = load_series(name, score_dir)
    if scores is None or len(scores) < min_detectors:
        return False
    if labels.max() == 0 or labels.min() == 1:
        return False
    ev = events(labels)
    return len(ev) >= min_events and max(e - s for s, e in ev) >= min_event_len
