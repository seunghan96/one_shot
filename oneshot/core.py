"""The method itself.

A deployment already runs several anomaly detectors and keeps their score
vectors. An operator confirms one anomaly. OneShot turns that single event
into a weight over the whole pool, instead of using it to pick one detector.

    scores  -> accuracy on the confirmed window -> softmax -> weighted sum

Nothing here is trained, and no detector is ever refit.
"""
from __future__ import annotations

import numpy as np

from .metrics import f1_at, vus_pr

DEFAULT_TEMPERATURE = 0.02


def observed_window(labels, events_, k=1, rng=None, earliest=False):
    """Build the observed part O from ``k`` confirmed events.

    The operator confirms whole events, so O holds every point of those
    events plus an equal number of points drawn from the rest of the series
    and treated as normal. A window of positives alone leaves every ranking
    measure undefined, which is why the negatives are needed.

    Returns a boolean mask that is True on the observed part.
    """
    rng = np.random.default_rng() if rng is None else rng
    T = len(labels)
    idx = np.arange(k) if earliest else rng.choice(len(events_), k, replace=False)
    seen = np.zeros(T, bool)
    for i in idx:
        s, e = events_[i]
        seen[s:e] = True
    seen[rng.choice(T, int(seen.sum()), replace=False)] = True
    return seen


def window_accuracy(scores, labels, seen, metric=vus_pr):
    """Score every detector on the confirmed window.

    ``scores`` is (n_detectors, T). Detectors the window cannot separate get
    the lowest observed value rather than NaN, so one bad detector never
    poisons the weights.
    """
    acc = np.array([metric(labels[seen], s[seen]) for s in scores], float)
    if not np.isfinite(acc).any():
        raise ValueError('the window gives no usable score for any detector')
    return np.nan_to_num(acc, nan=float(np.nanmin(acc)))


def soft_weights(accuracy, temperature=DEFAULT_TEMPERATURE):
    """Softmax over the window accuracies.

    A small temperature concentrates the weight on the window winner, and a
    large one flattens it toward the plain score average. The value is a
    property of the method rather than of the series, and we set it once on
    other series (see ``scripts/select_temperature.py``).
    """
    a = np.asarray(accuracy, float)
    w = np.exp((a - a.max()) / float(temperature))
    return w / w.sum()


def combine(scores, weights):
    """Weighted sum of the standardized detector scores."""
    w = np.asarray(weights, float)
    return (w[:, None] * np.asarray(scores, float)).sum(0)


def pick_threshold(combined, labels, seen, n_candidates=50):
    """Read an alarm threshold off the same confirmed window.

    The label that chooses the detectors also says where to raise the alarm,
    which costs nothing beyond the label a deployment already has.
    """
    z = combined[seen]
    best_t, best_f1 = float(np.max(z)), -1.0
    for t in np.quantile(z, np.linspace(0.5, 0.999, n_candidates)):
        f1 = f1_at(labels[seen], z, t)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def oneshot(scores, labels, events_, k=1, temperature=DEFAULT_TEMPERATURE,
            rng=None, earliest=False):
    """Run the whole method once and return the combined score and threshold.

    >>> z, tau, w = oneshot(scores, labels, events(labels))
    >>> alarms = z > tau
    """
    seen = observed_window(labels, events_, k=k, rng=rng, earliest=earliest)
    acc = window_accuracy(scores, labels, seen)
    w = soft_weights(acc, temperature)
    z = combine(scores, w)
    return z, pick_threshold(z, labels, seen), w
