#!/usr/bin/env python3
"""OneShot on a synthetic series, with no benchmark and no downloads.

Five detectors watch the same signal, and each of them misses a different
anomaly, which is the situation a deployment is actually in. An operator
confirms a single event, and that one event is all OneShot reads.

The detector that looks best on the confirmed event is not the one that does
best on the rest of the series, and spreading the weight recovers what that
mistake costs.

    python examples/quickstart.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from oneshot.core import combine, observed_window, soft_weights, window_accuracy
from oneshot.data import events
from oneshot.metrics import auc_pr

rng = np.random.default_rng(0)
T, STARTS, LENGTH = 4000, (400, 1200, 2000, 2800, 3500), 60

labels = np.zeros(T, int)
for start in STARTS:
    labels[start:start + LENGTH] = 1

# Detector i is blind to one of the events and sees the others with a
# strength of its own, so no detector dominates the series.
detectors = []
for i in range(5):
    s = np.zeros(T)
    for j, start in enumerate(STARTS):
        if (j + i) % 5 != 0:
            s[start:start + LENGTH] += rng.uniform(0.6, 1.4)
    detectors.append(s + rng.normal(0, 0.9, T))
scores = np.stack([(d - d.mean()) / d.std() for d in detectors])
names = [f'detector {i}' for i in range(5)]

# The operator confirms one event. Everything below reads that window alone.
seen = observed_window(labels, events(labels), k=1, rng=rng)
accuracy = window_accuracy(scores, labels, seen, metric=auc_pr)
weights = soft_weights(accuracy, temperature=0.05)
mixture = combine(scores, weights)

print('what the confirmed event says')
for name, a, w in zip(names, accuracy, weights):
    print(f'  {name}   window accuracy {a:.3f}   weight {w:.2f}')

hidden = ~seen
winner = int(np.argmax(accuracy))
print('\nwhat the rest of the series says')
for name, s in zip(names, scores):
    print(f'  {name}   {auc_pr(labels[hidden], s[hidden]):.3f}')
print(f'\n  best on the window ({names[winner]})   '
      f'{auc_pr(labels[hidden], scores[winner][hidden]):.3f}')
print(f'  OneShot                        '
      f'{auc_pr(labels[hidden], mixture[hidden]):.3f}')
