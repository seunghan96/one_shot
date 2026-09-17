<div align="center">

# OneShot

**One confirmed anomaly is enough to weight a pool of anomaly detectors.**

Code for *One Confirmed Anomaly Is Enough: Label-Efficient Detector Weighting
for Time Series Anomaly Detection*, under double-blind review.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

</div>

---

## The problem

Time series anomaly detection has no universal detector. The best one is
decided by the series, not by the algorithm, and a deployment cannot know in
advance which of its detectors to trust.

<div align="center">

| | |
|---|---|
| **Pick one detector** | needs to know which one, before any label exists |
| **Average them all** | throws the good ones in with the useless ones |

</div>

Both assume that no label ever arrives. A deployed monitor does receive
labels, though, because **an operator confirms the anomalies it raises**.

OneShot uses that single confirmed event to weight the whole pool, rather
than to choose one detector from it.

## The method

```
                confirmed event
                      |
   detector scores -> accuracy on the window -> softmax -> weighted sum
                                                              |
                                                     one combined score
```

Three lines is the whole of it.

```python
accuracy = window_accuracy(scores, labels, seen)   # score each detector on the event
weights  = soft_weights(accuracy, temperature)     # turn scores into weights
combined = combine(scores, weights)                # one score for the series
```

Nothing is trained, no detector is refit, and the same confirmed event also
says where to raise the alarm.

## Results

On **152 univariate series** of TSB-AD, every row reads the same confirmed
event and is scored on the same hidden part.

| Method | AUC-PR | AUC-ROC | VUS-PR | VUS-ROC | ρ |
|---|---|---|---|---|---|
| *No labels* | | | | | |
| Random detector | 0.337 | 0.733 | 0.384 | 0.784 | 0.0% |
| Score average | 0.551 | 0.915 | 0.596 | 0.942 | 53.6% |
| Rank average | 0.371 | 0.861 | 0.413 | 0.899 | 7.5% |
| *One confirmed event* | | | | | |
| Best on labeled window | 0.597 | 0.911 | 0.672 | 0.935 | 73.0% |
| Top-2 average | 0.622 | 0.932 | 0.702 | 0.954 | 80.5% |
| Top-3 average | 0.628 | **0.941** | 0.704 | 0.958 | 81.2% |
| Top-10 average | 0.598 | 0.931 | 0.649 | 0.954 | 67.3% |
| Linear weighting | 0.633 | **0.941** | 0.689 | **0.962** | 77.3% |
| **OneShot** | **0.642** | **0.941** | **0.710** | 0.961 | **82.8%** |
| *Upper bound* | | | | | |
| Oracle (best detector for each) | 0.701 | 0.960 | 0.779 | 0.975 | 100.0% |

ρ is the share of the gap between a randomly drawn detector and the best one
that a row closes. OneShot beats every alternative that uses the same label,
and matches the best fixed top-*K* average **without having to choose *K***.

Two more things the experiments show.

* **Where the gain comes from.** Split the series by how well the ranking on
  the window agrees with the ranking on the hidden part. The gain is large on
  every quarter whose rankings disagree, and vanishes where they agree, i.e.
  OneShot insures against a window that ranks the detectors wrongly.
* **What it costs.** 45 ms on top of a pool that already spends 44.9 s, which
  is less than a single detector at the median of that pool.

## Install

```bash
pip install -r requirements.txt
```

Only `numpy`, `scipy` and `scikit-learn` are needed to run the method.
[TSB-AD](https://github.com/TheDatumOrg/TSB-AD) is needed only to reproduce
the benchmark numbers, since it supplies both the detectors and VUS-PR.

## Try it in 10 seconds

```bash
python examples/quickstart.py
```

Five synthetic detectors, each blind to a different anomaly. The detector
that looks best on the confirmed event turns out to be the **worst** on the
rest of the series, and the mixture recovers what that mistake costs.

```
  best on the window (detector 4)   0.144
  OneShot                           0.221
```

## Use it on your own detectors

OneShot reads score vectors, so any detector that produces one per time point
fits. Write them as `<series>__<detector>.npy` next to a `<series>__label.npy`
and the loader picks them up.

```python
from oneshot import combine, events, load_series, observed_window, soft_weights
from oneshot.core import pick_threshold, window_accuracy

labels, scores, names = load_series('my_series', 'scores/')
seen = observed_window(labels, events(labels), k=1)   # the confirmed event

weights  = soft_weights(window_accuracy(scores, labels, seen))
combined = combine(scores, weights)
alarm    = combined > pick_threshold(combined, labels, seen)
```

## Reproduce the paper

```bash
# 1. score every detector on every series, once (repeat per detector)
python scripts/prepare_scores.py --tsb /path/to/TSB-AD \
       --detector IForest --out scores_u

# 2. the main comparison, which can be sharded across machines
python scripts/run_main.py --scores scores_u --out out/main.json

# 3. the temperature, chosen without reading the series it scores
python scripts/select_temperature.py out/main.json

# 4. the table
python scripts/summarize.py out/main.json
```

`run_main.py` writes after every series and skips what it already has, so an
interrupted run resumes where it stopped.

## Layout

```
oneshot/
├── oneshot/
│   ├── core.py       the method, i.e. window, weights, mixture, threshold
│   ├── data.py       reading a detector pool and finding its events
│   └── metrics.py    VUS-PR, with average precision as a fallback
├── scripts/
│   ├── prepare_scores.py       run TSB-AD detectors into score files
│   ├── run_main.py             the main comparison
│   ├── select_temperature.py   leave-one-out over series
│   └── summarize.py            the table, with signed-rank tests
└── examples/
    └── quickstart.py           a runnable synthetic example
```

## A note on the temperature

It is the only quantity OneShot does not read from the confirmed event. We
hold out one series at a time, compare eight candidates on all the others,
and apply the winner to the series we held out, so **no series takes part in
the choice that scores it**. All 152 choices agree on a single value,
τ = 0.02, and any value across an order of magnitude keeps the lead over the
window winner.

## License

MIT. See [LICENSE](LICENSE).

*Author information is omitted while the paper is under review.*
