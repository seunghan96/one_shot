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

Only `numpy`, `scipy` and `scikit-learn` are needed to run the method on your
own detectors. Reproducing the benchmark numbers also needs TSB-AD, which the
next section walks through.

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

### 1. Get the benchmark

The numbers above come from [TSB-AD](https://github.com/TheDatumOrg/TSB-AD),
which supplies the detectors, the curated series and the VUS-PR implementation.

```bash
git clone https://github.com/TheDatumOrg/TSB-AD
cd TSB-AD && pip install -e . && cd ..
```

The series are hosted outside GitHub, so download and unpack them into the
checkout.

```bash
# univariate, the collection the main table reads
wget https://www.thedatum.org/datasets/TSB-AD-U.zip
unzip TSB-AD-U.zip -d TSB-AD/Datasets/

# multivariate, used in the appendix
wget https://www.thedatum.org/datasets/TSB-AD-M.zip
unzip TSB-AD-M.zip -d TSB-AD/Datasets/
```

`TSB-AD/Datasets/File_List/TSB-AD-U-Eva.csv` then lists the 350 univariate
series of the evaluation split, which is the split we read. The 152 series of
the table are what remains after the eligibility filter, i.e. a series needs a
second event to evaluate on once the first is confirmed, a label that is not
constant, and a confirmed event long enough to score.

### 2. Score the detectors once

```bash
bash scripts/build_pool.sh TSB-AD scores_u
```

This runs the 18 statistical detectors of the first pool over every series and
writes `scores_u/<series>__<detector>.npy`. It is the slow step, a few hours on
one machine, and it only has to happen once. `prepare_scores.py` skips files it
already wrote, so an interrupted run resumes.

To reproduce the wider pools of the paper, run the same script with the deep
detectors and foundation models, or add window-length variants of a detector
with `--hp` and `--tag`.

```bash
python scripts/prepare_scores.py --tsb TSB-AD --detector PaAno_PAI \
       --out scores_u --hp '{"win_size": 150}' --tag w150
```

### 3. Run the comparison

```bash
python scripts/run_main.py --scores scores_u --out out/main.json
```

Every row of the table is measured inside this one run, on the same confirmed
event and the same hidden part. It writes after every series and skips what it
already has, so it can be interrupted, and it can be sharded across machines
with `--shard i --nshard n` into separate output files.

### 4. Read the table

```bash
python scripts/select_temperature.py out/main.json   # the leave-one-out choice
python scripts/summarize.py 'out/*.json'             # the table, with tests
```

`summarize.py` prints the rows of the table above, together with a Wilcoxon
signed-rank test of each row against OneShot on the same series.

### Multivariate

The same four steps run on the multivariate collection.

```bash
bash scripts/build_pool.sh TSB-AD scores_m M
python scripts/run_main.py --scores scores_m --out out/main_mv.json --min-det 5
```

## Layout

```
oneshot/
├── oneshot/
│   ├── core.py       the method, i.e. window, weights, mixture, threshold
│   ├── data.py       reading a detector pool and finding its events
│   └── metrics.py    VUS-PR, with average precision as a fallback
├── scripts/
│   ├── prepare_scores.py       run one TSB-AD detector into score files
│   ├── build_pool.sh           run the whole statistical pool
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
