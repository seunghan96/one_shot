#!/usr/bin/env bash
# Score every detector of the statistical pool on every univariate series.
#
# This is the pool the main table reads. Each detector runs once per series,
# before any label arrives, and writes one score file per series.
#
#   bash scripts/build_pool.sh /path/to/TSB-AD scores_u
#
# The 18 detectors below are the statistical block of TSB-AD. Add the deep
# detectors and foundation models the same way to reproduce the "+ DL" pool,
# e.g. AnomalyTransformer, CNN, FITS, LSTMAD, MOMENT_ZS, OFA, OmniAnomaly,
# PatchTST, TimesNet, TranAD, USAD.
set -euo pipefail

TSB=${1:?usage: build_pool.sh <TSB-AD path> <out dir> [U|M]}
OUT=${2:?usage: build_pool.sh <TSB-AD path> <out dir> [U|M]}
SET=${3:-U}

# The multivariate block is smaller, since a detector that reads one channel
# at a time, e.g. a subsequence variant or a matrix profile, does not apply.
if [ "$SET" = "M" ]; then
  DETECTORS=(CBLOF COPOD EIF HBOS IForest KMeansAD KNN LOF OCSVM PCA RobustPCA)
else
  DETECTORS=(CBLOF COPOD EIF FFT HBOS IForest KMeansAD_U KNN LOF MatrixProfile
             PCA POLY SR Sub_HBOS Sub_IForest Sub_KNN Sub_LOF Sub_PCA)
fi

for det in "${DETECTORS[@]}"; do
  echo "== $det"
  python3 scripts/prepare_scores.py --tsb "$TSB" --detector "$det" \
          --out "$OUT" --dataset "$SET"
done

echo
echo "pool written to $OUT"
ls "$OUT" | sed 's/.*__//' | sort -u | wc -l | xargs echo "distinct detectors:"
