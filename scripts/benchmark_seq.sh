#!/usr/bin/env bash
# Scores the sequence model that train_seq.sh trained (NW_SEQ_RUN) on OLiMPiC scanned dev
# systems: SER over all dev crops, note F1 over the songs with no PDMX copy.
# Result: $NW_RUNS/${NW_SEQ_RUN}_benchmark.txt; compare runs with
#   scorecard.py $NW_RUNS/$NW_SEQ_RUN/benchmark $NW_RUNS/benchmark
set -euo pipefail
source "$(dirname "$0")/common.sh"

run="${NW_SEQ_RUN:-seq512}"
predictions="$NW_RUNS/$run/benchmark/olimpic-systems"
(cd "$ROOT/eval" && "$PY" seq_predict.py "$NW_RUNS/$run/best.pt" --split dev --out "$predictions" \
  && "$PY" olimpic_benchmark.py systems --predictions "$predictions" --out "$predictions") | tee "$NW_RUNS/${run}_benchmark.txt"
