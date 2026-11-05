#!/usr/bin/env bash
# Trains the system-crop sequence model (train/seq) and scores it on OLiMPiC scanned dev
# against Zeus. Training data: OLiMPiC synthetic, GrandStaff-LMX and our own PDMX system
# crops (build_seq_dataset.py). Result: $NW_RUNS/seq/best.pt
set -euo pipefail
source "$(dirname "$0")/common.sh"

olimpic="$RAW/olimpic"
clean_dev="$NW_DATA/olimpic_dev_clean.txt"

(cd "$ROOT/train" && "$PY" build_seq_dataset.py --scores "$NW_DATA/render_scores.txt" --out "$NW_DATA/seq_pdmx" \
  --workers "$NW_RENDER_WORKERS" 2>&1 | grep -v '^\[Warning\]')
# PDMX holds copies of some OLiMPiC songs: monitor on the dev songs without one.
(cd "$ROOT/eval" && "$PY" olimpic_benchmark.py list --split dev > "$clean_dev")
(cd "$ROOT/train" && "$PY" train_seq.py --data "$olimpic/olimpic-1.0-synthetic:train:2" "$olimpic/grandstaff-lmx:train" \
  "$NW_DATA/seq_pdmx:train" --dev "$olimpic/olimpic-1.0-scanned:dev" --dev-samples "$clean_dev" --dev-count 658 \
  --out "$NW_RUNS/seq" --steps "$NW_SEQ_STEPS" --workers "$NW_TRAIN_WORKERS") 2>&1 | tee -a "$NW_RUNS/seq.log"

# seq_predict's SER covers all dev crops, including songs with a PDMX copy; the note F1
# below counts only the clean ones.
predictions="$NW_RUNS/seq/benchmark/olimpic-systems"   # compare: scorecard.py $NW_RUNS/seq/benchmark $NW_RUNS/benchmark
(cd "$ROOT/eval" && "$PY" seq_predict.py "$NW_RUNS/seq/best.pt" --split dev --out "$predictions" \
  && "$PY" olimpic_benchmark.py systems --predictions "$predictions" --out "$predictions") | tee "$NW_RUNS/seq_benchmark.txt"
