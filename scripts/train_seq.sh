#!/usr/bin/env bash
# Trains the system-crop sequence model (train/seq) and scores it on OLiMPiC scanned dev
# against Zeus, on the data build_seq_data.sh makes. Result: $NW_RUNS/$NW_SEQ_RUN/ (NW_SEQ_RUN
# defaults to seq512): best.pt, and the averaged weights every 10,000 steps. A new run takes its
# own NW_SEQ_RUN and NW_SEQ_DATA.
# usage: train_seq.sh [--resume]   --resume uses the data already built and goes on from last.pt
set -euo pipefail
source "$(dirname "$0")/common.sh"

olimpic="$RAW/olimpic"
clean_dev="$NW_DATA/olimpic_dev_clean.txt"

run="${NW_SEQ_RUN:-seq512}"
data="${NW_SEQ_DATA:-$NW_DATA}"
crops="$data/seq2"
written="$data/olimpic_written"
grandstaff="$data/grandstaff_written"
if [[ "${1:-}" != --resume ]]; then
  "$(dirname "$0")/build_seq_data.sh"
fi
# PDMX holds copies of some OLiMPiC songs: monitor on the dev songs without one.
(cd "$ROOT/eval" && "$PY" olimpic_benchmark.py list --split dev > "$clean_dev")
(cd "$ROOT/train" && "$PY" train_seq.py --data "$written:train:2" "$grandstaff:train" \
  "$crops/pdmx:train" "$crops/beams:train" "$crops/round:train" "$crops/marks:train" "$crops/musescore:train" --dev "$olimpic/olimpic-1.0-scanned:dev" --dev-samples "$clean_dev" --dev-count 615 \
  --out "$NW_RUNS/$run" --height 512 --batch 10 --lr 3e-4 --warmup 4800 --steps "$NW_SEQ_STEPS" \
  --workers "$NW_TRAIN_WORKERS") 2>&1 | tee -a "$NW_RUNS/$run.log"

# seq_predict's SER covers all dev crops, including songs with a PDMX copy; the note F1
# below counts only the clean ones.
predictions="$NW_RUNS/$run/benchmark/olimpic-systems"   # compare: scorecard.py $NW_RUNS/$run/benchmark $NW_RUNS/benchmark
(cd "$ROOT/eval" && "$PY" seq_predict.py "$NW_RUNS/$run/best.pt" --split dev --out "$predictions" \
  && "$PY" olimpic_benchmark.py systems --predictions "$predictions" --out "$predictions") | tee "$NW_RUNS/${run}_benchmark.txt"
