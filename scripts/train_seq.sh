#!/usr/bin/env bash
# Trains the system-crop sequence model (train/seq) and scores it on OLiMPiC scanned dev
# against Zeus. Training data: OLiMPiC synthetic relabelled with pitches as printed,
# GrandStaff-LMX and our own PDMX system crops (build_seq_dataset.py) in four styles.
# Result: $NW_RUNS/seq512/best.pt
set -euo pipefail
source "$(dirname "$0")/common.sh"

olimpic="$RAW/olimpic"
clean_dev="$NW_DATA/olimpic_dev_clean.txt"

crops="$NW_DATA/seq2"
build_crops() {
  local name=$1; shift
  (cd "$ROOT/train" && "$PY" build_seq_dataset.py --scores "$NW_DATA/render_scores.txt" --out "$crops/$name" \
    --workers "$NW_RENDER_WORKERS" "$@" 2>&1 | grep -v '^\[Warning\]')
}
build_crops pdmx
build_crops beams --systems-per-score 2 --thick-beams
build_crops round --systems-per-score 2 --thick-beams --round-heads
build_crops marks --systems-per-score 2 --thick-beams --marks
(cd "$ROOT/train" && "$PY" build_olimpic_written.py "$olimpic/olimpic-1.0-synthetic" "$NW_DATA/olimpic_written")
# PDMX holds copies of some OLiMPiC songs: monitor on the dev songs without one.
(cd "$ROOT/eval" && "$PY" olimpic_benchmark.py list --split dev > "$clean_dev")
(cd "$ROOT/train" && "$PY" train_seq.py --data "$NW_DATA/olimpic_written:train:2" "$olimpic/grandstaff-lmx:train" \
  "$crops/pdmx:train" "$crops/beams:train" "$crops/round:train" "$crops/marks:train" --dev "$olimpic/olimpic-1.0-scanned:dev" --dev-samples "$clean_dev" --dev-count 615 \
  --out "$NW_RUNS/seq512" --height 512 --batch 10 --lr 3e-4 --warmup 4800 --steps "$NW_SEQ_STEPS" \
  --workers "$NW_TRAIN_WORKERS") 2>&1 | tee -a "$NW_RUNS/seq512.log"

# seq_predict's SER covers all dev crops, including songs with a PDMX copy; the note F1
# below counts only the clean ones.
predictions="$NW_RUNS/seq512/benchmark/olimpic-systems"   # compare: scorecard.py $NW_RUNS/seq512/benchmark $NW_RUNS/benchmark
(cd "$ROOT/eval" && "$PY" seq_predict.py "$NW_RUNS/seq512/best.pt" --split dev --out "$predictions" \
  && "$PY" olimpic_benchmark.py systems --predictions "$predictions" --out "$predictions") | tee "$NW_RUNS/seq512_benchmark.txt"
