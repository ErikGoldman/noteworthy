#!/usr/bin/env bash
# Trains the system-crop sequence model (train/seq) on the data build_seq_data.sh makes,
# monitoring it on OLiMPiC scanned dev. Result: $NW_RUNS/$NW_SEQ_RUN/: best.pt, and the
# averaged weights every 10,000 steps and at the last step ($SEQ_CHECKPOINT, what is scored). A run that stopped goes on from
# its last.pt; a new run takes its own NW_SEQ_RUN and NW_SEQ_DATA. benchmark_seq.sh scores it.
set -euo pipefail
source "$(dirname "$0")/common.sh"

olimpic="$RAW/olimpic"
clean_dev="$NW_DATA/olimpic_dev_clean.txt"

data="${NW_SEQ_DATA:-$NW_DATA}"
crops="$data/seq2"
written="$data/olimpic_written"
grandstaff="$data/grandstaff_written"
# Handwriting, about 8% of samples: drawn by Smashcima, and real (MusiCorpus: 53 grand staves,
# about 0.5%, each seen often; more would teach those pages rather than handwriting).
SMASHCIMA_WEIGHT=2
MUSICORPUS_WEIGHT=30
# PDMX holds copies of some OLiMPiC songs: monitor on the dev songs without one.
(cd "$ROOT/eval" && "$PY" olimpic_benchmark.py list --split dev > "$clean_dev")
(cd "$ROOT/train" && "$PY" train_seq.py --data "$written:train:2" "$grandstaff:train" \
  "$crops/pdmx:train" "$crops/beams:train" "$crops/round:train" "$crops/marks:train" "$crops/musescore:train" "$crops/unmarked:train" \
  "$crops/smashcima:train:$SMASHCIMA_WEIGHT" "$data/musicorpus:train:$MUSICORPUS_WEIGHT" --dev "$olimpic/olimpic-1.0-scanned:dev" --dev-samples "$clean_dev" --dev-count 615 \
  --out "$NW_RUNS/$NW_SEQ_RUN" --height 512 --batch 10 --lr 3e-4 --warmup 4800 --steps "$NW_SEQ_STEPS" \
  --workers "$NW_TRAIN_WORKERS") 2>&1 | tee -a "$NW_RUNS/$NW_SEQ_RUN.log"
