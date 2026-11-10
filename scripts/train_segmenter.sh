#!/usr/bin/env bash
# Trains the segmentation network in two stages and exports it for the C++ recognizer:
#   base      from scratch on uniformly sampled crops
#   finetune  half the crops centred on a class present on the page, so rare symbols
#             (time-signature digits, tuplets) get enough examples; lower learning rate
# Result: $NW_RUNS/finetune/model.onnx ($SEGMENTER)
set -euo pipefail
source "$(dirname "$0")/common.sh"

# Handwritten pages are few (100 MUSCIMA++ pages); each counts this many times, so that
# they are about 7% of the 12,000-page mix: roughly their share of real-world scores.
MUSCIMA_WEIGHT=9

# train_stage NAME STEPS [train.py options]. A stage with a checkpoint resumes from it;
# train.py returns at once when that checkpoint already reached STEPS.
train_stage() {
  local name=$1 steps=$2
  shift 2
  local out="$NW_RUNS/$name"
  local start=("$@")
  [ -f "$out/last.pt" ] && start=(--resume "$out/last.pt")
  (cd "$ROOT/train" && "$PY" train.py --data "$NW_DATA/synth" "$NW_DATA/muscima:$MUSCIMA_WEIGHT" --out "$out" \
    --steps "$steps" --batch 16 --workers "$NW_TRAIN_WORKERS" --eval-every "$NW_EVAL_EVERY" "${start[@]}") \
    2>&1 | tee -a "$NW_RUNS/$name.log"
}

mkdir -p "$NW_RUNS"
train_stage base "$NW_BASE_STEPS"
train_stage finetune "$NW_FINETUNE_STEPS" --init "$NW_RUNS/base/last.pt" --lr 5e-4 --balanced-crops 0.5
model="$SEGMENTER"
if [ ! "$model" -nt "$NW_RUNS/finetune/last.pt" ]; then
  "$PY" "$ROOT/train/export_onnx.py" "$NW_RUNS/finetune/last.pt" "$model"
fi
log "model ready: $model"
