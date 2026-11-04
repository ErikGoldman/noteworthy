#!/usr/bin/env bash
# Trains the segmentation network in two stages and exports it for the C++ recognizer:
#   base      from scratch on uniformly sampled crops
#   finetune  half the crops centred on a class present on the page, so rare symbols
#             (time-signature digits, tuplets) get enough examples; lower learning rate
# Result: $NW_RUNS/finetune/model.onnx
set -euo pipefail
source "$(dirname "$0")/common.sh"

MUSCIMA_WEIGHT=20  # handwritten pages are few; each counts this many times

# train_stage NAME STEPS [train.py options]. An interrupted stage resumes from its last checkpoint.
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

export_model() {
  "$PY" "$ROOT/train/export_onnx.py" "$NW_RUNS/finetune/last.pt" "$NW_RUNS/finetune/model.onnx"
}

mkdir -p "$NW_RUNS"
step train-base train_stage base "$NW_BASE_STEPS"
step train-finetune train_stage finetune "$NW_FINETUNE_STEPS" --init "$NW_RUNS/base/last.pt" --lr 5e-4 --balanced-crops 0.5
step export-model export_model
