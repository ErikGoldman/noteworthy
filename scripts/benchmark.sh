#!/usr/bin/env bash
# Benchmarks the trained model on held-out PDMX scores (see EXPERIMENTS.md), and on
# Night Album when its PDF is present (it is not distributed).
set -euo pipefail
source "$(dirname "$0")/common.sh"

model="$NW_RUNS/finetune/model.onnx"
"$ROOT/eval/run_benchmarks.sh" "$model" "$NW_RUNS/benchmark" 3 "$(( $(nproc) / 3 ))" | tee "$NW_RUNS/benchmark.txt"
if [ -f "$ROOT/Night Album.pdf" ]; then
  "$ROOT/cpp/build/noteworthy" "$ROOT/Night Album.pdf" --model "$model" -o "$NW_RUNS/night_album.musicxml" 2>/dev/null
  (cd "$ROOT/eval" && "$PY" evaluate.py sleepless.gt "$NW_RUNS/night_album.musicxml") | tee -a "$NW_RUNS/benchmark.txt"
fi
