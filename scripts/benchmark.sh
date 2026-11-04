#!/usr/bin/env bash
# Benchmarks the trained model (see EXPERIMENTS.md):
#   verovio/musescore-100  held-out PDMX scores, rendered
#   OLiMPiC dev            real IMSLP scans, as system crops and as full pages
#   Night Album            when its PDF is present (it is not distributed)
# OLiMPiC test is held out like Night Album: eval/olimpic_benchmark.py --split test.
set -euo pipefail
source "$(dirname "$0")/common.sh"

model="$NW_RUNS/finetune/model.onnx"
binary="$ROOT/cpp/build/noteworthy"
report="$NW_RUNS/benchmark.txt"
jobs=3
threads=$(( $(nproc) / jobs ))

"$ROOT/eval/run_benchmarks.sh" "$model" "$NW_RUNS/benchmark" "$jobs" "$threads" | tee "$report"
for mode in systems pages; do
  echo "== olimpic dev $mode" | tee -a "$report"
  (cd "$ROOT/eval" && "$PY" olimpic_benchmark.py "$mode" --binary "$binary" --model "$model" \
    --out "$NW_RUNS/benchmark/olimpic-$mode" --jobs "$jobs" --threads "$threads") | tee -a "$report"
done
if [ -f "$ROOT/Night Album.pdf" ]; then
  echo "== night album" | tee -a "$report"
  "$binary" "$ROOT/Night Album.pdf" --model "$model" -o "$NW_RUNS/night_album.musicxml" 2>/dev/null
  (cd "$ROOT/eval" && "$PY" evaluate.py sleepless.gt "$NW_RUNS/night_album.musicxml") | tee -a "$report"
fi
