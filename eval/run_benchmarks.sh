#!/usr/bin/env bash
# Runs the held-out benchmarks for one model and prints F1 plus the error breakdown.
#
# usage: eval/run_benchmarks.sh MODEL.onnx RENDER_CACHE_DIR [JOBS] [THREADS_PER_JOB]
#
# RENDER_CACHE_DIR keeps rendered pages between runs (one subdirectory per renderer),
# so repeated runs only redo recognition. The MuseScore benchmark runs only where
# MuseScore 4 is found (see musescore.py).
set -euo pipefail
source "$(dirname "$0")/../scripts/common.sh"

model=$(realpath "$1")
cache=$(realpath -m "$2")
jobs=${3:-3}
threads=${4:-0}
scores="$NW_DATA/pdmx_piano.txt"

cd "$ROOT/eval"
renderers=(verovio)
if "$PY" -c "import sys, musescore; sys.exit(musescore.find() is None)"; then
  renderers+=(musescore)
else
  echo "MuseScore 4 not found (set MUSESCORE or put it on PATH): skipping the musescore benchmark"
fi

mkdir -p "$cache"
for renderer in "${renderers[@]}"; do
  out="$cache/$renderer"
  "$PY" synthetic_benchmark.py --renderer "$renderer" --scores "$scores" \
    --binary "$ROOT/cpp/build/noteworthy" --model "$model" --out "$out" --count "$NW_BENCH_COUNT" --jobs "$jobs" --threads "$threads" \
    > "$out.log" 2>&1 || { echo "$renderer benchmark failed, see $out.log"; exit 1; }
  echo "== $renderer: $(grep 'overall F1' "$out.log")"
  "$PY" diagnose.py "$out" "$scores" --count "$NW_BENCH_COUNT" | head -13
done
