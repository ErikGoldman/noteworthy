#!/usr/bin/env bash
# Runs the held-out benchmarks for one model and prints F1 plus the error breakdown.
#
# usage: eval/run_benchmarks.sh MODEL.onnx RENDER_CACHE_DIR [JOBS] [THREADS_PER_JOB]
#
# RENDER_CACHE_DIR keeps rendered pages between runs (one subdirectory per renderer),
# so repeated runs only redo recognition.
set -euo pipefail

model=$(realpath "$1")
cache=$(realpath -m "$2")
jobs=${3:-3}
threads=${4:-0}
root=$(cd "$(dirname "$0")/.." && pwd)
python="$root/.venv/bin/python"

cd "$root/eval"
for renderer in verovio musescore; do
  out="$cache/$renderer"
  "$python" synthetic_benchmark.py --renderer "$renderer" --scores ../data/pdmx_piano.txt \
    --binary ../cpp/build/noteworthy --model "$model" --out "$out" --count 100 --jobs "$jobs" --threads "$threads" \
    > "$out.log" 2>&1 || { echo "$renderer benchmark failed, see $out.log"; exit 1; }
  echo "== $renderer: $(grep 'overall F1' "$out.log")"
  "$python" diagnose.py "$out" ../data/pdmx_piano.txt | head -13
done
