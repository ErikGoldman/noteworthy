#!/usr/bin/env bash
# Benchmarks the trained model (see EXPERIMENTS.md):
#   verovio/musescore-100  held-out PDMX scores, rendered
#   OLiMPiC dev            real IMSLP scans, as system crops and as full pages
#   mutopia dev            piano pieces beyond OLiMPiC's genre, engraved by LilyPond
#   handwritten dev        MUSCIMA++ pages (writer-dependent test set)
#   Night Album            when its PDF is present (it is not distributed)
# OLiMPiC test and the writer-independent MUSCIMA++ pages are held out like Night Album
# (--split test of olimpic_benchmark.py, handwritten_benchmark.py).
set -euo pipefail
source "$(dirname "$0")/common.sh"

model="$SEGMENTER"
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
echo "== mutopia dev" | tee -a "$report"
(cd "$ROOT/eval" && "$PY" mutopia_benchmark.py --binary "$binary" --model "$model" \
  --out "$NW_RUNS/benchmark/mutopia" --jobs "$jobs" --threads "$threads") | tee -a "$report"
echo "== handwritten dev" | tee -a "$report"
(cd "$ROOT/eval" && "$PY" handwritten_benchmark.py --binary "$binary" --model "$model" \
  --out "$NW_RUNS/benchmark/handwritten" --jobs "$jobs" --threads "$threads") | tee -a "$report"
if [ -f "$ROOT/Night Album.pdf" ]; then
  echo "== night album" | tee -a "$report"
  "$binary" "$ROOT/Night Album.pdf" --model "$model" -o "$NW_RUNS/night_album.musicxml" 2>/dev/null
  (cd "$ROOT/eval" && "$PY" evaluate.py sleepless.gt "$NW_RUNS/night_album.musicxml" \
    --results "$NW_RUNS/benchmark/night-album/results.json") | tee -a "$report"
fi
echo "== scorecard" | tee -a "$report"
(cd "$ROOT/eval" && "$PY" scorecard.py "$NW_RUNS/benchmark") | tee -a "$report"
