#!/usr/bin/env bash
# Compares our recognizer with Zeus, the OLiMPiC authors' end-to-end model, on the
# OLiMPiC scanned test split (held out: numbers only, never its errors).
#
# Zeus's published score is TEDn on all 1,493 test systems; recomputing it here checks
# the TEDn pipeline. Our model trained on PDMX, which holds copies of many Lieder songs,
# so the head-to-head numbers are on the songs with no copy in our training sources
# (olimpic_benchmark.py). Both are scored on the same system crops with TEDn and with our
# note F1, which counts only what we recognize (pitch, onset, duration), where TEDn also
# counts stems, beams, slurs and articulations, which we do not output.
#
# TEDn costs seconds to minutes per system: about two hours on 32 cores.
set -euo pipefail
source "$(dirname "$0")/common.sh"

model="$SEGMENTER"
binary="$ROOT/cpp/build/noteworthy"
ours="$NW_RUNS/benchmark/olimpic-test-systems"
zeus="$NW_RUNS/zeus/test"
report="$NW_RUNS/zeus_comparison.txt"
jobs=3
threads=$(( $(nproc) / jobs ))
cd "$ROOT/eval"

echo "== ours, note F1" | tee "$report"
"$PY" olimpic_benchmark.py systems --split test --binary "$binary" --model "$model" --out "$ours" \
  --jobs "$jobs" --threads "$threads" | tee -a "$report"

[ -f "$zeus/work/run/scanned-test.predicted.lmx" ] || "$ZEUS_PY" zeus_comparison.py predict --split test --out "$zeus"
echo "== zeus, note F1" | tee -a "$report"
"$PY" olimpic_benchmark.py systems --split test --predictions "$zeus" --out "$zeus" | tee -a "$report"

tedn() { "$ZEUS_PY" zeus_comparison.py tedn "$1" --samples "$2"; }
echo "== TEDn-lmx (the paper reports 14.85% for Zeus on all test systems)" | tee -a "$report"
echo "zeus, all test systems: $(tedn "$zeus" "$NW_DATA/raw/olimpic/olimpic-1.0-scanned/samples.test.txt")" | tee -a "$report"
echo "zeus, scored songs:     $(tedn "$zeus" "$ours/scored_samples.txt")" | tee -a "$report"
echo "ours, scored songs:     $(tedn "$ours" "$ours/scored_samples.txt")" | tee -a "$report"
