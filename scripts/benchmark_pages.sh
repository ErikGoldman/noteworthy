#!/usr/bin/env bash
# Benchmarks the full recognizer on whole pages, as docs/EVALUATION_PROTOCOL.md defines it: the
# segmentation network (train_segmenter.sh) finds the systems, the sequence model's final
# averaged weights (train_seq.sh; no checkpoint chosen on real scans) read them.
#
#   Polish   OMR-NED against the gold as published, Corrected and Print standard
#            (eval/gold_overrides.py), each with the authors' bar alignment and over the whole
#            page; the recognizers we compare against run on the same pages
#   Mutopia  note F1
#   OLiMPiC  note F1 on whole scanned pages
#
# usage: benchmark_pages.sh [dev|test]
# test is sealed: it runs once, for the final evaluation, and only its numbers are read.
# Result: $NW_RUNS/$NW_SEQ_RUN/pages-SPLIT/report.txt. Every system's pages are kept: a rerun
# reads only what is missing, and scores everything again.
set -euo pipefail
source "$(dirname "$0")/common.sh"

split=${1:-dev}
binary="$ROOT/cpp/build/noteworthy"
out="$NW_RUNS/$NW_SEQ_RUN/pages-$split"
polish="$RAW/polish_scores"
baselines="$NW_RUNS/polish"   # other recognizers' pages do not depend on our run
mkdir -p "$out"

# Gold and pages of the split only, so that nothing outside it is read or scored. Gold is
# rebuilt every time: it follows eval/polish_overrides.jsonl.
[ -d "$polish/gold_terminated" ] || (cd "$ROOT/eval" && "$MUSICDIFF_PY" unpack_polish.py "$polish/polish.parquet" "$polish")
rm -rf "${out:?}/gold" "${out:?}/pages" && mkdir -p "$out/gold/published" "$out/pages"
for standard in corrected print; do
  (cd "$ROOT/eval" && "$MUSICDIFF_PY" gold_overrides.py "$polish/gold_terminated" "$ROOT/eval/polish_overrides.jsonl" \
    "$out/gold/all_$standard" --standard "$standard")
  mkdir -p "$out/gold/$standard"
done
pages=()
while read -r half page; do
  [ "$half" = "$split" ] || continue
  ln -s "$polish/gold/$page.krn" "$out/gold/published/$page.krn"
  for standard in corrected print; do ln -s "$out/gold/all_$standard/$page.krn" "$out/gold/$standard/$page.krn"; done
  ln -s "$polish/pages/$page.png" "$out/pages/$page.png"
  pages+=("$out/pages/$page.png")
done < "$ROOT/eval/polish_split.txt"

(cd "$ROOT/eval" && "$PY" seq_pages.py "$SEQ_CHECKPOINT" pages --pages "$out/pages" --binary "$binary" --model "$SEGMENTER" \
  --out "$out/ours")
"$TRANSCODA_PY" "$ROOT/eval/transcoda_predict.py" "$RAW/transcoda/b529f8a" --out "$baselines/transcoda" --beams 3 "${pages[@]}"
"$ROOT/eval/copisteria_pages.sh" "$baselines/copisteria" "${pages[@]}"
"$ROOT/eval/homr_pages.sh" "$baselines/homr" "0-$(( $(nproc) - 1 ))" "${pages[@]}"
"$PY" "$ROOT/eval/audiveris_pages.py" "$RAW/audiveris/unpacked/opt/audiveris/bin/Audiveris" "$baselines/audiveris" "${pages[@]}"

# score SYSTEM PREDICTIONS FORMAT: every gold standard, both alignments
score() {
  local system=$1 predictions=$2 format=$3 standard alignment flags
  for standard in published corrected print; do
    for alignment in authors whole-page; do
      flags=()
      [ "$alignment" = whole-page ] && flags+=(--whole-page-alignment)
      [ "$standard" != published ] && flags+=(--ambiguous-marks --either-ottava)
      [ "$standard" = corrected ] && flags+=(--either-display)
      (cd "$ROOT/eval" && "$MUSICDIFF_PY" omr_ned.py "$predictions" "$out/gold/$standard" --pred-format "$format" \
        --out "$out/scores/$system.$standard.$alignment.json" --jobs "$NW_RENDER_WORKERS" "${flags[@]}") >/dev/null
    done
  done
}
mkdir -p "$out/scores"
score ours "$out/ours" musicxml
score transcoda "$baselines/transcoda" kern
score copisteria "$baselines/copisteria" musicxml
score homr "$baselines/homr" musicxml
score audiveris "$baselines/audiveris" musicxml

{
  echo "== Polish $split, OMR-NED (lower is better)"
  (cd "$ROOT/eval" && "$PY" polish_report.py "$out/scores" "$out/gold/corrected")
  echo "== Mutopia $split, note F1"
  (cd "$ROOT/eval" && "$PY" seq_pages.py "$SEQ_CHECKPOINT" mutopia --binary "$binary" --model "$SEGMENTER" \
    --out "$out/mutopia" --split "$split")
  echo "== OLiMPiC pages $split, note F1"
  (cd "$ROOT/eval" && "$PY" olimpic_benchmark.py pages --split "$split" --binary "$binary" --model "$SEGMENTER" \
    --out "$out/olimpic-layouts" --jobs 3 --threads $(( $(nproc) / 3 )) >/dev/null \
    && "$PY" seq_pages.py "$SEQ_CHECKPOINT" olimpic --layouts "$out/olimpic-layouts" --out "$out/olimpic" --split "$split")
} | tee "$out/report.txt"
