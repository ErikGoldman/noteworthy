#!/usr/bin/env bash
# Reads a score with the models train.sh made: a PDF or page images in, MusicXML out. The
# segmentation network finds the systems, the sequence model reads each one. Piano music (two
# staves per system) is what it was trained for.
#
# usage: ./recognize.sh SCORE.pdf -o SCORE.musicxml
#        ./recognize.sh PAGE1.png PAGE2.png ... -o SCORE.musicxml
set -euo pipefail
source "$(dirname "$0")/scripts/common.sh"

inputs=() out=""
while [ $# -gt 0 ]; do
  case "$1" in
    -o) out=$(realpath -m "$2"); shift 2 ;;
    *) inputs+=("$(realpath "$1")"); shift ;;
  esac
done
if [ -z "$out" ] || [ ${#inputs[@]} -eq 0 ]; then
  sed -n 's/^# usage: /usage: /p; s/^#        /       /p' "$0" >&2
  exit 1
fi
(cd "$ROOT/eval" && "$PY" seq_pages.py "$SEQ_CHECKPOINT" score --inputs "${inputs[@]}" \
  --binary "$ROOT/cpp/build/noteworthy" --model "$SEGMENTER" --out "$out")
