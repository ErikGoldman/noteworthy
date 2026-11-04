#!/usr/bin/env bash
# Turns the raw sources into training pages: synthetic renders ($NW_DATA/synth) and
# converted handwritten pages ($NW_DATA/muscima).
set -euo pipefail
source "$(dirname "$0")/common.sh"

OTHER_SCORES=4000  # non-piano scores mixed in for layout variety

list_scores() {
  (cd "$ROOT/train" && "$PY" select_pdmx.py "$RAW/pdmx" "$NW_DATA/pdmx_piano.txt" "$NW_DATA/pdmx_other.txt")
  find "$RAW/asap" -name xml_score.musicxml | sort > "$NW_DATA/asap_scores.txt"
  cat "$NW_DATA/pdmx_piano.txt" "$NW_DATA/asap_scores.txt" <(head -n "$OTHER_SCORES" "$NW_DATA/pdmx_other.txt") \
    > "$NW_DATA/render_scores.txt"
}

# build_dataset.py skips pages already rendered, so an interrupted render resumes.
render_pages() {
  (cd "$ROOT/train" && "$PY" build_dataset.py --scores "$NW_DATA/render_scores.txt" --out "$NW_DATA/synth" \
    --pages "$NW_PAGES" --workers "$NW_RENDER_WORKERS")
}

convert_muscima() {
  rm -rf "$NW_DATA/muscima"
  (cd "$ROOT/train" && "$PY" convert_muscima.py "$RAW/muscima" "$NW_DATA/muscima")
}

step score-lists list_scores
step render-pages render_pages
step convert-muscima convert_muscima
