#!/usr/bin/env bash
# Turns the raw sources into training pages: synthetic renders ($NW_DATA/synth) and
# converted handwritten pages ($NW_DATA/muscima).
set -euo pipefail
source "$(dirname "$0")/common.sh"

make_score_list

# build_dataset.py renders only the pages not yet on disk.
(cd "$ROOT/train" && "$PY" build_dataset.py --scores "$NW_DATA/render_scores.txt" --out "$NW_DATA/synth" \
  --pages "$NW_PAGES" --workers "$NW_RENDER_WORKERS")

if [ ! -d "$NW_DATA/muscima" ]; then
  rm -rf "$NW_DATA/muscima.partial"
  (cd "$ROOT/train" && "$PY" convert_muscima.py "$RAW/muscima" "$NW_DATA/muscima.partial")
  mv "$NW_DATA/muscima.partial" "$NW_DATA/muscima"
fi
log "training pages ready"
