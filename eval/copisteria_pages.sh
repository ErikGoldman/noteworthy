#!/bin/bash
# copisteria with copista-28m on page images, with its default settings, for comparison.
#
# usage: copisteria_pages.sh OUT_DIR PAGE.png [...]
#
# Writes OUT_DIR/<page>.musicxml. It caches detections beside its input, so pages are copied
# to OUT_DIR first, and finds models/ from its repository's root, so it runs there. Each page
# gets its own process: one that fails or passes PAGE_TIMEOUT_S leaves no MusicXML and scores
# as failed, and the others still run.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PAGE_TIMEOUT_S=600
JOBS=4
out=$(realpath -m "$1")
shift
mkdir -p "$out"
cp "$@" "$out"
cd "${NW_DATA:-$ROOT/data}/raw/copisteria/src" || exit 1
for page in "$@"; do
  [ -e "$out/$(basename "$page" .png).musicxml" ] || echo "$out/$(basename "$page")"
done | xargs -P "$JOBS" -I{} timeout "$PAGE_TIMEOUT_S" "$ROOT/.venv-copisteria/bin/python" -m copisteria.pipeline {} --out "$out"
rm -f "$out"/*.png
