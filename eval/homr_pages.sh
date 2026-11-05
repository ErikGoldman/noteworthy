#!/bin/bash
# homr on page images, with its default settings, for comparison.
#
# usage: homr_pages.sh OUT_DIR CPU_LIST PAGE.png [...]
#
# homr writes beside its input, so each page is copied into OUT_DIR. A page already read is
# skipped; one that fails or passes PAGE_TIMEOUT_S leaves no MusicXML and scores as failed.
# homr fills every core it may use, so CPU_LIST keeps it off those a training run needs.
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PAGE_TIMEOUT_S=900
out=$1 cpus=$2
shift 2
mkdir -p "$out"
cd "$out" || exit 1
for page in "$@"; do
  name=$(basename "$page" .png)
  [ -e "$name.musicxml" ] && continue
  cp "$page" "$name.png"
  taskset -c "$cpus" timeout "$PAGE_TIMEOUT_S" "$ROOT/.venv-homr/bin/homr" "$name.png" > "$name.log" 2>&1 \
    || echo "$name failed: exit $?"
  rm -f "$name.png" "${name}_teaser.png"
done
