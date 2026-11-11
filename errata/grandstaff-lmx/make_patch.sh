#!/usr/bin/env bash
# Makes this folder's lmx.patch.gz and systems.tsv.gz, and WORK_DIR/musicxml.patch.gz (28 MB, too
# big for the repository: MuseScore lays each mended system out again), from GrandStaff-LMX
# (2024-02-12) with GrandStaff extracted in it, as scripts/fetch_training_data.sh lays them out.
# Needs the training environment (scripts/setup_env.sh) and MuseScore 3.6.2 (fetch_training_data.sh).
# usage: errata/grandstaff-lmx/make_patch.sh GRANDSTAFF_LMX_ROOT WORK_DIR
set -euo pipefail
source "$(dirname "$0")/../../scripts/common.sh"
here=$(cd "$(dirname "$0")" && pwd)
root=$(realpath "$1")
work=$(realpath -m "$2")
mkdir -p "$work"
(cd "$ROOT/train" && with_display "$PY" rebuild_grandstaff_lmx.py "$root" "$work/rebuilt")
patch_of() {  # patch_of EXT: the unified diff of every mended system's .EXT
  awk -F'\t' '$3 == "mended" {print $1}' "$work/rebuilt/systems.tsv" | sort | while read -r name; do
    diff -u --label "a/$name.$1" --label "b/$name.$1" "$root/$name.$1" "$work/rebuilt/$name.$1" || true
  done | gzip -9n
}
patch_of lmx > "$here/lmx.patch.gz"
patch_of musicxml > "$work/musicxml.patch.gz"
sort "$work/rebuilt/systems.tsv" | gzip -9n > "$here/systems.tsv.gz"
