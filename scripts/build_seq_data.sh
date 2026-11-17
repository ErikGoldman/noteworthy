#!/usr/bin/env bash
# Builds the sequence model's training data in $NW_SEQ_DATA (default $NW_DATA): our PDMX system
# crops engraved by Verovio (build_seq_dataset.py) in four styles and by MuseScore
# (build_musescore_crops.py; without a display, on a virtual one), handwritten crops drawn by our
# Smashcima fork (build_smashcima_crops.py), real piano grand staves from MusiCorpus
# (build_musicorpus_crops.py), OLiMPiC synthetic relabelled with pitches as
# printed, and GrandStaff-LMX rebuilt with its sources' clefs, keys and meters
# (rebuild_grandstaff_lmx.py), time symbols and barlines as tokens; and every crop's staff positions
# (find_staff_positions.py), for StaffSeq2Seq.
# Safe to stop and run again: the crop builders skip scores already built, and the relabelled sets
# are rewritten whole. A changed builder needs a fresh data directory. Before training on new or
# changed crops, check them by eye: docs/TRAINING_DATA_CHECKS.md.
# usage: build_seq_data.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"

olimpic="$RAW/olimpic"
data="${NW_SEQ_DATA:-$NW_DATA}"
crops="$data/seq2"
build_crops() {
  local name=$1; shift
  (cd "$ROOT/train" && "$PY" build_seq_dataset.py --scores "$NW_DATA/render_scores.txt" --out "$crops/$name" \
    --workers "$NW_RENDER_WORKERS" "$@" 2>&1 | grep -v '^\[Warning\]')
}
make_score_list
build_crops pdmx
build_crops beams --systems-per-score 2 --thick-beams
build_crops round --systems-per-score 2 --thick-beams --round-heads
build_crops marks --systems-per-score 2 --thick-beams --marks
# Prints often number the first tuplets of a run and no more: systems with tuplets, numbers hidden.
build_crops unmarked --systems-per-score 2 --unmarked-tuplets
# v2: tuplet numbers drawn as the scores draw them (2026-11-17). The sets before it, olimpic_written
# and olimpic_written_dev, are built by this script's history (git log scripts/build_seq_data.sh).
(cd "$ROOT/train" && "$PY" build_olimpic_written.py "$olimpic/olimpic-1.0-synthetic" "$olimpic/lieder" "$data/olimpic_written_v2")
# OLiMPiC's dev systems, synthetic and scanned, labelled the same way: eval/outer_notes.py's set.
outer_dev="$NW_DATA/olimpic_written_dev_v2"
for source in synthetic scanned; do
  (cd "$ROOT/train" && "$PY" build_olimpic_written.py "$olimpic/olimpic-1.0-$source" "$olimpic/lieder" "$outer_dev/$source" dev)
done
(cd "$ROOT/train" && with_display "$PY" rebuild_grandstaff_lmx.py "$olimpic/grandstaff-lmx" "$data/grandstaff_rebuilt")
(cd "$ROOT/train" && "$PY" build_grandstaff_written.py "$olimpic/grandstaff-lmx" "$data/grandstaff_rebuilt" "$data/grandstaff_written")
(cd "$ROOT/train" && with_display "$PY" build_musescore_crops.py --scores "$NW_DATA/render_scores.txt" \
  --out "$crops/musescore")
(cd "$ROOT/train" && "$SMASHCIMA_PY" build_smashcima_crops.py --scores "$NW_DATA/render_scores.txt" \
  --out "$crops/smashcima" --systems-per-score 1 --workers "$NW_SMASHCIMA_WORKERS")
(cd "$ROOT/train" && "$PY" build_musicorpus_crops.py "$RAW/musicorpus/musicorpus_v1.0" --out "$data/musicorpus")
# Each crop's staff positions, for StaffSeq2Seq (train_seq.py --staves), and OLiMPiC scanned dev's.
(cd "$ROOT/train" && "$PY" find_staff_positions.py "$data/staff_positions" "$data/olimpic_written_v2:train" \
  "$data/grandstaff_written:train" "$crops/pdmx:train" "$crops/beams:train" "$crops/round:train" "$crops/marks:train" \
  "$crops/musescore:train" "$crops/unmarked:train" "$crops/smashcima:train" "$data/musicorpus:train" \
  "$olimpic/olimpic-1.0-scanned:dev")
(cd "$ROOT/train" && "$PY" find_staff_positions.py "$outer_dev/staff_positions" "$outer_dev/synthetic:dev" \
  "$outer_dev/scanned:dev")
