#!/usr/bin/env bash
# One staff at a time (EXPERIMENTS_STAFF_MODEL.md, (k)): every two-staff system of the staff
# experiments' sets taken apart into one crop and label per staff (train/build_one_staff.py), in
# OUT laid out as SYSTEMS is, so that staff_experiment.sh trains on it with NW_SEQ_DATA=OUT:
# staff_experiment.sh's mix, OLiMPiC scanned dev's systems with no copy in PDMX (train_seq.py's dev)
# and outer_notes.py's set. Positions in OUT/staff_positions. Each set's OUT/.../outcomes.json says
# how many systems split and why the others did not. Safe to run again (each set is rewritten
# whole, the same); a changed builder needs a fresh OUT.
# usage: build_onestaff_data.sh SYSTEMS POSITIONS OUT
#   e.g. build_onestaff_data.sh "$NW_DATA/seq516g_data" "$NW_DATA/seq516g_data/staff_positions_v2" \
#          "$NW_DATA/seq516g_onestaff"
set -euo pipefail
source "$(dirname "$0")/common.sh"

systems=$(realpath "$1") positions=$(realpath "$2") out=$(realpath -m "$3")
split() {
  (cd "$ROOT/train" && "$PY" build_one_staff.py "$@" --workers "$NW_RENDER_WORKERS")
}
for set in olimpic_written grandstaff_written_v2 seq2/pdmx seq2/beams seq2/round seq2/marks seq2/musescore \
           seq2/unmarked seq2/smashcima musicorpus; do
  split "$systems/$set" train "$positions" "$out/$set" "$out/staff_positions"
done
split "$RAW/olimpic/olimpic-1.0-scanned" dev "$positions" "$out/olimpic-1.0-scanned" "$out/staff_positions" \
  --names "$NW_DATA/olimpic_dev_clean.txt"
for source in synthetic scanned; do
  split "$NW_DATA/olimpic_written_dev/$source" dev "$NW_DATA/olimpic_written_dev/staff_positions" \
    "$out/olimpic_written_dev/$source" "$out/olimpic_written_dev/staff_positions"
done
