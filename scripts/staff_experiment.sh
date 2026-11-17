#!/usr/bin/env bash
# One short run of docs/history/EXPERIMENTS_STAFF_MODEL_PLAN.md: the race's mix (train_seq.sh's sets of
# $NW_SEQ_DATA, GrandStaff with no tie across an edge), 20,000 steps with the whole schedule,
# then its averaged weights scored on OLiMPiC dev's written labels (outer_notes.py, tuplet_bars.py)
# into $NW_RUNS/NAME/scores.txt. Polish dev is scored apart (it needs the C++ segmenter).
# The model's own options follow NAME, e.g.
#   staff_experiment.sh t0_staff --staves "$NW_SEQ_DATA/staff_positions_v2"
#   staff_experiment.sh t4_system --staff-samples-only "$NW_SEQ_DATA/staff_positions_v2" --height 512
# A run that stopped goes on from its last.pt; one already scored is left alone.
# NW_EXPERIMENT_STEPS sets the length (20,000), NW_UNMARKED_WEIGHT the weight of the crops whose
# tuplet numbers are hidden (1). NW_EXPERIMENT_DEV and NW_EXPERIMENT_DEV_SAMPLES set the dev crops
# and which of them are scored (OLiMPiC scanned, its 615 systems with no copy in PDMX), and
# NW_EXPERIMENT_OUTER_DEV outer_notes.py's set (OLiMPiC dev with pitches as printed): the one-staff
# runs take each from build_onestaff_data.sh's output.
# usage: staff_experiment.sh NAME TRAIN_SEQ_OPTIONS...
set -euo pipefail
source "$(dirname "$0")/common.sh"

name=$1; shift
steps=${NW_EXPERIMENT_STEPS:-20000}
unmarked_weight=${NW_UNMARKED_WEIGHT:-1}
data="${NW_SEQ_DATA:-$NW_DATA/seq516g_data}"
crops="$data/seq2"
dev="${NW_EXPERIMENT_DEV:-$RAW/olimpic/olimpic-1.0-scanned}"
dev_samples="${NW_EXPERIMENT_DEV_SAMPLES:-$NW_DATA/olimpic_dev_clean.txt}"
outer_dev="${NW_EXPERIMENT_OUTER_DEV:-$NW_DATA/olimpic_written_dev}"
# OLiMPiC's training labels: olimpic_written_v2 draws tuplet numbers as the scores do (TODO 3c (i)).
olimpic_written="${NW_EXPERIMENT_OLIMPIC:-$data/olimpic_written}"
run="$NW_RUNS/$name"
checkpoint="$run/average_$(printf '%06d' "$steps").pt"
[ -f "$run/scores.txt" ] && { log "$name: already scored"; exit 0; }
mkdir -p "$run"

if [ ! -f "$checkpoint" ]; then
  (cd "$ROOT/train" && "$PY" train_seq.py --data "$olimpic_written:train:2" "$data/grandstaff_written_v2:train" \
    "$crops/pdmx:train" "$crops/beams:train" "$crops/round:train" "$crops/marks:train" "$crops/musescore:train" \
    "$crops/unmarked:train:$unmarked_weight" "$crops/smashcima:train:2" "$data/musicorpus:train:30" \
    --dev "$dev:dev" --dev-samples "$dev_samples" --dev-count "$(wc -w < "$dev_samples")" \
    --out "$run" --batch 10 --lr 3e-4 --warmup 4800 --steps "$steps" --eval-every 5000 \
    --workers "$NW_TRAIN_WORKERS" --seed 0 "$@") 2>&1 | tee -a "$run/train.log"
fi

(cd "$ROOT/eval" && "$PY" outer_notes.py "$checkpoint" --data "$outer_dev" --out "$run/outer_readings.jsonl" > "$run/outer_notes.txt")
(cd "$ROOT/eval" && "$PY" tuplet_bars.py "$run/outer_readings.jsonl" > "$run/tuplet_bars.txt")
{ echo "== dev SER ($(wc -w < "$dev_samples") crops of $dev)"; grep "dev SER" "$run/train.log" | tail -1 || true
  cat "$run/outer_notes.txt" "$run/tuplet_bars.txt"; } > "$run/scores.partial"
mv "$run/scores.partial" "$run/scores.txt"
log "$name: scored"; cat "$run/scores.txt"
