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
# tuplet numbers are hidden (1), NW_EXPERIMENT_SMASHCIMA_WEIGHT and NW_EXPERIMENT_MUSICORPUS_WEIGHT
# those of the handwritten crops (2) and MusiCorpus's scans (30); 0 leaves a set out (TODO 3d). NW_EXPERIMENT_DEV and NW_EXPERIMENT_DEV_SAMPLES set the dev crops
# and which of them are scored (OLiMPiC scanned, its 615 systems with no copy in PDMX), and
# NW_EXPERIMENT_OUTER_DEV outer_notes.py's set (OLiMPiC dev with pitches as printed): the one-staff
# runs take each from build_onestaff_data.sh's output. NW_EXPERIMENT_OLIMPIC and
# NW_EXPERIMENT_GRANDSTAFF choose the labels of those two training sets. NW_EXPERIMENT_ENGRAVED names
# the Verovio and MuseScore sets to train on in place of pdmx, beams, round, marks and musescore:
# engraver_hiding or tuplet_looks_v2 takes build_tuplet_experiments.sh's <set>_engraver_hiding or
# <set>_tuplet_looks_v2 (TODO 3c). NW_EXPERIMENT_EXTRA_DATA adds train_seq.py --data entries to the
# mix, as "features:train features_musescore:train" for the synthetic pieces (sets in $crops).
# usage: staff_experiment.sh NAME TRAIN_SEQ_OPTIONS...
set -euo pipefail
source "$(dirname "$0")/common.sh"

name=$1; shift
steps=${NW_EXPERIMENT_STEPS:-20000}
unmarked_weight=${NW_UNMARKED_WEIGHT:-1}
smashcima_weight=${NW_EXPERIMENT_SMASHCIMA_WEIGHT:-2}
musicorpus_weight=${NW_EXPERIMENT_MUSICORPUS_WEIGHT:-30}
data="${NW_SEQ_DATA:-$NW_DATA/seq516g_data}"
crops="$data/seq2"
dev="${NW_EXPERIMENT_DEV:-$RAW/olimpic/olimpic-1.0-scanned}"
dev_samples="${NW_EXPERIMENT_DEV_SAMPLES:-$NW_DATA/olimpic_dev_clean.txt}"
outer_dev="${NW_EXPERIMENT_OUTER_DEV:-$NW_DATA/olimpic_written_dev}"
# OLiMPiC's training labels: olimpic_written_v2 draws tuplet numbers as the scores do (TODO 3c (i)).
olimpic_written="${NW_EXPERIMENT_OLIMPIC:-$data/olimpic_written}"
# GrandStaff's: grandstaff_release has the release's labels, its cleaning left out (TODO 3d).
grandstaff="${NW_EXPERIMENT_GRANDSTAFF:-$data/grandstaff_written_v2}"
engraved_sets=(pdmx beams round marks musescore)
if [ -n "${NW_EXPERIMENT_ENGRAVED:-}" ]; then engraved_sets=("${engraved_sets[@]/%/_$NW_EXPERIMENT_ENGRAVED}"); fi
engraved=("${engraved_sets[@]/#/$crops/}")
read -r -a extra_data <<< "${NW_EXPERIMENT_EXTRA_DATA:-}"
run="$NW_RUNS/$name"
checkpoint="$run/average_$(printf '%06d' "$steps").pt"
[ -f "$run/scores.txt" ] && { log "$name: already scored"; exit 0; }
mkdir -p "$run"

if [ ! -f "$checkpoint" ]; then
  (cd "$ROOT/train" && "$PY" train_seq.py --data "$olimpic_written:train:2" "$grandstaff:train" \
    "${engraved[@]/%/:train}" "${extra_data[@]/#/$crops/}" \
    "$crops/unmarked:train:$unmarked_weight" "$crops/smashcima:train:$smashcima_weight" "$data/musicorpus:train:$musicorpus_weight" \
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
