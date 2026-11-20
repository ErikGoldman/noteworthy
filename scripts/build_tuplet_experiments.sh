#!/usr/bin/env bash
# Training data for TODO step 3c's tuplet runs and the synthetic-bars run, beside the baseline's
# (build_seq_data.sh) in $NW_SEQ_DATA (default $NW_DATA/seq516g_data), each set under its own
# name: train_seq.py finds a set's staff positions by its directory's name, so a set rebuilt under
# the baseline's name would take the baseline's positions.
#   - <set>_engraver_hiding: the Verovio and MuseScore sets with tuplet numbers hidden as engravers
#     do (--engraver-hiding, seq.tuplet_display), not by the blind rule; `unmarked` stays.
#   - <set>_tuplet_looks_v2: the same sets with printed numbers set in varied ways (--varied-tuplet-looks):
#     above or below, bracketed or not, 3 or 3:2, far from their notes, small, arcs. (<set>_tuplet_looks,
#     2026-11-19, set some far numbers off the page, drawing none where the label printed one: not used.)
#   - features, features_musescore: synthetic_bars.py's pieces (SYNTHETIC_SCORES of them), each in
#     one feature (tuplets, two voices, dense chords, or none), two systems a piece from each
#     engraver, about a tenth of the mix. (bars and bars_musescore, of 2026-11-19, were its first
#     design, a tuplet in 96% of bars: not used.)
# Each set's options are the baseline's, so a set differs from its baseline by its option alone
# (checked 2026-11-19: the builders reproduce the 2026-11-13 sets file for file). Staff positions
# go to $data/staff_positions_v2, where the runs take them. Check crops by eye before training
# (docs/TRAINING_DATA_CHECKS.md). Safe to stop and run again: the builders skip finished scores.
# usage: build_tuplet_experiments.sh [engraver_hiding] [tuplet_looks] [features]   (default: all three)
set -euo pipefail
source "$(dirname "$0")/common.sh"

data="${NW_SEQ_DATA:-$NW_DATA/seq516g_data}"
crops="$data/seq2"
positions="$data/staff_positions_v2"
SYNTHETIC_SCORES=8000

verovio() {
  local scores=$1 out=$2; shift 2
  (cd "$ROOT/train" && "$PY" build_seq_dataset.py --scores "$scores" --out "$crops/$out" \
    --workers "$NW_RENDER_WORKERS" "$@" 2>&1 | grep -v '^\[Warning\]')
}
musescore() {
  local scores=$1 out=$2; shift 2
  (cd "$ROOT/train" && with_display "$PY" build_musescore_crops.py --scores "$scores" --out "$crops/$out" "$@")
}
staff_positions() {
  (cd "$ROOT/train" && "$PY" find_staff_positions.py "$positions" "${@/%/:train}")
}

# The baseline's four Verovio sets and its MuseScore set (build_seq_data.sh), with one option added.
engraved_sets() {
  local suffix=$1 option=$2 scores="$NW_DATA/render_scores.txt"
  verovio "$scores" "pdmx_$suffix" "$option"
  verovio "$scores" "beams_$suffix" --systems-per-score 2 --thick-beams "$option"
  verovio "$scores" "round_$suffix" --systems-per-score 2 --thick-beams --round-heads "$option"
  verovio "$scores" "marks_$suffix" --systems-per-score 2 --thick-beams --marks "$option"
  musescore "$scores" "musescore_$suffix" "$option"
  staff_positions "$crops/pdmx_$suffix" "$crops/beams_$suffix" "$crops/round_$suffix" "$crops/marks_$suffix" \
    "$crops/musescore_$suffix"
}

synthetic_sets() {
  local scores="$data/synthetic_features/scores.txt"
  (cd "$ROOT/train" && "$PY" synthetic_bars.py --out "$data/synthetic_features" --scores "$SYNTHETIC_SCORES")
  verovio "$scores" features --systems-per-score 2
  musescore "$scores" features_musescore --systems-per-score 2
  staff_positions "$crops/features" "$crops/features_musescore"
}

experiments=("$@")
[ ${#experiments[@]} -gt 0 ] || experiments=(engraver_hiding tuplet_looks features)
for name in "${experiments[@]}"; do
  case $name in
    engraver_hiding) engraved_sets engraver_hiding --engraver-hiding ;;
    tuplet_looks) engraved_sets tuplet_looks_v2 --varied-tuplet-looks ;;
    features) synthetic_sets ;;
    *) echo "unknown experiment: $name" >&2; exit 2 ;;
  esac
done
