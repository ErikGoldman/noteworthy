#pragma once

#include <vector>

#include "chords.h"
#include "rational.h"

namespace omr {

struct MeasureItem {
  const RhythmItem* item = nullptr;
  int staff = 1;         // 1-based staff number within the system
  NoteValue value;       // as read in this measure: implied tuplets are added here
  Rational duration;     // value.duration(), except that a whole-measure rest lasts the measure
};

// The measure being solved: its duration and where its barlines are on the page.
struct MeasureSpan {
  Rational length;
  float x_start = 0;
  float x_end = 0;
};

struct Timing {
  Rational onset;
  int voice = 1;
};

// A fingering 3 reads like a tuplet number. A staff's tuplets are dropped when its notes
// do not fill the measure with them but fill it exactly without them.
void drop_false_tuplets(std::vector<MeasureItem>& items, Rational measure_length, double interline);

// Engravers mark the first few triplets of a passage and leave the "3" off the rest.
// An unmarked beam group of 3 or 6 notes of one plain value is made a triplet (or
// sextuplet) when the staff's notes overfill the measure and making all such groups of
// the staff tuplets fills it exactly. The exact fill keeps plain groups of three, as in
// 3/8 or after an eighth rest in 4/4, untouched.
void imply_tuplets(std::vector<MeasureItem>& items, Rational measure_length, double interline);

// Resolves when each item of one measure starts and which voice it belongs to.
//
// Onsets come from constraints, strongest first, each kept only if consistent with
// those already accepted:
//   1. consecutive items under one beam follow each other (also across staves),
//   2. items in the same vertical column start together,
//   3. consecutive items of one staff and voice side (stem direction, or a rest moved
//      up or down for its voice) follow each other.
// A constraint that would make a group span more than the measure is skipped.
// Groups left unconnected are placed by interpolating x against placed items and the
// barlines, then snapped to a multiple of the group's shortest duration.
std::vector<Timing> solve_measure(const std::vector<MeasureItem>& items, const MeasureSpan& span, double interline);

}  // namespace omr
