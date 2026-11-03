#pragma once

#include <vector>

#include "chords.h"
#include "rational.h"

namespace omr {

struct MeasureItem {
  const RhythmItem* item = nullptr;
  int staff = 1;         // 1-based staff number within the system
  Rational duration;     // as notated (whole-measure rests get the measure length)
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

// Resolves when each item of one measure starts and which voice it belongs to.
//
// Onsets come from constraints, strongest first, each kept only if consistent with
// those already accepted:
//   1. consecutive items under one beam follow each other (also across staves),
//   2. items in the same vertical column start together,
//   3. consecutive items of one staff and stem-direction voice follow each other.
// A constraint that would make a group span more than the measure is skipped.
// Groups left unconnected are placed by interpolating x against placed items and the
// barlines, then snapped to a multiple of the group's shortest duration.
std::vector<Timing> solve_measure(const std::vector<MeasureItem>& items, const MeasureSpan& span, double interline);

}  // namespace omr
