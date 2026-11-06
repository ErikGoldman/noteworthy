#pragma once

#include <optional>
#include <vector>

#include "rational.h"

namespace omr {

enum class ClefSign { G, F, C };

struct Clef {
  ClefSign sign = ClefSign::G;
  int line = 2;  // staff line the clef sits on, 1 = bottom

  // Diatonic step (C4 = 0, D4 = 1, ...) of the bottom staff line.
  int bottom_line_step() const;
};

struct Pitch {
  int diatonic = 0;  // diatonic step, C4 = 0
  int alter = 0;     // semitones: -1 flat, +1 sharp

  char step_name() const;  // 'C'..'B'
  int octave() const;
  int midi() const;
};

struct TimeSignature {
  int beats = 4;
  int beat_type = 4;
  Rational measure_length() const { return Rational(4 * beats, beat_type); }
};

// Notated duration: a base value (1 = whole, 2 = half, 4 = quarter, ...) plus dots,
// played `tuplet_actual` in the time of `tuplet_normal` (3 in 2 for a triplet).
struct NoteValue {
  int base = 4;
  int dots = 0;
  int tuplet_actual = 1;
  int tuplet_normal = 1;
  bool in_tuplet() const { return tuplet_actual != tuplet_normal; }
  Rational duration() const;  // in quarter notes
};

// A chord or rest occupying one rhythmic slot in one voice.
struct Event {
  Rational onset;           // quarter notes from the measure start
  Rational duration;        // usually value.duration(); a whole-measure rest lasts the measure
  NoteValue value;          // as drawn
  int staff = 1;            // 1-based, the staff it is drawn on
  int voice = 1;
  std::vector<Pitch> pitches;  // empty => rest
  std::vector<bool> tie_start; // parallel to pitches

  bool is_rest() const { return pitches.empty(); }
};

struct ClefChange {
  int staff = 1;
  Clef clef;
  Rational onset;  // 0 = at the measure start; the measure length = just before its end barline
};

// Vertical extent of a staff on the input page, in input pixels (top line to bottom line,
// at the middle of the staff).
struct StaffPlacement {
  float top = 0;
  float bottom = 0;
};

// Where a measure sits in the source's layout, for tools that relate output to pages.
struct LayoutBreak {
  bool new_system = false;
  std::optional<int> new_page;  // 1-based page number of the input, on a page's first measure
  int page = 0;                 // 1-based page number of the input
  double scale = 1;             // from input pixels to the page as the recognizer read it
  std::vector<StaffPlacement> staves;  // on a system's first measure: its staves, by staff number - 1
};

struct Measure {
  LayoutBreak layout;
  std::vector<Event> events;
  // Attributes taking effect at the start of this measure (only when they change).
  std::optional<int> key_fifths;
  std::optional<TimeSignature> time;
  std::vector<ClefChange> clefs;  // in onset order
};

struct Score {
  int staves = 2;
  std::vector<Measure> measures;
};

}  // namespace omr
