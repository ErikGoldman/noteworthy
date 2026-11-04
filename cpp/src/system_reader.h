#pragma once

#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "chords.h"
#include "layout.h"
#include "score.h"
#include "staff.h"
#include "symbols.h"

namespace omr {

struct ClefAt {
  float x;
  int staff_number;
  Clef clef;
};

// Everything the measures of one system need to know about their surroundings: clefs,
// key and time signatures, accidentals and ottava lines, located by measure.
class SystemReader {
 public:
  SystemReader(const std::vector<Staff>& staves, const PageSymbols& symbols, const System& sys,
               const std::vector<RhythmItem>& items, double interline);

  // Clef in force at x on a staff, given the clef carried in from earlier systems.
  Clef clef_at(int staff_number, float x, const Clef& carried) const;

  // Clefs drawn within measure k, left to right.
  std::vector<ClefAt> clefs_in_measure(int k) const;

  // Key signature drawn at the start of measure k. Every system restates its key, so a
  // system start counts its key accidentals (C when none); counting tolerates glyphs
  // whose reference point is off a step in other engravers' fonts. Mid-system,
  // accidentals right after a barline usually belong to notes, so a change must be
  // spelled in order and cancel what it drops (see plausible_key_change).
  std::optional<int> key_signature(int k, int current_fifths, const std::map<int, Clef>& carried) const;

  // Time signature drawn at the start of measure k.
  std::optional<TimeSignature> time_signature(int k) const;

  // Octaves an ottava line shifts a notehead by: up when drawn above its staff (8va),
  // down when below (8vb).
  int ottava_octaves(const Notehead& head) const;

  // Explicit accidental alteration of each notehead that has one, by head index. A head
  // in reach of several accidentals (its own, and a key signature's just before the
  // first note) takes the nearest.
  std::map<int, int> attached_accidentals() const;

 private:
  struct HeadMatch {
    int head;
    float score;  // px: horizontal gap plus vertical mismatch; lower is closer
  };

  // Measure k's region starts at its barline; the first measure's also takes in the
  // system's opening clef, which reaches left of the detected staff start.
  float measure_left(int k) const;
  // Left edge of the first note or rest of measure k on a staff; clef, key and time
  // changes sit between the measure's barline and this point.
  float content_start(int k, int staff_number) const;

  std::optional<HeadMatch> head_right_of(const Blob& acc) const;
  // Accidentals at the start of measure k on a staff, after any clef and before the
  // first note, that do not belong to a note.
  std::vector<Blob> key_accidentals(int k, int staff_number) const;
  // (alteration, diatonic step) of those accidentals, left to right.
  std::vector<std::pair<int, int>> key_accidental_steps(int k, int staff_number, const std::map<int, Clef>& carried) const;

  const std::vector<Staff>& staves_;
  const PageSymbols& symbols_;
  const System& sys_;
  const std::vector<RhythmItem>& items_;
  float il_;
  std::vector<ClefAt> clefs_;
};

}  // namespace omr
