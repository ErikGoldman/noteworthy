#pragma once

#include <vector>

#include "score.h"
#include "symbols.h"

namespace omr {

// One rhythmic unit as seen on the page: a chord (heads sharing a stem, or stemless
// whole notes) or a rest. Pitches and timing are resolved later.
struct RhythmItem {
  float x = 0;              // horizontal position of the notehead column / rest centre
  int page_staff = -1;
  NoteValue value;
  std::vector<int> heads;   // indices into PageSymbols::heads; empty for a rest
  int stem_dir = 0;         // +1 up, -1 down, 0 stemless or rest
  // Which of two voices sharing a staff this belongs to: +1 upper, -1 lower, 0 unknown.
  // Notes show it by stem direction, rests by being raised or lowered off the centre.
  int voice_side = 0;
  int beam_group = 0;       // beam component label, 0 if not beamed
  std::string rest_kind;    // channel name for rests, e.g. "rest_whole"

  bool is_rest() const { return heads.empty(); }
};

std::vector<RhythmItem> build_items(const PageSymbols& symbols, const std::vector<Staff>& staves, double interline);

}  // namespace omr
