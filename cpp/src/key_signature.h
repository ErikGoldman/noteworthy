#pragma once

#include <utility>
#include <vector>

namespace omr {

// Alteration (-1, 0, +1) a key signature of `fifths` (> 0 sharps, < 0 flats) gives to a
// diatonic step (C4 = 0, any octave).
int key_alter(int fifths, int diatonic);

// How a row of accidentals reads as a key signature. `fifths` is the longest run of
// sharps or flats following the standard order, `exact` when every sharp and flat is
// part of it. `counted_fifths` just counts the more common of sharps and flats, which
// tolerates misplaced glyph positions.
struct KeySpelling {
  int fifths = 0;
  bool exact = true;
  int counted_fifths = 0;
  std::vector<int> natural_steps;
};

// `alter_and_step`: (alteration, step C = 0 ... B = 6) of each accidental, left to right.
KeySpelling spell_key(const std::vector<std::pair<int, int>>& alter_and_step);

// Whether naturals on these steps cancel exactly the key of `fifths`.
bool cancels_key(const std::vector<int>& natural_steps, int fifths);

}  // namespace omr
