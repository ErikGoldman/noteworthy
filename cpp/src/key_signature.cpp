#include "key_signature.h"

#include <algorithm>
#include <cstdlib>

namespace omr {
namespace {

int floor_mod(int a, int b) { return ((a % b) + b) % b; }

// Steps (C = 0 ... B = 6) a key signature alters, in the order it writes them.
constexpr int kSharpOrder[7] = {3, 0, 4, 1, 5, 2, 6};  // F C G D A E B
constexpr int kFlatOrder[7] = {6, 2, 5, 1, 4, 0, 3};   // B E A D G C F

}  // namespace

int key_alter(int fifths, int diatonic) {
  const int step = floor_mod(diatonic, 7);
  for (int i = 0; i < std::abs(fifths); ++i) {
    if ((fifths > 0 ? kSharpOrder : kFlatOrder)[i] == step) return fifths > 0 ? 1 : -1;
  }
  return 0;
}

KeySpelling spell_key(const std::vector<std::pair<int, int>>& alter_and_step) {
  KeySpelling out;
  int altered = 0, run = 0, sign = 0, sharps = 0, flats = 0;
  for (const auto& [alter, step] : alter_and_step) {
    if (alter == 0) {
      out.natural_steps.push_back(step);
      continue;
    }
    ++altered;
    (alter > 0 ? sharps : flats)++;
    if (sign == 0) sign = alter > 0 ? 1 : -1;
    const int* order = sign > 0 ? kSharpOrder : kFlatOrder;
    if ((alter > 0) == (sign > 0) && run < 7 && order[run] == step) ++run;
  }
  out.fifths = sign * run;
  out.exact = run == altered;
  out.counted_fifths = sharps >= flats ? sharps : -flats;
  return out;
}

bool cancels_key(const std::vector<int>& natural_steps, int fifths) {
  if (fifths == 0 || natural_steps.size() != static_cast<size_t>(std::abs(fifths))) return false;
  return std::all_of(natural_steps.begin(), natural_steps.end(), [&](int step) { return key_alter(fifths, step) != 0; });
}

bool plausible_key_change(const KeySpelling& spelling, int current_fifths) {
  if (!spelling.exact || spelling.fifths == current_fifths) return false;
  if (spelling.fifths == 0) return cancels_key(spelling.natural_steps, current_fifths);
  for (int step = 0; step < 7; ++step) {
    const bool dropped = key_alter(current_fifths, step) != 0 && key_alter(spelling.fifths, step) != key_alter(current_fifths, step);
    if (dropped && std::find(spelling.natural_steps.begin(), spelling.natural_steps.end(), step) == spelling.natural_steps.end()) {
      return false;
    }
  }
  return true;
}

}  // namespace omr
