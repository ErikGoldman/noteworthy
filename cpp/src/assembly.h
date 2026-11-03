#pragma once

#include <map>
#include <optional>
#include <vector>

#include "layout.h"
#include "rhythm.h"
#include "score.h"
#include "staff.h"
#include "symbols.h"

namespace omr {

class SystemReader;

// Turns the symbols of successive pages into one score. Clef, key and time signature
// persist across systems and pages until they change.
class ScoreBuilder {
 public:
  void add_page(const std::vector<Staff>& staves, const PageSymbols& symbols, double interline);
  Score build() &&;

 private:
  void apply_attribute_changes(const SystemReader& reader, int k, Measure& measure);
  std::map<int, Pitch> resolve_pitches(const std::vector<MeasureItem>& items, const System& sys, const SystemReader& reader,
                                       const std::map<int, int>& explicit_alter, const PageSymbols& symbols);

  Score score_;
  std::map<int, Clef> clefs_;  // by staff number within a system
  std::optional<int> key_fifths_;
  std::optional<TimeSignature> time_;
};

}  // namespace omr
