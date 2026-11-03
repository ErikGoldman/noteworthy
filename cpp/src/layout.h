#pragma once

#include <vector>

#include "staff.h"
#include "symbols.h"

namespace omr {

struct System {
  std::vector<int> staves;  // page staff indices, top to bottom
  float x_start = 0;        // left edge of the staves
  float x_end = 0;
  // Measure boundaries left to right: measure k spans [bounds[k], bounds[k + 1]].
  // bounds.front() is x_start; the first measure also holds clef/key/time.
  std::vector<float> bounds;

  int measure_count() const { return static_cast<int>(bounds.size()) - 1; }
  // 1-based staff number within the system, or 0 if `page_staff` is not part of it.
  int staff_number(int page_staff) const;
};

// Groups staves into systems (staves joined by barlines or a brace) and finds the
// measures of each system from its barlines.
std::vector<System> find_systems(const std::vector<Staff>& staves, const PageSymbols& symbols, double interline);

}  // namespace omr
