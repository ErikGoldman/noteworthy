#include "layout.h"

#include <algorithm>
#include <cmath>

namespace omr {
namespace {

constexpr float kJoinSlackInterlines = 0.5f;      // barline may stop this short of a staff
constexpr float kEdgeSlackInterlines = 1.5f;     // brace tips overshoot the staff a little
constexpr float kStaffEndReachInterlines = 3.0f;
constexpr float kStaffCoverage = 0.8f;            // barline must cover this much of a staff
constexpr float kBarlineMergeInterlines = 3.5f;   // closer barlines (double, final, or doubly detected) count once: no measure is that narrow
constexpr float kSystemStartInterlines = 1.5f;    // a barline this close to the start opens the system
constexpr float kTrailingMeasureInterlines = 3.0f;

// A system's barlines and brace start at some staff's top line and end at some staff's
// bottom line. Long verticals that overshoot both (e.g. a scanned page border, which the
// network may read as a barline) would otherwise chain every staff on the page together.
bool ends_on_staff_edges(const Blob& b, const std::vector<Staff>& staves, double interline) {
  const float x = b.center.x;
  const float slack = static_cast<float>(kEdgeSlackInterlines * interline);
  const float top = static_cast<float>(b.box.y), bottom = static_cast<float>(b.box.br().y);
  const bool starts = std::any_of(staves.begin(), staves.end(), [&](const Staff& s) { return std::abs(top - s.top(x)) < slack; });
  const bool ends = std::any_of(staves.begin(), staves.end(), [&](const Staff& s) { return std::abs(bottom - s.bottom(x)) < slack; });
  return starts && ends;
}

// Staff x-extents come from detection strips, so the true ends lie a little further out.
bool within_span(float x, float x_start, float x_end, double interline) {
  const float reach = static_cast<float>(kStaffEndReachInterlines * interline);
  return x > x_start - reach && x < x_end + reach;
}

bool joins(const Blob& b, const Staff& upper, const Staff& lower, const std::vector<Staff>& staves, double interline) {
  const float x = b.center.x;
  const float slack = static_cast<float>(kJoinSlackInterlines * interline);
  return b.box.y <= upper.bottom(x) + slack && b.box.y + b.box.height >= lower.top(x) - slack &&
         within_span(x, std::max(upper.x_min(), lower.x_min()), std::min(upper.x_max(), lower.x_max()), interline) &&
         ends_on_staff_edges(b, staves, interline);
}

bool covers_staff(const Blob& b, const Staff& s) {
  const float x = b.center.x;
  const float overlap = std::min<float>(b.box.y + b.box.height, s.bottom(x)) - std::max<float>(b.box.y, s.top(x));
  return overlap >= kStaffCoverage * (s.bottom(x) - s.top(x));
}

// A barline drawn between two staves of a system. Seen faintly, the network may predict
// only part of a barline's length, but a vertical line bridging the gap between staves
// is still unmistakable.
bool bridges_gap(const Blob& b, const Staff& upper, const Staff& lower) {
  const float x = b.center.x;
  return b.box.y <= upper.bottom(x) && b.box.br().y >= lower.top(x);
}

bool on_system(const Blob& b, const System& sys, const std::vector<Staff>& staves) {
  for (size_t k = 0; k < sys.staves.size(); ++k) {
    if (covers_staff(b, staves[sys.staves[k]])) return true;
    if (k > 0 && bridges_gap(b, staves[sys.staves[k - 1]], staves[sys.staves[k]])) return true;
  }
  return false;
}

std::vector<std::vector<int>> group_staves(const std::vector<Staff>& staves, const PageSymbols& symbols, double interline) {
  auto any_joins = [&](const std::vector<Blob>& blobs, int i) {
    return std::any_of(blobs.begin(), blobs.end(), [&](const Blob& b) { return joins(b, staves[i - 1], staves[i], staves, interline); });
  };
  std::vector<std::vector<int>> groups;
  for (int i = 0; i < static_cast<int>(staves.size()); ++i) {
    if (i > 0 && (any_joins(symbols.barlines, i) || any_joins(symbols.braces, i))) {
      groups.back().push_back(i);
    } else {
      groups.push_back({i});
    }
  }
  return groups;
}

std::vector<float> barline_positions(const System& sys, const std::vector<Staff>& staves, const PageSymbols& symbols,
                                     double interline) {
  std::vector<float> xs;
  for (const Blob& b : symbols.barlines) {
    if (on_system(b, sys, staves) && within_span(b.center.x, sys.x_start, sys.x_end, interline)) xs.push_back(b.center.x);
  }
  std::sort(xs.begin(), xs.end());
  std::vector<float> merged;
  for (float x : xs) {
    if (!merged.empty() && x - merged.back() < kBarlineMergeInterlines * interline) {
      merged.back() = x;  // keep the rightmost line of a double/final barline
    } else {
      merged.push_back(x);
    }
  }
  return merged;
}

// Whether any note or rest of the system lies left of x.
bool has_content_before(float x, const System& sys, const PageSymbols& symbols) {
  const auto on_system = [&](int staff) { return sys.staff_number(staff) != 0; };
  return std::any_of(symbols.heads.begin(), symbols.heads.end(), [&](const Notehead& h) { return on_system(h.staff) && h.center.x < x; }) ||
         std::any_of(symbols.rests.begin(), symbols.rests.end(), [&](const Blob& r) { return on_system(r.staff) && r.center.x < x; });
}

}  // namespace

int System::staff_number(int page_staff) const {
  auto it = std::find(staves.begin(), staves.end(), page_staff);
  return it == staves.end() ? 0 : static_cast<int>(it - staves.begin()) + 1;
}

std::vector<System> find_systems(const std::vector<Staff>& staves, const PageSymbols& symbols, double interline) {
  std::vector<System> systems;
  for (auto& group : group_staves(staves, symbols, interline)) {
    System sys;
    sys.staves = group;
    sys.x_start = staves[group.front()].x_min();
    sys.x_end = staves[group.front()].x_max();
    for (int s : group) {
      sys.x_start = std::min(sys.x_start, staves[s].x_min());
      sys.x_end = std::max(sys.x_end, staves[s].x_max());
    }
    sys.bounds.push_back(sys.x_start);
    for (float x : barline_positions(sys, staves, symbols, interline)) {
      // A barline with nothing before it opens the system (often a start-repeat drawn
      // after the clef, key and time signature); it bounds no measure.
      if (x - sys.x_start < kSystemStartInterlines * interline || !has_content_before(x, sys, symbols)) continue;
      sys.bounds.push_back(x);
    }
    if (sys.x_end - sys.bounds.back() > kTrailingMeasureInterlines * interline) sys.bounds.push_back(sys.x_end);
    systems.push_back(std::move(sys));
  }
  return systems;
}

}  // namespace omr
