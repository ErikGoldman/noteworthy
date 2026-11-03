#include "rhythm.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

namespace omr {
namespace {

constexpr float kColumnInterlines = 0.4f;
constexpr int64_t kFinestOnsetGrid = 8;  // never snap finer than a 32nd (1/8 quarter)
constexpr int kVoicesPerStaff = 4;

// Union-find where each element stores its onset relative to its root, so relations
// "onset[b] - onset[a] = d" can be merged. Each group also tracks the time span it
// covers; a relation that would stretch a group beyond one measure is rejected, since
// it can only come from a misread symbol or a wrong grouping guess.
class RelativeOnsets {
 public:
  RelativeOnsets(const std::vector<Rational>& durations, Rational measure_length)
      : parent_(durations.size()), offset_(durations.size(), Rational(0)), first_(durations.size(), Rational(0)),
        last_end_(durations), limit_(measure_length) {
    std::iota(parent_.begin(), parent_.end(), 0);
  }

  int root(int i) {
    if (parent_[i] == i) return i;
    const int r = root(parent_[i]);
    offset_[i] += offset_[parent_[i]];
    parent_[i] = r;
    return r;
  }

  Rational offset(int i) {
    root(i);
    return offset_[i];
  }

  // Records onset[b] - onset[a] = d unless it contradicts what is already known.
  void relate(int a, int b, Rational d) {
    const int ra = root(a), rb = root(b);
    if (ra == rb) return;  // already related (consistently or not); first relation wins
    const Rational shift = offset(a) + d - offset(b);  // rb's frame -> ra's frame
    const Rational first = std::min(first_[ra], first_[rb] + shift);
    const Rational last_end = std::max(last_end_[ra], last_end_[rb] + shift);
    if (last_end - first > limit_) return;
    parent_[rb] = ra;
    offset_[rb] = shift;
    first_[ra] = first;
    last_end_[ra] = last_end;
  }

 private:
  std::vector<int> parent_;
  std::vector<Rational> offset_;
  // Per root, relative to the root's onset: earliest onset and latest end in the group.
  std::vector<Rational> first_;
  std::vector<Rational> last_end_;
  Rational limit_;
};

bool same_column(const MeasureItem& a, const MeasureItem& b, double interline) {
  return std::abs(a.item->x - b.item->x) < kColumnInterlines * interline;
}

bool staff_is_polyphonic(const std::vector<MeasureItem>& items, int staff, double interline) {
  for (size_t i = 0; i < items.size(); ++i) {
    for (size_t j = i + 1; j < items.size(); ++j) {
      const RhythmItem& a = *items[i].item;
      const RhythmItem& b = *items[j].item;
      if (items[i].staff == staff && items[j].staff == staff && a.stem_dir * b.stem_dir < 0 &&
          same_column(items[i], items[j], interline)) {
        return true;
      }
    }
  }
  return false;
}

void chain(RelativeOnsets& onsets, const std::vector<MeasureItem>& items, std::vector<int> members) {
  std::sort(members.begin(), members.end(), [&](int a, int b) { return items[a].item->x < items[b].item->x; });
  for (size_t k = 1; k < members.size(); ++k) onsets.relate(members[k - 1], members[k], items[members[k - 1]].duration);
}

void add_constraints(RelativeOnsets& onsets, const std::vector<MeasureItem>& items, double interline) {
  std::map<int, std::vector<int>> beams;
  for (size_t i = 0; i < items.size(); ++i) {
    if (items[i].item->beam_group) beams[items[i].item->beam_group].push_back(static_cast<int>(i));
  }
  for (auto& [group, members] : beams) chain(onsets, items, members);

  for (size_t i = 0; i < items.size(); ++i) {
    for (size_t j = i + 1; j < items.size(); ++j) {
      if (same_column(items[i], items[j], interline)) onsets.relate(static_cast<int>(i), static_cast<int>(j), Rational(0));
    }
  }

  std::map<std::pair<int, int>, std::vector<int>> voices;  // (staff, stem voice) -> items
  std::map<int, bool> polyphonic;
  for (size_t i = 0; i < items.size(); ++i) {
    const int staff = items[i].staff;
    if (!polyphonic.count(staff)) polyphonic[staff] = staff_is_polyphonic(items, staff, interline);
    const int voice = polyphonic[staff] ? items[i].item->stem_dir : 0;
    voices[{staff, voice}].push_back(static_cast<int>(i));
  }
  for (auto& [key, members] : voices) chain(onsets, items, members);
}

// Snaps to a multiple of `unit`: a group of eighths starts on an eighth boundary.
Rational snap(double quarters, Rational unit) {
  unit = std::max(unit, Rational(1, kFinestOnsetGrid));
  const double steps = std::max(0.0, quarters) / unit.to_double();
  return unit * Rational(static_cast<int64_t>(std::lround(steps)));
}

// Estimated onset at x, by linear interpolation between already placed items.
double interpolate_onset(float x, const std::vector<std::pair<float, Rational>>& placed) {
  if (placed.empty()) return 0;
  auto right = std::lower_bound(placed.begin(), placed.end(), x, [](const auto& p, float v) { return p.first < v; });
  if (right == placed.begin()) return 0;
  if (right == placed.end()) return std::prev(right)->second.to_double();
  auto left = std::prev(right);
  const double t = (x - left->first) / std::max(1e-3f, right->first - left->first);
  return left->second.to_double() * (1 - t) + right->second.to_double() * t;
}

std::vector<Rational> anchor_groups(RelativeOnsets& onsets, const std::vector<MeasureItem>& items, const MeasureSpan& span) {
  std::map<int, std::vector<int>> groups;
  for (size_t i = 0; i < items.size(); ++i) groups[onsets.root(static_cast<int>(i))].push_back(static_cast<int>(i));
  std::vector<std::vector<int>> ordered;
  for (auto& [root, members] : groups) ordered.push_back(members);
  auto min_x = [&](const std::vector<int>& g) {
    float m = 1e9f;
    for (int i : g) m = std::min(m, items[i].item->x);
    return m;
  };
  std::sort(ordered.begin(), ordered.end(), [&](const auto& a, const auto& b) { return min_x(a) < min_x(b); });

  std::vector<Rational> result(items.size());
  // (x, onset) reference points, kept sorted by x. The barlines bound the measure in time.
  std::vector<std::pair<float, Rational>> placed = {{span.x_start, Rational(0)}, {span.x_end, span.length}};
  for (size_t g = 0; g < ordered.size(); ++g) {
    const auto& members = ordered[g];
    Rational earliest = onsets.offset(members.front());
    Rational shortest = items[members.front()].duration;
    for (int i : members) {
      earliest = std::min(earliest, onsets.offset(i));
      shortest = std::min(shortest, items[i].duration);
    }
    // The leftmost group starts the measure; others are placed where x says they belong.
    Rational shift = Rational(0) - earliest;
    if (g > 0) {
      std::vector<double> estimates;
      for (int i : members) estimates.push_back(interpolate_onset(items[i].item->x, placed) - onsets.offset(i).to_double());
      std::nth_element(estimates.begin(), estimates.begin() + estimates.size() / 2, estimates.end());
      shift = snap(estimates[estimates.size() / 2] + earliest.to_double(), shortest) - earliest;
    }
    for (int i : members) {
      result[i] = onsets.offset(i) + shift;
      placed.emplace_back(items[i].item->x, result[i]);
    }
    std::sort(placed.begin(), placed.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
  }
  return result;
}

// Partitions each staff's items into non-overlapping voices, keeping stem directions
// apart when they overlap in time.
std::vector<int> assign_voices(const std::vector<MeasureItem>& items, const std::vector<Rational>& onsets) {
  std::vector<int> order(items.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) {
    if (onsets[a] != onsets[b]) return onsets[a] < onsets[b];
    return items[a].item->stem_dir > items[b].item->stem_dir;
  });
  struct Lane {
    int staff;
    int stem_dir;
    Rational end;
  };
  std::vector<Lane> lanes;
  std::vector<int> voice(items.size(), 1);
  std::map<int, int> lanes_per_staff;
  std::vector<int> lane_slot;
  for (int i : order) {
    const MeasureItem& it = items[i];
    int chosen = -1;
    for (int pass = 0; pass < 2 && chosen < 0; ++pass) {
      for (size_t l = 0; l < lanes.size(); ++l) {
        const bool free = lanes[l].staff == it.staff && lanes[l].end <= onsets[i];
        const bool same_dir = lanes[l].stem_dir == it.item->stem_dir || it.item->stem_dir == 0;
        if (free && (pass == 1 || same_dir)) {
          chosen = static_cast<int>(l);
          break;
        }
      }
    }
    if (chosen < 0) {
      lanes.push_back({it.staff, it.item->stem_dir, Rational(0)});
      lane_slot.push_back(lanes_per_staff[it.staff]++);
      chosen = static_cast<int>(lanes.size()) - 1;
    }
    lanes[chosen].end = onsets[i] + it.duration;
    voice[i] = (it.staff - 1) * kVoicesPerStaff + std::min(lane_slot[chosen], kVoicesPerStaff - 1) + 1;
  }
  return voice;
}

}  // namespace

std::vector<Timing> solve_measure(const std::vector<MeasureItem>& items, const MeasureSpan& span, double interline) {
  std::vector<Rational> durations;
  for (const MeasureItem& it : items) durations.push_back(it.duration);
  RelativeOnsets onsets(durations, span.length);
  add_constraints(onsets, items, interline);
  const std::vector<Rational> starts = anchor_groups(onsets, items, span);
  const std::vector<int> voices = assign_voices(items, starts);
  std::vector<Timing> out(items.size());
  for (size_t i = 0; i < items.size(); ++i) out[i] = {starts[i], voices[i]};
  return out;
}

}  // namespace omr
