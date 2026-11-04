#include "system_reader.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>

#include "key_signature.h"

namespace omr {
namespace {

// Distances in interlines.
constexpr float kAccidentalMinGap = 0.2f;
constexpr float kAccidentalMaxGap = 3.5f;
constexpr float kFlatBowlAboveBottom = 0.5f;  // a flat's note position sits this far above its box bottom
constexpr float kKeyAccidentalClearance = 2.0f;
constexpr float kHeadHalfWidth = 0.6f;

// Values a time signature's numbers may take.
const std::vector<int> kBeatCounts = [] {
  std::vector<int> v(32);
  std::iota(v.begin(), v.end(), 1);
  return v;
}();
const std::vector<int> kBeatTypes = {1, 2, 4, 8, 16, 32};

int floor_mod(int a, int b) { return ((a % b) + b) % b; }

int accidental_alter(const std::string& kind) {
  if (kind == "sharp") return 1;
  if (kind == "flat") return -1;
  if (kind == "double_sharp") return 2;
  if (kind == "double_flat") return -2;
  return 0;  // natural
}

// Staff position an accidental refers to. A flat's glyph extends well above its note,
// so its reference point is near the bottom of its box.
float accidental_position(const Blob& acc, const Staff& staff, float il) {
  const bool flat_like = acc.kind == "flat" || acc.kind == "double_flat";
  const float y = flat_like ? acc.box.br().y - kFlatBowlAboveBottom * il : acc.center.y;
  return staff.position(acc.center.x, y);
}

Clef clef_from_blob(const Blob& b, const Staff& staff) {
  if (b.kind == "clef_g") return {ClefSign::G, 2};
  if (b.kind == "clef_f") return {ClefSign::F, 4};
  const int line = std::clamp(static_cast<int>(std::lround(staff.position(b.center.x, b.center.y) / 2.0f)) + 1, 1, 5);
  return {ClefSign::C, line};
}

// The allowed value whose digits the glyphs support best. Glyphs overlapping
// horizontally are pieces of one digit (a thin stroke can split a glyph).
std::optional<int> decode_number(std::vector<const TimeGlyph*> glyphs, const std::vector<int>& allowed) {
  std::sort(glyphs.begin(), glyphs.end(), [](const TimeGlyph* a, const TimeGlyph* b) { return a->box.x < b->box.x; });
  std::vector<std::array<double, 10>> digits;
  std::optional<int> covered_until;
  for (const TimeGlyph* g : glyphs) {
    const bool same_digit = covered_until && g->box.x < *covered_until - g->box.width / 2;
    if (!same_digit) digits.push_back({});
    for (int d = 0; d < 10; ++d) digits.back()[d] += g->digit_mass[d];
    covered_until = std::max(covered_until.value_or(g->box.br().x), g->box.br().x);
  }
  std::optional<int> best;
  double best_score = -1;
  for (int value : allowed) {
    const std::string text = std::to_string(value);
    if (text.size() != digits.size()) continue;
    double score = 0;
    for (size_t i = 0; i < text.size(); ++i) score += digits[i][text[i] - '0'];
    if (score > best_score) {
      best_score = score;
      best = value;
    }
  }
  return best;
}

}  // namespace

SystemReader::SystemReader(const std::vector<Staff>& staves, const PageSymbols& symbols, const System& sys,
                           const std::vector<RhythmItem>& items, double interline)
    : staves_(staves), symbols_(symbols), sys_(sys), items_(items), il_(static_cast<float>(interline)) {
  for (const Blob& b : symbols.clefs) {
    if (const int n = sys.staff_number(b.staff)) clefs_.push_back({static_cast<float>(b.box.x), n, clef_from_blob(b, staves[b.staff])});
  }
  std::sort(clefs_.begin(), clefs_.end(), [](const ClefAt& a, const ClefAt& b) { return a.x < b.x; });
}

Clef SystemReader::clef_at(int staff_number, float x, const Clef& carried) const {
  Clef clef = carried;
  for (const ClefAt& c : clefs_) {
    if (c.staff_number == staff_number && c.x <= x) clef = c.clef;
  }
  return clef;
}

std::vector<ClefAt> SystemReader::clefs_in_measure(int k) const {
  std::vector<ClefAt> out;
  for (const ClefAt& c : clefs_) {
    if (c.x >= measure_left(k) && c.x < sys_.bounds[k + 1]) out.push_back(c);
  }
  return out;
}

std::optional<int> SystemReader::key_signature(int k, int current_fifths, const std::map<int, Clef>& carried) const {
  std::optional<int> best;
  for (int n = 1; n <= static_cast<int>(sys_.staves.size()); ++n) {
    const KeySpelling key = spell_key(key_accidental_steps(k, n, carried));
    std::optional<int> reading;
    if (k == 0) {
      reading = key.counted_fifths;
    } else if (key.fifths != 0 && key.exact) {
      reading = key.fifths;
    } else if (key.fifths == 0 && key.exact && cancels_key(key.natural_steps, current_fifths)) {
      reading = 0;
    }
    if (reading && (!best || std::abs(*reading) > std::abs(*best))) best = reading;
  }
  return best;
}

std::optional<TimeSignature> SystemReader::time_signature(int k) const {
  for (int n = 1; n <= static_cast<int>(sys_.staves.size()); ++n) {
    const int page_staff = sys_.staves[n - 1];
    std::vector<const TimeGlyph*> upper, lower;
    for (const TimeGlyph& d : symbols_.time_digits) {
      if (d.staff != page_staff || d.center.x < measure_left(k) || d.center.x >= content_start(k, n)) continue;
      if (d.kind == "ts_common") return TimeSignature{4, 4};
      if (d.kind == "ts_cut") return TimeSignature{2, 2};
      (d.center.y < staves_[page_staff].center_y(d.center.x) ? upper : lower).push_back(&d);
    }
    const std::optional<int> beats = decode_number(upper, kBeatCounts);
    const std::optional<int> beat_type = decode_number(lower, kBeatTypes);
    if (beats && beat_type) return TimeSignature{*beats, *beat_type};
  }
  return std::nullopt;
}

int SystemReader::ottava_octaves(const Notehead& head) const {
  for (const Blob& o : symbols_.ottavas) {
    const float reach = kHeadHalfWidth * il_;
    if (o.staff != head.staff || head.center.x < o.box.x - reach || head.center.x > o.box.br().x + reach) continue;
    const Staff& staff = staves_[o.staff];
    if (o.center.y < staff.top(o.center.x)) return 1;
    if (o.center.y > staff.bottom(o.center.x)) return -1;
  }
  return 0;
}

std::map<int, int> SystemReader::attached_accidentals() const {
  std::map<int, std::pair<float, int>> nearest;  // head -> (match score, alteration)
  for (const Blob& acc : symbols_.accidentals) {
    if (!sys_.staff_number(acc.staff)) continue;
    const std::optional<HeadMatch> match = head_right_of(acc);
    if (!match) continue;
    auto it = nearest.find(match->head);
    if (it == nearest.end() || match->score < it->second.first) nearest[match->head] = {match->score, accidental_alter(acc.kind)};
  }
  std::map<int, int> out;
  for (const auto& [head, match] : nearest) out[head] = match.second;
  return out;
}

float SystemReader::measure_left(int k) const { return k == 0 ? -std::numeric_limits<float>::infinity() : sys_.bounds[k]; }

float SystemReader::content_start(int k, int staff_number) const {
  float x = sys_.bounds[k + 1];
  for (const RhythmItem& it : items_) {
    if (sys_.staff_number(it.page_staff) == staff_number && it.x >= measure_left(k)) x = std::min(x, it.x - kHeadHalfWidth * il_);
  }
  return x;
}

std::optional<SystemReader::HeadMatch> SystemReader::head_right_of(const Blob& acc) const {
  const float pos = accidental_position(acc, staves_[acc.staff], il_);
  std::optional<HeadMatch> best;
  for (size_t h = 0; h < symbols_.heads.size(); ++h) {
    const Notehead& head = symbols_.heads[h];
    const float gap = head.center.x - kHeadHalfWidth * il_ - acc.box.br().x;
    if (head.staff != acc.staff || gap < -kAccidentalMinGap * il_ || gap > kAccidentalMaxGap * il_) continue;
    const float dpos = std::abs(head.position - pos);
    if (dpos > 1.0f) continue;
    const float score = dpos * il_ + gap;
    if (!best || score < best->score) best = HeadMatch{static_cast<int>(h), score};
  }
  return best;
}

std::vector<Blob> SystemReader::key_accidentals(int k, int staff_number) const {
  const int page_staff = sys_.staves[staff_number - 1];
  const float end = content_start(k, staff_number);
  float start = measure_left(k);
  for (const Blob& c : symbols_.clefs) {
    if (c.staff == page_staff && c.box.x >= start && c.box.x < end) start = std::max(start, static_cast<float>(c.box.br().x));
  }
  std::vector<Blob> out;
  for (const Blob& acc : symbols_.accidentals) {
    if (acc.staff != page_staff || acc.center.x < start || acc.center.x > end) continue;
    const std::optional<HeadMatch> match = head_right_of(acc);
    const bool belongs_to_note = match && symbols_.heads[match->head].center.x - acc.center.x < kKeyAccidentalClearance * il_;
    if (!belongs_to_note) out.push_back(acc);
  }
  return out;
}

std::vector<std::pair<int, int>> SystemReader::key_accidental_steps(int k, int staff_number,
                                                                    const std::map<int, Clef>& carried) const {
  std::vector<Blob> accidentals = key_accidentals(k, staff_number);
  std::sort(accidentals.begin(), accidentals.end(), [](const Blob& a, const Blob& b) { return a.center.x < b.center.x; });
  const auto it = carried.find(staff_number);
  const Clef before = it != carried.end() ? it->second : Clef{};
  std::vector<std::pair<int, int>> out;
  for (const Blob& acc : accidentals) {
    const Clef clef = clef_at(staff_number, acc.center.x, before);
    const int position = static_cast<int>(std::lround(accidental_position(acc, staves_[acc.staff], il_)));
    out.emplace_back(accidental_alter(acc.kind), floor_mod(clef.bottom_line_step() + position, 7));
  }
  return out;
}

}  // namespace omr
