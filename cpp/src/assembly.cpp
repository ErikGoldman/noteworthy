#include "assembly.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <string>

#include "chords.h"
#include "layout.h"
#include "rhythm.h"

namespace omr {
namespace {

// Distances in interlines.
constexpr float kAccidentalMinGap = 0.2f;
constexpr float kAccidentalMaxGap = 3.5f;
constexpr float kFlatBowlAboveBottom = 0.5f;  // a flat's note position sits this far above its box bottom
constexpr float kKeyAccidentalClearance = 2.0f;
constexpr float kTieEndReach = 1.2f;
constexpr float kTieVerticalReach = 1.5f;
constexpr float kHeadHalfWidth = 0.6f;
// Values a time signature's numbers may take.
const std::vector<int> kBeatCounts = [] {
  std::vector<int> v(32);
  std::iota(v.begin(), v.end(), 1);
  return v;
}();
const std::vector<int> kBeatTypes = {1, 2, 4, 8, 16, 32};

int floor_mod(int a, int b) { return ((a % b) + b) % b; }

// Alteration a key signature gives to a diatonic step.
int key_alter(int fifths, int diatonic) {
  static constexpr int kSharpOrder[7] = {3, 0, 4, 1, 5, 2, 6};  // F C G D A E B, as steps C=0
  const int step = floor_mod(diatonic, 7);
  for (int i = 0; i < std::abs(fifths); ++i) {
    if (kSharpOrder[fifths > 0 ? i : 6 - i] == step) return fifths > 0 ? 1 : -1;
  }
  return 0;
}

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

struct ClefAt {
  float x;
  int staff_number;
  Clef clef;
};

}  // namespace

// Everything the measures of one system need to know about their surroundings.
class SystemReader {
 public:
  SystemReader(const std::vector<Staff>& staves, const PageSymbols& symbols, const System& sys,
               const std::vector<RhythmItem>& items, double interline)
      : staves_(staves), symbols_(symbols), sys_(sys), items_(items), il_(static_cast<float>(interline)) {
    for (const Blob& b : symbols.clefs) {
      if (const int n = sys.staff_number(b.staff)) clefs_.push_back({static_cast<float>(b.box.x), n, clef_from_blob(b, staves[b.staff])});
    }
    std::sort(clefs_.begin(), clefs_.end(), [](const ClefAt& a, const ClefAt& b) { return a.x < b.x; });
  }

  // Clef in force at x on a staff, given the clef carried in from earlier systems.
  Clef clef_at(int staff_number, float x, const Clef& carried) const {
    Clef clef = carried;
    for (const ClefAt& c : clefs_) {
      if (c.staff_number == staff_number && c.x <= x) clef = c.clef;
    }
    return clef;
  }

  // Clefs drawn within measure k, left to right.
  std::vector<ClefAt> clefs_in_measure(int k) const {
    std::vector<ClefAt> out;
    for (const ClefAt& c : clefs_) {
      if (c.x >= measure_left(k) && c.x < sys_.bounds[k + 1]) out.push_back(c);
    }
    return out;
  }

  // Key signature drawn at the start of measure k. Every system restates its key, so
  // the first measure without one is in C; later measures without one keep the key.
  std::optional<int> key_signature(int k) const {
    std::optional<int> best;
    for (int n = 1; n <= static_cast<int>(sys_.staves.size()); ++n) {
      const std::vector<Blob> accidentals = key_accidentals(k, n);
      if (k > 0 && accidentals.empty()) continue;
      // A change to fewer accidentals may first cancel the old ones with naturals.
      int sharps = 0, flats = 0;
      for (const Blob& acc : accidentals) {
        if (accidental_alter(acc.kind) > 0) ++sharps;
        if (accidental_alter(acc.kind) < 0) ++flats;
      }
      const int fifths = sharps >= flats ? sharps : -flats;
      if (!best || std::abs(fifths) > std::abs(*best)) best = fifths;
    }
    return best;
  }

  // Time signature drawn at the start of measure k.
  std::optional<TimeSignature> time_signature(int k) const {
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

  // Octaves an ottava line shifts a notehead by: up when drawn above its staff (8va),
  // down when below (8vb).
  int ottava_octaves(const Notehead& head) const {
    for (const Blob& o : symbols_.ottavas) {
      const float reach = kHeadHalfWidth * il_;
      if (o.staff != head.staff || head.center.x < o.box.x - reach || head.center.x > o.box.br().x + reach) continue;
      const Staff& staff = staves_[o.staff];
      if (o.center.y < staff.top(o.center.x)) return 1;
      if (o.center.y > staff.bottom(o.center.x)) return -1;
    }
    return 0;
  }

  // Explicit accidental alteration of each notehead that has one, by head index.
  std::map<int, int> attached_accidentals() const {
    std::map<int, int> out;
    for (const Blob& acc : symbols_.accidentals) {
      if (!sys_.staff_number(acc.staff)) continue;
      const int head = head_right_of(acc);
      if (head >= 0) out[head] = accidental_alter(acc.kind);
    }
    return out;
  }

  // Measure k's region starts at its barline; the first measure's also takes in the
  // system's opening clef, which reaches left of the detected staff start.
  float measure_left(int k) const { return k == 0 ? -std::numeric_limits<float>::infinity() : sys_.bounds[k]; }

  // Left edge of the first note or rest of measure k on a staff; clef, key and time
  // changes sit between the measure's barline and this point.
  float content_start(int k, int staff_number) const {
    float x = sys_.bounds[k + 1];
    for (const RhythmItem& it : items_) {
      if (sys_.staff_number(it.page_staff) == staff_number && it.x >= measure_left(k)) x = std::min(x, it.x - kHeadHalfWidth * il_);
    }
    return x;
  }

 private:
  // The allowed value whose digits the glyphs support best. Glyphs overlapping
  // horizontally are pieces of one digit (a thin stroke can split a glyph).
  static std::optional<int> decode_number(std::vector<const TimeGlyph*> glyphs, const std::vector<int>& allowed) {
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

  int head_right_of(const Blob& acc) const {
    const float pos = accidental_position(acc, staves_[acc.staff], il_);
    int best = -1;
    float best_score = 1e9f;
    for (size_t h = 0; h < symbols_.heads.size(); ++h) {
      const Notehead& head = symbols_.heads[h];
      const float gap = head.center.x - kHeadHalfWidth * il_ - acc.box.br().x;
      if (head.staff != acc.staff || gap < -kAccidentalMinGap * il_ || gap > kAccidentalMaxGap * il_) continue;
      const float dpos = std::abs(head.position - pos);
      if (dpos > 1.0f) continue;
      const float score = dpos * il_ + gap;
      if (score < best_score) {
        best_score = score;
        best = static_cast<int>(h);
      }
    }
    return best;
  }

  // Accidentals at the start of measure k on a staff, after any clef and before the
  // first note, that do not belong to a note.
  std::vector<Blob> key_accidentals(int k, int staff_number) const {
    const int page_staff = sys_.staves[staff_number - 1];
    const float end = content_start(k, staff_number);
    float start = measure_left(k);
    for (const Blob& c : symbols_.clefs) {
      if (c.staff == page_staff && c.box.x >= start && c.box.x < end) start = std::max(start, static_cast<float>(c.box.br().x));
    }
    std::vector<Blob> out;
    for (const Blob& acc : symbols_.accidentals) {
      if (acc.staff != page_staff || acc.center.x < start || acc.center.x > end) continue;
      const int head = head_right_of(acc);
      const bool belongs_to_note = head >= 0 && symbols_.heads[head].center.x - acc.center.x < kKeyAccidentalClearance * il_;
      if (!belongs_to_note) out.push_back(acc);
    }
    return out;
  }

  const std::vector<Staff>& staves_;
  const PageSymbols& symbols_;
  const System& sys_;
  const std::vector<RhythmItem>& items_;
  float il_;
  std::vector<ClefAt> clefs_;
};

namespace {

// Where each notehead's pitch went: (event index in the measure, pitch index in it).
using HeadSlots = std::map<int, std::pair<size_t, size_t>>;

// Marks tie starts: a tie/slur arc whose ends sit beside two noteheads of equal pitch.
void mark_ties(std::vector<Event>& events, const HeadSlots& slots, const PageSymbols& symbols, float il) {
  auto pitch_at = [&](const std::pair<size_t, size_t>& slot) { return events[slot.first].pitches[slot.second].midi(); };
  for (const Blob& tie : symbols.ties) {
    const float left = static_cast<float>(tie.box.x), right = static_cast<float>(tie.box.br().x);
    for (const auto& [ha, slot_a] : slots) {
      const Notehead& a = symbols.heads[ha];
      if (std::abs(a.center.x + kHeadHalfWidth * il - left) > kTieEndReach * il ||
          std::abs(a.center.y - tie.center.y) > kTieVerticalReach * il) continue;
      for (const auto& [hb, slot_b] : slots) {
        const Notehead& b = symbols.heads[hb];
        if (hb == ha || std::abs(b.center.x - kHeadHalfWidth * il - right) > kTieEndReach * il) continue;
        if (pitch_at(slot_a) == pitch_at(slot_b)) events[slot_a.first].tie_start[slot_a.second] = true;
      }
    }
  }
}

// Items of measure k, with the duration each occupies. A lone whole rest fills the
// measure whatever its length.
std::vector<MeasureItem> measure_items(const std::vector<RhythmItem>& sys_items, const System& sys, int k, Rational measure_length) {
  std::vector<const RhythmItem*> inside;
  for (const RhythmItem& it : sys_items) {
    if (it.x >= sys.bounds[k] && it.x < sys.bounds[k + 1]) inside.push_back(&it);
  }
  std::map<int, int> items_per_staff;
  for (const RhythmItem* it : inside) items_per_staff[sys.staff_number(it->page_staff)]++;
  std::vector<MeasureItem> out;
  for (const RhythmItem* it : inside) {
    const int staff = sys.staff_number(it->page_staff);
    const bool measure_rest = it->rest_kind == "rest_whole" && items_per_staff[staff] == 1;
    out.push_back({it, staff, measure_rest ? measure_length : it->value.duration()});
  }
  return out;
}

}  // namespace

// Key and time changes drawn at the start of measure k of a system.
void ScoreBuilder::apply_attribute_changes(const SystemReader& reader, int k, Measure& measure) {
  const std::optional<int> key = reader.key_signature(k);
  if (key && key != key_fifths_) measure.key_fifths = key_fifths_ = key;
  const std::optional<TimeSignature> time = reader.time_signature(k);
  if (time && (!time_ || time->beats != time_->beats || time->beat_type != time_->beat_type)) measure.time = time_ = time;
  if (score_.measures.empty()) {
    if (!key_fifths_) measure.key_fifths = key_fifths_ = 0;
    if (!time_) measure.time = time_ = TimeSignature{};
  }
}

// The clef in force on staff n, as carried from earlier measures; treble (the MusicXML
// default) before any clef was read.
Clef ScoreBuilder::current_clef(int n) const {
  const auto it = clefs_.find(n);
  return it != clefs_.end() ? it->second : Clef{};
}

// Clef changes drawn in measure k, each taking effect at the first item after it on its
// staff (or at the measure end when none follows, as for a clef before a barline).
// Pitches do not depend on this: clef_at() reads clefs by position.
void ScoreBuilder::record_clef_changes(const SystemReader& reader, int k, const std::vector<MeasureItem>& items,
                                       const std::vector<Timing>& timing, Rational length, Measure& measure) {
  for (const ClefAt& c : reader.clefs_in_measure(k)) {
    const auto current = clefs_.find(c.staff_number);
    if (current != clefs_.end() && current->second.sign == c.clef.sign && current->second.line == c.clef.line) continue;
    Rational onset = length;
    for (size_t i = 0; i < items.size(); ++i) {
      if (items[i].staff == c.staff_number && items[i].item->x > c.x) onset = std::min(onset, timing[i].onset);
    }
    measure.clefs.push_back({c.staff_number, c.clef, onset});
    clefs_[c.staff_number] = c.clef;
  }
  std::stable_sort(measure.clefs.begin(), measure.clefs.end(), [](const ClefChange& a, const ClefChange& b) { return a.onset < b.onset; });
}

// Pitch of every notehead in the measure. Accidentals carry to later notes of the same
// staff and step within the measure; otherwise the key signature applies.
std::map<int, Pitch> ScoreBuilder::resolve_pitches(const std::vector<MeasureItem>& items, const System& sys,
                                                   const SystemReader& reader, const std::map<int, int>& explicit_alter,
                                                   const PageSymbols& symbols) {
  std::vector<int> heads;
  for (const MeasureItem& it : items) heads.insert(heads.end(), it.item->heads.begin(), it.item->heads.end());
  std::sort(heads.begin(), heads.end(), [&](int a, int b) { return symbols.heads[a].center.x < symbols.heads[b].center.x; });
  std::map<int, Pitch> pitch_of;
  std::map<std::pair<int, int>, int> carried_alter;  // (staff, diatonic) -> alter
  for (int h : heads) {
    const Notehead& head = symbols.heads[h];
    const int n = sys.staff_number(head.staff);
    Pitch p;
    p.diatonic = reader.clef_at(n, head.center.x, current_clef(n)).bottom_line_step() + head.position;
    if (auto e = explicit_alter.find(h); e != explicit_alter.end()) carried_alter[{n, p.diatonic}] = e->second;
    auto c = carried_alter.find({n, p.diatonic});
    p.alter = c != carried_alter.end() ? c->second : key_alter(*key_fifths_, p.diatonic);
    // Accidentals carry by written position, so the ottava shift comes last.
    p.diatonic += 7 * reader.ottava_octaves(head);
    pitch_of[h] = p;
  }
  return pitch_of;
}

void ScoreBuilder::add_page(const std::vector<Staff>& staves, const PageSymbols& symbols, double interline) {
  const std::vector<RhythmItem> items = build_items(symbols, staves, interline);
  for (const System& sys : find_systems(staves, symbols, interline)) {
    score_.staves = std::max(score_.staves, static_cast<int>(sys.staves.size()));
    std::vector<RhythmItem> sys_items;
    for (const RhythmItem& it : items) {
      if (sys.staff_number(it.page_staff)) sys_items.push_back(it);
    }
    const SystemReader reader(staves, symbols, sys, sys_items, interline);
    const std::map<int, int> explicit_alter = reader.attached_accidentals();
    for (int k = 0; k < sys.measure_count(); ++k) {
      Measure measure;
      apply_attribute_changes(reader, k, measure);
      const Rational length = time_->measure_length();
      const std::vector<MeasureItem> mitems = measure_items(sys_items, sys, k, length);
      const std::vector<Timing> timing = solve_measure(mitems, {length, sys.bounds[k], sys.bounds[k + 1]}, interline);
      const std::map<int, Pitch> pitch_of = resolve_pitches(mitems, sys, reader, explicit_alter, symbols);
      record_clef_changes(reader, k, mitems, timing, length, measure);
      HeadSlots slots;
      for (size_t i = 0; i < mitems.size(); ++i) {
        const RhythmItem& it = *mitems[i].item;
        Event e;
        e.onset = timing[i].onset;
        e.duration = mitems[i].duration;
        e.value = mitems[i].duration == it.value.duration() ? it.value : NoteValue{1, 0};
        e.staff = mitems[i].staff;
        e.voice = timing[i].voice;
        for (size_t j = 0; j < it.heads.size(); ++j) {
          e.pitches.push_back(pitch_of.at(it.heads[j]));
          slots[it.heads[j]] = {measure.events.size(), j};
        }
        e.tie_start.assign(e.pitches.size(), false);
        measure.events.push_back(std::move(e));
      }
      mark_ties(measure.events, slots, symbols, static_cast<float>(interline));
      score_.measures.push_back(std::move(measure));
    }
  }
}

Score ScoreBuilder::build() && { return std::move(score_); }

}  // namespace omr
