#include "assembly.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

#include "chords.h"
#include "key_signature.h"
#include "layout.h"
#include "rhythm.h"
#include "system_reader.h"

namespace omr {
namespace {

// Distances in interlines.
constexpr float kTieEndReach = 1.2f;
constexpr float kTieVerticalReach = 1.5f;
constexpr float kHeadHalfWidth = 0.6f;

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
    out.push_back({it, staff, it->value, measure_rest ? measure_length : it->value.duration()});
  }
  return out;
}

std::vector<StaffPlacement> staff_placements(const std::vector<Staff>& staves, const System& sys, double scale) {
  std::vector<StaffPlacement> out;
  const float mid_x = (sys.x_start + sys.x_end) / 2;
  for (int s : sys.staves) {
    out.push_back({static_cast<float>(staves[s].top(mid_x) / scale), static_cast<float>(staves[s].bottom(mid_x) / scale)});
  }
  return out;
}

}  // namespace

// Key and time changes drawn at the start of measure k of a system.
void ScoreBuilder::apply_attribute_changes(const SystemReader& reader, int k, Measure& measure) {
  const std::optional<int> key = reader.key_signature(k, key_fifths_.value_or(0), clefs_);
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

void ScoreBuilder::add_page(const InputPage& page, const std::vector<Staff>& staves, const PageSymbols& symbols, double interline) {
  const std::vector<RhythmItem> items = build_items(symbols, staves, interline);
  bool first_on_page = true;
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
      measure.layout.new_system = k == 0;
      measure.layout.page = page.number;
      measure.layout.scale = page.scale;
      if (k == 0) measure.layout.staves = staff_placements(staves, sys, page.scale);
      if (std::exchange(first_on_page, false)) measure.layout.new_page = page.number;
      apply_attribute_changes(reader, k, measure);
      const Rational length = time_->measure_length();
      std::vector<MeasureItem> mitems = measure_items(sys_items, sys, k, length);
      drop_false_tuplets(mitems, length, interline);
      imply_tuplets(mitems, length, interline);
      const std::vector<Timing> timing = solve_measure(mitems, {length, sys.bounds[k], sys.bounds[k + 1]}, interline);
      const std::map<int, Pitch> pitch_of = resolve_pitches(mitems, sys, reader, explicit_alter, symbols);
      record_clef_changes(reader, k, mitems, timing, length, measure);
      HeadSlots slots;
      for (size_t i = 0; i < mitems.size(); ++i) {
        const RhythmItem& it = *mitems[i].item;
        Event e;
        e.onset = timing[i].onset;
        e.duration = mitems[i].duration;
        e.value = mitems[i].duration == mitems[i].value.duration() ? mitems[i].value : NoteValue{1, 0};
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
