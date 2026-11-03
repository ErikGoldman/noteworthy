#include "chords.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace omr {
namespace {

// All distances in interlines.
constexpr float kStemReach = 0.9f;          // max horizontal head-centre-to-stem distance
constexpr float kStemEndSlack = 0.6f;       // head may sit this far beyond a stem end
constexpr float kHeadHalfWidth = 0.6f;
constexpr float kBeamSearchBeyondEnd = 0.5f;
constexpr float kBeamSearchAlongStem = 3.5f;
constexpr float kMinBeamRun = 0.2f;
constexpr float kBeamSampleHalfWidth = 0.2f;
constexpr float kBeamPitch = 0.75f;         // beam thickness + gap: merged beams count by length
constexpr float kFlagSearch = 1.5f;
constexpr float kFlagBeyondStem = 2.0f;
constexpr float kDotMinGap = 0.2f;
constexpr float kDotMaxGap = 2.0f;
constexpr float kDotMaxDy = 0.8f;
constexpr float kStemlessChordDx = 0.6f;
constexpr float kSecondDisplacementDx = 1.6f;  // a head displaced beside its neighbour a step away
constexpr float kTupletBeamOverhang = 1.0f;    // a tuplet number may sit slightly past its group's ends
constexpr int kNotesInTriplet = 3;

int find_stem(const Notehead& h, const std::vector<Stem>& stems, float il) {
  int best = -1;
  float best_dx = kStemReach * il;
  for (size_t i = 0; i < stems.size(); ++i) {
    const Stem& s = stems[i];
    const float dx = std::abs(s.x - h.center.x);
    if (dx > best_dx) continue;
    if (h.center.y < s.top - kStemEndSlack * il || h.center.y > s.bottom + kStemEndSlack * il) continue;
    best_dx = dx;
    best = static_cast<int>(i);
  }
  return best;
}

struct BeamInfo {
  int count = 0;
  int group = 0;
};

// Counts beams crossing the stem near its free end. Touching beams merge into one run
// in the mask, so a run's length decides how many beams it holds.
BeamInfo beams_at(const cv::Mat& labels, const Stem& stem, int dir, float il) {
  const float free_end = dir > 0 ? stem.top : stem.bottom;
  const float toward_heads = dir > 0 ? 1.0f : -1.0f;
  const int y_a = static_cast<int>(free_end - toward_heads * kBeamSearchBeyondEnd * il);
  const int y_b = static_cast<int>(free_end + toward_heads * kBeamSearchAlongStem * il);
  const int y0 = std::clamp(std::min(y_a, y_b), 0, labels.rows - 1);
  const int y1 = std::clamp(std::max(y_a, y_b), 0, labels.rows - 1);

  BeamInfo best;
  // A beam stops at the stem's outer edge, so sample a few columns either side of it.
  const int half_width = std::max(1, static_cast<int>(std::lround(kBeamSampleHalfWidth * il)));
  for (int dx = -half_width; dx <= half_width; ++dx) {
    const int x = std::clamp(static_cast<int>(std::lround(stem.x)) + dx, 0, labels.cols - 1);
    BeamInfo info;
    int run = 0, run_label = 0;
    auto close_run = [&]() {
      if (run >= kMinBeamRun * il) {
        info.count += std::max(1, static_cast<int>(std::lround((run + 0.25f * il) / (kBeamPitch * il))));
        if (!info.group) info.group = run_label;
      }
      run = 0;
    };
    for (int y = y0; y <= y1; ++y) {
      const int label = labels.at<int>(y, x);
      if (label) {
        if (!run) run_label = label;
        ++run;
      } else {
        close_run();
      }
    }
    close_run();
    if (info.count > best.count) best = info;
  }
  return best;
}

int flag_count(const std::vector<Blob>& flags, const Stem& stem, int dir, float il) {
  for (const Blob& f : flags) {
    if (f.box.x < stem.x - 0.4f * il || f.box.x > stem.x + 0.6f * il) continue;
    // The flag glyph covers the stem's end, so the detected stem stops short of the flag's tip.
    const bool at_end = dir > 0 ? (f.box.y >= stem.top - kFlagBeyondStem * il && f.box.y <= stem.top + kFlagSearch * il)
                                : (f.box.br().y <= stem.bottom + kFlagBeyondStem * il && f.box.br().y >= stem.bottom - kFlagSearch * il);
    if (!at_end) continue;
    if (f.kind == "flag_8") return 1;
    if (f.kind == "flag_16") return 2;
    return 3;
  }
  return 0;
}

// Augmentation dots right of the item: 0, 1 or 2 (distinct dot columns).
int dot_count(const std::vector<Blob>& dots, const std::vector<cv::Point2f>& anchors, float right_edge, float il) {
  std::vector<float> columns;
  for (const Blob& d : dots) {
    const float gap = d.center.x - right_edge;
    if (gap < kDotMinGap * il || gap > kDotMaxGap * il) continue;
    const bool beside = std::any_of(anchors.begin(), anchors.end(), [&](cv::Point2f a) { return std::abs(d.center.y - a.y) < kDotMaxDy * il; });
    if (!beside) continue;
    const bool new_column = std::none_of(columns.begin(), columns.end(), [&](float x) { return std::abs(x - d.center.x) < 0.3f * il; });
    if (new_column) columns.push_back(d.center.x);
  }
  return std::min<int>(2, static_cast<int>(columns.size()));
}

int base_for(HeadKind kind, int beams_or_flags) {
  switch (kind) {
    case HeadKind::Whole: return 1;
    case HeadKind::Half: return 2;
    case HeadKind::Filled: return 4 << std::min(beams_or_flags, 3);
  }
  return 4;
}

int rest_base(const std::string& kind) {
  static const std::map<std::string, int> bases = {{"rest_whole", 1}, {"rest_half", 2}, {"rest_quarter", 4},
                                                   {"rest_8", 8},     {"rest_16", 16},  {"rest_32", 32}};
  return bases.at(kind);
}

std::vector<cv::Point2f> head_centers(const PageSymbols& symbols, const std::vector<int>& heads) {
  std::vector<cv::Point2f> pts;
  for (int h : heads) pts.push_back(symbols.heads[h].center);
  return pts;
}

void add_stemmed_chords(const PageSymbols& symbols, float il, std::vector<int>& stem_of_head, std::vector<RhythmItem>& items) {
  std::map<int, std::vector<int>> heads_by_stem;
  for (size_t h = 0; h < symbols.heads.size(); ++h) {
    stem_of_head[h] = find_stem(symbols.heads[h], symbols.stems, il);
    if (stem_of_head[h] >= 0) heads_by_stem[stem_of_head[h]].push_back(static_cast<int>(h));
  }
  for (auto& [stem_index, heads] : heads_by_stem) {
    const Stem& stem = symbols.stems[stem_index];
    float min_y = 1e9f, max_x = -1e9f;
    for (int h : heads) {
      min_y = std::min(min_y, symbols.heads[h].center.y);
      max_x = std::max(max_x, symbols.heads[h].center.x);
    }
    RhythmItem item;
    item.heads = heads;
    item.stem_dir = stem.top < min_y - 0.5f * il ? 1 : -1;
    item.x = stem.x - item.stem_dir * kHeadHalfWidth * il;
    // Chords take the page staff of the head nearest the stem's head end.
    const int anchor = item.stem_dir > 0 ? *std::max_element(heads.begin(), heads.end(), [&](int a, int b) {
      return symbols.heads[a].center.y < symbols.heads[b].center.y;
    }) : *std::min_element(heads.begin(), heads.end(), [&](int a, int b) {
      return symbols.heads[a].center.y < symbols.heads[b].center.y;
    });
    item.page_staff = symbols.heads[anchor].staff;
    const BeamInfo beams = beams_at(symbols.beam_labels, stem, item.stem_dir, il);
    item.beam_group = beams.group;
    const int strokes = beams.count ? beams.count : flag_count(symbols.flags, stem, item.stem_dir, il);
    item.value.base = base_for(symbols.heads[anchor].kind, strokes);
    item.value.dots = dot_count(symbols.dots, head_centers(symbols, heads), max_x + kHeadHalfWidth * il, il);
    items.push_back(std::move(item));
  }
}

void add_stemless_chords(const PageSymbols& symbols, float il, const std::vector<int>& stem_of_head,
                         std::vector<RhythmItem>& items) {
  std::vector<int> loose;
  for (size_t h = 0; h < symbols.heads.size(); ++h) {
    if (stem_of_head[h] < 0) loose.push_back(static_cast<int>(h));
  }
  std::sort(loose.begin(), loose.end(), [&](int a, int b) { return symbols.heads[a].center.x < symbols.heads[b].center.x; });
  std::vector<bool> used(loose.size(), false);
  for (size_t i = 0; i < loose.size(); ++i) {
    if (used[i]) continue;
    const Notehead& first = symbols.heads[loose[i]];
    RhythmItem item;
    for (size_t j = i; j < loose.size(); ++j) {
      const Notehead& h = symbols.heads[loose[j]];
      const float dx = std::abs(h.center.x - first.center.x);
      const bool stacked = dx < kStemlessChordDx * il;
      const bool displaced_second = dx < kSecondDisplacementDx * il && std::abs(h.position - first.position) == 1;
      if (!used[j] && h.staff == first.staff && (stacked || displaced_second)) {
        item.heads.push_back(loose[j]);
        used[j] = true;
      }
    }
    float sum_x = 0, max_x = -1e9f;
    for (int h : item.heads) {
      sum_x += symbols.heads[h].center.x;
      max_x = std::max(max_x, symbols.heads[h].center.x);
    }
    item.x = sum_x / item.heads.size();
    item.page_staff = first.staff;
    item.value.base = base_for(first.kind, 0);
    item.value.dots = dot_count(symbols.dots, head_centers(symbols, item.heads), max_x + kHeadHalfWidth * il, il);
    items.push_back(std::move(item));
  }
}

void add_rests(const PageSymbols& symbols, float il, std::vector<RhythmItem>& items) {
  for (const Blob& r : symbols.rests) {
    if (r.staff < 0) continue;
    RhythmItem item;
    item.x = r.center.x;
    item.page_staff = r.staff;
    item.rest_kind = r.kind;
    item.value.base = rest_base(r.kind);
    item.value.dots = dot_count(symbols.dots, {r.center}, static_cast<float>(r.box.br().x), il);
    items.push_back(std::move(item));
  }
}

float vertical_distance(const RhythmItem& item, const PageSymbols& symbols, cv::Point2f p) {
  if (item.is_rest()) return 0;
  float d = std::numeric_limits<float>::max();
  for (int h : item.heads) d = std::min(d, std::abs(symbols.heads[h].center.y - p.y));
  return d;
}

// Items a tuplet number belongs to: the beam group it sits over, else the notes and
// rests nearest to it on its staff (a bracketed or bare tuplet).
std::vector<RhythmItem*> tuplet_members(const Blob& number, const PageSymbols& symbols, float il, std::vector<RhythmItem>& items) {
  std::map<int, std::vector<RhythmItem*>> groups;
  for (RhythmItem& it : items) {
    if (it.beam_group && it.page_staff == number.staff) groups[it.beam_group].push_back(&it);
  }
  std::vector<RhythmItem*> best;
  float best_dy = std::numeric_limits<float>::max();
  for (auto& [label, members] : groups) {
    const auto [lo, hi] = std::minmax_element(members.begin(), members.end(), [](auto* a, auto* b) { return a->x < b->x; });
    if (number.center.x < (*lo)->x - kTupletBeamOverhang * il || number.center.x > (*hi)->x + kTupletBeamOverhang * il) continue;
    float dy = std::numeric_limits<float>::max();
    for (const RhythmItem* m : members) dy = std::min(dy, vertical_distance(*m, symbols, number.center));
    if (dy < best_dy) {
      best_dy = dy;
      best = members;
    }
  }
  if (!best.empty()) return best;
  std::vector<RhythmItem*> nearest;
  for (RhythmItem& it : items) {
    if (it.page_staff == number.staff && !it.value.in_tuplet()) nearest.push_back(&it);
  }
  const size_t n = std::min<size_t>(kNotesInTriplet, nearest.size());
  std::partial_sort(nearest.begin(), nearest.begin() + n, nearest.end(),
                    [&](auto* a, auto* b) { return std::abs(a->x - number.center.x) < std::abs(b->x - number.center.x); });
  nearest.resize(n);
  return nearest;
}

// Only 3 and 6 (both 2/3 time) are told apart; other tuplet numbers are left alone.
void apply_tuplets(const PageSymbols& symbols, float il, std::vector<RhythmItem>& items) {
  for (const Blob& number : symbols.tuplets) {
    if (number.kind != "tuplet_3" || number.staff < 0) continue;
    for (RhythmItem* it : tuplet_members(number, symbols, il, items)) {
      it->value.tuplet_actual = 3;
      it->value.tuplet_normal = 2;
    }
  }
}

}  // namespace

std::vector<RhythmItem> build_items(const PageSymbols& symbols, double interline) {
  const float il = static_cast<float>(interline);
  std::vector<RhythmItem> items;
  std::vector<int> stem_of_head(symbols.heads.size(), -1);
  add_stemmed_chords(symbols, il, stem_of_head, items);
  add_stemless_chords(symbols, il, stem_of_head, items);
  add_rests(symbols, il, items);
  apply_tuplets(symbols, il, items);
  std::sort(items.begin(), items.end(), [](const RhythmItem& a, const RhythmItem& b) { return a.x < b.x; });
  return items;
}

}  // namespace omr
