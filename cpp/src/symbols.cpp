#include "symbols.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

#include <opencv2/imgproc.hpp>

namespace omr {
namespace {

constexpr uint8_t kPresent = 128;  // probability threshold (0..255)
constexpr float kHeadKindRadiusInterlines = 0.35f;
constexpr float kMinStemHeightInterlines = 1.5f;
constexpr float kMaxStemWidthInterlines = 0.6f;
constexpr float kStemColumnInterlines = 0.25f;
constexpr float kStemGapInterlines = 1.3f;  // a notehead's height plus margin
constexpr float kStaffReachInterlines = 6.0f;   // how far outside a staff a symbol may sit
constexpr float kStaffXReachInterlines = 4.0f;  // detected staff ends fall short of the drawn ones
constexpr float kOttavaDashGapInterlines = 1.0f;
constexpr float kRivalOverlap = 0.5f;  // box IoU above which two blobs are one symbol
// Even the reduced mid-system clefs stand >= 2.4 interlines tall; shorter clef blobs are
// specks on other symbols (time-signature digits) or the separate dots of an F clef.
constexpr float kMinClefHeightInterlines = 2.0f;
constexpr float kNoLedgerNeededInterlines = 0.8f;  // heads this close to a staff edge need no ledger line

cv::Mat binarize(const cv::Mat& prob) {
  cv::Mat mask;
  cv::threshold(prob, mask, kPresent - 1, 255, cv::THRESH_BINARY);
  return mask;
}

std::vector<Blob> components(const cv::Mat& prob, const std::string& kind, int min_area) {
  cv::Mat labels, stats, centroids;
  const int n = cv::connectedComponentsWithStats(binarize(prob), labels, stats, centroids, 8, CV_32S);
  std::vector<Blob> out;
  for (int i = 1; i < n; ++i) {
    const int area = stats.at<int>(i, cv::CC_STAT_AREA);
    if (area < min_area) continue;
    out.push_back({kind,
                   cv::Rect(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP),
                            stats.at<int>(i, cv::CC_STAT_WIDTH), stats.at<int>(i, cv::CC_STAT_HEIGHT)),
                   cv::Point2f(static_cast<float>(centroids.at<double>(i, 0)), static_cast<float>(centroids.at<double>(i, 1))),
                   area, -1});
  }
  return out;
}

float box_iou(const cv::Rect& a, const cv::Rect& b) {
  const float inter = static_cast<float>((a & b).area());
  return inter / static_cast<float>(a.area() + b.area() - inter);
}

// The network often fires several channels of one family on a symbol (an eighth rest
// also lights up rest_16 and rest_32). Of blobs of different kinds covering the same
// spot, keep the one with the most probability mass. Distinct symbols that merely
// touch overlap far less and survive.
std::vector<Blob> suppress_rival_kinds(std::vector<Blob> blobs, const SymbolMaps& maps) {
  std::vector<double> mass(blobs.size());
  for (size_t i = 0; i < blobs.size(); ++i) mass[i] = cv::sum(maps[blobs[i].kind](blobs[i].box))[0];
  std::vector<size_t> order(blobs.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return mass[a] > mass[b]; });
  std::vector<Blob> kept;
  for (size_t i : order) {
    const bool rival_kept = std::any_of(kept.begin(), kept.end(), [&](const Blob& k) {
      return k.kind != blobs[i].kind && box_iou(k.box, blobs[i].box) > kRivalOverlap;
    });
    if (!rival_kept) kept.push_back(std::move(blobs[i]));
  }
  return kept;
}

// Components of one family of mutually exclusive channels.
void append_components(std::vector<Blob>& out, const SymbolMaps& maps, std::initializer_list<const char*> kinds,
                       int min_area) {
  std::vector<Blob> family;
  for (const char* kind : kinds) {
    auto blobs = components(maps[kind], kind, min_area);
    family.insert(family.end(), blobs.begin(), blobs.end());
  }
  family = suppress_rival_kinds(std::move(family), maps);
  out.insert(out.end(), family.begin(), family.end());
}

float mean_in_disk(const cv::Mat& prob, cv::Point2f c, float radius) {
  const int r = static_cast<int>(std::ceil(radius));
  const cv::Rect roi = cv::Rect(cv::Point(c) - cv::Point(r, r), cv::Size(2 * r + 1, 2 * r + 1)) & cv::Rect(0, 0, prob.cols, prob.rows);
  if (roi.empty()) return 0;
  cv::Mat disk = cv::Mat::zeros(roi.size(), CV_8U);
  cv::circle(disk, cv::Point(c) - roi.tl(), r, 255, -1);
  return static_cast<float>(cv::mean(prob(roi), disk)[0]);
}

HeadKind classify_head(const SymbolMaps& maps, cv::Point2f c, double interline) {
  const float r = static_cast<float>(interline * kHeadKindRadiusInterlines);
  const float filled = mean_in_disk(maps["head_filled"], c, r);
  const float half = mean_in_disk(maps["head_half"], c, r);
  const float whole = mean_in_disk(maps["head_whole"], c, r);
  if (filled >= half && filled >= whole) return HeadKind::Filled;
  return half >= whole ? HeadKind::Half : HeadKind::Whole;
}

// Stem fragments: noteheads in the middle of a chord, and faint print, split one stem
// into several pieces along the same column.
std::vector<Stem> merge_collinear(std::vector<Stem> pieces, double interline) {
  std::sort(pieces.begin(), pieces.end(), [](const Stem& a, const Stem& b) { return a.top < b.top; });
  std::vector<Stem> merged;
  for (const Stem& p : pieces) {
    auto joinable = std::find_if(merged.begin(), merged.end(), [&](const Stem& m) {
      return std::abs(m.x - p.x) < kStemColumnInterlines * interline && p.top - m.bottom < kStemGapInterlines * interline;
    });
    if (joinable == merged.end()) {
      merged.push_back(p);
    } else {
      joinable->bottom = std::max(joinable->bottom, p.bottom);
    }
  }
  return merged;
}

std::vector<Stem> extract_stems(const cv::Mat& prob, double interline) {
  std::vector<Stem> pieces;
  for (const Blob& b : components(prob, "stem", 1)) {
    if (b.box.width > kMaxStemWidthInterlines * interline) continue;
    pieces.push_back({b.center.x, static_cast<float>(b.box.y), static_cast<float>(b.box.y + b.box.height)});
  }
  std::vector<Stem> stems;
  for (const Stem& s : merge_collinear(std::move(pieces), interline)) {
    if (s.bottom - s.top >= kMinStemHeightInterlines * interline) stems.push_back(s);
  }
  return stems;
}

// Whether a notehead at `c` can belong to `staff`: heads well outside a staff hang on
// ledger lines, so the first ledger line beyond the staff edge must be drawn there.
bool reaches_by_ledgers(const Staff& staff, cv::Point2f c, const cv::Mat& ledger_prob) {
  const float il = staff.interline(c.x);
  const float top = staff.top(c.x), bottom = staff.bottom(c.x);
  if (c.y > top - kNoLedgerNeededInterlines * il && c.y < bottom + kNoLedgerNeededInterlines * il) return true;
  const float ledger_y = c.y < top ? top - il : bottom + il;
  const int half = static_cast<int>(std::ceil(il * 0.6f));
  const cv::Rect probe = cv::Rect(static_cast<int>(c.x) - half, static_cast<int>(ledger_y) - 2, 2 * half + 1, 5) &
                         cv::Rect(0, 0, ledger_prob.cols, ledger_prob.rows);
  if (probe.empty()) return false;
  double max_prob = 0;
  cv::minMaxLoc(ledger_prob(probe), nullptr, &max_prob);
  return max_prob >= kPresent;
}

// Nearest staff that the head can reach by ledger lines; else simply the nearest staff.
// Notes high above a bass staff often sit closer to the treble staff than their own.
int notehead_staff(const std::vector<Staff>& staves, cv::Point2f c, const cv::Mat& ledger_prob) {
  int best = -1;
  float best_dist = std::numeric_limits<float>::max();
  for (size_t i = 0; i < staves.size(); ++i) {
    const Staff& s = staves[i];
    if (c.x < s.x_min() - kStaffXReachInterlines * s.interline(c.x) || c.x > s.x_max() + kStaffXReachInterlines * s.interline(c.x)) continue;
    const float dist = std::max({s.top(c.x) - c.y, c.y - s.bottom(c.x), 0.0f});
    if (dist > kStaffReachInterlines * s.interline(c.x) || !reaches_by_ledgers(s, c, ledger_prob)) continue;
    if (dist < best_dist) {
      best_dist = dist;
      best = static_cast<int>(i);
    }
  }
  return best >= 0 ? best : nearest_staff(staves, c, kStaffReachInterlines);
}

void assign_staff(std::vector<Blob>& blobs, const std::vector<Staff>& staves) {
  for (Blob& b : blobs) b.staff = nearest_staff(staves, b.center, kStaffReachInterlines);
}

// Time-signature digits. The two numbers of a signature often touch at the middle line,
// and a digit's mask can carry specks of another digit's channel, so digits are the
// components of all digit channels together, cut apart at each staff's middle line,
// and each is named by the channel holding most of its probability mass.
std::vector<Blob> extract_time_digits(const SymbolMaps& maps, const std::vector<Staff>& staves, double interline, int min_area) {
  static constexpr const char* kKinds[] = {"ts_0", "ts_1", "ts_2", "ts_3", "ts_4", "ts_5", "ts_6",
                                           "ts_7", "ts_8", "ts_9", "ts_common", "ts_cut"};
  cv::Mat any_digit = cv::Mat::zeros(maps.size(), CV_8U);
  for (const char* kind : kKinds) any_digit = cv::max(any_digit, maps[kind]);
  any_digit = binarize(any_digit);
  const int margin = static_cast<int>(kStaffXReachInterlines * interline);
  for (const Staff& staff : staves) {
    const int x_end = std::min(any_digit.cols, static_cast<int>(staff.x_max()) + margin);
    for (int x = std::max(0, static_cast<int>(staff.x_min()) - margin); x < x_end; ++x) {
      const int y = static_cast<int>(std::lround(staff.center_y(static_cast<float>(x))));
      if (y >= 1 && y + 1 < any_digit.rows) any_digit.rowRange(y - 1, y + 2).col(x).setTo(0);
    }
  }
  cv::Mat labels, stats, centroids;
  const int n = cv::connectedComponentsWithStats(any_digit, labels, stats, centroids, 8, CV_32S);
  std::vector<Blob> out;
  for (int i = 1; i < n; ++i) {
    if (stats.at<int>(i, cv::CC_STAT_AREA) < min_area) continue;
    const cv::Rect box(stats.at<int>(i, cv::CC_STAT_LEFT), stats.at<int>(i, cv::CC_STAT_TOP), stats.at<int>(i, cv::CC_STAT_WIDTH),
                       stats.at<int>(i, cv::CC_STAT_HEIGHT));
    const cv::Mat mask = labels(box) == i;
    const char* best = kKinds[0];
    double best_mass = -1;
    for (const char* kind : kKinds) {
      const double mass = cv::sum(maps[kind](box) & mask)[0];
      if (mass > best_mass) {
        best_mass = mass;
        best = kind;
      }
    }
    out.push_back({best, box,
                   cv::Point2f(static_cast<float>(centroids.at<double>(i, 0)), static_cast<float>(centroids.at<double>(i, 1))),
                   stats.at<int>(i, cv::CC_STAT_AREA), -1});
  }
  return out;
}

// An ottava's number and the dashes of its extension line are separate marks; closing
// the gaps along the line makes one blob spanning every note it affects.
std::vector<Blob> extract_ottavas(const cv::Mat& prob, double interline, int min_area) {
  const int gap = static_cast<int>(std::ceil(kOttavaDashGapInterlines * interline));
  cv::Mat joined;
  cv::morphologyEx(binarize(prob), joined, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(gap, 1)));
  return components(joined, "ottava", min_area);
}

}  // namespace

int nearest_staff(const std::vector<Staff>& staves, cv::Point2f p, float max_interlines) {
  int best = -1;
  float best_dist = std::numeric_limits<float>::max();
  for (size_t i = 0; i < staves.size(); ++i) {
    const Staff& s = staves[i];
    if (p.x < s.x_min() - kStaffXReachInterlines * s.interline(p.x) || p.x > s.x_max() + kStaffXReachInterlines * s.interline(p.x)) continue;
    const float top = s.top(p.x), bottom = s.bottom(p.x);
    const float dist = p.y < top ? top - p.y : (p.y > bottom ? p.y - bottom : 0.0f);
    if (dist < best_dist && dist <= max_interlines * s.interline(p.x)) {
      best_dist = dist;
      best = static_cast<int>(i);
    }
  }
  return best;
}

PageSymbols extract_symbols(const SymbolMaps& maps, const std::vector<Staff>& staves, double interline) {
  PageSymbols out;
  const int tiny = static_cast<int>(interline * interline * 0.05);  // ignore specks

  for (const Blob& c : components(maps["head_center"], "head_center", tiny)) {
    Notehead h;
    h.center = c.center;
    h.kind = classify_head(maps, c.center, interline);
    h.staff = notehead_staff(staves, c.center, maps["ledger_line"]);
    if (h.staff < 0) continue;
    h.position = static_cast<int>(std::lround(staves[h.staff].position(c.center.x, c.center.y)));
    out.heads.push_back(h);
  }
  out.stems = extract_stems(maps["stem"], interline);
  cv::connectedComponents(binarize(maps["beam"]), out.beam_labels, 8, CV_32S);
  append_components(out.flags, maps, {"flag_8", "flag_16", "flag_32"}, tiny);
  append_components(out.dots, maps, {"aug_dot"}, 1);
  append_components(out.accidentals, maps, {"sharp", "flat", "natural", "double_sharp", "double_flat"}, tiny);
  append_components(out.rests, maps, {"rest_whole", "rest_half", "rest_quarter", "rest_8", "rest_16", "rest_32"}, tiny);
  append_components(out.clefs, maps, {"clef_g", "clef_f", "clef_c"}, tiny);
  std::erase_if(out.clefs, [&](const Blob& c) { return c.box.height < kMinClefHeightInterlines * interline; });
  out.time_digits = extract_time_digits(maps, staves, interline, tiny);
  append_components(out.barlines, maps, {"barline"}, tiny);
  append_components(out.braces, maps, {"brace"}, tiny);
  append_components(out.ties, maps, {"tie_slur"}, tiny);
  if (maps.has("tuplet_3")) append_components(out.tuplets, maps, {"tuplet_3", "tuplet_other"}, tiny);
  if (maps.has("ottava")) out.ottavas = extract_ottavas(maps["ottava"], interline, tiny);
  for (auto* group : {&out.flags, &out.dots, &out.accidentals, &out.rests, &out.clefs, &out.time_digits, &out.barlines,
                      &out.tuplets, &out.ottavas}) {
    assign_staff(*group, staves);
  }
  return out;
}

}  // namespace omr
