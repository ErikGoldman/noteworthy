#include "staff.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace omr {
namespace {

constexpr int kMaxRunLength = 100;
constexpr double kStripWidthInterlines = 4.0;
constexpr double kStripStepInterlines = 2.0;
constexpr float kLinePresence = 0.35f;        // fraction of a strip's width a line must cover
constexpr float kGapTolerance = 0.3f;         // allowed relative deviation of line gaps
constexpr float kLinkToleranceInterlines = 0.5f;  // a staff misread one line off must not join
constexpr float kDuplicateStaffInterlines = 3.0f; // real staves are farther apart than this
constexpr double kMinStaffWidthInterlines = 10.0;

int histogram_mode(const std::vector<int>& hist, int from) {
  return static_cast<int>(std::max_element(hist.begin() + from, hist.end()) - hist.begin());
}

struct StripStaff {
  float x;
  std::array<float, 5> ys;
  float center() const { return ys[2]; }
};

struct LinePeak {
  float y;         // sub-pixel row
  float strength;  // mean staff-line probability across the strip, 0..1
};

// Row-profile peaks of one vertical strip of the staff-line probability map.
std::vector<LinePeak> strip_line_peaks(const cv::Mat& prob, const cv::Rect& strip) {
  cv::Mat profile;
  cv::reduce(prob(strip), profile, 1, cv::REDUCE_AVG, CV_32F);  // rows x 1, in 0..255
  profile /= 255.0f;
  std::vector<LinePeak> peaks;
  for (int y = 1; y + 1 < profile.rows; ++y) {
    const float v = profile.at<float>(y);
    if (v < kLinePresence || v < profile.at<float>(y - 1) || v <= profile.at<float>(y + 1)) continue;
    // Centre of mass of the line's rows, for sub-pixel accuracy on thick lines.
    float sum = 0, wsum = 0;
    for (int dy = -2; dy <= 2; ++dy) {
      const int yy = std::clamp(y + dy, 0, profile.rows - 1);
      const float w = profile.at<float>(yy);
      sum += w * yy;
      wsum += w;
    }
    peaks.push_back({sum / wsum, v});
  }
  return peaks;
}

// Maximal runs of evenly spaced peaks (peaks too close to the previous one are skipped).
std::vector<std::vector<LinePeak>> evenly_spaced_runs(const std::vector<LinePeak>& peaks, double interline) {
  std::vector<std::vector<LinePeak>> runs;
  std::vector<LinePeak> run;
  for (const LinePeak& p : peaks) {
    if (!run.empty()) {
      const float gap = p.y - run.back().y;
      if (gap < interline * (1 - kGapTolerance)) {
        if (p.strength > run.back().strength) run.back() = p;  // two peaks for one line
        continue;
      }
      if (gap > interline * (1 + kGapTolerance)) {
        runs.push_back(std::move(run));
        run.clear();
      }
    }
    run.push_back(p);
  }
  if (!run.empty()) runs.push_back(std::move(run));
  return runs;
}

// One staff per run of at least five lines. A longer run is a staff plus line-like
// neighbours (e.g. a beam along the top line), so keep the five lines whose weakest
// member is strongest: real staff lines are uniformly confident.
std::vector<std::array<float, 5>> group_into_staves(const std::vector<LinePeak>& peaks, double interline) {
  std::vector<std::array<float, 5>> staves;
  for (const auto& run : evenly_spaced_runs(peaks, interline)) {
    if (run.size() < 5) continue;
    size_t best = 0;
    float best_min = -1;
    for (size_t i = 0; i + 5 <= run.size(); ++i) {
      float weakest = 1;
      for (size_t k = i; k < i + 5; ++k) weakest = std::min(weakest, run[k].strength);
      if (weakest > best_min) {
        best_min = weakest;
        best = i;
      }
    }
    std::array<float, 5> ys;
    for (int k = 0; k < 5; ++k) ys[k] = run[best + k].y;
    staves.push_back(ys);
  }
  return staves;
}

std::vector<StripStaff> find_strip_staves(const cv::Mat& prob, double interline) {
  const int strip_w = static_cast<int>(interline * kStripWidthInterlines);
  const int step = static_cast<int>(interline * kStripStepInterlines);
  std::vector<StripStaff> found;
  for (int x0 = 0; x0 + strip_w <= prob.cols; x0 += step) {
    const auto peaks = strip_line_peaks(prob, cv::Rect(x0, 0, strip_w, prob.rows));
    for (const auto& ys : group_into_staves(peaks, interline)) found.push_back({x0 + strip_w / 2.0f, ys});
  }
  return found;
}

// Chains per-strip detections left to right into page-wide staves.
std::vector<std::vector<StripStaff>> link_strips(std::vector<StripStaff> strips, double interline) {
  std::sort(strips.begin(), strips.end(), [](const StripStaff& a, const StripStaff& b) { return a.x < b.x; });
  std::vector<std::vector<StripStaff>> chains;
  for (const StripStaff& s : strips) {
    std::vector<StripStaff>* best = nullptr;
    float best_dist = static_cast<float>(interline * kLinkToleranceInterlines);
    for (auto& chain : chains) {
      if (chain.back().x >= s.x) continue;  // already has a sample in this strip
      const float d = std::abs(chain.back().center() - s.center());
      if (d < best_dist) {
        best_dist = d;
        best = &chain;
      }
    }
    if (best) {
      best->push_back(s);
    } else {
      chains.push_back({s});
    }
  }
  return chains;
}

float chain_center_at(const std::vector<StripStaff>& chain, float x) {
  const auto& nearest = *std::min_element(chain.begin(), chain.end(), [x](const StripStaff& a, const StripStaff& b) {
    return std::abs(a.x - x) < std::abs(b.x - x);
  });
  return nearest.center();
}

// Where thick beams run along a staff, some strips read a staff shifted by one line.
// Those form short chains next to the real one; keep only the best supported of
// chains that overlap.
std::vector<std::vector<StripStaff>> drop_duplicate_chains(std::vector<std::vector<StripStaff>> chains, double interline) {
  std::sort(chains.begin(), chains.end(), [](const auto& a, const auto& b) { return a.size() > b.size(); });
  std::vector<std::vector<StripStaff>> kept;
  for (auto& chain : chains) {
    const bool duplicate = std::any_of(kept.begin(), kept.end(), [&](const auto& k) {
      return std::any_of(chain.begin(), chain.end(), [&](const StripStaff& s) {
        return s.x >= k.front().x && s.x <= k.back().x &&
               std::abs(chain_center_at(k, s.x) - s.center()) < kDuplicateStaffInterlines * interline;
      });
    });
    if (!duplicate) kept.push_back(std::move(chain));
  }
  return kept;
}

}  // namespace

Staff::Staff(std::vector<float> xs, std::vector<std::array<float, 5>> ys) : xs_(std::move(xs)), ys_(std::move(ys)) {
  if (xs_.empty() || xs_.size() != ys_.size()) throw std::invalid_argument("Staff needs matching, non-empty samples");
}

float Staff::line_y(int line, float x) const {
  if (x <= xs_.front()) return ys_.front()[line];
  if (x >= xs_.back()) return ys_.back()[line];
  const size_t k = std::upper_bound(xs_.begin(), xs_.end(), x) - xs_.begin();
  const float t = (x - xs_[k - 1]) / (xs_[k] - xs_[k - 1]);
  return ys_[k - 1][line] * (1 - t) + ys_[k][line] * t;
}

double estimate_interline(const cv::Mat& gray) {
  cv::Mat binary;
  cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  std::vector<int> black(kMaxRunLength + 1, 0), white(kMaxRunLength + 1, 0);
  for (int x = 0; x < binary.cols; ++x) {
    int run = 0;
    bool ink = binary.at<uint8_t>(0, x) != 0;
    for (int y = 0; y < binary.rows; ++y) {
      const bool v = binary.at<uint8_t>(y, x) != 0;
      if (v == ink) {
        ++run;
        continue;
      }
      if (run <= kMaxRunLength) (ink ? black : white)[run]++;
      ink = v;
      run = 1;
    }
  }
  // Staff-line thickness is the commonest black run; the commonest white run that is
  // longer than it is the space between lines.
  const int thickness = histogram_mode(black, 1);
  const int space = histogram_mode(white, thickness + 1);
  return thickness + space;
}

std::vector<Staff> detect_staves(const cv::Mat& staff_line_prob, double interline) {
  std::vector<Staff> staves;
  for (auto& chain : drop_duplicate_chains(link_strips(find_strip_staves(staff_line_prob, interline), interline), interline)) {
    if (chain.back().x - chain.front().x < kMinStaffWidthInterlines * interline) continue;
    std::vector<float> xs;
    std::vector<std::array<float, 5>> ys;
    for (const StripStaff& s : chain) {
      xs.push_back(s.x);
      ys.push_back(s.ys);
    }
    staves.emplace_back(std::move(xs), std::move(ys));
  }
  std::sort(staves.begin(), staves.end(),
            [](const Staff& a, const Staff& b) { return a.center_y(a.x_min()) < b.center_y(b.x_min()); });
  return staves;
}

}  // namespace omr
