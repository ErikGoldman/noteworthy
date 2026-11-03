#pragma once

#include <array>
#include <vector>

#include <opencv2/core.hpp>

namespace omr {

// A five-line staff, sampled along x so skewed or slightly curved staves are followed.
class Staff {
 public:
  Staff(std::vector<float> xs, std::vector<std::array<float, 5>> ys);

  float x_min() const { return xs_.front(); }
  float x_max() const { return xs_.back(); }
  float line_y(int line, float x) const;  // line 0 = top, 4 = bottom
  float top(float x) const { return line_y(0, x); }
  float bottom(float x) const { return line_y(4, x); }
  float center_y(float x) const { return line_y(2, x); }
  float interline(float x) const { return (bottom(x) - top(x)) / 4.0f; }

  // Continuous staff position: 0 = bottom line, +1 per half interline upward.
  float position(float x, float y) const { return (bottom(x) - y) / (interline(x) / 2.0f); }
  float y_at_position(float x, float position) const { return bottom(x) - position * interline(x) / 2.0f; }

 private:
  std::vector<float> xs_;
  std::vector<std::array<float, 5>> ys_;
};

// Staff interline (distance between adjacent staff lines, px) of a raw page, from the
// most common vertical black and white run lengths.
double estimate_interline(const cv::Mat& gray);

// Staves found in a staff-line probability map (CV_8U) of a page normalized to
// `interline` px, ordered top to bottom.
std::vector<Staff> detect_staves(const cv::Mat& staff_line_prob, double interline);

}  // namespace omr
