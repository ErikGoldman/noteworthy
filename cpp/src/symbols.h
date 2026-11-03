#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "segmenter.h"
#include "staff.h"

namespace omr {

enum class HeadKind { Filled, Half, Whole };

struct Notehead {
  cv::Point2f center;
  HeadKind kind = HeadKind::Filled;
  int staff = -1;     // index into the page's staves
  int position = 0;   // staff position: 0 = bottom line, +1 per half interline upward
};

struct Stem {
  float x = 0;
  float top = 0;
  float bottom = 0;
};

// A connected component of one symbol channel.
struct Blob {
  std::string kind;  // channel name, e.g. "sharp", "rest_8", "ts_4"
  cv::Rect box;
  cv::Point2f center;
  int area = 0;
  int staff = -1;    // nearest staff, or -1 when far from every staff
};

struct PageSymbols {
  std::vector<Notehead> heads;
  std::vector<Stem> stems;
  std::vector<Blob> flags;        // kind flag_8 / flag_16 / flag_32
  std::vector<Blob> dots;
  std::vector<Blob> accidentals;  // kind sharp / flat / natural / double_*
  std::vector<Blob> rests;        // kind rest_*
  std::vector<Blob> clefs;        // kind clef_*
  std::vector<Blob> time_digits;  // kind ts_*
  std::vector<Blob> barlines;
  std::vector<Blob> braces;
  std::vector<Blob> ties;         // ties and slurs
  cv::Mat beam_labels;            // CV_32S connected-component labels of the beam map, 0 = none
};

// Index of the staff nearest to point (x, y), measured from the staff's outer lines;
// -1 if farther than `max_interlines` from every staff.
int nearest_staff(const std::vector<Staff>& staves, cv::Point2f p, float max_interlines);

PageSymbols extract_symbols(const SymbolMaps& maps, const std::vector<Staff>& staves, double interline);

}  // namespace omr
