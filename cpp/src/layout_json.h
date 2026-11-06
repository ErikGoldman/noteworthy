#pragma once

#include <string>

#include "score.h"

namespace omr {

// Where each output system lies on the input pages, as JSON:
//   {"systems": [{"page": 1, "scale": 2.2, "first_measure": 0, "staves": [[top, bottom], ...]}, ...]}
// Measures count from 0 in output order; staves are listed by staff number, with y in
// input-page pixels; scale takes those to the page as the recognizer read it, as in --debug.
// Lets evaluation relate recognized notes to regions of a page.
std::string to_layout_json(const Score& score);

}  // namespace omr
