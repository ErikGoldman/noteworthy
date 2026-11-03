#pragma once

#include <string>

#include "score.h"

namespace omr {

std::string to_musicxml(const Score& score, const std::string& title);

}  // namespace omr
