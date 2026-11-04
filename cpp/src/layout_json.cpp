#include "layout_json.h"

#include <sstream>

namespace omr {

std::string to_layout_json(const Score& score) {
  std::ostringstream out;
  out << "{\"systems\": [";
  const char* separator = "";
  for (size_t i = 0; i < score.measures.size(); ++i) {
    const LayoutBreak& layout = score.measures[i].layout;
    if (!layout.new_system) continue;
    out << separator << "\n  {\"page\": " << layout.page << ", \"first_measure\": " << i << ", \"staves\": [";
    for (size_t s = 0; s < layout.staves.size(); ++s) {
      out << (s ? ", " : "") << "[" << layout.staves[s].top << ", " << layout.staves[s].bottom << "]";
    }
    out << "]}";
    separator = ",";
  }
  out << "\n]}\n";
  return out.str();
}

}  // namespace omr
