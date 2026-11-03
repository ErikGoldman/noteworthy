#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "assembly.h"
#include "chords.h"
#include "layout.h"
#include "musicxml_writer.h"
#include "page_source.h"
#include "segmenter.h"
#include "staff.h"
#include "symbols.h"

namespace fs = std::filesystem;

namespace {

struct Options {
  fs::path input;
  fs::path model;
  fs::path output;
  fs::path debug_dir;
};

void usage() {
  std::cerr << "usage: noteworthy INPUT(.pdf|image) --model MODEL.onnx -o OUT.musicxml [--debug DIR]\n";
}

bool parse_args(int argc, char** argv, Options& opt) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> const char* { return i + 1 < argc ? argv[++i] : nullptr; };
    const char* v = nullptr;
    if (a == "--model" && (v = next())) {
      opt.model = v;
    } else if (a == "-o" && (v = next())) {
      opt.output = v;
    } else if (a == "--debug" && (v = next())) {
      opt.debug_dir = v;
    } else if (!a.starts_with("-") && opt.input.empty()) {
      opt.input = a;
    } else {
      return false;
    }
  }
  return !opt.input.empty() && !opt.model.empty() && !opt.output.empty();
}

cv::Mat normalize_scale(const cv::Mat& page, double target_interline, double& scale_out) {
  const double interline = omr::estimate_interline(page);
  scale_out = target_interline / interline;
  cv::Mat scaled;
  cv::resize(page, scaled, cv::Size(), scale_out, scale_out, scale_out < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
  return scaled;
}

void write_staff_debug(const fs::path& path, const cv::Mat& page, const std::vector<omr::Staff>& staves) {
  cv::Mat vis;
  cv::cvtColor(page, vis, cv::COLOR_GRAY2BGR);
  for (const auto& s : staves) {
    for (int line = 0; line < 5; ++line) {
      for (float x = s.x_min(); x < s.x_max(); x += 4) {
        cv::circle(vis, cv::Point2f(x, s.line_y(line, x)), 1, cv::Scalar(0, 0, 255), -1);
      }
    }
  }
  cv::imwrite(path.string(), vis);
}

void write_symbol_debug(const fs::path& path, const cv::Mat& page, const std::vector<omr::Staff>& staves,
                        const omr::PageSymbols& symbols, double interline) {
  cv::Mat vis;
  cv::cvtColor(page, vis, cv::COLOR_GRAY2BGR);
  vis = vis * 0.5 + cv::Scalar(127, 127, 127);
  for (const auto& sys : omr::find_systems(staves, symbols, interline)) {
    const float top = staves[sys.staves.front()].top(sys.x_start);
    const float bottom = staves[sys.staves.back()].bottom(sys.x_start);
    for (float x : sys.bounds) cv::line(vis, cv::Point2f(x, top - 20), cv::Point2f(x, bottom + 20), cv::Scalar(255, 0, 255), 2);
  }
  for (const auto& st : symbols.stems) cv::line(vis, cv::Point2f(st.x, st.top), cv::Point2f(st.x, st.bottom), cv::Scalar(0, 160, 0), 2);
  for (const auto& h : symbols.heads) {
    const cv::Scalar color = h.kind == omr::HeadKind::Filled ? cv::Scalar(0, 0, 255)
                             : h.kind == omr::HeadKind::Half ? cv::Scalar(255, 0, 0) : cv::Scalar(0, 140, 255);
    cv::circle(vis, h.center, 5, color, -1);
    cv::putText(vis, std::to_string(h.position), h.center + cv::Point2f(6, -6), cv::FONT_HERSHEY_SIMPLEX, 0.35, color, 1);
  }
  for (const auto& it : omr::build_items(symbols, interline)) {
    const std::string label = std::to_string(it.value.base) + std::string(it.value.dots, '.') + (it.beam_group ? "b" : "");
    const float y = it.is_rest() ? staves[it.page_staff].bottom(it.x) + 30 : symbols.heads[it.heads.front()].center.y + 22;
    cv::putText(vis, label, cv::Point2f(it.x - 8, y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(120, 0, 120), 1);
  }
  cv::imwrite(path.string(), vis);
}

}  // namespace

int main(int argc, char** argv) {
  Options opt;
  if (!parse_args(argc, argv, opt)) {
    usage();
    return 2;
  }
  try {
    omr::Segmenter segmenter(opt.model);
    const auto pages = omr::load_pages(opt.input);
    if (!opt.debug_dir.empty()) fs::create_directories(opt.debug_dir);
    omr::ScoreBuilder builder;
    for (size_t p = 0; p < pages.size(); ++p) {
      double scale = 1;
      const cv::Mat page = normalize_scale(pages[p], segmenter.normalized_interline(), scale);
      const omr::SymbolMaps maps = segmenter.run(page);
      const auto staves = omr::detect_staves(maps["staff_line"], segmenter.normalized_interline());
      const omr::PageSymbols symbols = omr::extract_symbols(maps, staves, segmenter.normalized_interline());
      builder.add_page(staves, symbols, segmenter.normalized_interline());
      std::cerr << "page " << p + 1 << ": scale " << scale << ", " << staves.size() << " staves, "
                << symbols.heads.size() << " noteheads\n";
      if (!opt.debug_dir.empty()) {
        const std::string stem = "page" + std::to_string(p + 1);
        cv::imwrite((opt.debug_dir / (stem + "_input.png")).string(), page);
        write_staff_debug(opt.debug_dir / (stem + "_staves.png"), page, staves);
        write_symbol_debug(opt.debug_dir / (stem + "_symbols.png"), page, staves, symbols, segmenter.normalized_interline());
        for (const auto& name : maps.names()) {
          cv::imwrite((opt.debug_dir / (stem + "_" + name + ".png")).string(), maps[name]);
        }
      }
    }
    const omr::Score score = std::move(builder).build();
    std::ofstream(opt.output) << omr::to_musicxml(score, opt.input.stem().string());
    std::cerr << score.measures.size() << " measures written to " << opt.output << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
