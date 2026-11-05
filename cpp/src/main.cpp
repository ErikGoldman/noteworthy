#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "assembly.h"
#include "chords.h"
#include "layout.h"
#include "layout_json.h"
#include "musicxml_writer.h"
#include "page_source.h"
#include "segmenter.h"
#include "staff.h"
#include "symbols.h"

namespace fs = std::filesystem;

namespace {

struct Options {
  std::vector<fs::path> inputs;  // consecutive pages of one score
  fs::path layout;
  fs::path model;
  fs::path output;
  fs::path debug_dir;
  int threads = 0;  // 0 = ONNX Runtime's default (all cores)
};

void usage() {
  std::cerr << "usage: noteworthy INPUT(.pdf|image)... --model MODEL.onnx -o OUT.musicxml\n"
               "                  [--layout OUT.json] [--debug DIR] [--threads N]\n"
               "Several inputs are read as consecutive pages of one score.\n";
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
    } else if (a == "--layout" && (v = next())) {
      opt.layout = v;
    } else if (a == "--debug" && (v = next())) {
      opt.debug_dir = v;
    } else if (a == "--threads" && (v = next())) {
      opt.threads = std::stoi(v);
    } else if (!a.starts_with("-")) {
      opt.inputs.push_back(a);
    } else {
      return false;
    }
  }
  return !opt.inputs.empty() && !opt.model.empty() && !opt.output.empty();
}

constexpr size_t kEnoughStaves = 2;
constexpr double kRetryInterlines[] = {7, 9, 11, 13, 16, 20, 26};  // px on the raw page

struct PageReading {
  double scale;
  cv::Mat page;
  omr::SymbolMaps maps;
  std::vector<omr::Staff> staves;
};

PageReading read_at_scale(const cv::Mat& raw, double scale, const omr::Segmenter& segmenter) {
  cv::Mat page;
  cv::resize(raw, page, cv::Size(), scale, scale, scale < 1 ? cv::INTER_AREA : cv::INTER_LINEAR);
  omr::SymbolMaps maps = segmenter.run(page);
  auto staves = omr::detect_staves(maps["staff_line"], segmenter.normalized_interline());
  return {scale, std::move(page), std::move(maps), std::move(staves)};
}

double mean_interline(const std::vector<omr::Staff>& staves) {
  double sum = 0;
  for (const auto& s : staves) {
    sum += s.interline((s.x_min() + s.x_max()) / 2);
  }
  return sum / staves.size();
}

// The first scale comes from run lengths after a global threshold, which fail on pale staff
// lines: the estimate lands far off, and at the wrong scale the segmenter finds no staves. It
// does tolerate a moderate error, so the search only has to land near the true spacing; the
// staves found there then give the exact scale. Pages that read well never come here.
PageReading read_page(const cv::Mat& raw, const omr::Segmenter& segmenter) {
  const double target = segmenter.normalized_interline();
  PageReading best = read_at_scale(raw, target / omr::estimate_interline(raw), segmenter);
  if (best.staves.size() >= kEnoughStaves) return best;
  bool retried = false;
  for (const double interline : kRetryInterlines) {
    PageReading trial = read_at_scale(raw, target / interline, segmenter);
    if (trial.staves.size() > best.staves.size()) {
      best = std::move(trial);
      retried = true;
    }
  }
  if (!retried) return best;
  PageReading exact = read_at_scale(raw, best.scale * target / mean_interline(best.staves), segmenter);
  return exact.staves.size() >= best.staves.size() ? exact : best;
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
  for (const auto& it : omr::build_items(symbols, staves, interline)) {
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
    omr::Segmenter segmenter(opt.model, opt.threads);
    std::vector<cv::Mat> pages;
    for (const fs::path& input : opt.inputs) {
      for (cv::Mat& page : omr::load_pages(input)) pages.push_back(std::move(page));
    }
    if (!opt.debug_dir.empty()) fs::create_directories(opt.debug_dir);
    omr::ScoreBuilder builder;
    for (size_t p = 0; p < pages.size(); ++p) {
      const auto [scale, page, maps, staves] = read_page(pages[p], segmenter);
      const omr::PageSymbols symbols = omr::extract_symbols(maps, staves, segmenter.normalized_interline());
      builder.add_page({static_cast<int>(p + 1), scale}, staves, symbols, segmenter.normalized_interline());
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
    std::ofstream(opt.output) << omr::to_musicxml(score, opt.inputs.front().stem().string());
    if (!opt.layout.empty()) std::ofstream(opt.layout) << omr::to_layout_json(score);
    std::cerr << score.measures.size() << " measures written to " << opt.output << "\n";
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
  return 0;
}
