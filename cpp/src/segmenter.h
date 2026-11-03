#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <opencv2/core.hpp>

namespace omr {

// Per-channel symbol probability maps (CV_8U, 0..255) for one page.
class SymbolMaps {
 public:
  SymbolMaps(std::vector<std::string> names, std::vector<cv::Mat> maps);

  // Throws if the model has no such channel.
  const cv::Mat& operator[](std::string_view name) const;
  // Models trained before a channel was added lack it.
  bool has(std::string_view name) const;
  const std::vector<std::string>& names() const { return names_; }
  cv::Size size() const { return maps_.empty() ? cv::Size() : maps_.front().size(); }

 private:
  std::vector<std::string> names_;
  std::vector<cv::Mat> maps_;
};

// Runs the exported segmentation network on a page. The page must already be scaled
// so that its staff interline equals normalized_interline().
class Segmenter {
 public:
  explicit Segmenter(const std::filesystem::path& model_path, int threads = 0);
  ~Segmenter();
  Segmenter(const Segmenter&) = delete;
  Segmenter& operator=(const Segmenter&) = delete;

  double normalized_interline() const { return normalized_interline_; }
  SymbolMaps run(const cv::Mat& gray) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  std::vector<std::string> channels_;
  double normalized_interline_ = 0;
  int size_multiple_ = 1;
};

}  // namespace omr
