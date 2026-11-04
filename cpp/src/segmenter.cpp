#include "segmenter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <onnxruntime_cxx_api.h>

namespace omr {
namespace {

// Tiles are processed with extra context on every side, so the network sees what is
// beyond a tile edge; only each tile's core is kept.
constexpr int kTileCore = 768;
constexpr int kTileContext = 96;

std::string metadata(const Ort::Session& session, const char* key) {
  Ort::AllocatorWithDefaultOptions alloc;
  auto value = session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, alloc);
  if (!value) throw std::runtime_error(std::string("model metadata missing: ") + key);
  return value.get();
}

// Parses the JSON list of channel names written by export_onnx.py (["a", "b", ...]).
std::vector<std::string> parse_name_list(const std::string& json) {
  std::vector<std::string> names;
  size_t pos = 0;
  while ((pos = json.find('"', pos)) != std::string::npos) {
    const size_t end = json.find('"', pos + 1);
    if (end == std::string::npos) throw std::runtime_error("malformed channel list: " + json);
    names.push_back(json.substr(pos + 1, end - pos - 1));
    pos = end + 1;
  }
  return names;
}

int round_up(int v, int multiple) { return (v + multiple - 1) / multiple * multiple; }

}  // namespace

SymbolMaps::SymbolMaps(std::vector<std::string> names, std::vector<cv::Mat> maps)
    : names_(std::move(names)), maps_(std::move(maps)) {}

const cv::Mat& SymbolMaps::operator[](std::string_view name) const {
  auto it = std::find(names_.begin(), names_.end(), name);
  if (it == names_.end()) throw std::out_of_range("no symbol channel named " + std::string(name));
  return maps_[it - names_.begin()];
}

bool SymbolMaps::has(std::string_view name) const {
  return std::find(names_.begin(), names_.end(), name) != names_.end();
}

struct Segmenter::Impl {
  Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "noteworthy"};
  Ort::Session session{nullptr};
};

Segmenter::Segmenter(const std::filesystem::path& model_path, int threads) : impl_(std::make_unique<Impl>()) {
  Ort::SessionOptions options;
  options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (threads > 0) options.SetIntraOpNumThreads(threads);
  impl_->session = Ort::Session(impl_->env, model_path.c_str(), options);
  channels_ = parse_name_list(metadata(impl_->session, "channels"));
  normalized_interline_ = std::stod(metadata(impl_->session, "normalized_interline"));
  size_multiple_ = std::stoi(metadata(impl_->session, "size_multiple"));
}

Segmenter::~Segmenter() = default;

SymbolMaps Segmenter::run(const cv::Mat& gray) const {
  CV_Assert(gray.type() == CV_8UC1);
  const int channels = static_cast<int>(channels_.size());
  std::vector<cv::Mat> maps(channels);
  for (cv::Mat& m : maps) m = cv::Mat::zeros(gray.size(), CV_8UC1);

  Ort::MemoryInfo cpu = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const char* input_names[] = {"image"};
  const char* output_names[] = {"probabilities"};

  for (int y0 = 0; y0 < gray.rows; y0 += kTileCore) {
    for (int x0 = 0; x0 < gray.cols; x0 += kTileCore) {
      const cv::Rect core(x0, y0, std::min(kTileCore, gray.cols - x0), std::min(kTileCore, gray.rows - y0));
      const cv::Rect with_context = (core + cv::Size(2 * kTileContext, 2 * kTileContext) - cv::Point(kTileContext, kTileContext)) &
                                    cv::Rect(0, 0, gray.cols, gray.rows);
      const int h = round_up(with_context.height, size_multiple_);
      const int w = round_up(with_context.width, size_multiple_);

      // Pad with paper (1.0) up to the size multiple.
      cv::Mat input(h, w, CV_32FC1, cv::Scalar(1.0f));
      gray(with_context).convertTo(input(cv::Rect(0, 0, with_context.width, with_context.height)), CV_32F, 1.0 / 255.0);

      const int64_t shape[] = {1, 1, h, w};
      Ort::Value tensor = Ort::Value::CreateTensor<float>(cpu, input.ptr<float>(), input.total(), shape, 4);
      auto outputs = impl_->session.Run(Ort::RunOptions{nullptr}, input_names, &tensor, 1, output_names, 1);
      const float* probabilities = outputs[0].GetTensorData<float>();

      const cv::Rect core_in_tile = core - with_context.tl();
      for (int c = 0; c < channels; ++c) {
        const cv::Mat prob(h, w, CV_32FC1, const_cast<float*>(probabilities + static_cast<size_t>(c) * h * w));
        prob(core_in_tile).convertTo(maps[c](core), CV_8U, 255.0);
      }
    }
  }
  return SymbolMaps(channels_, std::move(maps));
}

}  // namespace omr
