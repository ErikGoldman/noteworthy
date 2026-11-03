#pragma once

#include <filesystem>
#include <vector>

#include <opencv2/core.hpp>

namespace omr {

// Loads every page of a PDF (rendered with PDFium) or a single raster image, as
// 8-bit grayscale. PDF pages are rendered at `pdf_dpi`.
std::vector<cv::Mat> load_pages(const std::filesystem::path& path, double pdf_dpi = 300.0);

}  // namespace omr
