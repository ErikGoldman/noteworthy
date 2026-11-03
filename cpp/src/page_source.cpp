#include "page_source.h"

#include <memory>
#include <stdexcept>

#include <fpdfview.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace omr {
namespace {

constexpr double kPointsPerInch = 72.0;

struct PdfiumLibrary {
  PdfiumLibrary() {
    FPDF_LIBRARY_CONFIG config{};
    config.version = 2;
    FPDF_InitLibraryWithConfig(&config);
  }
  ~PdfiumLibrary() { FPDF_DestroyLibrary(); }
};

void ensure_pdfium() { static PdfiumLibrary library; }

using DocPtr = std::unique_ptr<std::remove_pointer_t<FPDF_DOCUMENT>, decltype(&FPDF_CloseDocument)>;
using PagePtr = std::unique_ptr<std::remove_pointer_t<FPDF_PAGE>, decltype(&FPDF_ClosePage)>;
using BitmapPtr = std::unique_ptr<std::remove_pointer_t<FPDF_BITMAP>, decltype(&FPDFBitmap_Destroy)>;

cv::Mat render_page(FPDF_DOCUMENT doc, int index, double dpi) {
  PagePtr page(FPDF_LoadPage(doc, index), &FPDF_ClosePage);
  if (!page) throw std::runtime_error("cannot load PDF page " + std::to_string(index));
  const int width = static_cast<int>(FPDF_GetPageWidthF(page.get()) / kPointsPerInch * dpi + 0.5);
  const int height = static_cast<int>(FPDF_GetPageHeightF(page.get()) / kPointsPerInch * dpi + 0.5);
  BitmapPtr bitmap(FPDFBitmap_Create(width, height, /*alpha=*/0), &FPDFBitmap_Destroy);
  FPDFBitmap_FillRect(bitmap.get(), 0, 0, width, height, 0xFFFFFFFF);
  FPDF_RenderPageBitmap(bitmap.get(), page.get(), 0, 0, width, height, /*rotate=*/0, FPDF_ANNOT | FPDF_PRINTING);
  cv::Mat bgra(height, width, CV_8UC4, FPDFBitmap_GetBuffer(bitmap.get()), FPDFBitmap_GetStride(bitmap.get()));
  cv::Mat gray;
  cv::cvtColor(bgra, gray, cv::COLOR_BGRA2GRAY);
  return gray;  // cvtColor allocated new storage, so the bitmap may be freed
}

std::vector<cv::Mat> load_pdf(const std::filesystem::path& path, double dpi) {
  ensure_pdfium();
  DocPtr doc(FPDF_LoadDocument(path.c_str(), nullptr), &FPDF_CloseDocument);
  if (!doc) throw std::runtime_error("cannot open PDF " + path.string());
  std::vector<cv::Mat> pages;
  for (int i = 0; i < FPDF_GetPageCount(doc.get()); ++i) pages.push_back(render_page(doc.get(), i, dpi));
  return pages;
}

}  // namespace

std::vector<cv::Mat> load_pages(const std::filesystem::path& path, double pdf_dpi) {
  if (path.extension() == ".pdf" || path.extension() == ".PDF") return load_pdf(path, pdf_dpi);
  cv::Mat image = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
  if (image.empty()) throw std::runtime_error("cannot read image " + path.string());
  return {image};
}

}  // namespace omr
