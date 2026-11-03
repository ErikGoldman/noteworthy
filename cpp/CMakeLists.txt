cmake_minimum_required(VERSION 3.20)
project(noteworthy CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

set(THIRD_PARTY ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
if(NOT EXISTS ${THIRD_PARTY}/onnxruntime OR NOT EXISTS ${THIRD_PARTY}/pdfium)
  message(FATAL_ERROR "Run third_party/fetch.sh first")
endif()

# The system OpenCV dev package ships without a CMake config, so declare it by hand.
add_library(opencv INTERFACE)
target_include_directories(opencv SYSTEM INTERFACE /usr/include/opencv4)
target_link_libraries(opencv INTERFACE opencv_core opencv_imgproc opencv_imgcodecs)

add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
  IMPORTED_LOCATION ${THIRD_PARTY}/onnxruntime/lib/libonnxruntime.so
  INTERFACE_INCLUDE_DIRECTORIES ${THIRD_PARTY}/onnxruntime/include)

add_library(pdfium SHARED IMPORTED)
set_target_properties(pdfium PROPERTIES
  IMPORTED_LOCATION ${THIRD_PARTY}/pdfium/lib/libpdfium.so
  INTERFACE_INCLUDE_DIRECTORIES ${THIRD_PARTY}/pdfium/include)

add_library(omr
  src/page_source.cpp
  src/segmenter.cpp
  src/staff.cpp
  src/symbols.cpp
  src/chords.cpp
  src/layout.cpp
  src/rhythm.cpp
  src/assembly.cpp
  src/musicxml_writer.cpp
  src/score.cpp
)
target_include_directories(omr PUBLIC src)
target_link_libraries(omr PUBLIC opencv onnxruntime pdfium)

add_executable(noteworthy src/main.cpp)
target_link_libraries(noteworthy PRIVATE omr)
set_target_properties(noteworthy PROPERTIES BUILD_RPATH "${THIRD_PARTY}/onnxruntime/lib;${THIRD_PARTY}/pdfium/lib")
