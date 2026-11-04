#!/usr/bin/env bash
# Builds the C++ recognizer (cpp/build/noteworthy) with its pinned prebuilt dependencies.
set -euo pipefail
source "$(dirname "$0")/common.sh"

"$ROOT/cpp/third_party/fetch.sh"
cmake -S "$ROOT/cpp" -B "$ROOT/cpp/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$ROOT/cpp/build" -j "$(nproc)"
