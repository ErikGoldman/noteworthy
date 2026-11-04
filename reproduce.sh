#!/usr/bin/env bash
# From a fresh clone to a trained, benchmarked recognizer in one command:
#
#   ./reproduce.sh
#
# Stages (each in scripts/): Python environment, raw data download, training pages,
# training, C++ build, benchmark. Each stage skips what is already done, so the command
# can be rerun at any time (after an interruption, or to pick up a new stage) without
# downloading or recomputing finished parts.
#
# A full run downloads about 6 GB, needs about 40 GB of disk and an NVIDIA GPU, and
# takes on the order of a day. Settings are environment variables (scripts/common.sh);
# a quick end-to-end check:
#
#   NW_DATA=/tmp/nw/data NW_RUNS=/tmp/nw/runs NW_PAGES=64 NW_BASE_STEPS=200 \
#   NW_FINETUNE_STEPS=100 NW_EVAL_EVERY=100 NW_BENCH_COUNT=5 ./reproduce.sh
set -euo pipefail
scripts="$(cd "$(dirname "$0")" && pwd)/scripts"

"$scripts/setup_env.sh"
"$scripts/fetch_data.sh"
"$scripts/build_data.sh"
"$scripts/train.sh"
"$scripts/build_cpp.sh"
"$scripts/benchmark.sh"
"$scripts/train_seq.sh"
"$scripts/compare_zeus.sh"
