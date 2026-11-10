#!/usr/bin/env bash
# From a fresh clone to a recognizer you can run on your own scores, in one command:
#
#   ./train.sh
#
# Stages (each in scripts/): Python environments, training data download, training pages and
# crops, the segmentation network (finds staves and systems), the C++ recognizer, the sequence
# model (reads each system). Each stage skips what is already done, so the command can be rerun
# at any time (after an interruption, or after a change) without downloading or recomputing
# finished parts; reproduce.sh runs it first, so a reproduction redoes none of it.
#
# A full run downloads about 30 GB, needs about 85 GB of disk and an NVIDIA GPU, and takes
# days. Settings are environment variables (scripts/common.sh).
set -euo pipefail
scripts="$(cd "$(dirname "$0")" && pwd)/scripts"

"$scripts/setup_env.sh"
"$scripts/fetch_training_data.sh"
"$scripts/build_data.sh"
"$scripts/train_segmenter.sh"
"$scripts/build_cpp.sh"
"$scripts/build_seq_data.sh"
"$scripts/train_seq.sh"
