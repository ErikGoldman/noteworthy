#!/usr/bin/env bash
# Reproduces our results from a fresh clone, in one command:
#
#   ./reproduce.sh
#
# First everything train.sh does (skipped where it is done already), then the benchmarks'
# environments and data, our benchmarks, and the comparisons with other recognizers. Each stage
# skips what is already done, so the command can be rerun at any time.
#
# Page benchmarks run on the dev halves. The sealed test halves are run once, for the final
# evaluation (docs/EVALUATION_PROTOCOL.md): scripts/benchmark_pages.sh test.
#
# On top of train.sh, a full run downloads about 3 GB more and takes about a day more. Settings
# are environment variables (scripts/common.sh); a shorter end-to-end check, which still builds
# every crop set (hours):
#
#   NW_DATA=/tmp/nw/data NW_RUNS=/tmp/nw/runs NW_PAGES=64 NW_BASE_STEPS=200 \
#   NW_FINETUNE_STEPS=100 NW_EVAL_EVERY=100 NW_SEQ_STEPS=200 NW_BENCH_COUNT=5 ./reproduce.sh
set -euo pipefail
root="$(cd "$(dirname "$0")" && pwd)"
scripts="$root/scripts"

"$scripts/fetch_smb.sh"  # first: it may ask whether to go on without SMB (gated on Hugging Face)
"$root/train.sh"
"$scripts/setup_compare_env.sh"
"$scripts/fetch_benchmark_data.sh"
"$scripts/benchmark.sh"
"$scripts/benchmark_seq.sh"
"$scripts/benchmark_pages.sh" dev
"$scripts/compare_zeus.sh"
