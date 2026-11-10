#!/usr/bin/env bash
# Checks system prerequisites and creates or updates the Python environments that training and
# recognition need: .venv for everything of ours, .venv-smashcima for the handwritten crops (our
# Smashcima fork, a submodule in third_party/smashcima). The benchmarks' and other recognizers'
# environments are setup_compare_env.sh's.
set -euo pipefail
source "$(dirname "$0")/common.sh"

TORCH_INDEX=https://download.pytorch.org/whl/cu128

missing=()
for tool in uv git curl unzip tar md5sum cmake g++; do
  command -v "$tool" >/dev/null || missing+=("$tool")
done
[ -f /usr/include/opencv4/opencv2/core.hpp ] || missing+=("OpenCV headers")
[ -n "${DISPLAY:-}" ] || command -v xvfb-run >/dev/null || missing+=("xvfb-run (no display for MuseScore)")
if [ ${#missing[@]} -gt 0 ]; then
  echo "missing: ${missing[*]}" >&2
  echo "on Debian/Ubuntu: sudo apt install git curl unzip cmake g++ libopencv-dev xvfb" >&2
  echo "uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi
command -v nvidia-smi >/dev/null || log "warning: no NVIDIA GPU found; training needs CUDA"

[ -x "$PY" ] || uv venv "$ROOT/.venv" --python 3.12 -q
# uv installs only what is missing or at another version. torch comes from its own CUDA index.
uv pip install -q --python "$PY" --index-url "$TORCH_INDEX" -r "$ROOT/requirements-torch.txt"
uv pip install -q --python "$PY" -r "$ROOT/requirements.txt"
git -C "$ROOT" submodule update --init third_party/smashcima
[ -x "$SMASHCIMA_PY" ] || uv venv "$ROOT/.venv-smashcima" --python 3.12 -q
uv pip install -q --python "$SMASHCIMA_PY" -r "$ROOT/requirements-smashcima.txt"
uv pip install -q --python "$SMASHCIMA_PY" --no-deps -e "$ROOT/third_party/smashcima"
log "python environments ready"
