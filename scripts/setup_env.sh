#!/usr/bin/env bash
# Checks system prerequisites and creates the Python environment (.venv).
set -euo pipefail
source "$(dirname "$0")/common.sh"

TORCH_INDEX=https://download.pytorch.org/whl/cu128

check_system() {
  local missing=()
  for tool in uv git curl unzip tar md5sum cmake g++; do
    command -v "$tool" >/dev/null || missing+=("$tool")
  done
  [ -f /usr/include/opencv4/opencv2/core.hpp ] || missing+=("OpenCV headers")
  if [ ${#missing[@]} -gt 0 ]; then
    echo "missing: ${missing[*]}" >&2
    echo "on Debian/Ubuntu: sudo apt install git curl unzip cmake g++ libopencv-dev" >&2
    echo "uv: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    exit 1
  fi
  command -v nvidia-smi >/dev/null || log "warning: no NVIDIA GPU found; training needs CUDA"
}

create_venv() {
  [ -x "$PY" ] || uv venv "$ROOT/.venv" --python 3.12 -q
  # torch comes from its own CUDA index; everything else from PyPI.
  uv pip install -q --python "$PY" --index-url "$TORCH_INDEX" -r "$ROOT/requirements-torch.txt"
  uv pip install -q --python "$PY" -r "$ROOT/requirements.txt"
}

check_system
step python-env create_venv
