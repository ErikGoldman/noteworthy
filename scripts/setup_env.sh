#!/usr/bin/env bash
# Checks system prerequisites and creates or updates the Python environments: .venv for
# everything of ours, .venv-zeus for the OLiMPiC authors' recognizer we compare against.
set -euo pipefail
source "$(dirname "$0")/common.sh"

TORCH_INDEX=https://download.pytorch.org/whl/cu128

missing=()
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

[ -x "$PY" ] || uv venv "$ROOT/.venv" --python 3.12 -q
# uv installs only what is missing or at another version. torch comes from its own CUDA index.
uv pip install -q --python "$PY" --index-url "$TORCH_INDEX" -r "$ROOT/requirements-torch.txt"
uv pip install -q --python "$PY" -r "$ROOT/requirements.txt"
[ -x "$ZEUS_PY" ] || uv venv "$ROOT/.venv-zeus" --python 3.11 -q
uv pip install -q --python "$ZEUS_PY" -r "$ROOT/requirements-zeus.txt"
[ -x "$TRANSCODA_PY" ] || uv venv "$ROOT/.venv-transcoda" --python 3.12 -q
uv pip install -q --python "$TRANSCODA_PY" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -r "$ROOT/requirements-transcoda.txt"

[ -x "$MUSICDIFF_PY" ] || uv venv "$ROOT/.venv-musicdiff" --python 3.12 -q
uv pip install -q --python "$MUSICDIFF_PY" -r "$ROOT/requirements-omr-ned.txt"
[ -x "$HOMR_PY" ] || uv venv "$ROOT/.venv-homr" --python 3.12 -q
uv pip install -q --python "$HOMR_PY" -r "$ROOT/requirements-homr.txt"
[ -x "$COPISTERIA_PY" ] || uv venv "$ROOT/.venv-copisteria" --python 3.12 -q
(cd "$ROOT" && uv pip install -q --python "$COPISTERIA_PY" -r requirements-copisteria.txt)
log "python environment ready"
