#!/usr/bin/env bash
# Checks system prerequisites and creates or updates the Python environments: .venv for
# everything of ours, .venv-smashcima for the handwritten crops (our Smashcima fork, a submodule
# in third_party/smashcima), .venv-zeus for the OLiMPiC authors' recognizer we compare against.
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
[ -x "$ZEUS_PY" ] || uv venv "$ROOT/.venv-zeus" --python 3.11 -q
uv pip install -q --python "$ZEUS_PY" -r "$ROOT/requirements-zeus.txt"
[ -x "$TRANSCODA_PY" ] || uv venv "$ROOT/.venv-transcoda" --python 3.12 -q
uv pip install -q --python "$TRANSCODA_PY" --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu -r "$ROOT/requirements-transcoda.txt"

[ -x "$MUSICDIFF_PY" ] || uv venv "$ROOT/.venv-musicdiff" --python 3.12 -q
uv pip install -q --python "$MUSICDIFF_PY" -r "$ROOT/requirements-omr-ned.txt"
[ -x "$HOMR_PY" ] || uv venv "$ROOT/.venv-homr" --python 3.12 -q
uv pip install -q --python "$HOMR_PY" -r "$ROOT/requirements-homr.txt"
# Copisteria (compared against on Polish pages, eval/copisteria_pages.sh) is installed editable from
# its checkout, as its README asks: it finds its models/ from the repository's root. fetch_data.sh
# downloads the weights into it.
checkout https://github.com/kobimusic/copisteria c685f280914d91a694494588d9f6a376d7917eba "$RAW/copisteria/src"
[ -x "$COPISTERIA_PY" ] || uv venv "$ROOT/.venv-copisteria" --python 3.12 -q
uv pip install -q --python "$COPISTERIA_PY" -e "$RAW/copisteria/src"
log "python environment ready"
