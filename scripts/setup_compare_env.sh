#!/usr/bin/env bash
# Creates or updates the environments of the benchmarks and of the recognizers we compare
# against: .venv-zeus (the OLiMPiC authors' Zeus), .venv-transcoda, .venv-musicdiff (the OMR-NED
# metric), .venv-homr and .venv-copisteria.
set -euo pipefail
source "$(dirname "$0")/common.sh"

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
# its checkout, as its README asks: it finds its models/ from the repository's root. fetch_benchmark_data.sh
# downloads the weights into it.
checkout https://github.com/kobimusic/copisteria c685f280914d91a694494588d9f6a376d7917eba "$RAW/copisteria/src"
[ -x "$COPISTERIA_PY" ] || uv venv "$ROOT/.venv-copisteria" --python 3.12 -q
uv pip install -q --python "$COPISTERIA_PY" -e "$RAW/copisteria/src"
log "comparison environments ready"
