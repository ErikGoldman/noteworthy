#!/usr/bin/env bash
# Downloads pinned prebuilt dependencies into this directory.
set -euo pipefail
cd "$(dirname "$0")"

ORT_VERSION=1.31.0
PDFIUM_TAG=chromium/8100

if [ ! -d onnxruntime ]; then
  curl -fsSL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz" | tar xz
  mv "onnxruntime-linux-x64-${ORT_VERSION}" onnxruntime
fi
if [ ! -d pdfium ]; then
  mkdir pdfium
  curl -fsSL "https://github.com/bblanchon/pdfium-binaries/releases/download/${PDFIUM_TAG}/pdfium-linux-x64.tgz" | tar xz -C pdfium
fi
echo "third_party ready"
