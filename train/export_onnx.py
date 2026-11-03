"""Export a trained checkpoint to ONNX for the C++ pipeline.

usage: export_onnx.py CHECKPOINT.pt OUT.onnx

The model's contract with C++ travels in ONNX metadata:
  channels               JSON list of output channel names, in output order
  normalized_interline   staff interline (px) the model expects
  size_multiple          input H and W must be multiples of this
  input                  description of the expected input encoding
"""
from __future__ import annotations

import json
import sys

import onnx
import torch

from omr.dataset import NORMALIZED_INTERLINE_PX
from omr.model import SIZE_MULTIPLE, UNet


def main() -> None:
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    channels = ckpt["channels"]
    model = UNet(len(channels))
    model.load_state_dict(ckpt["model"])
    model.eval()

    dummy = torch.zeros(1, 1, 256, 256)
    torch.onnx.export(model, dummy, out_path, input_names=["image"], output_names=["logits"],
                      dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                    "logits": {0: "batch", 2: "height", 3: "width"}},
                      opset_version=17, dynamo=False)

    proto = onnx.load(out_path)
    for key, value in {
        "channels": json.dumps(channels),
        "normalized_interline": str(NORMALIZED_INTERLINE_PX),
        "size_multiple": str(SIZE_MULTIPLE),
        "input": "float32 NCHW grayscale, value = pixel / 255 (paper = 1, ink = 0)",
    }.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, out_path)
    print(f"exported {len(channels)} channels to {out_path} (step {ckpt.get('step')})")


if __name__ == "__main__":
    main()
