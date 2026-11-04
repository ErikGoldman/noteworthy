"""Export a trained checkpoint to ONNX for the C++ pipeline.

usage: export_onnx.py CHECKPOINT.pt OUT.onnx

The exported graph outputs per-channel probabilities ("probabilities", N x C x H x W),
including any exclusive-family softmax (see omr.heads); checkpoints from before the
families existed export with plain sigmoids.

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

from omr.channels import CHANNEL_INDEX
from omr.dataset import NORMALIZED_INTERLINE_PX
from omr.heads import probabilities
from omr.model import SIZE_MULTIPLE, UNet


class WithProbabilities(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, families: list[list[int]]):
        super().__init__()
        self.model = model
        self.families = families

    def forward(self, x):
        return probabilities(self.model(x), self.families)


def main() -> None:
    ckpt_path, out_path = sys.argv[1], sys.argv[2]
    ckpt = torch.load(ckpt_path, map_location="cpu")
    channels = ckpt["channels"]
    model = UNet(len(channels))
    model.load_state_dict(ckpt["model"])
    model.eval()
    families = [[CHANNEL_INDEX[name] for name in family] for family in ckpt.get("families", [])]

    dummy = torch.zeros(1, 1, 256, 256)
    torch.onnx.export(WithProbabilities(model, families), dummy, out_path, input_names=["image"],
                      output_names=["probabilities"],
                      dynamic_axes={"image": {0: "batch", 2: "height", 3: "width"},
                                    "probabilities": {0: "batch", 2: "height", 3: "width"}},
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
