"""Export a sequence-model checkpoint (train_seq.py) to ONNX for the C++ pipeline.

usage: export_seq_onnx.py CHECKPOINT.pt OUT_DIR

Writes two graphs, run as greedy decoding by the caller:
  encoder.onnx   image (B, 1, H, W), widths (B) -> memory (B, S, D), memory_padding (B, S)
  decoder.onnx   tokens (B, N), memory, memory_padding -> logits (B, V) of the token after
                 the last one
The decoder recomputes the whole prefix each step (no key/value cache): simple, and a
system's few hundred tokens decode in well under a second.

The model's contract with C++ travels in encoder.onnx metadata:
  vocabulary     JSON list of token strings, by id
  bos, eos       ids starting and ending a sequence
  height         input height in pixels; widths are the scaled crop widths, the batch
                 padded on the right to a multiple of width_multiple
  max_width      wider crops are squeezed to this width
  max_tokens     longest sequence, <bos> and <eos> included
  input          description of the expected input encoding
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime
import torch

from seq.data import HEIGHT, MAX_WIDTH, WIDTH_MULTIPLE
from seq.model import Seq2Seq
from seq.vocab import BOS, EOS, Vocab
from train_seq import VOCABULARY


class Encoder(torch.nn.Module):
    def __init__(self, model: Seq2Seq):
        super().__init__()
        self.model = model

    def forward(self, image, widths):
        return self.model.encode(image, widths)


class NextToken(torch.nn.Module):
    def __init__(self, model: Seq2Seq):
        super().__init__()
        self.model = model

    def forward(self, tokens, memory, memory_padding):
        return self.model.decode(tokens, memory, memory_padding)[:, -1]


def export(model: Seq2Seq, out: Path) -> None:
    image, widths = torch.rand(2, 1, HEIGHT, 4 * WIDTH_MULTIPLE), torch.tensor([4 * WIDTH_MULTIPLE, 2 * WIDTH_MULTIPLE])
    memory, padding = model.encode(image, widths)
    torch.onnx.export(Encoder(model), (image, widths), out / "encoder.onnx", input_names=["image", "widths"],
                      output_names=["memory", "memory_padding"],
                      dynamic_axes={"image": {0: "batch", 3: "width"}, "widths": {0: "batch"},
                                    "memory": {0: "batch", 1: "cells"}, "memory_padding": {0: "batch", 1: "cells"}},
                      opset_version=18, dynamo=True)
    tokens = torch.full((2, 3), BOS)
    torch.onnx.export(NextToken(model), (tokens, memory, padding), out / "decoder.onnx",
                      input_names=["tokens", "memory", "memory_padding"], output_names=["logits"],
                      dynamic_axes={"tokens": {0: "batch", 1: "length"}, "memory": {0: "batch", 1: "cells"},
                                    "memory_padding": {0: "batch", 1: "cells"}, "logits": {0: "batch"}},
                      opset_version=18, dynamo=True)


def add_metadata(path: Path, vocab: Vocab, max_tokens: int) -> None:
    proto = onnx.load(path)
    for key, value in {
        "vocabulary": json.dumps(vocab.tokens), "bos": str(BOS), "eos": str(EOS), "height": str(HEIGHT),
        "width_multiple": str(WIDTH_MULTIPLE), "max_width": str(MAX_WIDTH), "max_tokens": str(max_tokens),
        "input": "float32 NCHW grayscale, value = 1 - pixel / 255 (paper = 0, ink = 1), zero padding",
    }.items():
        entry = proto.metadata_props.add()
        entry.key, entry.value = key, value
    onnx.save(proto, path)


def greedy_onnx(out: Path, image: np.ndarray, widths: np.ndarray, steps: int) -> list[list[int]]:
    encoder = onnxruntime.InferenceSession(out / "encoder.onnx", providers=["CPUExecutionProvider"])
    decoder = onnxruntime.InferenceSession(out / "decoder.onnx", providers=["CPUExecutionProvider"])
    memory, padding = encoder.run(None, {"image": image, "widths": widths})
    tokens = np.full((image.shape[0], 1), BOS, np.int64)
    for _ in range(steps):
        logits, = decoder.run(None, {"tokens": tokens, "memory": memory, "memory_padding": padding})
        tokens = np.concatenate([tokens, logits.argmax(-1)[:, None]], 1)
    return tokens[:, 1:].tolist()


def check_against_torch(model: Seq2Seq, out: Path, steps: int = 40) -> None:
    """Greedy tokens from ONNX Runtime must equal PyTorch's on a random batch, padding included."""
    torch.manual_seed(0)
    image = torch.rand(2, 1, HEIGHT, 12 * WIDTH_MULTIPLE)
    widths = torch.tensor([12 * WIDTH_MULTIPLE, 7 * WIDTH_MULTIPLE])
    image[1, :, :, 7 * WIDTH_MULTIPLE:] = 0
    expected = [t[:steps] for t in model.generate(image, widths, max_tokens=steps + 1)]
    got = greedy_onnx(out, image.numpy(), widths.numpy(), steps)
    for e, g in zip(expected, got):
        n = e.index(EOS) + 1 if EOS in e else len(e)
        assert e[:n] == g[:n], f"ONNX decoding differs from PyTorch:\n{e[:n]}\n{g[:n]}"


def main() -> None:
    checkpoint, out = Path(sys.argv[1]), Path(sys.argv[2])
    state = torch.load(checkpoint, map_location="cpu")
    vocab = Vocab.load(VOCABULARY)
    model = Seq2Seq(len(vocab), max_tokens=state["args"]["max_tokens"])
    model.load_state_dict(state["model"])
    model.eval()
    out.mkdir(parents=True, exist_ok=True)
    torch.backends.mha.set_fastpath_enabled(False)  # its fused layer kernels have no ONNX form
    with torch.no_grad():
        export(model, out)
    add_metadata(out / "encoder.onnx", vocab, model.max_tokens)
    check_against_torch(model, out)
    print(f"exported {checkpoint} (step {state['step']}) to {out}")


if __name__ == "__main__":
    main()
