"""System crops as StaffSeq2Seq reads them: one crop per staff (seq.staves), each scaled to the
staff height, with the system's LMX. Not yet augmented: augmenting must move the staves' positions
with the image, and comes with the positions training will take from the renderer and segmenter.
"""
from __future__ import annotations

import re

import cv2
import numpy as np
import torch

from .data import WIDTH_MULTIPLE, Sample, max_width, to_tensor
from .model import StaffImages
from .staves import staff_crops, system_staves
from .vocab import PAD, Vocab

STAFF_HEIGHT = 256


def staff_count(lmx: str) -> int:
    """The staves an LMX system writes on: its highest staff:N."""
    return max(int(n) for n in re.findall(r"\bstaff:(\d+)", lmx))


class StaffNotFound(ValueError):
    pass


def staff_images(image: np.ndarray, count: int, height: int = STAFF_HEIGHT) -> list[np.ndarray]:
    """A grayscale system crop's `count` staves, each crop scaled to `height`; all share a width."""
    staves = system_staves(image, count)
    if staves is None:
        raise StaffNotFound
    crops = staff_crops(image, staves)
    width = min(max_width(2 * height), max(1, round(crops[0].shape[1] * height / crops[0].shape[0])))
    return [cv2.resize(c, (width, height), interpolation=cv2.INTER_AREA) for c in crops]


class StaffDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[Sample], vocab: Vocab, max_tokens: int, height: int = STAFF_HEIGHT):
        self.items, self.vocab, self.max_tokens, self.height = items, vocab, max_tokens, height

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        sample = self.items[i]
        lmx = sample.lmx.read_text()
        image = cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE)
        staves = staff_images(image, staff_count(lmx), self.height)
        return torch.stack([to_tensor(s) for s in staves]), torch.tensor(self.vocab.encode(lmx)[: self.max_tokens])


def collate_staves(batch) -> tuple[StaffImages, torch.Tensor, torch.Tensor]:
    """Staves padded blank to the batch's most staves and widest crop; tokens padded with PAD."""
    staves, tokens = zip(*batch)
    width = -(-max(s.shape[-1] for s in staves) // WIDTH_MULTIPLE) * WIDTH_MULTIPLE
    pixels = torch.zeros(len(staves), max(len(s) for s in staves), 1, staves[0].shape[-2], width)
    for k, s in enumerate(staves):
        pixels[k, : len(s), :, :, : s.shape[-1]] = s
    token_batch = torch.full((len(tokens), max(map(len, tokens))), PAD)
    for k, t in enumerate(tokens):
        token_batch[k, : len(t)] = t
    counts = torch.tensor([len(s) for s in staves])
    widths = torch.tensor([s.shape[-1] for s in staves])
    return StaffImages(pixels, counts), widths, token_batch
