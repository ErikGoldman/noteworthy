"""System crops as StaffSeq2Seq reads them: one crop per staff, cut at the staff positions
find_staff_positions.py found, with the system's LMX.

A system is first scaled so that its interline is a staff crop's (STAFF_HEIGHT over
STAFF_INTERLINES: a staff at 256 px has the pixels per interline a system has at 512). In
training it is then augmented as Seq2Seq's crops are (seq.data.reframe and degrade: new margins
with the scale they bring, slight rotation, width, then ink and paper), the staff centres moved
with its rows, and each staff cut STAFF_HEIGHT tall around its centre: both staves of a system
share its paper, blur and tilt, as on a page.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import WIDTH_MULTIPLE, Sample, degrade, max_width, reframe, to_tensor
from .model import StaffImages
from .staves import STAFF_INTERLINES, Staves, staff_crops
from .vocab import PAD, Vocab

STAFF_HEIGHT = 256


def staff_count(lmx: str) -> int:
    """The staves an LMX system writes on: its highest staff:N."""
    return max(int(n) for n in re.findall(r"\bstaff:(\d+)", lmx))


def load_positions(directory: Path, root: Path) -> dict[str, Staves | None]:
    """find_staff_positions.py's positions for one data root, by sample name."""
    found = json.loads((directory / f"{root.name}.json").read_text())["staves"]
    return {name: None if p is None else Staves(tuple(p[:-1]), p[-1]) for name, p in found.items()}


def at_staff_scale(image: np.ndarray, staves: Staves, height: int = STAFF_HEIGHT) -> tuple[np.ndarray, Staves]:
    """The system scaled so that its interline is a staff crop's; its width at most a system's."""
    f = height / STAFF_INTERLINES / staves.interline
    width = min(max_width(2 * height), max(1, round(image.shape[1] * f)))
    scaled = cv2.resize(image, (width, max(1, round(image.shape[0] * f))), interpolation=cv2.INTER_AREA)
    return scaled, Staves(tuple(c * f for c in staves.centres), height / STAFF_INTERLINES)


def cut_staves(image: np.ndarray, staves: Staves, height: int = STAFF_HEIGHT) -> list[np.ndarray]:
    """Each staff `height` tall around its centre (an image at staff scale, so that is
    STAFF_INTERLINES of its interline)."""
    return staff_crops(image, Staves(staves.centres, height / STAFF_INTERLINES))


class StaffDataset(torch.utils.data.Dataset):
    """items with their staff positions (load_positions); every item must have them."""

    def __init__(self, items: list[tuple[Sample, Staves]], vocab: Vocab, train: bool, max_tokens: int,
                 height: int = STAFF_HEIGHT):
        self.items, self.vocab, self.train, self.max_tokens, self.height = items, vocab, train, max_tokens, height
        self.rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        sample, staves = self.items[i]
        image, staves = at_staff_scale(cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE), staves, self.height)
        if self.train:
            rng = self.augmentation_rng()
            interline = staves.interline
            image, moved = reframe(rng, image)
            staves = Staves(tuple(moved(c) for c in staves.centres), staves.interline)
            image = degrade(rng, image, interline)
        crops = cut_staves(image, staves, self.height)
        tokens = self.vocab.encode(sample.lmx.read_text())[: self.max_tokens]
        return torch.stack([to_tensor(c) for c in crops]), torch.tensor(tokens)

    def augmentation_rng(self) -> np.random.Generator:
        """One generator per loader worker, seeded with the worker's torch seed (as SystemDataset)."""
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed())
        return self.rng


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
