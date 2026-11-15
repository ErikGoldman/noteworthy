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
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import WIDTH_MULTIPLE, Sample, degrade, max_width, reframe, to_tensor
from .meter import with_meter
from .model import StaffImages, SystemImages
from .staves import STAFF_INTERLINES, Staves, staff_crops
from .vocab import PAD, Vocab

STAFF_HEIGHT = 256


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


def cut_staves(image: np.ndarray, staves: Staves, height: int = STAFF_HEIGHT, outward: float = 0.0,
               cut_between: bool = False) -> list[np.ndarray]:
    """Each staff `height` tall around its centre (an image at staff scale, so that is
    STAFF_INTERLINES of its interline), the outer staves' slid `outward`, white past the midpoints
    to its neighbours if cut_between (staves.staff_crops)."""
    return staff_crops(image, Staves(staves.centres, height / STAFF_INTERLINES), outward, cut_between)


class StaffDataset(torch.utils.data.Dataset):
    """items with their staff positions (load_positions); every item must have them."""

    def __init__(self, items: list[tuple[Sample, Staves]], vocab: Vocab, train: bool, max_tokens: int,
                 height: int = STAFF_HEIGHT, outward: float = 0.0, meter_context: bool = False, jitter: float = 0.0,
                 cut_between: bool = False):
        """jitter: in training, each staff's crop is centred up to this many interlines off its staff
        (uniformly, each staff apart), so that no row of a crop holds a fixed place on the staff.
        cut_between: crops white past the midpoints between staves (staves.staff_crops)."""
        self.items, self.vocab, self.train, self.max_tokens, self.height = items, vocab, train, max_tokens, height
        self.outward, self.meter_context, self.jitter, self.cut_between = outward, meter_context, jitter, cut_between
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
            staves = Staves(tuple(moved(c) + rng.uniform(-1, 1) * self.jitter * interline for c in staves.centres),
                            staves.interline)
            image = degrade(rng, image, interline)
        crops = cut_staves(image, staves, self.height, self.outward, self.cut_between)
        lmx = sample.lmx.read_text()
        if self.meter_context:
            lmx = with_meter(lmx, self.augmentation_rng() if self.train else None)
        tokens = self.vocab.encode(lmx)[: self.max_tokens]
        return torch.stack([to_tensor(c) for c in crops]), torch.tensor(tokens)

    def augmentation_rng(self) -> np.random.Generator:
        """One generator per loader worker, seeded with the worker's torch seed (as SystemDataset)."""
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed())
        return self.rng


ROW_MULTIPLE = 32           # the encoder's vertical stride: a system's height is padded to it
# Pixels at staff scale, about four staves; a taller system is shrunk to it. One crop sets its
# batch's height: at 2,048 the odd tall crop (PDMX's median is 406, its 95th percentile 502)
# took a batch of ten out of memory on a 16 GB card.
MAX_SYSTEM_HEIGHT = 1024


def at_interline(image: np.ndarray, staves: Staves, height: int = STAFF_HEIGHT) -> tuple[np.ndarray, Staves]:
    """The whole system at staff scale (at_staff_scale: an interline of height / STAFF_INTERLINES
    pixels), as tall as its staves and margins make it, at most MAX_SYSTEM_HEIGHT; its staves with it."""
    scaled, staves = at_staff_scale(image, staves, height)
    if scaled.shape[0] > MAX_SYSTEM_HEIGHT:
        f = MAX_SYSTEM_HEIGHT / scaled.shape[0]
        scaled = cv2.resize(scaled, (max(1, round(scaled.shape[1] * f)), MAX_SYSTEM_HEIGHT), interpolation=cv2.INTER_AREA)
        staves = Staves(tuple(c * f for c in staves.centres), staves.interline * f)
    return scaled, staves


class InterlineSystemDataset(StaffDataset):
    """Whole system crops for Seq2Seq at staff scale (at_interline), not cut into staves: every
    staff gets the pixels per interline a staff crop has, however many the system holds, and the
    crop's height follows its staves. Augmented as StaffDataset's (reframe's margins also scale
    the system by up to about a tenth, as an interline found on a page is never exact). An item is
    ((pixels, staves), tokens), the staves where the augmented crop has them."""

    def __getitem__(self, i: int):
        sample, staves = self.items[i]
        image, staves = at_interline(cv2.imread(str(sample.image), cv2.IMREAD_GRAYSCALE), staves, self.height)
        if self.train:
            rng = self.augmentation_rng()
            image, moved = reframe(rng, image)
            staves = Staves(tuple(moved(c) for c in staves.centres), staves.interline)
            image = degrade(rng, image, staves.interline)
        lmx = sample.lmx.read_text()
        if self.meter_context:
            lmx = with_meter(lmx, self.augmentation_rng() if self.train else None)
        tokens = self.vocab.encode(lmx)[: self.max_tokens]
        return (to_tensor(image), staves), torch.tensor(tokens)


def collate_systems(batch) -> tuple[SystemImages, torch.Tensor, torch.Tensor]:
    """((pixels, staves), tokens) items of different heights and widths: padded white below to
    the batch's tallest (a multiple of ROW_MULTIPLE) and right to its widest; tokens padded with
    PAD. Height padding is not masked: white rows hold nothing to read (unlike columns past a
    crop's width, which the encoder masks)."""
    inputs, tokens = zip(*batch)
    images, staves = zip(*inputs)
    height = -(-max(im.shape[-2] for im in images) // ROW_MULTIPLE) * ROW_MULTIPLE
    width = -(-max(im.shape[-1] for im in images) // WIDTH_MULTIPLE) * WIDTH_MULTIPLE
    image_batch = torch.zeros(len(images), 1, height, width)
    for k, im in enumerate(images):
        image_batch[k, :, : im.shape[-2], : im.shape[-1]] = im
    centres = torch.zeros(len(staves), max(len(s.centres) for s in staves))
    for k, s in enumerate(staves):
        centres[k, : len(s.centres)] = torch.tensor(s.centres)
    systems = SystemImages(image_batch, centres, torch.tensor([len(s.centres) for s in staves]),
                           torch.tensor([s.interline for s in staves], dtype=torch.float32))
    token_batch = torch.full((len(tokens), max(map(len, tokens))), PAD)
    for k, t in enumerate(tokens):
        token_batch[k, : len(t)] = t
    return systems, torch.tensor([im.shape[-1] for im in images]), token_batch


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
