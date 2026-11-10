"""System crops and their LMX transcriptions, as model inputs.

A sample is a directory entry `<root>/<sample>.png|.jpg` + `<sample>.lmx` (the OLiMPiC
and GrandStaff-LMX layout). Images are scaled to a fixed height, ink bright on black, and
padded on the right to the widest crop of the batch.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from omr.augment import capture_degradation, stroke_weight

from .vocab import PAD, Vocab

HEIGHT = 256
MAX_WIDTH = 2048        # at HEIGHT; wider crops are squeezed horizontally
WIDTH_MULTIPLE = 32     # the encoder's horizontal stride
# A system crop is roughly this many interlines tall (two staves plus margins); the
# degradations are sized in interlines.
CROP_INTERLINES = 30


@dataclass(frozen=True)
class Sample:
    image: Path
    lmx: Path


def samples(root: Path, names: list[str]) -> list[Sample]:
    """OLiMPiC images are PNG, GrandStaff's JPEG."""
    extension = ".png" if (root / f"{names[0]}.png").exists() else ".jpg"
    return [Sample(root / f"{n}{extension}", root / f"{n}.lmx") for n in names]


def split_names(root: Path, split: str) -> list[str]:
    return (root / f"samples.{split}.txt").read_text().split()


def max_width(height: int) -> int:
    return MAX_WIDTH * height // HEIGHT


def scale_to_height(image: np.ndarray, height: int = HEIGHT) -> np.ndarray:
    width = min(max_width(height), max(1, round(image.shape[1] * height / image.shape[0])))
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def load_scaled(path: Path, height: int = HEIGHT) -> np.ndarray:
    """Grayscale, black ink on white, scaled to `height`."""
    return scale_to_height(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), height)


def augment(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    """What separates engraved scans from clean renders: margins and scale (crops are cut
    by hand), slight rotation, stroke weight, scan tone, blur, noise, JPEG."""
    h, w = image.shape
    interline = h / CROP_INTERLINES
    pad_top, pad_bottom = (int(rng.uniform(-0.08, 0.12) * h) for _ in range(2))
    image = cv2.copyMakeBorder(image[max(0, -pad_top):h - max(0, -pad_bottom)], max(0, pad_top), max(0, pad_bottom),
                               0, 0, cv2.BORDER_CONSTANT, value=255)
    angle = rng.uniform(-1.0, 1.0)
    m = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1.0)
    image = cv2.warpAffine(image, m, (image.shape[1], image.shape[0]), borderValue=255)
    width = min(max_width(h), round(image.shape[1] * h / image.shape[0] * rng.uniform(0.85, 1.15)))
    image = cv2.resize(image, (width, h), interpolation=cv2.INTER_AREA)
    if rng.random() < 0.5:
        image = stroke_weight(rng, image, interline * 2)
    if rng.random() < 0.3:
        image = np.where(image < rng.uniform(100, 200), 0, 255).astype(np.uint8)  # binarized scan
    if rng.random() < 0.9:
        image = capture_degradation(rng, image, interline)
    return image


def to_tensor(image: np.ndarray) -> torch.Tensor:
    """(1, H, W) float, ink bright on black."""
    return torch.from_numpy(255 - image).float().div(255).unsqueeze(0)


class SystemDataset(torch.utils.data.Dataset):
    def __init__(self, items: list[Sample], vocab: Vocab, train: bool, max_tokens: int, height: int = HEIGHT):
        self.height = height
        self.items = items
        self.vocab = vocab
        self.train = train
        self.max_tokens = max_tokens
        self.rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        sample = self.items[i]
        image = load_scaled(sample.image, self.height)
        if self.train:
            image = augment(self.augmentation_rng(), image)
        tokens = self.vocab.encode(sample.lmx.read_text())[: self.max_tokens]
        return to_tensor(image), torch.tensor(tokens)

    def augmentation_rng(self) -> np.random.Generator:
        """One generator per loader worker, seeded with the worker's torch seed: the loader derives
        those from the training seed, and hands each worker its batches in a fixed order"""
        if self.rng is None:
            self.rng = np.random.default_rng(torch.initial_seed())
        return self.rng


def collate(batch):
    images, tokens = zip(*batch)
    width = max(im.shape[-1] for im in images)
    width = -(-width // WIDTH_MULTIPLE) * WIDTH_MULTIPLE
    image_batch = torch.zeros(len(images), 1, images[0].shape[1], width)
    widths = torch.tensor([im.shape[-1] for im in images])
    for k, im in enumerate(images):
        image_batch[k, :, :, : im.shape[-1]] = im
    token_batch = torch.full((len(tokens), max(map(len, tokens))), PAD)
    for k, t in enumerate(tokens):
        token_batch[k, : len(t)] = t
    return image_batch, widths, token_batch
