"""PyTorch dataset over rendered pages: random scale-jittered crops, augmented."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import augment
from .channels import CHANNELS, map_packed, unpack_masks
from .held_out import is_held_out

# The C++ pipeline rescales every page so its staff interline equals this.
NORMALIZED_INTERLINE_PX = 20.0
MIN_INK_FRACTION = 0.02
CROP_ATTEMPTS = 8


def channels_present(packed: np.ndarray) -> np.ndarray:
    """Indices of the channels with any pixel set, from packed masks."""
    any_bits = np.bitwise_or.reduce(packed.reshape(-1, packed.shape[-1]), axis=0)
    return np.flatnonzero(np.unpackbits(any_bits)[: len(CHANNELS)])


def random_pixel_of(rng, packed: np.ndarray, channel: int) -> tuple[int, int]:
    """(y, x) of a random pixel of one channel; packbits stores channel c as bit 7 - c % 8."""
    ys, xs = np.nonzero(packed[..., channel // 8] & (0x80 >> (channel % 8)))
    k = rng.integers(len(ys))
    return int(ys[k]), int(xs[k])


def init_worker(_worker_id: int) -> None:
    """DataLoader worker setup: one thread each. Otherwise every worker's OpenCV and
    torch spin up a thread per core, and many workers oversubscribe the CPU badly."""
    cv2.setNumThreads(1)
    torch.set_num_threads(1)


def list_pages(root: str | Path) -> list[Path]:
    return sorted(p.with_suffix("") for p in Path(root).glob("*.json"))


def split_by_source(pages: list[Path]) -> tuple[list[Path], list[Path]]:
    """(train, val): pages of held-out source scores go to val, so no score is in both."""
    train, val = [], []
    for page in pages:
        source = Path(json.loads(page.with_suffix(".json").read_text())["source"])
        (val if is_held_out(source) else train).append(page)
    return train, val


class PageCrops(Dataset):
    """Each item is `crops_per_page` crops from one page: decoding a full page costs far
    more than cropping and augmenting it, so each load is amortised over several crops.
    Use `collate_crops` with a DataLoader.

    Symbol classes are very unevenly common (a 32nd flag is on ~2% of pages and covers a
    few dozen pixels), so `balanced_fraction` of crops are centred on a pixel of a class
    drawn uniformly from those on the page; the rest are placed at random."""

    def __init__(self, pages: list[Path], crop: int, length: int, crops_per_page: int = 4,
                 scale_range=(0.8, 1.25), augment_crops=True, seed=0, balanced_fraction=0.0):
        self.pages = pages
        self.crop = crop
        self.length = length
        self.crops_per_page = crops_per_page
        self.scale_range = scale_range
        self.augment_crops = augment_crops
        self.seed = seed
        self.balanced_fraction = balanced_fraction

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int):
        rng = np.random.default_rng((self.seed, i, torch.initial_seed() % 2**32) if self.augment_crops else (self.seed, i))
        page = self.pages[rng.integers(len(self.pages))]
        meta = json.loads(page.with_suffix(".json").read_text())
        image = cv2.imread(str(page.with_suffix(".png")), cv2.IMREAD_GRAYSCALE)
        packed = np.load(page.with_suffix(".npz"))["masks"]
        present = channels_present(packed)
        crops = [self._one_crop(rng, meta, image, packed, present) for _ in range(self.crops_per_page)]
        return torch.stack([c[0] for c in crops]), torch.stack([c[1] for c in crops])

    def _one_crop(self, rng, meta, image, packed, present):
        # Output interline = page interline * scale; jitter around the normalized size.
        target_interline = NORMALIZED_INTERLINE_PX * rng.uniform(*self.scale_range)
        scale = target_interline / meta["interline_px"]
        src = int(round(self.crop / scale))
        if len(present) and rng.random() < self.balanced_fraction:
            centre = random_pixel_of(rng, packed, int(rng.choice(present)))
            img_crop, mask_crop = self._crop_around(rng, image, packed, src, centre)
        else:
            img_crop, mask_crop = self._pick_crop(rng, image, packed, src)
        img_crop = cv2.resize(img_crop, (self.crop, self.crop), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
        mask_crop = map_packed(mask_crop, lambda m: cv2.resize(m, (self.crop, self.crop), interpolation=cv2.INTER_NEAREST))

        if self.augment_crops:
            img_crop, mask_crop = augment(rng, img_crop, mask_crop, target_interline, meta["font"] == "Petaluma")

        # uint8 on the CPU side: float conversion happens on the GPU (see to_model_input),
        # which keeps worker -> trainer transfers 4x smaller.
        x = torch.from_numpy(img_crop).unsqueeze(0)
        y = torch.from_numpy(unpack_masks(mask_crop).view(np.uint8)).permute(2, 0, 1)
        return x, y

    @staticmethod
    def _pad_to(image, packed, size):
        pad_h, pad_w = max(0, size - image.shape[0]), max(0, size - image.shape[1])
        if pad_h or pad_w:
            image = np.pad(image, ((0, pad_h), (0, pad_w)), constant_values=255)
            packed = np.pad(packed, ((0, pad_h), (0, pad_w), (0, 0)))
        return image, packed

    def _crop_around(self, rng, image, packed, size, centre):
        """A crop containing `centre`, at a random offset so the symbol is not always mid-crop."""
        image, packed = self._pad_to(image, packed, size)
        h, w = image.shape
        cy, cx = centre
        y0 = int(np.clip(cy - rng.integers(size // 4, 3 * size // 4 + 1), 0, h - size))
        x0 = int(np.clip(cx - rng.integers(size // 4, 3 * size // 4 + 1), 0, w - size))
        return image[y0:y0 + size, x0:x0 + size], packed[y0:y0 + size, x0:x0 + size]

    def _pick_crop(self, rng, image, packed, size):
        image, packed = self._pad_to(image, packed, size)
        h, w = image.shape
        for _ in range(CROP_ATTEMPTS):
            y0, x0 = rng.integers(0, h - size + 1), rng.integers(0, w - size + 1)
            img = image[y0:y0 + size, x0:x0 + size]
            if (img < 128).mean() >= MIN_INK_FRACTION:
                break
        return img, packed[y0:y0 + size, x0:x0 + size]



def collate_crops(items):
    """Flattens a list of (K, ...) crop stacks into one batch."""
    return torch.cat([x for x, _ in items]), torch.cat([y for _, y in items])


def to_model_input(x: torch.Tensor, y: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor]:
    """uint8 batch from the loader -> float image in [0, 1] (paper = 1) and float targets."""
    x = x.to(device, non_blocking=True).float().div_(255.0)
    y = y.to(device, non_blocking=True).float()
    return x, y
