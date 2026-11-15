"""Tests for seq.staff_data: staff crops centred on their staves, augmented or not.
Run from train/: python -m unittest test_staff_data"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from seq.data import Sample
from seq.staff_data import ROW_MULTIPLE, STAFF_HEIGHT, InterlineSystemDataset, StaffDataset, collate_staves, collate_systems
from seq.staff_positions import staff_count
from seq.staves import STAFF_INTERLINES, Staves, find_staves
from seq.vocab import Vocab
from test_staves import drawn
from train_seq import VOCABULARY

LMX = "measure clef:G2 staff:1 clef:F4 staff:2 C5 voice:1 whole staff:1 backup whole C3 voice:5 whole staff:2"


class StaffCrops(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        root = Path(cls.folder.name)
        cv2.imwrite(str(root / "s.png"), drawn([150, 270], height=420, width=900))
        (root / "s.lmx").write_text(LMX)
        cls.sample = Sample(root / "s.png", root / "s.lmx")
        cls.staves = find_staves(cv2.imread(str(root / "s.png"), cv2.IMREAD_GRAYSCALE))
        cls.vocab = Vocab.load(VOCABULARY)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def crop_centres(self, train: bool, seed: int) -> list[float]:
        torch.manual_seed(seed)
        dataset = StaffDataset([(self.sample, self.staves)], self.vocab, train, 64)
        staves, _ = dataset[0]
        crops = [(255 - s[0].numpy() * 255).astype(np.uint8) for s in staves]
        self.assertEqual([c.shape[0] for c in crops], [STAFF_HEIGHT, STAFF_HEIGHT])
        return [find_staves(c).centres for c in crops]

    def test_staff_count(self):
        self.assertEqual(staff_count(LMX), 2)

    def test_each_crop_centred_on_its_staff(self):
        for found in self.crop_centres(train=False, seed=0):
            self.assertEqual(len(found), 1)
            self.assertAlmostEqual(found[0], STAFF_HEIGHT / 2, delta=2)

    def test_augmented_crops_still_centred(self):
        """Margins change and scale the system, the staff positions with it: within an interline."""
        interline = STAFF_HEIGHT / 15
        for seed in range(20):
            for found in self.crop_centres(train=True, seed=seed):
                self.assertTrue(found, f"seed {seed}: no staff found in a crop")
                nearest = min(found, key=lambda c: abs(c - STAFF_HEIGHT / 2))
                self.assertLess(abs(nearest - STAFF_HEIGHT / 2), interline, f"seed {seed}")

    def test_collate(self):
        dataset = StaffDataset([(self.sample, self.staves)] * 3, self.vocab, False, 64)
        staves, widths, tokens = collate_staves([dataset[i] for i in range(3)])
        self.assertEqual(tuple(staves.pixels.shape[:3]), (3, 2, 1))
        self.assertEqual(staves.counts.tolist(), [2, 2, 2])
        self.assertEqual(tokens.shape[0], 3)

    def test_meter_context_puts_the_bar_in_front(self):
        """A whole note a bar: the label starts `meter beats:4 beat-type:4` (after <bos>), as
        SystemDataset's do; without the option, the label as it is."""
        with_meter = StaffDataset([(self.sample, self.staves)], self.vocab, False, 64, meter_context=True)[0][1]
        plain = StaffDataset([(self.sample, self.staves)], self.vocab, False, 64)[0][1]
        self.assertEqual(self.vocab.decode(with_meter[1:4].tolist()).split(), ["meter", "beats:4", "beat-type:4"])
        torch.testing.assert_close(with_meter[4:], plain[1:])


class InterlineSystems(unittest.TestCase):
    """Whole systems at staff scale: the height follows the staves, the interline is fixed."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root, cls.vocab = Path(cls.tmp.name), Vocab.load(VOCABULARY)
        cls.items = []
        for name, centres, height in (("two", [120, 240], 360), ("three", [120, 240, 360], 480)):
            cv2.imwrite(str(root / f"{name}.png"), drawn(centres, height=height))
            (root / f"{name}.lmx").write_text(LMX)
            cls.items.append((Sample(root / f"{name}.png", root / f"{name}.lmx"), Staves(tuple(map(float, centres)), 10.0)))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_height_follows_the_staves(self):
        dataset = InterlineSystemDataset(self.items, self.vocab, False, 64)
        two, three = (dataset[i][0][0] for i in range(2))
        scale = STAFF_HEIGHT / STAFF_INTERLINES / 10.0          # interline 10 px drawn
        self.assertEqual(two.shape[-2], round(360 * scale))
        self.assertEqual(three.shape[-2], round(480 * scale))

    def test_batched_to_one_height(self):
        dataset = InterlineSystemDataset(self.items, self.vocab, True, 64)
        images, widths, tokens = collate_systems([dataset[0], dataset[1]])
        self.assertEqual(images.pixels.shape[-2] % ROW_MULTIPLE, 0)
        self.assertGreaterEqual(images.pixels.shape[-2], max(dataset[i][0][0].shape[-2] for i in range(2)))
        self.assertEqual(images.counts.tolist(), [2, 3])
        self.assertEqual(images.centres.shape, (2, 3))

    def test_staves_follow_the_augmented_crop(self):
        """Wherever augmentation puts a staff, the staves found in the crop are where it says."""
        dataset = InterlineSystemDataset(self.items, self.vocab, True, 64)
        for _ in range(5):
            (ink, staves), _ = dataset[1]
            found = find_staves((255 - ink[0].numpy() * 255).astype(np.uint8), 3)    # to_tensor's ink is bright
            self.assertIsNotNone(found)
            for given, seen in zip(staves.centres, found.centres):
                self.assertLess(abs(given - seen), staves.interline)


class StaffJitter(unittest.TestCase):
    """--staff-jitter: training crops centred off their staff; reading them never."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cv2.imwrite(str(root / "two.png"), drawn([120, 240], height=360))
        (root / "two.lmx").write_text(LMX)
        cls.items = [(Sample(root / "two.png", root / "two.lmx"), Staves((120.0, 240.0), 10.0))]
        cls.vocab = Vocab.load(VOCABULARY)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def staff_line_rows(self, dataset) -> list[int]:
        """Each staff crop's most inked row (to_tensor's ink is bright): where its staff sits in it."""
        crops, _ = dataset[0]
        return [int(c[0].mean(dim=1).argmax()) for c in crops]

    def test_reading_is_never_jittered(self):
        plain = StaffDataset(self.items, self.vocab, False, 64)
        jittered = StaffDataset(self.items, self.vocab, False, 64, jitter=2.0)
        torch.testing.assert_close(plain[0][0], jittered[0][0])

    def test_training_crops_move_on_their_staff(self):
        torch.manual_seed(0)
        jittered = StaffDataset(self.items, self.vocab, True, 64, jitter=2.0)
        rows = {tuple(self.staff_line_rows(jittered)) for _ in range(8)}
        self.assertGreater(len(rows), 1)


if __name__ == "__main__":
    unittest.main()
