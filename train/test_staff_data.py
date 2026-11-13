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
from seq.staff_data import STAFF_HEIGHT, StaffDataset, collate_staves
from seq.staff_positions import staff_count
from seq.staves import find_staves
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


if __name__ == "__main__":
    unittest.main()
