"""Tests for seq.data's augmentation: it may move, scale, blur and fade the music, never remove it.
Run from train/: python -m unittest test_seq_data"""
from __future__ import annotations

import unittest

import numpy as np

from omr.augment import stroke_weight
from seq.data import reframe
from test_staves import drawn


class Reframe(unittest.TestCase):
    def test_margins_never_cut_ink(self):
        """A crop cut tight to its staves (OLiMPiC's are) keeps every staff's outer lines."""
        tight = drawn([20, 80], height=101, width=600)
        for seed in range(200):
            image, moved = reframe(np.random.default_rng(seed), tight)
            for row in (moved(0), moved(100)):
                self.assertGreaterEqual(row, 0, f"seed {seed}")
                self.assertLess(row, image.shape[0], f"seed {seed}")


class StrokeWeight(unittest.TestCase):
    def test_never_thinned_never_lighter(self):
        image = drawn([150, 270], height=420, width=900)
        for seed in range(20):
            out = stroke_weight(np.random.default_rng(seed), image, 34, thin=False)
            self.assertTrue((out <= image).all(), f"seed {seed}")


if __name__ == "__main__":
    unittest.main()
