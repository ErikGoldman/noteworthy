"""Tests for the training augmentation: omr.augment's smooth fields and seq.data.binarize.
Run from train/: python -m unittest test_augment"""
from __future__ import annotations

import unittest

import numpy as np

from omr.augment import _smooth_field, capture_degradation
from seq.data import binarize

AMPLITUDE = 25.0


class SmoothFieldTest(unittest.TestCase):
    def test_lighting_field_stays_near_its_amplitude_on_wide_crops(self):
        """The lighting field's sigma is a third of the crop's width: a grid of about 3 by 8."""
        rng = np.random.default_rng(0)
        for width in (300, 981, 2048):
            peaks = [np.abs(_smooth_field(rng, (256, width), width / 3, AMPLITUDE)).max() for _ in range(500)]
            self.assertLess(max(peaks), 6 * AMPLITUDE, f"width {width}")

    def test_capture_never_blanks_a_crop(self):
        rng = np.random.default_rng(1)
        crop = np.full((256, 2048), 255, np.uint8)
        crop[100:110] = 0
        for _ in range(300):
            out = capture_degradation(rng, crop, 256 / 30).astype(int)
            self.assertGreater(np.median(out[150:]) - np.median(out[102:108]), 50)


class BinarizeTest(unittest.TestCase):
    def test_grey_paper_stays_white(self):
        """Smashcima's and MusiCorpus's paper is about 173-215: a fixed threshold over it blackened the page."""
        rng = np.random.default_rng(2)
        crop = np.clip(np.random.default_rng(3).normal(182, 6, (256, 800)), 0, 255).astype(np.uint8)
        crop[100:104] = 3
        for _ in range(300):
            out = binarize(rng, crop)
            self.assertLess(np.mean(out[150:] == 0), 0.01)
            self.assertTrue((out[100:104] == 0).all())


if __name__ == "__main__":
    unittest.main()
