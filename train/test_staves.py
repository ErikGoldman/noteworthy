"""Tests for seq.staves on drawn staves: found where they are, with one line faint, a song's voice
staff set aside, and crops centred on each staff. Run from train/: python -m unittest test_staves"""
from __future__ import annotations

import unittest

import numpy as np

from seq.staves import STAFF_INTERLINES, Staves, find_staves, staff_crops, system_staves

INTERLINE = 10


def drawn(centres: list[int], height: int = 400, width: int = 600, faint_line: int | None = None) -> np.ndarray:
    """White crop with five-line staves centred on `centres`; noteheads scattered between lines."""
    image = np.full((height, width), 255, np.uint8)
    for staff, centre in enumerate(centres):
        for k in range(-2, 3):
            row = centre + k * INTERLINE
            image[row, :] = 200 if (staff, k) == (0, faint_line) else 0
    rng = np.random.default_rng(0)
    for _ in range(40):
        y, x = rng.integers(0, height - 6), rng.integers(0, width - 8)
        image[y:y + 6, x:x + 8] = 0
    return image


class FindStaves(unittest.TestCase):
    def test_two_staves(self):
        staves = find_staves(drawn([120, 240]))
        self.assertEqual(len(staves.centres), 2)
        np.testing.assert_allclose(staves.centres, [120, 240], atol=1)
        self.assertAlmostEqual(staves.interline, INTERLINE, delta=0.5)

    def test_a_faint_line(self):
        np.testing.assert_allclose(find_staves(drawn([120, 240], faint_line=1)).centres, [120, 240], atol=1)

    def test_no_staves(self):
        self.assertIsNone(find_staves(np.full((300, 400), 255, np.uint8)))


class SystemStaves(unittest.TestCase):
    def test_a_voice_staff_above_is_set_aside(self):
        """A song's crop: the voice staff's lower half at the top, the piano pair in the middle."""
        staves = system_staves(drawn([25, 200, 320], height=520), 2)
        np.testing.assert_allclose(staves.centres, [200, 320], atol=1)

    def test_too_few_staves(self):
        self.assertIsNone(system_staves(drawn([200]), 2))


class StaffCrops(unittest.TestCase):
    def test_each_crop_centred_on_its_staff_and_padded_white(self):
        image = drawn([60, 240])
        crops = staff_crops(image, Staves((60.0, 240.0), INTERLINE))
        half = round(STAFF_INTERLINES * INTERLINE / 2)
        self.assertEqual([c.shape for c in crops], [(2 * half, 600)] * 2)
        middle = crops[0][half - 2 * INTERLINE:half + 2 * INTERLINE + 1:INTERLINE]
        self.assertTrue((middle[:, :] == 0).all())                    # its five lines, centred
        self.assertTrue((crops[0][:half - 60 - 1] == 255).all())       # white above the image's top


if __name__ == "__main__":
    unittest.main()
