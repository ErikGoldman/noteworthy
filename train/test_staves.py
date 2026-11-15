"""Tests for seq.staves on drawn staves: found where they are, with one line faint, a song's voice
staff set aside, and crops centred on each staff. Run from train/: python -m unittest test_staves"""
from __future__ import annotations

import unittest

import numpy as np

from seq.staves import STAFF_INTERLINES, Staves, find_staves, staff_crops, system_staves

INTERLINE = 10
# Crops taller than FINDING_HEIGHT are looked at shrunk, their centres found to a row of that.
CENTRE_TOLERANCE = INTERLINE / 4


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
        np.testing.assert_allclose(staves.centres, [120, 240], atol=CENTRE_TOLERANCE)
        self.assertAlmostEqual(staves.interline, INTERLINE, delta=0.5)

    def test_a_faint_line(self):
        np.testing.assert_allclose(find_staves(drawn([120, 240], faint_line=1)).centres, [120, 240], atol=CENTRE_TOLERANCE)

    def test_no_staves(self):
        self.assertIsNone(find_staves(np.full((300, 400), 255, np.uint8)))


class SystemStaves(unittest.TestCase):
    def test_a_voice_staff_above_is_set_aside(self):
        """A song's crop: the voice staff's lower half at the top, the piano pair in the middle."""
        staves = system_staves(drawn([25, 200, 320], height=520), 2)
        np.testing.assert_allclose(staves.centres, [200, 320], atol=CENTRE_TOLERANCE)

    def test_a_faint_staff_beside_a_strong_one(self):
        """A system's second staff matching at a third of the first (blur, or beams over its lines)."""
        image = drawn([120, 240])
        image[240 - 2 * INTERLINE:240 + 2 * INTERLINE + 1:INTERLINE] = 170
        np.testing.assert_allclose(system_staves(image, 2).centres, [120, 240], atol=CENTRE_TOLERANCE)

    def test_ledger_lines_are_not_a_staff(self):
        """Chords high above the staves, each with five short ledger lines an interline apart: in
        rows, half as dark as a staff's lines, and taken for a staff, the pair of it and the
        upper staff is nearer the crop's middle than the system is; but none runs across."""
        image = drawn([230, 350], height=440)
        for x in range(20, 600, 40):
            image[90:131:INTERLINE, x:x + 22] = 0
        np.testing.assert_allclose(system_staves(image, 2).centres, [230, 350], atol=CENTRE_TOLERANCE)

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

    def test_outer_staves_slide_outward(self):
        """Slid one interline: the top staff's crop sits a line lower in its window (more room
        above), the bottom staff's a line higher; a middle staff stays centred."""
        image = drawn([100, 220, 340], height=460)
        half = round(STAFF_INTERLINES * INTERLINE / 2)
        crops = staff_crops(image, Staves((100.0, 220.0, 340.0), INTERLINE), outward=1.0)
        lines = [[r for r in range(len(c)) if (c[r, :] == 0).mean() > 0.9] for c in crops]
        self.assertEqual(lines[0][2], half + INTERLINE)    # top staff's middle line, a line lower
        self.assertEqual(lines[1][2], half)                # middle staff, centred
        self.assertEqual(lines[2][2], half - INTERLINE)    # bottom staff's, a line higher

    def test_a_staff_alone_stays_centred(self):
        crops = staff_crops(drawn([200]), Staves((200.0,), INTERLINE), outward=1.0)
        lines = [r for r in range(len(crops[0])) if (crops[0][r, :] == 0).mean() > 0.9]
        self.assertEqual(lines[2], round(STAFF_INTERLINES * INTERLINE / 2))

    def test_a_staff_centred_beyond_the_image_still_full_height(self):
        """Augmentation's margins may crop a tall system through its lower staff."""
        image = drawn([60, 200], height=250)
        crops = staff_crops(image, Staves((60.0, 280.0), INTERLINE))
        self.assertEqual([c.shape[0] for c in crops], [2 * round(STAFF_INTERLINES * INTERLINE / 2)] * 2)
        self.assertTrue((crops[1][-30:] == 255).all())

    def test_cut_between_shares_no_ink_with_a_neighbour(self):
        """Staves 120 apart overlap in their crops (each 2 * half = 150 tall); cut between, each
        crop is white past the midpoint (row 120 of the image), its own side untouched."""
        image = drawn([60, 180], height=300)
        image[110:131, :] = 0                         # ink astride the midpoint
        half = round(STAFF_INTERLINES * INTERLINE / 2)
        whole = staff_crops(image, Staves((60.0, 180.0), INTERLINE))
        top, bottom = staff_crops(image, Staves((60.0, 180.0), INTERLINE), cut_between=True)
        in_top, in_bottom = 120 - (60 - half), 120 - (180 - half)    # the midpoint in each crop's rows
        self.assertTrue(0 < in_top < 2 * half and 0 < in_bottom < 2 * half)
        self.assertTrue((top[in_top:] == 255).all())
        self.assertTrue((top[:in_top] == whole[0][:in_top]).all())
        self.assertTrue((bottom[:in_bottom] == 255).all())
        self.assertTrue((bottom[in_bottom:] == whole[1][in_bottom:]).all())
        self.assertTrue((whole[0][in_top - 5:in_top] == 0).all())    # there was ink to cut

    def test_cut_between_leaves_a_staff_alone_whole(self):
        image = drawn([200])
        self.assertTrue((staff_crops(image, Staves((200.0,), INTERLINE), cut_between=True)[0]
                         == staff_crops(image, Staves((200.0,), INTERLINE))[0]).all())


if __name__ == "__main__":
    unittest.main()
