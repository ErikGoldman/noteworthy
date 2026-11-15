"""Tests for tuplet_bars. Run from eval/: python -m unittest test_tuplet_bars"""
from __future__ import annotations

import unittest

from tuplet_bars import right_bars

TRIPLETS = "measure C4 eighth 3in2 tuplet:start D4 eighth 3in2 E4 eighth 3in2 tuplet:stop"
HIDDEN = "measure C4 eighth 3in2 tuplet:start tuplet:hidden D4 eighth 3in2 E4 eighth 3in2 tuplet:stop"
PLAIN = "measure C4 eighth D4 eighth E4 eighth"
QUARTERS = "measure C4 quarter D4 quarter"


class RightBars(unittest.TestCase):
    def test_all_read_right(self):
        label = f"{TRIPLETS} {HIDDEN} {QUARTERS}"
        self.assertEqual(right_bars(label, label), [("tuplets, numbers shown", True),
                                                    ("tuplets, a number hidden", True), ("no tuplets", True)])

    def test_unnumbered_triplets_read_as_plain_eighths(self):
        """The Polish failure: the hidden bar read without its tuplet; the others still right."""
        found = right_bars(f"{TRIPLETS} {PLAIN} {QUARTERS}", f"{TRIPLETS} {HIDDEN} {QUARTERS}")
        self.assertEqual([ok for _, ok in found], [True, False, True])

    def test_pitches_do_not_count(self):
        self.assertTrue(right_bars(QUARTERS.replace("C4", "G4"), QUARTERS)[0][1])

    def test_a_dropped_bar_shifts_nothing_else(self):
        found = right_bars(f"{TRIPLETS} {QUARTERS}", f"{TRIPLETS} {HIDDEN} {QUARTERS}")
        self.assertEqual([ok for _, ok in found], [True, False, True])


if __name__ == "__main__":
    unittest.main()
