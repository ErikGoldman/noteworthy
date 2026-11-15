"""Tests for outer_notes' note kinds. Run from eval/: python -m unittest test_outer_notes"""
from __future__ import annotations

import unittest

from outer_notes import note_kinds

HEAD = "measure clef:G2 staff:1 clef:F4 staff:2"


def kinds(lmx: str) -> dict[str, str]:
    tokens = f"{HEAD} {lmx}".split()
    return {f"{tokens[i]}@{i}": kind for i, kind in note_kinds(tokens).items()}


class NoteKinds(unittest.TestCase):
    def test_far_above_the_top_staff(self):
        """F5 is the treble's top line; E6 sits on the third ledger line above, D6 just below it."""
        self.assertEqual(list(kinds("E6 voice:1 quarter staff:1 D6 quarter").values()), ["far outside", "elsewhere"])

    def test_far_below_the_bottom_staff(self):
        """G2 is the bass's bottom line; A1 sits on the third ledger line below, B1 just above it."""
        self.assertEqual(list(kinds("A1 voice:5 quarter staff:2 chord B1 quarter").values()), ["far outside", "elsewhere"])

    def test_between_the_staves_is_never_outside(self):
        """Far below the treble, or far above the bass, is between the staves."""
        self.assertEqual(set(kinds("G3 voice:1 quarter staff:1 E4 voice:5 quarter staff:2").values()), {"elsewhere"})

    def test_a_chord_note_keeps_its_staff(self):
        """The chord's second note names no staff: it is on the bass staff, far below it."""
        found = kinds("A2 voice:5 quarter staff:2 chord A1 quarter")
        self.assertEqual(found["A1@10"], "far outside")

    def test_a_system_with_no_bass_notes_still_has_a_bass_staff(self):
        """Every note on the treble: one far below it is still between the staves."""
        self.assertEqual(list(kinds("C3 voice:1 quarter staff:1").values()), ["elsewhere"])

    def test_under_a_sign(self):
        self.assertEqual(list(kinds("C6 ottava:8va voice:1 quarter staff:1").values()), ["under a sign"])

    def test_one_staff_is_both_the_top_and_the_bottom(self):
        """A one-staff label names no staff: far above and far below its staff are both outside."""
        found = note_kinds("measure clef:G2 E6 voice:1 quarter F3 quarter C5 quarter".split())
        self.assertEqual(list(found.values()), ["far outside", "far outside", "elsewhere"])


if __name__ == "__main__":
    unittest.main()
