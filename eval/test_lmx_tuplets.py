"""Tests for lmx_tuplets: tuplets restored and cleared by the bar's arithmetic. Run from eval/: python -m unittest test_lmx_tuplets"""
from __future__ import annotations

import unittest
from fractions import Fraction

from lmx_tuplets import dots_mended, extra_beams_dropped, imply_tuplets

THREE_FOUR = Fraction(3)


def measure(*notes: str) -> str:
    return "measure " + " ".join(notes)


def beamed(pitches: list[str], length: str, tuplet: str | None = None) -> str:
    marks = f" {tuplet}" if tuplet else ""
    notes = [f"{p} {length}{marks}" for p in pitches]
    notes[0] += " beam:begin"
    notes[-1] += " beam:end"
    return " ".join(notes)


class UnmarkedTuplets(unittest.TestCase):
    def test_overfull_triplet_groups_become_triplets(self):
        lmx = measure(*[beamed(["C4", "D4", "E4"], "eighth")] * 3)
        self.assertEqual(imply_tuplets(lmx, THREE_FOUR).count("3in2"), 9)

    def test_three_16ths_under_an_eighths_second_beam_become_a_triplet(self):
        # Polish 055 bar 17: an eighth, then a 16th triplet under the group's second beam
        group = "F5 eighth beam:begin A4 16th beam:begin G5 16th F5 16th beam:end beam:end"
        tuplets = imply_tuplets(measure("C6 quarter", "C5 quarter", group), THREE_FOUR)
        self.assertEqual(tuplets, measure("C6 quarter", "C5 quarter",
                                          "F5 eighth beam:begin A4 16th 3in2 beam:begin G5 16th 3in2 F5 16th 3in2 beam:end beam:end"))

    def test_full_bar_of_plain_eighths_stays_plain(self):
        lmx = measure(*[beamed(["C4", "D4"], "eighth")] * 3)
        self.assertEqual(imply_tuplets(lmx, THREE_FOUR), lmx)


class RunawayTuplets(unittest.TestCase):
    def test_four_16ths_marked_as_a_quintuplet_are_plain(self):
        # Polish 018 bar 6: the bar falls short by a fifth of a beat
        quintuplet = ("B4 16th 5in4 beam:begin tuplet:start C5 16th 5in4 D5 16th 5in4 "
                      "C5 16th 5in4 beam:end tuplet:stop")
        lmx = measure(quintuplet, "G5 eighth", "rest eighth", "rest quarter")
        self.assertEqual(imply_tuplets(lmx, THREE_FOUR),
                         measure("B4 16th beam:begin C5 16th D5 16th C5 16th beam:end",
                                 "G5 eighth", "rest eighth", "rest quarter"))

    def test_whole_quintuplet_in_a_full_bar_stays(self):
        lmx = measure(beamed(["B4", "C5", "D5", "C5", "B4"], "16th", "5in4"), "G5 half")
        self.assertEqual(imply_tuplets(lmx, THREE_FOUR).count("5in4"), 5)

    def test_plain_notes_before_triplets_lose_their_marks(self):
        lmx = measure(beamed(["C4", "D4"], "eighth", "3in2"), *[beamed(["E4", "F4", "G4"], "eighth", "3in2")] * 2)
        self.assertEqual(imply_tuplets(lmx, THREE_FOUR).count("3in2"), 6)

    def test_cleared_group_loses_its_hidden_number_token(self):
        cleared = "C4 eighth 3in2 tuplet:start tuplet:hidden beam:begin D4 eighth 3in2 tuplet:stop beam:end"
        kept = "E4 eighth 3in2 tuplet:start tuplet:hidden beam:begin F4 eighth 3in2 G4 eighth 3in2 tuplet:stop beam:end"
        lmx = imply_tuplets(measure(cleared, kept, kept), THREE_FOUR)
        self.assertEqual(lmx.count("tuplet:hidden"), 2)
        self.assertTrue(lmx.startswith("measure C4 eighth beam:begin"))

    def test_marked_notes_in_a_full_bar_stay(self):
        lmx = measure(*[beamed(["C4", "D4", "E4"], "eighth", "3in2")] * 2, "F4 quarter")
        self.assertEqual(imply_tuplets(lmx, THREE_FOUR), lmx)


class ExtraBeams(unittest.TestCase):
    def test_triplet_32nds_beamed_with_an_eighth_become_16ths(self):
        # Polish 042 bar 3: the triplet has three beams where the print has two
        group = ("G4 32nd 3in2 beam:begin beam:begin beam:begin tuplet:start A4 32nd 3in2 "
                 "B4 32nd 3in2 beam:end beam:end tuplet:stop C5 eighth beam:end")
        dropped = extra_beams_dropped(measure(group, "D5 quarter", "E5 half"), Fraction(4))
        self.assertIn("G4 16th 3in2 beam:begin beam:begin tuplet:start", dropped)
        self.assertNotIn("32nd", dropped)

    def test_32nds_that_should_be_16ths_lose_a_beam(self):
        lmx = measure(beamed(["C5", "D5", "E5", "F5"], "32nd"), "G5 quarter", "A5 quarter")
        self.assertIn("16th", extra_beams_dropped(lmx, THREE_FOUR))



class Dots(unittest.TestCase):
    THREE_EIGHT = Fraction(3, 2)

    def test_dot_read_where_there_is_none_goes(self):
        # Polish 030 bar 21 in 3/8: a quarter read as dotted, then two 16ths
        lower = "backup quarter backup eighth D3 eighth E3 quarter"
        lmx = measure("C4 quarter dot", "B3 16th beam:begin beam:begin", "F4 16th beam:end beam:end", lower)
        self.assertEqual(dots_mended(lmx, self.THREE_EIGHT), measure("C4 quarter", "B3 16th beam:begin beam:begin",
                                                                     "F4 16th beam:end beam:end", lower))

    def test_short_voice_is_not_dotted(self):
        # a second voice may stop before the barline; OLiMPiC crops got worse when it was dotted
        lower = "backup quarter D3 quarter dot"
        lmx = measure("C4 quarter", "chord E4 quarter", lower)
        self.assertEqual(dots_mended(lmx, self.THREE_EIGHT), lmx)

    def test_two_events_that_could_lose_the_dot_leave_it_alone(self):
        lower = "backup quarter backup eighth D3 quarter dot"
        lmx = measure("C4 eighth dot", "D4 eighth dot", "E4 16th", lower)
        self.assertEqual(dots_mended(lmx, self.THREE_EIGHT), lmx)

    def test_bar_no_voice_fills_is_left_alone(self):
        # Polish 069 bar 12: a short bar before a repeat, short on both staves
        lmx = measure("C4 quarter", "backup quarter", "D3 eighth", "E3 16th")
        self.assertEqual(dots_mended(lmx, self.THREE_EIGHT), lmx)

    def test_full_bar_is_untouched(self):
        lmx = measure("C4 quarter dot")
        self.assertEqual(dots_mended(lmx, self.THREE_EIGHT), lmx)


if __name__ == "__main__":
    unittest.main()
