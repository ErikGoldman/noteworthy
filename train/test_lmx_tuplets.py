"""Tests for seq.lmx_tuplets: tuplets restored and cleared by the bar's arithmetic. Run from train/: python -m unittest test_lmx_tuplets"""
from __future__ import annotations

import unittest
from fractions import Fraction

from seq.lmx_tuplets import (ALL_RULES, PageRule, bar_at_end, bars_in_force, dots_mended, extra_beams_dropped, imply_tuplets,
                              length_at, measures_of, runs_as_tuplets, with_page_rules)

THREE_FOUR = Fraction(3)


def measure(*notes: str) -> str:
    return "measure " + " ".join(notes)


def beamed(pitches: list[str], length: str, tuplet: str | None = None) -> str:
    marks = f" {tuplet}" if tuplet else ""
    notes = [f"{p} {length}{marks}" for p in pitches]
    notes[0] += " beam:begin"
    notes[-1] += " beam:end"
    return " ".join(notes)


class Lengths(unittest.TestCase):
    def test_dotted_tuplet_note_keeps_its_dot_after_the_ratio(self):
        self.assertEqual(length_at("A4 half 2in1 dot".split(), 1), (Fraction(3, 2), 2))

    def test_dots_before_a_ratio_count_too(self):
        self.assertEqual(length_at("A4 eighth dot 3in2".split(), 1), (Fraction(1, 2), 3))

    def test_plain_dotted_note(self):
        self.assertEqual(length_at("A4 quarter dot dot".split(), 1), (Fraction(7, 4), None))


# A second voice that fills a 3/4 bar: it confirms the meter, which implying tuplets needs.
FILLING = "backup half dot C3 voice:2 half dot"


class UnmarkedTuplets(unittest.TestCase):
    def test_overfull_triplet_groups_become_triplets(self):
        lmx = measure(*[beamed(["C4", "D4", "E4"], "eighth")] * 3, FILLING)
        self.assertEqual(imply_tuplets(lmx, THREE_FOUR).count("3in2"), 9)

    def test_three_16ths_under_an_eighths_second_beam_become_a_triplet(self):
        # Polish 055 bar 17: an eighth, then a 16th triplet under the group's second beam
        group = "F5 eighth beam:begin A4 16th beam:begin G5 16th F5 16th beam:end beam:end"
        tuplets = imply_tuplets(measure("C6 quarter", "C5 quarter", group, FILLING), THREE_FOUR)
        self.assertEqual(tuplets, measure("C6 quarter", "C5 quarter",
                                          "F5 eighth beam:begin A4 16th 3in2 beam:begin G5 16th 3in2 F5 16th 3in2 beam:end beam:end",
                                          FILLING))

    def test_no_tuplets_where_every_voice_overflows(self):
        """Polish 051 bar 22, 071: a barline missed or a meter misread, not triplets."""
        lmx = measure(*[beamed(["C4", "D4", "E4"], "eighth")] * 3, "backup half dot C3 voice:2 whole")
        self.assertEqual(imply_tuplets(lmx, THREE_FOUR), lmx)

    def test_full_bar_of_plain_eighths_stays_plain(self):
        lmx = measure(*[beamed(["C4", "D4"], "eighth")] * 3)
        self.assertEqual(imply_tuplets(lmx, THREE_FOUR), lmx)


class RulesChosen(unittest.TestCase):
    OVERFULL = measure(*[beamed(["C4", "D4", "E4"], "eighth")] * 3, FILLING)
    RUNAWAY = measure(beamed(["C4", "D4"], "eighth", "3in2"), *[beamed(["E4", "F4", "G4"], "eighth", "3in2")] * 2)

    def test_implied_tuplets_left_out(self):
        self.assertEqual(imply_tuplets(self.OVERFULL, THREE_FOUR, ALL_RULES - {PageRule.IMPLIED_TUPLETS}), self.OVERFULL)

    def test_wrong_marks_kept_when_clearing_is_left_out(self):
        self.assertEqual(imply_tuplets(self.RUNAWAY, THREE_FOUR, ALL_RULES - {PageRule.WRONG_MARKS_CLEARED}), self.RUNAWAY)

    def test_no_rules_leaves_a_reading_as_read(self):
        for lmx in (self.OVERFULL, self.RUNAWAY):
            self.assertEqual(with_page_rules(lmx, THREE_FOUR, frozenset()), lmx)

    def test_every_rule_but_those_on_trial_is_the_default(self):
        self.assertEqual(with_page_rules(self.OVERFULL, THREE_FOUR), imply_tuplets(self.OVERFULL, THREE_FOUR))


class MeterInForce(unittest.TestCase):
    """A system can change meter partway (OLiMPiC 6259566 alternates 6/8 and 2/4)."""
    CHANGING = ("measure time beats:6 beat-type:8 C4 quarter dot D4 quarter dot "
                "measure E4 quarter dot F4 quarter dot measure time beats:2 beat-type:4 G4 half")

    def bars(self, lmx: str, carried=None):
        tokens = lmx.split()
        return bars_in_force(tokens, carried, measures_of(tokens))

    def test_each_measure_takes_the_signature_before_it(self):
        self.assertEqual(self.bars(self.CHANGING), [3, 3, 2])

    def test_carried_meter_holds_until_one_is_written(self):
        lmx = "measure C4 half dot measure time beats:2 beat-type:4 D4 half"
        self.assertEqual(self.bars(lmx, carried=THREE_FOUR), [3, 2])

    def test_the_last_signature_is_carried_on(self):
        self.assertEqual(bar_at_end(self.CHANGING.split(), None), 2)
        self.assertEqual(bar_at_end("measure C4 half dot".split(), THREE_FOUR), 3)

    def test_triplets_implied_by_the_measures_own_meter(self):
        """2/4 after 6/8: two groups of three eighths over a half note are triplets. Judged by the
        system's first signature (6/8), they fit as written and were left plain."""
        triplets = " ".join([beamed(["E4", "F4", "G4"], "eighth")] * 2)
        lmx = ("measure time beats:6 beat-type:8 C4 quarter dot D4 quarter dot "
               f"measure time beats:2 beat-type:4 {triplets} backup half C3 voice:2 half")
        self.assertEqual(imply_tuplets(lmx).count("3in2"), 6)


class RunsOfThree(unittest.TestCase):
    """runs_as_tuplets: triplets the model wrote plain, whatever their beaming, in both hands."""
    def ratios(self, lmx: str) -> list[str]:
        return [t for t in lmx.split() if t in ("3in2", "6in4")]

    def test_quarter_then_three_eighths_in_one_beam_becomes_a_triplet(self):
        bar = measure("C4 quarter", beamed(["D4", "E4", "F4"], "eighth"), "G4 quarter")  # 3/4 overfull by one eighth
        self.assertEqual(self.ratios(runs_as_tuplets(bar, THREE_FOUR)), ["3in2"] * 3)

    def test_two_hands_of_plain_triplets_both_mended(self):
        upper = " ".join([beamed(["C5", "D5", "E5", "F5", "G5", "A5"], "eighth")] + [beamed(["B4", "C5", "D5"], "eighth")])
        lower = " ".join([beamed(["C3", "D3", "E3", "F3", "G3", "A3"], "eighth")] + [beamed(["B2", "C3", "D3"], "eighth")])
        bar = measure(upper, "backup half dot", lower)  # every voice overfull: imply_tuplets leaves it
        self.assertEqual(imply_tuplets(bar, THREE_FOUR), bar)
        self.assertEqual(self.ratios(runs_as_tuplets(bar, THREE_FOUR)).count("3in2") + self.ratios(runs_as_tuplets(bar, THREE_FOUR)).count("6in4"), 18)

    def test_a_beamed_six_becomes_a_sextuplet(self):
        bar = measure(beamed(["C4", "D4", "E4", "F4", "G4", "A4"], "16th"), "B4 quarter", "C5 quarter")  # 3/4 + an eighth
        self.assertEqual(self.ratios(runs_as_tuplets(bar, THREE_FOUR)), ["6in4"] * 6)

    def test_a_bar_that_fits_is_left_alone(self):
        bar = measure(*[beamed(["C4", "D4"], "eighth")] * 3)
        self.assertEqual(runs_as_tuplets(bar, THREE_FOUR), bar)

    def test_nothing_marked_unless_the_bar_comes_out_exact(self):
        bar = measure("C4 quarter", beamed(["D4", "E4", "F4", "G4"], "eighth"))  # 3/4 + a quarter: no triplets fit
        self.assertEqual(runs_as_tuplets(bar, THREE_FOUR), bar)

    def test_on_trial_so_not_a_default_rule(self):
        self.assertNotIn(PageRule.RUNS_OF_THREE, ALL_RULES)


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

    def test_a_grace_eighth_read_into_a_16ths_beam_leaves_it_alone(self):
        # Polish 065 (seq514): every note in the group would lose a beam, and the grace has none to lose
        lmx = measure("grace:slash A4 eighth beam:begin G4 16th beam:end beam:backward-hook",
                      "B4 half", "C5 eighth")
        self.assertEqual(extra_beams_dropped(lmx, THREE_FOUR), lmx)


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
