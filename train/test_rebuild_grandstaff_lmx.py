"""Tests for seq.kern and rebuild_grandstaff_lmx: GrandStaff-LMX's clefs, keys and meters as its
**kern sources have them. Run from train/: python -m unittest test_rebuild_grandstaff_lmx"""
from __future__ import annotations

import unittest
from fractions import Fraction

import music21

import rebuild_grandstaff_lmx  # noqa: F401  (mends music21's ConverterHumdrum.parseData)
from seq import kern
from seq.kern import CLEF, KEY, METER, Change

# Two measures of 2/4 in two flats; the lower staff (left spine) splits into two voices before
# its signatures, as in the systems the release labels treble and C major there, and goes to
# treble a beat into measure 2.
SPLIT_FIRST = """**kern\t**kern
*^\t*
*clefF4\t*clefF4\t*clefG2
*k[b-e-]\t*k[b-e-]\t*k[b-e-]
*M2/4\t*M2/4\t*M2/4
=-\t=-\t=-
4C\t4CC\t4c
4D\t4DD\t4d
=\t=\t=
*v\t*v\t*
4E-\t4e-
*clefG2\t*
4g\t4f
=\t=
*-\t*-
"""


def parsed(text: str) -> music21.stream.Score:
    converter = music21.converter.subConverters.ConverterHumdrum()
    converter.parseData(text)
    return converter.stream


def lower_staff(score: music21.stream.Score) -> music21.stream.Part:
    return next(part for part in score.parts if part.id.endswith("_0"))


class Kern(unittest.TestCase):
    def setUp(self):
        self.kern = kern.read(SPLIT_FIRST)

    def test_opening_signatures_after_a_split(self):
        self.assertEqual(self.kern.opening, {(CLEF, 2): "F4", (CLEF, 1): "G2", (KEY, 2): -2, (KEY, 1): -2,
                                             (METER, 2): "2/4", (METER, 1): "2/4"})

    def test_a_change_at_its_beat(self):
        self.assertEqual(self.kern.changes, [Change(Fraction(3), 2, CLEF, "G2")])
        self.assertEqual(self.kern.measure_at(Fraction(3)), 1)

    def test_measures(self):
        self.assertEqual(self.kern.measure_count, 2)

    def test_pitches_by_measure_and_staff(self):
        self.assertEqual(self.kern.pitches[(1, 2, "G", 0, 4)], 1)
        self.assertEqual(self.kern.pitches[(1, 2, "E", -1, 3)], 1)

    def test_a_signature_with_nothing_after_it_is_no_change(self):
        courtesy = SPLIT_FIRST.replace("4g\t4f\n=\t=", "4g\t4f\n*clefF4\t*\n*M3/4\t*M3/4\n=\t=")
        self.assertEqual(kern.read(courtesy).changes, self.kern.changes)

    def test_a_restated_signature_is_no_change(self):
        restated = SPLIT_FIRST.replace("4E-\t4e-\n", "*k[b-e-]\t*k[b-e-]\n4E-\t4e-\n")
        self.assertEqual(kern.read(restated).changes, self.kern.changes)

    def test_key_and_meter_changes(self):
        changed = SPLIT_FIRST.replace("*v\t*v\t*\n", "*v\t*v\t*\n*k[f#]\t*k[f#]\n*M3/4\t*M3/4\n")
        self.assertIn(Change(Fraction(2), 1, KEY, 1), kern.read(changed).changes)
        self.assertIn(Change(Fraction(2), 2, METER, "3/4"), kern.read(changed).changes)

    def test_grace_notes_take_no_time(self):
        graced = SPLIT_FIRST.replace("4E-\t4e-\n", "8Eq\t.\n4E-\t4e-\n")
        self.assertEqual(kern.read(graced).changes, self.kern.changes)

    def test_grace_notes_counted_apart(self):
        graced = kern.read(SPLIT_FIRST.replace("4E-\t4e-\n", "8Eq\t.\n4E-\t4e-\n"))
        self.assertEqual(graced.grace_notes, 1)
        self.assertEqual(graced.pitches, self.kern.pitches)

    def test_notes_after_a_merger_of_the_staves(self):
        merged = SPLIT_FIRST.replace("=\t=\n*-\t*-", "*v\t*v\n4C 4c\n=\n*-")
        self.assertTrue(kern.read(merged).cut_at_merger)
        self.assertFalse(self.kern.cut_at_merger)

    def test_nothing_after_a_merger_of_the_staves(self):
        merged = SPLIT_FIRST.replace("=\t=\n*-\t*-", "=\t=\n*v\t*v\n*-")
        self.assertFalse(kern.read(merged).cut_at_merger)

    def test_ties_closed_within_the_system(self):
        tied = SPLIT_FIRST.replace("4D\t4DD\t4d\n", "4D\t4DD\t4d[\n").replace("4g\t4f\n", "4g\t4d]\n")
        self.assertEqual(kern.read(tied).ties_at_edges, 0)

    def test_a_tie_into_the_next_system(self):
        self.assertEqual(kern.read(SPLIT_FIRST.replace("4g\t4f\n", "4g\t4f[\n")).ties_at_edges, 1)

    def test_a_tie_from_the_system_before(self):
        self.assertEqual(kern.read(SPLIT_FIRST.replace("4C\t4CC\t4c\n", "4C\t4CC\t4c]\n")).ties_at_edges, 1)

    def test_a_tie_middle_open_at_both_ends(self):
        """'_' ends one tie and starts the next: alone, it runs in from before and on past the end."""
        self.assertEqual(kern.read(SPLIT_FIRST.replace("4C\t4CC\t4c\n", "4C\t4CC\t4c_\n")).ties_at_edges, 2)

    def test_a_tie_in_a_chord_on_the_other_staff_is_not_paired(self):
        """Ties pair within a staff: a C starting on the lower staff is not ended by one on the upper."""
        crossed = SPLIT_FIRST.replace("4D\t4DD\t4d\n", "4D[\t4DD\t4d\n").replace("4E-\t4e-\n", "4E-\t4e- 4d]\n")
        self.assertEqual(kern.read(crossed).ties_at_edges, 2)

    def test_durations(self):
        self.assertEqual(kern.duration("8.cc"), Fraction(3, 4))
        self.assertEqual(kern.duration("12a"), Fraction(1, 3))
        self.assertEqual(kern.duration("8aq"), 0)

    def test_pitch(self):
        self.assertEqual(kern.pitch("4cc#L"), ("C", 1, 5))
        self.assertEqual(kern.pitch("8BB-"), ("B", -1, 2))
        self.assertIsNone(kern.pitch("4r"))


class Mended(unittest.TestCase):
    """music21's parse as OLiMPiC's build calls it, mended."""

    def setUp(self):
        self.measures = list(lower_staff(parsed(SPLIT_FIRST)).getElementsByClass(music21.stream.Measure))

    def signatures(self, measure: int) -> list[tuple[float, str]]:
        kinds = (music21.clef.Clef, music21.key.KeySignature, music21.meter.TimeSignature)
        return [(float(e.offset), type(e).__name__) for e in self.measures[measure].getElementsByClass(kinds)]

    def test_opening_signatures_in_the_measure(self):
        self.assertEqual(sorted(self.signatures(0)), [(0.0, "BassClef"), (0.0, "KeySignature"), (0.0, "TimeSignature")])

    def test_none_left_in_the_voices(self):
        for voice in self.measures[0].voices:
            self.assertFalse(voice.getElementsByClass(music21.clef.Clef))

    def test_the_change_at_its_beat(self):
        self.assertEqual(self.signatures(1), [(1.0, "TrebleClef")])


if __name__ == "__main__":
    unittest.main()
