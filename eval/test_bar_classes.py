"""Tests for bar_classes and cause_bars. Run from eval/: python -m unittest test_bar_classes"""
from __future__ import annotations

import unittest

from bar_classes import CHORDS, NONE, TUPLET_HIDDEN, TUPLET_SHOWN, TWO_VOICES, bars_drawn, classes_of, right_bars
from cause_bars import MIN_BARS, by_piece, summary

PLAIN = "measure C4 voice:1 quarter stem:up staff:1 D4 quarter E4 quarter F4 quarter"
CHORD = "measure C4 voice:1 quarter stem:up staff:1 chord E4 quarter chord G4 quarter D4 quarter E4 half"
TWO = ("measure E5 voice:1 half stem:up staff:1 E5 half backup whole "
       "C4 voice:2 whole stem:down staff:1")
TRIPLETS = ("measure C4 voice:1 eighth 3in2 stem:up staff:1 beam:begin tuplet:start D4 eighth 3in2 "
            "E4 eighth 3in2 beam:end tuplet:stop F4 quarter G4 half")


class Classes(unittest.TestCase):
    def classes(self, lmx: str) -> set[str]:
        return classes_of(bars_drawn(lmx)[0])

    def test_each_class_from_the_label(self):
        self.assertEqual(self.classes(PLAIN), {NONE})
        self.assertEqual(self.classes(CHORD), {CHORDS})
        self.assertEqual(self.classes(TWO), {TWO_VOICES})
        self.assertEqual(self.classes(TRIPLETS), {TUPLET_SHOWN})
        hidden = TRIPLETS.replace("tuplet:start", "tuplet:start tuplet:hidden")
        self.assertEqual(self.classes(hidden), {TUPLET_HIDDEN})

    def test_a_voice_opening_with_a_grace_note_keeps_its_staff(self):
        """The upper staff's melody, then one voice below: not two voices on the lower staff."""
        bar = ("measure grace grace:slash F5 voice:1 eighth stem:up staff:1 G5 half G5 half backup whole "
               "G3 voice:5 whole stem:up staff:2")
        self.assertEqual(self.classes(bar), {NONE})

    def test_a_bar_is_right_by_its_rhythm(self):
        label = f"{PLAIN} {CHORD}"
        self.assertEqual([right for _, right in right_bars(label, label)], [True, True])
        a_chord_tone_missing = CHORD.replace(" chord G4 quarter", "")
        self.assertEqual([right for _, right in right_bars(label, f"{PLAIN} {a_chord_tone_missing}")], [True, False])
        another_pitch = CHORD.replace("G4", "A4")
        self.assertEqual([right for _, right in right_bars(label, f"{PLAIN} {another_pitch}")], [True, True])


class Pieces(unittest.TestCase):
    def test_a_piece_with_too_few_bars_of_a_class_is_left_out(self):
        rows = [{"name": "samples/1/s0", "label": " ".join([CHORD] * MIN_BARS), "reading": " ".join([CHORD] * MIN_BARS)},
                {"name": "samples/2/s0", "label": CHORD, "reading": CHORD}]
        share, pieces = summary(by_piece(rows))[CHORDS]
        self.assertEqual(pieces, 1)
        self.assertIsNone(share)  # under MIN_PIECES pieces: too few to decide


if __name__ == "__main__":
    unittest.main()
