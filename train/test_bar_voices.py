"""Tests for seq.bar_voices. Run from train/: python -m unittest test_bar_voices

BAR_CASES holds real label bars of the kinds that broke a rule before, each with its voices per staff
from the MusicXML it was linearized from (eval/check_bar_voices.py --fixture); the hand-made bars
below name each rule."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from seq.bar_voices import has_two_voices_on_a_staff, voices_with_notes, voices_written
from seq.label_form import voices_per_staff

BAR_CASES = Path(__file__).with_name("test_bar_voices_cases.jsonl")


def bar(text: str) -> list[str]:
    return text.split()


class Rules(unittest.TestCase):
    def test_a_voice_opening_with_a_grace_note_keeps_its_staff(self):
        tokens = bar("measure grace grace:slash F5 voice:1 eighth stem:up staff:1 G5 half G5 half backup whole "
                     "G3 voice:5 whole stem:up staff:2")
        self.assertEqual(voices_with_notes(tokens), {1: 1, 2: 1})

    def test_a_clef_opening_the_bar_names_no_note_its_staff(self):
        tokens = bar("measure clef:F4 staff:2 rest voice:1 quarter staff:1 E4 half dot backup whole "
                     "C3 voice:5 whole staff:2")
        self.assertEqual(voices_with_notes(tokens), {1: 1, 2: 1})
        self.assertEqual(voices_written(tokens), {1: 1, 2: 1})

    def test_a_clef_inside_the_bar_does_not_move_the_note_before_it(self):
        tokens = bar("measure E4 voice:1 half staff:1 clef:F4 staff:2 F4 half backup whole C3 voice:5 whole staff:2")
        self.assertEqual(voices_with_notes(tokens), {1: 1, 2: 1})

    def test_a_cross_staff_note_puts_its_voice_on_both_staves(self):
        tokens = bar("measure C5 voice:1 whole staff:1 backup whole forward quarter B4 voice:6 eighth staff:1 "
                     "G2 eighth staff:2 forward half")
        self.assertEqual(voices_with_notes(tokens), {1: 2, 2: 1})
        self.assertEqual(voices_written(tokens), {1: 2})

    def test_hidden_rests_are_written_but_hold_no_notes(self):
        tokens = bar("measure C5 voice:1 whole staff:1 backup whole print-object:no rest voice:2 whole staff:1")
        self.assertEqual(voices_with_notes(tokens), {1: 1})
        self.assertEqual(voices_written(tokens), {1: 2})
        self.assertFalse(has_two_voices_on_a_staff(tokens))

    def test_a_voice_that_only_rests_holds_no_notes(self):
        tokens = bar("measure C5 voice:1 whole staff:1 backup whole rest voice:2 rest:measure staff:1")
        self.assertEqual(voices_with_notes(tokens), {1: 1})
        self.assertEqual(voices_written(tokens), {1: 2})

    def test_a_stretch_of_forwards_alone_writes_no_voice(self):
        tokens = bar("measure C5 voice:1 whole staff:1 backup whole clef:F4 staff:2 forward whole")
        self.assertEqual(voices_written(tokens), {1: 1})

    def test_the_label_forms_counts_cut_at_spelled_backups(self):
        tokens = bar("C5 voice:1 whole staff:1 backup:bar C3 voice:5 whole staff:2 backup:bar A2 voice:6 whole staff:2")
        self.assertEqual(voices_per_staff(tokens), {1: 1, 2: 2})


class RealBars(unittest.TestCase):
    def test_each_case_as_its_musicxml_has_it(self):
        cases = [json.loads(line) for line in BAR_CASES.read_text().splitlines()]
        self.assertGreater(len({c["kind"] for c in cases}), 6)
        for case in cases:
            tokens = bar(case["bar"])
            with self.subTest(kind=case["kind"], source=case["source"]):
                self.assertEqual(voices_with_notes(tokens), {int(k): v for k, v in case["notes"].items()})
                self.assertEqual(voices_written(tokens), {int(k): v for k, v in case["written"].items()})


if __name__ == "__main__":
    unittest.main()
