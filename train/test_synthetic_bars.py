"""Tests for synthetic_bars. Run from train/: python -m unittest test_synthetic_bars"""
from __future__ import annotations

import random
import unittest
import xml.etree.ElementTree as ET
from collections import Counter
from fractions import Fraction

from build_seq_dataset import break_into_systems, linearize, piano_part, split_part_to_systems
from synthetic_bars import DENSE_CHORDS, DIVISIONS, PLAIN, TUPLETS, TWO_VOICES, synthetic_score

SCORES = 200


def voices_on_a_staff(score: ET.Element) -> int:
    staves = Counter()
    for staff, voice in {(n.findtext("staff"), n.findtext("voice")) for n in score.iter("note")}:
        staves[staff] += 1
    return max(staves.values())


def largest_chord(score: ET.Element, staff: str) -> int:
    largest = size = 0
    for note in score.iter("note"):
        if note.findtext("staff") == staff:
            size = size + 1 if note.find("chord") is not None else 1
            largest = max(largest, size)
    return largest


class SyntheticScores(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scores = [synthetic_score(k, 8) for k in range(SCORES)]

    def test_every_voice_fills_its_bar(self):
        for score, _ in self.scores:
            part = score.find("part")
            time = part.find("measure/attributes/time")
            bar = Fraction(4 * int(time.findtext("beats")), int(time.findtext("beat-type"))) * DIVISIONS
            for measure in part.findall("measure"):
                filled = Counter()
                for note in measure.findall("note"):
                    if note.find("chord") is None:
                        filled[note.findtext("voice")] += int(note.findtext("duration"))
                self.assertTrue(all(v == bar for v in filled.values()), (dict(filled), bar))

    def test_beams_and_tuplets_open_and_close(self):
        for score, _ in self.scores:
            for level in ("1", "2"):
                beams = Counter(b.text for b in score.iter("beam") if b.get("number") == level)
                self.assertEqual(beams["begin"], beams["end"])
            kinds = Counter(t.get("type") for t in score.iter("tuplet"))
            self.assertEqual(kinds["start"], kinds["stop"])

    def test_features_do_not_overlap(self):
        for score, feature in self.scores:
            has_tuplets = next(score.iter("tuplet"), None) is not None
            self.assertEqual(has_tuplets, feature == TUPLETS, feature)
            self.assertEqual(voices_on_a_staff(score) > 1, feature == TWO_VOICES, feature)
            # The right hand's chords: two notes at most but in dense-chord pieces (a left hand's
            # bass and triad is plain accompaniment).
            self.assertEqual(largest_chord(score, "1") > 2, feature == DENSE_CHORDS, feature)

    def test_each_feature_is_drawn(self):
        self.assertEqual({f for _, f in self.scores}, {PLAIN, TUPLETS, TWO_VOICES, DENSE_CHORDS})

    def test_systems_linearize(self):
        for score, _ in self.scores[:60]:
            part = piano_part(score)
            break_into_systems(part, random.Random(0))
            for page in split_part_to_systems(part):
                for system in page.systems:
                    self.assertIn("measure", linearize(system.part))

    def test_a_score_depends_on_its_index_alone(self):
        self.assertEqual(ET.tostring(synthetic_score(7, 8)[0]), ET.tostring(self.scores[7][0]))
        self.assertNotEqual(ET.tostring(synthetic_score(8, 8)[0]), ET.tostring(self.scores[7][0]))


if __name__ == "__main__":
    unittest.main()
