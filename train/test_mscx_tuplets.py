"""Tests for mscx_tuplets. Run from train/: python -m unittest test_mscx_tuplets"""
from __future__ import annotations

import random
import unittest
import xml.etree.ElementTree as ET

from mscx_tuplets import ROOMY_PAGE, draw_as_musicxml_says, paired, with_roomy_page
from seq.tuplet_display import PLAIN_LOOK, TupletLook
from test_tuplet_display import part, triplet

C_D_E = (60, 62, 64)  # triplet()'s pitches


def mscx_tuplet(pitches=C_D_E, ratio=(3, 2)) -> str:
    chords = "".join(f"<Chord><durationType>eighth</durationType><Note><pitch>{p}</pitch></Note></Chord>" for p in pitches)
    return (f"<Tuplet><normalNotes>{ratio[1]}</normalNotes><actualNotes>{ratio[0]}</actualNotes>"
            f"<baseNote>eighth</baseNote><Number><style>Tuplet</style><text>3</text></Number></Tuplet>{chords}<endTuplet/>")


def mscx_score(*measures: str) -> ET.Element:
    upper = "".join(f"<Measure><voice>{m}</voice></Measure>" for m in measures)
    lower = "<Measure><voice/></Measure>" * len(measures)
    return ET.fromstring(f"<Score><Staff id=\"1\">{upper}</Staff><Staff id=\"2\">{lower}</Staff></Score>")


class Pairing(unittest.TestCase):
    def test_groups_pair_in_order(self):
        pairs = paired(part(triplet() * 2), mscx_score(mscx_tuplet() * 2))
        self.assertEqual(len(pairs), 2)

    def test_another_ratio_does_not_pair(self):
        self.assertIsNone(paired(part(triplet()), mscx_score(mscx_tuplet(ratio=(6, 4)))))

    def test_other_notes_do_not_pair(self):
        """A group of another voice with the same ratio: the pitches tell them apart."""
        self.assertIsNone(paired(part(triplet()), mscx_score(mscx_tuplet(pitches=(65, 67, 69)))))

    def test_an_octave_away_pairs(self):
        """Under an ottava the MusicXML gives pitches as printed, MuseScore as they sound."""
        self.assertEqual(len(paired(part(triplet()), mscx_score(mscx_tuplet(pitches=(72, 74, 76))))), 1)

    def test_a_group_musescore_dropped_does_not_pair(self):
        self.assertIsNone(paired(part(triplet() * 2), mscx_score(mscx_tuplet())))


class Drawing(unittest.TestCase):
    def drawn(self, start_attributes: dict, look: TupletLook = PLAIN_LOOK) -> ET.Element:
        (group, tuplet), = paired(part(triplet()), mscx_score(mscx_tuplet()))
        for name, value in start_attributes.items():
            group.start.set(name, value)
        draw_as_musicxml_says(group, tuplet, look, random.Random(0))
        return tuplet

    def test_hidden_number_and_bracket(self):
        tuplet = self.drawn({"show-number": "none", "bracket": "no"})
        self.assertEqual((tuplet.findtext("numberType"), tuplet.findtext("bracketType")), ("2", "2"))

    def test_ratio_bracket_and_side(self):
        tuplet = self.drawn({"show-number": "both", "bracket": "yes", "placement": "below"})
        self.assertEqual([tuplet.findtext(t) for t in ("numberType", "bracketType", "direction")], ["1", "1", "down"])

    def test_far_number_moves_away_from_the_notes(self):
        tuplet = self.drawn({"placement": "above"}, TupletLook(far=1.0))
        self.assertLess(float(tuplet.find("Number/offset").get("y")), 0)

    def test_nothing_set_leaves_the_tuplet_alone(self):
        tuplet = self.drawn({})
        self.assertEqual([c.tag for c in tuplet], ["normalNotes", "actualNotes", "baseNote", "Number"])


class Page(unittest.TestCase):
    def test_a_far_look_gets_a_page_with_room_above_and_below(self):
        score = mscx_score(mscx_tuplet())
        with_roomy_page(score)
        self.assertEqual({tag: score.findtext(f"Style/{tag}") for tag in ROOMY_PAGE}, ROOMY_PAGE)


if __name__ == "__main__":
    unittest.main()
