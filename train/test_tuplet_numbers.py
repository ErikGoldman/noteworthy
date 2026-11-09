"""Tests for seq.tuplet_numbers: hidden tuplet numbers as an LMX token and back. Run from train/: python -m unittest test_tuplet_numbers"""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from build_seq_dataset import linearize
from seq.tuplet_numbers import hide_numbers, without_hidden_numbers


def triplet_part(start_attributes: str) -> ET.Element:
    def note(step: str, mark: str) -> str:
        return (f'<note><pitch><step>{step}</step><octave>5</octave></pitch><duration>1</duration><voice>1</voice>'
                '<type>eighth</type><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes>'
                f'</time-modification><staff>1</staff>{mark}</note>')
    return ET.fromstring(
        '<part><measure number="1"><attributes><divisions>3</divisions><time><beats>1</beats><beat-type>4</beat-type></time>'
        '<staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef>'
        '</attributes>' + note("C", f'<notations><tuplet type="start"{start_attributes}/></notations>') + note("D", "")
        + note("E", '<notations><tuplet type="stop"/></notations>') + '</measure></part>')


class Labels(unittest.TestCase):
    def test_hidden_number_follows_the_start(self):
        self.assertIn("tuplet:start tuplet:hidden", linearize(triplet_part(' show-number="none" bracket="no"')))

    def test_shown_number_takes_no_token(self):
        self.assertNotIn("tuplet:hidden", linearize(triplet_part("")))


class Output(unittest.TestCase):
    def test_token_comes_out_with_its_start(self):
        plain, hidden = without_hidden_numbers("C4 eighth 3in2 tuplet:start E4 eighth 3in2 tuplet:stop "
                                               "F4 eighth 3in2 tuplet:start tuplet:hidden A4 eighth 3in2 tuplet:stop")
        self.assertNotIn("tuplet:hidden", plain)
        self.assertEqual(hidden, [1])

    def test_stray_token_is_dropped(self):
        self.assertEqual(without_hidden_numbers("C4 eighth tuplet:hidden"), ("C4 eighth", []))

    def test_number_hidden_on_its_start(self):
        p = ET.fromstring('<part><note><notations><tuplet type="start"/></notations></note>'
                          '<note><notations><tuplet type="stop"/><tuplet type="start"/></notations></note></part>')
        hide_numbers(p, [1, 4])
        self.assertEqual([t.get("show-number") for t in p.iter("tuplet") if t.get("type") == "start"], [None, "none"])


if __name__ == "__main__":
    unittest.main()
