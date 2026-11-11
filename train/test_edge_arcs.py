"""Tests for the half arcs at system breaks (build_seq_dataset). Run from train/:
python -m unittest test_edge_arcs"""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from build_seq_dataset import notes_only, open_at_an_edge


class OpenAtAnEdgeTest(unittest.TestCase):
    def test_ties_and_slurs_closed_inside_are_not(self):
        self.assertFalse(open_at_an_edge("measure C4 half tied:start measure C4 half tied:stop"))
        self.assertFalse(open_at_an_edge("measure C4 quarter slur:start D4 quarter slur:stop"))

    def test_a_tie_or_slur_running_on_at_the_end_is(self):
        self.assertTrue(open_at_an_edge("measure C4 half tied:start"))
        self.assertTrue(open_at_an_edge("measure C4 quarter slur:start D4 quarter"))

    def test_one_coming_in_at_the_start_is(self):
        self.assertTrue(open_at_an_edge("measure C4 half tied:stop D4 half"))
        self.assertTrue(open_at_an_edge("measure C4 quarter slur:stop"))

    def test_a_tie_stops_only_on_its_own_pitch(self):
        self.assertTrue(open_at_an_edge("measure C4 half tied:start measure D4 half tied:stop"))


class NotesOnlyTest(unittest.TestCase):
    def test_signatures_and_directions_go_divisions_stay(self):
        measure = ET.fromstring(
            "<measure><attributes><divisions>2</divisions><key><fifths>1</fifths></key><time><beats>3</beats>"
            "<beat-type>4</beat-type></time><clef><sign>G</sign></clef></attributes>"
            "<direction><direction-type><octave-shift type='down'/></direction-type></direction>"
            "<note><pitch><step>C</step><octave>4</octave></pitch><duration>2</duration></note></measure>")
        kept = notes_only(measure)
        self.assertEqual([c.tag for c in kept], ["attributes", "note"])
        self.assertEqual([c.tag for c in kept.find("attributes")], ["divisions"])
        self.assertIsNotNone(measure.find("direction"))  # the original is left alone


if __name__ == "__main__":
    unittest.main()
