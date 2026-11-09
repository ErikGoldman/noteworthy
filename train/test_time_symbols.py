"""Tests for seq.time_symbols: C and cut time as LMX tokens and back. Run from train/: python -m unittest test_time_symbols"""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from build_seq_dataset import linearize
from seq.time_symbols import add_time_symbols, kern_time_symbols, with_kern_time_symbols, without_time_symbols


def part(time_attributes: str, beats: str = "4", beat_type: str = "4") -> ET.Element:
    return ET.fromstring(
        f'<part><measure number="1"><attributes><divisions>1</divisions><time{time_attributes}><beats>{beats}</beats>'
        f'<beat-type>{beat_type}</beat-type></time><staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef>'
        '<clef number="2"><sign>F</sign><line>4</line></clef></attributes>'
        '<note><pitch><step>C</step><octave>5</octave></pitch><duration>4</duration><voice>1</voice><type>whole</type>'
        '<staff>1</staff></note></measure></part>')


class Labels(unittest.TestCase):
    def test_common_time_follows_the_time_token(self):
        self.assertIn("time time:common beats:4 beat-type:4", linearize(part(' symbol="common"')))

    def test_cut_time(self):
        self.assertIn("time time:cut beats:2 beat-type:2", linearize(part(' symbol="cut"', "2", "2")))

    def test_numbers_take_no_token(self):
        self.assertNotIn("time:", linearize(part("")))


class Output(unittest.TestCase):
    def test_tokens_come_out_with_their_time_signature(self):
        plain, symbols = without_time_symbols("measure time beats:3 beat-type:4 measure time time:cut beats:2 beat-type:2")
        self.assertEqual(plain, "measure time beats:3 beat-type:4 measure time beats:2 beat-type:2")
        self.assertEqual(symbols, [(1, "time:cut")])

    def test_symbol_set_on_its_time_signature(self):
        p = ET.fromstring("<part><measure><attributes><time/></attributes></measure>"
                          "<measure><attributes><time/></attributes></measure></part>")
        add_time_symbols(p, [(1, "time:common"), (5, "time:cut")])
        self.assertEqual([t.get("symbol") for t in p.iter("time")], [None, "common"])


class GrandStaffKern(unittest.TestCase):
    KERN = "**kern\t**kern\n*M4/4\t*M4/4\n*met(c)\t*met(c)\n4c\t4e\n*M3/4\t*M3/4\n4d\t4f\n*-\t*-\n"

    def test_mensuration_belongs_to_the_signature_before_it(self):
        self.assertEqual(kern_time_symbols(self.KERN), ["time:common", None])

    def test_mensuration_after_data_belongs_to_no_signature(self):
        self.assertEqual(kern_time_symbols("**kern\n*M2/2\n4c\n*met(c|)\n4d\n*-\n"), [None])

    def test_tokens_added_by_order(self):
        lmx = "measure time beats:4 beat-type:4 C4 measure time beats:3 beat-type:4 D4"
        self.assertEqual(with_kern_time_symbols(lmx, self.KERN),
                         "measure time time:common beats:4 beat-type:4 C4 measure time beats:3 beat-type:4 D4")

    def test_no_symbol_leaves_the_label_whatever_the_counts(self):
        self.assertEqual(with_kern_time_symbols("measure C4", "**kern\n*M3/4\n4c\n*-\n"), "measure C4")

    def test_mismatched_counts_give_no_label(self):
        self.assertIsNone(with_kern_time_symbols("measure time beats:4 beat-type:4 C4", self.KERN))


if __name__ == "__main__":
    unittest.main()
