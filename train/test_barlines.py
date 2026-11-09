"""Tests for seq.barlines: barline signs as LMX tokens and back. Run from train/: python -m unittest test_barlines"""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from seq.barlines import (add_barlines, barline_tokens_of, kern_measure_barlines, with_barline_tokens,
                          with_kern_barline_tokens, without_barlines)


def measure(*barlines: str) -> ET.Element:
    return ET.fromstring("<measure>" + "".join(barlines) + "<note/></measure>")


def barline(location: str, style: str, repeat: str | None = None) -> str:
    repeat_xml = f'<repeat direction="{repeat}"/>' if repeat else ""
    return f'<barline location="{location}"><bar-style>{style}</bar-style>{repeat_xml}</barline>'


class TokensFromMusicXML(unittest.TestCase):
    def test_double_bar(self):
        self.assertEqual(barline_tokens_of(measure(barline("right", "light-light"))), ([], ["barline:double"]))

    def test_final_bar(self):
        self.assertEqual(barline_tokens_of(measure(barline("right", "light-heavy"))), ([], ["barline:final"]))

    def test_backward_repeat_is_a_repeat_not_a_final_bar(self):
        self.assertEqual(barline_tokens_of(measure(barline("right", "light-heavy", "backward"))), ([], ["repeat:end"]))

    def test_both_repeats(self):
        bar = measure(barline("left", "heavy-light", "forward"), barline("right", "light-heavy", "backward"))
        self.assertEqual(barline_tokens_of(bar), (["repeat:start"], ["repeat:end"]))

    def test_plain_bar_takes_none(self):
        self.assertEqual(barline_tokens_of(measure(barline("right", "regular"))), ([], []))
        self.assertEqual(barline_tokens_of(measure()), ([], []))

    def test_tokens_go_after_measure_and_at_the_end(self):
        self.assertEqual(with_barline_tokens(["measure", "C4", "quarter"], ["repeat:start"], ["barline:final"]),
                         ["measure", "repeat:start", "C4", "quarter", "barline:final"])


class BackToMusicXML(unittest.TestCase):
    def test_tokens_come_out_with_their_measures(self):
        plain, barlines = without_barlines("measure C4 whole barline:double measure repeat:start D4 whole barline:final")
        self.assertEqual(plain, "measure C4 whole measure D4 whole")
        self.assertEqual(barlines, [(0, "barline:double"), (1, "repeat:start"), (1, "barline:final")])

    def test_drawn_back_as_the_signs_they_came_from(self):
        part = ET.fromstring("<part><measure><note/></measure><measure><note/></measure></part>")
        add_barlines(part, [(0, "barline:double"), (1, "repeat:start"), (1, "barline:final")])
        first, second = part.findall("measure")
        self.assertEqual(barline_tokens_of(first), ([], ["barline:double"]))
        self.assertEqual(barline_tokens_of(second), (["repeat:start"], ["barline:final"]))
        self.assertEqual(second[0].tag, "barline")
        self.assertEqual(second[-1].tag, "barline")

    def test_a_measure_the_delinearizer_lost_takes_none(self):
        part = ET.fromstring("<part><measure><note/></measure></part>")
        add_barlines(part, [(3, "barline:final")])
        self.assertIsNone(part.find(".//barline"))


class GrandStaffKern(unittest.TestCase):
    KERN = "**kern\n*M3/4\n=-\n4c\n=\n4d\n=:|!|:\n4e\n==\n*-\n"

    def test_repeat_closes_one_measure_and_opens_the_next(self):
        self.assertEqual(kern_measure_barlines(self.KERN),
                         [([], []), ([], ["repeat:end"]), (["repeat:start"], ["barline:final"])])

    def test_tokens_go_into_their_measures(self):
        self.assertEqual(with_kern_barline_tokens("measure C4 measure D4 measure E4", self.KERN),
                         "measure C4 measure D4 repeat:end measure repeat:start E4 barline:final")

    def test_plain_barlines_leave_the_label(self):
        self.assertEqual(with_kern_barline_tokens("measure C4", "**kern\n4c\n=\n4d\n*-\n"), "measure C4")

    def test_mismatched_measures_give_no_label(self):
        self.assertIsNone(with_kern_barline_tokens("measure C4 measure D4", self.KERN))


if __name__ == "__main__":
    unittest.main()
