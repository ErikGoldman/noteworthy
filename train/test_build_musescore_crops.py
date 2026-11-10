"""Tests for build_musescore_crops: finding the system on MuseScore's page, the spare measure, and
the style variants. Run from train/ (NW_DATA set): python -m unittest test_build_musescore_crops"""
from __future__ import annotations

import random
import unittest
import xml.etree.ElementTree as ET

import numpy as np
from lxml import etree

from build_seq_dataset import without_octave_clefs
from build_musescore_crops import (STYLE_VARIANTS, ExpectedDrawing, SystemOnPage, barlines_drawn, batches,
                                   closes_with_own_barline, engraving_style, ink_crop, staff_lines, staves, style_variant,
                                   show_every_tuplet_number, system_on_page, with_label_ottavas, with_spare_measure)

INTERLINE = 25.0
UPPER_TOP, LOWER_TOP = 600.0, 860.0


def staff_svg(top: float, left: float = 300.0, right: float = 2800.0) -> str:
    return "".join(f'<polyline class="StaffLines" stroke-width="2.75" points="{left},{top + k * INTERLINE} {right},{top + k * INTERLINE}"/>'
                   for k in range(5))


def stroke(x: float, top: float, bottom: float, width: float = 4.5) -> str:
    return f'<polyline class="BarLine" stroke-width="{width}" points="{x},{top} {x},{bottom}"/>'


def barline_svg(x: float, final: bool = False, upper_only: bool = False) -> str:
    """A barline as MuseScore draws it: each staff's strokes from its top line, the upper staff's
    down to the lower's. A final bar is a thin stroke and a thick one 17 units right."""
    def strokes(top: float, bottom: float, with_final: bool) -> str:
        return stroke(x, top, bottom) + (stroke(x + 17, top, bottom, 13.6) if with_final else "")
    return (strokes(UPPER_TOP, LOWER_TOP, final) + strokes(LOWER_TOP, LOWER_TOP + 4 * INTERLINE, final and not upper_only))


def page_svg(*parts: str) -> bytes:
    return f'<svg xmlns="http://www.w3.org/2000/svg">{"".join(parts)}</svg>'.encode()


def system_svg(*barlines: str) -> bytes:
    return page_svg(staff_svg(UPPER_TOP), staff_svg(LOWER_TOP), *barlines)


class Staves(unittest.TestCase):
    def test_two_staves_of_five_lines(self):
        found = staves(staff_lines(system_svg()))
        self.assertEqual([s.top for s in found], [UPPER_TOP, LOWER_TOP])
        self.assertEqual(found[0].interline, INTERLINE)

    def test_indented_first_system_is_its_own_extent(self):
        svg = page_svg(staff_svg(100, left=400), staff_svg(360, left=400), staff_svg(UPPER_TOP), staff_svg(LOWER_TOP))
        self.assertEqual(len(staves(staff_lines(svg))), 4)

    def test_lines_short_of_a_staff(self):
        lines = staff_lines(system_svg())[:7]
        with self.assertRaises(ValueError):
            staves(lines)


TWO_MEASURES_THEN_SPARE = ExpectedDrawing(barlines=3, spare_measure=True)


class SystemOnMuseScorePage(unittest.TestCase):
    def test_crop_ends_at_the_barline_before_the_spare_measure(self):
        svg = system_svg(barline_svg(300), barline_svg(1200), barline_svg(2000), barline_svg(2780, final=True))
        self.assertEqual(system_on_page(svg, TWO_MEASURES_THEN_SPARE), SystemOnPage(INTERLINE, 2002.25))

    def test_crop_ends_at_the_systems_own_final_bar(self):
        svg = system_svg(barline_svg(300), barline_svg(1200), barline_svg(2000, final=True))
        self.assertEqual(system_on_page(svg, ExpectedDrawing(3, spare_measure=False)).right, 2017 + 6.8)

    def test_a_measure_more_than_the_label_is_refused(self):
        svg = system_svg(barline_svg(300), barline_svg(1200), barline_svg(2000), barline_svg(2400), barline_svg(2780, final=True))
        with self.assertRaisesRegex(ValueError, "do not match"):
            system_on_page(svg, TWO_MEASURES_THEN_SPARE)

    def test_a_final_bar_on_the_upper_staff_only_is_refused(self):
        svg = system_svg(barline_svg(300), barline_svg(1200, final=True, upper_only=True), barline_svg(2000),
                         barline_svg(2780, final=True))
        with self.assertRaisesRegex(ValueError, "differs"):
            system_on_page(svg, TWO_MEASURES_THEN_SPARE)

    def test_the_spare_measures_own_bar_may_differ(self):
        svg = system_svg(barline_svg(300), barline_svg(1200), barline_svg(2000), barline_svg(2780, final=True, upper_only=True))
        self.assertEqual(system_on_page(svg, TWO_MEASURES_THEN_SPARE).right, 2002.25)

    def test_a_system_broken_in_two_is_refused(self):
        svg = page_svg(staff_svg(100), staff_svg(360), staff_svg(UPPER_TOP), staff_svg(LOWER_TOP))
        with self.assertRaisesRegex(ValueError, "4 staves"):
            system_on_page(svg, TWO_MEASURES_THEN_SPARE)

    def test_unreadable_svg(self):
        with self.assertRaisesRegex(ValueError, "unreadable"):
            system_on_page(b"<svg", TWO_MEASURES_THEN_SPARE)


class InkCrop(unittest.TestCase):
    @staticmethod
    def page(ink_rows: slice, ink_cols: slice) -> np.ndarray:
        rgba = np.zeros((300, 400, 4), np.uint8)
        rgba[ink_rows, ink_cols, 3] = 255
        return rgba

    def test_ink_right_of_the_closing_barline_is_cut_off(self):
        crop = ink_crop(self.page(slice(100, 200), slice(50, 350)), SystemOnPage(10.0, 149.0), random.Random(0))
        self.assertEqual(crop.min(), 0)
        self.assertEqual(crop.max(), 255)
        inked_columns = np.nonzero((crop < 128).any(axis=0))[0]
        self.assertLess(len(inked_columns), 300 * 2.6)  # 100 ink columns at most 2.6 times enlarged

    def test_ink_touching_the_page_edge_is_refused(self):
        with self.assertRaisesRegex(ValueError, "page edge"):
            ink_crop(self.page(slice(0, 200), slice(50, 350)), SystemOnPage(10.0, 349.0), random.Random(0))

    def test_blank_page_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no ink"):
            ink_crop(self.page(slice(0, 0), slice(0, 0)), SystemOnPage(10.0, 349.0), random.Random(0))


class SpareMeasure(unittest.TestCase):
    def part(self, last_measure: str) -> ET.Element:
        return ET.fromstring(f"<part><measure><note/></measure>{last_measure}</part>")

    def test_plain_ending_takes_a_spare_measure(self):
        part = self.part("<measure><note/></measure>")
        self.assertFalse(closes_with_own_barline(part))
        with_spare_measure(part)
        self.assertEqual(len(part.findall("measure")), 3)
        self.assertEqual(len(part.findall("measure")[-1]), 0)

    def test_final_bar_and_repeat_close_the_system_themselves(self):
        for style, repeat in (("light-heavy", ""), ("light-light", ""), ("light-heavy", '<repeat direction="backward"/>')):
            part = self.part(f'<measure><note/><barline location="right"><bar-style>{style}</bar-style>{repeat}</barline></measure>')
            self.assertTrue(closes_with_own_barline(part))

    def test_an_opening_repeat_does_not_close(self):
        part = self.part(OPENING_REPEAT)
        self.assertFalse(closes_with_own_barline(part))


OPENING_REPEAT = '<measure><barline location="left"><bar-style>heavy-light</bar-style><repeat direction="forward"/></barline><note/></measure>'


def note(octave: int) -> str:
    return (f"<note><pitch><step>C</step><octave>{octave}</octave></pitch><duration>1</duration>"
            "<voice>1</voice><type>quarter</type><staff>1</staff></note>")


class Ottavas(unittest.TestCase):
    def test_a_sign_running_into_the_next_system_is_closed_within_this_one(self):
        start = ('<direction placement="above"><direction-type><octave-shift type="down" size="8"/>'
                 '</direction-type><staff>1</staff></direction>')
        part = ET.fromstring(f"<part><measure>{note(5)}{start}{note(6)}</measure><measure>{note(6)}</measure></part>")
        self.assertTrue(with_label_ottavas(part, "measure C5 quarter C5 ottava:8va quarter measure C5 ottava:8va quarter"))
        self.assertEqual([s.get("type") for s in part.iter("octave-shift")], ["down", "stop"])
        last = part.findall("measure")[-1]
        self.assertEqual([e.tag for e in last], ["note", "direction"])

    def test_signs_follow_the_label_not_the_source(self):
        stop = ('<direction><direction-type><octave-shift type="stop" size="8"/></direction-type>'
                '<direction-type><words>dolce</words></direction-type><staff>1</staff></direction>')
        part = ET.fromstring(f"<part><measure>{stop}{note(5)}</measure></part>")
        self.assertTrue(with_label_ottavas(part, "measure C5 quarter"))
        self.assertIsNone(next(part.iter("octave-shift"), None))
        self.assertIsNotNone(next(part.iter("words"), None))

    def test_a_label_the_notes_do_not_match_is_refused(self):
        part = ET.fromstring(f"<part><measure>{note(6)}</measure></part>")
        self.assertFalse(with_label_ottavas(part, "measure C5 ottava:8va quarter C5 ottava:8va quarter"))


class OctaveClefs(unittest.TestCase):
    def test_an_octave_clef_is_drawn_plain_and_its_notes_keep_their_pitch(self):
        part = ET.fromstring("<part><measure><attributes><clef><sign>G</sign><line>2</line>"
                             f"<clef-octave-change>-1</clef-octave-change></clef></attributes>{note(4)}</measure></part>")
        without_octave_clefs(part)
        self.assertIsNone(part.find(".//clef-octave-change"))
        self.assertEqual(part.find(".//clef/sign").text, "G")
        self.assertEqual(part.findtext(".//octave"), "4")


class TupletNumbers(unittest.TestCase):
    def test_a_hidden_number_is_shown_as_musescore_draws_it(self):
        part = ET.fromstring('<part><measure><note><notations><tuplet type="start" show-number="none" bracket="no"/>'
                             '</notations></note></measure></part>')
        show_every_tuplet_number(part)
        self.assertEqual(part.find(".//tuplet").attrib, {"type": "start", "bracket": "no"})


class BarlinesDrawn(unittest.TestCase):
    def test_opening_barline_and_one_per_measure(self):
        self.assertEqual(barlines_drawn(ET.fromstring("<part><measure/><measure/><measure/></part>")), 4)

    def test_a_repeat_opening_the_system_stands_apart(self):
        self.assertEqual(barlines_drawn(ET.fromstring(f"<part>{OPENING_REPEAT}<measure/></part>")), 4)

    def test_a_repeat_further_on_joins_the_barline_before_it(self):
        self.assertEqual(barlines_drawn(ET.fromstring(f"<part><measure/>{OPENING_REPEAT}</part>")), 3)


class Styles(unittest.TestCase):
    def test_a_scores_variant_does_not_depend_on_its_batch(self):
        scores = [(i, f"score{i}") for i in range(300)]
        grouped = batches(scores, 4)
        self.assertEqual(sorted(s for batch in grouped for s in batch.scores), scores)
        for batch in grouped:
            self.assertLessEqual(len(batch.scores), 4)
            self.assertTrue(all(style_variant(i) == batch.variant for i, _ in batch.scores))

    def test_style_file_names_a_music_font_and_a_wide_page(self):
        style = etree.fromstring(engraving_style(3).encode()).find("Style")
        self.assertIn(style.findtext("musicalSymbolFont"), {"Emmentaler", "Bravura", "Leland", "Gonville", "MuseJazz", "Petaluma"})
        self.assertEqual(style.findtext("lastSystemFillLimit"), "1")
        self.assertEqual(engraving_style(3), engraving_style(3))
        self.assertEqual(len({engraving_style(v) for v in range(STYLE_VARIANTS)}), STYLE_VARIANTS)


if __name__ == "__main__":
    unittest.main()
