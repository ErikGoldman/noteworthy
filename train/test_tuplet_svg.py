"""Tests for seq.tuplet_svg. Run from train/: python -m unittest test_tuplet_svg"""
from __future__ import annotations

import random
import unittest

from lxml import etree

from omr.render import SVG_NS
from seq.tuplet_display import TupletLook
from seq.tuplet_svg import restyle_tuplets

INTERLINE = 180.0
# Verovio's layout: a page of 2970 units with the music inside a 500-unit margin; a triplet's number
# drawn just above the top staff (row 300 of the margin group), its notes below it.
PAGE = f"""<svg xmlns="{SVG_NS}" xmlns:xlink="http://www.w3.org/1999/xlink" width="3613px" height="297px">
<svg class="definition-scale" viewBox="0 0 36130 2970"><g class="page-margin" transform="translate(500, 500)">
<g class="tuplet"><g class="note"><g class="notehead"><use xlink:href="#E0A4" transform="translate(100, 900) scale(0.72, 0.72)"/></g></g>
<g class="tupletNum"><use xlink:href="#E883" transform="translate(150, 300) scale(0.72, 0.72)"/></g></g>
</g></svg></svg>"""


def view_box(svg: bytes) -> list[float]:
    return [float(v) for v in etree.fromstring(svg).find(f"{{{SVG_NS}}}svg").get("viewBox").split()]


class FarNumbers(unittest.TestCase):
    def test_a_number_moved_off_the_page_grows_the_page_to_hold_it(self):
        svg = restyle_tuplets(PAGE.encode(), TupletLook(far=1.0), INTERLINE, random.Random(0))
        x, top, width, height = view_box(svg)
        number = etree.fromstring(svg).find(f".//{{{SVG_NS}}}g[@class='tupletNum']")
        moved_by = float(number.get("transform").split(",")[1].split(")")[0])
        self.assertLess(moved_by, -1.5 * INTERLINE)  # moved up, away from its notes
        self.assertLessEqual(top, 500 + 300 + moved_by - 2 * INTERLINE + 1e-6)  # its row, in the page, and room above
        self.assertGreaterEqual(top + height, 2970)

    def test_a_look_with_nothing_drawn_leaves_the_svg_as_it_was(self):
        self.assertEqual(restyle_tuplets(PAGE.encode(), TupletLook(), INTERLINE, random.Random(0)), PAGE.encode())


if __name__ == "__main__":
    unittest.main()
