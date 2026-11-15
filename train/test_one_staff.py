"""Tests for seq.one_staff: a two-staff system's label and image taken apart by staff. The labels
are checked against OLiMPiC's linearizer on the same music written as two one-staff parts.
Run from train/: python -m unittest test_one_staff"""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

import numpy as np

from build_seq_dataset import linearize
from seq.one_staff import Unsplittable, staff_images, staff_labels
from seq.staff_positions import Staves


def note(pitch: str, kind: str, duration: int, voice: int, staff: int | None, stem: str = "up", chord: bool = False) -> str:
    head = "<rest/>" if pitch == "rest" else f"<pitch><step>{pitch[0]}</step><octave>{pitch[1]}</octave></pitch>"
    return (f"<note>{'<chord/>' if chord else ''}{head}<duration>{duration}</duration><voice>{voice}</voice>"
            f"<type>{kind}</type><stem>{stem}</stem>{'' if staff is None else f'<staff>{staff}</staff>'}</note>")


def bar_rest(voice: int, staff: int | None) -> str:
    return (f"<note><rest measure=\"yes\"/><duration>8</duration><voice>{voice}</voice>"
            f"{'' if staff is None else f'<staff>{staff}</staff>'}</note>")


def clef(sign: str, line: int, staff: int | None) -> str:
    number = "" if staff is None else f' number="{staff}"'
    return f"<clef{number}><sign>{sign}</sign><line>{line}</line></clef>"


def backup(duration: int) -> str:
    return f"<backup><duration>{duration}</duration></backup>"


def part(*measures: str) -> ET.Element:
    return ET.fromstring("<part>" + "".join(f'<measure number="{i}">{m}</measure>' for i, m in enumerate(measures, 1))
                         + "</part>")


HEADER = "<divisions>2</divisions><key><fifths>-1</fifths></key><time><beats>4</beats><beat-type>4</beat-type></time>"

# Bar 1: two voices on the upper staff, a whole-bar rest on the lower. Bar 2: the upper staff
# rests the bar, the lower changes clef between its notes.
TWO_STAVES = part(
    f"<attributes>{HEADER}<staves>2</staves>{clef('G', 2, 1)}{clef('F', 4, 2)}</attributes>"
    + note("C5", "half", 4, 1, 1) + note("D5", "half", 4, 1, 1) + backup(8)
    + note("A4", "whole", 8, 2, 1, "down") + backup(8) + bar_rest(5, 2),
    bar_rest(1, 1) + backup(8) + note("C3", "quarter", 2, 5, 2) + note("E3", "quarter", 2, 5, 2, chord=True)
    + note("D3", "quarter", 2, 5, 2) + f"<attributes>{clef('G', 2, 2)}</attributes>" + note("G4", "half", 4, 5, 2, "down"))
UPPER_ALONE = part(
    f"<attributes>{HEADER}{clef('G', 2, None)}</attributes>" + note("C5", "half", 4, 1, None)
    + note("D5", "half", 4, 1, None) + backup(8) + note("A4", "whole", 8, 2, None, "down"),
    bar_rest(1, None))
LOWER_ALONE = part(
    f"<attributes>{HEADER}{clef('F', 4, None)}</attributes>" + bar_rest(5, None),
    note("C3", "quarter", 2, 5, None) + note("E3", "quarter", 2, 5, None, chord=True) + note("D3", "quarter", 2, 5, None)
    + f"<attributes>{clef('G', 2, None)}</attributes>" + note("G4", "half", 4, 5, None, "down"))


class Labels(unittest.TestCase):
    def test_each_staff_as_the_linearizer_writes_it_alone(self):
        self.assertEqual(staff_labels(linearize(TWO_STAVES)), (linearize(UPPER_ALONE), linearize(LOWER_ALONE)))

    def test_a_voice_crossing_between_the_staves_is_left_out(self):
        crossing = part(f"<attributes>{HEADER}<staves>2</staves>{clef('G', 2, 1)}{clef('F', 4, 2)}</attributes>"
                        + note("C4", "half", 4, 1, 1) + note("G3", "half", 4, 1, 2) + backup(8) + bar_rest(5, 2))
        with self.assertRaisesRegex(Unsplittable, "both staves"):
            staff_labels(linearize(crossing))

    def test_backups_recounted_over_a_dotted_triplet(self):
        # A dotted tuplet note's ratio comes before its dot; a run of 3 dotted triplet quarters
        # (3/2 * 2/3 each) fills 3 quarters, and the lower staff's voice starts at the bar's start.
        lmx = ("measure clef:G2 staff:1 clef:F4 staff:2 C5 voice:1 quarter 3in2 dot stem:up staff:1 C5 quarter 3in2 dot "
               "C5 quarter 3in2 dot backup half backup quarter C3 voice:5 half dot stem:up staff:2")
        self.assertEqual(staff_labels(lmx)[1], "measure clef:F4 C3 voice:5 half dot stem:up")

    def test_barlines_go_to_both_staves(self):
        lmx = ("measure repeat:start clef:G2 staff:1 clef:F4 staff:2 C5 voice:1 whole stem:up staff:1 backup whole "
               "C3 voice:5 whole stem:up staff:2 barline:final")
        self.assertEqual(staff_labels(lmx), ("measure repeat:start clef:G2 C5 voice:1 whole stem:up barline:final",
                                             "measure repeat:start clef:F4 C3 voice:5 whole stem:up barline:final"))

    def test_a_whole_bar_rest_backed_up_less_than_the_bar_is_left_out(self):
        lmx = ("measure clef:G2 staff:1 clef:F4 staff:2 rest voice:1 rest:measure staff:1 backup half "
               "C3 voice:5 whole stem:up staff:2")
        with self.assertRaisesRegex(Unsplittable, "whole-bar rest"):
            staff_labels(lmx)

    def test_an_ottava_sign_between_the_staves_is_left_out(self):
        lmx = ("measure clef:G2 staff:1 clef:F4 staff:2 C5 ottava:8vb voice:1 whole stem:up staff:1 backup whole "
               "C3 voice:5 whole stem:up staff:2")
        with self.assertRaisesRegex(Unsplittable, "ottava"):
            staff_labels(lmx)


INTERLINE = 10


def system() -> tuple[np.ndarray, Staves]:
    """Staves centred on rows 60 and 180 (gap rows 86-154), a barline through both, a stem from the
    upper staff past the middle (to row 125), a ledger-line note above the lower staff with its
    stem into it (from row 125), and a dot alone on each side of the middle."""
    image = np.full((240, 300), 255, np.uint8)
    for centre in (60, 180):
        image[[centre + k * INTERLINE for k in range(-2, 3)], :] = 0
    image[40:201, 10] = 0  # barline
    image[60:126, 50] = 0  # upper stem
    image[125:181, 150] = 0  # lower stem, its note on ledger lines above the staff
    image[130:133, 140:160] = 0
    image[112:115, 250:253] = 0  # a dot just above the middle (row 120)
    image[122:125, 250:253] = 0  # and one just below
    return image, Staves((60.0, 180.0), INTERLINE)


class Images(unittest.TestCase):
    def setUp(self):
        (self.upper, self.upper_staves), (self.lower, self.lower_staves) = staff_images(*system())

    def test_each_crop_ends_at_its_own_ink(self):
        self.assertEqual(len(self.upper), 126)
        self.assertEqual(len(self.lower), 240 - 122)
        self.assertEqual(self.lower_staves.centres, (180 - 122,))

    def test_the_other_staffs_ink_is_painted_out(self):
        self.assertTrue((self.upper[125:, 140:160] == 255).all())
        self.assertTrue((self.upper[86:, 150] == 255).all())
        self.assertTrue((self.lower[:155 - 122, 50] == 255).all())

    def test_ink_alone_goes_to_the_staff_on_its_side(self):
        self.assertEqual(self.upper[113, 251], 0)
        self.assertTrue((self.upper[122:, 250:253] == 255).all())
        self.assertEqual(self.lower[123 - 122, 251], 0)

    def test_a_barline_stays_in_both(self):
        self.assertEqual(self.upper[110, 10], 0)
        self.assertEqual(self.lower[0, 10], 0)

    def test_staves_too_close(self):
        image, _ = system()
        with self.assertRaises(Unsplittable):
            staff_images(image, Staves((60.0, 100.0), INTERLINE))


if __name__ == "__main__":
    unittest.main()
