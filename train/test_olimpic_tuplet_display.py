import unittest
import xml.etree.ElementTree as ET

from olimpic_tuplet_display import ScoreTuplet, Unpaired, exported_measure_numbers, tuplets_in, with_score_display


def staff(*measures: str) -> ET.Element:
    return ET.fromstring(f"<Staff>{''.join(measures)}</Staff>")


SECTION_BREAK = "<LayoutBreak><subtype>section</subtype></LayoutBreak>"


class MeasureNumbers(unittest.TestCase):
    def test_a_pickup_is_zero_and_the_count_starts_after_it(self):
        self.assertEqual(exported_measure_numbers(staff("<Measure><irregular>1</irregular></Measure>",
                                                        "<Measure/>", "<Measure/>")), ["0", "1", "2"])

    def test_later_irregular_measures_are_x_numbered(self):
        self.assertEqual(exported_measure_numbers(staff("<Measure/>", "<Measure><irregular/></Measure>", "<Measure/>")),
                         ["1", "X1", "2"])

    def test_a_section_break_restarts_the_count(self):
        self.assertEqual(exported_measure_numbers(staff("<Measure/>", f"<Measure>{SECTION_BREAK}</Measure>",
                                                        f"<VBox>{SECTION_BREAK}</VBox>", "<Measure/>")),
                         ["1", "2", "1"])


class ScoreTuplets(unittest.TestCase):
    def test_invisible_or_number_type_two_hides(self):
        measure = ET.fromstring("<Measure><voice>"
                                "<Tuplet><actualNotes>3</actualNotes><normalNotes>2</normalNotes><visible>0</visible></Tuplet>"
                                "<Tuplet><actualNotes>3</actualNotes><normalNotes>2</normalNotes></Tuplet>"
                                "<Tuplet><actualNotes>3</actualNotes><normalNotes>2</normalNotes><numberType>2</numberType></Tuplet>"
                                "</voice></Measure>")
        self.assertEqual([t.hidden for t in tuplets_in(measure, "0")], [True, False, True])
        self.assertTrue(tuplets_in(measure, "2")[1].hidden)  # the score's style hides by default


def system(*measures: tuple[str, int]) -> ET.Element:
    """A part of measures (number, tuplet starts), each start a 3-in-2 group labelled printed."""
    note = ("<note><time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>"
            "<notations><tuplet type=\"start\" bracket=\"yes\"/></notations></note>")
    return ET.fromstring("<part>" + "".join(f'<measure number="{n}">{note * k}</measure>' for n, k in measures) + "</part>")


SHOWN, HIDDEN = ScoreTuplet("3in2", False), ScoreTuplet("3in2", True)


def displays(part: ET.Element) -> list[str | None]:
    return [t.get("show-number") for t in part.iter("tuplet")]


class Pairing(unittest.TestCase):
    def test_the_score_decides_each_group(self):
        part = system(("1", 2))
        self.assertEqual(with_score_display([part], {"1": [(0, [HIDDEN, SHOWN])]}), 1)
        self.assertEqual(displays(part), ["none", None])
        self.assertEqual(next(part.iter("tuplet")).get("bracket"), "no")

    def test_a_repeated_number_pairs_past_the_last_measure_paired(self):
        # a section break restarts at 1; the first section's measure 1 is not among the systems
        later = system(("2", 0), ("1", 1))
        score = {"1": [(0, [SHOWN]), (2, [HIDDEN])], "2": [(1, [])]}
        with_score_display([later], score)
        self.assertEqual(displays(later), ["none"])

    def test_a_differing_ratio_changes_nothing(self):
        part = system(("1", 1), ("2", 1))
        with self.assertRaises(Unpaired):
            with_score_display([part], {"1": [(0, [HIDDEN])], "2": [(1, [ScoreTuplet("5in4", True)])]})
        self.assertEqual(displays(part), [None, None])

    def test_a_group_the_score_lacks_is_unpaired(self):
        with self.assertRaises(Unpaired):
            with_score_display([system(("1", 1))], {"1": [(0, [])]})


if __name__ == "__main__":
    unittest.main()
