import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from build_smashcima_crops import (check_manifest, drawable_system, has_hidden_note,
                                   has_measure_drawing_nothing, label_drawn,
                                   note_columns)


def part(measures: str) -> ET.Element:
    return ET.fromstring(f"<part>{measures}</part>")


class DrawableSystemTest(unittest.TestCase):
    def test_grace_notes_are_dropped(self):
        p = part("<measure><note><grace/><pitch><step>C</step><octave>5</octave></pitch><type>eighth</type></note>"
                 "<note><pitch><step>D</step><octave>5</octave></pitch><duration>4</duration><type>quarter</type></note></measure>")
        drawable_system(p)
        self.assertEqual([n.findtext("pitch/step") for n in p.iter("note")], ["D"])

    def test_undrawn_marks_go_and_drawn_articulations_stay(self):
        p = part("<measure><note><notations><fermata/><arpeggiate/><ornaments><trill-mark/></ornaments>"
                 "<articulations><staccato/><strong-accent/><tenuto/></articulations><slur type='start'/></notations></note></measure>")
        drawable_system(p)
        notations = p.find("measure/note/notations")
        self.assertEqual([c.tag for c in notations], ["articulations", "slur"])
        self.assertEqual([c.tag for c in notations.find("articulations")], ["staccato", "tenuto"])

    def test_only_a_triplet_keeps_its_number(self):
        p = part("<measure>"
                 "<note><time-modification><actual-notes>3</actual-notes></time-modification><notations><tuplet type='start'/></notations></note>"
                 "<note><time-modification><actual-notes>6</actual-notes></time-modification><notations><tuplet type='start'/></notations></note>"
                 "</measure>")
        drawable_system(p)
        self.assertEqual([t.get("show-number") for t in p.iter("tuplet")], [None, "none"])

    def test_ottava_signs_are_dropped(self):
        p = part("<measure><direction><direction-type><octave-shift type='down' size='8'/></direction-type></direction></measure>")
        drawable_system(p)
        self.assertIsNone(p.find(".//octave-shift"))


class LabelDrawnTest(unittest.TestCase):
    def test_a_label_with_a_32nd_note_is_not_drawn(self):
        self.assertFalse(label_drawn("measure " + "C4 quarter " * 5 + "C4 32nd"))

    def test_a_label_of_drawn_tokens_is(self):
        self.assertTrue(label_drawn("measure key:fifths:2 clef:G2 " + "C4 quarter tied:start " * 4))


class NoteColumnsTest(unittest.TestCase):
    def test_voices_share_columns_at_the_same_onset(self):
        p = part("<measure>"
                 "<note><duration>2</duration></note><note><chord/><duration>2</duration></note><note><duration>2</duration></note>"
                 "<backup><duration>4</duration></backup>"
                 "<note><duration>1</duration></note><note><duration>3</duration></note>"
                 "</measure>")
        self.assertEqual(note_columns(p), 3)  # onsets 0, 1 and 2


if __name__ == "__main__":
    unittest.main()


class HiddenNotesTest(unittest.TestCase):
    def test_a_measure_of_only_forwards_draws_nothing(self):
        self.assertTrue(has_measure_drawing_nothing(part(
            "<measure><note><rest/><duration>4</duration></note></measure>"
            "<measure><forward><duration>4</duration></forward></measure>")))

    def test_a_measure_of_only_hidden_rests_draws_nothing(self):
        self.assertTrue(has_measure_drawing_nothing(part(
            "<measure><note print-object='no'><rest/><duration>4</duration></note></measure>")))

    def test_a_measure_with_a_visible_rest_draws_it(self):
        self.assertFalse(has_measure_drawing_nothing(part(
            "<measure><note print-object='no'><rest/><duration>2</duration></note>"
            "<note><rest/><duration>2</duration></note></measure>")))

    def test_a_hidden_rest_is_drawable_and_a_hidden_pitch_is_not(self):
        self.assertFalse(has_hidden_note(part("<measure><note print-object='no'><rest/></note></measure>")))
        self.assertTrue(has_hidden_note(part(
            "<measure><note print-object='no'><pitch><step>C</step><octave>4</octave></pitch></note></measure>")))


class ManifestTest(unittest.TestCase):
    def test_a_restart_with_the_same_code_continues(self):
        with tempfile.TemporaryDirectory() as out:
            check_manifest(Path(out), {"smashcima": "a"})
            check_manifest(Path(out), {"smashcima": "a"})

    def test_a_restart_with_other_code_stops(self):
        with tempfile.TemporaryDirectory() as out:
            check_manifest(Path(out), {"smashcima": "a"})
            with self.assertRaises(SystemExit):
                check_manifest(Path(out), {"smashcima": "b"})
