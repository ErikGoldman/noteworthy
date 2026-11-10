import unittest
import xml.etree.ElementTree as ET

from build_smashcima_crops import drawable_system, label_drawn, note_columns


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
