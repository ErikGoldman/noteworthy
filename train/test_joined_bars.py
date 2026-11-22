import random
import unittest
import xml.etree.ElementTree as ET

from seq.joined_bars import events_of, joinable, joined, runs_of, with_bars_joined


def note(step: str, duration: int = 1, voice: int = 1, staff: int = 1, chord: bool = False) -> str:
    return (f"<note>{'<chord/>' if chord else ''}<pitch><step>{step}</step><octave>4</octave></pitch>"
            f"<duration>{duration}</duration><voice>{voice}</voice><staff>{staff}</staff></note>")


def part(*measures: str) -> ET.Element:
    return ET.fromstring("<part>" + "".join(f'<measure number="{k + 1}">{m}</measure>' for k, m in enumerate(measures)) + "</part>")


PIANO = note("C", 2) + note("D", 2) + "<backup><duration>4</duration></backup>" + note("E", 4, voice=5, staff=2)
SHORT_TREBLE = note("C", 2) + "<backup><duration>2</duration></backup>" + note("E", 4, voice=5, staff=2)


def onsets(measure: ET.Element) -> list[tuple[str, int]]:
    events, _, _ = events_of(measure, 0)
    return [(e.elements[-1].findtext("pitch/step"), e.onset) for e in events]


class JoinedBars(unittest.TestCase):
    def test_each_note_keeps_its_onset_from_the_joined_bars_start(self):
        measure = joined(part(PIANO, PIANO).findall("measure"))
        self.assertEqual(sorted(onsets(measure)), sorted([("C", 0), ("D", 2), ("E", 0), ("C", 4), ("D", 6), ("E", 4)]))

    def test_written_voice_by_voice_with_one_backup_between(self):
        measure = joined(part(PIANO, PIANO).findall("measure"))
        self.assertEqual([(c.tag, c.findtext("voice")) for c in measure],
                         [("note", "1")] * 4 + [("backup", None)] + [("note", "5")] * 2)

    def test_a_voice_silent_for_a_while_is_carried_by_a_forward_in_its_own_voice(self):
        measure = joined(part(SHORT_TREBLE, PIANO).findall("measure"))
        treble = [(c.tag, c.findtext("duration")) for c in measure if c.findtext("voice") == "1"]
        self.assertEqual(treble, [("note", "2"), ("forward", "2"), ("note", "2"), ("note", "2")])
        self.assertIn(("C", 4), onsets(measure))

    def test_chord_notes_share_their_notes_onset(self):
        measure = joined(part(note("C", 4) + note("E", 4, chord=True), note("D", 4)).findall("measure"))
        self.assertEqual(onsets(measure), [("C", 0), ("E", 0), ("D", 4)])

    def test_directions_travel_with_the_note_after_them(self):
        direction = "<direction><direction-type><words>dolce</words></direction-type></direction>"
        measure = joined(part(note("C", 4), direction + note("D", 4)).findall("measure"))
        self.assertEqual([c.tag for c in measure], ["note", "direction", "note"])

    def test_boundaries_with_something_written_at_them_are_kept(self):
        plain = note("C", 4)
        rest = '<note><rest measure="yes"/><duration>4</duration></note>'
        self.assertTrue(joinable(*part(plain, plain).findall("measure")))
        self.assertFalse(joinable(*part(plain + '<barline location="right"><bar-style>light-heavy</bar-style></barline>', plain).findall("measure")))
        self.assertFalse(joinable(*part(plain, "<attributes><key><fifths>1</fifths></key></attributes>" + plain).findall("measure")))
        self.assertFalse(joinable(*part(plain, '<print new-system="yes"/>' + plain).findall("measure")))
        self.assertFalse(joinable(*part(rest, plain).findall("measure")))
        self.assertTrue(joinable(*part(plain, "<print/>" + plain).findall("measure")))
        late_clef = plain + "<attributes><clef number=\"2\"><sign>G</sign></clef></attributes>" + plain
        self.assertFalse(joinable(*part(late_clef, plain).findall("measure")))
        ottava = '<direction><direction-type><octave-shift type="down" size="8"/></direction-type></direction>'
        self.assertFalse(joinable(*part(plain, ottava + plain).findall("measure")))

    def test_consecutive_boundaries_join_into_one_bar(self):
        self.assertEqual(runs_of([0, 2, 3, 5]), [[0], [2, 3], [5]])

    def test_neighbours_are_never_joined_and_at_least_one_boundary_is(self):
        p = part(*[note(s, 4) for s in "CDEFG"])
        count = with_bars_joined(p, random.Random(1), skip=(1, 1))
        measures = p.findall("measure")
        self.assertGreaterEqual(count, 1)
        self.assertEqual(len(measures), 5 - count)
        self.assertEqual(onsets(measures[0]), [("C", 0)])
        self.assertEqual(onsets(measures[-1]), [("G", 0)])


if __name__ == "__main__":
    unittest.main()
