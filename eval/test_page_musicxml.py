"""Tests for page_musicxml's finishing rules. Run from eval/: python -m unittest test_page_musicxml"""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from page_musicxml import clefs_before_barlines, mend_ties, pair_slurs, state_tuplet_brackets, ties_from_slurs


def note(pitch: str, *, chord: bool = False, staff: int = 1, voice: int = 1, grace: bool = False,
         slur: str | None = None, tie: str | None = None, staccato: bool = False) -> str:
    """One <note> in the delinearizer's shape. `pitch` is step and octave, such as "C4"."""
    notations = ""
    if slur or tie or staccato:
        notations = ("<notations>" + (f'<slur type="{slur}"/>' if slur else "") + (f'<tied type="{tie}"/>' if tie else "")
                     + ("<articulations><staccato/></articulations>" if staccato else "") + "</notations>")
    return (f"<note>{'<grace/>' if grace else ''}{'<chord/>' if chord else ''}"
            f"<pitch><step>{pitch[0]}</step><octave>{pitch[1]}</octave></pitch>"
            f"<duration>2</duration>{f'<tie type=\"{tie}\"/>' if tie else ''}<voice>{voice}</voice>"
            f"<staff>{staff}</staff>{notations}</note>")


def part(*measures: str) -> ET.Element:
    return ET.fromstring("<part>" + "".join(f"<measure>{m}</measure>" for m in measures) + "</part>")


def ties(element: ET.Element, kind: str) -> list[str]:
    """The pitches whose notes carry a tie of that kind, in both places MusicXML keeps it."""
    tied = []
    for n in element.iter("note"):
        in_note = any(t.get("type") == kind for t in n.findall("tie"))
        in_notations = any(t.get("type") == kind for t in n.findall("notations/tied"))
        assert in_note == in_notations, "a tie must be in <tie> and <tied> alike"
        if in_note:
            tied.append(n.findtext("pitch/step") + n.findtext("pitch/octave"))
    return tied


def slur_count(element: ET.Element) -> int:
    return len(element.findall(".//slur"))


def finished(p: ET.Element) -> ET.Element:
    """The slur and tie rules in finish_score's order."""
    pair_slurs(p)
    ties_from_slurs(p)
    mend_ties(p)
    return p


class TiesFromSlurs(unittest.TestCase):
    def test_slur_between_same_notes_becomes_tie(self):
        p = finished(part(note("C4", slur="start"), note("C4", slur="stop")))
        self.assertEqual(slur_count(p), 0)
        self.assertEqual(ties(p, "start"), ["C4"])
        self.assertEqual(ties(p, "stop"), ["C4"])

    def test_slur_between_same_chords_ties_every_note(self):
        p = finished(part(note("C4", slur="start") + note("E4", chord=True) + note("G4", chord=True),
                          note("C4", slur="stop") + note("E4", chord=True) + note("G4", chord=True)))
        self.assertEqual(slur_count(p), 0)
        self.assertEqual(sorted(ties(p, "start")), ["C4", "E4", "G4"])

    def test_chord_moving_under_a_slur_stays_slurred(self):
        # Polish 049: F4+D5 to F4+C5 shares F4, but the print's arc is a slur
        p = finished(part(note("F4", slur="start") + note("D5", chord=True),
                          note("F4", slur="stop") + note("C5", chord=True)))
        self.assertEqual(slur_count(p), 2)
        self.assertEqual(ties(p, "start"), [])

    def test_slur_to_a_different_pitch_stays(self):
        p = finished(part(note("C4", slur="start"), note("D4", slur="stop")))
        self.assertEqual(slur_count(p), 2)
        self.assertEqual(ties(p, "start"), [])

    def test_grace_note_slurred_to_its_pitch_stays_slurred(self):
        # Polish 017: a grace C slurred to a C
        p = finished(part(note("C5", grace=True, slur="start"), note("C5", slur="stop")))
        self.assertEqual(slur_count(p), 2)
        self.assertEqual(ties(p, "start"), [])

    def test_portato_stays_slurred(self):
        p = finished(part(note("C4", slur="start", staccato=True), note("C4", slur="stop", staccato=True)))
        self.assertEqual(slur_count(p), 2)

    def test_slur_over_three_notes_stays(self):
        p = finished(part(note("C4", slur="start"), note("C4"), note("C4", slur="stop")))
        self.assertEqual(slur_count(p), 2)
        self.assertEqual(ties(p, "start"), [])

    def test_slur_on_the_lower_staff_becomes_tie(self):
        p = finished(part(note("A3", staff=2, slur="start"), note("A3", staff=2, slur="stop")))
        self.assertEqual(ties(p, "stop"), ["A3"])


class MendTies(unittest.TestCase):
    def test_half_tie_is_completed(self):
        p = finished(part(note("C4", tie="start"), note("C4")))
        self.assertEqual(ties(p, "stop"), ["C4"])

    def test_tie_to_a_different_pitch_goes(self):
        p = finished(part(note("C4", tie="start"), note("D4")))
        self.assertEqual(ties(p, "start"), [])


class ClefsBeforeBarlines(unittest.TestCase):
    CLEF = '<attributes><clef number="2"><sign>G</sign><line>2</line></clef></attributes>'

    def test_clef_opening_a_bar_moves_to_the_end_of_the_bar_before(self):
        p = part(note("C3", staff=2), self.CLEF + note("C4", staff=2))
        clefs_before_barlines(p)
        first, second = p.findall("measure")
        self.assertEqual([c.tag for c in first][-1], "attributes")
        self.assertIsNotNone(first.find("attributes/clef"))
        self.assertIsNone(second.find("attributes"))

    def test_clef_inside_a_bar_stays(self):
        p = part(note("C3", staff=2), note("C3", staff=2) + self.CLEF + note("C4", staff=2))
        clefs_before_barlines(p)
        self.assertIsNotNone(p.findall("measure")[1].find("attributes/clef"))

    def test_first_bar_keeps_its_clefs(self):
        p = part(self.CLEF + note("C4", staff=2))
        clefs_before_barlines(p)
        self.assertIsNotNone(p.find("measure/attributes/clef"))

    def test_other_attributes_stay_with_their_bar(self):
        p = part(note("C3", staff=2), '<attributes><time><beats>3</beats><beat-type>4</beat-type></time>'
                 '<clef number="2"><sign>G</sign><line>2</line></clef></attributes>' + note("C4", staff=2))
        clefs_before_barlines(p)
        second = p.findall("measure")[1]
        self.assertIsNotNone(second.find("attributes/time"))
        self.assertIsNone(second.find("attributes/clef"))


def tuplet_note(content: str, tuplet: str | None = None, beam: str | None = None) -> str:
    """A triplet eighth: a pitch such as "C4", or "rest"."""
    sound = "<rest/>" if content == "rest" else f"<pitch><step>{content[0]}</step><octave>{content[1]}</octave></pitch>"
    return (f"<note>{sound}<duration>1</duration><voice>1</voice><type>eighth</type>"
            "<time-modification><actual-notes>3</actual-notes><normal-notes>2</normal-notes></time-modification>"
            f"{f'<beam number=\"1\">{beam}</beam>' if beam else ''}<staff>1</staff>"
            f"{f'<notations><tuplet type=\"{tuplet}\"/></notations>' if tuplet else ''}</note>")


class TupletBrackets(unittest.TestCase):
    def bracket(self, *notes: str) -> str:
        p = part("".join(notes))
        state_tuplet_brackets(p)
        return p.find(".//tuplet[@type='start']").get("bracket")

    def test_beamed_triplet_takes_no_bracket(self):
        self.assertEqual(self.bracket(tuplet_note("C4", "start", "begin"), tuplet_note("D4", None, "continue"),
                                      tuplet_note("E4", "stop", "end")), "no")

    def test_triplet_opening_with_a_rest_before_its_beam_takes_none(self):
        # Polish 003: rest, then two beamed eighths
        self.assertEqual(self.bracket(tuplet_note("rest", "start"), tuplet_note("D4", None, "begin"),
                                      tuplet_note("E4", "stop", "end")), "no")

    def test_unbeamed_triplet_takes_a_bracket(self):
        self.assertEqual(self.bracket(tuplet_note("C4", "start"), tuplet_note("D4"), tuplet_note("E4", "stop")), "yes")


if __name__ == "__main__":
    unittest.main()
