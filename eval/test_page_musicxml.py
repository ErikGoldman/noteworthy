"""Tests for page_musicxml's finishing rules. Run from eval/: python -m unittest test_page_musicxml"""
from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET

from page_musicxml import (clefs_before_barlines, mend_ties, pair_slurs, shared_heads_shown_alike, slurs_from_ties_between_chords,
                           state_tuplet_brackets, ties_across_system_breaks, ties_from_slurs, tuplets_closed)


def note(pitch: str, *, chord: bool = False, staff: int = 1, voice: int = 1, grace: bool = False,
         slur: str | None = None, tie: str | None = None, staccato: bool = False, accent: bool = False) -> str:
    """One <note> in the delinearizer's shape. `pitch` is step and octave, such as "C4"."""
    notations = ""
    if slur or tie or staccato or accent:
        notations = ("<notations>" + (f'<slur type="{slur}"/>' if slur else "") + (f'<tied type="{tie}"/>' if tie else "")
                     + ("<articulations><staccato/></articulations>" if staccato else "")
                     + ("<articulations><accent/></articulations>" if accent else "") + "</notations>")
    return (f"<note>{'<grace/>' if grace else ''}{'<chord/>' if chord else ''}"
            f"<pitch><step>{pitch[0]}</step><octave>{pitch[1]}</octave></pitch>"
            f"<duration>2</duration>{f'<tie type=\"{tie}\"/>' if tie else ''}<voice>{voice}</voice>"
            f"<staff>{staff}</staff>{notations}</note>")


def altered(xml: str, alter: int) -> str:
    """A note string from note() with an <alter> after its step."""
    return xml.replace("</step>", f"</step><alter>{alter}</alter>", 1)


def written(xml: str, accidental: str) -> str:
    return xml.replace("<voice>", f"<accidental>{accidental}</accidental><voice>", 1)


def alters(element: ET.Element) -> list[str]:
    return [n.findtext("pitch/alter", "0") for n in element.iter("note")]


def part(*measures: str) -> ET.Element:
    return ET.fromstring("<part>" + "".join(f"<measure>{m}</measure>" for m in measures) + "</part>")


def new_system(measure: str) -> str:
    return '<print new-system="yes"/>' + measure


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
    slurs_from_ties_between_chords(p)
    ties_across_system_breaks(p)
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

    def test_held_note_joined_by_a_chord_is_tied_alone(self):
        # Polish 096 bar 8: C4 slurred into C4+E-flat4; gold ties the C
        p = finished(part(note("C4", slur="start"), note("C4", slur="stop") + note("E4", chord=True)))
        self.assertEqual(slur_count(p), 0)
        self.assertEqual(ties(p, "start"), ["C4"])
        self.assertEqual(ties(p, "stop"), ["C4"])

    def test_chord_into_a_smaller_one_stays_slurred(self):
        p = finished(part(note("C4", slur="start") + note("E4", chord=True), note("C4", slur="stop")))
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

    def test_accented_chord_slurred_to_itself_becomes_tie(self):
        # Polish 064, 069, 077: an accent says nothing about the arc
        p = finished(part(note("A4", slur="start", accent=True) + note("C5", chord=True), note("A4", slur="stop") + note("C5", chord=True)))
        self.assertEqual(slur_count(p), 0)
        self.assertEqual(sorted(ties(p, "start")), ["A4", "C5"])

    def test_tie_over_a_barline_keeps_the_accidental(self):
        # Polish 039: C natural slurred over the barline to a C the key makes sharp
        p = finished(part(note("C5", slur="start"), altered(note("C5", slur="stop"), 1)))
        self.assertEqual(ties(p, "stop"), ["C5"])
        self.assertEqual(alters(p), ["0", "0"])

    def test_sharp_carried_over_a_barline_into_a_chord(self):
        p = finished(part(altered(note("G3", slur="start"), 1) + note("B3", chord=True), note("G3", slur="stop") + note("B3", chord=True)))
        self.assertEqual(sorted(ties(p, "stop")), ["B3", "G3"])
        self.assertEqual(alters(p), ["1", "0", "1", "0"])

    def test_written_accidental_after_the_barline_stays_slurred(self):
        p = finished(part(altered(note("G3", slur="start"), 1), written(note("G3", slur="stop"), "natural")))
        self.assertEqual(slur_count(p), 2)
        self.assertEqual(alters(p), ["1", "0"])

    def test_different_alterations_in_one_bar_stay_slurred(self):
        p = finished(part(note("C5", slur="start") + altered(note("C5", slur="stop"), 1)))
        self.assertEqual(slur_count(p), 2)

    def test_slur_over_three_notes_stays(self):
        p = finished(part(note("C4", slur="start"), note("C4"), note("C4", slur="stop")))
        self.assertEqual(slur_count(p), 2)
        self.assertEqual(ties(p, "start"), [])

    def test_slur_on_the_lower_staff_becomes_tie(self):
        p = finished(part(note("A3", staff=2, slur="start"), note("A3", staff=2, slur="stop")))
        self.assertEqual(ties(p, "stop"), ["A3"])


class TiesAcrossSystemBreaks(unittest.TestCase):
    def test_stray_slur_stop_after_the_break_is_a_tie(self):
        # Polish 070: the next system's half arc read as a slur stop
        p = finished(part(note("B1", staff=2, voice=5), new_system(note("B1", staff=2, voice=5, slur="stop"))))
        self.assertEqual(slur_count(p), 0)
        self.assertEqual(ties(p, "start"), ["B1"])
        self.assertEqual(ties(p, "stop"), ["B1"])

    def test_stray_slur_start_before_the_break_is_a_tie(self):
        p = finished(part(note("F3", slur="start") + note("A3", chord=True), new_system(note("F3") + note("A3", chord=True))))
        self.assertEqual(slur_count(p), 0)
        self.assertEqual(sorted(ties(p, "stop")), ["A3", "F3"])

    def test_stray_stop_no_longer_closes_an_unrelated_slur(self):
        p = finished(part(note("D4", voice=2, slur="start") + note("B1", voice=1), new_system(note("B1", voice=1, slur="stop"))))
        self.assertEqual(ties(p, "stop"), ["B1"])
        self.assertEqual(slur_count(p), 1)

    def test_slur_to_another_pitch_over_the_break_stays(self):
        p = finished(part(note("C4", slur="start"), new_system(note("D4", slur="stop"))))
        self.assertEqual(slur_count(p), 2)
        self.assertEqual(ties(p, "start"), [])

    def test_no_half_arc_no_tie(self):
        p = finished(part(note("C4"), new_system(note("C4"))))
        self.assertEqual(ties(p, "start"), [])

    def test_cross_staff_notes_of_a_voice_are_not_neighbours(self):
        # Polish 062: a staff-1 voice drops one note a beat to staff 2, each slurred from above
        p = finished(part(note("D4", slur="start") + note("B3", staff=2),
                          new_system(note("E4") + note("B3", staff=2, slur="stop"))))
        self.assertEqual(ties(p, "start"), [])

    def test_same_stray_stop_inside_a_system_is_left_to_pair_slurs(self):
        p = finished(part(note("B1"), note("B1", slur="stop")))
        self.assertEqual(ties(p, "start"), [])


def backup(duration: int = 2) -> str:
    return f"<backup><duration>{duration}</duration></backup>"


def forward(duration: int = 2) -> str:
    return f"<forward><duration>{duration}</duration></forward>"


def rest(*, staff: int = 1, voice: int = 1) -> str:
    return f"<note><rest/><duration>2</duration><voice>{voice}</voice><staff>{staff}</staff></note>"


def tie_marks(element: ET.Element) -> int:
    """Every <tie> and <tied> on the page: a rule must not write a tie twice."""
    return len(element.findall(".//tie")) + len(element.findall(".//tied"))


class TiesSwitchingVoice(unittest.TestCase):
    """A tie over a system break whose second note opens another voice of the staff (Polish 030 bars 26-27)."""

    def break_into_two_voices(self, before: str, voice1: str, voice2: str) -> ET.Element:
        """`before` ends the system in voice 1; the next system opens with voice1 and, after a backup, voice2."""
        return part(before, new_system(voice1 + backup() + voice2))

    def test_polish_030_open_slur_into_the_second_voice_becomes_a_tie(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start"), note("E4"), note("G3", voice=2, tie="stop")))
        self.assertEqual(slur_count(p), 0)
        self.assertEqual(ties(p, "start"), ["G3"])
        self.assertEqual(ties(p, "stop"), ["G3"])

    def test_without_the_tie_stop_the_slur_stays(self):
        # no evidence the arc lands on that note: a phrase slur may start where a voice holds the same pitch
        p = finished(self.break_into_two_voices(note("G3", slur="start"), note("E4"), note("G3", voice=2)))
        self.assertEqual(ties(p, "start"), [])

    def test_a_tie_the_model_wrote_across_voices_is_kept(self):
        # mend_ties pairs by voice, and used to drop both halves
        p = finished(self.break_into_two_voices(note("G3", tie="start"), note("E4"), note("G3", voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), ["G3"])
        self.assertEqual(ties(p, "stop"), ["G3"])
        self.assertEqual(tie_marks(p), 4)

    def test_another_pitch_is_no_partner(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start"), note("E4"), note("A3", voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), [])

    def test_the_other_staff_is_no_partner(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start"), note("E4"),
                                                note("G3", staff=2, voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), [])

    def test_a_partner_after_a_rest_in_its_voice_is_not_at_the_break(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start"), note("E4"),
                                                rest(voice=2) + note("G3", voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), [])

    def test_a_partner_after_a_forward_is_not_at_the_break(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start"), note("E4"),
                                                forward() + note("G3", voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), [])

    def test_two_voices_opening_on_the_pitch_are_ambiguous(self):
        p = finished(part(note("G3", slur="start"),
                          new_system(note("E4") + backup() + note("G3", voice=2, tie="stop")
                                     + backup() + note("G3", voice=3, tie="stop"))))
        self.assertEqual(ties(p, "start"), [])
        self.assertEqual(slur_count(p), 1)

    def test_two_voices_ending_on_the_pitch_are_ambiguous(self):
        p = finished(part(note("G3", slur="start") + backup() + note("G3", voice=3, slur="start"),
                          new_system(note("E4") + backup() + note("G3", voice=2, tie="stop"))))
        self.assertEqual(ties(p, "start"), [])

    def test_a_slur_closed_later_in_the_next_system_is_a_phrase_and_stays(self):
        p = finished(part(note("G3", slur="start"),
                          new_system(note("E4") + note("F4", slur="stop") + backup(4) + note("G3", voice=2, tie="stop"))))
        self.assertEqual(ties(p, "start"), ["G3"])
        self.assertEqual(slur_count(p), 2)

    def test_a_slur_closed_on_the_other_staff_does_not_count(self):
        p = finished(part(note("G3", slur="start") + backup() + note("C3", staff=2, voice=5, slur="start"),
                          new_system(note("E4") + backup() + note("G3", voice=2, tie="stop")
                                     + backup() + note("D3", staff=2, voice=5, slur="stop"))))
        self.assertEqual(ties(p, "start"), ["G3"])
        self.assertEqual(slur_count(p), 2)

    def test_a_chord_tied_on_whole_loses_its_slur(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start") + note("B3", chord=True), note("E4"),
                                                note("G3", voice=2, tie="stop") + note("B3", voice=2, chord=True, tie="stop")))
        self.assertEqual(slur_count(p), 0)
        self.assertEqual(sorted(ties(p, "start")), ["B3", "G3"])

    def test_a_chord_tied_on_in_part_keeps_its_slur(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start") + note("D4", chord=True), note("E4"),
                                                note("G3", voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), ["G3"])
        self.assertEqual(slur_count(p), 1)

    def test_the_tie_carries_the_accidental_over_the_barline(self):
        p = finished(self.break_into_two_voices(altered(note("F4", slur="start"), 1), note("A4"),
                                                note("F4", voice=2, tie="stop")))
        self.assertEqual(ties(p, "stop"), ["F4"])
        self.assertEqual(alters(p)[-1], "1")

    def test_a_written_accidental_after_the_break_is_another_note(self):
        p = finished(self.break_into_two_voices(altered(note("F4", slur="start"), 1), note("A4"),
                                                written(note("F4", voice=2, tie="stop"), "natural")))
        self.assertEqual(ties(p, "start"), [])

    def test_a_barline_inside_a_system_is_not_a_break(self):
        p = finished(part(note("G3", slur="start"), note("E4") + backup() + note("G3", voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), [])

    def test_grace_notes_are_never_partners(self):
        p = finished(self.break_into_two_voices(note("G3", grace=True, slur="start") + note("A3"), note("E4"),
                                                note("G3", voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), [])

    def test_a_grace_note_opening_the_voice_is_no_partner(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start"), note("E4"),
                                                note("G3", voice=2, grace=True, tie="stop") + note("A3", voice=2)))
        self.assertEqual(ties(p, "start"), [])

    def test_only_the_last_note_of_the_voice_counts(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start") + note("A3"), note("E4"),
                                                note("G3", voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), [])

    def test_a_rest_ending_the_voice_is_left_alone(self):
        p = finished(self.break_into_two_voices(rest(), note("E4"), note("G3", voice=2, tie="stop")))
        self.assertEqual(ties(p, "start"), [])

    def test_the_same_voice_is_left_to_the_older_rule(self):
        p = finished(part(note("G3", slur="start"), new_system(note("G3", tie="stop"))))
        self.assertEqual(ties(p, "start"), ["G3"])
        self.assertEqual(tie_marks(p), 4)

    def test_a_voice_crossing_to_the_other_staff_is_no_partner(self):
        p = finished(self.break_into_two_voices(note("G3", slur="start"), note("E4"),
                                                note("G3", staff=2, voice=1, tie="stop")))
        self.assertEqual(ties(p, "start"), [])

    def test_the_last_bar_of_the_page_keeps_its_open_slur(self):
        # it may run on to the next page
        p = finished(part(note("C4"), note("G3", slur="start")))
        self.assertEqual(ties(p, "start"), [])
        self.assertEqual(slur_count(p), 1)

    def test_a_page_of_one_bar_or_none(self):
        self.assertEqual(ties(finished(part(note("C4", slur="start"))), "start"), [])
        self.assertEqual(ties(finished(part()), "start"), [])

    def test_unpitched_and_empty_measures_do_not_break_it(self):
        p = finished(part(rest() + backup() + rest(voice=2), new_system(""), new_system(rest())))
        self.assertEqual(tie_marks(p), 0)

    def test_running_twice_changes_nothing_more(self):
        once = finished(self.break_into_two_voices(note("G3", slur="start"), note("E4"), note("G3", voice=2, tie="stop")))
        before = ET.tostring(once)
        self.assertEqual(ET.tostring(finished(once)), before)

    def test_two_breaks_in_a_row_each_get_their_tie(self):
        p = finished(part(note("G3", slur="start"),
                          new_system(note("E4") + backup() + note("G3", voice=2, tie="stop")
                                     + note("C4", voice=2, chord=False, slur="start")),
                          new_system(note("D4") + backup() + note("C4", voice=3, tie="stop"))))
        self.assertEqual(sorted(ties(p, "stop")), ["C4", "G3"])
        self.assertEqual(slur_count(p), 0)


def chord(*pitches: str, **marks) -> str:
    """A chord: the first pitch with `marks`, the rest marked chord, each with the same tie."""
    tie = marks.get("tie")
    return note(pitches[0], **marks) + "".join(note(p, chord=True, tie=tie, staff=marks.get("staff", 1),
                                                     voice=marks.get("voice", 1)) for p in pitches[1:])


class SlursFromTiesBetweenChords(unittest.TestCase):
    """Polish 057 bar 10: two chords under one arc above and one below, read as ties from each note."""

    def test_ties_into_a_chord_of_other_pitches_become_one_slur(self):
        p = finished(part(chord("G4", "E5", tie="start") + chord("F4", "F5", tie="stop")))
        self.assertEqual(tie_marks(p), 0)
        self.assertEqual(slur_count(p), 2)

    def test_the_slur_runs_from_the_first_chord_to_the_second(self):
        p = finished(part(chord("G4", "E5", tie="start") + chord("F4", "F5")))
        notes = list(p.iter("note"))
        self.assertEqual([s.get("type") for s in notes[0].iter("slur")], ["start"])
        self.assertEqual([s.get("type") for s in notes[2].iter("slur")], ["stop"])

    def test_true_ties_between_the_same_chord_stay(self):
        p = finished(part(chord("G4", "E5", tie="start") + chord("G4", "E5", tie="stop")))
        self.assertEqual(sorted(ties(p, "stop")), ["E5", "G4"])
        self.assertEqual(slur_count(p), 0)

    def test_one_shared_pitch_leaves_it_to_mend_ties(self):
        p = finished(part(chord("G4", "E5", tie="start") + chord("G4", "F5", tie="stop")))
        self.assertEqual(ties(p, "start"), ["G4"])
        self.assertEqual(slur_count(p), 0)

    def test_a_shared_step_and_octave_with_another_alteration_counts_as_shared(self):
        p = finished(part(chord("G4", "E5", tie="start") + altered(note("G4"), 1) + note("C5", chord=True)))
        self.assertEqual(slur_count(p), 0)

    def test_a_chord_tied_in_part_is_left(self):
        p = finished(part(note("G4", tie="start") + note("E5", chord=True) + chord("F4", "F5")))
        self.assertEqual(slur_count(p), 0)

    def test_a_single_note_tied_to_another_pitch_is_left(self):
        # a misread pitch is as likely as a misread arc: mend_ties drops the tie, as before
        p = finished(part(note("G4", tie="start") + note("F4", tie="stop")))
        self.assertEqual(slur_count(p), 0)
        self.assertEqual(tie_marks(p), 0)

    def test_a_chord_into_a_single_note_is_left(self):
        p = finished(part(chord("G4", "E5", tie="start") + note("F4")))
        self.assertEqual(slur_count(p), 0)

    def test_a_rest_after_the_chord_is_left(self):
        p = finished(part(chord("G4", "E5", tie="start") + rest()))
        self.assertEqual(slur_count(p), 0)

    def test_another_voice_is_not_the_next_chord(self):
        p = finished(part(chord("G4", "E5", tie="start") + backup() + chord("F4", "F5", voice=2)))
        self.assertEqual(slur_count(p), 0)

    def test_over_a_barline_too(self):
        p = finished(part(chord("G4", "E5", tie="start"), chord("F4", "F5", tie="stop")))
        self.assertEqual(slur_count(p), 2)
        self.assertEqual(tie_marks(p), 0)

    def test_an_existing_slur_start_is_not_doubled(self):
        p = finished(part(chord("G4", "E5", tie="start", slur="start") + chord("F4", "F5")))
        self.assertEqual(len(list(p.iter("note"))[0].findall("notations/slur")), 1)

    def test_the_last_chord_of_the_page_is_left(self):
        p = finished(part(note("C4"), chord("G4", "E5", tie="start")))
        self.assertEqual(slur_count(p), 0)

    def test_each_pair_in_a_run_of_chords(self):
        p = finished(part(chord("G4", "E5", tie="start") + chord("F4", "D5", tie="start") + chord("E4", "C5")))
        self.assertEqual(slur_count(p), 4)
        self.assertEqual(tie_marks(p), 0)


class MendTies(unittest.TestCase):
    def test_half_tie_is_completed(self):
        p = finished(part(note("C4", tie="start"), note("C4")))
        self.assertEqual(ties(p, "stop"), ["C4"])

    def test_tie_to_a_different_pitch_goes(self):
        p = finished(part(note("C4", tie="start"), note("D4")))
        self.assertEqual(ties(p, "start"), [])

    def test_tie_off_the_page_end_stays(self):
        # Polish 081: the last chord's ties run on to the next page
        p = finished(part(note("C4"), note("F4", tie="start")))
        self.assertEqual(ties(p, "start"), ["F4"])

    def test_voice_ending_mid_page_drops_its_open_tie(self):
        p = finished(part(note("C4", tie="start", voice=2) + "<backup><duration>2</duration></backup>" + note("E4"),
                          note("F4")))
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

    def test_lower_staff_clef_after_the_upper_staff_moves_too(self):
        # Polish 081: a system opens with the lower staff in a new clef, after the upper staff's notes
        p = part(note("C3", staff=2), note("C5") + "<backup><duration>2</duration></backup>" + self.CLEF
                 + note("C4", staff=2))
        clefs_before_barlines(p)
        first, second = p.findall("measure")
        self.assertIsNotNone(first.find("attributes/clef"))
        self.assertIsNone(second.find("attributes"))

    def test_lower_staff_clef_after_its_own_notes_stays(self):
        p = part(note("C3", staff=2), note("C5") + "<backup><duration>2</duration></backup>" + note("C3", staff=2)
                 + self.CLEF + note("C4", staff=2))
        clefs_before_barlines(p)
        self.assertIsNotNone(p.findall("measure")[1].find("attributes/clef"))

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

    def test_unbeamed_triplet_with_its_number_hidden_takes_none(self):
        hidden = tuplet_note("C4", "start").replace('<tuplet type="start"/>', '<tuplet type="start" show-number="none"/>')
        self.assertEqual(self.bracket(hidden, tuplet_note("D4"), tuplet_note("E4", "stop")), "no")


class SharedHeads(unittest.TestCase):
    def test_accidental_of_a_shared_head_shows_in_both_voices(self):
        # Polish 037: F5 natural in voices 1 and 2 at once, the natural read in voice 1 only
        p = part(written(note("F5"), "natural") + "<backup><duration>2</duration></backup>" + note("F5", voice=2))
        shared_heads_shown_alike(p)
        self.assertEqual([n.findtext("accidental") for n in p.iter("note")], ["natural", "natural"])

    def test_later_note_of_the_other_voice_keeps_none(self):
        p = part(written(note("F5"), "natural") + note("G5") + "<backup><duration>4</duration></backup>"
                 + note("G5", voice=2) + note("F5", voice=2))
        shared_heads_shown_alike(p)
        self.assertEqual([n.findtext("accidental") for n in p.iter("note")], ["natural", None, None, None])

    def test_other_staff_keeps_none(self):
        p = part(written(note("F4"), "natural") + "<backup><duration>2</duration></backup>" + note("F4", staff=2, voice=5))
        shared_heads_shown_alike(p)
        self.assertEqual([n.findtext("accidental") for n in p.iter("note")], ["natural", None])


def unmarked(content: str, beam: str | None = None) -> str:
    """A triplet eighth with its ratio but no start or stop, as imply_tuplets leaves it."""
    return tuplet_note(content, None, beam)


def tuplet_marks(p: ET.Element) -> list[tuple[str, str | None]]:
    return [(t.get("type"), t.get("show-number")) for t in p.iter("tuplet")]


class TupletsClosed(unittest.TestCase):
    def test_unmarked_triplets_close_every_three_with_no_number(self):
        # Polish 077 bar 4: twelve eighths the model wrote with no tuplet marks
        p = part("".join(unmarked(pitch) for pitch in ["C4", "D4", "E4", "F4", "G4", "A4"]))
        tuplets_closed(p)
        self.assertEqual(tuplet_marks(p), [("start", "none"), ("stop", None)] * 2)

    def test_group_the_model_opened_is_left_alone(self):
        # Polish 087 bar 5: a start with no stop, then unmarked notes inside it
        p = part(tuplet_note("C4", "start") + "".join(unmarked(pitch) for pitch in ["D4", "E4", "F4"]))
        tuplets_closed(p)
        self.assertEqual(tuplet_marks(p), [("start", None)])

    def test_group_does_not_run_over_the_barline(self):
        p = part(unmarked("C4") + unmarked("D4"), unmarked("E4") + unmarked("F4") + unmarked("G4"))
        tuplets_closed(p)
        first, second = p.findall("measure")
        self.assertEqual(tuplet_marks(first), [])
        self.assertEqual(tuplet_marks(second), [("start", "none"), ("stop", None)])


if __name__ == "__main__":
    unittest.main()
