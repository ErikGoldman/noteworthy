"""Tests for seq.tuplet_display. Run from train/: python -m unittest test_tuplet_display"""
from __future__ import annotations

import random
import unittest
import xml.etree.ElementTree as ET

from seq.tuplet_display import (BY_RUN, HIDDEN_BY_RUN_PLACE, HIDE_ALL, SHOW_ALL, TupletLook, hidden_share,
                                hide_as_engravers_do, set_look, tuplet_groups)


def note(step: str, kind: str = "eighth", tuplet: str | None = None, voice: str = "1", ratio=(3, 2)) -> str:
    modification = (f"<time-modification><actual-notes>{ratio[0]}</actual-notes><normal-notes>{ratio[1]}"
                    f"</normal-notes></time-modification>") if ratio else ""
    notations = f'<notations><tuplet type="{tuplet}"/></notations>' if tuplet else ""
    return (f"<note><pitch><step>{step}</step><octave>4</octave></pitch><duration>1</duration>"
            f"<voice>{voice}</voice><type>{kind}</type>{modification}<staff>1</staff>{notations}</note>")


def triplet(voice: str = "1", ratio=(3, 2), kind: str = "eighth") -> str:
    return (note("C", kind, "start", voice, ratio) + note("D", kind, None, voice, ratio)
            + note("E", kind, "stop", voice, ratio))


def plain(voice: str = "1") -> str:
    return note("F", "quarter", None, voice, None)


def part(*measures: str) -> ET.Element:
    return ET.fromstring("<part>" + "".join(f"<measure>{m}</measure>" for m in measures) + "</part>")


class Groups(unittest.TestCase):
    def test_runs_count_like_groups_back_to_back(self):
        groups = tuplet_groups(part(triplet() * 3 + plain(), triplet()))
        self.assertEqual([g.run_place for g in groups], [1, 2, 3, 1])

    def test_a_run_crosses_the_barline(self):
        self.assertEqual([g.run_place for g in tuplet_groups(part(triplet() * 2, triplet()))], [1, 2, 3])

    def test_another_ratio_or_value_starts_a_new_run(self):
        groups = tuplet_groups(part(triplet() + triplet(ratio=(6, 4)) + triplet(kind="16th")))
        self.assertEqual([g.run_place for g in groups], [1, 1, 1])

    def test_first_in_voice_after_plain_notes(self):
        groups = tuplet_groups(part(plain() + triplet() + triplet()))
        self.assertEqual([g.first_in_voice for g in groups], [True, False])

    def test_voices_are_apart(self):
        groups = tuplet_groups(part(triplet("1") + "<backup><duration>3</duration></backup>" + triplet("2"), triplet("1")))
        self.assertEqual([(g.run_place, g.first_in_voice) for g in groups], [(1, True), (1, True), (2, False)])

    def test_alone_in_bar(self):
        groups = tuplet_groups(part(triplet() + plain(), triplet() * 2))
        self.assertEqual([g.alone_in_bar for g in groups], [True, False, False])


class Hiding(unittest.TestCase):
    def hidden(self, p: ET.Element) -> list[bool]:
        return [g.start.get("show-number") == "none" for g in tuplet_groups(p)]

    def test_show_all_and_hide_all(self):
        for hiding, expected in ((SHOW_ALL, False), (HIDE_ALL, True)):
            p = part(triplet() * 4)
            hide_as_engravers_do(p, hiding, random.Random(0))
            self.assertEqual(self.hidden(p), [expected] * 4)

    def test_by_run_hides_late_groups_more(self):
        rng, counts = random.Random(0), [0] * 6
        for _ in range(2000):
            p = part(plain() + triplet() * 6)
            hide_as_engravers_do(p, BY_RUN, rng)
            counts = [c + h for c, h in zip(counts, self.hidden(p))]
        self.assertLess(counts[0], counts[5])
        self.assertAlmostEqual(counts[5] / 2000, HIDDEN_BY_RUN_PLACE[-1], delta=0.03)

    def test_odd_groupings_mostly_printed(self):
        group = tuplet_groups(part(triplet() * 4 + triplet(ratio=(5, 4))))[-1]
        self.assertLessEqual(hidden_share(group), 0.1)

    def test_a_hidden_number_keeps_its_hiding_under_a_look(self):
        p = part(triplet() * 2)
        tuplet_groups(p)[0].start.set("show-number", "none")
        set_look(p, TupletLook(side="above", ratio_shown=True))
        starts = [g.start for g in tuplet_groups(p)]
        self.assertEqual([s.get("show-number") for s in starts], ["none", "both"])
        self.assertEqual(starts[1].get("placement"), "above")


if __name__ == "__main__":
    unittest.main()
