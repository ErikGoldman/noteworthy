"""Tests for gold_overrides: line-local ops, staff moves, slurs per chord. Run from eval/: python -m unittest test_gold_overrides"""
from __future__ import annotations

import unittest

from gold_overrides import AMBIGUOUS_DECLARATION, corrected, midi, one_slur_per_chord

PAGE = "\n".join(["**kern\t**kern", "*M2/4\t*M2/4", "=1\t=1", "4C\t16qqccL", ".\t16qqddJ", "4E\t8ccL",
                  ".\t8dd", "=2\t=2", "4G\t4g# 4b", "*-\t*-", ""])


def override(line: int, **op) -> dict:
    return {"page": "000", "line": line, "standard": "print", "reason": "test", **op}


def line(text: str, number: int) -> str:
    return text.split("\n")[number - 1]


class LineLocalOps(unittest.TestCase):
    def test_slur_marks_land_on_their_notes(self):
        text = corrected(PAGE, [override(4, slur={"column": 1, "marks": "("}),
                                override(6, slur={"column": 1, "marks": ")("})])
        self.assertEqual(line(text, 4), "4C\t16qqccL(")
        self.assertEqual(line(text, 6), "4E\t8ccL)(")

    def test_show_accidental_marks_the_chosen_chord_note(self):
        text = corrected(PAGE, [override(9, show_accidental={"column": 1, "note": 0})])
        self.assertEqual(line(text, 9), "4G\t4g#X 4b")

    def test_ambiguous_articulation_is_flagged_and_declared(self):
        text = corrected(PAGE, [override(4, articulate={"column": 0, "mark": "'", "ambiguous": True})])
        self.assertEqual(line(text, 4), "4C'i\t16qqccL")
        self.assertIn(AMBIGUOUS_DECLARATION, text)

    def test_ops_on_one_line_all_apply(self):
        text = corrected(PAGE, [override(6, articulate={"column": 0, "mark": "^"}),
                                override(6, slur={"column": 1, "marks": ")"})])
        self.assertEqual(line(text, 6), "4E^\t8ccL)")

    def test_no_overrides_leaves_the_page_alone(self):
        self.assertEqual(corrected(PAGE, []), PAGE)


class OneSlurPerChord(unittest.TestCase):
    def test_chord_keeps_its_first_slur_only(self):
        self.assertEqual(one_slur_per_chord("4E( 4G( 4e(\t4C("), "4E( 4G 4e\t4C(")

    def test_closing_marks_too(self):
        self.assertEqual(one_slur_per_chord("4E) 4G)"), "4E) 4G")

    def test_interpretation_and_barlines_untouched(self):
        self.assertEqual(one_slur_per_chord("*\t*\n=1\t=1"), "*\t*\n=1\t=1")


MOVES = "\n".join(["**kern\t**kern", "*clefF4\t*clefG2", "=1\t=1", "8C 8cL\t4e", "8D 8dJ\t.", "4E 4e\t4f",
                   "=2\t=2", "4F\t4g", "*-\t*-", ""])


def spines_consistent(text: str) -> bool:
    """Every line has as many fields as the spine splits (*^) and merges (*v) before it allow."""
    width = None
    for row in filter(None, text.split("\n")):
        fields = row.split("\t")
        if row.startswith("!!"):
            continue
        if width is not None and len(fields) != width:
            return False
        width = len(fields) + fields.count("*^")
        merges, run = 0, 0
        for field in fields + [""]:
            if field == "*v":
                run += 1
            else:
                merges, run = merges + max(0, run - 1), 0
        width -= merges
    return True


def columns(text: str, number: int) -> list[str]:
    return line(text, number).split("\t")


class StaffMoves(unittest.TestCase):
    def test_move_up_puts_high_left_notes_in_a_new_upper_voice(self):
        text = corrected(MOVES, [override(4, move_up={"last_line": 6, "lowest": 60})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(line(text, 4), "*\t*^")
        self.assertEqual(columns(text, 5), ["8C", "4e", "8c"])
        self.assertEqual(columns(text, 7), ["4E", "4f", "4e"])
        self.assertEqual(line(text, 8), "*\t*v\t*v")

    def test_move_up_drops_the_beam_it_splits_between_staves(self):
        # no kern spine holds a beam across staves (062, 067)
        text = corrected(MOVES, [override(4, move_up={"last_line": 6, "lowest": 60})])
        self.assertNotIn("L", line(text, 5) + line(text, 6))
        self.assertNotIn("J", line(text, 5) + line(text, 6))

    def test_move_up_keeps_a_beam_whose_group_moves_whole(self):
        text = corrected(MOVES, [override(4, move_up={"last_line": 6, "lowest": 36})])
        self.assertEqual(columns(text, 5)[2], "8C 8cL")
        self.assertTrue(columns(text, 5)[0].endswith("ryy"))

    def test_move_down_puts_upper_notes_in_a_new_lower_voice(self):
        text = corrected(MOVES, [override(4, move_down={"last_line": 6})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(columns(text, 5), ["8C 8cL", "4e", "4ryy"])
        self.assertEqual(columns(text, 7), ["4E 4e", "4f", "4ryy"])

    def test_staff_down_leaves_the_upper_staff_blank_and_its_clef_on_it(self):
        page = MOVES.replace("8D 8dJ\t.", "*\t*clefG2\n8D 8dJ\t.")
        text = corrected(page, [override(4, staff_down={"last_line": 7, "lower_columns": 1})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(columns(text, 5), ["8C 8cL", "4e", "4ryy"])
        clef_line = next(row for row in text.split("\n") if "clefG2" in row and not row.startswith("*clefF4"))
        self.assertEqual(clef_line.split("\t")[:2], ["*", "*"])

    def test_lines_outside_the_range_are_unchanged(self):
        text = corrected(MOVES, [override(4, move_down={"last_line": 6})])
        self.assertEqual(line(text, 10), "4F\t4g")


class Pitch(unittest.TestCase):
    def test_kern_octaves_and_accidentals(self):
        self.assertEqual([midi("c"), midi("CC"), midi("b-"), midi("f#"), midi("cc")], [60, 36, 70, 66, 72])
        self.assertIsNone(midi("4r"))


if __name__ == "__main__":
    unittest.main()
