"""Tests for gold_overrides: line-local ops, staff moves, slurs per chord. Run from eval/: python -m unittest test_gold_overrides"""
from __future__ import annotations

import unittest

from gold_overrides import AMBIGUOUS_DECLARATION, beams_closed, corrected, midi, one_slur_per_chord

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

    def test_slash_grace_slashes_every_grace_in_the_field_only(self):
        text = corrected(PAGE.replace("4E\t8ccL", "16qqEE\t8qqcc 8qqee"), [override(6, slash_grace={"column": 1})])
        self.assertEqual(line(text, 6), "16qqEE\t8qcc 8qee")

    def test_hide_rest_hides_that_column_only(self):
        page = PAGE.replace("4E\t8ccL", "4r\t8r")
        text = corrected(page, [override(6, hide_rest={"column": 1})])
        self.assertEqual(line(text, 6), "4r\t8ryy")

    def test_hide_rest_leaves_a_hidden_rest_alone(self):
        page = PAGE.replace("4E\t8ccL", "4ryy\t8r")
        self.assertEqual(line(corrected(page, [override(6, hide_rest={"column": 0})]), 6), "4ryy\t8r")

    def test_ambiguous_articulation_is_flagged_and_declared(self):
        text = corrected(PAGE, [override(4, articulate={"column": 0, "mark": "'", "ambiguous": True})])
        self.assertEqual(line(text, 4), "4C'i\t16qqccL")
        self.assertIn(AMBIGUOUS_DECLARATION, text)

    def test_ops_on_one_line_all_apply(self):
        text = corrected(PAGE, [override(6, articulate={"column": 0, "mark": "^"}),
                                override(6, slur={"column": 1, "marks": ")"})])
        self.assertEqual(line(text, 6), "4E^\t8ccL)")

    def test_replace_swaps_a_passage_for_a_fresh_transcription(self):
        text = corrected(PAGE, [override(9, replace={"last_line": 9, "lines": ["4G\t4g#", "4A\t4a"]})])
        self.assertEqual(text.split("\n")[8:11], ["4G\t4g#", "4A\t4a", "*-\t*-"])

    def test_replace_and_a_later_mark_both_apply(self):
        # overrides apply from the bottom up, so line numbers after the passage still hold
        text = corrected(PAGE, [override(4, replace={"last_line": 5, "lines": ["4C\t4cc"]}),
                                override(6, slur={"column": 1, "marks": "("})])
        self.assertEqual(text.split("\n")[3:5], ["4C\t4cc", "4E\t8ccL("])

    def test_tuplet_numbers_hidden_then_shown_again(self):
        text = corrected(PAGE, [override(4, tuplet_numbers="hidden"), override(8, tuplet_numbers="shown")])
        self.assertEqual(line(text, 4), "*Xtuplet\t*Xtuplet")
        self.assertEqual(line(text, 5), "4C\t16qqccL")
        self.assertEqual(line(text, 9), "*tuplet\t*tuplet")
        self.assertEqual(line(text, 10), "=2\t=2")

    def test_tuplet_numbers_go_before_an_edit_of_the_same_line(self):
        text = corrected(PAGE, [override(4, tuplet_numbers="hidden"), override(4, slur={"column": 1, "marks": "("})])
        self.assertEqual(line(text, 4), "*Xtuplet\t*Xtuplet")
        self.assertEqual(line(text, 5), "4C\t16qqccL(")

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
        text = corrected(MOVES, [override(4, move_up={"last_line": 6, "notes": {"4": [1], "5": [1], "6": [1]}})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(line(text, 4), "*\t*^")
        self.assertEqual(columns(text, 5), ["8C", "4e", "8c"])
        self.assertEqual(columns(text, 7), ["4E", "4f", "4e"])
        self.assertEqual(line(text, 8), "*\t*v\t*v")

    def test_move_up_drops_the_beam_it_splits_between_staves(self):
        # no kern spine holds a beam across staves (062, 067)
        text = corrected(MOVES, [override(4, move_up={"last_line": 6, "notes": {"4": [1], "5": [1], "6": [1]}})])
        self.assertNotIn("L", line(text, 5) + line(text, 6))
        self.assertNotIn("J", line(text, 5) + line(text, 6))

    def test_move_up_keeps_a_beam_whose_group_moves_whole(self):
        text = corrected(MOVES, [override(4, move_up={"last_line": 6, "notes": {"4": [0, 1], "5": [0, 1], "6": [0, 1]}})])
        self.assertEqual(columns(text, 5)[2], "8C 8cL")
        self.assertTrue(columns(text, 5)[0].endswith("ryy"))

    def test_move_up_moves_only_the_listed_notes_whatever_their_pitch(self):
        # 062 bar 6: one B-flat 3 is drawn on the upper staff, the bar's others on the lower
        page = MOVES.replace("4E 4e\t4f", "4E 4e\t4f\n4E\t4g")
        text = corrected(page, [override(4, move_up={"last_line": 7, "notes": {"6": [0, 1]}})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(columns(text, 7), ["4ryy", "4f", "4E 4e"])
        self.assertEqual(columns(text, 8), ["4E", "4g", "4ryy"])

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


def kern_page(*rows: str) -> str:
    return "\n".join(["**kern", "*M2/4", "=1", *rows, "=2", "*-", ""])


class StrayBeamEnds(unittest.TestCase):
    def test_group_that_ended_a_note_early_runs_on(self):
        # Polish 042: two 16ths and an eighth under one beam, gold closing it after the 16ths too
        page = kern_page("16ccL", "16ddJ", "8eeJ", "4f")
        self.assertEqual(beams_closed(page), kern_page("16ccL", "16dd", "8eeJ", "4f"))

    def test_stray_end_with_no_group_before_goes(self):
        page = kern_page("4c", "8dJ", "8e")
        self.assertEqual(beams_closed(page), kern_page("4c", "8d", "8e"))

    def test_matched_groups_are_untouched(self):
        page = kern_page("8cL", "8dJ", "8eL", "8fJ")
        self.assertEqual(beams_closed(page), page)


class Pitch(unittest.TestCase):
    def test_kern_octaves_and_accidentals(self):
        self.assertEqual([midi("c"), midi("CC"), midi("b-"), midi("f#"), midi("cc")], [60, 36, 70, 66, 72])
        self.assertIsNone(midi("4r"))


if __name__ == "__main__":
    unittest.main()
