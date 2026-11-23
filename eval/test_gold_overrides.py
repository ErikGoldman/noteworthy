"""Tests for gold_overrides: line-local ops, staff moves, slurs per chord. Run from eval/: python -m unittest test_gold_overrides"""
from __future__ import annotations

import unittest

import tempfile
from pathlib import Path

from gold_overrides import AMBIGUOUS_DECLARATION, beams_closed, corrected, midi, one_slur_per_chord, staves_of_columns

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

    def test_beams_split_a_run_the_print_draws_as_two_groups(self):
        text = corrected(PAGE, [override(5, beams={"column": 1, "marks": "JJ"}), override(6, beams={"column": 1, "marks": "L"})])
        self.assertEqual(line(text, 5), ".\t16qqddJJ")
        self.assertEqual(line(text, 6), "4E\t8ccL")

    def test_beams_go_on_a_chords_last_note_only(self):
        text = corrected(PAGE.replace("4E\t8ccL", "4E\t8ccL 8eeL"), [override(6, beams={"column": 1, "marks": ""})])
        self.assertEqual(line(text, 6), "4E\t8cc 8ee")

    def test_show_accidental_marks_the_chosen_chord_note(self):
        text = corrected(PAGE, [override(9, show_accidental={"column": 1, "note": 0})])
        self.assertEqual(line(text, 9), "4G\t4g#X 4b")

    def test_show_accidental_on_a_note_written_without_one_draws_a_natural(self):
        text = corrected(PAGE, [override(9, show_accidental={"column": 1, "note": 1})])
        self.assertEqual(line(text, 9), "4G\t4g# 4bn")

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

    def test_a_mark_lands_on_the_line_a_replace_writes(self):
        # 065 bars 39-40: chords transcribed afresh where marks were already reviewed
        text = corrected(PAGE, [override(6, articulate={"column": 1, "mark": "'"}),
                                override(6, replace={"last_line": 6, "lines": ["4E\t8cc 8eeL"]})])
        self.assertEqual(line(text, 6), "4E\t8cc 8eeL'")

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

    def test_a_rest_hidden_on_a_line_that_moves_is_hidden_whatever_the_order(self):
        # Polish 060 bar 29: move_up and hide_rest on one line; the move once put its spine line there first
        page = MOVES.replace("8D 8dJ\t.", "8D 8dJ\t4r")
        moves = override(5, move_up={"last_line": 5, "notes": {"5": [1]}})
        hide = override(5, hide_rest={"column": 1})
        for order in ([moves, hide], [hide, moves]):
            text = corrected(page, order)
            self.assertTrue(spines_consistent(text))
            self.assertIn("4ryy", text)
            self.assertNotIn("\t4r\t", text + "\t")

    def test_hide_rest_on_a_field_with_no_drawn_rest_stops(self):
        with self.assertRaisesRegex(ValueError, "found no rest"):
            corrected(MOVES, [override(4, hide_rest={"column": 1})])

    def test_move_up_takes_notes_from_a_named_lower_column(self):
        # 077 bars 18-20: a spine split from the lower staff carries the upper staff's melody
        page = MOVES.replace("*clefF4\t*clefG2", "*clefF4\t*clefG2\n*^\t*").replace(
            "8C 8cL\t4e", "8C\t4aa\t4e").replace("8D 8dJ\t.", "8D\t.\t.").replace("4E 4e\t4f", "4E\t4gg\t4f").replace(
            "=1\t=1", "=1\t=1\t=1").replace("=2\t=2", "*v\t*v\t*\n=2\t=2")
        text = corrected(page, [override(5, move_up={"last_line": 7, "notes": {"5": [0], "7": [0]}, "column": 1})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(columns(text, 6), ["8C", "4ryy", "4e", "4aa"])
        self.assertEqual(columns(text, 8), ["4E", "4ryy", "4f", "4gg"])

    def test_a_split_chord_keeps_its_one_slur_with_the_notes_that_stay(self):
        # 030 bars 17-18: octaves joined by one stem across the staves, one slur over the chord
        page = MOVES.replace("4E 4e\t4f", "4E 4e\t4f( 4a(")
        text = corrected(page, [override(6, move_down={"last_line": 6, "notes": {"6": [0]}, "column": 1})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(columns(text, 7)[1], "4f")
        self.assertEqual(columns(text, 7)[2], "4a(")

    def test_move_down_puts_upper_notes_in_a_new_lower_voice(self):
        text = corrected(MOVES, [override(4, move_down={"last_line": 6})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(columns(text, 5), ["8C 8cL", "4e", "4ryy"])
        self.assertEqual(columns(text, 7), ["4E 4e", "4f", "4ryy"])

    def test_move_down_takes_only_the_listed_notes(self):
        # 110 bars 23-27: each triplet's first note is drawn on the lower staff, the rest above
        page = MOVES.replace("4E 4e\t4f", "4E 4e\t4f\n4E\t4g 4a")
        text = corrected(page, [override(4, move_down={"last_line": 7, "notes": {"7": [1]}})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(columns(text, 7), ["4E 4e", "4ryy", "4f"])
        self.assertEqual(columns(text, 8), ["4E", "4a", "4g"])

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


# The upper staff in two voices (columns: lower, upper voice 1, upper voice 2), as 030 bars 29 and
# 37 and 056 bar 17 have it where a voice's notes are drawn on the lower staff.
UPPER_SPLIT = "\n".join(["**kern\t**kern", "*clefF4\t*clefG2", "*M2/4\t*M2/4", "=1\t=1", "*\t*^",
                         "4C\t4cc\t4e", "4D\t4dd\t4f", "*\t*v\t*v", "=2\t=2", "4E\t2g", "4F\t.", "*-\t*-", ""])


def parsed_staves(text: str) -> list[list[str]]:
    """The pitches music21 reads on each staff, upper first, as the scorer parses gold."""
    from omr_ned import authors_metric, parsed_as_authors_do  # musicdiff's venv
    with tempfile.TemporaryDirectory() as folder:
        page = Path(folder) / "page.krn"
        page.write_text(text)
        score = parsed_as_authors_do(authors_metric(), page, prediction=False)
    return [[pitch.nameWithOctave for note in part.recurse().notes for pitch in note.pitches] for part in score.parts]


class StavesOfColumns(unittest.TestCase):
    def test_one_column_a_staff_from_the_header(self):
        self.assertEqual(staves_of_columns(UPPER_SPLIT.split("\n")[:3]), [0, 1])

    def test_a_split_voice_stays_on_its_staff_and_a_merge_ends_it(self):
        lines = UPPER_SPLIT.split("\n")
        self.assertEqual(staves_of_columns(lines[:5]), [0, 1, 1])
        self.assertEqual(staves_of_columns(lines[:8]), [0, 1])

    def test_the_lower_staff_split_too(self):
        lines = ["**kern\t**kern", "*^\t*", "*\t*\t*^"]
        self.assertEqual(staves_of_columns(lines), [0, 0, 1, 1])


class MoveDownFromTwoVoices(unittest.TestCase):
    def test_a_named_upper_column_goes_to_a_new_voice_after_the_lower_staff(self):
        text = corrected(UPPER_SPLIT, [override(6, move_down={"last_line": 7, "column": 1})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(line(text, 6), "*^\t*\t*")
        self.assertEqual(columns(text, 7), ["4C", "4cc", "4ryy", "4e"])
        self.assertEqual(columns(text, 8), ["4D", "4dd", "4ryy", "4f"])
        self.assertEqual(line(text, 9), "*v\t*v\t*\t*")

    def test_left_unnamed_it_takes_the_last_column(self):
        text = corrected(UPPER_SPLIT, [override(6, move_down={"last_line": 7})])
        self.assertEqual(columns(text, 7), ["4C", "4e", "4cc", "4ryy"])

    def test_only_the_listed_notes_of_that_column_move(self):
        page = UPPER_SPLIT.replace("4C\t4cc\t4e", "4C\t4cc\t4e 4g")
        text = corrected(page, [override(6, move_down={"last_line": 7, "column": 2, "notes": {"6": [0]}})])
        self.assertEqual(columns(text, 7), ["4C", "4e", "4cc", "4g"])
        self.assertEqual(columns(text, 8), ["4D", "4ryy", "4dd", "4f"])

    def test_the_moved_notes_are_read_on_the_lower_staff(self):
        before = parsed_staves(UPPER_SPLIT)
        after = parsed_staves(corrected(UPPER_SPLIT, [override(6, move_down={"last_line": 7, "column": 1})]))
        self.assertEqual(sorted(before[0]), ["C5", "D5", "E4", "F4", "G4"])
        self.assertEqual(sorted(after[0]), ["E4", "F4", "G4"])
        self.assertEqual(sorted(after[1]), sorted(before[1] + ["C5", "D5"]))

    def test_the_new_voice_follows_a_lower_staff_already_in_two(self):
        page = UPPER_SPLIT.replace("*\t*^\n4C\t4cc\t4e\n4D\t4dd\t4f\n*\t*v\t*v",
                                   "*^\t*\n4C\t4GG\t4cc\n4D\t4AA\t4dd\n*v\t*v\t*")
        text = corrected(page, [override(6, move_down={"last_line": 7})])
        self.assertTrue(spines_consistent(text))
        self.assertEqual(line(text, 6), "*\t*^\t*")
        self.assertEqual(columns(text, 7), ["4C", "4GG", "4cc", "4ryy"])

    def test_an_upper_clef_change_in_the_range_stays_on_the_upper_staff(self):
        page = UPPER_SPLIT.replace("4D\t4dd\t4f", "*\t*\t*clefF4\n4D\t4dd\t4f")
        text = corrected(page, [override(6, move_down={"last_line": 8, "column": 1})])
        self.assertEqual(columns(text, 8), ["*", "*", "*", "*clefF4"])

    def test_a_barline_in_the_range_runs_through_the_new_voice(self):
        page = UPPER_SPLIT.replace("4D\t4dd\t4f", "=1a\t=1a\t=1a\n4D\t4dd\t4f")
        text = corrected(page, [override(6, move_down={"last_line": 8, "column": 1})])
        self.assertEqual(columns(text, 8), ["=1a"] * 4)

    def test_a_note_held_across_the_split_stops_the_build(self):
        # the scorer's parser reads no parts at all from a page split under a sounding note
        page = UPPER_SPLIT.replace("4C\t4cc\t4e\n4D", "2C\t4cc\t4e\n.")
        with self.assertRaisesRegex(ValueError, "line 7: column 0 holds a note across the split"):
            corrected(page, [override(7, move_down={"last_line": 7})])

    def test_voices_split_inside_the_range_stop_the_build(self):
        with self.assertRaisesRegex(ValueError, "voices split or merged inside the range"):
            corrected(UPPER_SPLIT, [override(4, move_down={"last_line": 7})])


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
