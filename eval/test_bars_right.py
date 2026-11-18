"""Tests for bars_right's alignment walk. Run from eval/ in .venv-musicdiff."""
import unittest

from bars_right import charged_bars, reading_to_gold


def edit(op: str, gold: int | None = None, pred: int | None = None, part: int = 0) -> dict:
    return {"op": op, **({"gold": {"part": part, "bar": gold}} if gold is not None else {}),
            **({"pred": {"part": part, "bar": pred}} if pred is not None else {})}


class Walk(unittest.TestCase):
    def test_bars_pair_in_order_when_none_is_missing(self):
        self.assertEqual(reading_to_gold(3, set(), set()), {0: 0, 1: 1, 2: 2})

    def test_a_gold_bar_the_reading_lacks_is_skipped(self):
        self.assertEqual(reading_to_gold(4, {1}, set()), {0: 0, 1: 2, 2: 3})

    def test_a_reading_bar_the_gold_lacks_goes_to_the_gold_bar_before_it(self):
        self.assertEqual(reading_to_gold(3, set(), {1}), {0: 0, 1: 0, 2: 1, 3: 2})

    def test_an_extra_bar_at_the_start_goes_to_the_first(self):
        self.assertEqual(reading_to_gold(2, set(), {0}), {0: 0, 1: 0, 2: 1})

    def test_an_extra_bar_at_the_end(self):
        self.assertEqual(reading_to_gold(2, set(), {2}), {0: 0, 1: 1, 2: 1})


class Charged(unittest.TestCase):
    def test_a_note_only_the_reading_has_charges_its_aligned_gold_bar(self):
        page = {"edits": [edit("insbar", gold=0), edit("notedel", pred=1)]}
        self.assertEqual(charged_bars(page, [3]), ({(0, 0), (0, 2)}, 0))

    def test_an_edit_naming_both_sides_checks_the_walk(self):
        page = {"edits": [edit("pitchnameedit", gold=1, pred=1), edit("insbar", gold=0)]}
        self.assertEqual(charged_bars(page, [3])[1], 1)

    def test_staves_walk_apart(self):
        page = {"edits": [edit("delbar", pred=0, part=1), edit("notedel", pred=1, part=1), edit("notedel", pred=1, part=0)]}
        self.assertEqual(charged_bars(page, [2, 2])[0], {(1, 0), (0, 1)})


if __name__ == "__main__":
    unittest.main()
