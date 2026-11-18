"""Tests for held_out_copies. Run from eval/ in .venv."""
import tempfile
from fractions import Fraction
import unittest
from pathlib import Path
from unittest import mock

import held_out_copies
from held_out_copies import (MIN_SHARED, MIN_SKELETON, MIN_UNIT_FRAGMENTS, features_of, grouped_units, is_unseen,
                             melody_matches, normal_title, on_beats, set_held_out)
from omr.held_out import is_held_out, score_key
from score_events import NoteEvent

NOTE = "<note><pitch><step>{}</step><octave>4</octave></pitch><duration>1</duration></note>"


def musicxml(steps: str) -> str:
    notes = "".join(NOTE.format(s) for s in steps)
    return (f"<score-partwise><part id='P1'><measure><attributes><divisions>1</divisions></attributes>"
            f"{notes}</measure></part></score-partwise>")


def grams(start: int, count: int) -> set[tuple[int, ...]]:
    return {(i,) for i in range(start, start + count)}


def features(long=(), short=(), skeleton=()) -> dict:
    return {"long": set(long), "short": set(short), "skeleton": set(skeleton)}


class Rules(unittest.TestCase):
    def test_runs16_takes_either_side_s_share(self):
        set_held_out([features(long=grams(0, 100))])
        self.assertEqual(melody_matches(features(long=grams(0, 2))), {0: (["runs16"], "runs16 0.02/1.00")})

    def test_count8_needs_enough_shared_fragments(self):
        set_held_out([features(short=grams(0, 1000))])
        unit = grams(0, MIN_SHARED - 1) | grams(5000, MIN_UNIT_FRAGMENTS)
        self.assertEqual(melody_matches(features(short=unit)), {})
        unit = grams(0, MIN_SHARED) | grams(5000, MIN_UNIT_FRAGMENTS)
        self.assertEqual(melody_matches(features(short=unit))[0][0], ["count8"])

    def test_count8_ignores_a_tiny_training_unit(self):
        set_held_out([features(short=grams(0, 1000))])
        self.assertEqual(melody_matches(features(short=grams(0, MIN_UNIT_FRAGMENTS - 1))), {})

    def test_skeleton_needs_both_sides_large_enough(self):
        set_held_out([features(skeleton=grams(0, MIN_SKELETON)), features(skeleton=grams(0, MIN_SKELETON - 1))])
        found = melody_matches(features(skeleton=grams(0, MIN_SKELETON)))
        self.assertEqual(list(found), [0])
        self.assertEqual(found[0], (["skeleton"], "skeleton 1.00/1.00"))

    def test_rules_that_fire_together_are_all_named(self):
        set_held_out([features(long=grams(0, 10), skeleton=grams(0, MIN_SKELETON))])
        rules, evidence = melody_matches(features(long=grams(0, 10), skeleton=grams(0, MIN_SKELETON)))[0]
        self.assertEqual(rules, ["runs16", "skeleton"])
        self.assertEqual(evidence, "runs16 1.00/1.00; skeleton 1.00/1.00")


class Skeleton(unittest.TestCase):
    def test_only_notes_starting_on_a_beat_are_kept(self):
        on, off = (NoteEvent(0, Fraction(1), 60, Fraction(1, 2), 1), NoteEvent(0, Fraction(3, 2), 62, Fraction(1, 2), 1))
        self.assertEqual(on_beats([on, off]), [on])


class Titles(unittest.TestCase):
    def test_arrangement_words_accents_and_asides_go(self):
        self.assertEqual(normal_title("Säkkijärven Polkka (Piano) arr. for solo piano"), "sakkijarven polkka")

    def test_numbers_stay(self):
        self.assertEqual(normal_title("Sonata No. 14, Op. 27"), "sonata 14 27")


class Units(unittest.TestCase):
    def test_samples_group_by_unit_in_name_order(self):
        units = grouped_units("grandstaff", Path("/r"), ["b/x/1", "a/y/2", "a/y/1"], lambda n: "/".join(n.split("/")[:2]))
        self.assertEqual(units, [("grandstaff", "a/y", [Path("/r/a/y/2.musicxml"), Path("/r/a/y/1.musicxml")]),
                                 ("grandstaff", "b/x", [Path("/r/b/x/1.musicxml")])])


class Features(unittest.TestCase):
    def test_a_transposed_copy_has_the_same_fragments(self):
        with tempfile.TemporaryDirectory() as d:
            low, high = Path(d, "low.musicxml"), Path(d, "high.musicxml")
            low.write_text(musicxml("CDEFGABAGFEDCDEFGAB"))
            high.write_text(musicxml("CDEFGABAGFEDCDEFGAB").replace("<octave>4</octave>", "<octave>5</octave>"))
            self.assertEqual(features_of([low]), features_of([high]))
            self.assertTrue(all(features_of([low]).values()))

    def test_a_score_too_short_for_a_fragment_says_so(self):
        with tempfile.TemporaryDirectory() as d:
            short = Path(d, "short.musicxml")
            short.write_text(musicxml("CDE"))
            self.assertEqual(features_of([short]), "no melody fragments")

    def test_an_unreadable_file_says_why(self):
        with tempfile.TemporaryDirectory() as d:
            bad = Path(d, "bad.musicxml")
            bad.write_text("<score-partwise>")
            self.assertTrue(features_of([bad]).startswith("ParseError"))


class Unseen(unittest.TestCase):
    def test_a_listed_held_out_score_is_not_unseen(self):
        with tempfile.TemporaryDirectory() as d:
            scores = [Path(d, f"{i}.mxl") for i in range(200)]
            for score in scores:
                score.write_text("score")
            listed, other = [s for s in scores if is_held_out(s)][:2]
            trained = next(s for s in scores if not is_held_out(s))
            copies = Path(d, "copies.txt")
            copies.write_text(f"{score_key(listed)}\traw/{listed.name}\tpdmx\traw/copy.mxl\truns16\truns16 0.50/0.40\n")
            with mock.patch.object(held_out_copies, "COPIES", copies):
                held_out_copies.keys_with_copies.cache_clear()
                self.assertFalse(is_unseen(listed))
                self.assertTrue(is_unseen(other))
                self.assertFalse(is_unseen(trained))
            held_out_copies.keys_with_copies.cache_clear()


class Review(unittest.TestCase):
    def test_the_list_is_the_pairs_the_review_judged_copies(self):
        rows = [line.split("\t") for line in held_out_copies.COPIES.with_name("held_out_copies_review.tsv")
                .read_text().splitlines()[1:]]
        self.assertTrue(all(len(row) == 8 and row[6] in ("copy", "false") and row[7] for row in rows))
        copies = [row[:6] for row in rows if row[6] == "copy"]
        self.assertEqual([line.split("\t") for line in held_out_copies.COPIES.read_text().splitlines()], copies)


if __name__ == "__main__":
    unittest.main()
