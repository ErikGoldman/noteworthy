"""Tests for seq.label_form. Run from train/: python -m unittest test_label_form"""
from __future__ import annotations

import unittest

from fractions import Fraction

from seq.label_form import (BACKUP_TO_BAR, LabelForm, spelled, voices_per_staff, with_backups_spelled,
                             with_canonical_gaps, with_voice_counts, without_voice_counts)
from seq.vocab import UNK, Vocab
from train_seq import VOCABULARY

# Crop 27 of OLiMPiC dev, bar 7: one voice on the upper staff, two on the lower.
TWO_BASS_VOICES = ("measure D4 voice:1 quarter stem:up staff:1 chord F4 quarter E4 eighth G4 eighth "
                   "backup half B3 voice:5 quarter stem:up staff:2 G3 eighth E3 eighth "
                   "backup half A3 voice:6 quarter stem:down staff:2 forward quarter")
ONE_EACH = "measure key:fifths:0 clef:G2 staff:1 C4 voice:1 half staff:1 backup half C3 voice:5 half staff:2"


class VoiceCounts(unittest.TestCase):
    def test_counts_each_staffs_voices(self):
        self.assertEqual(voices_per_staff(TWO_BASS_VOICES.split()[1:]), {1: 1, 2: 2})

    def test_a_whole_bar_rest_is_a_voice(self):
        self.assertEqual(voices_per_staff("forward voice:1 rest:measure staff:1".split()), {1: 1})

    def test_tokens_follow_each_measure_in_staff_order(self):
        both = with_voice_counts(f"{ONE_EACH} {TWO_BASS_VOICES}")
        self.assertTrue(both.startswith("measure voices:1 voices:1 key:fifths:0"))
        self.assertIn("measure voices:1 voices:2 D4", both)

    def test_round_trip(self):
        lmx = f"{ONE_EACH} {TWO_BASS_VOICES}"
        self.assertEqual(without_voice_counts(with_voice_counts(lmx)), lmx)

    def test_many_voices_share_the_cap(self):
        bar = "measure " + " backup half ".join(f"C4 voice:{v} half staff:1" for v in range(1, 7))
        self.assertTrue(with_voice_counts(bar).startswith("measure voices:4 C4"))

    def test_the_vocabulary_knows_the_tokens(self):
        ids = Vocab.load(VOCABULARY).encode(with_voice_counts(TWO_BASS_VOICES))
        self.assertNotIn(UNK, ids)


# A hidden rest that opens voice 5 on the lower staff, after backups spelled in two tokens (3/4).
HIDDEN_REST = ("measure E5 voice:1 quarter staff:1 C6 half backup half backup quarter "
               "print-object:no rest voice:5 quarter staff:2 C3 half")


class CanonicalGaps(unittest.TestCase):
    def test_a_backup_to_the_bars_start_is_one_token(self):
        self.assertEqual(with_canonical_gaps(TWO_BASS_VOICES).split().count(BACKUP_TO_BAR), 2)
        self.assertNotIn("backup half", with_canonical_gaps(TWO_BASS_VOICES))

    def test_a_hidden_rest_becomes_a_forward_and_hands_on_its_voice_and_staff(self):
        canonical = with_canonical_gaps(HIDDEN_REST)
        self.assertIn("backup:bar forward quarter C3 voice:5 half staff:2", canonical)
        self.assertNotIn("print-object:no", canonical)

    def test_a_run_of_forwards_is_spelled_one_way(self):
        bar = "measure C4 voice:1 half staff:1 backup half forward eighth forward 16th forward 16th D4 voice:2 quarter staff:1"
        self.assertIn("backup:bar forward quarter D4", with_canonical_gaps(bar))

    def test_a_backup_short_of_the_bars_start_stays(self):
        bar = "measure C4 voice:1 half staff:1 backup quarter D4 voice:2 quarter staff:1"
        self.assertEqual(with_canonical_gaps(bar), bar)

    def test_after_a_whole_bar_rest_backups_stay(self):
        bar = "measure rest voice:1 rest:measure staff:1 backup quarter clef:F4 staff:2 E4 voice:5 quarter staff:2 backup quarter rest voice:6 quarter staff:2"
        self.assertEqual(with_canonical_gaps(bar), bar)

    def test_a_hidden_whole_bar_rest_round_trips(self):
        """Its hidden head goes as a gap's would; reading writes it back whole (Polish 038, 047:
        a headless `rest:measure` became a note with neither pitch nor length)."""
        form = LabelForm(canonical_gaps=True)
        label = "measure C4 voice:1 half staff:1 D4 half backup whole print-object:no rest voice:5 rest:measure staff:2"
        self.assertIn("backup:bar rest:measure voice:5", form.label(label))
        self.assertEqual(form.reading(form.label(label)), label)

    def test_a_drawn_whole_bar_rest_is_left_alone(self):
        form = LabelForm(canonical_gaps=True)
        label = "measure C4 voice:1 whole staff:1 backup whole rest voice:5 rest:measure staff:2"
        self.assertEqual(form.reading(form.label(label)), label)

    def test_readings_get_their_backups_spelled_again(self):
        self.assertEqual(with_backups_spelled(with_canonical_gaps(TWO_BASS_VOICES)), TWO_BASS_VOICES)
        self.assertIn("backup half backup quarter forward quarter", with_backups_spelled(with_canonical_gaps(HIDDEN_REST)))

    def test_lengths_are_spelled_plain_first_then_with_triplets(self):
        self.assertEqual(spelled(Fraction(3), "forward"), "forward half forward quarter".split())
        self.assertEqual(spelled(Fraction(5, 3), "forward"), "forward quarter 3in2 forward quarter".split())
        self.assertIsNone(spelled(Fraction(1, 5), "forward"))

    def test_voices_are_counted_across_backups_to_the_bar(self):
        self.assertTrue(LabelForm(voice_counts=True, canonical_gaps=True).label(TWO_BASS_VOICES)
                        .startswith("measure voices:1 voices:2 D4"))

    def test_the_vocabulary_knows_backup_to_the_bar(self):
        self.assertNotIn(UNK, Vocab.load(VOCABULARY).encode(with_canonical_gaps(HIDDEN_REST)))


class Form(unittest.TestCase):
    def test_plain_form_leaves_labels_alone(self):
        self.assertEqual(LabelForm().label(TWO_BASS_VOICES), TWO_BASS_VOICES)

    def test_options_round_trip_through_a_checkpoints_args(self):
        form = LabelForm(voice_counts=True)
        self.assertEqual(LabelForm.from_options(form.options()), form)
        self.assertEqual(LabelForm.from_options({}), LabelForm())

    def test_both_options_read_back(self):
        form = LabelForm(voice_counts=True, canonical_gaps=True)
        self.assertEqual(form.reading(form.label(TWO_BASS_VOICES)), TWO_BASS_VOICES)

    def test_reading_takes_the_counts_out(self):
        form = LabelForm(voice_counts=True)
        self.assertEqual(form.reading(form.label(TWO_BASS_VOICES)), TWO_BASS_VOICES)


if __name__ == "__main__":
    unittest.main()
