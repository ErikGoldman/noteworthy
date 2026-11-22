"""Tests for seq.label_form. Run from train/: python -m unittest test_label_form"""
from __future__ import annotations

import unittest

from fractions import Fraction

from seq.label_form import (BACKUP_TO_BAR, LabelForm, UnplacedPosition, spelled, voices_per_staff,
                             with_backups_spelled, with_canonical_gaps, with_gaps_from_positions, with_positions,
                             with_clock, with_onsets, with_voice_counts, without_clock, without_onsets,
                             without_voice_counts, LANE_TOKENS, interleaving_failures, with_interleaving,
                             without_interleaving)
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



class Clock(unittest.TestCase):
    def clocks(self, lmx: str) -> list[str]:
        return [t for t in with_clock(lmx).split() if t.startswith("clock:")]

    def test_each_quarter_reached_is_marked_and_the_bar_ends_on_its_length(self):
        bar = "measure C4 quarter D4 eighth E4 eighth F4 half"
        self.assertEqual(with_clock(bar), "measure C4 quarter clock:1 D4 eighth E4 eighth clock:2 F4 half clock:4")

    def test_triplets_reach_the_beat_on_their_third_note(self):
        triplet = "C4 eighth 3in2 D4 eighth 3in2 E4 eighth 3in2"
        self.assertEqual(with_clock(f"measure {triplet}").split(), ["measure", *triplet.split(), "clock:1"])

    def test_the_same_notes_plain_reach_it_a_note_early(self):
        self.assertEqual(with_clock("measure C4 eighth D4 eighth E4 eighth").split()[:6],
                         ["measure", "C4", "eighth", "D4", "eighth", "clock:1"])

    def test_a_second_voice_counts_again_from_the_bars_start(self):
        bar = "measure C5 half backup half C3 quarter D3 quarter"
        self.assertEqual(self.clocks(bar), ["clock:2", "clock:1", "clock:2"])

    def test_chord_notes_and_graces_do_not_move_the_clock(self):
        self.assertEqual(self.clocks("measure C4 quarter chord E4 quarter grace D4 eighth F4 quarter"), ["clock:1", "clock:2"])

    def test_no_clock_after_a_whole_bar_rest(self):
        self.assertEqual(self.clocks("measure rest:measure rest whole backup whole C3 quarter"), [])

    def test_reading_takes_the_clock_out(self):
        form = LabelForm(clock=True)
        self.assertEqual(form.reading(form.label(TWO_BASS_VOICES)), TWO_BASS_VOICES)
        self.assertEqual(without_clock(with_clock(TWO_BASS_VOICES)), TWO_BASS_VOICES)

    def test_clock_tokens_are_in_the_vocabulary(self):
        vocab = Vocab.load(VOCABULARY)
        self.assertNotIn(UNK, vocab.encode(with_clock("measure C4 whole measure D4 half D4 half")))

    def test_option_round_trips_through_a_checkpoints_args(self):
        self.assertEqual(LabelForm.from_options(LabelForm(clock=True).options()), LabelForm(clock=True))


class Onsets(unittest.TestCase):
    def onsets(self, lmx: str) -> list[str]:
        tokens = with_onsets(lmx).split()
        return [f"{a} {b}" for a, b in zip(tokens, tokens[1:]) if a.startswith("beat:")]

    def test_each_note_is_preceded_by_where_it_starts(self):
        self.assertEqual(with_onsets("measure clef:G2 C4 quarter D4 eighth E4 eighth"),
                         "measure clef:G2 beat:0 sub:0 C4 quarter beat:1 sub:0 D4 eighth beat:1 sub:1/2 E4 eighth")

    def test_triplets_start_on_thirds(self):
        triplet = "measure C4 eighth 3in2 D4 eighth 3in2 E4 eighth 3in2 F4 quarter"
        self.assertEqual(self.onsets(triplet), ["beat:0 sub:0", "beat:0 sub:1/3", "beat:0 sub:2/3", "beat:1 sub:0"])

    def test_chord_members_and_graces_take_none(self):
        self.assertEqual(self.onsets("measure C4 quarter chord E4 quarter grace D4 eighth F4 quarter"),
                         ["beat:0 sub:0", "beat:1 sub:0"])

    def test_a_second_voice_starts_again_from_the_bars_start(self):
        self.assertEqual(self.onsets("measure C5 half backup half C3 quarter dot D3 eighth"),
                         ["beat:0 sub:0", "beat:0 sub:0", "beat:1 sub:1/2"])

    def test_rests_and_gaps_have_onsets_too(self):
        self.assertEqual(self.onsets("measure print-object:no rest half forward half"), ["beat:0 sub:0", "beat:2 sub:0"])

    def test_none_after_a_whole_bar_rest_or_off_the_grid(self):
        self.assertEqual(self.onsets("measure rest:measure rest whole backup whole C3 quarter"), ["beat:0 sub:0"])
        quintuplet = " ".join(f"C4 16th 5in4" for _ in range(5))
        self.assertEqual(self.onsets(f"measure {quintuplet}"), ["beat:0 sub:0"])

    def test_reading_takes_the_onsets_out(self):
        for form in (LabelForm(onsets=True), LabelForm(onsets=True, clock=True)):
            self.assertEqual(form.reading(form.label(TWO_BASS_VOICES)), TWO_BASS_VOICES)
        self.assertEqual(without_onsets(with_onsets(TWO_BASS_VOICES)), TWO_BASS_VOICES)

    def test_onset_tokens_are_in_the_vocabulary(self):
        vocab = Vocab.load(VOCABULARY)
        bar = "measure C4 eighth 3in2 D4 eighth 3in2 E4 eighth 3in2 F4 16th G4 16th A4 eighth B4 half"
        self.assertNotIn(UNK, vocab.encode(with_onsets(bar)))

    def test_option_round_trips_through_a_checkpoints_args(self):
        self.assertEqual(LabelForm.from_options(LabelForm(onsets=True).options()), LabelForm(onsets=True))


class Positions(unittest.TestCase):
    def test_each_backup_and_closing_gap_becomes_where_the_next_event_starts(self):
        self.assertEqual(with_positions(TWO_BASS_VOICES),
                         "measure D4 voice:1 quarter stem:up staff:1 chord F4 quarter E4 eighth G4 eighth "
                         "at:0 B3 voice:5 quarter stem:up staff:2 G3 eighth E3 eighth "
                         "at:0 A3 voice:6 quarter stem:down staff:2 at:2")

    def test_written_out_again_as_backups_and_forwards(self):
        self.assertEqual(with_gaps_from_positions(with_positions(TWO_BASS_VOICES)), (TWO_BASS_VOICES, []))

    def test_backups_and_a_hidden_rest_are_one_position_that_hands_on_voice_and_staff(self):
        positioned = with_positions(HIDDEN_REST)
        self.assertIn("C6 half at:1 C3 voice:5 half staff:2", positioned)
        self.assertIn("C6 half backup half C3 voice:5 half staff:2", with_gaps_from_positions(positioned)[0])

    def test_a_position_behind_its_voice_going_on_is_no_gap_and_reported(self):
        self.assertEqual(with_gaps_from_positions("measure C4 voice:1 half at:0 D4 quarter"),
                         ("measure C4 voice:1 half D4 quarter", [UnplacedPosition(0, "voice:1", Fraction(0), Fraction(2))]))

    def test_a_voice_backing_up_into_itself_keeps_its_backup(self):
        bar = "measure C4 voice:1 half backup quarter D4 quarter"
        self.assertEqual(with_positions(bar), bar)

    def test_a_bar_of_unknown_time_keeps_its_backup(self):
        bar = "measure print-object:no rest voice:1 rest:measure backup whole C3 voice:5 whole"
        self.assertEqual(with_positions(bar), bar)

    def test_a_dotted_hidden_rest_is_a_gap_too(self):
        self.assertEqual(with_positions("measure print-object:no rest voice:1 quarter dot C4 eighth"),
                         "measure at:3/2 C4 voice:1 eighth")



class Interleaved(unittest.TestCase):
    TWO_HANDS = ("measure clef:G2 staff:1 C4 voice:1 half stem:up staff:1 D4 half backup whole "
                 "E3 voice:5 quarter staff:2 F3 quarter G3 half")

    def test_voices_come_in_time_order(self):
        form = LabelForm(canonical_gaps=True, interleaved=True)
        self.assertEqual(form.label(self.TWO_HANDS),
                         "measure clef:G2 staff:1 C4 voice:1 half stem:up staff:1 lane:2 E3 voice:5 quarter staff:2 "
                         "F3 quarter lane:1 D4 half lane:2 G3 half")

    def test_reading_gives_the_canonical_form_back(self):
        canonical = LabelForm(canonical_gaps=True)
        form = LabelForm(canonical_gaps=True, interleaved=True)
        self.assertEqual(form.reading(form.label(self.TWO_HANDS)), canonical.reading(canonical.label(self.TWO_HANDS)))

    def test_a_chord_stays_after_its_head(self):
        lmx = "measure C4 voice:1 half chord E4 half D4 half backup whole G3 voice:5 whole staff:2"
        self.assertIn("C4 voice:1 half chord E4 half", with_interleaving(with_canonical_gaps(lmx)))

    def test_a_voice_coming_in_partway_keeps_its_backup(self):
        lmx = "measure C4 voice:1 half D4 half backup half E4 voice:2 half"
        canonical = with_canonical_gaps(lmx)
        interleaved = with_interleaving(canonical)
        self.assertEqual(interleaving_failures(canonical), 0)
        self.assertEqual(interleaved, "measure C4 voice:1 half D4 half lane:2 backup half E4 voice:2 half")
        self.assertEqual(without_interleaving(interleaved), canonical)

    def test_a_grace_chord_stays_after_its_grace_note(self):
        lmx = ("measure C5 voice:1 half D5 half backup whole grace grace:slash B1 eighth staff:2 "
               "grace grace:slash chord B2 eighth B3 voice:5 whole")
        canonical = with_canonical_gaps(lmx)
        self.assertEqual(without_interleaving(with_interleaving(canonical)), canonical)

    def test_needs_canonical_gaps(self):
        with self.assertRaises(ValueError):
            LabelForm(interleaved=True)

    def test_lane_tokens_are_in_the_vocabulary(self):
        vocab = Vocab.load(VOCABULARY)
        self.assertTrue(all(token in vocab.index for token in LANE_TOKENS))


if __name__ == "__main__":
    unittest.main()
