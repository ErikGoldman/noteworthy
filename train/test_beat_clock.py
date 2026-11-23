import unittest
from fractions import Fraction

import torch

from seq.beat_clock import FEATURES, BeatClock, clock_features


def time_after(lmx: str) -> list[Fraction]:
    return [Fraction(f[2] * 4).limit_denominator(48) for f in clock_features(lmx.split())]


def left_after(lmx: str) -> list[Fraction]:
    return [Fraction(f[3] * 4).limit_denominator(48) for f in clock_features(lmx.split())]


class BeatClockTest(unittest.TestCase):
    def test_a_tuplet_ratio_shortens_the_note_it_follows(self):
        times = time_after("measure E4 voice:1 eighth 3in2 F4 eighth 3in2 G4 eighth 3in2")
        self.assertEqual(times[3:5], [Fraction(1, 2), Fraction(1, 3)])
        self.assertEqual(times[-1], 1)

    def test_chords_and_grace_notes_take_no_time(self):
        times = time_after("measure C4 voice:1 quarter chord E4 quarter grace D4 eighth E4 half")
        self.assertEqual(times[3], 1)
        self.assertEqual(times[6], 1)
        self.assertEqual(times[9], 1)
        self.assertEqual(times[-1], 3)

    def test_dots_count_as_soon_as_written(self):
        self.assertEqual(time_after("measure C4 voice:1 half dot")[-2:], [2, 3])

    def test_backups_go_back_and_backup_bar_returns_to_the_start(self):
        self.assertEqual(time_after("measure C4 voice:1 whole backup half")[-1], 2)
        self.assertEqual(time_after("measure C4 voice:1 whole backup:bar")[-1], 0)

    def test_measure_starts_the_clock_again(self):
        self.assertEqual(time_after("measure C4 voice:1 whole measure")[-1], 0)

    def test_time_left_needs_the_meter_and_goes_negative_past_the_bar(self):
        unknown = clock_features("measure C4 voice:1 whole".split())[-1]
        self.assertEqual(unknown[0], 0.0)
        lmx = "meter beats:3 beat-type:4 measure C4 voice:1 half D4 quarter E4 quarter"
        left = left_after(lmx)
        self.assertEqual([left[k] for k in (6, 8, 10)], [1, 0, -1])
        last = clock_features(lmx.split())[-1]
        self.assertEqual((last[4], last[5]), (1.0, 0.0))

    def test_a_time_signature_in_the_label_sets_the_bar(self):
        self.assertEqual(left_after("measure time beats:6 beat-type:8 C4 voice:1 quarter")[-1], 2)

    def test_a_whole_bar_rest_makes_the_clock_unknown(self):
        self.assertEqual(clock_features("meter beats:4 beat-type:4 measure rest:measure".split())[-1][1], 0.0)

    def test_ids_in_batches(self):
        names = ["<pad>", "measure", "C4", "quarter"]
        features = BeatClock(names)(torch.tensor([[1, 2, 3], [1, 0, 0]]))
        self.assertEqual(tuple(features.shape), (2, 3, FEATURES))
        self.assertAlmostEqual(features[0, 2, 2].item(), 0.25)


if __name__ == "__main__":
    unittest.main()
