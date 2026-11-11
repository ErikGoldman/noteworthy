"""Tests for seq.meter and the meter prefix in training and decoding. Run from train/:
python -m unittest test_meter"""
from __future__ import annotations

import random
import unittest
from fractions import Fraction

import torch

from seq.meter import METER, meter_tokens, with_meter, without_meter
from seq.model import Seq2Seq
from seq.vocab import PAD, Vocab
from train_seq import VOCABULARY, meter_starts, targets_of

THREE_FOUR_BAR = "measure key:fifths:0 clef:G2 staff:1 C4 voice:1 half stem:up staff:1 D4 quarter stem:up staff:1"


class MeterTest(unittest.TestCase):
    def test_bar_lengths_spelled_with_time_signature_tokens(self):
        self.assertEqual(meter_tokens(Fraction(3)), [METER, "beats:3", "beat-type:4"])
        self.assertEqual(meter_tokens(Fraction(3, 2)), [METER, "beats:3", "beat-type:8"])
        self.assertEqual(meter_tokens(Fraction(9, 4)), [METER, "beats:9", "beat-type:16"])
        self.assertEqual(meter_tokens(Fraction(1, 3)), [])

    def test_prefix_from_the_label_and_off_again(self):
        prefixed = with_meter(THREE_FOUR_BAR)
        self.assertTrue(prefixed.startswith("meter beats:3 beat-type:4 measure"))
        self.assertEqual(without_meter(prefixed), THREE_FOUR_BAR)
        self.assertEqual(without_meter(THREE_FOUR_BAR), THREE_FOUR_BAR)

    def test_training_leaves_some_labels_without(self):
        rng = random.Random(0)
        kept = sum(with_meter(THREE_FOUR_BAR, rng).startswith(METER) for _ in range(1000))
        self.assertTrue(800 < kept < 900)

    def test_prefix_is_given_not_predicted(self):
        vocab = Vocab.load(VOCABULARY)
        tokens = torch.tensor([vocab.encode(with_meter(THREE_FOUR_BAR)), vocab.encode(THREE_FOUR_BAR) + [PAD] * 3])
        targets = targets_of(tokens, vocab)
        self.assertEqual(targets[0, :3].tolist(), [PAD] * 3)
        self.assertEqual(targets[0, 3:].tolist(), tokens[0, 4:].tolist())
        self.assertEqual(targets[1].tolist(), tokens[1, 1:].tolist())
        self.assertEqual(meter_starts(tokens, vocab), [tokens[0, 1:4].tolist(), []])

    def test_generation_starts_after_the_prefix_and_keeps_the_order(self):
        torch.manual_seed(0)
        model = Seq2Seq(40, dim=32, encoder_layers=1, decoder_layers=1, heads=2, max_tokens=12, height=64).eval()
        images, widths = torch.rand(3, 1, 64, 96), torch.tensor([96, 96, 96])
        starts = [[5, 6, 7], [], [8, 9, 10]]
        out = model.generate(images, widths, starts=starts)
        self.assertEqual([o[:3] for o in (out[0], out[2])], [[5, 6, 7], [8, 9, 10]])
        alone = model.generate(images[1:2], widths[1:2])
        self.assertEqual(out[1], alone[0])


if __name__ == "__main__":
    unittest.main()
