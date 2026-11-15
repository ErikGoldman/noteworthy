"""Tests for train_seq.with_own_readings: the decoder's input with some label tokens replaced by
its own sampled reading (scheduled sampling). Run from train/: python -m unittest test_own_readings"""
from __future__ import annotations

import unittest

import torch

from seq.meter import with_meter
from seq.vocab import BOS, PAD, Vocab
from train_seq import VOCABULARY, own_readings_share, with_own_readings

BAR = "measure key:fifths:0 clef:G2 beats:3 beat-type:4 staff:1 C4 half staff:1 D4 quarter"


class ReadsOneToken:
    """A stand-in decoder that is sure every next token is `token`."""

    def __init__(self, token: int, vocab_size: int) -> None:
        self.token, self.vocab_size = token, vocab_size

    def decode(self, inputs, memory, padding):
        logits = torch.full((*inputs.shape, self.vocab_size), -1e9)
        logits[..., self.token] = 0.0
        return logits


class OwnReadings(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = Vocab.load(VOCABULARY)
        self.guess = self.vocab.index["E4"]
        self.model = ReadsOneToken(self.guess, len(self.vocab))

    def batch(self, *labels: str) -> torch.Tensor:
        encoded = [self.vocab.encode(label) for label in labels]
        longest = max(map(len, encoded))
        return torch.tensor([e + [PAD] * (longest - len(e)) for e in encoded])

    def mixed(self, tokens: torch.Tensor, share: float) -> torch.Tensor:
        return with_own_readings(self.model, None, None, tokens, self.vocab, share)

    def test_nothing_replaced_at_share_zero(self):
        tokens = self.batch(BAR)
        self.assertTrue(torch.equal(self.mixed(tokens, 0.0), tokens[:, :-1]))

    def test_every_label_token_replaced_at_share_one(self):
        inputs = self.mixed(self.batch(BAR), 1.0)
        self.assertEqual(inputs[0, 0].item(), BOS)
        self.assertTrue((inputs[0, 1:] == self.guess).all())

    def test_bos_meter_prefix_and_padding_kept(self):
        tokens = self.batch(with_meter(BAR), BAR)
        inputs = self.mixed(tokens, 1.0)
        self.assertTrue(torch.equal(inputs[:, 0], tokens[:, 0]))
        self.assertTrue(torch.equal(inputs[0, 1:4], tokens[0, 1:4]))
        self.assertTrue((inputs[0, 4:] == self.guess).all())
        padded = tokens[1, :-1] == PAD
        self.assertTrue(padded.any())
        self.assertTrue((inputs[1][padded] == PAD).all())

    def test_share_rises_over_the_first_half(self):
        self.assertEqual(own_readings_share(0, 100, 0.2), 0.0)
        self.assertAlmostEqual(own_readings_share(25, 100, 0.2), 0.1)
        self.assertEqual(own_readings_share(50, 100, 0.2), 0.2)
        self.assertEqual(own_readings_share(90, 100, 0.2), 0.2)


if __name__ == "__main__":
    unittest.main()
