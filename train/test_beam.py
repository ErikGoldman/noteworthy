"""Tests for beam search in seq.model's shared decoder. Run from train/: python -m unittest test_beam"""
from __future__ import annotations

import math
import unittest

import torch

from seq.model import _TokenDecoder
from seq.vocab import EOS

A, B, C, D = 4, 5, 6, 7
VOCAB = 8


class Scripted(_TokenDecoder):
    """A decoder whose next-token probabilities depend only on the tokens so far: `script` maps
    the sequence after <bos> to {token: probability}, `otherwise` any sequence it does not list
    (such as one a beam fills with a token of no probability)."""

    def __init__(self, script: dict[tuple, dict[int, float]], otherwise: dict[int, float] | None = None) -> None:
        super().__init__()
        self.script, self.otherwise, self.max_tokens = script, otherwise or {EOS: 1.0}, 16

    def decode(self, tokens, memory, padding):
        logits = torch.full((*tokens.shape, VOCAB), -1e9)
        for row, seq in enumerate(tokens.tolist()):
            for token, p in self.script.get(tuple(seq[1:]), self.otherwise).items():
                logits[row, -1, token] = math.log(p)
        return logits


# Greedy takes A (0.6), then has only poor choices; B (0.4) ends at once with certainty.
TRAP = {(): {A: 0.6, B: 0.4}, (A,): {C: 0.34, D: 0.33, EOS: 0.33}, (B,): {EOS: 1.0},
        (A, C): {EOS: 1.0}, (A, D): {EOS: 1.0}}
NO_MEMORY = (torch.zeros(1, 1, 1), torch.zeros(1, 1, dtype=torch.bool))


class BeamSearch(unittest.TestCase):
    def test_greedy_falls_into_the_trap(self):
        tokens = Scripted(TRAP)._generate_from(*NO_MEMORY, None, None)[0]
        self.assertEqual(tokens[:3], [A, C, EOS])

    def test_beam_finds_the_likelier_reading(self):
        self.assertEqual(Scripted(TRAP)._beam_from(*NO_MEMORY, None, [], 2), [B, EOS])

    def test_start_tokens_are_kept_and_not_scored(self):
        script = {(C,): {A: 0.9, B: 0.1}, (C, A): {EOS: 1.0}, (C, B): {EOS: 1.0}}
        self.assertEqual(Scripted(script)._beam_from(*NO_MEMORY, None, [C], 2), [C, A, EOS])

    def test_a_longer_reading_finishing_later_is_not_cut_off(self):
        # two short readings end first; the long one, likelier per token, ends after them (tuplet ratios, 2026-11-23)
        script = {(): {A: 0.4, B: 0.3, C: 0.3}, (B,): {EOS: 1.0}, (C,): {EOS: 1.0},
                  (A,): {A: 1.0}, (A, A): {A: 1.0}, (A, A, A): {EOS: 1.0}}
        self.assertEqual(Scripted(script)._beam_from(*NO_MEMORY, None, [], 2), [A, A, A, EOS])

    def test_stops_at_the_length_limit_with_the_best_unfinished_reading(self):
        self.assertEqual(Scripted({}, otherwise={A: 0.9, B: 0.1})._beam_from(*NO_MEMORY, 4, [], 2), [A, A, A])


if __name__ == "__main__":
    unittest.main()
