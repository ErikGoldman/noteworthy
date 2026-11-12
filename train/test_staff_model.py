"""Tests for seq.model.StaffSeq2Seq: shapes for any number of staves, padding that changes
nothing, staves that see each other only through the across-staves layers, a decoder that
cannot see later tokens, and a size matching Seq2Seq's. Run from train/: python -m unittest test_staff_model"""
from __future__ import annotations

import unittest

import torch

from seq.model import COL_STRIDE, Seq2Seq, StaffImages, StaffSeq2Seq

VOCAB, HEIGHT, DIM = 50, 64, 32  # small enough for the CPU; the real model is 240 tokens, 256 px a staff, 384 wide


def small_model(**kwargs) -> StaffSeq2Seq:
    torch.manual_seed(0)
    return StaffSeq2Seq(VOCAB, dim=DIM, heads=4, decoder_layers=2, max_tokens=64, staff_height=HEIGHT,
                        **kwargs).eval()


def staves(counts: list[int], width: int = 8 * COL_STRIDE, seed: int = 1) -> StaffImages:
    g = torch.Generator().manual_seed(seed)
    pixels = torch.rand(len(counts), max(counts), 1, HEIGHT, width, generator=g)
    for k, n in enumerate(counts):
        pixels[k, n:] = 0
    return StaffImages(pixels, torch.tensor(counts))


def tokens(batch: int, length: int = 10) -> torch.Tensor:
    return torch.randint(4, VOCAB, (batch, length), generator=torch.Generator().manual_seed(2))


class Shapes(unittest.TestCase):
    def test_systems_of_two_and_three_staves_in_one_batch(self):
        model, systems = small_model(), staves([2, 3])
        widths = torch.tensor([8 * COL_STRIDE, 5 * COL_STRIDE])
        memory, padding = model.encode(systems, widths)
        rows, cols = HEIGHT // 32, 8
        self.assertEqual(memory.shape, (2, 3 * rows * cols, DIM))
        self.assertEqual(padding.shape, (2, 3 * rows * cols))
        self.assertEqual(model(systems, widths, tokens(2)).shape, (2, 9, VOCAB))

    def test_padding_marks_blank_staves_and_columns_past_the_width(self):
        model = small_model()
        _, padding = model.encode(staves([2, 3]), torch.tensor([8 * COL_STRIDE, 5 * COL_STRIDE]))
        grid = padding.reshape(2, 3, HEIGHT // 32, 8)
        self.assertTrue(grid[0, 2].all())                                     # the 2-staff system's third staff
        self.assertFalse(grid[0, :2].any())
        self.assertTrue(grid[1, :, :, 5:].all() and not grid[1, :, :, :5].any())   # columns past width 5

    def test_generate(self):
        model = small_model()
        out = model.generate(staves([2, 3]), torch.tensor([8 * COL_STRIDE] * 2), max_tokens=6, starts=[[5], []])
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0][0], 5)


class Padding(unittest.TestCase):
    def test_a_blank_staff_changes_nothing(self):
        """A 2-staff system read alone, and padded to 3 staves beside a 3-staff one: same logits."""
        model = small_model()
        width = torch.tensor([8 * COL_STRIDE])
        alone = model(staves([2]), width, tokens(1))
        padded = staves([2, 3])
        padded.pixels[0, :2] = staves([2]).pixels[0]
        beside = model(padded, width.repeat(2), tokens(2))[:1]
        torch.testing.assert_close(beside, alone, atol=1e-5, rtol=1e-5)

    def test_what_a_blank_staff_holds_changes_nothing(self):
        model, systems, width = small_model(), staves([2, 3]), torch.tensor([8 * COL_STRIDE] * 2)
        before = model(systems, width, tokens(2))
        systems.pixels[0, 2] = torch.rand(1, HEIGHT, 8 * COL_STRIDE)
        torch.testing.assert_close(model(systems, width, tokens(2)), before)


class StavesSeeEachOther(unittest.TestCase):
    def setUp(self):
        self.model, self.width = small_model(), torch.tensor([8 * COL_STRIDE])
        self.systems = staves([2])
        self.changed = staves([2])
        self.changed.pixels[0, 0] = torch.rand(1, HEIGHT, 8 * COL_STRIDE, generator=torch.Generator().manual_seed(9))

    def test_not_before_the_across_staves_layers(self):
        grid, changed = self.model.staff_grid(self.systems), self.model.staff_grid(self.changed)
        self.assertFalse(torch.allclose(grid[0, 0], changed[0, 0]))
        torch.testing.assert_close(grid[0, 1], changed[0, 1])

    def test_after_them(self):
        memory, _ = self.model.encode(self.systems, self.width)
        changed, _ = self.model.encode(self.changed, self.width)
        per_staff = memory.shape[1] // 2
        self.assertFalse(torch.allclose(memory[0, per_staff:], changed[0, per_staff:]))


class Decoder(unittest.TestCase):
    def test_a_later_token_does_not_change_earlier_logits(self):
        model, systems, width = small_model(), staves([2]), torch.tensor([8 * COL_STRIDE])
        sequence = tokens(1)
        later = sequence.clone()
        later[0, 6] = (later[0, 6] + 1) % VOCAB
        a, b = model(systems, width, sequence), model(systems, width, later)
        torch.testing.assert_close(a[0, :6], b[0, :6])
        self.assertFalse(torch.allclose(a[0, 6:], b[0, 6:]))


class Size(unittest.TestCase):
    def test_as_many_parameters_as_seq2seq(self):
        """One block is Seq2Seq's two encoder layers; only the row and staff embeddings differ."""
        count = lambda m: sum(p.numel() for p in m.parameters())
        staff, system = StaffSeq2Seq(240), Seq2Seq(240, height=512)
        extra = 8 * 384 - (16 - 8) * 384   # 8 staff embeddings, 8 rows a staff against 16 a system
        self.assertEqual(count(staff) - count(system), extra)


if __name__ == "__main__":
    unittest.main()
