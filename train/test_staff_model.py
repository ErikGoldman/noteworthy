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


class HowFarTheStavesSee(unittest.TestCase):
    """The attention across staves alone, on a feature grid of two staves: what the lower staff
    takes from one column of the upper staff, by setting. (Through the whole encoder a change
    anywhere spreads: the CNN sees neighbouring columns, and attention within a staff runs first.)"""

    COLS = 8

    def lower_columns_changed(self, across_columns: int, width_columns: int = COLS) -> list[int]:
        model = small_model(across_columns=across_columns)
        g = torch.Generator().manual_seed(3)
        grid = torch.randn(1, 2, HEIGHT // 32, self.COLS, DIM, generator=g)
        changed = grid.clone()
        changed[0, 0, :, 3] += torch.randn(HEIGHT // 32, DIM, generator=g)  # not uniform: LayerNorm removes that
        blank = torch.tensor([[False, False]])
        past = torch.arange(self.COLS)[None, :] >= width_columns
        layer = model.across_staves[0]
        before = model.attend_across(layer, grid, blank, past)[0, 1]
        after = model.attend_across(layer, changed, blank, past)[0, 1]
        self.assertTrue(torch.isfinite(after).all())
        return [c for c in range(self.COLS) if not torch.allclose(before[:, c], after[:, c], atol=1e-6)]

    def test_one_column(self):
        self.assertEqual(self.lower_columns_changed(1), [3])

    def test_groups_of_columns(self):
        """Groups of three: columns 3, 4 and 5 attend across the staves together."""
        self.assertEqual(self.lower_columns_changed(3), [3, 4, 5])

    def test_none(self):
        self.assertEqual(len(small_model(across_columns=0).across_staves), 0)

    def test_meter_context_is_the_models_own(self):
        self.assertTrue(small_model(meter_context=True).meter_context)
        self.assertFalse(small_model().meter_context)

    def test_two_blocks_two_layers_across(self):
        self.assertEqual(len(small_model(blocks=2).across_staves), 2)

    def test_a_group_running_past_the_width(self):
        """Width 5 columns, groups of three: column 3's group (3, 4, 5) holds a padding column,
        which is masked; nothing becomes NaN, and the change still reaches 3 and 4."""
        self.assertEqual(self.lower_columns_changed(3, width_columns=5)[:2], [3, 4])

    def test_a_group_wholly_past_the_width_stays_finite(self):
        model = small_model(across_columns=3)
        grid = torch.randn(1, 2, HEIGHT // 32, self.COLS, DIM)
        out = model.attend_across(model.across_staves[0], grid, torch.tensor([[False, False]]),
                                  torch.arange(self.COLS)[None, :] >= 2)
        self.assertTrue(torch.isfinite(out).all())


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
