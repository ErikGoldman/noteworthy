"""Tests for seq.model.Seq2Seq's relative rows: no embedding of where a row is, a bias by the
distance between rows in its place, and padding still kept out. Run from train/: python -m
unittest test_seq_model"""
from __future__ import annotations

import unittest

import torch

from seq.model import COL_STRIDE, Seq2Seq, SystemImages, _down_columns

VOCAB, HEIGHT = 50, 128


def model(**kwargs) -> Seq2Seq:
    torch.manual_seed(0)
    return Seq2Seq(VOCAB, dim=32, heads=4, decoder_layers=2, max_tokens=32, height=HEIGHT, **kwargs).eval()


def images(width: int = 8 * COL_STRIDE) -> torch.Tensor:
    return torch.rand(2, 1, HEIGHT, width, generator=torch.Generator().manual_seed(1))


def systems(height: int, centres: list[list[float]], interline: float = 17.0, width: int = 4 * COL_STRIDE,
            pixels: torch.Tensor | None = None) -> SystemImages:
    """A batch of systems at staff scale, the staves the centres give (rows of fewer padded with 0)."""
    most = max(map(len, centres))
    return SystemImages(torch.rand(len(centres), 1, height, width, generator=torch.Generator().manual_seed(1))
                        if pixels is None else pixels,
                        torch.tensor([c + [0.0] * (most - len(c)) for c in centres]),
                        torch.tensor([len(c) for c in centres]), torch.full((len(centres),), interline))


class RelativeRows(unittest.TestCase):
    def test_no_embedding_of_where_a_row_is(self):
        names = {n for n, _ in model(relative_rows=True).named_parameters()}
        self.assertFalse(any(n.startswith("row_embedding") for n in names))
        self.assertIn("row_embedding", {n for n, _ in model().named_parameters()})

    def test_scores_depend_on_rows_apart_only(self):
        """A query on row 2 against a key on row 5 scores as on rows 6 and 9: rotation by row."""
        layer = model(relative_rows=True).row_layers[0]
        g = torch.Generator().manual_seed(3)
        q, k = torch.randn(1, 4, 1, 8, generator=g), torch.randn(1, 4, 1, 8, generator=g)
        score = lambda qr, kr: (layer.rotate(q, torch.tensor([qr])) * layer.rotate(k, torch.tensor([kr]))).sum()
        torch.testing.assert_close(score(2, 5), score(6, 9))
        self.assertFalse(torch.allclose(score(2, 5), score(2, 6)))

    def test_it_learns(self):
        m = model(relative_rows=True).train()
        memory, _ = m.encode(images(), torch.tensor([8 * COL_STRIDE] * 2))
        memory.sum().backward()
        self.assertGreater(m.row_layers[0].qkv.weight.grad.abs().sum().item(), 0)

    def test_padding_changes_nothing(self):
        """Columns past a crop's width are kept out, as without relative rows."""
        m = model(relative_rows=True)
        a, b = images(), images()
        b[:, :, :, 5 * COL_STRIDE:] = torch.rand(2, 1, HEIGHT, 3 * COL_STRIDE)
        widths = torch.tensor([5 * COL_STRIDE] * 2)
        (ma, pad), (mb, _) = m.encode(a, widths), m.encode(b, widths)
        # The CNN's reach crosses the width a little (about 2e-5 at column 0, with or without
        # relative rows); padding the attention let through would move the first columns by 0.1s.
        rows = HEIGHT // 32
        keep = torch.arange(rows * 8).reshape(rows, 8)[:, :3].flatten()
        torch.testing.assert_close(ma[:, keep], mb[:, keep], atol=1e-3, rtol=0)
        self.assertTrue(torch.isfinite(ma[~pad]).all())

    def test_under_bf16_autocast(self):
        m = model(relative_rows=True)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            memory, _ = m.encode(images(), torch.tensor([8 * COL_STRIDE] * 2))
        self.assertTrue(torch.isfinite(memory.float()).all())


class RowWindows(unittest.TestCase):
    """The windowed encoder: bands of rows, shifted bands, then columns."""

    ROWS, COLS = 8, 6

    def setUp(self):
        self.model = model(relative_rows=True, row_windows=4)
        g = torch.Generator().manual_seed(4)
        self.grid = torch.randn(1, self.ROWS, self.COLS, 32, generator=g)
        self.changed = self.grid.clone()
        self.changed[0, 1, 2] += torch.randn(32, generator=g)      # row 1, column 2
        self.past = torch.zeros(1, self.COLS, dtype=torch.bool)

    def rows_changed(self, out_a, out_b) -> list[int]:
        return [r for r in range(self.ROWS) if not torch.allclose(out_a[0, r], out_b[0, r], atol=1e-6)]

    def test_a_band_keeps_to_itself(self):
        """Bands of 4 rows: row 1 reaches rows 0-3, not 4-7."""
        layer = self.model.row_layers[0]
        a = self.model.attend_in_bands(layer, self.grid, self.past, shift=0)
        b = self.model.attend_in_bands(layer, self.changed, self.past, shift=0)
        self.assertEqual(self.rows_changed(a, b), [0, 1, 2, 3])

    def test_shifted_bands_straddle_the_edges(self):
        """Shifted by 2: the first band is rows 0-1, the next 2-5; row 1 reaches rows 0-1 only."""
        layer = self.model.row_layers[1]
        a = self.model.attend_in_bands(layer, self.grid, self.past, shift=2)
        b = self.model.attend_in_bands(layer, self.changed, self.past, shift=2)
        self.assertEqual(self.rows_changed(a, b), [0, 1])

    def test_a_column_reaches_every_row_of_it_only(self):
        layer = self.model.row_layers[2]
        run = lambda x, m: layer(x, torch.arange(x.shape[1]), m)
        a, b = _down_columns(self.grid, self.past, run), _down_columns(self.changed, self.past, run)
        self.assertEqual(self.rows_changed(a[:, :, 2:3], b[:, :, 2:3]), list(range(self.ROWS)))
        torch.testing.assert_close(a[:, :, [0, 1, 3, 4, 5]], b[:, :, [0, 1, 3, 4, 5]])

    def test_any_height_any_width(self):
        """Crops of different heights (a multiple of 32) read end to end, padding past widths kept out."""
        m = model(relative_rows=True, row_windows=4)
        for height in (128, 224):
            images = torch.rand(2, 1, height, 8 * COL_STRIDE, generator=torch.Generator().manual_seed(height))
            memory, padding = m.encode(images, torch.tensor([8 * COL_STRIDE, 5 * COL_STRIDE]))
            self.assertTrue(torch.isfinite(memory).all())
            self.assertEqual(memory.shape[1], height // 32 * 8)
        self.assertEqual(len(m.generate(images, torch.tensor([8 * COL_STRIDE] * 2), max_tokens=6)), 2)

    def test_absolute_rows_at_staff_scale_read_any_height(self):
        """Interline scaling with today's attention: a row embedding as tall as the tallest crop."""
        from seq.staff_data import MAX_SYSTEM_HEIGHT
        torch.manual_seed(0)
        m = Seq2Seq(VOCAB, dim=32, heads=4, decoder_layers=2, max_tokens=32, height=MAX_SYSTEM_HEIGHT,
                    interline_scaled=True).eval()
        self.assertEqual(len(m.row_embedding), MAX_SYSTEM_HEIGHT // 32)
        for height in (128, 384, MAX_SYSTEM_HEIGHT):
            memory, _ = m.encode(systems(height, [[height / 2]]), torch.tensor([4 * COL_STRIDE]))
            self.assertEqual(memory.shape[1], height // 32 * 4)

    def test_needs_relative_rows(self):
        with self.assertRaises(ValueError):
            model(row_windows=4)


if __name__ == "__main__":
    unittest.main()


class StaffFrame(unittest.TestCase):
    """staff_anchored: a row's embedding says which staff it is nearest and how far from its
    centre, not where it is in the crop; staff_bands: attention within each staff's rows first."""

    def anchored(self, **kwargs) -> Seq2Seq:
        from seq.staff_data import MAX_SYSTEM_HEIGHT
        torch.manual_seed(0)
        return Seq2Seq(VOCAB, dim=32, heads=4, decoder_layers=2, max_tokens=32, height=MAX_SYSTEM_HEIGHT,
                       interline_scaled=True, staff_anchored=True, **kwargs).eval()

    def test_rows_go_to_their_nearest_staff(self):
        nearest, offset = self.anchored().staff_frame(systems(256, [[80.0, 176.0]], interline=16.0), 8)
        self.assertEqual(nearest[0].tolist(), [0, 0, 0, 0, 1, 1, 1, 1])     # row centres 16, 48, ... 240
        torch.testing.assert_close(offset[0, :2], torch.tensor([-4.0, -2.0]))
        torch.testing.assert_close(offset[0, 5:7], torch.tensor([0.0, 2.0]))

    def test_a_staff_moved_by_a_row_keeps_its_rows_embedding(self):
        m = self.anchored()
        here = m.frame_embedding(*m.staff_frame(systems(256, [[80.0, 176.0]]), 8))
        lower = m.frame_embedding(*m.staff_frame(systems(256, [[112.0, 208.0]]), 8))
        torch.testing.assert_close(here[0, :6], lower[0, 1:7])

    def test_a_staff_the_system_lacks_owns_no_rows(self):
        nearest, _ = self.anchored().staff_frame(systems(256, [[80.0, 176.0], [128.0]]), 8)
        self.assertEqual(set(nearest[1].tolist()), {0})

    def test_no_absolute_row_embedding(self):
        names = {n for n, _ in self.anchored().named_parameters()}
        self.assertFalse(any(n.startswith("row_embedding") for n in names))

    def test_a_band_sees_its_own_staff_only(self):
        m = self.anchored(staff_bands=True)
        images = systems(256, [[80.0, 176.0]])
        nearest, _ = m.staff_frame(images, 8)
        grid = torch.rand(1, 8, 4, 32, generator=torch.Generator().manual_seed(2))
        changed = grid.clone()
        changed[0, 6] += 1.0                                   # a row of the lower staff
        past = torch.zeros(1, 4, dtype=torch.bool)
        a, b = m.attend_in_staff_bands(grid, nearest, past), m.attend_in_staff_bands(changed, nearest, past)
        torch.testing.assert_close(a[0, :4], b[0, :4])          # the upper staff's rows
        self.assertFalse(torch.allclose(a[0, 4:], b[0, 4:]))

    def test_bands_of_systems_of_different_staves_read_end_to_end(self):
        m = self.anchored(staff_bands=True)
        width = 6 * COL_STRIDE
        images = systems(384, [[96.0, 192.0, 288.0], [130.0, 250.0]], width=width)
        memory, padding = m.encode(images, torch.tensor([width, 3 * COL_STRIDE]))
        self.assertFalse(memory.isnan().any())
        self.assertEqual(memory.shape[1], 12 * 6)
        out = m.generate(images, torch.tensor([width, 3 * COL_STRIDE]), max_tokens=5)
        self.assertEqual(len(out), 2)

    def windows(self, centres: list[list[float]], grid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        m = self.anchored(staff_windows=True)
        images = systems(grid.shape[1] * 32, centres, interline=256 / 15)
        return m.encode_staff_windows(grid, images, torch.zeros(len(centres), grid.shape[2], dtype=torch.bool))

    def test_windows_hold_a_staff_crops_rows_and_overlap(self):
        """At staff scale a window is 15 interlines, 8 feature rows; staves 160 px apart (5 rows)
        share 3 rows, and a staff 64 px from the top has 2 of its window's rows above the system."""
        grid = torch.rand(1, 14, 3, 32, generator=torch.Generator().manual_seed(2))
        memory, padding = self.windows([[64.0, 224.0]], grid)
        self.assertEqual(memory.shape, (1, 2 * 8 * 3, 32))
        rows_padded = padding.reshape(2, 8, 3)[:, :, 0]
        self.assertEqual(rows_padded[0].tolist(), [True, True] + [False] * 6)
        self.assertFalse(rows_padded[1].any())
        self.assertFalse(memory.isnan().any())

    def test_a_system_moved_by_a_row_reads_the_same(self):
        """Each row is embedded by its offset from its window's own staff, so the system moved a
        feature row down (its grid with it) gives the same memory."""
        grid = torch.rand(1, 14, 3, 32, generator=torch.Generator().manual_seed(2))
        lower = torch.cat([torch.rand(1, 1, 3, 32), grid[:, :-1]], dim=1)
        here, _ = self.windows([[128.0, 288.0]], grid)
        moved, _ = self.windows([[160.0, 320.0]], lower)
        torch.testing.assert_close(here, moved)

    def test_windows_of_systems_of_different_staves_read_end_to_end(self):
        m = self.anchored(staff_windows=True)
        width = 6 * COL_STRIDE
        images = systems(384, [[96.0, 192.0, 288.0], [130.0, 250.0]], interline=256 / 15, width=width)
        memory, padding = m.encode(images, torch.tensor([width, 3 * COL_STRIDE]))
        self.assertFalse(memory.isnan().any())
        self.assertEqual(memory.shape[1], 3 * 8 * 6)
        self.assertTrue(padding.reshape(2, 3, 8, 6)[1, 2].all())      # the staff the second lacks
        self.assertEqual(len(m.generate(images, torch.tensor([width, 3 * COL_STRIDE]), max_tokens=5)), 2)

    def test_windows_need_the_frame_and_replace_bands(self):
        with self.assertRaises(ValueError):
            model(staff_windows=True, interline_scaled=True)
        with self.assertRaises(ValueError):
            self.anchored(staff_windows=True, staff_bands=True)

    def test_needs_interline_scale_and_its_own_rows(self):
        with self.assertRaises(ValueError):
            model(staff_anchored=True)
        with self.assertRaises(ValueError):
            model(staff_bands=True, interline_scaled=True)
