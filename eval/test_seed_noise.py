"""Tests for seed_noise's statistics. Run from eval/: python -m unittest test_seed_noise"""
from __future__ import annotations

import random
import unittest

from seed_noise import Form, excess_of


def form(scores: list[dict[str, float]], name: str = "f") -> Form:
    return Form(name, [f"seed{k}" for k in range(len(scores))], scores)


def noisy_seeds(rng: random.Random, units: int, seeds: int, page_noise: float, every_unit: float) -> list[dict[str, float]]:
    """Each unit's true score, plus each seed's own noise on every unit, plus a shift of that
    seed's whole set."""
    truth = [rng.uniform(0, 20) for _ in range(units)]
    out = []
    for _ in range(seeds):
        shift = rng.gauss(0, every_unit)
        out.append({f"u{k}": t + shift + rng.gauss(0, page_noise) for k, t in enumerate(truth)})
    return out


class PageByPage(unittest.TestCase):
    def test_one_unit_differing(self):
        """Three seeds equal but on one of four units (0, 3, 6): its variance 9, over 4 squared."""
        f = form([{"a": 1, "b": 2, "c": 3, "d": x} for x in (0, 3, 6)])
        self.assertAlmostEqual(f.page_by_page_variance(), 9 / 16)
        self.assertAlmostEqual(excess_of([f]).ratio, 1.0)  # the seed means differ by that unit alone

    def test_resampled_standard_error(self):
        """Differences 1, 3 over two units: their standard deviation over the root of two."""
        f = form([{"a": 2, "b": 4}, {"a": 1, "b": 1}])
        self.assertAlmostEqual(f.resampled_standard_error(0, 1), (2 ** 0.5) / (2 ** 0.5))


class Excess(unittest.TestCase):
    def test_unit_by_unit_noise_gives_a_ratio_near_one(self):
        rng = random.Random(0)
        forms = [form(noisy_seeds(rng, 400, 3, 2.0, 0.0), f"f{k}") for k in range(40)]
        excess = excess_of(forms)
        self.assertEqual(excess.degrees, 80)
        self.assertTrue(0.7 < excess.ratio < 1.4, excess.ratio)
        self.assertGreater(excess.chance, 0.01)

    def test_a_whole_set_shift_shows_as_excess(self):
        """Seeds that move every unit together: far more spread than unit-by-unit noise gives."""
        rng = random.Random(1)
        forms = [form(noisy_seeds(rng, 49, 3, 2.0, 1.0), f"f{k}") for k in range(2)]
        excess = excess_of(forms)
        self.assertGreater(excess.ratio, 3)
        self.assertLess(excess.chance, 0.05)


if __name__ == "__main__":
    unittest.main()
