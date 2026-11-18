"""Tests for synthetic_benchmark's render cache. Run from eval/ in .venv."""
import tempfile
import unittest
from pathlib import Path

from synthetic_benchmark import cached_measures  # puts train/ on the path, for omr

from omr.held_out import score_key  # noqa: E402


class Cache(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.out = Path(self.dir.name)
        self.score, self.other = self.out / "a.mxl", self.out / "b.mxl"
        self.score.write_text("score a")
        self.other.write_text("score b")

    def tearDown(self):
        self.dir.cleanup()

    def cache(self, key: str | None) -> None:
        if key is not None:
            (self.out / "000.score").write_text(key)
        (self.out / "000.measures").write_text("12")

    def test_no_cached_page(self):
        self.assertIsNone(cached_measures(self.out, 0, self.score))

    def test_a_page_of_the_same_score_is_reused(self):
        self.cache(score_key(self.score))
        self.assertEqual(cached_measures(self.out, 0, self.score), 12)

    def test_a_page_of_another_score_stops_the_run(self):
        self.cache(score_key(self.other))
        with self.assertRaisesRegex(RuntimeError, "fresh directory"):
            cached_measures(self.out, 0, self.score)

    def test_a_page_with_no_recorded_score_stops_the_run(self):
        self.cache(None)
        with self.assertRaisesRegex(RuntimeError, "unrecorded score"):
            cached_measures(self.out, 0, self.score)


if __name__ == "__main__":
    unittest.main()
