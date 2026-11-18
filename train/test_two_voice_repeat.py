import tempfile
import unittest
from pathlib import Path

from seq.data import Sample
import json

from train_seq import dataset_items, has_two_voices, training_items

ONE_VOICE = "measure clef:G2 staff:1 C4 voice:1 half staff:1 D4 half"
TWO_VOICES = "measure clef:G2 staff:1 C5 voice:1 whole staff:1 backup whole C4 voice:2 whole staff:1"
TWO_STAVES = "measure clef:G2 staff:1 C5 voice:1 whole staff:1 backup whole C3 voice:5 whole staff:2"


def sample(directory: Path, name: str, lmx: str) -> Sample:
    (directory / f"{name}.lmx").write_text(lmx)
    (directory / f"{name}.png").write_bytes(b"")
    return Sample(directory / f"{name}.png", directory / f"{name}.lmx")


class TwoVoices(unittest.TestCase):
    def test_two_voices_on_one_staff_count_one_on_each_does_not(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(has_two_voices(sample(Path(d), "a", TWO_VOICES)))
            self.assertFalse(has_two_voices(sample(Path(d), "b", TWO_STAVES)))
            self.assertFalse(has_two_voices(sample(Path(d), "c", ONE_VOICE)))

    def test_repeat_multiplies_only_two_voice_systems(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            sample(root, "a", TWO_VOICES), sample(root, "b", ONE_VOICE)
            (root / "samples.train.txt").write_text("a b")
            names = [s.lmx.stem for s in dataset_items(f"{root}:train", two_voice_repeat=3)]
            self.assertEqual(sorted(names), ["a", "a", "a", "b"])
            self.assertEqual(sorted(s.lmx.stem for s in dataset_items(f"{root}:train")), ["a", "b"])


class TrainingItems(unittest.TestCase):
    """The mix the training loop gets, on each path: plain, and with staff positions (the path the
    system runs take, which once dropped the repeat)."""
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name) / "set"
        self.root.mkdir()
        sample(self.root, "a", TWO_VOICES), sample(self.root, "b", ONE_VOICE), sample(self.root, "c", ONE_VOICE)
        (self.root / "samples.train.txt").write_text("a b c")
        self.positions = Path(self.folder.name) / "positions"
        self.positions.mkdir()
        staves = {"a": [10, 50, 8], "b": [10, 50, 8], "c": None}  # centres..., interline; c has none
        (self.positions / "set.json").write_text(json.dumps({"staves": staves}))

    def tearDown(self):
        self.folder.cleanup()

    def names(self, items) -> list[str]:
        return sorted((item[0] if isinstance(item, tuple) else item).lmx.stem for item in items)

    def test_plain_path_repeats_two_voice_systems(self):
        self.assertEqual(self.names(training_items([f"{self.root}:train"], 3, None)), ["a", "a", "a", "b", "c"])

    def test_staff_position_path_repeats_them_too(self):
        items = training_items([f"{self.root}:train"], 3, self.positions)
        self.assertEqual(self.names(items), ["a", "a", "a", "b"])

    def test_a_set_weight_multiplies_after_the_repeat(self):
        self.assertEqual(self.names(training_items([f"{self.root}:train:2"], 2, self.positions)), ["a"] * 4 + ["b"] * 2)


if __name__ == "__main__":
    unittest.main()
