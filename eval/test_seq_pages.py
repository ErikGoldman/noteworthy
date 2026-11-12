"""Tests for seq_pages' layout handling. Run from eval/: python -m unittest test_seq_pages"""
from __future__ import annotations

import unittest

from seq_pages import without_duplicate_staves

PIANO = [[100, 180], [400, 480]]


class DuplicateStaves(unittest.TestCase):
    def test_a_copy_in_a_system_of_its_own_goes(self):
        """Polish 097: the last system's top staff also found alone, a little lower."""
        systems = [{"page": 1, "staves": [[2785, 2867]]}, {"page": 1, "staves": [[2794, 2879], [3062, 3146]]}]
        self.assertEqual([s["staves"] for s in without_duplicate_staves(systems)], [[], [[2794, 2879], [3062, 3146]]])

    def test_two_copies_in_one_system_leave_one(self):
        """Polish 106: a song's voice staff found twice, the copies a system of their own."""
        systems = [{"page": 1, "staves": [[2789, 2873], [2800, 2887]]}]
        self.assertEqual(len(without_duplicate_staves(systems)[0]["staves"]), 1)

    def test_real_staves_untouched(self):
        systems = [{"page": 1, "staves": PIANO}, {"page": 1, "staves": [[700, 780], [1000, 1080]]},
                   {"page": 2, "staves": PIANO}]
        self.assertEqual(without_duplicate_staves(systems), systems)


if __name__ == "__main__":
    unittest.main()
