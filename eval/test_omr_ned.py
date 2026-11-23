"""Tests for omr_ned's voice options on built bars, and print fixes set on the gold. Run from eval/ in .venv-musicdiff:
../.venv-musicdiff/bin/python -m unittest test_omr_ned"""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import music21 as m21

from omr_ned import Options, authors_metric, compare_parsed, listed_note, with_open_voice_choices_alike

METRIC = authors_metric()


def bar(*voices) -> m21.stream.Score:
    """A 3/4 bar of voices, each a list of (pitch or tuple of pitches, onset, quarter length)."""
    score, part, measure = m21.stream.Score(), m21.stream.Part(), m21.stream.Measure(number=1)
    measure.append(m21.meter.TimeSignature("3/4"))
    for k, items in enumerate(voices):
        voice = m21.stream.Voice(id=str(k + 1))
        for pitches, onset, length in items:
            voice.insert(onset, m21.chord.Chord(list(pitches), quarterLength=length) if isinstance(pitches, tuple)
                         else m21.note.Note(pitches, quarterLength=length))
        measure.insert(0, voice)
    part.append(measure)
    score.append(part)
    return score


def cost(pred: m21.stream.Score, gold: m21.stream.Score, voicing: bool, either_voices: bool) -> int:
    if either_voices:
        with_open_voice_choices_alike(pred, gold, m21)
    detail = Options(voicing=voicing).detail()
    return METRIC.Comparison.annotated_scores_diff(METRIC.AnnScore(pred, detail), METRIC.AnnScore(gold, detail))[1]


# Polish 110 bar 8: the gold splits E-flat 2 and 3 (one stem in print) by line, the reading a chord.
LINE_SPLIT = lambda: bar([("E-3", 0, 1), ("C-4", 1, 1), ("E-4", 2, 1)], [("E-2", 0, 1), (("A-3", "C-4"), 1, 2)])
AS_CHORD = lambda: bar([(("E-2", "E-3"), 0, 1), (("A-3", "C-4"), 1, 2)], [("C-4", 1, 1), ("E-4", 2, 1)])
# Two voices of different rhythms; G5 filed in the lower voice.
CLEAR = lambda: bar([("E5", 0, 1), ("F5", 1, 1), ("G5", 2, 1)], [("C4", 0, 2)])
WRONG_VOICE = lambda: bar([("E5", 0, 1), ("F5", 1, 1)], [("C4", 0, 2), ("G5", 2, 1)])


class EitherVoices(unittest.TestCase):
    def test_voicing_charges_a_split_the_print_leaves_open(self):
        self.assertGreater(cost(AS_CHORD(), LINE_SPLIT(), voicing=True, either_voices=False), 0)

    def test_either_voices_allows_it(self):
        self.assertEqual(cost(AS_CHORD(), LINE_SPLIT(), voicing=True, either_voices=True), 0)

    def test_a_shared_note_once_or_in_both_voices_costs_the_same(self):
        both = bar([("D-5", 0, 1), ("B-5", 1, 2)], [("D-5", 0, 1), ("G-4", 1, 2)])
        once = bar([("D-5", 0, 1), ("B-5", 1, 2)], [("G-4", 1, 2)])
        self.assertEqual(cost(once, both, voicing=True, either_voices=True), 0)

    def test_a_note_in_a_voice_the_print_decides_is_still_charged(self):
        self.assertGreater(cost(WRONG_VOICE(), CLEAR(), voicing=True, either_voices=True), 0)

    def test_default_omr_ned_charges_no_wrong_voice(self):
        self.assertEqual(cost(WRONG_VOICE(), CLEAR(), voicing=False, either_voices=False), 0)


if __name__ == "__main__":
    unittest.main()


# A beam from the lower staff's C3 E3 up to the upper staff's G4 C5, as the print draws it; the gold,
# like kern gold, holds four unbeamed eighths.
ACROSS = """<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1"><part-list><score-part id="P1"><part-name>Piano</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>2</divisions><time><beats>2</beats><beat-type>4</beat-type></time>
<staves>2</staves><clef number="1"><sign>G</sign><line>2</line></clef><clef number="2"><sign>F</sign><line>4</line></clef></attributes>
<note><pitch><step>C</step><octave>3</octave></pitch><duration>1</duration><voice>1</voice><type>eighth</type><staff>2</staff><beam number="1">begin</beam></note>
<note><pitch><step>E</step><octave>3</octave></pitch><duration>1</duration><voice>1</voice><type>eighth</type><staff>2</staff><beam number="1">continue</beam></note>
<note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>eighth</type><staff>1</staff><beam number="1">continue</beam></note>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><voice>1</voice><type>eighth</type><staff>1</staff><beam number="1">end</beam></note>
</measure></part></score-partwise>"""
UNBEAMED = re.sub(r"<beam [^>]*>\w+</beam>", "", ACROSS)


def beam_correction(pitch: str = "C5") -> dict:
    note = lambda staff, onset, pitch, kind: {"staff": staff, "onset": onset, "pitch": pitch, "types": [kind]}
    return {"page": "x", "bar": 0, "kind": "beams", "reason": "test", "notes": [note("lower", "0", "C3", "start"), note("lower", "1/2", "E3", "continue"),
                                                              note("upper", "1", "G4", "continue"), note("upper", "3/2", pitch, "stop")]}


class GoldBeams(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        folder = Path(self.folder.name)
        self.pred, self.gold, self.beams = folder / "reading" / "x.musicxml", folder / "x.musicxml", folder / "beams.jsonl"
        self.pred.parent.mkdir()
        self.pred.write_text(ACROSS)
        self.gold.write_text(UNBEAMED)

    def tearDown(self):
        self.folder.cleanup()

    def score(self, correction: dict | None) -> tuple[float | None, str | None]:
        if correction is not None:
            self.beams.write_text(json.dumps(correction) + "\n")
        return compare_parsed(METRIC, self.pred, self.gold, Options(gold_print_fixes=self.beams if correction else None))

    def test_a_beam_across_the_staves_is_charged_against_kern_gold(self):
        self.assertGreater(self.score(None)[0], 0)

    def test_set_on_the_gold_it_matches_the_reading(self):
        self.assertEqual(self.score(beam_correction())[0], 0)

    def test_a_listed_note_not_in_the_gold_stops_the_page(self):
        with self.assertRaisesRegex(ValueError, "C6 at 3/2: 0 notes"):
            self.score(beam_correction("C6"))


def g_major_bar(sharp_drawn: bool) -> str:
    """F natural, then F-sharp with its sign drawn or not."""
    sign = "<accidental>sharp</accidental>" if sharp_drawn else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1"><part-list><score-part id="P1"><part-name>P</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>1</divisions><key><fifths>1</fifths></key><time><beats>2</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>F</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type><accidental>natural</accidental></note>
<note><pitch><step>F</step><alter>1</alter><octave>4</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type>{sign}</note>
</measure></part></score-partwise>"""


class HiddenAccidental(unittest.TestCase):
    """The print leaves off the sharp the parser would draw after F natural; the reading too."""
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        folder = Path(self.folder.name)
        self.pred, self.gold, self.fixes = folder / "reading" / "x.musicxml", folder / "x.musicxml", folder / "fixes.jsonl"
        self.pred.parent.mkdir()
        self.pred.write_text(g_major_bar(sharp_drawn=False))
        self.gold.write_text(g_major_bar(sharp_drawn=True))

    def tearDown(self):
        self.folder.cleanup()

    def score(self, pitch: str | None) -> float:
        if pitch:
            self.fixes.write_text(json.dumps({"page": "x", "bar": 0, "kind": "hide_accidental", "reason": "test",
                                              "notes": [{"staff": "upper", "onset": "1", "pitch": pitch}]}) + "\n")
        return compare_parsed(METRIC, self.pred, self.gold, Options(gold_print_fixes=self.fixes if pitch else None))[0]

    def test_the_drawn_sign_is_charged(self):
        self.assertGreater(self.score(None), 0)

    def test_hidden_on_the_gold_it_matches(self):
        self.assertEqual(self.score("F#4"), 0)


def accented_bar(mark: str) -> str:
    """One C5 quarter with the articulation `mark` (accent or strong-accent)."""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<score-partwise version="3.1"><part-list><score-part id="P1"><part-name>P</part-name></score-part></part-list>
<part id="P1"><measure number="1"><attributes><divisions>1</divisions><time><beats>1</beats><beat-type>4</beat-type></time><clef><sign>G</sign><line>2</line></clef></attributes>
<note><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><voice>1</voice><type>quarter</type><notations><articulations><{mark}/></articulations></notations></note>
</measure></part></score-partwise>"""


class StrongAccent(unittest.TestCase):
    """The print draws a strong accent; the gold writes kern's plain accent; the reading says strong accent."""
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        folder = Path(self.folder.name)
        self.pred, self.gold, self.fixes = folder / "reading" / "x.musicxml", folder / "x.musicxml", folder / "fixes.jsonl"
        self.pred.parent.mkdir()
        self.pred.write_text(accented_bar("strong-accent"))
        self.gold.write_text(accented_bar("accent"))

    def tearDown(self):
        self.folder.cleanup()

    def score(self, pitch: str | None) -> float:
        if pitch:
            self.fixes.write_text(json.dumps({"page": "x", "bar": 0, "kind": "strong_accent", "reason": "test",
                                              "notes": [{"staff": "upper", "onset": "0", "pitch": pitch}]}) + "\n")
        return compare_parsed(METRIC, self.pred, self.gold, Options(gold_print_fixes=self.fixes if pitch else None))[0]

    def test_the_plain_accent_is_charged(self):
        self.assertGreater(self.score(None), 0)

    def test_made_strong_on_the_gold_it_matches(self):
        self.assertEqual(self.score("C5"), 0)

    def test_a_note_with_no_accent_stops_the_page(self):
        self.gold.write_text(accented_bar("staccato"))
        with self.assertRaisesRegex(ValueError, "no accent to make strong"):
            self.score("C5")


class ListedNote(unittest.TestCase):
    def test_length_tells_apart_two_notes_of_one_onset_and_pitch(self):
        score = bar([("E-5", 0, 1)], [("E-5", 0, Fraction(1, 3)), ("A4", Fraction(1, 3), Fraction(1, 3))])
        self.assertEqual(listed_note(score, 0, {"staff": "upper", "onset": "0", "pitch": "E-5", "length": "1/3"}).quarterLength,
                         Fraction(1, 3))
        with self.assertRaisesRegex(ValueError, "2 notes"):
            listed_note(score, 0, {"staff": "upper", "onset": "0", "pitch": "E-5"})
