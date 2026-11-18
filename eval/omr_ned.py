"""OMR-NED, the metric of the Transcoda paper, with its authors' code unchanged.

usage: omr_ned.py PREDICTIONS GOLD --out RESULTS.json [--pred-format kern|musicxml]
                  [--gold-format kern|musicxml] [--jobs N] [--ambiguous-marks] [--either-ottava]
                  [--either-display] [--whole-page-alignment] [--voicing] [--either-voices]
                  [--gold-print-fixes FILE]

Runs in .venv-musicdiff (requirements-omr-ned.txt). PREDICTIONS and GOLD are directories of
<sample>.krn or <sample>.musicxml. A page that cannot be parsed or compared scores 100, the
authors' policy, as does one over the time or memory limit here, and a missing prediction.
The score is the mean over pages; lower is better. Each page runs in its own process:
musicdiff's edit distance can take tens of gigabytes on a bad prediction.

--either-ottava scores a MusicXML page with ottava signs twice, as drawn and with the signs folded
into the pitch, and keeps the better: the protocol's rule for both Corrected and Print standard.

--either-display is for the standards that do not hold a system to how the page draws two things
(As published, Corrected): a tuplet's number and bracket, and C or cut time against the numbers
4/4 or 2/2. Both pages are scored as if they drew them alike, so a hidden 3 costs nothing against a
shown one, nor C against 4/4. The tuplet's ratio and the meter are still charged.

--ambiguous-marks is for Print standard gold: where a reviewer could not tell a dot from a wedge
(gold_overrides.py), a prediction's wedge is refunded the edit a staccato would not have cost. A
missing mark is still charged. Everything else is the authors' metric.

--whole-page-alignment aligns bars by the authors' own edit costs over the whole page. Their
musicdiff first pins down the longest run of identical bars (Myers' diff) and aligns only between
those pins. That maximises identical bars, not the lowest cost: on a page of repeated bars it can
pin a bar to the wrong copy, and every bar between then counts as deleted and inserted. A correct
change that makes one more bar identical can then raise the score (Polish 017: 452 to 1,063 where
the whole-page alignment goes 376 to 352). Never higher than the authors' number; slower.

--voicing is not OMR-NED: a diagnostic beside it (decided with the user, 2026-11-17). musicdiff's
Voicing detail compares each bar voice by voice, each voice's notes as a sequence (voices paired
at least cost, their numbers ignored), and a chord as one object. The authors' default pools a
bar's notes and pairs them by pitch and onset, so it charges a note in the wrong voice nothing,
and one early error in a bar again for every later note it shifts (each deleted and inserted);
Voicing charges the wrong voice and the first error once (a voice's entry moved costs 1).

--either-voices allows the voice choices the print leaves open (2026-11-17, with the user; to be
revisited): notes of one onset and length split across voices, which the print can draw as one
chord (Polish 110 bar 8), and a shared note written in both voices (060 bar 25). Wherever either
page splits notes of one onset and length across voices, both pages move all their notes at that
place into one pooled voice, as one chord, each pitch once; so neither the split nor which voice
the chord continues is charged. With --voicing, voices elsewhere are still compared; without it,
what remains is a doubled note counted once. Undercharges two voices the print does draw with
opposite stems in one rhythm: the kern gold records no stems to tell them apart.

--gold-print-fixes FILE sets on the parsed gold what the print draws and kern cannot say
(2026-11-17, with the user), before it is compared. One JSON object per line
(eval/polish_print_fixes.jsonl): page, bar (from 0, as the scorer counts), kind, reason, and
notes, each {"staff": "upper" or "lower", "onset": quarters into the bar (a string such as
"1/3"), "pitch": one it holds, as "B-3", and, where another note shares onset and pitch (a
tied note's second voice, a grace note), "length" in quarters, "0" for a grace}. Kinds:
  beams            a beam joining notes on both staves, which music21 keeps as each note's beam
                   types on its own staff (lower: start, continue; upper: continue, stop): each
                   note also has "types", its beam per level, from "start", "continue", "stop",
                   "partial-right", "partial-left"
  hide_accidental  the listed pitch's sign not drawn, where music21 draws one the print leaves
                   off (a note tied into a new system, a chord repeated in the bar); its parser
                   ignores kern's `y`, and musicdiff charges a sign only where it is drawn
A listed note found nowhere, or twice, or a hidden sign on a pitch with none, fails the page: a
correction never silently does nothing.
"""
from __future__ import annotations

import argparse
import dataclasses
import functools
import importlib.util
import json
import multiprocessing as mp
import os
import resource
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from gold_overrides import AMBIGUOUS_COLOR

ROOT = Path(__file__).resolve().parent.parent
AUTHORS_METRIC = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw/transcoda_repo/src/src/evaluation/omr_ned.py"
PAGE_MEMORY_BYTES = 6 << 30
PAGE_TIMEOUT_S = 900
FAILED = 100.0
SUFFIX = {"kern": ".krn", "musicxml": ".musicxml"}


@dataclasses.dataclass(frozen=True)
class Options:
    ambiguous_marks: bool = False
    whole_page_alignment: bool = False
    either_display: bool = False
    voicing: bool = False
    either_voices: bool = False
    gold_print_fixes: Path | None = None

    def detail(self) -> int:
        from musicdiff import DetailLevel  # polish_report.py imports this module from the training venv, which lacks musicdiff
        return DetailLevel.Default | DetailLevel.Voicing if self.voicing else DetailLevel.Default


def one_block_of_bars(original, compare_to):
    return [{"original": list(original), "compare_to": list(compare_to)}]


def authors_metric():
    spec = importlib.util.spec_from_file_location("transcoda_omr_ned", AUTHORS_METRIC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def ambiguous_notes(score) -> set[int]:
    """Ids of the notes and chords flagged in gold; the metric's annotations keep these ids."""
    flagged = lambda note: note.hasStyleInformation and note.style.color == AMBIGUOUS_COLOR
    return {general.id for general in score.recurse().notes
            if any(flagged(note) for note in (general.notes if general.isChord else [general]))}


def wedge_refund(operations, ambiguous: set[int]) -> int:
    return sum(operation[3] for operation in operations if operation[0] == "editarticulation"
               and operation[2].general_note in ambiguous and "staccatissimo" in operation[1].articulations)


def parsed_as_authors_do(metric, path: Path, prediction: bool):
    """A score as compute_omr_ned would read it. Their **kern often has no header line (the
    published gold, Transcoda's output), and without one the parser silently returns an empty
    score, which then scores 100. A prediction's syntax errors are repaired and counted."""
    m21 = metric.m21
    if path.suffix != SUFFIX["kern"]:
        return m21.converter.parse(str(path), forceSource=True)
    with tempfile.NamedTemporaryFile("w", suffix=SUFFIX["kern"], encoding="utf-8") as kern:
        kern.write(metric._ensure_kern_header(path.read_text()))
        kern.flush()
        return m21.converter.parse(kern.name, forceSource=True, acceptSyntaxErrors=prediction)


def draw_display_alike(score) -> None:
    """musicdiff reads a tuplet's display (number shown, bracket) from its first note, and a time
    signature's symbol in place of its numbers."""
    for note in score.recurse().notesAndRests:
        for tuplet in note.duration.tuplets:
            tuplet.tupletActualShow, tuplet.tupletNormalShow, tuplet.bracket, tuplet.placement = "number", None, False, None
    for time in score.recurse().getElementsByClass("TimeSignature"):
        time.symbol = ""


Place = tuple[int, int, "Fraction", "Fraction"]   # part, measure, onset, length


def timed_notes(score):
    """Each voiced note's place, with the note and its voice (grace notes take no place)."""
    for p, part in enumerate(score.parts):
        for m, measure in enumerate(part.getElementsByClass("Measure")):
            for voice in measure.voices:
                for note in voice.notes:
                    if not note.duration.isGrace:
                        yield (p, m, Fraction(note.getOffsetInHierarchy(measure)), Fraction(note.quarterLength)), note, voice


def split_places(score) -> set[Place]:
    """Where notes of one onset and length stand in more than one voice of a staff."""
    voices_at = defaultdict(set)
    for place, _, voice in timed_notes(score):
        voices_at[place].add(id(voice))
    return {place for place, voices in voices_at.items() if len(voices) > 1}


def all_in_voices(score, m21) -> None:
    """A measure's notes and rests outside any voice put into one, so a pooled voice beside them
    leaves them compared (musicdiff reads only a measure's voices once it has any)."""
    for measure in score.recurse().getElementsByClass("Measure"):
        loose = list(measure.getElementsByClass("GeneralNote"))
        if loose and not measure.voices:
            voice = m21.stream.Voice(id="1")
            for element in loose:
                offset = element.getOffsetBySite(measure)
                measure.remove(element)
                voice.insert(offset, element)
            measure.insert(0, voice)


def with_places_pooled(score, places: set[Place], m21) -> None:
    pooled = defaultdict(set)
    for place, note, voice in list(timed_notes(score)):
        if place in places:
            pooled[place].update(p.nameWithOctave for p in note.pitches)
            voice.remove(note)
    measures = {(p, m): measure for p, part in enumerate(score.parts) for m, measure in enumerate(part.getElementsByClass("Measure"))}
    pools = {}
    for (p, m, onset, length), pitches in sorted(pooled.items()):
        pool = pools.get((p, m))
        if pool is None:
            pool = pools[(p, m)] = m21.stream.Voice(id="pooled")
            measures[(p, m)].insert(0, pool)
        pool.insert(float(onset), m21.chord.Chord(sorted(pitches, key=lambda name: m21.pitch.Pitch(name).ps),
                                                    quarterLength=length))


def with_open_voice_choices_alike(pred_score, gold_score, m21) -> None:
    for score in (pred_score, gold_score):
        all_in_voices(score, m21)
    places = split_places(pred_score) | split_places(gold_score)
    for score in (pred_score, gold_score):
        with_places_pooled(score, places, m21)


STAVES = {"upper": 0, "lower": 1}


def print_fixes(path: Path, page: str) -> list[dict]:
    return [c for c in map(json.loads, filter(str.strip, path.read_text().splitlines())) if c["page"] == page]


def listed_note(score, bar: int, note: dict):
    measure = list(score.parts[STAVES[note["staff"]]].getElementsByClass("Measure"))[bar]
    onset = Fraction(note["onset"])
    found = [n for n in measure.recurse().notes
             if Fraction(n.getOffsetInHierarchy(measure)) == onset and note["pitch"] in (p.nameWithOctave for p in n.pitches)
             and ("length" not in note or Fraction(n.quarterLength) == Fraction(note["length"]))]
    if len(found) != 1:
        raise ValueError(f"bar {bar}, {note['staff']} staff, {note['pitch']} at {note['onset']}: {len(found)} notes, not 1")
    return found[0]


def with_beams_set(found, note: dict, m21) -> None:
    beams = m21.beam.Beams()
    for kind in note["types"]:
        beams.append(*kind.split("-"))
    found.beams = beams


def with_accidental_hidden(found, note: dict, m21) -> None:
    pitch = next(p for p in found.pitches if p.nameWithOctave == note["pitch"])
    if pitch.accidental is None:
        raise ValueError(f"{note['pitch']} at {note['onset']}: no accidental to hide")
    pitch.accidental.displayStatus = False


PRINT_FIXES = {"beams": with_beams_set, "hide_accidental": with_accidental_hidden}


def with_print_fixes(score, fixes: list[dict], m21) -> None:
    for fix in fixes:
        for note in fix["notes"]:
            PRINT_FIXES[fix["kind"]](listed_note(score, fix["bar"], note), note, m21)


def compare_parsed(metric, pred: Path, gold: Path, options: Options) -> tuple[float | None, str | None]:
    """The authors' formula on scores parsed here, also for two files of different formats, which
    they never compare."""
    gold_score = parsed_as_authors_do(metric, gold, prediction=False)
    pred_score = parsed_as_authors_do(metric, pred, prediction=True)
    if options.gold_print_fixes:
        with_print_fixes(gold_score, print_fixes(options.gold_print_fixes, gold.stem), metric.m21)
    if options.either_display:
        draw_display_alike(gold_score)
        draw_display_alike(pred_score)
    if options.either_voices:
        with_open_voice_choices_alike(pred_score, gold_score, metric.m21)
    pred_ann = metric.AnnScore(pred_score, options.detail())
    gold_ann = metric.AnnScore(gold_score, options.detail())
    operations, distance = metric.Comparison.annotated_scores_diff(pred_ann, gold_ann)
    if options.ambiguous_marks:
        distance -= wedge_refund(operations, ambiguous_notes(gold_score))
    total = pred_ann.notation_size() + gold_ann.notation_size()
    return (min(100.0 * (distance + pred_ann.num_syntax_errors_fixed) / total, 100.0) if total else 0.0), None


def score_page(pred: Path, gold: Path, pred_format: str, gold_format: str, options: Options, answer) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (PAGE_MEMORY_BYTES, PAGE_MEMORY_BYTES))
    try:
        metric = authors_metric()
        if options.whole_page_alignment:
            metric.Comparison._non_common_subsequences_of_measures = staticmethod(one_block_of_bars)
        if pred_format != gold_format or options.ambiguous_marks or options.either_display or options.voicing or options.either_voices \
                or options.gold_print_fixes:
            answer.send(compare_parsed(metric, pred, gold, options))
            return
        compute = metric.compute_omr_ned if pred_format == "kern" else metric.compute_omr_ned_from_musicxml
        result = compute(pred.read_text(), gold.read_text())
        answer.send((result.omr_ned, result.parse_error))
    except BaseException as e:  # noqa: B902  MemoryError and RecursionError count as failed pages
        answer.send((None, f"{type(e).__name__}: {str(e)[:150]}"))


def page_score(job: tuple[Path, Path, str, str, Options]) -> tuple[str, float, str | None]:
    pred, gold, pred_format, gold_format, options = job
    if not pred.exists():
        return gold.stem, FAILED, "no prediction"
    receive, send = mp.Pipe(duplex=False)
    worker = mp.get_context("spawn").Process(target=score_page, args=(pred, gold, pred_format, gold_format, options, send))
    worker.start()
    send.close()
    answered = receive.poll(PAGE_TIMEOUT_S)
    try:
        score, error = receive.recv() if answered else (None, "timed out")
    except EOFError:
        score, error = None, "worker died"
    worker.kill()
    worker.join()
    return gold.stem, FAILED if score is None else score, error


def without_octave_shifts(musicxml: Path, folder: Path) -> Path:
    """A copy of the page with its ottava signs taken out. MusicXML keeps the sounding pitch
    under a sign, so what is left is the same music with the ottava folded into the pitch."""
    tree = ET.parse(musicxml)
    for measure in tree.getroot().iter("measure"):
        for direction in measure.findall("direction"):
            if direction.find("direction-type/octave-shift") is not None:
                measure.remove(direction)
    folded = folder / musicxml.name
    tree.write(folded)
    return folded


def better_of_signed_and_folded(job: tuple[Path, Path, str, str, Options], folder: Path) -> tuple[str, float, str | None]:
    """The page's better score of its ottavas as signs and folded into the pitch: gold signs
    some and folds others, and Corrected gold folds all (docs/EVALUATION_PROTOCOL.md)."""
    signed = page_score(job)
    pred = job[0]
    if pred.suffix != SUFFIX["musicxml"] or not pred.exists() or "<octave-shift" not in pred.read_text():
        return signed
    folded = page_score((without_octave_shifts(pred, folder), *job[1:]))
    return min(signed, folded, key=lambda page: page[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("gold", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pred-format", choices=SUFFIX, default="kern")
    ap.add_argument("--gold-format", choices=SUFFIX, default="kern")
    ap.add_argument("--ambiguous-marks", action="store_true")
    ap.add_argument("--either-ottava", action="store_true",
                    help="each page's better score of its ottava signs kept and folded into the pitch")
    ap.add_argument("--either-display", action="store_true",
                    help="tuplet numbers shown or hidden, and C or 4/4, cost the same")
    ap.add_argument("--whole-page-alignment", action="store_true",
                    help="align bars by edit cost over the whole page, not between runs of identical bars")
    ap.add_argument("--voicing", action="store_true",
                    help="not OMR-NED: compare voice by voice (musicdiff's Voicing detail), a diagnostic")
    ap.add_argument("--either-voices", action="store_true",
                    help="notes of one onset and length split across voices or not, and a shared note once or twice, cost the same")
    ap.add_argument("--gold-print-fixes", type=Path,
                    help="what the print draws and kern cannot say (beams across the staves, signs left off), set on the parsed gold")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    golds = sorted(args.gold.glob("*" + SUFFIX[args.gold_format]))
    options = Options(args.ambiguous_marks, args.whole_page_alignment, args.either_display, args.voicing, args.either_voices,
                      args.gold_print_fixes)
    jobs = [(args.predictions / (g.stem + SUFFIX[args.pred_format]), g, args.pred_format, args.gold_format, options) for g in golds]
    from multiprocessing.pool import ThreadPool
    with ThreadPool(args.jobs) as pool, tempfile.TemporaryDirectory() as folder:
        score = functools.partial(better_of_signed_and_folded, folder=Path(folder)) if args.either_ottava else page_score
        pages = pool.map(score, jobs)
    scores = {name: score for name, score, _ in pages}
    errors = {name: error for name, _, error in pages if error}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scores": scores, "errors": errors}, indent=0))
    print(f"OMR-NED {sum(scores.values()) / len(scores):.2f} over {len(scores)} pages, {len(errors)} failed")


if __name__ == "__main__":
    main()
