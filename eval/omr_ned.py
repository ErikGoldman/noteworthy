"""OMR-NED, the metric of the Transcoda paper, with its authors' code unchanged.

usage: omr_ned.py PREDICTIONS GOLD --out RESULTS.json [--pred-format kern|musicxml]
                  [--gold-format kern|musicxml] [--jobs N] [--ambiguous-marks] [--either-ottava]

Runs in .venv-musicdiff (requirements-omr-ned.txt). PREDICTIONS and GOLD are directories of
<sample>.krn or <sample>.musicxml. A page that cannot be parsed or compared scores 100, the
authors' policy, as does one over the time or memory limit here, and a missing prediction.
The score is the mean over pages; lower is better. Each page runs in its own process:
musicdiff's edit distance can take tens of gigabytes on a bad prediction.

--either-ottava scores a MusicXML page with ottava signs twice, as drawn and with the signs folded
into the pitch, and keeps the better: the protocol's rule for both Corrected and Print standard.

--ambiguous-marks is for Print standard gold: where a reviewer could not tell a dot from a wedge
(gold_overrides.py), a prediction's wedge is refunded the edit a staccato would not have cost. A
missing mark is still charged. Everything else is the authors' metric.
"""
from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import multiprocessing as mp
import os
import resource
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from gold_overrides import AMBIGUOUS_COLOR

ROOT = Path(__file__).resolve().parent.parent
AUTHORS_METRIC = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw/transcoda_repo/src/src/evaluation/omr_ned.py"
PAGE_MEMORY_BYTES = 6 << 30
PAGE_TIMEOUT_S = 900
FAILED = 100.0
SUFFIX = {"kern": ".krn", "musicxml": ".musicxml"}


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


def compare_across_formats(metric, pred: Path, gold: Path, ambiguous_marks: bool = False) -> tuple[float | None, str | None]:
    """The authors' formula on two files of different formats, which they never compare."""
    gold_score = parsed_as_authors_do(metric, gold, prediction=False)
    pred_ann = metric.AnnScore(parsed_as_authors_do(metric, pred, prediction=True))
    gold_ann = metric.AnnScore(gold_score)
    operations, distance = metric.Comparison.annotated_scores_diff(pred_ann, gold_ann)
    if ambiguous_marks:
        distance -= wedge_refund(operations, ambiguous_notes(gold_score))
    total = pred_ann.notation_size() + gold_ann.notation_size()
    return (min(100.0 * (distance + pred_ann.num_syntax_errors_fixed) / total, 100.0) if total else 0.0), None


def score_page(pred: Path, gold: Path, pred_format: str, gold_format: str, ambiguous_marks: bool, answer) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (PAGE_MEMORY_BYTES, PAGE_MEMORY_BYTES))
    try:
        metric = authors_metric()
        if pred_format != gold_format or ambiguous_marks:
            answer.send(compare_across_formats(metric, pred, gold, ambiguous_marks))
            return
        compute = metric.compute_omr_ned if pred_format == "kern" else metric.compute_omr_ned_from_musicxml
        result = compute(pred.read_text(), gold.read_text())
        answer.send((result.omr_ned, result.parse_error))
    except BaseException as e:  # noqa: B902  MemoryError and RecursionError count as failed pages
        answer.send((None, f"{type(e).__name__}: {str(e)[:150]}"))


def page_score(job: tuple[Path, Path, str, str, bool]) -> tuple[str, float, str | None]:
    pred, gold, pred_format, gold_format, ambiguous_marks = job
    if not pred.exists():
        return gold.stem, FAILED, "no prediction"
    receive, send = mp.Pipe(duplex=False)
    worker = mp.get_context("spawn").Process(target=score_page, args=(pred, gold, pred_format, gold_format, ambiguous_marks, send))
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


def better_of_signed_and_folded(job: tuple[Path, Path, str, str, bool], folder: Path) -> tuple[str, float, str | None]:
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
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    golds = sorted(args.gold.glob("*" + SUFFIX[args.gold_format]))
    jobs = [(args.predictions / (g.stem + SUFFIX[args.pred_format]), g, args.pred_format, args.gold_format, args.ambiguous_marks) for g in golds]
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
