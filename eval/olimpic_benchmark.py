"""Benchmarks on real scans: OLiMPiC, the piano accompaniments of OpenScore Lieder songs
(CC0 transcriptions) paired with the IMSLP scans they were transcribed from (CC BY-SA).

usage: olimpic_benchmark.py systems|pages --binary NOTEWORTHY --model MODEL.onnx --out DIR
                                          [--split dev|test] [--jobs N] [--threads T]
       olimpic_benchmark.py systems --predictions DIR --out DIR [--split dev|test]
       olimpic_benchmark.py list [--split dev|test]

  systems  each OLiMPiC system crop is recognized alone and scored against its piano
           ground truth. Tests reading real engravings and scan noise. A crop shows no
           time signature unless the system starts the song, so the recognizer assumes
           4/4 there; onsets and durations are scored, which mostly do not depend on it.
           With --predictions, another recognizer's MusicXML for each crop is scored
           instead (DIR/samples/<song>/<system>.musicxml; see zeus_comparison.py).
           The samples scored are listed in OUT/scored_samples.txt.
  list     prints the samples that may be scored (no copy in our training sources), e.g.
           for monitoring a model trained on PDMX (train_seq.py --dev-samples).
  pages    each song's full IMSLP pages are recognized in order as one score, so page
           layout (finding staves and systems among the voice part, lyrics, titles and
           neighbouring songs) and context carried across systems and pages are tested
           too. Only the piano staves OLiMPiC annotated are scored: recognized staves
           outside its system boxes (the voice, other songs) are dropped.

Develop against dev; test is held out like Night Album (look at its number, never at its
errors). Songs whose piece also exists in the training sources are left out: songs on
PDMX (by MuseScore id) and songs with a copy in PDMX or ASAP (omr.pieces).

Data (scripts/fetch_data.sh): $NW_DATA/raw/olimpic/{olimpic-1.0-scanned,sources,lieder_scores.yaml}.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from note_scoring import report, score
from results_file import save_results
from score_events import NoteEvent, parse_musicxml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from omr.pieces import Piece, PieceIndex, asap_pieces, pdmx_pieces, surname, title_words  # noqa: E402

RAW = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw"
OLIMPIC = RAW / "olimpic"
SCANNED = OLIMPIC / "olimpic-1.0-scanned"
SOURCES = OLIMPIC / "sources"


# --- which songs count -----------------------------------------------------------------

# Retitled in PDMX, so metadata misses them; found by comparing melodies staff by staff.
COPIES_FOUND_BY_MELODY = {"6472842"}  # Haydn, The Mermaid's Song: "Now The Dancing Sunbeams Play"


def lieder_piece(path: str) -> Piece:
    """Piece of a Lieder song from its repository path, "Last,_First/set/NN_Title"."""
    composer, _, song = path.split("/")
    return Piece(title_words(re.sub(r"^\d+[a-z]?_", "", song).replace("_", " ")), surname(composer.split(",")[0]))


def pdmx_musescore_ids() -> set[str]:
    csv.field_size_limit(1 << 30)
    with (RAW / "pdmx" / "PDMX.csv").open() as f:
        return {Path(row["metadata"]).stem for row in csv.DictReader(f)}


def songs_in_training_sources() -> set[str]:
    """MuseScore ids of Lieder songs that are, or have a copy, in PDMX or ASAP."""
    lieder = yaml.safe_load((OLIMPIC / "lieder_scores.yaml").read_text())
    known = PieceIndex([p for pieces in (pdmx_pieces(RAW / "pdmx") | asap_pieces(RAW / "asap")).values() for p in pieces])
    on_pdmx = pdmx_musescore_ids()
    by_metadata = {str(k) for k, v in lieder.items() if str(k) in on_pdmx or known.contains(lieder_piece(v["path"]))}
    return by_metadata | COPIES_FOUND_BY_MELODY


def split_samples(split: str) -> tuple[list[str], int]:
    """Samples ("samples/<song>/p<page>-s<system>") of a split that may be scored, and
    how many were left out as copies of training pieces."""
    samples = (SCANNED / f"samples.{split}.txt").read_text().split()
    excluded = songs_in_training_sources()
    kept = [s for s in samples if song_of(s) not in excluded]
    return kept, len(samples) - len(kept)


def song_of(sample: str) -> str:
    return sample.split("/")[1]


def recognize(binary: str, model: str, threads: int, images: list[Path], out: Path) -> str | None:
    """Runs the recognizer (writing OUT and its layout, OUT.json); returns an error message on failure."""
    out.parent.mkdir(parents=True, exist_ok=True)
    run = subprocess.run([binary, *map(str, images), "--model", model, "-o", str(out), "--layout", str(out.with_suffix(".json")),
                          "--threads", str(threads)], capture_output=True, text=True)
    return None if run.returncode == 0 else run.stderr.strip()[-200:]


def bench_systems(args: argparse.Namespace) -> None:
    samples, excluded = split_samples(args.split)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "scored_samples.txt").write_text("\n".join(samples) + "\n")

    def one(sample: str) -> Counter | str:
        if args.predictions:
            prediction = args.predictions / f"{sample}.musicxml"
        else:
            prediction = args.out / f"{sample}.musicxml"
            if error := recognize(args.binary, args.model, args.threads, [SCANNED / f"{sample}.png"], prediction):
                return error
        return score(parse_musicxml(SCANNED / f"{sample}.musicxml"), parse_musicxml(prediction))

    results = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        for sample, result in zip(samples, pool.map(one, samples)):
            if isinstance(result, str):
                print(f"{sample}: recognizer failed: {result}")
            else:
                results[sample] = result
    print(f"{excluded} samples left out as copies of training pieces")
    save_results(results, args.out / "results.json", song_of)
    report(results, "systems", show_worst=args.split != "test")


# --- pages -----------------------------------------------------------------------------

def song_systems(song: str) -> list[tuple[str, int, int]]:
    """(sample, IMSLP page, system on that page) of each annotated system of a song, in
    reading order."""
    mapping = yaml.safe_load((SOURCES / "corpus_to_imslp" / f"{song}.yaml").read_text())
    order = sorted(mapping.items(), key=lambda kv: (kv[1]["imslpPage"], kv[1]["imslpSystem"]))
    return [(f"samples/{sample}", place["imslpPage"], place["imslpSystem"]) for sample, place in order]


def song_document(song: str) -> dict:
    """Pages of a song's IMSLP document (each song is from one), with their images and
    piano system boxes."""
    mapping = yaml.safe_load((SOURCES / "corpus_to_imslp" / f"{song}.yaml").read_text())
    doc = next(iter(mapping.values()))["imslpDocument"].lstrip("#")
    return yaml.safe_load((SOURCES / "imslp_systems" / f"IMSLP{doc}.yaml").read_text())["pages"]


def song_truth(systems: list[tuple[str, int, int]]) -> list[NoteEvent]:
    """The song's annotated systems as one score: measures numbered on across systems."""
    events, offset = [], 0
    for sample, _, _ in systems:
        system_events = parse_musicxml(SCANNED / f"{sample}.musicxml")
        events += [NoteEvent(e.measure + offset, e.onset, e.midi, e.duration, e.staff) for e in system_events]
        offset += 1 + max((e.measure for e in system_events), default=-1)
    return events


def annotated_piano_events(prediction: Path, page_boxes: list[list[dict]]) -> list[NoteEvent]:
    """Recognized notes on staves inside an annotated piano system box of their page, with
    measures renumbered over the systems kept. page_boxes[i]: boxes of input page i + 1."""
    layout = json.loads(prediction.with_suffix(".json").read_text())["systems"]
    events = parse_musicxml(prediction)
    n_measures = 1 + max((e.measure for e in events), default=-1)
    kept: dict[int, int] = {}       # output measure -> renumbered measure
    piano_staves: dict[int, set[int]] = {}  # output measure -> staff numbers kept
    for i, system in enumerate(layout):
        end = layout[i + 1]["first_measure"] if i + 1 < len(layout) else n_measures
        boxes = page_boxes[system["page"] - 1]
        inside = {n for n, (top, bottom) in enumerate(system["staves"], 1)
                  if any(b["top"] <= (top + bottom) / 2 <= b["top"] + b["height"] for b in boxes)}
        if not inside:
            continue
        for m in range(system["first_measure"], end):
            kept[m] = len(kept)
            piano_staves[m] = inside
    return [NoteEvent(kept[e.measure], e.onset, e.midi, e.duration, e.staff)
            for e in events if e.measure in kept and e.staff in piano_staves[e.measure]]


def bench_pages(args: argparse.Namespace) -> None:
    samples, excluded = split_samples(args.split)
    songs = sorted(set(map(song_of, samples)))

    def one(song: str) -> Counter | str:
        systems = song_systems(song)
        pages = song_document(song)
        page_numbers = sorted({page for _, page, _ in systems})
        images = [SOURCES / "imslp_pngs" / pages[p]["image"] for p in page_numbers]
        # A page may also hold annotated systems of a neighbouring song; keep this song's.
        boxes = [[pages[p]["systems"][n - 1]["boundingBox"] for _, page, n in systems if page == p] for p in page_numbers]
        prediction = args.out / f"{song}.musicxml"
        error = recognize(args.binary, args.model, args.threads, images, prediction)
        return error or score(song_truth(systems), annotated_piano_events(prediction, boxes))

    results = {}
    with ThreadPoolExecutor(args.jobs) as pool:
        for song, result in zip(songs, pool.map(one, songs)):
            if isinstance(result, str):
                print(f"{song}: recognizer failed: {result}")
            else:
                results[song] = result
    print(f"{excluded} samples left out as copies of training pieces")
    save_results(results, args.out / "results.json", lambda song: song)
    report(results, "songs", show_worst=args.split != "test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["systems", "pages", "list"])
    ap.add_argument("--binary")
    ap.add_argument("--model")
    ap.add_argument("--predictions", type=Path, help="score these instead of recognizing (systems only)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    if args.mode == "list":
        print("\n".join(split_samples(args.split)[0]))
        return
    if not args.predictions and not (args.binary and args.model):
        ap.error("--binary and --model are needed unless --predictions is given")
    if args.out is None:
        ap.error("--out is needed")
    if args.predictions and args.mode != "systems":
        ap.error("--predictions works with systems only")
    (bench_systems if args.mode == "systems" else bench_pages)(args)


if __name__ == "__main__":
    main()
