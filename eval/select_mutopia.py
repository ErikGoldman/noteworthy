"""Chooses the Mutopia pieces of the benchmark: run once, its output is committed.

usage: select_mutopia.py --work DIR --out mutopia_pieces.txt [--workers W]

Candidates are single-file piano pieces by composers absent from ASAP and GrandStaff,
whose MIDI gives the printed measures (mutopia_truth.midi_matches_print). A
title match cannot prove a piece absent from PDMX (retitled copies pass, as Chopin's
Prelude 14 did), so a candidate is dropped when a training score shares much of its melody:
the top note of each onset, as intervals so that transpositions match too.

Output lines: split, path under ftp/, md5 of its PDF, md5 of its MIDI. A composer's pieces
share a split: dev holds COMPOSERS_SEEN, test only composers nobody has looked at.
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing as mp
import random
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from mutopia_truth import midi_events, midi_matches_print, onsets_on_grid, pickup_quarters
from score_events import NoteEvent, parse_musicxml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
FTP = RAW / "mutopia" / "repo" / "ftp"
SITE = "https://www.mutopiaproject.org/ftp"
TRAINING_SCORES = ROOT / "data" / "render_scores.txt"
COMPOSERS_IN_TRAINING = ("Bach", "Balakirev", "Beethoven", "Brahms", "Chopin", "Debussy", "Glinka", "Haydn", "Liszt",
                         "Mozart", "Prokofiev", "Rachmaninoff", "Ravel", "Schubert", "Schumann", "Scriabin", "Hummel",
                         "Joplin", "Scarlatti")
# Composers whose pieces had results or errors read while this benchmark was built.
COMPOSERS_SEEN = {
    "AdamA", "AlbenizIMF", "AscherJ", "BizetG", "BurgmullerJFF", "GottschalkLM", "HandelGF",
    "IlievGK", "KumarR", "MagnenatS", "MinkusLA", "SatieE", "SidwellA", "SousaJP",
    "Traditional", "VerdiG",
}
PIECES_PER_COMPOSER = 2
CANDIDATES = 44
GRAM = 8                 # intervals per melody fragment
COPY_SHARE = 0.10        # of a candidate's fragments in one score makes it a copy; copies ran from 0.26 up, 0.16-0.17 was unclear
SECONDS_BETWEEN_DOWNLOADS = 2

_candidate_grams: dict[str, set[tuple[int, ...]]] = {}


def melody_grams(events: list[NoteEvent]) -> set[tuple[int, ...]]:
    top: dict[tuple[int, object], int] = {}
    for e in events:
        key = (e.measure, e.onset)
        top[key] = max(top.get(key, 0), e.midi)
    pitches = [top[k] for k in sorted(top)]
    intervals = [b - a for a, b in zip(pitches, pitches[1:])]
    return {tuple(intervals[i:i + GRAM]) for i in range(len(intervals) - GRAM + 1)}


def candidates() -> list[str]:
    by_composer = defaultdict(list)
    for ly in sorted(FTP.rglob("*.ly")):
        text = ly.read_text(errors="replace")
        instrument = re.search(r'mutopiainstrument\s*=\s*"([^"]*)"', text)
        composer = ly.relative_to(FTP).parts[0]
        if (instrument and instrument[1] == "Piano" and ly.parent.name == ly.stem
                and not composer.startswith(COMPOSERS_IN_TRAINING) and midi_matches_print(text)):
            by_composer[composer].append(str(ly.parent.relative_to(FTP)))
    rng = random.Random(0)
    chosen = [p for pieces in by_composer.values() for p in rng.sample(pieces, min(PIECES_PER_COMPOSER, len(pieces)))]
    rng.shuffle(chosen)
    return sorted(set(chosen[:CANDIDATES]))


def fetch(piece: str, work: Path) -> tuple[Path, Path]:
    name = Path(piece).name
    paths = work / f"{name}-a4.pdf", work / f"{name}.mid"
    for path in paths:
        if not path.exists():
            subprocess.run(["curl", "-sfL", "-o", str(path), f"{SITE}/{piece}/{path.name}"], check=True)
            time.sleep(SECONDS_BETWEEN_DOWNLOADS)
    return paths


def shared_share(score: str) -> dict[str, float]:
    """Share of each candidate's fragments present in a training score, where any are."""
    try:
        grams = melody_grams(parse_musicxml(score))
    except Exception:
        return {}
    return {piece: len(g & grams) / len(g) for piece, g in _candidate_grams.items() if g & grams}


def set_candidates(grams: dict[str, set[tuple[int, ...]]]) -> None:
    global _candidate_grams
    _candidate_grams = grams


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    grams, files = {}, {}
    for piece in candidates():
        try:
            files[piece] = fetch(piece, args.work)
        except subprocess.CalledProcessError:
            print(f"{piece}: no PDF or MIDI", flush=True)
            continue
        ly = (FTP / piece / f"{Path(piece).name}.ly").read_text(errors="replace")
        events = midi_events(files[piece][1], pickup_quarters(ly))
        if onsets_on_grid(events):
            grams[piece] = melody_grams(events)
        else:
            print(f"{piece}: onsets off the grid", flush=True)
    print(f"{len(grams)} candidates fetched", flush=True)
    scores = TRAINING_SCORES.read_text().split()
    worst: dict[str, float] = defaultdict(float)
    with mp.Pool(args.workers, set_candidates, (grams,)) as pool:
        for done, shares in enumerate(pool.imap_unordered(shared_share, scores, chunksize=32), 1):
            for piece, share in shares.items():
                worst[piece] = max(worst[piece], share)
            if done % 2000 == 0:
                print(f"{done}/{len(scores)} training scores", flush=True)
    kept = [p for p in grams if grams[p] and worst[p] < COPY_SHARE]
    for piece in sorted(grams, key=lambda p: -worst[p]):
        print(f"{worst[piece]:.2f} {'copy' if piece not in kept else 'kept'} {piece}")
    lines = [f"{'dev' if p.split('/')[0] in COMPOSERS_SEEN else 'test'} {p} {md5(files[p][0])} {md5(files[p][1])}"
             for p in sorted(kept)]
    args.out.write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} pieces: {sum(l.startswith('dev') for l in lines)} dev")


if __name__ == "__main__":
    main()
