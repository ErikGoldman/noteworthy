"""SMB's zero-shot pages: its two-staff piano pages (texture Pianoform) of pieces no training source
holds (docs/EVALUATION_PROTOCOL.md, Overlap; decided 2026-11-19, the user: only these are scored).
A piece is held, and every page of it left out, if any of its pages shares COPY_SHARE of its melody
fragments with one training piece, or if it is a GrandStaff training piece by name: GrandStaff
crops only some of a piece's bars, so a page of bars never cropped shares nothing by melody.
Reads only SMB's gold; names and shares are all it writes.

Melody: select_mutopia.melody_grams, the top note of each onset as runs of 8 intervals, against
the render list (PDMX, ASAP) and GrandStaff's crops (one crop of each bar range, in whichever key
it comes: the fragments are intervals; some pieces have no untransposed crop). A score that will
not parse is counted, an SMB page that will not parse leaves its piece out, and both are listed.

usage: smb_benchmark.py PAGES.txt SHARES.tsv [WORKERS]   (in .venv-musicdiff, for the **kern reader)
"""
from __future__ import annotations

import json
import multiprocessing as mp
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import converter21
import music21
import pyarrow.parquet as pq

from score_events import NoteEvent, parse_musicxml
from select_mutopia import COPY_SHARE, TRAINING_SCORES, melody_grams

ROOT = Path(__file__).resolve().parents[1]
SMB = ROOT / "data" / "raw" / "smb" / "data" / "test-00000-of-00001.parquet"
GRANDSTAFF = ROOT / "data" / "raw" / "olimpic" / "grandstaff-lmx"
TEXTURE = "Pianoform"

_index: dict[tuple, list[str]] = {}  # fragment: SMB pages holding it


def kern_events(kern: str) -> list[NoteEvent]:
    """Through MusicXML, so the page's events come from the parser the training scores go through."""
    score = music21.converter.parse(kern, format="humdrum", forceSource=True)
    with tempfile.TemporaryDirectory() as work:
        path = Path(work) / "page.musicxml"
        score.write("musicxml", fp=path)
        return parse_musicxml(path)


def smb_piece(original_score: str) -> str:
    """haydn_sonata15_1_p0 -> haydn_sonata15_1"""
    return re.sub(r"_p\d+$", "", original_score)


def name_key(composer: str, piece: str) -> str:
    return re.sub(r"[^a-z0-9]", "", f"{composer.split('-')[0]}/{piece}".lower())


def grandstaff_training_names() -> set[str]:
    """GrandStaff's training pieces (composer/collection/piece), as SMB's composer_piece would name them."""
    pieces = {tuple(Path(sample).parts[:3]) for sample in (GRANDSTAFF / "samples.train.txt").read_text().split()}
    return {name_key(composer, piece) for composer, _, piece in pieces}


def smb_pages() -> tuple[dict[str, str], dict[str, set], list[str]]:
    """(page: piece, page: fragments, pages that would not parse), two-staff piano pages only."""
    rows = pq.read_table(SMB, columns=["id", "page", "page_texture", "original_score"]).to_pylist()
    pieces, grams, unparsed = {}, {}, []
    for row in rows:
        if row["page_texture"] != TEXTURE:
            continue
        pieces[row["id"]] = smb_piece(row["original_score"])
        try:
            grams[row["id"]] = melody_grams(kern_events(json.loads(row["page"])["kern"]))
        except Exception:  # noqa: BLE001 - listed, and its piece left out
            unparsed.append(row["id"])
    return pieces, grams, unparsed


def training_pieces() -> list[tuple[str, list[str]]]:
    """(name, files): a render-list score is one file; a GrandStaff piece is one crop per bar range."""
    pieces = [(path, [path]) for path in TRAINING_SCORES.read_text().split()]
    by_range: dict[tuple[str, str], str] = {}
    for crop in sorted(GRANDSTAFF.glob("*/*/*/*.musicxml")):
        by_range.setdefault((str(crop.parent.relative_to(GRANDSTAFF)), crop.stem[crop.stem.index("m-"):]), str(crop))
    by_piece = defaultdict(list)
    for (piece, _), crop in by_range.items():
        by_piece[piece].append(crop)
    return pieces + sorted(by_piece.items())


def piece_grams(files: list[str]) -> set:
    """Each file's fragments, pooled: crops number their bars from 0, so their events cannot be
    pooled before melody_grams orders them. A fragment across two crops is left out."""
    return set().union(*(melody_grams(parse_musicxml(f)) for f in files))


def set_index(index: dict[tuple, list[str]]) -> None:
    global _index
    _index = index
    converter21.register()


def shared_counts(piece: tuple[str, list[str]]) -> tuple[str, Counter | None]:
    """How many of each SMB page's fragments this training piece holds; None if it will not parse."""
    name, files = piece
    try:
        grams = piece_grams(files)
    except Exception:  # noqa: BLE001 - counted by the caller
        return name, None
    return name, Counter(page for fragment in grams & _index.keys() for page in _index[fragment])


def largest_shares(grams: dict[str, set], workers: int) -> tuple[dict[str, tuple[float, str]], list[str]]:
    index = defaultdict(list)
    for page, fragments in grams.items():
        for fragment in fragments:
            index[fragment].append(page)
    best: dict[str, tuple[float, str]] = {page: (0.0, "") for page in grams}
    pieces, unparsed = training_pieces(), []
    with mp.Pool(workers, set_index, (dict(index),)) as pool:
        for done, (name, counts) in enumerate(pool.imap_unordered(shared_counts, pieces, chunksize=16), 1):
            if counts is None:
                unparsed.append(name)
                continue
            for page, n in counts.items():
                best[page] = max(best[page], (n / len(grams[page]), name))
            if done % 5000 == 0:
                print(f"{done}/{len(pieces)} training pieces", flush=True)
    print(f"training pieces {len(pieces)}, unparsed {len(unparsed)}: {' '.join(unparsed) or '-'}", flush=True)
    return best, unparsed


def main() -> None:
    pages_out, shares_out = Path(sys.argv[1]), Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    converter21.register()
    pieces, grams, unparsed = smb_pages()
    print(f"SMB {TEXTURE}: {len(pieces)} pages, {len(unparsed)} unparsed: {' '.join(unparsed) or '-'}", flush=True)
    best, _ = largest_shares(grams, workers)
    gs_names = grandstaff_training_names()
    reasons: dict[str, str] = {}  # piece: why it is held
    for page, piece in pieces.items():
        if page in unparsed:
            reasons.setdefault(piece, "a page unparsed")
        elif best[page][0] >= COPY_SHARE:
            reasons[piece] = "melody"
    for piece in set(pieces.values()):
        if piece not in reasons and name_key(*piece.split("_", 1)) in gs_names:
            reasons[piece] = "GrandStaff by name"
    kept = sorted(page for page, piece in pieces.items() if piece not in reasons)
    pages_out.write_text("".join(f"{page}\n" for page in kept))
    shares_out.write_text("page\tpiece\tlargest_share\tnearest\tverdict\n" + "".join(
        f"{page}\t{piece}\t{best[page][0]:.3f}\t{best[page][1]}\t{reasons.get(piece, 'kept')}\n"
        if page in best else f"{page}\t{piece}\t-\t-\t{reasons[piece]}\n" for page, piece in sorted(pieces.items())))
    held = Counter(reasons.values())
    print(f"kept {len(kept)} pages of {len(set(pieces.values())) - len(reasons)} pieces; "
          f"held {len(reasons)} pieces: " + ", ".join(f"{n} by {why}" for why, n in sorted(held.items())))


if __name__ == "__main__":
    main()
