"""Piece identity across score sources, so that other copies of a held-out piece (PDMX
re-uploads and arrangements, the same work in ASAP or OpenScore Lieder) stay out of
training.

Metadata is messy (PDMX composers are often missing, titles carry catalogue noise), so a
piece is the set of its title's distinctive words plus, when known, the composer's
surname. Two scores are the same piece when their word sets are equal and their
composers do not contradict each other. This errs towards false matches, which only cost
a few training scores, over missed ones, which leak test data into training.
"""
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# Words that vary between copies of the same piece without telling pieces apart.
_FILLER = {"op", "opus", "no", "nr", "nro", "num", "number", "in", "the", "a", "an", "for", "and", "of",
           "major", "minor", "dur", "moll", "flat", "sharp", "piano", "solo", "version", "arr", "arranged",
           "score", "sheet", "music", "easy", "simple"}


def _words(text: str) -> list[str]:
    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode().lower()
    return re.findall(r"[a-z]+|\d+", ascii_text)


def title_words(title: str) -> frozenset[str]:
    """Distinctive words of a title; single letters are mostly key names (C, F#)."""
    return frozenset(w for w in _words(title) if w not in _FILLER and len(w) > 1)


def surname(composer: str) -> str | None:
    """Last word of a composer's name, without dates or notes in parentheses."""
    words = [w for w in _words(re.sub(r"\(.*?\)", " ", composer)) if not w.isdigit()]
    return words[-1] if words and composer.strip() not in ("", "NA") else None


@dataclass(frozen=True)
class Piece:
    title: frozenset[str]
    composer: str | None

    def same_as(self, other: Piece) -> bool:
        composers_agree = self.composer is None or other.composer is None or self.composer == other.composer
        return bool(self.title) and self.title == other.title and composers_agree


class PieceIndex:
    """Pieces by title, for asking whether a score is a copy of any of them."""

    def __init__(self, pieces: list[Piece]):
        self._by_title: dict[frozenset[str], list[Piece]] = {}
        for p in pieces:
            self._by_title.setdefault(p.title, []).append(p)

    def contains(self, piece: Piece) -> bool:
        return any(piece.same_as(p) for p in self._by_title.get(piece.title, []))


def pdmx_pieces(pdmx_root: Path) -> dict[str, list[Piece]]:
    """Pieces of each PDMX score, by MXL file name (unique: it is a content hash). A PDMX
    row has both a title and a song name; either may identify the piece."""
    csv.field_size_limit(1 << 30)
    out = {}
    with (pdmx_root / "PDMX.csv").open() as f:
        for row in csv.DictReader(f):
            composer = surname(row["composer_name"])
            out[Path(row["mxl"]).name] = [Piece(title_words(t), composer) for t in {row["title"], row["song_name"]} if t != "NA"]
    return out


def pdmx_license_conflicts(pdmx_root: Path) -> set[str]:
    """MXL file names of scores whose public license disagrees with the one inside the file.
    PDMX's authors recommend leaving these out."""
    csv.field_size_limit(1 << 30)
    with (pdmx_root / "PDMX.csv").open() as f:
        return {Path(row["mxl"]).name for row in csv.DictReader(f) if row["license_conflict"] == "True"}


def asap_pieces(asap_root: Path) -> dict[str, list[Piece]]:
    """Pieces of each ASAP score, by score path relative to the ASAP root."""
    with (asap_root / "metadata.csv").open() as f:
        return {row["xml_score"]: [Piece(title_words(row["title"].replace("_", " ")), surname(row["composer"]))]
                for row in csv.DictReader(f)}
