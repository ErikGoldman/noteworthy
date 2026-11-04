"""MUSCIMA++ v2 pages and their dev and test splits.

Each annotated page is a CVC-MUSCIMA image: 50 writers each copied the same 20 pieces,
and MUSCIMA++ annotates 7 copies of every piece. Its writer-independent test set holds
one copy of each piece by 7 writers who appear nowhere else. Those pages are kept out of
training (convert_muscima.py) and scored by eval/handwritten_benchmark.py. The pieces
themselves are in training, in other writers' hands; that cannot be avoided. Its
writer-dependent test set (writers whose other pages are in training) is our dev set.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_DOCUMENT_NAME = re.compile(r"CVC-MUSCIMA_W-(\d+)_N-(\d+)_D-ideal")


@dataclass(frozen=True)
class Page:
    annotation: Path  # MuNG XML
    image: Path       # the undistorted scan, white ink on black
    writer: int
    piece: int

    @property
    def name(self) -> str:
        return f"w{self.writer:02d}_p{self.piece:03d}"


def pages(root: Path) -> list[Page]:
    """All annotated pages; root holds v2.0/ (MuNG) and CvcMuscima-Distortions/ (images)."""
    out = []
    for xml_path in sorted((root / "v2.0" / "data" / "annotations").glob("CVC-MUSCIMA_*.xml")):
        m = _DOCUMENT_NAME.search(xml_path.name)
        writer, piece = int(m.group(1)), int(m.group(2))
        image = root / "CvcMuscima-Distortions" / "ideal" / f"w-{writer:02d}" / "image" / f"p{piece:03d}.png"
        out.append(Page(xml_path, image, writer, piece))
    return out


def _listed(root: Path, spec: str) -> list[Page]:
    names = set((root / "v2.0" / "specifications" / spec).read_text().split())
    return [p for p in pages(root) if p.annotation.stem in names]


def dev_pages(root: Path) -> list[Page]:
    """MUSCIMA++'s writer-dependent test set: pages kept out of training, by writers whose
    other pages are in it. For development; errors may be looked at."""
    return _listed(root, "testset-dependent.txt")


def test_pages(root: Path) -> list[Page]:
    """MUSCIMA++'s writer-independent test set, held out like Night Album: numbers only."""
    return _listed(root, "testset-independent.txt")


def training_pages(root: Path) -> list[Page]:
    held_out = {p.annotation for p in dev_pages(root) + test_pages(root)}
    return [p for p in pages(root) if p.annotation not in held_out]
