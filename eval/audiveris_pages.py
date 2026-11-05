"""Audiveris on page images, for comparison.

usage: audiveris_pages.py AUDIVERIS OUT_DIR PAGE.png [...]

Audiveris rejects a sheet with 9 pixels between staff lines as having no staves, and asks
for 300 DPI, so pages are enlarged first (Polish's are 1050 wide). Writes
OUT_DIR/<page>.musicxml. Audiveris can split a page into movements; they are joined back,
part by part, as the page's one score.
A page it cannot read, or that passes PAGE_TIMEOUT_S, leaves no file and scores as failed.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2

UPSCALE = 2
PAGE_TIMEOUT_S = 600
JOBS = 4


def musicxml_of(mxl: Path) -> ET.Element:
    with zipfile.ZipFile(mxl) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".xml") and not n.startswith("META-INF"))
        return ET.fromstring(archive.read(name))


def joined(movements: list[ET.Element]) -> ET.Element:
    """The first movement with the later ones' measures appended; parts pair up in order, and
    a later movement's extra parts are dropped, so scoring sees their notes as missed."""
    first, *later = movements
    for movement in later:
        for target, part in zip(first.findall("part"), movement.findall("part")):
            target.extend(part.findall("measure"))
    return first


def read_page(audiveris: str, out: Path, page: Path) -> str:
    target = out / f"{page.stem}.musicxml"
    if target.exists():
        return f"{page.stem}: done before"
    with tempfile.TemporaryDirectory() as work_dir:
        work = Path(work_dir)
        image = cv2.imread(str(page))
        enlarged = work / page.name
        cv2.imwrite(str(enlarged), cv2.resize(image, None, fx=UPSCALE, fy=UPSCALE, interpolation=cv2.INTER_CUBIC))
        try:
            subprocess.run([audiveris, "-batch", "-export", "-output", str(work), "--", str(enlarged)],
                           capture_output=True, timeout=PAGE_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return f"{page.stem}: timed out"
        scores = sorted(work.rglob("*.mxl"))
        if not scores:
            return f"{page.stem}: no score"
        ET.ElementTree(joined([musicxml_of(score) for score in scores])).write(target, encoding="utf-8", xml_declaration=True)
        return f"{page.stem}: {len(scores)} score(s)"


def main() -> None:
    audiveris, out, pages = sys.argv[1], Path(sys.argv[2]), [Path(p) for p in sys.argv[3:]]
    out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(JOBS) as pool:
        for result in pool.map(lambda page: read_page(audiveris, out, page), pages):
            print(result, flush=True)


if __name__ == "__main__":
    main()
