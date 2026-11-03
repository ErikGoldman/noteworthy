"""Select PDMX scores for rendering: mostly solo piano (one part, grand staff).

usage: select_pdmx.py PDMX_ROOT OUT_PIANO.txt OUT_OTHER.txt
OUT_OTHER gets a sample of non-piano scores, used sparingly for layout variety.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import random
import re
import sys
import zipfile
from pathlib import Path

MIN_NOTES_PER_BAR = 3
OTHER_SAMPLE = 5000


def _score_xml(mxl_path: Path) -> str | None:
    try:
        with zipfile.ZipFile(mxl_path) as z:
            name = next(n for n in z.namelist() if n.endswith((".xml", ".musicxml")) and not n.startswith("META-INF"))
            return z.read(name).decode("utf-8", errors="replace")
    except (zipfile.BadZipFile, StopIteration, OSError):
        return None


def _is_grand_staff_piano(mxl_path: Path) -> bool:
    xml = _score_xml(mxl_path)
    return xml is not None and len(re.findall(r"<score-part\s+id=", xml)) == 1 and "<staves>2</staves>" in xml


def main() -> None:
    root, out_piano, out_other = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
    csv.field_size_limit(10**9)
    rows = [r for r in csv.DictReader((root / "PDMX.csv").open())
            if r["subset:deduplicated"] == "True" and r["subset:all_valid"] == "True"
            and float(r["notes_per_bar"] or 0) >= MIN_NOTES_PER_BAR]
    paths = [(root / r["mxl"]).resolve() for r in rows if r["tracks"] in ("0", "0-0")]
    with mp.Pool() as pool:
        keep = pool.map(_is_grand_staff_piano, paths, chunksize=256)
    piano = [str(p) for p, k in zip(paths, keep) if k]
    piano_set = set(piano)
    others = [str((root / r["mxl"]).resolve()) for r in rows]
    others = [p for p in others if p not in piano_set]
    random.Random(0).shuffle(others)
    Path(out_piano).write_text("\n".join(piano) + "\n")
    Path(out_other).write_text("\n".join(others[:OTHER_SAMPLE]) + "\n")
    print(f"{len(piano)} grand-staff piano scores, {min(OTHER_SAMPLE, len(others))} others")


if __name__ == "__main__":
    main()
