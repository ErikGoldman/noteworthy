"""List the scores to render for training: mostly solo piano (one part, grand staff).

usage: list_scores.py PDMX_ROOT ASAP_ROOT OUT_DIR

Writes to OUT_DIR:
  pdmx_piano.txt      grand-staff piano scores from PDMX (the benchmarks draw on these)
  render_scores.txt   what build_dataset.py renders: the piano scores, ASAP, and a sample
                      of non-piano PDMX scores for layout variety
  dropped_copies.txt  scores left out because they are another copy of a held-out piece

Held-out scores (omr.held_out) stay in render_scores.txt; training splits them off as
validation pages. Other copies of the same pieces (omr.pieces) are dropped, so neither
validation nor the benchmarks measure memorized music.
"""
from __future__ import annotations

import csv
import multiprocessing as mp
import random
import re
import sys
import zipfile
from pathlib import Path

from omr.held_out import is_held_out
from omr.pieces import PieceIndex, asap_pieces, pdmx_pieces

MIN_NOTES_PER_BAR = 3
OTHER_SAMPLE = 4000


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


def pdmx_scores(root: Path) -> tuple[list[Path], list[Path]]:
    """(grand-staff piano scores, sample of other scores) from PDMX's deduplicated subset."""
    csv.field_size_limit(1 << 30)
    with (root / "PDMX.csv").open() as f:
        rows = [r for r in csv.DictReader(f)
                if r["subset:deduplicated"] == "True" and r["subset:all_valid"] == "True"
                and float(r["notes_per_bar"] or 0) >= MIN_NOTES_PER_BAR]
    candidates = [(root / r["mxl"]).resolve() for r in rows if r["tracks"] in ("0", "0-0")]
    with mp.Pool() as pool:
        keep = pool.map(_is_grand_staff_piano, candidates, chunksize=256)
    piano = [p for p, k in zip(candidates, keep) if k]
    piano_set = set(piano)
    others = [p for p in ((root / r["mxl"]).resolve() for r in rows) if p not in piano_set]
    random.Random(0).shuffle(others)
    return piano, others[:OTHER_SAMPLE]


def drop_copies_of_held_out(scores: list[Path], pieces_of: dict[Path, list]) -> tuple[list[Path], list[Path]]:
    """(kept, dropped): training-side scores that share a piece with a held-out score go."""
    held_out = PieceIndex([p for s in scores if is_held_out(s) for p in pieces_of[s]])
    kept, dropped = [], []
    for s in scores:
        copy = not is_held_out(s) and any(held_out.contains(p) for p in pieces_of[s])
        (dropped if copy else kept).append(s)
    return kept, dropped


def write_list(path: Path, scores: list[Path]) -> None:
    partial = path.with_suffix(".partial")
    partial.write_text("".join(f"{s}\n" for s in scores))
    partial.replace(path)


def main() -> None:
    pdmx_root, asap_root, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve(), Path(sys.argv[3])
    piano, others = pdmx_scores(pdmx_root)
    asap = sorted(asap_root.glob("**/xml_score.musicxml"))
    by_mxl_name = pdmx_pieces(pdmx_root)
    by_asap_path = asap_pieces(asap_root)
    pieces_of = {s: by_mxl_name.get(s.name, []) for s in piano + others}
    pieces_of |= {s: by_asap_path.get(str(s.relative_to(asap_root)), []) for s in asap}

    kept, dropped = drop_copies_of_held_out(piano + asap + others, pieces_of)
    dropped_set = set(dropped)
    out.mkdir(parents=True, exist_ok=True)
    write_list(out / "dropped_copies.txt", dropped)
    write_list(out / "pdmx_piano.txt", [s for s in piano if s not in dropped_set])
    write_list(out / "render_scores.txt", kept)  # written last: its presence means the lists are complete
    print(f"{len(piano)} piano, {len(asap)} ASAP, {len(others)} other scores; "
          f"{sum(map(is_held_out, kept))} held out; {len(dropped)} copies of held-out pieces dropped")


if __name__ == "__main__":
    main()
