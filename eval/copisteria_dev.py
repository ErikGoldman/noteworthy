"""Copisteria (copista-28m, defaults) on the dev benchmarks, for comparison with ours.

usage: copisteria_dev.py systems --out DIR [--split dev]
       copisteria_dev.py mutopia --pages RUN/mutopia/work --out DIR [--split dev]

Writes predictions for the benchmarks' own --predictions scoring: DIR/samples/.../<crop>.musicxml
for olimpic_benchmark.py systems, DIR/<piece>.musicxml for mutopia_benchmark.py. Mutopia reads
the page images seq_pages.py left in its work directory, so both systems see the same pages;
copisteria reads a page at a time, so a piece's pages are joined part by part.

OLiMPiC is read from its piano crops. Whole scans would need the piano staves of each system,
which copisteria's MusicXML does not locate; a crop is a page of one system to it, without the
neighbours its evidence model reads, so this may understate it.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from mutopia_benchmark import pieces
from olimpic_benchmark import SCANNED, split_samples

ROOT = Path(__file__).resolve().parent.parent
REPOSITORY = ROOT / "data" / "raw" / "copisteria" / "src"
PYTHON = ROOT / ".venv-copisteria" / "bin" / "python"
JOB_TIMEOUT_S = 3600
JOBS = 4


def read_images(images: list[Path], out: Path) -> None:
    """Writes out/<image>.musicxml for each image; its detection cache lands beside the copies."""
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    copies = [shutil.copy(image, out) for image in images]
    # In one process, so the models load once; it finds models/ from its repository's root.
    run = subprocess.run([PYTHON, "-m", "copisteria.pipeline", *map(str, copies), "--out", str(out)], cwd=REPOSITORY,
                         capture_output=True, text=True, timeout=JOB_TIMEOUT_S)
    if run.returncode != 0:
        print(f"{out}: {run.stderr.strip()[-200:]}", flush=True)
    for copy in copies:
        Path(copy).unlink()


def joined(pages: list[Path]) -> ET.ElementTree:
    first, *later = [ET.parse(page) for page in pages]
    for page in later:
        for target, part in zip(first.getroot().findall("part"), page.getroot().findall("part")):
            target.extend(part.findall("measure"))
    return first


def read_systems(args: argparse.Namespace) -> None:
    samples, _ = split_samples(args.split)
    by_directory = defaultdict(list)
    for sample in samples:
        by_directory[Path(sample).parent].append(SCANNED / f"{sample}.png")
    with ThreadPoolExecutor(JOBS) as pool:
        list(pool.map(lambda item: read_images(item[1], args.out / item[0]), by_directory.items()))


def page_number(path: Path) -> int:
    return int(path.name[4:].split("_")[0])


def read_piece(name: str, args: argparse.Namespace) -> None:
    work = args.out / "work" / name
    read_images(list((args.pages / name / "debug").glob("page*_input.png")), work)
    pages = sorted(work.glob("*.musicxml"), key=page_number)
    if pages:
        joined(pages).write(args.out / f"{name}.musicxml", encoding="utf-8", xml_declaration=True)


def read_mutopia(args: argparse.Namespace) -> None:
    with ThreadPoolExecutor(JOBS) as pool:
        list(pool.map(lambda piece: read_piece(Path(piece).name, args), pieces(args.split)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("benchmark", choices=["systems", "mutopia"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pages", type=Path)
    ap.add_argument("--split", choices=["dev"], default="dev")
    args = ap.parse_args()
    {"systems": read_systems, "mutopia": read_mutopia}[args.benchmark](args)


if __name__ == "__main__":
    main()
