"""The sequence model (train/seq) on whole pages: the recognizer's layout finds the systems,
each is cropped and transcribed, and their measures are joined into one score.

usage: seq_pages.py CHECKPOINT mutopia --binary NOTEWORTHY --model MODEL.onnx --out DIR
                    [--split dev|test] [--batch N]
       seq_pages.py CHECKPOINT olimpic --layouts DIR --out DIR [--split dev|test] [--batch N]

Mutopia PDFs are rendered by the recognizer, so its normalized debug pages are cropped; OLiMPiC
scans are cropped as they are, with the layouts olimpic_benchmark.py pages left behind.
The model reads two-staff systems, so a system's last two staves are taken (the voice of a
song sits above the piano). The U-Net model is used only for finding systems. Results go
to DIR/results.json for scorecard.py, in the row's own directory:
  DIR=$RUN/mutopia or $RUN/olimpic-pages
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from mutopia_benchmark import MUTOPIA, pieces
from mutopia_truth import midi_events, pickup_quarters
from note_scoring import report, score
from olimpic_benchmark import SOURCES, song_document, song_of, song_systems, song_truth, split_samples
from results_file import save_results
from score_events import NoteEvent, parse_musicxml
from seq_predict import delinearize, load_model

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from seq.data import collate, scale_to_height, to_tensor  # noqa: E402
from seq.vocab import BOS, EOS, Vocab  # noqa: E402
from train_seq import VOCABULARY  # noqa: E402

NORMALIZED_INTERLINE_PX = 20.0  # the U-Net's; see train/omr/dataset.py
MARGIN_INTERLINES = 7.0     # how far from the staves a crop may reach for high notes
SIDE_MARGIN_INTERLINES = 1.0
INK = 128


@dataclass(frozen=True)
class SystemCrop:
    page: int                  # 1-based, in the order given to the recognizer
    top_staff: tuple[float, float]       # in the coordinates of the page as given
    bottom_staff: tuple[float, float]
    scale: float               # from those to the pixels of the page image being cropped


def page_scales(layout: dict) -> dict[int, float]:
    """The layout is in input coordinates; debug pages are scaled to the normalized interline."""
    heights = defaultdict(list)
    for system in layout["systems"]:
        heights[system["page"]] += [bottom - top for top, bottom in system["staves"]]
    return {page: 4 * NORMALIZED_INTERLINE_PX / float(np.median(h)) for page, h in heights.items()}


def piano_systems(layout: dict, scales: dict[int, float]) -> list[SystemCrop]:
    return [SystemCrop(s["page"], tuple(s["staves"][-2]), tuple(s["staves"][-1]), scales[s["page"]])
            for s in layout["systems"] if len(s["staves"]) >= 2]


def emptiest_row(page: np.ndarray, start: float, end: float) -> int:
    """The row with least ink between two heights, nearest the middle among equals: notes
    can reach far from their staff, so no fixed distance separates two systems."""
    rows = np.arange(max(0, int(start)), min(page.shape[0], int(end) + 1))
    if len(rows) == 0:
        return max(0, min(page.shape[0], int(start)))
    ink = (page[rows] < INK).sum(axis=1)
    candidates = rows[ink == ink.min()]
    return int(candidates[np.abs(candidates - (start + end) / 2).argmin()])


def crop_system(page: np.ndarray, system: SystemCrop, above: float, below: float) -> np.ndarray:
    """The system's band, kept off its neighbours (`above` and `below`: their nearest staff
    edges, in pixels of `page`), trimmed to the ink at the sides."""
    interline = (system.bottom_staff[1] - system.bottom_staff[0]) / 4 * system.scale
    first_line, last_line = system.top_staff[0] * system.scale, system.bottom_staff[1] * system.scale
    reach = MARGIN_INTERLINES * interline
    top = emptiest_row(page, max(above, first_line - reach), first_line)
    bottom = emptiest_row(page, last_line, min(below, last_line + reach))
    band = page[top:bottom]
    columns = np.nonzero((band < INK).any(axis=0))[0]
    if len(columns) == 0:
        return band
    margin = int(SIDE_MARGIN_INTERLINES * interline)
    return band[:, max(0, columns[0] - margin):columns[-1] + margin]


def page_crops(pages: list[np.ndarray], systems: list[SystemCrop]) -> list[np.ndarray]:
    crops = []
    for i, system in enumerate(systems):
        page = pages[system.page - 1]
        same_page = lambda k: 0 <= k < len(systems) and systems[k].page == system.page
        above = systems[i - 1].bottom_staff[1] * system.scale if same_page(i - 1) else 0.0
        below = systems[i + 1].top_staff[0] * system.scale if same_page(i + 1) else float(page.shape[0])
        crops.append(crop_system(page, system, above, below))
    return crops


def transcribe(model, vocab: Vocab, crops: list[np.ndarray], device, batch: int) -> list[str]:
    """LMX of each crop; widest together, so a batch pads little."""
    order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1] / crops[i].shape[0])
    out = [""] * len(crops)
    empty = torch.tensor([BOS, EOS])
    for start in range(0, len(order), batch):
        chosen = order[start:start + batch]
        images, widths, _ = collate([(to_tensor(scale_to_height(crops[i])), empty) for i in chosen])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.generate(images.to(device), widths)
        for i, tokens in zip(chosen, predicted):
            out[i] = vocab.decode(tokens)
    return out


def measure_count(musicxml: str) -> int:
    part = ET.fromstring(musicxml).find("part")
    return 0 if part is None else len(part.findall("measure"))


def joined_events(lmx_of_systems: list[str], work: Path) -> list[NoteEvent]:
    """The systems as one score, measures numbered on across them."""
    events, offset = [], 0
    work.mkdir(parents=True, exist_ok=True)
    for k, lmx in enumerate(lmx_of_systems):
        path = work / f"system{k}.musicxml"
        path.with_suffix(".lmx").write_text(lmx + "\n")
        musicxml = delinearize(lmx)
        path.write_text(musicxml)
        events += [NoteEvent(e.measure + offset, e.onset, e.midi, e.duration, e.staff) for e in parse_musicxml(path)]
        offset += measure_count(musicxml)
    return events


def layout_of(args, inputs: list[Path], work: Path) -> tuple[dict, list[np.ndarray]]:
    """Systems found by the recognizer, and the pages they refer to, as it saw them."""
    work.mkdir(parents=True, exist_ok=True)
    layout, debug = work / "layout.json", work / "debug"
    if not layout.exists():
        subprocess.run([args.binary, *map(str, inputs), "--model", args.model, "-o", str(work / "v5.musicxml"), "--layout", str(layout),
                        "--debug", str(debug), "--threads", "6"], capture_output=True, check=True)
    names = sorted(debug.glob("page*_input.png"), key=lambda p: int(p.name[4:].split("_")[0]))
    return json.loads(layout.read_text()), [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in names]


def bench_mutopia(args, model, vocab, device) -> dict[str, Counter]:
    results = {}
    for piece in pieces(args.split):
        name = Path(piece).name
        layout, pages = layout_of(args, [MUTOPIA / "pieces" / f"{name}-a4.pdf"], args.out / "work" / name)
        systems = piano_systems(layout, page_scales(layout))
        lmx = transcribe(model, vocab, page_crops(pages, systems), device, args.batch)
        source = (MUTOPIA / "repo" / "ftp" / piece / f"{name}.ly").read_text(errors="replace")
        truth = midi_events(MUTOPIA / "pieces" / f"{name}.mid", pickup_quarters(source))
        results[piece] = score(truth, joined_events(lmx, args.out / "work" / name / "systems"))
        print(piece, dict(results[piece].most_common(3)), flush=True)
    return results


def bench_olimpic(args, model, vocab, device) -> dict[str, Counter]:
    samples, _ = split_samples(args.split)
    results = {}
    for song in sorted(set(map(song_of, samples))):
        annotated = song_systems(song)
        document = song_document(song)
        numbers = sorted({page for _, page, _ in annotated})
        images = [SOURCES / "imslp_pngs" / document[p]["image"] for p in numbers]
        boxes = [[document[p]["systems"][n - 1]["boundingBox"] for _, page, n in annotated if page == p] for p in numbers]
        layout = json.loads((args.layouts / f"{song}.json").read_text())
        pages = [cv2.imread(str(image), cv2.IMREAD_GRAYSCALE) for image in images]
        inside = lambda s: any(b["top"] <= sum(s.bottom_staff) / 2 <= b["top"] + b["height"] for b in boxes[s.page - 1])
        all_systems = piano_systems(layout, defaultdict(lambda: 1.0))
        crops = page_crops(pages, all_systems)
        kept = [c for s, c in zip(all_systems, crops) if inside(s)]
        lmx = transcribe(model, vocab, kept, device, args.batch)
        results[song] = score(song_truth(annotated), joined_events(lmx, args.out / "work" / song / "systems"))
        print(song, dict(results[song].most_common(3)), flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("source", choices=["mutopia", "olimpic"])
    ap.add_argument("--binary", help="mutopia: finds the systems")
    ap.add_argument("--model", help="mutopia: the recognizer's model")
    ap.add_argument("--layouts", type=Path, help="olimpic: <song>.json from olimpic_benchmark.py pages")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = torch.device("cuda")
    vocab = Vocab.load(VOCABULARY)
    model = load_model(args.checkpoint, vocab, device)
    results = (bench_mutopia if args.source == "mutopia" else bench_olimpic)(args, model, vocab, device)
    save_results(results, args.out / "results.json", lambda item: item)
    report(results, "pieces", show_worst=args.split != "test")


if __name__ == "__main__":
    main()
