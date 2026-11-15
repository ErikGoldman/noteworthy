"""The sequence model (train/seq) on whole pages: the recognizer's layout finds the systems,
each is cropped and transcribed, and their measures are joined into one score.

usage: seq_pages.py CHECKPOINT mutopia --binary NOTEWORTHY --model MODEL.onnx --out DIR
                    [--split dev|test] [--batch N]
       seq_pages.py CHECKPOINT olimpic --layouts DIR --out DIR [--split dev|test] [--batch N]
       seq_pages.py CHECKPOINT pages --pages DIR --binary NOTEWORTHY --model MODEL.onnx --out DIR
       seq_pages.py CHECKPOINT rendered --pages DIR --scores LIST --binary NOTEWORTHY --model MODEL.onnx
                    --out DIR [--count N] [--split dev|test]
       seq_pages.py CHECKPOINT score --inputs SCORE.pdf|PAGE.png... --binary NOTEWORTHY --model MODEL.onnx
                    --out SCORE.musicxml

Mutopia PDFs are rendered by the recognizer, so its normalized debug pages are cropped; OLiMPiC
scans are cropped as they are, with the layouts olimpic_benchmark.py pages left behind.
The model reads two-staff systems (piano_staff_pairs chooses them: the voice of a
song sits above the piano), at each of CROP_HEIGHT_RATIOS. The U-Net model is used only for finding systems. Results go
to DIR/results.json for scorecard.py, in the row's own directory:
  DIR=$RUN/mutopia or $RUN/olimpic-pages
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
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
from page_musicxml import finish_score, mark_new_system
from seq_predict import delinearize, load_model
from synthetic_benchmark import f1_counts, held_out_scores, split_scores

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from seq.data import collate, scale_to_height, to_tensor  # noqa: E402
from seq.lmx_tuplets import bar_fill, written_bar  # noqa: E402
from seq.meter import meter_tokens, without_meter  # noqa: E402
from seq.model import StaffSeq2Seq  # noqa: E402
from seq.ottava import sounding  # noqa: E402
from seq.staff_data import at_interline, at_staff_scale, collate_staves, collate_systems, cut_staves  # noqa: E402
from seq.staves import Staves  # noqa: E402
from seq.vocab import BOS, EOS, Vocab  # noqa: E402
from train_seq import VOCABULARY  # noqa: E402

MARGIN_INTERLINES = 7.0     # how far from the staves a crop may reach for high notes
SIDE_MARGIN_INTERLINES = 1.0
# White left of the brace, over crop height. A signature against the crop's edge reads as no
# key: training crops have a margin there, and on Polish this alone took keys of 3 and 4
# accidentals from 66 to 75 of 75. More adds nothing.
LEFT_MARGIN_HEIGHTS = 0.1
# Crop height over system height, tried in this order. 2.06 is OLiMPiC's crops, median over
# the dev songs: crops are scaled to one height, so a tighter crop shows larger staves than
# the model trained on. It helps most systems and wrecks a few, which the tight crop (0: no
# padding) reads well, so each system is read at both and the better reading kept.
CROP_HEIGHT_RATIOS = (2.06, 0.0)
INK = 128


@dataclass(frozen=True)
class SystemCrop:
    page: int                  # 1-based, in the order given to the recognizer
    top_staff: tuple[float, float]       # in the coordinates of the page as given
    bottom_staff: tuple[float, float]
    scale: float               # from those to the pixels of the page image being cropped


@dataclass(frozen=True)
class Crop:
    image: np.ndarray
    staves: Staves  # the system's staff centres and interline, in the crop's rows: StaffSeq2Seq's cut


def page_scales(layout: dict) -> dict[int, float]:
    """The layout is in input coordinates; debug pages are as the recognizer read them. Inferring
    this from staff heights drifts by up to a tenth, which moves a low system off its crop."""
    return {system["page"]: system["scale"] for system in layout["systems"]}


def piano_staff_pairs(staves: list[list[float]], piano_page: bool) -> list[tuple[list[float], list[float]]]:
    """The two-staff systems in one of the layout's systems. On a page of piano systems, more
    than two staves are piano systems the layout ran together, which a page without brackets
    invites. Elsewhere the piano is the last two: the voice of a song sits above it."""
    if piano_page and len(staves) % 2 == 0:
        return list(zip(staves[::2], staves[1::2]))
    return [(staves[-2], staves[-1])]


def without_duplicate_staves(systems: list[dict]) -> list[dict]:
    """The layout's systems without a staff found twice. Real staves never overlap; the layout can
    find one tilted staff as two chains side by side, a little apart in height, which its own
    duplicate test (one chain within the other's extent) misses (Polish 097, 106: the second copy
    in a system of its own, or both in one). Of staves covering over half of each other's height
    on a page, the one in the system with more staves stays; systems keep their order."""
    kept_spans: dict[int, list[tuple[float, float]]] = {}
    kept: dict[int, list[list[float]]] = {}
    for i in sorted(range(len(systems)), key=lambda i: -len(systems[i]["staves"])):
        spans = kept_spans.setdefault(systems[i]["page"], [])
        kept[i] = []
        for top, bottom in systems[i]["staves"]:
            if all(min(bottom, b) - max(top, t) <= (bottom - top) / 2 for t, b in spans):
                kept[i].append([top, bottom])
                spans.append((top, bottom))
    return [{**system, "staves": kept[i]} for i, system in enumerate(systems)]


def piano_systems(layout: dict, scales: dict[int, float]) -> list[SystemCrop]:
    systems = [s for s in without_duplicate_staves(layout["systems"]) if len(s["staves"]) >= 2]
    piano_pages = {s["page"] for s in systems if len(s["staves"]) == 2}
    return [SystemCrop(s["page"], tuple(top), tuple(bottom), scales[s["page"]])
            for s in systems for top, bottom in piano_staff_pairs(s["staves"], s["page"] in piano_pages)]


def emptiest_row(page: np.ndarray, start: float, end: float, aim: float | None = None) -> int:
    """The row with least ink between two heights, nearest `aim` (the middle unless given) among
    equals: notes can reach far from their staff, so no fixed distance separates two systems."""
    rows = np.arange(max(0, int(start)), min(page.shape[0], int(end) + 1))
    if len(rows) == 0:
        return max(0, min(page.shape[0], int(start)))
    ink = (page[rows] < INK).sum(axis=1)
    candidates = rows[ink == ink.min()]
    aim = (start + end) / 2 if aim is None else aim
    return int(candidates[np.abs(candidates - aim).argmin()])


def crop_system(page: np.ndarray, system: SystemCrop, above: float, below: float, height_ratio: float) -> Crop:
    """The system's band, kept off its neighbours (`above` and `below`: their nearest staff
    edges, in pixels of `page`), trimmed to the ink at the sides; with its staves' positions."""
    interline = (system.bottom_staff[1] - system.bottom_staff[0]) / 4 * system.scale
    first_line, last_line = system.top_staff[0] * system.scale, system.bottom_staff[1] * system.scale
    reach = MARGIN_INTERLINES * interline
    # Between two systems, aim at the middle of the whole gap, not of the reach: an 8va line
    # sits high above its staff, and the reach's middle cuts it off (Polish 023).
    top = emptiest_row(page, max(above, first_line - reach), first_line, (above + first_line) / 2 if above else None)
    bottom = emptiest_row(page, last_line, min(below, last_line + reach),
                          (last_line + below) / 2 if below < page.shape[0] else None)
    band = page[top:bottom]
    centres = tuple(sum(staff) / 2 * system.scale - top for staff in (system.top_staff, system.bottom_staff))
    columns = np.nonzero((band < INK).any(axis=0))[0]
    if len(columns) == 0:
        return Crop(band, Staves(centres, interline))
    margin = int(SIDE_MARGIN_INTERLINES * interline)
    trimmed = band[:, max(0, columns[0] - margin):columns[-1] + margin]
    tall = padded_to_height(trimmed, round(height_ratio * (last_line - first_line)))
    added_above = (tall.shape[0] - trimmed.shape[0]) // 2
    return Crop(with_left_margin(tall), Staves(tuple(c + added_above for c in centres), interline))


def padded_to_height(crop: np.ndarray, height: int) -> np.ndarray:
    """White above and below (the odd row below) rather than more of the page, which would bring
    in the neighbours."""
    missing = max(0, height - crop.shape[0])
    return cv2.copyMakeBorder(crop, missing // 2, missing - missing // 2, 0, 0, cv2.BORDER_CONSTANT, value=255)


def with_left_margin(crop: np.ndarray) -> np.ndarray:
    return cv2.copyMakeBorder(crop, 0, 0, round(LEFT_MARGIN_HEIGHTS * crop.shape[0]), 0, cv2.BORDER_CONSTANT, value=255)


def page_crops(pages: list[np.ndarray], systems: list[SystemCrop], height_ratio: float) -> list[Crop]:
    crops = []
    for i, system in enumerate(systems):
        page = pages[system.page - 1]
        same_page = lambda k: 0 <= k < len(systems) and systems[k].page == system.page
        above = systems[i - 1].bottom_staff[1] * system.scale if same_page(i - 1) else 0.0
        below = systems[i + 1].top_staff[0] * system.scale if same_page(i + 1) else float(page.shape[0])
        crops.append(crop_system(page, system, above, below, height_ratio))
    return crops


def model_input(model, crop: Crop):
    """A crop as the model reads it: scaled to its height, at staff scale with its staves, or for
    StaffSeq2Seq cut into staves, as in training (seq.staff_data)."""
    if isinstance(model, StaffSeq2Seq):
        staves = cut_staves(*at_staff_scale(crop.image, crop.staves), outward=model.staff_outward,
                            cut_between=model.staff_cut_between)
        return torch.stack([to_tensor(s) for s in staves])
    if model.interline_scaled:
        image, staves = at_interline(crop.image, crop.staves)
        return to_tensor(image), staves
    return to_tensor(scale_to_height(crop.image, model.height))


def transcribe(model, vocab: Vocab, crops: list[Crop], device, batch: int,
               bars: list[Fraction | None] | None = None) -> list[str]:
    """LMX of each crop; widest together, so a batch pads little. bars: each crop's meter in force,
    for a model that reads one (seq.meter)."""
    batched = (collate_staves if isinstance(model, StaffSeq2Seq) else collate_systems if model.interline_scaled
               else collate)
    order = sorted(range(len(crops)), key=lambda i: crops[i].image.shape[1] / crops[i].image.shape[0])
    out = [""] * len(crops)
    empty = torch.tensor([BOS, EOS])
    for start in range(0, len(order), batch):
        chosen = order[start:start + batch]
        images, widths, _ = batched([(model_input(model, crops[i]), empty) for i in chosen])
        starts = [vocab.encode(" ".join(meter_tokens(bars[i])))[1:-1] if bars and bars[i] else [] for i in chosen]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.generate(images.to(device), widths, starts=starts)
        for i, tokens in zip(chosen, predicted):
            out[i] = without_meter(vocab.decode(tokens))
    return out


def meters_in_force(lmx_of_systems: list[str]) -> list[Fraction | None]:
    """Each system's meter: its own time signature, or the last one read before it"""
    bars, in_force = [], None
    for lmx in lmx_of_systems:
        in_force = written_bar(lmx.split()) or in_force
        bars.append(in_force)
    return bars


def best_readings(model, vocab: Vocab, pages: list[np.ndarray], systems: list[SystemCrop], wanted, device,
                  batch: int) -> list[str]:
    """LMX of each wanted system: of its reading at each crop height, the one that fills its
    bars best, the earlier height among equals. Neighbours bound a crop, wanted or not."""
    def read(bars=None) -> list[str]:
        readings = []
        for ratio in CROP_HEIGHT_RATIOS:
            crops = [c for s, c in zip(systems, page_crops(pages, systems, ratio)) if wanted(s)]
            readings.append(transcribe(model, vocab, crops, device, batch, bars))
        return [max(reads, key=lambda lmx: bar_fill(sounding(lmx))) for reads in zip(*readings)]
    chosen = read()
    # A model that reads the meter reads again, given what the first reading found in force.
    return read(meters_in_force(chosen)) if model.meter_context else chosen


def measure_count(musicxml: str) -> int:
    part = ET.fromstring(musicxml).find("part")
    return 0 if part is None else len(part.findall("measure"))


def joined_events(lmx_of_systems: list[str], work: Path) -> list[NoteEvent]:
    """The systems as one score, measures numbered on across them."""
    events, offset, bar = [], 0, None
    work.mkdir(parents=True, exist_ok=True)
    for k, lmx in enumerate(lmx_of_systems):
        path = work / f"system{k}.musicxml"
        path.with_suffix(".lmx").write_text(lmx + "\n")
        bar = written_bar(lmx.split()) or bar
        musicxml = delinearize(lmx, bar)
        path.write_text(musicxml)
        events += [NoteEvent(e.measure + offset, e.onset, e.midi, e.duration, e.staff) for e in parse_musicxml(path)]
        offset += measure_count(musicxml)
    return events


def drop_repeated_signatures(part: ET.Element, in_force: dict[tuple, str]) -> None:
    """Removes clefs, keys and times equal to those in force: a system restates its clefs and
    key, which in one continuous score are not new symbols."""
    for attributes in part.iter("attributes"):
        for element in list(attributes):
            if element.tag not in ("clef", "key", "time"):
                continue
            slot = (element.tag, element.get("number"))
            content = ET.tostring(element, encoding="unicode").strip()
            if in_force.get(slot) == content:
                attributes.remove(element)
            in_force[slot] = content


def joined_musicxml(lmx_of_systems: list[str]) -> str:
    """The systems as one score. Each keeps its own <divisions>, which apply until replaced."""
    joined, in_force, bar = None, {}, None
    for lmx in lmx_of_systems:
        bar = written_bar(lmx.split()) or bar
        score = ET.fromstring(delinearize(lmx, bar))
        part = score.find("part")
        drop_repeated_signatures(part, in_force)
        if joined is None:
            joined, target = score, part
        else:
            measures = part.findall("measure")
            if measures:
                mark_new_system(measures[0])
            target.extend(measures)
    if joined is None:
        return ""
    for number, measure in enumerate(target.findall("measure"), 1):
        measure.set("number", str(number))
    finish_score(joined)
    return ET.tostring(joined, encoding="unicode", xml_declaration=True)


def transcribe_pages(args, model, vocab, device) -> None:
    """<page>.musicxml for each image of a directory, for a scorer outside this environment."""
    args.out.mkdir(parents=True, exist_ok=True)
    for image in sorted(args.pages.glob("*.png")):
        layout, pages = layout_of(args, [image], args.out / "work" / image.stem)
        systems = piano_systems(layout, page_scales(layout))
        lmx = best_readings(model, vocab, pages, systems, lambda _: True, device, args.batch)
        (args.out / f"{image.stem}.musicxml").write_text(joined_musicxml(lmx))
        print(image.stem, f"{len(systems)} systems", flush=True)


def transcribe_score(args, model, vocab, device) -> None:
    """One MusicXML score from a PDF or page images, in the order given. The layout is found
    afresh in a temporary directory: a cached one may belong to another input of the same name."""
    with tempfile.TemporaryDirectory() as work:
        layout, pages = layout_of(args, args.inputs, Path(work))
        systems = piano_systems(layout, page_scales(layout))
        lmx = best_readings(model, vocab, pages, systems, lambda _: True, device, args.batch)
    args.out.write_text(joined_musicxml(lmx))
    print(f"{len(systems)} systems: {args.out}")


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
        lmx = best_readings(model, vocab, pages, systems, lambda _: True, device, args.batch)
        source = (MUTOPIA / "repo" / "ftp" / piece / f"{name}.ly").read_text(errors="replace")
        truth = midi_events(MUTOPIA / "pieces" / f"{name}.mid", pickup_quarters(source))
        results[piece] = score(truth, joined_events(lmx, args.out / "work" / name / "systems"))
        print(piece, dict(results[piece].most_common(3)), flush=True)
    return results


def bench_rendered(args, model, vocab, device) -> dict[str, Counter]:
    """synthetic_benchmark.py's pages, truth and matching, so its row compares with the recognizer's."""
    scores = split_scores(held_out_scores(args.scores), args.count, args.split)
    results = {}
    for i, score_path in enumerate(scores):
        page, measures = args.pages / f"{i:03d}.png", args.pages / f"{i:03d}.measures"
        if not measures.exists():
            continue
        layout, pages = layout_of(args, [page], args.out / "work" / page.stem)
        systems = piano_systems(layout, page_scales(layout))
        lmx = best_readings(model, vocab, pages, systems, lambda _: True, device, args.batch)
        truth = [e for e in parse_musicxml(score_path) if e.measure < int(measures.read_text())]
        correct, predicted, gold = f1_counts(truth, joined_events(lmx, args.out / "work" / page.stem / "systems"))
        results[score_path.name] = Counter(correct=correct, pred=predicted, missed=gold - correct)
        print(page.stem, dict(results[score_path.name]), flush=True)
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
        lmx = best_readings(model, vocab, pages, all_systems, inside, device, args.batch)
        results[song] = score(song_truth(annotated), joined_events(lmx, args.out / "work" / song / "systems"))
        print(song, dict(results[song].most_common(3)), flush=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("source", choices=["mutopia", "olimpic", "pages", "rendered", "score"])
    ap.add_argument("--binary", help="mutopia: finds the systems")
    ap.add_argument("--model", help="mutopia: the recognizer's model")
    ap.add_argument("--layouts", type=Path, help="olimpic: <song>.json from olimpic_benchmark.py pages")
    ap.add_argument("--pages", type=Path, help="pages, rendered: a directory of page images")
    ap.add_argument("--inputs", type=Path, nargs="+", help="score: a PDF or page images, in order")
    ap.add_argument("--scores", help="rendered: the list given to synthetic_benchmark.py")
    ap.add_argument("--count", type=int, default=100, help="rendered: as for synthetic_benchmark.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    device = torch.device("cuda")
    vocab = Vocab.load(VOCABULARY)
    model = load_model(args.checkpoint, device)
    if args.source == "pages":
        transcribe_pages(args, model, vocab, device)
        return
    if args.source == "score":
        transcribe_score(args, model, vocab, device)
        return
    bench = {"mutopia": bench_mutopia, "olimpic": bench_olimpic, "rendered": bench_rendered}[args.source]
    results = bench(args, model, vocab, device)
    save_results(results, args.out / "results.json", lambda item: item)
    report(results, "pieces", show_worst=args.split != "test")


if __name__ == "__main__":
    main()
