"""End-to-end benchmark on rendered pages of held-out scores.

usage: synthetic_benchmark.py --scores LIST.txt --binary NOTEWORTHY --model MODEL.onnx --out DIR
                              [--count N] [--degrade] [--renderer verovio|musescore]
                              [--split dev|test]

Scores are restricted to held-out ones (omr.held_out), which the network never trained
on; copies of the same pieces are kept out of training too (see train/list_scores.py). Page 1 of each score is rendered, recognized by the C++
tool, and compared with the source MusicXML on the measures of that page.

Training pages all come from Verovio, so the musescore renderer (a different engraver
and default font) is the better proxy for real-world printed music.

dev is the first N held-out scores, the pages developed against. test is the next N
that share no piece with them, held out like Night Album (look at its number, never at
its errors), so give it its own output directory.

Renders are cached in the output directory (NNN.png plus NNN.measures), so re-running
after a recognizer change only re-runs recognition; delete the directory to re-render.
"""
from __future__ import annotations

import argparse
import io
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from pathlib import Path

import cairosvg
import numpy as np
from lxml import etree
from PIL import Image

from results_file import save_results
from score_events import parse_musicxml

import musescore

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from omr.augment import capture_degradation  # noqa: E402
from omr.held_out import is_held_out  # noqa: E402
from omr.pieces import PieceIndex, pdmx_pieces  # noqa: E402
from omr.render import seeded_toolkit  # noqa: E402

ROOT_DATA = Path(os.environ.get("NW_DATA", Path(__file__).resolve().parent.parent / "data"))
INTERLINE_PX = 24.0
UNITS_PER_INTERLINE = 180.0  # Verovio default unit (9) * 20
MUSESCORE_SPATIUM_MM = 1.75  # default interline
MUSESCORE_DPI = round(INTERLINE_PX * 25.4 / MUSESCORE_SPATIUM_MM)


def held_out_scores(score_list: str | Path) -> list[Path]:
    """Held-out scores of a list, in list order."""
    return [Path(line) for line in Path(score_list).read_text().split("\n") if line and is_held_out(Path(line))]


def split_scores(scores: list[Path], count: int, split: str) -> list[Path]:
    dev = scores[:count]
    if split == "dev":
        return dev
    pieces_of = pdmx_pieces(ROOT_DATA / "raw" / "pdmx")
    pieces = lambda score: pieces_of.get(score.name, [])
    in_dev = PieceIndex([p for score in dev for p in pieces(score)])
    return [s for s in scores[count:] if not any(in_dev.contains(p) for p in pieces(s))][:count]


def render_verovio(score: Path) -> tuple[np.ndarray, int]:
    """Page 1 as grayscale, and how many measures it holds."""
    tk = seeded_toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "breaks": "auto", "header": "none", "footer": "none"})
    if not tk.loadFile(str(score)):
        raise RuntimeError("verovio could not load")
    svg = tk.renderToSVG(1)
    root = etree.fromstring(svg.encode())
    measures = sum(1 for g in root.iter("{http://www.w3.org/2000/svg}g") if g.get("class", "").split()[:1] == ["measure"])
    inner = root.find("{http://www.w3.org/2000/svg}svg")
    width = float(inner.get("viewBox").split()[2]) * INTERLINE_PX / UNITS_PER_INTERLINE
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=int(width), background_color="white")
    return np.asarray(Image.open(io.BytesIO(png)).convert("L")), measures


def render_musescore(scores: dict[int, Path], work: Path) -> dict[int, tuple[np.ndarray, int] | str]:
    """Page 1 of each score via one MuseScore batch job (a process start costs seconds)."""
    conversions = [(score, [work / f"{i:03d}.png", work / f"{i:03d}.mpos"]) for i, score in scores.items()]
    musescore.run_batch(conversions, work / "job.json", ["-r", str(MUSESCORE_DPI)])
    out: dict[int, tuple[np.ndarray, int] | str] = {}
    for i in scores:
        page, positions = work / f"{i:03d}-1.png", work / f"{i:03d}.mpos"
        if not page.exists() or not positions.exists():
            out[i] = "skip (musescore could not render)"
            continue
        # MuseScore writes transparent pages.
        rgba = Image.open(page).convert("RGBA")
        flat = Image.new("RGBA", rgba.size, "white")
        flat.alpha_composite(rgba)
        measures = sum(1 for e in etree.parse(str(positions)).iter("element") if e.get("page") == "0")
        out[i] = (np.asarray(flat.convert("L")), measures)
    return out


def prepare_pages(scores: list[Path], count: int, out: Path, renderer: str, degrade: bool) -> dict[int, int | str]:
    """Renders (or reuses) NNN.png per score; returns measures on the page, or why it was skipped."""
    counts: dict[int, int | str] = {}
    todo: dict[int, Path] = {}
    for i in range(min(count, len(scores))):
        count_file = out / f"{i:03d}.measures"
        if count_file.exists():
            counts[i] = int(count_file.read_text())
        else:
            todo[i] = scores[i]
    if renderer == "musescore":
        with tempfile.TemporaryDirectory(dir=out) as work:
            rendered = render_musescore(todo, Path(work))
    else:
        rendered = {}
        for i, score in todo.items():  # Verovio is not thread-safe, so this stays serial
            try:
                rendered[i] = render_verovio(score)
            except RuntimeError as e:
                rendered[i] = f"skip ({e})"
    for i, result in rendered.items():
        if isinstance(result, str):
            counts[i] = result
            continue
        image, measures = result
        if degrade:
            image = capture_degradation(np.random.default_rng(i), image, INTERLINE_PX)
        Image.fromarray(image).save(out / f"{i:03d}.png")
        (out / f"{i:03d}.measures").write_text(str(measures))
        counts[i] = measures
    return dict(sorted(counts.items()))


def f1_counts(gt, pred) -> tuple[int, int, int]:
    key = lambda e: (e.measure, e.onset, e.midi, e.duration)
    tp = sum((Counter(map(key, gt)) & Counter(map(key, pred))).values())
    return tp, len(pred), len(gt)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--binary", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--threads", type=int, default=0, help="inference threads per recognizer (0 = all cores)")
    ap.add_argument("--renderer", choices=["verovio", "musescore"], default="verovio")
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = split_scores(held_out_scores(args.scores), args.count, args.split)

    measure_counts = prepare_pages(scores, args.count, out, args.renderer, args.degrade)

    def recognize(i: int) -> str | tuple[int, int, int, int]:
        n_measures = measure_counts[i]
        if isinstance(n_measures, str):
            return n_measures
        png, xml = out / f"{i:03d}.png", out / f"{i:03d}.musicxml"
        run = subprocess.run([args.binary, str(png), "--model", args.model, "-o", str(xml), "--threads", str(args.threads)], capture_output=True, text=True)
        if run.returncode != 0:
            return f"recognizer failed: {run.stderr.strip()[-200:]}"
        gt = [e for e in parse_musicxml(scores[i]) if e.measure < n_measures]
        return (n_measures, *f1_counts(gt, parse_musicxml(xml)))

    total_tp = total_pred = total_gt = 0
    results = {}
    indices = sorted(measure_counts)
    with ThreadPoolExecutor(args.jobs) as pool:
        for i, result in zip(indices, pool.map(recognize, indices)):
            if isinstance(result, str):
                print(f"{i:03d} {result}")
                continue
            n_measures, tp, n_pred, n_gt = result
            total_tp, total_pred, total_gt = total_tp + tp, total_pred + n_pred, total_gt + n_gt
            results[scores[i].name] = Counter(correct=tp, pred=n_pred, missed=n_gt - tp)
            f1 = 2 * tp / max(1, n_pred + n_gt)
            print(f"{i:03d} measures={n_measures:3d} gt={n_gt:4d} pred={n_pred:4d} F1={f1:.3f}  {scores[i].name}", flush=True)
    save_results(results, out / "results.json", lambda name: name)
    print(f"overall F1 = {2 * total_tp / max(1, total_pred + total_gt):.3f} "
          f"(P={total_tp / max(1, total_pred):.3f} R={total_tp / max(1, total_gt):.3f})")


if __name__ == "__main__":
    main()
