"""End-to-end benchmark on rendered pages of held-out scores.

usage: synthetic_benchmark.py --scores LIST.txt --binary NOTEWORTHY --model MODEL.onnx --out DIR [--count N] [--degrade]

Scores are restricted to the validation split used in training (same hash rule), so the
network has never seen them. Page 1 of each score is rendered with Verovio, recognized
by the C++ tool, and compared with the source MusicXML on the measures of that page.
"""
from __future__ import annotations

import argparse
import io
import subprocess
import sys
import zlib
from collections import Counter
from pathlib import Path

import cairosvg
import numpy as np
import verovio
from lxml import etree
from PIL import Image

from score_events import parse_musicxml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from omr.augment import capture_degradation  # noqa: E402

VAL_FRACTION = 0.05  # must match train.py
INTERLINE_PX = 24.0
UNITS_PER_INTERLINE = 180.0  # Verovio default unit (9) * 20


def is_validation(source: str) -> bool:
    return (zlib.crc32(source.encode()) % 1000) / 1000.0 < VAL_FRACTION


def render_first_page(score: Path, out_png: Path, degrade: bool, seed: int) -> int:
    """Renders page 1; returns how many measures it holds."""
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": 2100, "pageHeight": 2970, "breaks": "auto", "header": "none", "footer": "none"})
    if not tk.loadFile(str(score)):
        raise RuntimeError("verovio could not load")
    svg = tk.renderToSVG(1)
    root = etree.fromstring(svg.encode())
    measures = sum(1 for g in root.iter("{http://www.w3.org/2000/svg}g") if g.get("class", "").split()[:1] == ["measure"])
    inner = root.find("{http://www.w3.org/2000/svg}svg")
    width = float(inner.get("viewBox").split()[2]) * INTERLINE_PX / UNITS_PER_INTERLINE
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=int(width), background_color="white")
    image = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
    if degrade:
        image = capture_degradation(np.random.default_rng(seed), image, INTERLINE_PX)
    Image.fromarray(image).save(out_png)
    return measures


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
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = [Path(l.strip()) for l in Path(args.scores).read_text().splitlines() if l.strip() and is_validation(l.strip())]
    total_tp = total_pred = total_gt = 0
    for i, score in enumerate(scores[: args.count]):
        png, xml = out / f"{i:03d}.png", out / f"{i:03d}.musicxml"
        try:
            n_measures = render_first_page(score, png, args.degrade, i)
        except RuntimeError as e:
            print(f"{i:03d} skip ({e})")
            continue
        run = subprocess.run([args.binary, str(png), "--model", args.model, "-o", str(xml)], capture_output=True, text=True)
        if run.returncode != 0:
            print(f"{i:03d} recognizer failed: {run.stderr.strip()[-200:]}")
            continue
        gt = [e for e in parse_musicxml(score) if e.measure < n_measures]
        pred = parse_musicxml(xml)
        tp, n_pred, n_gt = f1_counts(gt, pred)
        total_tp, total_pred, total_gt = total_tp + tp, total_pred + n_pred, total_gt + n_gt
        f1 = 2 * tp / max(1, n_pred + n_gt)
        print(f"{i:03d} measures={n_measures:3d} gt={n_gt:4d} pred={n_pred:4d} F1={f1:.3f}  {score.name}")
    print(f"overall F1 = {2 * total_tp / max(1, total_pred + total_gt):.3f} "
          f"(P={total_tp / max(1, total_pred):.3f} R={total_tp / max(1, total_gt):.3f})")


if __name__ == "__main__":
    main()
