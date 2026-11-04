"""Re-rasterize a rendered dataset after a labeling or drawing change.

usage: relabel_dataset.py DIR [--workers K]

Labeling and rasterization (see omr.render.label_page) are redone from each page's
stored NNNNNN.svg.gz, so channel changes need no Verovio layout. Both the image
(NNNNNN.png) and the masks (NNNNNN.npz) are rewritten, since drawing fixes (e.g.
fonts) change the image too.
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from omr.render import label_page


def _relabel(svg_path: Path) -> str | None:
    stem = svg_path.parent / svg_path.name.removesuffix(".svg.gz")
    try:
        meta = json.loads(stem.with_suffix(".json").read_text())
        image, masks = label_page(gzip.decompress(svg_path.read_bytes()), meta["font"], meta["interline_px"])
    except Exception as e:  # noqa: BLE001 - report and continue with the other pages
        return f"{svg_path.name}: {type(e).__name__}: {e}"
    # Write-then-rename, so an interrupted run never leaves a half-written page.
    tmp = stem.with_name(stem.name + ".tmp")
    Image.fromarray(image).save(tmp.with_suffix(".png"))
    np.savez_compressed(tmp.with_suffix(".npz"), masks=masks)
    tmp.with_suffix(".png").replace(stem.with_suffix(".png"))
    tmp.with_suffix(".npz").replace(stem.with_suffix(".npz"))
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    args = ap.parse_args()

    pages = sorted(Path(args.dir).glob("*.svg.gz"))
    failures = 0
    with mp.Pool(args.workers) as pool:
        for done, error in enumerate(pool.imap_unordered(_relabel, pages, chunksize=8), 1):
            if error:
                failures += 1
                print(f"[fail] {error}", file=sys.stderr)
            if done % 500 == 0:
                print(f"{done}/{len(pages)} redone, {failures} failures", flush=True)
    print(f"done: {len(pages) - failures} pages redone, {failures} failures")


if __name__ == "__main__":
    main()
