"""Render a synthetic training set of labeled score pages.

usage: build_dataset.py --scores LIST.txt --out DIR --pages N [--workers K]

LIST.txt holds one MusicXML/MXL path per line. Each job picks a random score and
random layout options and renders one page. Output per page:
  NNNNNN.png    grayscale page image
  NNNNNN.npz    packed channel masks ("masks") (H, W, ceil(C/8)) uint8
  NNNNNN.json   render metadata (source, font, interline, options)
  NNNNNN.svg.gz the laid-out page; relabel_dataset.py regenerates the masks from it
"""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from omr.render import random_layout_options, render_score_pages

BASE_INTERLINE_PX = 20.0
# Some scores make Verovio loop inside C++, where no signal can interrupt it. The parent
# stops waiting for such a job after this long; its worker stays stuck until the pool
# is terminated at the end, costing one worker per hang.
HUNG_JOB_S = 300


def _render_job(args: tuple[int, str, Path]) -> tuple[int, str | None]:
    index, score_path, out_dir = args
    rng = random.Random(index)
    options = random_layout_options(rng)
    try:
        pages = render_score_pages(score_path, options, BASE_INTERLINE_PX, max_pages=1, rng=rng)
    except Exception as e:  # noqa: BLE001 - a bad score must not kill the build
        return index, f"{score_path}: {type(e).__name__}: {e}"
    if not pages:
        return index, f"{score_path}: verovio could not load"
    page = pages[0]
    stem = out_dir / f"{index:06d}"
    Image.fromarray(page.image).save(stem.with_suffix(".png"))
    np.savez_compressed(stem.with_suffix(".npz"), masks=page.masks)
    stem.with_suffix(".svg.gz").write_bytes(gzip.compress(page.svg))
    stem.with_suffix(".json").write_text(json.dumps({
        "source": score_path, "font": page.font, "interline_px": page.interline_px, "options": options,
    }))
    return index, None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pages", type=int, required=True)
    ap.add_argument("--workers", type=int, default=mp.cpu_count())
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    scores = [line.strip() for line in Path(args.scores).read_text().splitlines() if line.strip()]
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pick = random.Random(args.seed)
    jobs = [(args.seed * 10_000_000 + i, pick.choice(scores), out_dir) for i in range(args.pages)
            if not (out_dir / f"{args.seed * 10_000_000 + i:06d}.json").exists()]

    failures = 0
    # "spawn", not fork: Verovio keeps its font resources in C++ static state that
    # pool workers forked during recycling cannot use, so every job after the first
    # worker restart fails with "font resources are not available".
    with mp.get_context("spawn").Pool(args.workers, maxtasksperchild=20) as pool:
        pending = [(job, pool.apply_async(_render_job, (job,))) for job in jobs]
        for done, (job, result) in enumerate(pending, 1):
            try:
                _, error = result.get(timeout=HUNG_JOB_S)
            except mp.TimeoutError:
                error = f"{job[1]}: hung for {HUNG_JOB_S}s"
            if error:
                failures += 1
                print(f"[fail] {error}", file=sys.stderr)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} rendered, {failures} failures", flush=True)
    print(f"done: {len(jobs) - failures} pages, {failures} failures")


if __name__ == "__main__":
    main()
