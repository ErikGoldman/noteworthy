"""Render a synthetic training set of labeled score pages.

usage: build_dataset.py --scores LIST.txt --out DIR --pages N [--workers K]

LIST.txt holds one MusicXML/MXL path per line. Each job picks a random score and
random layout options and renders one page. Output per page:
  NNNNNN.png    grayscale page image
  NNNNNN.npz    packed channel masks ("masks") (H, W, ceil(C/8)) uint8
  NNNNNN.json   render metadata (source, font, interline, options)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import signal
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from omr.render import random_layout_options, render_score_pages

BASE_INTERLINE_PX = 20.0
JOB_TIMEOUT_S = 120


class _Timeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise _Timeout()


def _render_job(args: tuple[int, str, Path]) -> tuple[int, str | None]:
    index, score_path, out_dir = args
    rng = random.Random(index)
    options = random_layout_options(rng)
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.alarm(JOB_TIMEOUT_S)
    try:
        pages = render_score_pages(score_path, options, BASE_INTERLINE_PX, max_pages=1, rng=rng)
    except Exception as e:  # noqa: BLE001 - a bad score must not kill the build
        return index, f"{score_path}: {type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
    if not pages:
        return index, f"{score_path}: verovio could not load"
    page = pages[0]
    stem = out_dir / f"{index:06d}"
    Image.fromarray(page.image).save(stem.with_suffix(".png"))
    np.savez_compressed(stem.with_suffix(".npz"), masks=page.masks)
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
        for done, (index, error) in enumerate(pool.imap_unordered(_render_job, jobs), 1):
            if error:
                failures += 1
                print(f"[fail] {error}", file=sys.stderr)
            if done % 100 == 0:
                print(f"{done}/{len(jobs)} rendered, {failures} failures", flush=True)
    print(f"done: {len(jobs) - failures} pages, {failures} failures")


if __name__ == "__main__":
    main()
