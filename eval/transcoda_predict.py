"""Transcoda (btrkeks/transcoda-59M-zeroshot-v1) on page images, for comparison.

usage: transcoda_predict.py MODEL_DIR --out DIR [--joined NAME] PAGE.png [...]

Runs in its own environment, .venv-transcoda (requirements-transcoda.txt): the model's
code needs torchtune, which needs an older torch than ours. MODEL_DIR is a download of
the model repository; loading it executes that repository's Python.

Writes DIR/<page>.krn, the model's **kern, and DIR/<page>.musicxml converted by music21,
the form our benchmarks score. --joined also writes DIR/NAME.musicxml: the pages in the
given order as one score, for a multi-page ground truth. Without the grammar constraint
the model can break **kern's spine rules, and the 2048-token limit can cut a line short;
only the part of a page before the first such error is converted.

Decoding is the model card's headline setting: beam search of width 3, repetition
penalty 1.1, at most 2048 tokens, without the project repository's grammar constraint.
Preprocessing follows the card: width 1050, white-padded or top-cropped to height 1485.
"""
from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import music21
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

WIDTH, HEIGHT = 1050, 1485


def preprocess(path: Path) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((WIDTH, max(1, int(image.height * WIDTH / image.width))), Image.BILINEAR)
    pixels = np.array(image)[:HEIGHT]
    pixels = np.concatenate([pixels, np.full((HEIGHT - pixels.shape[0], WIDTH, 3), 255, pixels.dtype)])
    return ((torch.from_numpy(pixels).permute(2, 0, 1).float() / 255 - 0.5) / 0.5)[None]


def transcribe(model, tokenizer, page: Path) -> str:
    with torch.no_grad():
        out = model.generate(pixel_values=preprocess(page), image_sizes=torch.tensor([[HEIGHT, WIDTH]]),
                             max_length=2048, do_sample=False, num_beams=3, length_penalty=1.0,
                             repetition_penalty=1.1)
    return tokenizer.decode(out[0], skip_special_tokens=True)


def spines_after(fields: list[str]) -> int | None:
    """How many spines follow a line of spine operations, None if its merges are invalid."""
    count, i = 0, 0
    while i < len(fields):
        if fields[i] == "*v":
            merged = next((j for j in range(i, len(fields)) if fields[j] != "*v"), len(fields)) - i
            if merged < 2:
                return None
            count, i = count + 1, i + merged
        else:
            count, i = count + (2 if fields[i] == "*^" else 1), i + 1
    return count


def valid_prefix(lines: list[str]) -> tuple[list[str], int]:
    """The lines before the first break of the spine rules, and the spines open after them."""
    spines = len(lines[0].split("\t"))
    for k, line in enumerate(lines):
        fields = line.split("\t")
        following = spines_after(fields) if line.startswith("*") else spines
        if len(fields) != spines or following is None or "" in fields:
            return lines[:k], spines
        spines = following
    return lines, spines


def as_humdrum(kern: str) -> str:
    """The model emits the spines' body only: no **kern header, no *- terminators."""
    lines = [line for line in kern.strip().split("\n") if line]
    body, spines = valid_prefix(lines)
    return "\n".join(["\t".join(["**kern"] * len(lines[0].split("\t"))), *body, "\t".join(["*-"] * spines)])


def to_score(kern: str) -> music21.stream.Score:
    try:
        return music21.converter.parse(as_humdrum(kern), format="humdrum")
    except Exception as e:  # music21 raises many kinds on malformed **kern
        print(f"  music21 could not parse it: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
        return music21.stream.Score()


def write_musicxml(score: music21.stream.Score, path: Path) -> None:
    """Without the file, the page scores as empty; the **kern beside it is still good."""
    try:
        score.write("musicxml", path)
    except Exception as e:  # music21 raises many kinds on what it parsed from malformed **kern
        print(f"  no MusicXML for {path.stem}: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)


def joined(scores: list[music21.stream.Score]) -> music21.stream.Score:
    """The pages' parts concatenated measure by measure; a page with fewer parts leaves the
    other parts' measures out, so measure alignment sees them as missed."""
    parts = max((len(s.parts) for s in scores), default=0)
    out = music21.stream.Score([music21.stream.Part() for _ in range(parts)])
    for score in scores:
        for target, part in zip(out.parts, score.parts):
            for measure in part.getElementsByClass(music21.stream.Measure):
                target.append(copy.deepcopy(measure))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("pages", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--joined")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True).eval()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.model)
    args.out.mkdir(parents=True, exist_ok=True)
    scores = []
    for page in args.pages:
        kern_path = args.out / f"{page.stem}.krn"
        if not kern_path.exists():
            started = time.time()
            kern_path.write_text(transcribe(model, tokenizer, page) + "\n")
            print(f"{page.stem}: {time.time() - started:.0f} s", flush=True)
        score = to_score(kern_path.read_text())
        write_musicxml(score, args.out / f"{page.stem}.musicxml")
        scores.append(score)
    if args.joined:
        joined(scores).write("musicxml", args.out / f"{args.joined}.musicxml")


if __name__ == "__main__":
    main()
