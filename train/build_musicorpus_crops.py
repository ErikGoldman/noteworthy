"""Real handwritten piano systems from MusiCorpus 1.0 (CC BY-NC-SA 4.0), for the sequence model
(train_seq.py): each grand staff of a page tagged as piano music in modern notation, labelled
from its own MusicXML. It is cut from the page around its staves and every symbol annotated as
its own (up to 4 ledger lines away), with about a staff space of margin: the dataset's cut (Grandstaves/<staves>/image.jpg) follows
the staves and clips notes on ledger lines.

usage: build_musicorpus_crops.py MUSICORPUS_DIR --out DIR

Writes DIR/samples/<part>/<page>/<staves>.png and .lmx, and the lists:
  samples.train.txt          OmniOMR's train split
  samples.dev.txt            OmniOMR's validation split
  samples.dev-dolores.txt    Dolores' validation split: its transcriptions leave out the slurs
                             the pages draw (its README: expression slurs not annotated), so it
                             is not trained on, and its slur tokens say nothing; read its notes
and DIR/outcomes.json, what became of every grand staff. The test splits are sealed
(benchmark only, numbers only): never read here.

Left out, as the label could not say what the page draws: parts not named as a piano, octave
clefs (our clef tokens have no octave), hidden notes that are not rests, hidden whole-measure
rests (the transcriptions put one where the page draws a measure-repeat sign), and labels with
a token outside our vocabulary or of a length the model does not take. The transcriptions do not
say which tuplet numbers are drawn; the symbol annotations do (tuplet3 and the like, beside the
groups): with none drawn, every number is labelled hidden, and with some, which ones is not
known here, so the grand staff is left out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from build_seq_dataset import MAX_TOKENS, linearize, piano_part
from build_smashcima_crops import MIN_TOKENS, has_hidden_note, is_hidden
from seq.vocab import Vocab
from train_seq import VOCABULARY

PARTS = ("UFAL.OmniOMR", "CVC.Dolores")
SEALED = "test"
TUPLET_NUMBER = re.compile(r"^tuplet\d+$")
PIANO_NAME = re.compile(r"^\s*(piano|klavír)", re.IGNORECASE)
# Of the grand staff's height: two staves and the gap between, about 11 staff spaces.
MARGIN = 0.09
# How far its symbols may widen the cut beyond its staves: notes on 4 ledger lines. Some symbols
# (a long slur, a dynamic) reach the next part's staff, which would then fill half the crop.
REACH = 0.36
LISTS = {("UFAL.OmniOMR", "train"): "train", ("UFAL.OmniOMR", "validation"): "dev",
         ("CVC.Dolores", "validation"): "dev-dolores"}


def page_splits(part_dir: Path) -> dict[str, str]:
    return {page: split for split, pages in json.loads((part_dir / "splits.json").read_text()).items()
            for page in pages}


def is_piano_page(page: Path) -> bool:
    metadata = json.loads((page / "metadata.json").read_text())
    return metadata.get("notation") == "CWMN" and metadata.get("notation_complexity") == "pianoform"


def has_hidden_measure_rest(part: ET.Element) -> bool:
    return any(is_hidden(n) and n.find("rest[@measure='yes']") is not None for n in part.iter("note"))


def cut_from_page(page: Path, staff: Path) -> np.ndarray:
    """The grand staff and all its symbols, with a margin, from the page image"""
    x, y, width, height = json.loads((page / "subdivisions.image.json").read_text())["Grandstaves"][staff.name]["bbox"]
    boxes = [a["bbox"] for a in json.loads((staff / "coco-object-detection.json").read_text())["annotations"]]
    margin, reach = MARGIN * height, REACH * height
    left = x + min([0] + [b[0] for b in boxes]) - margin
    top = y + max(-reach, min([0] + [b[1] for b in boxes])) - margin
    right = x + max([width] + [b[0] + b[2] for b in boxes]) + margin
    bottom = y + min(height + reach, max([height] + [b[1] + b[3] for b in boxes])) + margin
    image = cv2.imread(str(page / "image.jpg"), cv2.IMREAD_GRAYSCALE)
    rows, columns = image.shape
    return image[max(0, int(top)):min(rows, int(np.ceil(bottom))), max(0, int(left)):min(columns, int(np.ceil(right)))]


def drawn_tuplet_numbers(staff: Path) -> int:
    coco = json.loads((staff / "coco-object-detection.json").read_text())
    numbers = {c["id"] for c in coco["categories"] if TUPLET_NUMBER.match(c["name"])}
    return sum(a["category_id"] in numbers for a in coco["annotations"])


def hide_tuplet_numbers(part: ET.Element) -> None:
    for tuplet in part.iter("tuplet"):
        if tuplet.get("type") == "start":
            tuplet.set("show-number", "none")


def label(staff: Path, vocabulary: set[str]) -> tuple[str | None, str]:
    """The grand staff's LMX, or None and why it is left out"""
    score = ET.parse(staff / "transcription.musicxml").getroot()
    part = piano_part(score)
    if part is None:
        return None, "not two staves"
    name = score.findtext(f".//score-part[@id='{part.get('id')}']/part-name") or ""
    if not PIANO_NAME.match(name):
        return None, "not a piano part"
    if part.find(".//clef-octave-change") is not None:
        return None, "octave clef"
    if has_hidden_note(part):
        return None, "hidden note"
    if has_hidden_measure_rest(part):
        return None, "hidden measure rest"
    drawn = drawn_tuplet_numbers(staff)
    if drawn == 0:
        hide_tuplet_numbers(part)
    lmx = linearize(part)
    tokens = lmx.split()
    groups, hidden = tokens.count("tuplet:start"), tokens.count("tuplet:hidden")
    if groups and not (drawn == groups and hidden == 0 or drawn == 0 and hidden == groups):
        return None, "tuplet numbers drawn on some groups"
    if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
        return None, "label length"
    if not vocabulary.issuperset(tokens):
        return None, "token outside the vocabulary"
    return lmx, "kept"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("musicorpus", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if (args.out / "outcomes.json").exists():
        print(f"{args.out} is built; a changed builder needs a fresh directory")
        return
    vocabulary = set(Vocab.load(VOCABULARY).tokens)
    partial = args.out.with_name(args.out.name + ".partial")
    shutil.rmtree(partial, ignore_errors=True)
    lists: dict[str, list[str]] = {name: [] for name in LISTS.values()}
    outcomes: Counter = Counter()
    for part in PARTS:
        splits = page_splits(args.musicorpus / part)
        for page_name, split in sorted(splits.items()):
            if split == SEALED or (part, split) not in LISTS:
                continue
            page = args.musicorpus / part / page_name
            if not is_piano_page(page):
                continue
            for staff in sorted((page / "Grandstaves").glob("*")):
                lmx, outcome = label(staff, vocabulary)
                outcomes[f"{part} {split}: {outcome}"] += 1
                if lmx is None:
                    continue
                name = f"samples/{part}/{page_name}/{staff.name}"
                (partial / name).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(partial / f"{name}.png"), cut_from_page(page, staff))
                (partial / f"{name}.lmx").write_text(lmx + "\n")
                lists[LISTS[part, split]].append(name)
    for list_name, names in lists.items():
        (partial / f"samples.{list_name}.txt").write_text("".join(f"{n}\n" for n in names))
    (partial / "outcomes.json").write_text(json.dumps(dict(sorted(outcomes.items())), indent=1) + "\n")
    os.replace(partial, args.out)
    for line, count in sorted(outcomes.items()):
        print(f"{count:4d}  {line}")


if __name__ == "__main__":
    main()
