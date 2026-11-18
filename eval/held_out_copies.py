"""Shortlists the held-out scores that may have a copy in a training set, for review by eye.

usage: held_out_copies.py --out SHORTLIST.tsv [--workers W]    (in .venv, from eval/)

Held-out scores (omr.held_out, among data/render_scores.txt) are never trained on, but the same
music can reach training another way: under another title in PDMX, as another edition in ASAP,
or in GrandStaff, OLiMPiC or MusiCorpus, which list_scores.py never compares. Here each held-out
score is compared with every training unit, and a pair is shortlisted when any rule fires:

  runs16    COPY_SHARE of either side's melody fragments of 16 intervals are in the other
            (select_mutopia.melody_grams: the top note of each onset, as intervals, so
            transpositions match). Precise, but an arrangement breaks the runs.
  count8    MIN_SHARED fragments of 8 intervals in common, with a training unit of at least
            MIN_UNIT_FRAGMENTS (a tiny score of scales or repeated notes matches everything).
  skeleton  COPY_SHARE of either side's 8-interval fragments of the melody skeleton (the top
            note starting on each quarter beat: the notes an arrangement adds between beats are
            dropped), both sides having at least MIN_SKELETON of them.
  title     the same normalised title (normal_title).

The shortlist is not the verdict. Pairs were looked at by eye (titles, and both melodies aligned
at their longest common runs): every runs16 pair, and for each score with no copy found by then,
the strongest pair of each other rule. Each decision and its reason is in
held_out_copies_review.tsv; held_out_copies.txt holds the pairs judged copies, and is what
is_unseen reads. Evaluation sets select with is_unseen in place of is_held_out. Training is not
changed: a held-out score with a copy stays out of it, as its copy is already in.

Training units, as trained on:
  pdmx, asap    each score of render_scores.txt that is not held out
  grandstaff    a movement: the MusicXML of its GrandStaff-LMX train samples (every transposition)
  olimpic       a song: the MusicXML of its OLiMPiC synthetic train systems (the piano part)
  musicorpus    a manuscript: its pages build_musicorpus_crops.py trains on

Output: one line per shortlisted pair, tab-separated: the score's key (omr.held_out.score_key),
its path under the data directory, the training source, the training unit, the rules that
fired, and their evidence; sorted. held_out_copies.txt has the same columns.
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import re
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from functools import cache
from pathlib import Path

import yaml

from score_events import NoteEvent, parse_musicxml
from select_mutopia import COPY_SHARE, melody_grams

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "train"))
from omr.held_out import is_held_out, score_key  # noqa: E402

DATA = Path(os.environ.get("NW_DATA", Path(__file__).resolve().parent.parent / "data"))
PDMX = DATA / "raw" / "pdmx"
OLIMPIC = DATA / "raw" / "olimpic"
MUSICORPUS = DATA / "raw" / "musicorpus" / "musicorpus_v1.0"
COPIES = Path(__file__).with_name("held_out_copies.txt")

# Rule thresholds, measured on 2026-11-18 against the 92 held-out scores with a copy confirmed by
# eye (held_out_copies_review.tsv). Runs of 8 at COPY_SHARE (select_mutopia's rule, made for whole
# pieces) shortlisted 425 of 1023 held-out scores, mostly false: short hymns share generic
# 8-interval runs. Of the 92, the rules catch: runs16 59 (27 false scores flagged), count8 75 (44),
# skeleton 73 (96), title 16 (12); count8 alone catches 10, title alone 7, skeleton alone 5,
# runs16 alone none. count8's recall is flattered: 22 of the 92 were first found by a count of 15.
# MIN_SHARED, MIN_UNIT_FRAGMENTS and MIN_SKELETON were chosen, not tuned; the 40 keeps tiny scale
# sheets, which match everything, out of count8.
LONG_GRAM = 16
SHORT_GRAM = 8
MIN_SHARED = 15
MIN_UNIT_FRAGMENTS = 40
MIN_SKELETON = 20
# Words that say how a score is arranged or catalogued, not what it is.
TITLE_FILLER = frozenset(
    "a an and arr arranged arrangement by easy for from in of op opus no nr piano simple solo the "
    "transcription transcribed version ver".split())

Grams = set[tuple[int, ...]]
Features = dict[str, Grams]  # by kind: "long", "short", "skeleton"
Unit = tuple[str, str, list[Path]]  # source, name, files
Match = tuple[list[str], str]  # rules that fired, their evidence

_held_out_index: dict[str, dict[tuple[int, ...], list[int]]] = {}
_held_out_sizes: dict[str, list[int]] = {}


@cache
def keys_with_copies() -> frozenset[str]:
    return frozenset(line.split("\t")[0] for line in COPIES.read_text().splitlines() if line)


def is_unseen(source: Path) -> bool:
    """Held out, and no copy of it is in a training set."""
    return is_held_out(source) and score_key(source) not in keys_with_copies()


def render_list_units(scores: list[Path]) -> list[Unit]:
    return [("asap" if "asap" in s.parts else "pdmx", str(s.relative_to(DATA)), [s]) for s in scores if not is_held_out(s)]


def grouped_units(source: str, root: Path, names: list[str], unit_of) -> list[Unit]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for name in names:
        groups[unit_of(name)].append(root / f"{name}.musicxml")
    return [(source, unit, files) for unit, files in sorted(groups.items())]


def grandstaff_units() -> list[Unit]:
    root = OLIMPIC / "grandstaff-lmx"
    names = (root / "samples.train.txt").read_text().split()
    return grouped_units("grandstaff", root, names, lambda name: "/".join(name.split("/")[:3]))


def olimpic_units() -> list[Unit]:
    root = OLIMPIC / "olimpic-1.0-synthetic"
    names = (root / "samples.train.txt").read_text().split()
    return grouped_units("olimpic", root, names, lambda name: name.split("/")[1])


def musicorpus_units() -> list[Unit]:
    # Imported here: the builder brings in torch, which is_unseen's users should not pay for.
    from build_musicorpus_crops import LISTS, PARTS, SEALED, is_piano_page, page_splits
    units = []
    for part in PARTS:
        trained = [page for page, split in sorted(page_splits(MUSICORPUS / part).items())
                   if split != SEALED and LISTS.get((part, split)) == "train" and is_piano_page(MUSICORPUS / part / page)]
        units += grouped_units("musicorpus", MUSICORPUS / part, [f"{page}/transcription" for page in trained],
                               lambda name: f"{part}/{name.split('_')[0]}")
    return units


def on_beats(events: list[NoteEvent]) -> list[NoteEvent]:
    return [e for e in events if e.onset.denominator == 1]


def features_of(files: list[Path]) -> Features | str:
    """The files' melody fragments of each kind, or why there are none."""
    try:
        scores = [parse_musicxml(f) for f in files]
    except Exception as e:  # noqa: BLE001 -- the caller prints and counts it
        return f"{type(e).__name__}: {e}"
    features = {"long": set().union(*(melody_grams(s, LONG_GRAM) for s in scores)),
                "short": set().union(*(melody_grams(s, SHORT_GRAM) for s in scores)),
                "skeleton": set().union(*(melody_grams(on_beats(s), SHORT_GRAM) for s in scores))}
    return features if features["short"] else "no melody fragments"


def set_held_out(features: list[Features]) -> None:
    global _held_out_index, _held_out_sizes
    _held_out_index = {kind: defaultdict(list) for kind in features[0]} if features else {}
    for i, f in enumerate(features):
        for kind, grams in f.items():
            for gram in grams:
                _held_out_index[kind][gram].append(i)
    _held_out_sizes = {kind: [len(f[kind]) for f in features] for kind in _held_out_index}


def shared_counts(kind: str, grams: Grams) -> Counter:
    return Counter(i for gram in grams for i in _held_out_index[kind].get(gram, ()))


def melody_matches(unit: Features) -> dict[int, Match]:
    """The melody rules that fire between this training unit and each held-out score."""
    found: dict[int, Match] = defaultdict(lambda: ([], ""))
    def fire(i: int, rule: str, evidence: str) -> None:
        rules, before = found[i]
        found[i] = (rules + [rule], f"{before}; {evidence}" if before else evidence)
    for i, n in shared_counts("long", unit["long"]).items():
        held_side, training_side = n / _held_out_sizes["long"][i], n / len(unit["long"])
        if max(held_side, training_side) >= COPY_SHARE:
            fire(i, "runs16", f"runs16 {held_side:.2f}/{training_side:.2f}")
    if len(unit["short"]) >= MIN_UNIT_FRAGMENTS:
        for i, n in shared_counts("short", unit["short"]).items():
            if n >= MIN_SHARED:
                fire(i, "count8", f"count8 n={n}")
    if len(unit["skeleton"]) >= MIN_SKELETON:
        for i, n in shared_counts("skeleton", unit["skeleton"]).items():
            held_side, training_side = n / _held_out_sizes["skeleton"][i], n / len(unit["skeleton"])
            if _held_out_sizes["skeleton"][i] >= MIN_SKELETON and max(held_side, training_side) >= COPY_SHARE:
                fire(i, "skeleton", f"skeleton {held_side:.2f}/{training_side:.2f}")
    return dict(found)


def compare_unit(unit: Unit) -> tuple[Unit, dict[int, Match] | str]:
    features = features_of(unit[2])
    return unit, features if isinstance(features, str) else melody_matches(features)


def normal_title(title: str) -> str:
    """Lowercase words and numbers without accents, bracketed asides or TITLE_FILLER."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", title.lower()) if not unicodedata.combining(c))
    plain = re.sub(r"\([^)]*\)|\[[^]]*\]", " ", plain)
    return " ".join(w for w in re.findall(r"[a-z]+|\d+", plain) if w not in TITLE_FILLER)


def pdmx_titles() -> dict[str, str]:
    """PDMX's title of each score, by file name."""
    csv.field_size_limit(1 << 30)
    with (PDMX / "PDMX.csv").open() as f:
        return {Path(r["mxl"]).name: r["title"] or r["song_name"] for r in csv.DictReader(f)}


def unit_title(source: str, name: str, pdmx: dict[str, str], songs: dict[str, str]) -> str:
    if source == "pdmx":
        return pdmx.get(Path(name).name, "")
    if source == "olimpic":
        return songs.get(name, "").split("/")[-1]
    if source == "musicorpus":
        part, document = name.split("/")
        page = next((MUSICORPUS / part).glob(f"{document}_*"))
        return json.loads((page / "metadata.json").read_text()).get("title_description") or ""
    if source == "asap":
        return " ".join(Path(name).parts[2:-1])
    return name  # grandstaff: composer/collection/movement


def title_matches(held_out: list[Path], units: list[Unit]) -> dict[tuple[int, str, str], Match]:
    pdmx = pdmx_titles()
    songs = {str(k): v["path"] for k, v in yaml.safe_load((OLIMPIC / "lieder" / "data" / "scores.yaml").read_text()).items()}
    by_title = defaultdict(list)
    for i, score in enumerate(held_out):
        source = "asap" if "asap" in score.parts else "pdmx"
        title = normal_title(unit_title(source, str(score.relative_to(DATA)), pdmx, songs))
        if title:
            by_title[title].append(i)
    found = {}
    for source, name, _ in units:
        title = normal_title(unit_title(source, name, pdmx, songs))
        for i in by_title.get(title, []):
            found[i, source, name] = (["title"], f"title {title!r}")
    return found


def held_out_features(scores: list[Path], workers: int) -> tuple[list[Path], list[Features]]:
    """The held-out scores that have melody fragments, and their fragments."""
    held_out = [s for s in scores if is_held_out(s)]
    with mp.Pool(workers) as pool:
        results = pool.map(features_of, [[s] for s in held_out], chunksize=8)
    for s, features in zip(held_out, results):
        if isinstance(features, str):
            print(f"held out, not compared ({features}): {s.relative_to(DATA)}", flush=True)
    kept = [(s, f) for s, f in zip(held_out, results) if not isinstance(f, str)]
    return [s for s, _ in kept], [f for _, f in kept]


def melody_pairs(units: list[Unit], features: list[Features], workers: int) -> dict[tuple[int, str, str], Match]:
    found = {}
    failed: Counter = Counter()
    with mp.Pool(workers, set_held_out, (features,)) as pool:
        for done, ((source, name, _), result) in enumerate(pool.imap(compare_unit, units, chunksize=16), 1):
            if isinstance(result, str):
                failed[source] += 1
                print(f"{source} unit not compared ({result}): {name}", flush=True)
                continue
            for i, match in result.items():
                found[i, source, name] = match
            if done % 5000 == 0:
                print(f"{done}/{len(units)} training units", flush=True)
    print(f"training units: {dict(Counter(u[0] for u in units))}; not compared: {dict(failed)}")
    return found


def merged(*pair_sets: dict[tuple[int, str, str], Match]) -> dict[tuple[int, str, str], Match]:
    out: dict[tuple[int, str, str], Match] = {}
    for pairs in pair_sets:
        for pair, (rules, evidence) in pairs.items():
            before_rules, before = out.get(pair, ([], ""))
            out[pair] = (before_rules + rules, f"{before}; {evidence}" if before else evidence)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    start = time.monotonic()
    scores = [Path(line) for line in (DATA / "render_scores.txt").read_text().split()]
    held_out, features = held_out_features(scores, args.workers)
    print(f"{len(held_out)} held-out scores compared ({sum(map(is_held_out, scores))} held out)", flush=True)
    units = render_list_units(scores) + grandstaff_units() + olimpic_units() + musicorpus_units()
    pairs = merged(melody_pairs(units, features, args.workers), title_matches(held_out, units))
    lines = sorted("\t".join([score_key(held_out[i]), str(held_out[i].relative_to(DATA)), source, name,
                              ",".join(rules), evidence]) for (i, source, name), (rules, evidence) in pairs.items())
    args.out.write_text("".join(f"{line}\n" for line in lines))
    by_rule = Counter(rule for _, (rules, _) in pairs.items() for rule in rules)
    scores_by_rule = {rule: len({i for (i, _, _), (rules, _) in pairs.items() if rule in rules}) for rule in by_rule}
    print(f"{len(lines)} pairs shortlisted, {len({i for i, _, _ in pairs})} of {len(held_out)} held-out scores; "
          f"pairs by rule {dict(by_rule)}, scores by rule {scores_by_rule}; {time.monotonic() - start:.0f} s")


if __name__ == "__main__":
    main()
