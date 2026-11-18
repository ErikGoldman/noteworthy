"""Gold for OLiMPiC scanned dev by page (docs/EVAL_SETS.md, spec 2): one MusicXML for each IMSLP
page of the clean dev songs (data/olimpic_dev_clean.txt, songs with no copy in a training
source), its annotated piano systems in reading order, from their labels with pitches as
printed and ottava signs, tuplet numbers drawn or hidden as the song's score draws them
(olimpic_written_dev_v2, train/olimpic_tuplet_display.py). The labels are joined as a reading's systems are
(seq_pages.joined_musicxml), with no page rule: gold is what the label says. A page holding a
system with no as-printed label (one opening inside a sign) is left out and named.

With --rules, the labels go through those page rules instead (all, or one rule's name), for the
check that a rule leaves a correct reading as it is.

usage: olimpic_pages_gold.py OUT [--rules all|RULE]
  writes OUT/<song>-p<IMSLP page>.musicxml and OUT/pages.tsv (page, song, its systems)"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from olimpic_benchmark import clean_dev_songs, song_systems
from seq_pages import joined_musicxml
from seq_predict import ROOT

sys.path.insert(0, str(ROOT / "train"))
from seq.lmx_tuplets import ALL_RULES, PageRule  # noqa: E402

LABELS = ROOT / "data" / "olimpic_written_dev_v2" / "scanned"


def pages_of(song: str) -> dict[int, list[str]]:
    """The song's annotated systems by IMSLP page, each in reading order."""
    pages = defaultdict(list)
    for sample, page, _ in song_systems(song):
        pages[page].append(sample)
    return dict(pages)


def label_of(sample: str) -> str | None:
    path = LABELS / f"{sample}.lmx"
    return path.read_text().strip() if path.exists() else None


def rules_named(name: str | None) -> frozenset[PageRule]:
    if name is None:
        return frozenset()
    return ALL_RULES if name == "all" else frozenset({PageRule(name)})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--rules", choices=["all", *(r.value for r in PageRule)],
                    help="page rules to apply to the labels (default none: the gold)")
    args = ap.parse_args()
    rules = rules_named(args.rules)
    args.out.mkdir(parents=True, exist_ok=False)
    songs = clean_dev_songs()
    written, left_out = [], []
    for song in songs:
        for page, samples in sorted(pages_of(song).items()):
            labels = [label_of(s) for s in samples]
            name = f"{song}-p{page}"
            if None in labels:
                left_out.append((name, [s for s, lmx in zip(samples, labels) if lmx is None]))
                continue
            (args.out / f"{name}.musicxml").write_text(joined_musicxml(labels, rules))
            written.append(f"{name}\t{song}\t{' '.join(samples)}\n")
    (args.out / "pages.tsv").write_text("".join(written))
    print(f"{len(songs)} songs, {len(written)} pages written, {len(left_out)} left out:")
    for name, samples in left_out:
        print(f"   {name}: {', '.join(samples)} has no as-printed label")


if __name__ == "__main__":
    main()
