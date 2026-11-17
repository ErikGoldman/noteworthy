"""An OLiMPiC split (by default the synthetic training set) with pitches as printed, ottava
tokens (seq.ottava), and tuplet numbers drawn or not as the song's own MuseScore file draws them
(olimpic_tuplet_display).

usage: build_olimpic_written.py OLIMPIC_ROOT LIEDER OUT_DIR [SPLIT]

OLIMPIC_ROOT is olimpic-1.0-synthetic or olimpic-1.0-scanned (whose systems are the synthetic
dev and test systems, printed and scanned); LIEDER the OpenScore Lieder repository at the commit
OLiMPiC was made from (scripts/fetch_training_data.sh); SPLIT train, dev or test, written to
OUT_DIR/samples.SPLIT.txt.

Labels come from each system's MusicXML; images are linked, not copied. A system that opens
with the stop of a sign from an earlier system is left out: its file does not say which way
the sign went, and the image shows the notes shifted. A system wholly inside a long sign
has no trace of it, so it stays, with its sounding pitches. A song whose tuplet groups cannot
all be paired with its score's is left out whole, each one named in OUT_DIR/left_out.SPLIT.tsv.
"""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import yaml

from build_seq_dataset import linearize
from olimpic_tuplet_display import Unpaired, has_tuplets, score_tuplets, song_score, with_score_display


def opens_inside_a_sign(part: ET.Element) -> bool:
    first_on_staff = {}
    for direction in part.iter("direction"):
        for sign in direction.iter("octave-shift"):
            first_on_staff.setdefault(direction.findtext("staff", "1"), sign.get("type"))
    return "stop" in first_on_staff.values()


def song_of(name: str) -> str:
    return name.split("/")[1]


def page_order(name: str) -> tuple[int, int]:
    page, system = Path(name).name[1:].split("-s")
    return int(page), int(system)


def with_tuplets_as_drawn(parts: dict[str, ET.Element], lieder: Path, song: str, path: str | None) -> int:
    """Sets the song's tuplet starts (its systems' parts, by name) as its score draws them; returns
    how many changed. Raises Unpaired."""
    if not any(map(has_tuplets, parts.values())):
        return 0
    if path is None:
        raise Unpaired("not in the Lieder corpus's scores.yaml")
    score = score_tuplets(song_score(lieder, song, path))
    return with_score_display([parts[name] for name in sorted(parts, key=page_order)], score)


def write_label(root: Path, out: Path, name: str, part: ET.Element) -> None:
    target = out / f"{name}.lmx"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(linearize(part) + "\n")
    target.with_suffix(".png").unlink(missing_ok=True)
    target.with_suffix(".png").symlink_to((root / f"{name}.png").resolve())


def main() -> None:
    root, lieder, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    split = sys.argv[4] if len(sys.argv) > 4 else "train"
    paths = {str(song): entry["path"] for song, entry in yaml.safe_load((lieder / "data" / "scores.yaml").read_text()).items()}
    names = (root / f"samples.{split}.txt").read_text().split()
    by_song = defaultdict(list)
    for name in names:
        by_song[song_of(name)].append(name)
    written, changed, left_out = set(), 0, {}
    for song, systems in by_song.items():
        parts = {name: ET.parse(root / f"{name}.musicxml").find("part") for name in systems}
        try:
            changed += with_tuplets_as_drawn(parts, lieder, song, paths.get(song))
        except Unpaired as reason:
            left_out[song] = reason
            continue
        for name, part in parts.items():
            if not opens_inside_a_sign(part):
                write_label(root, out, name, part)
                written.add(name)
    (out / f"samples.{split}.txt").write_text("\n".join(n for n in names if n in written) + "\n")
    (out / f"left_out.{split}.tsv").write_text("".join(
        f"{song}\t{len(by_song[song])} systems\t{reason}\n" for song, reason in sorted(left_out.items())))
    print(f"{len(written)} systems kept; tuplet display set from the scores on {changed} groups; "
          f"{len(left_out)} songs left out, their groups unpaired ({sum(len(by_song[s]) for s in left_out)} systems, "
          f"left_out.{split}.tsv):")
    for song, reason in sorted(left_out.items()):
        print(f"   {song}: {reason}")


if __name__ == "__main__":
    main()
