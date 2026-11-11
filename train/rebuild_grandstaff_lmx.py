"""GrandStaff-LMX rebuilt from its **kern sources as OLiMPiC built it, with the clefs, keys and
meters it lost put back. errata/grandstaff-lmx/README.md says what was wrong.

usage: rebuild_grandstaff_lmx.py GRANDSTAFF_LMX_ROOT OUT_DIR [--workers N] [--limit N]

GRANDSTAFF_LMX_ROOT is GrandStaff-LMX extracted with GrandStaff in it (.krn, .musicxml, .lmx per
system). OLiMPiC's build (app/datasets/grandstaff/build.py) is run through its own functions:
music21 reads the **kern and exports a crude MusicXML, MuseScore 3.6.2 normalises it, OLiMPiC's
Linearizer writes the LMX; rerun as it is, it gives back the release byte for byte. The one
change: after music21's parse, every clef, key and meter it made is taken out and the source's
(seq.kern) put in each staff's measures where the source has them. For every system of the
train, dev and test lists, OUT_DIR gets a line in systems.tsv (system, split, one of STATUSES),
and <system>.musicxml and .lmx where the rebuild mends the release's.
"""
from __future__ import annotations

import argparse
import contextlib
import filecmp
import io
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import music21

from build_musescore_crops import STARTUP_TIMEOUT_S, TIMEOUT_PER_SYSTEM_S, MuseScore, as_musescore_child
from seq import kern as kern_reader

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("NW_DATA", ROOT / "data"))
sys.path.insert(0, str(DATA / "raw" / "olimpic" / "code" / "repo"))
from app.datasets.grandstaff import build as olimpic_build  # noqa: E402
from app.linearization.vocabulary import ACCIDENTAL_TOKENS, NOTE_TYPE_TOKENS, STEM_TOKENS  # noqa: E402

SPLITS = ("train", "dev", "test")
CORRECT, MENDED = "correct", "mended"
NOTES_DIFFER = "notes differ from the source"
MEASURES_DIFFER = "measure count differs from the source"
UNKNOWN_OPENING = "a staff's first clef, key or meter comes after notes in the source"
UNREADABLE = "the source's spines cannot be followed"
NOT_BUILT = "OLiMPiC's build gives no MusicXML"
MORE_CHANGED = "the rebuild changes more than signatures, accidentals and stems"
STATUSES = (CORRECT, MENDED, NOTES_DIFFER, MEASURES_DIFFER, UNKNOWN_OPENING, UNREADABLE, NOT_BUILT, MORE_CHANGED)
# LMX tokens a mended signature may change: a key and a meter; the accidentals MuseScore draws
# or not under the right key; the stems it turns by where a note sits on the staff, which a clef
# moves. And a clef with its staff, the backups before it and the forwards after it, which take
# MuseScore to a clef's beat and back when no note of its staff starts there.
SIGNATURE_TOKENS = ("key:", "time", "beats:", "beat-type:")
MAY_CHANGE = set(ACCIDENTAL_TOKENS) | set(STEM_TOKENS)
CHUNK = 100


class MeasuresDiffer(Exception):
    pass


def signature(kind: str, value: str | int) -> music21.base.Music21Object:
    if kind == kern_reader.CLEF:
        return music21.clef.clefFromString(value.replace("v", "").replace("^", ""),
                                           octaveShift=value.count("^") - value.count("v"))
    if kind == kern_reader.KEY:
        return music21.key.KeySignature(value)
    return music21.meter.TimeSignature(value)


def with_source_signatures(score: music21.stream.Score, kern: kern_reader.Kern) -> None:
    """music21 puts a signature after a spine split in the split's voices, and a change at the
    start of the first measure or in no measure at all; its MusicXML export loses all those."""
    for part in score.parts:
        staff = kern_reader.STAFF_OF_SPINE[int(part.id.split("_")[-1])]
        measures = list(part.getElementsByClass(music21.stream.Measure))
        if len(measures) != kern.measure_count:
            raise MeasuresDiffer
        made = (music21.clef.Clef, music21.key.KeySignature, music21.meter.TimeSignature)
        for element in list(part.recurse().getElementsByClass(made)):
            element.activeSite.remove(element)
        for (kind, on), value in kern.opening.items():
            if on == staff:
                measures[0].insert(0, signature(kind, value))
        for change in kern.changes:
            if change.staff == staff:
                measure = kern.measure_at(change.time)
                measures[measure].insert(float(change.time - kern.measure_start(measure)),
                                         signature(change.kind, change.value))


def parse_with_source_signatures(self, data: str, *args, **kwargs):
    """music21's ConverterHumdrum.parseData, as OLiMPiC's build calls it, followed by the mend."""
    _parse_data(self, data, *args, **kwargs)
    with_source_signatures(self.stream, kern_reader.read(data))


_parse_data = music21.converter.subConverters.ConverterHumdrum.parseData
music21.converter.subConverters.ConverterHumdrum.parseData = parse_with_source_signatures


def notes_of(part: ET.Element) -> Counter:
    """(measure, staff, step, alter, octave) of every note but grace notes, as kern.Kern.pitches."""
    pitches = Counter()
    for measure, element in enumerate(part.findall("measure")):
        for note in element.findall("note"):
            if note.find("pitch") is None or note.find("grace") is not None:
                continue
            pitches[(measure, int(note.findtext("staff", "1")), note.findtext("pitch/step"),
                     int(float(note.findtext("pitch/alter", "0"))), int(note.findtext("pitch/octave")))] += 1
    return pitches


def without_signatures(lmx: str) -> list[str]:
    tokens = lmx.split()
    dropped = {i for i, t in enumerate(tokens) if t.startswith(SIGNATURE_TOKENS) or t in MAY_CHANGE}
    for i in (i for i, t in enumerate(tokens) if t.startswith("clef:")):
        dropped |= {i, i + 1} | _cursor_moves(tokens, i - 2, -2, "backup") | _cursor_moves(tokens, i + 2, 2, "forward")
    return [t for i, t in enumerate(tokens) if i not in dropped]


def _cursor_moves(tokens: list[str], start: int, step: int, move: str) -> set[int]:
    """Indices of the moves (a move and its note type) from start on, walking by step: -2 the
    backups just before a clef, 2 the forwards just after it."""
    moves, i = set(), start
    while 0 <= i < len(tokens) - 1 and tokens[i] == move and tokens[i + 1] in NOTE_TYPE_TOKENS:
        moves |= {i, i + 1}
        i += step
    return moves


def crude_musicxml(base: Path) -> str | None:
    """OLiMPiC's first step with the mend; a status when it gives no crude MusicXML."""
    try:
        kern = kern_reader.read(olimpic_build._load_kern_file(f"{base}.krn"))
    except kern_reader.Unreadable:
        return UNREADABLE
    if kern.after_notes:
        return UNKNOWN_OPENING
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            olimpic_build._kern_to_crude_musicxml([str(base)], soft=False)
    except MeasuresDiffer:
        return MEASURES_DIFFER
    except Exception:  # music21 fails on the file as it did for OLiMPiC
        return NOT_BUILT
    return None if Path(f"{base}.crude.musicxml").exists() else NOT_BUILT


def normalise_with_musescore(musescore: MuseScore, bases: list[Path], work: Path) -> None:
    """MuseScore stops at a file it fails on: the files after it go to a new process."""
    pending = [b for b in bases if Path(f"{b}.crude.musicxml").exists()]
    while pending:
        job = [{"in": f"{b}.crude.musicxml", "out": f"{b}.musescore.musicxml"} for b in pending]
        (work / "job.json").write_text(json.dumps(job))
        with contextlib.suppress(subprocess.TimeoutExpired):
            subprocess.run([str(musescore.binary), "-c", str(work / "config"), "-j", str(work / "job.json")],
                           env=musescore.environment(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=STARTUP_TIMEOUT_S + TIMEOUT_PER_SYSTEM_S * len(pending),
                           preexec_fn=as_musescore_child)
        failed = next((i for i, b in enumerate(pending) if not Path(f"{b}.musescore.musicxml").exists()), None)
        pending = [] if failed is None else pending[failed + 1:]


def judged(name: str, base: Path, release: Path) -> str:
    """The rebuilt system's status against the release and its source."""
    if filecmp.cmp(f"{base}.lmx", release / f"{name}.lmx", shallow=False) and \
            filecmp.cmp(f"{base}.musicxml", release / f"{name}.musicxml", shallow=False):
        return CORRECT
    kern = kern_reader.read(olimpic_build._load_kern_file(str(release / f"{name}.krn")))
    if notes_of(ET.parse(f"{base}.musicxml").getroot().find("part")) != kern.pitches:
        return NOTES_DIFFER
    if without_signatures(Path(f"{base}.lmx").read_text()) != without_signatures((release / f"{name}.lmx").read_text()):
        return MORE_CHANGED
    return MENDED


def rebuild_chunk(job: tuple[Path, Path, list[str]]) -> list[tuple[str, str]]:
    release, out, names = job
    musescore = MuseScore(ROOT / "tools" / "musescore" / "squashfs-root")
    statuses = {}
    with tempfile.TemporaryDirectory(dir=out) as temporary:
        work = Path(temporary)
        (work / "config").mkdir()
        bases = {name: work / str(i) for i, name in enumerate(names)}
        for name, base in bases.items():
            shutil.copy(release / f"{name}.krn", f"{base}.krn")
            if (status := crude_musicxml(base)) is not None:
                statuses[name] = status
        normalise_with_musescore(musescore, [bases[n] for n in names if n not in statuses], work)
        for name, base in bases.items():
            if name in statuses:
                continue
            if not Path(f"{base}.musescore.musicxml").exists():
                statuses[name] = NOT_BUILT
                continue
            with contextlib.redirect_stdout(io.StringIO()):
                olimpic_build._finalize_lmx_and_musicxml_annotations([str(base)])
            statuses[name] = judged(name, base, release)
            if statuses[name] == MENDED:
                for ext in ("musicxml", "lmx"):
                    target = out / f"{name}.{ext}"
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(f"{base}.{ext}", target)
    return [(name, statuses[name]) for name in names]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("release", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, help="the first N systems of each list, to try it")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    systems = [(name, split) for split in SPLITS
               for name in (args.release / f"samples.{split}.txt").read_text().split()[:args.limit]]
    split_of = dict(systems)
    names = [name for name, _ in systems]
    jobs = [(args.release.resolve(), args.out.resolve(), names[i:i + CHUNK]) for i in range(0, len(names), CHUNK)]
    counts = Counter()
    with mp.get_context("spawn").Pool(args.workers) as pool, (args.out / "systems.tsv").open("w") as table:
        for chunk in pool.imap(rebuild_chunk, jobs):
            for name, status in chunk:
                table.write(f"{name}\t{split_of[name]}\t{status}\n")
                counts[split_of[name], status] += 1
            table.flush()
    for split in SPLITS:
        print(split, {status: counts[split, status] for status in STATUSES if counts[split, status]})


if __name__ == "__main__":
    main()
