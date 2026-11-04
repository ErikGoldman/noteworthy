"""Held-out benchmark on real scans: OpenScore Lieder transcriptions scored against the
IMSLP scans they were transcribed from.

usage:
  lieder_benchmark.py select DIR [--count 50]
  lieder_benchmark.py download DIR
  lieder_benchmark.py transcribe DIR
  lieder_benchmark.py run DIR --binary NOTEWORTHY --model MODEL.onnx --out OUT [--jobs 3]

OpenScore Lieder (CC0) transcribes each song from one IMSLP scan and keeps its system
and page breaks, so a transcription is exact ground truth for the scan's pages: old
plates, scanner noise, skew and yellowed paper, none of which the synthetic benchmarks
show. The songs are voice and piano; all staves are scored.

`download` accepts IMSLP's disclaimer (files are public domain in Canada, maybe not
elsewhere), so it is a step for the user to run, not an automated one.

Held out: songs on PDMX (MuseScore.com, our training source) are excluded by MuseScore
id, and so are songs whose piece (omr.pieces) has any copy in PDMX or ASAP, whether or
not that copy is in a training list.

A scan may hold a whole volume, so `run` recognizes every page and scores each song on
the run of consecutive pages (as many as the transcription has) that matches it best.
Measures are aligned as in diagnose.py; unmatched measures at either end of that run
belong to neighbouring songs and are not counted.
"""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import yaml

import musescore
from diagnose import align_measures, by_measure, classify_measure
from score_events import NoteEvent, _read_musicxml_root, parse_musicxml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from omr.pieces import Piece, PieceIndex, asap_pieces, pdmx_pieces, surname, title_words  # noqa: E402

RAW = ROOT / "data" / "raw"
LIEDER_RAW = "https://raw.githubusercontent.com/OpenScore/Lieder/main"
USER_AGENT = "noteworthy-omr-benchmark/0.1 (research; low volume)"
IMSLP_DELAY_S = 5.0
SELECTION = "selection.tsv"


@dataclass(frozen=True)
class Song:
    musescore_id: str
    imslp_file: str  # IMSLP file index, e.g. "412044"
    path: str        # composer/set/song in the Lieder repository

    @property
    def stem(self) -> str:
        return f"lc{self.musescore_id}"


def read_selection(d: Path) -> list[Song]:
    return [Song(*line.split("\t")) for line in (d / SELECTION).read_text().splitlines() if line]


def scan_path(d: Path, song: Song) -> Path:
    return d / "scans" / f"imslp{song.imslp_file}.pdf"


def truth_path(d: Path, song: Song) -> Path:
    return d / "truth" / f"{song.stem}.musicxml"


def fetch(url: str, headers: dict[str, str] | None = None) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read(), response.headers.get_content_type()


# --- select -----------------------------------------------------------------------------

def pdmx_musescore_ids(pdmx_root: Path) -> set[str]:
    csv.field_size_limit(1 << 30)
    with (pdmx_root / "PDMX.csv").open() as f:
        return {Path(row["metadata"]).stem for row in csv.DictReader(f)}


def lieder_piece(path: str) -> Piece:
    """Piece of a Lieder song from its repository path, "Last,_First/set/NN_Title"."""
    composer, _, song = path.split("/")
    return Piece(title_words(re.sub(r"^\d+[a-z]?_", "", song).replace("_", " ")), surname(composer.split(",")[0]))


def select(d: Path, count: int) -> None:
    """Voice-and-piano songs (no `instruments` entry) with no copy in the training
    sources, at most one per composer, in a fixed pseudo-random order. Extra candidates
    cover transcriptions that turn out not to fit (see transcribe)."""
    scores = yaml.safe_load(fetch(f"{LIEDER_RAW}/data/scores.yaml")[0])
    on_pdmx = pdmx_musescore_ids(RAW / "pdmx")
    known = PieceIndex([p for pieces in (pdmx_pieces(RAW / "pdmx") | asap_pieces(RAW / "asap")).values() for p in pieces])
    candidates = sorted(((str(k), v) for k, v in scores.items()
                         if str(k) not in on_pdmx and "instruments" not in v and v.get("imslp", "").startswith("#")
                         and not known.contains(lieder_piece(v["path"]))),
                        key=lambda kv: zlib.crc32(kv[0].encode()))
    chosen, composers = [], set()
    for musescore_id, v in candidates:
        composer = v["path"].split("/")[0]
        if composer not in composers:
            composers.add(composer)
            chosen.append(Song(musescore_id, v["imslp"].lstrip("#"), v["path"]))
    chosen = chosen[: count * 3 // 2]
    d.mkdir(parents=True, exist_ok=True)
    (d / SELECTION).write_text("".join(f"{s.musescore_id}\t{s.imslp_file}\t{s.path}\n" for s in chosen))
    print(f"{len(chosen)} candidates")


# --- download ---------------------------------------------------------------------------

_PDF_LINK = re.compile(r'href="([^"]+\.pdf)"', re.IGNORECASE)


def download_scan(song: Song, target: Path) -> str:
    cookie = {"Cookie": "imslpdisclaimeraccepted=yes"}
    body, kind = fetch(f"https://imslp.org/wiki/Special:ImagefromIndex/{song.imslp_file}", cookie)
    if kind != "application/pdf":
        # A landing page rather than the file: follow its link to the PDF.
        link = _PDF_LINK.search(body.decode("utf-8", "replace"))
        if not link:
            return f"no PDF link on the IMSLP page ({kind})"
        time.sleep(IMSLP_DELAY_S)
        body, kind = fetch(urllib.parse.urljoin("https://imslp.org/", link.group(1)), cookie)
        if kind != "application/pdf":
            return f"link did not give a PDF ({kind})"
    partial = target.with_suffix(".partial")
    partial.write_bytes(body)
    partial.replace(target)
    return f"{len(body) / 1e6:.1f} MB"


def download(d: Path) -> None:
    (d / "scans").mkdir(parents=True, exist_ok=True)
    for song in read_selection(d):
        target = scan_path(d, song)
        if target.exists():
            continue
        try:
            status = download_scan(song, target)
        except OSError as e:
            status = f"failed: {e}"
        print(f"{song.stem} imslp{song.imslp_file}: {status}", flush=True)
        time.sleep(IMSLP_DELAY_S)


# --- transcribe -------------------------------------------------------------------------

def staves_per_part(xml: Path) -> list[int]:
    root = _read_musicxml_root(xml)
    return [max((int(s.text) for s in part.iter("staves")), default=1) for part in root.iter("part")]


def page_count(xml: Path) -> int:
    root = _read_musicxml_root(xml)
    first_part = next(root.iter("part"))
    return 1 + sum(1 for p in first_part.iter("print") if p.get("new-page") == "yes")


def transcribe(d: Path) -> None:
    """Fetches the transcriptions and converts them with MuseScore. Keeps only voice plus
    a two-staff piano, the layout the recognizer reads."""
    work = d / "mscx"
    work.mkdir(parents=True, exist_ok=True)
    (d / "truth").mkdir(exist_ok=True)
    songs = read_selection(d)
    for song in songs:
        source = work / f"{song.stem}.mscx"
        if not source.exists():
            url = f"{LIEDER_RAW}/scores/{urllib.parse.quote(song.path)}/{song.stem}.mscx"
            source.write_bytes(fetch(url)[0])
    conversions = [(work / f"{s.stem}.mscx", [truth_path(d, s)]) for s in songs if not truth_path(d, s).exists()]
    if conversions:
        musescore.run_batch(conversions, work / "job.json")
    for song in songs:
        xml = truth_path(d, song)
        layout = staves_per_part(xml) if xml.exists() else None
        if layout != [1, 2]:
            print(f"{song.stem}: dropped (staves per part {layout})")
            xml.unlink(missing_ok=True)


# --- run --------------------------------------------------------------------------------

def measure_pages(xml: Path) -> list[int]:
    """Input page of each measure of a recognizer output (from its new-page prints)."""
    root = _read_musicxml_root(xml)
    pages, page = [], 0
    for measure in next(root.iter("part")).iter("measure"):
        for p in measure.iter("print"):
            if p.get("page-number"):
                page = int(p.get("page-number"))
        pages.append(page)
    return pages


@dataclass
class SongResult:
    tally: Counter
    pages: tuple[int, int]
    n_pred: int


def score_window(gt: list[list[NoteEvent]], pred: list[list[NoteEvent]]) -> tuple[Counter, int]:
    """Error tally of the aligned measures, and the number of predicted notes counted."""
    pairs = align_measures(gt, pred)
    matched = [k for k, (g, p) in enumerate(pairs) if g is not None and p is not None]
    if not matched:
        return Counter(missed=sum(map(len, gt))), 0
    tally, n_pred = Counter(), 0
    for k, (g, p) in enumerate(pairs):
        if g is not None and p is not None:
            classify_measure(gt[g], pred[p], tally)
            n_pred += len(pred[p])
        elif g is not None:
            tally["missed measure"] += 1
            tally["missed"] += len(gt[g])
        elif matched[0] < k < matched[-1]:
            tally["extra measure"] += 1
            tally["extra"] += len(pred[p])
            n_pred += len(pred[p])
    return tally, n_pred


def score_song(truth: Path, prediction: Path) -> SongResult:
    gt = by_measure(parse_musicxml(truth))
    n_pages = page_count(truth)
    events, pages = parse_musicxml(prediction), measure_pages(prediction)
    pred_all = by_measure(events)
    pred_all += [[] for _ in range(len(pages) - len(pred_all))]
    best: SongResult | None = None
    for first in sorted(set(pages)):
        window = [pred_all[m] for m, page in enumerate(pages) if first <= page < first + n_pages]
        tally, n_pred = score_window(gt, window)
        if best is None or tally["correct"] > best.tally["correct"]:
            best = SongResult(tally, (first, first + n_pages - 1), n_pred)
    return best or SongResult(Counter(missed=sum(map(len, gt))), (0, 0), 0)


def run(d: Path, binary: str, model: str, out: Path, jobs: int, threads: int) -> None:
    out.mkdir(parents=True, exist_ok=True)
    songs = [s for s in read_selection(d) if truth_path(d, s).exists() and scan_path(d, s).exists()]

    def recognize(song: Song) -> SongResult | str:
        prediction = out / f"{song.stem}.musicxml"
        result = subprocess.run([binary, str(scan_path(d, song)), "--model", model, "-o", str(prediction),
                                 "--threads", str(threads)], capture_output=True, text=True)
        if result.returncode != 0:
            return f"recognizer failed: {result.stderr.strip()[-200:]}"
        return score_song(truth_path(d, song), prediction)

    total = Counter()
    total_pred = 0
    with ThreadPoolExecutor(jobs) as pool:
        for song, result in zip(songs, pool.map(recognize, songs)):
            if isinstance(result, str):
                print(f"{song.stem} {result}")
                continue
            t = result.tally
            n_gt = sum(v for k, v in t.items() if "measure" not in k and k != "extra")
            total += t
            total_pred += result.n_pred
            errors = ", ".join(f"{k}={v}" for k, v in t.most_common() if k != "correct")
            print(f"{song.stem} pages {result.pages[0]}-{result.pages[1]} gt={n_gt:4d} pred={result.n_pred:4d} "
                  f"F1={2 * t['correct'] / max(1, n_gt + result.n_pred):.3f}  {song.path}  [{errors}]", flush=True)
    n_gt = sum(v for k, v in total.items() if "measure" not in k and k != "extra")
    print(f"overall F1 = {2 * total['correct'] / max(1, n_gt + total_pred):.3f} over {len(songs)} songs, {n_gt} gt notes "
          f"(P={total['correct'] / max(1, total_pred):.3f} R={total['correct'] / max(1, n_gt):.3f})")
    for k, v in total.most_common():
        print(f"  {k:16s} {v}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    for name in ("select", "download", "transcribe", "run"):
        p = sub.add_parser(name)
        p.add_argument("dir", type=Path)
        if name == "select":
            p.add_argument("--count", type=int, default=50)
        if name == "run":
            p.add_argument("--binary", required=True)
            p.add_argument("--model", required=True)
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--jobs", type=int, default=3)
            p.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.command == "select":
        select(args.dir, args.count)
    elif args.command == "download":
        download(args.dir)
    elif args.command == "transcribe":
        transcribe(args.dir)
    else:
        run(args.dir, args.binary, args.model, args.out, args.jobs, args.threads)


if __name__ == "__main__":
    main()
