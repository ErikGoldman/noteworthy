"""OMR-NED, the metric of the Transcoda paper, with its authors' code unchanged.

usage: omr_ned.py PREDICTIONS GOLD --out RESULTS.json [--pred-format kern|musicxml]
                  [--gold-format kern|musicxml] [--jobs N]

Runs in .venv-musicdiff (requirements-omr-ned.txt). PREDICTIONS and GOLD are directories of
<sample>.krn or <sample>.musicxml. A page that cannot be parsed or compared scores 100, the
authors' policy, as does one over the time or memory limit here, and a missing prediction.
The score is the mean over pages; lower is better. Each page runs in its own process:
musicdiff's edit distance can take tens of gigabytes on a bad prediction.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import resource
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUTHORS_METRIC = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw/transcoda_repo/src/src/evaluation/omr_ned.py"
PAGE_MEMORY_BYTES = 6 << 30
PAGE_TIMEOUT_S = 900
FAILED = 100.0
SUFFIX = {"kern": ".krn", "musicxml": ".musicxml"}


def authors_metric():
    spec = importlib.util.spec_from_file_location("transcoda_omr_ned", AUTHORS_METRIC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up by name
    spec.loader.exec_module(module)
    return module


def compare_across_formats(metric, pred: Path, gold: Path) -> tuple[float | None, str | None]:
    """The authors' formula on two files of different formats, which they never compare."""
    m21 = metric.m21
    pred_ann = metric.AnnScore(m21.converter.parse(str(pred), forceSource=True))
    gold_ann = metric.AnnScore(m21.converter.parse(str(gold), forceSource=True))
    _, distance = metric.Comparison.annotated_scores_diff(pred_ann, gold_ann)
    total = pred_ann.notation_size() + gold_ann.notation_size()
    return (min(100.0 * (distance + pred_ann.num_syntax_errors_fixed) / total, 100.0) if total else 0.0), None


def score_page(pred: Path, gold: Path, pred_format: str, gold_format: str, answer) -> None:
    resource.setrlimit(resource.RLIMIT_AS, (PAGE_MEMORY_BYTES, PAGE_MEMORY_BYTES))
    try:
        metric = authors_metric()
        if pred_format != gold_format:
            answer.send(compare_across_formats(metric, pred, gold))
            return
        compute = metric.compute_omr_ned if pred_format == "kern" else metric.compute_omr_ned_from_musicxml
        result = compute(pred.read_text(), gold.read_text())
        answer.send((result.omr_ned, result.parse_error))
    except BaseException as e:  # noqa: B902  MemoryError and RecursionError count as failed pages
        answer.send((None, f"{type(e).__name__}: {str(e)[:150]}"))


def page_score(job: tuple[Path, Path, str, str]) -> tuple[str, float, str | None]:
    pred, gold, pred_format, gold_format = job
    if not pred.exists():
        return gold.stem, FAILED, "no prediction"
    receive, send = mp.Pipe(duplex=False)
    worker = mp.get_context("spawn").Process(target=score_page, args=(pred, gold, pred_format, gold_format, send))
    worker.start()
    send.close()
    answered = receive.poll(PAGE_TIMEOUT_S)
    try:
        score, error = receive.recv() if answered else (None, "timed out")
    except EOFError:
        score, error = None, "worker died"
    worker.kill()
    worker.join()
    return gold.stem, FAILED if score is None else score, error


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", type=Path)
    ap.add_argument("gold", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pred-format", choices=SUFFIX, default="kern")
    ap.add_argument("--gold-format", choices=SUFFIX, default="kern")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()
    golds = sorted(args.gold.glob("*" + SUFFIX[args.gold_format]))
    jobs = [(args.predictions / (g.stem + SUFFIX[args.pred_format]), g, args.pred_format, args.gold_format) for g in golds]
    from multiprocessing.pool import ThreadPool
    with ThreadPool(args.jobs) as pool:
        pages = pool.map(page_score, jobs)
    scores = {name: score for name, score, _ in pages}
    errors = {name: error for name, _, error in pages if error}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"scores": scores, "errors": errors}, indent=0))
    print(f"OMR-NED {sum(scores.values()) / len(scores):.2f} over {len(scores)} pages, {len(errors)} failed")


if __name__ == "__main__":
    main()
