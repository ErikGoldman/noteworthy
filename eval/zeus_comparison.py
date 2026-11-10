"""Zeus, the OLiMPiC authors' end-to-end recognizer (Mayer et al., ICDAR 2024), on the same
OLiMPiC system crops as ours, and their TEDn metric for any recognizer's output.

Runs in .venv-zeus (TensorFlow 2.12, see scripts/setup_compare_env.sh) with the OLiMPiC code
(scripts/fetch_training_data.sh) and the Zeus model (scripts/fetch_benchmark_data.sh); it does not
import our own modules.

usage:
  zeus_comparison.py predict --split dev|test --out DIR
      The released Zeus model on every system crop of the split. Writes its raw output,
      DIR/samples/<song>/<system>.lmx, and the same delinearized to .musicxml, which our
      scorer reads (olimpic_benchmark.py systems --predictions DIR).
  zeus_comparison.py tedn PREDICTIONS_DIR --samples LIST [--workers N]
      TEDn of the predictions of the listed samples in the "lmx" flavour of the paper's
      Table 3 (gold pruned to what LMX can express), micro-averaged as the paper does:
      total edit cost over total gold cost. A sample's .lmx is used when present, else
      its .musicxml is linearized, so ours goes through the same LMX round trip as Zeus's.
      Costs are cached per sample in PREDICTIONS_DIR/tedn.json: a system takes seconds
      to minutes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OLIMPIC = Path(os.environ.get("NW_DATA", ROOT / "data")) / "raw" / "olimpic"
SCANNED = OLIMPIC / "olimpic-1.0-scanned"
CODE = OLIMPIC / "code"
ZEUS_MODEL = CODE / "zeus-olimpic-1.0-2024-02-12.model"

sys.path.insert(0, str(CODE / "repo"))
from app.evaluation.TEDn_lmx_xml import TEDn_lmx_xml  # noqa: E402
from app.linearization.Delinearizer import Delinearizer  # noqa: E402
from app.linearization.Linearizer import Linearizer  # noqa: E402
from app.symbolic.part_to_score import part_to_score  # noqa: E402


def split_samples(split: str) -> list[str]:
    return (SCANNED / f"samples.{split}.txt").read_text().split()


# --- predict ---------------------------------------------------------------------------

def run_zeus(split: str, work: Path) -> list[str]:
    """Zeus's LMX for each sample of the split, in split order."""
    work.mkdir(parents=True, exist_ok=True)
    dataset = work / "scanned"
    if not dataset.exists():
        dataset.symlink_to(SCANNED)
    zeus = CODE / "repo" / "zeus"
    # create_pickle.py writes "<dataset>-<split>.pickle" into its working directory.
    subprocess.run([sys.executable, zeus / "create_pickle.py", "scanned", split], cwd=work, check=True)
    subprocess.run([sys.executable, zeus / "zeus.py", "--load", ZEUS_MODEL, "--exp", "run", "--test", f"scanned-{split}"],
                   cwd=work, check=True)
    return (work / "run" / f"scanned-{split}.predicted.lmx").read_text().splitlines()


def delinearize(lmx: str) -> str:
    delinearizer = Delinearizer(errout=None)
    delinearizer.process_text(lmx)
    return ET.tostring(part_to_score(delinearizer.part_element).getroot(), encoding="unicode", xml_declaration=True)


def predict(args: argparse.Namespace) -> None:
    samples = split_samples(args.split)
    predictions = run_zeus(args.split, args.out / "work")
    assert len(predictions) == len(samples), f"{len(predictions)} predictions for {len(samples)} samples"
    for sample, lmx in zip(samples, predictions):
        path = args.out / sample
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".lmx").write_text(lmx + "\n")
        path.with_suffix(".musicxml").write_text(delinearize(lmx))


# --- tedn ------------------------------------------------------------------------------

def linearize(musicxml: Path) -> str:
    # Our MusicXML has a single part; the OLiMPiC loader looks for a part named piano.
    linearizer = Linearizer(errout=None)
    linearizer.process_part(ET.parse(musicxml).getroot().find("part"))
    return " ".join(linearizer.output_tokens)


def predicted_lmx(predictions: Path, sample: str) -> str:
    raw = predictions / f"{sample}.lmx"
    if raw.exists():
        return raw.read_text().strip()
    prediction = predictions / f"{sample}.musicxml"
    return linearize(prediction) if prediction.exists() else ""


def sample_cost(job: tuple[Path, str]) -> tuple[str, int, int]:
    predictions, sample = job
    result = TEDn_lmx_xml(predicted_lmx(predictions, sample), (SCANNED / f"{sample}.musicxml").read_text(), flavor="lmx")
    return sample, result.edit_cost, result.gold_cost


def tedn(args: argparse.Namespace) -> None:
    samples = args.samples.read_text().split()
    cache_path = args.predictions / "tedn.json"
    costs = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    todo = [(args.predictions, s) for s in samples if s not in costs]
    # Largest first, so the slowest systems do not end up running alone at the end.
    todo.sort(key=lambda job: (SCANNED / f"{job[1]}.musicxml").stat().st_size, reverse=True)
    with Pool(args.workers) as pool:
        for done, (sample, edit, gold) in enumerate(pool.imap_unordered(sample_cost, todo), 1):
            costs[sample] = [edit, gold]
            if done % 50 == 0 or done == len(todo):
                cache_path.write_text(json.dumps(costs))
                print(f"  {done}/{len(todo)} systems", file=sys.stderr)
    edit = sum(costs[s][0] for s in samples)
    gold = sum(costs[s][1] for s in samples)
    print(f"TEDn-lmx = {100 * edit / gold:.2f}% over {len(samples)} systems ({args.samples.name})")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("predict")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--out", type=Path, required=True)
    t = sub.add_parser("tedn")
    t.add_argument("predictions", type=Path)
    t.add_argument("--samples", type=Path, required=True)
    t.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    (predict if args.command == "predict" else tedn)(args)


if __name__ == "__main__":
    main()
