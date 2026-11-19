"""Frees disk taken by runs whose files are safe in R2. Prints what it would remove; --apply removes it.

usage: prune_runs.py checkpoints RUN... [--apply]
           From each finished run, the checkpoints R2 holds at the same size, except best.pt, the
           averages runs are judged at (JUDGED_STEPS) and the run's last average. A run without
           exit_status (still training, or copied home part way) keeps last.pt, which resumes it.
       prune_runs.py retire RUN... [--apply]
           Pushes each run to R2, then removes it here once R2 holds every file at its size.

RUN is a path from the repo root (runs/NAME). A local copy pulled part way through a run is not
what R2 holds; only files R2 holds at the same size are ever removed, so such a copy loses nothing
R2 lacks. Either way, `r2_data.py pull runs/NAME` brings a run back whole.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import r2_data

JUDGED_STEPS = {60000, 80000, 100000}  # TODO.md, "How every run is judged"


def average_step(checkpoint: Path) -> int:
    return int(checkpoint.stem.removeprefix("average_"))


def removable_checkpoints(run: Path, stored: dict[str, int]) -> list[Path]:
    root = r2_data.ROOT
    averages = sorted((root / run).glob("average_*.pt"), key=average_step)
    kept = {root / run / "best.pt", *averages[-1:]}
    kept |= {a for a in averages if average_step(a) in JUDGED_STEPS}
    if not (root / run / "exit_status").exists():
        kept.add(root / run / "last.pt")
    return [p for p in sorted((root / run).glob("*.pt"))
            if p not in kept and stored.get(str(p.relative_to(root))) == p.stat().st_size]


def prune_checkpoints(client, runs: list[Path], apply: bool) -> None:
    total = 0
    for run in runs:
        for checkpoint in removable_checkpoints(run, r2_data.stored_files(client, run)):
            total += checkpoint.stat().st_size
            print(f"{'removing' if apply else 'would remove'} {checkpoint.relative_to(r2_data.ROOT)}", flush=True)
            if apply:
                checkpoint.unlink()
    print(f"{total / 1e9:.1f} GB {'freed' if apply else 'to free'}")


def retire(client, runs: list[Path], apply: bool) -> None:
    for run in runs:
        if not apply:
            print(f"would push and remove {run}")
            continue
        r2_data.push(client, run)
        missing = r2_data.not_stored(client, run)
        if missing:
            print(f"{run}: kept, R2 lacks {len(missing)} files, e.g. {missing[0].name}", flush=True)
            continue
        shutil.rmtree(r2_data.ROOT / run)
        print(f"{run}: every file in R2, removed here", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["checkpoints", "retire"])
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--apply", action="store_true", help="remove, rather than say what would be removed")
    args = ap.parse_args()
    client = r2_data.bucket_client()
    (prune_checkpoints if args.action == "checkpoints" else retire)(client, args.runs, args.apply)


if __name__ == "__main__":
    main()
