"""Locate and run MuseScore 4 (benchmark renders, Lieder conversions).

The executable is $MUSESCORE when set, otherwise the first of the usual names on PATH.
Under WSL the Windows build works: set MUSESCORE to its MuseScore4.exe; paths handed to
a Windows executable are translated with wslpath.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

_NAMES = ["mscore4portable", "mscore", "musescore", "MuseScore4", "MuseScore4.exe"]


def find() -> Path | None:
    configured = os.environ.get("MUSESCORE")
    if configured:
        return Path(configured) if Path(configured).exists() else None
    found = next((shutil.which(n) for n in _NAMES if shutil.which(n)), None)
    return Path(found) if found else None


def _path_for(executable: Path, p: Path) -> str:
    if executable.suffix.lower() != ".exe":
        return str(p)
    return subprocess.run(["wslpath", "-w", str(p)], capture_output=True, text=True, check=True).stdout.strip()


def run_batch(conversions: list[tuple[Path, list[Path]]], job_file: Path, options: list[str] = ()) -> None:
    """Runs (input, outputs) conversions as one batch job, since each MuseScore start costs
    seconds. Failed conversions simply leave their outputs missing."""
    executable = find()
    if executable is None:
        raise RuntimeError("MuseScore 4 not found: set MUSESCORE or put it on PATH")
    job = [{"in": _path_for(executable, src), "out": [_path_for(executable, o) for o in outs]} for src, outs in conversions]
    job_file.write_text(json.dumps(job))
    subprocess.run([str(executable), *options, "-j", _path_for(executable, job_file)], capture_output=True)
