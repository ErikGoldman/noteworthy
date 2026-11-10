"""Installs the assets our Smashcima fork draws with into $MC_ASSETS_CACHE, and checks that
they are byte for byte the ones our crops were built with.

Smashcima downloads MUSCIMA++ 1.0 from LINDAT and paper patches from the Moravian Library's
image server, without checksums, then extracts glyphs from MUSCIMA++. Any change upstream (or
in the extraction code) would silently change every handwritten crop; here it stops the build.
A deliberate change (a fork commit that bumps an asset's version) updates EXPECTED.

usage: install_smashcima_assets.py      (run in .venv-smashcima)
"""
import hashlib
import sys
from pathlib import Path

from smashcima.assets.AssetRepository import AssetRepository
from smashcima.assets.glyphs.muscima_pp.MuscimaPPGlyphs import MuscimaPPGlyphs
from smashcima.assets.textures.MzkPaperPatches import MzkPaperPatches
from smashcima.config import MC_ASSETS_CACHE

EXPECTED = {
    "MuscimaPP": "db2c67738e1307f3d42749c49dea346c97845d36ce89fe995f41e795222bee0a",
    "MuscimaPPGlyphs": "fbb66e1f6c9ab84d190834a494c30bf06c3ea95581b3efdac1fe06bed5463bb8",
    "MzkPaperPatches": "a024e20d22ceb414b003d5035efa0daf515e4ffeca9c00a9ca255d7f2a62d560",
}


def tree_digest(directory: Path) -> str:
    """sha256 over every file's path and contents, in path order"""
    digest = hashlib.sha256()
    for path in sorted(p for p in directory.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(directory)).encode() + b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def main() -> None:
    assets = AssetRepository(MC_ASSETS_CACHE)
    for bundle in (MuscimaPPGlyphs, MzkPaperPatches):
        assets.resolve_bundle(bundle)
    wrong = []
    for name, expected in EXPECTED.items():
        found = tree_digest(MC_ASSETS_CACHE / name)
        if found != expected:
            wrong.append(f"{name}: expected {expected}, found {found}")
    if wrong:
        sys.exit("Smashcima assets differ from the ones our crops were built with:\n  " + "\n  ".join(wrong))
    print("Smashcima assets ready")


if __name__ == "__main__":
    main()
