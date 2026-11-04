"""LMX token vocabulary: OLiMPiC's, plus padding, start, end and unknown."""
from __future__ import annotations

from pathlib import Path

PAD, BOS, EOS, UNK = 0, 1, 2, 3
_SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class Vocab:
    def __init__(self, tokens: list[str]):
        self.tokens = _SPECIALS + tokens
        self.index = {t: i for i, t in enumerate(self.tokens)}

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(path.read_text().split())

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, lmx: str) -> list[int]:
        return [BOS] + [self.index.get(t, UNK) for t in lmx.split()] + [EOS]

    def decode(self, ids: list[int]) -> str:
        out = []
        for i in ids:
            if i == EOS:
                break
            if i >= len(_SPECIALS):
                out.append(self.tokens[i])
        return " ".join(out)
