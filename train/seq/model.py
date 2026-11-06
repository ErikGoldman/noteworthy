"""Image-to-sequence model for one system crop: a ResNet encoder, a transformer encoder over
its 2D feature grid, and a transformer decoder emitting LMX tokens."""
from __future__ import annotations

import math

import torch
from torch import nn

from .vocab import BOS, EOS, PAD

# Encoder strides: the feature grid has height / row_stride rows and width / 16 columns.
COL_STRIDE = 16
# Strides of the last three stages, by row stride; the stem halves twice before them.
STAGE_STRIDES = {32: ((2, 2), (2, 2), (2, 1)), 16: ((2, 2), (2, 2), (1, 1)), 8: ((2, 2), (1, 2), (1, 1))}


class BasicBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: tuple[int, int]):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = (nn.Identity() if cin == cout and stride == (1, 1)
                     else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))

    def forward(self, x):
        return torch.relu(self.body(x) + self.skip(x))


def _stage(cin: int, cout: int, stride: tuple[int, int], blocks: int = 2) -> nn.Sequential:
    return nn.Sequential(BasicBlock(cin, cout, stride), *[BasicBlock(cout, cout, (1, 1)) for _ in range(blocks - 1)])


def _sinusoid(n: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    out = torch.zeros(n, dim, device=device)
    out[:, 0::2] = torch.sin(pos * freq)
    out[:, 1::2] = torch.cos(pos * freq)
    return out


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 384, encoder_layers: int = 2, decoder_layers: int = 6,
                 heads: int = 8, max_tokens: int = 1024, height: int = 256, row_stride: int = 32, dropout: float = 0.1):
        super().__init__()
        self.height = height
        rows = height // row_stride
        into_128, into_256, into_dim = STAGE_STRIDES[row_stride]
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            _stage(64, 64, (1, 1)), _stage(64, 128, into_128), _stage(128, 256, into_256), _stage(256, dim, into_dim))
        self.row_embedding = nn.Parameter(torch.zeros(rows, dim))
        layer = lambda: nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer(), encoder_layers, enable_nested_tensor=False)
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_tokens, dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True), decoder_layers)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size, bias=False)
        self.out.weight = self.token_embedding.weight
        self.max_tokens = max_tokens

    def encode(self, images: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Memory (B, rows * cols, dim) and its padding mask (True = padding)."""
        f = self.cnn(images)                                   # (B, dim, rows, cols)
        b, d, rows, cols = f.shape
        f = f.permute(0, 2, 3, 1) + self.row_embedding[:rows, None, :] + _sinusoid(cols, d, f.device)[None]
        valid_cols = (widths.to(f.device) + COL_STRIDE - 1) // COL_STRIDE
        padding = torch.arange(cols, device=f.device)[None, :] >= valid_cols[:, None]   # (B, cols)
        padding = padding[:, None, :].expand(b, rows, cols).reshape(b, rows * cols)
        memory = self.encoder(f.reshape(b, rows * cols, d), src_key_padding_mask=padding)
        return memory, padding

    def decode(self, tokens: torch.Tensor, memory: torch.Tensor, memory_padding: torch.Tensor) -> torch.Tensor:
        n = tokens.shape[1]
        x = self.token_embedding(tokens) + self.position_embedding(torch.arange(n, device=tokens.device))[None]
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tokens.device), 1)
        x = self.decoder(x, memory, tgt_mask=causal, tgt_key_padding_mask=tokens == PAD,
                         memory_key_padding_mask=memory_padding, tgt_is_causal=True)
        return self.out(self.norm(x))

    def forward(self, images, widths, tokens):
        """Logits predicting tokens[:, 1:] from tokens[:, :-1]."""
        memory, padding = self.encode(images, widths)
        return self.decode(tokens[:, :-1], memory, padding)

    @torch.no_grad()
    def generate(self, images, widths, max_tokens: int | None = None) -> list[list[int]]:
        """Greedy decoding."""
        memory, padding = self.encode(images, widths)
        b = images.shape[0]
        tokens = torch.full((b, 1), BOS, device=images.device)
        done = torch.zeros(b, dtype=torch.bool, device=images.device)
        for _ in range((max_tokens or self.max_tokens) - 1):
            nxt = self.decode(tokens, memory, padding)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            tokens = torch.cat([tokens, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return [t[1:].tolist() for t in tokens]
