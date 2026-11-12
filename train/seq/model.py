"""Image-to-sequence models for one system crop, both ending in a transformer decoder that emits
LMX tokens. Seq2Seq reads the system as one image: a ResNet encoder and a transformer encoder over
its 2D feature grid. StaffSeq2Seq reads each staff as its own image at a fixed height, whatever
the number of staves, then lets the staves attend to each other column by column."""
from __future__ import annotations

import math
from dataclasses import dataclass

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


def _cnn(dim: int, row_stride: int) -> nn.Sequential:
    into_128, into_256, into_dim = STAGE_STRIDES[row_stride]
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        _stage(64, 64, (1, 1)), _stage(64, 128, into_128), _stage(128, 256, into_256), _stage(256, dim, into_dim))


def _encoder_layer(dim: int, heads: int, dropout: float) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True)


def _sinusoid(n: int, dim: int, device) -> torch.Tensor:
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    out = torch.zeros(n, dim, device=device)
    out[:, 0::2] = torch.sin(pos * freq)
    out[:, 1::2] = torch.cos(pos * freq)
    return out


def _column_padding(widths: torch.Tensor, cols: int, device) -> torch.Tensor:
    """(B, cols), True where a column lies past the crop's width."""
    valid_cols = (widths.to(device) + COL_STRIDE - 1) // COL_STRIDE
    return torch.arange(cols, device=device)[None, :] >= valid_cols[:, None]


class _TokenDecoder(nn.Module):
    """The decoder and greedy generation both models share. A subclass builds its encoder, then
    calls _init_decoder (so Seq2Seq creates its parameters in the order its checkpoints and seeded
    runs were made with), and defines encode(images, widths) -> (memory, memory_padding)."""

    def _init_decoder(self, vocab_size: int, dim: int, decoder_layers: int, heads: int, max_tokens: int,
                      dropout: float) -> None:
        self.token_embedding = nn.Embedding(vocab_size, dim, padding_idx=PAD)
        self.position_embedding = nn.Embedding(max_tokens, dim)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout, batch_first=True, norm_first=True), decoder_layers)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size, bias=False)
        self.out.weight = self.token_embedding.weight
        self.max_tokens = max_tokens

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
    def generate(self, images, widths, max_tokens: int | None = None,
                 starts: list[list[int]] | None = None) -> list[list[int]]:
        """Greedy decoding, each sequence after <bos> and its start tokens (a context the model
        reads, such as seq.meter's), which are returned with it."""
        memory, padding = self.encode(images, widths)
        if not starts or len({len(s) for s in starts}) == 1:
            return self._generate_from(memory, padding, max_tokens, starts)
        out: list[list[int]] = [[] for _ in starts]  # one decoding batch per length of start tokens
        for length in sorted({len(s) for s in starts}):
            chosen = [i for i, s in enumerate(starts) if len(s) == length]
            for i, tokens in zip(chosen, self._generate_from(memory[chosen], padding[chosen], max_tokens,
                                                             [starts[i] for i in chosen])):
                out[i] = tokens
        return out

    def _generate_from(self, memory, padding, max_tokens, starts) -> list[list[int]]:
        b, device = memory.shape[0], memory.device
        tokens = torch.tensor([[BOS] + s for s in starts] if starts else [[BOS]] * b, device=device)
        done = torch.zeros(b, dtype=torch.bool, device=device)
        for _ in range((max_tokens or self.max_tokens) - tokens.shape[1]):
            nxt = self.decode(tokens, memory, padding)[:, -1].argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            tokens = torch.cat([tokens, nxt[:, None]], 1)
            done |= nxt == EOS
            if done.all():
                break
        return [t[1:].tolist() for t in tokens]


class Seq2Seq(_TokenDecoder):
    def __init__(self, vocab_size: int, dim: int = 384, encoder_layers: int = 2, decoder_layers: int = 6,
                 heads: int = 8, max_tokens: int = 1024, height: int = 256, row_stride: int = 32, dropout: float = 0.1,
                 meter_context: bool = False):
        super().__init__()
        self.height = height
        self.meter_context = meter_context  # trained to read a seq.meter prefix
        self.cnn = _cnn(dim, row_stride)
        self.row_embedding = nn.Parameter(torch.zeros(height // row_stride, dim))
        self.encoder = nn.TransformerEncoder(_encoder_layer(dim, heads, dropout), encoder_layers,
                                             enable_nested_tensor=False)
        self._init_decoder(vocab_size, dim, decoder_layers, heads, max_tokens, dropout)

    def encode(self, images: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Memory (B, rows * cols, dim) and its padding mask (True = padding)."""
        f = self.cnn(images)                                   # (B, dim, rows, cols)
        b, d, rows, cols = f.shape
        f = f.permute(0, 2, 3, 1) + self.row_embedding[:rows, None, :] + _sinusoid(cols, d, f.device)[None]
        padding = _column_padding(widths, cols, f.device)[:, None, :].expand(b, rows, cols).reshape(b, rows * cols)
        memory = self.encoder(f.reshape(b, rows * cols, d), src_key_padding_mask=padding)
        return memory, padding


@dataclass
class StaffImages:
    """A batch of systems as staves: pixels (B, S, 1, staff_height, W), S the most staves in the
    batch; counts (B,) the staves each system has, the rest blank padding. A system's staves share
    its width (widths, beside it), since they share its time axis."""
    pixels: torch.Tensor
    counts: torch.Tensor

    def to(self, device) -> StaffImages:
        return StaffImages(self.pixels.to(device), self.counts.to(device))


class StaffSeq2Seq(_TokenDecoder):
    """Each staff is read as its own image at staff_height, so every staff keeps its resolution
    however many a system has. Its feature grid gets row, column and staff embeddings; then each
    block attends within each staff (all its rows and columns), and across the staves at each
    column (all staves' rows there: the same moment in the music). The decoder reads every
    staff's grid. blocks=1 has as many encoder layers, and parameters, as Seq2Seq's two."""

    def __init__(self, vocab_size: int, dim: int = 384, blocks: int = 1, decoder_layers: int = 6, heads: int = 8,
                 max_tokens: int = 1024, staff_height: int = 256, row_stride: int = 32, max_staves: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.staff_height = staff_height
        self.cnn = _cnn(dim, row_stride)
        self.row_embedding = nn.Parameter(torch.zeros(staff_height // row_stride, dim))
        self.staff_embedding = nn.Parameter(torch.zeros(max_staves, dim))
        self.within_staff = nn.ModuleList(_encoder_layer(dim, heads, dropout) for _ in range(blocks))
        self.across_staves = nn.ModuleList(_encoder_layer(dim, heads, dropout) for _ in range(blocks))
        self._init_decoder(vocab_size, dim, decoder_layers, heads, max_tokens, dropout)

    def staff_grid(self, staves: StaffImages) -> torch.Tensor:
        """Each staff's embedded feature grid, (B, S, rows, cols, dim): no staff has seen another."""
        b, s = staves.pixels.shape[:2]
        f = self.cnn(staves.pixels.flatten(0, 1))              # (B * S, dim, rows, cols)
        _, d, rows, cols = f.shape
        f = f.permute(0, 2, 3, 1) + self.row_embedding[:rows, None, :] + _sinusoid(cols, d, f.device)[None]
        return f.reshape(b, s, rows, cols, d) + self.staff_embedding[:s, None, None, :]

    def encode(self, staves: StaffImages, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Memory (B, S * rows * cols, dim), staff by staff, and its padding mask (True = padding:
        a column past the crop's width, or a staff the system does not have)."""
        f = self.staff_grid(staves)
        b, s, rows, cols, d = f.shape
        blank_staff = torch.arange(s, device=f.device)[None, :] >= staves.counts.to(f.device)[:, None]   # (B, S)
        past_width = _column_padding(widths, cols, f.device)                                            # (B, cols)
        within_padding = past_width[:, None, None, :].expand(b, s, rows, cols).reshape(b * s, rows * cols)
        across_padding = blank_staff[:, None, :, None].expand(b, cols, s, rows).reshape(b * cols, s * rows)
        for within, across in zip(self.within_staff, self.across_staves):
            f = within(f.reshape(b * s, rows * cols, d), src_key_padding_mask=within_padding)
            f = f.reshape(b, s, rows, cols, d).permute(0, 3, 1, 2, 4).reshape(b * cols, s * rows, d)
            f = across(f, src_key_padding_mask=across_padding)
            f = f.reshape(b, cols, s, rows, d).permute(0, 2, 3, 1, 4)
        padding = (blank_staff[:, :, None, None] | past_width[:, None, None, :]).expand(b, s, rows, cols)
        return f.reshape(b, s * rows * cols, d), padding.reshape(b, s * rows * cols)
