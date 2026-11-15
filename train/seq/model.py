"""Image-to-sequence models for one system crop, both ending in a transformer decoder that emits
LMX tokens. Seq2Seq reads the system as one image: a ResNet encoder and a transformer encoder over
its 2D feature grid. StaffSeq2Seq reads each staff as its own image at a fixed height, whatever
the number of staves, then lets the staves attend to each other column by column."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .staves import STAFF_INTERLINES
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
                 starts: list[list[int]] | None = None, beam: int = 1) -> list[list[int]]:
        """Each sequence after <bos> and its start tokens (a context the model reads, such as
        seq.meter's), which are returned with it: greedy, or the best of `beam` readings kept
        side by side (beam search), one crop at a time."""
        memory, padding = self.encode(images, widths)
        if beam > 1:
            return [self._beam_from(memory[i:i + 1], padding[i:i + 1], max_tokens, starts[i] if starts else [], beam)
                    for i in range(memory.shape[0])]
        if not starts or len({len(s) for s in starts}) == 1:
            return self._generate_from(memory, padding, max_tokens, starts)
        out: list[list[int]] = [[] for _ in starts]  # one decoding batch per length of start tokens
        for length in sorted({len(s) for s in starts}):
            chosen = [i for i, s in enumerate(starts) if len(s) == length]
            for i, tokens in zip(chosen, self._generate_from(memory[chosen], padding[chosen], max_tokens,
                                                             [starts[i] for i in chosen])):
                out[i] = tokens
        return out

    def _beam_from(self, memory, padding, max_tokens, start: list[int], width: int) -> list[int]:
        return self.beam_readings(memory, padding, max_tokens, start, width)[0]

    def beam_readings(self, memory, padding, max_tokens, start: list[int], width: int) -> list[list[int]]:
        """Readings of one crop, the best mean log-probability per token first, among `width` kept
        at each step; a reading that ends stops there, and the search when `width` have ended."""
        prefix = [BOS] + start
        live, ended = [(prefix, 0.0)], []
        while live and len(live[0][0]) < (max_tokens or self.max_tokens) and len(ended) < width:
            tokens = torch.tensor([seq for seq, _ in live], device=memory.device)
            n = len(live)
            logits = self.decode(tokens, memory.expand(n, -1, -1), padding.expand(n, -1))[:, -1].float()
            scores = torch.tensor([score for _, score in live], device=memory.device)[:, None] + logits.log_softmax(-1)
            top = scores.flatten().topk(width)
            live = []
            for score, index in zip(top.values.tolist(), top.indices.tolist()):
                which, token = divmod(index, scores.shape[1])
                seq = tokens[which].tolist() + [token]
                (ended if token == EOS else live).append((seq, score))
        ranked = sorted(ended or live, key=lambda kept: -kept[1] / (len(kept[0]) - len(prefix)))
        return [seq[1:] for seq, _ in ranked]

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


class RowRotaryEncoderLayer(nn.Module):
    """A pre-norm encoder layer (as _encoder_layer's) whose queries and keys are rotated by their
    feature row (rotary position embedding, on the row axis only): the score of two positions
    depends on how many rows apart they are, not on where they are. Unlike a learned bias added
    to the scores, it leaves PyTorch's fused attention usable: a bias table over every pair of a
    wide crop's positions took 5 GB a layer and half the speed (EXPERIMENTS_STAFF_MODEL.md)."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads, self.dropout = heads, dropout
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.qkv, self.out = nn.Linear(dim, 3 * dim), nn.Linear(dim, dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(4 * dim, dim))
        self.drop1, self.drop2 = nn.Dropout(dropout), nn.Dropout(dropout)
        half = dim // heads // 2
        self.register_buffer("frequencies", 1.0 / 10000 ** (torch.arange(half, dtype=torch.float32) / half), persistent=False)

    def rotate(self, x: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """x (B, heads, L, head_dim) rotated pairwise by each position's row (L,)."""
        angle = rows[:, None].float() * self.frequencies[None, :]       # (L, head_dim / 2)
        cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)

    def forward(self, x: torch.Tensor, rows: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
        b, length, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, length, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        keep = ~padding[:, None, None, :]                               # (B, 1, 1, L): True = attend
        attended = nn.functional.scaled_dot_product_attention(self.rotate(q, rows), self.rotate(k, rows), v,
                                                              attn_mask=keep, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop1(self.out(attended.transpose(1, 2).reshape(b, length, d)))
        return x + self.drop2(self.ff(self.norm2(x)))


@dataclass
class SystemImages:
    """A batch of whole systems at staff scale (Seq2Seq with interline_scaled): pixels (B, 1, H, W);
    centres (B, S), each system's staff centres in pixel rows, top to bottom, S the most staves in
    the batch (the rest padding); counts (B,) the staves each has; interlines (B,) in pixels."""
    pixels: torch.Tensor
    centres: torch.Tensor
    counts: torch.Tensor
    interlines: torch.Tensor

    def to(self, device, non_blocking: bool = False) -> SystemImages:
        return SystemImages(*(t.to(device, non_blocking=non_blocking)
                              for t in (self.pixels, self.centres, self.counts, self.interlines)))


# A row's offset from its staff's centre is embedded from waves of these periods, in interlines:
# from the half interline between a line and a space up to past a staff's reach.
OFFSET_PERIODS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
# Rows farther than this from their staff (wide margins) all read as this far.
OFFSET_REACH = 16.0


class Seq2Seq(_TokenDecoder):
    """relative_rows: instead of each feature row's own embedding (where it is in the crop), the
    encoder's queries and keys are rotated by their row (RowRotaryEncoderLayer), so heights are
    read relative to one another and a staff means the same wherever it sits in the crop, as a
    StaffSeq2Seq staff crop's centred staff does (EXPERIMENTS_STAFF_MODEL.md, "a fixed frame for
    pitch").

    row_windows: with relative rows, an encoder whose attention grows with the crop's height, not
    its square, for systems of many staves: attention within bands of row_windows feature rows
    (all columns), within the same bands shifted by half, then down each column (all rows).
    Read at staff scale (interline_scaled, seq.staff_data.at_interline) a band of 8 rows is about
    15 interlines, about a staff. interline_scaled is how the model's crops are scaled (each
    system so that its interline is a staff crop's), not a weight, recorded so that everything
    that reads it scales the same. With absolute rows it needs `height` to be the tallest such
    crop (seq.staff_data.MAX_SYSTEM_HEIGHT): at a fixed interline a row is then a fixed distance
    from the crop's top, which at a fixed height it is not.

    staff_anchored (interline_scaled): StaffSeq2Seq's fixed frame without cutting the system up.
    In place of where a row is in the crop, each feature row's embedding says which staff is
    nearest it and how far from that staff's centre it lies, in interlines (staff_frame), so a row
    on a staff's top line reads the same in every system, as it does in a staff crop.

    staff_bands (with staff_anchored): StaffSeq2Seq's attention too, one layer within each staff's
    band (the rows nearer it than any other staff), then one down each column across them all.
    Unlike staff crops the bands do not overlap: the gap between two staves is split between them,
    and the column layer joins the halves.

    staff_windows (with staff_anchored, in place of staff_bands): StaffSeq2Seq's encoder on the
    whole system's features. Each staff's window is the feature rows its staff crop would hold
    (STAFF_INTERLINES), so neighbours' windows overlap as their crops do, each row embedded by its
    offset from the window's own staff; one layer within each window, then one down each column
    across all windows. What is left of StaffSeq2Seq's difference is that the convolutions see the
    whole system, not each staff alone."""

    def __init__(self, vocab_size: int, dim: int = 384, encoder_layers: int = 2, decoder_layers: int = 6,
                 heads: int = 8, max_tokens: int = 1024, height: int = 256, row_stride: int = 32, dropout: float = 0.1,
                 meter_context: bool = False, relative_rows: bool = False, row_windows: int = 0,
                 interline_scaled: bool = False, staff_anchored: bool = False, staff_bands: bool = False,
                 staff_windows: bool = False, max_staves: int = 8):
        super().__init__()
        if row_windows and not relative_rows:
            raise ValueError("row_windows needs relative_rows: a band's rows have no fixed place in the crop")
        if staff_anchored and (relative_rows or not interline_scaled):
            raise ValueError("staff_anchored replaces relative_rows, and needs interline_scaled for the staves")
        if (staff_bands or staff_windows) and not staff_anchored:
            raise ValueError("staff_bands and staff_windows need staff_anchored: the bands are the staves'")
        if staff_bands and staff_windows:
            raise ValueError("staff_windows replaces staff_bands")
        self.height, self.row_stride = height, row_stride
        self.meter_context = meter_context  # trained to read a seq.meter prefix
        self.relative_rows, self.row_windows, self.interline_scaled = relative_rows, row_windows, interline_scaled
        self.staff_anchored, self.staff_bands, self.staff_windows = staff_anchored, staff_bands, staff_windows
        self.cnn = _cnn(dim, row_stride)
        if staff_anchored:
            self.offset_embedding = nn.Linear(2 * len(OFFSET_PERIODS), dim)
            self.staff_embedding = nn.Parameter(torch.zeros(max_staves, dim))
        if staff_bands or staff_windows:
            self.band_layer, self.column_layer = _encoder_layer(dim, heads, dropout), _encoder_layer(dim, heads, dropout)
        elif staff_anchored:
            self.encoder = nn.TransformerEncoder(_encoder_layer(dim, heads, dropout), encoder_layers,
                                                 enable_nested_tensor=False)
        elif relative_rows:
            layers = 3 if row_windows else encoder_layers    # band, shifted band, column
            self.row_layers = nn.ModuleList(RowRotaryEncoderLayer(dim, heads, dropout) for _ in range(layers))
        else:
            self.row_embedding = nn.Parameter(torch.zeros(height // row_stride, dim))
            self.encoder = nn.TransformerEncoder(_encoder_layer(dim, heads, dropout), encoder_layers,
                                                 enable_nested_tensor=False)
        self._init_decoder(vocab_size, dim, decoder_layers, heads, max_tokens, dropout)

    def encode(self, images: torch.Tensor | SystemImages, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Memory (B, rows * cols, dim) and its padding mask (True = padding). An interline_scaled
        model reads SystemImages, the others a (B, 1, H, W) tensor."""
        f = self.cnn(images.pixels if self.interline_scaled else images)   # (B, dim, rows, cols)
        b, d, rows, cols = f.shape
        f = f.permute(0, 2, 3, 1) + _sinusoid(cols, d, f.device)[None]
        past_width = _column_padding(widths, cols, f.device)  # (B, cols)
        padding = past_width[:, None, :].expand(b, rows, cols).reshape(b, rows * cols)
        if self.staff_windows:
            return self.encode_staff_windows(f, images, past_width)
        if self.staff_anchored:
            nearest, offset = self.staff_frame(images, rows)
            f = f + self.frame_embedding(nearest, offset)[:, :, None, :]
            if self.staff_bands:
                f = self.attend_in_staff_bands(f, nearest, past_width)
                memory = _down_columns(f, past_width, lambda x, m: self.column_layer(x, src_key_padding_mask=m))
                return memory.reshape(b, rows * cols, d), padding
            return self.encoder(f.reshape(b, rows * cols, d), src_key_padding_mask=padding), padding
        if self.row_windows:
            f = self.attend_in_bands(self.row_layers[0], f, past_width, shift=0)
            f = self.attend_in_bands(self.row_layers[1], f, past_width, shift=self.row_windows // 2)
            memory = _down_columns(f, past_width, lambda x, m: self.row_layers[2](x, torch.arange(x.shape[1], device=x.device), m))
            memory = memory.reshape(b, rows * cols, d)
        elif self.relative_rows:
            memory = f.reshape(b, rows * cols, d)
            row_of = torch.arange(rows, device=f.device).repeat_interleave(cols)
            for layer in self.row_layers:
                memory = layer(memory, row_of, padding)
        else:
            f = f + self.row_embedding[:rows, None, :]
            memory = self.encoder(f.reshape(b, rows * cols, d), src_key_padding_mask=padding)
        return memory, padding

    def attend_in_bands(self, layer: nn.Module, f: torch.Tensor, past_width: torch.Tensor, shift: int) -> torch.Tensor:
        """One layer over bands of row_windows rows, the first band `shift` rows short (so a shifted
        pass straddles the last one's band edges); f (B, rows, cols, d) in and out."""
        b, rows, cols, d = f.shape
        w = self.row_windows
        below = -(rows + shift) % w
        x = nn.functional.pad(f, (0, 0, 0, 0, shift, below))   # padded rows: masked, then cut off
        bands = (rows + shift + below) // w
        row_padding = torch.ones(rows + shift + below, dtype=torch.bool, device=f.device)
        row_padding[shift:shift + rows] = False
        mask = row_padding[None, :, None] | past_width[:, None, :]           # (B, all rows, cols)
        x = x.reshape(b * bands, w * cols, d)
        mask = mask.reshape(b, bands, w, cols).reshape(b * bands, w * cols)
        mask = mask & ~mask.all(dim=1, keepdim=True)        # a band wholly padding would attend to nothing
        row_of = torch.arange(w, device=f.device).repeat_interleave(cols)
        x = layer(x, row_of, mask).reshape(b, bands * w, cols, d)
        return x[:, shift:shift + rows]

    def staff_frame(self, images: SystemImages, rows: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Each feature row's nearest staff (B, rows) and its offset from that staff's centre in
        interlines (B, rows), below positive, clamped to OFFSET_REACH."""
        device = images.centres.device
        middle = (torch.arange(rows, device=device, dtype=torch.float32) + 0.5) * self.row_stride   # a row's centre, px
        apart = (middle[None, None, :] - images.centres[:, :, None].float())                        # (B, S, rows)
        lacking = torch.arange(images.centres.shape[1], device=device)[None, :] >= images.counts[:, None]
        nearest = apart.abs().masked_fill(lacking[:, :, None], float("inf")).argmin(dim=1)        # (B, rows)
        offset = apart.gather(1, nearest[:, None, :])[:, 0] / images.interlines[:, None].float()
        return nearest, offset.clamp(-OFFSET_REACH, OFFSET_REACH)

    def frame_embedding(self, staff: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
        """(..., dim): which staff a row belongs to, and where on it; staff and offset (...)."""
        angles = 2 * math.pi * offset[..., None] / offset.new_tensor(OFFSET_PERIODS)
        waves = torch.cat([angles.sin(), angles.cos()], dim=-1)
        return self.offset_embedding(waves.to(self.offset_embedding.weight.dtype)) + self.staff_embedding[staff]

    def attend_in_staff_bands(self, f: torch.Tensor, nearest: torch.Tensor, past_width: torch.Tensor) -> torch.Tensor:
        """band_layer within each staff's band; f (B, rows, cols, d) in and out. A staff's rows are
        contiguous (each row goes to its nearest centre), so a band is a start and a length: bands
        are gathered into (B * S, longest band * cols), padded and masked, and scattered back."""
        b, rows, cols, d = f.shape
        staves = int(nearest.max()) + 1
        member = nearest[:, None, :] == torch.arange(staves, device=f.device)[None, :, None]    # (B, S, rows)
        length = member.sum(-1)                                                                 # (B, S)
        start = member.int().argmax(-1)                                                         # (B, S)
        longest = int(length.max())
        step = torch.arange(longest, device=f.device)
        taken = (start[..., None] + step).clamp(max=rows - 1)                                   # (B, S, longest)
        x = f.gather(1, taken.reshape(b, -1, 1, 1).expand(-1, -1, cols, d)).reshape(b * staves, longest * cols, d)
        outside = (step >= length[..., None])[..., None] | past_width[:, None, None, :]        # (B, S, longest, cols)
        mask = outside.reshape(b * staves, longest * cols)
        mask = mask & ~mask.all(dim=1, keepdim=True)        # a staff the system lacks would attend to nothing
        x = self.band_layer(x, src_key_padding_mask=mask).reshape(b, staves * longest, cols, d)
        place = nearest * longest + torch.arange(rows, device=f.device)[None] - start.gather(1, nearest)
        return x.gather(1, place[:, :, None, None].expand(-1, -1, cols, d))

    def encode_staff_windows(self, f: torch.Tensor, images: SystemImages,
                             past_width: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Memory (B, S * window * cols, dim), staff by staff, and its padding mask: a window's rows
        beyond the system's, a staff it lacks, a column past its width."""
        b, rows, cols, d = f.shape
        staves = images.centres.shape[1]
        window = round(STAFF_INTERLINES * float(images.interlines.max()) / self.row_stride)
        centres = images.centres.float()[..., None]                                          # (B, S, 1) px
        start = (centres / self.row_stride - window / 2).round().long()
        taken = start + torch.arange(window, device=f.device)                                # (B, S, window)
        lacking = torch.arange(staves, device=f.device)[None, :] >= images.counts[:, None]   # (B, S)
        outside = (taken < 0) | (taken >= rows) | lacking[..., None]
        x = f.gather(1, taken.clamp(0, rows - 1).reshape(b, -1, 1, 1).expand(-1, -1, cols, d))
        offset = ((taken.float() + 0.5) * self.row_stride - centres) / images.interlines.float()[:, None, None]
        staff = torch.arange(staves, device=f.device)[None, :, None].expand(b, staves, window)
        frame = self.frame_embedding(staff, offset.clamp(-OFFSET_REACH, OFFSET_REACH))      # (B, S, window, d)
        x = x.reshape(b, staves, window, cols, d) + frame[:, :, :, None, :]
        padding = outside[..., None] | past_width[:, None, None, :]                          # (B, S, window, cols)
        within = padding.reshape(b * staves, window * cols)
        within = within & ~within.all(dim=1, keepdim=True)        # a staff the system lacks would attend to nothing
        x = self.band_layer(x.reshape(b * staves, window * cols, d), src_key_padding_mask=within)
        x = _down_columns(x.reshape(b, staves * window, cols, d), past_width,
                          lambda y, m: self.column_layer(y, src_key_padding_mask=m), outside.reshape(b, -1))
        return x.reshape(b, -1, d), padding.reshape(b, -1)


def _down_columns(f: torch.Tensor, past_width: torch.Tensor, layer, row_padding: torch.Tensor | None = None) -> torch.Tensor:
    """layer(x, padding) down each column, all its rows but row_padding's (B, rows); f (B, rows,
    cols, d) in and out."""
    b, rows, cols, d = f.shape
    x = f.permute(0, 2, 1, 3).reshape(b * cols, rows, d)
    mask = past_width[:, :, None].expand(b, cols, rows)
    if row_padding is not None:
        mask = mask | row_padding[:, None, :]
    mask = mask.reshape(b * cols, rows)
    mask = mask & ~mask.all(dim=1, keepdim=True)
    return layer(x, mask).reshape(b, cols, rows, d).permute(0, 2, 1, 3)


@dataclass
class StaffImages:
    """A batch of systems as staves: pixels (B, S, 1, staff_height, W), S the most staves in the
    batch; counts (B,) the staves each system has, the rest blank padding. A system's staves share
    its width (widths, beside it), since they share its time axis."""
    pixels: torch.Tensor
    counts: torch.Tensor

    def to(self, device, non_blocking: bool = False) -> StaffImages:
        return StaffImages(self.pixels.to(device, non_blocking=non_blocking), self.counts.to(device, non_blocking=non_blocking))


class StaffSeq2Seq(_TokenDecoder):
    """Each staff is read as its own image at staff_height, so every staff keeps its resolution
    however many a system has. Its feature grid gets row, column and staff embeddings; then each
    block attends within each staff (all its rows and columns), and across the staves at each
    column (all staves' rows there: the same moment in the music). The decoder reads every
    staff's grid. blocks=1 has as many encoder layers, and parameters, as Seq2Seq's two.

    across_columns: how many neighbouring columns the attention across staves takes together.
    1 is a single column; k > 1 groups of k columns side by side (a group, not a window around
    each column, so the layer stays a plain one over a reshaped grid); 0 none, each staff
    encoded alone and only the decoder reading them together. staff_outward: how many interlines
    the top and bottom staves' crops are slid away from the system (seq.staves.staff_crops); not
    a weight, but the crops the model was trained on, so everything that reads it cuts the same;
    staff_cut_between likewise (crops white past the midpoints between staves).
    meter_context: trained to read a seq.meter prefix, as Seq2Seq's option."""

    def __init__(self, vocab_size: int, dim: int = 384, blocks: int = 1, decoder_layers: int = 6, heads: int = 8,
                 max_tokens: int = 1024, staff_height: int = 256, row_stride: int = 32, max_staves: int = 8,
                 dropout: float = 0.1, across_columns: int = 1, staff_outward: float = 0.0, meter_context: bool = False,
                 staff_cut_between: bool = False):
        super().__init__()
        self.staff_height, self.across_columns, self.staff_outward = staff_height, across_columns, staff_outward
        self.staff_cut_between = staff_cut_between
        self.meter_context = meter_context
        self.cnn = _cnn(dim, row_stride)
        self.row_embedding = nn.Parameter(torch.zeros(staff_height // row_stride, dim))
        self.staff_embedding = nn.Parameter(torch.zeros(max_staves, dim))
        self.within_staff = nn.ModuleList(_encoder_layer(dim, heads, dropout) for _ in range(blocks))
        self.across_staves = nn.ModuleList(_encoder_layer(dim, heads, dropout) for _ in range(blocks if across_columns else 0))
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
        for k, within in enumerate(self.within_staff):
            f = within(f.reshape(b * s, rows * cols, d), src_key_padding_mask=within_padding).reshape(b, s, rows, cols, d)
            if self.across_staves:
                f = self.attend_across(self.across_staves[k], f, blank_staff, past_width)
        padding = (blank_staff[:, :, None, None] | past_width[:, None, None, :]).expand(b, s, rows, cols)
        return f.reshape(b, s * rows * cols, d), padding.reshape(b, s * rows * cols)

    def attend_across(self, layer: nn.Module, f: torch.Tensor, blank_staff: torch.Tensor,
                      past_width: torch.Tensor) -> torch.Tensor:
        """One layer of attention across the staves, over groups of across_columns columns."""
        b, s, rows, cols, d = f.shape
        group = self.across_columns
        extra = -cols % group
        if extra:
            f = nn.functional.pad(f, (0, 0, 0, extra))
            past_width = nn.functional.pad(past_width, (0, extra), value=True)
        groups = (cols + extra) // group
        # (B, S, rows, groups, group, d) -> (B * groups, S * rows * group, d)
        x = f.reshape(b, s, rows, groups, group, d).permute(0, 3, 1, 2, 4, 5).reshape(b * groups, s * rows * group, d)
        if group == 1:  # as trained before groups: only staves the system lacks are masked
            mask = blank_staff[:, None, :, None].expand(b, groups, s, rows)
        else:  # a group may run past the crop's width: those columns are masked too
            masked = blank_staff[:, None, :, None, None] | past_width.reshape(b, groups, 1, 1, group)
            mask = masked.expand(b, groups, s, rows, group)
        mask = mask.reshape(b * groups, -1)
        # A group wholly past the width would attend to nothing (NaN); its output is padding anyway.
        mask = mask & ~mask.all(dim=1, keepdim=True)
        x = layer(x, src_key_padding_mask=mask)
        x = x.reshape(b, groups, s, rows, group, d).permute(0, 2, 3, 1, 4, 5).reshape(b, s, rows, groups * group, d)
        return x[:, :, :, :cols]
