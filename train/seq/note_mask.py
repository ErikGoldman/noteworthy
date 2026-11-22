"""Hiding a note from the decoder while it writes the note's length, so that only the bar around it
(the time already written, the other staff, the barline) can tell it: the model made to count
(TODO, priority 1). The decoder is run layer by layer as nn.TransformerDecoder runs it (norm first),
with a cross-attention mask per output position: where the next token is a length or a tuplet
ratio, the image columns the decoder looked at most for it, and a margin either side, are hidden in
every row. Where it looked comes from a pass without the mask: the cross-attention of the layers
that follow the notes (LOCATING_LAYERS, heads averaged). In (l) layers 1 and 2 move along the system
note by note; layer 0 wanders and layers 3-5 rest on the clef at the left edge (2026-11-20). They
look at the note while writing its pitch (or `rest`), not its length, so a length or ratio is hidden
at the columns its note's pitch step looked at. Memory is (rows * cols) positions laid out row by row (Seq2Seq.encode), so column c is
every position k with k % cols == c.

The encoder mixes columns, so a hidden column's neighbours may still carry the note: hidden_accuracy
in eval measures how much of it the mask really hides."""
from __future__ import annotations

import re

import torch
import torch.nn.functional as F

from .lmx_tuplets import LENGTHS
from .vocab import PAD, Vocab

RATIO = re.compile(r"^\d+in\d+$")
LOCATING_LAYERS = (1, 2)


PITCH = re.compile(r"^[A-G][#b]*-?\d$")


def note_starts(vocab: Vocab) -> torch.Tensor:
    """(vocab,) True for the tokens that name a note or rest, where the decoder looks at it."""
    out = torch.zeros(len(vocab.tokens), dtype=torch.bool)
    for token, index in vocab.index.items():
        out[index] = bool(PITCH.match(token)) or token == "rest"
    return out


def hidden_targets(vocab: Vocab) -> torch.Tensor:
    """(vocab,) True for the tokens whose prediction hides the note: lengths and tuplet ratios."""
    out = torch.zeros(len(vocab.tokens), dtype=torch.bool)
    for token, index in vocab.index.items():
        out[index] = token in LENGTHS or bool(RATIO.match(token))
    return out


def decode_layers(model, tokens: torch.Tensor, memory: torch.Tensor, memory_padding: torch.Tensor,
                  memory_mask: torch.Tensor | None = None, want_attention: bool = False):
    """model.decode's logits (and, if wanted, the cross-attention (B, T, S) of LOCATING_LAYERS, heads
    averaged and summed), with memory_mask (B * heads, T, S), True = hidden, on every layer's."""
    n = tokens.shape[1]
    x = model.token_embedding(tokens) + model.position_embedding(torch.arange(n, device=tokens.device))[None]
    causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=tokens.device), 1)
    padding = tokens == PAD
    attention = None
    layers = model.decoder.layers
    for k, layer in enumerate(layers):
        x = x + layer._sa_block(layer.norm1(x), causal, padding, is_causal=True)
        query = layer.norm2(x)
        if want_attention and k in LOCATING_LAYERS:
            attended, weights = layer.multihead_attn(query, memory, memory, attn_mask=memory_mask,
                                                     key_padding_mask=memory_padding, need_weights=True,
                                                     average_attn_weights=True)
            attention = weights if attention is None else attention + weights
            x = x + layer.dropout2(attended)
        else:
            x = x + layer._mha_block(query, memory, memory_mask, memory_padding)
        x = x + layer._ff_block(layer.norm3(x))
    logits = model.out(model.norm(x))
    return (logits, attention) if want_attention else logits


def column_attention(attention: torch.Tensor, cols: int) -> torch.Tensor:
    """(B, T, cols): each output position's attention summed over the rows of each column."""
    b, t, s = attention.shape
    return attention.reshape(b, t, s // cols, cols).sum(2)


def note_columns(attention: torch.Tensor, cols: int, starts: torch.Tensor) -> torch.Tensor:
    """(B, T) the column of each position's note: where the decoder looked most at the latest
    position at or before it whose next token named a note or rest (starts (B, T))."""
    peak = column_attention(attention, cols).argmax(-1)                       # (B, T)
    t = peak.shape[1]
    index = torch.where(starts, torch.arange(t, device=peak.device)[None, :], torch.zeros_like(peak))
    latest = torch.cummax(index, dim=1).values
    return peak.gather(1, latest)


def note_mask(attention: torch.Tensor, cols: int, hide: torch.Tensor, starts: torch.Tensor, margin: int,
              heads: int) -> torch.Tensor:
    """(B * heads, T, S) cross-attention mask: for each output position in hide (B, T), the columns
    within margin of its note's, in every row."""
    b, t, s = attention.shape
    peak = note_columns(attention, cols, starts)
    columns = torch.arange(cols, device=attention.device)
    near = (columns[None, None, :] - peak[..., None]).abs() <= margin         # (B, T, cols)
    hidden = near & hide[..., None]
    mask = hidden[:, :, None, :].expand(b, t, s // cols, cols).reshape(b, t, s)
    return mask.repeat_interleave(heads, dim=0)


def masked_logits(model, tokens: torch.Tensor, memory: torch.Tensor, memory_padding: torch.Tensor,
                  hide: torch.Tensor, starts: torch.Tensor, margin: int) -> torch.Tensor:
    """Logits with the note hidden at the output positions in hide (B, T): a pass without gradient
    to find where each position looks, then the pass with those columns hidden."""
    cols = memory.shape[1] // model.row_embedding.shape[0]  # a Seq2Seq's memory: its rows, then columns
    with torch.no_grad():
        _, attention = decode_layers(model, tokens, memory, memory_padding, want_attention=True)
    mask = note_mask(attention, cols, hide, starts, margin, model.decoder.layers[0].multihead_attn.num_heads)
    return decode_layers(model, tokens, memory, memory_padding, memory_mask=mask)


def hidden_positions(tokens: torch.Tensor, is_target: torch.Tensor) -> torch.Tensor:
    """(B, T-1) True at the input positions whose next token is a length or ratio (tokens (B, T))."""
    return is_target.to(tokens.device)[tokens[:, 1:]]


def masked_loss(logits: torch.Tensor, targets: torch.Tensor, hide: torch.Tensor) -> torch.Tensor:
    """Cross-entropy at the hidden positions only."""
    chosen = hide & (targets != PAD)
    return F.cross_entropy(logits[chosen].float(), targets[chosen]) if chosen.any() else logits.sum() * 0
