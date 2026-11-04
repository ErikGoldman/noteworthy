"""How the network's output logits become per-channel probabilities, and its loss.

Most channels are independent (a stem pixel may also be a beam pixel), so each gets its
own sigmoid. Some channels form families of mutually exclusive glyphs: a time-signature
glyph is one of 0-9, common or cut time, never several. Independent sigmoids let such
families fire together (ts_5, ts_6 and ts_9 on one "6"), and penalizing false positives
harder only made the network give up on the confusable classes. A family therefore
shares one softmax, with an implicit "none of these" class whose logit is fixed at 0:

    p_k = exp(l_k) / (1 + sum_j exp(l_j))

so the output keeps exactly one probability per channel.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .channels import CHANNEL_INDEX, CHANNELS

EXCLUSIVE_FAMILIES = [
    ["head_filled", "head_half", "head_whole"],
    ["flag_8", "flag_16", "flag_32"],
    ["sharp", "flat", "natural", "double_sharp", "double_flat"],
    ["clef_g", "clef_f", "clef_c"],
    ["rest_whole", "rest_half", "rest_quarter", "rest_8", "rest_16", "rest_32"],
    ["ts_0", "ts_1", "ts_2", "ts_3", "ts_4", "ts_5", "ts_6", "ts_7", "ts_8", "ts_9", "ts_common", "ts_cut"],
    ["tuplet_3", "tuplet_other"],
]
FAMILY_INDICES = [[CHANNEL_INDEX[name] for name in family] for family in EXCLUSIVE_FAMILIES]
INDEPENDENT_INDICES = sorted(set(range(len(CHANNELS))) - {i for family in FAMILY_INDICES for i in family})


def _with_null(logits: torch.Tensor, family: list[int]) -> torch.Tensor:
    """(N, 1 + K, H, W): a zero "none" logit followed by the family's logits."""
    member_logits = logits[:, family]
    return torch.cat([torch.zeros_like(member_logits[:, :1]), member_logits], 1)


def probabilities(logits: torch.Tensor, families: list[list[int]] = FAMILY_INDICES) -> torch.Tensor:
    """Per-channel probabilities from (N, C, H, W) logits. `families=[]` gives plain
    sigmoids, as models trained before the families existed expect."""
    channels: list[torch.Tensor | None] = [None] * logits.shape[1]
    for family in families:
        family_prob = torch.softmax(_with_null(logits, family), 1)
        for k, c in enumerate(family):
            channels[c] = family_prob[:, k + 1:k + 2]
    for c, existing in enumerate(channels):
        if existing is None:
            channels[c] = torch.sigmoid(logits[:, c:c + 1])
    return torch.cat(channels, 1)


def loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Classification loss (BCE for independent channels, cross-entropy within each family)
    plus soft Dice over the channels present in the batch, so rare symbols are not drowned
    out by background."""
    # Weighted by share of channels, so the total matches a per-channel BCE in scale.
    terms = [F.binary_cross_entropy_with_logits(logits[:, INDEPENDENT_INDICES], target[:, INDEPENDENT_INDICES])
             * len(INDEPENDENT_INDICES) / len(CHANNELS)]
    for family in FAMILY_INDICES:
        members = target[:, family]
        # Rendered glyphs may share an anti-aliased boundary pixel; split such pixels evenly.
        claimed = members.sum(1, keepdim=True)
        soft_target = torch.cat([(1 - claimed).clamp(min=0), members / claimed.clamp(min=1)], 1)
        log_prob = torch.log_softmax(_with_null(logits, family), 1)
        terms.append(-(soft_target * log_prob).sum(1).mean() * len(family) / len(CHANNELS))
    prob = probabilities(logits)
    dims = (0, 2, 3)
    inter = (prob * target).sum(dims)
    denom = prob.sum(dims) + target.sum(dims)
    present = target.sum(dims) > 0
    dice = 1 - (2 * inter + 1) / (denom + 1)
    return sum(terms) + (dice[present].mean() if present.any() else 0)
