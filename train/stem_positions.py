"""Where handwritten stems end: data from MUSCIMA++ and the models that predict it, for the
Smashcima fork's stem synthesis (../Smashcima, smashcima/synthesis/notation/StemModel.py).

After Shatri, Mayer and Hajič, "Data-Driven Glyph Positioning for Handwritten Music Synthesis"
(arXiv 2610.28899), with one change: their GNN takes every glyph's position as input, the stem's
too, so it cannot run when the stem has yet to be drawn. Here each chord is a node holding only
what Smashcima knows before it draws stems: its noteheads' staff positions, the stem direction,
its beams or flags, its beam group, where it sits along the staff. The model predicts the stem's
length (from the notehead nearest the tip to the tip) and its sideways offset from that
notehead, both in millimetres.

usage (from the repo root, CPU):
  .venv/bin/python train/stem_positions.py evaluate     # writer-held-out folds, against baselines
  .venv/bin/python train/stem_positions.py export OUT.npz   # train on every training page
"""
from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from omr.muscima import Page, training_pages  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MUSCIMA = ROOT / "data" / "raw" / "muscima"
MM_PER_PX = 25.4 / 300  # MUSCIMA++ scans are 300 DPI
NOTEHEADS = {"noteheadFull", "noteheadHalf"}
FLAGS = {"flag8thUp", "flag8thDown", "flag16thUp", "flag16thDown", "flag32ndUp", "flag32ndDown"}

FEATURES = ["top_position", "bottom_position", "noteheads", "stem_up", "filled", "beams", "flags",
            "beamed", "group_size", "group_index", "x_mm"]
"""Per chord, in this order. Positions in staff steps from the middle line (up positive);
group_index runs 0 to 1 along a beam group; x_mm from the staff's left end, only to order and
space neighbours."""
TARGETS = ["length_mm", "dx_mm"]
SEQUENCE, BEAM_GROUP = 0, 1
EDGE_TYPES = 2


@dataclass
class Staff:
    """Chords of one staff, left to right: features, targets, and edges by type"""
    writer: int
    features: np.ndarray  # chords x FEATURES
    targets: np.ndarray   # chords x TARGETS
    edges: list[tuple[int, int, int]]  # (source, target, type), both directions


def centre(node) -> tuple[float, float]:
    return node.left + node.width / 2, node.top + node.height / 2


def staff_geometry(staff, by_id) -> tuple[float, float] | None:
    """(middle line y, staff space) in pixels, or None without five staff lines"""
    lines = sorted(centre(by_id[i])[1] for i in staff.outlinks if by_id[i].class_name == "staffLine")
    if len(lines) != 5:
        return None
    return lines[2], (lines[4] - lines[0]) / 4


def page_staves(page: Page) -> list[Staff]:
    from mung.io import read_nodes_from_file
    nodes = read_nodes_from_file(str(page.annotation))
    by_id = {n.id: n for n in nodes}
    chords_by_staff = defaultdict(list)
    heads_by_stem = defaultdict(list)
    for n in nodes:
        if n.class_name not in NOTEHEADS:
            continue
        stems = [by_id[o] for o in n.outlinks if by_id[o].class_name == "stem"]
        staves = [by_id[o] for o in n.outlinks if by_id[o].class_name == "staff"]
        if len(stems) == 1 and len(staves) == 1:
            heads_by_stem[stems[0].id].append((n, staves[0]))
    for stem_id, heads in heads_by_stem.items():
        staff = heads[0][1]
        if any(s.id != staff.id for _, s in heads):
            continue  # a chord across staves
        geometry = staff_geometry(staff, by_id)
        if geometry is None:
            continue
        middle, space = geometry
        stem = by_id[stem_id]
        linked = {o for h, _ in heads for o in h.outlinks}
        beams = {o for o in linked if by_id[o].class_name == "beam"}
        flags = sum(by_id[o].class_name in FLAGS for o in linked)
        ys = [centre(h)[1] for h, _ in heads]
        stem_up = centre(stem)[1] < np.mean(ys)
        nearest_tip = min(heads, key=lambda h: centre(h[0])[1]) if stem_up \
            else max(heads, key=lambda h: centre(h[0])[1])
        tip_x, tip_y = centre(nearest_tip[0])
        tip = stem.top if stem_up else stem.bottom
        positions = [(middle - y) / (space / 2) for y in ys]
        chords_by_staff[staff.id].append(dict(
            stem=stem_id, beams=beams, x=tip_x,
            features=[max(positions), min(positions), len(heads), float(stem_up),
                      float(any(h.class_name == "noteheadFull" for h, _ in heads)), len(beams), flags,
                      float(len(beams) > 0), 1.0, 0.0, (tip_x - staff.left) * MM_PER_PX],
            targets=[abs(tip_y - tip) * MM_PER_PX, (centre(stem)[0] - tip_x) * MM_PER_PX]))
    return [staff_of(page.writer, chords) for chords in chords_by_staff.values() if len(chords) > 0]


def beam_groups(chords: list[dict]) -> list[int]:
    """A group number per chord: chords sharing a beam, transitively"""
    group = list(range(len(chords)))

    def root(i):
        while group[i] != i:
            i = group[i]
        return i
    for i in range(len(chords)):
        for j in range(i):
            if chords[i]["beams"] & chords[j]["beams"]:
                group[root(i)] = root(j)
    return [root(i) for i in range(len(chords))]


def staff_of(writer: int, chords: list[dict]) -> Staff:
    chords.sort(key=lambda c: c["x"])
    groups = beam_groups(chords)
    members = defaultdict(list)
    for i, g in enumerate(groups):
        members[g].append(i)
    features = np.array([c["features"] for c in chords], dtype=np.float32)
    for g, indices in members.items():
        if len(indices) > 1 and features[indices[0], FEATURES.index("beamed")]:
            for k, i in enumerate(indices):
                features[i, FEATURES.index("group_size")] = len(indices)
                features[i, FEATURES.index("group_index")] = k / (len(indices) - 1)
    edges = []
    for i in range(len(chords) - 1):
        edges += [(i, i + 1, SEQUENCE), (i + 1, i, SEQUENCE)]
    for indices in members.values():
        edges += [(i, j, BEAM_GROUP) for i in indices for j in indices if i != j]
    return Staff(writer, features, np.array([c["targets"] for c in chords], dtype=np.float32), edges)


def all_staves() -> list[Staff]:
    return [s for page in training_pages(MUSCIMA) for s in page_staves(page)]


# ---- models ------------------------------------------------------------------------------------

def normalised(features: np.ndarray) -> np.ndarray:
    """Features scaled to about unit range; x as the gap to the previous chord"""
    f = features.copy()
    x = f[:, FEATURES.index("x_mm")]
    f[:, FEATURES.index("x_mm")] = np.diff(x, prepend=x[:1]) / 10.0
    f[:, :2] /= 8.0
    f[:, FEATURES.index("group_size")] /= 8.0
    return f


class StemGNN(nn.Module):
    """Relational GraphSAGE: per layer, a chord's state plus the mean of its neighbours' along
    each edge type, residual and layer-normalised; a head predicts (length, dx)."""

    def __init__(self, hidden: int = 128, layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(len(FEATURES), hidden)
        self.self_weights = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(layers))
        self.neighbour_weights = nn.ModuleList(
            nn.ModuleList(nn.Linear(hidden, hidden, bias=False) for _ in range(EDGE_TYPES)) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(layers))
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, len(TARGETS)))

    def forward(self, x: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.embed(x))
        for self_w, neighbour_w, norm in zip(self.self_weights, self.neighbour_weights, self.norms):
            update = self_w(h)
            for t in range(EDGE_TYPES):
                e = edges[edges[:, 2] == t]
                total = torch.zeros_like(h).index_add_(0, e[:, 1], h[e[:, 0]])
                count = torch.zeros(len(h), 1).index_add_(0, e[:, 1], torch.ones(len(e), 1)).clamp(min=1)
                update = update + neighbour_w[t](total / count)
            h = norm(h + torch.relu(update))
        return self.head(h)


def batched(staves: list[Staff]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xs, ys, es, offset = [], [], [], 0
    for s in staves:
        xs.append(normalised(s.features))
        ys.append(s.targets)
        es += [(a + offset, b + offset, t) for a, b, t in s.edges]
        offset += len(s.features)
    edges = torch.tensor(es, dtype=torch.long).reshape(-1, 3)
    return torch.tensor(np.concatenate(xs)), torch.tensor(np.concatenate(ys)), edges


def train_gnn(staves: list[Staff], epochs: int = 300, seed: int = 0) -> StemGNN:
    torch.manual_seed(seed)
    model = StemGNN()
    x, y, edges = batched(staves)
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, epochs)
    loss = nn.HuberLoss(delta=1.0)
    for _ in range(epochs):
        optimiser.zero_grad()
        loss(model(x, edges), y).backward()
        optimiser.step()
        schedule.step()
    return model


def context_key(features: np.ndarray) -> tuple:
    """Stem direction, beamed or not, and beam/flag count: the context means' cells"""
    f = dict(zip(FEATURES, features))
    return f["stem_up"], f["beamed"], min(int(f["beams"] + f["flags"]), 3)


def context_means(staves: list[Staff]):
    cells = defaultdict(list)
    for s in staves:
        for f, t in zip(s.features, s.targets):
            cells[context_key(f)].append(t)
    overall = np.concatenate([s.targets for s in staves]).mean(0)
    means = {k: np.mean(v, 0) for k, v in cells.items()}
    return lambda f: means.get(context_key(f), overall)


SMASHCIMA_STEM_MM = 5.0
"Smashcima's constant stem length; it attaches the stem at the notehead's edge"


def evaluate(folds: int = 5) -> None:
    staves = all_staves()
    writers = sorted({s.writer for s in staves})
    random.Random(0).shuffle(writers)
    errors = defaultdict(list)
    for k in range(folds):
        held = set(writers[k::folds])
        train = [s for s in staves if s.writer not in held]
        test = [s for s in staves if s.writer in held]
        truth = np.concatenate([s.targets for s in test])
        features = np.concatenate([s.features for s in test])
        # Smashcima's dx: half a notehead, about 0.8 mm, to the stem's side
        smashcima = np.stack([np.full(len(truth), SMASHCIMA_STEM_MM),
                              np.where(features[:, FEATURES.index("stem_up")] > 0, 0.8, -0.8)], 1)
        global_mean = np.tile(np.concatenate([s.targets for s in train]).mean(0), (len(truth), 1))
        context = context_means(train)
        contextual = np.array([context(f) for f in features])
        model = train_gnn(train)
        with torch.no_grad():
            x, _, edges = batched(test)
            gnn = model(x, edges).numpy()
        for name, prediction in [("smashcima", smashcima), ("global mean", global_mean),
                                 ("context means", contextual), ("gnn", gnn)]:
            errors[name].append(prediction - truth)
        print(f"fold {k}: {len(train)} train staves, {len(test)} test staves ({len(truth)} stems)", flush=True)
    beamed = np.concatenate([np.concatenate([s.features for s in staves if s.writer in set(writers[k::folds])])
                             for k in range(folds)])[:, FEATURES.index("beamed")] > 0
    print(f"{'model':>14s} {'length RMSE':>12s} {'dx RMSE':>8s} {'tip RMSE':>9s} {'single':>7s} {'beamed':>7s}  (mm)")
    for name, parts in errors.items():
        e = np.concatenate(parts)
        tip = np.sqrt((e ** 2).sum(1))
        rmse = lambda v: float(np.sqrt((v ** 2).mean()))
        print(f"{name:>14s} {rmse(e[:, 0]):12.2f} {rmse(e[:, 1]):8.2f} {rmse(tip):9.2f} "
              f"{rmse(tip[~beamed]):7.2f} {rmse(tip[beamed]):7.2f}")


def residual_spread(model: StemGNN, staves: list[Staff]) -> np.ndarray:
    with torch.no_grad():
        x, y, edges = batched(staves)
        return (model(x, edges) - y).std(0).numpy()


def export(out: Path, held_out_spread: np.ndarray | None = None) -> None:
    """Weights as plain arrays for the fork's numpy inference, with the residual spread to
    sample around the prediction (from writer-held-out folds when given)"""
    staves = all_staves()
    model = train_gnn(staves)
    arrays = {name: p.detach().numpy() for name, p in model.state_dict().items()}
    spread = held_out_spread if held_out_spread is not None else residual_spread(model, staves)
    np.savez(out, features=np.array(FEATURES), targets=np.array(TARGETS), residual_std=spread, **arrays)
    print(f"wrote {out} from {len(staves)} staves; residual std {spread}")


def held_out_spread(folds: int = 5) -> np.ndarray:
    """Residual spread of models on writers they did not see: how much to vary a prediction"""
    staves = all_staves()
    writers = sorted({s.writer for s in staves})
    random.Random(0).shuffle(writers)
    residuals = []
    for k in range(folds):
        held = set(writers[k::folds])
        model = train_gnn([s for s in staves if s.writer not in held])
        with torch.no_grad():
            x, y, edges = batched([s for s in staves if s.writer in held])
            residuals.append((model(x, edges) - y).numpy())
    return np.concatenate(residuals).std(0)


def main() -> None:
    torch.set_num_threads(4)
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["evaluate", "export"])
    ap.add_argument("out", nargs="?", type=Path)
    args = ap.parse_args()
    if args.command == "evaluate":
        evaluate()
    else:
        export(args.out, held_out_spread())


if __name__ == "__main__":
    main()
