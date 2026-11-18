"""Trains the system-crop image-to-LMX model (seq/).

usage: train_seq.py --data ROOT:SPLIT[:WEIGHT] [...] --dev ROOT:SPLIT --out DIR
                    [--steps N] [--batch B] [--lr LR] [--dev-count N] [--eval-every N]
                    [--init CHECKPOINT]

Each --data entry is a directory of <sample>.png + <sample>.lmx with a
samples.<SPLIT>.txt list (OLiMPiC, GrandStaff-LMX); WEIGHT repeats it in the mix. --dev is
decoded greedily every --eval-every steps (its first --dev-count samples) and scored by
symbol error rate, SER: token edit distance over gold length, as in the OLiMPiC paper.
--dev-samples restricts it to listed samples: for a model trained on PDMX, the OLiMPiC
songs without a copy there (eval/olimpic_benchmark.py list).
Resumes from OUT/last.pt; --init starts a new run from another run's weights.
--staves POSITIONS_DIR trains StaffSeq2Seq instead, on each system's staff crops cut at the
positions find_staff_positions.py wrote there (one file per --data and --dev root); samples
without positions are left out. Its staves are seq.staff_data.STAFF_HEIGHT tall (--height is
Seq2Seq's). --staff-samples-only POSITIONS_DIR trains Seq2Seq on the samples such a run keeps, to
compare the two on the same data.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import time
from pathlib import Path

import Levenshtein
import torch
from torch.nn import functional as F

from seq.data import SystemDataset, collate, samples, split_names
from seq.label_form import LabelForm, bars_of, voices_per_staff
from seq.meter import METER, PREFIX_LENGTH, without_meter
from seq.model import Seq2Seq, StaffSeq2Seq
from seq.ottava import sounding
from seq.staff_data import (MAX_SYSTEM_HEIGHT, InterlineSystemDataset, StaffDataset, collate_staves,
                             collate_systems, load_positions)
from seq.barlines import without_barlines
from seq.time_symbols import without_time_symbols
from seq.tuplet_numbers import without_hidden_numbers
from seq.vocab import PAD, Vocab

ROOT = Path(__file__).resolve().parent.parent
VOCABULARY = ROOT / "train" / "seq" / "vocabulary.txt"
GPU_MEMORY_FRACTION = 0.85


# Note errors concentrate in rhythm (durations, beam counts, tuplets) on OLiMPiC's scans and Polish
# alike, and few labels have the hard cases: 6% of PDMX crops, 24% of GrandStaff's.
RHYTHM_HEAVY_REPEAT = 2
RHYTHM_HEAVY = re.compile(r"(?:^| )(?:\d+in\d+|32nd|64th)(?: |$)")


def rhythm_heavy(sample) -> bool:
    return RHYTHM_HEAVY.search(sample.lmx.read_text()) is not None


def has_two_voices(sample) -> bool:
    """A staff holds two or more voices in some bar."""
    return any(count >= 2 for bar in bars_of(sample.lmx.read_text().split()) for count in voices_per_staff(bar).values())


def dataset_items(spec: str, two_voice_repeat: int = 1):
    root, split, *weight = spec.split(":")
    items = samples(Path(root), split_names(Path(root), split))
    items += [s for s in items if rhythm_heavy(s)] * (RHYTHM_HEAVY_REPEAT - 1)
    if two_voice_repeat > 1:
        items += [s for s in items if has_two_voices(s)] * (two_voice_repeat - 1)
    return items * int(weight[0] if weight else 1)


def training_items(specs: list[str], two_voice_repeat: int, positions: Path | None) -> list:
    """Each --data set's samples as the mix weights them; with positions, each paired with its staff
    positions (samples without any left out). Built in one place, so every path gets the same mix:
    the staff paths once rebuilt it without the two-voice repeat (l_system_two_voice_100k, 2026-11-18)."""
    weighted = [(Path(spec.split(":")[0]), dataset_items(spec, two_voice_repeat)) for spec in specs]
    if positions is None:
        return [s for _, items in weighted for s in items]
    return [pair for root, items in weighted for pair in with_staves(items, root, positions)]


def with_staves(items: list, root: Path, positions: Path) -> list:
    """(sample, its staff positions) for each sample of root that has them."""
    found = load_positions(positions, root)
    paired = [(s, found[str(s.image.relative_to(root).with_suffix(""))]) for s in items]
    kept = [(s, staves) for s, staves in paired if staves is not None]
    print(f"{root.name}: {len(kept)} of {len(items)} samples with staff positions")
    return kept


def targets_of(tokens: torch.Tensor, vocab: Vocab) -> torch.Tensor:
    """What each position predicts: the next token, but nothing for a meter prefix, which the
    model is given and never asked to write"""
    targets = tokens[:, 1:].clone()
    targets[tokens[:, 1] == vocab.index[METER], :PREFIX_LENGTH] = PAD
    return targets


def with_own_readings(model, memory, padding, tokens: torch.Tensor, vocab: Vocab, share: float) -> torch.Tensor:
    """The decoder's input (tokens[:, :-1]) with `share` of its label tokens replaced by tokens
    sampled from the model's own reading of the batch, labels fed: training on its own mistakes, so
    one wrong token does not lead it astray (scheduled sampling, two passes). <bos> and a meter
    prefix, which the model is given, are kept, and so is padding."""
    inputs = tokens[:, :-1]
    # Uncached: autocast keeps each weight's cast for the rest of the step, and before torch 2.11 one
    # first made under no_grad carries no gradient, so the second pass would leave the decoder's
    # weights untrained (the first run of this on torch 2.8, 2026-11-15).
    device = inputs.device.type
    with torch.no_grad(), torch.autocast(device, dtype=torch.get_autocast_dtype(device),
                                         enabled=torch.is_autocast_enabled(device), cache_enabled=False):
        logits = model.decode(inputs, memory, padding)
        sampled = torch.distributions.Categorical(logits=logits.float()).sample()
    swap = torch.rand(inputs.shape, device=inputs.device) < share
    swap[tokens[:, 1] == vocab.index[METER], 1:PREFIX_LENGTH + 1] = False
    swap &= inputs != PAD
    # Input position t holds label token t, which the model predicts at position t - 1.
    return torch.cat([inputs[:, :1], torch.where(swap[:, 1:], sampled[:, :-1], inputs[:, 1:])], 1)


def own_readings_share(step: int, steps: int, share: float) -> float:
    return share * min(1.0, step / (steps / 2))


def plain(options: dict) -> dict:
    """Paths as strings: torch.load's default weights_only mode rejects other classes."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in options.items()}


def weights_for_model(weights: dict, model: Seq2Seq) -> dict:
    """Weights of another run, its row embedding stretched to this model's rows, and tokens
    added to the vocabulary since then keeping this model's fresh embedding."""
    rows = model.row_embedding.shape[0]
    stretched = F.interpolate(weights["row_embedding"].T[None], size=rows, mode="linear", align_corners=True)[0].T
    embedding = model.token_embedding.weight.detach().clone()
    embedding[:len(weights["token_embedding.weight"])] = weights["token_embedding.weight"]
    return {**weights, "row_embedding": stretched, "token_embedding.weight": embedding, "out.weight": embedding}


# The weights' moving average, about the last 2,000 steps: a run's checkpoints swing in dev SER
# by as much as two runs differ (seq512: 0.082 to 0.077 within 5,000 steps), and the average
# is steadier and usually better than any one of them.
EMA_DECAY = 0.9995
SNAPSHOT_EVERY = 10000  # averaged weights kept along the way, to score a run's course


@torch.no_grad()
def update_average(average: torch.nn.Module, model: torch.nn.Module) -> None:
    for kept, current in zip(average.state_dict().values(), model.state_dict().values()):
        if kept.dtype.is_floating_point:
            kept.mul_(EMA_DECAY).add_(current, alpha=1 - EMA_DECAY)
        else:
            kept.copy_(current)


def save_whole(state: dict, path: Path) -> None:
    """Saved under another name and renamed into place: anything copying the run directory while
    training (rsync, a push to R2) finds the old checkpoint or the new one, never half of one."""
    partial = path.with_name(path.name + ".partial")
    torch.save(state, partial)
    partial.replace(path)


def comparable(lmx: str) -> str:
    """LMX in OLiMPiC's form, as its dev labels are: ottavas applied, none of our tokens for
    barlines, time symbols or hidden tuplet numbers."""
    return without_hidden_numbers(without_time_symbols(without_barlines(sounding(lmx))[0])[0])[0]


def meter_starts(tokens: torch.Tensor, vocab: Vocab) -> list[list[int]]:
    """Each label's meter prefix (seq.meter), given to the model as it was in training"""
    meter = vocab.index[METER]
    return [t[1:1 + PREFIX_LENGTH] if t[1] == meter else [] for t in tokens.tolist()]


def as_lmx(model, vocab: Vocab, ids: list[int]) -> str:
    """A decoded reading or label as plain LMX: the meter prefix and its label form's additions out."""
    return model.label_form.reading(without_meter(vocab.decode(ids)))


def ser(model: Seq2Seq, loader, vocab: Vocab, device) -> float:
    model.eval()
    errors = total = 0
    for images, widths, tokens in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.generate(images.to(device), widths, starts=meter_starts(tokens, vocab))
        for p, g in zip(predicted, tokens.tolist()):
            gold = comparable(as_lmx(model, vocab, g[1:])).split()
            errors += Levenshtein.distance(comparable(as_lmx(model, vocab, p)).split(), gold)
            total += len(gold)
    model.train()
    return errors / max(1, total)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--dev-samples", type=Path, help="score these samples of --dev instead of its split")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--dev-count", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--init", type=Path, help="weights to start from, for fine-tuning")
    ap.add_argument("--height", type=int, default=256, help="crop height at the model's input")
    ap.add_argument("--row-stride", type=int, default=32, choices=[8, 16, 32])
    ap.add_argument("--seed", type=int, default=0, help="decides the initial weights, batches and augmentation")
    ap.add_argument("--meter-context", action="store_true", help="labels start with the meter in force (seq.meter)")
    ap.add_argument("--voice-counts", action="store_true",
                    help="labels say how many voices each staff holds after each measure (seq.label_form)")
    ap.add_argument("--canonical-gaps", action="store_true",
                    help="labels write each gap and each return to the bar's start one way (seq.label_form)")
    ap.add_argument("--two-voice-repeat", type=int, default=1,
                    help="systems with a two-voice staff-bar repeated in the mix: 4 brings two-voice "
                         "staff-bars from 16% of the mix to about Polish's 28% (TODO 3b)")
    ap.add_argument("--own-readings", type=float, default=0.0,
                    help="share of the decoder's input tokens replaced by ones sampled from its own reading "
                         "of the batch (scheduled sampling), rising from 0 over the first half of training")
    ap.add_argument("--staves", type=Path, help="train StaffSeq2Seq on staff crops, positions from this directory")
    ap.add_argument("--staff-samples-only", type=Path, help="train Seq2Seq on the samples with staff positions here")
    ap.add_argument("--relative-rows", action="store_true",
                    help="Seq2Seq: attention biased by vertical distance in place of each row's own embedding")
    ap.add_argument("--row-windows", type=int, default=0,
                    help="Seq2Seq, with --relative-rows: encoder attention in bands of this many feature rows")
    ap.add_argument("--interline-scale", action="store_true",
                    help="Seq2Seq: each system at staff scale, its height following its staves "
                         "(needs --staff-samples-only for the positions; --height is then the tallest crop)")
    ap.add_argument("--staff-anchored", action="store_true",
                    help="Seq2Seq, with --interline-scale: rows embedded by their nearest staff and offset from it")
    ap.add_argument("--staff-bands", action="store_true",
                    help="Seq2Seq, with --staff-anchored: attention within each staff's rows, then down columns")
    ap.add_argument("--staff-windows", action="store_true",
                    help="Seq2Seq, with --staff-anchored: attention within each staff's overlapping window "
                         "(a staff crop's rows), then down columns")
    ap.add_argument("--staff-jitter", type=float, default=0.0,
                    help="StaffSeq2Seq: in training, each staff crop centred up to this many interlines off its staff")
    ap.add_argument("--staff-cut-between", action="store_true",
                    help="StaffSeq2Seq: each staff crop white past the midpoints to its neighbours")
    ap.add_argument("--blocks", type=int, default=1, help="StaffSeq2Seq: encoder blocks (within, then across staves)")
    ap.add_argument("--across-columns", type=int, default=1,
                    help="StaffSeq2Seq: columns the attention across staves takes together; 0: none")
    ap.add_argument("--staff-outward", type=float, default=0.0,
                    help="StaffSeq2Seq: interlines the top and bottom staves' crops slide away from the system")
    args = ap.parse_args()
    if args.staves and args.init:
        ap.error("--staves trains StaffSeq2Seq from scratch")
    if not args.staves and (args.blocks != 1 or args.across_columns != 1 or args.staff_outward or args.staff_jitter
                            or args.staff_cut_between):
        ap.error("--blocks, --across-columns, --staff-outward, --staff-jitter and --staff-cut-between "
                 "are StaffSeq2Seq's (--staves)")
    if args.staff_anchored and not args.interline_scale:
        ap.error("--staff-anchored needs --interline-scale")
    if args.staff_anchored and args.relative_rows:
        ap.error("--staff-anchored replaces --relative-rows")
    if (args.staff_bands or args.staff_windows) and not args.staff_anchored:
        ap.error("--staff-bands and --staff-windows need --staff-anchored")
    if args.staff_bands and args.staff_windows:
        ap.error("--staff-windows replaces --staff-bands")
    if args.row_windows and not args.relative_rows:
        ap.error("--row-windows needs --relative-rows")
    if args.interline_scale:
        args.height = MAX_SYSTEM_HEIGHT  # the row embedding's reach (and what the checkpoint records)
    if args.interline_scale and not args.staff_samples_only:
        ap.error("--interline-scale needs staff positions: --staff-samples-only DIR")
    if args.staves and args.relative_rows:
        ap.error("--relative-rows is Seq2Seq's: a staff crop's rows already stand at fixed heights")
    if args.staves and args.staff_samples_only:
        ap.error("--staff-samples-only is for Seq2Seq; --staves already keeps only those samples")
    torch.manual_seed(args.seed)

    device = torch.device("cuda")
    # Under WSL the driver silently backs GPU allocations beyond the card with system
    # memory, and training then runs several times slower. Batches vary in width and
    # length, so PyTorch's cache keeps growing until it gets there; a cap makes it free
    # cached blocks instead.
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    vocab = Vocab.load(VOCABULARY)
    positions = args.staves or args.staff_samples_only
    train_items = training_items(args.data, args.two_voice_repeat, positions)
    dev_root, dev_split = args.dev.split(":")
    dev_names = args.dev_samples.read_text().split() if args.dev_samples else split_names(Path(dev_root), dev_split)
    dev_items = samples(Path(dev_root), dev_names[: args.dev_count])
    if positions:
        dev_items = with_staves(dev_items, Path(dev_root), positions)
    if args.staff_samples_only and not args.interline_scale:
        train_items, dev_items = [s for s, _ in train_items], [s for s, _ in dev_items]
    form = LabelForm(voice_counts=args.voice_counts, canonical_gaps=args.canonical_gaps)
    if args.staves:
        dataset = lambda items, train: StaffDataset(items, vocab, train, args.max_tokens, outward=args.staff_outward,
                                                    meter_context=args.meter_context, jitter=args.staff_jitter,
                                                    cut_between=args.staff_cut_between, form=form)
        batched = collate_staves
        model = StaffSeq2Seq(len(vocab), max_tokens=args.max_tokens, row_stride=args.row_stride, blocks=args.blocks,
                             across_columns=args.across_columns, staff_outward=args.staff_outward,
                             meter_context=args.meter_context, staff_cut_between=args.staff_cut_between, label_form=form)
    elif args.interline_scale:
        dataset = lambda items, train: InterlineSystemDataset(items, vocab, train, args.max_tokens,
                                                              meter_context=args.meter_context, form=form)
        batched = collate_systems
        model = Seq2Seq(len(vocab), max_tokens=args.max_tokens, height=args.height, row_stride=args.row_stride,
                        meter_context=args.meter_context, relative_rows=args.relative_rows,
                        row_windows=args.row_windows, interline_scaled=True, staff_anchored=args.staff_anchored,
                        staff_bands=args.staff_bands, staff_windows=args.staff_windows, label_form=form)
    else:
        dataset = lambda items, train: SystemDataset(items, vocab, train, args.max_tokens, args.height, args.meter_context,
                                                     form)
        batched = collate
        model = Seq2Seq(len(vocab), max_tokens=args.max_tokens, height=args.height, row_stride=args.row_stride,
                        meter_context=args.meter_context, relative_rows=args.relative_rows, row_windows=args.row_windows,
                        label_form=form)
    train = torch.utils.data.DataLoader(dataset(train_items, True), batch_size=args.batch,
                                        shuffle=True, num_workers=args.workers, collate_fn=batched, drop_last=True,
                                        persistent_workers=True)
    dev = torch.utils.data.DataLoader(dataset(dev_items, False), batch_size=32, num_workers=4, collate_fn=batched)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = lambda s: min(1.0, s / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    args.out.mkdir(parents=True, exist_ok=True)
    step, best = 0, float("inf")
    if args.init and not (args.out / "last.pt").exists():
        model.load_state_dict(weights_for_model(torch.load(args.init, map_location=device)["model"], model))
    average = copy.deepcopy(model)
    if (args.out / "last.pt").exists():
        state = torch.load(args.out / "last.pt", map_location=device)
        model.load_state_dict(state["model"]), opt.load_state_dict(state["opt"]), sched.load_state_dict(state["sched"])
        average.load_state_dict(state.get("average", state["model"]))
        step, best = state["step"], state["best"]
    if step >= args.steps:
        print(f"already trained to step {step}; nothing to do")
        return
    # A resumed run cannot pick up the loader's order where it stopped; it starts a new order,
    # the same every time it resumes at this step.
    torch.manual_seed(args.seed + step)
    print(f"{len(train_items)} training samples, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    log = (args.out / "log.jsonl").open("a")
    started, seen = time.time(), 0
    while step < args.steps:
        for images, widths, tokens in train:
            images, tokens = images.to(device, non_blocking=True), tokens.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                memory, padding = model.encode(images, widths)
                inputs = (with_own_readings(model, memory, padding, tokens, vocab,
                                            own_readings_share(step, args.steps, args.own_readings))
                          if args.own_readings else tokens[:, :-1])
                logits = model.decode(inputs, memory, padding)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets_of(tokens, vocab).reshape(-1),
                                   ignore_index=PAD, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            update_average(average, model)
            step += 1
            seen += tokens.shape[0]
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{seen / (time.time() - started):.0f} samples/s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                dev_ser = ser(average, dev, vocab, device)
                print(f"step {step} dev SER {dev_ser:.4f} (averaged weights)", flush=True)
                log.write(json.dumps({"step": step, "loss": loss.item(), "dev_ser": dev_ser}) + "\n")
                log.flush()
                state = {"model": model.state_dict(), "average": average.state_dict(), "opt": opt.state_dict(),
                         "sched": sched.state_dict(), "step": step, "best": min(best, dev_ser), "args": plain(vars(args))}
                save_whole(state, args.out / "last.pt")
                # best.pt and the snapshots hold the averaged weights as "model": what is scored.
                scored = {"model": average.state_dict(), "step": step, "dev_ser": dev_ser, "args": plain(vars(args))}
                if dev_ser < best:
                    best = dev_ser
                    save_whole(scored, args.out / "best.pt")
                if step % SNAPSHOT_EVERY == 0 or step == args.steps:
                    save_whole(scored, args.out / f"average_{step:06d}.pt")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
