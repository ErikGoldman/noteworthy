"""Trains the system-crop image-to-LMX model (seq/).

usage: train_seq.py --data ROOT:SPLIT[:WEIGHT] [...] --dev ROOT:SPLIT --out DIR
                    [--steps N] [--batch B] [--lr LR] [--dev-count N] [--eval-every N]

Each --data entry is a directory of <sample>.png + <sample>.lmx with a
samples.<SPLIT>.txt list (OLiMPiC, GrandStaff-LMX); WEIGHT repeats it in the mix. --dev is
decoded greedily every --eval-every steps (its first --dev-count samples) and scored by
symbol error rate, SER: token edit distance over gold length, as in the OLiMPiC paper.
--dev-samples restricts it to listed samples: for a model trained on PDMX, the OLiMPiC
songs without a copy there (eval/olimpic_benchmark.py list).
Resumes from OUT/last.pt.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import Levenshtein
import torch
from torch.nn import functional as F

from seq.data import SystemDataset, collate, samples, split_names
from seq.model import Seq2Seq
from seq.vocab import PAD, Vocab

ROOT = Path(__file__).resolve().parent.parent
VOCABULARY = ROOT / "train" / "seq" / "vocabulary.txt"
GPU_MEMORY_FRACTION = 0.85


def dataset_items(spec: str):
    root, split, *weight = spec.split(":")
    return samples(Path(root), split_names(Path(root), split)) * int(weight[0] if weight else 1)


def plain(options: dict) -> dict:
    """Paths as strings: torch.load's default weights_only mode rejects other classes."""
    return {k: str(v) if isinstance(v, Path) else v for k, v in options.items()}


def ser(model: Seq2Seq, loader, vocab: Vocab, device) -> float:
    model.eval()
    errors = total = 0
    for images, widths, tokens in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            predicted = model.generate(images.to(device), widths)
        for p, g in zip(predicted, tokens.tolist()):
            gold = vocab.decode(g[1:]).split()
            errors += Levenshtein.distance(vocab.decode(p).split(), gold)
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
    args = ap.parse_args()

    device = torch.device("cuda")
    # Under WSL the driver silently backs GPU allocations beyond the card with system
    # memory, and training then runs several times slower. Batches vary in width and
    # length, so PyTorch's cache keeps growing until it gets there; a cap makes it free
    # cached blocks instead.
    torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION)
    vocab = Vocab.load(VOCABULARY)
    train_items = [s for spec in args.data for s in dataset_items(spec)]
    dev_root, dev_split = args.dev.split(":")
    dev_names = args.dev_samples.read_text().split() if args.dev_samples else split_names(Path(dev_root), dev_split)
    dev_items = samples(Path(dev_root), dev_names[: args.dev_count])
    train = torch.utils.data.DataLoader(SystemDataset(train_items, vocab, True, args.max_tokens), batch_size=args.batch,
                                        shuffle=True, num_workers=args.workers, collate_fn=collate, drop_last=True,
                                        persistent_workers=True)
    dev = torch.utils.data.DataLoader(SystemDataset(dev_items, vocab, False, args.max_tokens), batch_size=32,
                                      num_workers=4, collate_fn=collate)
    model = Seq2Seq(len(vocab), max_tokens=args.max_tokens).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    schedule = lambda s: min(1.0, s / args.warmup) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, schedule)
    args.out.mkdir(parents=True, exist_ok=True)
    step, best = 0, float("inf")
    if (args.out / "last.pt").exists():
        state = torch.load(args.out / "last.pt", map_location=device)
        model.load_state_dict(state["model"]), opt.load_state_dict(state["opt"]), sched.load_state_dict(state["sched"])
        step, best = state["step"], state["best"]
    if step >= args.steps:
        print(f"already trained to step {step}; nothing to do")
        return
    print(f"{len(train_items)} training samples, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    log = (args.out / "log.jsonl").open("a")
    started, seen = time.time(), 0
    while step < args.steps:
        for images, widths, tokens in train:
            images, tokens = images.to(device, non_blocking=True), tokens.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(images, widths, tokens)
            loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1),
                                   ignore_index=PAD, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            seen += images.shape[0]
            if step % 100 == 0:
                print(f"step {step} loss {loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{seen / (time.time() - started):.0f} samples/s", flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                dev_ser = ser(model, dev, vocab, device)
                print(f"step {step} dev SER {dev_ser:.4f}", flush=True)
                log.write(json.dumps({"step": step, "loss": loss.item(), "dev_ser": dev_ser}) + "\n")
                log.flush()
                state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                         "step": step, "best": min(best, dev_ser), "args": plain(vars(args))}
                torch.save(state, args.out / "last.pt")
                if dev_ser < best:
                    best = dev_ser
                    torch.save(state, args.out / "best.pt")
            if step >= args.steps:
                break


if __name__ == "__main__":
    main()
