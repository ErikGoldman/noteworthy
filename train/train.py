"""Train the symbol segmentation U-Net.

usage: train.py --data DIR[:WEIGHT] [DIR[:WEIGHT] ...] --out DIR [--steps N] [--batch B]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from omr.channels import CHANNELS
from omr.heads import EXCLUSIVE_FAMILIES, loss, probabilities
from omr.dataset import PageCrops, collate_crops, init_worker, list_pages, split_by_source, to_model_input
from omr.model import UNet

VAL_CROPS = 512
CROPS_PER_PAGE = 4


def weighted_pages(spec: str) -> list[Path]:
    """"DIR" or "DIR:N": the pages of DIR, repeated N times in the sampling pool."""
    directory, _, weight = spec.partition(":")
    return list_pages(directory) * int(weight or 1)


@torch.no_grad()
def evaluate(model, loader, device) -> dict[str, float]:
    model.eval()
    tp = torch.zeros(len(CHANNELS), device=device)
    fp = torch.zeros_like(tp)
    fn = torch.zeros_like(tp)
    for x, y in loader:
        x, y = to_model_input(x, y, device)
        y = y.bool()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred = probabilities(model(x).float()) > 0.5
        tp += (pred & y).sum((0, 2, 3))
        fp += (pred & ~y).sum((0, 2, 3))
        fn += (~pred & y).sum((0, 2, 3))
    model.train()
    # Channels absent from the val crops and never predicted have no meaningful F1.
    return {name: round((2 * t / (2 * t + p + n)).item(), 4)
            for name, t, p, n in zip(CHANNELS, tp, fp, fn) if (t + p + n) > 0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=60_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--crop", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--resume", default=None, help="checkpoint to continue (weights, optimizer, schedule, step)")
    ap.add_argument("--init", default=None, help="checkpoint whose weights start a new schedule (fine-tuning)")
    ap.add_argument("--balanced-crops", type=float, default=0.0,
                    help="fraction of training crops centred on a uniformly chosen class present on the page")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pages = [p for spec in args.data for p in weighted_pages(spec)]
    train_pages, val_pages = split_by_source(pages)
    print(f"{len(train_pages)} train pages, {len(val_pages)} val pages")

    if args.batch % CROPS_PER_PAGE:
        raise SystemExit(f"--batch must be a multiple of {CROPS_PER_PAGE}")
    pages_per_batch = args.batch // CROPS_PER_PAGE
    train_ds = PageCrops(train_pages, args.crop, length=args.steps * pages_per_batch, crops_per_page=CROPS_PER_PAGE,
                         balanced_fraction=args.balanced_crops)
    val_ds = PageCrops(val_pages, args.crop, length=VAL_CROPS // CROPS_PER_PAGE, crops_per_page=CROPS_PER_PAGE, seed=12345)
    train_loader = DataLoader(train_ds, pages_per_batch, shuffle=False, num_workers=args.workers, collate_fn=collate_crops, worker_init_fn=init_worker,
                              pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(val_ds, pages_per_batch, num_workers=8, collate_fn=collate_crops, worker_init_fn=init_worker, pin_memory=True)

    device = torch.device("cuda")
    model = UNet(len(CHANNELS)).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.03)
    start_step = 0
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=device)["model"])
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["opt"])
        sched.load_state_dict(ckpt["sched"])
        start_step = ckpt["step"]
        if start_step >= args.steps:
            print(f"{args.resume} already trained to step {start_step}; nothing to do")
            return
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    log = (out / "log.jsonl").open("a")
    best_f1 = -1.0
    t0 = time.time()
    for step, (x, y) in enumerate(train_loader, start=start_step + 1):
        if step > args.steps:
            break
        x, y = to_model_input(x, y, device)
        x = x.to(memory_format=torch.channels_last)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        batch_loss = loss(logits.float(), y)
        opt.zero_grad(set_to_none=True)
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            rate = 100 * args.batch / (time.time() - t0)
            t0 = time.time()
            print(f"step {step} loss {batch_loss.item():.4f} lr {sched.get_last_lr()[0]:.2e} {rate:.0f} crops/s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            f1 = evaluate(model, val_loader, device)
            mean_f1 = sum(f1.values()) / len(f1)
            log.write(json.dumps({"step": step, "loss": batch_loss.item(), "mean_f1": mean_f1, "f1": f1}) + "\n")
            log.flush()
            print(f"[eval] step {step} mean F1 {mean_f1:.4f} " + " ".join(f"{k}={v:.2f}" for k, v in f1.items()), flush=True)
            state = {"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "step": step, "channels": CHANNELS, "families": EXCLUSIVE_FAMILIES}
            torch.save(state, out / "last.pt")
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                torch.save(state, out / "best.pt")


if __name__ == "__main__":
    main()
