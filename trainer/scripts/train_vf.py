#!/usr/bin/env python3
"""Supervised training on viriformat (.vf) data: distillation from a teacher
network, or learning from the data's own labels.

Positions come from any viriformat files (PeaZero's runs/<name>/<name>.vf,
or another engine's NNUE data such as Pea's), streamed at random through a
shuffle buffer. Every --holdout-every-th game is held out for evaluation.

Two modes:

  Distillation (--teacher <checkpoint.pt>): the teacher network labels every
  position on the fly with its policy over the legal moves and its W/D/L
  value; the student learns to copy them (optionally blended with the game
  result, --result-weight). No search, no new games. Typical use: make a
  small, fast network (64x6) from a big, slow one (96x8):

    python scripts\\train_vf.py --data ..\\runs\\fifth\\fifth.vf ..\\runs\\fourth\\fourth.vf ^
        --teacher ..\\runs\\fifth\\checkpoint.pt --init ..\\runs\\fourth\\teacher_gen3000.pt ^
        --out ..\\runs\\distill64 --positions 200000000

  Data labels (no --teacher): value = mix of the game result and the stored
  score (--score-weight; scores are mapped to W/D/L with a table fitted on
  the data itself), policy = the move played, slightly smoothed
  (--policy-smoothing) and down-weighted (--policy-weight), since one move
  is a much weaker target than a search distribution:

    python scripts\\train_vf.py --data D:\\pea_data\\*.vf --net 96x8 --out ..\\runs\\pea96 --positions 500000000

The student starts from --init (a checkpoint: its size and weights) or a new
--net CxB. The best held-out point is saved to --out as checkpoint.pt
(train.py format: copy it into a new run folder and use --resume to continue
with self-play) and net.pzn (engine / SPRT); log.jsonl has every evaluation.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import peazero_rs as pz  # noqa: E402
from peazero.model import MuZeroNet, NetConfig, export_pzn  # noqa: E402


def load_net(path, dev):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = MuZeroNet(NetConfig(**ck["config"]))
    net.load_state_dict(ck["model"])
    net.set_qat(False)
    return net.to(dev), ck


def to_dev(b, dev):
    return {k: torch.from_numpy(v).to(dev, non_blocking=True) for k, v in b.items()}


def fit_wdl_table(stream, positions, bin_cp=25, max_cp=2000):
    """P(win/draw/loss | score) for the side to move, fitted on the data:
    per score bin the frequency of each game result, lightly smoothed."""
    nb = 2 * max_cp // bin_cp + 1
    counts = np.ones((nb, 3))  # Laplace prior
    done = 0
    while done < positions:
        b = stream.batch(8192)
        idx = np.clip(np.round(b["score"] / bin_cp).astype(np.int64) + max_cp // bin_cp, 0, nb - 1)
        np.add.at(counts, (idx, b["result"]), 1)
        done += len(idx)
    kernel = np.array([0.25, 0.5, 0.25])
    sm = np.stack([np.convolve(np.pad(counts[:, c], 1, mode="edge"), kernel, mode="valid") for c in range(3)], 1)
    table = sm / sm.sum(1, keepdims=True)
    return table, bin_cp, max_cp


class Targets:
    """Policy (over the legal moves) and W/D/L value targets for a batch."""

    def __init__(self, args, dev, teacher=None, wdl=None):
        self.args, self.dev, self.teacher = args, dev, teacher
        if wdl is not None:
            table, self.bin_cp, self.max_cp = wdl
            self.table = torch.tensor(table, dtype=torch.float32, device=dev)

    def score_wdl(self, score):
        idx = torch.clamp(torch.round(score / self.bin_cp).long() + self.max_cp // self.bin_cp, 0,
                          self.table.shape[0] - 1)
        return self.table[idx]

    @torch.no_grad()
    def __call__(self, t, amp):
        a, mask = self.args, t["legal_mask"]
        result = F.one_hot(t["result"], 3).float()
        if self.teacher is not None:
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                _, p, v = self.teacher.initial_inference(t["obs"])
            logits = p.float().gather(1, t["legal_index"]) / a.teacher_temp
            logits = logits.masked_fill(mask == 0, -1e9)
            policy = torch.softmax(logits, -1)
            value = torch.softmax(v.float(), -1)
            if a.result_weight > 0:
                value = (1 - a.result_weight) * value + a.result_weight * result
        else:
            played = (t["legal_index"] == t["played"][:, None]) & (mask > 0)
            onehot = played.float()
            onehot = onehot / onehot.sum(-1, keepdim=True).clamp(min=1)
            uniform = mask / mask.sum(-1, keepdim=True).clamp(min=1)
            policy = (1 - a.policy_smoothing) * onehot + a.policy_smoothing * uniform
            value = a.score_weight * self.score_wdl(t["score"]) + (1 - a.score_weight) * result
        return policy, value


def losses(net, t, policy, value, amp):
    """(policy CE over the legal moves, value CE, top-1 agreement, value CE vs result)."""
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        _, p, v = net.initial_inference(t["obs"])
    logits = p.float().gather(1, t["legal_index"]).masked_fill(t["legal_mask"] == 0, -1e9)
    logp = torch.log_softmax(logits, -1)
    pl = -(policy * logp).sum(-1).mean()
    vlogp = torch.log_softmax(v.float(), -1)
    vl = -(value * vlogp).sum(-1).mean()
    agree = (logits.argmax(-1) == policy.argmax(-1)).float().mean()
    vres = F.nll_loss(vlogp, t["result"])
    return pl, vl, agree, vres


@torch.no_grad()
def evaluate(net, batches, targets, amp):
    net.eval()
    out = np.zeros(4)
    for t in batches:
        policy, value = targets(t, amp)
        out += np.array([float(x) for x in losses(net, t, policy, value, amp)])
    return dict(zip(["policy", "value", "agree", "value_vs_result"], out / len(batches)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="viriformat files or globs")
    ap.add_argument("--out", required=True, help="output folder (checkpoint.pt, net.pzn, log.jsonl)")
    ap.add_argument("--teacher", help="checkpoint.pt of the teacher network (distillation mode)")
    ap.add_argument("--init", help="checkpoint.pt to start the student from (its size and weights)")
    ap.add_argument("--net", default="64x6", help="student size CxB when not --init (default 64x6)")
    ap.add_argument("--positions", type=float, default=100e6, help="training positions in total")
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3, help="peak learning rate (cosine decay to 1%%)")
    ap.add_argument("--warmup", type=int, default=1000, help="warm-up steps")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--buffer", type=int, default=1_000_000, help="shuffle buffer (positions)")
    ap.add_argument("--skip-plies", type=int, default=0, help="skip the first N plies of every game record")
    ap.add_argument("--holdout-every", type=int, default=200, help="every N-th game is held out")
    ap.add_argument("--eval-every", type=int, default=2000, help="steps between held-out evaluations")
    ap.add_argument("--eval-batches", type=int, default=20)
    # distillation
    ap.add_argument("--teacher-temp", type=float, default=1.0, help="teacher policy softmax temperature")
    ap.add_argument("--result-weight", type=float, default=0.0,
                    help="distillation: blend this much game result into the teacher's value")
    # data labels
    ap.add_argument("--score-weight", type=float, default=0.5,
                    help="data labels: value = w * W/D/L(score) + (1 - w) * game result")
    ap.add_argument("--policy-smoothing", type=float, default=0.1,
                    help="data labels: share of the policy target spread over all legal moves")
    ap.add_argument("--policy-weight", type=float, default=None,
                    help="scale of the policy loss (default 1 for distillation, 0.5 for data labels)")
    ap.add_argument("--fit-positions", type=int, default=2_000_000,
                    help="data labels: positions used to fit the score -> W/D/L table")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    files = sorted({f for pat in args.data for f in (glob.glob(pat) or [pat])})
    dev = torch.device(args.device)
    amp = not args.no_amp and dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)
    distill = args.teacher is not None
    if args.policy_weight is None:
        args.policy_weight = 1.0 if distill else 0.5

    train = pz.VfStream(files, buffer=args.buffer, seed=args.seed, holdout_every=args.holdout_every,
                        holdout=False, skip_plies=args.skip_plies)
    test = pz.VfStream(files, buffer=min(args.buffer, 200_000), seed=args.seed + 1,
                       holdout_every=args.holdout_every, holdout=True, skip_plies=args.skip_plies)
    sizes = sum(os.path.getsize(f) for f in files)
    print(f"data: {len(files)} file(s), {sizes / 1e9:.2f} GB (~{sizes / 4e6:.0f}M positions at ~4 bytes each); "
          f"every {args.holdout_every}th game held out")

    teacher, wdl = None, None
    if distill:
        teacher, tck = load_net(args.teacher, dev)
        teacher.eval()
        tc = tck["config"]
        print(f"distillation: teacher {args.teacher} ({tc['channels']}x{tc['repr_blocks']}, "
              f"gen {tck.get('generation')}), temperature {args.teacher_temp}, result weight {args.result_weight}")
    else:
        t0 = time.time()
        wdl = fit_wdl_table(train, args.fit_positions)
        table, bin_cp, max_cp = wdl
        mid = max_cp // bin_cp
        print(f"score -> W/D/L fitted on {args.fit_positions:,} positions ({time.time() - t0:.0f}s): "
              + ", ".join(f"{cp:+d}cp W{table[mid + cp // bin_cp][0]:.2f} D{table[mid + cp // bin_cp][1]:.2f}"
                          for cp in (-300, -100, 0, 100, 300)))
        np.save(os.path.join(args.out, "wdl_table.npy"), table)
    targets = Targets(args, dev, teacher, wdl)

    if args.init:
        net, ick = load_net(args.init, dev)
        cfg = NetConfig(**ick["config"])
        print(f"student: {args.init} ({cfg.channels}x{cfg.repr_blocks}, gen {ick.get('generation')})")
    else:
        c, b = (int(x) for x in args.net.split("x"))
        cfg = NetConfig(channels=c, repr_blocks=b, dyn_blocks=b)
        net = MuZeroNet(cfg).to(dev)
        print(f"student: new {args.net}")
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    test_batches = [to_dev(test.batch(args.batch), dev) for _ in range(args.eval_batches)]
    if distill:
        ref = evaluate(teacher, test_batches, targets, amp)
        print(f"teacher on held-out games: value vs result {ref['value_vs_result']:.4f}")
    start = evaluate(net, test_batches, targets, amp)
    print(f"student at start: policy {start['policy']:.4f} value {start['value']:.4f} agree {start['agree']:.3f} "
          f"value vs result {start['value_vs_result']:.4f}")

    steps = int(args.positions // args.batch)
    log = open(os.path.join(args.out, "log.jsonl"), "a")
    best, t0, seen = None, time.time(), 0
    sums = np.zeros(2)
    print(f"training {steps:,} steps of {args.batch} ({args.positions / 1e6:.0f}M positions)", flush=True)
    for step in range(1, steps + 1):
        lr = args.lr * min(1.0, step / args.warmup) * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        t = to_dev(train.batch(args.batch), dev)
        policy, value = targets(t, amp)
        net.train()
        pl, vl, _, _ = losses(net, t, policy, value, amp)
        loss = args.policy_weight * pl + vl
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        if step % 50 == 0:
            sums += [float(pl.detach()), float(vl.detach())]
        seen += args.batch
        if step % args.eval_every == 0 or step == steps:
            ev = evaluate(net, test_batches, targets, amp)
            score = args.policy_weight * ev["policy"] + ev["value"]
            el = time.time() - t0
            rec = {"step": step, "positions": seen, "lr": lr, "epochs": train.epochs, "seconds": round(el),
                   "train_policy": sums[0] / max(1, args.eval_every // 50), "train_value": sums[1] / max(1, args.eval_every // 50),
                   **{f"test_{k}": v for k, v in ev.items()}}
            sums[:] = 0
            mark = ""
            if best is None or score < best:
                best, mark = score, "  *best"
                torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "config": cfg.__dict__,
                            "generation": 0, "step": 0}, os.path.join(args.out, "checkpoint.pt"))
                export_pzn(net.eval(), os.path.join(args.out, "net.pzn"))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(f"step {step:7d} | {seen / 1e6:7.1f}M pos ({seen / el:6.0f}/s) | held-out policy {ev['policy']:.4f} "
                  f"value {ev['value']:.4f} agree {ev['agree']:.3f} vs result {ev['value_vs_result']:.4f} | "
                  f"lr {lr:.1e}{mark}", flush=True)
    print(f"done: best checkpoint and net.pzn in {args.out}")


if __name__ == "__main__":
    main()
