#!/usr/bin/env python3
"""PeaZero training loop: alternate root-only-Gumbel self-play and training.

    python train.py --run runs/first --net 64x6
    python train.py --run runs/first --resume          # continue a run

Each generation:
  1. self-play until --positions-per-gen new positions are recorded
  2. train ~positions × --reuse / --batch steps on the replay buffer
  3. save the checkpoint and export the network

Run directory layout:
  <run>/config.json            arguments of the first start (for reference)
  <run>/checkpoint.pt          latest model + optimizer (used by --resume)
  <run>/metrics.jsonl          one line per generation
  <run>/latest.pzn             copy of the newest network
  <run>/nets/gen_XXXX.pzn      network after every generation
  <run>/data/gen_XXXX/*.bin    self-play games (data format v0)

Old data is deleted automatically once the replay buffer can no longer use
it (see --keep-data-gens); networks are always kept.
"""
import argparse
import glob
import json
import math
import os
import re
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peazero_rs as pz  # noqa: E402  (pip install ./crates/py)
from peazero import encoding as enc  # noqa: E402
from peazero.model import MuZeroNet, NetConfig, export_pzn  # noqa: E402
from peazero.train_lib import muzero_loss, selfplay_step  # noqa: E402

SIZES = {"32x4": (32, 4), "64x6": (64, 6), "128x10": (128, 10)}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--net", default="64x6", choices=SIZES)
    ap.add_argument("--generations", type=int, default=1000)
    # self-play
    ap.add_argument("--games", type=int, default=1024, help="concurrent self-play games")
    ap.add_argument("--considered", type=int, default=16, help="Gumbel m: actions evaluated per move")
    ap.add_argument("--max-plies", type=int, default=512)
    ap.add_argument("--positions-per-gen", type=int, default=100_000)
    # training
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--unroll", type=int, default=5)
    ap.add_argument("--reuse", type=float, default=4.0, help="avg times each new position is sampled")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--replay", type=int, default=1_000_000, help="replay buffer size in positions")
    ap.add_argument("--keep-data-gens", type=int, default=-1,
                    help="generations of self-play data kept on disk; -1 = just enough to refill the "
                         "replay buffer on --resume (default), 0 = keep everything")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on CUDA")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def log(run, record):
    with open(os.path.join(run, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def data_gen_dirs(run):
    """(generation, path) of every data/gen_XXXX folder, oldest first."""
    out = []
    for path in glob.glob(os.path.join(run, "data", "gen_*")):
        m = re.fullmatch(r"gen_(\d+)", os.path.basename(path))
        if m and os.path.isdir(path):
            out.append((int(m.group(1)), path))
    return sorted(out)


def migrate_flat_data(run):
    """Move files from the old flat layout (data/gen0001_000.bin) into
    data/gen_0001/part_000.bin."""
    moved = 0
    for path in glob.glob(os.path.join(run, "data", "gen*_*.bin")):
        m = re.fullmatch(r"gen(\d+)_(\d+)\.bin", os.path.basename(path))
        if not m:
            continue
        target_dir = os.path.join(run, "data", f"gen_{int(m.group(1)):04d}")
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(path, os.path.join(target_dir, f"part_{int(m.group(2)):03d}.bin"))
        moved += 1
    if moved:
        print(f"moved {moved} data files into per-generation folders")


def replay_gens(args):
    """Generations of data needed to fill the replay buffer, plus one spare."""
    return math.ceil(args.replay / args.positions_per_gen) + 1


def data_gens_to_keep(args):
    if args.keep_data_gens >= 0:
        return args.keep_data_gens  # 0 = keep everything
    return replay_gens(args)


def prune_data(run, keep):
    if keep <= 0:
        return
    dirs = data_gen_dirs(run)
    for _, path in dirs[:-keep]:
        shutil.rmtree(path, ignore_errors=True)


def main():
    args = parse_args()
    dev = torch.device(args.device)
    amp = not args.no_amp and dev.type == "cuda"
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    for d in ("data", "nets"):
        os.makedirs(os.path.join(args.run, d), exist_ok=True)
    migrate_flat_data(args.run)
    config_path = os.path.join(args.run, "config.json")
    if not os.path.exists(config_path):
        with open(config_path, "w") as f:
            json.dump(vars(args), f, indent=2)
    keep_gens = data_gens_to_keep(args)
    assert pz.ENCODING_VERSION == enc.ENCODING_VERSION, "rebuild peazero_rs: encoding version mismatch"

    ckpt_path = os.path.join(args.run, "checkpoint.pt")
    c, blocks = SIZES[args.net]
    cfg = NetConfig(channels=c, repr_blocks=blocks, dyn_blocks=blocks)
    gen, step = 0, 0
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = NetConfig(**ckpt["config"])
        gen, step = ckpt["generation"], ckpt["step"]
    net = MuZeroNet(cfg).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if args.resume and os.path.exists(ckpt_path):
        net.load_state_dict(ckpt["model"])
        opt.load_state_dict(ckpt["optimizer"])
        print(f"resumed {args.run} at generation {gen}, step {step}")

    replay = pz.Replay(capacity=args.replay, seed=args.seed + gen)
    if args.resume:  # refill the replay buffer, oldest data first (newest must survive eviction)
        dirs = data_gen_dirs(args.run)
        refill = dirs[-replay_gens(args):]
        for _, d in refill:
            for path in sorted(glob.glob(os.path.join(d, "*.bin"))):
                with open(path, "rb") as f:
                    replay.add_chunk(f.read())
        print(f"replay buffer: {replay.positions:,} positions from {len(refill)} generations on disk")

    sp = pz.SelfPlay(games=args.games, considered=args.considered, max_plies=args.max_plies, seed=args.seed + gen)
    params = sum(p.numel() for p in net.parameters()) / 1e6
    print(f"net {args.net} ({params:.2f}M params) on {dev}{' fp16' if amp else ''}; "
          f"{args.games} games × {args.considered} considered")

    while gen < args.generations:
        gen += 1
        # ---------------------------------------------------------------- self-play
        net.eval()
        t0 = time.time()
        new_positions, new_games, part = 0, 0, 0
        gen_dir = os.path.join(args.run, "data", f"gen_{gen:04d}")
        os.makedirs(gen_dir, exist_ok=True)
        stats_before = sp.stats()
        moves = 0
        while new_positions < args.positions_per_gen:
            selfplay_step(net, sp, dev, amp)
            moves += sp.num_games
            if moves % (sp.num_games * 32) == 0:
                chunk, games, positions = sp.take_chunk()
                if games:
                    path = os.path.join(gen_dir, f"part_{part:03d}.bin")
                    with open(path, "wb") as f:
                        f.write(chunk)
                    replay.add_chunk(chunk)
                    part += 1
                    new_positions += positions
                    new_games += games
        sp_time = time.time() - t0
        st = sp.stats()
        d = {k: st[k] - stats_before[k] for k in ("games", "white_wins", "black_wins", "draws")}
        terms = {k: st["terminations"][k] - stats_before["terminations"][k] for k in st["terminations"]}

        # ---------------------------------------------------------------- training
        net.train()
        t1 = time.time()
        n_steps = max(1, int(new_positions * args.reuse / args.batch))
        sums = {}
        for _ in range(n_steps):
            step += 1
            lr = args.lr * min(1.0, step / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            batch = replay.sample(args.batch, args.unroll)
            loss, m = muzero_loss(net, batch, dev, amp)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            for k, v in m.items():
                sums[k] = sums.get(k, 0.0) + v
        train_time = time.time() - t1
        avg = {k: v / n_steps for k, v in sums.items()}

        # ---------------------------------------------------------------- save
        net.eval()
        net_path = os.path.join(args.run, "nets", f"gen_{gen:04d}.pzn")
        export_pzn(net, net_path)
        shutil.copyfile(net_path, os.path.join(args.run, "latest.pzn"))
        prune_data(args.run, keep_gens)
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "config": cfg.__dict__,
                    "generation": gen, "step": step}, ckpt_path)

        record = {
            "generation": gen, "step": step, "lr": lr,
            "selfplay_seconds": round(sp_time, 1), "train_seconds": round(train_time, 1),
            "moves_per_second": round(moves / sp_time), "new_games": new_games, "new_positions": new_positions,
            "avg_game_length": round(new_positions / max(new_games, 1), 1),
            "results": d, "terminations": terms, "replay_positions": replay.positions, **avg,
        }
        log(args.run, record)
        print(f"gen {gen:4d} | selfplay {sp_time:6.1f}s {moves / sp_time:7.0f} mv/s "
              f"{new_games:5d} games (W{d['white_wins']} D{d['draws']} B{d['black_wins']}, "
              f"len {record['avg_game_length']}) | train {n_steps} steps {train_time:6.1f}s "
              f"loss {avg['loss']:.3f} (p {avg['policy_loss']:.3f} v {avg['value_loss']:.3f} "
              f"agree {avg['policy_agreement']:.2f}) | {net_path}")


if __name__ == "__main__":
    main()
