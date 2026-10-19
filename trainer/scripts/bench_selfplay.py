#!/usr/bin/env python3
"""Self-play speed benchmark: the plain loop vs. pinned buffers, CUDA graphs
and two pipelined pools (peazero/selfplay_gpu.py), on the same network.

    python scripts/bench_selfplay.py --run ..\\runs\\fourth          # the run's network and settings
    python scripts/bench_selfplay.py --net 64x6 --threads 8          # random 64x6 network

Prints, per variant: moves/s and where the time goes (waiting for the GPU,
Rust search work, Python launch overhead); first the raw network speed with
and without CUDA graphs, and a check that the graph gives the same outputs.
Takes ~3-4 minutes. Stop the training first (it competes for GPU and CPU).
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import peazero_rs as pz  # noqa: E402
from peazero.model import MuZeroNet, NetConfig  # noqa: E402
from peazero.selfplay_gpu import SearchRunner  # noqa: E402
from peazero.train_lib import search_step  # noqa: E402

SIZES = {"32x4": (32, 4), "64x6": (64, 6), "128x10": (128, 10)}


def load_net(args, dev):
    if args.run:
        ckpt = torch.load(os.path.join(args.run, "checkpoint.pt"), map_location="cpu", weights_only=False)
        net = MuZeroNet(NetConfig(**ckpt["config"]))
        net.load_state_dict(ckpt["model"])
        desc = f"{args.run} gen {ckpt['generation']}"
    else:
        c, b = SIZES[args.net]
        net = MuZeroNet(NetConfig(c, b, b))
        desc = f"random {args.net}"
    return net.to(dev).eval(), desc


def make_pools(args, games, n_pools):
    sizes = [games // n_pools + (i < games % n_pools) for i in range(n_pools)]
    return [pz.SearchSelfPlay(games=n, simulations=args.simulations, considered=args.considered,
                              max_plies=args.max_plies, seed=1 + 1_000_003 * i, threads=args.threads,
                              noise_plies=args.noise_plies)
            for i, n in enumerate(sizes)]


def raw_forward(net, dev, batch, graphs, amp, reps=50, cl=False):
    """Network-only throughput (evaluations/s) at one batch size."""
    pool = make_pools(ARGS, batch, 1)
    r = SearchRunner(pool, dev, amp, cuda_graphs=graphs, channels_last=cl)
    r.set_net(net)
    s = r.slots[0]
    s.obs_h.zero_()
    s.legal_h.zero_()  # valid gather indices
    for _ in range(5):
        r._launch(s)
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(reps):
        r._launch(s)
    torch.cuda.synchronize()
    return batch * reps / (time.perf_counter() - t)


def check_graph(net, dev, amp, cl=False):
    """Same requests through eager and graphed inference: max |difference|."""
    pools = make_pools(ARGS, 256, 1)
    eager = SearchRunner(pools, dev, amp, cuda_graphs=False)
    eager.set_net(net)
    graph = SearchRunner(pools, dev, amp, cuda_graphs=True, channels_last=cl)
    graph.set_net(net)
    if not graph.graphs:
        return None
    se, sg = eager.slots[0], graph.slots[0]
    pools[0].requests_into(se.obs_np, se.legal_np)
    sg.obs_np[...] = se.obs_np
    sg.legal_np[...] = se.legal_np
    eager._launch(se)
    graph._launch(sg)
    torch.cuda.synchronize()
    return max(float((se.logits_h - sg.logits_h).abs().max()), float((se.value_h - sg.value_h).abs().max()))


def run_variant(name, net, dev, amp, games, pools, graphs, plain=False, cl=False):
    ps = make_pools(ARGS, games, max(1, pools))
    if plain:
        sp = ps[0]
        step = lambda: search_step(net, sp, dev, amp)  # noqa: E731
        stats = sp.stats
    else:
        r = SearchRunner(ps, dev, amp, cuda_graphs=graphs, channels_last=cl)
        r.set_net(net)
        step, stats = r.step, r.stats
    t_end = time.perf_counter() + ARGS.warmup
    while time.perf_counter() < t_end:
        step()
    if not plain:
        r.reset_timing()
    plies0, t0 = stats()["plies"], time.perf_counter()
    while time.perf_counter() - t0 < ARGS.seconds:
        step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mps = (stats()["plies"] - plies0) / dt
    split = ""
    out = {"variant": name, "moves_per_second": round(mps), "games": games}
    if not plain:
        tm = r.timing
        steps = max(tm["steps"], 1)
        out.update({k: round(tm[k] / dt * 100, 1) for k in ("gpu_wait", "rust", "launch")})
        out["ms_per_step"] = round(dt / steps * 1000, 2)
        split = (f"  | per step {dt / steps * 1000:6.2f} ms: gpu wait {tm['gpu_wait'] / dt * 100:4.1f}%  "
                 f"rust {tm['rust'] / dt * 100:4.1f}%  launch {tm['launch'] / dt * 100:4.1f}%")
        if graphs and r.cuda and not r.graphs:
            split += "  (graph capture failed → eager)"
    print(f"{name:34s} {mps:7.0f} mv/s{split}", flush=True)
    del ps
    return out


def main():
    global ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", help="use this run's checkpoint and settings")
    ap.add_argument("--net", default="64x6", choices=SIZES)
    ap.add_argument("--games", type=int)
    ap.add_argument("--simulations", type=int)
    ap.add_argument("--considered", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--max-plies", type=int, default=512)
    ap.add_argument("--noise-plies", type=int, default=30)
    ap.add_argument("--seconds", type=float, default=20, help="measured seconds per variant")
    ap.add_argument("--warmup", type=float, default=8, help="unmeasured seconds per variant first")
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--only-new", action="store_true",
                    help="self-play: only the best variant and the channels_last variant")
    ap.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)  # plumbing test without a GPU
    ARGS = ap.parse_args()
    cfg = {}
    if ARGS.run and os.path.exists(os.path.join(ARGS.run, "config.json")):
        with open(os.path.join(ARGS.run, "config.json")) as f:
            cfg = json.load(f)
    for k, d in (("games", 1024), ("simulations", 64), ("considered", 16), ("threads", 0)):
        if getattr(ARGS, k) is None:
            setattr(ARGS, k, cfg.get(k, d))
    dev = torch.device("cuda" if torch.cuda.is_available() and not ARGS.cpu else "cpu")
    if dev.type != "cuda" and not ARGS.cpu:
        sys.exit("needs a CUDA GPU")
    torch.backends.cudnn.benchmark = True
    amp = not ARGS.no_amp and dev.type == "cuda"
    net, desc = load_net(ARGS, dev)
    gpu = torch.cuda.get_device_name() if dev.type == "cuda" else "CPU"
    print(f"{desc}; {ARGS.games} games, {ARGS.simulations} simulations, {ARGS.considered} considered, "
          f"threads {ARGS.threads or 'all'}; {gpu}; fp16 {amp}")
    if dev.type == "cuda":
        network_only(net, dev, amp)
    g = ARGS.games
    print(f"\nself-play ({ARGS.warmup:.0f}s warm-up + {ARGS.seconds:.0f}s each):")
    if ARGS.only_new:
        a = run_variant("pinned + graphs, 2 pools", net, dev, amp, g, 2, True)
        b = run_variant("graphs + channels_last, 2 pools", net, dev, amp, g, 2, True, cl=True)
        print(f"\nchannels_last: {b['moves_per_second'] / a['moves_per_second']:.2f}x")
        return
    results = [
        run_variant("plain loop (current)", net, dev, amp, g, 1, False, plain=True),
        run_variant("pinned, 1 pool", net, dev, amp, g, 1, False),
        run_variant("pinned + graphs, 1 pool", net, dev, amp, g, 1, True),
        run_variant("pinned, 2 pools", net, dev, amp, g, 2, False),
        run_variant("pinned + graphs, 2 pools", net, dev, amp, g, 2, True),
        run_variant(f"pinned + graphs, 2 pools, {2 * g} games", net, dev, amp, 2 * g, 2, True),
        run_variant("graphs + channels_last, 2 pools", net, dev, amp, g, 2, True, cl=True),
    ]
    base = results[0]["moves_per_second"]
    print("\nspeed-up vs plain loop: " + ", ".join(
        f"{r['variant']} {r['moves_per_second'] / base:.2f}x" for r in results[1:]))


def network_only(net, dev, amp):
    print("\nnetwork only (evaluations/s):")
    for b in (512, 1024, 2048):
        e = raw_forward(net, dev, b, False, amp)
        g = raw_forward(net, dev, b, True, amp)
        c = raw_forward(net, dev, b, True, amp, cl=True)
        print(f"  batch {b:4d}: eager {e:9,.0f}   CUDA graph {g:9,.0f} ({g / e:.2f}x)   "
              f"graph + channels_last {c:9,.0f} ({c / e:.2f}x)", flush=True)
    diff = check_graph(net, dev, amp)
    print("  graph vs eager outputs: " + ("capture failed" if diff is None else f"max |diff| {diff:.2e}"))
    diff = check_graph(net, dev, amp, cl=True)
    print("  graph + channels_last vs eager: " + ("capture failed" if diff is None else f"max |diff| {diff:.2e}")
          + " (other kernels: small fp16 differences are expected)")


if __name__ == "__main__":
    main()
