#!/usr/bin/env python3
"""GPU spike for the RTX 3070: how fast can we run self-play inference and
training steps for each candidate network size?

    pip install torch onnxruntime-gpu   # onnxruntime-gpu is optional
    python scripts/bench_gpu.py                 # all sizes
    python scripts/bench_gpu.py --sizes 64x6

Reports:
  * inference  : recurrent (dynamics + prediction) evals/s at several batch
                 sizes, fp32 and fp16 (autocast). This bounds self-play speed.
  * onnxruntime: same via CUDAExecutionProvider on the exported ONNX file,
                 i.e. roughly what the Rust self-play (ort crate) will get.
  * training   : positions/s for a full K=5 unroll forward+backward step.
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from peazero import encoding as enc  # noqa: E402
from peazero.model import MuZeroNet, NetConfig, export_onnx  # noqa: E402

SIZES = {"32x4": (32, 4), "64x6": (64, 6), "128x10": (128, 10)}


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def bench_inference(net, dev, batch, half, seconds=2.0):
    c = net.cfg.channels
    lat = torch.rand(batch, c, 8, 8, device=dev)
    act = torch.zeros(batch, enc.ACTION_PLANES, 8, 8, device=dev)
    ctx = torch.autocast(dev.type, dtype=torch.float16, enabled=half and dev.type == "cuda")
    with torch.inference_mode(), ctx:
        for _ in range(5):
            net.recurrent_inference(lat, act)
        sync(dev)
        n, t0 = 0, time.perf_counter()
        while time.perf_counter() - t0 < seconds:
            net.recurrent_inference(lat, act)
            n += 1
        sync(dev)
    return n * batch / (time.perf_counter() - t0)


def bench_ort(onnx_dir, channels, batch, seconds=2.0):
    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        return None
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return None
    s = ort.InferenceSession(os.path.join(onnx_dir, "recurrent.onnx"), providers=["CUDAExecutionProvider"])
    feed = {
        "latent_in": np.random.rand(batch, channels, 8, 8).astype(np.float32),
        "action": np.zeros((batch, enc.ACTION_PLANES, 8, 8), np.float32),
    }
    for _ in range(5):
        s.run(None, feed)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        s.run(None, feed)
        n += 1
    return n * batch / (time.perf_counter() - t0)


def bench_training(net, dev, batch, unroll=5, steps=20, half=True):
    """One MuZero training step: h on obs, then `unroll` × (g, f), losses on
    every step, backward, optimizer step. Random data; only speed matters."""
    net.train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler(dev.type, enabled=half and dev.type == "cuda")
    obs = torch.rand(batch, enc.OBS_PLANES, 8, 8, device=dev)
    acts = torch.zeros(unroll, batch, enc.ACTION_PLANES, 8, 8, device=dev)
    tgt_p = torch.softmax(torch.randn(unroll + 1, batch, enc.NUM_ACTIONS, device=dev), -1)
    tgt_v = torch.randint(0, 3, (unroll + 1, batch), device=dev)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=half and dev.type == "cuda"):
            s, p, v = net.initial_inference(obs)
            loss = F.cross_entropy(p, tgt_p[0]) + F.cross_entropy(v, tgt_v[0])
            for k in range(unroll):
                s = s * 0.5 + s.detach() * 0.5  # MuZero gradient scaling
                s, p, v = net.recurrent_inference(s, acts[k])
                loss = loss + (F.cross_entropy(p, tgt_p[k + 1]) + F.cross_entropy(v, tgt_v[k + 1])) / unroll
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(3):
        step()
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    sync(dev)
    return steps * batch / (time.perf_counter() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="*", default=list(SIZES))
    ap.add_argument("--batches", nargs="*", type=int, default=[64, 256, 1024])
    ap.add_argument("--train-batch", type=int, default=512)
    ap.add_argument("--cpu", action="store_true", help="force CPU (sanity check only)")
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    name = torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"
    print(f"device: {name}   torch {torch.__version__}")
    if dev.type == "cpu":
        print("WARNING: no CUDA device — numbers are only a smoke test")
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    for size in args.sizes:
        c, b = SIZES[size]
        net = MuZeroNet(NetConfig(channels=c, repr_blocks=b, dyn_blocks=b)).to(dev).eval()
        params = sum(p.numel() for p in net.parameters()) / 1e6
        print(f"\n== {size}  ({params:.2f}M params)")
        for batch in args.batches:
            fp32 = bench_inference(net, dev, batch, half=False)
            fp16 = bench_inference(net, dev, batch, half=True)
            print(f"  inference batch {batch:>5}: fp32 {fp32:>10,.0f}  fp16 {fp16:>10,.0f} evals/s")
        if dev.type == "cuda":
            onnx_dir = os.path.join("..", "models", f"bench-{size}")
            export_onnx(net.cpu(), onnx_dir)
            net.to(dev)
            for batch in args.batches:
                r = bench_ort(onnx_dir, c, batch)
                if r is None:
                    print("  onnxruntime-gpu not available (pip install onnxruntime-gpu) — skipped")
                    break
                print(f"  ort-cuda  batch {batch:>5}: {r:>10,.0f} evals/s")
        tps = bench_training(net, dev, args.train_batch, steps=20 if dev.type == "cuda" else 2)
        print(f"  training  batch {args.train_batch}, K=5 unroll: {tps:,.0f} positions/s")


if __name__ == "__main__":
    main()
