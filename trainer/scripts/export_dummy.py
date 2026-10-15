#!/usr/bin/env python3
"""Export a randomly initialised MuZero net to ONNX (+ a bundled net.pzn),
plus reference outputs so the Rust side can check its inference matches PyTorch.

    python scripts/export_dummy.py --channels 64 --repr-blocks 6 --dyn-blocks 6 --out ../models/dummy-64x6
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from peazero import encoding as enc  # noqa: E402
from peazero.model import MuZeroNet, NetConfig, export_onnx, write_pzn  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--repr-blocks", type=int, default=6)
    ap.add_argument("--dyn-blocks", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    cfg = NetConfig(channels=args.channels, repr_blocks=args.repr_blocks, dyn_blocks=args.dyn_blocks)
    net = MuZeroNet(cfg)
    # Randomise BN statistics so the check isn't trivially "identity BN".
    for m in net.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.running_mean.uniform_(-0.1, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    net.eval()
    export_onnx(net, args.out)
    write_pzn(os.path.join(args.out, "initial.onnx"), os.path.join(args.out, "recurrent.onnx"),
              os.path.join(args.out, "net.pzn"))

    # Reference: start position observation + action e2e4.
    import chess

    obs = torch.from_numpy(enc.encode_board(chess.Board(), 0)).unsqueeze(0)
    action = enc.move_to_action(chess.E2, chess.E4, None, True)
    act = torch.from_numpy(enc.encode_action_planes(action)).unsqueeze(0)
    with torch.no_grad():
        s, p, v = net.initial_inference(obs)
        s2, p2, v2 = net.recurrent_inference(s, act)
    ref = {
        "config": cfg.__dict__,
        "action": action,
        "initial": {"policy_head": p[0, :16].tolist(), "policy_sum": float(p.sum()), "value": v[0].tolist()},
        "recurrent": {"policy_head": p2[0, :16].tolist(), "policy_sum": float(p2.sum()), "value": v2[0].tolist()},
    }
    with open(os.path.join(args.out, "reference.json"), "w") as f:
        json.dump(ref, f, indent=1)
    params = sum(x.numel() for x in net.parameters())
    print(f"exported {args.out}: {params / 1e6:.2f}M params")


if __name__ == "__main__":
    main()
