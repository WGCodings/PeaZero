#!/usr/bin/env python3
"""Capacity check: is the network too small for the data?

Trains networks of different sizes from scratch, supervised, on the same
self-play data (the run's data/gen_* folders still on disk), with the same
number of steps, and compares their loss on held-out positions (the newest
--holdout-gens generations, never trained on). The run's own network
(checkpoint.pt) is measured on the same positions as a reference.

    python scripts\\capacity_check.py --run ..\\runs\\fourth                 # 64x6 vs 128x10
    python scripts\\capacity_check.py --run ..\\runs\\fourth --nets 64x6,96x8,128x10

Reading the result:
  * the bigger net reaches a clearly lower held-out policy/value loss than
    the same-budget 64x6 -> the 64x6 is capacity-limited; a bigger net pays.
  * about the same -> the data/targets are the limit, not the network.
  * train loss far below held-out loss -> over-fitting (too little data for
    that size); the best held-out point (early stopping) is what counts.

Every network's best checkpoint (by held-out loss) is saved to
<run>/capacity/<CxB>/ as checkpoint.pt (train.py format, usable as a warm
start) and net.pzn (for an SPRT). Results: <run>/capacity/results.json.
Stop the training first: it needs the GPU.
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import peazero_rs as pz  # noqa: E402
from peazero.model import MuZeroNet, NetConfig, export_pzn  # noqa: E402
from peazero.train_lib import autocast, batch_to_device, calibrate_bn, muzero_loss  # noqa: E402


def parse_net(s):
    m = re.fullmatch(r"(\d+)x(\d+)", s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"bad net size {s!r} (want e.g. 128x10)")
    return int(m.group(1)), int(m.group(2))


def gen_dirs(run):
    out = []
    for p in glob.glob(os.path.join(run, "data", "gen_*")):
        m = re.fullmatch(r"gen_(\d+)", os.path.basename(p))
        if m and os.path.isdir(p):
            out.append((int(m.group(1)), p))
    return sorted(out)


def load(dirs):
    replay = pz.Replay(capacity=10**9, seed=1)
    for _, d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*.bin"))):
            with open(f, "rb") as fh:
                replay.add_chunk(fh.read())
    return replay


@torch.no_grad()
def evaluate(net, batches, dev, amp):
    """Mean policy loss (vs search target), value loss (vs game result) and
    top-1 agreement over fixed batches, eval mode."""
    net.eval()
    tot = {"policy": 0.0, "value": 0.0, "agree": 0.0}
    n = 0
    for t in batches:
        with autocast(dev, amp):
            _, p, v = net.initial_inference(t["obs"])
        logp = F.log_softmax(p.float(), dim=-1)
        target = torch.zeros_like(logp).scatter_add_(1, t["policy_index"][:, 0], t["policy_prob"][:, 0])
        pm, vm = t["policy_mask"][:, 0], t["value_mask"][:, 0]
        tot["policy"] += float((-(target * logp).sum(-1) * pm).sum() / pm.sum().clamp(min=1))
        tot["value"] += float((F.cross_entropy(v.float(), t["value_target"][:, 0], reduction="none") * vm).sum()
                              / vm.sum().clamp(min=1))
        tot["agree"] += float(((p.argmax(-1) == target.argmax(-1)).float() * pm).sum() / pm.sum().clamp(min=1))
        n += 1
    return {k: v / n for k, v in tot.items()}


def bn_ready(net, replay, dev, amp):
    """Eval-mode copy with BatchNorm statistics recomputed on training data
    (same procedure as the exported networks)."""
    init_net, _ = calibrate_bn(net, replay, dev, amp)
    return init_net


def train_one(c, b, args, train, test_b, train_b, dev, amp, out_dir):
    torch.manual_seed(args.seed)
    cfg = NetConfig(channels=c, repr_blocks=b, dyn_blocks=b)
    net = MuZeroNet(cfg).to(dev)
    params = sum(p.numel() for p in net.representation.parameters()) + \
        sum(p.numel() for p in net.prediction.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            fused=dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    steps = max(1, int(args.epochs * train.positions / args.batch))
    evals = max(1, steps // args.evals)
    name = f"{c}x{b}"
    print(f"\n{name}: {params / 1e6:.2f}M parameters (representation + heads), {steps} steps "
          f"({args.epochs} epochs of {train.positions:,} samples), batch {args.batch}", flush=True)
    best, history, t0 = None, [], time.time()
    batch = batch_to_device(train.sample(args.batch, 0), dev)
    for step in range(1, steps + 1):
        # warm-up then cosine decay to 1 % of --lr
        warm = min(1.0, step / args.warmup)
        lr = args.lr * warm * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        for g in opt.param_groups:
            g["lr"] = lr
        net.train()
        loss, _ = muzero_loss(net, batch, dev, amp, q_ratio=args.q_ratio)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if step < steps:
            batch = batch_to_device(train.sample(args.batch, 0), dev)
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        scaler.step(opt)
        scaler.update()
        if step % evals == 0 or step == steps:
            ev_net = bn_ready(net, train, dev, amp)
            te = evaluate(ev_net, test_b, dev, amp)
            tr = evaluate(ev_net, train_b, dev, amp)
            score = te["policy"] + te["value"]
            row = {"step": step, "epoch": round(step * args.batch / train.positions, 2),
                   "test": te, "train": tr, "seconds": round(time.time() - t0)}
            history.append(row)
            mark = ""
            if best is None or score < best["test"]["policy"] + best["test"]["value"]:
                best = row
                mark = "  *best"
                os.makedirs(out_dir, exist_ok=True)
                torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "config": cfg.__dict__,
                            "generation": 0, "step": 0}, os.path.join(out_dir, "checkpoint.pt"))
                export_pzn(ev_net, os.path.join(out_dir, "net.pzn"))
            print(f"  epoch {row['epoch']:5.2f}  held-out p {te['policy']:.4f} v {te['value']:.4f} "
                  f"agree {te['agree']:.3f} | train p {tr['policy']:.4f} v {tr['value']:.4f} | "
                  f"{row['seconds']}s{mark}", flush=True)
    return {"net": name, "params": params, "steps": steps, "best": best, "history": history,
            "seconds": round(time.time() - t0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--nets", default="64x6,128x10", help="comma-separated CxB sizes to train from scratch")
    ap.add_argument("--holdout-gens", type=int, default=2, help="newest generations held out as test data")
    ap.add_argument("--epochs", type=float, default=12, help="passes over the training samples per network")
    ap.add_argument("--evals", type=int, default=12, help="held-out evaluations per network")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--q-ratio", type=float, help="value target mix (default: the run's setting)")
    ap.add_argument("--test-batches", type=int, default=40, help="fixed held-out batches of 1024")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()
    nets = [parse_net(s) for s in args.nets.split(",")]
    dev = torch.device(args.device)
    amp = not args.no_amp and dev.type == "cuda"
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    cfg_path = os.path.join(args.run, "config.json")
    run_cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {}
    if args.q_ratio is None:
        args.q_ratio = run_cfg.get("q_ratio", 0.5)

    dirs = gen_dirs(args.run)
    if len(dirs) <= args.holdout_gens:
        sys.exit(f"only {len(dirs)} data generations on disk")
    train_dirs, test_dirs = dirs[:-args.holdout_gens], dirs[-args.holdout_gens:]
    train, test = load(train_dirs), load(test_dirs)
    print(f"data: training gens {train_dirs[0][0]}-{train_dirs[-1][0]} ({train.positions:,} samples, "
          f"{train.games:,} games); held-out gens {test_dirs[0][0]}-{test_dirs[-1][0]} "
          f"({test.positions:,} samples); q_ratio {args.q_ratio}")
    test_b = [batch_to_device(test.sample(1024, 0), dev) for _ in range(args.test_batches)]
    train_b = [batch_to_device(train.sample(1024, 0), dev) for _ in range(max(4, args.test_batches // 4))]

    results = {"data": {"train_samples": train.positions, "test_samples": test.positions,
                        "train_gens": [train_dirs[0][0], train_dirs[-1][0]],
                        "test_gens": [test_dirs[0][0], test_dirs[-1][0]]}, "nets": []}
    ck = os.path.join(args.run, "checkpoint.pt")
    if os.path.exists(ck):
        state = torch.load(ck, map_location="cpu", weights_only=False)
        ref = MuZeroNet(NetConfig(**state["config"])).to(dev)
        ref.load_state_dict(state["model"])
        r = evaluate(bn_ready(ref, train, dev, amp), test_b, dev, amp)
        rc = state["config"]
        results["reference"] = {"net": f"{rc['channels']}x{rc['repr_blocks']}", "generation": state["generation"],
                                "test": r}
        print(f"reference: the run's network ({rc['channels']}x{rc['repr_blocks']}, gen {state['generation']}, "
              f"trained by self-play on all its data): held-out p {r['policy']:.4f} v {r['value']:.4f} "
              f"agree {r['agree']:.3f}")

    out_root = os.path.join(args.run, "capacity")
    for c, b in nets:
        res = train_one(c, b, args, train, test_b, train_b, dev, amp, os.path.join(out_root, f"{c}x{b}"))
        results["nets"].append(res)
        os.makedirs(out_root, exist_ok=True)
        with open(os.path.join(out_root, "results.json"), "w") as f:
            json.dump(results, f, indent=1)

    print("\nsummary (best held-out point of each network; lower loss = better):")
    rows = [(r["net"], r["params"], r["best"]) for r in results["nets"]]
    base = rows[0][2]["test"]
    for name, params, b in rows:
        te, tr = b["test"], b["train"]
        print(f"  {name:>7s} {params / 1e6:5.2f}M  policy {te['policy']:.4f} ({te['policy'] - base['policy']:+.4f})  "
              f"value {te['value']:.4f} ({te['value'] - base['value']:+.4f})  agree {te['agree']:.3f}  "
              f"[train p {tr['policy']:.4f}, epoch {b['epoch']}]")
    if "reference" in results:
        r = results["reference"]["test"]
        print(f"  run net (gen {results['reference']['generation']}): policy {r['policy']:.4f}  "
              f"value {r['value']:.4f}  agree {r['agree']:.3f}")
    print(f"\nsaved: {os.path.join(out_root, '<net>', 'checkpoint.pt')} and net.pzn "
          f"(best held-out point), {os.path.join(out_root, 'results.json')}")


if __name__ == "__main__":
    main()
