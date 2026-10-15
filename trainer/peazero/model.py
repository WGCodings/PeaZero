"""MuZero networks for chess (v0 architecture).

    representation h: obs (21×8×8)                   → latent s (C×8×8)
    dynamics       g: latent s + action planes (5×8×8) → next latent s'
    prediction     f: latent s                       → policy logits (4672), value logits (W/D/L)

Exported to ONNX as two graphs, matching how search calls them:
    initial   : obs                 → latent, policy, value   (h then f)
    recurrent : latent, action      → latent, policy, value   (g then f)

No reward head: in chess every non-terminal reward is 0, and the game
outcome is learned through the value head (see docs/MUZERO.md).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encoding as enc


@dataclass(frozen=True)
class NetConfig:
    channels: int = 64
    repr_blocks: int = 6
    dyn_blocks: int = 6
    policy_channels: int = 32
    value_channels: int = 8
    value_hidden: int = 128


class ResBlock(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


def conv_bn_relu(cin: int, cout: int, k: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False), nn.BatchNorm2d(cout), nn.ReLU())


def normalize_latent(s: torch.Tensor) -> torch.Tensor:
    """Min-max scale each sample's latent to [0, 1] (MuZero appendix G)."""
    flat = s.flatten(1)
    lo = flat.min(dim=1, keepdim=True).values
    hi = flat.max(dim=1, keepdim=True).values
    flat = (flat - lo) / (hi - lo + 1e-5)
    return flat.view_as(s)


class Representation(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.stem = conv_bn_relu(enc.OBS_PLANES, cfg.channels, 3)
        self.blocks = nn.Sequential(*[ResBlock(cfg.channels) for _ in range(cfg.repr_blocks)])

    def forward(self, obs):
        return normalize_latent(self.blocks(self.stem(obs)))


class Dynamics(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.stem = conv_bn_relu(cfg.channels + enc.ACTION_PLANES, cfg.channels, 3)
        self.blocks = nn.Sequential(*[ResBlock(cfg.channels) for _ in range(cfg.dyn_blocks)])

    def forward(self, latent, action_planes):
        x = torch.cat([latent, action_planes], dim=1)
        return normalize_latent(self.blocks(self.stem(x)))


class Prediction(nn.Module):
    def __init__(self, cfg: NetConfig):
        super().__init__()
        self.policy = nn.Sequential(
            conv_bn_relu(cfg.channels, cfg.policy_channels, 1),
            nn.Conv2d(cfg.policy_channels, enc.POLICY_PLANES, 1),
        )
        self.value_conv = conv_bn_relu(cfg.channels, cfg.value_channels, 1)
        self.value_fc = nn.Sequential(
            nn.Linear(cfg.value_channels * 64, cfg.value_hidden),
            nn.ReLU(),
            nn.Linear(cfg.value_hidden, 3),
        )

    def forward(self, latent):
        # (N, 73, 8, 8) flattened NCHW → index = plane * 64 + square, as in encoding.rs
        policy = self.policy(latent).flatten(1)
        value = self.value_fc(self.value_conv(latent).flatten(1))
        return policy, value


class MuZeroNet(nn.Module):
    def __init__(self, cfg: NetConfig = NetConfig()):
        super().__init__()
        self.cfg = cfg
        self.representation = Representation(cfg)
        self.dynamics = Dynamics(cfg)
        self.prediction = Prediction(cfg)

    def initial_inference(self, obs):
        s = self.representation(obs)
        p, v = self.prediction(s)
        return s, p, v

    def recurrent_inference(self, latent, action_planes):
        s = self.dynamics(latent, action_planes)
        p, v = self.prediction(s)
        return s, p, v


class _Initial(nn.Module):
    def __init__(self, net: MuZeroNet):
        super().__init__()
        self.net = net

    def forward(self, obs):
        return self.net.initial_inference(obs)


class _Recurrent(nn.Module):
    def __init__(self, net: MuZeroNet):
        super().__init__()
        self.net = net

    def forward(self, latent, action_planes):
        return self.net.recurrent_inference(latent, action_planes)


PZN_MAGIC = b"PZNN"
PZN_VERSION = 1


def write_pzn(initial_onnx: str, recurrent_onnx: str, out_path: str) -> None:
    """Bundle the two ONNX graphs into one .pzn file (format: crates/nn/src/lib.rs)."""
    import struct

    with open(initial_onnx, "rb") as f:
        ini = f.read()
    with open(recurrent_onnx, "rb") as f:
        rec = f.read()
    with open(out_path, "wb") as f:
        f.write(PZN_MAGIC)
        f.write(struct.pack("<II", PZN_VERSION, enc.ENCODING_VERSION))
        f.write(struct.pack("<Q", len(ini)))
        f.write(ini)
        f.write(struct.pack("<Q", len(rec)))
        f.write(rec)


def export_pzn(net: MuZeroNet, out_path: str) -> None:
    """Export straight to a single .pzn file (works for a net on any device)."""
    import copy
    import os
    import tempfile

    cpu_net = copy.deepcopy(net).float().cpu()
    with tempfile.TemporaryDirectory() as tmp:
        export_onnx(cpu_net, tmp)
        write_pzn(os.path.join(tmp, "initial.onnx"), os.path.join(tmp, "recurrent.onnx"), out_path)


def export_onnx(net: MuZeroNet, out_dir: str, opset: int = 17) -> None:
    """Write initial.onnx and recurrent.onnx with a dynamic batch axis."""
    import os

    os.makedirs(out_dir, exist_ok=True)
    was_training = net.training
    # Wrappers must be in eval mode too: torch.onnx.export restores the
    # wrapper's mode afterwards, and .train() would propagate into `net`.
    initial, recurrent = _Initial(net).eval(), _Recurrent(net).eval()
    c = net.cfg.channels
    obs = torch.zeros(1, enc.OBS_PLANES, 8, 8)
    latent = torch.zeros(1, c, 8, 8)
    act = torch.zeros(1, enc.ACTION_PLANES, 8, 8)
    outs = ["latent", "policy", "value"]
    dyn = {name: {0: "batch"} for name in ["obs", "action", *outs]}
    dyn["latent_in"] = {0: "batch"}
    with torch.no_grad():
        torch.onnx.export(
            initial, (obs,), os.path.join(out_dir, "initial.onnx"),
            input_names=["obs"], output_names=outs,
            dynamic_axes={k: v for k, v in dyn.items() if k in ["obs", *outs]},
            opset_version=opset, dynamo=False,
        )
        torch.onnx.export(
            recurrent, (latent, act), os.path.join(out_dir, "recurrent.onnx"),
            input_names=["latent_in", "action"], output_names=outs,
            dynamic_axes={k: v for k, v in dyn.items() if k in ["latent_in", "action", *outs]},
            opset_version=opset, dynamo=False,
        )
    net.train(was_training)
