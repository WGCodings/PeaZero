"""Self-play driver and MuZero loss (docs/MUZERO.md §2-3).

Rust (`peazero_rs`) owns the games, Gumbel selection, data format and batch
sampling; this file only runs the network.
"""
from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from .model import MuZeroNet


def autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return nullcontext()


@torch.inference_mode()
def selfplay_step(net: MuZeroNet, sp, device: torch.device, amp: bool) -> None:
    """Play one move in every game of the pool (root-only Gumbel)."""
    obs = torch.from_numpy(sp.observations()).to(device, non_blocking=True)
    with autocast(device, amp):
        latent, policy, value = net.initial_inference(obs)
    actions = sp.expand(policy.float().cpu().numpy())
    acts = torch.from_numpy(actions).to(device, non_blocking=True)
    latent_rep = latent.repeat_interleave(sp.considered, dim=0)
    with autocast(device, amp):
        _, _, child_value = net.recurrent_inference(latent_rep, acts)
    sp.finish(
        np.ascontiguousarray(value.float().cpu().numpy()),
        np.ascontiguousarray(child_value.float().cpu().numpy()),
    )


def muzero_loss(net: MuZeroNet, batch: dict, device: torch.device, amp: bool):
    """K-step unrolled loss. Returns (loss tensor, metrics dict of floats)."""
    t = {k: torch.from_numpy(v).to(device, non_blocking=True) for k, v in batch.items()}
    obs, actions = t["obs"], t["actions"]
    pidx, pprob, pmask = t["policy_index"], t["policy_prob"], t["policy_mask"]
    vtgt, vmask = t["value_target"], t["value_mask"]
    b, unroll = actions.shape[0], actions.shape[1]

    def step(k: int, policy: torch.Tensor, value: torch.Tensor):
        logp = F.log_softmax(policy.float(), dim=-1)
        target = torch.zeros_like(logp).scatter_add_(1, pidx[:, k], pprob[:, k])
        pl = (-(target * logp).sum(-1) * pmask[:, k]).sum() / b
        vl = (F.cross_entropy(value.float(), vtgt[:, k], reduction="none") * vmask[:, k]).sum() / b
        agree = ((policy.argmax(-1) == target.argmax(-1)).float() * pmask[:, k]).sum() / pmask[:, k].sum().clamp(min=1)
        return pl, vl, agree

    with autocast(device, amp):
        latent, policy, value = net.initial_inference(obs)
        pl0, vl0, agree0 = step(0, policy, value)
        loss = pl0 + vl0
        pl_u = vl_u = torch.zeros((), device=device)
        for k in range(1, unroll + 1):
            latent = 0.5 * latent + 0.5 * latent.detach()  # MuZero gradient scaling
            latent, policy, value = net.recurrent_inference(latent, actions[:, k - 1])
            pl, vl, _ = step(k, policy, value)
            pl_u = pl_u + pl / unroll
            vl_u = vl_u + vl / unroll
        loss = loss + pl_u + vl_u

    metrics = {
        "loss": float(loss.detach()),
        "policy_loss": float(pl0.detach()),
        "value_loss": float(vl0.detach()),
        "policy_loss_unrolled": float(pl_u.detach()),
        "value_loss_unrolled": float(vl_u.detach()),
        "policy_agreement": float(agree0.detach()),
    }
    return loss, metrics
