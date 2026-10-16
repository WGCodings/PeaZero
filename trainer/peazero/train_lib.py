"""Self-play driver and MuZero loss (docs/MUZERO.md §2-3).

Rust (`peazero_rs`) owns the games, Gumbel selection, data format and batch
sampling; this file only runs the network.
"""
from __future__ import annotations

import copy
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
def selfplay_step(net: MuZeroNet, sp, device: torch.device, amp: bool, rec_net: MuZeroNet | None = None) -> None:
    """Play one move in every game of the pool (root-only Gumbel).

    `net` runs the root (initial inference), `rec_net` the children
    (recurrent inference); see calibrate_bn for why they can differ."""
    rec_net = net if rec_net is None else rec_net
    obs = torch.from_numpy(sp.observations()).to(device, non_blocking=True)
    with autocast(device, amp):
        latent, policy, value = net.initial_inference(obs)
    actions = sp.expand(policy.float().cpu().numpy())
    acts = torch.from_numpy(actions).to(device, non_blocking=True)
    latent_rep = latent.repeat_interleave(sp.considered, dim=0)
    with autocast(device, amp):
        _, _, child_value = rec_net.recurrent_inference(latent_rep, acts)
    sp.finish(
        np.ascontiguousarray(value.float().cpu().numpy()),
        np.ascontiguousarray(child_value.float().cpu().numpy()),
    )


def batch_to_device(batch: dict, device: torch.device) -> dict:
    """numpy batch → tensors on `device`. On CUDA the copy goes through pinned
    memory and is asynchronous, so it overlaps with work already queued."""
    out = {}
    for k, v in batch.items():
        t = torch.from_numpy(v)
        if device.type == "cuda":
            t = t.pin_memory()
        out[k] = t.to(device, non_blocking=True)
    return out


def muzero_loss(net: MuZeroNet, batch: dict, device: torch.device, amp: bool):
    """K-step unrolled loss. `batch` is a numpy batch from Replay.sample or
    the output of batch_to_device. Returns (loss tensor, metrics dict of
    detached 0-dim tensors; call float() on them only when needed, every
    float() waits for the GPU)."""
    t = batch if isinstance(next(iter(batch.values())), torch.Tensor) else batch_to_device(batch, device)
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
        "loss": loss.detach().float(),
        "policy_loss": pl0.detach().float(),
        "value_loss": vl0.detach().float(),
        "policy_loss_unrolled": pl_u.detach().float(),
        "value_loss_unrolled": vl_u.detach().float(),
        "policy_agreement": agree0.detach().float(),
    }
    return loss, metrics


def _reset_bn(modules) -> None:
    for m in modules:
        if isinstance(m, torch.nn.BatchNorm2d):
            m.reset_running_stats()
            m.momentum = None  # cumulative average over the calibration batches


@torch.no_grad()
def calibrate_bn(net: MuZeroNet, replay, device: torch.device, amp: bool,
                 batches: int = 16, batch_size: int = 1024):
    """Eval-mode copies of `net` with BatchNorm statistics recomputed for
    how each exported graph is actually used: (init_net, rec_net).

    During training every call (root, unroll step 1..K) normalises with its
    own batch statistics, but the running averages mix all of them: the
    prediction head is shared by root latents and dynamics latents, whose
    distributions differ. In eval mode (self-play, engine) that mix fits
    neither, which badly corrupted the root value and the children's Q.

    init_net: representation + prediction statistics from root positions.
    rec_net:  dynamics + prediction statistics from first-step children
              (the only recurrent call self-play makes).
    """
    if replay.positions == 0:
        return copy.deepcopy(net).eval(), copy.deepcopy(net).eval()
    samples = [replay.sample(batch_size, 1) for _ in range(batches)]
    tensors = [(torch.from_numpy(b["obs"]).to(device), torch.from_numpy(b["actions"][:, 0]).to(device))
               for b in samples]

    init_net = copy.deepcopy(net)
    _reset_bn(init_net.modules())
    init_net.train()
    with autocast(device, amp):
        for obs, _ in tensors:
            init_net.initial_inference(obs)
    init_net.eval()

    rec_net = copy.deepcopy(init_net)
    _reset_bn(list(rec_net.dynamics.modules()) + list(rec_net.prediction.modules()))
    rec_net.dynamics.train()
    rec_net.prediction.train()
    with autocast(device, amp):
        for obs, act in tensors:
            latent = rec_net.representation(obs)  # eval mode: root statistics
            rec_net.recurrent_inference(latent, act)
    rec_net.eval()
    return init_net, rec_net


@torch.no_grad()
def eval_metrics(init_net: MuZeroNet, rec_net: MuZeroNet, replay, device: torch.device, amp: bool,
                 batches: int = 4, batch_size: int = 1024) -> dict:
    """Losses of the networks exactly as exported (eval mode): root and the
    first dynamics step. Should be close to the training losses."""
    out = {"eval_policy_loss": 0.0, "eval_value_loss": 0.0,
           "eval_policy_loss_k1": 0.0, "eval_value_loss_k1": 0.0}
    if replay.positions == 0:
        return {}
    for _ in range(batches):
        b = replay.sample(batch_size, 1)
        t = {k: torch.from_numpy(v).to(device) for k, v in b.items()}
        with autocast(device, amp):
            latent, p0, v0 = init_net.initial_inference(t["obs"])
            _, p1, v1 = rec_net.recurrent_inference(latent, t["actions"][:, 0])
        for k, (p, v, pk, vk) in enumerate(((p0, v0, "eval_policy_loss", "eval_value_loss"),
                                             (p1, v1, "eval_policy_loss_k1", "eval_value_loss_k1"))):
            logp = F.log_softmax(p.float(), dim=-1)
            target = torch.zeros_like(logp).scatter_add_(1, t["policy_index"][:, k], t["policy_prob"][:, k])
            pm, vm = t["policy_mask"][:, k], t["value_mask"][:, k]
            out[pk] += float((-(target * logp).sum(-1) * pm).sum() / pm.sum().clamp(min=1)) / batches
            vl = F.cross_entropy(v.float(), t["value_target"][:, k], reduction="none")
            out[vk] += float((vl * vm).sum() / vm.sum().clamp(min=1)) / batches
    return out
