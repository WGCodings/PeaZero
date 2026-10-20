"""Fast GPU driver for tree-search self-play (SearchSelfPlay pools).

One self-play step = one simulation for every game: Rust encodes the
positions to evaluate (`requests`), the network evaluates them, Rust feeds
the results back into the searches (`respond`). The plain loop
(`train_lib.search_step`) does these strictly one after another and pays
for a fresh allocation, a pageable host→GPU copy and ~50 separate kernel
launches every step. `SearchRunner` removes those costs:

* Pinned buffers: Rust writes the requests straight into page-locked
  host memory (`requests_into`) and the results come back into
  page-locked memory, so the copies are asynchronous DMA transfers and
  nothing is allocated per step.
* CUDA graphs (`cuda_graphs=True`): the whole forward pass + the gather
  of the legal-move logits is recorded once per network and replayed
  with a single launch. The batch is always the pool size (`requests`
  returns every game every step), so the shapes never change.
* Pipelining (`len(pools) == 2`): the games are split over two pools.
  While the GPU evaluates pool A, the Rust threads process pool B's
  results and prepare B's next requests, and the other way round, so
  GPU and CPU work overlap instead of taking turns.

Results are the same as the plain loop: the same network evaluates the
same positions (CUDA graphs replay the same kernels). With two pools the
games are split differently, so the individual games differ, not their
statistics.

`timing` accumulates where the self-play time goes (seconds):
  gpu_wait  blocked waiting for the GPU (GPU slower than the CPU side)
  rust      requests + respond in Rust (the tree searches)
  launch    copies + kernel launches issued from Python
"""
from __future__ import annotations

import copy
import time
import warnings
from types import SimpleNamespace

import torch

from . import encoding as enc

MAX_LEGAL = 218  # crates/selfplay/src/replay.rs


def merge_stats(stats: list[dict]) -> dict:
    """Sum the stats dicts of several pools (nested dicts too)."""
    out: dict = {}
    for s in stats:
        for k, v in s.items():
            if isinstance(v, dict):
                out[k] = merge_stats([out.get(k, {}), v])
            else:
                out[k] = out.get(k, 0) + v
    return out


class SearchRunner:
    def __init__(self, pools, device: torch.device, amp: bool, cuda_graphs: bool = True,
                 channels_last: bool = False):
        self.pools = list(pools)
        self.dev = device
        self.amp = amp and device.type == "cuda"
        self.cuda = device.type == "cuda"
        self.graphs_wanted = cuda_graphs and self.cuda
        # NHWC layout: lets cuDNN pick its tensor-core convolution kernels
        self.channels_last = channels_last and self.cuda
        self.graphs = False  # True once capture succeeded
        self.net = None
        self.slots = [self._slot(p) for p in self.pools]
        self.started = False
        self.timing = {"gpu_wait": 0.0, "rust": 0.0, "launch": 0.0, "steps": 0}

    # ------------------------------------------------------------ setup
    def _slot(self, pool):
        n = pool.num_games
        pin = self.cuda
        s = SimpleNamespace(pool=pool, n=n, graph=None, out=None)
        s.obs_h = torch.empty((n, enc.OBS_PLANES, 8, 8), dtype=torch.float32, pin_memory=pin)
        s.legal_h = torch.empty((n, MAX_LEGAL), dtype=torch.int32, pin_memory=pin)
        s.logits_h = torch.empty((n, MAX_LEGAL), dtype=torch.float32, pin_memory=pin)
        s.value_h = torch.empty((n, 3), dtype=torch.float32, pin_memory=pin)
        s.obs_np, s.legal_np = s.obs_h.numpy(), s.legal_h.numpy()
        s.logits_np, s.value_np = s.logits_h.numpy(), s.value_h.numpy()
        if self.cuda:
            s.obs_d = torch.empty((n, enc.OBS_PLANES, 8, 8), dtype=torch.float32, device=self.dev)
            s.legal_d = torch.empty((n, MAX_LEGAL), dtype=torch.int32, device=self.dev)
            s.event = torch.cuda.Event()
        return s

    def _forward(self, s, obs):
        """Network + legal-move gather; returns (logits (n,218), value (n,3)) float32."""
        if self.amp:
            # no weight-cast cache: inside a CUDA graph the casts must be recorded
            ctx = torch.autocast("cuda", dtype=torch.float16, cache_enabled=False)
        else:
            ctx = torch.autocast(self.dev.type, enabled=False)
        if self.channels_last:
            obs = obs.contiguous(memory_format=torch.channels_last)
        with ctx:
            _, policy, value = self.net.initial_inference(obs)
        legal = (s.legal_d if self.cuda else s.legal_h).long()
        return policy.float().gather(1, legal), value.float()

    def set_net(self, net) -> None:
        """Use `net` (eval mode) from the next launch on; records the CUDA
        graphs for it. Call between generations, after drain()."""
        self.drain()
        if self.cuda:
            torch.cuda.synchronize()
        for s in self.slots:
            s.graph, s.out = None, None
        # channels_last: a private copy, so the caller's network (export,
        # eval_metrics) keeps its normal layout
        self.net = copy.deepcopy(net).to(memory_format=torch.channels_last) if self.channels_last else net
        self.graphs = False
        if not self.graphs_wanted:
            return
        try:
            for s in self.slots:
                self._capture(s)
            self.graphs = True
        except Exception as e:  # e.g. an unsupported driver: fall back to eager
            warnings.warn(f"CUDA graph capture failed ({e}); using eager inference")
            for s in self.slots:
                s.graph, s.out = None, None
            self.graphs_wanted = False
            torch.cuda.synchronize()

    def _capture(self, s) -> None:
        s.obs_d.zero_()
        s.legal_d.zero_()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):  # warm-up: cuDNN algorithm choice, allocator
                self._forward(s, s.obs_d)
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(g):
            s.out = self._forward(s, s.obs_d)
        s.graph = g

    # ------------------------------------------------------------ stepping
    def _request(self, s) -> None:
        t = time.perf_counter()
        s.pool.requests_into(s.obs_np, s.legal_np)
        self.timing["rust"] += time.perf_counter() - t

    def _launch(self, s) -> None:
        t = time.perf_counter()
        if self.cuda:
            s.obs_d.copy_(s.obs_h, non_blocking=True)
            s.legal_d.copy_(s.legal_h, non_blocking=True)
            if s.graph is not None:
                s.graph.replay()
                logits, value = s.out
            else:
                with torch.no_grad():
                    logits, value = self._forward(s, s.obs_d)
            s.logits_h.copy_(logits, non_blocking=True)
            s.value_h.copy_(value, non_blocking=True)
            s.event.record()
        else:
            with torch.no_grad():
                logits, value = self._forward(s, s.obs_h)
            s.logits_h.copy_(logits)
            s.value_h.copy_(value)
        self.timing["launch"] += time.perf_counter() - t

    def _finish(self, s) -> int:
        t = time.perf_counter()
        if self.cuda:
            s.event.synchronize()
        t2 = time.perf_counter()
        done = s.pool.respond(s.logits_np, s.value_np)
        self.timing["gpu_wait"] += t2 - t
        self.timing["rust"] += time.perf_counter() - t2
        return done

    def start(self) -> None:
        if not self.started:
            for s in self.slots:
                self._request(s)
                self._launch(s)
            self.started = True

    def step(self) -> int:
        """One simulation for every game of every pool. Returns games finished."""
        self.start()
        done = 0
        for s in self.slots:
            done += self._finish(s)  # meanwhile the GPU works on the other pool
            self._request(s)
            self._launch(s)
        self.timing["steps"] += 1
        return done

    def drain(self) -> int:
        """Finish the evaluations still in flight (end of a self-play phase)."""
        done = 0
        if self.started:
            for s in self.slots:
                done += self._finish(s)
            self.started = False
        return done

    # ------------------------------------------------------------ pool access
    def stats(self) -> dict:
        return merge_stats([p.stats() for p in self.pools])

    def take_chunks(self):
        """[(chunk bytes, games, training positions, plies)] of every pool with finished games."""
        return [c for c in (p.take_chunk() for p in self.pools) if c[1]]

    @property
    def num_threads(self) -> int:
        return self.pools[0].num_threads

    def reset_timing(self) -> dict:
        t, self.timing = self.timing, {"gpu_wait": 0.0, "rust": 0.0, "launch": 0.0, "steps": 0}
        return t
