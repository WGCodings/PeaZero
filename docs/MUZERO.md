# PeaZero — MuZero design notes (Sprint 2)

This is the reference for how PeaZero's MuZero works and why. Sprint 3 implements the search from §2 and the self-play generator; Sprint 4 implements the trainer from §3.

---

## 1. MuZero in one page (chess edition)

Three networks, all trained together:

| net | input → output | role |
|---|---|---|
| representation **h** | observation of the real position → latent state `s₀` (C×8×8) | "understand the board" |
| dynamics **g** | (`sₖ`, action) → `sₖ₊₁` | "imagine the position after a move". Never sees the rules. |
| prediction **f** | `sₖ` → policy logits (4672) + value (W/D/L) | "what's good here, who's winning" |

**Search** (MCTS) runs entirely in latent space: the root is `h(real position)`, and every expansion calls `g` then `f`. Chess rules are used in exactly two places:
1. legal-move masking at the **root** only (deeper nodes may pick any of the 4672 actions; the policy learns to give illegal ones ~0 prior);
2. deciding when the real game is over.

**Self-play** plays games with search and stores, for every move: the position, the move played, and the search's improved policy. At game end it stores the result.

**Training** samples a position from a game, runs `h` on it, then unrolls `g` K=5 times using the moves that were *actually played*. At each step `f`'s policy is compared with the search policy recorded at that real future position, and its value with the game result. The gradients flow through the whole unroll, which is what teaches `g` to imagine positions usefully.

## 2. Search: Gumbel MuZero

We use **Gumbel MuZero** (Danihelka et al., ICLR 2022), ported from DeepMind's reference implementation `mctx` (`policies.py`, `action_selection.py`, `qtransforms.py`, `seq_halving.py`). Its key property: it guarantees policy improvement even with very few simulations (16–32), where classic pUCT MuZero needs hundreds. That makes self-play affordable on one GPU.

### 2.1 Root: Gumbel-Top-k + Sequential Halving

1. Mask illegal moves (logit = −∞).
2. Draw Gumbel noise `g(a)` per action (`gumbel_scale = 1` in self-play; `0` when playing on OpenBench → deterministic).
3. Consider the top `m = min(16, #legal)` actions by `g(a) + logits(a)`.
4. Spend the `n` simulations with **Sequential Halving**: visit each considered action equally; halve the set, keeping the best by `g(a) + logits(a) + σ(q̂(a))`; repeat until the budget is used. mctx precomputes this as a table of "which visit count is allowed next" (`get_sequence_of_considered_visits`); the root picks the argmax of `g + logits + σ(q̂)` among actions whose visit count equals the currently allowed one.
5. **Move to play** = argmax over the considered actions with the maximal visit count of `g(a) + logits(a) + σ(q̂(a))`.
6. **Policy training target** = `softmax(logits + σ(completedQ))` over legal actions (mctx `action_weights`). This replaces visit-count targets.

### 2.2 Interior nodes: deterministic selection

`π' = softmax(logits + σ(completedQ))`, then pick `argmax_a [ π'(a) − N(a) / (1 + Σ_b N(b)) ]`. This makes visit frequencies track `π'` without any noise.

### 2.3 Completed Q and σ (`qtransform_completed_by_mix_value`)

* Unvisited actions get the **mixed value**: `v_mix = (v̂ + Σ_b N(b) · Σ_{visited} π(a)q(a) / Σ_{visited} π(a)) / (1 + Σ_b N(b))`, where `v̂` is the node's raw network value.
* Min-max rescale the completed Q-values of the node to [0, 1].
* `σ(q) = (c_visit + max_b N(b)) · c_scale · q` with **c_visit = 50, c_scale = 0.1** (mctx defaults).

### 2.4 Two-player bookkeeping

Values are always from the side to move's view. Backing up a value from child to parent negates it (mctx uses `discount = −1` for board games). The scalar value of a W/D/L head is `P(win) − P(loss)`.

### 2.5 Engine (OpenBench) use

Sequential Halving needs the simulation budget `n` up front. In UCI play, `n` = measured simulations/s × soft time budget (`timeman.rs`), with `gumbel_scale = 0`. `go nodes N` sets `n = N` directly. `bench` becomes a fixed-`n` search over the bench positions (deterministic, since tract inference is deterministic single-threaded).

## 3. Training

### 3.1 Targets per sampled position `t`, unroll steps `k = 0..K` (K = 5)

| head | target | notes |
|---|---|---|
| policy at step k | Gumbel `action_weights` recorded at real ply `t+k` | masked if `t+k` is at or past the end of the game |
| value at step k | final result from the side to move's view at ply `t+k`, as W/D/L class | at the terminal ply itself (`t+k = T`): the result (e.g. "loss" for the mated side); beyond `T`: masked |

**No reward head.** In chess every non-terminal reward is 0, and the board-game MuZero setup uses Monte-Carlo targets (`td_steps` = game length, discount 1), so the value target is simply the game result. Adding a reward head would only learn "0".

### 3.2 Loss and tricks (from the MuZero paper, mirrored in LightZero)

* `L = Σ_k w_k · [CE(policy_k, π_target) + CE(value_k, z)] + λ‖θ‖²`, with `w_0 = 1`, `w_k = 1/K` for k ≥ 1.
* **Gradient scaling**: the gradient entering the dynamics function is scaled by ½ at every unroll step (`s = 0.5·s + 0.5·s.detach()`).
* **Latent normalisation**: min-max scale each latent state to [0, 1] after `h` and after every `g` (in `model.py`, so it's also in the exported ONNX).
* BatchNorm: always export with the module in eval mode (Sprint 2 hit exactly this bug: `torch.onnx.export` restored the wrapper's training mode, silently corrupting the reference outputs).

### 3.3 Starting hyperparameters (to be tuned)

| | value |
|---|---|
| network | 64 channels, 6 + 6 residual blocks (≈1.0M params) |
| self-play | n = 32 simulations, m = 16 considered, 256–512 games in parallel |
| game length | adjudicate a draw at 512 plies; no resignation at first |
| batch | 512 positions, K = 5 |
| optimiser | AdamW, lr 1e-3 with cosine decay per generation, weight decay 1e-4 |
| replay buffer | last ~1M positions |
| sample reuse | each position sampled ~4× on average |
| new network | export every ~1000 training steps; self-play picks it up |

## 4. Encodings (Sprint 2, `crates/core/src/encoding.rs`)

All side-to-move relative (Black's board mirrored with `sq ^ 56`).

* **Actions**: AlphaZero 8×8×73 = 4672, `index = plane·64 + from`. Planes: 56 queen-like (8 directions × 7 distances), 8 knight, 9 under-promotions. Queen promotion = the pawn's queen-like move; castling = king's 2-square move. Matches a 73-channel conv policy head.
* **Observation**: 21 planes: own/opponent pieces (12), repetition ≥1 / ≥2, four castling rights, en-passant square, halfmove clock / 100, all-ones. No move history in v0 (MuZero's dynamics can't see history inside the tree anyway); revisit later.
* **Action planes** for `g`: from-square, to-square (empty if off-board), 3 under-promotion flags = 5 planes.
* Python mirror: `trainer/peazero/encoding.py`, tested against Rust-generated `tests/encoding/vectors.jsonl` (486 checks). A Rust test fails if the vectors file is stale. **Changing the encoding = new `ENCODING_VERSION` = all old networks are invalid.**

## 5. Inference spike results

### 5.1 CPU (tract, what OpenBench workers run)

Single thread, Intel Xeon @ 2.8 GHz (AVX2), recurrent (g + f) evals/s:

| net | params | tract b=1 | tract b=8 | onnxruntime b=1 | onnxruntime b=8 |
|---|---|---|---|---|---|
| 32×4 | 0.24M | 1,958 | 1,268 | 4,388 | 6,327 |
| 64×6 | 1.01M | 610 | 419 | 1,223 | 1,679 |
| 128×10 | 6.16M | 102 | 75 | 232 | 281 |

* tract matches PyTorch to ~1e-7.
* tract is ~2× slower than onnxruntime at batch 1, and gets *slower* per eval with batching. So the engine searches with batch 1, and a hand-written SIMD forward pass is a clear later win (~2× from ORT parity alone).
* At OpenBench STC (8+0.08, roughly 0.2–0.3 s per move): 64×6 gives ~150 simulations/move, 32×4 ~500. Gumbel search is designed to be useful in exactly that range.

### 5.2 GPU (RTX 3070)

Run `python trainer/scripts/bench_gpu.py` on the 3070 and fill in:

| net | infer b=256 fp16 | infer b=1024 fp16 | ort-cuda b=1024 | train pos/s (b=512, K=5) |
|---|---|---|---|---|
| 32×4 | | | | |
| 64×6 | | | | |
| 128×10 | | | | |

## 6. Decisions

| topic | decision | why |
|---|---|---|
| self-play & training | local, RTX 3070 | simplest loop; OpenBench workers are CPU-only |
| trainer | own small PyTorch trainer; losses and tricks ported from LightZero's `gumbel_muzero` + the MuZero paper | frameworks are general-purpose and heavy; our chess-specific encodings/data format would fight them |
| search | Rust port of mctx Gumbel MuZero | one implementation for self-play and the engine |
| CPU inference (engine, OpenBench) | `tract` (pure Rust) | builds with plain `make` on any worker; SIMD forward pass later |
| GPU inference (self-play only) | ONNX Runtime CUDA via the `ort` crate, behind a Cargo feature | OpenBench never builds it |
| network format | two ONNX graphs (`initial.onnx`, `recurrent.onnx`) | exactly the two calls search makes |
| first network | 64×6 (fallback 32×4 if self-play is too slow) | spike numbers above |
| value head | W/D/L classes | chess draws are a real outcome; also gives UCI WDL output for free |

### 6.1 Self-play data format v0 (implemented in Sprint 3)

Binary chunk files, little-endian, written by Rust, read by Python with numpy:

```
file header : magic "PZSP" | u32 format version | u32 ENCODING_VERSION | u32 game count
per game    : u16 ply count | i8 result (+1 white win, 0 draw, −1 black win) | u8 termination
per ply     : 12 × u64 piece bitboards (absolute, P N B R Q K white then black)
              u8 side to move | u8 castling | u8 ep square (64 = none) | u8 halfmove clock
              u8 repetitions | u16 action played
              u8 legal count L | L × (u16 action, f16 target probability)
```

≈ 250 bytes per position; the trainer turns the compact board into observation planes itself (vectorised numpy), checked against the same Rust test vectors.

## 7. Reference implementations reviewed

| project | what we take | what we don't |
|---|---|---|
| **mctx** (DeepMind, JAX) | the exact Gumbel root/interior selection, completed-Q transform, sequential-halving table | JAX batching machinery |
| **LightZero** (OpenDILab, PyTorch) | MuZero model structure, Gumbel MuZero training loss, latent normalisation, gradient scaling | the framework itself. Its chess env only ships AlphaZero configs, so there is no MuZero-chess setup to reuse anyway |
| **MuZero paper + pseudocode** | targets, unroll, board-game settings | 800-simulation pUCT search |
| **muzero-general** | readable sanity reference | unmaintained, slow |

## 8. Known pitfalls to test for

* **Perspective/sign errors** in value backup and targets. Unit-test with mate-in-1 positions and a hand-made toy network.
* **Train/eval BatchNorm mismatch.** `nnbench` checks tract against a PyTorch reference on every export.
* **Encoding drift** between Rust and Python. Guarded by the vectors test on both sides.
* **Draw-dominated early games** (random play ends by the 50-move rule or insufficient material). Expected; the value head learns "draw" first.
* **Policy collapse onto illegal actions deep in the tree.** Monitor the share of root prior mass on illegal moves during training.

## Sources

* Schrittwieser et al., *Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model* (MuZero), arXiv:1911.08265
* Danihelka et al., *Policy improvement by planning with Gumbel*, ICLR 2022 — https://openreview.net/forum?id=bERaNdoegnO
* mctx — https://github.com/google-deepmind/mctx
* LightZero — https://github.com/opendilab/LightZero
* tract — https://github.com/sonos/tract
