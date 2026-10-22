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

Sequential Halving needs the simulation budget `n` up front. In UCI play, `n` = measured simulations/s × soft time budget (`timeman.rs`), with `gumbel_scale = 0`. `go nodes N` sets `n = N` directly. `bench` becomes a fixed-`n` search over the bench positions (deterministic: inference is single-threaded and identical across kernel sets, §5.1b).

**Search modes in the engine (implemented).** `engine/src/search/` holds one file per method behind the `NetSearch` trait; `PolicyEngine` (policy.rs) owns the network and switches with the UCI option `SearchMode` (combo):

| mode | what it does |
|---|---|
| `Mcts` (default) | plain AlphaZero-style MCTS: PUCT (`c_puct` 1.5, unvisited Q = 0) with the policy softmax as priors, each new leaf evaluated once with the network (W/D/L backed up, sides flipped per ply), exact mate/stalemate/draw values, most-visited move played. Score/`wdl` = mean W/D/L of the chosen move; `depth` = average playout depth. `go nodes N` = N playouts; `go depth N` searches until the average playout depth reaches N (cap 100,000 playouts, e.g. in forced mates); plain `go` without limits = 400 playouts. Info lines whenever depth or seldepth grows, at least once a second. No tree reuse, virtual loss, batching, noise or temperature. ~1,200 (AVX2) to ~1,700 (AVX-512) playouts/s per core with 64×6 |
| `AlphaBeta` | plain iterative-deepening negamax with alpha-beta pruning; moves ordered by the policy, network value at the leaves (`q_to_cp(W−L)`, clamped ±3000); exact mate/stalemate/draw detection. Nothing else (no TT, no cache, no extensions, no quiescence): every node is one network evaluation, ~1.2–1.8k nodes/s per core with 64×6, enough for short mates (Morphy's mate in 2 at depth 3 in <1 s). `wdl` in the info line = network W/D/L at the last PV position (exact for mate/draw), from the root side's view. Plain `go` without limits searches depth 3 |
| `Searchless` | one evaluation, best legal policy move |

A new mode is a new file in `search/`, a variant in `SearchMode` and an arm in `SearchMode::create`. `bench` runs `go depth 4` with `Mcts` (capped at 2,000 playouts per position) on the 9 bench positions and prints total playouts and overall nps. The count is deterministic but depends on the network (MCTS depth = average playout depth, which grows faster with a sharper policy): 17686 with the untrained `networks/default.pzn`, 4,000–7,000 with trained 64×6 networks (5–10 s). A new embedded network means a new bench number on OpenBench. On OpenBench, `option.SearchMode=AlphaBeta` or `option.SearchMode=Searchless` selects the other modes in games.

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

### 5.1b Hand-written SIMD inference (`crates/nn/src/fast.rs`, `winograd.rs`; what the engine uses now)

The engine evaluates the `initial` network (h + f) with its own kernels on x86-64; tract remains for the dynamics graph, the tests and non-x86 CPUs. Weights are read straight from the ONNX graph in the `.pzn` (minimal protobuf reader, `onnx.rs`; BatchNorm is already folded by the export), so every existing network works unchanged.

* Activations square-major on a zero-padded 10×10 board (no edge checks). Kernel sets chosen at runtime, so one binary runs on every x86-64 CPU: AVX-512, AVX2/FMA/F16C, SSE2 floor. `PEAZERO_SIMD=sse2|avx2` forces a lower set, `PEAZERO_WINOGRAD=0` the direct convolutions (testing).
* **Residual blocks: Winograd F(2×2, 3×3)**, as in Lc0's CPU backend: 16 tiles of 2×2 outputs, 2.25× fewer multiply-adds. Input/output transforms are SIMD adds over channels; the 16 per-position matrix products use register-blocked kernels (AVX2 6 tiles × 16 channels, AVX-512 8 × 32). F(2×2) rather than F(4×4) because the larger tiles need 4× the weight data, which makes batch-1 inference memory-bound.
* **Winograd weights stored as half floats** (widened with F16C while loading; arithmetic stays f32): 12 layers of 64×6 take ~1.5 MB instead of ~3 MB and stay in L2. Without this, a layer's products took 31 µs instead of 18 µs once all 12 layers ran in turn. Effect on the output vs exact f32 (2,000 positions from random games, gen 325 of `fourth`): same best move 99.95%, move probabilities ±0.26 percentage points at most, value (W−L) ±0.001 at most.
* Stem (21 input planes) and the 1×1 heads use direct convolutions (AVX-512 8 squares × 32 channels, AVX2 6 × 16, SSE2 3 × 16).
* AVX2 and AVX-512 are bit-identical (same operations per lane in the same order; our f32→f16 conversion is tested to round exactly like F16C), and the bench node count is the same with every kernel set and as with tract (17686 with `networks/default.pzn`).
* Batch 1, 64×6, best of 5 (Xeon @ 2.1 GHz, shared VM): tract ~950–1,150 µs/eval; direct AVX2 ~730 µs, direct AVX-512 ~480–570 µs; **Winograd AVX2 ~400–520 µs, Winograd AVX-512 ~400 µs (2.3–2.8× tract)**; SSE2 ~1,250 µs. Engine `bench`: 962 nps (tract) → 1,737 (AVX2) / 2,198 (AVX-512).
* The engine binary no longer contains tract on x86-64: builds take seconds instead of minutes, need no C compiler, and the binary is 5 MB instead of 34 MB.
* `cargo run --release -p peazero-nn --example latency -- net.pzn` compares every kernel set (direct and Winograd) with tract; `--example accuracy` measures Winograd vs exact f32 on 2,000 positions.

### 5.1c Int8 residual blocks (`crates/nn/src/int8.rs`)

When the CPU has AVX2, the residual-block convolutions run in int8 (float otherwise; `PEAZERO_INT8=0` forces float, `=avx2` / `=avxvnni` caps the kernel, for testing). Stem, heads, bias, residual and ReLU stay f32.

* Weights int8 with one scale per output channel; activations (ReLU outputs) quantised per conv input and per evaluation to 0..127 (7 bits, scale = max / 127). `y = s_x · s_w[co] · Σ q_x q_w + bias`.
* Kernels: AVX2 `vpmaddubsw` + `vpmaddwd` (7-bit activations: a pair sum ≤ 2·127·127 never saturates the i16 step), AVX-VNNI `vpdpbusd` (Intel 12th gen+), AVX-512 VNNI `vpdpbusd` on zmm (Zen 4, Intel servers). All three give the same integers and the same f32 epilogue → bit-identical; bench 17686 with every kernel.
* Accuracy vs exact f32 (2,000 random-game positions, gen 325 of `fourth`): same best move 96.4%, move probabilities up to 18 percentage points off, value (W−L) 0.011 mean / 0.083 max. Tried and not better: 7-bit weights (worse), clipping activation outliers (much worse), per-channel calibrated activation scales (no gain), any single layer in float (no gain: the error is spread over all 12). 8-bit activations gave 97.35% but can saturate the AVX2 kernel. The fix is quantisation-aware training: `train.py --qat` (docs/TRAINING.md) simulates exactly this rounding in training.
* Speed, batch 1, 64×6 (Xeon @ 2.1 GHz): AVX2 int8 ~455 µs vs AVX2 Winograd ~535 µs; AVX-VNNI ~260 µs; AVX-512 VNNI ~170 µs (vs AVX-512 Winograd ~400 µs). Engine `bench`: AVX2 int8 ~1,900 nps vs AVX2 Winograd ~1,650; AVX-512 VNNI ~5,700 nps.

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
| CPU inference (engine, OpenBench) | hand-written SIMD (`fast.rs`, x86-64), tract elsewhere | §5.1b; tract stays as reference and for the dynamics graph |
| GPU inference (self-play only) | ONNX Runtime CUDA via the `ort` crate, behind a Cargo feature | OpenBench never builds it |
| network format | two ONNX graphs (`initial.onnx`, `recurrent.onnx`) | exactly the two calls search makes |
| first network | 64×6 (fallback 32×4 if self-play is too slow) | spike numbers above |
| value head | W/D/L classes | chess draws are a real outcome; also gives UCI WDL output for free |

### 6.1 Self-play data format v0/v1 (implemented in Sprint 3; header version 1, superseded by v2, §6.4)

Binary chunk files, little-endian, written by Rust, read by Python with numpy:

```
file header : magic "PZSP" | u32 format version | u32 ENCODING_VERSION | u32 game count
per game    : u16 ply count | i8 result (+1 white win, 0 draw, −1 black win) | u8 termination
per ply     : 12 × u64 piece bitboards (absolute, P N B R Q K white then black)
              u8 side to move | u8 castling | u8 ep square (64 = none) | u8 halfmove clock
              u8 repetitions | u16 action played
              u8 legal count L | L × (u16 action, u16 target probability × 65535)
```

≈ 215 bytes per position (measured). Implemented in `crates/selfplay/src/record.rs`. The trainer never decodes it in Python: the Rust replay buffer (`peazero_rs.Replay`) samples batches and encodes observations with the engine's own encoder.

## 6.2 Sprint 4: root-only Gumbel self-play (engine stays searchless)

Per self-play move: one `initial` call on the root, Gumbel-Top-k picks `m = 16` legal moves, each is evaluated **once** with `recurrent` (dynamics + prediction), then §2.1 steps 5–6 pick the move and produce the policy target. With one visit per considered action, Sequential Halving is a single round (mctx with `n == m`). That's 17 network calls per move, so a 1024-game pool means batches of 1024 and 16,384.

Deviation from mctx: the min-max rescale of completed Q uses an epsilon of 1e-4 instead of 1e-8, and the mixed value is summed in f64. Otherwise f32 rounding noise between equal Q-values (~1e-8) gets blown up into a full-scale preference.

Architecture: Rust (`crates/selfplay`, exposed to Python as `peazero_rs`) owns the games, Gumbel selection, data files and batch sampling; Python only runs the network in PyTorch (fp16 on CUDA was ~2.5× faster than onnxruntime-CUDA in the spike).

## 6.3 Sprint 5: Gumbel tree search on the real board (self-play only)

Root-only Gumbel (§6.2) looks one ply deep and takes its child values from the dynamics network, whose values correlated only ~0.3 with the network's own evaluation of the real child positions. That made the policy targets barely better than the prior, so learning was slow (after ~50M positions a queen was worth only +0.13 W−L to the value head).

`crates/selfplay/src/search.rs` now runs the full mctx `gumbel_muzero_policy` per move, with `n` simulations (default 64):

* every node is a real position (moves made with our movegen); leaves are evaluated with `initial` (representation + prediction), AlphaZero-style; the dynamics network is not used in self-play;
* root: Gumbel noise + Sequential Halving schedule over `m = 16` candidates (`get_table_of_considered_visits`; 64 simulations → 16 candidates × 1 visit, 8 × 2 more, 4 × 4 more, the final 2 the rest);
* interior: deterministic `argmax π'(a) − N(a)/(1+ΣN)`, `π' = softmax(logits + σ(q̂))`;
* completed Q with the mixed value, min-max rescaled, `σ = (50 + max N) · 0.1 · q̂` (epsilon 1e-4, see §6.2);
* checkmate / draw leaves (incl. repetition and 50-move rule from the game history) get their exact value and cost no network call;
* move: Sequential Halving winner; policy target: `softmax(logits + σ(q̂))` at the root.

Tests (`search::tests`): the schedule matches mctx's Python implementation; with a material-only value and a uniform policy the search finds a back-rank mate in one, takes a hanging rook, and refuses a pawn defended by a pawn (Qxd5?? exd5), which a one-ply search with the same networks plays.

Training is unchanged (K-step unroll still trains the dynamics network; with `--unroll 0` training becomes AlphaZero-style and much cheaper). The engine stays searchless.

## 6.4 Sprint 5b: value targets and noise schedule (run `fourth`)

* **Data format v2** (`record.rs`): per ply an optional search value, stored as W and D × 65535 for the side to move. W − L = root search value (mctx node value: mean of the root's network value and all backed-up values), D = the network's draw probability at the root, so L = 1 − W − D. v1 chunks (no value) are still read.
* **Value target** (`--q-ratio`, default 0.5 for new runs): `(1 − q)·one-hot(game result) + q·search W/D/L` per position, as soft cross-entropy. The logged `value_loss` stays the loss against the game result, so it remains comparable with earlier runs.
* **Noise schedule** (`--noise-plies 30 --late-noise 0`): full Gumbel noise for the first 30 plies, then none (AlphaZero's temperature drop). The policy target doesn't depend on the noise, only the move played does.
* **`--unroll 0`** for new runs: AlphaZero-style loss on the root position only; the dynamics network is still in the model but no longer trained.
* Run settings are stored in `config.json` and reused by `--resume`; settings missing from an older run's config keep that run's original behaviour.

## 6.5 Playout cap randomization (KataGo; `--fast-simulations`)

Policy targets want deep searches, value targets want many independent games. With `--fast-simulations N` (0 = off) each move is a **full** search (`--simulations`, `--considered`) with probability `--full-prob` (0.25), otherwise a **fast** search (N simulations, `--fast-considered` 8). Only full-search plies are training samples; every ply is still stored, so viriformat export and unrolling see complete games.

* `search.rs`: the slot draws full/fast when its root is evaluated (with PCR off nothing is drawn, so games are byte-identical to before), using separate Sequential Halving tables. Both use the same Gumbel noise schedule (noise in the first `--noise-plies` keeps openings varied).
* **Data format v3** (`record.rs`): v2's per-ply `has_value` byte becomes a flags byte (bit 0 = search value present, bit 1 = fast search). v1 and v2 are still read (all plies full); older binaries reject v3 instead of misreading it.
* `replay.rs`: samples are drawn only at full-search plies; `positions()` and the capacity count training samples. Fast plies keep their move and value targets for unrolling (policy masked) but drop their policy targets in memory.
* Cost per move (evaluations): full share × (full + 1) + fast share × (fast + 1); 25 % × 201 + 75 % × 33 ≈ 75 for 200/32, vs 65 for 64 on every move. Training samples per move: 25 %.

## 6.6 Game variety: openings, random moves, move temperature

Gumbel noise on the root logits barely varies the games once the search is deep: the played move is `argmax(g + logits + σ(q̂))` and σ reaches ±15 at 200 simulations while the noise is ~1. `fourth`'s data was 69 % unique, most duplicates in the first 40 plies. `search.rs` now has (all off by default):

* **Openings** (`openings`, `book_prob`): a new game starts after 1..all moves of a random book line; **random moves** (`random_prob`, `random_plies`) follow. Neither is recorded; `Game` history is kept so repetitions still count.
* **Move temperature** (lc0's temperature with value cutoff): the played move is sampled from `π'(a)^(1/T)` over visited root actions whose Q is within `temp_value_cutoff` of the best; `T` decays linearly over `temp_decay_plies` (game plies, book moves included) to `temp_endgame`. The policy target is unchanged.
* With all settings off no random numbers are drawn, so games are byte-identical to before.

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
