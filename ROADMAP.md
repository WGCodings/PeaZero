# PeaZero — Roadmap

Goal: a UCI chess engine whose move choice comes from **pure MuZero**. That means a learned representation, dynamics and prediction network, with MCTS that plans inside the learned model. The networks are trained by self-play.

**Decisions so far**
- Movegen: written from scratch in Rust (bitboards), verified by perft.
- Hardware: NVIDIA GeForce RTX 3070 (8 GB VRAM). Small networks and batched inference are mandatory.
- Algorithm: pure MuZero, using **Gumbel MuZero** search (Danihelka et al. 2022). It guarantees policy improvement even with very few simulations (16–32), which cuts self-play compute by an order of magnitude compared to 800-sim pUCT.
- Data: **pure self-play** only, with no supervised bootstrap.
- Testing: every change, whether to search or network, is validated by **SPRT on PeaOB** (OpenBench).
- Self-play + training: locally on the RTX 3070. OpenBench workers only *play* (CPU inference).
- Trainer: our own small PyTorch trainer; losses/tricks ported from LightZero (`gumbel_muzero`) and the MuZero paper. Search ported from DeepMind `mctx` to Rust.
- Inference: `tract` (pure Rust) on CPU for the engine/OpenBench; hand-written SIMD later. ONNX Runtime CUDA (`ort` crate, Cargo feature) for self-play only.
- Repo: public at github.com/WGCodings/PeaZero, registered on PeaOB (`Engines/PeaZero.json`).
- Details and rationale: [docs/MUZERO.md](docs/MUZERO.md).
- Language: Rust for everything that ships. Python or an external project is allowed for training.
- First network milestone: **a small model that reliably plays legal chess** (and beats random), not strength.
- The project is open-ended: after the first net, it's a continuous loop of search and network improvements.

## Way of working: sprints
Roughly 5 days/week. Each sprint ends with: test what we have → review → plan the next sprint (scope may change).

| Sprint | Scope | Done when |
|---|---|---|
| 1 ✅ | Phases 1–4: board, legal movegen, perft suite, UCI, random mover, OpenBench hooks (`bench`, Makefile) | done: 4.8B perft nodes correct, 1000-game stress clean, OB progtest ran |
| 2 ✅ | Phase 5 research + decisions, action/observation encodings (Rust + Python mirror), PyTorch→ONNX→tract inference spike | done: docs/MUZERO.md, 486 Rust↔Python encoding checks, tract matches PyTorch to 1e-7. Open: run `bench_gpu.py` on the 3070 |
| 3 ✅ | **Searchless** engine: one network eval per move, play the best legal policy move. `.pzn` network file, embedded via `EVALFILE`, `EvalFile` UCI option, network-independent bench, thin LTO for faster OB builds | done: untrained 64×6 net plays 300/300 legal games vs random (−120 Elo: the baseline training must beat) |
| 4 ✅ | Root-only Gumbel self-play (Rust pool, PyO3 bindings) + PyTorch trainer (K=5 unroll loss, data format v0, export every generation), live dashboard. Engine stays searchless | done: runs on the 3070 (~11k moves/s). Fixed: BatchNorm statistics mixed root and dynamics latents → exported net broken in eval mode (run `first` peaked at gen 200); per-graph recalibration since run `second` (+100 Elo at gen 100). Learning still slow: one-ply targets through the dynamics net |
| **5** | Self-play with Gumbel **tree search** (64 simulations) on the real board, leaves evaluated with `initial` (docs/MUZERO.md §6.3). Engine stays searchless | implemented + tested (finds mate in 1, avoids defended-pawn capture a one-ply search plays); **open: run `third` on the 3070, SPRT vs `second`** |
| 6+ | Policy net in an alpha-beta engine (own framework; a simple `SearchMode=AlphaBeta` test bed exists: policy ordering + value leaves, finds short mates); value-target improvements (search value mixing, lower sampling noise after the opening); network/search improvement loop, every change SPRT-tested on OB | — |

### OpenBench requirements (apply from Sprint 1)
- `Makefile` at repo root that honours `EXE=` (OB builds with `make EXE=<name>`).
- `<engine> bench` prints a final line `<nodes> nodes <nps> nps` and exits. The node count is the build signature, so it must be **deterministic**.
- Engine must support `Threads` and `Hash` UCI options (OB sets them even if unused).
- Later, networks will be distributed through OB's network upload, and the engine must be able to embed a default net at compile time (`include_bytes!`, path via `EVALFILE`-style Makefile variable).

---

## Repository layout (proposed)

```
PeaZero/
├─ Cargo.toml            # workspace
├─ crates/
│  ├─ core/              # board, bitboards, attacks, movegen, zobrist, perft
│  ├─ uci/               # UCI protocol loop, time management, engine trait
│  ├─ mcts/              # MuZero tree search (Phase 7), batching
│  ├─ nn/                # network inference backend (Phase 6/7)
│  └─ selfplay/          # self-play game generator binary (Phase 6)
├─ engine/               # final binary: `peazero`
├─ trainer/              # training code (Python or Rust, decided in Phase 5)
├─ tests/perft/          # EPD suites
└─ ROADMAP.md
```

The point is that the UCI layer only depends on an `Engine` trait:
`fn search(&mut self, pos: &Position, limits: &SearchLimits, info: &mut dyn InfoSink) -> Move`.
The random mover (Phase 4) and the MuZero engine (Phase 7) are both implementations of it, so the switch in Phase 7 doesn't touch UCI code.

---

## Phase 1 — Board representation & move generation

**Build**
- `Bitboard(u64)` and a `Square`/`Piece`/`Color` type.
- `Position`: piece bitboards, occupancy, side to move, castling rights, en passant square, halfmove clock, fullmove number, Zobrist hash.
- FEN parse/serialize (round-trip tested).
- Attack tables: knight, king, pawn (precomputed); sliders via **magic bitboards** (fancy magics; PEXT optional later behind a feature flag).
- `Move` as a packed `u16` (from, to, flags: promo piece, castle, en passant, double push).
- **Pseudolegal** generator writing into a fixed-capacity `MoveList` (no `Vec`, max 218 moves).
- **Legal** generator: pins + checkers mask approach (generate only legal moves directly), with `is_legal(move)` as a fallback/cross-check.
- `make_move` / `unmake_move` (or copy-make; benchmark both), with incremental hash.
- Draw detection: 50-move rule, threefold repetition (hash history), insufficient material.
- Game result detection: checkmate / stalemate. Needed later for self-play labels.

**Exit criteria:** FEN round-trip on all suite positions; legal gen == pseudolegal gen filtered by `is_legal` on random positions.

## Phase 2 — Exhaustive perft testing

**Build**
- `perft(depth)` and `divide(depth)` (per-move breakdown), exposed as a CLI subcommand **and** as non-standard UCI command `go perft N`.
- Test harness reading EPD suites (`fen ;D1 n ;D2 n ...`) and running in `cargo test --release`.
- Suites:
  - The 6 standard chessprogramming.org positions (startpos, Kiwipete, positions 3–6) to depth 5–6.
  - A large community suite (the ~6000-position "standard.epd"-style suites used by Ethereal/Stockfish devs) at shallow depth.
  - Special cases: en passant discovered check, castling through/out of check, promotions with capture, the "illegal ep" positions.
- Debug workflow: diff `divide` output against Stockfish's `go perft` to find the bad subtree.
- Hash consistency test: incremental Zobrist == recomputed-from-scratch Zobrist after every make/unmake during perft.
- Perft NPS benchmark (self-play speed matters later, but MuZero self-play is NN-bound, so correctness > speed here).

**Exit criteria:** 100% of suite positions pass; hash test passes; a startpos perft(6) = 119,060,324.

## Phase 3 — UCI protocol

**Commands to support**
| Command | Behaviour |
|---|---|
| `uci` | print `id name PeaZero`, `id author WG`, all `option`s, then `uciok` |
| `isready` | `readyok` (must answer even mid-search, and after loading a network) |
| `ucinewgame` | clear game state / search tree / caches |
| `position startpos [moves ...]` | |
| `position fen <fen> [moves ...]` | |
| `go` with `wtime btime winc binc movestogo movetime depth nodes infinite` | `depth` maps to simulation count later; `nodes` = MCTS simulations |
| `go ponder` + `ponderhit` | optional, low priority |
| `stop` | end search, print `bestmove` immediately |
| `setoption name X value Y` | `Hash`, `Threads`, later `NetworkPath`, `Simulations`, `Temperature`, `CPuct` |
| `quit` | exit cleanly |
| `debug on/off` | accepted, can be a no-op |
| non-standard: `d` (print board), `go perft N`, `bench` | dev conveniences |

**Implementation notes**
- Read stdin on its own thread; search runs on a worker thread; `stop` is an `AtomicBool`.
- Always reply `bestmove` (even when there are no legal moves: `bestmove 0000`).
- Moves in long algebraic notation (`e7e8q`), castling as king move (`e1g1`). Plan `UCI_Chess960` later if wanted.
- Time manager: simple `time/movestogo + inc*0.75` style budget with a hard cap. Make it a separate module so MCTS can reuse it.
- `info` output: `depth`, `nodes`, `nps`, `time`, `score cp`, `pv`. For MCTS, the value gets mapped to centipawns and the PV is the most-visited path.

**Exit criteria:** engine passes a cutechess-cli / fastchess sanity run (no illegal moves, no time losses, no hangs on `stop`).

## Phase 4 — Random mover

- `RandomEngine` implementing the `Engine` trait: uniform random legal move (seedable RNG via `setoption name Seed`).
- Playable by hand in a GUI (Arena, Cute Chess, En Croissant, BanksiaGUI).
- 1000-game run with fastchess, random vs random: zero illegal moves, zero crashes, every game reaches a proper result (mate/stalemate/50-move/repetition/insufficient). This stress-tests the whole Phase 1–3 stack.
- Optionally hook it into PeaOB as the first "engine version", so later nets can be SPRT-tested against it.

**Exit criteria:** you can play a full game against it in a GUI; stress run clean.

## Phase 5 — MuZero research & trainer decision

**Understand (write findings to the project)**
- The three networks:
  - **Representation** `h(obs) → s₀`: encodes the real position (+ history planes) into a latent state.
  - **Dynamics** `g(sₖ, aₖ) → (sₖ₊₁, rₖ)`: predicts the next latent state and reward (reward is always 0 in chess until the end, so this head can be dropped or kept trivially).
  - **Prediction** `f(sₖ) → (pₖ, vₖ)`: policy over 4672 actions (8×8×73 AlphaZero encoding) and value.
- MCTS in latent space: pUCT selection, expansion calls `g` then `f`, backup of value with sign flipping per ply. Legal-move masking happens **only at the root**; deeper nodes allow all actions (the model learns to give illegal ones low prior).
- Training: unroll K=5 steps from a sampled position. Targets are the MCTS visit distribution (policy), the final game outcome (value, since chess has no intermediate rewards), and optionally reward. Includes gradient scaling of ½ on the dynamics and scaling of the latent state to [0,1].
- Self-play details: Dirichlet noise at root, temperature schedule (sample for first ~30 plies, then greedy), resign threshold, max game length.
- Variants worth knowing: **Gumbel MuZero** (strong with very few simulations, which helps a lot on a single GPU), **Reanalyze / MuZero Unplugged** (re-run search on old games with newer nets for fresh targets), **EfficientZero** (consistency loss on latent states; aimed at sample efficiency).

**Evaluate existing trainers**
- **LightZero** (OpenDILab, PyTorch): MuZero, Gumbel MuZero, EfficientZero, Stochastic MuZero; has board-game envs. Most likely to be useful as a reference or base.
- **muzero-general** (PyTorch): very readable, but slow and unmaintained. Good for learning only.
- **DeepMind mctx** (JAX): MuZero and Gumbel MuZero *search* implementations, not a full trainer. A reference for getting the search exactly right.
- **Rust options:** `burn` / `candle` for an all-Rust trainer; `ort` (ONNX Runtime) or `tch-rs` (libtorch) for inference from Rust.

**Decide (exit criteria):** a written choice for:
1. Trainer: own PyTorch trainer, adapted LightZero, or all-Rust.
2. Inference backend in the Rust engine (ONNX via `ort`, `tch-rs`, or `burn`).
3. Data format between self-play and trainer (e.g. compressed binary chunks of `(obs, actions[K], policy targets, outcome)`).
4. Search variant: classic pUCT MuZero vs Gumbel MuZero.

## Phase 6 — Basic self-play training

**Pipeline**
```
 [selfplay (Rust)] --games--> [replay buffer on disk] --> [trainer] --checkpoint--> [export] --> back to selfplay
```
- Self-play generator: plays many games **in parallel** (hundreds) and batches leaf evaluations into one GPU call. Without batching the GPU sits idle and self-play is too slow to train anything.
- Start small, so a full loop runs in hours, not weeks:
  - Latent: small ResNet, e.g. 6 blocks × 64 channels on 8×8.
  - Simulations: 50–100 per move (or 16–32 with Gumbel).
  - Input: current position + a few history planes, side-to-move, castling, 50-move counter, repetition planes.
- Training loop: sample positions from the last N games, unroll K=5, losses = policy CE + value MSE (or WDL CE) (+ reward), plus L2. Checkpoint every X steps.
- Evaluation gate: each new net plays the previous net + the random mover (fastchess, SPRT on PeaOB). Track Elo over generations.
- Sanity checks along the way:
  - Overfit the network on a tiny fixed dataset (loss should go to ~0).
  - A "mate in 1" test set: does the policy head find the mates?
  - Value head on clearly won/lost positions.
  - Check that the dynamics model stays sane: compare `f(g(h(obs), a))` against `f(h(obs_after_a))`.

**Exit criteria:** net beats the random mover convincingly (e.g. >95% score) and shows monotonic-ish Elo gain over generations.

## Phase 7 — Network in the UCI engine

- `MuZeroEngine` implementing `Engine`:
  - Load network at startup / via `setoption name NetworkPath`.
  - Root: `h(obs)`, mask illegal moves, add no noise (play mode).
  - MCTS loop until time manager / `nodes` / `stop` says to stop; pick most-visited child.
  - `info` lines: simulations as `nodes`, value→cp mapping, most-visited PV.
  - Optional: tree reuse between moves (keep the subtree after the opponent's reply when it was explored).
- Batching in play mode: virtual loss + batch of leaves per GPU call, so a single game still uses the GPU well.
- Fallback: CPU inference path for machines without CUDA (slow, but it should still play legal chess).
- Release check: fastchess runs vs random mover and vs a weak classical engine; no illegal moves, no time losses.

**Exit criteria:** you can play PeaZero in a GUI and it clearly plays "intentional" chess.

---

## Open questions
- [x] GPU: RTX 3070 8 GB
- [x] Time budget: ~5 days/week, sprint-based
- [x] "Done": never. First milestone = small model that plays legal chess
- [x] Supervised bootstrapping: **no**, pure self-play
- [x] Testing: PeaOB SPRT for everything
- [ ] Chess960 support: in scope or not?
- [x] Self-play location: locally on the 3070 (OB datagen workloads with a tiny net = possible later)
- [x] Trainer: own PyTorch trainer using LightZero/mctx as references
- [x] CPU inference: tract now, SIMD later

## Risks
- **Compute.** Pure MuZero for chess from zero is very expensive. Mitigations: tiny nets, Gumbel search, reanalyze, optional supervised bootstrap.
- **Silent learned-model bugs.** With learned dynamics, a broken model still produces moves. That's why the Phase 6 sanity checks matter, and why the random-mover baseline stays in the eval set.
- **Action encoding mismatches** between the Rust side and the trainer (4672 index ↔ move). Write one shared spec and a round-trip test on both sides early.
