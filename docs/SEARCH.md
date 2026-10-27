# Engine search: improvement plan

This plan covers the engine's MCTS (`engine/src/search/mcts.rs`), the search
that plays games on OpenBench. It does not cover self-play: that uses the
Gumbel search in `crates/selfplay`, which is a separate code path.

The ideas are ranked by expected Elo per hour of work, and every item is
meant to be its own SPRT.

## 0. Current search features (MCTS, `SearchMode Mcts`)

**Selection and evaluation**
- PUCT selection: `Q + U`, with
  ```
  U          = cpuct(N) · gini_scale · N^CPuctExponent · P / (1 + n)
  cpuct(N)   = CPuct + CPuctFactor · ln((N + CPuctBase) / CPuctBase)
  gini_scale = clamp(CPuctGiniBase − CPuctGiniFactor · ln(gini + 0.001), 0, CPuctGiniMax)
  gini       = 1 − Σ prior² of the node's children
  ```
- First-play urgency: an unvisited child scores
  `FpuParentWeight · Q(parent) − FpuReduction · √(prior mass of the visited children)`, with separate `...Root` values at the root.
- Policy priors: a softmax over the legal moves' logits, with temperature `PolicyTemp`, and `PolicyTempRoot` at the root.
- The defaults reproduce the original search: constant cpuct 1.5 · √N, unvisited children at Q = 0, temperature 1. Bench is unchanged.
- One network evaluation per new leaf: policy plus W/D/L.
- W/D/L is backed up per node and flipped each ply. Q = W − L.

**Exact terminal values**
- Checkmate and stalemate.
- 50-move rule and insufficient material.
- Repetition: the first repetition counts as a draw, including positions from the game history.

**Move choice and output**
- Final move: most visits; ties are broken by prior.
- PV, and a ponder move from the PV.
- `info` lines show:
  - depth (the average playout depth) and seldepth;
  - score in cp, from W−L through a tangent curve: about 10,800 cp at the ±0.999 cap, which is the "very large value" shown near a mate;
  - `wdl` (UCI_ShowWDL);
  - nodes, nps and pv.

**Limits**
- `go nodes`, `go depth`, `go movetime`, clock plus increment, `movestogo`, `infinite`, `searchmoves`, and `stop`.
- 400 playouts when no limit is given.
- Time manager: soft = time/movestogo (30 by default) + ¾·inc, capped at time/2. The search stops at the soft limit; the hard limit is computed but not used yet.

**Network and modes**
- Int8 SIMD inference (FastNet) on x86-64, tract on other CPUs. Batch 1, single-threaded.
- Other modes: `AlphaBeta` (iterative deepening with policy move ordering, with real `mate N` scores) and `Searchless`.

**Tuning**
- All search parameters are UCI spin options, declared in one table in `engine/src/search/params.rs`.
- The UCI command `spsa` prints them in OpenBench's SPSA input format.
- To add a parameter: add one line to the `tunables!` table.

**Not present yet**
- Tree reuse between moves.
- A network evaluation cache or transposition table.
- Batching, virtual loss or multithreading.
- Propagation of proven wins, losses and draws, so no `mate N` scores.
- Smart pruning, early stopping or extending past `soft`.

## 1. Node budget

- **How many playouts.** At 8+0.08 the soft limit is about 8000/30 + 60 ≈ 330 ms per move. The 64×6 int8 network takes about 264 µs per evaluation, so the search sees only about 1,000–1,300 playouts per move.
- **What matters at that budget.** Anything that makes those few playouts count is worth more here than for an engine searching 10⁵ nodes per move: FPU, cpuct, policy temperature, tree reuse and stopping rules.
- **What can wait.** Throughput tricks only pay if the CPU network gets faster per position when it evaluates in batches. That has to be measured first (§5.1).

## 2. Phase 1: search parameters (done; now tune them)

### 2.1 First-play urgency

**Problem with the default.** An unvisited move scores Q = 0, which is a draw.
- In a lost position, every unexplored move looks better than every explored one, so the search spreads its few playouts over junk moves.
- In a won position it is the reverse: good alternatives are never tried.

**Options.**
- `FpuParentWeight` = 1: unvisited moves start at the parent's own value.
- `FpuReduction` > 0: they start lower the more prior mass has already been visited.
- Typical starting values: weight 1, reduction about 0.4 in the tree and about 1.0 at the root.

**Sign convention.** In our `Node`, Q is stored from the view of the side that moved into the node. A child of `node` is scored from the view of the side to move at `node`, which is `−node.q()`. A unit test covers this.

### 2.2 Exploration term

- **`CPuct`:** the base constant.
- **`CPuctFactor` / `CPuctBase`:** logarithmic growth with the parent's visits. With a large base (around 18k) the growth only matters at long time controls.
- **`CPuctExponent`:** the power of N; 0.5 = √N. Values around 0.8 with a small CPuct (around 0.16) are a different but equally valid regime: much more exploration at large N.
- **Gini scaling:** more exploration where the policy is sharp. One working setting is base 0.68, factor 1.63, max 2.1.
- **Why our value is open:** 1.5 was never tuned, and our policy comes from Gumbel training, which changes how sharp it is.

### 2.3 Policy temperature

- `softmax(logit / T)`, with one T for the root and one for the tree. Values between 1.3 and 1.6 are common.
- **Why it may matter here:** our policy targets are π' = softmax(logits + σ(completed Q)), which tend to be sharp. A sharp prior plus low cpuct means good second moves get almost no visits; the castling story was this effect inside self-play.

### 2.4 Tune them with SPSA

**How to run it:**
1. Start the engine and type `spsa`. It prints the current values, including any `setoption` you made, so you can start a tune from any point.
2. Paste the printed lines into the SPSA field of a new OpenBench tune. `c_end` is 7% of each value.
3. When the tune finishes, copy the final values into the `default` column of the `tunables!` table in `engine/src/search/params.rs`.
4. SPRT the tuned values against the old defaults.

**Which parameters to tune.** Leave out the lines of parameters you don't want tuned. Fewer parameters converge faster.

**SPSA is a local search.** It tunes around its starting point and won't move between very different regimes by itself. For example, CPuct 1.5 with exponent 0.84 explodes. To compare regimes:
1. SPRT a few hand-picked starting points against each other, set through the dev engine's options. Two examples:
   - √N with log growth: CPuct 215, CPuctFactor 282, FpuParentWeight 100, FpuReduction 44, FpuParentWeightRoot 100, FpuReductionRoot 100, PolicyTemp 161, PolicyTempRoot 161.
   - N^0.84 with gini scaling: CPuct 16, CPuctExponent 84, CPuctGiniBase 68, CPuctGiniFactor 163, CPuctGiniMax 210, FpuParentWeight 100, PolicyTemp 130, PolicyTempRoot 130.
2. Run the SPSA from the winner.

**Practical notes:**
- Tune at the time control you test at (8+0.08). Settings tuned at fixed nodes don't carry over exactly.
- Budget about 20–40k games for 5–7 parameters.
- Whenever the network changes a lot (a new run, or the distilled net), re-tune at least cpuct and the temperatures. The best values depend on how sharp the policy is.

## 3. Phase 2: get more out of each second

### 3.1 Tree reuse

- **Change:** keep the tree after `go`. On the next `position ... moves`:
  1. Find the two plies played (ours and the opponent's).
  2. Make that grandchild the new root.
  3. Compact the arena: copy that subtree to a fresh `Vec` with new indices.
- **Fallback:** if the moves don't match (new game, `ucinewgame`, a position that isn't a continuation), start fresh.
- **Gain:** the subtree of the reply that was actually played usually holds 20–50% of the previous search, and the expected reply holds the most. That is close to a free 1.3–1.5× in nodes.
- **Caveats:**
  - Priors inside the tree were computed with `PolicyTemp`. When a node becomes the root, recompute its children's priors with `PolicyTempRoot`.
  - Repetition draws depend on game history, which changes. The simplest safe rule is to clear terminal flags for draws by repetition when reusing.
- **Expected:** +20 to +50 Elo at short time controls. Effort: medium (about 150 lines, plus tests).

### 3.2 Time management: use `hard`, stop early when the answer is settled

Today the search always stops at exactly `soft`; `hard` (3× soft) is computed and never used.

**a) Early stop (saves time).** Estimate the playouts left before `soft` from the current nodes per second.
- If the second-best root move can't catch the best even if it gets every remaining playout, stop now: `visits_best − visits_second > remaining · factor`, with a factor of about 1.3.
- Optionally, during the search, skip root moves that can no longer catch the leader.
- The time saved stays on the clock for later moves.

**b) Extend in unclear positions.** Measure at the soft limit:
```
bm_frac  = visits(best) / visits(root)
m        = (a − bm_frac) · b                       // spread visits → more time
m       *= 1 + c · |Q(best) − Q(end of best PV)|   // unstable PV → more time
stop when elapsed ≥ soft · clamp(m, m_min, m_max)  (never past hard)
```
- Reasonable starting values: a 1.4, b 1.4, c 1.2, m_min 0.1, m_max 5.
- A simpler form also works: keep going past `soft` up to `hard` while the most-visited move is not also the move with the best Q, or while its Q dropped noticeably during the last part of the search.

**c) Tuning.** The time manager constants go into the same `tunables!` table and the same SPSA.

**Expected:** +15 to +40 Elo combined. Test at the real time control, never at fixed nodes. Effort: small to medium.

### 3.3 Network evaluation cache

A hash map from the Zobrist key (plus the key of the history planes the network sees) to the policy and WDL output.
- **Where it pays:**
  - transpositions inside the tree, which are rare at 1k nodes but common in endgames;
  - repeated positions;
  - a tree that was lost, because tree reuse missed or a move was unexpected.
- **Size:** around 200k entries. A fixed-size table that just overwrites is enough. The `Hash` UCI option sizes it.
- **Caution about the key:** the network's input includes history planes, so a cache keyed by the current position alone returns slightly wrong evaluations for transpositions reached by different move orders. Either accept that or hash the last few positions in too. SPRT it.
- **Expected:** +5 to +15 Elo at short time controls, more at long ones. Effort: small.

A full DAG search (shared position nodes, statistics on the edges) is the long-term version. It's a big rewrite, and at our node counts it isn't worth it before the items above.

### 3.4 Cheap per-playout speed-ups (profile first)

At 264 µs per evaluation, inference probably dominates, so measure before working on these:
- `Tree::evaluate` calls `pos.legal_moves()` for the terminal check, then `legal_actions()` generates the moves again inside `PolicyNet::evaluate`. Generate them once and pass them in.
- Every playout clones the whole `Game` (with its history) and allocates `path`. Reuse one scratch game and path buffer, or make and unmake moves.
- `Node` is about 64 bytes. Moving `terminal`, `expanded` and `count` into a compact form, and storing priors as `f16`, matters for long time controls, where memory grows by about 35 nodes per playout.
- Run `perf` or `cargo flamegraph` on `bench` first. If inference is above 90% of the time, skip all of this.

## 4. Phase 3: correctness in decisive positions

### 4.1 Certainty propagation (MCTS-solver)

**Current behaviour.**
- Terminal nodes get exact values, but nothing propagates them, so the `info` score near a mate is about 10,800 cp instead of `mate N`.
- A mate in one at the root still has to win on visits against moves the policy likes.
- A move that walks into a forced mate keeps being searched, because its average Q is only "quite bad".

**Rules:**
- A node is a proven **win** (for the side to move) if **any** child is a proven loss for the opponent.
- It is a proven **loss** if **all** children are proven wins for the opponent.
- It is a proven **draw** if no child wins and the best child is a proven draw.

**Use of proven results:**
- Back up the exact value instead of the average.
- Never select a child that is proven lost when another option exists.
- Stop the search immediately when the root is proven: play the mating move, and report `score mate N` with the distance.
- Prefer the shortest win and the longest loss, which needs plies-to-mate stored on proven nodes.

**Expected:** +5 to +20 Elo, mostly by converting won positions and avoiding blunders into mates. Effort: medium.

### 4.2 Final move choice

- **Problem:** "most visits" is right in most positions. But at 1k nodes, a move that became good late can lose on visits while having a clearly better Q.
- **Options:**
  - most visits, with ties broken by Q;
  - maximise `Q − c / √N`, a lower confidence bound, so a move with few visits only wins if its Q is clearly higher. `c` becomes a tunable.
- **Test:** cheap to try; SPRT it after the early stop, because the two interact.
- **Expected:** 0 to +15 Elo.

### 4.3 Syzygy tablebases

- **Change:** probe WDL at leaves with ≤ 5–6 pieces, and use DTZ at the root. Rust crates such as `shakmaty-syzygy` or `pyrrhic-rs` can be adapted to our board.
- **Value:** small unless OpenBench workers have the tables. They would need `SyzygyPath` set, and the tests should be configured to use them; most aren't by default.
- **Priority:** low until everything above is done.

## 5. Phase 4: larger engineering (measure first)

### 5.1 Batched evaluation with virtual loss

- **Change:** collect several leaves per step with virtual loss, evaluate them as one batch, then back them all up.
- **When it pays:** only if the int8 CPU network gets faster per position at batch 4–16 than at batch 1. Convolutions on a 64-channel 8×8 board often do, through better GEMM shapes and amortised weight loads, by roughly 1.3–2×.
- **First measure:** a micro-benchmark of evaluations per second at batch sizes 1, 2, 4, 8, 16 (§7). If the gain is below 1.2×, skip this.
- **Costs:** virtual loss also slightly degrades the search, more so at small node counts like ours.
- **Expected:** −10 to +40 Elo, depending entirely on that benchmark. Effort: high.

### 5.2 Multi-threaded search

- **Change:** the same virtual-loss machinery with N threads, each evaluating its own leaves.
- **Value:** only useful for `Threads > 1` tests and play. OpenBench tests normally run 1 thread, so this doesn't show in SPRTs unless you add SMP tests.
- **Priority:** after 5.1, which builds most of the infrastructure.

### 5.3 Gumbel root search in the engine

- **Idea:** the network was trained with Gumbel MuZero, and Danihelka et al. show that Sequential Halving at the root with completed-Q policy improvement beats PUCT at small budgets (≤ 100s of simulations). Our engine sits at about 1,000 nodes per move.
- **Test design:** an experimental root-search mode.
  - Gumbel noise turned off for play.
  - Root candidates: the top k by policy.
  - Sequential Halving over the playout budget.
  - Final move = argmax of logits + σ(completed Q).
  - PUCT, or the deterministic Gumbel non-root rule, below the root.
- **Expected:** uncertain: could be +30 or −30. It interacts with time management, because Sequential Halving needs the budget known in advance; use `soft` as the budget. Effort: medium; the logic already exists in `crates/selfplay/src/search.rs` to copy from.

### 5.4 Not recommended now

- **Moves-left utility.** Needs a moves-left head the network doesn't have.
- **Contempt / draw score.** Neutral in self-SPRTs; only useful against weaker opponents.
- **Trend-based cpuct adjustment and per-thread cpuct jitter.** Second-order; revisit after SPSA, if at all.
- **Full DAG search.** Large rewrite; see §3.3.

## 6. Recommended order

| # | Change | Test | Expected | Effort |
|---|---|---|---|---|
| 1 | Tunable FPU, exploration term and policy temperatures as UCI options **(done; defaults = original search)** | — | — | small |
| 2 | Pick a starting point (§2.4), then SPSA | SPRT tuned vs original | +30–80 | games only |
| 3 | Tree reuse | SPRT 8+0.08 | +20–50 | medium |
| 4 | Early stop + use `hard` / extension multiplier | SPRT 8+0.08, then SPSA TM constants | +15–40 | small–medium |
| 5 | Certainty propagation + mate scores | SPRT 8+0.08 | +5–20 | medium |
| 6 | Network evaluation cache | SPRT 8+0.08 and 40+0.4 | +5–15 | small |
| 7 | Final-move LCB | SPRT | 0–15 | trivial |
| 8 | Batch benchmark → batched search if ≥ 1.2× | benchmark, then SPRT | −10–+40 | high |
| 9 | Gumbel root search (experiment) | SPRT | ±30 | medium |

The Elo figures are rough guesses based on what such changes typically give
MCTS engines at comparable node counts. They are not promises: the gains
overlap (for example, FPU and cpuct partly fix the same problem), so they
won't simply add up.

## 7. Testing notes

- **Elo gain per node versus per second.**
  - Selection changes (#2, #5, #7) improve play at a fixed number of nodes. A fixed-nodes SPRT (for example N=1000) is a quick first check, but the result that counts is at the time control.
  - Reuse, time management and the cache (#3, #4, #6) only show up in timed tests.
- **One change per SPRT.** Bounds as usual (for example [0, 5] for gains, or [−3, 1] non-regression for refactors such as §3.4).
- **Batch benchmark (for §5.1).** Add a `bench-batch` mode that times `PolicyNet` inference at batch sizes 1, 2, 4, 8 and 16 on a fixed set of positions. That needs a batched `evaluate` in the int8 backend.
- **Correctness tests to keep green:**
  - `mate_in_one`, `go_depth_stops_at_depth_and_is_deterministic`, `single_playout_plays_policy_move`;
  - the FPU tests (`fpu_follows_the_parent_value`, `fpu_changes_which_move_is_explored`);
  - a bench equal to the original search's with default parameters.
- **Tests to add with later items:**
  - a tree-reuse round-trip test: the reused subtree's visits equal those of the old child;
  - solver tests: mate in two found and reported as `mate 2`; the root proven and stopped early.
