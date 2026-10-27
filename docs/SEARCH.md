# Engine search: improvement plan

This plan covers the engine's MCTS (`engine/src/search/mcts.rs`), the search
that plays games on OpenBench. It does not cover self-play: that uses the
Gumbel search in `crates/selfplay`, which is a separate code path.

Each idea is checked against what lc0 and princhess do. The ideas are ranked
by expected Elo per hour of work, and every item is meant to be its own SPRT.

## 0. Current search features (MCTS, `SearchMode Mcts`)

**Selection and evaluation**
- PUCT selection: `Q + cpuct(N) · P · √N / (1 + n)`.
- cpuct grows with visits: `CPuct + CPuctFactor · ln((N + CPuctBase) / CPuctBase)`. *(new)*
- FPU reduction: an unvisited child scores `Q(parent) − FpuReduction · √(visited policy)`; the root uses `FpuReductionRoot`. *(new)*
- Policy priors are a softmax over the legal moves' logits, with temperature `PolicyTemp`, and `PolicyTempRoot` at the root. *(new)*
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
  - score in cp, using lc0's W−L → cp mapping;
  - `wdl` (UCI_ShowWDL);
  - nodes, nps and pv.

**Limits**
- `go nodes`, `go depth`, `go movetime`, clock plus increment, `movestogo`, `infinite`, `searchmoves`, and `stop`.
- 400 playouts when no limit is given.
- Time manager: soft = time/movestogo (30 by default) + ¾·inc, capped at time/2. The search stops at the soft limit; the hard limit is computed but not used yet.

**Network and modes**
- Int8 SIMD inference (FastNet) on x86-64, tract on other CPUs. Batch 1, single-threaded.
- Other modes: `AlphaBeta` (iterative deepening with policy move ordering) and `Searchless`.

**Tuning**
- All search parameters are UCI spin options, declared in one table in `engine/src/search/params.rs`.
- The UCI command `spsa` prints them in OpenBench's SPSA input format.
- To add a parameter: add one line to the `tunables!` table.

## 1. Where the search stands today

| Aspect | PeaZero now | lc0 (defaults) | princhess |
|---|---|---|---|
| Selection | PUCT, constant `C_PUCT = 1.5` | PUCT, cpuct grows with N: `2.15 + 2.82·ln((N+18368)/18368)` | PUCT, cpuct grows with N: `cpuct·(N+1)^(τ/10000)`, τ = 84; scaled by the Gini impurity of the policy |
| Unvisited child (FPU) | Q = 0 (a draw) | parent Q − 0.44·√(policy mass already visited); root uses reduction 1.0 | parent Q; root Q = 0 |
| Policy softmax | temperature 1 | temperature 1.61 | 1.0 for children, a separate higher temperature at the root |
| Tree between moves | thrown away | reused | reused, via the graph and table |
| Transpositions / NN cache | none | NN cache, 200k entries | DAG: shared position nodes, statistics on edges, warm start from the table |
| Batching / threads | batch 1, 1 thread | minibatch 256, out-of-order evaluation, virtual loss | threads with virtual loss |
| Stopping | the soft limit only; `hard` is never used | smart pruning (factor 1.33), KLD-gain stop, time manager | soft × adaptive multiplier (clamped to 0.1–5), hard limit, see §3.2 |
| Proven results | terminal nodes are exact, but not propagated | sticky endgames (certainty propagation) | tablebases, no solver |
| Final move | most visits | most visits, with proven results and Q as tie-breaks | Q − c·2/√N (a lower confidence bound) |

What this means in practice:

- **Node budget.** At 8+0.08 the soft limit is about 8000/30 + 60 ≈ 330 ms per move. The 64×6 int8 network takes about 264 µs per evaluation, so the search sees only about 1,000–1,300 playouts per move.
- **What matters at that budget.** Anything that makes those few playouts count is worth more here than it would be for lc0 at 10⁵ nodes: FPU, cpuct, policy temperature, tree reuse and stopping rules.
- **What can wait.** Throughput tricks only pay if the CPU network gets faster per position when it evaluates in batches. That has to be measured first (§5.1).

## 2. Phase 1: search parameters (cheap, probably the largest gain)

Make each parameter a UCI spin option. OpenBench SPSA needs that, and it lets
you SPRT a new default without a rebuild. Integer options with a scale (for
example CPuct 215 = 2.15) are the usual OpenBench convention.

### 2.1 FPU reduction (highest priority)

**Problem.** Today an unvisited move scores Q = 0, which is a draw.
- In a lost position, every unexplored move looks better than every explored one, so the search spreads its few playouts over junk moves.
- In a won position it is the reverse: good alternatives are never tried.

**Change.** Use lc0's rule:

```
visited_policy = Σ prior(c) over children with visits > 0
q_fpu = Q(parent, from the mover's view) − fpu_reduction · √visited_policy
```

- lc0's defaults are reduction 0.44 inside the tree and 1.0 at the root (the root gets a stronger reduction).
- A simpler alternative is princhess's FPU = parent Q.

**Expected:** +20 to +60 Elo at this node count; this is one of lc0's biggest historical search gains. Effort: about 20 lines.

**Sign convention.** In our `Node`, Q is stored from the view of the side that moved into the node. A child of `node` is scored from the view of the side to move at `node`, which is `−node.q()`. Check this with a unit test.

### 2.2 cpuct that grows with visits

Replace the constant with lc0's form:

```
cpuct(N) = c_init + c_factor · ln((N + c_base) / c_base)
```

- Start from lc0's c_init 2.15, c_base 18368, c_factor 2.82.
- At 1k nodes the log term is almost 0, so the effective value is c_init. The growth only matters at long time controls.
- Our 1.5 was never tuned, and our network's policy came from Gumbel training. That changes how sharp the policy is, so the best value may sit well away from lc0's.

**Expected:** +10 to +40 Elo after tuning. Effort: trivial.

### 2.3 Policy temperature, separate at the root

The engine applies a softmax to the legal logits as they are (`policy.rs`).

- **Change:** use `softmax(logit / T)`, with one T for the root and one for the tree. lc0 uses T = 1.61 everywhere; princhess uses a higher root T.
- **Why it may matter here:** our policy targets are π' = softmax(logits + σ(completed Q)), which tend to be sharp. A sharp prior plus low cpuct means good second moves get almost no visits; the castling story was this effect inside self-play.

**Expected:** +10 to +30 Elo. Effort: trivial; apply it where the priors are computed in `Tree::evaluate`.

### 2.4 Tune them together with SPSA

**How to run it:**
1. Start the engine and type `spsa`.
2. Paste the printed lines into the SPSA field of a new OpenBench tune.
3. When the tune finishes, copy the final values into the `default` column of the `tunables!` table in `engine/src/search/params.rs`.

Parameters you don't want tuned: just leave their line out of the tune.

- **Method:** run an OpenBench SPSA workload over CPuct, CPuctFactor, CPuctBase (optional), FpuReduction, FpuReductionRoot, PolicyTemp and PolicyTempRoot.
  - Tune at the time control you test at (8+0.08). Settings tuned at fixed nodes do not carry over exactly.
  - Budget about 20–40k games for 5–7 parameters.
- **Afterwards:** SPRT the tuned values against the hand-set defaults.
- **Re-tune after changes:** whenever the network changes a lot (a new run, or the distilled net), re-tune at least cpuct and the temperatures. The best values depend on how sharp the policy is.
- **princhess's approach:** it tuned CPuct, CPuctTau and PolicyTemperatureRoot with Bayesian optimisation (chess-tuning-tools). SPSA on OpenBench does the same job with tools you already run.

## 3. Phase 2: get more out of each second

### 3.1 Tree reuse

- **Change:** keep the tree after `go`. On the next `position ... moves`:
  1. Find the two plies played (ours and the opponent's).
  2. Make that grandchild the new root.
  3. Compact the arena: copy that subtree to a fresh `Vec` with new indices.
- **Fallback:** if the moves don't match (new game, `ucinewgame`, a position that isn't a continuation), start fresh.
- **Gain:** the subtree of the reply that was actually played usually holds 20–50% of the previous search, and the expected reply holds the most. That is close to a free 1.3–1.5× in nodes.
- **Caveats:**
  - The root's priors must be the plain policy. With the separate root temperature (§2.3), re-apply it when a node becomes the root.
  - Repetition rules depend on game history, which changes; the same holds for terminal flags on nodes whose status depends on the path. lc0 handles this by re-checking draws on the path; the simplest safe rule is to clear terminal flags for draws by repetition when reusing.
- **Expected:** +20 to +50 Elo at short time controls. Effort: medium (about 150 lines, plus tests).

### 3.2 Time management: use `hard`, stop early when the answer is settled

Today the search always stops at exactly `soft`; `hard` (3× soft) is computed and never used. lc0 and princhess both spend time unevenly.

**a) Smart pruning (lc0) saves time.** Estimate the playouts left before `soft` from the current nodes per second.
- If the second-best root move can't catch the best even if it gets every remaining playout, stop now: `visits_best − visits_second > remaining · factor`, with lc0's factor 1.33.
- Optionally, during the search, skip root moves that can no longer catch the leader.
- The time saved stays on the clock for later moves.

**b) Extend in unclear positions (princhess's multiplier).** Measure at the soft limit:
```
bm_frac  = visits(best) / visits(root)
m        = (1.40 − bm_frac) · 1.39              // spread visits → more time
m       *= 1 + 1.21 · |Q(best) − Q(end of best PV)|   // unstable PV → more time
stop when elapsed ≥ soft · clamp(m, 0.1, 5)  (never past hard)
```
- A simpler form also works: keep going past `soft` up to `hard` while the most-visited move is not also the move with the best Q, or while its Q dropped noticeably during the last part of the search.

**c) Tuning.** The time manager constants go into the same SPSA (TMVisitsBase, TMVisitsM and so on).

**Expected:** +15 to +40 Elo combined. Test at the real time control, never at fixed nodes. Effort: small to medium.

### 3.3 NN evaluation cache

A hash map from the Zobrist key (plus the key of the history planes the network sees) to the policy and WDL output.
- **Where it pays:**
  - transpositions inside the tree, which are rare at 1k nodes but common in endgames;
  - repeated positions;
  - a tree that was lost, because tree reuse missed or a move was unexpected.
- **Size:** lc0 uses 200k entries. A fixed-size table that just overwrites is enough. The `Hash` UCI option sizes it.
- **Caution about the key:** the network's input includes history planes, so a cache keyed by the current position alone returns slightly wrong evaluations for transpositions reached by different move orders. lc0 accepts this, but hashes the last few positions in too. Decide which, and SPRT it.
- **Expected:** +5 to +15 Elo at short time controls, more at long ones. Effort: small.

The full DAG (princhess-style shared nodes with edge statistics) is the long-term version. It's a big rewrite, and at our node counts it isn't worth it before the items above.

### 3.4 Cheap per-playout speed-ups (profile first)

At 264 µs per evaluation, inference probably dominates, so measure before working on these:
- `Tree::evaluate` calls `pos.legal_moves()` for the terminal check, then `legal_actions()` generates the moves again inside `PolicyNet::evaluate`. Generate them once and pass them in.
- Every playout clones the whole `Game` (with its history) and allocates `path`. Reuse one scratch game and path buffer, or make and unmake moves.
- `Node` is about 64 bytes. Moving `terminal`, `expanded` and `count` into a compact form, and storing priors as `f16`, matters for long time controls, where memory grows by about 35 nodes per playout.
- Run `perf` or `cargo flamegraph` on `bench` first. If inference is above 90% of the time, skip all of this.

## 4. Phase 3: correctness in decisive positions

### 4.1 Certainty propagation (MCTS-solver, lc0's "sticky endgames")

**Current behaviour.** Terminal nodes get exact values, but nothing propagates them. A mate in one at the root still has to win on visits against moves the policy likes. A move that walks into a forced mate keeps being searched because its average Q is only "quite bad".

**Rules:**
- A node is a proven **win** (for the side to move) if **any** child is a proven loss for the opponent.
- It is a proven **loss** if **all** children are proven wins for the opponent.
- It is a proven **draw** if no child wins and the best child is a proven draw.

**Use of proven results:**
- Back up the exact value instead of the average.
- Never select a child that is proven lost when another option exists.
- Stop the search immediately when the root is proven: play the mating move, and report `score mate N` with the distance.
- Prefer the shortest win and the longest loss, which needs plies-to-mate stored on proven nodes.

**Expected:** +5 to +20 Elo, mostly by converting won positions and avoiding blunders into mates. It also fixes the mate-display issue in `info` lines. Effort: medium.

### 4.2 Final move choice

- **Problem:** "most visits" is right in most positions. But at 1k nodes, a move that became good late can lose on visits while having a clearly better Q.
- **Options:**
  - lc0 style: most visits, with ties broken by Q.
  - princhess style: maximise `Q − c / √N`, a lower confidence bound, so a move with few visits only wins if its Q is clearly higher.
- **Test:** cheap to try; SPRT it after smart pruning, because the two interact.
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
- **Costs:** virtual loss also slightly degrades the search, more so at small node counts. lc0 compensates with very high node counts, which we don't have.
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

- **Moves-left utility (lc0).** Needs a moves-left head the network doesn't have.
- **Contempt / draw score.** Neutral in self-SPRTs; only useful against weaker opponents.
- **Princhess's Gini-scaled cpuct and trend adjustment.** Second-order; revisit after SPSA, if at all.
- **Full DAG search.** Large rewrite; see §3.3.

## 6. Recommended order

| # | Change | Test | Expected | Effort |
|---|---|---|---|---|
| 1 | FPU reduction + cpuct(N) + policy temperatures as UCI options, lc0 defaults **(done)** | SPRT 8+0.08 vs current | +30–80 | small |
| 2 | SPSA over #1's parameters | SPRT tuned vs #1 | +10–30 | games only |
| 3 | Tree reuse | SPRT 8+0.08 | +20–50 | medium |
| 4 | Smart pruning + use `hard` / adaptive multiplier | SPRT 8+0.08, then SPSA TM constants | +15–40 | small–medium |
| 5 | Certainty propagation + mate scores | SPRT 8+0.08 | +5–20 | medium |
| 6 | NN cache | SPRT 8+0.08 and 40+0.4 | +5–15 | small |
| 7 | Final-move LCB | SPRT | 0–15 | trivial |
| 8 | Batch benchmark → batched search if ≥ 1.2× | benchmark, then SPRT | −10–+40 | high |
| 9 | Gumbel root search (experiment) | SPRT | ±30 | medium |

The Elo figures are rough guesses based on how much these changes gave other
MCTS engines at comparable node counts. They are not promises: the gains
overlap (for example, FPU and cpuct partly fix the same problem), so they
won't simply add up.

## 7. Testing notes

- **Elo gain per node versus per second.**
  - Selection changes (#1, #2, #5, #7) improve play at a fixed number of nodes. A fixed-nodes SPRT (for example N=1000) is a quick first check, but the result that counts is at the time control.
  - Reuse, time management and the cache (#3, #4, #6) only show up in timed tests.
- **One change per SPRT.** Bounds as usual (for example [0, 5] for gains, or [−3, 1] non-regression for refactors such as §3.4).
- **Batch benchmark (for §5.1).** Add a `bench-batch` mode that times `PolicyNet` inference at batch sizes 1, 2, 4, 8 and 16 on a fixed set of positions. That needs a batched `evaluate` in the int8 backend.
- **Correctness tests to keep green:** `mate_in_one`, `go_depth_stops_at_depth_and_is_deterministic`, `single_playout_plays_policy_move`. Also add:
  - FPU sign tests: a lost and a won parent;
  - a tree-reuse round-trip test: the reused subtree's visits equal those of the old child;
  - solver tests: mate in two found and reported as `mate 2`; the root proven and stopped early.
