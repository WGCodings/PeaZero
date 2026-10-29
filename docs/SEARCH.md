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
- Defaults:
  - cpuct 2.15, growing logarithmically (factor 2.82, base 18368), times √N; gini scaling off;
  - FPU = parent Q − 0.44 · √(visited prior mass) in the tree, and parent Q − 1.0 · √(...) at the root;
  - policy temperature 1.61.
- The original search (constant cpuct 1.5, unvisited children at Q = 0, temperature 1) is still reachable: CPuct 150, CPuctFactor 0, all four FPU values 0, both temperatures 100.
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

**Tree reuse**
- The subtree of the moves played since the last search becomes the new root, with its visits and values (§3.1).

**Transposition table (network evaluations)**
- A position seen before (any move order) costs a table probe instead of a network call (§3.3).

**Memory (`Hash` is the whole budget)**
- 75% for the tree (24-byte nodes in one arena, allocated once), 25% for the evaluation table (80-byte entries). Peak memory is `Hash` plus about 14 MB for the program and network.
- When the tree is full, the search prunes and keeps going until its time is up:
  - The least-visited expanded nodes are collapsed back into leaves until the tree fits in half the arena. A collapsed node keeps its own visits and W/L; only its children are dropped.
  - Most of the arena is unvisited children (each expansion allocates a slot per legal move), so this frees a lot, while the well-visited part of the tree that decides the move is untouched.
  - A collapsed node that becomes interesting again is simply expanded again, usually from the evaluation table.
  - Only if pruning can't make room for one expansion (a tiny `Hash`) does the search continue without growing the tree.
  - `info string tree full (Hash N MB): pruned K times` reports it. `hashfull` in the `info` lines is the tree's fill.
- Example: a 60k-playout search with `Hash` 16 pruned 6 times, peaked at 30 MB, and played the same move as with `Hash` 256 (122 MB).
- `Hash` 16 holds about 16k playouts and `Hash` 64 about 64k; a move at 8+0.08 uses 1–3k.

**Limits**
- `go nodes`, `go depth`, `go movetime`, clock plus increment, `movestogo`, `infinite`, `searchmoves`, and `stop`.
- 400 playouts when no limit is given.
- Time manager (§3.2): target = time/movestogo (`TmMovesToGo`, 30) + `TmIncPercent` of the increment, at most half the remaining time. The search stops at target × a factor from how settled it is (more time when the best move holds a small share of the visits, less when it holds most), never past the hard limit (`TmHardMult` × target, at most ¾ of the remaining time). A single legal move is played at once; `go movetime` uses exactly its time.

**Network and modes**
- Int8 SIMD inference (FastNet) on x86-64, tract on other CPUs. Batch 1, single-threaded.
- Other modes: `AlphaBeta` (iterative deepening with policy move ordering, with real `mate N` scores) and `Searchless`.

**Tuning**
- All search parameters are UCI spin options, declared in one table in `engine/src/search/params.rs`.
- The UCI command `spsa` prints them in OpenBench's SPSA input format.
- To add a parameter: add one line to the `tunables!` table.

**Not present yet**
- Shared statistics between transpositions (a full transposition graph).
- Batching, virtual loss or multithreading.
- Propagation of proven wins, losses and draws, so no `mate N` scores.
- Smart pruning, early stopping or extending past `soft`.

## 1. Node budget

- **How many playouts.** At 8+0.08 the soft limit is about 8000/30 + 60 ≈ 330 ms per move. The 64×6 int8 network takes about 264 µs per evaluation, so the search sees only about 1,000–1,300 playouts per move.
- **What matters at that budget.** Anything that makes those few playouts count is worth more here than for an engine searching 10⁵ nodes per move: FPU, cpuct, policy temperature, tree reuse and stopping rules.
- **What can wait.** Throughput tricks only pay if the CPU network gets faster per position when it evaluates in batches. That has to be measured first (§5.1).

## 2. Phase 1: search parameters (done; now tune them)

### 2.1 First-play urgency

**Problem with the original search.** An unvisited move scored Q = 0, which is a draw.
- In a lost position, every unexplored move looks better than every explored one, so the search spreads its few playouts over junk moves.
- In a won position it is the reverse: good alternatives are never tried.

**Options.**
- `FpuParentWeight` = 1: unvisited moves start at the parent's own value.
- `FpuReduction` > 0: they start lower the more prior mass has already been visited.
- Defaults: weight 1; reduction 0.44 in the tree and 1.0 at the root.

**Sign convention.** In our `Node`, Q is stored from the view of the side that moved into the node. A child of `node` is scored from the view of the side to move at `node`, which is `−node.q()`. A unit test covers this.

### 2.2 Exploration term

- **`CPuct`:** the base constant.
- **`CPuctFactor` / `CPuctBase`:** logarithmic growth with the parent's visits. With a large base (around 18k) the growth only matters at long time controls.
- **`CPuctExponent`:** the power of N; 0.5 = √N. Values around 0.8 with a small CPuct (around 0.16) are a different but equally valid regime: much more exploration at large N.
- **Gini scaling:** more exploration where the policy is sharp. One working setting is base 0.68, factor 1.63, max 2.1.
- **Why it's worth tuning:** neither the original 1.5 nor the current 2.15 was tuned for our network, and our policy comes from Gumbel training, which changes how sharp it is.

### 2.3 Policy temperature

- `softmax(logit / T)`, with one T for the root and one for the tree. Values between 1.3 and 1.6 are common.
- **Why it may matter here:** our policy targets are π' = softmax(logits + σ(completed Q)), which tend to be sharp. A sharp prior plus low cpuct means good second moves get almost no visits; the castling story was this effect inside self-play.

### 2.4 Tune them with SPSA

**How to run it:**
1. Start the engine and type `spsa`. It prints the current values, including any `setoption` you made, so you can start a tune from any point.
2. Paste the printed lines into the SPSA field of a new OpenBench tune. `c_end` is 7% of each value.
3. When the tune finishes, copy the final values into the `default` column of the `tunables!` table in `engine/src/search/params.rs`.
4. SPRT the tuned values against the current defaults.

**Which parameters to tune.** Leave out the lines of parameters you don't want tuned. Fewer parameters converge faster.

**SPSA is a local search.** It tunes around its starting point and won't move between very different regimes by itself. For example, CPuct 2.15 with exponent 0.84 explodes. The ranges are wide enough for other regimes. To try one:
1. SPRT it against the defaults, set through the dev engine's options. For example, N^0.84 with gini scaling: CPuct 16, CPuctExponent 84, CPuctFactor 0, CPuctGiniBase 68, CPuctGiniFactor 163, CPuctGiniMax 210, FpuReduction 0, FpuParentWeightRoot 0, FpuReductionRoot 0, PolicyTemp 130, PolicyTempRoot 130.
2. Run the SPSA from the winner.

**Practical notes:**
- Tune at the time control you test at (8+0.08). Settings tuned at fixed nodes don't carry over exactly.
- Budget about 20–40k games for 5–7 parameters.
- Whenever the network changes a lot (a new run, or the distilled net), re-tune at least cpuct and the temperatures. The best values depend on how sharp the policy is.

## 3. Phase 2: get more out of each second

### 3.1 Tree reuse (done)

**How it works** (`Mcts::take_reusable` in `mcts.rs`):
1. After each `go`, the tree is kept, together with the game it was searched for and the parameters used.
2. On the next `go`, if the new game is the saved game plus some moves (normally our move and the opponent's reply), those moves are followed down the tree from the old root.
3. If the node reached is expanded and not terminal, its subtree becomes the new root. Its visits and W/D/L sums carry over, so the search starts with that work already done.
4. **Compaction, in place:** the kept subtree is slid to the front of the same arena, with the new root at index 0.
   - Children always come after their parent in the arena, so one forward pass over a bitset marks the subtree.
   - A second forward pass moves each kept node to its rank among the kept nodes and remaps its child index. A node only ever moves to a lower index, so nothing is overwritten before it is read.
   - No second buffer, nothing freed node by node, and both passes are linear (microseconds).
5. **Root temperature:** the new root's children got their priors with `PolicyTemp`, but the root uses `PolicyTempRoot`. Because p ∝ exp(logit / T), the conversion is exact without storing logits: p_root ∝ p_tree^(T_tree / T_root), then renormalised.

**Starts fresh when:**
- there is no saved tree (first move, `ucinewgame`, a new `EvalFile`);
- the position isn't a continuation of the saved game;
- the parameters changed (`setoption` between moves);
- `go searchmoves` is used, which trims the root;
- the reached node was never expanded, or is terminal. A node marked as a draw by repetition is not a draw at the root, where a move must be found.

**Why repetition flags stay valid:** every node in the kept subtree has exactly the same game history as before, because the new root was reached along the path that was actually played. So its draw-by-repetition flags are still correct.

**Limits and output:**
- `go nodes N` counts this search's playouts, not the kept ones, as do `nodes` and `nps` in the `info` lines.
- `go depth` uses the average depth of this search's playouts.
- `info string tree reuse: N playouts kept` shows how much was carried over.
- `bench` always searches from a fresh tree, so its node count doesn't change.

**Why a tree and not a graph.** A transposition graph shares a position reached by different move orders, and lets a whole table survive between moves. Transpositions are common (§3.3: 5–40% of evaluations at 1,000 playouts), but the expensive part, the network call, is already shared through the transposition table without changing the tree. What a graph would add is sharing visits and values between the copies of a position. That needs a different node layout, a backup rule for positions with several parents, repetition and 50-move counters in the key, and garbage collection instead of compaction.

**Expected:** +20 to +50 Elo at short time controls. Test at the time control: fixed-node tests would overstate it, because reused playouts are free there.

### 3.2 Time management (basic version done)

**Budget per move** (`engine/src/timeman.rs`):
```
target = remaining / TmMovesToGo + TmIncPercent · increment     (≤ remaining / 2)
hard   = TmHardMult · target                                    (≤ 3/4 · remaining)
```
The GUI's `movestogo` replaces `TmMovesToGo` when it is sent. `go movetime` uses exactly its time.

**When to stop** (`Tree::time_up` in `mcts.rs`), checked after every playout:
```
share  = visits of the most visited root move / visits of all root moves
factor = clamp((TmVisitsBase − share) · TmVisitsMult, TmMinMult, TmMaxMult)
stop when elapsed ≥ target · factor, and always at hard
```
- A clear best move (a large share) means less time; visits spread over several moves mean more.
- Defaults: base 1.40 and multiplier 1.39, so a share of 70% gives 0.97 × target, 47% gives 1.29 × and 20% gives 1.67 ×. The clamp is 0.30–3.00, and `hard` (3 × target) caps it anyway.
- With tree reuse, the share includes the kept visits, so a move that was already clear after the previous search is played quickly.
- A single legal move is played after one playout.

**Output:** `info string time: X ms used, target T ms x factor F (best move P% of visits), hard H ms` after each timed move.

**One 8+0.08 self-play game (120 plies):** the factor ranged from 0.57 to 1.75 (mean 1.16), and the lowest clock was 0.7 s with the increment keeping it stable. Engine start-up (about 0.3 s, mostly reading the network) happens before `readyok`, so it isn't on the clock.

**Tunable** (`spsa`): `TmMovesToGo`, `TmIncPercent`, `TmHardMult`, `TmVisitsBase`, `TmVisitsMult`, `TmMinMult`, `TmMaxMult`. Tune at the time control you test at.

**Next steps** (each its own SPRT):
1. **PV stability:** more time when the best move's Q dropped during the search, or when the best move by visits isn't the best by Q:
   ```
   factor *= 1 + c · |Q(best) − Q(end of best PV)|
   ```
2. **Early stop:** stop before the target when the second-best move can't catch the best even with every remaining playout (`visits_best − visits_second > remaining · factor`, with a factor of about 1.3). The time saved stays on the clock.
3. **Best-move changes:** more time after the most visited move changed late in the search.

**Expected:** +15 to +40 Elo with all parts. Test at the real time control, never at fixed nodes.

### 3.3 Transposition table for network evaluations (done)

**Measured first** (per search, share of network evaluations whose exact network input had already been evaluated in that search):

| Bench position | 1,000 playouts | 10,000 playouts |
|---|---|---|
| Start position | 22% | 34% |
| Kiwipete | 16% | 29% |
| QGD | 23% | 40% |
| Italian middlegame | 6% | 13% |
| Middlegame (pos 5) | 3% | 7% |
| Rook endgame | 6% | 25% |
| K+P vs K+P | 38% | 71% |

**Why one key is enough.** The network sees only the current position, its repetition count and the 50-move counter (21 planes, no move history). So the key is (Zobrist hash, repetition count, 50-move counter), and every move order reaching that position gets the same entry.

**Compact entries (80 bytes).** Measured on 85k searched positions (mean 27 legal moves, max 62), the share of prior mass outside the best K moves:

| Moves kept | Tree, T 1.62 (mean / p99) | Root, T 2.12 (mean / p99) |
|---|---|---|
| top 8 | 15% / 39% | 23% / 48% |
| top 16 | 4.3% / 17% | 8.1% / 25% |
| top 24 | 1.1% / 7.3% | 2.5% / 13% |
| top 32 | 0.2% / 2.8% | 0.6% / 5.7% |

The tuned temperatures make the policy flat enough that dropping moves loses real exploration mass, and storing the top 32 with their move indices costs as much as storing all 64 at one byte each. So each entry keeps every legal move:
- 64-bit key, W and L as u16, the move count, and 64 one-byte logits.
- Each logit is its distance below the best logit, in steps of 1/16, clamped at about 16: at most 1/32 off, about 2% on a prior. A move 16 below the best has a prior under 0.01% after the temperature.
- The search always uses the quantised values, on a miss too. So it decides the same way whether or not a position was in the table, and the table size can't change the search, only its speed. (Bench moved once, from 2825 to 2827, when this was introduced.)

**How it works** (`engine/src/search/tt.rs`):
- A fixed array of entries, a quarter of `Hash` (16 MB → 4 MB ≈ 52k positions).
- A position with more than 64 legal moves uses the network's values directly and isn't stored (none in 85k searched positions).
- Always-replace; the slot index is the key scaled to the table size.
- A hit regenerates the legal moves and pairs them with the stored logits. A different move count means a key collision and is treated as a miss.
- Temperatures are applied after the lookup, so the table stays valid when the parameters change.
- The table survives between moves, which helps when tree reuse misses. It's cleared on `ucinewgame` and when the network changes, and reallocated (with the tree) when `Hash` changes.

**Output:** after each move, `info string tree reuse: N playouts kept; eval table: H of P lookups hit, F permille full`.

**Speed:** bench nps went from about 5,300–5,700 to 6,100–6,200. In a UCI test, 22–27% of the lookups hit in the opening and 58% in a K+P endgame.

**Expected:** +5 to +20 Elo at 8+0.08, more at longer time controls and in endgames. Test at the time control: at fixed nodes it changes nothing, by design.

### 3.4 Cheap per-playout speed-ups (profile first)

At 264 µs per evaluation, inference probably dominates, so measure before working on these:
- `Tree::evaluate` calls `pos.legal_moves()` for the terminal check, then `legal_actions()` generates the moves again inside `PolicyNet::evaluate`. Generate them once and pass them in.
- Every playout clones the whole `Game` (with its history) and allocates `path`. Reuse one scratch game and path buffer, or make and unmake moves.
- `Node` is already compact (24 bytes: move, child count, flags, prior, first-child index, visits, W and L sums).
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
| 1 | Tunable FPU, exploration term and policy temperatures as UCI options **(done; bench 5572)** | SPRT vs original | done (gain) | small |
| 2 | SPSA from the defaults (§2.4) | SPRT tuned vs #1 | +10–30 | games only |
| 3 | Tree reuse **(done)** | SPRT 8+0.08 | +20–50 | medium |
| 4 | Time management: visit-share factor + `hard` **(basic version done)**; then PV stability, early stop | SPRT 8+0.08, then SPSA TM constants | +15–40 | small–medium |
| 5 | Certainty propagation + mate scores | SPRT 8+0.08 | +5–20 | medium |
| 6 | Transposition table for evaluations **(done)** | SPRT 8+0.08 and 40+0.4 | +5–20 | small |
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
  - bench 5572 with the current defaults and network (4736 with the original-search values).
- **Tests to add with later items:**
  - a tree-reuse round-trip test: the reused subtree's visits equal those of the old child;
  - solver tests: mate in two found and reported as `mate 2`; the root proven and stopped early.
