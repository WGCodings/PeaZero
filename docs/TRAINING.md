# Training PeaZero (Sprint 4)

Self-play and training run on the RTX 3070 machine. The engine itself stays searchless: it plays the network's best legal policy move.

## One-time setup (Windows, PowerShell)

```powershell
cd C:\Users\warre\RustroverProjects\PeaZero
pip install -r trainer\requirements.txt
# torch must be the CUDA build (see pytorch.org), e.g.:
#   pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install .\crates\py          # compiles the Rust self-play core into the `peazero_rs` module
python -c "import peazero_rs, torch; print(peazero_rs.ENCODING_VERSION, torch.cuda.is_available())"
```

Re-run `pip install .\crates\py` whenever anything under `crates/` changes.

## Run

```powershell
cd trainer
python train.py --run ..\runs\fourth
```

Stop it any time with Ctrl+C. Continue with:

```powershell
python train.py --run ..\runs\fourth --resume
```

**Settings live with the run.** The first start stores all settings in `runs/<name>/config.json`; `--resume` reuses them, so you don't repeat flags. A flag on the command line overrides the stored value and is saved (the startup `settings:` line marks it "changed"), e.g. `--resume --generations 2000`. Settings an older run never recorded keep that run's original behaviour, so resuming `second` or `third` doesn't change how they train.

Defaults for a new run (`python train.py --help`):

| setting | default | meaning |
|---|---|---|
| `--simulations` | 64 | tree-search simulations per move (0 = old root-only mode) |
| `--considered` | 16 | root candidates (Sequential Halving width) |
| `--noise-plies` / `--late-noise` | 30 / 0.0 | full Gumbel noise for the first 30 plies of a game, then none: opening variety, but games are played out properly so the result reflects who was better |
| `--fast-simulations` / `--full-prob` / `--fast-considered` | 0 (off) / 0.25 / 8 | playout cap randomization: 25 % of moves get the full search (`--simulations`), the rest a fast search that is not a training sample; see below |
| `--q-ratio` | 0.5 | value target = 50 % game result + 50 % search value of the position (much less noisy than the result alone) |
| `--unroll` | 0 | AlphaZero-style training: no dynamics unroll (self-play and the engine don't use the dynamics network); training ~4× cheaper than K = 5 |
| `--reuse` | 2.0 | average times each new position is trained on |
| `--games`, `--positions-per-gen`, `--batch`, `--replay` | 1024, 100k, 512, 1M | |
| `--qat` | off | quantisation-aware training: the trainer simulates the engine's int8 residual blocks, so the network learns weights that still play well after rounding (below); `--no-qat` turns it off again |
| `--viriformat` | off | keep all self-play games: each generation is appended to `runs/<name>/<name>.vf` (viriformat) before its data folder is deleted; `--no-viriformat` turns it off again |
| `--keep-nets` | 25 | keep the network of every 25th generation (plus every evaluation milestone and the newest), delete the rest, also the ones already on disk; 0 = keep all |
| `--eval-every` | 25 | strength evaluation every 25 generations (match + puzzles, below); 0 = off |
| `--match-games` | 100 | games per evaluation match (0 = puzzles only) |
| `--match-tc` / `--match-search-mode` | 8+0.08 / Mcts | evaluation matches: the engine binary with its own search at this clock; `none` = in-process match without clock |
| `--match-simulations` | 0 | only with `--match-tc none`: search simulations per move (0 = searchless) |
| `--eval-threads` | 4 | CPU threads for the evaluation = games played at the same time |
| `--engine` | newest `target/release/peazero` or `./peazero` | engine binary for the matches (this session only) |
| `--puzzles` | `trainer/data/puzzles.csv` | puzzle set (this session only; `none` = no puzzles) |
| `--pools` | 2 | self-play speed, same results (this session only): the games are split over 2 pools so the GPU evaluates one while the CPU searches the other; 1 = one pool; 0 = the old plain loop. See Speed |
| `--cuda-graphs` | on | self-play inference as recorded CUDA graphs (one launch per batch); `--no-cuda-graphs` = eager PyTorch (this session only) |
| `--channels-last` | on | self-play inference in NHWC memory layout (cuDNN tensor-core kernels, 1.25× on the 3070); `--no-channels-last` = normal layout (this session only) |

## Keeping the games (viriformat)

With `--viriformat` every generation's games are converted to **viriformat** (the Viridithas game format that bullet and other NNUE trainers read) and appended to one file per run, `runs/<name>/<name>.vf`, right before the generation's `.bin` folder is deleted. The `.bin` files keep getting deleted as before; nothing is lost.

* Per game: the start position with the result (black win / draw / white win; games stopped at the ply limit count as draws), then every move with a **white-relative score** in centipawns: the position's search value from self-play (W − L of the 64-simulation search), converted with the same curve as the engine's `score cp`.
* Size: ~4 bytes per position, about 0.4 MB per generation (100k positions).
* Checked against the reference implementation (the `viriformat` crate): the spec example byte for byte, and every move of 300 random games decodes legally to the same positions.
* Appends are crash-safe (`<name>.vf.json` records the size after each complete generation; an interrupted append is cut off next time). If you move the `.vf` away, a new one starts; generations already exported aren't repeated.
* Data folders still on disk (the newest ~20 generations) are exported when they are deleted later, or by hand while training is stopped: `python -m peazero.vf_export --run ..\runs\fourth`.
* Generations deleted before you turned this on can't be recovered.

## Playout cap randomization (`--fast-simulations`)

Deep searches give better policy targets; many games give better value targets. Playout cap randomization (KataGo) gets both: each move is a **full** search with probability `--full-prob`, otherwise a cheap **fast** search that only keeps the game going and is not a training sample (docs/MUZERO.md §6.5).

* Suggested start: `--simulations 200 --fast-simulations 32 --full-prob 0.25 --fast-considered 8`. That costs ~75 network evaluations per move (vs 65 now), but every policy target comes from a 200-simulation search instead of 64.
* Only 1 in 4 moves is a training sample, so with the same `--positions-per-gen` a generation would take ~4× more moves. `--positions-per-gen` and `--replay` count training samples: use ~25,000 per generation (similar self-play time per generation as 100k now) and `--replay 500000` (the same ~20 generations of history).
* Old data (format v2) still loads: every ply of it is a training sample. New data is format v3; the viriformat export still contains every move.
* `metrics.jsonl` logs `full_search_share` (should be ≈ `--full-prob`); `new_positions` counts training samples, `avg_game_length` all moves.
* Measure the trade-off on your machine first: `python scripts\bench_selfplay.py --run ..\runs\fourth --pcr` prints moves/s and training samples/s for the current setting, 200 simulations on every move, and two PCR settings.

## Quantisation-aware training (`--qat`)

On CPUs with AVX2 the engine runs the residual blocks in int8 (docs/MUZERO.md §5.1c). Rounding a float-trained net costs accuracy: gen 2000 of `fourth` matches its own float output's best move in only 95 % of positions. With `--qat` the trainer rounds the same way in every training step, so the network adapts to the rounding instead of being surprised by it.

* What it simulates, exactly like `crates/nn/src/int8.rs`: weights rounded to −127..127 with one scale per output channel (max |w| / 127), activations rounded to 0..127 with one scale per convolution input and per position (max / 127). Only the 12 residual-block convolutions of the representation network: stem, heads, bias, residual and ReLU stay float in the engine too. The PyTorch simulation matches the engine's int8 output to ~1e-5 (checked on 300 positions).
* How it trains through rounding: straight-through estimator. The forward pass sees the rounded values, the backward pass treats rounding as the identity, so the float "shadow" weights keep getting gradients and drift toward values whose rounded version gives the lowest loss.
* Only the training steps simulate int8; self-play uses the float network at full speed (the training loss is what teaches the weights to survive rounding; simulating it in self-play too only costs speed). BatchNorm statistics are still measured through the int8 simulation. The exported `.pzn` keeps float weights; the engine rounds them to the same integers. Float-only CPUs (no AVX2) run the float weights, which stay a good network.
* Cost: about 10–25 % slower training steps (a few extra element-wise operations per convolution); self-play unchanged. No change to the checkpoint format, so it can be switched on or off on `--resume` at any time.
* When: for the last stretch of a run, together with a low learning rate: `python train.py --run ..\runs\fourth --resume --lr 1e-4 --qat`. Rounding steps are small (~1/127 of a channel's largest weight), so the net can only settle into them when the weights stop moving much; at high learning rates the updates just jump across them.
* Measured (offline, gen 2000 of `fourth` fine-tuned toward its own float output on random-game positions, Adam lr 1e-5, 600 steps of 256): int8 vs float policy KL 0.0096 → 0.0073 (−25 %), value |W−L| difference 0.025 → 0.011 (−55 %); a float-trained control stayed at 0.0094 / 0.026. Same best move stayed ~95 %: most of the remaining disagreements are near-ties between two moves. At lr 1e-4 this fine-tuning got worse (the loss being matched is tiny, so batch noise dominates). In real training the gain should be larger than in this small test, but the SPRT decides: QAT net in int8 vs the plain net in int8.
* Every generation logs `int8_agreement` (same best move, simulated int8 vs float, on replay positions) and `int8_value_diff` (mean |W−L| difference), with or without `--qat`; the console line shows `int8 xx.x%`. The `exported p/v` losses are the float network's.

## Strength evaluation

The losses can't tell you how strong the network is: the targets get harder as the search gets better, and the value loss even rises as games become more decisive. So every `--eval-every` generations (a *milestone*: 25, 50, 75, …) a background process measures it directly, **on the CPU** (the engine's own SIMD inference) at lower priority, while the GPU keeps training:

* **Match:** gen N vs gen N − 25, `--match-games` games with the **real engine** over UCI, each side with its own `EvalFile`, its own search (`SearchMode`, MCTS by default) and its own time management at `--match-tc 8+0.08`, like an OpenBench test. Random 6-ply openings (the same set at every milestone), each played twice with colours swapped; `--eval-threads` games at a time, one engine thread each. The clock is kept like cutechess does (a side whose time runs out loses; time losses and engine errors are flagged in the console). The result is an Elo difference with a 95 % interval from the game pairs. Needs the engine built (`cargo build --release` or `make`; the binary is copied before each match, so rebuilding meanwhile is fine) and python-chess (`pip install chess`). Without a binary the evaluator uses the in-process match (no clock, searchless or `--match-simulations`) and switches over as soon as the binary exists; `--match-tc none` always uses the in-process match.
* **Elo ladder:** the matches played under the same conditions (time control + search mode, or in-process settings), chained (gen 25 = 0, gen 50 = Elo(50 vs 25), gen 75 = that + Elo(75 vs 50), …). As long as it keeps climbing, the network keeps getting stronger. Self-play Elo is inflated compared to Elo against other engines, so read it as a trend, not a rating. The interval widens along the ladder (the errors add up).
* **Puzzles:** a fixed set of Lichess puzzles solved by the raw policy (searchless): the share solved, the share whose first move is right, the share per rating band, and a **puzzle rating** (the Lichess puzzle rating at which that share would be expected). This compares against an outside reference, not the run's own targets, so it is the steadiest progress signal between matches.

One-time setup for the puzzles (Lichess puzzle database, CC0; streams ~250 MB and keeps ~3,250 puzzles, 250 per 200-point band from 400 to 3000):

```powershell
pip install zstandard
python scripts\make_puzzles.py          # writes trainer\data\puzzles.csv
```

Keep that file fixed: the curves are only comparable on the same set. Without it the evaluator still runs the matches.

Results go to `runs/<name>/evals.jsonl` and the dashboard's **Strength** section (Elo ladder, Elo per milestone, puzzle curves, a table of the latest evaluations); the console shows one `[eval]` line per result. The evaluator catches up on missed milestones after a restart. When training reaches `--generations` it waits for the last milestones (Ctrl+C skips). Evaluate an existing run, e.g. backfill `third` (from `trainer/`):

```powershell
python -m peazero.evaluator --run ..\runs\third --once                  # every 25 generations, then exit
python -m peazero.evaluator --run ..\runs\third --once --every 100 --games 200
```

Cost with 4 threads: the 100-game 8+0.08 match takes about 8–10 minutes (fits easily in 25 generations of ~1.5 minutes each); the puzzles a few seconds. The engines run at lower priority next to self-play, so on a busy CPU they search a bit less than on an idle machine. That hits both sides equally, so the Elo difference stays fair, but don't compare node counts with OpenBench. The in-process searchless match takes seconds, but deterministic searchless play often repeats into draws.

## Dashboard

`train.py` opens a live dashboard in your browser at <http://localhost:8765>: generation progress and ETA, the current phase (self-play positions / training steps with running loss), totals (positions trained and generated, games, replay buffer), and charts over all generations (total / policy / value loss incl. unrolled, agreement, W/D/B, how games end, game length, time per generation, self-play speed, learning rate), plus a table of the last 15 generations. Hover a chart to read exact values.

* `--no-browser` starts the server without opening a tab; `--dashboard-port N` picks another port (0 = off).
* Look at any run, also when it isn't training: `python -m peazero.dashboard --run ..\runs\first` (from `trainer/`).
* The ETA counts towards `--generations` (the *total* target, also on `--resume`) and uses the average of the last 5 generations.

## Run folder

```
runs/first/
  config.json          the run's settings (reused by --resume)
  checkpoint.pt        latest model + optimizer (used by --resume)
  metrics.jsonl        one line per generation
  status.json          live progress, read by the dashboard
  latest.pzn           copy of the newest network
  nets/gen_0025.pzn    every 25th generation's network (--keep-nets), ~4 MB each
  first.vf             all exported games in viriformat (--viriformat)
  first.vf.json        which generations are in it
  evals.jsonl          strength evaluation: one line per match / puzzle run
  eval_status.json     what the evaluator is doing, read by the dashboard
  data/gen_0012/       self-play games of generation 12 (part_000.bin, ...)
```

Self-play data is only needed to refill the replay buffer on `--resume`, so older generations are deleted automatically. By default the newest `replay / positions-per-gen + 1` generations are kept (11 with defaults, roughly 250 MB). Use `--keep-data-gens 0` to keep everything, for example to re-analyse old games later, or `--keep-data-gens N` for a fixed number. Runs using the old flat `data/genXXXX_YYY.bin` layout are reorganised automatically on the next start.

## What a generation does

1. **Self-play.** Every move runs a Gumbel MuZero tree search with `--simulations` (default 64) leaf evaluations on the *real* board: Rust makes the moves, the `initial` network evaluates each new leaf, checkmates and draws are scored exactly. The root uses Sequential Halving over the top `--considered` (16) moves, so the best candidates are searched several plies deep. The move played and the policy target come from the search (docs/MUZERO.md §6.3). All games advance in lock-step, one simulation per step, so the GPU always sees batches of `--games` positions. `--simulations 0` selects the old root-only mode (one dynamics-network evaluation per candidate, §6.2). Every recorded position also stores the search's value (W/D/L for the side to move: W−L from the root search value, D from the network), used by `--q-ratio`. Finished games are written to `runs/<name>/data/` (data format v3; v1/v2 files from older runs are still read). With `--fast-simulations` only some moves get the full search (playout cap randomization, below).
2. **Training.** Batches are sampled from the replay buffer and trained with the K = 5 unrolled MuZero loss.
3. **Export.** BatchNorm statistics are recalibrated (see below), then `runs/<name>/nets/gen_XXXX.pzn` plus `checkpoint.pt` are written.

### BatchNorm calibration

Training normalises every network call with its own batch statistics, but BatchNorm's running averages mix root positions and dynamics (unrolled) latents, which the shared prediction head sees with very different distributions. In eval mode (self-play, engine) that mix fits neither. In run `first` this made the exported value head worse than guessing (value loss 1.6 vs. 0.7 in training) and the children's Q in self-play uncorrelated with the real positions, so the Gumbel policy targets were mostly noise and strength peaked around generation 200.

Since then, after each generation `calibrate_bn` recomputes the statistics for each graph separately: `initial` from root positions, `recurrent` from first-step children. Self-play and the exported `.pzn` use these copies. `metrics.jsonl` logs the exported net's losses as `eval_policy_loss` / `eval_value_loss` (+ `_k1` for the first dynamics step); they should stay close to the training losses, and the dashboard plots them next to them.

## Reading the log

```
gen   12 | selfplay  20.1s  5100 mv/s  520 games (W80 D360 B80, len 192.3) | train 781 steps 46.0s loss 6.2 (p 4.9 v 0.8 agree 0.31) | ...
```

| field | meaning | what to expect early on |
|---|---|---|
| `W/D/B`, `len` | self-play results and average game length | mostly draws at first (50-move rule, ply limit); decisive games and shorter games are a sign of progress |
| `p` | policy loss at the root (cross-entropy vs. the Gumbel target) | falls steadily |
| `v` | value loss at the root (W/D/L cross-entropy) | low at first because "draw" is easy to predict; may *rise* when decisive games appear, which is fine |
| `agree` | how often the network's top move equals the target's top move | rises |
| `policy_loss_unrolled`, `value_loss_unrolled` (in `metrics.jsonl`) | the same losses on the dynamics network's imagined positions | should follow the root losses; if they stay much higher, the dynamics network isn't learning |

## Testing a network

* Locally: `setoption name EvalFile value C:\...\runs\first\nets\gen_0010.pzn`, or embed it with `make EXE=pz EVALFILE=...`.
* On OpenBench: upload the `.pzn` as a network for PeaZero and SPRT it against the previous one (or the random-init `networks/default.pzn`). The bench (231) doesn't depend on the network.

## Speed

Where a generation's time goes (64×6, RTX 3070): self-play ~12 s, training ~30–38 s.

* **Self-play is GPU-bound.** With tree search every move costs `simulations + 1` network evaluations (65 by default), in batches of `--games` positions; expect roughly 2k moves/s on the 3070 instead of ~11k in root-only mode, so self-play takes ~4–5× longer per generation. The Rust side (tree, move making, encoding) is ~2 ms per round of 1024 games with 8 threads; `--threads N` spreads it over CPU cores (0 = all, the default) without changing the games played. Larger `--games` (e.g. 2048) gives the GPU bigger batches and may raise moves/s.
* **Training** samples the next batch while the GPU is still busy, keeps metrics on the GPU and uses fused AdamW, so the GPU no longer waits for the CPU between steps.
* **Bigger batches** are the main remaining lever for a small net: `--batch 1024` halves the number of steps (steps = new positions × reuse / batch). Check the gain first with `python scripts/bench_gpu.py --sizes 64x6 --train-batch 1024`; when switching, raise the learning rate a bit (e.g. `--lr 1.4e-3`) and SPRT the result, since it changes how the net learns. Both flags can change on `--resume`.

Stored settings: `--resume` reuses `config.json`; a flag on the command line overrides it and is saved. Starting **without** `--resume` writes the defaults into `config.json` (and starts a new network), so always pass `--resume` for an existing run.

### Self-play pipeline (`peazero/selfplay_gpu.py`)

One self-play step = one simulation for every game: Rust encodes the positions to evaluate, the GPU evaluates them, Rust feeds the results into the searches. Three things make a step cheaper; none changes what is played (checked: with one pool the games are byte-identical to the old loop):

* **Pinned buffers**: Rust writes the requests straight into page-locked memory (`requests_into`) and results come back into page-locked memory, so the GPU copies are asynchronous and nothing is allocated per step.
* **CUDA graphs** (`--cuda-graphs`, default on): the forward pass and the legal-move gather are recorded once per network and replayed with one launch instead of ~50; re-recorded after every generation (milliseconds). Falls back to normal inference if recording fails.
* **Two pools** (`--pools 2`, default): while the GPU evaluates pool A, Rust works on pool B, and vice versa. Each pool has half of `--games`, so the GPU batch halves; `--games 2048` keeps batches of 1024.

The console line shows where self-play time went: `gpu wait` (blocked on the GPU: GPU-bound), `rust` (tree search work: CPU-bound), `launch` (Python issuing copies and kernels); `metrics.jsonl` has them as `sp_*_seconds`.

Measure on your machine (stop training first, ~4 minutes):
```powershell
python scripts\bench_selfplay.py --run ..\runs\fourth
```
It prints the network's raw speed with and without CUDA graphs, checks that both give the same outputs, and the moves/s of the old loop vs. each combination (`--only-new`: just the default vs. `--channels-last`).

Measured on the RTX 3070 (64×6, gen 2261 of `fourth`, 1024 games, 64 simulations, 8 threads):

| variant | moves/s | |
|---|---|---|
| plain loop (before) | 2014 | |
| pinned, 1 pool | 2262 | 1.12× |
| pinned + graphs, 1 pool | 2318 | 1.15× |
| pinned, 2 pools (no graphs) | 2003 | 0.99×: Python spends 82 % of the time launching kernels |
| **pinned + graphs, 2 pools (default)** | **2740** | **1.36×** |
| same with 2048 games | 2867 | 1.42× |
| **+ channels_last (default)** | **3501** | **1.74×** |

Network alone at batch 512: eager 156k evaluations/s, CUDA graph 177k, graph + channels_last 228k (identical outputs). Self-play is limited by the network itself: 3501 moves/s × 65 evaluations ≈ 228k evaluations/s, exactly that speed; the Rust work is fully hidden behind it (63 % of the time is waiting for the GPU). Further self-play gains need a faster network on the GPU or fewer evaluations per move (tree reuse, playout cap randomization). 2048 games adds 5 % but games then span more generations; 1024 stays the default.

## Known limits (next sprint candidates)

* Self-play and training alternate. Overlapping them gains little while both are GPU-bound; it pays off once self-play is cheaper per move or a second GPU is available.
