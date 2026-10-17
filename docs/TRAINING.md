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
| `--q-ratio` | 0.5 | value target = 50 % game result + 50 % search value of the position (much less noisy than the result alone) |
| `--unroll` | 0 | AlphaZero-style training: no dynamics unroll (self-play and the engine don't use the dynamics network); training ~4× cheaper than K = 5 |
| `--reuse` | 2.0 | average times each new position is trained on |
| `--games`, `--positions-per-gen`, `--batch`, `--replay` | 1024, 100k, 512, 1M | |
| `--eval-every` | 25 | strength evaluation every 25 generations (match + puzzles, below); 0 = off |
| `--match-games` / `--match-simulations` | 100 / 0 | games per evaluation match (0 = puzzles only) / search simulations in those games (0 = searchless) |
| `--eval-threads` | 4 | CPU threads for the evaluation |
| `--puzzles` | `trainer/data/puzzles.csv` | puzzle set (this session only; `none` = no puzzles) |

## Strength evaluation

The losses can't tell you how strong the network is: the targets get harder as the search gets better, and the value loss even rises as games become more decisive. So every `--eval-every` generations (a *milestone*: 25, 50, 75, …) a background process measures it directly, **on the CPU** (tract, the engine's own inference) at lower priority, while the GPU keeps training:

* **Match:** gen N vs gen N − 25, `--match-games` games. Random 6-ply openings (the same set at every milestone), each played twice with colours swapped; searchless by default, like your OpenBench searchless tests. The result is an Elo difference with a 95 % interval from the game pairs.
* **Elo ladder:** the matches chained (gen 25 = 0, gen 50 = Elo(50 vs 25), gen 75 = that + Elo(75 vs 50), …). As long as it keeps climbing, the network keeps getting stronger. Self-play Elo is inflated compared to Elo against other engines, so read it as a trend, not a rating. The interval widens along the ladder (the errors add up).
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

Cost with 4 threads (64×6, ~800 evaluations/s per core): the searchless 100-game match is ~12k evaluations, a few seconds; the puzzles about as much. `--match-simulations 64` gives more decisive games (deterministic searchless play often repeats into draws) but needs 65× the evaluations, roughly 4–5 minutes per milestone, still fine every 25 generations.

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
  nets/gen_0001.pzn    every generation's network (kept, ~4 MB each)
  evals.jsonl          strength evaluation: one line per match / puzzle run
  eval_status.json     what the evaluator is doing, read by the dashboard
  data/gen_0012/       self-play games of generation 12 (part_000.bin, ...)
```

Self-play data is only needed to refill the replay buffer on `--resume`, so older generations are deleted automatically. By default the newest `replay / positions-per-gen + 1` generations are kept (11 with defaults, roughly 250 MB). Use `--keep-data-gens 0` to keep everything, for example to re-analyse old games later, or `--keep-data-gens N` for a fixed number. Runs using the old flat `data/genXXXX_YYY.bin` layout are reorganised automatically on the next start.

## What a generation does

1. **Self-play.** Every move runs a Gumbel MuZero tree search with `--simulations` (default 64) leaf evaluations on the *real* board: Rust makes the moves, the `initial` network evaluates each new leaf, checkmates and draws are scored exactly. The root uses Sequential Halving over the top `--considered` (16) moves, so the best candidates are searched several plies deep. The move played and the policy target come from the search (docs/MUZERO.md §6.3). All games advance in lock-step, one simulation per step, so the GPU always sees batches of `--games` positions. `--simulations 0` selects the old root-only mode (one dynamics-network evaluation per candidate, §6.2). Every recorded position also stores the search's value (W/D/L for the side to move: W−L from the root search value, D from the network), used by `--q-ratio`. Finished games are written to `runs/<name>/data/` (data format v2; v1 files from older runs are still read).
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

All flags are read from the command line on every start (`config.json` only records the first one), so repeat them on `--resume`.

## Known limits (next sprint candidates)

* Self-play and training alternate. Overlapping them gains little while both are GPU-bound; it pays off once self-play is cheaper per move or a second GPU is available.
