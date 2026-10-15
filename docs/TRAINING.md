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
python train.py --run ..\runs\first --net 64x6
```

Stop it any time with Ctrl+C. Continue with:

```powershell
python train.py --run ..\runs\first --resume
```

Defaults: 1024 concurrent games, 16 considered moves per position, 100,000 new positions per generation, batch 512, K = 5, each position trained on ~4× on average, 1M-position replay buffer. See `python train.py --help`.

## Dashboard

`train.py` opens a live dashboard in your browser at <http://localhost:8765>: generation progress and ETA, the current phase (self-play positions / training steps with running loss), totals (positions trained and generated, games, replay buffer), and charts over all generations (total / policy / value loss incl. unrolled, agreement, W/D/B, how games end, game length, time per generation, self-play speed, learning rate), plus a table of the last 15 generations. Hover a chart to read exact values.

* `--no-browser` starts the server without opening a tab; `--dashboard-port N` picks another port (0 = off).
* Look at any run, also when it isn't training: `python -m peazero.dashboard --run ..\runs\first` (from `trainer/`).
* The ETA counts towards `--generations` (the *total* target, also on `--resume`) and uses the average of the last 5 generations.

## Run folder

```
runs/first/
  config.json          arguments of the first start
  checkpoint.pt        latest model + optimizer (used by --resume)
  metrics.jsonl        one line per generation
  status.json          live progress, read by the dashboard
  latest.pzn           copy of the newest network
  nets/gen_0001.pzn    every generation's network (kept, ~4 MB each)
  data/gen_0012/       self-play games of generation 12 (part_000.bin, ...)
```

Self-play data is only needed to refill the replay buffer on `--resume`, so older generations are deleted automatically. By default the newest `replay / positions-per-gen + 1` generations are kept (11 with defaults, roughly 250 MB). Use `--keep-data-gens 0` to keep everything, for example to re-analyse old games later, or `--keep-data-gens N` for a fixed number. Runs using the old flat `data/genXXXX_YYY.bin` layout are reorganised automatically on the next start.

## What a generation does

1. **Self-play.** All 1024 games make one move per step. The root is evaluated once (`initial`), Gumbel-Top-k picks 16 candidate moves, each candidate is evaluated once with the dynamics network (`recurrent`), and the move and policy target come from the Gumbel rules (docs/MUZERO.md §2, §6.2). Finished games are written to `runs/<name>/data/*.bin`.
2. **Training.** Batches are sampled from the replay buffer and trained with the K = 5 unrolled MuZero loss.
3. **Export.** `runs/<name>/nets/gen_XXXX.pzn` plus `checkpoint.pt`.

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

## Known limits (next sprint candidates)

* The Rust side of self-play is single-threaded (~11k moves/s on one core). With a 64×6 network the GPU and Rust are roughly balanced; threading the pool is the next easy speed-up.
* Self-play and training alternate in one process; overlapping them would make better use of the GPU.
