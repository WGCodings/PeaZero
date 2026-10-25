#!/usr/bin/env python3
"""PeaZero training loop: alternate tree-search self-play and training.

    python train.py --run runs/fourth                  # new run, default settings
    python train.py --run runs/fourth --resume         # continue with the run's settings
    python train.py --run runs/fourth --resume --generations 2000   # ... and change one

Each generation:
  1. self-play until --positions-per-gen new positions are recorded
  2. train ~positions × --reuse / --batch steps on the replay buffer
  3. save the checkpoint and export the network

Run directory layout:
  <run>/config.json            the run's settings (reused by --resume)
  <run>/checkpoint.pt          latest model + optimizer (used by --resume)
  <run>/metrics.jsonl          one line per generation
  <run>/status.json            live progress (read by the dashboard)
  <run>/latest.pzn             copy of the newest network
  <run>/nets/gen_XXXX.pzn      network of every --keep-nets-th generation (+ the newest)
  <run>/evals.jsonl            strength evaluation (matches, puzzles; see below)
  <run>/data/gen_XXXX/*.bin    self-play games (data format v2)

--qat (quantisation-aware training): the engine runs the residual blocks in
int8 (crates/nn/src/int8.rs); with --qat the trainer rounds weights and
activations the same way in every training step, so the network learns to
play well after rounding (self-play stays float, at full speed). Best switched on for a run's last stretch, e.g.
together with the final learning-rate drop. Every generation logs
int8_agreement (same best move, int8 vs float) either way.

Settings: every run stores its settings in config.json. --resume reuses
them; a flag given on the command line overrides the stored value and is
saved. Settings that an older run never recorded keep that run's original
behaviour (LEGACY below), so resuming an old run doesn't change it.

Old data is deleted automatically once the replay buffer can no longer use
it (see --keep-data-gens); with --viriformat it is appended to
<run>/<run name>.vf first (peazero/vf_export.py). Networks: every --keep-nets-th generation and
every evaluation milestone are kept, plus the newest; the rest is deleted.

A live dashboard (statistics, charts, ETA) opens at http://localhost:8765
while training; see --dashboard-port / --no-browser.

Strength: every --eval-every generations a background CPU process
(peazero/evaluator.py) plays gen N against gen N - eval-every
(--match-games games with the engine binary at --match-tc 8+0.08, MCTS)
and runs the puzzle set (trainer/data/puzzles.csv, made by
scripts/make_puzzles.py). The GPU keeps training meanwhile; the
Elo ladder and puzzle curves appear on the dashboard. --eval-every 0 = off.
"""
import argparse
import collections
import copy
import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import peazero_rs as pz  # noqa: E402  (pip install ./crates/py)
from peazero import dashboard  # noqa: E402
from peazero import encoding as enc  # noqa: E402
from peazero.model import MuZeroNet, NetConfig, export_pzn  # noqa: E402
from peazero.selfplay_gpu import SearchRunner  # noqa: E402
from peazero.train_lib import (  # noqa: E402
    batch_to_device, calibrate_bn, compare_losses, eval_metrics, muzero_loss, search_step, selfplay_step)

SIZES = {"32x4": (32, 4), "64x6": (64, 6), "96x8": (96, 8), "128x6": (128, 6), "128x10": (128, 10)}


def net_size(s):
    """'96x8' -> (96, 8): channels x residual blocks (any size, not only SIZES)."""
    m = re.fullmatch(r"(\d+)x(\d+)", s)
    if not m:
        raise argparse.ArgumentTypeError(f"--net: want CHANNELSxBLOCKS like 96x8, got {s!r}")
    return int(m.group(1)), int(m.group(2))


# Settings of a new run. All of these are stored in <run>/config.json.
DEFAULTS = {
    "net": "64x6", "generations": 1000,
    # self-play
    "games": 1024, "simulations": 64, "considered": 16, "max_plies": 512, "threads": 0,
    "fast_simulations": 0, "full_prob": 0.25, "fast_considered": 8,
    # game variety (all off by default): opening book + random moves, lc0-style move temperature
    "book": None, "book_prob": 0.75, "random_prob": 0.0, "random_plies": 2,
    "temperature": 0.0, "temp_decay_plies": 30, "temp_endgame": 0.0, "temp_value_cutoff": 0.1,
    # exploration floor at the root: no move's prior can collapse beyond rediscovery
    "root_explore": 0.25,
    "positions_per_gen": 100_000, "noise_plies": 30, "late_noise": 0.0,
    # training
    "batch": 512, "unroll": 0, "reuse": 2.0, "lr": 1e-3, "weight_decay": 1e-4, "warmup": 500,
    "replay": 1_000_000, "keep_data_gens": -1, "keep_nets": 25, "viriformat": False,
    "q_ratio": 0.5, "qat": False, "seed": 0,
    # warm start: self-play by a stronger existing network until this one catches up
    "teacher": None, "teacher_gens": 200,
    # background strength evaluation (CPU)
    "eval_every": 25, "match_games": 100, "match_tc": "8+0.08", "match_search_mode": "Mcts",
    "match_simulations": 0, "eval_threads": 4,
}
# What runs started before a setting existed did (used when their
# config.json lacks the key).
LEGACY = {"root_explore": 0.0, "simulations": 0, "unroll": 5, "reuse": 4.0, "threads": 0,
          "noise_plies": 0, "late_noise": 0.0, "q_ratio": 0.0}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run directory")
    ap.add_argument("--resume", action="store_true", help="continue the run with its stored settings")
    ap.add_argument("--net", help="network size CHANNELSxBLOCKS, e.g. 64x6, 96x8, 128x10")
    ap.add_argument("--generations", type=int, help="total generations to reach")
    # self-play
    ap.add_argument("--games", type=int, help="concurrent self-play games")
    ap.add_argument("--simulations", type=int,
                    help="tree-search simulations per move on the real board (Gumbel MuZero search, "
                         "AlphaZero-style evaluation); 0 = old root-only mode through the dynamics network")
    ap.add_argument("--considered", type=int,
                    help="Gumbel m: root candidates (Sequential Halving width; root-only mode: evaluations per move)")
    ap.add_argument("--fast-simulations", type=int,
                    help="playout cap randomization: each move is a full search (--simulations, --considered) "
                         "with probability --full-prob, otherwise a fast search with this many simulations "
                         "that is not a training sample; 0 = off (every move full)")
    ap.add_argument("--full-prob", type=float, help="share of moves with the full search (playout cap randomization)")
    ap.add_argument("--fast-considered", type=int, help="Sequential Halving width of a fast search")
    ap.add_argument("--book", help="opening book (.pgn lines or .epd/.fen positions): with probability --book-prob a "
                                   "new game starts after 1..all moves of a random line ('none' = no book)")
    ap.add_argument("--book-prob", type=float, help="share of games that start from the book")
    ap.add_argument("--random-prob", type=float,
                    help="share of games that then play 1..--random-plies uniformly random moves (not recorded)")
    ap.add_argument("--random-plies", type=int, help="at most this many random moves after the opening")
    ap.add_argument("--temperature", type=float,
                    help="lc0-style: sample the move played from the search policy with this temperature "
                         "(decays over --temp-decay-plies to --temp-endgame); 0 = always the best move")
    ap.add_argument("--temp-decay-plies", type=int, help="plies over which the temperature decays to 0 (0 = constant)")
    ap.add_argument("--temp-endgame", type=float, help="lowest temperature for the rest of the game")
    ap.add_argument("--temp-value-cutoff", type=float,
                    help="only sample moves whose search value (W-L, -1..1) is within this of the best move")
    ap.add_argument("--root-explore", type=float,
                    help="root exploration floor: root candidates are drawn from (1-e)*prior + e/legal moves, "
                         "so a move the policy has written off (e.g. castling) still gets searched; "
                         "0.25 for new runs, 0 = off")
    ap.add_argument("--max-plies", type=int)
    ap.add_argument("--threads", type=int,
                    help="CPU threads for the Rust self-play work (0 = all logical CPUs); "
                         "does not change the games played")
    ap.add_argument("--positions-per-gen", type=int)
    ap.add_argument("--noise-plies", type=int,
                    help="plies per game with full Gumbel noise, then --late-noise (0 = full noise all game)")
    ap.add_argument("--late-noise", type=float, help="Gumbel noise scale after --noise-plies (0 = none)")
    # training
    ap.add_argument("--batch", type=int)
    ap.add_argument("--unroll", type=int,
                    help="dynamics unroll steps K in the loss; 0 = AlphaZero-style (no dynamics training)")
    ap.add_argument("--reuse", type=float, help="avg times each new position is sampled")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--weight-decay", type=float)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--replay", type=int, help="replay buffer size in positions")
    ap.add_argument("--keep-nets", type=int,
                    help="keep the network of every N-th generation (and every evaluation milestone, "
                         "and the newest); others are deleted. 0 = keep all")
    ap.add_argument("--viriformat", action=argparse.BooleanOptionalAction,
                    help="append every generation's games to <run>/<run name>.vf (viriformat) before its "
                         "data folder is deleted; --no-viriformat turns it off")
    ap.add_argument("--keep-data-gens", type=int,
                    help="generations of self-play data kept on disk; -1 = just enough to refill the "
                         "replay buffer on --resume, 0 = keep everything")
    ap.add_argument("--q-ratio", type=float,
                    help="value target = (1-q)·game result + q·search value (needs data format v2)")
    ap.add_argument("--qat", action=argparse.BooleanOptionalAction,
                    help="quantisation-aware training: simulate the engine's int8 residual blocks in the "
                         "training steps (self-play stays float) so the net learns weights that survive "
                         "rounding; --no-qat turns it off")
    ap.add_argument("--teacher",
                    help="warm start: checkpoint.pt of an existing (stronger) network that plays the self-play "
                         "games while this network trains on them; self-play switches to this network once its "
                         "loss on fresh data beats the teacher's, or after --teacher-gens generations")
    ap.add_argument("--teacher-gens", type=int, help="at most this many generations of teacher self-play")
    ap.add_argument("--seed", type=int)
    # strength evaluation
    ap.add_argument("--eval-every", type=int,
                    help="every N generations: match vs the net N generations older + puzzles, in a "
                         "background CPU process (0 = off)")
    ap.add_argument("--match-games", type=int, help="games per evaluation match (0 = puzzles only)")
    ap.add_argument("--match-tc",
                    help="evaluation matches: time control (base+inc seconds) for the engine binary "
                         "playing with its own search; 'none' = in-process match without clock")
    ap.add_argument("--match-search-mode", choices=["Mcts", "AlphaBeta", "Searchless"],
                    help="engine SearchMode in --match-tc matches")
    ap.add_argument("--match-simulations", type=int,
                    help="--match-tc none: search simulations per move (0 = searchless)")
    ap.add_argument("--eval-threads", type=int,
                    help="CPU threads for the evaluation (= games at the same time in --match-tc matches)")
    # this session only (not stored)
    ap.add_argument("--puzzles", help="puzzle CSV for the evaluation (default trainer/data/puzzles.csv; "
                                      "'none' = no puzzles)")
    ap.add_argument("--engine", help="engine binary for --match-tc matches (default: newest of "
                                     "target/release/peazero and ./peazero)")
    ap.add_argument("--pools", type=int, default=2,
                    help="self-play speed (this session only, same results): split the games over N pools "
                         "so the GPU evaluates one while the CPU searches the other (2, default); 1 = one "
                         "pool; 0 = the old plain loop (no pinned buffers / CUDA graphs)")
    ap.add_argument("--cuda-graphs", action=argparse.BooleanOptionalAction, default=True,
                    help="self-play inference as recorded CUDA graphs (one launch per evaluation batch); "
                         "--no-cuda-graphs = eager PyTorch")
    ap.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=True,
                    help="self-play inference in NHWC memory layout (cuDNN tensor-core kernels; 1.25x "
                         "on an RTX 3070); --no-channels-last = normal layout (this session only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true", help="disable fp16 autocast on CUDA")
    ap.add_argument("--dashboard-port", type=int, default=8765, help="live dashboard port; 0 = no dashboard")
    ap.add_argument("--no-browser", action="store_true", help="don't open the dashboard in a browser")
    return resolve_settings(ap.parse_args())


def resolve_settings(args):
    """Fill unset settings from the run's config.json (--resume) or the
    defaults, store the result in config.json and print it."""
    config_path = os.path.join(args.run, "config.json")
    saved = {}
    if args.resume and os.path.exists(config_path):
        with open(config_path) as f:
            saved = json.load(f)
    overridden = []
    for key, default in DEFAULTS.items():
        value = getattr(args, key)
        if value is not None:
            if args.resume and saved.get(key, value) != value:
                overridden.append(key)
            continue
        if args.resume and os.path.exists(config_path):
            value = saved.get(key, LEGACY.get(key, default))
        else:
            value = default
        setattr(args, key, value)
    if args.simulations == 0 and args.unroll == 0:
        print("root-only self-play needs the dynamics network: using --unroll 5")
        args.unroll = 5
    os.makedirs(args.run, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump({k: getattr(args, k) for k in DEFAULTS}, f, indent=2)
    print("settings: " + ", ".join(f"{k}={getattr(args, k)}" + (" (changed)" if k in overridden else "")
                                   for k in DEFAULTS))
    return args


def log(run, record):
    with open(os.path.join(run, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


def data_gen_dirs(run):
    """(generation, path) of every data/gen_XXXX folder, oldest first."""
    out = []
    for path in glob.glob(os.path.join(run, "data", "gen_*")):
        m = re.fullmatch(r"gen_(\d+)", os.path.basename(path))
        if m and os.path.isdir(path):
            out.append((int(m.group(1)), path))
    return sorted(out)


def migrate_flat_data(run):
    """Move files from the old flat layout (data/gen0001_000.bin) into
    data/gen_0001/part_000.bin."""
    moved = 0
    for path in glob.glob(os.path.join(run, "data", "gen*_*.bin")):
        m = re.fullmatch(r"gen(\d+)_(\d+)\.bin", os.path.basename(path))
        if not m:
            continue
        target_dir = os.path.join(run, "data", f"gen_{int(m.group(1)):04d}")
        os.makedirs(target_dir, exist_ok=True)
        shutil.move(path, os.path.join(target_dir, f"part_{int(m.group(2)):03d}.bin"))
        moved += 1
    if moved:
        print(f"moved {moved} data files into per-generation folders")


def replay_gens(args):
    """Generations of data needed to fill the replay buffer, plus one spare."""
    return math.ceil(args.replay / args.positions_per_gen) + 1


def data_gens_to_keep(args):
    if args.keep_data_gens >= 0:
        return args.keep_data_gens  # 0 = keep everything
    return replay_gens(args)


def prune_nets(args, newest):
    """Delete nets/gen_XXXX.pzn except every --keep-nets-th generation,
    the evaluation milestones (the evaluator needs gen N and N - eval_every)
    and `newest`. Returns the number of files deleted."""
    if args.keep_nets <= 0:
        return 0
    removed = 0
    for path in glob.glob(os.path.join(args.run, "nets", "gen_*.pzn")):
        m = re.fullmatch(r"gen_(\d+)\.pzn", os.path.basename(path))
        if not m:
            continue  # hand-made copies like gen_0839fix.pzn stay
        g = int(m.group(1))
        if g >= newest or g % args.keep_nets == 0 or (args.eval_every > 0 and g % args.eval_every == 0):
            continue
        try:
            os.remove(path)
            removed += 1
        except OSError:
            pass
    return removed


def prune_data(run, keep, exporter=None):
    """Delete all but the newest `keep` data folders; with `exporter`
    (--viriformat) each one is appended to the run's .vf file first, and
    kept if that fails."""
    if keep <= 0:
        return
    dirs = data_gen_dirs(run)
    for gen, path in dirs[:-keep]:
        if exporter is not None:
            try:
                exporter.export_gen(gen, path)
            except Exception as e:
                print(f"viriformat export of {path} failed ({e}); keeping the folder")
                continue
        shutil.rmtree(path, ignore_errors=True)


def start_evaluator(args):
    """Background strength evaluation (peazero/evaluator.py) at lower CPU
    priority. It exits when we do: it watches its stdin, which we hold."""
    if args.eval_every <= 0:
        return None
    cmd = [sys.executable, "-m", "peazero.evaluator", "--run", os.path.abspath(args.run),
           "--every", str(args.eval_every), "--games", str(args.match_games),
           "--tc", str(args.match_tc), "--search-mode", args.match_search_mode,
           "--simulations", str(args.match_simulations), "--threads", str(args.eval_threads),
           "--seed", str(args.seed), "--parent-stdin"]
    if args.engine:
        cmd += ["--engine", os.path.abspath(args.engine)]
    if args.puzzles:
        cmd += ["--puzzles", args.puzzles if args.puzzles.lower() == "none" else os.path.abspath(args.puzzles)]
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        kw["preexec_fn"] = lambda: os.nice(10)
    try:
        return subprocess.Popen(cmd, cwd=os.path.dirname(os.path.abspath(__file__)), stdin=subprocess.PIPE, **kw)
    except OSError as e:
        print(f"could not start the evaluator: {e}")
        return None


def stop_evaluator(proc, finish=False):
    """finish=True (training done): let it evaluate the last milestones
    first; otherwise stop it now."""
    if proc is None or proc.poll() is not None:
        return
    try:
        if finish:
            print("waiting for the evaluator to finish the last milestones (Ctrl+C to skip)")
            proc.stdin.write(b"finish\n")
            proc.stdin.flush()
            proc.wait()
        else:
            proc.stdin.close()  # the evaluator exits on end of file
            proc.wait(timeout=5)
    except (Exception, KeyboardInterrupt):
        proc.kill()


def main():
    args = parse_args()
    for d in ("data", "nets"):
        os.makedirs(os.path.join(args.run, d), exist_ok=True)
    status = dashboard.StatusWriter(
        args.run, run=os.path.abspath(args.run), target_generations=args.generations, batch=args.batch,
        unroll=args.unroll, reuse=args.reuse, games=args.games, considered=args.considered,
        positions_target=args.positions_per_gen, replay_capacity=args.replay)
    status.update(force=True, phase="loading", generation=0)
    if args.dashboard_port:
        url = dashboard.start(args.run, args.dashboard_port, not args.no_browser)
        if url:
            print(f"dashboard: {url}")
    evaluator = start_evaluator(args)
    finished = False
    try:
        train(args, status)
        finished = True
    except KeyboardInterrupt:
        print("\nstopped (Ctrl+C); continue with --resume")
    finally:
        status.update(force=True, phase="stopped")
        stop_evaluator(evaluator, finish=finished)


class PlainPool:
    """One self-play pool driven by the plain loop (train_lib.search_step /
    selfplay_step), with the SearchRunner interface used below."""

    def __init__(self, pool):
        self.pool = pool

    def stats(self):
        return self.pool.stats()

    def take_chunks(self):
        chunk = self.pool.take_chunk()
        return [chunk] if chunk[1] else []

    @property
    def num_threads(self):
        return self.pool.num_threads


def load_book(path, cache_dir):
    """Opening lines [(FEN or None, [UCI moves])] from a .pgn (move lines) or
    .epd/.fen (positions) file; parsed PGNs are cached in cache_dir."""
    import chess
    path = os.path.abspath(path)
    st = os.stat(path)
    cache = os.path.join(cache_dir, "book_cache.json")
    try:
        with open(cache) as f:
            c = json.load(f)
        if c["path"] == path and c["size"] == st.st_size and c["mtime"] == st.st_mtime:
            return [(fen, moves) for fen, moves in c["lines"]]
    except (OSError, ValueError, KeyError):
        pass
    t = time.time()
    lines = []
    if path.lower().endswith(".pgn"):
        import chess.pgn
        with open(path, encoding="utf-8", errors="replace") as f:
            while (g := chess.pgn.read_game(f)) is not None:
                fen = g.headers.get("FEN")
                moves = [m.uci() for m in g.mainline_moves()]
                if moves or fen:
                    lines.append((fen, moves))
    else:  # one EPD/FEN per line
        with open(path, encoding="utf-8", errors="replace") as f:
            for raw in f:
                raw = raw.strip()
                if not raw or raw.startswith("#"):
                    continue
                fields = raw.split()
                try:
                    board = chess.Board(" ".join(fields[:6])) if len(fields) >= 6 and fields[4].isdigit() \
                        else chess.Board.from_epd(raw)[0]
                except ValueError:
                    continue
                lines.append((board.fen(), []))
    if not lines:
        sys.exit(f"no openings found in {path}")
    print(f"book: {len(lines):,} openings from {path} ({time.time() - t:.0f}s)")
    try:
        with open(cache, "w") as f:
            json.dump({"path": path, "size": st.st_size, "mtime": st.st_mtime, "lines": lines}, f)
    except OSError:
        pass
    return lines


def variety_settings(args):
    """SearchSelfPlay keyword arguments for the opening book, random moves and
    move temperature (empty when all are off)."""
    kw = {}
    if args.book and args.book.lower() != "none":
        kw.update(openings=load_book(args.book, args.run), book_prob=args.book_prob)
    if args.random_prob > 0 and args.random_plies > 0:
        kw.update(random_prob=args.random_prob, random_plies=args.random_plies)
    if args.root_explore > 0:
        kw["root_explore"] = args.root_explore
    if args.temperature > 0 or args.temp_endgame > 0:
        kw.update(temperature=args.temperature, temp_decay_plies=args.temp_decay_plies,
                  temp_endgame=args.temp_endgame, temp_value_cutoff=args.temp_value_cutoff)
    return kw


def selfplay_nets(net, replay, dev, amp):
    """calibrate_bn copies for self-play and export. With --qat the
    BatchNorm statistics are measured through the int8 simulation (what the
    engine runs), but self-play itself uses the float network: QAT only in
    the training steps, so self-play keeps its full speed."""
    init_net, rec_net = calibrate_bn(net, replay, dev, amp)
    init_net.set_qat(False)
    rec_net.set_qat(False)
    return init_net, rec_net


def train(args, status):
    dev = torch.device(args.device)
    amp = not args.no_amp and dev.type == "cuda"
    torch.manual_seed(args.seed)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    migrate_flat_data(args.run)
    keep_gens = data_gens_to_keep(args)
    exporter = None
    if args.viriformat:
        from peazero.vf_export import VfExporter
        if not hasattr(pz, "chunk_to_viriformat"):
            sys.exit("peazero_rs is out of date (no viriformat export); run `pip install ./crates/py`")
        exporter = VfExporter(args.run)
        print(f"viriformat: {exporter.summary()}")
    assert pz.ENCODING_VERSION == enc.ENCODING_VERSION, "rebuild peazero_rs: encoding version mismatch"

    ckpt_path = os.path.join(args.run, "checkpoint.pt")
    c, blocks = net_size(args.net)
    cfg = NetConfig(channels=c, repr_blocks=blocks, dyn_blocks=blocks)
    gen, step = 0, 0
    teacher_until = None  # generation at which self-play switched from the teacher to this net
    if args.resume and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg = NetConfig(**ckpt["config"])
        gen, step = ckpt["generation"], ckpt["step"]
        teacher_until = ckpt.get("teacher_until")
    net = MuZeroNet(cfg).to(dev)
    net.set_qat(args.qat)
    if args.qat and not net.qat:
        print(f"--qat: {cfg.channels} channels is not a multiple of 32, the engine uses float; QAT off")
    # fused: one CUDA kernel for the whole update; same optimizer state as the default
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
                            fused=dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    if args.resume and os.path.exists(ckpt_path):
        net.load_state_dict(ckpt["model"])
        # The saved group options (incl. fused=None from checkpoints written
        # before fused AdamW) replace ours on load, and they also decide where
        # each parameter's step counter goes: fused AdamW needs it on the GPU.
        opt_state = ckpt["optimizer"]
        for g in opt_state["param_groups"]:
            g["fused"] = dev.type == "cuda" or None
            g["foreach"] = None
        opt.load_state_dict(opt_state)
        print(f"resumed {args.run} at generation {gen}, step {step}")

    replay = pz.Replay(capacity=args.replay, seed=args.seed + gen)
    if args.resume:  # refill the replay buffer, oldest data first (newest must survive eviction)
        dirs = data_gen_dirs(args.run)
        refill = dirs[-replay_gens(args):]
        for _, d in refill:
            for path in sorted(glob.glob(os.path.join(d, "*.bin"))):
                with open(path, "rb") as f:
                    replay.add_chunk(f.read())
        print(f"replay buffer: {replay.positions:,} positions from {len(refill)} generations on disk")
    removed = prune_nets(args, gen)
    if removed:
        print(f"deleted {removed} old networks: keeping every {args.keep_nets}th generation, the evaluation "
              "milestones and the newest (--keep-nets 0 keeps all)")

    runner = None
    if args.simulations > 0:
        if not hasattr(pz, "SearchSelfPlay"):
            sys.exit("peazero_rs is out of date (no tree search); run `pip install ./crates/py`")
        n_pools = max(1, args.pools)
        pcr = variety_settings(args)
        if args.fast_simulations > 0:
            pcr.update(fast_simulations=args.fast_simulations, full_prob=args.full_prob,
                       fast_considered=args.fast_considered)
        sizes = [args.games // n_pools + (i < args.games % n_pools) for i in range(n_pools)]
        try:
            # pool i gets its own seed; with one pool this is the old single pool
            pools = [pz.SearchSelfPlay(games=n, simulations=args.simulations, considered=args.considered,
                                       max_plies=args.max_plies, seed=args.seed + gen + 1_000_003 * i,
                                       threads=args.threads, noise_plies=args.noise_plies,
                                       late_gumbel_scale=args.late_noise, **pcr)
                     for i, n in enumerate(sizes)]
        except TypeError:
            sys.exit("peazero_rs is out of date (no noise schedule / data format v2); run `pip install ./crates/py`")
        if args.pools > 0:
            if not hasattr(pools[0], "requests_into"):
                sys.exit("peazero_rs is out of date (no requests_into); run `pip install ./crates/py`")
            runner = SearchRunner(pools, dev, amp, cuda_graphs=args.cuda_graphs,
                                  channels_last=args.channels_last)
            sp = runner
        else:
            sp = PlainPool(pools[0])
        mode = f"{args.simulations} simulations, {args.considered} considered"
        if pcr.get("openings"):
            mode += f"; book {len(pcr['openings']):,} lines ({args.book_prob:.0%} of games)"
        if pcr.get("random_prob"):
            mode += f"; {args.random_prob:.0%} 1-{args.random_plies} random moves"
        if pcr.get("root_explore"):
            mode += f"; root exploration floor {args.root_explore:g}"
        if pcr.get("temperature"):
            mode += (f"; temperature {args.temperature:g} -> {args.temp_endgame:g} over {args.temp_decay_plies} "
                     f"plies, cutoff {args.temp_value_cutoff:g}")
        if args.fast_simulations > 0:
            mode += (f" on {args.full_prob:.0%} of moves, else {args.fast_simulations} simulations, "
                     f"{args.fast_considered} considered (playout cap randomization)")
        if runner:
            mode += (f", {n_pools} pipelined pools" if n_pools > 1 else ", 1 pool")
            mode += ", CUDA graphs" if runner.graphs_wanted else ""
            mode += ", channels_last" if runner.channels_last else ""
        else:
            mode += ", plain loop"
    else:
        if args.fast_simulations > 0:
            sys.exit("--fast-simulations needs tree-search self-play (--simulations > 0)")
        sp = PlainPool(pz.SelfPlay(games=args.games, considered=args.considered, max_plies=args.max_plies,
                                   seed=args.seed + gen, threads=args.threads))
        mode = f"root-only, {args.considered} considered"
    sp_threads = sp.num_threads
    params = sum(p.numel() for p in net.parameters()) / 1e6
    net_name = f"{cfg.channels}x{cfg.repr_blocks}"
    print(f"net {net_name} ({params:.2f}M params) on {dev}{' fp16' if amp else ''}; "
          f"{args.games} games, {mode}, {sp_threads} self-play threads, batch {args.batch}")
    status.update(force=True, net=net_name,
                  params=params, device=dev.type, amp=amp, step=step, generation=gen,
                  session_start_generation=gen, replay_positions=replay.positions)
    # Self-play and export use eval-mode copies with recalibrated BatchNorm
    # statistics (train_lib.calibrate_bn), never `net` itself in eval mode.
    init_net, rec_net = selfplay_nets(net, replay, dev, amp)

    # Warm start: a stronger existing network plays the self-play games
    # until this one's loss on fresh data beats it (or --teacher-gens).
    teacher = None
    if args.teacher and teacher_until is None and gen < args.teacher_gens:
        if args.simulations <= 0:
            sys.exit("--teacher needs tree-search self-play (--simulations > 0)")
        tck = torch.load(args.teacher, map_location="cpu", weights_only=False)
        teacher = MuZeroNet(NetConfig(**tck["config"])).to(dev)
        teacher.load_state_dict(tck["model"])
        teacher.set_qat(False)
        tc = tck["config"]
        print(f"teacher: {args.teacher} ({tc['channels']}x{tc['repr_blocks']}, gen {tck.get('generation')}) plays "
              f"self-play until this net's loss beats it (at most until generation {args.teacher_gens})")
    elif args.teacher and teacher_until is not None:
        print(f"teacher phase ended at generation {teacher_until}; self-play by this network")

    def teacher_net():
        """Eval-mode teacher with BatchNorm statistics measured on the replay data."""
        return calibrate_bn(teacher, replay, dev, amp)[0] if replay.positions else copy.deepcopy(teacher).eval()

    teacher_sp = teacher_net() if teacher is not None else None
    teacher_wins = 0  # consecutive generations in which this net beat the teacher on unseen games
    keys_ok = hasattr(pz, "chunk_position_keys")
    recent_keys = collections.deque(maxlen=5)  # unique position hashes of the last generations
    sp_net = teacher_sp if teacher_sp is not None else init_net
    if runner:
        runner.set_net(sp_net)
        if runner.graphs_wanted and not runner.graphs:
            print("CUDA graphs unavailable here; eager inference")

    while gen < args.generations:
        gen += 1
        # ---------------------------------------------------------------- self-play
        t0 = time.time()
        new_positions, new_games, new_plies, part = 0, 0, 0, 0
        gen_keys = []  # position hashes of this generation's games (uniqueness metric)
        # this generation's games only: unseen by both networks when compared below
        fresh = pz.Replay(capacity=10**9, seed=args.seed + gen) if teacher is not None else None
        gen_dir = os.path.join(args.run, "data", f"gen_{gen:04d}")
        os.makedirs(gen_dir, exist_ok=True)
        stats_before = sp.stats()
        status.update(force=True, phase="selfplay", generation=gen, phase_start=t0, positions=0,
                      games_finished=0, moves_per_second=None)
        # root-only: one step = one move per game; search: one step = one
        # simulation per game (~simulations + 1 steps per move)
        chunk_every = 32 if args.simulations == 0 else 8 * (args.simulations + 1)
        steps = 0
        if runner:
            runner.reset_timing()
        while new_positions < args.positions_per_gen:
            if runner:
                runner.step()
            elif args.simulations > 0:
                search_step(sp_net, sp.pool, dev, amp)
            else:
                selfplay_step(init_net, sp.pool, dev, amp, rec_net)
            steps += 1
            if time.time() - status.last_write >= 1.0:
                moves = sp.stats()["plies"] - stats_before["plies"]
                status.update(positions=new_positions, games_finished=new_games,
                              moves_per_second=moves / max(time.time() - t0, 1e-6),
                              replay_positions=replay.positions)
            if steps % chunk_every == 0:
                for chunk, games, positions, plies in sp.take_chunks():
                    path = os.path.join(gen_dir, f"part_{part:03d}.bin")
                    with open(path, "wb") as f:
                        f.write(chunk)
                    replay.add_chunk(chunk)
                    if keys_ok:
                        gen_keys.append(pz.chunk_position_keys(chunk))
                    if fresh is not None:
                        fresh.add_chunk(chunk)
                    part += 1
                    new_positions += positions
                    new_plies += plies
                    new_games += games
        sp_timing = None
        if runner:
            runner.drain()  # evaluations still in flight belong to this generation's network
            sp_timing = runner.reset_timing()
        sp_time = time.time() - t0
        st = sp.stats()
        uniq = {}
        if gen_keys:
            # unique share within this generation, and share not seen in the previous generations kept
            keys = np.unique(np.concatenate(gen_keys))
            total = sum(len(k) for k in gen_keys)
            uniq["unique_positions"] = round(len(keys) / max(total, 1), 4)
            if recent_keys:
                seen = np.isin(keys, np.concatenate(list(recent_keys)), assume_unique=False)
                uniq["new_vs_recent"] = round(float((~seen).sum()) / max(total, 1), 4)
            recent_keys.append(keys)
        moves = st["plies"] - stats_before["plies"]
        d = {k: st[k] - stats_before[k] for k in ("games", "white_wins", "black_wins", "draws")}
        terms = {k: st["terminations"][k] - stats_before["terminations"][k] for k in st["terminations"]}

        # Teacher phase: both networks on this generation's games, before the
        # student trains on them (the student as of the previous generation).
        teacher_info = {}
        if teacher is not None and fresh is not None and fresh.positions:
            (sp_p, sp_v), (tp, tv) = compare_losses([init_net, teacher_sp], fresh, dev, amp, batches=16)
            teacher_wins = teacher_wins + 1 if sp_p + sp_v < tp + tv else 0
            teacher_info = {"selfplay_by": "teacher", "student_policy_loss": round(sp_p, 4),
                            "student_value_loss": round(sp_v, 4), "teacher_policy_loss": round(tp, 4),
                            "teacher_value_loss": round(tv, 4), "student_wins_in_a_row": teacher_wins}

        # ---------------------------------------------------------------- training
        net.train()
        t1 = time.time()
        n_steps = max(1, int(new_positions * args.reuse / args.batch))
        sums = {}
        status.update(force=True, phase="training", phase_start=t1, steps_done=0, steps_total=n_steps,
                      running=None, positions=new_positions, games_finished=new_games, replay_positions=replay.positions)
        # The next batch is sampled (CPU, Rust) and copied while the GPU is still
        # busy with the current step's backward pass; metrics stay on the GPU
        # and are only read once a second for the dashboard.
        batch = batch_to_device(replay.sample(args.batch, args.unroll), dev)
        for i in range(n_steps):
            step += 1
            lr = args.lr * min(1.0, step / args.warmup)
            for g in opt.param_groups:
                g["lr"] = lr
            loss, m = muzero_loss(net, batch, dev, amp, q_ratio=args.q_ratio)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if i + 1 < n_steps:
                batch = batch_to_device(replay.sample(args.batch, args.unroll), dev)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            for k, v in m.items():
                sums[k] = sums[k] + v if k in sums else v
            if time.time() - status.last_write >= 1.0:
                status.update(steps_done=i + 1, step=step,
                              running={k: float(v) / (i + 1) for k, v in sums.items()},
                              steps_per_second=(i + 1) / max(time.time() - t1, 1e-6))
        avg = {k: float(v) / n_steps for k, v in sums.items()}
        train_time = time.time() - t1

        # ---------------------------------------------------------------- save
        status.update(force=True, phase="saving", steps_done=n_steps, step=step)
        net.eval()
        init_net, rec_net = selfplay_nets(net, replay, dev, amp)
        ev = eval_metrics(init_net, rec_net, replay, dev, amp)
        if teacher is not None and (teacher_wins >= 2 or gen >= args.teacher_gens):
            why = ("its loss beat the teacher's on 2 generations of unseen games in a row" if teacher_wins >= 2
                   else f"--teacher-gens {args.teacher_gens}")
            print(f"teacher phase over at generation {gen} ({why}): self-play by this network from now on")
            teacher, teacher_sp, teacher_until = None, None, gen
        elif teacher is not None:
            teacher_sp = teacher_net()
        sp_net = teacher_sp if teacher_sp is not None else init_net
        if runner:
            runner.set_net(sp_net)
        net_path = os.path.join(args.run, "nets", f"gen_{gen:04d}.pzn")
        export_pzn(init_net, net_path, rec_net)
        shutil.copyfile(net_path, os.path.join(args.run, "latest.pzn"))
        prune_data(args.run, keep_gens, exporter)
        prune_nets(args, gen)
        torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(), "config": cfg.__dict__,
                    "teacher_until": teacher_until,
                    "generation": gen, "step": step}, ckpt_path)

        record = {
            "generation": gen, "step": step, "lr": lr, "batch": args.batch, "time": round(time.time(), 1),
            "selfplay_seconds": round(sp_time, 1), "train_seconds": round(train_time, 1),
            "moves_per_second": round(moves / sp_time), "new_games": new_games, "new_positions": new_positions,
            "avg_game_length": round(new_plies / max(new_games, 1), 1),
            "full_search_share": round((st["full_plies"] - stats_before["full_plies"]) / max(moves, 1), 3)
            if "full_plies" in st else 1.0,
            **({"sp_gpu_wait_seconds": round(sp_timing["gpu_wait"], 1),
                "sp_rust_seconds": round(sp_timing["rust"], 1),
                "sp_launch_seconds": round(sp_timing["launch"], 1)} if sp_timing else {}),
            **teacher_info, **uniq,
            "results": d, "terminations": terms, "replay_positions": replay.positions, "qat": net.qat, **avg, **ev,
        }
        log(args.run, record)
        print(f"gen {gen:4d} | selfplay {sp_time:6.1f}s {moves / sp_time:7.0f} mv/s "
              + (f"(gpu wait {sp_timing['gpu_wait']:.0f}s rust {sp_timing['rust']:.0f}s "
                 f"launch {sp_timing['launch']:.0f}s) " if sp_timing else "")
              + f"{new_games:5d} games (W{d['white_wins']} D{d['draws']} B{d['black_wins']}, "
              + (f"unique {uniq['unique_positions']:.0%}, " if uniq else "")
              + f"len {record['avg_game_length']}) | train {n_steps} steps {train_time:6.1f}s "
              f"loss {avg['loss']:.3f} (p {avg['policy_loss']:.3f} v {avg['value_loss']:.3f} "
              f"agree {avg['policy_agreement']:.2f}) | exported p {ev.get('eval_policy_loss', 0):.3f} "
              f"v {ev.get('eval_value_loss', 0):.3f}"
              + (f" | int8 {ev['int8_agreement'] * 100:.1f}%" if "int8_agreement" in ev else "")
              + (f" | teacher self-play, unseen games: this net p {teacher_info['student_policy_loss']:.3f} "
                 f"v {teacher_info['student_value_loss']:.3f} vs teacher p {teacher_info['teacher_policy_loss']:.3f} "
                 f"v {teacher_info['teacher_value_loss']:.3f}" if teacher_info else "")
              + f" | {net_path}")


if __name__ == "__main__":
    main()
