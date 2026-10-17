#!/usr/bin/env python3
"""Background strength evaluation of a training run, on the CPU.

Every --every generations (a "milestone"):
  * match:   gen N vs gen N - every, --games games (random openings, each
             played twice with colours swapped). Default: the real engine
             binary over UCI at --tc 8+0.08 with its own search (--search-mode,
             MCTS), like OpenBench (uci_match.py). --tc none: in-process
             match without clock, searchless or --simulations
  * puzzles: gen N on a fixed puzzle set (Lichess format, see
             scripts/make_puzzles.py), searchless by default

Results are appended to <run>/evals.jsonl (one line per match / puzzle
run) and shown on the training dashboard. Milestones already in
evals.jsonl are skipped, so it catches up after a restart and can backfill
an old run:

    python -m peazero.evaluator --run runs/fourth --once        # evaluate what exists, then exit
    python -m peazero.evaluator --run runs/fourth               # keep watching for new networks

train.py starts it automatically (see --eval-every there). Everything runs
on the CPU (tract, the engine's inference), so the GPU keeps training. The
time-control match needs the engine built: `cargo build --release` (or
`make`); without it the evaluator falls back to the in-process match.
"""
import argparse
import csv
import json
import math
import os
import re
import sys
import threading
import time
import traceback

TRAINER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(TRAINER_DIR)
DEFAULT_PUZZLES = os.path.join(TRAINER_DIR, "data", "puzzles.csv")
# Rating bands for the per-band puzzle curves (upper bound exclusive).
BANDS = [(0, 1000), (1000, 1400), (1400, 1800), (1800, 2200), (2200, 10_000)]


def band_name(lo, hi):
    if lo == 0:
        return f"<{hi}"
    if hi >= 10_000:
        return f"{lo}+"
    return f"{lo}-{hi}"


# --------------------------------------------------------------------- statistics
def score_to_elo(s):
    return -400.0 * math.log10(1.0 / s - 1.0) + 0.0  # + 0.0: no "-0"


def match_stats(games):
    """Elo of A vs B with a 95% interval from the game pairs (pentanomial:
    the two games of an opening are one sample, which removes the opening's
    colour bias from the error estimate)."""
    n = len(games)
    w = sum(1 for g in games if g["score_a"] == 1.0)
    l = sum(1 for g in games if g["score_a"] == 0.0)
    d = n - w - l
    pairs = {}
    for g in games:
        pairs.setdefault(g["opening"], []).append(g["score_a"])
    pair_scores = [sum(v) / len(v) for v in pairs.values()]
    m = len(pair_scores)
    s = (w + 0.5 * d) / n
    var = sum((p - s) ** 2 for p in pair_scores) / m / m if m > 1 else 0.25 / n
    # Floor of one game's worth of score: identical pairs (e.g. all draws)
    # would otherwise claim a zero-width interval.
    se = max(math.sqrt(var), 0.5 / n)
    lo_clip, hi_clip = 0.5 / n, 1.0 - 0.5 / n  # 100-0 would be infinite Elo
    clip = lambda x: min(hi_clip, max(lo_clip, x))  # noqa: E731
    penta = [0] * 5
    for v in pairs.values():
        if len(v) == 2:
            penta[int(round(2 * sum(v)))] += 1
    terms = {}
    for g in games:
        terms[g["termination"]] = terms.get(g["termination"], 0) + 1
    return {
        "games": n, "wins": w, "draws": d, "losses": l, "score": round(s, 4),
        "elo": round(score_to_elo(clip(s)), 1),
        "elo_lo": round(score_to_elo(clip(s - 1.96 * se)), 1),
        "elo_hi": round(score_to_elo(clip(s + 1.96 * se)), 1),
        "pentanomial": penta, "terminations": terms,
        "avg_plies": round(sum(g["plies"] for g in games) / n, 1),
    }


def puzzle_performance(ratings, solved):
    """Rating R at which the expected number of solved puzzles,
    sum 1 / (1 + 10^((r - R) / 400)), equals the actual number."""
    k = sum(solved)
    if k == 0:
        return min(ratings) - 400
    if k == len(ratings):
        return max(ratings) + 400
    lo, hi = -1000.0, 5000.0
    for _ in range(60):
        mid = (lo + hi) / 2
        expected = sum(1.0 / (1.0 + 10 ** ((r - mid) / 400.0)) for r in ratings)
        lo, hi = (mid, hi) if expected < k else (lo, mid)
    return (lo + hi) / 2


def puzzle_stats(puzzles, results):
    out = {"puzzles": len(puzzles)}
    first = [r[0] for r in results]
    solved = [r[1] for r in results]
    out["first_move"] = round(sum(first) / len(first), 4)
    out["solved"] = round(sum(solved) / len(solved), 4)
    out["rating"] = round(puzzle_performance([p["rating"] for p in puzzles], solved))
    bands = {}
    for lo, hi in BANDS:
        idx = [i for i, p in enumerate(puzzles) if lo <= p["rating"] < hi]
        if idx:
            bands[band_name(lo, hi)] = {
                "n": len(idx),
                "first_move": round(sum(first[i] for i in idx) / len(idx), 4),
                "solved": round(sum(solved[i] for i in idx) / len(idx), 4),
            }
    out["bands"] = bands
    return out


# --------------------------------------------------------------------- files
def load_puzzles(path):
    """Lichess puzzle CSV (full database or the subset from
    scripts/make_puzzles.py): needs FEN, Moves and Rating columns."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return [{"fen": r["FEN"], "moves": r["Moves"], "rating": int(r["Rating"])} for r in rows]


def net_files(run):
    """{generation: path} of <run>/nets/gen_XXXX.pzn."""
    out = {}
    d = os.path.join(run, "nets")
    if os.path.isdir(d):
        for name in os.listdir(d):
            m = re.fullmatch(r"gen_(\d+)\.pzn", name)
            if m:
                out[int(m.group(1))] = os.path.join(d, name)
    return out


def read_evals(run):
    out = []
    path = os.path.join(run, "evals.jsonl")
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def append_eval(run, record):
    with open(os.path.join(run, "evals.jsonl"), "a") as f:
        f.write(json.dumps(record) + "\n")


class Status:
    """<run>/eval_status.json: what the evaluator is doing (dashboard)."""

    def __init__(self, run, settings):
        self.path = os.path.join(run, "eval_status.json")
        self.settings = settings

    def set(self, state, task=None):
        data = {"state": state, "task": task, "updated": time.time(), **self.settings}
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except OSError:
            pass


# --------------------------------------------------------------------- evaluation
# Set when train.py has finished: evaluate what is left, then exit.
FINISHING = threading.Event()


def stable(path, age=10.0):
    """A network file the trainer has finished writing."""
    if FINISHING.is_set():
        return True
    try:
        return time.time() - os.path.getmtime(path) > age
    except OSError:
        return False


def record_mode(e):
    """Match conditions of an evals.jsonl match record (older records: in-process)."""
    return e.get("mode") or (f"{e['simulations']} sims" if e.get("simulations") else "searchless")


def match_mode(args):
    """Match conditions; only matches with the same mode form one ladder."""
    if args.tc:
        return f"{args.tc} {args.search_mode}"
    return f"{args.simulations} sims" if args.simulations else "searchless"


def pending_work(args, nets, evals, have_puzzles):
    """(kind, generation) tasks still to do, oldest first."""
    mode = match_mode(args)
    matches = {(e["generation"], e["opponent"]) for e in evals
               if e.get("type") == "match" and record_mode(e) == mode}
    puzzles = {e["generation"] for e in evals if e.get("type") == "puzzles"}
    todo = []
    for g in sorted(nets):
        if g % args.every or not stable(nets[g]):
            continue
        if have_puzzles and g not in puzzles:
            todo.append(("puzzles", g))
        opp = g - args.every
        if args.games > 0 and opp in nets and (g, opp) not in matches:
            todo.append(("match", g))
    return todo


def run_match(pz, args, nets, g, status):
    opp = g - args.every
    t = time.time()
    if args.tc:
        from peazero.uci_match import play_tc_match
        # The engine keeps its old network if EvalFile fails to load; make
        # sure both load (same tract code as the engine) before playing.
        for net in (nets[g], nets[opp]):
            pz.solve_puzzles(net, [])
        task = f"gen {g} vs {opp}"
        games = play_tc_match(args.engine, nets[g], nets[opp], games=args.games, tc=args.tc,
                              concurrency=args.concurrency, opening_plies=args.opening_plies,
                              seed=args.seed, max_plies=args.max_plies,
                              options={"SearchMode": args.search_mode},
                              progress=lambda d, n: status.set("running", f"{task} ({d}/{n} games)"))
    else:
        games = pz.play_match(nets[g], nets[opp], games=args.games, opening_plies=args.opening_plies,
                              simulations=args.simulations, considered=args.considered,
                              max_plies=args.max_plies, seed=args.seed, threads=args.threads)
    st = match_stats(games)
    rec = {"type": "match", "generation": g, "opponent": opp, "mode": match_mode(args),
           "simulations": 0 if args.tc else args.simulations,
           "seconds": round(time.time() - t, 1), "time": round(time.time(), 1), **st}
    bad = {k: v for k, v in st["terminations"].items() if k in ("time_forfeit", "engine_error", "illegal_move")}
    print(f"[eval] gen {g} vs {opp} ({rec['mode']}): {st['elo']:+.0f} Elo [{st['elo_lo']:+.0f}, {st['elo_hi']:+.0f}] "
          f"(+{st['wins']} ={st['draws']} -{st['losses']}, {rec['seconds']:.0f}s)"
          + (f" WARNING {bad}" if bad else ""), flush=True)
    return rec


def run_puzzles(pz, args, nets, g, puzzles):
    t = time.time()
    res = pz.solve_puzzles(nets[g], [(p["fen"], p["moves"]) for p in puzzles],
                           simulations=args.puzzle_simulations, considered=args.considered,
                           threads=args.threads)
    st = puzzle_stats(puzzles, res)
    rec = {"type": "puzzles", "generation": g, "simulations": args.puzzle_simulations,
           "seconds": round(time.time() - t, 1), "time": round(time.time(), 1), **st}
    print(f"[eval] gen {g} puzzles: {100 * st['solved']:.1f}% solved, {100 * st['first_move']:.1f}% first move, "
          f"rating {st['rating']} ({rec['seconds']:.0f}s)", flush=True)
    return rec


def refresh_engine(args, first=False):
    """Use time-control matches whenever the engine binary exists, the
    in-process match otherwise (checked again before every milestone, so
    building the engine later switches over)."""
    if not args.tc_wanted:
        return
    from peazero.uci_match import find_engine
    exe = args.engine_arg or find_engine(REPO_ROOT)
    if exe and os.path.isfile(exe):
        if args.tc is None and not first:
            print(f"[eval] engine found: {exe}; time-control matches from now on", flush=True)
        args.tc, args.engine = args.tc_wanted, exe
    else:
        if args.tc is not None or first:
            where = args.engine_arg or f"target/release in {REPO_ROOT}"
            print(f"[eval] no engine binary at {where} (build it with `cargo build --release`); "
                  "in-process match until it exists", flush=True)
        args.tc = None


def watch_parent():
    """train.py holds our stdin open: a line "finish" = training is done,
    evaluate the rest and exit; end of file = exit now."""
    def wait():
        try:
            for line in sys.stdin:
                if line.strip() == "finish":
                    FINISHING.set()
        except Exception:
            pass
        os._exit(0)
    threading.Thread(target=wait, daemon=True).start()


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--every", type=int, default=25, help="milestone interval in generations")
    ap.add_argument("--games", type=int, default=100, help="match games per milestone (0 = no matches)")
    ap.add_argument("--tc", default="8+0.08",
                    help="match time control (base+increment seconds) with the engine binary; "
                         "'none' = in-process match without clock (--simulations)")
    ap.add_argument("--engine", help="engine binary (default: newest of target/release/peazero and ./peazero)")
    ap.add_argument("--search-mode", default="Mcts", help="engine SearchMode for --tc matches")
    ap.add_argument("--concurrency", type=int, help="--tc games at the same time (default --threads)")
    ap.add_argument("--simulations", type=int, default=0,
                    help="in-process match (--tc none): search simulations (0 = searchless)")
    ap.add_argument("--puzzles", default=DEFAULT_PUZZLES, help="puzzle CSV ('none' = no puzzles)")
    ap.add_argument("--puzzle-simulations", type=int, default=0, help="0 = searchless (the raw policy)")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads")
    ap.add_argument("--considered", type=int, default=16)
    ap.add_argument("--opening-plies", type=int, default=6, help="random plies of every opening")
    ap.add_argument("--max-plies", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0, help="opening seed (same openings at every milestone)")
    ap.add_argument("--poll", type=float, default=30.0, help="seconds between looks for new networks")
    ap.add_argument("--once", action="store_true", help="evaluate what exists now, then exit")
    ap.add_argument("--parent-stdin", action="store_true", help=argparse.SUPPRESS)
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.parent_stdin:
        watch_parent()
    import peazero_rs as pz
    if not hasattr(pz, "play_match"):
        sys.exit("[eval] peazero_rs is out of date (no play_match); run `pip install ./crates/py`")
    if args.tc and args.tc.lower() == "none":
        args.tc = None
    if args.tc:
        from peazero.uci_match import parse_tc
        parse_tc(args.tc)  # fail early on a typo
        try:
            import chess.engine  # noqa: F401
        except ImportError:
            print("[eval] time-control matches need python-chess (pip install chess); "
                  "using the in-process match", flush=True)
            args.tc = None
    args.tc_wanted, args.engine_arg, args.tc = args.tc, args.engine, None
    refresh_engine(args, first=True)
    args.concurrency = args.concurrency or args.threads

    puzzles = []
    if args.puzzles and args.puzzles.lower() != "none":
        if os.path.exists(args.puzzles):
            puzzles = load_puzzles(args.puzzles)
            print(f"[eval] {len(puzzles)} puzzles from {args.puzzles}", flush=True)
        else:
            print(f"[eval] no puzzle file at {args.puzzles} (make one with trainer/scripts/make_puzzles.py); "
                  "matches only", flush=True)
    settings = {"every": args.every, "games": args.games, "simulations": args.simulations,
                "mode": match_mode(args), "puzzles": len(puzzles), "threads": args.threads}
    status = Status(args.run, settings)
    print(f"[eval] every {args.every} generations: "
          + (f"{args.games}-game match vs gen N-{args.every} ({match_mode(args)}"
             + (f", {args.concurrency} at a time, {args.engine})" if args.tc else ")")
             if args.games > 0 else "no matches")
          + (f", {len(puzzles)} puzzles" if puzzles else "")
          + f", {args.threads} CPU threads", flush=True)

    failures = {}
    while True:
        refresh_engine(args)
        status.settings["mode"] = match_mode(args)
        nets = net_files(args.run)
        todo = [t for t in pending_work(args, nets, read_evals(args.run), bool(puzzles))
                if failures.get(t, 0) < 3]
        if not todo:
            if args.once or FINISHING.is_set():
                break
            status.set("waiting")
            FINISHING.wait(args.poll)
            continue
        kind, g = todo[0]
        try:
            if kind == "match":
                status.set("running", f"gen {g} vs {g - args.every}")
                rec = run_match(pz, args, nets, g, status)
            else:
                status.set("running", f"puzzles gen {g}")
                rec = run_puzzles(pz, args, nets, g, puzzles)
            append_eval(args.run, rec)
        except Exception:
            traceback.print_exc()
            failures[(kind, g)] = failures.get((kind, g), 0) + 1
            again = "retrying" if failures[(kind, g)] < 3 else "giving up on it"
            print(f"[eval] {kind} for gen {g} failed; {again}", flush=True)
            status.set("error", f"{kind} gen {g}")
            FINISHING.wait(args.poll)
    status.set("idle")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
