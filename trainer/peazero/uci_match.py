"""Time-control matches between two networks with the real engine (UCI).

Both sides are the PeaZero engine binary, each with its own `EvalFile`,
playing with their own search (`SearchMode`, MCTS by default) and time
management at a clock like `8+0.08`, as on OpenBench. Used by the
evaluator (evaluator.py).

* Openings: random `opening_plies` legal moves from the start position,
  seeded, so every milestone of a run uses the same set. Each opening is
  played twice with colours swapped.
* Clocks are kept here (wall time per move, like cutechess): a side whose
  clock runs out loses (`time_forfeit`). Draws by threefold repetition,
  50 moves, stalemate and insufficient material; `max_plies` = draw.
* `concurrency` games run at the same time, each with its own pair of
  engine processes (Threads = 1). The binary is copied to a temporary
  folder first, so rebuilding the engine meanwhile is no problem.
"""
import os
import random
import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

# Allowed overrun of the clock before a loss on time (pipe and Python
# overhead between the engine sending `bestmove` and us reading it).
TIME_MARGIN = 0.1

ENGINE_NAMES = ("peazero.exe", "peazero") if os.name == "nt" else ("peazero",)


def find_engine(repo_root):
    """Newest engine binary from `cargo build --release` (target/release)
    or `make` (repo root), or None."""
    found = []
    for d in (os.path.join(repo_root, "target", "release"), repo_root):
        for name in ENGINE_NAMES:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                found.append(p)
    return max(found, key=os.path.getmtime) if found else None


def parse_tc(tc):
    """'8+0.08' -> (8.0, 0.08); '10' -> (10.0, 0.0)."""
    base, _, inc = tc.partition("+")
    base, inc = float(base), float(inc or 0)
    if base <= 0 or inc < 0:
        raise ValueError(f"bad time control {tc!r}")
    return base, inc


def random_openings(n, plies, seed):
    """n lists of `plies` random legal moves (re-drawn if the game ends)."""
    rng = random.Random(seed ^ 0x0BE7_1A65)
    out = []
    while len(out) < n:
        b = chess.Board()
        moves = []
        for _ in range(plies):
            legal = list(b.legal_moves)
            if not legal:
                break
            m = rng.choice(legal)
            b.push(m)
            moves.append(m)
        if len(moves) == plies and b.outcome(claim_draw=True) is None:
            out.append(moves)
    return out


TERMINATIONS = {
    chess.Termination.CHECKMATE: "checkmate",
    chess.Termination.STALEMATE: "stalemate",
    chess.Termination.INSUFFICIENT_MATERIAL: "insufficient_material",
    chess.Termination.FIFTY_MOVES: "fifty_moves",
    chess.Termination.SEVENTYFIVE_MOVES: "fifty_moves",
    chess.Termination.THREEFOLD_REPETITION: "repetition",
    chess.Termination.FIVEFOLD_REPETITION: "repetition",
}


class Player:
    """One engine process with one network; restarted after a crash."""

    def __init__(self, exe, net, options):
        self.exe, self.net, self.options = exe, net, options
        self.engine = None
        self.start()

    def start(self):
        self.close()
        self.engine = chess.engine.SimpleEngine.popen_uci(self.exe)
        self.engine.configure({"EvalFile": self.net, **self.options})

    def close(self):
        if self.engine is not None:
            try:
                self.engine.quit()
            except Exception:
                try:
                    self.engine.close()
                except Exception:
                    pass
            self.engine = None


def play_game(players, opening, a_white, base, inc, max_plies, game_id):
    """players = (A, B). Returns (score_a, termination, plies after the opening)."""
    board = chess.Board()
    for m in opening:
        board.push(m)
    side = {chess.WHITE: players[0] if a_white else players[1],
            chess.BLACK: players[1] if a_white else players[0]}
    clock = {chess.WHITE: base, chess.BLACK: base}
    plies = 0

    def result(winner, termination):  # winner: chess.WHITE / chess.BLACK / None
        if winner is None:
            return 0.5, termination, plies
        a_won = (winner == chess.WHITE) == a_white
        return (1.0 if a_won else 0.0), termination, plies

    while True:
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            return result(outcome.winner, TERMINATIONS.get(outcome.termination, "other"))
        if plies >= max_plies:
            return result(None, "max_plies")
        mover = board.turn
        player = side[mover]
        limit = chess.engine.Limit(white_clock=max(clock[chess.WHITE], 0.001),
                                   black_clock=max(clock[chess.BLACK], 0.001),
                                   white_inc=inc, black_inc=inc)
        t0 = time.perf_counter()
        try:
            res = player.engine.play(board, limit, game=game_id)
        except (chess.engine.EngineError, chess.engine.EngineTerminatedError):
            player.start()
            return result(not mover, "engine_error")
        clock[mover] -= time.perf_counter() - t0
        if clock[mover] < -TIME_MARGIN:
            return result(not mover, "time_forfeit")
        clock[mover] += inc
        if res.move is None or res.move not in board.legal_moves:
            return result(not mover, "illegal_move")
        board.push(res.move)
        plies += 1


def play_tc_match(engine, net_a, net_b, games=100, tc="8+0.08", concurrency=4, opening_plies=6,
                  seed=0, max_plies=512, options=None, progress=None):
    """Play `games` (rounded up to even) between `net_a` and `net_b`.
    Returns one dict per game (opening, a_white, score_a, termination,
    plies), in the order opening 0 A-white, opening 0 B-white, ...
    `progress(done, total)` is called after every game."""
    base, inc = parse_tc(tc)
    openings = random_openings(max(1, (games + 1) // 2), opening_plies, seed)
    jobs = [(i, a_white) for i in range(len(openings)) for a_white in (True, False)]
    results = [None] * len(jobs)
    options = dict(options or {})
    tmp = tempfile.mkdtemp(prefix="peazero-eval-")
    exe = os.path.join(tmp, os.path.basename(engine))
    shutil.copy2(engine, exe)
    lock = threading.Lock()
    next_job = iter(range(len(jobs)))
    done = [0]

    def worker():
        players = (Player(exe, net_a, options), Player(exe, net_b, options))
        try:
            while True:
                with lock:
                    k = next(next_job, None)
                if k is None:
                    return
                i, a_white = jobs[k]
                score, term, plies = play_game(players, openings[i], a_white, base, inc, max_plies, object())
                results[k] = {"opening": i, "a_white": a_white, "score_a": score,
                              "termination": term, "plies": plies}
                with lock:
                    done[0] += 1
                    if progress:
                        progress(done[0], len(jobs))
        finally:
            for p in players:
                p.close()

    try:
        n = max(1, min(concurrency, len(jobs)))
        with ThreadPoolExecutor(n) as ex:
            for f in [ex.submit(worker) for _ in range(n)]:
                f.result()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results
