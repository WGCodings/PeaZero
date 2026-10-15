#!/usr/bin/env python3
"""Differential perft test: PeaZero vs python-chess on random-game positions.

    pip install chess
    python tools/perft_diff.py --engine target/release/peazero --games 200 --depth 3

Plays random games with python-chess, samples positions, and compares the
per-move perft breakdown (`peazero perft <depth> <fen>`) against python-chess.
On a mismatch it prints the FEN and the differing root moves, ready to dig
into with `go perft` / Stockfish.
"""
import argparse
import random
import subprocess
import sys

import chess


def py_perft(board: chess.Board, depth: int) -> int:
    if depth == 0:
        return 1
    if depth == 1:
        return board.legal_moves.count()
    n = 0
    for m in board.legal_moves:
        board.push(m)
        n += py_perft(board, depth - 1)
        board.pop()
    return n


def py_divide(board: chess.Board, depth: int) -> dict:
    out = {}
    for m in board.legal_moves:
        board.push(m)
        out[m.uci()] = py_perft(board, depth - 1)
        board.pop()
    return out


def engine_divide(engine: str, fen: str, depth: int) -> dict:
    res = subprocess.run([engine, "perft", str(depth), *fen.split()], capture_output=True, text=True, check=True)
    out = {}
    for line in res.stdout.splitlines():
        if ": " in line and not line.startswith(("Nodes", "Time")):
            mv, n = line.split(": ")
            out[mv] = int(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="target/release/peazero")
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    checked = 0
    for g in range(args.games):
        board = chess.Board()
        stop_at = rng.randint(1, 160)
        for _ in range(stop_at):
            moves = list(board.legal_moves)
            if not moves or board.is_game_over(claim_draw=False):
                break
            board.push(rng.choice(moves))
        fen = board.fen()
        want = py_divide(board, args.depth)
        got = engine_divide(args.engine, fen, args.depth)
        checked += 1
        if want != got:
            print(f"MISMATCH game {g}: {fen}")
            for mv in sorted(set(want) | set(got)):
                if want.get(mv) != got.get(mv):
                    print(f"  {mv}: python-chess={want.get(mv)} peazero={got.get(mv)}")
            return 1
    print(f"OK: {checked} random positions match python-chess at depth {args.depth}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
