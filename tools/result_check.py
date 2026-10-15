#!/usr/bin/env python3
"""Check PeaZero's game-result detection against python-chess.

    python tools/result_check.py --engine target/release/peazero games.pgn

For every game in the PGN, replays it in PeaZero (`position startpos moves
...` + the non-standard `result` command) and compares the verdict with
python-chess's `outcome(claim_draw=True)` at the final position.
"""
import argparse
import subprocess
import sys

import chess
import chess.pgn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="target/release/peazero")
    ap.add_argument("pgn")
    args = ap.parse_args()

    games = []
    with open(args.pgn) as f:
        while (g := chess.pgn.read_game(f)) is not None:
            games.append(g)

    script = []
    expected = []
    for g in games:
        board = g.board()
        moves = []
        for m in g.mainline_moves():
            board.push(m)
            moves.append(m.uci())
        o = board.outcome(claim_draw=True)
        expected.append("*" if o is None else o.result())
        script.append(f"position fen {g.board().fen()} moves {' '.join(moves)}")
        script.append("result")
    script.append("quit")

    out = subprocess.run([args.engine], input="\n".join(script) + "\n", capture_output=True, text=True, check=True)
    got = [line.split()[1] for line in out.stdout.splitlines() if line.startswith("result ")]
    assert len(got) == len(games), f"expected {len(games)} results, got {len(got)}"

    bad = [(i, e, r) for i, (e, r) in enumerate(zip(expected, got)) if e != r]
    for i, e, r in bad[:20]:
        print(f"game {i + 1}: python-chess={e} peazero={r}")
    print(f"{len(games) - len(bad)}/{len(games)} game results agree")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
