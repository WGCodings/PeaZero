"""Statistics of the background evaluator (no networks needed)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from peazero.evaluator import match_stats, puzzle_performance, puzzle_stats  # noqa: E402


def games(scores):
    """Pairs of games: scores[i] = (game A white, game B white) for opening i."""
    out = []
    for i, pair in enumerate(scores):
        for j, s in enumerate(pair):
            out.append({"opening": i, "a_white": j == 0, "score_a": s, "termination": "checkmate", "plies": 80})
    return out


def test_even_match_is_zero_elo_with_an_interval():
    st = match_stats(games([(1.0, 0.0)] * 25 + [(0.5, 0.5)] * 25))
    assert st["elo"] == 0.0
    assert st["elo_lo"] < 0 < st["elo_hi"]
    assert st["pentanomial"] == [0, 0, 50, 0, 0]
    assert (st["wins"], st["draws"], st["losses"]) == (25, 50, 25)


def test_winning_match_is_positive_and_sweeps_are_finite():
    st = match_stats(games([(1.0, 0.5)] * 50))
    assert 150 < st["elo"] < 250 and st["elo_lo"] > 0
    sweep = match_stats(games([(1.0, 1.0)] * 50))
    assert 500 < sweep["elo"] < 1000
    assert match_stats(games([(0.5, 0.5)] * 50))["elo_hi"] > 0  # never a zero-width interval


def test_puzzle_performance():
    assert round(puzzle_performance([1000] * 10, [True] * 5 + [False] * 5)) == 1000
    r = puzzle_performance([800, 1200, 1600, 2000], [True, True, False, False])
    assert 1300 < r < 1500
    assert puzzle_performance([1000, 1500], [False, False]) < 1000
    assert puzzle_performance([1000, 1500], [True, True]) > 1500


def test_puzzle_bands():
    puzzles = [{"rating": r} for r in (600, 900, 1200, 2500)]
    st = puzzle_stats(puzzles, [(True, True), (True, False), (False, False), (True, True)])
    assert st["solved"] == 0.5 and st["first_move"] == 0.75
    assert st["bands"]["<1000"] == {"n": 2, "first_move": 1.0, "solved": 0.5}
    assert st["bands"]["2200+"]["solved"] == 1.0
    assert "1400-1800" not in st["bands"]
