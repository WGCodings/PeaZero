"""Check the Python encoding mirror against vectors produced by Rust.

    cd trainer && python -m pytest -q
"""
import json
import pathlib

import chess
import numpy as np
import pytest

from peazero import encoding as enc

VECTORS = pathlib.Path(__file__).resolve().parents[2] / "tests" / "encoding" / "vectors.jsonl"


def load():
    with open(VECTORS) as f:
        return [json.loads(line) for line in f if line.strip()]


ALL = load()
POSITIONS = [v for v in ALL if "fen" in v]
ACTIONS = [v for v in ALL if "action" in v]


def test_version_matches():
    assert {v["version"] for v in ALL} == {enc.ENCODING_VERSION}
    assert len(POSITIONS) > 150 and len(ACTIONS) > 100


@pytest.mark.parametrize("v", POSITIONS, ids=lambda v: v["fen"])
def test_observation(v):
    board = chess.Board(v["fen"])
    obs = enc.encode_board(board, v["repetitions"]).reshape(-1)
    expected = np.zeros(enc.OBS_SIZE, dtype=np.float32)
    for i, val in v["obs"]:
        expected[i] = val
    diff = np.nonzero(obs != expected)[0]
    assert diff.size == 0, f"planes differ at {sorted(set((diff // 64).tolist()))}"


@pytest.mark.parametrize("v", POSITIONS, ids=lambda v: v["fen"])
def test_actions(v):
    board = chess.Board(v["fen"])
    white = board.turn == chess.WHITE
    got = sorted(
        (enc.move_to_action(m.from_square, m.to_square, m.promotion, white), m.uci()) for m in board.legal_moves
    )
    expected = sorted((a, uci) for uci, a in v["actions"])
    assert got == expected


@pytest.mark.parametrize("v", ACTIONS, ids=lambda v: str(v["action"]))
def test_action_planes(v):
    planes = enc.encode_action_planes(v["action"]).reshape(-1)
    assert sorted(np.nonzero(planes)[0].tolist()) == v["planes"]
