//! Network encodings shared by the engine, self-play and (mirrored in
//! Python) the trainer. **Changing anything here invalidates every trained
//! network** — bump `ENCODING_VERSION` and regenerate the test vectors.
//!
//! Everything is from the side to move's point of view: for Black the board
//! is mirrored vertically (`sq ^ 56`) and colours are swapped, so the network
//! always "plays upwards".
//!
//! # Actions (AlphaZero 8×8×73 = 4672)
//! `action = plane * 64 + from_square_relative`
//!
//! | planes | meaning |
//! |---|---|
//! | 0..56  | queen-like moves: `dir * 7 + (distance - 1)`, dir = N, NE, E, SE, S, SW, W, NW |
//! | 56..64 | knight moves in order (1,2) (2,1) (2,-1) (1,-2) (-1,-2) (-2,-1) (-2,1) (-1,2) as (file, rank) deltas |
//! | 64..73 | under-promotions: `64 + piece * 3 + (file_delta + 1)`, piece = N, B, R |
//!
//! Queen promotions use the queen-like plane of the pawn move; castling is
//! the king's two-square move. This layout matches a convolutional policy
//! head with 73 output channels flattened as NCHW.
//!
//! # Observation (21 planes × 64 squares, f32)
//! `index = plane * 64 + square_relative`
//!
//! | plane | content |
//! |---|---|
//! | 0..6   | own P, N, B, R, Q, K |
//! | 6..12  | opponent P, N, B, R, Q, K |
//! | 12     | position occurred before (≥1 repetition) — all ones |
//! | 13     | position occurred twice before (≥2) — all ones |
//! | 14..18 | castling rights: own O-O, own O-O-O, opp O-O, opp O-O-O — all ones |
//! | 18     | en passant target square (one-hot) |
//! | 19     | halfmove clock / 100 (clamped to 1) — constant plane |
//! | 20     | all ones (lets convolutions see the board edge) |

use crate::game::Game;
use crate::moves::{Move, MoveList};
use crate::position::{Position, BLACK_OO, BLACK_OOO, WHITE_OO, WHITE_OOO};
use crate::types::{Color, PieceType, Square};

pub const ENCODING_VERSION: u32 = 1;

pub const POLICY_PLANES: usize = 73;
pub const NUM_ACTIONS: usize = POLICY_PLANES * 64; // 4672

pub const OBS_PLANES: usize = 21;
pub const OBS_SIZE: usize = OBS_PLANES * 64; // 1344

const KNIGHT_DELTAS: [(i8, i8); 8] = [
    (1, 2),
    (2, 1),
    (2, -1),
    (1, -2),
    (-1, -2),
    (-2, -1),
    (-2, 1),
    (-1, 2),
];
const DIRECTIONS: [(i8, i8); 8] = [
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
];

/// Square seen from `stm`'s side of the board.
#[inline(always)]
pub fn relative_square(sq: Square, stm: Color) -> Square {
    match stm {
        Color::White => sq,
        Color::Black => sq.flip_rank(),
    }
}

/// Policy index (0..4672) of a move played by `stm`.
pub fn move_to_action(m: Move, stm: Color) -> usize {
    let from = relative_square(m.from(), stm);
    let to = relative_square(m.to(), stm);
    let df = to.file() as i8 - from.file() as i8;
    let dr = to.rank() as i8 - from.rank() as i8;

    let plane = match m.promo_piece() {
        Some(p) if p != PieceType::Queen => {
            let piece = match p {
                PieceType::Knight => 0,
                PieceType::Bishop => 1,
                PieceType::Rook => 2,
                _ => unreachable!(),
            };
            64 + piece * 3 + (df + 1) as usize
        }
        _ => {
            if let Some(k) = KNIGHT_DELTAS.iter().position(|&d| d == (df, dr)) {
                56 + k
            } else {
                let dir = DIRECTIONS
                    .iter()
                    .position(|&d| d == (df.signum(), dr.signum()))
                    .expect("move is neither a line nor a knight move");
                let dist = df.abs().max(dr.abs()) as usize;
                debug_assert!(df == 0 || dr == 0 || df.abs() == dr.abs());
                dir * 7 + dist - 1
            }
        }
    };
    plane * 64 + from.idx()
}

/// All legal moves with their policy indices.
pub fn legal_actions(pos: &Position) -> Vec<(usize, Move)> {
    let mut list = MoveList::new();
    pos.generate_legal(&mut list);
    let stm = pos.side_to_move();
    list.iter().map(|&m| (move_to_action(m, stm), m)).collect()
}

/// Inverse of `move_to_action` for a concrete position (None if the index
/// doesn't correspond to a legal move there).
pub fn action_to_move(pos: &Position, action: usize) -> Option<Move> {
    let mut list = MoveList::new();
    pos.generate_legal(&mut list);
    let stm = pos.side_to_move();
    list.iter()
        .copied()
        .find(|&m| move_to_action(m, stm) == action)
}

/// Encode a position into `out` (length `OBS_SIZE`, overwritten).
/// `repetitions` = how many times this exact position occurred before
/// (see `Game::repetitions`).
pub fn encode_observation(pos: &Position, repetitions: usize, out: &mut [f32]) {
    assert_eq!(out.len(), OBS_SIZE);
    out.fill(0.0);
    let us = pos.side_to_move();
    let them = !us;

    for (side_offset, color) in [(0, us), (6, them)] {
        for pt in PieceType::ALL {
            let plane = side_offset + pt.idx();
            for sq in pos.pieces_of(color, pt) {
                out[plane * 64 + relative_square(sq, us).idx()] = 1.0;
            }
        }
    }

    let fill = |out: &mut [f32], plane: usize, v: f32| out[plane * 64..(plane + 1) * 64].fill(v);

    if repetitions >= 1 {
        fill(out, 12, 1.0);
    }
    if repetitions >= 2 {
        fill(out, 13, 1.0);
    }

    let cr = pos.castling_rights();
    let (own_oo, own_ooo, opp_oo, opp_ooo) = match us {
        Color::White => (WHITE_OO, WHITE_OOO, BLACK_OO, BLACK_OOO),
        Color::Black => (BLACK_OO, BLACK_OOO, WHITE_OO, WHITE_OOO),
    };
    for (plane, bit) in [(14, own_oo), (15, own_ooo), (16, opp_oo), (17, opp_ooo)] {
        if cr & bit != 0 {
            fill(out, plane, 1.0);
        }
    }

    if let Some(ep) = pos.ep_square() {
        out[18 * 64 + relative_square(ep, us).idx()] = 1.0;
    }

    fill(out, 19, (pos.halfmove_clock() as f32 / 100.0).min(1.0));
    fill(out, 20, 1.0);
}

pub const ACTION_PLANES: usize = 5;
pub const ACTION_SIZE: usize = ACTION_PLANES * 64; // 320

/// Relative (from, to) squares of an action index. `to` is None when the
/// index points off the board (such actions are never legal, but the
/// dynamics network can still be asked about them deep in the tree).
pub fn action_squares(action: usize) -> (Square, Option<Square>) {
    assert!(action < NUM_ACTIONS);
    let plane = action / 64;
    let from = Square((action % 64) as u8);
    let (df, dr) = match plane {
        0..=55 => {
            let (dx, dy) = DIRECTIONS[plane / 7];
            let dist = (plane % 7 + 1) as i8;
            (dx * dist, dy * dist)
        }
        56..=63 => KNIGHT_DELTAS[plane - 56],
        _ => (((plane - 64) % 3) as i8 - 1, 1),
    };
    let f = from.file() as i8 + df;
    let r = from.rank() as i8 + dr;
    let to = ((0..8).contains(&f) && (0..8).contains(&r)).then(|| Square::new(f as u8, r as u8));
    (from, to)
}

/// Encode an action for the dynamics network (`ACTION_SIZE` floats):
/// plane 0 = from square, plane 1 = to square (empty if off-board),
/// planes 2..5 = under-promotion to N / B / R (all ones).
pub fn encode_action_planes(action: usize, out: &mut [f32]) {
    assert_eq!(out.len(), ACTION_SIZE);
    out.fill(0.0);
    let (from, to) = action_squares(action);
    out[from.idx()] = 1.0;
    if let Some(to) = to {
        out[64 + to.idx()] = 1.0;
    }
    let plane = action / 64;
    if plane >= 64 {
        let piece = (plane - 64) / 3;
        out[(2 + piece) * 64..(3 + piece) * 64].fill(1.0);
    }
}

/// Encode the current position of a game (repetitions taken from history).
pub fn encode_game(game: &Game, out: &mut [f32]) {
    encode_observation(game.position(), game.repetitions(), out);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn known_actions() {
        let pos = Position::startpos();
        let m = pos.parse_uci_move("e2e4").unwrap();
        // N, distance 2 → plane 1; from e2 = 12
        assert_eq!(move_to_action(m, Color::White), 64 + 12);
        let m = pos.parse_uci_move("g1f3").unwrap();
        // knight (-1, 2) → plane 56 + 7; from g1 = 6
        assert_eq!(move_to_action(m, Color::White), 63 * 64 + 6);

        // Black's e7e5 mirrors White's e2e4.
        let mut g = Game::default();
        g.push_uci("e2e4");
        let m = g.position().parse_uci_move("e7e5").unwrap();
        assert_eq!(move_to_action(m, Color::Black), 64 + 12);

        // Under-promotion with capture to the left: b7xa8=N → plane 64 + 0*3 + 0
        let pos = Position::from_fen("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let m = pos.parse_uci_move("b7a8n").unwrap();
        assert_eq!(move_to_action(m, Color::White), 64 * 64 + 49);
        // Queen promotion uses the queen-like plane (NW, distance 1 → plane 7*7)
        let m = pos.parse_uci_move("b7a8q").unwrap();
        assert_eq!(move_to_action(m, Color::White), 49 * 64 + 49);
    }

    /// Every legal move in many random positions maps to a unique in-range
    /// index and back.
    #[test]
    fn roundtrip_random_games() {
        let mut rng = Rng::new(2024);
        let mut checked = 0;
        for _ in 0..300 {
            let mut g = Game::default();
            for _ in 0..200 {
                let pos = *g.position();
                let acts = legal_actions(&pos);
                if acts.is_empty() {
                    break;
                }
                let mut seen = std::collections::HashSet::new();
                for &(a, m) in &acts {
                    assert!(a < NUM_ACTIONS);
                    assert!(seen.insert(a), "duplicate action {a} in {}", pos.to_fen());
                    assert_eq!(action_to_move(&pos, a), Some(m));
                    checked += 1;
                }
                let (_, m) = acts[rng.below(acts.len() as u64) as usize];
                g.push(m);
                if g.result() != crate::GameResult::Ongoing {
                    break;
                }
            }
        }
        assert!(checked > 100_000);
    }

    #[test]
    fn action_squares_match_moves() {
        let mut rng = Rng::new(99);
        for _ in 0..100 {
            let mut g = Game::default();
            for _ in 0..120 {
                let pos = *g.position();
                let acts = legal_actions(&pos);
                if acts.is_empty() {
                    break;
                }
                let stm = pos.side_to_move();
                for &(a, m) in &acts {
                    let (from, to) = action_squares(a);
                    assert_eq!(from, relative_square(m.from(), stm));
                    assert_eq!(
                        to,
                        Some(relative_square(m.to(), stm)),
                        "{m} in {}",
                        pos.to_fen()
                    );
                }
                g.push(acts[rng.below(acts.len() as u64) as usize].1);
                if g.result() != crate::GameResult::Ongoing {
                    break;
                }
            }
        }
        // Off-board target: a1 "north-west" by 1.
        assert_eq!(action_squares(49 * 64), (Square::A1, None));
        let mut planes = vec![0.0; ACTION_SIZE];
        encode_action_planes(64 * 64 + 49, &mut planes); // b7xa8=N
        assert_eq!(planes.iter().filter(|&&v| v == 1.0).count(), 2 + 64);
    }

    #[test]
    fn observation_is_side_relative() {
        // After 1. e4 e5, both sides see a mirrored-identical pawn structure
        // except for move-order details; the start position is symmetric.
        let mut a = vec![0.0; OBS_SIZE];
        let mut b = vec![0.0; OBS_SIZE];
        let mut g = Game::default();
        encode_game(&g, &mut a);
        g.push_uci("g1f3");
        g.push_uci("g8f6");
        g.push_uci("f3g1");
        g.push_uci("f6g8");
        encode_game(&g, &mut b);
        // Same position again → only the repetition plane differs.
        let diff: Vec<usize> = (0..OBS_SIZE).filter(|&i| a[i] != b[i]).collect();
        assert!(
            diff.iter().all(|&i| i / 64 == 12 || i / 64 == 19),
            "{diff:?}"
        );
        assert!(diff.iter().any(|&i| i / 64 == 12));

        // Black to move after 1. e4: own pawns on rank 2 (relative), opponent
        // pawn on relative e5 → absolute e4 mirrored = e5.
        let mut g = Game::default();
        g.push_uci("e2e4");
        encode_game(&g, &mut a);
        let own_pawns: Vec<usize> = (0..64).filter(|&s| a[s] == 1.0).collect();
        assert_eq!(own_pawns, (8..16).collect::<Vec<_>>());
        let opp_pawns: Vec<usize> = (0..64).filter(|&s| a[6 * 64 + s] == 1.0).collect();
        assert!(opp_pawns.contains(&Square::parse("e5").unwrap().idx()));
        assert_eq!(a[20 * 64], 1.0);
        assert_eq!(a[14 * 64], 1.0); // own O-O still allowed
    }
}
