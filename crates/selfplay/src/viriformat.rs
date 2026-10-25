//! Export self-play games to viriformat, the game format of Viridithas that
//! bullet and other NNUE trainers read (spec: github.com/cosmobobak/viriformat).
//!
//! A game is a marlinformat `PackedBoard` (32 bytes: start position and the
//! result) followed by one (move, score) pair per ply (4 bytes) and a
//! 4-byte zero terminator. All integers little-endian, squares a1 = 0.
//!
//! * Move: from | to << 6 | promo << 12 | kind << 14, kind 1 = en passant,
//!   2 = castling (written king-takes-rook: e1h1 / e1a1), 3 = promotion
//!   (promo: knight 0, bishop 1, rook 2, queen 3).
//! * Score: the position's search value from self-play (W − L of the
//!   root search, side to move) converted with the engine's `q_to_cp`
//!   curve, white-relative. Positions without a recorded value (data
//!   format v0/v1) get 0.
//! * Result: black win 0, draw 1 (also games stopped at the ply limit),
//!   white win 2.
//! * Fullmove counter: self-play games start from the start position, so
//!   the packed start position carries fullmove 1.

use peazero_core::encoding::action_to_move;
use peazero_core::{Color, Move, PieceType};

use crate::record::{read_chunk, GameRecord, PlyRecord};

pub const PACKED_BOARD_SIZE: usize = 32;
const UNMOVED_ROOK: u8 = 6;

/// Same curve as the engine's `q_to_cp` (engine/src/policy.rs).
pub fn q_to_cp(q: f32) -> i32 {
    let q = q.clamp(-0.999, 0.999) as f64;
    (111.714_640_912 * (1.562_068_842_1 * q).tan()).round() as i32
}

/// White-relative centipawn score of a recorded position.
pub fn white_score(ply: &PlyRecord) -> i16 {
    let Some([w, _, l]) = ply.value_wdl() else {
        return 0;
    };
    let cp = q_to_cp(w - l).clamp(-32_000, 32_000);
    let cp = if ply.stm == Color::White { cp } else { -cp };
    cp as i16
}

/// marlinformat `PackedBoard` of a recorded position.
pub fn pack_board(ply: &PlyRecord, fullmove: u16, eval: i16, wdl: u8) -> [u8; PACKED_BOARD_SIZE] {
    use peazero_core::position::{BLACK_OO, BLACK_OOO, WHITE_OO, WHITE_OOO};
    let mut occupancy = 0u64;
    for bb in ply.pieces {
        occupancy |= bb;
    }
    let mut nibbles = [0u8; 16];
    let mut bits = occupancy;
    let mut i = 0;
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        let k = (0..12).find(|&k| ply.pieces[k] >> sq & 1 != 0).unwrap();
        let (black, pt) = (k >= 6, (k % 6) as u8);
        let mut code = pt;
        if pt == PieceType::Rook.idx() as u8 {
            // Castling rights: the rook that may still castle is "unmoved".
            let (oo, ooo, home) = if black {
                (BLACK_OO, BLACK_OOO, 56)
            } else {
                (WHITE_OO, WHITE_OOO, 0)
            };
            if (sq == home + 7 && ply.castling & oo != 0) || (sq == home && ply.castling & ooo != 0)
            {
                code = UNMOVED_ROOK;
            }
        }
        let piece = code | (black as u8) << 3;
        nibbles[i / 2] |= piece << ((i % 2) * 4);
        i += 1;
    }
    let mut out = [0u8; PACKED_BOARD_SIZE];
    out[0..8].copy_from_slice(&occupancy.to_le_bytes());
    out[8..24].copy_from_slice(&nibbles);
    let ep = ply.ep.map_or(64, |s| s.idx() as u8);
    out[24] = ((ply.stm == Color::Black) as u8) << 7 | ep;
    out[25] = ply.halfmove;
    out[26..28].copy_from_slice(&fullmove.to_le_bytes());
    out[28..30].copy_from_slice(&eval.to_le_bytes());
    out[30] = wdl;
    out[31] = 0;
    out
}

/// viriformat 16-bit move.
pub fn encode_move(m: Move) -> u16 {
    let from = m.from().idx() as u16;
    let mut to = m.to().idx() as u16;
    let kind = if m.is_castle() {
        // king-takes-rook: the rook stands on the h-file (kingside) or a-file
        to = if to > from {
            from - from % 8 + 7
        } else {
            from - from % 8
        };
        2
    } else if m.is_promo() {
        3
    } else if m.is_ep() {
        1
    } else {
        0
    };
    let promo = if m.is_promo() { m.flag() & 3 } else { 0 };
    from | to << 6 | promo << 12 | kind << 14
}

/// Append one game; returns the number of positions written.
pub fn write_game(g: &GameRecord, out: &mut Vec<u8>) -> Result<usize, String> {
    let Some(first) = g.plies.first() else {
        return Ok(0);
    };
    let wdl = match g.result {
        1 => 2,
        -1 => 0,
        _ => 1,
    };
    out.extend_from_slice(&pack_board(first, 1, 0, wdl));
    for p in &g.plies {
        let pos = p.position();
        let m = action_to_move(&pos, p.action as usize)
            .ok_or_else(|| format!("action {} is not legal in {}", p.action, pos.to_fen()))?;
        out.extend_from_slice(&encode_move(m).to_le_bytes());
        out.extend_from_slice(&white_score(p).to_le_bytes());
    }
    out.extend_from_slice(&[0; 4]);
    Ok(g.plies.len())
}

/// Convert a self-play data chunk (any data format version) to viriformat
/// bytes. Returns (bytes, games, positions).
pub fn chunk_to_viriformat(chunk: &[u8]) -> Result<(Vec<u8>, usize, usize), String> {
    let games = read_chunk(chunk)?;
    let mut out = Vec::with_capacity(chunk.len());
    let mut positions = 0;
    for g in &games {
        positions += write_game(g, &mut out)?;
    }
    Ok((out, games.len(), positions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{write_chunk, Termination};
    use peazero_core::encoding::move_to_action;
    use peazero_core::Game;

    /// Record a game from UCI moves (every ply gets a value), like self-play.
    fn record(moves: &[&str], value_white: f32) -> GameRecord {
        let mut game = Game::default();
        let mut plies = Vec::new();
        for u in moves {
            let m = game.position().parse_uci_move(u).unwrap();
            let stm = game.position().side_to_move();
            let a = move_to_action(m, stm);
            let q = if stm == Color::White {
                value_white
            } else {
                -value_white
            };
            let w = (1.0 + q) / 2.0;
            plies.push(
                PlyRecord::new(game.position(), game.repetitions(), a, &[(a, 1.0)]).with_value([
                    w,
                    0.0,
                    1.0 - w,
                ]),
            );
            game.push(m);
        }
        let (result, termination) = match Termination::from_result(game.result()) {
            Some(x) => x,
            None => (0, Termination::MaxPlies),
        };
        GameRecord {
            plies,
            result,
            termination,
        }
    }

    #[test]
    fn matches_the_spec_example() {
        // README example: 1. e4 e5 2. Qh5 Ke7 3. Qxe5#, white wins. Scores
        // differ (ours come from the search value), so compare the rest.
        peazero_core::init();
        let g = record(&["e2e4", "e7e5", "d1h5", "e8e7", "h5e5"], 0.0);
        assert_eq!(g.result, 1);
        let mut out = Vec::new();
        write_game(&g, &mut out).unwrap();
        let expected_board: [u8; 32] = [
            0xff, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0xff, // occupancy
            0x16, 0x42, 0x25, 0x61, 0x00, 0x00, 0x00, 0x00, // pieces
            0x88, 0x88, 0x88, 0x88, 0x9e, 0xca, 0xad, 0xe9, //
            0x40, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, // stm/ep, clocks, score, result
        ];
        assert_eq!(&out[..32], &expected_board);
        let moves: Vec<u16> = out[32..52]
            .chunks(4)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        assert_eq!(moves, vec![0x070c, 0x0934, 0x09c3, 0x0d3c, 0x0927]);
        assert_eq!(&out[52..], &[0, 0, 0, 0]);
    }

    #[test]
    fn scores_are_white_relative() {
        peazero_core::init();
        let g = record(&["e2e4", "e7e5", "g1f3"], 0.5); // white is better throughout
        let mut out = Vec::new();
        write_game(&g, &mut out).unwrap();
        let scores: Vec<i16> = out[32..44]
            .chunks(4)
            .map(|c| i16::from_le_bytes([c[2], c[3]]))
            .collect();
        assert!(scores.iter().all(|&s| s > 50), "{scores:?}");
        assert_eq!(scores[0], scores[1]);
        assert_eq!(q_to_cp(0.0), 0);
    }

    #[test]
    fn castling_ep_and_promotion_moves() {
        peazero_core::init();
        let mut game = Game::default();
        for u in ["e2e4", "g8f6", "g1f3", "f6e4", "f1c4", "d7d5"] {
            assert!(game.push_uci(u));
        }
        let castle = game.position().parse_uci_move("e1g1").unwrap();
        assert_eq!(encode_move(castle), 4 | 7 << 6 | 2 << 14); // e1 takes h1
        let g = Game::from_fen("r3k2r/8/8/8/8/8/8/4K3 b kq - 0 1").unwrap();
        let ooo = g.position().parse_uci_move("e8c8").unwrap();
        assert_eq!(encode_move(ooo), 60 | 56 << 6 | 2 << 14); // e8 takes a8
        let g = Game::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let ep = g.position().parse_uci_move("e5d6").unwrap();
        assert_eq!(encode_move(ep), 36 | 43 << 6 | 1 << 14);
        let g = Game::from_fen("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let q = g.position().parse_uci_move("a7a8q").unwrap();
        assert_eq!(encode_move(q), 48 | 56 << 6 | 3 << 12 | 3 << 14);
        let n = g.position().parse_uci_move("a7b8n").unwrap();
        assert_eq!(encode_move(n), 48 | 57 << 6 | 3 << 14);
    }

    #[test]
    fn castling_rights_mark_unmoved_rooks() {
        peazero_core::init();
        // White may castle short only, black long only.
        let g = Game::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1").unwrap();
        let p = PlyRecord::new(g.position(), 0, 0, &[]);
        let b = pack_board(&p, 1, 0, 1);
        let nib = |i: usize| (b[8 + i / 2] >> ((i % 2) * 4)) & 0xF;
        // occupancy order: a1, e1, h1, a8, e8, h8
        assert_eq!(
            [nib(0), nib(1), nib(2), nib(3), nib(4), nib(5)],
            [3, 5, 6, 8 | 6, 8 | 5, 8 | 3]
        );
    }

    #[test]
    fn chunk_converts_every_game() {
        peazero_core::init();
        let a = record(&["e2e4", "e7e5", "d1h5", "e8e7", "h5e5"], 0.2);
        let b = record(&["d2d4", "d7d5"], -0.1);
        let (bytes, games, positions) = chunk_to_viriformat(&write_chunk(&[a, b])).unwrap();
        assert_eq!((games, positions), (2, 7));
        assert_eq!(bytes.len(), 2 * 32 + 7 * 4 + 2 * 4);
        assert_eq!(bytes[32 + 5 * 4 + 4 + 30], 1); // second game: draw (ply limit)
    }
}

/// Cross-check against the reference implementation (the `viriformat`
/// crate): our bytes must decode to the same positions, with every move
/// legal, over many random games (castling, en passant, promotions).
#[cfg(test)]
mod reference {
    use super::*;
    use crate::record::Termination;
    use peazero_core::encoding::{legal_actions, move_to_action};
    use peazero_core::rng::Rng;
    use peazero_core::Game;
    use std::io::BufReader;
    use viriformat::dataformat::Game as ViriGame;

    fn random_game(rng: &mut Rng) -> (GameRecord, Vec<String>) {
        let mut game = Game::default();
        let mut plies = Vec::new();
        let mut fens = Vec::new();
        while game.result() == peazero_core::GameResult::Ongoing && plies.len() < 400 {
            let moves = legal_actions(game.position());
            let (a, m) = moves[rng.below(moves.len() as u64) as usize];
            assert_eq!(move_to_action(m, game.position().side_to_move()), a);
            let w = rng.below(1000) as f32 / 1000.0;
            fens.push(game.position().to_fen());
            plies.push(
                PlyRecord::new(game.position(), game.repetitions(), a, &[(a, 1.0)]).with_value([
                    w,
                    0.0,
                    1.0 - w,
                ]),
            );
            game.push(m);
        }
        let (result, termination) =
            Termination::from_result(game.result()).unwrap_or((0, Termination::MaxPlies));
        (
            GameRecord {
                plies,
                result,
                termination,
            },
            fens,
        )
    }

    /// Placement, side to move, castling (ep is written differently by
    /// different engines: only when a capture is possible, or always).
    fn key(fen: &str) -> String {
        fen.split(' ').take(3).collect::<Vec<_>>().join(" ")
    }

    #[test]
    fn reference_crate_reads_our_games() {
        peazero_core::init();
        let mut rng = Rng::new(11);
        let (mut castles, mut eps, mut promos) = (0, 0, 0);
        for _ in 0..300 {
            let (g, fens) = random_game(&mut rng);
            let mut bytes = Vec::new();
            write_game(&g, &mut bytes).unwrap();
            let vg =
                ViriGame::deserialise_from(&mut BufReader::new(&bytes[..]), Vec::new()).unwrap();
            let want = match g.result {
                1 => viriformat::dataformat::WDL::Win,
                -1 => viriformat::dataformat::WDL::Loss,
                _ => viriformat::dataformat::WDL::Draw,
            };
            assert_eq!(vg.outcome(), want);
            let mut board = vg.initial_position();
            assert_eq!(vg.len(), fens.len());
            for (i, (mv, _)) in vg.moves.iter().enumerate() {
                assert_eq!(key(&board.to_string()), key(&fens[i]), "ply {i}");
                assert!(
                    board.legal_moves().contains(mv),
                    "illegal {mv:?} at {}",
                    fens[i]
                );
                castles += mv.is_castle() as u32;
                eps += mv.is_ep() as u32;
                promos += mv.is_promo() as u32;
                board.make_move_simple(*mv);
            }
            let scores: Vec<i16> = vg.moves.iter().map(|(_, s)| s.get()).collect();
            let ours: Vec<i16> = g.plies.iter().map(white_score).collect();
            assert_eq!(scores, ours);
        }
        assert!(
            castles > 0 && eps > 0 && promos > 0,
            "castles {castles} ep {eps} promos {promos}"
        );
    }

    /// Games written by the reference implementation (as other engines' data
    /// is, e.g. Pea's NNUE data) read back correctly with vf_reader.
    #[test]
    fn vf_reader_reads_reference_written_games() {
        peazero_core::init();
        let mut rng = Rng::new(23);
        let (mut castles, mut eps, mut promos) = (0, 0, 0);
        for _ in 0..300 {
            let (g, fens) = random_game(&mut rng);
            let mut ours = Vec::new();
            write_game(&g, &mut ours).unwrap();
            let vg =
                ViriGame::deserialise_from(&mut BufReader::new(&ours[..]), Vec::new()).unwrap();
            let mut theirs = Vec::new();
            vg.serialise_into(&mut theirs).unwrap();
            let back = crate::vf_reader::read_all(&theirs).unwrap();
            assert_eq!(back.len(), 1);
            let b = &back[0];
            assert_eq!(b.len(), fens.len());
            for (i, (v, p)) in b.iter().zip(&g.plies).enumerate() {
                assert_eq!(key(&v.pos.to_fen()), key(&fens[i]), "ply {i}");
                assert_eq!(
                    move_to_action(v.mv, v.pos.side_to_move()) as u16,
                    p.action,
                    "ply {i}"
                );
                castles += v.mv.is_castle() as u32;
                eps += v.mv.is_ep() as u32;
                promos += v.mv.is_promo() as u32;
                let cp = white_score(p);
                assert_eq!(v.score, if p.stm == Color::White { cp } else { -cp });
                assert_eq!(
                    v.result,
                    if p.stm == Color::White {
                        g.result
                    } else {
                        -g.result
                    }
                );
            }
        }
        assert!(
            castles > 0 && eps > 0 && promos > 0,
            "castles {castles} ep {eps} promos {promos}"
        );
    }
}
