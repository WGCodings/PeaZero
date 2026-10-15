use std::fmt;

use crate::attacks::{bishop_attacks, king_attacks, knight_attacks, pawn_attacks, rook_attacks};
use crate::bitboard::Bitboard;
use crate::moves::Move;
use crate::types::{Color, Piece, PieceType, Square};
use crate::zobrist::KEYS;

pub const START_FEN: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub const WHITE_OO: u8 = 1;
pub const WHITE_OOO: u8 = 2;
pub const BLACK_OO: u8 = 4;
pub const BLACK_OOO: u8 = 8;

/// Castling rights that survive a move touching each square.
const CASTLE_KEEP: [u8; 64] = {
    let mut t = [0xFu8; 64];
    t[0] = !WHITE_OOO & 0xF; // a1
    t[4] = !(WHITE_OO | WHITE_OOO) & 0xF; // e1
    t[7] = !WHITE_OO & 0xF; // h1
    t[56] = !BLACK_OOO & 0xF; // a8
    t[60] = !(BLACK_OO | BLACK_OOO) & 0xF; // e8
    t[63] = !BLACK_OO & 0xF; // h8
    t
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FenError(pub String);

impl fmt::Display for FenError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid FEN: {}", self.0)
    }
}

impl std::error::Error for FenError {}

/// Full board state. `Copy` on purpose: the engine uses copy-make
/// (`let mut child = *pos; child.make_move(m);`), which is simple, fast
/// enough, and ideal for tree search where nodes keep their own position.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Position {
    pieces: [Bitboard; 6],
    colors: [Bitboard; 2],
    board: [Piece; 64],
    stm: Color,
    castling: u8,
    /// Only set when the side to move has a pawn that could capture en
    /// passant (pseudo-legally). Keeps hashing/repetition exact.
    ep: Option<Square>,
    halfmove: u16,
    fullmove: u16,
    hash: u64,
}

impl Default for Position {
    fn default() -> Self {
        Position::startpos()
    }
}

impl Position {
    fn empty() -> Position {
        Position {
            pieces: [Bitboard::EMPTY; 6],
            colors: [Bitboard::EMPTY; 2],
            board: [Piece::NONE; 64],
            stm: Color::White,
            castling: 0,
            ep: None,
            halfmove: 0,
            fullmove: 1,
            hash: 0,
        }
    }

    pub fn startpos() -> Position {
        Position::from_fen(START_FEN).expect("start FEN is valid")
    }

    // ----------------------------------------------------------- accessors

    #[inline(always)]
    pub fn side_to_move(&self) -> Color {
        self.stm
    }
    #[inline(always)]
    pub fn castling_rights(&self) -> u8 {
        self.castling
    }
    #[inline(always)]
    pub fn ep_square(&self) -> Option<Square> {
        self.ep
    }
    #[inline(always)]
    pub fn halfmove_clock(&self) -> u16 {
        self.halfmove
    }
    #[inline(always)]
    pub fn fullmove_number(&self) -> u16 {
        self.fullmove
    }
    #[inline(always)]
    pub fn hash(&self) -> u64 {
        self.hash
    }
    #[inline(always)]
    pub fn piece_on(&self, sq: Square) -> Piece {
        self.board[sq.idx()]
    }
    #[inline(always)]
    pub fn pieces(&self, pt: PieceType) -> Bitboard {
        self.pieces[pt.idx()]
    }
    #[inline(always)]
    pub fn color_bb(&self, c: Color) -> Bitboard {
        self.colors[c.idx()]
    }
    #[inline(always)]
    pub fn pieces_of(&self, c: Color, pt: PieceType) -> Bitboard {
        self.pieces[pt.idx()] & self.colors[c.idx()]
    }
    #[inline(always)]
    pub fn occupied(&self) -> Bitboard {
        self.colors[0] | self.colors[1]
    }
    #[inline(always)]
    pub fn king_sq(&self, c: Color) -> Square {
        self.pieces_of(c, PieceType::King).lsb()
    }

    /// Pieces of either colour attacking `sq` given occupancy `occ`.
    #[inline]
    pub fn attackers_to(&self, sq: Square, occ: Bitboard) -> Bitboard {
        let rq = self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen);
        let bq = self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen);
        (pawn_attacks(Color::White, sq) & self.pieces_of(Color::Black, PieceType::Pawn))
            | (pawn_attacks(Color::Black, sq) & self.pieces_of(Color::White, PieceType::Pawn))
            | (knight_attacks(sq) & self.pieces(PieceType::Knight))
            | (king_attacks(sq) & self.pieces(PieceType::King))
            | (rook_attacks(sq, occ) & rq)
            | (bishop_attacks(sq, occ) & bq)
    }

    /// Is `sq` attacked by any piece of colour `by`, with occupancy `occ`?
    #[inline]
    pub fn is_attacked(&self, sq: Square, by: Color, occ: Bitboard) -> bool {
        let them = self.colors[by.idx()];
        if (pawn_attacks(!by, sq) & self.pieces(PieceType::Pawn) & them).any()
            || (knight_attacks(sq) & self.pieces(PieceType::Knight) & them).any()
            || (king_attacks(sq) & self.pieces(PieceType::King) & them).any()
        {
            return true;
        }
        let rq = (self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen)) & them;
        let bq = (self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen)) & them;
        (rook_attacks(sq, occ) & rq).any() || (bishop_attacks(sq, occ) & bq).any()
    }

    /// Enemy pieces giving check to the side to move.
    #[inline]
    pub fn checkers(&self) -> Bitboard {
        let us = self.stm;
        self.attackers_to(self.king_sq(us), self.occupied()) & self.colors[(!us).idx()]
    }

    #[inline]
    pub fn in_check(&self) -> bool {
        self.is_attacked(self.king_sq(self.stm), !self.stm, self.occupied())
    }

    // ------------------------------------------------------ piece updates

    #[inline(always)]
    fn add_piece(&mut self, p: Piece, sq: Square) {
        let b = sq.bb();
        self.pieces[p.kind().idx()] |= b;
        self.colors[p.color().idx()] |= b;
        self.board[sq.idx()] = p;
        self.hash ^= KEYS.piece_sq[p.idx()][sq.idx()];
    }

    #[inline(always)]
    fn remove_piece(&mut self, sq: Square) -> Piece {
        let p = self.board[sq.idx()];
        debug_assert!(!p.is_none());
        let b = sq.bb();
        self.pieces[p.kind().idx()] ^= b;
        self.colors[p.color().idx()] ^= b;
        self.board[sq.idx()] = Piece::NONE;
        self.hash ^= KEYS.piece_sq[p.idx()][sq.idx()];
        p
    }

    #[inline(always)]
    fn move_piece(&mut self, from: Square, to: Square) {
        let p = self.remove_piece(from);
        self.add_piece(p, to);
    }

    /// Would `sq` be a usable en passant square for `stm` (an enemy pawn of
    /// `stm` can capture onto it)?
    #[inline(always)]
    fn ep_capturable(&self, sq: Square, stm: Color) -> bool {
        (pawn_attacks(!stm, sq) & self.pieces_of(stm, PieceType::Pawn)).any()
    }

    // ------------------------------------------------------------- moves

    /// Apply a (pseudo-)legal move in place. No legality check is done;
    /// callers use legal movegen or `is_legal` first.
    pub fn make_move(&mut self, m: Move) {
        let us = self.stm;
        let from = m.from();
        let to = m.to();
        let moving = self.board[from.idx()];
        debug_assert!(
            !moving.is_none() && moving.color() == us,
            "bad move {m} in {}",
            self.to_fen()
        );

        self.hash ^= KEYS.castling[self.castling as usize];
        if let Some(ep) = self.ep.take() {
            self.hash ^= KEYS.ep_file[ep.file() as usize];
        }
        self.halfmove += 1;

        if m.is_capture() {
            let cap_sq = if m.is_ep() { Square(to.0 ^ 8) } else { to };
            self.remove_piece(cap_sq);
            self.halfmove = 0;
        }

        self.move_piece(from, to);

        match moving.kind() {
            PieceType::Pawn => {
                self.halfmove = 0;
                if let Some(promo) = m.promo_piece() {
                    self.remove_piece(to);
                    self.add_piece(Piece::new(us, promo), to);
                } else if m.is_double_push() {
                    let ep = Square((from.0 + to.0) / 2);
                    if self.ep_capturable(ep, !us) {
                        self.ep = Some(ep);
                        self.hash ^= KEYS.ep_file[ep.file() as usize];
                    }
                }
            }
            PieceType::King if m.is_castle() => {
                let (rf, rt) = if m.flag() == Move::KING_CASTLE {
                    (Square(to.0 + 1), Square(to.0 - 1))
                } else {
                    (Square(to.0 - 2), Square(to.0 + 1))
                };
                self.move_piece(rf, rt);
            }
            _ => {}
        }

        self.castling &= CASTLE_KEEP[from.idx()] & CASTLE_KEEP[to.idx()];
        self.hash ^= KEYS.castling[self.castling as usize];

        if us == Color::Black {
            self.fullmove += 1;
        }
        self.stm = !us;
        self.hash ^= KEYS.side;
    }

    /// Pass the turn (for future null-move style uses / debugging).
    pub fn make_null_move(&mut self) {
        if let Some(ep) = self.ep.take() {
            self.hash ^= KEYS.ep_file[ep.file() as usize];
        }
        self.halfmove += 1;
        if self.stm == Color::Black {
            self.fullmove += 1;
        }
        self.stm = !self.stm;
        self.hash ^= KEYS.side;
    }

    /// Hash recomputed from scratch (used to verify incremental updates).
    pub fn compute_hash(&self) -> u64 {
        let mut h = 0u64;
        for sq in self.occupied() {
            h ^= KEYS.piece_sq[self.board[sq.idx()].idx()][sq.idx()];
        }
        h ^= KEYS.castling[self.castling as usize];
        if let Some(ep) = self.ep {
            h ^= KEYS.ep_file[ep.file() as usize];
        }
        if self.stm == Color::Black {
            h ^= KEYS.side;
        }
        h
    }

    /// Insufficient mating material for both sides (FIDE dead positions
    /// that are easy to detect: K v K, K+minor v K, K+B v K+B same colour).
    pub fn insufficient_material(&self) -> bool {
        let heavy = self.pieces(PieceType::Pawn)
            | self.pieces(PieceType::Rook)
            | self.pieces(PieceType::Queen);
        if heavy.any() {
            return false;
        }
        let knights = self.pieces(PieceType::Knight);
        let bishops = self.pieces(PieceType::Bishop);
        let minors = knights | bishops;
        if minors.count() <= 1 {
            return true;
        }
        // Only bishops, all on the same colour complex (either side).
        knights.is_empty()
            && ((bishops & Bitboard::LIGHT_SQUARES).is_empty()
                || (bishops & Bitboard::DARK_SQUARES).is_empty())
    }

    // --------------------------------------------------------------- FEN

    pub fn from_fen(fen: &str) -> Result<Position, FenError> {
        let mut parts = fen.split_whitespace();
        let placement = parts.next().ok_or_else(|| FenError("empty".into()))?;
        let stm = parts.next().unwrap_or("w");
        let castling = parts.next().unwrap_or("-");
        let ep = parts.next().unwrap_or("-");
        let halfmove = parts.next().unwrap_or("0");
        let fullmove = parts.next().unwrap_or("1");

        let mut pos = Position::empty();

        let ranks: Vec<&str> = placement.split('/').collect();
        if ranks.len() != 8 {
            return Err(FenError(format!("expected 8 ranks, got {}", ranks.len())));
        }
        for (i, rank_str) in ranks.iter().enumerate() {
            let rank = 7 - i as u8;
            let mut file = 0u8;
            for c in rank_str.chars() {
                if let Some(d) = c.to_digit(10) {
                    if !(1..=8).contains(&d) {
                        return Err(FenError(format!("bad digit '{c}'")));
                    }
                    file += d as u8;
                } else {
                    let p =
                        Piece::from_char(c).ok_or_else(|| FenError(format!("bad piece '{c}'")))?;
                    if file >= 8 {
                        return Err(FenError(format!("rank {} too long", rank + 1)));
                    }
                    pos.add_piece(p, Square::new(file, rank));
                    file += 1;
                }
                if file > 8 {
                    return Err(FenError(format!("rank {} too long", rank + 1)));
                }
            }
            if file != 8 {
                return Err(FenError(format!("rank {} has {} files", rank + 1, file)));
            }
        }

        for c in Color::BOTH {
            if pos.pieces_of(c, PieceType::King).count() != 1 {
                return Err(FenError(format!("{c:?} must have exactly one king")));
            }
        }
        if (pos.pieces(PieceType::Pawn) & (Bitboard::RANK_1 | Bitboard::RANK_8)).any() {
            return Err(FenError("pawn on back rank".into()));
        }

        pos.stm = match stm {
            "w" => Color::White,
            "b" => Color::Black,
            _ => return Err(FenError(format!("bad side to move '{stm}'"))),
        };

        if castling != "-" {
            for c in castling.chars() {
                pos.castling |= match c {
                    'K' => WHITE_OO,
                    'Q' => WHITE_OOO,
                    'k' => BLACK_OO,
                    'q' => BLACK_OOO,
                    _ => return Err(FenError(format!("bad castling char '{c}'"))),
                };
            }
        }
        // Drop rights that don't match king/rook placement.
        let wk = pos.piece_on(Square::E1) == Piece::new(Color::White, PieceType::King);
        let bk = pos.piece_on(Square::E8) == Piece::new(Color::Black, PieceType::King);
        let wr = Piece::new(Color::White, PieceType::Rook);
        let br = Piece::new(Color::Black, PieceType::Rook);
        if !(wk && pos.piece_on(Square::H1) == wr) {
            pos.castling &= !WHITE_OO;
        }
        if !(wk && pos.piece_on(Square::A1) == wr) {
            pos.castling &= !WHITE_OOO;
        }
        if !(bk && pos.piece_on(Square::H8) == br) {
            pos.castling &= !BLACK_OO;
        }
        if !(bk && pos.piece_on(Square::A8) == br) {
            pos.castling &= !BLACK_OOO;
        }

        if ep != "-" {
            let sq = Square::parse(ep).ok_or_else(|| FenError(format!("bad ep square '{ep}'")))?;
            let expected_rank = if pos.stm == Color::White { 5 } else { 2 };
            if sq.rank() == expected_rank && pos.ep_capturable(sq, pos.stm) {
                pos.ep = Some(sq);
            }
        }

        pos.halfmove = halfmove
            .parse()
            .map_err(|_| FenError(format!("bad halfmove '{halfmove}'")))?;
        pos.fullmove = fullmove
            .parse::<u16>()
            .map_err(|_| FenError(format!("bad fullmove '{fullmove}'")))?
            .max(1);

        // The side that just moved must not be in check.
        let them = !pos.stm;
        if pos.is_attacked(pos.king_sq(them), pos.stm, pos.occupied()) {
            return Err(FenError("side not to move is in check".into()));
        }

        pos.hash = pos.compute_hash();
        Ok(pos)
    }

    pub fn to_fen(&self) -> String {
        let mut s = String::with_capacity(90);
        for rank in (0..8).rev() {
            let mut empty = 0;
            for file in 0..8 {
                let p = self.board[Square::new(file, rank).idx()];
                if p.is_none() {
                    empty += 1;
                } else {
                    if empty > 0 {
                        s.push(char::from(b'0' + empty));
                        empty = 0;
                    }
                    s.push(p.to_char());
                }
            }
            if empty > 0 {
                s.push(char::from(b'0' + empty));
            }
            if rank > 0 {
                s.push('/');
            }
        }
        s.push(' ');
        s.push(if self.stm == Color::White { 'w' } else { 'b' });
        s.push(' ');
        if self.castling == 0 {
            s.push('-');
        } else {
            for (bit, c) in [
                (WHITE_OO, 'K'),
                (WHITE_OOO, 'Q'),
                (BLACK_OO, 'k'),
                (BLACK_OOO, 'q'),
            ] {
                if self.castling & bit != 0 {
                    s.push(c);
                }
            }
        }
        s.push(' ');
        match self.ep {
            Some(sq) => s.push_str(&sq.to_string()),
            None => s.push('-'),
        }
        s.push_str(&format!(" {} {}", self.halfmove, self.fullmove));
        s
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, " +---+---+---+---+---+---+---+---+")?;
        for rank in (0..8).rev() {
            for file in 0..8 {
                write!(
                    f,
                    " | {}",
                    self.board[Square::new(file, rank).idx()].to_char()
                )?;
            }
            writeln!(f, " | {}", rank + 1)?;
            writeln!(f, " +---+---+---+---+---+---+---+---+")?;
        }
        writeln!(f, "   a   b   c   d   e   f   g   h")?;
        writeln!(f)?;
        writeln!(f, "Fen: {}", self.to_fen())?;
        write!(f, "Key: {:016X}", self.hash)
    }
}

impl fmt::Debug for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Position({})", self.to_fen())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fen_roundtrip() {
        for fen in [
            START_FEN,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ] {
            assert_eq!(Position::from_fen(fen).unwrap().to_fen(), fen);
        }
    }

    #[test]
    fn useless_ep_square_is_dropped() {
        let p = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
            .unwrap();
        assert_eq!(p.ep_square(), None);
    }

    #[test]
    fn rejects_bad_fens() {
        for fen in [
            "",
            "8/8/8/8/8/8/8/8 w - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP w KQkq - 0 1",
            "rnbqkbnr/pppppppp/9/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR x KQkq - 0 1",
            "4k3/8/8/8/8/8/8/4K2r b - - 0 1", // white (not to move) in check
        ] {
            assert!(Position::from_fen(fen).is_err(), "{fen}");
        }
    }

    #[test]
    fn insufficient() {
        let f = |s| Position::from_fen(s).unwrap().insufficient_material();
        assert!(f("8/8/4k3/8/8/3K4/8/8 w - - 0 1"));
        assert!(f("8/8/4k3/8/8/3KN3/8/8 w - - 0 1"));
        assert!(f("8/8/4kb2/8/8/3KB3/8/8 w - - 0 1")); // f6 dark, e3 dark
        assert!(!f("8/8/4k1b1/8/8/3KB3/8/8 w - - 0 1")); // g6 light, e3 dark
        assert!(!f("8/8/4k3/8/8/3KNN2/8/8 w - - 0 1"));
        assert!(!f("8/8/4k3/8/8/3KP3/8/8 w - - 0 1"));
    }
}
