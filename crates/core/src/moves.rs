use std::fmt;
use std::ops::Deref;

use crate::types::{PieceType, Square};

/// Packed move: bits 0-5 from, 6-11 to, 12-15 flags.
///
/// | flag | meaning                    |
/// |------|----------------------------|
/// | 0    | quiet                      |
/// | 1    | double pawn push           |
/// | 2    | king-side castle           |
/// | 3    | queen-side castle          |
/// | 4    | capture                    |
/// | 5    | en passant capture         |
/// | 8-11 | promotion (N, B, R, Q)     |
/// | 12-15| capture-promotion (N,B,R,Q)|
///
/// Castling is encoded as the king's move (e1g1, e1c1), which is also its
/// UCI notation.
#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Move(u16);

impl Move {
    pub const NULL: Move = Move(0);

    pub const QUIET: u16 = 0;
    pub const DOUBLE_PUSH: u16 = 1;
    pub const KING_CASTLE: u16 = 2;
    pub const QUEEN_CASTLE: u16 = 3;
    pub const CAPTURE: u16 = 4;
    pub const EP_CAPTURE: u16 = 5;
    pub const PROMO: u16 = 8;
    pub const PROMO_CAPTURE: u16 = 12;

    #[inline(always)]
    pub const fn new(from: Square, to: Square, flag: u16) -> Move {
        Move(from.0 as u16 | (to.0 as u16) << 6 | flag << 12)
    }

    /// Promotion move; `promo` must be N, B, R or Q.
    #[inline(always)]
    pub const fn new_promo(from: Square, to: Square, promo: PieceType, capture: bool) -> Move {
        let base = if capture {
            Self::PROMO_CAPTURE
        } else {
            Self::PROMO
        };
        Move::new(from, to, base | (promo as u16 - 1))
    }

    #[inline(always)]
    pub const fn raw(self) -> u16 {
        self.0
    }

    #[inline(always)]
    pub const fn from_raw(raw: u16) -> Move {
        Move(raw)
    }

    #[inline(always)]
    pub const fn from(self) -> Square {
        Square((self.0 & 63) as u8)
    }

    #[inline(always)]
    pub const fn to(self) -> Square {
        Square((self.0 >> 6 & 63) as u8)
    }

    #[inline(always)]
    pub const fn flag(self) -> u16 {
        self.0 >> 12
    }

    #[inline(always)]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        self.flag() & 4 != 0
    }

    #[inline(always)]
    pub const fn is_ep(self) -> bool {
        self.flag() == Self::EP_CAPTURE
    }

    #[inline(always)]
    pub const fn is_castle(self) -> bool {
        matches!(self.flag(), Self::KING_CASTLE | Self::QUEEN_CASTLE)
    }

    #[inline(always)]
    pub const fn is_double_push(self) -> bool {
        self.flag() == Self::DOUBLE_PUSH
    }

    #[inline(always)]
    pub const fn is_promo(self) -> bool {
        self.flag() & 8 != 0
    }

    #[inline(always)]
    pub const fn promo_piece(self) -> Option<PieceType> {
        if self.is_promo() {
            Some(PieceType::from_index((self.flag() & 3) as usize + 1))
        } else {
            None
        }
    }

    /// Long algebraic UCI string, e.g. `e2e4`, `e7e8q`; null move is `0000`.
    pub fn to_uci(self) -> String {
        self.to_string()
    }
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_null() {
            return write!(f, "0000");
        }
        write!(f, "{}{}", self.from(), self.to())?;
        if let Some(p) = self.promo_piece() {
            write!(f, "{}", p.to_char())?;
        }
        Ok(())
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self}")
    }
}

/// Upper bound on legal moves in any chess position is 218.
pub const MAX_MOVES: usize = 256;

/// Fixed-capacity, stack-allocated move list (no heap allocation in movegen).
#[derive(Clone)]
pub struct MoveList {
    moves: [Move; MAX_MOVES],
    len: usize,
}

impl MoveList {
    #[inline(always)]
    pub const fn new() -> MoveList {
        MoveList {
            moves: [Move::NULL; MAX_MOVES],
            len: 0,
        }
    }

    #[inline(always)]
    pub fn push(&mut self, m: Move) {
        debug_assert!(self.len < MAX_MOVES);
        self.moves[self.len] = m;
        self.len += 1;
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        &self.moves[..self.len]
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl Deref for MoveList {
    type Target = [Move];
    #[inline(always)]
    fn deref(&self) -> &[Move] {
        self.as_slice()
    }
}

impl fmt::Debug for MoveList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.as_slice()).finish()
    }
}
