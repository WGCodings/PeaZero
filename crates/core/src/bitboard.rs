use std::fmt;
use std::ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not};

use crate::types::Square;

#[derive(Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    pub const EMPTY: Bitboard = Bitboard(0);
    pub const ALL: Bitboard = Bitboard(!0);
    pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
    pub const FILE_H: Bitboard = Bitboard(0x8080_8080_8080_8080);
    pub const RANK_1: Bitboard = Bitboard(0xFF);
    pub const RANK_2: Bitboard = Bitboard(0xFF << 8);
    pub const RANK_7: Bitboard = Bitboard(0xFF << 48);
    pub const RANK_8: Bitboard = Bitboard(0xFF << 56);
    pub const LIGHT_SQUARES: Bitboard = Bitboard(0x55AA_55AA_55AA_55AA);
    pub const DARK_SQUARES: Bitboard = Bitboard(0xAA55_AA55_AA55_AA55);

    #[inline(always)]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub const fn any(self) -> bool {
        self.0 != 0
    }

    #[inline(always)]
    pub const fn count(self) -> u32 {
        self.0.count_ones()
    }

    #[inline(always)]
    pub const fn more_than_one(self) -> bool {
        self.0 & self.0.wrapping_sub(1) != 0
    }

    #[inline(always)]
    pub const fn contains(self, sq: Square) -> bool {
        self.0 >> sq.0 & 1 != 0
    }

    /// Least significant square. Undefined (returns 64-ish) on empty boards.
    #[inline(always)]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.0 != 0);
        Square(self.0.trailing_zeros() as u8)
    }

    #[inline(always)]
    pub fn pop_lsb(&mut self) -> Square {
        let sq = self.lsb();
        self.0 &= self.0 - 1;
        sq
    }
}

impl Iterator for Bitboard {
    type Item = Square;
    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0 == 0 {
            None
        } else {
            Some(self.pop_lsb())
        }
    }
}

macro_rules! impl_op {
    ($tr:ident, $f:ident, $tra:ident, $fa:ident, $op:tt) => {
        impl $tr for Bitboard {
            type Output = Bitboard;
            #[inline(always)]
            fn $f(self, rhs: Bitboard) -> Bitboard { Bitboard(self.0 $op rhs.0) }
        }
        impl $tra for Bitboard {
            #[inline(always)]
            fn $fa(&mut self, rhs: Bitboard) { self.0 = self.0 $op rhs.0; }
        }
    };
}
impl_op!(BitAnd, bitand, BitAndAssign, bitand_assign, &);
impl_op!(BitOr, bitor, BitOrAssign, bitor_assign, |);
impl_op!(BitXor, bitxor, BitXorAssign, bitxor_assign, ^);

impl Not for Bitboard {
    type Output = Bitboard;
    #[inline(always)]
    fn not(self) -> Bitboard {
        Bitboard(!self.0)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f)?;
        for rank in (0..8).rev() {
            for file in 0..8 {
                let c = if self.contains(Square::new(file, rank)) {
                    'X'
                } else {
                    '.'
                };
                write!(f, "{c} ")?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}
