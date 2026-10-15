use std::fmt;

use crate::bitboard::Bitboard;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    pub const BOTH: [Color; 2] = [Color::White, Color::Black];

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn flip(self) -> Color {
        match self {
            Color::White => Color::Black,
            Color::Black => Color::White,
        }
    }
}

impl std::ops::Not for Color {
    type Output = Color;
    #[inline(always)]
    fn not(self) -> Color {
        self.flip()
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
}

impl PieceType {
    pub const ALL: [PieceType; 6] = [
        PieceType::Pawn,
        PieceType::Knight,
        PieceType::Bishop,
        PieceType::Rook,
        PieceType::Queen,
        PieceType::King,
    ];

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self as usize
    }

    #[inline(always)]
    pub const fn from_index(i: usize) -> PieceType {
        Self::ALL[i]
    }

    /// Lowercase letter (p, n, b, r, q, k).
    pub const fn to_char(self) -> char {
        match self {
            PieceType::Pawn => 'p',
            PieceType::Knight => 'n',
            PieceType::Bishop => 'b',
            PieceType::Rook => 'r',
            PieceType::Queen => 'q',
            PieceType::King => 'k',
        }
    }
}

/// Coloured piece packed as `color * 6 + kind`; `Piece::NONE` (12) is an empty square.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub struct Piece(u8);

impl Piece {
    pub const NONE: Piece = Piece(12);

    #[inline(always)]
    pub const fn new(color: Color, kind: PieceType) -> Piece {
        Piece(color as u8 * 6 + kind as u8)
    }

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        self.0 == 12
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn color(self) -> Color {
        if self.0 < 6 {
            Color::White
        } else {
            Color::Black
        }
    }

    #[inline(always)]
    pub const fn kind(self) -> PieceType {
        PieceType::from_index((self.0 % 6) as usize)
    }

    /// FEN letter: uppercase for white, lowercase for black, '.' for empty.
    pub fn to_char(self) -> char {
        if self.is_none() {
            return '.';
        }
        let c = self.kind().to_char();
        match self.color() {
            Color::White => c.to_ascii_uppercase(),
            Color::Black => c,
        }
    }

    pub fn from_char(c: char) -> Option<Piece> {
        let color = if c.is_ascii_uppercase() {
            Color::White
        } else {
            Color::Black
        };
        let kind = match c.to_ascii_lowercase() {
            'p' => PieceType::Pawn,
            'n' => PieceType::Knight,
            'b' => PieceType::Bishop,
            'r' => PieceType::Rook,
            'q' => PieceType::Queen,
            'k' => PieceType::King,
            _ => return None,
        };
        Some(Piece::new(color, kind))
    }
}

/// Square index 0..64, a1 = 0, h1 = 7, a8 = 56, h8 = 63.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, PartialOrd, Ord)]
pub struct Square(pub u8);

impl Square {
    pub const A1: Square = Square(0);
    pub const B1: Square = Square(1);
    pub const C1: Square = Square(2);
    pub const D1: Square = Square(3);
    pub const E1: Square = Square(4);
    pub const F1: Square = Square(5);
    pub const G1: Square = Square(6);
    pub const H1: Square = Square(7);
    pub const A8: Square = Square(56);
    pub const B8: Square = Square(57);
    pub const C8: Square = Square(58);
    pub const D8: Square = Square(59);
    pub const E8: Square = Square(60);
    pub const F8: Square = Square(61);
    pub const G8: Square = Square(62);
    pub const H8: Square = Square(63);

    #[inline(always)]
    pub const fn new(file: u8, rank: u8) -> Square {
        Square(rank * 8 + file)
    }

    #[inline(always)]
    pub const fn idx(self) -> usize {
        self.0 as usize
    }

    #[inline(always)]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    #[inline(always)]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    #[inline(always)]
    pub const fn bb(self) -> Bitboard {
        Bitboard(1u64 << self.0)
    }

    /// Same square seen from the other side (rank mirrored).
    #[inline(always)]
    pub const fn flip_rank(self) -> Square {
        Square(self.0 ^ 56)
    }

    /// Rank from `color`'s point of view (0 = own back rank).
    #[inline(always)]
    pub const fn relative_rank(self, color: Color) -> u8 {
        match color {
            Color::White => self.rank(),
            Color::Black => 7 - self.rank(),
        }
    }

    pub fn parse(s: &str) -> Option<Square> {
        let b = s.as_bytes();
        if b.len() != 2 {
            return None;
        }
        let f = b[0].wrapping_sub(b'a');
        let r = b[1].wrapping_sub(b'1');
        if f < 8 && r < 8 {
            Some(Square::new(f, r))
        } else {
            None
        }
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}{}",
            (b'a' + self.file()) as char,
            (b'1' + self.rank()) as char
        )
    }
}
