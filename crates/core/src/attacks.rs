//! Attack tables.
//!
//! * Leapers (pawn, knight, king), `between` and `line` are computed at compile
//!   time as `const` tables.
//! * Sliders use "fancy" magic bitboards. Magics are searched at startup with a
//!   fixed seed (deterministic, a few ms) and stored in one shared table.

use std::sync::LazyLock;

use crate::bitboard::Bitboard;
use crate::rng::Rng;
use crate::types::{Color, Square};

const FILE_A: u64 = 0x0101_0101_0101_0101;
const FILE_B: u64 = FILE_A << 1;
const FILE_G: u64 = FILE_A << 6;
const FILE_H: u64 = FILE_A << 7;
const RANK_1: u64 = 0xFF;
const RANK_8: u64 = 0xFF << 56;

const ROOK_DIRS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];
const BISHOP_DIRS: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
const ALL_DIRS: [(i8, i8); 8] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
];

// ---------------------------------------------------------------- leapers

const fn knight_mask(sq: usize) -> u64 {
    let b = 1u64 << sq;
    let not_a = !FILE_A;
    let not_ab = !(FILE_A | FILE_B);
    let not_h = !FILE_H;
    let not_gh = !(FILE_G | FILE_H);
    ((b << 17) & not_a)
        | ((b << 15) & not_h)
        | ((b << 10) & not_ab)
        | ((b << 6) & not_gh)
        | ((b >> 17) & not_h)
        | ((b >> 15) & not_a)
        | ((b >> 10) & not_gh)
        | ((b >> 6) & not_ab)
}

const fn king_mask(sq: usize) -> u64 {
    let b = 1u64 << sq;
    let not_a = !FILE_A;
    let not_h = !FILE_H;
    (b << 8)
        | (b >> 8)
        | ((b << 1) & not_a)
        | ((b >> 1) & not_h)
        | ((b << 9) & not_a)
        | ((b << 7) & not_h)
        | ((b >> 7) & not_a)
        | ((b >> 9) & not_h)
}

const fn pawn_mask(color: usize, sq: usize) -> u64 {
    let b = 1u64 << sq;
    if color == 0 {
        ((b << 7) & !FILE_H) | ((b << 9) & !FILE_A)
    } else {
        ((b >> 9) & !FILE_H) | ((b >> 7) & !FILE_A)
    }
}

const KNIGHT: [u64; 64] = {
    let mut t = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = knight_mask(i);
        i += 1;
    }
    t
};

const KING: [u64; 64] = {
    let mut t = [0u64; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = king_mask(i);
        i += 1;
    }
    t
};

const PAWN: [[u64; 64]; 2] = {
    let mut t = [[0u64; 64]; 2];
    let mut i = 0;
    while i < 64 {
        t[0][i] = pawn_mask(0, i);
        t[1][i] = pawn_mask(1, i);
        i += 1;
    }
    t
};

// ------------------------------------------------------- ray utilities

/// Slow reference slider attacks: walk each direction until blocked.
const fn slide(sq: usize, occ: u64, dirs: &[(i8, i8)]) -> u64 {
    let mut att = 0u64;
    let mut d = 0;
    while d < dirs.len() {
        let (df, dr) = dirs[d];
        let mut f = (sq % 8) as i8 + df;
        let mut r = (sq / 8) as i8 + dr;
        while f >= 0 && f < 8 && r >= 0 && r < 8 {
            let s = 1u64 << (r * 8 + f);
            att |= s;
            if occ & s != 0 {
                break;
            }
            f += df;
            r += dr;
        }
        d += 1;
    }
    att
}

const fn ray(sq: usize, df: i8, dr: i8) -> u64 {
    slide(sq, 0, &[(df, dr)])
}

const BETWEEN_LINE: ([[u64; 64]; 64], [[u64; 64]; 64]) = {
    let mut between = [[0u64; 64]; 64];
    let mut line = [[0u64; 64]; 64];
    let mut a = 0;
    while a < 64 {
        let mut d = 0;
        while d < 8 {
            let (df, dr) = ALL_DIRS[d];
            let full = ray(a, df, dr) | ray(a, -df, -dr) | (1u64 << a);
            let mut acc = 0u64;
            let mut f = (a % 8) as i8 + df;
            let mut r = (a / 8) as i8 + dr;
            while f >= 0 && f < 8 && r >= 0 && r < 8 {
                let b = (r * 8 + f) as usize;
                between[a][b] = acc;
                line[a][b] = full;
                acc |= 1u64 << b;
                f += df;
                r += dr;
            }
            d += 1;
        }
        a += 1;
    }
    (between, line)
};

// ---------------------------------------------------------------- magics

struct Magic {
    mask: u64,
    magic: u64,
    shift: u32,
    offset: usize,
}

struct Sliders {
    rook: Vec<Magic>,
    bishop: Vec<Magic>,
    table: Box<[u64]>,
}

static SLIDERS: LazyLock<Sliders> = LazyLock::new(build_sliders);

fn find_magic(sq: usize, dirs: &[(i8, i8)], rng: &mut Rng, table: &mut Vec<u64>) -> Magic {
    let rank_bb = RANK_1 << (8 * (sq / 8));
    let file_bb = FILE_A << (sq % 8);
    let edges = ((RANK_1 | RANK_8) & !rank_bb) | ((FILE_A | FILE_H) & !file_bb);
    let mask = slide(sq, 0, dirs) & !edges;
    let bits = mask.count_ones();
    let n = 1usize << bits;
    let shift = 64 - bits;

    // Enumerate all occupancy subsets of the mask (Carry-Rippler).
    let mut occs = Vec::with_capacity(n);
    let mut atts = Vec::with_capacity(n);
    let mut sub = 0u64;
    loop {
        occs.push(sub);
        atts.push(slide(sq, sub, dirs));
        sub = sub.wrapping_sub(mask) & mask;
        if sub == 0 {
            break;
        }
    }

    let mut used = vec![0u64; n];
    let mut epoch = vec![0u32; n];
    let mut attempt = 0u32;
    loop {
        let magic = rng.sparse_u64();
        if (mask.wrapping_mul(magic) >> 56).count_ones() < 6 {
            continue;
        }
        attempt += 1;
        let mut ok = true;
        for i in 0..n {
            let idx = (occs[i].wrapping_mul(magic) >> shift) as usize;
            if epoch[idx] < attempt {
                epoch[idx] = attempt;
                used[idx] = atts[i];
            } else if used[idx] != atts[i] {
                ok = false;
                break;
            }
        }
        if ok {
            let offset = table.len();
            table.extend_from_slice(&used);
            return Magic {
                mask,
                magic,
                shift,
                offset,
            };
        }
    }
}

fn build_sliders() -> Sliders {
    let mut rng = Rng::new(0x9EA2_E500_C0FF_EE01);
    let mut table = Vec::with_capacity(107_648);
    let rook = (0..64)
        .map(|sq| find_magic(sq, &ROOK_DIRS, &mut rng, &mut table))
        .collect();
    let bishop = (0..64)
        .map(|sq| find_magic(sq, &BISHOP_DIRS, &mut rng, &mut table))
        .collect();
    Sliders {
        rook,
        bishop,
        table: table.into_boxed_slice(),
    }
}

/// Build the slider tables now (otherwise built on first use).
pub fn init() {
    LazyLock::force(&SLIDERS);
}

// ------------------------------------------------------------ public API

#[inline(always)]
pub fn pawn_attacks(color: Color, sq: Square) -> Bitboard {
    Bitboard(PAWN[color.idx()][sq.idx()])
}

#[inline(always)]
pub fn knight_attacks(sq: Square) -> Bitboard {
    Bitboard(KNIGHT[sq.idx()])
}

#[inline(always)]
pub fn king_attacks(sq: Square) -> Bitboard {
    Bitboard(KING[sq.idx()])
}

#[inline(always)]
pub fn rook_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let s = &*SLIDERS;
    let m = &s.rook[sq.idx()];
    Bitboard(s.table[m.offset + ((occ.0 & m.mask).wrapping_mul(m.magic) >> m.shift) as usize])
}

#[inline(always)]
pub fn bishop_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    let s = &*SLIDERS;
    let m = &s.bishop[sq.idx()];
    Bitboard(s.table[m.offset + ((occ.0 & m.mask).wrapping_mul(m.magic) >> m.shift) as usize])
}

#[inline(always)]
pub fn queen_attacks(sq: Square, occ: Bitboard) -> Bitboard {
    rook_attacks(sq, occ) | bishop_attacks(sq, occ)
}

/// Squares strictly between `a` and `b` if they share a line, else empty.
#[inline(always)]
pub fn between(a: Square, b: Square) -> Bitboard {
    Bitboard(BETWEEN_LINE.0[a.idx()][b.idx()])
}

/// Full board-edge-to-edge line through `a` and `b` (incl. both), else empty.
#[inline(always)]
pub fn line(a: Square, b: Square) -> Bitboard {
    Bitboard(BETWEEN_LINE.1[a.idx()][b.idx()])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn magics_match_reference() {
        let mut rng = Rng::new(42);
        for sq in 0..64u8 {
            for _ in 0..2000 {
                let occ = rng.next_u64() & rng.next_u64();
                assert_eq!(
                    rook_attacks(Square(sq), Bitboard(occ)).0,
                    slide(sq as usize, occ, &ROOK_DIRS)
                );
                assert_eq!(
                    bishop_attacks(Square(sq), Bitboard(occ)).0,
                    slide(sq as usize, occ, &BISHOP_DIRS)
                );
            }
        }
    }

    #[test]
    fn between_and_line() {
        let a1 = Square::A1;
        let h8 = Square::H8;
        assert_eq!(between(a1, h8).count(), 6);
        assert_eq!(line(a1, h8).count(), 8);
        assert!(between(a1, Square::B1).is_empty());
        assert!(between(a1, Square(10)).is_empty()); // a1-c2 not aligned
        assert!(line(a1, Square(10)).is_empty());
        assert_eq!(knight_attacks(Square::B1).count(), 3);
        assert_eq!(king_attacks(Square::E1).count(), 5);
    }
}
