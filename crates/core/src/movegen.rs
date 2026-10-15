//! Move generation.
//!
//! * `generate_legal`: fully legal moves, generated directly using check and
//!   pin masks (no make/unmake filtering). This is what search and UCI use.
//! * `generate_pseudo`: pseudolegal moves (may leave the king in check;
//!   castling still respects "not out of / through check"). Combined with
//!   `is_legal` it is the reference the legal generator is tested against.

use crate::attacks::{
    between, bishop_attacks, king_attacks, knight_attacks, line, pawn_attacks, queen_attacks,
    rook_attacks,
};
use crate::bitboard::Bitboard;
use crate::moves::{Move, MoveList};
use crate::position::{Position, BLACK_OO, BLACK_OOO, WHITE_OO, WHITE_OOO};
use crate::types::{Color, PieceType, Square};

const PROMOS: [PieceType; 4] = [
    PieceType::Queen,
    PieceType::Rook,
    PieceType::Bishop,
    PieceType::Knight,
];

#[inline(always)]
fn push_pawn_moves(list: &mut MoveList, from: Square, to: Square, capture: bool, promo_rank: bool) {
    if promo_rank {
        for p in PROMOS {
            list.push(Move::new_promo(from, to, p, capture));
        }
    } else {
        list.push(Move::new(
            from,
            to,
            if capture { Move::CAPTURE } else { Move::QUIET },
        ));
    }
}

#[inline(always)]
fn push_targets(list: &mut MoveList, from: Square, targets: Bitboard, enemy: Bitboard) {
    for to in targets {
        let flag = if enemy.contains(to) {
            Move::CAPTURE
        } else {
            Move::QUIET
        };
        list.push(Move::new(from, to, flag));
    }
}

impl Position {
    /// Own pieces pinned to our king.
    pub fn pinned(&self) -> Bitboard {
        let us = self.side_to_move();
        let ksq = self.king_sq(us);
        let them = self.color_bb(!us);
        let own = self.color_bb(us);
        let occ = self.occupied();
        let rq = (self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen)) & them;
        let bq = (self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen)) & them;
        let snipers =
            (rook_attacks(ksq, Bitboard::EMPTY) & rq) | (bishop_attacks(ksq, Bitboard::EMPTY) & bq);
        let mut pinned = Bitboard::EMPTY;
        for s in snipers {
            let b = between(ksq, s) & occ;
            if !b.more_than_one() && (b & own).any() {
                pinned |= b;
            }
        }
        pinned
    }

    /// Fill `list` with all legal moves (list is cleared first).
    pub fn generate_legal(&self, list: &mut MoveList) {
        list.clear();
        let us = self.side_to_move();
        let them = !us;
        let own = self.color_bb(us);
        let enemy = self.color_bb(them);
        let occ = own | enemy;
        let ksq = self.king_sq(us);

        // King moves: target must not be attacked once the king has left ksq
        // (so sliders "see through" the king's old square).
        let occ_no_king = occ ^ ksq.bb();
        for to in king_attacks(ksq) & !own {
            if !self.is_attacked(to, them, occ_no_king) {
                let flag = if enemy.contains(to) {
                    Move::CAPTURE
                } else {
                    Move::QUIET
                };
                list.push(Move::new(ksq, to, flag));
            }
        }

        let checkers = self.checkers();
        if checkers.more_than_one() {
            return; // double check: only king moves
        }
        let check_mask = if checkers.any() {
            between(ksq, checkers.lsb()) | checkers
        } else {
            Bitboard::ALL
        };
        let pinned = self.pinned();
        let targets_mask = !own & check_mask;

        // Castling (only when not in check).
        if checkers.is_empty() {
            self.gen_castling(list, us, occ, true);
        }

        // Knights (a pinned knight can never move).
        for from in self.pieces_of(us, PieceType::Knight) & !pinned {
            push_targets(list, from, knight_attacks(from) & targets_mask, enemy);
        }

        // Sliders.
        let bq = (self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen)) & own;
        let rq = (self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen)) & own;
        for from in bq {
            let mut t = bishop_attacks(from, occ) & targets_mask;
            if pinned.contains(from) {
                t &= line(ksq, from);
            }
            push_targets(list, from, t, enemy);
        }
        for from in rq {
            let mut t = rook_attacks(from, occ) & targets_mask;
            if pinned.contains(from) {
                t &= line(ksq, from);
            }
            push_targets(list, from, t, enemy);
        }

        // Pawns.
        let up: i8 = if us == Color::White { 8 } else { -8 };
        let start_rank = if us == Color::White {
            Bitboard::RANK_2
        } else {
            Bitboard::RANK_7
        };
        for from in self.pieces_of(us, PieceType::Pawn) {
            let pin_mask = if pinned.contains(from) {
                line(ksq, from)
            } else {
                Bitboard::ALL
            };
            let allowed = check_mask & pin_mask;
            let promo_rank = from.relative_rank(us) == 6;

            let one = Square((from.0 as i8 + up) as u8);
            if !occ.contains(one) {
                if allowed.contains(one) {
                    push_pawn_moves(list, from, one, false, promo_rank);
                }
                if start_rank.contains(from) {
                    let two = Square((one.0 as i8 + up) as u8);
                    if !occ.contains(two) && allowed.contains(two) {
                        list.push(Move::new(from, two, Move::DOUBLE_PUSH));
                    }
                }
            }

            let att = pawn_attacks(us, from);
            for to in att & enemy & allowed {
                push_pawn_moves(list, from, to, true, promo_rank);
            }

            if let Some(ep) = self.ep_square() {
                if att.contains(ep) {
                    // Rare: verify by simulating the capture (covers checks,
                    // pins, and the horizontal "two pawns vanish" case).
                    let cap = Square(ep.0 ^ 8);
                    let occ2 = (occ ^ from.bb() ^ cap.bb()) | ep.bb();
                    let enemy2 = enemy ^ cap.bb();
                    let rq_t =
                        (self.pieces(PieceType::Rook) | self.pieces(PieceType::Queen)) & enemy2;
                    let bq_t =
                        (self.pieces(PieceType::Bishop) | self.pieces(PieceType::Queen)) & enemy2;
                    let safe = (rook_attacks(ksq, occ2) & rq_t).is_empty()
                        && (bishop_attacks(ksq, occ2) & bq_t).is_empty()
                        && (knight_attacks(ksq) & self.pieces(PieceType::Knight) & enemy2)
                            .is_empty()
                        && (pawn_attacks(us, ksq) & self.pieces(PieceType::Pawn) & enemy2)
                            .is_empty();
                    if safe {
                        list.push(Move::new(from, ep, Move::EP_CAPTURE));
                    }
                }
            }
        }
    }

    fn gen_castling(&self, list: &mut MoveList, us: Color, occ: Bitboard, check_path: bool) {
        let rights = self.castling_rights();
        let them = !us;
        let (oo, ooo, k) = match us {
            Color::White => (WHITE_OO, WHITE_OOO, Square::E1),
            Color::Black => (BLACK_OO, BLACK_OOO, Square::E8),
        };
        if rights & oo != 0 {
            let f = Square(k.0 + 1);
            let g = Square(k.0 + 2);
            if !occ.contains(f)
                && !occ.contains(g)
                && (!check_path
                    || (!self.is_attacked(k, them, occ)
                        && !self.is_attacked(f, them, occ)
                        && !self.is_attacked(g, them, occ)))
            {
                list.push(Move::new(k, g, Move::KING_CASTLE));
            }
        }
        if rights & ooo != 0 {
            let d = Square(k.0 - 1);
            let c = Square(k.0 - 2);
            let b = Square(k.0 - 3);
            if !occ.contains(d)
                && !occ.contains(c)
                && !occ.contains(b)
                && (!check_path
                    || (!self.is_attacked(k, them, occ)
                        && !self.is_attacked(d, them, occ)
                        && !self.is_attacked(c, them, occ)))
            {
                list.push(Move::new(k, c, Move::QUEEN_CASTLE));
            }
        }
    }

    /// Pseudolegal moves: everything a piece can do ignoring pins/checks.
    /// Castling through or out of check is still excluded (it can't be fixed
    /// by a post-hoc "king in check" test).
    pub fn generate_pseudo(&self, list: &mut MoveList) {
        list.clear();
        let us = self.side_to_move();
        let own = self.color_bb(us);
        let enemy = self.color_bb(!us);
        let occ = own | enemy;

        for from in self.pieces_of(us, PieceType::King) {
            push_targets(list, from, king_attacks(from) & !own, enemy);
        }
        self.gen_castling(list, us, occ, true);
        for from in self.pieces_of(us, PieceType::Knight) {
            push_targets(list, from, knight_attacks(from) & !own, enemy);
        }
        for from in self.pieces_of(us, PieceType::Bishop) {
            push_targets(list, from, bishop_attacks(from, occ) & !own, enemy);
        }
        for from in self.pieces_of(us, PieceType::Rook) {
            push_targets(list, from, rook_attacks(from, occ) & !own, enemy);
        }
        for from in self.pieces_of(us, PieceType::Queen) {
            push_targets(list, from, queen_attacks(from, occ) & !own, enemy);
        }

        let up: i8 = if us == Color::White { 8 } else { -8 };
        let start_rank = if us == Color::White {
            Bitboard::RANK_2
        } else {
            Bitboard::RANK_7
        };
        for from in self.pieces_of(us, PieceType::Pawn) {
            let promo_rank = from.relative_rank(us) == 6;
            let one = Square((from.0 as i8 + up) as u8);
            if !occ.contains(one) {
                push_pawn_moves(list, from, one, false, promo_rank);
                let two = Square((one.0 as i8 + up) as u8);
                if start_rank.contains(from) && !occ.contains(two) {
                    list.push(Move::new(from, two, Move::DOUBLE_PUSH));
                }
            }
            let att = pawn_attacks(us, from);
            for to in att & enemy {
                push_pawn_moves(list, from, to, true, promo_rank);
            }
            if let Some(ep) = self.ep_square() {
                if att.contains(ep) {
                    list.push(Move::new(from, ep, Move::EP_CAPTURE));
                }
            }
        }
    }

    /// Is a pseudolegal move legal (doesn't leave own king in check)?
    pub fn is_legal(&self, m: Move) -> bool {
        let us = self.side_to_move();
        let mut child = *self;
        child.make_move(m);
        !child.is_attacked(child.king_sq(us), !us, child.occupied())
    }

    /// Parse a UCI move string against the legal moves of this position.
    pub fn parse_uci_move(&self, s: &str) -> Option<Move> {
        let mut list = MoveList::new();
        self.generate_legal(&mut list);
        list.iter().copied().find(|m| m.to_uci() == s)
    }

    /// Convenience: legal moves as a fresh list.
    pub fn legal_moves(&self) -> MoveList {
        let mut list = MoveList::new();
        self.generate_legal(&mut list);
        list
    }
}
