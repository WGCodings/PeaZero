//! Transposition table for network evaluations; its size is a share of
//! the UCI `Hash` option (the tree gets the rest, see mcts.rs).
//!
//! The network sees only the current position, its repetition count and
//! the 50-move counter (no move history), so a position reached by another
//! move order has exactly the same network input. The table maps
//! (Zobrist hash, repetition count, 50-move counter) to the network's
//! output for that position: W/D/L and the logits of all legal moves, in
//! move-generation order. A probe costs well under a microsecond; an
//! evaluation ~260 µs.
//!
//! Compact entries (80 bytes): W and L as u16, each logit as one byte —
//! its distance below the best logit in steps of 1/16, clamped at 255/16 ≈
//! 16 (a move that far below the best has a prior under 0.01 % after the
//! temperature). That is at most 1/32 off per logit, ~2 % on a prior.
//! Every move is kept: with the tuned temperatures the moves outside the
//! top 16 still hold ~4–8 % of the prior mass on average, and storing the
//! top moves with their indices would cost as much as storing all of them
//! at one byte each.
//!
//! The search always uses the quantised values, on a miss too, so it makes
//! the same decisions whether or not a position was in the table (the
//! table size can't change the search, only its speed). Positions with
//! more than `MAX_MOVES` legal moves (none in 85k searched positions; the
//! maximum seen was 62) use the network's values directly and are not
//! stored.
//!
//! Fixed-size array, one entry per slot, always replace. The table
//! survives between moves (it helps when tree reuse misses) and is cleared
//! on `ucinewgame` and when the network changes.

use peazero_core::encoding::legal_actions;
use peazero_core::{Game, Move};

use crate::policy::{Evaluation, PolicyNet};

/// Legal moves an entry can hold.
pub const MAX_MOVES: usize = 64;
/// Logit quantisation steps per unit.
const STEPS: f32 = 16.0;

#[derive(Clone, Copy)]
struct Entry {
    /// 0 = empty.
    key: u64,
    /// W and L in units of 1/65535.
    w: u16,
    l: u16,
    count: u8,
    /// (best logit − logit) · 16, clamped to 255.
    logits: [u8; MAX_MOVES],
}

const _: () = assert!(std::mem::size_of::<Entry>() == 80);

const EMPTY: Entry = Entry {
    key: 0,
    w: 0,
    l: 0,
    count: 0,
    logits: [0; MAX_MOVES],
};

impl Entry {
    fn pack(key: u64, eval: &Evaluation) -> Entry {
        let best = eval
            .moves
            .iter()
            .map(|m| m.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let mut e = Entry {
            key,
            w: (eval.wdl[0].clamp(0.0, 1.0) * 65535.0).round() as u16,
            l: (eval.wdl[2].clamp(0.0, 1.0) * 65535.0).round() as u16,
            count: eval.moves.len() as u8,
            logits: [0; MAX_MOVES],
        };
        for (q, m) in e.logits.iter_mut().zip(&eval.moves) {
            *q = ((best - m.1) * STEPS).round().min(255.0) as u8;
        }
        e
    }

    /// The evaluation, with the moves in `legal` (move-generation order).
    fn unpack(&self, legal: impl Iterator<Item = Move>) -> Evaluation {
        let w = self.w as f32 / 65535.0;
        let l = self.l as f32 / 65535.0;
        let moves = legal
            .zip(&self.logits)
            .map(|(m, &q)| (m, -(q as f32) / STEPS))
            .collect();
        Evaluation {
            moves,
            wdl: [w, (1.0 - w - l).max(0.0), l],
        }
    }
}

pub struct EvalTable {
    entries: Vec<Entry>,
    /// Occupied slots (for `hashfull`).
    used: usize,
    pub probes: u64,
    pub hits: u64,
}

fn mix(mut x: u64) -> u64 {
    // splitmix64 finaliser
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

impl EvalTable {
    /// A table of about `bytes` bytes.
    pub fn new(bytes: usize) -> EvalTable {
        let n = (bytes / std::mem::size_of::<Entry>()).max(1);
        EvalTable {
            entries: vec![EMPTY; n],
            used: 0,
            probes: 0,
            hits: 0,
        }
    }

    pub fn clear(&mut self) {
        self.entries.fill(EMPTY);
        self.used = 0;
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Occupied slots in permille.
    pub fn hashfull(&self) -> usize {
        self.used * 1000 / self.entries.len()
    }

    /// Everything the network input depends on: the position (Zobrist
    /// hash), its repetition count and the 50-move counter. Never 0.
    pub fn key(game: &Game) -> u64 {
        let pos = game.position();
        let extra = (pos.halfmove_clock() as u64) | ((game.repetitions().min(3) as u64) << 16);
        (pos.hash() ^ mix(extra.wrapping_add(0x9e37_79b9_7f4a_7c15))) | 1
    }

    fn slot(&self, key: u64) -> usize {
        ((key as u128 * self.entries.len() as u128) >> 64) as usize
    }

    /// The evaluation of `game`'s position: from the table, or from the
    /// network (then stored). Always the quantised values (see the module
    /// documentation), except for positions too large for an entry.
    pub fn evaluate(&mut self, net: &mut PolicyNet, game: &Game) -> Evaluation {
        let key = EvalTable::key(game);
        let i = self.slot(key);
        self.probes += 1;
        let e = &self.entries[i];
        if e.key == key {
            let legal = legal_actions(game.position());
            // a different move count: a key collision, not this position
            if legal.len() == e.count as usize {
                self.hits += 1;
                return e.unpack(legal.into_iter().map(|(_, m)| m));
            }
        }
        let eval = net.evaluate(game);
        if eval.moves.len() > MAX_MOVES {
            return eval;
        }
        let entry = Entry::pack(key, &eval);
        if self.entries[i].key == 0 {
            self.used += 1;
        }
        self.entries[i] = entry;
        entry.unpack(eval.moves.iter().map(|m| m.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transposition_hits_return_the_same_values_as_the_miss() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let mut tt = EvalTable::new(1 << 20);
        // two move orders reaching the same position
        let mut a = Game::default();
        for m in ["g1f3", "g8f6", "b1c3"] {
            assert!(a.push_uci(m));
        }
        let mut b = Game::default();
        for m in ["b1c3", "g8f6", "g1f3"] {
            assert!(b.push_uci(m));
        }
        assert_eq!(EvalTable::key(&a), EvalTable::key(&b));
        let miss = tt.evaluate(&mut net, &a);
        let hit = tt.evaluate(&mut net, &b);
        assert_eq!((tt.probes, tt.hits), (2, 1));
        assert_eq!(hit.wdl, miss.wdl);
        assert_eq!(hit.moves, miss.moves);
        assert!(tt.hashfull() <= 1000);
    }

    #[test]
    fn quantisation_is_close_to_the_network() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let mut tt = EvalTable::new(1 << 20);
        let g =
            Game::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        let exact = net.evaluate(&g);
        let q = tt.evaluate(&mut net, &g);
        let best = exact
            .moves
            .iter()
            .map(|m| m.1)
            .fold(f32::NEG_INFINITY, f32::max);
        for (e, q) in exact.moves.iter().zip(&q.moves) {
            assert_eq!(e.0, q.0);
            let rel = e.1 - best; // ≤ 0
            if rel > -15.0 {
                assert!(
                    (q.1 - rel).abs() <= 0.5 / STEPS + 1e-4,
                    "{} vs {}",
                    q.1,
                    rel
                );
            }
        }
        for k in 0..3 {
            assert!((q.wdl[k] - exact.wdl[k]).abs() < 1e-4);
        }
    }

    #[test]
    fn key_includes_the_counters_the_network_sees() {
        peazero_core::init();
        // same pieces, different 50-move counter
        let a = Game::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let b = Game::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 7 1").unwrap();
        assert_ne!(EvalTable::key(&a), EvalTable::key(&b));
        // a repetition: same position, repetition count 1
        let mut g = Game::default();
        let k0 = EvalTable::key(&g);
        for m in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            assert!(g.push_uci(m));
        }
        assert_ne!(EvalTable::key(&g), k0);
    }

    #[test]
    fn sized_in_bytes() {
        let tt = EvalTable::new(4 << 20);
        assert_eq!(tt.len(), (4 << 20) / 80);
        assert_eq!(tt.hashfull(), 0);
    }
}
