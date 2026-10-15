use crate::moves::{Move, MoveList};
use crate::position::Position;

/// Count leaf nodes at `depth` using the legal generator (bulk counting at
/// depth 1).
pub fn perft(pos: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut list = MoveList::new();
    pos.generate_legal(&mut list);
    if depth == 1 {
        return list.len() as u64;
    }
    let mut nodes = 0;
    for &m in list.iter() {
        let mut child = *pos;
        child.make_move(m);
        nodes += perft(&child, depth - 1);
    }
    nodes
}

/// Per-root-move breakdown, sorted by UCI string (matches Stockfish's
/// `go perft` order closely enough for diffing).
pub fn divide(pos: &Position, depth: u32) -> Vec<(Move, u64)> {
    let mut list = MoveList::new();
    pos.generate_legal(&mut list);
    let mut out: Vec<(Move, u64)> = list
        .iter()
        .map(|&m| {
            let mut child = *pos;
            child.make_move(m);
            (
                m,
                if depth <= 1 {
                    1
                } else {
                    perft(&child, depth - 1)
                },
            )
        })
        .collect();
    out.sort_by_key(|(m, _)| m.to_uci());
    out
}

/// Perft using pseudolegal generation + `is_legal` filtering, with full
/// hash verification at every node. Slow; only for tests.
pub fn perft_checked(pos: &Position, depth: u32) -> u64 {
    assert_eq!(
        pos.hash(),
        pos.compute_hash(),
        "incremental hash mismatch at {}",
        pos.to_fen()
    );
    if depth == 0 {
        return 1;
    }
    let mut legal = MoveList::new();
    let mut pseudo = MoveList::new();
    pos.generate_legal(&mut legal);
    pos.generate_pseudo(&mut pseudo);

    let mut filtered: Vec<u16> = pseudo
        .iter()
        .filter(|&&m| pos.is_legal(m))
        .map(|m| m.raw())
        .collect();
    let mut direct: Vec<u16> = legal.iter().map(|m| m.raw()).collect();
    filtered.sort_unstable();
    direct.sort_unstable();
    assert_eq!(
        direct,
        filtered,
        "legal != filtered pseudo at {}",
        pos.to_fen()
    );
    let before = direct.len();
    direct.dedup();
    assert_eq!(before, direct.len(), "duplicate moves at {}", pos.to_fen());

    let mut nodes = 0;
    for &m in legal.iter() {
        let mut child = *pos;
        child.make_move(m);
        nodes += perft_checked(&child, depth - 1);
    }
    nodes
}
