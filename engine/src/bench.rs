//! `peazero bench` — deterministic node count used by OpenBench as the build
//! signature, plus an NPS figure for worker speed scaling.
//!
//! Sprint 1 benches movegen via perft. Once MCTS exists this switches to a
//! fixed-simulation search over the same positions (the bench number will
//! change then — that's expected).

use std::time::Instant;

use peazero_core::perft::perft;
use peazero_core::Position;

const BENCH_POSITIONS: &[(&str, u32)] = &[
    (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        5,
    ),
    (
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        4,
    ),
    ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5),
    (
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
        4,
    ),
    (
        "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
        4,
    ),
    (
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        4,
    ),
    (
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
        4,
    ),
    (
        "rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
        4,
    ),
    ("8/8/4k3/3p4/3P1K2/8/8/8 w - - 0 1", 6),
];

/// `depth_override` replaces every position's depth (handy for quick runs).
pub fn run(depth_override: Option<u32>) -> u64 {
    let start = Instant::now();
    let mut total = 0u64;
    for (i, &(fen, depth)) in BENCH_POSITIONS.iter().enumerate() {
        let pos = Position::from_fen(fen).expect("bench FEN");
        let d = depth_override.unwrap_or(depth);
        let nodes = perft(&pos, d);
        total += nodes;
        println!("position {:>2} depth {d} nodes {nodes:>10}  {fen}", i + 1);
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let nps = (total as f64 / elapsed) as u64;
    println!("{total} nodes {nps} nps");
    total
}
