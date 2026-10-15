//! `peazero bench` — deterministic node count used by OpenBench as the build
//! signature, plus an NPS figure for worker speed scaling.
//!
//! Searchless engine: one "node" = one network evaluation. The bench
//! evaluates every bench position and every position one legal move later,
//! so the node count depends only on the positions, not on the network
//! weights (swapping networks on OpenBench doesn't change the bench).

use std::time::Instant;

use peazero_core::{Game, GameResult};

use crate::policy::PolicyNet;

pub const BENCH_FENS: &[&str] = &[
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "rnbqkb1r/pp1p1ppp/4pn2/2p5/2PP4/2N5/PP2PPPP/R1BQKBNR w KQkq - 0 4",
    "8/8/4k3/3p4/3P1K2/8/8/8 w - - 0 1",
];

pub fn run(net: &mut PolicyNet) -> u64 {
    let start = Instant::now();
    let mut nodes = 0u64;
    for (i, fen) in BENCH_FENS.iter().enumerate() {
        let root = Game::from_fen(fen).expect("bench FEN");
        let before = nodes;
        let eval = net.evaluate(&root);
        nodes += 1;
        for &(m, _) in &eval.moves {
            let mut child = root.clone();
            child.push(m);
            if child.result() == GameResult::Ongoing {
                net.evaluate(&child);
                nodes += 1;
            }
        }
        println!("position {:>2} nodes {:>5}  {fen}", i + 1, nodes - before);
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let nps = (nodes as f64 / elapsed) as u64;
    println!("{nodes} nodes {nps} nps");
    nodes
}
