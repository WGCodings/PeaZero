//! `peazero bench` — deterministic node count used by OpenBench as the build
//! signature, plus an NPS figure for worker speed scaling.
//!
//! Runs exactly what `go depth 3` does with the MCTS search (the default
//! mode) on every bench position, silently, and prints the total node count
//! (playouts) and the overall speed (total nodes / total time).
//!
//! The node count is deterministic (no randomness in the search,
//! single-threaded tract inference) but depends on the network: MCTS depth
//! is the average playout depth, and how fast that grows depends on the
//! policy and values. A new embedded network gives a new bench number.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use peazero_core::Game;

use crate::engine::{CommonOptions, SearchLimits};
use crate::policy::PolicyNet;
use crate::search::{mcts, SearchContext};

pub const BENCH_DEPTH: u32 = 10;
/// Per-position playout cap: a flat (e.g. untrained) policy deepens the
/// tree very slowly, so without it the bench could run for minutes.
pub const BENCH_MAX_NODES: u64 = 20_000;

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
    let limits = SearchLimits {
        depth: Some(BENCH_DEPTH),
        nodes: Some(BENCH_MAX_NODES),
        ..Default::default()
    };
    let common = CommonOptions::default();
    let stop = AtomicBool::new(false);
    let ctx = SearchContext {
        limits: &limits,
        common: &common,
        stop: &stop,
        show_wdl: false,
        verbose_move_stats: false,
    };

    let start = Instant::now();
    let mut total = 0u64;
    for (i, fen) in BENCH_FENS.iter().enumerate() {
        let game = Game::from_fen(fen).expect("bench FEN");
        let t = Instant::now();
        let (_, nodes) = mcts::run(net, &game, &ctx, false);
        let ms = t.elapsed().as_millis();
        total += nodes;
        println!(
            "position {:>2} nodes {nodes:>6} nps {:>6} time {ms:>6} ms  {fen}",
            i + 1,
            nodes as u128 * 1000 / ms.max(1)
        );
    }
    let elapsed = start.elapsed().as_secs_f64().max(1e-6);
    let nps = (total as f64 / elapsed) as u64;
    println!("{total} nodes {nps} nps");
    total
}
