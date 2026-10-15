//! Perft test-suite.
//!
//! * `standard_suite_fast`    — every position in tests/perft/standard.epd, all
//!   depths whose expected count is ≤ 5M nodes. Runs by default.
//! * `standard_suite_full`    — every position at every listed depth (billions
//!   of nodes). `cargo test --release -- --ignored`.
//! * `pseudo_vs_legal_hash`   — for every suite position, depth 3: legal gen ==
//!   filtered pseudolegal gen, no duplicates, incremental hash == recomputed.
//! * `tricky_positions`       — classic edge cases (ep pins, castling checks,
//!   promotions, stalemate/mate traps).

use peazero_core::perft::{perft, perft_checked};
use peazero_core::Position;

struct Entry {
    fen: String,
    depths: Vec<(u32, u64)>,
}

fn load_suite() -> Vec<Entry> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/perft/standard.epd"
    );
    let text = std::fs::read_to_string(path).expect("read standard.epd");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let mut parts = line.split(';');
            let fen = parts.next().unwrap().trim().to_string();
            let depths = parts
                .map(|p| {
                    let mut it = p.split_whitespace();
                    let d = it.next().unwrap().trim_start_matches('D').parse().unwrap();
                    let n = it.next().unwrap().parse().unwrap();
                    (d, n)
                })
                .collect();
            Entry { fen, depths }
        })
        .collect()
}

fn run_suite(max_nodes: u64) -> u64 {
    peazero_core::init();
    let suite = load_suite();
    assert!(suite.len() >= 120, "suite looks truncated");
    let mut total = 0;
    let mut failures = Vec::new();
    for e in &suite {
        let pos = Position::from_fen(&e.fen).unwrap_or_else(|err| panic!("{err}: {}", e.fen));
        for &(d, expected) in &e.depths {
            if expected > max_nodes {
                continue;
            }
            let got = perft(&pos, d);
            total += got;
            if got != expected {
                failures.push(format!("{} D{d}: expected {expected}, got {got}", e.fen));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "perft failures:\n{}",
        failures.join("\n")
    );
    total
}

#[test]
fn standard_suite_fast() {
    run_suite(5_000_000);
}

#[test]
#[ignore = "billions of nodes; run with --release -- --ignored"]
fn standard_suite_full() {
    let start = std::time::Instant::now();
    let nodes = run_suite(u64::MAX);
    let secs = start.elapsed().as_secs_f64();
    eprintln!(
        "full suite: {nodes} nodes in {secs:.1}s ({:.0} Mnps)",
        nodes as f64 / secs / 1e6
    );
}

#[test]
fn pseudo_vs_legal_hash() {
    peazero_core::init();
    for e in load_suite() {
        let pos = Position::from_fen(&e.fen).unwrap();
        let expected = e.depths.iter().find(|(d, _)| *d == 3).map(|x| x.1);
        let got = perft_checked(&pos, 3);
        if let Some(exp) = expected {
            assert_eq!(got, exp, "{}", e.fen);
        }
    }
}

#[test]
fn tricky_positions() {
    peazero_core::init();
    let cases: &[(&str, u32, u64)] = &[
        // illegal ep moves (horizontal / diagonal pins)
        ("3k4/3p4/8/K1P4r/8/8/8/8 b - - 0 1", 6, 1_134_888),
        ("8/8/4k3/8/2p5/8/B2P2K1/8 w - - 0 1", 6, 1_015_133),
        // ep capture checks opponent
        ("8/8/1k6/2b5/2pP4/8/5K2/8 b - d3 0 1", 6, 1_440_467),
        // castling gives check
        ("5k2/8/8/8/8/8/8/4K2R w K - 0 1", 6, 661_072),
        ("3k4/8/8/8/8/8/8/R3K3 w Q - 0 1", 6, 803_711),
        // castling rights / prevented
        ("r3k2r/1b4bq/8/8/8/8/7B/R3K2R w KQkq - 0 1", 4, 1_274_206),
        ("r3k2r/8/3Q4/8/8/5q2/8/R3K2R b KQkq - 0 1", 4, 1_720_476),
        // promote out of check / discovered check
        ("2K2r2/4P3/8/8/8/8/8/3k4 w - - 0 1", 6, 3_821_001),
        ("8/8/1P2K3/8/2n5/1q6/8/5k2 b - - 0 1", 5, 1_004_658),
        // (under)promote to give check
        ("4k3/1P6/8/8/8/8/K7/8 w - - 0 1", 6, 217_342),
        ("8/P1k5/K7/8/8/8/8/8 w - - 0 1", 6, 92_683),
        // self stalemate / stalemate & checkmate
        ("K1k5/8/P7/8/8/8/8/8 w - - 0 1", 6, 2_217),
        ("8/k1P5/8/1K6/8/8/8/8 w - - 0 1", 7, 567_584),
        ("8/8/2k5/5q2/5n2/8/5K2/8 b - - 0 1", 4, 23_527),
        // standard chessprogramming.org positions 3-6 at depth 5
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 5, 674_624),
        (
            "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
            4,
            422_333,
        ),
        (
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
            4,
            422_333,
        ),
        (
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            4,
            2_103_487,
        ),
        (
            "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
            4,
            3_894_594,
        ),
    ];
    for &(fen, depth, expected) in cases {
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(perft(&pos, depth), expected, "{fen} depth {depth}");
    }
}

#[test]
fn startpos_depth_6() {
    peazero_core::init();
    assert_eq!(perft(&Position::startpos(), 6), 119_060_324);
}
