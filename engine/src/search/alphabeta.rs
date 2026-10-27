//! Plain alpha-beta on top of the network: iterative deepening, negamax
//! with alpha-beta pruning, moves ordered by the policy, the value head
//! scores the leaves. Checkmate, stalemate and draws (repetition, 50-move
//! rule, insufficient material) are detected exactly at every node, so
//! mates are found regardless of how good the network is.
//!
//! Nothing else on purpose: no transposition table, no evaluation cache, no
//! extensions, no quiescence, no pruning beyond alpha-beta. Every node costs
//! one network evaluation.
//!
//! Scores are centipawns from the side to move (`q_to_cp` of `W - L`,
//! clamped to ±3000); mate scores are `±(MATE - ply)`. The `wdl` in the info
//! line is the network's W/D/L at the last position of the PV (exact for a
//! mate or draw), seen from the side to move at the root.

use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use peazero_core::{Game, GameResult, Move};

use super::{NetSearch, SearchContext};
use crate::engine::SearchResult;
use crate::policy::{q_to_cp, wdl_string, PolicyNet};
use crate::timeman;

pub const MATE: i32 = 32_000;
/// `|score| >= MATE_BOUND` means a forced mate.
pub const MATE_BOUND: i32 = MATE - 1_000;
const INF: i32 = MATE + 1;
/// Network scores are clamped so they never look like mates.
const MAX_EVAL: i32 = 3_000;
const MAX_PLY: usize = 64;
/// Depth used when `go` has no time, depth or node limit at all.
const DEFAULT_DEPTH: i32 = 3;

pub struct AlphaBeta;

/// UCI score string: `cp X` or `mate N` (negative = getting mated).
pub fn score_string(score: i32) -> String {
    if score >= MATE_BOUND {
        format!("mate {}", (MATE - score + 1) / 2)
    } else if score <= -MATE_BOUND {
        format!("mate -{}", (MATE + score) / 2)
    } else {
        format!("cp {score}")
    }
}

struct Worker<'a> {
    net: &'a mut PolicyNet,
    ctx: &'a SearchContext<'a>,
    start: Instant,
    hard: Option<Duration>,
    nodes: u64,
    seldepth: usize,
    aborted: bool,
    pv: Vec<[Move; MAX_PLY]>,
    pv_len: [usize; MAX_PLY],
}

impl Worker<'_> {
    fn should_stop(&self) -> bool {
        self.ctx.stop.load(Ordering::Relaxed)
            || self.ctx.limits.nodes.is_some_and(|n| self.nodes >= n)
            || self.hard.is_some_and(|h| self.start.elapsed() >= h)
    }

    fn negamax(&mut self, game: &Game, depth: i32, mut alpha: i32, beta: i32, ply: usize) -> i32 {
        self.pv_len[ply] = ply;
        if self.should_stop() {
            self.aborted = true;
            return 0;
        }
        self.nodes += 1;
        self.seldepth = self.seldepth.max(ply);
        let pos = game.position();

        // draws (a single repetition inside the search counts as a draw)
        if ply > 0
            && (pos.halfmove_clock() >= 100
                || pos.insufficient_material()
                || game.repetitions() >= 1)
        {
            return 0;
        }
        if pos.legal_moves().is_empty() {
            return if pos.in_check() {
                -MATE + ply as i32
            } else {
                0
            };
        }

        let eval = self.net.evaluate(game);
        if depth <= 0 || ply >= MAX_PLY - 1 {
            let [w, _, l] = eval.wdl;
            return q_to_cp(w - l).clamp(-MAX_EVAL, MAX_EVAL);
        }

        // move ordering: policy logits, best first
        let mut moves = eval.moves;
        moves.sort_by(|a, b| b.1.total_cmp(&a.1));
        if ply == 0 && !self.ctx.limits.searchmoves.is_empty() {
            let allowed = &self.ctx.limits.searchmoves;
            moves.retain(|(m, _)| allowed.contains(&m.to_uci()));
        }

        let mut best = -INF;
        for (m, _) in moves {
            let mut child = game.clone();
            child.push(m);
            let score = -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1);
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
                if score > alpha {
                    alpha = score;
                    self.pv[ply][ply] = m;
                    let child_len = self.pv_len[ply + 1];
                    for i in ply + 1..child_len {
                        self.pv[ply][i] = self.pv[ply + 1][i];
                    }
                    self.pv_len[ply] = child_len.max(ply + 1);
                    if score >= beta {
                        break;
                    }
                }
            }
        }
        best
    }

    /// W/D/L at the end of `pv`, from the root side to move's point of view.
    fn pv_leaf_wdl(&mut self, game: &Game, pv: &[Move]) -> [f32; 3] {
        let mut leaf = game.clone();
        for &m in pv {
            leaf.push(m);
        }
        let wdl = match leaf.result() {
            GameResult::Checkmate(_) => [0.0, 0.0, 1.0], // side to move is mated
            GameResult::Draw(_) => [0.0, 1.0, 0.0],
            GameResult::Ongoing if !pv.is_empty() && leaf.repetitions() >= 1 => [0.0, 1.0, 0.0],
            GameResult::Ongoing => self.net.evaluate(&leaf).wdl,
        };
        if pv.len() % 2 == 1 {
            [wdl[2], wdl[1], wdl[0]] // leaf side to move is the opponent
        } else {
            wdl
        }
    }
}

impl NetSearch for AlphaBeta {
    fn search(&mut self, net: &mut PolicyNet, game: &Game, ctx: &SearchContext) -> SearchResult {
        run(net, game, ctx, true).0
    }
}

/// Iterative deepening. Returns the result and the number of nodes
/// searched; `print = false` suppresses the info lines (used by `bench`).
pub fn run(
    net: &mut PolicyNet,
    game: &Game,
    ctx: &SearchContext,
    print: bool,
) -> (SearchResult, u64) {
    let start = Instant::now();
    let limits = ctx.limits;
    let budget = timeman::budget(
        limits,
        game.position().side_to_move(),
        ctx.common.move_overhead_ms,
    );
    let unlimited = budget.is_none() && limits.nodes.is_none() && !limits.infinite;
    let max_depth = limits
        .depth
        .map(|d| d as i32)
        .or(limits.mate.map(|m| 2 * m as i32 - 1))
        .unwrap_or(if unlimited {
            DEFAULT_DEPTH
        } else {
            MAX_PLY as i32 - 1
        })
        .clamp(1, MAX_PLY as i32 - 1);

    let mut w = Worker {
        net,
        ctx,
        start,
        hard: budget.map(|b| b.hard),
        nodes: 0,
        seldepth: 0,
        aborted: false,
        pv: vec![[Move::NULL; MAX_PLY]; MAX_PLY],
        pv_len: [0; MAX_PLY],
    };

    let mut best = Move::NULL;
    let mut ponder = None;
    for depth in 1..=max_depth {
        w.seldepth = 0;
        let score = w.negamax(game, depth, -INF, INF, 0);
        if w.aborted {
            break;
        }
        let pv: Vec<Move> = w.pv[0][..w.pv_len[0]].to_vec();
        let Some(&first) = pv.first() else {
            break; // no legal moves
        };
        best = first;
        ponder = pv.get(1).copied();
        if print {
            print_info(&mut w, game, depth, score, &pv);
        }
        if score.abs() >= MATE_BOUND {
            break; // forced mate found (for either side)
        }
        if let Some(b) = budget {
            // the next iteration takes several times longer
            if start.elapsed() >= b.soft / 2 {
                break;
            }
        }
    }

    if best.is_null() {
        // depth 1 didn't finish in time: play the policy's choice
        best = w
            .net
            .evaluate(game)
            .moves
            .iter()
            .filter(|(m, _)| {
                limits.searchmoves.is_empty() || limits.searchmoves.contains(&m.to_uci())
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|&(m, _)| m)
            .unwrap_or(Move::NULL);
        ponder = None;
    }
    (SearchResult { best, ponder }, w.nodes)
}

fn print_info(w: &mut Worker, game: &Game, depth: i32, score: i32, pv: &[Move]) {
    let wdl = if w.ctx.show_wdl {
        wdl_string(w.pv_leaf_wdl(game, pv))
    } else {
        String::new()
    };
    let ms = w.start.elapsed().as_millis();
    let pv_str: Vec<String> = pv.iter().map(|m| m.to_uci()).collect();
    println!(
        "info depth {depth} seldepth {} multipv 1 score {}{wdl} nodes {} nps {} time {ms} pv {}",
        w.seldepth,
        score_string(score),
        w.nodes,
        w.nodes as u128 * 1000 / ms.max(1),
        pv_str.join(" ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CommonOptions, SearchLimits};
    use std::sync::atomic::AtomicBool;

    fn run(fen: &str, depth: u32) -> (Move, Game) {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let game = Game::from_fen(fen).unwrap();
        let limits = SearchLimits {
            depth: Some(depth),
            ..Default::default()
        };
        let common = CommonOptions::default();
        let stop = AtomicBool::new(false);
        let ctx = SearchContext {
            limits: &limits,
            common: &common,
            stop: &stop,
            show_wdl: true,
            verbose_move_stats: false,
            params: &crate::search::SearchParams::default(),
        };
        let r = AlphaBeta.search(&mut net, &game, &ctx);
        (r.best, game)
    }

    fn uci(game: &Game, m: &str) -> Move {
        game.position().parse_uci_move(m).unwrap()
    }

    #[test]
    fn mate_in_one() {
        // back-rank mate, whatever the (untrained) network thinks
        let (m, g) = run("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 2);
        assert_eq!(m, uci(&g, "a1a8"));
    }

    #[test]
    fn mate_in_two() {
        // Morphy: 1. Ra6! (bxa6 2. b7#); no mate in one, and Ra6 is the
        // only move that forces mate in two (verified with python-chess)
        let fen = "kbK5/pp6/1P6/8/8/8/8/R7 w - - 0 1";
        let (m, g) = run(fen, 3);
        assert_eq!(m, uci(&g, "a1a6"));
    }

    #[test]
    fn wdl_of_mating_pv_is_a_win() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let game = Game::from_fen("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1").unwrap();
        let limits = SearchLimits::default();
        let common = CommonOptions::default();
        let stop = AtomicBool::new(false);
        let ctx = SearchContext {
            limits: &limits,
            common: &common,
            stop: &stop,
            show_wdl: true,
            verbose_move_stats: false,
            params: &crate::search::SearchParams::default(),
        };
        let mut w = Worker {
            net: &mut net,
            ctx: &ctx,
            start: Instant::now(),
            hard: None,
            nodes: 0,
            seldepth: 0,
            aborted: false,
            pv: vec![[Move::NULL; MAX_PLY]; MAX_PLY],
            pv_len: [0; MAX_PLY],
        };
        let mate = uci(&game, "a1a8");
        assert_eq!(w.pv_leaf_wdl(&game, &[mate]), [1.0, 0.0, 0.0]);
    }

    /// Plain minimax over the same tree, same leaf score, same draw/mate
    /// rules, no pruning. Alpha-beta must return exactly the same value.
    fn minimax(net: &mut PolicyNet, game: &Game, depth: i32, ply: usize) -> i32 {
        let pos = game.position();
        if ply > 0
            && (pos.halfmove_clock() >= 100
                || pos.insufficient_material()
                || game.repetitions() >= 1)
        {
            return 0;
        }
        let legal = pos.legal_moves();
        if legal.is_empty() {
            return if pos.in_check() {
                -MATE + ply as i32
            } else {
                0
            };
        }
        if depth <= 0 {
            let [w, _, l] = net.evaluate(game).wdl;
            return q_to_cp(w - l).clamp(-MAX_EVAL, MAX_EVAL);
        }
        let mut best = -INF;
        for &m in legal.iter() {
            let mut child = game.clone();
            child.push(m);
            best = best.max(-minimax(net, &child, depth - 1, ply + 1));
        }
        best
    }

    #[test]
    fn alphabeta_equals_minimax() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let limits = SearchLimits::default();
        let common = CommonOptions::default();
        let stop = AtomicBool::new(false);
        let ctx = SearchContext {
            limits: &limits,
            common: &common,
            stop: &stop,
            show_wdl: false,
            verbose_move_stats: false,
            params: &crate::search::SearchParams::default(),
        };
        for (fen, depth) in [
            (
                "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
                2,
            ),
            (
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                2,
            ),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", 3),
            ("kbK5/pp6/1P6/8/8/8/8/R7 w - - 0 1", 3),
        ] {
            let game = Game::from_fen(fen).unwrap();
            let mut w = Worker {
                net: &mut net,
                ctx: &ctx,
                start: Instant::now(),
                hard: None,
                nodes: 0,
                seldepth: 0,
                aborted: false,
                pv: vec![[Move::NULL; MAX_PLY]; MAX_PLY],
                pv_len: [0; MAX_PLY],
            };
            let ab = w.negamax(&game, depth, -INF, INF, 0);
            let ab_nodes = w.nodes;
            let mm = minimax(&mut net, &game, depth, 0);
            assert_eq!(ab, mm, "{fen} depth {depth}");
            println!("{fen} d{depth}: score {ab} (alpha-beta {ab_nodes} nodes)");
        }
    }

    #[test]
    fn score_strings() {
        assert_eq!(score_string(MATE - 1), "mate 1");
        assert_eq!(score_string(MATE - 3), "mate 2");
        assert_eq!(score_string(-(MATE - 2)), "mate -1");
        assert_eq!(score_string(42), "cp 42");
    }
}
