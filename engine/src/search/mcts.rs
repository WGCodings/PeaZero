//! Plain MCTS (AlphaZero-style PUCT) on top of the network.
//!
//! Each playout walks down the tree with PUCT, evaluates the new leaf with
//! the network once (policy softmax → priors of its children, W/D/L →
//! value) and backs the W/D/L up, flipping sides each ply. Checkmate,
//! stalemate and draws (repetition, 50-move rule, insufficient material)
//! get their exact value. The move played is the most visited one.
//!
//! Nothing else on purpose: no tree reuse between moves, no virtual loss or
//! batching, no noise or temperature, unvisited moves start at Q = 0.

use std::sync::atomic::Ordering;
use std::time::Instant;

use peazero_core::{Game, Move};

use super::{NetSearch, SearchContext};
use crate::engine::SearchResult;
use crate::policy::{q_to_cp, wdl_string, PolicyNet};
use crate::timeman;

/// Exploration constant of PUCT.
const C_PUCT: f32 = 1.5;
/// Playouts when `go` has no time, node or depth limit.
const DEFAULT_PLAYOUTS: u64 = 400;
/// `go depth N` stops at this many playouts even if the average depth
/// can't reach N (e.g. every line ends in a forced mate).
const DEPTH_PLAYOUT_CAP: u64 = 100_000;
/// At least one info line this often.
const INFO_MS: u128 = 1000;

const ROOT: usize = 0;

struct Node {
    /// Move that leads here from the parent.
    mv: Move,
    prior: f32,
    /// Children are `first..first + count` in the arena.
    first: usize,
    count: usize,
    expanded: bool,
    /// Exact W/D/L for the side to move here, if the game is over.
    terminal: Option<[f32; 3]>,
    visits: u32,
    /// Summed W/D/L from the point of view of the side that played `mv`.
    wdl: [f32; 3],
}

impl Node {
    fn new(mv: Move, prior: f32) -> Node {
        Node {
            mv,
            prior,
            first: 0,
            count: 0,
            expanded: false,
            terminal: None,
            visits: 0,
            wdl: [0.0; 3],
        }
    }

    /// Mean `W - L` for the side that played `mv`.
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            (self.wdl[0] - self.wdl[2]) / self.visits as f32
        }
    }
}

fn flip([w, d, l]: [f32; 3]) -> [f32; 3] {
    [l, d, w]
}

pub struct Mcts;

struct Tree<'a> {
    net: &'a mut PolicyNet,
    nodes: Vec<Node>,
    seldepth: usize,
    depth_sum: u64,
}

impl Tree<'_> {
    /// Evaluate `game` at `node`: exact result if the game is over,
    /// otherwise expand with the network. Returns W/D/L for its side to move.
    fn evaluate(&mut self, node: usize, game: &Game, ply: usize) -> [f32; 3] {
        let pos = game.position();
        let draw = [0.0, 1.0, 0.0];
        let terminal = if ply > 0
            && (pos.halfmove_clock() >= 100
                || pos.insufficient_material()
                || game.repetitions() >= 1)
        {
            Some(draw)
        } else if pos.legal_moves().is_empty() {
            Some(if pos.in_check() {
                [0.0, 0.0, 1.0]
            } else {
                draw
            })
        } else {
            None
        };
        if let Some(t) = terminal {
            self.nodes[node].terminal = Some(t);
            return t;
        }

        let eval = self.net.evaluate(game);
        let max = eval
            .moves
            .iter()
            .map(|m| m.1)
            .fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = eval.moves.iter().map(|m| (m.1 - max).exp()).sum();
        let first = self.nodes.len();
        for &(m, logit) in &eval.moves {
            self.nodes.push(Node::new(m, (logit - max).exp() / sum));
        }
        let n = &mut self.nodes[node];
        n.first = first;
        n.count = eval.moves.len();
        n.expanded = true;
        eval.wdl
    }

    fn select(&self, node: usize) -> usize {
        let n = &self.nodes[node];
        let sqrt_n = (n.visits.max(1) as f32).sqrt();
        (n.first..n.first + n.count)
            .max_by(|&a, &b| {
                let score = |c: usize| {
                    let c = &self.nodes[c];
                    c.q() + C_PUCT * c.prior * sqrt_n / (1.0 + c.visits as f32)
                };
                score(a).total_cmp(&score(b))
            })
            .unwrap()
    }

    fn playout(&mut self, root: &Game) {
        let mut path = vec![ROOT];
        let mut game = root.clone();
        let mut node = ROOT;
        while self.nodes[node].expanded && self.nodes[node].terminal.is_none() {
            node = self.select(node);
            game.push(self.nodes[node].mv);
            path.push(node);
        }
        let ply = path.len() - 1;
        let wdl = match self.nodes[node].terminal {
            Some(t) => t,
            None => self.evaluate(node, &game, ply),
        };
        self.seldepth = self.seldepth.max(ply);
        self.depth_sum += ply as u64;

        // each node stores results for the side that moved into it
        let mut v = flip(wdl);
        for &i in path.iter().rev() {
            let n = &mut self.nodes[i];
            n.visits += 1;
            for (acc, x) in n.wdl.iter_mut().zip(v) {
                *acc += x;
            }
            v = flip(v);
        }
    }

    fn best_child(&self, node: usize) -> Option<usize> {
        let n = &self.nodes[node];
        (n.first..n.first + n.count).max_by(|&a, &b| {
            let (a, b) = (&self.nodes[a], &self.nodes[b]);
            a.visits.cmp(&b.visits).then(a.prior.total_cmp(&b.prior))
        })
    }

    fn pv(&self) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut node = ROOT;
        while let Some(c) = self.best_child(node) {
            if self.nodes[c].visits == 0 {
                break;
            }
            pv.push(self.nodes[c].mv);
            node = c;
        }
        pv
    }

    /// Average playout depth (rounded down): the `depth` of MCTS.
    fn depth(&self) -> u64 {
        self.depth_sum / (self.nodes[ROOT].visits as u64).max(1)
    }

    fn print_info(&self, start: Instant, show_wdl: bool) {
        let Some(best) = self.best_child(ROOT) else {
            return;
        };
        let b = &self.nodes[best];
        let playouts = self.nodes[ROOT].visits as u64;
        let wdl = if b.visits > 0 {
            b.wdl.map(|x| x / b.visits as f32)
        } else {
            [0.0, 1.0, 0.0]
        };
        let wdl_str = if show_wdl {
            wdl_string(wdl)
        } else {
            String::new()
        };
        let ms = start.elapsed().as_millis();
        let pv: Vec<String> = self.pv().iter().map(|m| m.to_uci()).collect();
        println!(
            "info depth {} seldepth {} multipv 1 score cp {}{wdl_str} nodes {playouts} nps {} time {ms} pv {}",
            self.depth(),
            self.seldepth,
            q_to_cp(wdl[0] - wdl[2]),
            playouts as u128 * 1000 / ms.max(1),
            pv.join(" ")
        );
    }
}

impl NetSearch for Mcts {
    fn search(&mut self, net: &mut PolicyNet, game: &Game, ctx: &SearchContext) -> SearchResult {
        run(net, game, ctx, true).0
    }
}

/// Run the search; returns the result and the number of playouts.
/// `print = false` suppresses the info lines (used by `bench`).
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
    let max_playouts = limits.nodes.unwrap_or(if limits.depth.is_some() {
        DEPTH_PLAYOUT_CAP
    } else if budget.is_none() && !limits.infinite {
        DEFAULT_PLAYOUTS
    } else {
        u64::MAX
    });

    let mut tree = Tree {
        net,
        nodes: vec![Node::new(Move::NULL, 1.0)],
        seldepth: 0,
        depth_sum: 0,
    };
    tree.playout(game); // expands the root
    if !limits.searchmoves.is_empty() {
        // searchmoves: keep only those root moves (they have no subtrees
        // yet, so they can be moved around in the arena)
        let (first, count) = (tree.nodes[ROOT].first, tree.nodes[ROOT].count);
        let mut kept = 0;
        for c in first..first + count {
            if limits.searchmoves.contains(&tree.nodes[c].mv.to_uci()) {
                tree.nodes.swap(first + kept, c);
                kept += 1;
            }
        }
        if kept > 0 {
            tree.nodes[ROOT].count = kept;
        }
    }

    let target_depth = limits.depth.map(u64::from);
    let (mut shown_depth, mut shown_seldepth, mut last_info) = (0, 0, 0u128);
    let mut shown_playouts = 0;
    while (tree.nodes[ROOT].visits as u64) < max_playouts
        && tree.nodes[ROOT].count > 0
        && target_depth.is_none_or(|d| tree.depth() < d)
        && !ctx.stop.load(Ordering::Relaxed)
        && budget.is_none_or(|b| start.elapsed() < b.soft)
    {
        tree.playout(game);
        if print {
            let ms = start.elapsed().as_millis();
            if tree.depth() > shown_depth
                || tree.seldepth > shown_seldepth
                || ms - last_info >= INFO_MS
            {
                shown_depth = tree.depth();
                shown_seldepth = tree.seldepth;
                shown_playouts = tree.nodes[ROOT].visits;
                last_info = ms;
                tree.print_info(start, ctx.show_wdl);
            }
        }
    }

    let playouts = tree.nodes[ROOT].visits as u64;
    let Some(best) = tree.best_child(ROOT) else {
        if print {
            println!("info depth 0 score cp 0");
        }
        let none = SearchResult {
            best: Move::NULL,
            ponder: None,
        };
        return (none, playouts);
    };
    if print && tree.nodes[ROOT].visits != shown_playouts {
        tree.print_info(start, ctx.show_wdl);
    }
    let ponder = tree.pv().get(1).copied();
    let result = SearchResult {
        best: tree.nodes[best].mv,
        ponder,
    };
    (result, playouts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{CommonOptions, SearchLimits};
    use std::sync::atomic::AtomicBool;

    fn run(fen: &str, playouts: u64) -> (Move, Game) {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let game = Game::from_fen(fen).unwrap();
        let limits = SearchLimits {
            nodes: Some(playouts),
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
        };
        let r = Mcts.search(&mut net, &game, &ctx);
        (r.best, game)
    }

    fn depth_search(fen: &str, depth: u32) -> u64 {
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
            show_wdl: false,
            verbose_move_stats: false,
        };
        super::run(&mut net, &game, &ctx, false).1
    }

    #[test]
    fn go_depth_stops_at_depth_and_is_deterministic() {
        peazero_core::init();
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let a = depth_search(fen, 2);
        assert_eq!(a, depth_search(fen, 2));
        assert!(depth_search(fen, 1) < a);
    }

    #[test]
    fn mate_in_one() {
        // back-rank mate; the exact terminal value makes it win the visits
        let (m, g) = run("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 300);
        assert_eq!(m, g.position().parse_uci_move("a1a8").unwrap());
    }

    #[test]
    fn single_playout_plays_policy_move() {
        // with one playout (root expansion only) it plays the top prior,
        // i.e. the searchless move
        peazero_core::init();
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let (m, g) = run(fen, 1);
        let mut net = PolicyNet::embedded();
        let eval = net.evaluate(&g);
        let top = eval
            .moves
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0;
        assert_eq!(m, top);
    }
}
