//! Plain MCTS (AlphaZero-style PUCT) on top of the network.
//!
//! Each playout walks down the tree with PUCT, evaluates the new leaf with
//! the network once (policy softmax → priors of its children, W/D/L →
//! value) and backs the W/D/L up, flipping sides each ply. Checkmate,
//! stalemate and draws (repetition, 50-move rule, insufficient material)
//! get their exact value. The move played is the most visited one.
//!
//! Tunable (UCI options, see params.rs):
//! * exploration U = cpuct(N) · gini_scale · N^exponent · P / (1 + n), with
//!   optional logarithmic growth of cpuct and gini scaling;
//! * first-play urgency: an unvisited child is scored as a weighted parent
//!   Q minus a reduction that grows with the policy mass already visited,
//!   with separate settings at the root;
//! * policy softmax temperature, separate at the root.
//!
//! The original search (cpuct 1.5, unvisited children at Q = 0,
//! temperature 1) is still reachable through the options.
//!
//! Tree reuse: after a search the tree is kept with the game it was built
//! for. When the next `go` comes for the same game plus some moves (usually
//! our move and the opponent's reply), those moves are followed down the
//! tree and that node's subtree becomes the new root: its visits and values
//! carry over, so the search starts with work already done. The subtree is
//! copied into a second buffer (compaction), so memory stays bounded and
//! nothing else has to be freed. The rest of the tree is thrown away.
//!
//! Not (yet): virtual loss or batching, NN cache, proven-result
//! propagation, smart pruning.

use std::sync::atomic::Ordering;
use std::time::Instant;

use peazero_core::{Game, Move};

use super::{NetSearch, SearchContext, SearchParams};
use crate::engine::SearchResult;
use crate::policy::{q_to_cp, wdl_string, PolicyNet};
use crate::timeman;

/// Playouts when `go` has no time, node or depth limit.
const DEFAULT_PLAYOUTS: u64 = 400;
/// `go depth N` stops at this many playouts even if the average depth
/// can't reach N (e.g. every line ends in a forced mate).
const DEPTH_PLAYOUT_CAP: u64 = 100_000;
/// At least one info line this often.
const INFO_MS: u128 = 1000;

const ROOT: usize = 0;

#[derive(Clone)]
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

/// Priors from raw logits: softmax(logit / temp) over the legal moves.
fn priors(moves: &[(Move, f32)], temp: f32) -> Vec<(Move, f32)> {
    let inv_t = 1.0 / temp.max(1e-3);
    let max = moves.iter().map(|m| m.1).fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = moves.iter().map(|m| ((m.1 - max) * inv_t).exp()).collect();
    let sum: f32 = exps.iter().sum();
    moves
        .iter()
        .zip(exps)
        .map(|(m, e)| (m.0, e / sum))
        .collect()
}

/// The MCTS search mode; keeps the last tree for reuse.
#[derive(Default)]
pub struct Mcts {
    saved: Option<Saved>,
    /// Spare arena for compaction (keeps its allocation between moves).
    spare: Vec<Node>,
}

/// A finished search's tree and what it was built for.
struct Saved {
    nodes: Vec<Node>,
    game: Game,
    params: SearchParams,
}

struct Tree<'a> {
    net: &'a mut PolicyNet,
    params: SearchParams,
    nodes: Vec<Node>,
    seldepth: usize,
    depth_sum: u64,
    /// Playouts of this search (the root's visits include reused ones).
    playouts: u64,
}

/// Copy the subtree of `old[root]` into `out` (cleared first) with `root`
/// at index 0. Breadth-first, so every node's children stay contiguous.
fn compact(old: &[Node], root: usize, out: &mut Vec<Node>) {
    out.clear();
    let mut src = vec![root]; // src[i] = index in `old` of out[i]
    out.push(old[root].clone());
    let mut i = 0;
    while i < out.len() {
        let o = &old[src[i]];
        if o.count > 0 {
            out[i].first = out.len();
            out.extend_from_slice(&old[o.first..o.first + o.count]);
            src.extend(o.first..o.first + o.count);
        }
        i += 1;
    }
}

/// Root children got their priors with the tree temperature `t_tree`;
/// convert them to the root temperature: p ∝ exp(logit / T), so
/// p_root ∝ p_tree^(t_tree / t_root).
fn retemper(children: &mut [Node], t_tree: f32, t_root: f32) {
    if (t_tree - t_root).abs() < 1e-6 {
        return;
    }
    let k = t_tree / t_root.max(1e-3);
    let mut sum = 0.0;
    for c in children.iter_mut() {
        c.prior = c.prior.powf(k);
        sum += c.prior;
    }
    if sum > 0.0 {
        for c in children.iter_mut() {
            c.prior /= sum;
        }
    }
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
        let temp = if node == ROOT {
            self.params.policy_temp_root
        } else {
            self.params.policy_temp
        };
        let first = self.nodes.len();
        for (m, prior) in priors(&eval.moves, temp) {
            self.nodes.push(Node::new(m, prior));
        }
        let n = &mut self.nodes[node];
        n.first = first;
        n.count = eval.moves.len();
        n.expanded = true;
        eval.wdl
    }

    /// (FPU, gini) of `node`. FPU: value of an unvisited child, from the
    /// view of the side to move at `node`: the parent's own Q minus the FPU
    /// reduction times √(policy mass of the children already visited).
    /// gini = 1 − Σ prior² of the children.
    fn fpu_and_gini(&self, node: usize) -> (f32, f32) {
        let n = &self.nodes[node];
        let (mut visited, mut squares) = (0.0f32, 0.0f32);
        for c in &self.nodes[n.first..n.first + n.count] {
            squares += c.prior * c.prior;
            if c.visits > 0 {
                visited += c.prior;
            }
        }
        let p = &self.params;
        let (weight, reduction) = if node == ROOT {
            (p.fpu_parent_weight_root, p.fpu_reduction_root)
        } else {
            (p.fpu_parent_weight, p.fpu_reduction)
        };
        // `n.q()` is from the view of the side that moved into `node`
        let fpu = -weight * n.q() - reduction * visited.sqrt();
        (fpu, 1.0 - squares)
    }

    #[cfg(test)]
    fn fpu(&self, node: usize) -> f32 {
        self.fpu_and_gini(node).0
    }

    fn select(&self, node: usize) -> usize {
        let n = &self.nodes[node];
        let (fpu, gini) = self.fpu_and_gini(node);
        let u = self.params.explore_coef(n.visits, gini);
        let mut best = (n.first, f32::NEG_INFINITY);
        for c in n.first..n.first + n.count {
            let child = &self.nodes[c];
            let q = if child.visits == 0 { fpu } else { child.q() };
            let score = q + u * child.prior / (1.0 + child.visits as f32);
            if score > best.1 {
                best = (c, score);
            }
        }
        best.0
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
        self.playouts += 1;

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

    /// Average depth of this search's playouts (rounded down): the
    /// `depth` of MCTS.
    fn depth(&self) -> u64 {
        self.depth_sum / self.playouts.max(1)
    }

    fn print_info(&self, start: Instant, show_wdl: bool) {
        let Some(best) = self.best_child(ROOT) else {
            return;
        };
        let b = &self.nodes[best];
        let playouts = self.playouts;
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

impl Mcts {
    /// The saved tree, re-rooted at `game`, if `game` continues the saved
    /// game and the search can start from it; `None` means a fresh tree.
    /// Returns the nodes and how many playouts were kept.
    fn take_reusable(&mut self, game: &Game, ctx: &SearchContext) -> Option<(Vec<Node>, u32)> {
        let saved = self.saved.take()?;
        if saved.params != *ctx.params
            || !ctx.limits.searchmoves.is_empty()
            || saved.game.start_position() != game.start_position()
            || !game.moves().starts_with(saved.game.moves())
        {
            return None;
        }
        let nodes = saved.nodes;
        let mut node = ROOT;
        for &m in &game.moves()[saved.game.moves().len()..] {
            let n = &nodes[node];
            node = (n.first..n.first + n.count).find(|&c| nodes[c].mv == m)?;
        }
        // the new root must be an expanded, non-terminal position (a node
        // marked as a draw by repetition is not a draw at the root)
        let n = &nodes[node];
        if !n.expanded || n.terminal.is_some() || n.count == 0 {
            return None;
        }
        if node == ROOT {
            let kept = n.visits;
            return Some((nodes, kept)); // same position again
        }
        let mut out = std::mem::take(&mut self.spare);
        compact(&nodes, node, &mut out);
        self.spare = nodes;
        let count = out[ROOT].count;
        out[ROOT].mv = Move::NULL;
        let p = ctx.params;
        retemper(&mut out[1..1 + count], p.policy_temp, p.policy_temp_root);
        let kept = out[ROOT].visits;
        Some((out, kept))
    }
}

impl NetSearch for Mcts {
    fn new_game(&mut self) {
        self.saved = None;
    }

    fn search(&mut self, net: &mut PolicyNet, game: &Game, ctx: &SearchContext) -> SearchResult {
        let (reused, kept) = match self.take_reusable(game, ctx) {
            Some((nodes, kept)) => (Some(nodes), kept),
            None => (None, 0),
        };
        if kept > 0 {
            println!("info string tree reuse: {kept} playouts kept");
        }
        let (result, _, nodes) = search_tree(net, game, ctx, true, reused);
        // searchmoves trims the root: don't reuse such a tree
        if ctx.limits.searchmoves.is_empty() {
            self.saved = Some(Saved {
                nodes,
                game: game.clone(),
                params: *ctx.params,
            });
        }
        result
    }
}

/// Run a search on a fresh tree; returns the result and the number of
/// playouts. `print = false` suppresses the info lines (used by `bench`).
pub fn run(
    net: &mut PolicyNet,
    game: &Game,
    ctx: &SearchContext,
    print: bool,
) -> (SearchResult, u64) {
    let (result, playouts, _) = search_tree(net, game, ctx, print, None);
    (result, playouts)
}

/// Search from `reused` (a re-rooted tree from the previous move) or a
/// fresh tree. Returns the result, this search's playouts and the tree.
fn search_tree(
    net: &mut PolicyNet,
    game: &Game,
    ctx: &SearchContext,
    print: bool,
    reused: Option<Vec<Node>>,
) -> (SearchResult, u64, Vec<Node>) {
    let start = Instant::now();
    let limits = ctx.limits;
    let budget = timeman::budget(
        limits,
        game.position().side_to_move(),
        ctx.common.move_overhead_ms,
    );
    // node limits count this search's playouts, not the reused ones
    let max_playouts = limits.nodes.unwrap_or(if limits.depth.is_some() {
        DEPTH_PLAYOUT_CAP
    } else if budget.is_none() && !limits.infinite {
        DEFAULT_PLAYOUTS
    } else {
        u64::MAX
    });

    let mut tree = Tree {
        net,
        params: *ctx.params,
        nodes: Vec::new(),
        seldepth: 0,
        depth_sum: 0,
        playouts: 0,
    };
    match reused {
        Some(nodes) => tree.nodes = nodes,
        None => {
            tree.nodes.push(Node::new(Move::NULL, 1.0));
            tree.playout(game); // expands the root
        }
    }
    if !limits.searchmoves.is_empty() {
        // searchmoves: keep only those root moves (always a fresh tree, so
        // they have no subtrees yet and can be moved around in the arena)
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
    while tree.playouts < max_playouts
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
                shown_playouts = tree.playouts;
                last_info = ms;
                tree.print_info(start, ctx.show_wdl);
            }
        }
    }

    let playouts = tree.playouts;
    let Some(best) = tree.best_child(ROOT) else {
        if print {
            println!("info depth 0 score cp 0");
        }
        let none = SearchResult {
            best: Move::NULL,
            ponder: None,
        };
        return (none, playouts, tree.nodes);
    };
    if print && tree.playouts != shown_playouts {
        tree.print_info(start, ctx.show_wdl);
    }
    let ponder = tree.pv().get(1).copied();
    let result = SearchResult {
        best: tree.nodes[best].mv,
        ponder,
    };
    (result, playouts, tree.nodes)
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
            params: &crate::search::SearchParams::default(),
        };
        let r = Mcts::default().search(&mut net, &game, &ctx);
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
            params: &crate::search::SearchParams::default(),
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

    /// A root with three children (priors 0.5 / 0.3 / 0.2); only the first
    /// is visited (4 times, mean W-L `q` for the side to move at the root).
    fn small_tree(net: &mut PolicyNet, q: f32, params: SearchParams) -> Tree<'_> {
        let mut nodes = vec![Node::new(Move::NULL, 1.0)];
        for p in [0.5, 0.3, 0.2] {
            nodes.push(Node::new(Move::NULL, p));
        }
        let (w, l) = ((1.0 + q) / 2.0, (1.0 - q) / 2.0);
        nodes[1].visits = 4;
        nodes[1].wdl = [4.0 * w, 0.0, 4.0 * l];
        // the root stores results for the side that moved into it
        nodes[0].visits = 5;
        nodes[0].wdl = [5.0 * l, 0.0, 5.0 * w];
        nodes[0].first = 1;
        nodes[0].count = 3;
        nodes[0].expanded = true;
        Tree {
            net,
            params,
            nodes,
            seldepth: 0,
            depth_sum: 0,
            playouts: 0,
        }
    }

    #[test]
    fn fpu_follows_the_parent_value() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        // weight 0, reduction 0: unvisited moves are a draw, whatever the
        // parent's value (the original search)
        let d = SearchParams {
            fpu_parent_weight_root: 0.0,
            fpu_reduction_root: 0.0,
            ..Default::default()
        };
        assert_eq!(small_tree(&mut net, -0.8, d).fpu(ROOT), 0.0);
        assert_eq!(small_tree(&mut net, 0.8, d).fpu(ROOT), 0.0);
        // weight 1, reduction 1: the parent's Q minus √(visited prior mass)
        let p = SearchParams {
            fpu_parent_weight_root: 1.0,
            fpu_reduction_root: 1.0,
            ..Default::default()
        };
        let r = 0.5f32.sqrt();
        let lost = small_tree(&mut net, -0.8, p).fpu(ROOT);
        assert!((lost - (-0.8 - r)).abs() < 1e-5, "{lost}");
        let won = small_tree(&mut net, 0.8, p).fpu(ROOT);
        assert!((won - (0.8 - r)).abs() < 1e-5, "{won}");
        // half weight, no reduction
        let h = SearchParams {
            fpu_parent_weight_root: 0.5,
            fpu_reduction_root: 0.0,
            ..Default::default()
        };
        assert!((small_tree(&mut net, 0.3, h).fpu(ROOT) - 0.15).abs() < 1e-5);
    }

    #[test]
    fn fpu_changes_which_move_is_explored() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        // FPU = draw (the original search): in a lost position every
        // unvisited move looks better than the visited one
        let d = SearchParams {
            fpu_parent_weight_root: 0.0,
            fpu_reduction_root: 0.0,
            ..Default::default()
        };
        assert_eq!(small_tree(&mut net, -0.8, d).select(ROOT), 2);
        // parent-Q FPU with a huge reduction: stay on the visited move
        let p = SearchParams {
            fpu_reduction_root: 3.0,
            ..Default::default()
        };
        assert_eq!(small_tree(&mut net, -0.8, p).select(ROOT), 1);
        assert_eq!(small_tree(&mut net, 0.8, p).select(ROOT), 1);
    }

    #[test]
    fn policy_temperature_flattens_priors() {
        let m = Move::NULL;
        let moves = [(m, 2.0), (m, 0.0), (m, -1.0)];
        let p1 = priors(&moves, 1.0);
        let p2 = priors(&moves, 2.0);
        for p in [&p1, &p2] {
            assert!((p.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);
        }
        assert!((p1[0].1 - 2f32.exp() / (2f32.exp() + 1.0 + (-1f32).exp())).abs() < 1e-5);
        assert!(p2[0].1 < p1[0].1 && p2[2].1 > p1[2].1);
    }

    #[test]
    fn default_params_find_mate_at_small_budget() {
        // mate in one at a small budget
        let (m, g) = run("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", 100);
        assert_eq!(m, g.position().parse_uci_move("a1a8").unwrap());
    }

    // ---------------------------------------------------------- tree reuse

    /// Every expanded, non-terminal node was visited once when it was
    /// expanded and once more for every visit of one of its children.
    fn check_tree(nodes: &[Node]) {
        for (i, n) in nodes.iter().enumerate() {
            assert!(
                n.first + n.count <= nodes.len(),
                "node {i} children out of range"
            );
            if n.expanded && n.terminal.is_none() {
                let below: u32 = nodes[n.first..n.first + n.count]
                    .iter()
                    .map(|c| c.visits)
                    .sum();
                assert_eq!(n.visits, 1 + below, "node {i}");
            }
        }
    }

    /// Search `game` with `m` for `playouts` new playouts.
    fn search_with(
        m: &mut Mcts,
        net: &mut PolicyNet,
        game: &Game,
        playouts: u64,
        params: &SearchParams,
    ) -> Move {
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
            show_wdl: false,
            verbose_move_stats: false,
            params,
        };
        m.search(net, game, &ctx).best
    }

    fn ctx_for<'a>(
        limits: &'a SearchLimits,
        common: &'a CommonOptions,
        stop: &'a AtomicBool,
        params: &'a SearchParams,
    ) -> SearchContext<'a> {
        SearchContext {
            limits,
            common,
            stop,
            show_wdl: false,
            verbose_move_stats: false,
            params,
        }
    }

    #[test]
    fn reuse_keeps_the_subtree_of_the_moves_played() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let params = SearchParams::default();
        let mut m = Mcts::default();
        let mut game = Game::default();
        let best = search_with(&mut m, &mut net, &game, 400, &params);
        let saved = m.saved.as_ref().unwrap();
        check_tree(&saved.nodes);
        // the expected reply: the most visited grandchild under our move
        let nodes = &saved.nodes;
        let ours = (nodes[ROOT].first..nodes[ROOT].first + nodes[ROOT].count)
            .find(|&c| nodes[c].mv == best)
            .unwrap();
        let o = &nodes[ours];
        let reply = (o.first..o.first + o.count)
            .max_by_key(|&c| nodes[c].visits)
            .unwrap();
        let (reply_mv, reply_visits) = (nodes[reply].mv, nodes[reply].visits);
        let reply_children: Vec<Move> = (nodes[reply].first
            ..nodes[reply].first + nodes[reply].count)
            .map(|c| nodes[c].mv)
            .collect();
        assert!(reply_visits > 1, "the reply should have a subtree");

        game.push(best);
        game.push(reply_mv);
        let limits = SearchLimits::default();
        let (common, stop) = (CommonOptions::default(), AtomicBool::new(false));
        let ctx = ctx_for(&limits, &common, &stop, &params);
        let (nodes, kept) = m.take_reusable(&game, &ctx).expect("reusable");
        assert_eq!(kept, reply_visits);
        assert_eq!(nodes[ROOT].visits, reply_visits);
        check_tree(&nodes);
        // same children, in the same order, priors re-tempered to the root
        // temperature and still summing to 1
        let root = &nodes[ROOT];
        let kids = &nodes[root.first..root.first + root.count];
        assert_eq!(
            kids.iter().map(|c| c.mv).collect::<Vec<_>>(),
            reply_children
        );
        assert!((kids.iter().map(|c| c.prior).sum::<f32>() - 1.0).abs() < 1e-4);
        // only the subtree was copied: smaller than the old arena, which
        // is kept as the spare buffer for the next compaction
        assert!(nodes.len() < m.spare.len());
    }

    #[test]
    fn retemper_matches_a_fresh_softmax() {
        let logits = [2.0f32, 0.5, -1.0, -3.0];
        let moves: Vec<(Move, f32)> = logits.iter().map(|&l| (Move::NULL, l)).collect();
        let tree = priors(&moves, 1.6);
        let root = priors(&moves, 2.1);
        let mut nodes: Vec<Node> = tree.iter().map(|&(m, p)| Node::new(m, p)).collect();
        retemper(&mut nodes, 1.6, 2.1);
        for (n, r) in nodes.iter().zip(&root) {
            assert!((n.prior - r.1).abs() < 1e-5, "{} vs {}", n.prior, r.1);
        }
    }

    #[test]
    fn no_reuse_for_other_games_or_settings() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let params = SearchParams::default();
        let limits = SearchLimits::default();
        let (common, stop) = (CommonOptions::default(), AtomicBool::new(false));
        let ctx = ctx_for(&limits, &common, &stop, &params);
        let start = Game::default();

        // a different game (not a continuation)
        let mut m = Mcts::default();
        let mut g = start.clone();
        g.push_uci("e2e4");
        search_with(&mut m, &mut net, &g, 100, &params);
        let mut other = start.clone();
        other.push_uci("d2d4");
        assert!(m.take_reusable(&other, &ctx).is_none());

        // changed parameters
        search_with(&mut m, &mut net, &g, 100, &params);
        let changed = SearchParams {
            cpuct: params.cpuct + 0.5,
            ..params
        };
        let ctx2 = ctx_for(&limits, &common, &stop, &changed);
        assert!(m.take_reusable(&g, &ctx2).is_none());

        // ucinewgame
        search_with(&mut m, &mut net, &g, 100, &params);
        m.new_game();
        assert!(m.take_reusable(&g, &ctx).is_none());

        // same position again: the whole tree, nothing copied
        search_with(&mut m, &mut net, &g, 100, &params);
        let (nodes, kept) = m.take_reusable(&g, &ctx).unwrap();
        assert_eq!(kept, 100);
        check_tree(&nodes);
    }

    #[test]
    fn reuse_over_a_game_plays_legal_moves_and_counts_new_playouts() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let params = SearchParams::default();
        let mut m = Mcts::default();
        let mut game = Game::default();
        let mut reused = 0;
        for _ in 0..16 {
            let mv = search_with(&mut m, &mut net, &game, 150, &params);
            assert!(game.position().legal_moves().contains(&mv));
            let s = m.saved.as_ref().unwrap();
            check_tree(&s.nodes);
            // the root holds the kept visits plus exactly 150 new ones
            assert!(s.nodes[ROOT].visits >= 150);
            if s.nodes[ROOT].visits > 150 {
                reused += 1;
            }
            game.push(mv);
        }
        // the engine plays both sides here, so every search after the
        // first starts from the previous one's subtree
        assert!(reused >= 10, "reused {reused} times");
    }
}
