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
//! Memory: `Hash` is the whole budget. `TREE_SHARE_PERCENT` of it holds the
//! tree (24-byte nodes in one arena, allocated once), the rest the
//! evaluation table (tt.rs). When the arena is full the search prunes and
//! goes on: the least-visited expanded nodes are collapsed back into
//! leaves (they keep their own visits and W/L, only their children go)
//! until the tree fits in half the arena. Most of the arena is unvisited
//! children, so this frees a lot while the well-visited part of the tree,
//! which decides the move, is untouched. A collapsed node that becomes
//! interesting again is simply expanded again (usually a table hit).
//!
//! Tree reuse: after a search the tree is kept with the game it was built
//! for. When the next `go` comes for the same game plus some moves (usually
//! our move and the opponent's reply), those moves are followed down the
//! tree and that node's subtree becomes the new root: its visits and values
//! carry over, so the search starts with work already done. The subtree is
//! compacted in place to the front of the arena; the rest is dropped.
//!
//! Transpositions: network evaluations go through the evaluation table, so
//! a position reached again by another move order costs a table probe
//! instead of a network call. Each tree node still collects its own visits.
//!
//! Not (yet): virtual loss or batching, proven-result propagation, smart
//! pruning.

use std::ops::Range;
use std::sync::atomic::Ordering;
use std::time::Instant;

use peazero_core::{Game, Move};

use super::tt::EvalTable;
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
/// Share of `Hash` for the tree; the rest is the evaluation table.
const TREE_SHARE_PERCENT: usize = 75;

const ROOT: usize = 0;

/// `Node::flags` bits.
const EXPANDED: u8 = 1;
const TERMINAL_DRAW: u8 = 2;
const TERMINAL_LOSS: u8 = 4;

/// One tree node: 24 bytes.
#[derive(Clone, Copy)]
struct Node {
    /// Move that leads here from the parent.
    mv: Move,
    /// Number of children (at most 218 legal moves).
    count: u8,
    /// `EXPANDED`, `TERMINAL_DRAW`, `TERMINAL_LOSS`.
    flags: u8,
    prior: f32,
    /// Children are `first..first + count` in the arena; they always come
    /// after their parent.
    first: u32,
    visits: u32,
    /// Summed W and L from the point of view of the side that played `mv`
    /// (D = visits − W − L: every playout adds a W/D/L summing to 1).
    w: f32,
    l: f32,
}

const _: () = assert!(std::mem::size_of::<Node>() == 24);

impl Node {
    fn new(mv: Move, prior: f32) -> Node {
        Node {
            mv,
            count: 0,
            flags: 0,
            prior,
            first: 0,
            visits: 0,
            w: 0.0,
            l: 0.0,
        }
    }

    fn children(&self) -> Range<usize> {
        self.first as usize..self.first as usize + self.count as usize
    }

    fn expanded(&self) -> bool {
        self.flags & EXPANDED != 0
    }

    /// Exact W/D/L for the side to move here, if the game is over.
    fn terminal(&self) -> Option<[f32; 3]> {
        if self.flags & TERMINAL_LOSS != 0 {
            Some([0.0, 0.0, 1.0])
        } else if self.flags & TERMINAL_DRAW != 0 {
            Some([0.0, 1.0, 0.0])
        } else {
            None
        }
    }

    /// Mean `W - L` for the side that played `mv`.
    fn q(&self) -> f32 {
        if self.visits == 0 {
            0.0
        } else {
            (self.w - self.l) / self.visits as f32
        }
    }

    /// Mean W/D/L for the side that played `mv`.
    fn mean_wdl(&self) -> [f32; 3] {
        if self.visits == 0 {
            return [0.0, 1.0, 0.0];
        }
        let n = self.visits as f32;
        let (w, l) = (self.w / n, self.l / n);
        [w, (1.0 - w - l).max(0.0), l]
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

/// (tree nodes, evaluation-table bytes) for `Hash` = `mb`.
fn budget(mb: usize) -> (usize, usize) {
    let bytes = mb.max(1) << 20;
    let tree = bytes * TREE_SHARE_PERCENT / 100;
    (tree / std::mem::size_of::<Node>(), bytes - tree)
}

/// The MCTS search mode; keeps the arena (with the last tree, for reuse)
/// and the evaluation table between moves.
#[derive(Default)]
pub struct Mcts {
    /// The arena: allocated once with room for `max_nodes`.
    nodes: Vec<Node>,
    max_nodes: usize,
    /// What the tree in `nodes` was built for (None: nothing reusable).
    saved: Option<Saved>,
    /// Created on the first search, sized by `Hash`.
    tt: Option<EvalTable>,
    hash_mb: usize,
}

/// The game and parameters the saved tree was searched with.
struct Saved {
    game: Game,
    params: SearchParams,
}

struct Tree<'a> {
    net: &'a mut PolicyNet,
    tt: &'a mut EvalTable,
    params: SearchParams,
    nodes: &'a mut Vec<Node>,
    max_nodes: usize,
    /// The arena had no room for an expansion: prune before the next
    /// playout.
    full: bool,
    /// Pruning can't make room (a tiny `Hash`): search on without growing.
    frozen: bool,
    /// Prunes in this search, for the info string.
    prunes: u32,
    seldepth: usize,
    depth_sum: u64,
    /// Playouts of this search (the root's visits include reused ones).
    playouts: u64,
}

/// Keep only the subtree of `nodes[root]`, moved to the front of the arena
/// with `root` at index 0, in place. Children always come after their
/// parent in the arena, so one forward pass marks the subtree (a bitset)
/// and a second one slides the kept nodes down in their original order:
/// a node's new index (its rank among the kept nodes) is never above its
/// old one, so nothing is overwritten before it is read.
///
/// `collapse = Some(t)`: also turn every expanded node (other than the
/// root) with at most `t` visits back into a leaf, dropping its subtree;
/// it keeps its own visits and W/L.
fn compact_in_place(nodes: &mut Vec<Node>, root: usize, collapse: Option<u32>) {
    let n = nodes.len();
    let collapsed = |i: usize, node: &Node| i != root && collapse.is_some_and(|t| node.visits <= t);
    let mut kept = vec![0u64; n.div_ceil(64)];
    let set = |kept: &mut Vec<u64>, i: usize| kept[i / 64] |= 1 << (i % 64);
    let is = |kept: &Vec<u64>, i: usize| kept[i / 64] >> (i % 64) & 1 != 0;
    set(&mut kept, root);
    for (i, node) in nodes.iter().enumerate().skip(root) {
        if is(&kept, i) && !collapsed(i, node) {
            for c in node.children() {
                set(&mut kept, c);
            }
        }
    }
    // rank(i) = number of kept nodes before i
    let mut base = Vec::with_capacity(kept.len());
    let mut total = 0u32;
    for word in &kept {
        base.push(total);
        total += word.count_ones();
    }
    let rank = |i: usize| base[i / 64] + (kept[i / 64] & ((1u64 << (i % 64)) - 1)).count_ones();
    for i in root..n {
        if is(&kept, i) {
            let mut node = nodes[i];
            if node.count > 0 && collapsed(i, &node) {
                node.count = 0;
                node.first = 0;
                node.flags &= !EXPANDED;
            } else if node.count > 0 {
                node.first = rank(node.first as usize);
            }
            nodes[rank(i) as usize] = node;
        }
    }
    nodes.truncate(total as usize);
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
    /// otherwise expand with the network (through the evaluation table).
    /// Returns W/D/L for its side to move. If the arena has no room for the
    /// children, the node stays unexpanded (its value is still backed up)
    /// and `full` is set, so the search prunes before the next playout.
    fn evaluate(&mut self, node: usize, game: &Game, ply: usize) -> [f32; 3] {
        let pos = game.position();
        let draw = ply > 0
            && (pos.halfmove_clock() >= 100
                || pos.insufficient_material()
                || game.repetitions() >= 1);
        let terminal = if draw {
            Some(TERMINAL_DRAW)
        } else if pos.legal_moves().is_empty() {
            Some(if pos.in_check() {
                TERMINAL_LOSS
            } else {
                TERMINAL_DRAW
            })
        } else {
            None
        };
        if let Some(t) = terminal {
            let n = &mut self.nodes[node];
            n.flags |= t;
            return n.terminal().unwrap();
        }

        let eval = self.tt.evaluate(self.net, game);
        if self.nodes.len() + eval.moves.len() > self.max_nodes {
            self.full = !self.frozen;
            return eval.wdl;
        }
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
        n.first = first as u32;
        n.count = eval.moves.len() as u8;
        n.flags |= EXPANDED;
        eval.wdl
    }

    /// Make room in a full arena: collapse the least-visited expanded nodes
    /// (see the module documentation) until at most half the arena is used.
    /// Only between playouts (it moves nodes). Returns the nodes freed.
    fn prune(&mut self) -> usize {
        self.full = false;
        let before = self.nodes.len();
        let target = self.max_nodes / 2;
        if before > target {
            // collapsing every expanded node with ≤ t visits frees exactly
            // the children of those nodes (visits never grow downwards)
            let mut cand: Vec<(u32, usize)> = self.nodes[1..]
                .iter()
                .filter(|n| n.count > 0)
                .map(|n| (n.visits, n.count as usize))
                .collect();
            cand.sort_unstable_by_key(|c| c.0);
            let excess = before - target;
            let (mut freed, mut threshold) = (0, None);
            for (i, &(v, c)) in cand.iter().enumerate() {
                freed += c;
                threshold = Some(v);
                if freed >= excess && cand.get(i + 1).is_none_or(|next| next.0 != v) {
                    break;
                }
            }
            if threshold.is_some() {
                compact_in_place(self.nodes, ROOT, threshold);
                self.prunes += 1;
            }
        }
        // can't fit even the largest expansion: go on without growing
        if self.nodes.len() + 218 > self.max_nodes {
            self.frozen = true;
        }
        before - self.nodes.len()
    }

    /// (FPU, gini) of `node`. FPU: value of an unvisited child, from the
    /// view of the side to move at `node`: the parent's own Q minus the FPU
    /// reduction times √(policy mass of the children already visited).
    /// gini = 1 − Σ prior² of the children.
    fn fpu_and_gini(&self, node: usize) -> (f32, f32) {
        let n = &self.nodes[node];
        let (mut visited, mut squares) = (0.0f32, 0.0f32);
        for c in &self.nodes[n.children()] {
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
        let mut best = (n.first as usize, f32::NEG_INFINITY);
        for c in n.children() {
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
        while self.nodes[node].expanded() && self.nodes[node].terminal().is_none() {
            node = self.select(node);
            game.push(self.nodes[node].mv);
            path.push(node);
        }
        let ply = path.len() - 1;
        let wdl = match self.nodes[node].terminal() {
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
            n.w += v[0];
            n.l += v[2];
            v = flip(v);
        }
    }

    fn best_child(&self, node: usize) -> Option<usize> {
        let n = &self.nodes[node];
        n.children().max_by(|&a, &b| {
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

    /// The most visited root move's share of the root moves' visits.
    fn best_share(&self) -> f32 {
        let root = &self.nodes[ROOT];
        let total: u32 = self.nodes[root.children()].iter().map(|c| c.visits).sum();
        match self.best_child(ROOT) {
            Some(b) if total > 0 => self.nodes[b].visits as f32 / total as f32,
            _ => 0.0,
        }
    }

    /// Factor on the target time: a clear best move (a large share of the
    /// visits) gives less time, visits spread over several moves more.
    fn time_factor(&self) -> f32 {
        self.params.time_factor(self.best_share())
    }

    /// Whether to stop now, `elapsed` into the move: never past `hard`;
    /// `go movetime` uses exactly its time; with a single legal move there
    /// is nothing to think about; otherwise at `soft` × the time factor.
    fn time_up(&self, budget: &timeman::TimeBudget, elapsed: std::time::Duration) -> bool {
        if elapsed >= budget.hard {
            return true;
        }
        if budget.fixed {
            return elapsed >= budget.soft;
        }
        if self.nodes[ROOT].count == 1 {
            return true;
        }
        elapsed.as_secs_f32() >= budget.soft.as_secs_f32() * self.time_factor()
    }

    /// Arena fill in permille (UCI `hashfull`: the tree is what runs out).
    fn hashfull(&self) -> usize {
        self.nodes.len() * 1000 / self.max_nodes.max(1)
    }

    fn print_info(&self, start: Instant, show_wdl: bool) {
        let Some(best) = self.best_child(ROOT) else {
            return;
        };
        let wdl = self.nodes[best].mean_wdl();
        let playouts = self.playouts;
        let wdl_str = if show_wdl {
            wdl_string(wdl)
        } else {
            String::new()
        };
        let ms = start.elapsed().as_millis();
        let pv: Vec<String> = self.pv().iter().map(|m| m.to_uci()).collect();
        println!(
            "info depth {} seldepth {} multipv 1 score cp {}{wdl_str} nodes {playouts} nps {} hashfull {} time {ms} pv {}",
            self.depth(),
            self.seldepth,
            q_to_cp(wdl[0] - wdl[2]),
            playouts as u128 * 1000 / ms.max(1),
            self.hashfull(),
            pv.join(" ")
        );
    }
}

impl Mcts {
    /// Size the arena and the evaluation table for `Hash` = `mb` (only
    /// reallocates when it changed; a resize drops the saved tree).
    fn prepare(&mut self, mb: usize) {
        if self.tt.is_some() && mb == self.hash_mb {
            return;
        }
        let (max_nodes, tt_bytes) = budget(mb);
        self.nodes = Vec::with_capacity(max_nodes);
        self.max_nodes = max_nodes;
        self.saved = None;
        self.tt = Some(EvalTable::new(tt_bytes));
        self.hash_mb = mb;
    }

    /// Re-root the saved tree at `game` if `game` continues the saved game
    /// and the search can start from it; otherwise clear the arena (fresh
    /// tree). Returns how many playouts were kept.
    fn reuse_tree(&mut self, game: &Game, ctx: &SearchContext) -> u32 {
        let kept = self.find_reusable(game, ctx);
        match kept {
            Some(node) => {
                if node != ROOT {
                    compact_in_place(&mut self.nodes, node, None);
                    self.nodes[ROOT].mv = Move::NULL;
                    let children = self.nodes[ROOT].children();
                    let p = ctx.params;
                    retemper(&mut self.nodes[children], p.policy_temp, p.policy_temp_root);
                }
                self.nodes[ROOT].visits
            }
            None => {
                self.nodes.clear();
                0
            }
        }
    }

    /// The node of the saved tree that `game`'s position corresponds to,
    /// if the tree can be reused from there.
    fn find_reusable(&mut self, game: &Game, ctx: &SearchContext) -> Option<usize> {
        let saved = self.saved.take()?;
        if saved.params != *ctx.params
            || !ctx.limits.searchmoves.is_empty()
            || self.nodes.is_empty()
            || saved.game.start_position() != game.start_position()
            || !game.moves().starts_with(saved.game.moves())
        {
            return None;
        }
        let nodes = &self.nodes;
        let mut node = ROOT;
        for &m in &game.moves()[saved.game.moves().len()..] {
            node = nodes[node].children().find(|&c| nodes[c].mv == m)?;
        }
        // the new root must be an expanded, non-terminal position (a node
        // marked as a draw by repetition is not a draw at the root)
        let n = &nodes[node];
        if !n.expanded() || n.terminal().is_some() || n.count == 0 {
            return None;
        }
        Some(node)
    }
}

impl NetSearch for Mcts {
    fn new_game(&mut self) {
        self.saved = None;
        self.nodes.clear();
        // also called when the network changes: its evaluations are stale
        if let Some(tt) = &mut self.tt {
            tt.clear();
        }
    }

    fn search(&mut self, net: &mut PolicyNet, game: &Game, ctx: &SearchContext) -> SearchResult {
        self.prepare(ctx.common.hash_mb);
        let kept = self.reuse_tree(game, ctx);
        let max_nodes = self.max_nodes;
        let tt = self.tt.as_mut().unwrap();
        let (probes, hits) = (tt.probes, tt.hits);
        let (result, _) = search_tree(net, tt, &mut self.nodes, max_nodes, game, ctx, true);
        println!(
            "info string tree reuse: {kept} playouts kept; eval table: {} of {} lookups hit, {} permille full",
            tt.hits - hits,
            tt.probes - probes,
            tt.hashfull()
        );
        // searchmoves trims the root: don't reuse such a tree
        if ctx.limits.searchmoves.is_empty() {
            self.saved = Some(Saved {
                game: game.clone(),
                params: *ctx.params,
            });
        }
        result
    }
}

/// Run a search on a fresh tree with its own arena and table (sized by
/// `Hash`); returns the result and the number of playouts. `print = false`
/// suppresses the info lines (used by `bench`).
pub fn run(
    net: &mut PolicyNet,
    game: &Game,
    ctx: &SearchContext,
    print: bool,
) -> (SearchResult, u64) {
    let (max_nodes, tt_bytes) = budget(ctx.common.hash_mb);
    let mut tt = EvalTable::new(tt_bytes);
    let mut nodes = Vec::with_capacity(max_nodes);
    search_tree(net, &mut tt, &mut nodes, max_nodes, game, ctx, print)
}

/// Search from the tree in `nodes` (a re-rooted tree from the previous
/// move, or empty for a fresh tree). Returns the result and this search's
/// playouts; the tree stays in `nodes`.
fn search_tree(
    net: &mut PolicyNet,
    tt: &mut EvalTable,
    nodes: &mut Vec<Node>,
    max_nodes: usize,
    game: &Game,
    ctx: &SearchContext,
    print: bool,
) -> (SearchResult, u64) {
    let start = Instant::now();
    let limits = ctx.limits;
    let budget = timeman::budget_with(
        limits,
        game.position().side_to_move(),
        ctx.common.move_overhead_ms,
        &ctx.params.time_config(),
    );
    // node limits count this search's playouts, not the reused ones
    let max_playouts = limits.nodes.unwrap_or(if limits.depth.is_some() {
        DEPTH_PLAYOUT_CAP
    } else if budget.is_none() && !limits.infinite {
        DEFAULT_PLAYOUTS
    } else {
        u64::MAX
    });

    let fresh = nodes.is_empty();
    let mut tree = Tree {
        net,
        tt,
        params: *ctx.params,
        nodes,
        max_nodes,
        full: false,
        frozen: false,
        prunes: 0,
        seldepth: 0,
        depth_sum: 0,
        playouts: 0,
    };
    if fresh {
        tree.nodes.push(Node::new(Move::NULL, 1.0));
        tree.playout(game); // expands the root
    }
    if !limits.searchmoves.is_empty() {
        // searchmoves: keep only those root moves (always a fresh tree, so
        // they have no subtrees yet and can be moved around in the arena)
        let children = tree.nodes[ROOT].children();
        let first = children.start;
        let mut kept = 0;
        for c in children {
            if limits.searchmoves.contains(&tree.nodes[c].mv.to_uci()) {
                tree.nodes.swap(first + kept, c);
                kept += 1;
            }
        }
        if kept > 0 {
            tree.nodes[ROOT].count = kept as u8;
        }
    }

    let target_depth = limits.depth.map(u64::from);
    let (mut shown_depth, mut shown_seldepth, mut last_info) = (0, 0, 0u128);
    let mut shown_playouts = 0;
    while tree.playouts < max_playouts
        && tree.nodes[ROOT].count > 0
        && target_depth.is_none_or(|d| tree.depth() < d)
        && !ctx.stop.load(Ordering::Relaxed)
        && budget.is_none_or(|b| !tree.time_up(&b, start.elapsed()))
    {
        tree.playout(game);
        if tree.full {
            tree.prune();
        }
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
    if let (true, Some(b)) = (print, budget) {
        if !b.fixed {
            println!(
                "info string time: {} ms used, target {} ms x factor {:.2} (best move {:.0}% of visits), hard {} ms",
                start.elapsed().as_millis(),
                b.soft.as_millis(),
                tree.time_factor(),
                100.0 * tree.best_share(),
                b.hard.as_millis()
            );
        }
    }
    if print && tree.prunes > 0 {
        println!(
            "info string tree full (Hash {} MB): pruned {} times{}",
            ctx.common.hash_mb,
            tree.prunes,
            if tree.frozen {
                ", then searched on without growing"
            } else {
                ""
            }
        );
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
        return (none, playouts);
    };
    if print && tree.playouts != shown_playouts {
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
        (nodes[1].w, nodes[1].l) = (4.0 * w, 4.0 * l);
        // the root stores results for the side that moved into it
        nodes[0].visits = 5;
        (nodes[0].w, nodes[0].l) = (5.0 * l, 5.0 * w);
        nodes[0].first = 1;
        nodes[0].count = 3;
        nodes[0].flags = EXPANDED;
        Tree {
            net,
            // a small table and arena per test tree (leaked: tests only)
            tt: Box::leak(Box::new(EvalTable::new(1 << 16))),
            params,
            nodes: Box::leak(Box::new(nodes)),
            max_nodes: 1000,
            full: false,
            frozen: false,
            prunes: 0,
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
    /// expanded and once more for every visit of one of its children;
    /// children come after their parent.
    fn check_tree(nodes: &[Node]) {
        for (i, n) in nodes.iter().enumerate() {
            let c = n.children();
            assert!(c.end <= nodes.len(), "node {i} children out of range");
            assert!(n.count == 0 || c.start > i, "node {i} children before it");
            if n.expanded() && n.terminal().is_none() {
                // equal, or more after a collapsed node was expanded again
                let below: u32 = nodes[c].iter().map(|c| c.visits).sum();
                assert!(n.visits > below, "node {i}: {} vs {below}", n.visits);
            }
        }
    }

    /// Number of nodes in the subtree of `nodes[root]`.
    fn subtree_size(nodes: &[Node], root: usize) -> usize {
        1 + nodes[root]
            .children()
            .map(|c| subtree_size(nodes, c))
            .sum::<usize>()
    }

    /// The two subtrees hold the same nodes in the same shape.
    fn same_tree(a: &[Node], ra: usize, b: &[Node], rb: usize) {
        let (x, y) = (&a[ra], &b[rb]);
        assert_eq!((x.visits, x.count, x.flags), (y.visits, y.count, y.flags));
        assert_eq!((x.w, x.l, x.prior), (y.w, y.l, y.prior));
        for (ca, cb) in x.children().zip(y.children()) {
            assert!(a[ca].mv == b[cb].mv);
            same_tree(a, ca, b, cb);
        }
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
        let (common, stop) = (CommonOptions::default(), AtomicBool::new(false));
        m.search(net, game, &ctx_for(&limits, &common, &stop, params))
            .best
    }

    #[test]
    fn node_is_24_bytes_and_hash_is_the_whole_budget() {
        assert_eq!(std::mem::size_of::<Node>(), 24);
        for mb in [1, 16, 256] {
            let (nodes, tt) = budget(mb);
            assert!(nodes * 24 + tt <= mb << 20);
            assert!(nodes * 24 + tt + 24 > mb << 20);
        }
    }

    #[test]
    fn compaction_in_place_matches_a_copy() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let params = SearchParams::default();
        let mut m = Mcts::default();
        search_with(&mut m, &mut net, &Game::default(), 600, &params);
        let original = m.nodes.clone();
        check_tree(&original);
        // every node with a subtree, as the new root
        let roots: Vec<usize> = (1..original.len())
            .filter(|&i| original[i].count > 0)
            .collect();
        assert!(roots.len() > 20);
        for r in roots {
            let mut nodes = original.clone();
            compact_in_place(&mut nodes, r, None);
            assert_eq!(nodes.len(), subtree_size(&original, r));
            check_tree(&nodes);
            same_tree(&original, r, &nodes, ROOT);
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
        let old = m.nodes.clone();
        // the expected reply: the most visited grandchild under our move
        let ours = old[ROOT].children().find(|&c| old[c].mv == best).unwrap();
        let reply = old[ours].children().max_by_key(|&c| old[c].visits).unwrap();
        assert!(old[reply].visits > 1, "the reply should have a subtree");

        game.push(best);
        game.push(old[reply].mv);
        let limits = SearchLimits::default();
        let (common, stop) = (CommonOptions::default(), AtomicBool::new(false));
        let kept = m.reuse_tree(&game, &ctx_for(&limits, &common, &stop, &params));
        assert_eq!(kept, old[reply].visits);
        assert_eq!(m.nodes.len(), subtree_size(&old, reply));
        check_tree(&m.nodes);
        // same children in the same order, priors re-tempered to the root
        // temperature (still summing to 1)
        let root = m.nodes[ROOT];
        let moves = |n: &[Node], i: usize| n[i].children().map(|c| n[c].mv).collect::<Vec<_>>();
        assert!(moves(&m.nodes, ROOT) == moves(&old, reply));
        let sum: f32 = m.nodes[root.children()].iter().map(|c| c.prior).sum();
        assert!((sum - 1.0).abs() < 1e-4);
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
        let mut g = Game::default();
        g.push_uci("e2e4");

        // a different game (not a continuation)
        let mut m = Mcts::default();
        search_with(&mut m, &mut net, &g, 100, &params);
        let mut other = Game::default();
        other.push_uci("d2d4");
        assert_eq!(m.reuse_tree(&other, &ctx), 0);
        assert!(m.nodes.is_empty());

        // changed parameters
        search_with(&mut m, &mut net, &g, 100, &params);
        let changed = SearchParams {
            cpuct: params.cpuct + 0.5,
            ..params
        };
        assert_eq!(
            m.reuse_tree(&g, &ctx_for(&limits, &common, &stop, &changed)),
            0
        );

        // ucinewgame
        search_with(&mut m, &mut net, &g, 100, &params);
        m.new_game();
        assert_eq!(m.reuse_tree(&g, &ctx), 0);

        // same position again: the whole tree
        search_with(&mut m, &mut net, &g, 100, &params);
        let n = m.nodes.len();
        assert_eq!(m.reuse_tree(&g, &ctx), 100);
        assert_eq!(m.nodes.len(), n);
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
            check_tree(&m.nodes);
            // the root holds the kept visits plus exactly 150 new ones
            assert!(m.nodes[ROOT].visits >= 150);
            if m.nodes[ROOT].visits > 150 {
                reused += 1;
            }
            game.push(mv);
        }
        // the engine plays both sides here, so every search after the
        // first starts from the previous one's subtree
        assert!(reused >= 10, "reused {reused} times");
        assert!(m.nodes.capacity() <= m.max_nodes + 1);
    }

    #[test]
    fn pruning_collapses_the_least_visited_nodes_only() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let params = SearchParams::default();
        let mut m = Mcts::default();
        search_with(&mut m, &mut net, &Game::default(), 1500, &params);
        let original = m.nodes.clone();
        for t in [1, 3, 10] {
            let mut nodes = original.clone();
            compact_in_place(&mut nodes, ROOT, Some(t));
            check_tree(&nodes);
            assert!(nodes.len() < original.len());
            // freed = the children of every expanded node with ≤ t visits
            let freed: usize = original[1..]
                .iter()
                .filter(|n| n.count > 0 && n.visits <= t)
                .map(|n| n.count as usize)
                .sum();
            assert_eq!(original.len() - nodes.len(), freed);
            // statistics are kept; nodes above the threshold keep children
            assert_eq!(nodes[ROOT].visits, original[ROOT].visits);
            assert_eq!(nodes[ROOT].count, original[ROOT].count);
            for n in &nodes[1..] {
                if n.visits > t {
                    assert!(n.count > 0 || !n.expanded());
                } else {
                    assert_eq!(n.count, 0);
                }
            }
            let sum = |v: &[Node]| v.iter().map(|n| n.visits as u64).sum::<u64>();
            let kept_children: u64 = sum(&nodes[nodes[ROOT].children()]);
            assert_eq!(kept_children, sum(&original[original[ROOT].children()]));
        }
    }

    #[test]
    fn full_tree_prunes_and_keeps_searching() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let params = SearchParams::default();
        let mut m = Mcts::default();
        // Hash 1: room for ~1,000 playouts; ask for 6,000
        let limits = SearchLimits {
            nodes: Some(6000),
            ..Default::default()
        };
        let common = CommonOptions {
            hash_mb: 1,
            ..Default::default()
        };
        let stop = AtomicBool::new(false);
        let game = Game::default();
        let best = m
            .search(&mut net, &game, &ctx_for(&limits, &common, &stop, &params))
            .best;
        assert!(game.position().legal_moves().contains(&best));
        let (max_nodes, _) = budget(1);
        assert!(m.nodes.len() <= max_nodes);
        // every requested playout was done
        assert_eq!(m.nodes[ROOT].visits, 6000);
        check_tree(&m.nodes);
    }

    // ------------------------------------------------------- time management

    #[test]
    fn time_factor_follows_the_best_moves_share() {
        let p = SearchParams::default();
        // clear best move → less time; spread visits → more time
        assert!(p.time_factor(0.9) < p.time_factor(0.5));
        assert!(p.time_factor(0.5) < p.time_factor(0.2));
        assert!(p.time_factor(0.9) < 1.0 && p.time_factor(0.2) > 1.0);
        for share in [0.0, 0.5, 1.0] {
            let f = p.time_factor(share);
            assert!(f >= p.tm_min_mult && f <= p.tm_max_mult);
        }
    }

    fn timed_search(fen: &str, limits: SearchLimits) -> std::time::Duration {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let params = SearchParams::default();
        let (common, stop) = (CommonOptions::default(), AtomicBool::new(false));
        let game = Game::from_fen(fen).unwrap();
        let t = Instant::now();
        let r = Mcts::default().search(&mut net, &game, &ctx_for(&limits, &common, &stop, &params));
        assert!(!r.best.is_null());
        t.elapsed()
    }

    #[test]
    fn timed_search_stays_within_the_hard_limit() {
        let limits = SearchLimits {
            wtime: Some(3000),
            winc: Some(30),
            ..Default::default()
        };
        let b = timeman::budget_with(
            &limits,
            peazero_core::Color::White,
            CommonOptions::default().move_overhead_ms,
            &SearchParams::default().time_config(),
        )
        .unwrap();
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";
        let used = timed_search(fen, limits);
        assert!(
            used < b.hard + std::time::Duration::from_millis(50),
            "{used:?} vs {b:?}"
        );
    }

    #[test]
    fn single_legal_move_is_played_at_once() {
        // only Kxg2 is legal
        let limits = SearchLimits {
            wtime: Some(60_000),
            ..Default::default()
        };
        let used = timed_search("7k/8/8/8/8/8/6r1/7K w - - 0 1", limits);
        assert!(used < std::time::Duration::from_millis(100), "{used:?}");
    }
}
