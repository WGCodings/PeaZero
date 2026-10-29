//! Plain MCTS (AlphaZero-style PUCT) on top of the network.
//!
//! Each playout walks down the tree with PUCT, evaluates the new leaf with
//! the network once (policy softmax → priors of its children, W/D/L →
//! value) and backs the W/D/L up, flipping sides each ply. Checkmate,
//! stalemate and draws (repetition, 50-move rule, insufficient material)
//! get their exact value. The move played is chosen by a lower confidence
//! bound on Q among the well-visited root moves (`View::root_choice`).
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
//! Threads (UCI `Threads`): all threads search the same tree. Node fields
//! are atomics (W and L share one 64-bit word, so a backup is one
//! compare-and-swap per node); the calling thread is the main thread (time
//! management, info lines, stopping), the others are helpers with their own
//! network instance (shared weights). With more than one thread a selected
//! node gets its visit and a virtual loss (`VirtualLoss` losses) at once,
//! so other threads spread out instead of following the same line; the
//! backup replaces the virtual loss by the real result. A thread claims a
//! leaf before evaluating it; another thread reaching the same leaf takes
//! its virtual losses back and starts a new playout (a collision). Every
//! playout holds a shared lock on the arena; pruning takes it exclusively,
//! so nodes never move under a running playout. With one thread nothing
//! of this changes the search (no virtual loss, same results).
//!
//! Not (yet): proven-result propagation, smart pruning.

use std::ops::{Index, Range};
use std::sync::atomic::{
    AtomicBool, AtomicU16, AtomicU32, AtomicU64, AtomicU8, AtomicUsize, Ordering,
    Ordering::{Acquire, Relaxed, Release},
};
use std::sync::RwLock;
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
/// With a single legal move, search until it has this many visits (for a
/// real score in the info line), then play it.
const SINGLE_MOVE_VISITS: u32 = 8;
/// Share of `Hash` for the tree; the rest is the evaluation table.
const TREE_SHARE_PERCENT: usize = 75;
/// Most legal moves in a chess position (room an expansion may need).
const MAX_CHILDREN: usize = 218;

const ROOT: usize = 0;

/// `Node::flags` bits.
const EXPANDED: u8 = 1;
const TERMINAL_DRAW: u8 = 2;
const TERMINAL_LOSS: u8 = 4;
/// A thread is evaluating this leaf (claimed before expanding).
const EXPANDING: u8 = 8;

/// One tree node: 24 bytes, all fields atomic (shared by the threads).
#[derive(Default)]
#[repr(C)]
struct Node {
    /// Summed W (low 32 bits) and L (high 32 bits), f32 each, from the point
    /// of view of the side that played `mv` (D = visits − W − L: every
    /// playout adds a W/D/L summing to 1).
    wl: AtomicU64,
    /// f32 bits.
    prior: AtomicU32,
    /// Children are `first..first + count` in the arena; they always come
    /// after their parent.
    first: AtomicU32,
    visits: AtomicU32,
    /// Move that leads here from the parent (`Move::raw`).
    mv: AtomicU16,
    /// Number of children (at most 218 legal moves).
    count: AtomicU8,
    /// `EXPANDED`, `TERMINAL_DRAW`, `TERMINAL_LOSS`, `EXPANDING`.
    flags: AtomicU8,
}

const _: () = assert!(std::mem::size_of::<Node>() == 24);

/// A node's fields as plain values (copying nodes, tests).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct NodeData {
    mv: Move,
    count: u8,
    flags: u8,
    prior: f32,
    first: u32,
    visits: u32,
    w: f32,
    l: f32,
}

fn pack_wl(w: f32, l: f32) -> u64 {
    w.to_bits() as u64 | (l.to_bits() as u64) << 32
}

fn unpack_wl(x: u64) -> (f32, f32) {
    (f32::from_bits(x as u32), f32::from_bits((x >> 32) as u32))
}

impl NodeData {
    fn new(mv: Move, prior: f32) -> NodeData {
        NodeData {
            mv,
            prior,
            ..Default::default()
        }
    }

    #[cfg(test)]
    fn children(&self) -> Range<usize> {
        self.first as usize..self.first as usize + self.count as usize
    }

    #[cfg(test)]
    fn expanded(&self) -> bool {
        self.flags & EXPANDED != 0
    }

    #[cfg(test)]
    fn terminal(&self) -> bool {
        self.flags & (TERMINAL_DRAW | TERMINAL_LOSS) != 0
    }
}

impl Node {
    #[cfg(test)]
    fn from_data(d: NodeData) -> Node {
        let n = Node::default();
        n.store(d);
        n
    }

    fn load(&self) -> NodeData {
        let (w, l) = unpack_wl(self.wl.load(Relaxed));
        NodeData {
            mv: self.mv(),
            count: self.count(),
            flags: self.flags.load(Relaxed),
            prior: self.prior(),
            first: self.first.load(Relaxed),
            visits: self.visits(),
            w,
            l,
        }
    }

    fn store(&self, d: NodeData) {
        self.wl.store(pack_wl(d.w, d.l), Relaxed);
        self.prior.store(d.prior.to_bits(), Relaxed);
        self.first.store(d.first, Relaxed);
        self.visits.store(d.visits, Relaxed);
        self.mv.store(d.mv.raw(), Relaxed);
        self.count.store(d.count, Relaxed);
        self.flags.store(d.flags, Relaxed);
    }

    fn mv(&self) -> Move {
        Move::from_raw(self.mv.load(Relaxed))
    }

    fn prior(&self) -> f32 {
        f32::from_bits(self.prior.load(Relaxed))
    }

    fn set_prior(&self, p: f32) {
        self.prior.store(p.to_bits(), Relaxed);
    }

    fn visits(&self) -> u32 {
        self.visits.load(Relaxed)
    }

    fn count(&self) -> u8 {
        self.count.load(Relaxed)
    }

    fn children(&self) -> Range<usize> {
        let first = self.first.load(Relaxed) as usize;
        first..first + self.count() as usize
    }

    /// Acquire: a thread that sees `EXPANDED` also sees the children.
    fn expanded(&self) -> bool {
        self.flags.load(Acquire) & EXPANDED != 0
    }

    /// Exact W/D/L for the side to move here, if the game is over.
    fn terminal(&self) -> Option<[f32; 3]> {
        let f = self.flags.load(Relaxed);
        if f & TERMINAL_LOSS != 0 {
            Some([0.0, 0.0, 1.0])
        } else if f & TERMINAL_DRAW != 0 {
            Some([0.0, 1.0, 0.0])
        } else {
            None
        }
    }

    fn wl(&self) -> (f32, f32) {
        unpack_wl(self.wl.load(Relaxed))
    }

    fn add_wl(&self, dw: f32, dl: f32) {
        let _ = self.wl.fetch_update(Relaxed, Relaxed, |x| {
            let (w, l) = unpack_wl(x);
            Some(pack_wl(w + dw, l + dl))
        });
    }

    /// Mean `W - L` for the side that played `mv`.
    fn q(&self) -> f32 {
        let n = self.visits();
        if n == 0 {
            0.0
        } else {
            let (w, l) = self.wl();
            (w - l) / n as f32
        }
    }

    /// Mean W/D/L for the side that played `mv`.
    fn mean_wdl(&self) -> [f32; 3] {
        let n = self.visits();
        if n == 0 {
            return [0.0, 1.0, 0.0];
        }
        let (w, l) = self.wl();
        let (w, l) = (w / n as f32, l / n as f32);
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

/// The tree: a fixed array of nodes, the first `used` in use. Expansions
/// reserve their children with an atomic counter.
#[derive(Default)]
struct Arena {
    nodes: Box<[Node]>,
    used: AtomicUsize,
}

impl Index<usize> for Arena {
    type Output = Node;

    fn index(&self, i: usize) -> &Node {
        &self.nodes[i]
    }
}

impl Arena {
    fn new(capacity: usize) -> Arena {
        Arena {
            nodes: (0..capacity).map(|_| Node::default()).collect(),
            used: AtomicUsize::new(0),
        }
    }

    fn len(&self) -> usize {
        self.used.load(Relaxed)
    }

    fn capacity(&self) -> usize {
        self.nodes.len()
    }

    fn clear(&mut self) {
        *self.used.get_mut() = 0;
    }

    /// A fresh tree: just the root.
    fn reset_root(&mut self) {
        self.nodes[ROOT].store(NodeData::new(Move::NULL, 1.0));
        *self.used.get_mut() = 1;
    }

    /// Room for `n` more nodes: the index of the first, or None if full.
    fn reserve(&self, n: usize) -> Option<usize> {
        let mut cur = self.used.load(Relaxed);
        loop {
            if cur + n > self.capacity() {
                return None;
            }
            match self
                .used
                .compare_exchange_weak(cur, cur + n, Relaxed, Relaxed)
            {
                Ok(_) => return Some(cur),
                Err(now) => cur = now,
            }
        }
    }

    #[cfg(test)]
    fn snapshot(&self) -> Vec<NodeData> {
        self.nodes[..self.len()].iter().map(Node::load).collect()
    }

    #[cfg(test)]
    fn from_data(data: &[NodeData], capacity: usize) -> Arena {
        let a = Arena::new(capacity.max(data.len()));
        for (n, &d) in a.nodes.iter().zip(data) {
            n.store(d);
        }
        a.used.store(data.len(), Relaxed);
        a
    }
}

/// Keep only the subtree of `a[root]`, moved to the front of the arena
/// with `root` at index 0, in place. Children always come after their
/// parent in the arena, so one forward pass marks the subtree (a bitset)
/// and a second one slides the kept nodes down in their original order:
/// a node's new index (its rank among the kept nodes) is never above its
/// old one, so nothing is overwritten before it is read.
///
/// `collapse = Some(t)`: also turn every expanded node (other than the
/// root) with at most `t` visits back into a leaf, dropping its subtree;
/// it keeps its own visits and W/L.
fn compact_in_place(a: &mut Arena, root: usize, collapse: Option<u32>) {
    let n = a.len();
    let collapsed =
        |i: usize, node: &NodeData| i != root && collapse.is_some_and(|t| node.visits <= t);
    let mut kept = vec![0u64; n.div_ceil(64)];
    let set = |kept: &mut Vec<u64>, i: usize| kept[i / 64] |= 1 << (i % 64);
    let is = |kept: &Vec<u64>, i: usize| kept[i / 64] >> (i % 64) & 1 != 0;
    set(&mut kept, root);
    for i in root..n {
        let node = a[i].load();
        if is(&kept, i) && !collapsed(i, &node) {
            for c in node.first as usize..node.first as usize + node.count as usize {
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
            let mut node = a[i].load();
            if node.count > 0 && collapsed(i, &node) {
                node.count = 0;
                node.first = 0;
                node.flags &= !(EXPANDED | EXPANDING);
            } else if node.count > 0 {
                node.first = rank(node.first as usize);
            }
            a[rank(i) as usize].store(node);
        }
    }
    *a.used.get_mut() = total as usize;
}

/// Root children got their priors with the tree temperature `t_tree`;
/// convert them to the root temperature: p ∝ exp(logit / T), so
/// p_root ∝ p_tree^(t_tree / t_root).
fn retemper(children: &[Node], t_tree: f32, t_root: f32) {
    if (t_tree - t_root).abs() < 1e-6 {
        return;
    }
    let k = t_tree / t_root.max(1e-3);
    let powered: Vec<f32> = children.iter().map(|c| c.prior().powf(k)).collect();
    let sum: f32 = powered.iter().sum();
    if sum > 0.0 {
        for (c, p) in children.iter().zip(powered) {
            c.set_prior(p / sum);
        }
    }
}

/// Read-only questions about a tree (selection, move choice, PV, time).
struct View<'a> {
    a: &'a Arena,
    params: SearchParams,
}

impl View<'_> {
    /// (FPU, gini) of `node`. FPU: value of an unvisited child, from the
    /// view of the side to move at `node`: the parent's own Q minus the FPU
    /// reduction times √(policy mass of the children already visited).
    /// gini = 1 − Σ prior² of the children.
    fn fpu_and_gini(&self, node: usize) -> (f32, f32) {
        let n = &self.a[node];
        let (mut visited, mut squares) = (0.0f32, 0.0f32);
        for c in n.children() {
            let c = &self.a[c];
            let p = c.prior();
            squares += p * p;
            if c.visits() > 0 {
                visited += p;
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
        let n = &self.a[node];
        let (fpu, gini) = self.fpu_and_gini(node);
        let u = self.params.explore_coef(n.visits(), gini);
        let children = n.children();
        let mut best = (children.start, f32::NEG_INFINITY);
        for c in children {
            let child = &self.a[c];
            let visits = child.visits();
            let q = if visits == 0 { fpu } else { child.q() };
            let score = q + u * child.prior() / (1.0 + visits as f32);
            if score > best.1 {
                best = (c, score);
            }
        }
        best.0
    }

    fn best_child(&self, node: usize) -> Option<usize> {
        self.a[node].children().max_by(|&a, &b| {
            let (a, b) = (&self.a[a], &self.a[b]);
            a.visits()
                .cmp(&b.visits())
                .then(a.prior().total_cmp(&b.prior()))
        })
    }

    /// The move to play: among the root moves with at least
    /// `BestMoveMinShare` of the most visited move's visits, the highest
    /// lower confidence bound Q − `BestMoveLcbC` / √visits (ties: more
    /// visits). A move with fewer visits is chosen only if its Q is
    /// clearly better.
    fn root_choice(&self) -> Option<usize> {
        let most = self.best_child(ROOT)?;
        let max_visits = self.a[most].visits();
        if max_visits == 0 {
            return Some(most);
        }
        let p = &self.params;
        let min_visits = ((max_visits as f32 * p.best_move_min_share).ceil() as u32).max(1);
        let lcb = |c: usize| {
            let n = &self.a[c];
            n.q() - p.best_move_lcb_c / (n.visits() as f32).sqrt()
        };
        self.a[ROOT]
            .children()
            .filter(|&c| self.a[c].visits() >= min_visits)
            .max_by(|&a, &b| {
                lcb(a)
                    .total_cmp(&lcb(b))
                    .then(self.a[a].visits().cmp(&self.a[b].visits()))
            })
    }

    /// The principal variation: the chosen root move, then the most
    /// visited move at each step.
    fn pv(&self) -> Vec<Move> {
        let mut pv = Vec::new();
        let mut next = self.root_choice();
        while let Some(c) = next {
            if self.a[c].visits() == 0 {
                break;
            }
            pv.push(self.a[c].mv());
            next = self.best_child(c);
        }
        pv
    }

    /// The most visited root move's share of the root moves' visits.
    fn best_share(&self) -> f32 {
        let total: u32 = self.a[ROOT].children().map(|c| self.a[c].visits()).sum();
        match self.best_child(ROOT) {
            Some(b) if total > 0 => self.a[b].visits() as f32 / total as f32,
            _ => 0.0,
        }
    }

    /// |Q(best root move) − Q(last node of its PV)|, both from the root
    /// mover's view: how much the evaluation changes along the main line.
    fn pv_diff(&self) -> f32 {
        let Some(best) = self.root_choice() else {
            return 0.0;
        };
        let q_best = self.a[best].q();
        let (mut node, mut depth, mut q_end) = (best, 1, q_best);
        while let Some(c) = self.best_child(node) {
            if self.a[c].visits() == 0 {
                break;
            }
            node = c;
            depth += 1;
            // a node's Q is for the side that moved into it
            q_end = if depth % 2 == 1 {
                self.a[c].q()
            } else {
                -self.a[c].q()
            };
        }
        (q_best - q_end).abs()
    }

    /// Factor on the target time (visit component, PV component, clamped
    /// product): a clear best move (a large share of the visits) gives less
    /// time, visits spread over several moves more; an evaluation that
    /// changes along the PV gives more.
    fn time_factor(&self) -> (f32, f32, f32) {
        self.params.time_factor(self.best_share(), self.pv_diff())
    }

    /// Whether to stop now, `elapsed` into the move: never past `hard`;
    /// `go movetime` uses exactly its time; a single legal move is played
    /// as soon as it has a few visits (for a real score in the info line);
    /// otherwise at `soft` × the time factor.
    fn time_up(&self, budget: &timeman::TimeBudget, elapsed: std::time::Duration) -> bool {
        if elapsed >= budget.hard {
            return true;
        }
        if budget.fixed {
            return elapsed >= budget.soft;
        }
        let root = &self.a[ROOT];
        if root.count() == 1 {
            return self.a[root.children().start].visits() >= SINGLE_MOVE_VISITS;
        }
        elapsed.as_secs_f32() >= budget.soft.as_secs_f32() * self.time_factor().2
    }

    /// Arena fill in permille (UCI `hashfull`: the tree is what runs out).
    fn hashfull(&self) -> usize {
        self.a.len() * 1000 / self.a.capacity().max(1)
    }
}

/// State shared by the threads of one search.
struct Shared<'a> {
    arena: RwLock<Arena>,
    tt: &'a EvalTable,
    params: SearchParams,
    /// Virtual loss per in-flight visit (0 with one thread).
    virtual_loss: f32,
    max_playouts: u64,
    /// Set by the main thread: the helpers stop.
    stop: AtomicBool,
    playouts: AtomicU64,
    depth_sum: AtomicU64,
    seldepth: AtomicUsize,
    collisions: AtomicU64,
    prunes: AtomicU32,
    /// Pruning can't make room (a tiny `Hash`): search on without growing.
    frozen: AtomicBool,
}

/// How a playout ended.
#[derive(Debug, PartialEq, Eq)]
enum Playout {
    Done,
    /// The arena had no room for an expansion: prune before going on.
    Full,
    /// Another thread is evaluating the same leaf: nothing was changed.
    Collision,
}

impl Shared<'_> {
    fn view<'g>(&self, a: &'g Arena) -> View<'g> {
        View {
            a,
            params: self.params,
        }
    }

    /// Average depth of this search's playouts (rounded down): the
    /// `depth` of MCTS.
    fn depth(&self) -> u64 {
        self.depth_sum.load(Relaxed) / self.playouts.load(Relaxed).max(1)
    }

    /// One playout: select down to a leaf, evaluate it, back up.
    fn playout(&self, net: &mut PolicyNet, root: &Game, path: &mut Vec<usize>) -> Playout {
        let guard = self.arena.read().unwrap();
        let a: &Arena = &guard;
        let view = self.view(a);
        let vl = self.virtual_loss;
        path.clear();
        path.push(ROOT);
        let mut game = root.clone();
        let mut node = ROOT;
        if vl > 0.0 {
            a[ROOT].visits.fetch_add(1, Relaxed);
        }
        while a[node].expanded() && a[node].terminal().is_none() {
            node = view.select(node);
            if vl > 0.0 {
                // count the visit now, as a loss for the side choosing it
                a[node].visits.fetch_add(1, Relaxed);
                a[node].add_wl(0.0, vl);
            }
            game.push(a[node].mv());
            path.push(node);
        }
        let ply = path.len() - 1;
        let (wdl, full) = match a[node].terminal() {
            Some(t) => (t, false),
            None => {
                let before = a[node].flags.fetch_or(EXPANDING, Ordering::AcqRel);
                if before & (EXPANDING | EXPANDED | TERMINAL_DRAW | TERMINAL_LOSS) != 0 {
                    // another thread got here first: undo and try again
                    self.undo_virtual_loss(a, path);
                    self.collisions.fetch_add(1, Relaxed);
                    return Playout::Collision;
                }
                self.evaluate(a, net, node, &game, ply)
            }
        };
        self.seldepth.fetch_max(ply, Relaxed);
        self.depth_sum.fetch_add(ply as u64, Relaxed);
        self.playouts.fetch_add(1, Relaxed);

        // each node stores results for the side that moved into it
        let mut v = flip(wdl);
        for &i in path.iter().rev() {
            let n = &a[i];
            if vl > 0.0 {
                // the visit was counted on the way down; replace the
                // virtual loss by the result
                n.add_wl(v[0], v[2] - if i == ROOT { 0.0 } else { vl });
            } else {
                n.visits.fetch_add(1, Relaxed);
                n.add_wl(v[0], v[2]);
            }
            v = flip(v);
        }
        if full {
            Playout::Full
        } else {
            Playout::Done
        }
    }

    fn undo_virtual_loss(&self, a: &Arena, path: &[usize]) {
        let vl = self.virtual_loss;
        if vl > 0.0 {
            a[ROOT].visits.fetch_sub(1, Relaxed);
            for &i in &path[1..] {
                a[i].visits.fetch_sub(1, Relaxed);
                a[i].add_wl(0.0, -vl);
            }
        }
    }

    /// Evaluate the claimed leaf `node` (`game`, `ply` plies below the
    /// root): exact result if the game is over, otherwise expand with the
    /// network (through the evaluation table). Returns W/D/L for its side
    /// to move and whether the arena was too full to expand (the node then
    /// stays an unexpanded leaf; its value is still backed up).
    fn evaluate(
        &self,
        a: &Arena,
        net: &mut PolicyNet,
        node: usize,
        game: &Game,
        ply: usize,
    ) -> ([f32; 3], bool) {
        let n = &a[node];
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
            n.flags.fetch_or(t, Relaxed);
            return (n.terminal().unwrap(), false);
        }

        let eval = self.tt.evaluate(net, game);
        let Some(first) = a.reserve(eval.moves.len()) else {
            // no room: release the claim so it can be expanded later; prune
            // unless pruning already couldn't help (frozen)
            n.flags.fetch_and(!EXPANDING, Relaxed);
            return (eval.wdl, !self.frozen.load(Relaxed));
        };
        let temp = if node == ROOT {
            self.params.policy_temp_root
        } else {
            self.params.policy_temp
        };
        for (i, (m, prior)) in priors(&eval.moves, temp).into_iter().enumerate() {
            a[first + i].store(NodeData::new(m, prior));
        }
        n.first.store(first as u32, Relaxed);
        n.count.store(eval.moves.len() as u8, Relaxed);
        // Release: the children are visible before EXPANDED is
        n.flags.fetch_or(EXPANDED, Release);
        (eval.wdl, false)
    }

    /// Make room in a full arena (exclusively: no playout runs meanwhile):
    /// collapse the least-visited expanded nodes (see the module
    /// documentation) until at most half the arena is used.
    fn prune(&self) {
        let mut guard = self.arena.write().unwrap();
        let a: &mut Arena = &mut guard;
        let before = a.len();
        if before + MAX_CHILDREN <= a.capacity() {
            return; // another thread pruned already
        }
        let target = a.capacity() / 2;
        if before > target {
            // collapsing every expanded node with ≤ t visits frees exactly
            // the children of those nodes (visits never grow downwards)
            let mut cand: Vec<(u32, usize)> = (1..before)
                .map(|i| a[i].load())
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
                compact_in_place(a, ROOT, threshold);
                self.prunes.fetch_add(1, Relaxed);
            }
        }
        // can't fit even the largest expansion: go on without growing
        if a.len() + MAX_CHILDREN > a.capacity() {
            self.frozen.store(true, Relaxed);
        }
    }

    /// A helper thread: playouts until the main thread says stop.
    fn helper(&self, net: &mut PolicyNet, root: &Game) {
        let mut path = Vec::new();
        while !self.stop.load(Relaxed) && self.playouts.load(Relaxed) < self.max_playouts {
            match self.playout(net, root, &mut path) {
                Playout::Full => self.prune(),
                Playout::Collision => std::thread::yield_now(),
                Playout::Done => {}
            }
        }
    }

    fn print_info(&self, start: Instant, show_wdl: bool) {
        let guard = self.arena.read().unwrap();
        let view = self.view(&guard);
        let Some(best) = view.root_choice() else {
            return;
        };
        let wdl = guard[best].mean_wdl();
        let playouts = self.playouts.load(Relaxed);
        let wdl_str = if show_wdl {
            wdl_string(wdl)
        } else {
            String::new()
        };
        let ms = start.elapsed().as_millis();
        let pv: Vec<String> = view.pv().iter().map(|m| m.to_uci()).collect();
        println!(
            "info depth {} seldepth {} multipv 1 score cp {}{wdl_str} nodes {playouts} nps {} hashfull {} time {ms} pv {}",
            self.depth(),
            self.seldepth.load(Relaxed),
            q_to_cp(wdl[0] - wdl[2]),
            playouts as u128 * 1000 / ms.max(1),
            view.hashfull(),
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
        self.arena = Arena::new(max_nodes);
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
                let a = &mut self.arena;
                if node != ROOT {
                    compact_in_place(a, node, None);
                    a[ROOT].mv.store(Move::NULL.raw(), Relaxed);
                    let p = ctx.params;
                    retemper(
                        &a.nodes[a[ROOT].children()],
                        p.policy_temp,
                        p.policy_temp_root,
                    );
                }
                a[ROOT].visits()
            }
            None => {
                self.arena.clear();
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
            || self.arena.len() == 0
            || saved.game.start_position() != game.start_position()
            || !game.moves().starts_with(saved.game.moves())
        {
            return None;
        }
        let a = &self.arena;
        let mut node = ROOT;
        for &m in &game.moves()[saved.game.moves().len()..] {
            node = a[node].children().find(|&c| a[c].mv() == m)?;
        }
        // the new root must be an expanded, non-terminal position (a node
        // marked as a draw by repetition is not a draw at the root)
        let n = &a[node];
        if !n.expanded() || n.terminal().is_some() || n.count() == 0 {
            return None;
        }
        Some(node)
    }

    /// One network instance per helper thread (`threads` - 1 in all),
    /// forked from `net` (shared weights).
    fn helpers(&mut self, net: &PolicyNet, threads: usize) -> &mut [PolicyNet] {
        let n = threads.max(1) - 1;
        self.helper_nets.truncate(n);
        while self.helper_nets.len() < n {
            self.helper_nets.push(net.fork());
        }
        &mut self.helper_nets
    }
}

/// The MCTS search mode; keeps the arena (with the last tree, for reuse),
/// the evaluation table and the helper threads' networks between moves.
#[derive(Default)]
pub struct Mcts {
    /// The arena: allocated once with room for `max_nodes`.
    arena: Arena,
    max_nodes: usize,
    /// What the tree in `arena` was built for (None: nothing reusable).
    saved: Option<Saved>,
    /// Created on the first search, sized by `Hash`.
    tt: Option<EvalTable>,
    hash_mb: usize,
    /// Network instances of the helper threads.
    helper_nets: Vec<PolicyNet>,
}

/// The game and parameters the saved tree was searched with.
struct Saved {
    game: Game,
    params: SearchParams,
}

impl NetSearch for Mcts {
    fn new_game(&mut self) {
        self.saved = None;
        self.arena.clear();
        // also called when the network changes: its evaluations are stale,
        // and the helpers' networks are re-forked from the new one
        if let Some(tt) = &mut self.tt {
            tt.clear();
        }
        self.helper_nets.clear();
    }

    fn search(&mut self, net: &mut PolicyNet, game: &Game, ctx: &SearchContext) -> SearchResult {
        self.prepare(ctx.common.hash_mb);
        let kept = self.reuse_tree(game, ctx);
        let threads = ctx.common.threads;
        self.helpers(net, threads);
        let tt = self.tt.as_ref().unwrap();
        let (probes, hits) = (tt.probes(), tt.hits());
        let (result, _) = search_tree(
            net,
            &mut self.helper_nets,
            tt,
            &mut self.arena,
            game,
            ctx,
            true,
        );
        println!(
            "info string tree reuse: {kept} playouts kept; eval table: {} of {} lookups hit, {} permille full",
            tt.hits() - hits,
            tt.probes() - probes,
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
/// `Hash`, `Threads` threads); returns the result and the number of
/// playouts. `print = false` suppresses the info lines (used by `bench`).
pub fn run(
    net: &mut PolicyNet,
    game: &Game,
    ctx: &SearchContext,
    print: bool,
) -> (SearchResult, u64) {
    let (max_nodes, tt_bytes) = budget(ctx.common.hash_mb);
    let tt = EvalTable::new(tt_bytes);
    let mut arena = Arena::new(max_nodes);
    let mut helpers: Vec<PolicyNet> = (1..ctx.common.threads.max(1)).map(|_| net.fork()).collect();
    search_tree(net, &mut helpers, &tt, &mut arena, game, ctx, print)
}

/// Search from the tree in `arena` (a re-rooted tree from the previous
/// move, or empty for a fresh tree) with the calling thread plus one
/// helper thread per network in `helpers`. Returns the result and this
/// search's playouts; the tree stays in `arena`.
fn search_tree(
    net: &mut PolicyNet,
    helpers: &mut [PolicyNet],
    tt: &EvalTable,
    arena: &mut Arena,
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

    let fresh = arena.len() == 0;
    if fresh {
        arena.reset_root();
    }
    let shared = Shared {
        arena: RwLock::new(std::mem::take(arena)),
        tt,
        params: *ctx.params,
        virtual_loss: if helpers.is_empty() {
            0.0
        } else {
            ctx.params.virtual_loss
        },
        max_playouts,
        stop: AtomicBool::new(false),
        playouts: AtomicU64::new(0),
        depth_sum: AtomicU64::new(0),
        seldepth: AtomicUsize::new(0),
        collisions: AtomicU64::new(0),
        prunes: AtomicU32::new(0),
        frozen: AtomicBool::new(false),
    };
    let mut path = Vec::new();
    if fresh {
        shared.playout(net, game, &mut path); // expands the root
    }
    if !limits.searchmoves.is_empty() {
        // searchmoves: keep only those root moves (always a fresh tree, so
        // they have no subtrees yet and can be moved around in the arena;
        // no helper thread runs yet)
        let a = shared.arena.read().unwrap();
        let children = a[ROOT].children();
        let first = children.start;
        let mut kept = 0;
        for c in children {
            if limits.searchmoves.contains(&a[c].mv().to_uci()) {
                let (x, y) = (a[first + kept].load(), a[c].load());
                a[first + kept].store(y);
                a[c].store(x);
                kept += 1;
            }
        }
        if kept > 0 {
            a[ROOT].count.store(kept as u8, Relaxed);
        }
    }

    let target_depth = limits.depth.map(u64::from);
    std::thread::scope(|s| {
        for h in helpers.iter_mut() {
            let shared = &shared;
            s.spawn(move || shared.helper(h, game));
        }
        let (mut shown_depth, mut shown_seldepth, mut last_info) = (0, 0, 0u128);
        loop {
            let go_on = {
                let a = shared.arena.read().unwrap();
                let view = shared.view(&a);
                shared.playouts.load(Relaxed) < max_playouts
                    && a[ROOT].count() > 0
                    && target_depth.is_none_or(|d| shared.depth() < d)
                    && !ctx.stop.load(Relaxed)
                    && budget.is_none_or(|b| !view.time_up(&b, start.elapsed()))
            };
            if !go_on {
                break;
            }
            match shared.playout(net, game, &mut path) {
                Playout::Full => shared.prune(),
                Playout::Collision => std::thread::yield_now(),
                Playout::Done => {}
            }
            if print {
                let ms = start.elapsed().as_millis();
                let (depth, seldepth) = (shared.depth(), shared.seldepth.load(Relaxed));
                if depth > shown_depth || seldepth > shown_seldepth || ms - last_info >= INFO_MS {
                    shown_depth = depth;
                    shown_seldepth = seldepth;
                    last_info = ms;
                    shared.print_info(start, ctx.show_wdl);
                }
            }
        }
        shared.stop.store(true, Relaxed);
    });

    if print {
        let a = shared.arena.read().unwrap();
        let view = shared.view(&a);
        if let Some(b) = budget.filter(|b| !b.fixed) {
            let (visits, pv, factor) = view.time_factor();
            println!(
                "info string time: {} ms used, target {} ms x factor {factor:.2} (visits {visits:.2}: best move {:.0}% of visits; pv {pv:.2}: pv diff {:.3}), hard {} ms",
                start.elapsed().as_millis(),
                b.soft.as_millis(),
                100.0 * view.best_share(),
                view.pv_diff(),
                b.hard.as_millis()
            );
        }
        let prunes = shared.prunes.load(Relaxed);
        if prunes > 0 {
            println!(
                "info string tree full (Hash {} MB): pruned {prunes} times{}",
                ctx.common.hash_mb,
                if shared.frozen.load(Relaxed) {
                    ", then searched on without growing"
                } else {
                    ""
                }
            );
        }
        if !helpers.is_empty() {
            println!(
                "info string threads: {}, collisions {}",
                helpers.len() + 1,
                shared.collisions.load(Relaxed)
            );
        }
    }

    let playouts = shared.playouts.load(Relaxed);
    let choice = {
        let a = shared.arena.read().unwrap();
        shared.view(&a).root_choice()
    };
    let Some(best) = choice else {
        if print {
            println!("info depth 0 score cp 0");
        }
        *arena = shared.arena.into_inner().unwrap();
        let none = SearchResult {
            best: Move::NULL,
            ponder: None,
        };
        return (none, playouts);
    };
    // always a final info line (GUIs take the score from it), also when
    // no new playout was needed (a reused tree)
    if print {
        {
            let a = shared.arena.read().unwrap();
            if let Some(most) = shared.view(&a).best_child(ROOT).filter(|&m| m != best) {
                let (b, m) = (&a[best], &a[most]);
                println!(
                    "info string move choice: {} (Q {:+.3}, {} visits) over the most visited {} (Q {:+.3}, {} visits)",
                    b.mv(),
                    b.q(),
                    b.visits(),
                    m.mv(),
                    m.q(),
                    m.visits()
                );
            }
        }
        shared.print_info(start, ctx.show_wdl);
    }
    *arena = shared.arena.into_inner().unwrap();
    let view = View {
        a: arena,
        params: *ctx.params,
    };
    let ponder = view.pv().get(1).copied();
    let result = SearchResult {
        best: arena[best].mv(),
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

    /// A view of a tree built from `nodes` (leaked: tests only).
    fn view_of(nodes: &[NodeData], params: SearchParams) -> View<'static> {
        View {
            a: Box::leak(Box::new(Arena::from_data(nodes, 1000))),
            params,
        }
    }

    /// A root with three children (priors 0.5 / 0.3 / 0.2); only the first
    /// is visited (4 times, mean W-L `q` for the side to move at the root).
    fn small_tree(q: f32, params: SearchParams) -> View<'static> {
        let mut nodes = vec![NodeData::new(Move::NULL, 1.0)];
        for p in [0.5, 0.3, 0.2] {
            nodes.push(NodeData::new(Move::NULL, p));
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
        view_of(&nodes, params)
    }

    #[test]
    fn fpu_follows_the_parent_value() {
        // weight 0, reduction 0: unvisited moves are a draw, whatever the
        // parent's value (the original search)
        let d = SearchParams {
            fpu_parent_weight_root: 0.0,
            fpu_reduction_root: 0.0,
            ..Default::default()
        };
        assert_eq!(small_tree(-0.8, d).fpu(ROOT), 0.0);
        assert_eq!(small_tree(0.8, d).fpu(ROOT), 0.0);
        // weight 1, reduction 1: the parent's Q minus √(visited prior mass)
        let p = SearchParams {
            fpu_parent_weight_root: 1.0,
            fpu_reduction_root: 1.0,
            ..Default::default()
        };
        let r = 0.5f32.sqrt();
        let lost = small_tree(-0.8, p).fpu(ROOT);
        assert!((lost - (-0.8 - r)).abs() < 1e-5, "{lost}");
        let won = small_tree(0.8, p).fpu(ROOT);
        assert!((won - (0.8 - r)).abs() < 1e-5, "{won}");
        // half weight, no reduction
        let h = SearchParams {
            fpu_parent_weight_root: 0.5,
            fpu_reduction_root: 0.0,
            ..Default::default()
        };
        assert!((small_tree(0.3, h).fpu(ROOT) - 0.15).abs() < 1e-5);
    }

    #[test]
    fn fpu_changes_which_move_is_explored() {
        // FPU = draw (the original search): in a lost position every
        // unvisited move looks better than the visited one
        let d = SearchParams {
            fpu_parent_weight_root: 0.0,
            fpu_reduction_root: 0.0,
            ..Default::default()
        };
        assert_eq!(small_tree(-0.8, d).select(ROOT), 2);
        // parent-Q FPU with a huge reduction: stay on the visited move
        let p = SearchParams {
            fpu_reduction_root: 3.0,
            ..Default::default()
        };
        assert_eq!(small_tree(-0.8, p).select(ROOT), 1);
        assert_eq!(small_tree(0.8, p).select(ROOT), 1);
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
    fn check_tree(nodes: &[NodeData]) {
        for (i, n) in nodes.iter().enumerate() {
            let c = n.children();
            assert!(c.end <= nodes.len(), "node {i} children out of range");
            assert!(n.count == 0 || c.start > i, "node {i} children before it");
            if n.expanded() && !n.terminal() {
                // equal, or more after a collapsed node was expanded again
                let below: u32 = nodes[c].iter().map(|c| c.visits).sum();
                assert!(n.visits > below, "node {i}: {} vs {below}", n.visits);
            }
            // no virtual loss or claim left behind
            assert!(
                n.flags & EXPANDING == 0 || n.expanded() || n.terminal(),
                "node {i}"
            );
        }
    }

    /// Number of nodes in the subtree of `nodes[root]`.
    fn subtree_size(nodes: &[NodeData], root: usize) -> usize {
        1 + nodes[root]
            .children()
            .map(|c| subtree_size(nodes, c))
            .sum::<usize>()
    }

    /// The two subtrees hold the same nodes in the same shape.
    fn same_tree(a: &[NodeData], ra: usize, b: &[NodeData], rb: usize) {
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

    fn view(m: &Mcts) -> View<'_> {
        View {
            a: &m.arena,
            params: SearchParams::default(),
        }
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
        let original = m.arena.snapshot();
        check_tree(&original);
        // every node with a subtree, as the new root
        let roots: Vec<usize> = (1..original.len())
            .filter(|&i| original[i].count > 0)
            .collect();
        assert!(roots.len() > 20);
        for r in roots {
            let mut a = Arena::from_data(&original, original.len());
            compact_in_place(&mut a, r, None);
            let nodes = a.snapshot();
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
        let old = m.arena.snapshot();
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
        let new = m.arena.snapshot();
        assert_eq!(new.len(), subtree_size(&old, reply));
        check_tree(&new);
        // same children in the same order, priors re-tempered to the root
        // temperature (still summing to 1)
        let moves = |n: &[NodeData], i: usize| n[i].children().map(|c| n[c].mv).collect::<Vec<_>>();
        assert!(moves(&new, ROOT) == moves(&old, reply));
        let sum: f32 = new[new[ROOT].children()].iter().map(|c| c.prior).sum();
        assert!((sum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn retemper_matches_a_fresh_softmax() {
        let logits = [2.0f32, 0.5, -1.0, -3.0];
        let moves: Vec<(Move, f32)> = logits.iter().map(|&l| (Move::NULL, l)).collect();
        let tree = priors(&moves, 1.6);
        let root = priors(&moves, 2.1);
        let nodes: Vec<Node> = tree
            .iter()
            .map(|&(m, p)| Node::from_data(NodeData::new(m, p)))
            .collect();
        retemper(&nodes, 1.6, 2.1);
        for (n, r) in nodes.iter().zip(&root) {
            assert!((n.prior() - r.1).abs() < 1e-5, "{} vs {}", n.prior(), r.1);
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
        assert_eq!(m.arena.len(), 0);

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
        let n = m.arena.len();
        assert_eq!(m.reuse_tree(&g, &ctx), 100);
        assert_eq!(m.arena.len(), n);
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
            check_tree(&m.arena.snapshot());
            // the root holds the kept visits plus exactly 150 new ones
            let root_visits = m.arena[ROOT].visits();
            assert!(root_visits >= 150);
            if root_visits > 150 {
                reused += 1;
            }
            game.push(mv);
        }
        // the engine plays both sides here, so every search after the
        // first starts from the previous one's subtree
        assert!(reused >= 10, "reused {reused} times");
        assert_eq!(m.arena.capacity(), m.max_nodes);
    }

    #[test]
    fn pruning_collapses_the_least_visited_nodes_only() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let params = SearchParams::default();
        let mut m = Mcts::default();
        search_with(&mut m, &mut net, &Game::default(), 1500, &params);
        let original = m.arena.snapshot();
        for t in [1, 3, 10] {
            let mut a = Arena::from_data(&original, original.len());
            compact_in_place(&mut a, ROOT, Some(t));
            let nodes = a.snapshot();
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
            let sum = |v: &[NodeData]| v.iter().map(|n| n.visits as u64).sum::<u64>();
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
        assert!(m.arena.len() <= max_nodes);
        // every requested playout was done
        assert_eq!(m.arena[ROOT].visits(), 6000);
        check_tree(&m.arena.snapshot());
    }

    // ------------------------------------------------------- time management

    #[test]
    fn time_factor_follows_the_best_moves_share_and_the_pv() {
        let p = SearchParams::default();
        let f = |share, diff| p.time_factor(share, diff).2;
        // clear best move → less time; spread visits → more time
        assert!(f(0.9, 0.0) < f(0.5, 0.0) && f(0.5, 0.0) < f(0.2, 0.0));
        assert!(f(0.9, 0.0) < 1.0 && f(0.2, 0.0) > 1.0);
        // a stable PV leaves it alone; an unstable one adds time
        assert_eq!(p.time_factor(0.5, 0.0).1, 1.0);
        assert!(f(0.5, 0.2) > f(0.5, 0.0) && f(0.5, 0.5) > f(0.5, 0.2));
        let (v, pv, both) = p.time_factor(0.6, 0.1);
        assert!((both - v * pv).abs() < 1e-6);
        for share in [0.0, 0.5, 1.0] {
            for diff in [0.0, 1.0, 2.0] {
                let x = f(share, diff);
                assert!(x >= p.tm_min_mult && x <= p.tm_max_mult);
            }
        }
    }

    #[test]
    fn pv_diff_compares_the_root_move_with_the_pv_end() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let params = SearchParams::default();
        let mut m = Mcts::default();
        search_with(&mut m, &mut net, &Game::default(), 500, &params);
        let v = view(&m);
        // recompute by hand along the PV
        let best = v.root_choice().unwrap();
        let (mut node, mut depth) = (best, 1);
        let mut q = v.a[best].q();
        while let Some(c) = v.best_child(node) {
            if v.a[c].visits() == 0 {
                break;
            }
            node = c;
            depth += 1;
            q = if depth % 2 == 1 {
                v.a[c].q()
            } else {
                -v.a[c].q()
            };
        }
        assert!(depth > 3);
        let d = v.pv_diff();
        assert!((d - (v.a[best].q() - q).abs()).abs() < 1e-6);
        assert!((0.0..=2.0).contains(&d));
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
        // (it searches until the move has SINGLE_MOVE_VISITS visits, ~2 ms)
        // only Kxg2 is legal
        let limits = SearchLimits {
            wtime: Some(60_000),
            ..Default::default()
        };
        let used = timed_search("7k/8/8/8/8/8/6r1/7K w - - 0 1", limits);
        assert!(used < std::time::Duration::from_millis(100), "{used:?}");
    }

    // ---------------------------------------------------- final move choice

    /// A root with children (visits, Q for the side to move at the root).
    fn choice_tree(children: &[(u32, f32)], params: SearchParams) -> View<'static> {
        let mut nodes = vec![NodeData::new(Move::NULL, 1.0)];
        for &(v, q) in children {
            let mut c = NodeData::new(Move::NULL, 1.0 / children.len() as f32);
            c.visits = v;
            (c.w, c.l) = (v as f32 * (1.0 + q) / 2.0, v as f32 * (1.0 - q) / 2.0);
            nodes.push(c);
        }
        nodes[0].visits = 1 + children.iter().map(|c| c.0).sum::<u32>();
        nodes[0].first = 1;
        nodes[0].count = children.len() as u8;
        nodes[0].flags = EXPANDED;
        view_of(&nodes, params)
    }

    #[test]
    fn root_choice_prefers_a_clearly_better_q_when_visits_are_close() {
        let p = SearchParams::default();
        // 400 visits at Q 0.00 vs 300 visits at Q +0.10: the second wins
        assert_eq!(
            choice_tree(&[(400, 0.0), (300, 0.10)], p).root_choice(),
            Some(2)
        );
        // ... but not with only a slightly better Q
        assert_eq!(
            choice_tree(&[(400, 0.0), (300, 0.002)], p).root_choice(),
            Some(1)
        );
        // a move below the visit share never wins, however good its Q
        assert_eq!(
            choice_tree(&[(400, 0.0), (50, 0.9)], p).root_choice(),
            Some(1)
        );
        // BestMoveMinShare 100: always the most visited move
        let mut most = p;
        assert!(most.set("BestMoveMinShare", "100"));
        assert_eq!(
            choice_tree(&[(400, 0.0), (300, 0.5)], most).root_choice(),
            Some(1)
        );
        // the PV starts with the chosen move
        let t = choice_tree(&[(400, 0.0), (300, 0.10)], p);
        assert_eq!(t.root_choice(), Some(2));
        assert_eq!(t.pv().len(), 1);
    }

    // ------------------------------------------------------------- threads

    /// Search `game` with `threads` threads for `playouts` playouts.
    fn threaded(
        m: &mut Mcts,
        net: &mut PolicyNet,
        game: &Game,
        playouts: u64,
        threads: usize,
    ) -> Move {
        let limits = SearchLimits {
            nodes: Some(playouts),
            ..Default::default()
        };
        let common = CommonOptions {
            threads,
            ..Default::default()
        };
        let stop = AtomicBool::new(false);
        let params = SearchParams::default();
        m.search(net, game, &ctx_for(&limits, &common, &stop, &params))
            .best
    }

    #[test]
    fn wl_packing_and_atomic_backup() {
        let n = Node::default();
        n.add_wl(0.25, 0.5);
        n.add_wl(1.0, -0.5);
        assert_eq!(n.wl(), (1.25, 0.0));
        assert_eq!(unpack_wl(pack_wl(-3.5, 7.25)), (-3.5, 7.25));
        // many threads adding at once lose nothing
        let n = Node::default();
        std::thread::scope(|s| {
            for _ in 0..4 {
                s.spawn(|| {
                    for _ in 0..10_000 {
                        n.add_wl(1.0, 0.5);
                    }
                });
            }
        });
        assert_eq!(n.wl(), (40_000.0, 20_000.0));
    }

    #[test]
    fn arena_reservations_never_overlap() {
        let a = Arena::new(10_000);
        let firsts: Vec<Vec<usize>> = std::thread::scope(|s| {
            let hs: Vec<_> = (0..4)
                .map(|_| s.spawn(|| (0..500).filter_map(|_| a.reserve(5)).collect::<Vec<_>>()))
                .collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let mut all: Vec<usize> = firsts.into_iter().flatten().collect();
        all.sort_unstable();
        assert_eq!(all.len(), 2000);
        assert!(all.windows(2).all(|w| w[1] >= w[0] + 5));
        assert_eq!(a.len(), 10_000);
        assert!(a.reserve(1).is_none());
    }

    #[test]
    fn several_threads_search_one_consistent_tree() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let mut m = Mcts::default();
        let game =
            Game::from_fen("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1")
                .unwrap();
        let mv = threaded(&mut m, &mut net, &game, 3000, 3);
        assert!(game.position().legal_moves().contains(&mv));
        assert_eq!(m.helper_nets.len(), 2);
        let nodes = m.arena.snapshot();
        // every playout counted once, no virtual loss left over: the root's
        // visits are the playouts, and each node's W + L ≤ visits
        let root = nodes[ROOT];
        assert!(
            root.visits >= 3000 && root.visits < 3000 + 3 * 2,
            "{}",
            root.visits
        );
        for (i, n) in nodes.iter().enumerate() {
            assert!(n.w >= -1e-3 && n.l >= -1e-3, "node {i}: {n:?}");
            assert!(n.w + n.l <= n.visits as f32 + 1e-2, "node {i}: {n:?}");
        }
        check_tree(&nodes);
    }

    #[test]
    fn threads_with_reuse_and_pruning_over_a_game() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let mut m = Mcts::default();
        let mut game = Game::default();
        for _ in 0..6 {
            let limits = SearchLimits {
                nodes: Some(2500),
                ..Default::default()
            };
            // Hash 1: the threads have to prune several times
            let common = CommonOptions {
                threads: 3,
                hash_mb: 1,
                ..Default::default()
            };
            let stop = AtomicBool::new(false);
            let params = SearchParams::default();
            let mv = m
                .search(&mut net, &game, &ctx_for(&limits, &common, &stop, &params))
                .best;
            assert!(game.position().legal_moves().contains(&mv));
            check_tree(&m.arena.snapshot());
            assert!(m.arena.len() <= m.max_nodes);
            game.push(mv);
        }
    }

    #[test]
    fn one_thread_is_unchanged_by_the_threading_code() {
        // the same search twice with one thread: identical trees (no
        // virtual loss, no collisions)
        peazero_core::init();
        let game = Game::default();
        let snap = || {
            let mut net = PolicyNet::embedded();
            let mut m = Mcts::default();
            threaded(&mut m, &mut net, &game, 800, 1);
            m.arena.snapshot()
        };
        assert_eq!(snap(), snap());
    }

    #[test]
    fn stop_ends_all_threads() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let mut m = Mcts::default();
        let limits = SearchLimits {
            infinite: true,
            ..Default::default()
        };
        let common = CommonOptions {
            threads: 3,
            ..Default::default()
        };
        let stop = AtomicBool::new(false);
        let params = SearchParams::default();
        let game = Game::default();
        let t = Instant::now();
        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(300));
                stop.store(true, Relaxed);
            });
            let r = m.search(&mut net, &game, &ctx_for(&limits, &common, &stop, &params));
            assert!(!r.best.is_null());
        });
        assert!(t.elapsed() < std::time::Duration::from_millis(1500));
        check_tree(&m.arena.snapshot());
    }
}
