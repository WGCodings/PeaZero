//! Gumbel MuZero tree search on the real board (self-play, Sprint 5).
//!
//! The search follows mctx `gumbel_muzero_policy`, but every node is a real
//! position: moves are made with the engine's move generator and leaves are
//! evaluated with the representation + prediction networks (`initial`),
//! AlphaZero-style. The dynamics network is not used here.
//!
//! Per move, with `n` simulations and `m` considered actions:
//! * Root: Gumbel noise on the legal logits; Sequential Halving over the top
//!   `m` actions (mctx `gumbel_muzero_root_action_selection`), so the best
//!   candidates get most of the visits.
//! * Below the root: deterministic selection
//!   `argmax π'(a) − N(a) / (1 + ΣN)` with `π' = softmax(logits + σ(q̂))`
//!   (mctx `gumbel_muzero_interior_action_selection`).
//! * `q̂`: completed Q values (mixed value for unvisited actions), min-max
//!   rescaled; `σ(q̂) = (c_visit + max N) · c_scale · q̂`.
//! * Checkmate / draw leaves get their exact value without a network call.
//! * Move played: Sequential Halving winner; policy target:
//!   `softmax(logits + σ(q̂))` at the root.
//!
//! Playout cap randomization (KataGo; `fast_simulations > 0`): each move is
//! a *full* search (`simulations`, `considered`) with probability
//! `full_prob`, otherwise a *fast* search (`fast_simulations`,
//! `fast_considered`). Fast moves just keep the game going cheaply: they are
//! recorded with `full_search = false` and are not training samples, so
//! every policy target comes from a full search while games (the value
//! signal) stay cheap. Both use the same Gumbel noise schedule.
//!
//! Root exploration floor (`root_explore = ε > 0`): for choosing the root
//! candidates and the move played, the prior is mixed with a uniform share,
//! `(1 − ε)·π + ε/k` (AlphaZero's noise share), so every legal move keeps a
//! real chance of being considered. Without it a move whose logit has
//! fallen far below the 16th candidate is never visited again, its policy
//! target stays ~0 and training pushes it further down: castling collapsed
//! this way (logit −4 vs +7..+18 for every other move). The policy target
//! still uses the network's own logits.
//!
//! Temperature (lc0-style; `temperature > 0`): the move played is sampled
//! from the policy target `π'` as `π'(a)^(1/T)` among the root actions whose
//! search value is within `temp_value_cutoff` (W − L) of the best visited
//! action, instead of the Sequential Halving winner. `T` decays linearly
//! from `temperature` to 0 over `temp_decay_plies` plies of the game (0 =
//! no decay), but never below `temp_endgame`. The policy target is
//! unchanged; only which move is played differs. Gives game variety where
//! Gumbel noise cannot: with many simulations σ(q̂) dwarfs the noise.
//!
//! Openings (`openings`): each new game starts, with probability
//! `book_prob`, after a random number (1..=all) of moves of a random book
//! line, else from the start position; then, with probability
//! `random_prob`, 1..=`random_plies` uniformly random legal moves follow.
//! Book and random moves are not recorded (no search behind them): the game
//! record starts at the resulting position.
//!
//! The pool is stepped in lock-step rounds so the GPU sees full batches:
//! every round each game asks for exactly one evaluation (its next leaf, or
//! the root of its next move). A move takes about `n + 1` rounds.

use std::sync::Arc;

use peazero_core::encoding::{encode_game, legal_actions, OBS_SIZE};
use peazero_core::rng::Rng;
use peazero_core::{Game, GameResult, Move};
use rayon::prelude::*;

use crate::gumbel::GumbelConfig;
use crate::pool::PoolStats;
use crate::record::{GameRecord, PlyRecord, Termination};
use crate::replay::MAX_LEGAL;
use crate::{wdl_logits_to_probs, wdl_logits_to_q};

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub games: usize,
    /// Simulations (leaf evaluations) per move, not counting the root.
    pub simulations: usize,
    /// Games reaching this many plies are adjudicated as draws.
    pub max_plies: usize,
    /// `considered` = Sequential Halving width `m`.
    pub gumbel: GumbelConfig,
    pub seed: u64,
    /// Worker threads for the per-game work; 0 = one per logical CPU.
    pub threads: usize,
    /// Plies per game with the full Gumbel noise (`gumbel.gumbel_scale`);
    /// after that `late_gumbel_scale` is used. 0 = full noise all game.
    pub noise_plies: usize,
    pub late_gumbel_scale: f32,
    /// Playout cap randomization: simulations of a fast move; 0 = off
    /// (every move is a full search).
    pub fast_simulations: usize,
    /// Probability that a move gets the full search (with fast moves on).
    pub full_prob: f32,
    /// Sequential Halving width of a fast search.
    pub fast_considered: usize,
    /// Opening book lines (start position + moves); empty = no book.
    pub openings: Arc<Vec<Opening>>,
    /// Probability that a new game starts from a book line.
    pub book_prob: f32,
    /// Probability of random moves after the opening.
    pub random_prob: f32,
    /// At most this many random moves (uniform 1..=random_plies).
    pub random_plies: usize,
    /// Move-selection temperature at the start of the game; 0 = off (play
    /// the Sequential Halving winner).
    pub temperature: f32,
    /// Plies over which the temperature decays linearly to 0 (0 = constant).
    pub temp_decay_plies: usize,
    /// Lower bound of the temperature for the rest of the game.
    pub temp_endgame: f32,
    /// Only moves whose search value (W − L) is within this of the best
    /// visited move are sampled.
    pub temp_value_cutoff: f32,
    /// Root exploration floor ε: candidates are drawn from (1 − ε)·prior + ε/k.
    pub root_explore: f32,
}

/// One opening book line: a start position and the moves that follow
/// (empty for EPD/FEN books: the position itself is the opening).
#[derive(Clone, Debug)]
pub struct Opening {
    pub start: Game,
    pub moves: Vec<Move>,
}

impl Opening {
    /// From a FEN (None = start position) and UCI moves; errors on illegal moves.
    pub fn new(fen: Option<&str>, moves: &[String]) -> Result<Opening, String> {
        let start = match fen {
            Some(f) => Game::from_fen(f).map_err(|e| format!("bad FEN {f:?}: {e}"))?,
            None => Game::default(),
        };
        let mut g = start.clone();
        let mut out = Vec::with_capacity(moves.len());
        for m in moves {
            let mv = g
                .position()
                .parse_uci_move(m)
                .ok_or_else(|| format!("illegal move {m} in opening line"))?;
            g.push(mv);
            if g.result() != GameResult::Ongoing {
                return Err(format!("opening line ends the game at {m}"));
            }
            out.push(mv);
        }
        Ok(Opening { start, moves: out })
    }
}

fn uniform(rng: &mut Rng) -> f32 {
    (rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32
}

impl SearchConfig {
    /// Move-selection temperature at game ply `ply` (0 = play the best move).
    fn move_temperature(&self, ply: usize) -> f32 {
        let t = if self.temp_decay_plies > 0 {
            self.temperature * (1.0 - ply as f32 / self.temp_decay_plies as f32).max(0.0)
        } else {
            self.temperature
        };
        t.max(self.temp_endgame)
    }

    /// The start of a new game: book line and/or random moves, or the start
    /// position. Draws nothing from `rng` when neither is configured.
    fn new_game(&self, rng: &mut Rng) -> Game {
        let use_book = !self.openings.is_empty() && uniform(rng) < self.book_prob;
        let use_random =
            self.random_plies > 0 && self.random_prob > 0.0 && uniform(rng) < self.random_prob;
        if !use_book && !use_random {
            return Game::default();
        }
        for _attempt in 0..16 {
            let mut g = if use_book {
                let o = &self.openings[rng.below(self.openings.len() as u64) as usize];
                let mut g = o.start.clone();
                let n = if o.moves.is_empty() {
                    0
                } else {
                    1 + rng.below(o.moves.len() as u64) as usize
                };
                for &m in &o.moves[..n] {
                    g.push(m);
                }
                g
            } else {
                Game::default()
            };
            if use_random {
                let n = 1 + rng.below(self.random_plies as u64) as usize;
                for _ in 0..n {
                    let moves = legal_actions(g.position());
                    if moves.is_empty() {
                        break;
                    }
                    g.push(moves[rng.below(moves.len() as u64) as usize].1);
                }
            }
            if g.result() == GameResult::Ongoing {
                return g; // history kept: repetitions of book positions still count
            }
        }
        Game::default()
    }
}

impl SearchConfig {
    fn pcr(&self) -> bool {
        self.fast_simulations > 0 && self.full_prob < 1.0
    }
}

/// Considered-visit tables for the full and the fast search.
pub(crate) struct Tables {
    full: Vec<Vec<u32>>,
    fast: Vec<Vec<u32>>,
}

impl Tables {
    pub(crate) fn new(cfg: &SearchConfig) -> Tables {
        Tables {
            full: considered_visits_table(cfg.gumbel.considered, cfg.simulations),
            fast: if cfg.pcr() {
                considered_visits_table(cfg.fast_considered.max(1), cfg.fast_simulations)
            } else {
                Vec::new()
            },
        }
    }
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            games: 256,
            simulations: 64,
            max_plies: 512,
            gumbel: GumbelConfig::default(),
            seed: 0,
            threads: 1,
            noise_plies: 0,
            late_gumbel_scale: 0.0,
            fast_simulations: 0,
            full_prob: 0.25,
            fast_considered: 8,
            openings: Arc::new(Vec::new()),
            book_prob: 0.0,
            random_prob: 0.0,
            random_plies: 0,
            temperature: 0.0,
            temp_decay_plies: 0,
            temp_endgame: 0.0,
            temp_value_cutoff: 0.1,
            root_explore: 0.0,
        }
    }
}

const NONE: u32 = u32::MAX;

struct Node {
    actions: Vec<(usize, Move)>,
    /// Policy logits of `actions`, max-normalised.
    logits: Vec<f32>,
    children: Vec<u32>,
    visits: Vec<u32>,
    /// Sum of backed-up values per action, from this node's side to move.
    value_sum: Vec<f32>,
    /// Network value (`W - L`, side to move), or the exact terminal value.
    raw_value: f32,
    /// Network draw probability (exact for terminals).
    raw_draw: f32,
    terminal: bool,
}

impl Node {
    fn new(actions: Vec<(usize, Move)>, raw_logits: &[f32], raw_value: f32, raw_draw: f32) -> Node {
        let max = raw_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let k = actions.len();
        Node {
            logits: raw_logits.iter().map(|&l| l - max).collect(),
            children: vec![NONE; k],
            visits: vec![0; k],
            value_sum: vec![0.0; k],
            actions,
            raw_value,
            raw_draw,
            terminal: false,
        }
    }

    fn terminal(value: f32) -> Node {
        Node {
            actions: Vec::new(),
            logits: Vec::new(),
            children: Vec::new(),
            visits: Vec::new(),
            value_sum: Vec::new(),
            raw_value: value,
            raw_draw: if value == 0.0 { 1.0 } else { 0.0 },
            terminal: true,
        }
    }

    fn total_visits(&self) -> u32 {
        self.visits.iter().sum()
    }
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// `σ(completed Q)` for every action of `node` (mctx
/// `qtransform_completed_by_mix_value` with rescaling).
fn sigma(node: &Node, cfg: &GumbelConfig) -> Vec<f32> {
    let k = node.actions.len();
    let prior = softmax(&node.logits);
    let sum_visits: u32 = node.total_visits();
    let max_visits = node.visits.iter().copied().max().unwrap_or(0);
    let q: Vec<f32> = (0..k)
        .map(|a| {
            if node.visits[a] > 0 {
                node.value_sum[a] / node.visits[a] as f32
            } else {
                0.0
            }
        })
        .collect();
    // Mixed value (f64 so equal Qs give exactly that Q).
    let mut sum_prior = 0.0f64;
    let mut weighted = 0.0f64;
    for a in 0..k {
        if node.visits[a] > 0 {
            let p = prior[a].max(f32::MIN_POSITIVE) as f64;
            sum_prior += p;
            weighted += p * q[a] as f64;
        }
    }
    let mixed = if sum_visits > 0 {
        let s = sum_visits as f64;
        ((node.raw_value as f64 + s * weighted / sum_prior) / (s + 1.0)) as f32
    } else {
        node.raw_value
    };
    let completed: Vec<f32> = (0..k)
        .map(|a| if node.visits[a] > 0 { q[a] } else { mixed })
        .collect();
    // Min-max rescale; epsilon 1e-4 instead of mctx's 1e-8 so f32 rounding
    // noise is not blown up into a preference (see gumbel.rs).
    let lo = completed.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = completed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-4);
    let scale = (cfg.c_visit + max_visits as f32) * cfg.c_scale;
    completed.iter().map(|&v| scale * (v - lo) / span).collect()
}

/// mctx `get_sequence_of_considered_visits`.
fn considered_visits_sequence(m: usize, n: usize) -> Vec<u32> {
    if m <= 1 {
        return (0..n as u32).collect();
    }
    let log2max = (m as f64).log2().ceil() as usize;
    let mut seq = Vec::with_capacity(n + m);
    let mut visits = vec![0u32; m];
    let mut considered = m;
    while seq.len() < n {
        let extra = (n / (log2max * considered)).max(1);
        for _ in 0..extra {
            seq.extend_from_slice(&visits[..considered]);
            for v in visits.iter_mut().take(considered) {
                *v += 1;
            }
        }
        considered = (considered / 2).max(2);
    }
    seq.truncate(n);
    seq
}

/// Table indexed by `[num_considered][simulation]` (mctx
/// `get_table_of_considered_visits`), one extra column so the final
/// selection after `n` simulations can be looked up too.
pub(crate) fn considered_visits_table(m: usize, n: usize) -> Vec<Vec<u32>> {
    (0..=m)
        .map(|k| {
            let mut s = considered_visits_sequence(k, n);
            s.push(u32::MAX); // never used for selection
            s
        })
        .collect()
}

/// mctx `score_considered`: `g + logits + σ`, restricted to the actions whose
/// visit count equals `considered_visit`.
fn score_considered(node: &Node, gumbel: &[f32], sigma: &[f32], considered_visit: u32) -> usize {
    let score = |a: usize| (gumbel[a] + node.logits[a] + sigma[a]).max(-1e9);
    let best = (0..node.actions.len())
        .filter(|&a| node.visits[a] == considered_visit)
        .max_by(|&a, &b| score(a).total_cmp(&score(b)));
    // The schedule always leaves a candidate; fall back defensively.
    best.unwrap_or_else(|| {
        (0..node.actions.len())
            .max_by(|&a, &b| score(a).total_cmp(&score(b)))
            .unwrap()
    })
}

fn interior_action(node: &Node, cfg: &GumbelConfig) -> usize {
    let s = sigma(node, cfg);
    let improved: Vec<f32> = node.logits.iter().zip(&s).map(|(l, s)| l + s).collect();
    let probs = softmax(&improved);
    let denom = 1.0 + node.total_visits() as f32;
    (0..node.actions.len())
        .max_by(|&a, &b| {
            (probs[a] - node.visits[a] as f32 / denom)
                .total_cmp(&(probs[b] - node.visits[b] as f32 / denom))
        })
        .unwrap()
}

/// Exact value of a finished game for its side to move, or None if ongoing.
fn terminal_value(game: &Game) -> Option<f32> {
    match game.result() {
        GameResult::Ongoing => None,
        GameResult::Checkmate(_) => Some(-1.0), // side to move is mated
        GameResult::Draw(_) => Some(0.0),
    }
}

/// What the slot is waiting for.
enum Request {
    /// Evaluate the root of the next move.
    Root,
    /// Evaluate a new leaf reached by `path` (last entry = parent, action).
    Leaf { path: Vec<(u32, usize)> },
}

pub(crate) struct Slot {
    pub(crate) game: Game,
    pub(crate) plies: Vec<PlyRecord>,
    rng: Rng,
    nodes: Vec<Node>,
    root_gumbel: Vec<f32>,
    simulations_done: usize,
    /// This move gets the full search (playout cap randomization).
    full: bool,
    request: Request,
    /// Position to evaluate and its legal moves.
    pub(crate) eval_game: Game,
    pub(crate) eval_actions: Vec<(usize, Move)>,
}

pub(crate) struct Outcome {
    /// 1 white won, -1 black won, 0 draw.
    pub(crate) result: i8,
    pub(crate) termination: Termination,
    pub(crate) plies: Vec<PlyRecord>,
}

impl Slot {
    /// A slot that starts searching from `game` (evaluation games, puzzles).
    pub(crate) fn with_game(game: Game, rng: Rng) -> Slot {
        let eval_actions = legal_actions(game.position());
        Slot {
            eval_game: game.clone(),
            game,
            plies: Vec::with_capacity(256),
            rng,
            nodes: Vec::new(),
            root_gumbel: Vec::new(),
            simulations_done: 0,
            full: true,
            request: Request::Root,
            eval_actions,
        }
    }

    pub(crate) fn request_root(&mut self) {
        self.nodes.clear();
        self.simulations_done = 0;
        self.eval_game = self.game.clone();
        self.eval_actions = legal_actions(self.game.position());
        self.request = Request::Root;
    }

    fn backup(&mut self, path: &[(u32, usize)], leaf_value: f32) {
        let mut v = leaf_value;
        for &(node, a) in path.iter().rev() {
            v = -v;
            let n = &mut self.nodes[node as usize];
            n.visits[a] += 1;
            n.value_sum[a] += v;
        }
    }

    /// Run simulations until one needs a network evaluation (sets
    /// `request`) or the move's budget is used up (returns true).
    fn advance(&mut self, cfg: &SearchConfig, tables: &Tables) -> bool {
        let (budget, considered, table) = if self.full {
            (cfg.simulations, cfg.gumbel.considered, &tables.full)
        } else {
            (
                cfg.fast_simulations,
                cfg.fast_considered.max(1),
                &tables.fast,
            )
        };
        while self.simulations_done < budget {
            let mut path: Vec<(u32, usize)> = Vec::with_capacity(16);
            let mut node = 0u32;
            loop {
                let n = &self.nodes[node as usize];
                let a = if node == 0 {
                    let k = n.actions.len().min(considered).max(1);
                    let s = sigma(n, &cfg.gumbel);
                    let cv = table[k][self.simulations_done];
                    score_considered(n, &self.root_gumbel, &s, cv)
                } else {
                    interior_action(n, &cfg.gumbel)
                };
                path.push((node, a));
                let child = n.children[a];
                if child == NONE {
                    break;
                }
                if self.nodes[child as usize].terminal {
                    break;
                }
                node = child;
            }

            let &(parent, a) = path.last().unwrap();
            let existing = self.nodes[parent as usize].children[a];
            if existing != NONE {
                // Revisit of a known terminal: back up its exact value.
                let v = self.nodes[existing as usize].raw_value;
                self.backup(&path, v);
                self.simulations_done += 1;
                continue;
            }

            let mut leaf = self.game.clone();
            for &(n, act) in &path {
                leaf.push(self.nodes[n as usize].actions[act].1);
            }
            if let Some(v) = terminal_value(&leaf) {
                let idx = self.nodes.len() as u32;
                self.nodes.push(Node::terminal(v));
                self.nodes[parent as usize].children[a] = idx;
                self.backup(&path, v);
                self.simulations_done += 1;
                continue;
            }
            self.eval_actions = legal_actions(leaf.position());
            self.eval_game = leaf;
            self.request = Request::Leaf { path };
            return false;
        }
        true
    }

    /// Pick the move, record the ply, play it. Returns the finished game, if
    /// any, and sets up the next root request.
    fn play_move(&mut self, cfg: &SearchConfig) -> Option<Outcome> {
        let root = &self.nodes[0];
        let s = sigma(root, &cfg.gumbel);
        let max_visit = root.visits.iter().copied().max().unwrap_or(0);
        let mut chosen = score_considered(root, &self.root_gumbel, &s, max_visit);
        // With the exploration floor, a move the search visited gets at least
        // the floored prior in its target, so a collapsed move that turns out
        // good can regain policy mass (σ alone cannot lift a logit ~20 down).
        let floored = explore_logits(&root.logits, cfg.root_explore);
        let improved: Vec<f32> = (0..root.actions.len())
            .map(|a| match &floored {
                Some(f) if root.visits[a] > 0 => root.logits[a].max(f[a]) + s[a],
                _ => root.logits[a] + s[a],
            })
            .collect();
        let target = softmax(&improved);
        let temp = cfg.move_temperature(self.game.ply());
        if temp > 0.0 {
            chosen =
                sample_with_temperature(root, &target, temp, cfg.temp_value_cutoff, &mut self.rng)
                    .unwrap_or(chosen);
        }
        let targets: Vec<(usize, f32)> = root.actions.iter().map(|&(a, _)| a).zip(target).collect();
        let (action, mv) = root.actions[chosen];
        // Search value of the root: mean of the network value and all backed-up
        // values (mctx node value). W - L from it, D from the network.
        let visits: u32 = root.total_visits();
        let sum: f32 = root.value_sum.iter().sum();
        let q = ((root.raw_value + sum) / (1.0 + visits as f32)).clamp(-1.0, 1.0);
        let d = root.raw_draw.min(1.0 - q.abs());
        let value = [(1.0 - d + q) / 2.0, d, (1.0 - d - q) / 2.0];
        self.plies.push(
            PlyRecord::new(
                self.game.position(),
                self.game.repetitions(),
                action,
                &targets,
            )
            .with_value(value)
            .with_full_search(self.full),
        );
        self.game.push(mv);

        let outcome = match Termination::from_result(self.game.result()) {
            Some(x) => Some(x),
            None if self.plies.len() >= cfg.max_plies => Some((0, Termination::MaxPlies)),
            None => None,
        };
        let finished = outcome.map(|(result, termination)| {
            self.game = cfg.new_game(&mut self.rng);
            Outcome {
                result,
                termination,
                plies: std::mem::replace(&mut self.plies, Vec::with_capacity(256)),
            }
        });
        self.request_root();
        finished
    }

    /// Apply this slot's network output and advance to its next request.
    pub(crate) fn respond(
        &mut self,
        logits: &[f32],
        value: &[f32],
        cfg: &SearchConfig,
        tables: &Tables,
    ) -> Option<Outcome> {
        let k = self.eval_actions.len();
        let v = wdl_logits_to_q(value);
        let draw = wdl_logits_to_probs(value)[1];
        let actions = std::mem::take(&mut self.eval_actions);
        let node = Node::new(actions, &logits[..k], v, draw);
        match std::mem::replace(&mut self.request, Request::Root) {
            Request::Root => {
                // Drawn only with PCR on, so the games without it are unchanged.
                self.full = !cfg.pcr()
                    || ((self.rng.next_u64() >> 40) as f32 / (1u64 << 24) as f32) < cfg.full_prob;
                let scale = if cfg.noise_plies > 0 && self.plies.len() >= cfg.noise_plies {
                    cfg.late_gumbel_scale
                } else {
                    cfg.gumbel.gumbel_scale
                };
                self.root_gumbel = (0..k)
                    .map(|_| scale * sample_gumbel(&mut self.rng))
                    .collect();
                if let Some(f) = explore_logits(&node.logits, cfg.root_explore) {
                    // The root scores are gumbel + logit + σ: add floored − logit so
                    // candidates and the played move see the floored prior.
                    for ((g, &l), &fl) in self.root_gumbel.iter_mut().zip(&node.logits).zip(&f) {
                        *g += fl - l;
                    }
                }
                self.nodes.clear();
                self.nodes.push(node);
            }
            Request::Leaf { path } => {
                let &(parent, a) = path.last().unwrap();
                let idx = self.nodes.len() as u32;
                self.nodes.push(node);
                self.nodes[parent as usize].children[a] = idx;
                self.backup(&path, v);
                self.simulations_done += 1;
            }
        }
        if self.advance(cfg, tables) {
            self.play_move(cfg)
        } else {
            None
        }
    }
}

/// Logits of the floored prior `(1 − ε)·softmax(logits) + ε/k`, on the same
/// scale as `logits` (max-normalised logits are log-probabilities up to a
/// constant). None when ε = 0.
fn explore_logits(logits: &[f32], eps: f32) -> Option<Vec<f32>> {
    if eps <= 0.0 || logits.is_empty() {
        return None;
    }
    let eps = eps.min(1.0);
    let k = logits.len() as f32;
    let prior = softmax(logits);
    // ln of the mixed probability, shifted back to the logits' scale
    let shift = logits
        .iter()
        .zip(&prior)
        .map(|(&l, &p)| l - p.max(f32::MIN_POSITIVE).ln())
        .sum::<f32>()
        / k;
    Some(
        prior
            .iter()
            .map(|&p| ((1.0 - eps) * p + eps / k).ln() + shift)
            .collect(),
    )
}

/// lc0-style move sampling: `target(a)^(1/temp)` over the visited root
/// actions whose Q is within `cutoff` of the best visited action's Q.
fn sample_with_temperature(
    root: &Node,
    target: &[f32],
    temp: f32,
    cutoff: f32,
    rng: &mut Rng,
) -> Option<usize> {
    let q = |a: usize| root.value_sum[a] / root.visits[a] as f32;
    let visited: Vec<usize> = (0..root.actions.len())
        .filter(|&a| root.visits[a] > 0)
        .collect();
    let best = visited
        .iter()
        .map(|&a| q(a))
        .fold(f32::NEG_INFINITY, f32::max);
    let eligible: Vec<usize> = visited
        .into_iter()
        .filter(|&a| q(a) >= best - cutoff)
        .collect();
    // weights relative to the largest target, so 1/temp powers stay in range
    let top = eligible.iter().map(|&a| target[a]).fold(0.0f32, f32::max);
    if top <= 0.0 {
        return None;
    }
    let w: Vec<f64> = eligible
        .iter()
        .map(|&a| ((target[a] / top) as f64).powf(1.0 / temp as f64))
        .collect();
    let total: f64 = w.iter().sum();
    let mut x = uniform(rng) as f64 * total;
    for (i, &wi) in w.iter().enumerate() {
        if x < wi {
            return Some(eligible[i]);
        }
        x -= wi;
    }
    eligible.last().copied()
}

fn sample_gumbel(rng: &mut Rng) -> f32 {
    let u = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    (-(-u.ln()).ln()) as f32
}

pub struct SearchPool {
    cfg: SearchConfig,
    tables: Arc<Tables>,
    slots: Vec<Slot>,
    threads: rayon::ThreadPool,
    finished: Vec<GameRecord>,
    stats: PoolStats,
}

impl SearchPool {
    pub fn new(cfg: SearchConfig) -> SearchPool {
        assert!(cfg.games > 0 && cfg.gumbel.considered > 0);
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .thread_name(|i| format!("peazero-search-{i}"))
            .build()
            .expect("failed to start self-play threads");
        let tables = Arc::new(Tables::new(&cfg));
        SearchPool {
            slots: (0..cfg.games)
                .map(|i| {
                    let seed = (cfg.seed ^ 0x5EA8_C411)
                        .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    let mut rng = Rng::new(seed);
                    let game = cfg.new_game(&mut rng);
                    Slot::with_game(game, rng)
                })
                .collect(),
            tables,
            threads,
            finished: Vec::new(),
            stats: PoolStats::default(),
            cfg,
        }
    }

    pub fn num_games(&self) -> usize {
        self.slots.len()
    }

    pub fn num_threads(&self) -> usize {
        self.threads.current_num_threads()
    }

    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Positions to evaluate this round: observations (`num_games() *
    /// OBS_SIZE`) and the policy indices of their legal moves
    /// (`num_games() * MAX_LEGAL`, padded with 0).
    pub fn requests(&self, obs: &mut [f32], legal: &mut [i32]) {
        assert_eq!(obs.len(), self.slots.len() * OBS_SIZE);
        assert_eq!(legal.len(), self.slots.len() * MAX_LEGAL);
        let slots = &self.slots;
        self.threads.install(|| {
            slots
                .par_iter()
                .zip(obs.par_chunks_exact_mut(OBS_SIZE))
                .zip(legal.par_chunks_exact_mut(MAX_LEGAL))
                .for_each(|((slot, o), l)| {
                    encode_game(&slot.eval_game, o);
                    l.fill(0);
                    for (dst, &(a, _)) in l.iter_mut().zip(&slot.eval_actions) {
                        *dst = a as i32;
                    }
                });
        });
    }

    /// Network output for this round's requests: legal-move logits gathered
    /// at the indices from `requests` (`num_games() * MAX_LEGAL`) and W/D/L
    /// value logits (`num_games() * 3`). Returns the number of games that
    /// finished.
    pub fn respond(&mut self, logits: &[f32], value: &[f32]) -> usize {
        assert_eq!(logits.len(), self.slots.len() * MAX_LEGAL);
        assert_eq!(value.len(), self.slots.len() * 3);
        let cfg = &self.cfg;
        let tables = &self.tables;
        let slots = &mut self.slots;
        let (outcomes, moves): (Vec<Option<Outcome>>, Vec<u32>) = self.threads.install(|| {
            slots
                .par_iter_mut()
                .zip(logits.par_chunks_exact(MAX_LEGAL))
                .zip(value.par_chunks_exact(3))
                .map(|((slot, l), v)| {
                    let before = slot.plies.len();
                    let out = slot.respond(l, v, cfg, tables);
                    let moved = out.is_some() || slot.plies.len() != before;
                    // the move just played: last ply of the slot or of the finished game
                    let full = moved
                        && match &out {
                            Some(o) => o.plies.last(),
                            None => slot.plies.last(),
                        }
                        .is_some_and(|p| p.full_search);
                    (out, moved as u32 | (full as u32) << 1)
                })
                .unzip()
        });
        self.stats.plies += moves.iter().map(|&m| (m & 1) as u64).sum::<u64>();
        self.stats.full_plies += moves.iter().map(|&m| (m >> 1) as u64).sum::<u64>();
        let mut done = 0;
        for o in outcomes.into_iter().flatten() {
            self.stats.games_finished += 1;
            self.stats.terminations[o.termination as usize] += 1;
            match o.result {
                1 => self.stats.white_wins += 1,
                -1 => self.stats.black_wins += 1,
                _ => self.stats.draws += 1,
            }
            self.finished.push(GameRecord {
                plies: o.plies,
                result: o.result,
                termination: o.termination,
            });
            done += 1;
        }
        done
    }

    pub fn take_finished(&mut self) -> Vec<GameRecord> {
        std::mem::take(&mut self.finished)
    }

    pub fn plies_in_progress(&self) -> usize {
        self.slots.iter().map(|s| s.plies.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peazero_core::encoding::{move_to_action, NUM_ACTIONS};
    use peazero_core::Position;

    #[test]
    fn considered_visits_match_mctx() {
        // Reference values computed with mctx's Python implementation.
        assert_eq!(considered_visits_sequence(4, 6), vec![0, 0, 0, 0, 1, 1]);
        assert_eq!(considered_visits_sequence(2, 5), vec![0, 0, 1, 1, 2]);
        let s = considered_visits_sequence(16, 64);
        let expected: Vec<u32> = [
            vec![0; 16],
            vec![1; 8],
            vec![2; 8],
            vec![3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6],
            vec![7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14],
        ]
        .concat();
        assert_eq!(s, expected);
    }

    /// Drive a single-slot pool with an "oracle" network: uniform policy and
    /// a value that only knows material. The search must find wins the raw
    /// policy cannot see.
    fn search_once(fen: &str, simulations: usize) -> (Move, Vec<(usize, f32)>) {
        search_with(fen, simulations, GumbelConfig::default().considered)
    }

    fn search_with(fen: &str, simulations: usize, considered: usize) -> (Move, Vec<(usize, f32)>) {
        let (mv, targets, _) = search_full(fen, simulations, considered);
        (mv, targets)
    }

    /// Searches one move; returns move, policy target and recorded value.
    fn search_full(
        fen: &str,
        simulations: usize,
        considered: usize,
    ) -> (Move, Vec<(usize, f32)>, Option<[f32; 3]>) {
        peazero_core::init();
        let cfg = SearchConfig {
            games: 1,
            simulations,
            gumbel: GumbelConfig {
                gumbel_scale: 0.0,
                considered,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut pool = SearchPool::new(cfg);
        let game = Game::from_fen(fen).unwrap();
        pool.slots[0].game = game.clone();
        pool.slots[0].request_root();
        let mut obs = vec![0.0; OBS_SIZE];
        let mut legal = vec![0i32; MAX_LEGAL];
        loop {
            pool.requests(&mut obs, &mut legal);
            let pos = *pool.slots[0].eval_game.position();
            let v = material(&pos);
            // W/D/L logits giving W - L ≈ v
            let value = [v * 3.0, 0.0, -v * 3.0];
            let logits = vec![0.0f32; MAX_LEGAL];
            pool.respond(&logits, &value);
            let finished = pool.take_finished();
            let ply = finished
                .first()
                .and_then(|g| g.plies.last())
                .or(pool.slots[0].plies.last());
            if let Some(p) = ply {
                let mv = peazero_core::encoding::action_to_move(game.position(), p.action as usize)
                    .unwrap();
                let targets = p
                    .targets
                    .iter()
                    .map(|&(a, q)| (a as usize, q as f32 / 65535.0))
                    .collect();
                return (mv, targets, p.value_wdl());
            }
        }
    }

    #[test]
    fn records_search_value() {
        // White mates in one: the root search value must be clearly winning,
        // and a valid W/D/L distribution.
        let fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1";
        let (_, _, value) = search_full(fen, 64, 16);
        let [w, d, l] = value.unwrap();
        assert!((w + d + l - 1.0).abs() < 1e-3);
        assert!(w - l > 0.5, "search value {w} {d} {l}");
        // Black is a rook down with no tactic: the value is negative.
        let fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1";
        let (_, _, value) = search_full(fen, 64, 16);
        let [w, _, l] = value.unwrap();
        assert!(w - l < 0.0, "search value {w} {l}");
    }

    /// Material balance for the side to move, squashed to (-1, 1).
    fn material(pos: &Position) -> f32 {
        let fen = pos.to_fen();
        let board = fen.split(' ').next().unwrap();
        let mut s = 0i32;
        for c in board.chars() {
            let v = match c.to_ascii_lowercase() {
                'p' => 1,
                'n' | 'b' => 3,
                'r' => 5,
                'q' => 9,
                _ => 0,
            };
            s += if c.is_ascii_uppercase() { v } else { -v };
        }
        let white = fen.split(' ').nth(1) == Some("w");
        let s = if white { s } else { -s };
        (s as f32 / 10.0).tanh()
    }

    fn uci(pos: &Position, m: &str) -> Move {
        pos.parse_uci_move(m).unwrap()
    }

    #[test]
    fn finds_mate_in_one() {
        // Back-rank mate: Ra8#. Uniform policy, material value: only the
        // terminal check in the search can find it.
        let fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1";
        let (mv, targets) = search_once(fen, 64);
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(mv, uci(&pos, "a1a8"));
        let a = move_to_action(mv, pos.side_to_move());
        let best = targets.iter().map(|t| t.1).fold(0.0, f32::max);
        assert!(targets.iter().any(|&(x, q)| x == a && q == best));
    }

    #[test]
    fn avoids_poisoned_capture() {
        // Qxd5?? loses the queen to exd5; the pawn on d5 is defended. A
        // one-ply look (every move visited once, like root-only Gumbel) sees
        // +1 pawn and plays it; the tree sees the recapture.
        let fen = "4k3/8/4p3/3p4/8/8/3Q4/4K3 w - - 0 1";
        let pos = Position::from_fen(fen).unwrap();
        let n = pos.legal_moves().len();
        let (one_ply, _) = search_with(fen, n, n);
        assert_eq!(one_ply, uci(&pos, "d2d5"));
        let (mv, _) = search_once(fen, 64);
        assert_ne!(mv, uci(&pos, "d2d5"));
    }

    #[test]
    fn takes_free_piece() {
        // Black rook on h6 hangs to the bishop on c1.
        let fen = "4k3/8/7r/8/8/8/8/2B1K3 w - - 0 1";
        let (mv, _) = search_once(fen, 64);
        let pos = Position::from_fen(fen).unwrap();
        assert_eq!(mv, uci(&pos, "c1h6"));
    }

    #[test]
    fn playout_cap_randomization() {
        peazero_core::init();
        let run = |fast: usize| {
            let cfg = SearchConfig {
                games: 8,
                simulations: 32,
                max_plies: 80,
                threads: 2,
                fast_simulations: fast,
                full_prob: 0.25,
                fast_considered: 4,
                ..Default::default()
            };
            let mut pool = SearchPool::new(cfg);
            let n = pool.num_games();
            let mut rng = Rng::new(9);
            let (mut obs, mut legal) = (vec![0.0; n * OBS_SIZE], vec![0i32; n * MAX_LEGAL]);
            let mut rounds = 0u64;
            while pool.stats().games_finished < 16 {
                pool.requests(&mut obs, &mut legal);
                let logits: Vec<f32> = (0..n * MAX_LEGAL)
                    .map(|_| rng.below(100) as f32 / 40.0)
                    .collect();
                let value: Vec<f32> = (0..n * 3).map(|_| rng.below(100) as f32 / 40.0).collect();
                pool.respond(&logits, &value);
                rounds += 1;
            }
            let games = pool.take_finished();
            let plies: Vec<&PlyRecord> = games.iter().flat_map(|g| &g.plies).collect();
            let full = plies.iter().filter(|p| p.full_search).count();
            let s = pool.stats();
            assert!(s.full_plies <= s.plies);
            (
                full as f64 / plies.len() as f64,
                rounds as f64 * n as f64 / s.plies as f64,
            )
        };
        let (share_off, evals_off) = run(0);
        assert_eq!(share_off, 1.0);
        let (share, evals) = run(8);
        assert!((share - 0.25).abs() < 0.08, "full-search share {share}");
        // ~0.25 × 33 + 0.75 × 9 ≈ 15 evaluations per move instead of ~33
        assert!(
            evals < 0.6 * evals_off,
            "evaluations per move {evals} vs {evals_off}"
        );
    }

    #[test]
    fn root_explore_rediscovers_a_collapsed_move() {
        peazero_core::init();
        // O-O is mate, but the "network" gives castling a logit 20 below
        // every other move (what self-play training had done to it).
        let fen = "8/1P6/B7/8/R7/5k2/4N3/3NK2R w K - 0 1";
        let run = |root_explore: f32| {
            let cfg = SearchConfig {
                games: 64,
                simulations: 200,
                max_plies: 1,
                threads: 2,
                gumbel: GumbelConfig {
                    considered: 16,
                    ..Default::default()
                },
                openings: Arc::new(vec![Opening::new(Some(fen), &[]).unwrap()]),
                book_prob: 1.0,
                root_explore,
                ..Default::default()
            };
            let mut pool = SearchPool::new(cfg);
            let n = pool.num_games();
            let (mut obs, mut legal) = (vec![0.0; n * OBS_SIZE], vec![0i32; n * MAX_LEGAL]);
            while pool.stats().games_finished < 64 {
                pool.requests(&mut obs, &mut legal);
                let mut logits = vec![0.0f32; n * MAX_LEGAL];
                let mut value = Vec::with_capacity(n * 3);
                for (i, s) in pool.slots.iter().enumerate() {
                    for (j, (_, m)) in s.eval_actions.iter().enumerate() {
                        // a peaked prior like a trained net's: a few likely moves,
                        // a long tail; castling collapsed far below everything
                        logits[i * MAX_LEGAL + j] = if m.is_castle() {
                            -20.0
                        } else {
                            -0.35 * j as f32
                        };
                    }
                    // material on a linear scale (±40 = ±1): every white move is
                    // good here, mate (exact +1) clearly best
                    let m = material(s.eval_game.position()).clamp(-0.999, 0.999);
                    let v = (m.atanh() * 10.0 / 40.0).clamp(-0.99, 0.99);
                    value.extend_from_slice(&[v * 3.0, 0.0, -v * 3.0]);
                }
                pool.respond(&logits, &value);
            }
            let games = pool.take_finished();
            let pos = Position::from_fen(fen).unwrap();
            let castle = pos
                .legal_moves()
                .iter()
                .copied()
                .find(|m| m.is_castle())
                .unwrap();
            let a = move_to_action(castle, pos.side_to_move()) as u16;
            let played = games.iter().filter(|g| g.plies[0].action == a).count();
            let target: f32 = games
                .iter()
                .map(|g| {
                    g.plies[0]
                        .targets
                        .iter()
                        .find(|t| t.0 == a)
                        .map_or(0.0, |t| t.1 as f32 / 65535.0)
                })
                .sum::<f32>()
                / games.len() as f32;
            (played, target)
        };
        let (played_off, target_off) = run(0.0);
        assert_eq!(played_off, 0, "collapsed castling found without the floor");
        assert!(target_off < 1e-3);
        let (played_on, target_on) = run(0.25);
        // With a peaked prior the floor gives castling a fair lottery chance
        // among the ~25 low-prior moves for the last candidate slots (~1 in 7
        // positions); where it is considered, the mate is played and its target
        // is large, so training pulls the logit back up.
        assert!(
            played_on >= 4,
            "castling mate played in only {played_on}/64 games with the floor"
        );
        assert!(target_on > 0.015, "mean castling target {target_on}");
    }

    #[test]
    fn temperature_varies_the_moves_played() {
        peazero_core::init();
        // Every game sees the same evaluations, so without temperature (and
        // without Gumbel noise) all games play the same moves.
        let distinct_starts = |temperature: f32, cutoff: f32, fen: Option<&str>| {
            let openings = fen
                .map(|f| vec![Opening::new(Some(f), &[]).unwrap()])
                .unwrap_or_default();
            let cfg = SearchConfig {
                openings: Arc::new(openings),
                book_prob: 1.0,
                games: 48,
                simulations: 16,
                max_plies: 3,
                threads: 2,
                gumbel: GumbelConfig {
                    gumbel_scale: 0.0,
                    considered: 8,
                    ..Default::default()
                },
                temperature,
                temp_value_cutoff: cutoff,
                ..Default::default()
            };
            let mut pool = SearchPool::new(cfg);
            let n = pool.num_games();
            let (mut obs, mut legal) = (vec![0.0; n * OBS_SIZE], vec![0i32; n * MAX_LEGAL]);
            while pool.stats().games_finished < 48 {
                pool.requests(&mut obs, &mut legal);
                let logits: Vec<f32> = (0..n * MAX_LEGAL)
                    .map(|i| (((i % MAX_LEGAL) * 7919) % 13) as f32 / 4.0)
                    .collect();
                let mut value = Vec::with_capacity(n * 3);
                for s in &pool.slots {
                    let v = material(s.eval_game.position());
                    value.extend_from_slice(&[v * 3.0, 0.0, -v * 3.0]);
                }
                pool.respond(&logits, &value);
            }
            let games = pool.take_finished();
            let lines = games
                .iter()
                .map(|g| g.plies.iter().map(|p| p.action).collect::<Vec<_>>())
                .collect::<std::collections::HashSet<_>>()
                .len();
            let first = games
                .iter()
                .map(|g| g.plies[0].action)
                .collect::<std::collections::HashSet<_>>()
                .len();
            (lines, first)
        };
        assert_eq!(distinct_starts(0.0, 0.1, None).0, 1);
        let (varied, _) = distinct_starts(1.0, 1.0, None);
        assert!(varied > 10, "{varied} different games with temperature 1");
        // A free rook: the target puts almost everything on the capture. At a
        // high temperature other moves get sampled, unless the value cutoff
        // excludes them.
        let fen = Some("4k3/8/7r/8/8/8/8/2B1K3 w - - 0 1");
        assert_eq!(distinct_starts(5.0, 0.0, fen).1, 1);
        assert!(distinct_starts(5.0, 2.0, fen).1 > 1);
        // decay schedule
        let cfg = SearchConfig {
            temperature: 1.0,
            temp_decay_plies: 20,
            temp_endgame: 0.2,
            ..Default::default()
        };
        assert_eq!(cfg.move_temperature(0), 1.0);
        assert!((cfg.move_temperature(10) - 0.5).abs() < 1e-6);
        assert_eq!(cfg.move_temperature(30), 0.2);
    }

    #[test]
    fn openings_and_random_moves() {
        peazero_core::init();
        let line = |m: &str| m.split_whitespace().map(String::from).collect::<Vec<_>>();
        let book = vec![
            Opening::new(None, &line("e2e4 e7e5 g1f3 b8c6 f1b5 a7a6")).unwrap(),
            Opening::new(None, &line("d2d4 d7d5 c2c4 e7e6 b1c3 g8f6")).unwrap(),
            Opening::new(
                Some("rnbqkb1r/pppppppp/5n2/8/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2"),
                &[],
            )
            .unwrap(),
        ];
        assert!(Opening::new(None, &line("e2e5")).is_err());
        let cfg = SearchConfig {
            games: 64,
            simulations: 4,
            max_plies: 6,
            threads: 2,
            gumbel: GumbelConfig {
                considered: 4,
                ..Default::default()
            },
            openings: Arc::new(book),
            book_prob: 0.5,
            random_prob: 0.5,
            random_plies: 2,
            ..Default::default()
        };
        let mut pool = SearchPool::new(cfg);
        let n = pool.num_games();
        let mut rng = Rng::new(3);
        let (mut obs, mut legal) = (vec![0.0; n * OBS_SIZE], vec![0i32; n * MAX_LEGAL]);
        while pool.stats().games_finished < 200 {
            pool.requests(&mut obs, &mut legal);
            let logits: Vec<f32> = (0..n * MAX_LEGAL)
                .map(|_| rng.below(100) as f32 / 40.0)
                .collect();
            let value: Vec<f32> = (0..n * 3).map(|_| rng.below(100) as f32 / 40.0).collect();
            pool.respond(&logits, &value);
        }
        let games = pool.take_finished();
        let start = Position::startpos();
        let mut starts = std::collections::HashSet::new();
        let mut fresh = 0;
        for g in &games {
            // every game replays legally from its first recorded position
            let first = g.plies[0].position();
            fresh += (first.to_fen() == start.to_fen()) as usize;
            starts.insert(first.to_fen());
            let mut game = Game::new(first);
            for p in &g.plies {
                let key = |f: String| f.split(' ').take(5).collect::<Vec<_>>().join(" ");
                assert_eq!(key(game.position().to_fen()), key(p.position().to_fen()));
                let mv = peazero_core::encoding::action_to_move(game.position(), p.action as usize)
                    .unwrap();
                game.push(mv);
            }
        }
        // P(no book, no random moves) = 0.25
        let share = fresh as f64 / games.len() as f64;
        assert!((share - 0.25).abs() < 0.1, "fresh starts {share}");
        assert!(
            starts.len() > 40,
            "only {} different start positions",
            starts.len()
        );
    }

    #[test]
    fn pool_plays_complete_games() {
        peazero_core::init();
        let cfg = SearchConfig {
            games: 6,
            simulations: 8,
            max_plies: 60,
            threads: 2,
            ..Default::default()
        };
        let mut pool = SearchPool::new(cfg);
        let n = pool.num_games();
        let mut rng = Rng::new(4);
        let mut obs = vec![0.0; n * OBS_SIZE];
        let mut legal = vec![0i32; n * MAX_LEGAL];
        while pool.stats().games_finished < 6 {
            pool.requests(&mut obs, &mut legal);
            let logits: Vec<f32> = (0..n * MAX_LEGAL)
                .map(|_| rng.below(100) as f32 / 40.0)
                .collect();
            let value: Vec<f32> = (0..n * 3).map(|_| rng.below(100) as f32 / 40.0).collect();
            pool.respond(&logits, &value);
        }
        let games = pool.take_finished();
        for g in &games {
            let mut game = Game::default();
            for p in &g.plies {
                let sum: u32 = p.targets.iter().map(|&(_, q)| q as u32).sum();
                assert!((sum as i64 - 65535).abs() < 300);
                let mv = peazero_core::encoding::action_to_move(game.position(), p.action as usize)
                    .unwrap();
                game.push(mv);
            }
        }
        let s = pool.stats();
        assert_eq!(s.white_wins + s.black_wins + s.draws, s.games_finished);
        assert!(s.plies > 0);
        let _ = NUM_ACTIONS;
    }
}
