//! Strength evaluation for the background evaluator (trainer/peazero/evaluator.py):
//! matches between two networks and puzzle solving.
//!
//! Both reuse the self-play search slot (search.rs) with the Gumbel noise
//! switched off, so a network always plays the same move in the same
//! position. `simulations = 0` is searchless: the move with the highest
//! policy logit. With simulations it is the deterministic Gumbel tree search
//! (Sequential Halving at the root), like self-play without noise.
//!
//! Games of a match start from random openings; every opening is played
//! twice with colours swapped, so an unbalanced opening favours nobody.
//! Within a game, the side to move's network evaluates every position of
//! its search.

use peazero_core::encoding::{encode_game, legal_actions, move_to_action, NUM_ACTIONS, OBS_SIZE};
use peazero_core::rng::Rng;
use peazero_core::{Color, Game, GameResult, Move};
use rayon::prelude::*;

use crate::gumbel::GumbelConfig;
use crate::record::Termination;
use crate::search::{Outcome, SearchConfig, Slot, Tables};

/// Batched network evaluation: `(network index, observations of n
/// positions)` → `(policy logits n × NUM_ACTIONS, W/D/L value logits n × 3)`.
/// Called from inside the arena's thread pool, so it may use rayon itself.
pub trait Evaluate: Sync {
    fn evaluate(&self, net: usize, obs: &[f32]) -> (Vec<f32>, Vec<f32>);
}

impl<F> Evaluate for F
where
    F: Fn(usize, &[f32]) -> (Vec<f32>, Vec<f32>) + Sync,
{
    fn evaluate(&self, net: usize, obs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        self(net, obs)
    }
}

#[derive(Clone, Debug)]
pub struct ArenaConfig {
    /// Search simulations per move; 0 = searchless (best policy move).
    pub simulations: usize,
    /// Sequential Halving width at the root (with simulations).
    pub considered: usize,
    /// Games reaching this many plies (after the opening) are draws.
    pub max_plies: usize,
    /// Worker threads (0 = one per logical CPU).
    pub threads: usize,
}

impl Default for ArenaConfig {
    fn default() -> Self {
        ArenaConfig {
            simulations: 0,
            considered: 16,
            max_plies: 512,
            threads: 1,
        }
    }
}

impl ArenaConfig {
    fn search_config(&self) -> SearchConfig {
        SearchConfig {
            games: 1,
            simulations: self.simulations,
            max_plies: self.max_plies,
            gumbel: GumbelConfig {
                considered: self.considered.max(1),
                gumbel_scale: 0.0,
                ..Default::default()
            },
            seed: 0,
            threads: 1,
            noise_plies: 0,
            late_gumbel_scale: 0.0,
            fast_simulations: 0,
            full_prob: 1.0,
            fast_considered: 8,
        }
    }

    fn thread_pool(&self) -> rayon::ThreadPool {
        rayon::ThreadPoolBuilder::new()
            .num_threads(self.threads)
            .thread_name(|i| format!("peazero-arena-{i}"))
            .build()
            .expect("failed to start evaluation threads")
    }
}

/// A move a slot played this round.
struct Played {
    action: usize,
    outcome: Option<Outcome>,
}

/// One lock-step round: every active slot gets one network evaluation from
/// network `net_of[slot]`. Returns, per slot, the move it played (if its
/// search finished a move this round).
fn round<E: Evaluate + ?Sized>(
    slots: &mut [Slot],
    active: &[usize],
    net_of: &[usize],
    nets: usize,
    eval: &E,
    cfg: &SearchConfig,
    table: &Tables,
) -> Vec<Option<Played>> {
    let mut resp: Vec<Option<(Vec<f32>, [f32; 3])>> = (0..slots.len()).map(|_| None).collect();
    for net in 0..nets {
        let idx: Vec<usize> = active
            .iter()
            .copied()
            .filter(|&i| net_of[i] == net)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let mut obs = vec![0.0f32; idx.len() * OBS_SIZE];
        {
            let slots = &*slots;
            obs.par_chunks_exact_mut(OBS_SIZE)
                .zip(idx.par_iter())
                .for_each(|(o, &i)| encode_game(&slots[i].eval_game, o));
        }
        let (policy, value) = eval.evaluate(net, &obs);
        assert_eq!(policy.len(), idx.len() * NUM_ACTIONS, "policy output size");
        assert_eq!(value.len(), idx.len() * 3, "value output size");
        for (j, &i) in idx.iter().enumerate() {
            let p = &policy[j * NUM_ACTIONS..(j + 1) * NUM_ACTIONS];
            let logits = slots[i].eval_actions.iter().map(|&(a, _)| p[a]).collect();
            resp[i] = Some((logits, [value[j * 3], value[j * 3 + 1], value[j * 3 + 2]]));
        }
    }
    slots
        .par_iter_mut()
        .zip(resp.into_par_iter())
        .map(|(slot, r)| {
            let (logits, v) = r?;
            let before = slot.plies.len();
            let outcome = slot.respond(&logits, &v, cfg, table);
            let last = match &outcome {
                Some(o) => o.plies.last(),
                None if slot.plies.len() != before => slot.plies.last(),
                None => None,
            };
            let action = last?.action as usize;
            Some(Played { action, outcome })
        })
        .collect()
}

/// `n` distinct-enough openings: `plies` random legal moves from the start
/// position (re-drawn if the game ends inside the opening). Deterministic
/// for a seed, so every match of a run uses the same openings.
pub fn random_openings(n: usize, plies: usize, seed: u64) -> Vec<Game> {
    let mut rng = Rng::new(seed ^ 0x0BE7_1A65);
    (0..n)
        .map(|_| loop {
            let mut g = Game::default();
            for _ in 0..plies {
                let moves = legal_actions(g.position());
                if moves.is_empty() || g.result() != GameResult::Ongoing {
                    break;
                }
                g.push(moves[rng.below(moves.len() as u64) as usize].1);
            }
            if g.ply() == plies && g.result() == GameResult::Ongoing {
                break g;
            }
        })
        .collect()
}

#[derive(Clone, Debug)]
pub struct MatchGame {
    pub opening: usize,
    /// Network A (index 0) had white.
    pub a_white: bool,
    /// 1 = A won, 0.5 = draw, 0 = B won.
    pub score_a: f32,
    pub termination: Termination,
    /// Plies played after the opening.
    pub plies: usize,
}

/// Play every opening twice (A white, then B white) between network 0 (A)
/// and network 1 (B). Games are returned in the order opening 0 A-white,
/// opening 0 B-white, opening 1 A-white, ...
pub fn play_match<E: Evaluate + ?Sized>(
    openings: &[Game],
    cfg: &ArenaConfig,
    eval: &E,
) -> Vec<MatchGame> {
    let scfg = cfg.search_config();
    let table = Tables::new(&scfg);
    let pool = cfg.thread_pool();
    pool.install(|| {
        let n = openings.len() * 2;
        let a_white: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
        let mut slots: Vec<Slot> = (0..n)
            .map(|i| Slot::with_game(openings[i / 2].clone(), Rng::new(i as u64)))
            .collect();
        let mut results: Vec<Option<MatchGame>> = vec![None; n];
        let mut net_of = vec![0usize; n];
        loop {
            let active: Vec<usize> = (0..n).filter(|&i| results[i].is_none()).collect();
            if active.is_empty() {
                break;
            }
            for &i in &active {
                let white = slots[i].game.position().side_to_move() == Color::White;
                net_of[i] = if white == a_white[i] { 0 } else { 1 };
            }
            let played = round(&mut slots, &active, &net_of, 2, eval, &scfg, &table);
            for (i, p) in played.into_iter().enumerate() {
                let Some(Played {
                    outcome: Some(o), ..
                }) = p
                else {
                    continue;
                };
                let score_white = match o.result {
                    1 => 1.0,
                    -1 => 0.0,
                    _ => 0.5,
                };
                results[i] = Some(MatchGame {
                    opening: i / 2,
                    a_white: a_white[i],
                    score_a: if a_white[i] {
                        score_white
                    } else {
                        1.0 - score_white
                    },
                    termination: o.termination,
                    plies: o.plies.len(),
                });
            }
        }
        results.into_iter().map(Option::unwrap).collect()
    })
}

/// A puzzle: the solver is to move in `game`; `solution` alternates solver
/// and opponent moves, starting and ending with a solver move.
#[derive(Clone, Debug)]
pub struct Puzzle {
    pub game: Game,
    pub solution: Vec<Move>,
}

impl Puzzle {
    /// Lichess puzzle database format: `fen` is the position *before* the
    /// opponent's move, `moves` the UCI moves starting with that move.
    pub fn from_lichess(fen: &str, moves: &str) -> Result<Puzzle, String> {
        let mut game = Game::from_fen(fen).map_err(|e| format!("bad FEN {fen:?}: {e}"))?;
        let ucis: Vec<&str> = moves.split_whitespace().collect();
        if ucis.len() < 2 {
            return Err(format!("puzzle needs at least two moves: {moves:?}"));
        }
        if !game.push_uci(ucis[0]) {
            return Err(format!("illegal move {} in {fen}", ucis[0]));
        }
        let mut scratch = game.clone();
        let mut solution = Vec::with_capacity(ucis.len() - 1);
        for u in &ucis[1..] {
            let m = scratch
                .position()
                .parse_uci_move(u)
                .ok_or_else(|| format!("illegal move {u} in puzzle {fen} {moves}"))?;
            scratch.push(m);
            solution.push(m);
        }
        Ok(Puzzle { game, solution })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PuzzleResult {
    /// The solver's first move was right.
    pub first_move: bool,
    /// Every solver move was right (a different checkmating move counts as
    /// right, like on Lichess).
    pub solved: bool,
}

pub fn solve_puzzles<E: Evaluate + ?Sized>(
    puzzles: &[Puzzle],
    cfg: &ArenaConfig,
    eval: &E,
) -> Vec<PuzzleResult> {
    let scfg = cfg.search_config();
    let table = Tables::new(&scfg);
    let pool = cfg.thread_pool();
    pool.install(|| {
        let n = puzzles.len();
        let mut slots: Vec<Slot> = puzzles
            .iter()
            .enumerate()
            .map(|(i, p)| Slot::with_game(p.game.clone(), Rng::new(i as u64)))
            .collect();
        let expected_action =
            |slot: &Slot, m: Move| move_to_action(m, slot.game.position().side_to_move());
        let mut step = vec![0usize; n];
        let mut expected: Vec<usize> = (0..n)
            .map(|i| expected_action(&slots[i], puzzles[i].solution[0]))
            .collect();
        let mut results = vec![PuzzleResult::default(); n];
        let mut done = vec![false; n];
        let net_of = vec![0usize; n];
        loop {
            let active: Vec<usize> = (0..n).filter(|&i| !done[i]).collect();
            if active.is_empty() {
                break;
            }
            let played = round(&mut slots, &active, &net_of, 1, eval, &scfg, &table);
            for (i, p) in played.into_iter().enumerate() {
                let Some(p) = p else { continue };
                let sol = &puzzles[i].solution;
                let mate = p
                    .outcome
                    .as_ref()
                    .is_some_and(|o| o.termination == Termination::Checkmate);
                let correct = p.action == expected[i] || mate;
                if step[i] == 0 {
                    results[i].first_move = correct;
                }
                if !correct {
                    done[i] = true;
                    continue;
                }
                if mate || step[i] + 1 >= sol.len() {
                    results[i].solved = true;
                    done[i] = true;
                    continue;
                }
                if p.outcome.is_some() {
                    // Right move but the game ended (a draw) before the line did.
                    done[i] = true;
                    continue;
                }
                // The solver's move is on the board; play the opponent's reply.
                let slot = &mut slots[i];
                slot.game.push(sol[step[i] + 1]);
                step[i] += 2;
                if step[i] >= sol.len() || slot.game.result() != GameResult::Ongoing {
                    results[i].solved = step[i] >= sol.len();
                    done[i] = true;
                    continue;
                }
                slot.request_root();
                expected[i] = expected_action(slot, sol[step[i]]);
            }
        }
        results
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use peazero_core::Position;

    /// Material balance for the side to move as W/D/L logits.
    fn material_value(pos: &Position) -> [f32; 3] {
        let fen = pos.to_fen();
        let mut s = 0i32;
        for c in fen.split(' ').next().unwrap().chars() {
            let v = match c.to_ascii_lowercase() {
                'p' => 1,
                'n' | 'b' => 3,
                'r' => 5,
                'q' => 9,
                _ => 0,
            };
            s += if c.is_ascii_uppercase() { v } else { -v };
        }
        if pos.side_to_move() == Color::Black {
            s = -s;
        }
        let v = (s as f32 / 10.0).tanh();
        [v * 3.0, 0.0, -v * 3.0]
    }

    /// The same material value computed from the observation planes
    /// (0-5: side to move's P N B R Q K, 6-11: the opponent's).
    fn material_from_obs(o: &[f32]) -> [f32; 3] {
        let values = [1.0, 3.0, 3.0, 5.0, 9.0, 0.0];
        let mut s = 0.0f32;
        for (p, v) in values.iter().enumerate() {
            let mine: f32 = o[p * 64..(p + 1) * 64].iter().sum();
            let theirs: f32 = o[(p + 6) * 64..(p + 7) * 64].iter().sum();
            s += v * (mine - theirs);
        }
        let v = (s / 10.0).tanh();
        [v * 3.0, 0.0, -v * 3.0]
    }

    /// Uniform policy; network 0 knows material, network 1 thinks
    /// everything is equal.
    fn oracle(net: usize, obs: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let n = obs.len() / OBS_SIZE;
        let policy = vec![0.0f32; n * NUM_ACTIONS];
        let mut value = Vec::with_capacity(n * 3);
        for o in obs.chunks_exact(OBS_SIZE) {
            if net == 0 {
                value.extend_from_slice(&material_from_obs(o));
            } else {
                value.extend_from_slice(&[0.0, 0.0, 0.0]);
            }
        }
        (policy, value)
    }

    #[test]
    fn obs_material_matches_fen_material() {
        peazero_core::init();
        for fen in [
            "4k3/8/7r/8/8/8/8/2B1K3 w - - 0 1",
            "4k3/8/7r/8/8/8/8/2B1K3 b - - 0 1",
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        ] {
            let g = Game::from_fen(fen).unwrap();
            let mut o = vec![0.0; OBS_SIZE];
            encode_game(&g, &mut o);
            let a = material_from_obs(&o);
            let b = material_value(g.position());
            assert!((a[0] - b[0]).abs() < 1e-5, "{fen}: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn parses_lichess_puzzle() {
        peazero_core::init();
        // First line of the Lichess puzzle database (puzzle 00008).
        let p = Puzzle::from_lichess(
            "r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24",
            "f2g3 e6e7 b2b1 b3c1 b1c1 h6c1",
        )
        .unwrap();
        assert_eq!(p.solution.len(), 5);
        assert_eq!(p.game.position().side_to_move(), Color::White);
        assert!(Puzzle::from_lichess("8/8/8/8/8/8/8/K6k w - - 0 1", "a1a2").is_err());
    }

    #[test]
    fn solves_tactics_with_search() {
        peazero_core::init();
        let puzzles = vec![
            // Back-rank mate in one: the opponent's move h7h6, then Ra8#.
            Puzzle::from_lichess("6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1", "h7h6 a1a8").unwrap(),
            // Free rook: Bxh6 (after the opponent's Ke8-d8).
            Puzzle::from_lichess("4k3/8/7r/8/8/8/8/2B1K3 b - - 0 1", "e8d8 c1h6").unwrap(),
            // A "solution" that just gives the rook away: must fail.
            Puzzle::from_lichess("4k3/8/7r/8/8/8/8/2B1K3 b - - 0 1", "e8d8 c1d2").unwrap(),
        ];
        let cfg = ArenaConfig {
            simulations: 64,
            threads: 2,
            ..Default::default()
        };
        let r = solve_puzzles(&puzzles, &cfg, &oracle);
        assert!(r[0].solved && r[0].first_move, "{r:?}");
        assert!(r[1].solved && r[1].first_move, "{r:?}");
        assert!(!r[2].solved && !r[2].first_move, "{r:?}");
    }

    #[test]
    fn multi_move_puzzle_follows_the_line() {
        peazero_core::init();
        // After ...Ra8: 1. Rd8+ Rxd8 (forced) 2. Rxd8#. Searchless, with a
        // policy for the two solution moves.
        let fen = "6k1/r4ppp/8/8/8/8/3R1PPP/3R2K1 b - - 0 1";
        let p = Puzzle::from_lichess(fen, "a7a8 d2d8 a8d8 d1d8").unwrap();
        let first = move_to_action(p.solution[0], Color::White);
        let second = move_to_action(p.solution[2], Color::White);
        let eval = move |_net: usize, obs: &[f32]| {
            let n = obs.len() / OBS_SIZE;
            let mut policy = vec![0.0f32; n * NUM_ACTIONS];
            for j in 0..n {
                policy[j * NUM_ACTIONS + first] = 10.0;
                policy[j * NUM_ACTIONS + second] = 5.0;
            }
            (policy, vec![0.0f32; n * 3])
        };
        let r = solve_puzzles(std::slice::from_ref(&p), &ArenaConfig::default(), &eval);
        assert_eq!(
            r[0],
            PuzzleResult {
                first_move: true,
                solved: true
            }
        );
        // Preferring Rd7 is a different, non-mating move: fails.
        let pos = p.game.position();
        let wrong = move_to_action(pos.parse_uci_move("d2d7").unwrap(), Color::White);
        let eval_wrong = move |_net: usize, obs: &[f32]| {
            let n = obs.len() / OBS_SIZE;
            let mut policy = vec![0.0f32; n * NUM_ACTIONS];
            for j in 0..n {
                policy[j * NUM_ACTIONS + wrong] = 10.0;
            }
            (policy, vec![0.0f32; n * 3])
        };
        let r = solve_puzzles(&[p], &ArenaConfig::default(), &eval_wrong);
        assert_eq!(
            r[0],
            PuzzleResult {
                first_move: false,
                solved: false
            }
        );
    }

    #[test]
    fn openings_are_deterministic_and_legal() {
        peazero_core::init();
        let a = random_openings(20, 6, 7);
        let b = random_openings(20, 6, 7);
        for (x, y) in a.iter().zip(&b) {
            assert_eq!(x.position().to_fen(), y.position().to_fen());
            assert_eq!(x.ply(), 6);
            assert_eq!(x.result(), GameResult::Ongoing);
        }
        let c = random_openings(20, 6, 8);
        assert!(a
            .iter()
            .zip(&c)
            .any(|(x, y)| x.position().to_fen() != y.position().to_fen()));
    }

    #[test]
    fn match_is_paired_deterministic_and_favours_the_better_net() {
        peazero_core::init();
        let openings = random_openings(4, 4, 1);
        let cfg = ArenaConfig {
            simulations: 16,
            considered: 8,
            max_plies: 60,
            threads: 2,
        };
        let r1 = play_match(&openings, &cfg, &oracle);
        let r2 = play_match(&openings, &cfg, &oracle);
        assert_eq!(r1.len(), 8);
        for (i, g) in r1.iter().enumerate() {
            assert_eq!(g.opening, i / 2);
            assert_eq!(g.a_white, i % 2 == 0);
            assert!(g.plies <= 60);
        }
        let s1: Vec<f32> = r1.iter().map(|g| g.score_a).collect();
        let s2: Vec<f32> = r2.iter().map(|g| g.score_a).collect();
        assert_eq!(s1, s2, "matches must be deterministic");
        let score: f32 = s1.iter().sum();
        assert!(
            score >= 4.0,
            "material-aware net scored {score} / 8: {s1:?}"
        );
    }

    #[test]
    fn searchless_match_completes() {
        peazero_core::init();
        let openings = random_openings(3, 6, 2);
        let cfg = ArenaConfig {
            max_plies: 40,
            ..Default::default()
        };
        let r = play_match(&openings, &cfg, &oracle);
        assert_eq!(r.len(), 6);
    }
}
