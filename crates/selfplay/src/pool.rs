//! A pool of concurrent self-play games, stepped in lock-step so the network
//! always sees large batches. One step = one move in every game:
//!
//! ```text
//! observations()  → N root observations        → GPU: initial (h + f)
//! expand(policy)  → N·m action planes           → GPU: recurrent (g + f) on the
//!                                                  root latent repeated m times
//! finish(root_value, child_value)               → every game plays one move
//! take_finished() → completed games (data format v0 records)
//! ```
//! Finished games are immediately replaced by new ones, so the batch size
//! stays constant.
//!
//! Per-game work (encoding, Gumbel selection, making moves) runs on a rayon
//! thread pool (`PoolConfig::threads`). Every game slot has its own RNG, so
//! the games played do not depend on the number of threads.

use peazero_core::encoding::{
    encode_action_planes, encode_game, legal_actions, ACTION_SIZE, NUM_ACTIONS, OBS_SIZE,
};
use peazero_core::rng::Rng;
use peazero_core::{Game, GameResult};
use rayon::prelude::*;

use crate::gumbel::{self, GumbelConfig, RootCandidates};
use crate::record::{GameRecord, PlyRecord, Termination};
use crate::wdl_logits_to_q;

#[derive(Clone, Debug)]
pub struct PoolConfig {
    pub games: usize,
    /// Games reaching this many plies are adjudicated as draws.
    pub max_plies: usize,
    pub gumbel: GumbelConfig,
    pub seed: u64,
    /// Worker threads for the per-game work; 0 = one per logical CPU.
    pub threads: usize,
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            games: 256,
            max_plies: 512,
            gumbel: GumbelConfig::default(),
            seed: 0,
            threads: 1,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoolStats {
    pub games_finished: u64,
    pub plies: u64,
    /// Plies played after a full search (= training samples); equals
    /// `plies` without playout cap randomization.
    pub full_plies: u64,
    pub white_wins: u64,
    pub black_wins: u64,
    pub draws: u64,
    /// Indexed by `Termination as usize`.
    pub terminations: [u64; 6],
}

struct Slot {
    game: Game,
    plies: Vec<PlyRecord>,
    candidates: Option<RootCandidates>,
    rng: Rng,
}

impl Slot {
    fn new(rng: Rng) -> Slot {
        Slot {
            game: Game::default(),
            plies: Vec::with_capacity(256),
            candidates: None,
            rng,
        }
    }

    /// Start a new game in this slot, returning the finished one's plies.
    fn reset(&mut self) -> Vec<PlyRecord> {
        self.game = Game::default();
        self.candidates = None;
        std::mem::replace(&mut self.plies, Vec::with_capacity(256))
    }
}

/// What happened in one slot during `finish`.
struct Outcome {
    result: i8,
    termination: Termination,
    plies: Vec<PlyRecord>,
}

pub struct GamePool {
    cfg: PoolConfig,
    slots: Vec<Slot>,
    threads: rayon::ThreadPool,
    finished: Vec<GameRecord>,
    stats: PoolStats,
}

impl GamePool {
    pub fn new(cfg: PoolConfig) -> GamePool {
        assert!(cfg.games > 0 && cfg.gumbel.considered > 0);
        let threads = rayon::ThreadPoolBuilder::new()
            .num_threads(cfg.threads)
            .thread_name(|i| format!("peazero-selfplay-{i}"))
            .build()
            .expect("failed to start self-play threads");
        GamePool {
            slots: (0..cfg.games)
                .map(|i| {
                    let seed = (cfg.seed ^ 0x5E1F_9A1E)
                        .wrapping_add((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                    Slot::new(Rng::new(seed))
                })
                .collect(),
            threads,
            finished: Vec::new(),
            stats: PoolStats::default(),
            cfg,
        }
    }

    /// Number of worker threads actually used.
    pub fn num_threads(&self) -> usize {
        self.threads.current_num_threads()
    }

    pub fn num_games(&self) -> usize {
        self.slots.len()
    }

    /// `m`: dynamics evaluations per game per step (padded).
    pub fn considered(&self) -> usize {
        self.cfg.gumbel.considered
    }

    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Root observations, `num_games() * OBS_SIZE` floats.
    pub fn observations(&self, out: &mut [f32]) {
        assert_eq!(out.len(), self.slots.len() * OBS_SIZE);
        let slots = &self.slots;
        self.threads.install(|| {
            slots
                .par_iter()
                .zip(out.par_chunks_exact_mut(OBS_SIZE))
                .for_each(|(slot, chunk)| encode_game(&slot.game, chunk));
        });
    }

    /// Choose the considered actions from the root policy logits
    /// (`num_games() * NUM_ACTIONS`) and write their action planes into
    /// `out` (`num_games() * considered() * ACTION_SIZE`). Games with fewer
    /// legal moves than `m` are padded by repeating their first candidate.
    pub fn expand(&mut self, policy: &[f32], out: &mut [f32]) {
        let m = self.considered();
        assert_eq!(policy.len(), self.slots.len() * NUM_ACTIONS);
        assert_eq!(out.len(), self.slots.len() * m * ACTION_SIZE);
        let gcfg = &self.cfg.gumbel;
        let slots = &mut self.slots;
        self.threads.install(|| {
            slots
                .par_iter_mut()
                .zip(policy.par_chunks_exact(NUM_ACTIONS))
                .zip(out.par_chunks_exact_mut(m * ACTION_SIZE))
                .for_each(|((slot, row), planes)| {
                    let legal = legal_actions(slot.game.position());
                    let cand = gumbel::select(legal, row, gcfg, &mut slot.rng);
                    for (j, dst) in planes.chunks_exact_mut(ACTION_SIZE).enumerate() {
                        let k = cand.considered[j.min(cand.considered.len() - 1)];
                        encode_action_planes(cand.legal[k].0, dst);
                    }
                    slot.candidates = Some(cand);
                });
        });
    }

    /// Play one move in every game. `root_value`: `num_games() * 3` W/D/L
    /// logits of the roots; `child_value`: `num_games() * m * 3` W/D/L logits
    /// of the children (side to move = opponent). Returns games finished.
    pub fn finish(&mut self, root_value: &[f32], child_value: &[f32]) -> usize {
        let m = self.considered();
        assert_eq!(root_value.len(), self.slots.len() * 3);
        assert_eq!(child_value.len(), self.slots.len() * m * 3);
        let gcfg = &self.cfg.gumbel;
        let max_plies = self.cfg.max_plies;
        let slots = &mut self.slots;
        let outcomes: Vec<Option<Outcome>> = self.threads.install(|| {
            slots
                .par_iter_mut()
                .zip(root_value.par_chunks_exact(3))
                .zip(child_value.par_chunks_exact(m * 3))
                .map(|((slot, rv), cv)| {
                    let cand = slot
                        .candidates
                        .take()
                        .expect("finish() called without expand()");
                    let root_q = wdl_logits_to_q(rv);
                    let q: Vec<f32> = (0..cand.considered.len())
                        .map(|j| -wdl_logits_to_q(&cv[j * 3..j * 3 + 3]))
                        .collect();
                    let res = gumbel::finish(&cand, root_q, &q, gcfg);

                    let (action, mv) = cand.legal[res.chosen];
                    let targets: Vec<(usize, f32)> =
                        cand.legal.iter().map(|&(a, _)| a).zip(res.target).collect();
                    slot.plies.push(PlyRecord::new(
                        slot.game.position(),
                        slot.game.repetitions(),
                        action,
                        &targets,
                    ));
                    slot.game.push(mv);

                    let outcome = match Termination::from_result(slot.game.result()) {
                        Some(x) => Some(x),
                        None if slot.plies.len() >= max_plies => Some((0, Termination::MaxPlies)),
                        None => None,
                    };
                    outcome.map(|(result, termination)| Outcome {
                        result,
                        termination,
                        plies: slot.reset(),
                    })
                })
                .collect()
        });

        self.stats.plies += self.slots.len() as u64;
        self.stats.full_plies += self.slots.len() as u64;
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

    /// Positions recorded in games still in progress.
    pub fn plies_in_progress(&self) -> usize {
        self.slots.iter().map(|s| s.plies.len()).sum()
    }
}

/// Final result of a game as stored, for tests and tools.
pub fn replay_result(record: &GameRecord) -> GameResult {
    let mut g = Game::default();
    for p in &record.plies {
        let m = peazero_core::encoding::action_to_move(g.position(), p.action as usize)
            .expect("recorded action is not legal");
        g.push(m);
    }
    g.result()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::{read_chunk, write_chunk};

    /// Drive the pool with a deterministic "network" (random logits).
    fn run(pool: &mut GamePool, steps: usize, seed: u64) {
        let n = pool.num_games();
        let m = pool.considered();
        let mut rng = Rng::new(seed);
        let mut obs = vec![0.0; n * OBS_SIZE];
        let mut acts = vec![0.0; n * m * ACTION_SIZE];
        for _ in 0..steps {
            pool.observations(&mut obs);
            let policy: Vec<f32> = (0..n * NUM_ACTIONS)
                .map(|_| (rng.below(1000) as f32) / 500.0)
                .collect();
            pool.expand(&policy, &mut acts);
            let rv: Vec<f32> = (0..n * 3).map(|_| rng.below(100) as f32 / 50.0).collect();
            let cv: Vec<f32> = (0..n * m * 3)
                .map(|_| rng.below(100) as f32 / 50.0)
                .collect();
            pool.finish(&rv, &cv);
        }
    }

    #[test]
    fn games_finish_and_records_replay() {
        peazero_core::init();
        let mut pool = GamePool::new(PoolConfig {
            games: 8,
            max_plies: 120,
            ..Default::default()
        });
        run(&mut pool, 600, 5);
        let games = pool.take_finished();
        assert!(games.len() >= 8, "only {} games finished", games.len());

        // Round-trip through the file format.
        let bytes = write_chunk(&games);
        let back = read_chunk(&bytes).unwrap();
        assert_eq!(back.len(), games.len());

        for g in &back {
            // Replaying the recorded actions reproduces every recorded
            // position, and the stored result matches the rules.
            let mut game = Game::default();
            for p in &g.plies {
                assert_eq!(
                    p.position().to_fen().split(' ').take(4).collect::<Vec<_>>(),
                    game.position()
                        .to_fen()
                        .split(' ')
                        .take(4)
                        .collect::<Vec<_>>()
                );
                assert_eq!(p.repetitions as usize, game.repetitions());
                let sum: u32 = p.targets.iter().map(|&(_, q)| q as u32).sum();
                assert!((sum as i64 - 65535).abs() < 300, "targets sum {sum}");
                assert!(p.targets.iter().any(|&(a, _)| a == p.action));
                let mv = peazero_core::encoding::action_to_move(game.position(), p.action as usize)
                    .unwrap();
                game.push(mv);
            }
            match g.termination {
                Termination::MaxPlies => {
                    assert_eq!(g.plies.len(), 120);
                    assert_eq!(g.result, 0);
                }
                t => {
                    let (r, t2) = Termination::from_result(game.result()).unwrap();
                    assert_eq!((r, t2), (g.result, t));
                }
            }
        }
        let s = pool.stats();
        assert_eq!(s.games_finished as usize, back.len());
        assert_eq!(s.white_wins + s.black_wins + s.draws, s.games_finished);
    }

    #[test]
    fn thread_count_does_not_change_games() {
        peazero_core::init();
        let play = |threads| {
            let mut pool = GamePool::new(PoolConfig {
                games: 16,
                max_plies: 80,
                seed: 7,
                threads,
                ..Default::default()
            });
            run(&mut pool, 300, 9);
            write_chunk(&pool.take_finished())
        };
        let one = play(1);
        assert!(!one.is_empty());
        assert_eq!(one, play(4));
    }
}
