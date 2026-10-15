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

use peazero_core::encoding::{
    encode_action_planes, encode_game, legal_actions, ACTION_SIZE, NUM_ACTIONS, OBS_SIZE,
};
use peazero_core::rng::Rng;
use peazero_core::{Game, GameResult};

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
}

impl Default for PoolConfig {
    fn default() -> Self {
        PoolConfig {
            games: 256,
            max_plies: 512,
            gumbel: GumbelConfig::default(),
            seed: 0,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PoolStats {
    pub games_finished: u64,
    pub plies: u64,
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
}

impl Slot {
    fn new() -> Slot {
        Slot {
            game: Game::default(),
            plies: Vec::with_capacity(256),
            candidates: None,
        }
    }
}

pub struct GamePool {
    cfg: PoolConfig,
    slots: Vec<Slot>,
    rng: Rng,
    finished: Vec<GameRecord>,
    stats: PoolStats,
}

impl GamePool {
    pub fn new(cfg: PoolConfig) -> GamePool {
        assert!(cfg.games > 0 && cfg.gumbel.considered > 0);
        GamePool {
            slots: (0..cfg.games).map(|_| Slot::new()).collect(),
            rng: Rng::new(cfg.seed ^ 0x5E1F_9A1E),
            finished: Vec::new(),
            stats: PoolStats::default(),
            cfg,
        }
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
        for (slot, chunk) in self.slots.iter().zip(out.chunks_exact_mut(OBS_SIZE)) {
            encode_game(&slot.game, chunk);
        }
    }

    /// Choose the considered actions from the root policy logits
    /// (`num_games() * NUM_ACTIONS`) and write their action planes into
    /// `out` (`num_games() * considered() * ACTION_SIZE`). Games with fewer
    /// legal moves than `m` are padded by repeating their first candidate.
    pub fn expand(&mut self, policy: &[f32], out: &mut [f32]) {
        let m = self.considered();
        assert_eq!(policy.len(), self.slots.len() * NUM_ACTIONS);
        assert_eq!(out.len(), self.slots.len() * m * ACTION_SIZE);
        for (i, slot) in self.slots.iter_mut().enumerate() {
            let legal = legal_actions(slot.game.position());
            let row = &policy[i * NUM_ACTIONS..(i + 1) * NUM_ACTIONS];
            let cand = gumbel::select(legal, row, &self.cfg.gumbel, &mut self.rng);
            for j in 0..m {
                let k = cand.considered[j.min(cand.considered.len() - 1)];
                let action = cand.legal[k].0;
                let start = (i * m + j) * ACTION_SIZE;
                encode_action_planes(action, &mut out[start..start + ACTION_SIZE]);
            }
            slot.candidates = Some(cand);
        }
    }

    /// Play one move in every game. `root_value`: `num_games() * 3` W/D/L
    /// logits of the roots; `child_value`: `num_games() * m * 3` W/D/L logits
    /// of the children (side to move = opponent). Returns games finished.
    pub fn finish(&mut self, root_value: &[f32], child_value: &[f32]) -> usize {
        let m = self.considered();
        assert_eq!(root_value.len(), self.slots.len() * 3);
        assert_eq!(child_value.len(), self.slots.len() * m * 3);
        let mut done = 0;
        for i in 0..self.slots.len() {
            let slot = &mut self.slots[i];
            let cand = slot
                .candidates
                .take()
                .expect("finish() called without expand()");
            let root_q = wdl_logits_to_q(&root_value[i * 3..i * 3 + 3]);
            let q: Vec<f32> = (0..cand.considered.len())
                .map(|j| -wdl_logits_to_q(&child_value[(i * m + j) * 3..(i * m + j) * 3 + 3]))
                .collect();
            let res = gumbel::finish(&cand, root_q, &q, &self.cfg.gumbel);

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
            self.stats.plies += 1;

            let outcome = match Termination::from_result(slot.game.result()) {
                Some(x) => Some(x),
                None if slot.plies.len() >= self.cfg.max_plies => Some((0, Termination::MaxPlies)),
                None => None,
            };
            if let Some((result, termination)) = outcome {
                let finished = std::mem::replace(slot, Slot::new());
                self.finished.push(GameRecord {
                    plies: finished.plies,
                    result,
                    termination,
                });
                self.stats.games_finished += 1;
                self.stats.terminations[termination as usize] += 1;
                match result {
                    1 => self.stats.white_wins += 1,
                    -1 => self.stats.black_wins += 1,
                    _ => self.stats.draws += 1,
                }
                done += 1;
            }
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
}
