//! Replay buffer + training batch sampler (docs/MUZERO.md §3.1).
//!
//! For a sampled position `t` of a game with `T` plies and unroll steps
//! `k = 0..=K`:
//! * observation: encoded ply `t` (same encoder as the engine)
//! * actions `a_{t+k}` for k < K: the move actually played, or a random
//!   action past the end of the game (its targets are masked anyway)
//! * policy target at step k: recorded improved policy at ply `t+k`
//!   (masked when `t+k >= T`)
//! * value target at step k: final result from the side to move's view at
//!   ply `t+k` as a W/D/L class (0 = win, 1 = draw, 2 = loss); the terminal
//!   position `t+k == T` is included, beyond it is masked.

use std::collections::VecDeque;

use peazero_core::encoding::{
    encode_action_planes, encode_observation, ACTION_SIZE, NUM_ACTIONS, OBS_SIZE,
};
use peazero_core::rng::Rng;
use peazero_core::Color;

use crate::record::{read_chunk, GameRecord};

/// Upper bound on legal moves; policy targets are padded to this width.
pub const MAX_LEGAL: usize = 218;

pub struct Batch {
    pub batch: usize,
    pub unroll: usize,
    /// `B × OBS_SIZE`
    pub obs: Vec<f32>,
    /// `B × K × ACTION_SIZE`
    pub actions: Vec<f32>,
    /// `B × (K+1) × MAX_LEGAL` action indices (padding: 0 with prob 0)
    pub policy_index: Vec<i64>,
    /// `B × (K+1) × MAX_LEGAL` target probabilities
    pub policy_prob: Vec<f32>,
    /// `B × (K+1)`
    pub policy_mask: Vec<f32>,
    /// `B × (K+1)` W/D/L class
    pub value_target: Vec<i64>,
    /// `B × (K+1)`
    pub value_mask: Vec<f32>,
    /// `B × (K+1) × 3` search value (W/D/L, side to move) where recorded
    pub value_search: Vec<f32>,
    /// `B × (K+1)`: 1 where `value_search` is valid
    pub value_search_mask: Vec<f32>,
}

pub struct Replay {
    games: VecDeque<GameRecord>,
    /// cumulative ply counts, `cum[i]` = plies in games[..=i]
    cum: Vec<usize>,
    capacity: usize,
    rng: Rng,
    total_added: u64,
}

fn value_class(result: i8, stm: Color) -> i64 {
    match result {
        0 => 1,
        1 if stm == Color::White => 0,
        -1 if stm == Color::Black => 0,
        _ => 2,
    }
}

impl Replay {
    /// `capacity`: maximum number of positions kept (oldest games evicted).
    pub fn new(capacity: usize, seed: u64) -> Replay {
        Replay {
            games: VecDeque::new(),
            cum: Vec::new(),
            capacity,
            rng: Rng::new(seed ^ 0x2E91_A7B3),
            total_added: 0,
        }
    }

    pub fn positions(&self) -> usize {
        self.cum.last().copied().unwrap_or(0)
    }

    pub fn games(&self) -> usize {
        self.games.len()
    }

    /// Positions ever added (including evicted ones).
    pub fn total_added(&self) -> u64 {
        self.total_added
    }

    pub fn add_games(&mut self, games: Vec<GameRecord>) {
        for g in games {
            if g.plies.is_empty() {
                continue;
            }
            self.total_added += g.plies.len() as u64;
            self.games.push_back(g);
        }
        let mut total: usize = self.games.iter().map(|g| g.plies.len()).sum();
        while total > self.capacity && self.games.len() > 1 {
            total -= self.games.pop_front().unwrap().plies.len();
        }
        self.cum.clear();
        let mut acc = 0;
        for g in &self.games {
            acc += g.plies.len();
            self.cum.push(acc);
        }
    }

    pub fn add_chunk(&mut self, bytes: &[u8]) -> Result<usize, String> {
        let games = read_chunk(bytes)?;
        let n = games.len();
        self.add_games(games);
        Ok(n)
    }

    fn pick(&mut self) -> (usize, usize) {
        let total = self.positions();
        let idx = self.rng.below(total as u64) as usize;
        let g = self.cum.partition_point(|&c| c <= idx);
        let start = if g == 0 { 0 } else { self.cum[g - 1] };
        (g, idx - start)
    }

    pub fn sample(&mut self, batch: usize, unroll: usize) -> Batch {
        assert!(self.positions() > 0, "replay buffer is empty");
        let k1 = unroll + 1;
        let mut b = Batch {
            batch,
            unroll,
            obs: vec![0.0; batch * OBS_SIZE],
            actions: vec![0.0; batch * unroll * ACTION_SIZE],
            policy_index: vec![0; batch * k1 * MAX_LEGAL],
            policy_prob: vec![0.0; batch * k1 * MAX_LEGAL],
            policy_mask: vec![0.0; batch * k1],
            value_target: vec![1; batch * k1],
            value_mask: vec![0.0; batch * k1],
            value_search: vec![0.0; batch * k1 * 3],
            value_search_mask: vec![0.0; batch * k1],
        };
        for i in 0..batch {
            let (gi, t) = self.pick();
            let random_actions: Vec<usize> = (0..unroll)
                .map(|_| self.rng.below(NUM_ACTIONS as u64) as usize)
                .collect();
            let game = &self.games[gi];
            let plies = &game.plies;
            let total = plies.len();

            let root = &plies[t];
            encode_observation(
                &root.position(),
                root.repetitions as usize,
                &mut b.obs[i * OBS_SIZE..(i + 1) * OBS_SIZE],
            );

            for k in 0..k1 {
                let j = t + k;
                let slot = i * k1 + k;
                if k < unroll {
                    let action = if j < total {
                        plies[j].action as usize
                    } else {
                        random_actions.get(k).copied().unwrap_or(0)
                    };
                    let start = (i * unroll + k) * ACTION_SIZE;
                    encode_action_planes(action, &mut b.actions[start..start + ACTION_SIZE]);
                }
                if j < total {
                    let p = &plies[j];
                    let sum: f32 = p
                        .targets
                        .iter()
                        .map(|&(_, q)| q as f32)
                        .sum::<f32>()
                        .max(1.0);
                    for (n, &(a, q)) in p.targets.iter().enumerate().take(MAX_LEGAL) {
                        b.policy_index[slot * MAX_LEGAL + n] = a as i64;
                        b.policy_prob[slot * MAX_LEGAL + n] = q as f32 / sum;
                    }
                    b.policy_mask[slot] = 1.0;
                    b.value_target[slot] = value_class(game.result, p.stm);
                    b.value_mask[slot] = 1.0;
                    if let Some(wdl) = p.value_wdl() {
                        b.value_search[slot * 3..slot * 3 + 3].copy_from_slice(&wdl);
                        b.value_search_mask[slot] = 1.0;
                    }
                } else if j == total {
                    let stm = !plies[total - 1].stm;
                    b.value_target[slot] = value_class(game.result, stm);
                    b.value_mask[slot] = 1.0;
                }
            }
        }
        b
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pool::{GamePool, PoolConfig};
    use peazero_core::encoding::{action_squares, encode_game};
    use peazero_core::Game;

    fn make_games(n_games: usize, max_plies: usize) -> Vec<GameRecord> {
        let mut pool = GamePool::new(PoolConfig {
            games: 4,
            max_plies,
            ..Default::default()
        });
        let n = pool.num_games();
        let m = pool.considered();
        let mut rng = Rng::new(11);
        let mut obs = vec![0.0; n * OBS_SIZE];
        let mut acts = vec![0.0; n * m * ACTION_SIZE];
        let mut out = Vec::new();
        while out.len() < n_games {
            pool.observations(&mut obs);
            let policy: Vec<f32> = (0..n * NUM_ACTIONS)
                .map(|_| rng.below(100) as f32 / 30.0)
                .collect();
            pool.expand(&policy, &mut acts);
            pool.finish(&vec![0.0; n * 3], &vec![0.0; n * m * 3]);
            out.extend(pool.take_finished());
        }
        out
    }

    #[test]
    fn sample_matches_games() {
        peazero_core::init();
        let games = make_games(6, 60);
        let mut replay = Replay::new(1_000_000, 3);
        replay.add_games(games.clone());
        assert_eq!(
            replay.positions(),
            games.iter().map(|g| g.plies.len()).sum::<usize>()
        );

        let b = replay.sample(64, 5);
        assert_eq!(b.obs.len(), 64 * OBS_SIZE);
        for i in 0..64 {
            let pm: f32 = b.policy_mask[i * 6..i * 6 + 6].iter().sum();
            let vm: f32 = b.value_mask[i * 6..i * 6 + 6].iter().sum();
            assert!(pm >= 1.0 && vm >= pm && vm <= pm + 1.0);
            for k in 0..6 {
                let slot = i * 6 + k;
                if b.policy_mask[slot] == 1.0 {
                    let s: f32 = b.policy_prob[slot * MAX_LEGAL..(slot + 1) * MAX_LEGAL]
                        .iter()
                        .sum();
                    assert!((s - 1.0).abs() < 1e-3);
                }
            }
        }

        // Observations must be identical to encoding the replayed game.
        let g = &games[0];
        let mut game = Game::default();
        let mut expected = vec![0.0; OBS_SIZE];
        let mut got = vec![0.0; OBS_SIZE];
        for p in &g.plies {
            encode_game(&game, &mut expected);
            encode_observation(&p.position(), p.repetitions as usize, &mut got);
            assert_eq!(expected, got);
            game.push(
                peazero_core::encoding::action_to_move(game.position(), p.action as usize).unwrap(),
            );
        }
    }

    #[test]
    fn value_targets_follow_side_to_move() {
        assert_eq!(value_class(1, Color::White), 0);
        assert_eq!(value_class(1, Color::Black), 2);
        assert_eq!(value_class(-1, Color::Black), 0);
        assert_eq!(value_class(0, Color::Black), 1);
        // A checkmate game: the terminal position is a loss for its side to move.
        let mut replay = Replay::new(1000, 1);
        let mut g = Game::default();
        let mut plies = Vec::new();
        for m in ["f2f3", "e7e5", "g2g4", "d8h4"] {
            let mv = g.position().parse_uci_move(m).unwrap();
            let a = peazero_core::encoding::move_to_action(mv, g.position().side_to_move());
            plies.push(crate::record::PlyRecord::new(
                g.position(),
                0,
                a,
                &[(a, 1.0)],
            ));
            g.push(mv);
        }
        replay.add_games(vec![GameRecord {
            plies,
            result: -1,
            termination: crate::record::Termination::Checkmate,
        }]);
        for _ in 0..20 {
            let b = replay.sample(1, 5);
            // Find t from the observation's own-pawn count is awkward; instead
            // check consistency: every valid value target alternates W/L.
            let classes: Vec<i64> = (0..6)
                .filter(|&k| b.value_mask[k] == 1.0)
                .map(|k| b.value_target[k])
                .collect();
            for w in classes.windows(2) {
                assert_eq!(w[0] + w[1], 2, "{classes:?}"); // win/loss alternate
            }
            assert_eq!(*classes.last().unwrap(), 2); // mated side (white) to move at the end
            let _ = action_squares(0);
        }
    }
}
