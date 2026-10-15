//! Sprint 1 engine: plays a uniformly random legal move.

use std::sync::atomic::AtomicBool;

use peazero_core::rng::Rng;
use peazero_core::{Game, Move, MoveList};

use crate::engine::{CommonOptions, Engine, SearchLimits, SearchResult};

pub struct RandomEngine {
    /// 0 = seed from the clock (different games every run).
    seed: u64,
    rng: Rng,
}

impl RandomEngine {
    pub fn new() -> RandomEngine {
        RandomEngine {
            seed: 0,
            rng: Rng::from_time(),
        }
    }

    fn reseed(&mut self) {
        self.rng = if self.seed == 0 {
            Rng::from_time()
        } else {
            Rng::new(self.seed)
        };
    }
}

impl Engine for RandomEngine {
    fn new_game(&mut self) {
        // With a fixed seed, every game is reproducible from `ucinewgame`.
        if self.seed != 0 {
            self.reseed();
        }
    }

    fn uci_options(&self) -> Vec<String> {
        vec!["option name Seed type spin default 0 min 0 max 2147483647".to_string()]
    }

    fn set_option(&mut self, name: &str, value: &str) -> bool {
        if name.eq_ignore_ascii_case("seed") {
            if let Ok(v) = value.parse::<u64>() {
                self.seed = v;
                self.reseed();
            }
            return true;
        }
        false
    }

    fn search(
        &mut self,
        game: &Game,
        limits: &SearchLimits,
        _: &CommonOptions,
        _stop: &AtomicBool,
    ) -> SearchResult {
        let start = std::time::Instant::now();
        let mut list = MoveList::new();
        game.position().generate_legal(&mut list);

        let mut candidates: Vec<Move> = list.iter().copied().collect();
        if !limits.searchmoves.is_empty() {
            let restricted: Vec<Move> = candidates
                .iter()
                .copied()
                .filter(|m| limits.searchmoves.contains(&m.to_uci()))
                .collect();
            if !restricted.is_empty() {
                candidates = restricted;
            }
        }

        if candidates.is_empty() {
            println!("info depth 0 score cp 0");
            return SearchResult {
                best: Move::NULL,
                ponder: None,
            };
        }

        let best = candidates[self.rng.below(candidates.len() as u64) as usize];
        println!(
            "info depth 1 seldepth 1 multipv 1 score cp 0 nodes {} nps 0 time {} pv {}",
            candidates.len(),
            start.elapsed().as_millis(),
            best
        );
        SearchResult { best, ponder: None }
    }
}
