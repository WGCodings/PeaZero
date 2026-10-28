//! Search methods on top of the policy/value network.
//!
//! Every method implements [`NetSearch`]; `PolicyEngine` (policy.rs) owns
//! the network and picks the method with the UCI option `SearchMode`.
//! Adding a method = a new file here, a variant in [`SearchMode`] and an
//! arm in [`SearchMode::create`].
//!
//! * `Mcts` (default) — plain PUCT MCTS: policy priors, network value at
//!   new leaves, most visited move.
//! * `AlphaBeta` — iterative-deepening alpha-beta: policy for move
//!   ordering, network value at the leaves.
//! * `Searchless` — one evaluation, play the best legal policy move.

use std::sync::atomic::AtomicBool;

use peazero_core::Game;

use crate::engine::{CommonOptions, SearchLimits, SearchResult};
use crate::policy::PolicyNet;
pub use params::SearchParams;

pub mod alphabeta;
pub mod mcts;
pub mod params;
pub mod searchless;

/// Everything a search needs besides the network and the position.
pub struct SearchContext<'a> {
    pub limits: &'a SearchLimits,
    pub common: &'a CommonOptions,
    pub stop: &'a AtomicBool,
    pub show_wdl: bool,
    pub verbose_move_stats: bool,
    /// Tunable search parameters (UCI options, see params.rs).
    pub params: &'a SearchParams,
}

pub trait NetSearch: Send {
    /// Called on `ucinewgame` (clear tables etc.).
    fn new_game(&mut self) {}

    /// Choose a move; may print `info` lines, must not print `bestmove`.
    fn search(&mut self, net: &mut PolicyNet, game: &Game, ctx: &SearchContext) -> SearchResult;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchMode {
    Searchless,
    AlphaBeta,
    Mcts,
}

impl SearchMode {
    pub const ALL: [SearchMode; 3] = [
        SearchMode::Searchless,
        SearchMode::AlphaBeta,
        SearchMode::Mcts,
    ];

    pub fn name(self) -> &'static str {
        match self {
            SearchMode::Searchless => "Searchless",
            SearchMode::AlphaBeta => "AlphaBeta",
            SearchMode::Mcts => "Mcts",
        }
    }

    pub fn parse(s: &str) -> Option<SearchMode> {
        SearchMode::ALL
            .into_iter()
            .find(|m| m.name().eq_ignore_ascii_case(s))
    }

    pub fn create(self) -> Box<dyn NetSearch> {
        match self {
            SearchMode::Searchless => Box::new(searchless::Searchless),
            SearchMode::AlphaBeta => Box::new(alphabeta::AlphaBeta),
            SearchMode::Mcts => Box::new(mcts::Mcts::default()),
        }
    }

    /// `option name SearchMode type combo default ... var ...`
    pub fn uci_option(default: SearchMode) -> String {
        let vars: String = SearchMode::ALL
            .iter()
            .map(|m| format!(" var {}", m.name()))
            .collect();
        format!(
            "option name SearchMode type combo default {}{vars}",
            default.name()
        )
    }
}
