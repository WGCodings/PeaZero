//! The network engine: owns the policy/value network and hands each `go` to
//! the search method selected with the UCI option `SearchMode`
//! (see search/mod.rs): `Mcts` (default; PUCT with policy priors and network
//! values), `AlphaBeta` (policy move ordering, value at the leaves) or
//! `Searchless` (one evaluation, best legal policy move).

use std::sync::atomic::AtomicBool;

use peazero_core::encoding::{encode_game, legal_actions, NUM_ACTIONS, OBS_SIZE};
use peazero_core::{Game, Move};
use peazero_nn::wdl_probs;

use crate::engine::{CommonOptions, Engine, SearchLimits, SearchResult};
use crate::search::{NetSearch, SearchContext, SearchMode};

/// The network compiled into the binary (see build.rs / `EVALFILE`).
pub static EMBEDDED_NET: &[u8] = include_bytes!(env!("PEAZERO_EVALFILE"));

/// Result of evaluating one position with the policy/value heads.
pub struct Evaluation {
    /// Legal moves with their raw policy logits.
    pub moves: Vec<(Move, f32)>,
    /// Win / draw / loss probabilities for the side to move.
    pub wdl: [f32; 3],
}

/// x86-64: hand-written SIMD inference (peazero_nn::fast); other CPUs: tract.
#[cfg(target_arch = "x86_64")]
type Backend = peazero_nn::fast::FastNet;
#[cfg(not(target_arch = "x86_64"))]
type Backend = peazero_nn::MuZeroNet;

pub struct PolicyNet {
    net: Backend,
    obs: Vec<f32>,
    policy: Vec<f32>,
}

impl PolicyNet {
    pub fn from_pzn(bytes: &[u8]) -> Result<PolicyNet, String> {
        #[cfg(target_arch = "x86_64")]
        let net = peazero_nn::fast::FastNet::new(std::sync::Arc::new(
            peazero_nn::fast::FastWeights::from_pzn(bytes)?,
        ));
        #[cfg(not(target_arch = "x86_64"))]
        let net = peazero_nn::MuZeroNet::from_pzn(bytes, 1, false).map_err(|e| format!("{e:#}"))?;
        Ok(PolicyNet {
            net,
            obs: vec![0.0; OBS_SIZE],
            policy: vec![0.0; NUM_ACTIONS],
        })
    }

    /// Which inference kernels are in use (for `info string`).
    pub fn backend_name(&self) -> String {
        #[cfg(target_arch = "x86_64")]
        {
            self.net.describe()
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            "tract".to_string()
        }
    }

    pub fn embedded() -> PolicyNet {
        PolicyNet::from_pzn(EMBEDDED_NET).expect("embedded network is invalid")
    }

    /// Evaluate the current position of `game`. Returns no moves if the
    /// game has no legal moves.
    pub fn evaluate(&mut self, game: &Game) -> Evaluation {
        let legal = legal_actions(game.position());
        encode_game(game, &mut self.obs);
        let mut value = [0.0f32; 3];
        #[cfg(target_arch = "x86_64")]
        self.net.evaluate(&self.obs, &mut self.policy, &mut value);
        #[cfg(not(target_arch = "x86_64"))]
        {
            let out = self
                .net
                .initial(&self.obs)
                .expect("network inference failed");
            self.policy.copy_from_slice(&out.policy);
            value.copy_from_slice(&out.value);
        }
        Evaluation {
            moves: legal.iter().map(|&(a, m)| (m, self.policy[a])).collect(),
            wdl: wdl_probs(&value),
        }
    }
}

/// Map an expected score `q = W - L` in [-1, 1] to centipawns
/// (same shape as Lc0's default conversion).
pub fn q_to_cp(q: f32) -> i32 {
    let q = q.clamp(-0.999, 0.999) as f64;
    (111.714_640_912 * (1.562_068_842_1 * q).tan()).round() as i32
}

/// ` wdl W D L` in permille, as used in `info` lines.
pub fn wdl_string([w, d, _]: [f32; 3]) -> String {
    let permille = |x: f32| (x * 1000.0).round() as i32;
    format!(
        " wdl {} {} {}",
        permille(w),
        permille(d),
        1000 - permille(w) - permille(d)
    )
}

/// Legal moves with their policy probability (softmax over the legal
/// moves' logits), most likely first: (move, probability, raw logit).
pub fn move_probabilities(moves: &[(Move, f32)]) -> Vec<(Move, f32, f32)> {
    let max = moves.iter().map(|m| m.1).fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = moves.iter().map(|m| (m.1 - max).exp()).sum();
    let mut out: Vec<(Move, f32, f32)> = moves
        .iter()
        .map(|&(m, logit)| (m, (logit - max).exp() / sum, logit))
        .collect();
    out.sort_by(|a, b| b.1.total_cmp(&a.1));
    out
}

pub struct PolicyEngine {
    net: PolicyNet,
    net_name: String,
    show_wdl: bool,
    verbose_move_stats: bool,
    mode: SearchMode,
    search: Box<dyn NetSearch>,
}

impl PolicyEngine {
    pub const DEFAULT_MODE: SearchMode = SearchMode::Mcts;

    pub fn new() -> PolicyEngine {
        PolicyEngine {
            net: PolicyNet::embedded(),
            net_name: "<embedded>".to_string(),
            show_wdl: true,
            verbose_move_stats: false,
            mode: Self::DEFAULT_MODE,
            search: Self::DEFAULT_MODE.create(),
        }
    }
}

impl Engine for PolicyEngine {
    fn new_game(&mut self) {
        self.search.new_game();
    }

    fn uci_options(&self) -> Vec<String> {
        vec![
            format!("option name EvalFile type string default {}", self.net_name),
            "option name UCI_ShowWDL type check default true".to_string(),
            "option name VerboseMoveStats type check default false".to_string(),
            SearchMode::uci_option(Self::DEFAULT_MODE),
        ]
    }

    fn set_option(&mut self, name: &str, value: &str) -> bool {
        match name.to_ascii_lowercase().as_str() {
            "evalfile" => {
                if value.is_empty() || value == "<embedded>" {
                    self.net = PolicyNet::embedded();
                    self.net_name = "<embedded>".into();
                } else {
                    match std::fs::read(value)
                        .map_err(|e| e.to_string())
                        .and_then(|b| PolicyNet::from_pzn(&b))
                    {
                        Ok(net) => {
                            println!(
                                "info string loaded network {value} ({} inference)",
                                net.backend_name()
                            );
                            self.net = net;
                            self.net_name = value.to_string();
                        }
                        Err(e) => println!("info string failed to load {value}: {e}"),
                    }
                }
                // cached evaluations belong to the old network
                self.search.new_game();
                true
            }
            "uci_showwdl" => {
                self.show_wdl = value.eq_ignore_ascii_case("true");
                true
            }
            "verbosemovestats" => {
                self.verbose_move_stats = value.eq_ignore_ascii_case("true");
                true
            }
            "searchmode" => {
                match SearchMode::parse(value) {
                    Some(mode) if mode != self.mode => {
                        self.mode = mode;
                        self.search = mode.create();
                    }
                    Some(_) => {}
                    None => println!("info string unknown SearchMode {value}"),
                }
                true
            }
            _ => false,
        }
    }

    fn search(
        &mut self,
        game: &Game,
        limits: &SearchLimits,
        common: &CommonOptions,
        stop: &AtomicBool,
    ) -> SearchResult {
        let ctx = SearchContext {
            limits,
            common,
            stop,
            show_wdl: self.show_wdl,
            verbose_move_stats: self.verbose_move_stats,
        };
        self.search.search(&mut self.net, game, &ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn play(engine: &mut PolicyEngine, plies: usize, limits: &SearchLimits) {
        let stop = AtomicBool::new(false);
        let mut game = Game::default();
        for _ in 0..plies {
            let r = engine.search(&game, limits, &CommonOptions::default(), &stop);
            if r.best.is_null() {
                break;
            }
            assert!(game.position().legal_moves().contains(&r.best));
            game.push(r.best);
        }
    }

    #[test]
    fn mcts_is_default_and_plays_legal_moves() {
        peazero_core::init();
        let mut engine = PolicyEngine::new();
        assert_eq!(engine.mode, SearchMode::Mcts);
        let limits = SearchLimits {
            nodes: Some(50),
            ..Default::default()
        };
        play(&mut engine, 10, &limits);
    }

    #[test]
    fn alphabeta_mode_plays_legal_moves() {
        peazero_core::init();
        let mut engine = PolicyEngine::new();
        assert!(engine.set_option("SearchMode", "AlphaBeta"));
        assert_eq!(engine.mode, SearchMode::AlphaBeta);
        let limits = SearchLimits {
            depth: Some(2),
            ..Default::default()
        };
        play(&mut engine, 6, &limits);
    }

    #[test]
    fn searchless_mode_plays_legal_moves() {
        peazero_core::init();
        let mut engine = PolicyEngine::new();
        assert!(engine.set_option("SearchMode", "searchless"));
        assert_eq!(engine.mode, SearchMode::Searchless);
        play(&mut engine, 40, &SearchLimits::default());
        assert!(engine.set_option("SearchMode", "AlphaBeta"));
        assert_eq!(engine.mode, SearchMode::AlphaBeta);
    }

    #[test]
    fn search_mode_option_lists_all_modes() {
        let line = SearchMode::uci_option(PolicyEngine::DEFAULT_MODE);
        assert_eq!(
            line,
            "option name SearchMode type combo default Mcts var Searchless var AlphaBeta var Mcts"
        );
    }

    #[test]
    fn move_probabilities_sum_to_one() {
        peazero_core::init();
        let mut net = PolicyNet::embedded();
        let eval = net.evaluate(&Game::default());
        let probs = move_probabilities(&eval.moves);
        assert_eq!(probs.len(), 20);
        let sum: f32 = probs.iter().map(|p| p.1).sum();
        assert!((sum - 1.0).abs() < 1e-4);
        assert!(probs.windows(2).all(|w| w[0].1 >= w[1].1));
    }

    #[test]
    fn cp_mapping_is_monotone_and_bounded() {
        assert_eq!(q_to_cp(0.0), 0);
        assert!(q_to_cp(0.5) > 0 && q_to_cp(-0.5) < 0);
        assert!(q_to_cp(0.9) > q_to_cp(0.5));
        assert!(q_to_cp(1.0).abs() < 100_000);
    }
}
