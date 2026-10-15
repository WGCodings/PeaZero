//! Searchless engine: one network evaluation per move, play the legal move
//! with the highest policy logit. No MCTS yet — this validates the network
//! pipeline end to end (encoding → inference → move choice) on OpenBench.

use std::sync::atomic::AtomicBool;
use std::time::Instant;

use peazero_core::encoding::{encode_game, legal_actions, OBS_SIZE};
use peazero_core::{Game, Move};
use peazero_nn::{wdl_probs, MuZeroNet};

use crate::engine::{CommonOptions, Engine, SearchLimits, SearchResult};

/// The network compiled into the binary (see build.rs / `EVALFILE`).
pub static EMBEDDED_NET: &[u8] = include_bytes!(env!("PEAZERO_EVALFILE"));

/// Result of evaluating one position with the policy/value heads.
pub struct Evaluation {
    /// Legal moves with their raw policy logits.
    pub moves: Vec<(Move, f32)>,
    /// Win / draw / loss probabilities for the side to move.
    pub wdl: [f32; 3],
}

pub struct PolicyNet {
    net: MuZeroNet,
    obs: Vec<f32>,
}

impl PolicyNet {
    pub fn from_pzn(bytes: &[u8]) -> Result<PolicyNet, String> {
        let net = MuZeroNet::from_pzn(bytes, 1, false).map_err(|e| format!("{e:#}"))?;
        Ok(PolicyNet {
            net,
            obs: vec![0.0; OBS_SIZE],
        })
    }

    pub fn embedded() -> PolicyNet {
        PolicyNet::from_pzn(EMBEDDED_NET).expect("embedded network is invalid")
    }

    /// Evaluate the current position of `game`. Returns no moves if the
    /// game has no legal moves.
    pub fn evaluate(&mut self, game: &Game) -> Evaluation {
        let legal = legal_actions(game.position());
        encode_game(game, &mut self.obs);
        let out = self
            .net
            .initial(&self.obs)
            .expect("network inference failed");
        Evaluation {
            moves: legal.iter().map(|&(a, m)| (m, out.policy[a])).collect(),
            wdl: wdl_probs(&out.value),
        }
    }
}

/// Map an expected score `q = W - L` in [-1, 1] to centipawns
/// (same shape as Lc0's default conversion).
pub fn q_to_cp(q: f32) -> i32 {
    let q = q.clamp(-0.999, 0.999) as f64;
    (111.714_640_912 * (1.562_068_842_1 * q).tan()).round() as i32
}

pub struct PolicyEngine {
    net: PolicyNet,
    net_name: String,
    show_wdl: bool,
}

impl PolicyEngine {
    pub fn new() -> PolicyEngine {
        PolicyEngine {
            net: PolicyNet::embedded(),
            net_name: "<embedded>".to_string(),
            show_wdl: true,
        }
    }
}

impl Engine for PolicyEngine {
    fn new_game(&mut self) {}

    fn uci_options(&self) -> Vec<String> {
        vec![
            format!("option name EvalFile type string default {}", self.net_name),
            "option name UCI_ShowWDL type check default true".to_string(),
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
                            self.net = net;
                            self.net_name = value.to_string();
                            println!("info string loaded network {value}");
                        }
                        Err(e) => println!("info string failed to load {value}: {e}"),
                    }
                }
                true
            }
            "uci_showwdl" => {
                self.show_wdl = value.eq_ignore_ascii_case("true");
                true
            }
            _ => false,
        }
    }

    fn search(
        &mut self,
        game: &Game,
        limits: &SearchLimits,
        _: &CommonOptions,
        _stop: &AtomicBool,
    ) -> SearchResult {
        let start = Instant::now();
        let eval = self.net.evaluate(game);

        let best = eval
            .moves
            .iter()
            .filter(|(m, _)| {
                limits.searchmoves.is_empty() || limits.searchmoves.contains(&m.to_uci())
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|&(m, _)| m);

        let Some(best) = best else {
            println!("info depth 0 score cp 0");
            return SearchResult {
                best: Move::NULL,
                ponder: None,
            };
        };

        let [w, d, l] = eval.wdl;
        let ms = start.elapsed().as_millis();
        let wdl = if self.show_wdl {
            let permille = |x: f32| (x * 1000.0).round() as i32;
            format!(
                " wdl {} {} {}",
                permille(w),
                permille(d),
                1000 - permille(w) - permille(d)
            )
        } else {
            String::new()
        };
        println!(
            "info depth 1 seldepth 1 multipv 1 score cp {}{wdl} nodes 1 nps {} time {ms} pv {best}",
            q_to_cp(w - l),
            1000 / ms.max(1)
        );
        SearchResult { best, ponder: None }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_net_plays_legal_moves() {
        peazero_core::init();
        let mut engine = PolicyEngine::new();
        let stop = AtomicBool::new(false);
        let mut game = Game::default();
        for _ in 0..40 {
            let r = engine.search(
                &game,
                &SearchLimits::default(),
                &CommonOptions::default(),
                &stop,
            );
            if r.best.is_null() {
                break;
            }
            assert!(game.position().legal_moves().contains(&r.best));
            game.push(r.best);
        }
    }

    #[test]
    fn cp_mapping_is_monotone_and_bounded() {
        assert_eq!(q_to_cp(0.0), 0);
        assert!(q_to_cp(0.5) > 0 && q_to_cp(-0.5) < 0);
        assert!(q_to_cp(0.9) > q_to_cp(0.5));
        assert!(q_to_cp(1.0).abs() < 100_000);
    }
}
