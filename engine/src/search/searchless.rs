//! Searchless: one network evaluation, play the legal move with the highest
//! policy logit (the Sprint 3 engine).

use std::time::Instant;

use peazero_core::{Game, Move};

use super::{NetSearch, SearchContext};
use crate::engine::SearchResult;
use crate::policy::{move_probabilities, q_to_cp, wdl_string, PolicyNet};

pub struct Searchless;

impl NetSearch for Searchless {
    fn search(&mut self, net: &mut PolicyNet, game: &Game, ctx: &SearchContext) -> SearchResult {
        let start = Instant::now();
        let eval = net.evaluate(game);
        let limits = ctx.limits;

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

        if ctx.verbose_move_stats {
            for (m, p, logit) in move_probabilities(&eval.moves) {
                println!(
                    "info string {m:<6} P: {:6.2}%  logit: {logit:7.3}",
                    p * 100.0
                );
            }
        }

        let [w, _, l] = eval.wdl;
        let ms = start.elapsed().as_millis();
        let wdl = if ctx.show_wdl {
            wdl_string(eval.wdl)
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
