//! Writes encoding test vectors (JSON lines) to stdout. The Python trainer's
//! mirror of `encoding.rs` is tested against these; a Rust test checks that
//! the committed file is up to date.
//!
//!     cargo run --release -p peazero-core --example encoding_vectors > tests/encoding/vectors.jsonl

use peazero_core::encoding::{
    encode_action_planes, encode_game, legal_actions, ACTION_SIZE, ENCODING_VERSION, NUM_ACTIONS,
    OBS_SIZE,
};
use peazero_core::rng::Rng;
use peazero_core::{Game, GameResult};

fn main() {
    print!("{}", generate());
}

pub fn generate() -> String {
    peazero_core::init();
    let mut out = String::new();
    let mut emit = |game: &Game| {
        let mut obs = vec![0.0f32; OBS_SIZE];
        encode_game(game, &mut obs);
        let nonzero: Vec<String> = obs
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, v)| format!("[{i},{v}]"))
            .collect();
        let mut actions: Vec<(usize, String)> = legal_actions(game.position())
            .into_iter()
            .map(|(a, m)| (a, m.to_uci()))
            .collect();
        actions.sort();
        let actions: Vec<String> = actions
            .iter()
            .map(|(a, m)| format!("[\"{m}\",{a}]"))
            .collect();
        out.push_str(&format!(
            "{{\"version\":{ENCODING_VERSION},\"fen\":\"{}\",\"repetitions\":{},\"obs\":[{}],\"actions\":[{}]}}\n",
            game.position().to_fen(),
            game.repetitions(),
            nonzero.join(","),
            actions.join(",")
        ));
    };

    // 1. Perft suite positions (castling, ep, promotions, both colours).
    let suite = include_str!("../../../tests/perft/standard.epd");
    for line in suite.lines().filter(|l| !l.trim().is_empty()) {
        let fen = line.split(';').next().unwrap().trim();
        emit(&Game::from_fen(fen).unwrap());
    }

    // 2. Repetitions: knight shuffle back to the start position twice.
    let mut g = Game::default();
    emit(&g);
    for _ in 0..2 {
        for m in ["g1f3", "g8f6", "f3g1", "f6g8"] {
            g.push_uci(m);
            emit(&g);
        }
    }

    // 3. Action planes for a spread of indices (incl. off-board targets).
    let mut action_lines = String::new();
    for a in (0..NUM_ACTIONS)
        .step_by(37)
        .chain([0, 49 * 64, 4096 + 49, NUM_ACTIONS - 1])
    {
        let mut planes = vec![0.0f32; ACTION_SIZE];
        encode_action_planes(a, &mut planes);
        let nz: Vec<String> = planes
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(i, _)| i.to_string())
            .collect();
        action_lines.push_str(&format!(
            "{{\"version\":{ENCODING_VERSION},\"action\":{a},\"planes\":[{}]}}\n",
            nz.join(",")
        ));
    }

    // 4. Positions from seeded random games (halfmove clocks, odd material).
    let mut rng = Rng::new(7);
    for _ in 0..40 {
        let mut g = Game::default();
        let plies = 1 + rng.below(150) as usize;
        for _ in 0..plies {
            let acts = legal_actions(g.position());
            if acts.is_empty() || g.result() != GameResult::Ongoing {
                break;
            }
            g.push(acts[rng.below(acts.len() as u64) as usize].1);
        }
        emit(&g);
    }
    out.push_str(&action_lines);
    out
}
