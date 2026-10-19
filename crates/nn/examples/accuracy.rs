//! Accuracy of the Winograd (half-precision weights) path against the exact
//! direct path, on positions from random games.
//!     cargo run --release -p peazero-nn --example accuracy -- net.pzn
use std::sync::Arc;

use peazero_core::encoding::{encode_game, legal_actions, NUM_ACTIONS, OBS_SIZE};
use peazero_core::rng::Rng;
use peazero_core::{Game, GameResult};
use peazero_nn::fast::{FastNet, FastWeights};
use peazero_nn::wdl_probs;

fn main() {
    peazero_core::init();
    let path = std::env::args().nth(1).unwrap();
    let w = Arc::new(FastWeights::from_pzn(&std::fs::read(&path).unwrap()).unwrap());
    let (mut exact, mut fast) = (FastNet::new(w.clone()), FastNet::new(w));
    exact.set_winograd(false);
    fast.set_winograd(true);
    let mut rng = Rng::new(1);
    let (mut n, mut same_top, mut max_dl, mut sum_dl, mut max_dq, mut sum_dq, mut max_dp) =
        (0usize, 0usize, 0f32, 0f64, 0f32, 0f64, 0f32);
    let mut obs = vec![0.0f32; OBS_SIZE];
    let (mut pe, mut pf) = (vec![0.0f32; NUM_ACTIONS], vec![0.0f32; NUM_ACTIONS]);
    let (mut ve, mut vf) = ([0f32; 3], [0f32; 3]);
    while n < 2000 {
        let mut g = Game::default();
        let plies = rng.below(120) as usize;
        for _ in 0..plies {
            let moves = legal_actions(g.position());
            if moves.is_empty() || g.result() != GameResult::Ongoing {
                break;
            }
            g.push(moves[rng.below(moves.len() as u64) as usize].1);
        }
        let legal = legal_actions(g.position());
        if legal.is_empty() || g.result() != GameResult::Ongoing {
            continue;
        }
        encode_game(&g, &mut obs);
        exact.evaluate(&obs, &mut pe, &mut ve);
        fast.evaluate(&obs, &mut pf, &mut vf);
        let top = |p: &[f32]| {
            legal
                .iter()
                .max_by(|a, b| p[a.0].total_cmp(&p[b.0]))
                .unwrap()
                .0
        };
        same_top += usize::from(top(&pe) == top(&pf));
        // softmax over legal moves
        let probs = |p: &[f32]| {
            let m = legal
                .iter()
                .map(|a| p[a.0])
                .fold(f32::NEG_INFINITY, f32::max);
            let e: Vec<f32> = legal.iter().map(|a| (p[a.0] - m).exp()).collect();
            let s: f32 = e.iter().sum();
            e.into_iter().map(|x| x / s).collect::<Vec<f32>>()
        };
        let (qe, qf) = (probs(&pe), probs(&pf));
        for ((a, x), y) in legal.iter().zip(&qe).zip(&qf) {
            let dl = (pe[a.0] - pf[a.0]).abs();
            max_dl = max_dl.max(dl);
            sum_dl += dl as f64;
            max_dp = max_dp.max((x - y).abs());
        }
        let q = |v: &[f32; 3]| {
            let p = wdl_probs(v);
            p[0] - p[2]
        };
        let dq = (q(&ve) - q(&vf)).abs();
        max_dq = max_dq.max(dq);
        sum_dq += dq as f64;
        n += 1;
    }
    println!(
        "{n} positions: same best move {:.2}%, logit diff mean {:.1e} max {:.1e}, move probability diff max {:.4}, value (W-L) diff mean {:.1e} max {:.1e}",
        100.0 * same_top as f64 / n as f64,
        sum_dl / (n as f64 * 30.0),
        max_dl,
        max_dp,
        sum_dq / n as f64,
        max_dq
    );
}
