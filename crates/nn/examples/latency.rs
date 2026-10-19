//! Batch-1 latency of tract vs the hand-written kernels on one network.
//! (GFLOP/s* = direct-convolution FLOPs / time, so Winograd shows as an
//! "effective" rate above the CPU's peak.)
//!     cargo run --release -p peazero-nn --example latency -- net.pzn
use std::sync::Arc;
use std::time::Instant;

use peazero_core::encoding::{encode_game, NUM_ACTIONS, OBS_SIZE};
use peazero_nn::fast::{FastNet, FastWeights, Simd};

/// Best of 5 runs of `n` evaluations (µs per evaluation): robust to noise
/// from other processes.
fn time(mut f: impl FnMut(), n: usize) -> f64 {
    for _ in 0..n / 10 {
        f();
    }
    (0..5)
        .map(|_| {
            let t = Instant::now();
            for _ in 0..n {
                f();
            }
            t.elapsed().as_secs_f64() * 1e6 / n as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    peazero_core::init();
    let path = std::env::args().nth(1).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let tract = peazero_nn::MuZeroNet::from_pzn(&bytes, 1, false).unwrap();
    let w = Arc::new(FastWeights::from_pzn(&bytes).unwrap());
    let mut game = peazero_core::Game::default();
    for m in [
        "e2e4", "c7c5", "g1f3", "d7d6", "d2d4", "c5d4", "f3d4", "g8f6", "b1c3", "a7a6",
    ] {
        game.push_uci(m);
    }
    let mut obs = vec![0.0f32; OBS_SIZE];
    encode_game(&game, &mut obs);
    let macs = 21.0 * 64.0 * 9.0 * 64.0
        + (w.blocks() as f64) * 2.0 * 64.0 * 64.0 * 9.0 * 64.0
        + 64.0 * 32.0 * 64.0
        + 32.0 * 73.0 * 64.0
        + 64.0 * 8.0 * 64.0
        + 512.0 * 128.0
        + 384.0;
    let (mut p, mut v) = (vec![0.0f32; NUM_ACTIONS], [0.0f32; 3]);
    let want = tract.initial(&obs).unwrap();
    for (level, wino) in [Simd::Avx512, Simd::Avx2, Simd::Sse2]
        .into_iter()
        .flat_map(|l| [(l, true), (l, false)])
    {
        let mut net = FastNet::new(w.clone());
        net.set_simd(level);
        net.set_winograd(wino);
        if net.simd() != level {
            continue;
        }
        net.evaluate(&obs, &mut p, &mut v);
        let err = p
            .iter()
            .zip(&want.policy)
            .chain(v.iter().zip(&want.value))
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let us = time(|| net.evaluate(&obs, &mut p, &mut v), 1000);
        println!(
            "{:13}: {us:6.0} us/eval {:6.0} evals/s {:5.1} GFLOP/s* (max |diff| vs tract {err:.1e})",
            format!("{}{}", level.name(), if wino { "+wino" } else { "" }),
            1e6 / us,
            2.0 * macs / us / 1e3
        );
    }
    let us = time(
        || {
            tract.initial(&obs).unwrap();
        },
        1000,
    );
    println!(
        "{:13}: {us:6.0} us/eval {:6.0} evals/s {:5.1} GFLOP/s",
        "tract",
        1e6 / us,
        2.0 * macs / us / 1e3
    );
}
