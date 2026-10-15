//! Inference spike: check tract matches PyTorch, then measure CPU speed.
//!
//!     cargo run --release -p peazero-nn --bin nnbench -- models/dummy-64x6 [batches...]

use std::time::Instant;

use peazero_core::encoding::{encode_action_planes, encode_observation, ACTION_SIZE, OBS_SIZE};
use peazero_core::Position;
use peazero_nn::MuZeroNet;

fn number_list(json: &str, key: &str, section: &str) -> Vec<f64> {
    // Tiny ad-hoc extraction to avoid a JSON dependency in the spike.
    let s = &json[json.find(&format!("\"{section}\"")).expect("section")..];
    let s = &s[s.find(&format!("\"{key}\"")).expect("key")..];
    let s = &s[s.find('[').unwrap() + 1..];
    s[..s.find(']').unwrap()]
        .split(',')
        .map(|x| x.trim().parse().unwrap())
        .collect()
}

fn number(json: &str, key: &str, section: &str) -> f64 {
    let s = &json[json.find(&format!("\"{section}\"")).expect("section")..];
    let s = &s[s.find(&format!("\"{key}\"")).expect("key") + key.len() + 3..];
    let end = s.find([',', '}', '\n']).unwrap();
    s[..end].trim().parse().unwrap()
}

fn main() -> peazero_nn::TractResult<()> {
    peazero_core::init();
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = args
        .first()
        .map(String::as_str)
        .unwrap_or("models/dummy-64x6");
    let batches: Vec<usize> = if args.len() > 1 {
        args[1..].iter().map(|b| b.parse().unwrap()).collect()
    } else {
        vec![1, 8, 32]
    };

    // ---- correctness vs PyTorch reference
    let reference = std::fs::read_to_string(format!("{dir}/reference.json"))?;
    let action: usize = {
        let s = &reference[reference.find("\"action\"").unwrap() + 9..];
        s[..s.find(',').unwrap()].trim().parse().unwrap()
    };
    let net = MuZeroNet::load(dir, 1, true)?;
    let mut obs = vec![0.0f32; OBS_SIZE];
    encode_observation(&Position::startpos(), 0, &mut obs);
    let mut act = vec![0.0f32; ACTION_SIZE];
    encode_action_planes(action, &mut act);
    let a = net.initial(&obs)?;
    let b = net.recurrent(&a.latent, &act)?;
    let mut max_err = 0.0f64;
    for (section, out) in [("initial", &a), ("recurrent", &b)] {
        let head = number_list(&reference, "policy_head", section);
        let value = number_list(&reference, "value", section);
        let sum = number(&reference, "policy_sum", section);
        for (i, r) in head.iter().enumerate() {
            max_err = max_err.max((out.policy[i] as f64 - r).abs());
        }
        for (i, r) in value.iter().enumerate() {
            max_err = max_err.max((out.value[i] as f64 - r).abs());
        }
        let got_sum: f64 = out.policy.iter().map(|&x| x as f64).sum();
        max_err = max_err.max((got_sum - sum).abs() / (1.0 + sum.abs()));
    }
    println!(
        "{dir}: channels={} max |tract - pytorch| = {max_err:.2e}",
        net.channels()
    );
    assert!(max_err < 1e-3, "tract output does not match PyTorch");

    // ---- speed
    for &batch in &batches {
        let net = MuZeroNet::load(dir, batch, true)?;
        let obs = vec![0.0f32; batch * OBS_SIZE];
        let init = net.initial(&obs)?;
        let mut actions = vec![0.0f32; batch * ACTION_SIZE];
        for i in 0..batch {
            encode_action_planes(
                64 + 12,
                &mut actions[i * ACTION_SIZE..(i + 1) * ACTION_SIZE],
            );
        }
        // warm-up
        for _ in 0..3 {
            net.recurrent(&init.latent, &actions)?;
        }
        let iters = (2000 / batch).clamp(20, 400);
        let t = Instant::now();
        for _ in 0..iters {
            net.recurrent(&init.latent, &actions)?;
        }
        let rec = t.elapsed().as_secs_f64();
        let t = Instant::now();
        for _ in 0..iters {
            net.initial(&obs)?;
        }
        let ini = t.elapsed().as_secs_f64();
        println!(
            "  batch {batch:>3}: recurrent {:>8.0} evals/s ({:>6.2} ms/call)   initial {:>8.0} evals/s",
            (iters * batch) as f64 / rec,
            rec * 1000.0 / iters as f64,
            (iters * batch) as f64 / ini,
        );
    }
    Ok(())
}
