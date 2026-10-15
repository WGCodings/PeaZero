//! Root-only Gumbel MuZero (see docs/MUZERO.md §2 and mctx).
//!
//! Per move:
//! 1. `select`: Gumbel noise `g` on the legal-move logits; consider the
//!    top `m` actions by `g + logits`.
//! 2. The caller evaluates each considered action once with the dynamics +
//!    prediction networks: `q(a) = -value(child)` (child value is from the
//!    opponent's point of view).
//! 3. `finish`: completed Q (mixed value for unconsidered moves), min-max
//!    rescale, `σ(q) = (c_visit + max N) · c_scale · q`, play
//!    `argmax_{considered} g + logits + σ(q)`, train towards
//!    `softmax(logits + σ(completedQ))`.
//!
//! With one visit per considered action, Sequential Halving degenerates to a
//! single round, which is exactly what mctx does when `n == m`.

use peazero_core::rng::Rng;
use peazero_core::Move;

#[derive(Clone, Debug)]
pub struct GumbelConfig {
    /// Maximum number of considered actions `m` (= dynamics calls per move).
    pub considered: usize,
    /// Scale of the Gumbel noise (1.0 in self-play, 0.0 = deterministic).
    pub gumbel_scale: f32,
    pub c_visit: f32,
    pub c_scale: f32,
}

impl Default for GumbelConfig {
    fn default() -> Self {
        GumbelConfig {
            considered: 16,
            gumbel_scale: 1.0,
            c_visit: 50.0,
            c_scale: 0.1,
        }
    }
}

/// Root state between `select` and `finish`.
#[derive(Clone, Debug)]
pub struct RootCandidates {
    /// Legal moves and their policy indices.
    pub legal: Vec<(usize, Move)>,
    /// Policy logits of the legal moves (same order), max-normalised.
    pub logits: Vec<f32>,
    pub gumbel: Vec<f32>,
    /// Indices into `legal` of the considered actions, best `g + logit` first.
    pub considered: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct RootResult {
    /// Index into `legal` of the move to play.
    pub chosen: usize,
    /// Improved policy over `legal` (sums to 1).
    pub target: Vec<f32>,
}

fn sample_gumbel(rng: &mut Rng) -> f32 {
    // u in (0, 1), never exactly 0 or 1
    let u = ((rng.next_u64() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
    (-(-u.ln()).ln()) as f32
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = x.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.into_iter().map(|v| v / s).collect()
}

/// Step 1: pick the considered actions. `policy` is the full 4672-logit row.
pub fn select(
    legal: Vec<(usize, Move)>,
    policy: &[f32],
    cfg: &GumbelConfig,
    rng: &mut Rng,
) -> RootCandidates {
    assert!(!legal.is_empty());
    let raw: Vec<f32> = legal.iter().map(|&(a, _)| policy[a]).collect();
    let max = raw.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let logits: Vec<f32> = raw.iter().map(|&l| l - max).collect();
    let gumbel: Vec<f32> = (0..legal.len())
        .map(|_| cfg.gumbel_scale * sample_gumbel(rng))
        .collect();
    let mut order: Vec<usize> = (0..legal.len()).collect();
    order.sort_by(|&a, &b| (gumbel[b] + logits[b]).total_cmp(&(gumbel[a] + logits[a])));
    order.truncate(cfg.considered.min(legal.len()).max(1));
    RootCandidates {
        legal,
        logits,
        gumbel,
        considered: order,
    }
}

/// Step 3. `root_q`: network value of the root (`W - L`, side to move).
/// `q`: one value per considered action, from the root side's view.
pub fn finish(c: &RootCandidates, root_q: f32, q: &[f32], cfg: &GumbelConfig) -> RootResult {
    assert_eq!(q.len(), c.considered.len());
    let n = c.legal.len();
    let prior = softmax(&c.logits);

    // Mixed value for unvisited actions (mctx `_compute_mixed_value`).
    // Accumulated in f64 so equal Qs give a mixed value equal to them.
    let visits = c.considered.len() as f64;
    let sum_prior: f64 = c
        .considered
        .iter()
        .map(|&i| prior[i].max(f32::MIN_POSITIVE) as f64)
        .sum();
    let weighted_q: f64 = c
        .considered
        .iter()
        .zip(q)
        .map(|(&i, &qi)| prior[i].max(f32::MIN_POSITIVE) as f64 * qi as f64)
        .sum::<f64>()
        / sum_prior;
    let mixed = ((root_q as f64 + visits * weighted_q) / (visits + 1.0)) as f32;

    let mut completed = vec![mixed; n];
    for (&i, &qi) in c.considered.iter().zip(q) {
        completed[i] = qi;
    }
    // Rescale to [0, 1] over legal actions. mctx uses an epsilon of 1e-8;
    // in f32 the mixed value alone can differ from equal Qs by ~1e-8, which
    // that epsilon would blow up to a full-scale preference. 1e-4 keeps
    // rounding noise from being treated as a real difference.
    let lo = completed.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = completed.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let span = (hi - lo).max(1e-4);
    let max_visit = 1.0; // every considered action visited exactly once
    let scale = (cfg.c_visit + max_visit) * cfg.c_scale;
    let sigma: Vec<f32> = completed.iter().map(|&v| scale * (v - lo) / span).collect();

    let chosen = *c
        .considered
        .iter()
        .max_by(|&&a, &&b| {
            (c.gumbel[a] + c.logits[a] + sigma[a])
                .total_cmp(&(c.gumbel[b] + c.logits[b] + sigma[b]))
        })
        .unwrap();
    let improved: Vec<f32> = (0..n).map(|i| c.logits[i] + sigma[i]).collect();
    RootResult {
        chosen,
        target: softmax(&improved),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peazero_core::encoding::legal_actions;
    use peazero_core::Position;

    fn setup(m: usize, scale: f32) -> (RootCandidates, GumbelConfig) {
        let pos = Position::startpos();
        let legal = legal_actions(&pos);
        let policy = vec![0.0f32; 4672];
        let cfg = GumbelConfig {
            considered: m,
            gumbel_scale: scale,
            ..Default::default()
        };
        let mut rng = Rng::new(1);
        (select(legal, &policy, &cfg, &mut rng), cfg)
    }

    #[test]
    fn considers_m_distinct_moves() {
        let (c, _) = setup(16, 1.0);
        assert_eq!(c.considered.len(), 16);
        let mut s = c.considered.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 16);
        let (c, _) = setup(64, 1.0);
        assert_eq!(c.considered.len(), 20); // only 20 legal moves
    }

    #[test]
    fn best_q_wins_without_noise() {
        // Uniform prior, no Gumbel noise: the move with the highest Q must be
        // chosen and get the largest target probability.
        let (c, cfg) = setup(8, 0.0);
        let mut q = vec![-0.5f32; c.considered.len()];
        q[5] = 0.9;
        let r = finish(&c, 0.0, &q, &cfg);
        assert_eq!(r.chosen, c.considered[5]);
        let best = r.target.iter().copied().fold(0.0, f32::max);
        assert_eq!(r.target[c.considered[5]], best);
        assert!((r.target.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn equal_q_keeps_prior() {
        // All Q equal → σ is constant → target == prior.
        let pos = Position::startpos();
        let legal = legal_actions(&pos);
        let mut policy = vec![0.0f32; 4672];
        policy[legal[3].0] = 2.0;
        let cfg = GumbelConfig::default();
        let mut rng = Rng::new(2);
        let c = select(legal, &policy, &cfg, &mut rng);
        let q = vec![0.3f32; c.considered.len()];
        let r = finish(&c, 0.3, &q, &cfg);
        let prior = softmax(&c.logits);
        for (a, b) in r.target.iter().zip(&prior) {
            assert!(
                (a - b).abs() < 1e-5,
                "target {a} prior {b} logits {:?}",
                c.logits
            );
        }
    }

    #[test]
    fn single_legal_move() {
        let pos = Position::from_fen("7k/8/8/8/8/8/6q1/7K w - - 0 1").unwrap();
        let legal = legal_actions(&pos);
        assert_eq!(legal.len(), 1);
        let cfg = GumbelConfig::default();
        let c = select(legal, &vec![0.0; 4672], &cfg, &mut Rng::new(3));
        let r = finish(&c, 0.0, &[0.1], &cfg);
        assert_eq!(r.chosen, 0);
        assert!((r.target[0] - 1.0).abs() < 1e-6);
    }
}
