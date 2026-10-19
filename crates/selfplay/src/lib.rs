//! Self-play for PeaZero (Sprint 4: root-only Gumbel, engine stays searchless).
//!
//! * [`gumbel`]  — Gumbel-Top-k root selection and improved-policy targets,
//!   ported from DeepMind mctx (`gumbel_muzero_policy` with every considered
//!   action visited exactly once through the dynamics network).
//! * [`record`]  — self-play data format v0 (writer + reader).
//! * [`pool`]    — a pool of concurrent games driven in lock-step batches so
//!   the GPU always sees large batches.
//! * [`replay`]  — replay buffer that samples training batches (K-step
//!   unrolls) and encodes observations with the engine's own encoder.
//!
//! No Python or GPU code here; `crates/py` wraps this for the trainer.

pub mod arena;
pub mod gumbel;
pub mod pool;
pub mod record;
pub mod replay;
pub mod search;
pub mod viriformat;

/// W/D/L probabilities from W/D/L logits.
pub fn wdl_logits_to_probs(logits: &[f32]) -> [f32; 3] {
    let m = logits[0].max(logits[1]).max(logits[2]);
    let e = [
        (logits[0] - m).exp(),
        (logits[1] - m).exp(),
        (logits[2] - m).exp(),
    ];
    let s = e[0] + e[1] + e[2];
    [e[0] / s, e[1] / s, e[2] / s]
}

/// Scalar value `W - L` from W/D/L logits (side-to-move perspective).
pub fn wdl_logits_to_q(logits: &[f32]) -> f32 {
    let m = logits[0].max(logits[1]).max(logits[2]);
    let e = [
        (logits[0] - m).exp(),
        (logits[1] - m).exp(),
        (logits[2] - m).exp(),
    ];
    (e[0] - e[2]) / (e[0] + e[1] + e[2])
}
