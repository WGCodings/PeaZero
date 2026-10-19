//! MuZero network inference on CPU.
//!
//! * [`fast`] (x86-64): hand-written SIMD inference of the `initial` network
//!   (AVX-512 / AVX2 / SSE2, picked at runtime). The engine uses this.
//! * [`MuZeroNet`] (feature `tract`, on by default): both graphs through
//!   `tract`, any batch size; the engine falls back to it on other CPUs
//!   (ARM), and tests use it as the reference.
//!
//! A network is two ONNX graphs exported by `trainer/peazero/model.py`:
//!   initial   : obs (N,21,8,8)                     → latent, policy (N,4672), value (N,3)
//!   recurrent : latent (N,C,8,8), action (N,5,8,8) → latent, policy, value
//!
//! They ship as one `.pzn` file (what OpenBench uploads and `EVALFILE`
//! embeds):
//!
//! ```text
//! "PZNN" | u32 format version (1) | u32 ENCODING_VERSION
//! u64 initial.onnx length   | bytes
//! u64 recurrent.onnx length | bytes
//! ```
//! (all integers little-endian)

#[cfg(target_arch = "x86_64")]
pub mod fast;
pub mod onnx;
#[cfg(feature = "tract")]
mod tract_net;

#[cfg(feature = "tract")]
pub use tract_net::{MuZeroNet, NetOutput, TractResult};

use peazero_core::encoding::ENCODING_VERSION;

pub const PZN_MAGIC: &[u8; 4] = b"PZNN";
pub const PZN_VERSION: u32 = 1;

/// The two ONNX blobs inside a `.pzn` file.
pub struct PznParts<'a> {
    pub initial: &'a [u8],
    pub recurrent: &'a [u8],
}

pub fn parse_pzn(bytes: &[u8]) -> Result<PznParts<'_>, String> {
    let mut rest = bytes;
    let mut take = |n: usize| -> Result<&[u8], String> {
        if rest.len() < n {
            return Err("truncated .pzn file".into());
        }
        let (a, b) = rest.split_at(n);
        rest = b;
        Ok(a)
    };
    if take(4)? != PZN_MAGIC {
        return Err("not a PeaZero network file (bad magic)".into());
    }
    let u32le = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    let version = u32le(take(4)?);
    let encoding = u32le(take(4)?);
    if version != PZN_VERSION {
        return Err(format!(
            "unsupported .pzn version {version} (expected {PZN_VERSION})"
        ));
    }
    if encoding != ENCODING_VERSION {
        return Err(format!(
            "network uses encoding v{encoding}, engine uses v{ENCODING_VERSION}"
        ));
    }
    let len = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap()) as usize;
    let n = len(take(8)?);
    let initial = take(n)?;
    let n = len(take(8)?);
    let recurrent = take(n)?;
    Ok(PznParts { initial, recurrent })
}

/// Softmax of 3 W/D/L logits.
pub fn wdl_probs(value_logits: &[f32]) -> [f32; 3] {
    let m = value_logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, f32::max);
    let e: Vec<f32> = value_logits.iter().map(|&x| (x - m).exp()).collect();
    let s: f32 = e.iter().sum();
    [e[0] / s, e[1] / s, e[2] / s]
}
