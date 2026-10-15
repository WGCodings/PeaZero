//! MuZero network inference on CPU with `tract` (pure Rust, no system
//! libraries, so OpenBench workers can build it with plain `make`).
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

use std::path::Path;
use std::sync::Arc;

use peazero_core::encoding::{ACTION_PLANES, ENCODING_VERSION, NUM_ACTIONS, OBS_PLANES};
use tract_onnx::prelude::tract_data::internal::{anyhow, bail};
use tract_onnx::prelude::*;

pub use tract_onnx::prelude::TractResult;

pub const PZN_MAGIC: &[u8; 4] = b"PZNN";
pub const PZN_VERSION: u32 = 1;

type Plan = Arc<TypedRunnableModel>;

pub struct NetOutput {
    /// `batch * channels * 64` floats.
    pub latent: Vec<f32>,
    /// `batch * 4672` logits (index = encoding::move_to_action).
    pub policy: Vec<f32>,
    /// `batch * 3` logits: win, draw, loss for the side to move.
    pub value: Vec<f32>,
}

/// The two ONNX blobs inside a `.pzn` file.
pub struct PznParts<'a> {
    pub initial: &'a [u8],
    pub recurrent: &'a [u8],
}

pub fn parse_pzn(bytes: &[u8]) -> TractResult<PznParts<'_>> {
    let mut rest = bytes;
    let mut take = |n: usize| -> TractResult<&[u8]> {
        if rest.len() < n {
            bail!("truncated .pzn file");
        }
        let (a, b) = rest.split_at(n);
        rest = b;
        Ok(a)
    };
    if take(4)? != PZN_MAGIC {
        bail!("not a PeaZero network file (bad magic)");
    }
    let u32le = |b: &[u8]| u32::from_le_bytes(b.try_into().unwrap());
    let version = u32le(take(4)?);
    let encoding = u32le(take(4)?);
    if version != PZN_VERSION {
        bail!("unsupported .pzn version {version} (expected {PZN_VERSION})");
    }
    if encoding != ENCODING_VERSION {
        bail!("network uses encoding v{encoding}, engine uses v{ENCODING_VERSION}");
    }
    let len = |b: &[u8]| u64::from_le_bytes(b.try_into().unwrap()) as usize;
    let n = len(take(8)?);
    let initial = take(n)?;
    let n = len(take(8)?);
    let recurrent = take(n)?;
    Ok(PznParts { initial, recurrent })
}

/// Runnable plan + the concrete shape of output 0 (latent).
fn load_plan(onnx: &[u8], facts: &[Vec<usize>]) -> TractResult<(Plan, Vec<usize>)> {
    let mut model = tract_onnx::onnx().model_for_read(&mut &onnx[..])?;
    for (i, shape) in facts.iter().enumerate() {
        model = model.with_input_fact(i, f32::fact(shape).into())?;
    }
    let typed = model.into_optimized()?;
    let latent_shape = typed
        .output_fact(0)?
        .shape
        .as_concrete()
        .map(|s| s.to_vec())
        .ok_or_else(|| anyhow!("latent output shape is not concrete"))?;
    Ok((typed.into_runnable()?, latent_shape))
}

pub struct MuZeroNet {
    initial: Plan,
    recurrent: Option<Plan>,
    channels: usize,
    batch: usize,
}

impl MuZeroNet {
    /// Build from `.pzn` bytes for a fixed batch size (tract optimises for
    /// the concrete shape). `with_recurrent = false` skips the dynamics graph
    /// (searchless play only needs `initial`).
    pub fn from_pzn(bytes: &[u8], batch: usize, with_recurrent: bool) -> TractResult<MuZeroNet> {
        let parts = parse_pzn(bytes)?;
        Self::from_onnx(
            parts.initial,
            with_recurrent.then_some(parts.recurrent),
            batch,
        )
    }

    pub fn from_onnx(
        initial: &[u8],
        recurrent: Option<&[u8]>,
        batch: usize,
    ) -> TractResult<MuZeroNet> {
        let (initial, latent_shape) = load_plan(initial, &[vec![batch, OBS_PLANES, 8, 8]])?;
        let channels = latent_shape[1];
        let recurrent = match recurrent {
            Some(bytes) => Some(
                load_plan(
                    bytes,
                    &[
                        vec![batch, channels, 8, 8],
                        vec![batch, ACTION_PLANES, 8, 8],
                    ],
                )?
                .0,
            ),
            None => None,
        };
        Ok(MuZeroNet {
            initial,
            recurrent,
            channels,
            batch,
        })
    }

    /// Load either a `.pzn` file or a directory with `initial.onnx` +
    /// `recurrent.onnx`.
    pub fn load(
        path: impl AsRef<Path>,
        batch: usize,
        with_recurrent: bool,
    ) -> TractResult<MuZeroNet> {
        let path = path.as_ref();
        if path.is_dir() {
            let initial = std::fs::read(path.join("initial.onnx"))?;
            let recurrent = if with_recurrent {
                Some(std::fs::read(path.join("recurrent.onnx"))?)
            } else {
                None
            };
            Self::from_onnx(&initial, recurrent.as_deref(), batch)
        } else {
            Self::from_pzn(&std::fs::read(path)?, batch, with_recurrent)
        }
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn batch(&self) -> usize {
        self.batch
    }

    pub fn latent_size(&self) -> usize {
        self.channels * 64
    }

    fn collect(&self, out: TVec<TValue>) -> TractResult<NetOutput> {
        let get = |i: usize| -> TractResult<Vec<f32>> {
            Ok(out[i]
                .to_plain_array_view::<f32>()?
                .iter()
                .copied()
                .collect())
        };
        let o = NetOutput {
            latent: get(0)?,
            policy: get(1)?,
            value: get(2)?,
        };
        debug_assert_eq!(o.policy.len(), self.batch * NUM_ACTIONS);
        Ok(o)
    }

    /// `obs`: `batch * OBS_SIZE` floats (see `encoding::encode_observation`).
    pub fn initial(&self, obs: &[f32]) -> TractResult<NetOutput> {
        let t = Tensor::from_shape(&[self.batch, OBS_PLANES, 8, 8], obs)?;
        self.collect(self.initial.run(tvec!(t.into()))?)
    }

    /// `latent`: `batch * latent_size()`; `actions`: `batch * ACTION_SIZE`.
    pub fn recurrent(&self, latent: &[f32], actions: &[f32]) -> TractResult<NetOutput> {
        let Some(plan) = &self.recurrent else {
            bail!("network was loaded without the recurrent (dynamics) graph");
        };
        let l = Tensor::from_shape(&[self.batch, self.channels, 8, 8], latent)?;
        let a = Tensor::from_shape(&[self.batch, ACTION_PLANES, 8, 8], actions)?;
        self.collect(plan.run(tvec!(l.into(), a.into()))?)
    }
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
