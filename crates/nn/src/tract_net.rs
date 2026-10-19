//! Inference through `tract`: both graphs, any (fixed) batch size.

use std::path::Path;
use std::sync::Arc;

use peazero_core::encoding::{ACTION_PLANES, NUM_ACTIONS, OBS_PLANES};
use tract_onnx::prelude::tract_data::internal::{anyhow, bail};
use tract_onnx::prelude::*;

pub use tract_onnx::prelude::TractResult;

use crate::parse_pzn;

type Plan = Arc<TypedRunnableModel>;

pub struct NetOutput {
    /// `batch * channels * 64` floats.
    pub latent: Vec<f32>,
    /// `batch * 4672` logits (index = encoding::move_to_action).
    pub policy: Vec<f32>,
    /// `batch * 3` logits: win, draw, loss for the side to move.
    pub value: Vec<f32>,
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
        let parts = parse_pzn(bytes).map_err(|e| anyhow!(e))?;
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
