//! Hand-written CPU inference for the `initial` network (representation +
//! prediction), batch 1, x86-64 only: AVX-512, AVX2/FMA or SSE2 kernels,
//! picked at runtime. (Other CPUs use tract.)
//! Weights come straight from the ONNX graph in a `.pzn` file (BatchNorm is
//! already folded into the convolutions by the export).
//!
//! The residual-block convolutions use Winograd F(2×2, 3×3) by default
//! (winograd.rs: 2.25× fewer multiply-adds, weights stored as half floats so
//! they stay in L2); `PEAZERO_WINOGRAD=0` switches to the direct kernels.
//!
//! Network (trainer/peazero/model.py):
//!   stem   conv3x3 21→C, ReLU
//!   blocks B × [conv3x3, ReLU, conv3x3, + skip, ReLU]
//!   latent min-max normalised over all C×64 values
//!   policy conv1x1 C→32, ReLU, conv1x1 32→73 (+bias) → logits[plane*64 + sq]
//!   value  conv1x1 C→8, ReLU, linear 512→128, ReLU, linear 128→3 (W/D/L logits)
//!
//! Layout: activations are square-major on a zero-padded 10×10 board,
//! `buf[(padded square) * stride + channel]`, so a 3×3 convolution needs no
//! edge checks and channels are contiguous for SIMD. Convolution weights are
//! repacked per block of 16 output channels as `[k][cin][16]`; the kernel
//! computes 4 squares × 16 channels at a time (8 AVX2 accumulators), so every
//! weight vector loaded is used 4 times.

use std::sync::Arc;

use peazero_core::encoding::{NUM_ACTIONS, OBS_PLANES};

use crate::onnx::{parse_model, Graph, Node, Tensor};
use crate::winograd::{self, WinoConv, POS, TILES};

const CB: usize = 16; // output channels per kernel block
const PAD_SQUARES: usize = 100;
const POLICY_PLANES: usize = NUM_ACTIONS / 64;
const NORM_EPS: f32 = 1e-5;

/// Padded square index of board square `s` (a1 = 0, row-major like the
/// network's 8×8 planes).
#[inline(always)]
fn pad(s: usize) -> usize {
    (s / 8 + 1) * 10 + s % 8 + 1
}

/// One convolution with packed weights.
struct Conv {
    cin: usize,
    cout: usize,
    /// cout rounded up to a multiple of CB (= output stride).
    cout_pad: usize,
    /// Padded-board offsets of the kernel taps (9 for 3×3, 1 for 1×1).
    taps: Vec<isize>,
    /// `[cout_pad / CB][taps][cin][CB]`
    w: Vec<f32>,
    /// `[cout_pad]`
    bias: Vec<f32>,
}

impl Conv {
    fn new(w: &Tensor, bias: Option<&Tensor>) -> Result<Conv, String> {
        let [cout, cin, kh, kw] = w.dims[..] else {
            return Err(format!("conv weight must be 4-D, got {:?}", w.dims));
        };
        if kh != kw || !(kh == 1 || kh == 3) {
            return Err(format!("unsupported conv kernel {kh}x{kw}"));
        }
        let cout_pad = cout.div_ceil(CB) * CB;
        let r = (kh / 2) as isize;
        let mut taps = Vec::new();
        for dy in -r..=r {
            for dx in -r..=r {
                taps.push(dy * 10 + dx);
            }
        }
        let nt = taps.len();
        let mut packed = vec![0.0f32; cout_pad * nt * cin];
        for co in 0..cout {
            let (blk, lane) = (co / CB, co % CB);
            for ci in 0..cin {
                for ky in 0..kh {
                    for kx in 0..kw {
                        let t = ky * kw + kx;
                        let src = ((co * cin + ci) * kh + ky) * kw + kx;
                        packed[((blk * nt + t) * cin + ci) * CB + lane] = w.data[src];
                    }
                }
            }
        }
        let mut b = vec![0.0f32; cout_pad];
        if let Some(bias) = bias {
            if bias.data.len() != cout {
                return Err("conv bias size mismatch".into());
            }
            b[..cout].copy_from_slice(&bias.data);
        }
        Ok(Conv {
            cin,
            cout,
            cout_pad,
            taps,
            w: packed,
            bias: b,
        })
    }
}

/// All weights of one network, shared between evaluators (threads).
pub struct FastWeights {
    channels: usize,
    stem: Conv,
    blocks: Vec<(Conv, Conv)>,
    /// The same block convolutions in the Winograd domain (if the channel
    /// count is a multiple of 16).
    wino: Vec<(WinoConv, WinoConv)>,
    policy1: Conv,
    policy2: Conv,
    value_conv: Conv,
    /// value fc1 packed `[square * value_channels + c][hidden]`
    fc1_w: Vec<f32>,
    fc1_b: Vec<f32>,
    hidden: usize,
    /// value fc2 `[3][hidden]`
    fc2_w: Vec<f32>,
    fc2_b: [f32; 3],
}

impl FastWeights {
    /// Build from the `initial` ONNX graph.
    pub fn from_onnx(bytes: &[u8]) -> Result<FastWeights, String> {
        let g = parse_model(bytes)?;
        let init = |name: &str| {
            g.initializers
                .get(name)
                .ok_or_else(|| format!("missing tensor {name}"))
        };
        let conv_of = |n: &Node| -> Result<Conv, String> {
            let w = init(&n.inputs[1])?;
            let b = match n.inputs.get(2) {
                Some(name) if !name.is_empty() => Some(init(name)?),
                _ => None,
            };
            Conv::new(w, b)
        };
        let convs: Vec<&Node> = g.nodes.iter().filter(|n| n.op_type == "Conv").collect();
        let is3 = |n: &&Node| {
            g.initializers
                .get(&n.inputs[1])
                .is_some_and(|t| t.dims.get(2) == Some(&3))
        };
        let trunk: Vec<&Node> = convs.iter().copied().filter(is3).collect();
        let heads: Vec<&Node> = convs.iter().copied().filter(|n| !is3(n)).collect();
        if trunk.is_empty() || trunk.len() % 2 != 1 {
            return Err(format!(
                "expected a stem + residual blocks, found {} 3x3 convs",
                trunk.len()
            ));
        }
        let stem = conv_of(trunk[0])?;
        if stem.cin != OBS_PLANES {
            return Err(format!(
                "stem takes {} planes, the encoding has {OBS_PLANES}",
                stem.cin
            ));
        }
        let channels = stem.cout;
        let mut blocks = Vec::new();
        let mut wino = Vec::new();
        let wino_of = |n: &Node| -> Result<WinoConv, String> {
            let w = init(&n.inputs[1])?;
            let zeros = vec![0.0; w.dims[0]];
            let b = match n.inputs.get(2) {
                Some(name) if !name.is_empty() => &init(name)?.data,
                _ => &zeros,
            };
            Ok(WinoConv::new(&w.data, b, w.dims[0], w.dims[1]))
        };
        for pair in trunk[1..].chunks(2) {
            blocks.push((conv_of(pair[0])?, conv_of(pair[1])?));
            if channels % 16 == 0 {
                wino.push((wino_of(pair[0])?, wino_of(pair[1])?));
            }
        }
        // Heads: policy2 outputs the 73 move planes; policy1 feeds it (via a
        // ReLU); the remaining 1x1 conv is the value conv.
        let p2 = heads
            .iter()
            .find(|n| init(&n.inputs[1]).is_ok_and(|t| t.dims[0] == POLICY_PLANES))
            .ok_or("no policy output conv")?;
        let feeding = producer(&g, &p2.inputs[0]).ok_or("policy conv input not found")?;
        let p1_out = if feeding.op_type == "Relu" {
            &feeding.inputs[0]
        } else {
            &p2.inputs[0]
        };
        let p1 = heads
            .iter()
            .find(|n| &n.outputs[0] == p1_out)
            .ok_or("no policy hidden conv")?;
        let v = heads
            .iter()
            .find(|n| !std::ptr::eq(**n, *p1) && !std::ptr::eq(**n, *p2))
            .ok_or("no value conv")?;
        let (policy1, policy2, value_conv) = (conv_of(p1)?, conv_of(p2)?, conv_of(v)?);
        let gemms: Vec<&Node> = g.nodes.iter().filter(|n| n.op_type == "Gemm").collect();
        if gemms.len() != 2 {
            return Err(format!("expected 2 value layers, found {}", gemms.len()));
        }
        let (w1, b1) = (init(&gemms[0].inputs[1])?, init(&gemms[0].inputs[2])?);
        let (w2, b2) = (init(&gemms[1].inputs[1])?, init(&gemms[1].inputs[2])?);
        let (hidden, vin) = (w1.dims[0], w1.dims[1]);
        let vc = value_conv.cout;
        if vin != vc * 64 || w2.dims != [3, hidden] {
            return Err("unexpected value head shapes".into());
        }
        // PyTorch flattens the value conv output NCHW (index c*64 + sq); our
        // activations are square-major, so reorder the input dimension.
        let mut fc1_w = vec![0.0f32; vin * hidden];
        for o in 0..hidden {
            for c in 0..vc {
                for s in 0..64 {
                    fc1_w[(s * vc + c) * hidden + o] = w1.data[o * vin + c * 64 + s];
                }
            }
        }
        Ok(FastWeights {
            channels,
            stem,
            blocks,
            wino,
            policy1,
            policy2,
            value_conv,
            fc1_w,
            fc1_b: b1.data.clone(),
            hidden,
            fc2_w: w2.data.clone(),
            fc2_b: [b2.data[0], b2.data[1], b2.data[2]],
        })
    }

    /// From a `.pzn` file (uses its `initial` graph).
    pub fn from_pzn(bytes: &[u8]) -> Result<FastWeights, String> {
        let parts = crate::parse_pzn(bytes).map_err(|e| format!("{e:#}"))?;
        FastWeights::from_onnx(parts.initial)
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn blocks(&self) -> usize {
        self.blocks.len()
    }
}

fn producer<'a>(g: &'a Graph, output: &str) -> Option<&'a Node> {
    g.nodes
        .iter()
        .find(|n| n.outputs.iter().any(|o| o == output))
}

/// Evaluator: shared weights + this thread's scratch buffers.
pub struct FastNet {
    w: Arc<FastWeights>,
    simd: Simd,
    winograd: bool,
    wv: Vec<f32>,
    wm: Vec<f32>,
    obs: Vec<f32>,
    x: Vec<f32>,
    t: Vec<f32>,
    p1: Vec<f32>,
    p2: Vec<f32>,
    v1: Vec<f32>,
    hid: Vec<f32>,
}

impl FastNet {
    pub fn new(w: Arc<FastWeights>) -> FastNet {
        let c = w.channels;
        FastNet {
            simd: Simd::detect(),
            winograd: !w.wino.is_empty()
                && std::env::var("PEAZERO_WINOGRAD").map_or(true, |v| v != "0"),
            wv: vec![0.0; POS * TILES * c],
            wm: vec![0.0; POS * TILES * c],
            obs: vec![0.0; PAD_SQUARES * OBS_PLANES],
            x: vec![0.0; PAD_SQUARES * c],
            t: vec![0.0; PAD_SQUARES * c],
            p1: vec![0.0; PAD_SQUARES * w.policy1.cout_pad],
            p2: vec![0.0; PAD_SQUARES * w.policy2.cout_pad],
            v1: vec![0.0; PAD_SQUARES * w.value_conv.cout_pad],
            hid: vec![0.0; w.hidden],
            w,
        }
    }

    /// Kernel set in use.
    pub fn simd(&self) -> Simd {
        self.simd
    }

    /// Use a lower kernel set than the CPU supports (tests, benchmarks).
    /// Requests above what the CPU supports are ignored.
    pub fn set_simd(&mut self, simd: Simd) {
        self.simd = simd.min(Simd::supported());
    }

    /// Whether the residual blocks use Winograd convolutions.
    pub fn winograd(&self) -> bool {
        self.winograd
    }

    /// Use direct (`false`) or Winograd (`true`, if available) convolutions
    /// for the residual blocks (tests, benchmarks).
    pub fn set_winograd(&mut self, on: bool) {
        self.winograd = on && !self.w.wino.is_empty();
    }

    pub fn weights(&self) -> &Arc<FastWeights> {
        &self.w
    }

    /// `obs`: one encoded position (`OBS_SIZE`, planes × 64). Writes the
    /// policy logits (`NUM_ACTIONS`, index = plane * 64 + square) and the
    /// W/D/L value logits.
    pub fn evaluate(&mut self, obs: &[f32], policy: &mut [f32], value: &mut [f32; 3]) {
        assert_eq!(obs.len(), OBS_PLANES * 64);
        assert_eq!(policy.len(), NUM_ACTIONS);
        let w = Arc::clone(&self.w);
        let c = w.channels;
        for p in 0..OBS_PLANES {
            for s in 0..64 {
                self.obs[pad(s) * OBS_PLANES + p] = obs[p * 64 + s];
            }
        }
        self.conv(&w.stem, Src::Obs, Dst::X, false, true);
        if self.winograd {
            for (c1, c2) in &w.wino {
                winograd::input_transform(&self.x, c, &mut self.wv, self.simd);
                winograd::products(c1, &self.wv, &mut self.wm, self.simd);
                winograd::output_transform(&self.wm, c1, &mut self.t, false, true, self.simd);
                winograd::input_transform(&self.t, c, &mut self.wv, self.simd);
                winograd::products(c2, &self.wv, &mut self.wm, self.simd);
                winograd::output_transform(&self.wm, c2, &mut self.x, true, true, self.simd);
            }
        } else {
            for (c1, c2) in &w.blocks {
                self.conv(c1, Src::X, Dst::T, false, true);
                self.conv(c2, Src::T, Dst::X, true, true);
            }
        }
        // min-max normalisation of the latent (interior squares only)
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for s in 0..64 {
            for &v in &self.x[pad(s) * c..pad(s) * c + c] {
                lo = lo.min(v);
                hi = hi.max(v);
            }
        }
        let scale = 1.0 / (hi - lo + NORM_EPS);
        for s in 0..64 {
            for v in &mut self.x[pad(s) * c..pad(s) * c + c] {
                *v = (*v - lo) * scale;
            }
        }
        self.conv(&w.policy1, Src::X, Dst::P1, false, true);
        self.conv(&w.policy2, Src::P1, Dst::P2, false, false);
        let stride = w.policy2.cout_pad;
        for plane in 0..POLICY_PLANES {
            for s in 0..64 {
                policy[plane * 64 + s] = self.p2[pad(s) * stride + plane];
            }
        }
        self.conv(&w.value_conv, Src::X, Dst::V1, false, true);
        // value fc1: input = value conv output, square-major, real channels only
        let (vc, vstride, h) = (w.value_conv.cout, w.value_conv.cout_pad, w.hidden);
        self.hid.copy_from_slice(&w.fc1_b);
        for s in 0..64 {
            for ch in 0..vc {
                let x = self.v1[pad(s) * vstride + ch];
                if x != 0.0 {
                    let row = &w.fc1_w[(s * vc + ch) * h..(s * vc + ch + 1) * h];
                    axpy(&mut self.hid, x, row, self.simd >= Simd::Avx2);
                }
            }
        }
        for v in &mut self.hid {
            *v = v.max(0.0);
        }
        for (k, out) in value.iter_mut().enumerate() {
            let row = &w.fc2_w[k * h..(k + 1) * h];
            *out = w.fc2_b[k] + row.iter().zip(&self.hid).map(|(a, b)| a * b).sum::<f32>();
        }
    }

    fn buf(&self, s: Src) -> (&[f32], usize) {
        let w = &self.w;
        match s {
            Src::Obs => (&self.obs, OBS_PLANES),
            Src::X => (&self.x, w.channels),
            Src::T => (&self.t, w.channels),
            Src::P1 => (&self.p1, w.policy1.cout_pad),
        }
    }

    fn conv(&mut self, conv: &Conv, src: Src, dst: Dst, residual: bool, relu: bool) {
        // Split borrows: input and output are different buffers, except the
        // residual (read from the output buffer itself, same square/channel).
        let (input, in_stride) = {
            let (b, s) = self.buf(src);
            (b as *const [f32], s)
        };
        let out: &mut [f32] = match dst {
            Dst::X => &mut self.x,
            Dst::T => &mut self.t,
            Dst::P1 => &mut self.p1,
            Dst::P2 => &mut self.p2,
            Dst::V1 => &mut self.v1,
        };
        // SAFETY: src and dst are distinct buffers (checked by the callers'
        // fixed Src/Dst pairs above), so the shared and mutable borrows don't alias.
        let input = unsafe { &*input };
        // SAFETY: the level is never above what Simd::detect found, and SSE2
        // is part of every x86-64 CPU.
        unsafe {
            match self.simd {
                Simd::Avx512 => conv_avx512(conv, input, in_stride, out, residual, relu),
                Simd::Avx2 => conv_avx2(conv, input, in_stride, out, residual, relu),
                Simd::Sse2 => conv_sse2(conv, input, in_stride, out, residual, relu),
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Src {
    Obs,
    X,
    T,
    P1,
}

#[derive(Clone, Copy)]
enum Dst {
    X,
    T,
    P1,
    P2,
    V1,
}

/// Kernel sets, chosen at runtime from the CPU's features (so one binary
/// runs on every x86-64 CPU and uses the best kernels available).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Simd {
    Sse2,
    Avx2,
    Avx512,
}

impl Simd {
    /// Best kernel set this CPU supports.
    pub fn supported() -> Simd {
        use std::arch::is_x86_feature_detected as has;
        if has!("avx512f") && has!("avx2") && has!("fma") && has!("f16c") {
            Simd::Avx512
        } else if has!("avx2") && has!("fma") && has!("f16c") {
            Simd::Avx2
        } else {
            Simd::Sse2
        }
    }

    /// Kernel set to use: the best supported one, or a lower one requested
    /// with the environment variable `PEAZERO_SIMD=sse2|avx2|avx512`
    /// (testing).
    pub fn detect() -> Simd {
        let best = Simd::supported();
        match std::env::var("PEAZERO_SIMD")
            .map(|v| v.to_ascii_lowercase())
            .as_deref()
        {
            Ok("sse2") => Simd::Sse2,
            Ok("avx2") => best.min(Simd::Avx2),
            _ => best,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Simd::Sse2 => "SSE2",
            Simd::Avx2 => "AVX2",
            Simd::Avx512 => "AVX-512",
        }
    }
}

/// SSE2 (every x86-64 CPU): S squares × 16 channels (4 xmm each); with
/// S = 3 that is 12 accumulators + 4 weight vectors of the 16 xmm registers.
#[target_feature(enable = "sse2")]
unsafe fn conv_sse2(
    conv: &Conv,
    input: &[f32],
    in_stride: usize,
    out: &mut [f32],
    residual: bool,
    relu: bool,
) {
    let mut s0 = 0;
    while s0 + 3 <= 64 {
        conv_sse2_block::<3>(conv, input, in_stride, out, residual, relu, s0);
        s0 += 3;
    }
    conv_sse2_block::<1>(conv, input, in_stride, out, residual, relu, s0);
}

#[target_feature(enable = "sse2")]
#[inline]
unsafe fn conv_sse2_block<const S: usize>(
    conv: &Conv,
    input: &[f32],
    in_stride: usize,
    out: &mut [f32],
    residual: bool,
    relu: bool,
    s0: usize,
) {
    use std::arch::x86_64::*;
    let (nt, cin, os) = (conv.taps.len(), conv.cin, conv.cout_pad);
    let inp = input.as_ptr();
    let mut sq = [0usize; S];
    for (i, q) in sq.iter_mut().enumerate() {
        *q = SQUARES[s0 + i];
    }
    for blk in 0..conv.cout_pad / CB {
        let wb = conv.w.as_ptr().add(blk * nt * cin * CB);
        let bp = conv.bias.as_ptr().add(blk * CB);
        let bias = [
            _mm_loadu_ps(bp),
            _mm_loadu_ps(bp.add(4)),
            _mm_loadu_ps(bp.add(8)),
            _mm_loadu_ps(bp.add(12)),
        ];
        let mut acc = [bias; S];
        for (t, &off) in conv.taps.iter().enumerate() {
            let wt = wb.add(t * cin * CB);
            let mut xp = [inp; S];
            for i in 0..S {
                xp[i] = inp.offset((sq[i] as isize + off) * in_stride as isize);
            }
            for ci in 0..cin {
                let w = wt.add(ci * CB);
                let wv = [
                    _mm_loadu_ps(w),
                    _mm_loadu_ps(w.add(4)),
                    _mm_loadu_ps(w.add(8)),
                    _mm_loadu_ps(w.add(12)),
                ];
                for i in 0..S {
                    let v = _mm_set1_ps(*xp[i].add(ci));
                    for q in 0..4 {
                        acc[i][q] = _mm_add_ps(acc[i][q], _mm_mul_ps(v, wv[q]));
                    }
                }
            }
        }
        let zero = _mm_setzero_ps();
        for i in 0..S {
            let o = out.as_mut_ptr().add(sq[i] * os + blk * CB);
            for (q, &a) in acc[i].iter().enumerate() {
                let mut v = a;
                if residual {
                    v = _mm_add_ps(v, _mm_loadu_ps(o.add(4 * q)));
                }
                if relu {
                    v = _mm_max_ps(v, zero);
                }
                _mm_storeu_ps(o.add(4 * q), v);
            }
        }
    }
}

/// Padded indices of the 64 board squares, in board order.
const SQUARES: [usize; 64] = {
    let mut t = [0usize; 64];
    let mut s = 0;
    while s < 64 {
        t[s] = (s / 8 + 1) * 10 + s % 8 + 1;
        s += 1;
    }
    t
};

/// AVX2 + FMA: S squares × 16 channels (2 ymm each) per block. With S = 6
/// that is 12 accumulators + 2 weight vectors + 1 broadcast of the 16 ymm
/// registers; each weight vector loaded is used S times.
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_avx2(
    conv: &Conv,
    input: &[f32],
    in_stride: usize,
    out: &mut [f32],
    residual: bool,
    relu: bool,
) {
    conv_avx2_blocks(
        conv,
        0..conv.cout_pad / CB,
        input,
        in_stride,
        out,
        residual,
        relu,
    );
}

/// AVX2 kernel over the 16-channel output blocks `blks`.
#[target_feature(enable = "avx2,fma")]
unsafe fn conv_avx2_blocks(
    conv: &Conv,
    blks: std::ops::Range<usize>,
    input: &[f32],
    in_stride: usize,
    out: &mut [f32],
    residual: bool,
    relu: bool,
) {
    let mut s0 = 0;
    while s0 + 6 <= 64 {
        conv_avx2_block::<6>(
            conv,
            blks.clone(),
            input,
            in_stride,
            out,
            residual,
            relu,
            s0,
        );
        s0 += 6;
    }
    conv_avx2_block::<4>(conv, blks, input, in_stride, out, residual, relu, s0);
}

#[target_feature(enable = "avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_avx2_block<const S: usize>(
    conv: &Conv,
    blks: std::ops::Range<usize>,
    input: &[f32],
    in_stride: usize,
    out: &mut [f32],
    residual: bool,
    relu: bool,
    s0: usize,
) {
    use std::arch::x86_64::*;
    let (nt, cin, os) = (conv.taps.len(), conv.cin, conv.cout_pad);
    let inp = input.as_ptr();
    let mut sq = [0usize; S];
    for (i, q) in sq.iter_mut().enumerate() {
        *q = SQUARES[s0 + i];
    }
    for blk in blks {
        let wb = conv.w.as_ptr().add(blk * nt * cin * CB);
        let b0 = _mm256_loadu_ps(conv.bias.as_ptr().add(blk * CB));
        let b1 = _mm256_loadu_ps(conv.bias.as_ptr().add(blk * CB + 8));
        let mut lo = [b0; S];
        let mut hi = [b1; S];
        for (t, &off) in conv.taps.iter().enumerate() {
            let wt = wb.add(t * cin * CB);
            let mut xp = [inp; S];
            for i in 0..S {
                xp[i] = inp.offset((sq[i] as isize + off) * in_stride as isize);
            }
            for ci in 0..cin {
                let w0 = _mm256_loadu_ps(wt.add(ci * CB));
                let w1 = _mm256_loadu_ps(wt.add(ci * CB + 8));
                for i in 0..S {
                    let v = _mm256_broadcast_ss(&*xp[i].add(ci));
                    lo[i] = _mm256_fmadd_ps(v, w0, lo[i]);
                    hi[i] = _mm256_fmadd_ps(v, w1, hi[i]);
                }
            }
        }
        let zero = _mm256_setzero_ps();
        for i in 0..S {
            let o = out.as_mut_ptr().add(sq[i] * os + blk * CB);
            let (mut l, mut h) = (lo[i], hi[i]);
            if residual {
                l = _mm256_add_ps(l, _mm256_loadu_ps(o));
                h = _mm256_add_ps(h, _mm256_loadu_ps(o.add(8)));
            }
            if relu {
                l = _mm256_max_ps(l, zero);
                h = _mm256_max_ps(h, zero);
            }
            _mm256_storeu_ps(o, l);
            _mm256_storeu_ps(o.add(8), h);
        }
    }
}

/// AVX-512: S squares × 32 channels (two 16-channel weight blocks, 2 zmm
/// per square) per block; with S = 12 that is 24 accumulators of the 32 zmm
/// registers, and each weight vector is used 12 times. A trailing
/// 16-channel block (e.g. the value conv) goes through the AVX2 kernel.
#[target_feature(enable = "avx512f,avx2,fma")]
unsafe fn conv_avx512(
    conv: &Conv,
    input: &[f32],
    in_stride: usize,
    out: &mut [f32],
    residual: bool,
    relu: bool,
) {
    let blocks = conv.cout_pad / CB;
    for pair in 0..blocks / 2 {
        let mut s0 = 0;
        while s0 + AVX512_SQ <= 64 {
            conv_avx512_block::<AVX512_SQ>(conv, pair, input, in_stride, out, residual, relu, s0);
            s0 += AVX512_SQ;
        }
        if s0 < 64 {
            conv_avx512_block::<{ 64 % AVX512_SQ }>(
                conv, pair, input, in_stride, out, residual, relu, s0,
            );
        }
    }
    if blocks % 2 == 1 {
        // last single 16-channel block (e.g. the value conv): AVX2 kernel
        conv_avx2_blocks(
            conv,
            blocks - 1..blocks,
            input,
            in_stride,
            out,
            residual,
            relu,
        );
    }
}

const AVX512_SQ: usize = 8;

#[target_feature(enable = "avx512f,avx2,fma")]
#[inline]
#[allow(clippy::too_many_arguments)]
unsafe fn conv_avx512_block<const S: usize>(
    conv: &Conv,
    pair: usize,
    input: &[f32],
    in_stride: usize,
    out: &mut [f32],
    residual: bool,
    relu: bool,
    s0: usize,
) {
    use std::arch::x86_64::*;
    let (nt, cin, os) = (conv.taps.len(), conv.cin, conv.cout_pad);
    let inp = input.as_ptr();
    let (ba, bb) = (2 * pair, 2 * pair + 1);
    let wa = conv.w.as_ptr().add(ba * nt * cin * CB);
    let wbp = conv.w.as_ptr().add(bb * nt * cin * CB);
    let mut sq = [0usize; S];
    for (i, q) in sq.iter_mut().enumerate() {
        *q = SQUARES[s0 + i];
    }
    let mut lo = [_mm512_loadu_ps(conv.bias.as_ptr().add(ba * CB)); S];
    let mut hi = [_mm512_loadu_ps(conv.bias.as_ptr().add(bb * CB)); S];
    for (t, &off) in conv.taps.iter().enumerate() {
        let (ta, tb) = (wa.add(t * cin * CB), wbp.add(t * cin * CB));
        let mut xp = [inp; S];
        for i in 0..S {
            xp[i] = inp.offset((sq[i] as isize + off) * in_stride as isize);
        }
        for ci in 0..cin {
            let w0 = _mm512_loadu_ps(ta.add(ci * CB));
            let w1 = _mm512_loadu_ps(tb.add(ci * CB));
            for i in 0..S {
                let v = _mm512_set1_ps(*xp[i].add(ci));
                lo[i] = _mm512_fmadd_ps(v, w0, lo[i]);
                hi[i] = _mm512_fmadd_ps(v, w1, hi[i]);
            }
        }
    }
    let zero = _mm512_setzero_ps();
    for i in 0..S {
        let oa = out.as_mut_ptr().add(sq[i] * os + ba * CB);
        let ob = out.as_mut_ptr().add(sq[i] * os + bb * CB);
        let (mut l, mut h) = (lo[i], hi[i]);
        if residual {
            l = _mm512_add_ps(l, _mm512_loadu_ps(oa));
            h = _mm512_add_ps(h, _mm512_loadu_ps(ob));
        }
        if relu {
            l = _mm512_max_ps(l, zero);
            h = _mm512_max_ps(h, zero);
        }
        _mm512_storeu_ps(oa, l);
        _mm512_storeu_ps(ob, h);
    }
}

/// y += a * x
fn axpy(y: &mut [f32], a: f32, x: &[f32], simd: bool) {
    if simd && y.len().is_multiple_of(8) {
        // SAFETY: simd implies AVX2 + FMA; lengths checked.
        unsafe { axpy_avx2(y, a, x) };
        return;
    }
    for (yv, xv) in y.iter_mut().zip(x) {
        *yv += a * xv;
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_avx2(y: &mut [f32], a: f32, x: &[f32]) {
    use std::arch::x86_64::*;
    let av = _mm256_set1_ps(a);
    for i in (0..y.len()).step_by(8) {
        let yp = y.as_mut_ptr().add(i);
        _mm256_storeu_ps(
            yp,
            _mm256_fmadd_ps(av, _mm256_loadu_ps(x.as_ptr().add(i)), _mm256_loadu_ps(yp)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use peazero_core::encoding::{encode_game, OBS_SIZE};
    use peazero_core::Game;

    const NETS: &[&str] = &[
        "../../networks/default.pzn",
        "../../models/dummy-64x6",
        "../../models/dummy-32x4",
    ];

    fn positions() -> Vec<Vec<f32>> {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "r2q1rk1/pP1p2pp/Q4n2/bbp1p3/Np6/1B3NBn/pPPP1PPP/R3K2R b KQ - 0 1",
            "8/8/4k3/3p4/3P1K2/8/8/8 w - - 0 1",
        ];
        fens.iter()
            .map(|f| {
                let g = Game::from_fen(f).unwrap();
                let mut o = vec![0.0; OBS_SIZE];
                encode_game(&g, &mut o);
                o
            })
            .collect()
    }

    fn load(path: &str) -> Option<(FastWeights, crate::MuZeroNet)> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
        if !p.exists() {
            return None;
        }
        let tract = crate::MuZeroNet::load(&p, 1, false).unwrap();
        let fast = if p.is_dir() {
            FastWeights::from_onnx(&std::fs::read(p.join("initial.onnx")).unwrap()).unwrap()
        } else {
            FastWeights::from_pzn(&std::fs::read(&p).unwrap()).unwrap()
        };
        Some((fast, tract))
    }

    /// AVX2 and AVX-512 do the same fused multiply-adds in the same order,
    /// so their outputs must be bit-identical (the engine's bench node count
    /// must not depend on the CPU).
    #[test]
    fn avx2_and_avx512_are_bit_identical() {
        peazero_core::init();
        if Simd::supported() < Simd::Avx512 {
            return;
        }
        for path in NETS {
            let Some((w, _)) = load(path) else { continue };
            let w = Arc::new(w);
            for wino in [false, true] {
                let (mut a, mut b) = (FastNet::new(w.clone()), FastNet::new(w.clone()));
                a.set_simd(Simd::Avx512);
                b.set_simd(Simd::Avx2);
                a.set_winograd(wino);
                b.set_winograd(wino);
                for obs in positions() {
                    let (mut pa, mut pb) = (vec![0.0; NUM_ACTIONS], vec![0.0; NUM_ACTIONS]);
                    let (mut va, mut vb) = ([0.0; 3], [0.0; 3]);
                    a.evaluate(&obs, &mut pa, &mut va);
                    b.evaluate(&obs, &mut pb, &mut vb);
                    assert!(
                        pa == pb && va == vb,
                        "{path} winograd={wino}: AVX2 and AVX-512 differ"
                    );
                }
            }
        }
    }

    #[test]
    fn matches_tract() {
        peazero_core::init();
        let mut tested = 0;
        for path in NETS {
            let Some((w, tract)) = load(path) else {
                continue;
            };
            let w = Arc::new(w);
            for (level, wino) in [Simd::Avx512, Simd::Avx2, Simd::Sse2]
                .into_iter()
                .flat_map(|l| [(l, false), (l, true)])
            {
                let mut net = FastNet::new(w.clone());
                net.set_simd(level);
                net.set_winograd(wino);
                let portable = (net.simd(), net.winograd());
                for obs in positions() {
                    let want = tract.initial(&obs).unwrap();
                    let mut policy = vec![0.0; NUM_ACTIONS];
                    let mut value = [0.0; 3];
                    net.evaluate(&obs, &mut policy, &mut value);
                    let perr = policy
                        .iter()
                        .zip(&want.policy)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    let verr = value
                        .iter()
                        .zip(&want.value)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f32::max);
                    assert!(
                        perr < 2e-3 && verr < 2e-3,
                        "{path} {portable:?}: policy err {perr}, value err {verr}"
                    );
                }
            }
            tested += 1;
        }
        assert!(tested > 0, "no test networks found");
    }
}
