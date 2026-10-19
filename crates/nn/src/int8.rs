//! Int8 residual-block convolutions (x86-64, optional: UCI option `Int8`).
//!
//! * Weights: int8 with one scale per output channel, `w ≈ q_w · s_w[co]`.
//! * Activations: the block inputs are ReLU outputs (≥ 0); each conv input
//!   is quantised per evaluation to 0..127 with the layer's own scale,
//!   `x ≈ q_x · s_x`, `s_x = max(x) / 127`.
//! * `y[co] = s_x · s_w[co] · Σ q_x q_w + bias[co]` (+ residual, ReLU) in f32.
//!
//! Kernels (all give the same integers, so benches agree across CPUs):
//! * AVX2: `vpmaddubsw` (u8 × i8 pairs → i16) + `vpmaddwd` (→ i32). With
//!   7-bit activations a pair sum is at most 2·127·127 = 32258, so the i16
//!   step never saturates.
//! * AVX-VNNI (Intel 12th gen+): `vpdpbusd` on ymm, 4 products per i32 lane.
//! * AVX-512 VNNI (Zen 4, Intel servers): `vpdpbusd` on zmm.
//!
//! Weight layout per block of 16 output channels: `[tap][cin/4][16][4]`
//! bytes, i.e. one 64-byte row per group of 4 input channels: lane `co`
//! holds `w[co][4k..4k+4]`, matching a broadcast of 4 activation bytes.

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

const CB: usize = 16;
/// Squares per AVX2 / AVX-VNNI block.
const YMM_SQ: usize = 5;

/// Activations are quantised to 0..=QMAX (7 bits, see module docs).
const QMAX: f32 = 127.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Int8Kernel {
    Avx2,
    AvxVnni,
    Avx512Vnni,
}

impl Int8Kernel {
    /// Best int8 kernel this CPU supports (None without AVX2).
    pub fn detect() -> Option<Int8Kernel> {
        use std::arch::is_x86_feature_detected as has;
        if !(has!("avx2") && has!("fma")) {
            return None;
        }
        Some(if has!("avx512f") && has!("avx512vnni") {
            Int8Kernel::Avx512Vnni
        } else if has!("avxvnni") {
            Int8Kernel::AvxVnni
        } else {
            Int8Kernel::Avx2
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Int8Kernel::Avx2 => "AVX2 int8",
            Int8Kernel::AvxVnni => "AVX-VNNI int8",
            Int8Kernel::Avx512Vnni => "AVX-512 VNNI int8",
        }
    }
}

/// One quantised 3×3 convolution (cin multiple of 4, cout multiple of 16).
pub struct QConv {
    pub cin: usize,
    pub cout: usize,
    /// `[cout/16][9][cin/4][16][4]`
    w: Vec<i8>,
    /// per output channel weight scale
    wscale: Vec<f32>,

    bias: Vec<f32>,
}

const TAPS: [isize; 9] = [-11, -10, -9, -1, 0, 1, 9, 10, 11];

impl QConv {
    /// `w`: PyTorch layout `[cout][cin][3][3]`.
    pub fn new(w: &[f32], bias: &[f32], cout: usize, cin: usize) -> QConv {
        assert!(cin.is_multiple_of(4) && cout.is_multiple_of(CB));
        let mut packed = vec![0i8; cout * 9 * cin];
        let mut wscale = vec![0.0f32; cout];
        for co in 0..cout {
            let row = &w[co * cin * 9..(co + 1) * cin * 9];
            let max = row.iter().fold(0.0f32, |m, &x| m.max(x.abs()));
            let s = if max > 0.0 { max / 127.0 } else { 1.0 };
            wscale[co] = s;
            let (blk, lane) = (co / CB, co % CB);
            for ci in 0..cin {
                for t in 0..9 {
                    let q = (row[ci * 9 + t] / s).round().clamp(-127.0, 127.0) as i8;
                    let idx = (((blk * 9 + t) * (cin / 4) + ci / 4) * CB + lane) * 4 + ci % 4;
                    packed[idx] = q;
                }
            }
        }
        QConv {
            cin,
            cout,
            w: packed,
            wscale,

            bias: bias.to_vec(),
        }
    }
}

/// Quantise the interior of a padded board `x[100][c]` (values ≥ 0) into
/// `q[100][c]` (borders stay 0). Returns the scale `s_x` (0 if all zero).
/// One shared AVX2 routine, so every int8 kernel sees the same bytes.
///
/// # Safety
/// The CPU must support AVX2 (guaranteed when an `Int8Kernel` exists).
#[target_feature(enable = "avx2,fma")]
pub unsafe fn quantize(x: &[f32], c: usize, q: &mut [u8]) -> f32 {
    use std::arch::x86_64::*;
    assert!(c.is_multiple_of(32) && x.len() >= 100 * c && q.len() >= 100 * c);
    let xp = x.as_ptr();
    let mut m = _mm256_setzero_ps();
    for &sq in &SQUARES {
        for k in (0..c).step_by(8) {
            m = _mm256_max_ps(m, _mm256_loadu_ps(xp.add(sq * c + k)));
        }
    }
    let mut lanes = [0.0f32; 8];
    _mm256_storeu_ps(lanes.as_mut_ptr(), m);
    let max = lanes.iter().fold(0.0f32, |a, &b| a.max(b));
    if max <= 0.0 {
        for &sq in &SQUARES {
            q[sq * c..sq * c + c].fill(0);
        }
        return 0.0;
    }
    let inv = _mm256_set1_ps(QMAX / max);
    let hi = _mm256_set1_epi32(QMAX as i32);
    let lo = _mm256_setzero_si256();
    let order = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
    let qp = q.as_mut_ptr();
    for &sq in &SQUARES {
        for k in (0..c).step_by(32) {
            // round to nearest even, clamp to 0..127, pack 32 lanes to bytes
            let cv = |j: usize| {
                let v = _mm256_cvtps_epi32(_mm256_mul_ps(
                    _mm256_loadu_ps(xp.add(sq * c + k + 8 * j)),
                    inv,
                ));
                _mm256_max_epi32(_mm256_min_epi32(v, hi), lo)
            };
            let ab = _mm256_packs_epi32(cv(0), cv(1));
            let cd = _mm256_packs_epi32(cv(2), cv(3));
            let bytes = _mm256_permutevar8x32_epi32(_mm256_packus_epi16(ab, cd), order);
            _mm256_storeu_si256(qp.add(sq * c + k) as *mut __m256i, bytes);
        }
    }
    max / QMAX
}

/// Quantised convolution: `q` (padded `[100][cin]` bytes, scale `sx`) →
/// padded f32 `out[100][cout]`, plus bias, optional residual (read from
/// `out`) and ReLU.
///
/// # Safety
/// `kernel` must be supported by the CPU (from `Int8Kernel::detect`).
pub unsafe fn conv(
    qc: &QConv,
    q: &[u8],
    sx: f32,
    out: &mut [f32],
    residual: bool,
    relu: bool,
    kernel: Int8Kernel,
) {
    assert!(q.len() >= 100 * qc.cin && out.len() >= 100 * qc.cout);
    // output scale per channel; same scalar products for every kernel
    let mut scale = [0.0f32; 512];
    assert!(qc.cout <= scale.len());
    for (s, w) in scale.iter_mut().zip(&qc.wscale) {
        *s = sx * w;
    }
    match kernel {
        Int8Kernel::Avx512Vnni => conv_avx512vnni(qc, q, &scale, out, residual, relu),
        Int8Kernel::AvxVnni => conv_avxvnni(qc, q, &scale, out, residual, relu),
        Int8Kernel::Avx2 => conv_avx2(qc, q, &scale, out, residual, relu),
    }
}

/// AVX2 / AVX-VNNI kernels: S squares × 16 channels (2 ymm i32
/// accumulators each). Generated twice so the plain AVX2 copy is compiled
/// without VNNI (the compiler may not use VNNI instructions there).
macro_rules! ymm_kernel {
    ($conv:ident, $block:ident, $feat:literal, $dot:expr) => {
        #[target_feature(enable = $feat)]
        unsafe fn $conv(
            qc: &QConv,
            q: &[u8],
            scale: &[f32],
            out: &mut [f32],
            residual: bool,
            relu: bool,
        ) {
            let mut s0 = 0;
            while s0 + YMM_SQ <= 64 {
                $block::<YMM_SQ>(qc, q, scale, out, residual, relu, s0);
                s0 += YMM_SQ;
            }
            if s0 < 64 {
                $block::<{ 64 % YMM_SQ }>(qc, q, scale, out, residual, relu, s0);
            }
        }

        #[target_feature(enable = $feat)]
        #[inline]
        #[allow(clippy::too_many_arguments)]
        unsafe fn $block<const S: usize>(
            qc: &QConv,
            q: &[u8],
            scale: &[f32],
            out: &mut [f32],
            residual: bool,
            relu: bool,
            s0: usize,
        ) {
            use std::arch::x86_64::*;
            let (cin, cout) = (qc.cin, qc.cout);
            let groups = cin / 4;
            let (qp, wp) = (q.as_ptr(), qc.w.as_ptr());
            let mut sq = [0usize; S];
            for (i, s) in sq.iter_mut().enumerate() {
                *s = SQUARES[s0 + i];
            }
            for blk in 0..cout / CB {
                let mut lo = [_mm256_setzero_si256(); S];
                let mut hi = [_mm256_setzero_si256(); S];
                for (t, &off) in TAPS.iter().enumerate() {
                    let wt = wp.add((blk * 9 + t) * groups * CB * 4);
                    let mut xp = [qp; S];
                    for i in 0..S {
                        xp[i] = qp.offset((sq[i] as isize + off) * cin as isize);
                    }
                    for g in 0..groups {
                        let w0 = _mm256_loadu_si256(wt.add(g * 64) as *const __m256i);
                        let w1 = _mm256_loadu_si256(wt.add(g * 64 + 32) as *const __m256i);
                        for i in 0..S {
                            let x = _mm256_set1_epi32(
                                (xp[i].add(4 * g) as *const i32).read_unaligned(),
                            );
                            lo[i] = $dot(lo[i], x, w0);
                            hi[i] = $dot(hi[i], x, w1);
                        }
                    }
                }
                let c = blk * CB;
                let (s0v, s1v) = (
                    _mm256_loadu_ps(scale.as_ptr().add(c)),
                    _mm256_loadu_ps(scale.as_ptr().add(c + 8)),
                );
                let (b0, b1) = (
                    _mm256_loadu_ps(qc.bias.as_ptr().add(c)),
                    _mm256_loadu_ps(qc.bias.as_ptr().add(c + 8)),
                );
                let zero = _mm256_setzero_ps();
                for i in 0..S {
                    let o = out.as_mut_ptr().add(sq[i] * cout + c);
                    let mut y0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(lo[i]), s0v), b0);
                    let mut y1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(hi[i]), s1v), b1);
                    if residual {
                        y0 = _mm256_add_ps(y0, _mm256_loadu_ps(o));
                        y1 = _mm256_add_ps(y1, _mm256_loadu_ps(o.add(8)));
                    }
                    if relu {
                        y0 = _mm256_max_ps(y0, zero);
                        y1 = _mm256_max_ps(y1, zero);
                    }
                    _mm256_storeu_ps(o, y0);
                    _mm256_storeu_ps(o.add(8), y1);
                }
            }
        }
    };
}

/// u8 × i8 dot products of 4 bytes per i32 lane, accumulated: AVX2 via
/// maddubs (pairs → i16, never saturating with 7-bit activations) + madd.
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn dot_avx2(
    acc: std::arch::x86_64::__m256i,
    x: std::arch::x86_64::__m256i,
    w: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;
    _mm256_add_epi32(
        acc,
        _mm256_madd_epi16(_mm256_maddubs_epi16(x, w), _mm256_set1_epi16(1)),
    )
}

ymm_kernel!(conv_avx2, block_avx2, "avx2,fma", dot_avx2);
ymm_kernel!(
    conv_avxvnni,
    block_avxvnni,
    "avx2,fma,avxvnni",
    std::arch::x86_64::_mm256_dpbusd_avx_epi32
);

/// AVX-512 VNNI: one row of 8 squares × 32 channels (two 16-channel
/// blocks, 16 zmm accumulators); an odd last block is computed alone.
#[target_feature(enable = "avx512f,avx512vnni,avx2,fma")]
unsafe fn conv_avx512vnni(
    qc: &QConv,
    q: &[u8],
    scale: &[f32],
    out: &mut [f32],
    residual: bool,
    relu: bool,
) {
    use std::arch::x86_64::*;
    let (cin, cout) = (qc.cin, qc.cout);
    let groups = cin / 4;
    let (qp, wp) = (q.as_ptr(), qc.w.as_ptr());
    let blocks = cout / CB;
    for pair in 0..blocks.div_ceil(2) {
        let ba = 2 * pair;
        let two = ba + 1 < blocks;
        for row in 0..8 {
            let base = (row + 1) * 10 + 1;
            let mut lo = [_mm512_setzero_si512(); 8];
            let mut hi = [_mm512_setzero_si512(); 8];
            for (t, &off) in TAPS.iter().enumerate() {
                let wa = wp.add((ba * 9 + t) * groups * CB * 4);
                let wb = wp.add(((ba + usize::from(two)) * 9 + t) * groups * CB * 4);
                let x0 = qp.offset((base as isize + off) * cin as isize);
                for g in 0..groups {
                    let w0 = _mm512_loadu_si512(wa.add(g * 64) as *const _);
                    let w1 = _mm512_loadu_si512(wb.add(g * 64) as *const _);
                    for i in 0..8 {
                        let x = _mm512_set1_epi32(
                            (x0.add(i * cin + 4 * g) as *const i32).read_unaligned(),
                        );
                        lo[i] = _mm512_dpbusd_epi32(lo[i], x, w0);
                        hi[i] = _mm512_dpbusd_epi32(hi[i], x, w1);
                    }
                }
            }
            let zero = _mm512_setzero_ps();
            for (half, acc) in [(0usize, &lo), (1, &hi)] {
                if half == 1 && !two {
                    break;
                }
                let c = (ba + half) * CB;
                let sv = _mm512_loadu_ps(scale.as_ptr().add(c));
                let bv = _mm512_loadu_ps(qc.bias.as_ptr().add(c));
                for (i, a) in acc.iter().enumerate() {
                    let o = out.as_mut_ptr().add((base + i) * cout + c);
                    let mut y = _mm512_add_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(*a), sv), bv);
                    if residual {
                        y = _mm512_add_ps(y, _mm512_loadu_ps(o));
                    }
                    if relu {
                        y = _mm512_max_ps(y, zero);
                    }
                    _mm512_storeu_ps(o, y);
                }
            }
        }
    }
}
