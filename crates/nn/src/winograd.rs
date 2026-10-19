//! Winograd F(2×2, 3×3) for the residual-block convolutions (x86-64).
//!
//! A 3×3 convolution over the 8×8 board is computed on 16 tiles of 2×2
//! outputs, each from a 4×4 input tile (padded board, overlap 2):
//!
//!   Y = Aᵀ [ U ⊙ V ] A,   U = G g Gᵀ (weights, precomputed),   V = Bᵀ d B
//!
//! Per tile and channel pair that is 16 multiplications instead of 36, so
//! 2.25× fewer multiply-adds. The work becomes:
//! 1. input transform: V[pos][tile][ci] (adds only, vectorised over channels)
//! 2. 16 small matrix products (one per transform position):
//!    M[pos][tile][co] = Σ_ci V[pos][tile][ci] · U[pos][ci][co]
//! 3. output transform + bias (+ residual) (+ ReLU) back to the padded board.
//!
//! The smaller F(2×2) variant is used on purpose: F(4×4) saves more
//! multiplications but needs 4× the weight data, which makes batch-1
//! inference memory-bound. The transforms are plain element-wise adds in a
//! fixed order (identical on every CPU), and the matrix products do the same
//! fused multiply-adds per output lane with AVX2 and AVX-512, so those two
//! stay bit-identical.

use crate::fast::Simd;

/// Transform positions (4×4) and tiles per board (4×4 tiles of 2×2).
pub const POS: usize = 16;
pub const TILES: usize = 16;
const CB: usize = 16;

/// Weights of one 3×3 convolution in the Winograd domain, stored as
/// 16-bit floats: the 12 layers of a 64×6 net then take ~1.5 MB instead of
/// ~3 MB and stay in a typical L2 cache (batch-1 inference is otherwise
/// bound by streaming them in). The arithmetic stays 32-bit.
pub struct WinoConv {
    pub cin: usize,
    pub cout: usize,
    /// `[POS][cin][cout]`, IEEE half precision (widened with F16C on load)
    uh: Vec<u16>,
    /// The same rounded values as f32 (SSE2 path, no F16C)
    u: Vec<f32>,
    /// `[cout]`
    bias: Vec<f32>,
}

impl WinoConv {
    /// `w`: PyTorch layout `[cout][cin][3][3]`.
    pub fn new(w: &[f32], bias: &[f32], cout: usize, cin: usize) -> WinoConv {
        const G: [[f64; 3]; 4] = [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
        ];
        let mut u = vec![0.0f32; POS * cin * cout];
        for co in 0..cout {
            for ci in 0..cin {
                let g = |r: usize, c: usize| f64::from(w[((co * cin + ci) * 3 + r) * 3 + c]);
                // tmp = G g (4×3), U = tmp Gᵀ (4×4)
                let mut tmp = [[0.0f64; 3]; 4];
                for (i, row) in tmp.iter_mut().enumerate() {
                    for (c, v) in row.iter_mut().enumerate() {
                        *v = (0..3).map(|r| G[i][r] * g(r, c)).sum();
                    }
                }
                for i in 0..4 {
                    for j in 0..4 {
                        let v: f64 = (0..3).map(|c| tmp[i][c] * G[j][c]).sum();
                        u[((i * 4 + j) * cin + ci) * cout + co] = f16_to_f32(f32_to_f16(v as f32));
                    }
                }
            }
        }
        let uh = u.iter().map(|&x| f32_to_f16(x)).collect();
        WinoConv {
            cin,
            cout,
            uh,
            u,
            bias: bias.to_vec(),
        }
    }
}

/// f32 → IEEE half, round to nearest even (same as F16C `vcvtps2ph`).
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 255 {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    let round = |v: u32, rem: u32, halfway: u32| {
        v + u32::from(rem > halfway || (rem == halfway && v & 1 == 1))
    };
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32;
        return sign | round(m >> shift, m & ((1 << shift) - 1), 1 << (shift - 1)) as u16;
    }
    sign | round(((e as u32) << 10) | (man >> 13), man & 0x1fff, 0x1000) as u16
}

/// IEEE half → f32 (exact).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // subnormal: normalise
            let shift = man.leading_zeros() - 21;
            sign | ((113 - shift) << 23) | ((man << shift) & 0x3ff) << 13
        }
        31 => sign | 0x7f80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

/// Padded square index (10×10 board) of padded row r, column c.
#[inline(always)]
fn sq(r: usize, c: usize) -> usize {
    r * 10 + c
}

/// Input transform: padded board `input[100][cin]` → `v[POS][TILES][cin]`.
/// Per tile and group of channels the 16 values stay in registers.
pub fn input_transform(input: &[f32], cin: usize, v: &mut [f32], simd: Simd) {
    assert!(input.len() >= 100 * cin && v.len() >= POS * TILES * cin);
    // SAFETY: bounds asserted above; AVX2 only when the CPU has it; the
    // channel count is a multiple of 16 (checked when the weights are built).
    unsafe {
        if simd >= Simd::Avx2 {
            input_avx2(input, cin, v)
        } else {
            input_sse2(input, cin, v)
        }
    }
}

/// Output transform: `m[POS][TILES][cout]` → padded board `out[100][cout]`,
/// plus bias, optional residual (read from `out` itself) and ReLU.
pub fn output_transform(
    m: &[f32],
    conv: &WinoConv,
    out: &mut [f32],
    residual: bool,
    relu: bool,
    simd: Simd,
) {
    assert!(m.len() >= POS * TILES * conv.cout && out.len() >= 100 * conv.cout);
    // SAFETY: as in input_transform.
    unsafe {
        if simd >= Simd::Avx2 {
            output_avx2(m, conv, out, residual, relu)
        } else {
            output_sse2(m, conv, out, residual, relu)
        }
    }
}

/// Both ISAs run the same element-wise operations in the same order, so
/// the transforms give bit-identical results.
macro_rules! transforms {
    ($input:ident, $output:ident, $feat:literal, $lanes:expr, $load:ident, $store:ident, $add:ident, $sub:ident, $max:ident, $zero:ident) => {
        #[target_feature(enable = $feat)]
        unsafe fn $input(input: &[f32], cin: usize, v: &mut [f32]) {
            use std::arch::x86_64::*;
            let (ip, vp) = (input.as_ptr(), v.as_mut_ptr());
            for ty in 0..4 {
                for tx in 0..4 {
                    let tile = ty * 4 + tx;
                    let base = sq(2 * ty, 2 * tx);
                    for c in (0..cin).step_by($lanes) {
                        let d = |i: usize, j: usize| $load(ip.add((base + i * 10 + j) * cin + c));
                        // t = Bᵀ d (rows), then V = t B (columns), per column j / row i
                        let mut t = [[$zero(); 4]; 4];
                        for j in 0..4 {
                            let (d0, d1, d2, d3) = (d(0, j), d(1, j), d(2, j), d(3, j));
                            t[0][j] = $sub(d0, d2);
                            t[1][j] = $add(d1, d2);
                            t[2][j] = $sub(d2, d1);
                            t[3][j] = $sub(d1, d3);
                        }
                        for (i, r) in t.iter().enumerate() {
                            let o = |j: usize| vp.add(((4 * i + j) * TILES + tile) * cin + c);
                            $store(o(0), $sub(r[0], r[2]));
                            $store(o(1), $add(r[1], r[2]));
                            $store(o(2), $sub(r[2], r[1]));
                            $store(o(3), $sub(r[1], r[3]));
                        }
                    }
                }
            }
        }

        #[target_feature(enable = $feat)]
        unsafe fn $output(m: &[f32], conv: &WinoConv, out: &mut [f32], residual: bool, relu: bool) {
            use std::arch::x86_64::*;
            let co = conv.cout;
            let (mp, op, bp) = (m.as_ptr(), out.as_mut_ptr(), conv.bias.as_ptr());
            for ty in 0..4 {
                for tx in 0..4 {
                    let tile = ty * 4 + tx;
                    for c in (0..co).step_by($lanes) {
                        let mm = |i: usize, j: usize| {
                            $load(mp.add(((4 * i + j) * TILES + tile) * co + c))
                        };
                        // s = Aᵀ m (rows): s0 = m0 + m1 + m2, s1 = m1 - m2 - m3
                        let mut s = [[$zero(); 4]; 2];
                        for j in 0..4 {
                            let (m0, m1, m2, m3) = (mm(0, j), mm(1, j), mm(2, j), mm(3, j));
                            s[0][j] = $add($add(m0, m1), m2);
                            s[1][j] = $sub($sub(m1, m2), m3);
                        }
                        let bias = $load(bp.add(c));
                        for (dy, r) in s.iter().enumerate() {
                            // Y = s A (columns): y0 = s0 + s1 + s2, y1 = s1 - s2 - s3
                            let y = [$add($add(r[0], r[1]), r[2]), $sub($sub(r[1], r[2]), r[3])];
                            for (dx, &val) in y.iter().enumerate() {
                                let p = op.add(sq(2 * ty + dy + 1, 2 * tx + dx + 1) * co + c);
                                let mut x = $add(val, bias);
                                if residual {
                                    x = $add(x, $load(p));
                                }
                                if relu {
                                    x = $max(x, $zero());
                                }
                                $store(p, x);
                            }
                        }
                    }
                }
            }
        }
    };
}

transforms!(
    input_avx2,
    output_avx2,
    "avx2",
    8,
    _mm256_loadu_ps,
    _mm256_storeu_ps,
    _mm256_add_ps,
    _mm256_sub_ps,
    _mm256_max_ps,
    _mm256_setzero_ps
);
transforms!(
    input_sse2,
    output_sse2,
    "sse2",
    4,
    _mm_loadu_ps,
    _mm_storeu_ps,
    _mm_add_ps,
    _mm_sub_ps,
    _mm_max_ps,
    _mm_setzero_ps
);

/// The 16 matrix products `m[pos] = v[pos] · u[pos]` (`TILES × cin` times
/// `cin × cout`).
pub fn products(conv: &WinoConv, v: &[f32], m: &mut [f32], simd: Simd) {
    let (ci, co) = (conv.cin, conv.cout);
    for pos in 0..POS {
        let a = &v[pos * TILES * ci..(pos + 1) * TILES * ci];
        let range = pos * ci * co..(pos + 1) * ci * co;
        let (b, bh) = (&conv.u[range.clone()], &conv.uh[range]);
        let c = &mut m[pos * TILES * co..(pos + 1) * TILES * co];
        // SAFETY: the level never exceeds what the CPU supports (Simd::detect),
        // SSE2 is part of x86-64, and cout is a multiple of 16 (checked by the
        // caller), so all 16-wide accesses stay in bounds.
        unsafe {
            match simd {
                Simd::Avx512 => gemm_avx512(a, bh, c, ci, co),
                Simd::Avx2 => gemm_avx2(a, bh, c, ci, co, 0..co / CB),
                Simd::Sse2 => gemm_sse2(a, b, c, ci, co),
            }
        }
    }
}

/// AVX2: R rows × 16 columns per block (2R accumulators); weights are
/// half floats, widened with F16C as they are loaded.
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn gemm_avx2(
    a: &[f32],
    b: &[u16],
    c: &mut [f32],
    k: usize,
    n: usize,
    blks: std::ops::Range<usize>,
) {
    let mut r0 = 0;
    while r0 + 6 <= TILES {
        gemm_avx2_block::<6>(a, b, c, k, n, blks.clone(), r0);
        r0 += 6;
    }
    gemm_avx2_block::<4>(a, b, c, k, n, blks, r0);
}

#[target_feature(enable = "avx2,fma,f16c")]
#[inline]
unsafe fn gemm_avx2_block<const R: usize>(
    a: &[f32],
    b: &[u16],
    c: &mut [f32],
    k: usize,
    n: usize,
    blks: std::ops::Range<usize>,
    r0: usize,
) {
    use std::arch::x86_64::*;
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    for blk in blks {
        let mut lo = [_mm256_setzero_ps(); R];
        let mut hi = [_mm256_setzero_ps(); R];
        for kk in 0..k {
            let w0 = _mm256_cvtph_ps(_mm_loadu_si128(bp.add(kk * n + blk * CB) as *const __m128i));
            let w1 = _mm256_cvtph_ps(_mm_loadu_si128(
                bp.add(kk * n + blk * CB + 8) as *const __m128i
            ));
            for r in 0..R {
                let x = _mm256_broadcast_ss(&*ap.add((r0 + r) * k + kk));
                lo[r] = _mm256_fmadd_ps(x, w0, lo[r]);
                hi[r] = _mm256_fmadd_ps(x, w1, hi[r]);
            }
        }
        for r in 0..R {
            let o = c.as_mut_ptr().add((r0 + r) * n + blk * CB);
            _mm256_storeu_ps(o, lo[r]);
            _mm256_storeu_ps(o.add(8), hi[r]);
        }
    }
}

/// AVX-512: R rows × 32 columns per block (2R zmm accumulators); a trailing
/// 16-column block goes through the AVX2 kernel.
#[target_feature(enable = "avx512f,avx2,fma,f16c")]
unsafe fn gemm_avx512(a: &[f32], b: &[u16], c: &mut [f32], k: usize, n: usize) {
    let pairs = n / 32;
    let mut r0 = 0;
    while r0 + 8 <= TILES {
        gemm_avx512_block::<8>(a, b, c, k, n, pairs, r0);
        r0 += 8;
    }
    if r0 < TILES {
        gemm_avx512_block::<{ TILES % 8 }>(a, b, c, k, n, pairs, r0);
    }
    if !n.is_multiple_of(32) {
        gemm_avx2(a, b, c, k, n, n / CB - 1..n / CB);
    }
}

#[target_feature(enable = "avx512f,avx2,fma,f16c")]
#[inline]
unsafe fn gemm_avx512_block<const R: usize>(
    a: &[f32],
    b: &[u16],
    c: &mut [f32],
    k: usize,
    n: usize,
    pairs: usize,
    r0: usize,
) {
    use std::arch::x86_64::*;
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    for pair in 0..pairs {
        let col = pair * 32;
        let mut lo = [_mm512_setzero_ps(); R];
        let mut hi = [_mm512_setzero_ps(); R];
        for kk in 0..k {
            let w0 = _mm512_cvtph_ps(_mm256_loadu_si256(bp.add(kk * n + col) as *const __m256i));
            let w1 = _mm512_cvtph_ps(_mm256_loadu_si256(
                bp.add(kk * n + col + 16) as *const __m256i
            ));
            for r in 0..R {
                let x = _mm512_set1_ps(*ap.add((r0 + r) * k + kk));
                lo[r] = _mm512_fmadd_ps(x, w0, lo[r]);
                hi[r] = _mm512_fmadd_ps(x, w1, hi[r]);
            }
        }
        for r in 0..R {
            let o = c.as_mut_ptr().add((r0 + r) * n + col);
            _mm512_storeu_ps(o, lo[r]);
            _mm512_storeu_ps(o.add(16), hi[r]);
        }
    }
}

/// SSE2: 3 rows × 16 columns per block.
#[target_feature(enable = "sse2")]
unsafe fn gemm_sse2(a: &[f32], b: &[f32], c: &mut [f32], k: usize, n: usize) {
    use std::arch::x86_64::*;
    let (ap, bp) = (a.as_ptr(), b.as_ptr());
    let mut r0 = 0;
    while r0 < TILES {
        let rows = (TILES - r0).min(3);
        for blk in 0..n / CB {
            let mut acc = [[_mm_setzero_ps(); 4]; 3];
            for kk in 0..k {
                let w = bp.add(kk * n + blk * CB);
                let wv = [
                    _mm_loadu_ps(w),
                    _mm_loadu_ps(w.add(4)),
                    _mm_loadu_ps(w.add(8)),
                    _mm_loadu_ps(w.add(12)),
                ];
                for (r, row) in acc.iter_mut().enumerate().take(rows) {
                    let x = _mm_set1_ps(*ap.add((r0 + r) * k + kk));
                    for (q, a) in row.iter_mut().enumerate() {
                        *a = _mm_add_ps(*a, _mm_mul_ps(x, wv[q]));
                    }
                }
            }
            for (r, row) in acc.iter().enumerate().take(rows) {
                let o = c.as_mut_ptr().add((r0 + r) * n + blk * CB);
                for (q, a) in row.iter().enumerate() {
                    _mm_storeu_ps(o.add(4 * q), *a);
                }
            }
        }
        r0 += rows;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Our conversion must round exactly like the CPU (F16C), so every
    /// kernel set sees the same weights.
    #[test]
    fn half_conversion_matches_f16c() {
        if !(std::arch::is_x86_feature_detected!("f16c")
            && std::arch::is_x86_feature_detected!("avx"))
        {
            return;
        }
        let mut x = 0x1234_5678u32;
        let mut vals = vec![
            0.0f32, -0.0, 1.0, -2.5, 65504.0, 1e-5, -3e-7, 6.1e-5, 1e-8, 0.333, 70000.0, 5.9e-8,
        ];
        for _ in 0..1_000_000 {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            let v = f32::from_bits(x);
            if v.is_finite() {
                // mostly the weight range, some arbitrary finite values
                vals.push(if x.is_multiple_of(4) {
                    v
                } else {
                    (x as f32 / u32::MAX as f32 - 0.5) * 4.0
                });
            }
        }
        for chunk in vals.chunks_exact(8) {
            // SAFETY: f16c + avx checked above.
            let (hw_h, hw_f) = unsafe { hw(chunk) };
            for i in 0..8 {
                let h = f32_to_f16(chunk[i]);
                assert_eq!(h, hw_h[i], "rounding of {}", chunk[i]);
                assert_eq!(
                    f16_to_f32(h).to_bits(),
                    hw_f[i].to_bits(),
                    "widening of {h:#x}"
                );
            }
        }
    }

    #[target_feature(enable = "avx,f16c")]
    unsafe fn hw(v: &[f32]) -> ([u16; 8], [f32; 8]) {
        use std::arch::x86_64::*;
        let h = _mm256_cvtps_ph::<{ _MM_FROUND_TO_NEAREST_INT }>(_mm256_loadu_ps(v.as_ptr()));
        let mut hs = [0u16; 8];
        _mm_storeu_si128(hs.as_mut_ptr() as *mut __m128i, h);
        let mut fs = [0f32; 8];
        _mm256_storeu_ps(fs.as_mut_ptr(), _mm256_cvtph_ps(h));
        (hs, fs)
    }
}
