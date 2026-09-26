//! Minimal 2D discrete cosine transform (DCT-II forward / its exact
//! algebraic inverse), used by `pressure.rs`'s exact constant-density
//! Poisson solve (see that module's own doc for why an exact transform-based
//! solve replaced an iterative Jacobi/Gauss-Seidel one).
//!
//! O(N log N) per row/column via `rustfft`, replacing an earlier direct
//! O(N^2) summation. Real motivation, not a micro-optimization for its own
//! sake: found live 2026-08-09 via direct per-phase timing on
//! `fluid_pressure_projection_gui.rs` that this transform was 88% of the
//! total per-substep pressure-projection cost (~17ms of ~19.4ms on a 63x61
//! active region) -- the actual reason that demo, which structurally has NO
//! acoustic-CFL substep explosion at all (`eos_stiffness=0`,
//! incompressibility comes from this Poisson solve instead), still only ran
//! at ~2fps. `rustfft` (not a hand-rolled arbitrary-length FFT) specifically
//! because the active-region bounding box -- this transform's own N -- is
//! whatever size the fluid body currently occupies, not a power of two
//! chosen in advance.
//!
//! Construction: zero-pad the length-N real signal to 2N, take a 2N-point
//! complex FFT, and read the DCT-II coefficients off a per-bin phase twiddle
//! -- the direct algebraic consequence of `X_k = Re(exp(-i*pi*k/(2N)) *
//! DFT_{2N}(pad(x))[k])` (derived from the DCT-II sum by writing
//! `cos(theta) = Re(exp(-i*theta))` and splitting the phase). The inverse
//! (DCT-III) is the same relation solved the other way: a padded,
//! phase-twiddled spectrum fed through an unnormalized inverse 2N-point FFT.
//! Both derivations verified BY HAND for N=1 and N=2 (not just asserted) and
//! then checked numerically against the OLD O(N^2) direct-sum reference
//! (kept as a `#[cfg(test)]`-only oracle) across even/odd/prime sizes,
//! including a first version that was WRONG (a mis-remembered "Makhoul"
//! reordering with a spurious factor of 2, caught immediately by this same
//! test suite, not shipped) -- this is the corrected, re-derived version.

use std::cell::RefCell;

use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;

thread_local! {
    // Real, necessary reuse, not a micro-optimization: found live 2026-08-09
    // that building a FRESH `FftPlanner` per call (the first version of this
    // fix) gave ZERO fps improvement over the old O(N^2) code, despite
    // passing every correctness test -- planning a non-power-of-two length
    // (this active region's own size, e.g. 2*61=122) does real, repeated
    // setup work internally (Bluestein/mixed-radix strategy construction),
    // and `pressure.rs` calls this every substep. `rustfft`'s own planner
    // caches plans BY LENGTH within one planner instance -- this thread-
    // local keeps ONE instance alive for the process's lifetime so that
    // cache actually pays off across substeps/frames instead of being
    // rebuilt and discarded every single call.
    static PLANNER: RefCell<FftPlanner<f32>> = RefCell::new(FftPlanner::new());
}

/// 1D DCT-II via a 2N-point FFT: `X_k = Re(exp(-i*pi*k/(2N)) *
/// DFT_{2N}(pad(x))[k])` for `k=0..N-1`, where `pad(x)` is `x` zero-padded to
/// length `2N`. Unnormalized forward transform, same convention as the old
/// direct-sum version -- `idct_1d_fast` is its exact algebraic inverse.
/// `fft_2n` must be a forward plan of size `2*x.len()`.
fn dct_1d_fast(x: &[f32], fft_2n: &dyn rustfft::Fft<f32>) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return Vec::new();
    }
    let mut buf = vec![Complex32::new(0.0, 0.0); 2 * n];
    for (i, &xi) in x.iter().enumerate() {
        buf[i] = Complex32::new(xi, 0.0);
    }
    fft_2n.process(&mut buf);
    let mut out = vec![0.0f32; n];
    for (k, out_k) in out.iter_mut().enumerate() {
        let theta = std::f32::consts::PI * k as f32 / (2.0 * n as f32);
        let (sin_t, cos_t) = theta.sin_cos();
        // Re((a+bi) * (cos_t - i*sin_t)) = a*cos_t + b*sin_t.
        *out_k = buf[k].re * cos_t + buf[k].im * sin_t;
    }
    out
}

/// Exact inverse of `dct_1d_fast` (DCT-III). Derived from the same forward
/// relation solved for `x_i`: `x_i = (1/N) * Re(IFFT_{2N,unnormalized}(W)[i])`
/// where `W_k = c_k * X_k * exp(i*pi*k/(2N))` for `k=0..N-1` (zero-padded to
/// `2N`), `c_0=1`, `c_k=2` for `k>0` -- verified by hand for N=1 (recovers
/// x_0 exactly) and N=2 (recovers both x_0, x_1 exactly), then numerically
/// against the direct-sum reference for the full even/odd/prime sweep below.
/// `ifft_2n` must be an UNNORMALIZED inverse plan (`rustfft`'s own
/// convention -- it does not divide by N itself) of size `2*x_hat.len()`.
fn idct_1d_fast(x_hat: &[f32], ifft_2n: &dyn rustfft::Fft<f32>) -> Vec<f32> {
    let n = x_hat.len();
    if n == 0 {
        return Vec::new();
    }
    let mut w = vec![Complex32::new(0.0, 0.0); 2 * n];
    for (k, &xk) in x_hat.iter().enumerate() {
        let c_k = if k == 0 { 1.0 } else { 2.0 };
        let theta = std::f32::consts::PI * k as f32 / (2.0 * n as f32);
        let (sin_t, cos_t) = theta.sin_cos();
        w[k] = Complex32::new(c_k * xk * cos_t, c_k * xk * sin_t);
    }
    ifft_2n.process(&mut w);
    let mut out = vec![0.0f32; n];
    for (i, out_i) in out.iter_mut().enumerate() {
        *out_i = w[i].re / n as f32;
    }
    out
}

/// 2D forward transform, separable: DCT along y (fixed x, vary y), then
/// along x (fixed y, vary x). `data` is row-major, `nx * ny`
/// (`data[x*ny+y]`). Rectangular, not just square -- real requirement so
/// `pressure.rs` can scope this to a small bounding box of the actually
/// active fluid region instead of the full (possibly sparse, possibly huge)
/// grid resolution -- see that module's own doc for why locking this to a
/// dense full-domain transform would be a real regression against this
/// engine's sparse-grid design.
pub(super) fn dct2_forward(data: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let fft_2ny = PLANNER.with_borrow_mut(|p| p.plan_fft_forward(2 * ny));
    let mut tmp = vec![0.0f32; nx * ny];
    for x in 0..nx {
        let row_hat = dct_1d_fast(&data[x * ny..(x + 1) * ny], fft_2ny.as_ref());
        tmp[x * ny..(x + 1) * ny].copy_from_slice(&row_hat);
    }
    let fft_2nx = PLANNER.with_borrow_mut(|p| p.plan_fft_forward(2 * nx));
    let mut out = vec![0.0f32; nx * ny];
    for y in 0..ny {
        let col: Vec<f32> = (0..nx).map(|x| tmp[x * ny + y]).collect();
        let col_hat = dct_1d_fast(&col, fft_2nx.as_ref());
        for (x, &v) in col_hat.iter().enumerate() {
            out[x * ny + y] = v;
        }
    }
    out
}

/// 2D inverse transform, separable, exact inverse of `dct2_forward` --
/// reverses the axis order (column-then-row inverse undoes a row-then-column
/// forward), same real requirement any separable 2D transform pair has.
pub(super) fn dct2_inverse(data_hat: &[f32], nx: usize, ny: usize) -> Vec<f32> {
    let ifft_2nx = PLANNER.with_borrow_mut(|p| p.plan_fft_inverse(2 * nx));
    let mut tmp = vec![0.0f32; nx * ny];
    for y in 0..ny {
        let col: Vec<f32> = (0..nx).map(|x| data_hat[x * ny + y]).collect();
        let col_inv = idct_1d_fast(&col, ifft_2nx.as_ref());
        for (x, &v) in col_inv.iter().enumerate() {
            tmp[x * ny + y] = v;
        }
    }
    let ifft_2ny = PLANNER.with_borrow_mut(|p| p.plan_fft_inverse(2 * ny));
    let mut out = vec![0.0f32; nx * ny];
    for x in 0..nx {
        let row_inv = idct_1d_fast(&tmp[x * ny..(x + 1) * ny], ifft_2ny.as_ref());
        out[x * ny..(x + 1) * ny].copy_from_slice(&row_inv);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Direct O(N^2) reference, the module's ORIGINAL implementation before
    /// the FFT-based rewrite -- kept only as a `#[cfg(test)]` oracle to
    /// numerically validate the fast version against, not deleted outright.
    fn dct_1d_reference(x: &[f32]) -> Vec<f32> {
        let n = x.len();
        let mut out = vec![0.0f32; n];
        for (k, out_k) in out.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for (i, &xi) in x.iter().enumerate() {
                sum += xi * (std::f32::consts::PI / n as f32 * (i as f32 + 0.5) * k as f32).cos();
            }
            *out_k = sum;
        }
        out
    }

    fn dct_1d_fast_owned(x: &[f32]) -> Vec<f32> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(2 * x.len());
        dct_1d_fast(x, fft.as_ref())
    }

    fn idct_1d_fast_owned(x_hat: &[f32]) -> Vec<f32> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_inverse(2 * x_hat.len());
        idct_1d_fast(x_hat, fft.as_ref())
    }

    /// Real, direct numerical check that the FFT-based forward transform
    /// agrees with the direct O(N^2) reference -- for even, odd, AND prime
    /// sizes (the active-region bounding box in `pressure.rs` is an
    /// arbitrary runtime size, not chosen to be FFT-friendly), not just
    /// round-trip self-consistency (which a consistently-wrong pair could
    /// still pass -- exactly how the first, wrong version of this file
    /// would NOT have been caught by round-trip alone).
    #[test]
    fn fft_dct_matches_direct_sum_reference_for_various_sizes() {
        for n in [1usize, 2, 3, 4, 5, 7, 8, 11, 13, 16, 31, 61, 63] {
            let x: Vec<f32> = (0..n).map(|i| ((i * 7 + 3) % 13) as f32 - 6.0).collect();
            let expected = dct_1d_reference(&x);
            let actual = dct_1d_fast_owned(&x);
            for (k, (&e, &a)) in expected.iter().zip(actual.iter()).enumerate() {
                assert!(
                    (e - a).abs() < 1.0e-2 * n as f32,
                    "n={n} k={k}: reference={e} fast={a}"
                );
            }
        }
    }

    /// Real, direct numerical check that `idct_1d_fast` recovers the
    /// original signal for the same size sweep.
    #[test]
    fn dct_round_trip_recovers_original_signal_1d() {
        for n in [1usize, 2, 3, 4, 5, 7, 8, 11, 13, 16, 31, 61, 63] {
            let x: Vec<f32> = (0..n).map(|i| ((i * 5 + 1) % 9) as f32 - 4.0).collect();
            let x_hat = dct_1d_fast_owned(&x);
            let x_back = idct_1d_fast_owned(&x_hat);
            for (i, (&a, &b)) in x.iter().zip(x_back.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1.0e-2 * n as f32,
                    "n={n} i={i}: round trip mismatch: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    fn dct2_round_trip_recovers_original_signal() {
        let (nx, ny) = (8, 8);
        let mut data = vec![0.0f32; nx * ny];
        for x in 0..nx {
            for y in 0..ny {
                data[x * ny + y] = ((x * 3 + y * 7) % 11) as f32 - 5.0;
            }
        }
        let hat = dct2_forward(&data, nx, ny);
        let back = dct2_inverse(&hat, nx, ny);
        for (a, b) in data.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1.0e-1, "2D round trip mismatch: {a} vs {b}");
        }
    }

    /// Real check that rectangular (non-square) domains work too -- the
    /// whole point of generalizing past a fixed `n x n` was so `pressure.rs`
    /// can scope this to an arbitrary bounding box, not just a square one.
    /// Deliberately includes odd/prime dimensions (5, 11), matching the
    /// live-observed 63x61 active-region size this fix was measured against.
    #[test]
    fn dct2_round_trip_recovers_original_signal_rectangular() {
        let (nx, ny) = (5, 11);
        let mut data = vec![0.0f32; nx * ny];
        for x in 0..nx {
            for y in 0..ny {
                data[x * ny + y] = ((x * 2 + y * 5) % 7) as f32 - 3.0;
            }
        }
        let hat = dct2_forward(&data, nx, ny);
        let back = dct2_inverse(&hat, nx, ny);
        for (a, b) in data.iter().zip(back.iter()) {
            assert!(
                (a - b).abs() < 1.0e-1,
                "rectangular round trip mismatch: {a} vs {b}"
            );
        }
    }
}
