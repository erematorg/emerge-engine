//! Power terms of the equations of state (Tait liquids, ideal-gas adiabats),
//! evaluated exactly when the exponent is a whole number.

/// `x.powf(exp)` using exact integer exponentiation-by-squaring (`powi`)
/// when `exp` is a representable integer -- identical result, no
/// transcendental call. Every Tait-EOS exponent this engine ships (7.0,
/// Cole 1948) is a clean integer; this is a pure implementation-cost win in
/// a hot per-particle, per-substep path (kirchhoff_stress + timestep_bound
/// for every fluid material), not a physics change. Falls back to `powf`
/// exactly for any exponent that isn't a clean integer, so behavior for a
/// non-standard EOS power is unchanged.
#[inline]
pub(crate) fn fast_pow(x: f32, exp: f32) -> f32 {
    if exp.fract() == 0.0 && exp.abs() < 32.0 {
        x.powi(exp as i32)
    } else {
        x.powf(exp)
    }
}
