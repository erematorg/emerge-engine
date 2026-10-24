//! The volume ratio `J = det F`: how much a small piece of matter has grown or
//! shrunk compared with its rest state.
//!
//! Source: Gurtin, Fried & Anand (2010), ch. 8 (deformation of volume and area)
//! and ch. 16 (Reynolds' transport relation), which give `dJ/dt = J div v`, the
//! continuity equation used here.

/// Floor applied by legacy solid/plastic constitutive laws before using
/// `det(F)=J` in a singular expression. Strict WC-MPM liquid state does not
/// use this floor: it keeps positive `J` through its exponential continuity
/// update and reports an inadmissible state instead of clamping it.
pub(crate) const MIN_J: f32 = 1e-6;

/// One substep of the continuity equation for a material that owns its
/// own volume, carried in the log so a small increment is not absorbed.
///
/// The scalar twin of `advance_deformation_gradient`, for the same
/// reason. Reading `J` back from an isotropic `F` each substep and
/// multiplying, `J_new = det(F) * exp(dt div v)`, loses precision: near one, an
/// f32 has a resolution of about 1.2e-7, while a calm flow's own
/// increment is a thousandth of that, so each step loses a fixed
/// FRACTION of its increment to absorption and the smallest increments
/// vanish outright. Measured with no solver and no grid
/// (`tests/probes/fluid_j_rounding.rs`), against an f64 replica of the
/// same update that holds J at exactly 1.000000: f32 walked to 0.999468
/// at a divergence of 0.2 per second, and at the finest increment it
/// froze completely, 0.000 drift, J stuck.
///
/// Adding `dt div v` to `ln J` instead keeps the increment: near J = 1
/// the logarithm is near zero, where f32 resolution is not 1.2e-7 but
/// vanishingly small. The clamp is applied in the same place, in the log,
/// so a bounded material stays bounded.
///
/// Returns the carried logarithm and the `J` it means.
#[inline]
pub(crate) fn advance_log_volume_ratio(
    carried_log_j: f32,
    dt_div_v: f32,
    min_j: f32,
    max_j: f32,
) -> (f32, f32) {
    let advanced = carried_log_j + dt_div_v;
    let clamped = advanced.clamp(min_j.max(f32::MIN_POSITIVE).ln(), max_j.ln());
    (clamped, clamped.exp())
}
