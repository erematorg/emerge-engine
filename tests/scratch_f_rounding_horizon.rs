//! Does repeating the deformation-gradient update lose volume on its own?
//!
//! `det(exp(dt C) F) = exp(dt tr C) det(F)` holds exactly in real arithmetic,
//! so `ln det F` must equal the running sum of `dt tr(C)` whatever C does.
//! This drives one particle through a prescribed oscillation with NO solver,
//! no grid and no gravity, and prints both sides. Any gap is what f32 costs
//! per step. Knobs: ROUND_STEPS, ROUND_DT, ROUND_AMP (the size of C).
//!
//!   ROUND_AMP=1e-6 cargo test --profile quick --test scratch_f_update_rounding -- --ignored --nocapture
extern crate emerge_engine as emerge;

use emerge::{MaterialModel, NoCompressionMaterial, Particle, Particles};
use glam::Mat2;

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn f_update_volume_under_a_prescribed_oscillation() {
    let env = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let dt = env("ROUND_DT", 0.00437) as f32;
    let amp = env("ROUND_AMP", 1.0e-6) as f32;
    let steps = env("ROUND_STEPS", 90_000.0) as usize;
    // One oscillation per 200 steps, the order the anchored scene shows.
    let omega = std::f32::consts::TAU / 200.0;

    let material = NoCompressionMaterial::new(2000.0, 4000.0);
    let mut p = Particle::zeroed();
    p.deformation_gradient = Mat2::IDENTITY;
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    let mut particles = Particles::from(vec![p]);

    let mut exact = 0.0_f64;
    // Same law, same formula, everything in f64: separates what the
    // increment costs from what the repeated product costs.
    let mut f64_f = [1.0_f64, 0.0, 0.0, 1.0];
    // What the f32 increments alone say the volume should be: the sum of
    // ln det of each increment, before any product accumulates.
    let mut f32_increment_ln_det = 0.0_f64;
    // What the f32 increment MATRIX alone loses, before any product.
    let mut f32_increment_only = 0.0_f64;
    // The candidate fix, carried through its own full f32 product.
    let mut fixed_f = Mat2::IDENTITY;
    // Second candidate: f32 storage, but the product itself done in f64
    // and rounded back once -- prices what the accumulation costs.
    let mut mixed_f = Mat2::IDENTITY;
    // Third candidate: ordinary f32 product, then the step's volume
    // pinned onto what the continuity equation says it must be.
    let mut pinned_f = Mat2::IDENTITY;
    // The volume the continuity equation asks for, carried in its own f32
    // accumulator instead of being re-read from a near-singular det(F).
    let mut carried_ln_j = 0.0_f32;
    // Same idea, but carrying the volume ratio itself rather than its
    // logarithm: one multiply per step instead of an add plus an exp.
    let mut vol_f = Mat2::IDENTITY;
    let mut carried_j = 1.0_f32;
    for step in 1..=steps {
        let phase = omega * step as f32;
        // Deviatoric stretch plus a shear, both oscillating: the trace is not
        // zero every step, but its time average is, so an exact integrator
        // must come back to J = 1.
        let c = Mat2::from_cols_array(&[
            amp * phase.cos(),
            amp * (phase * 0.37).sin(),
            amp * (phase * 0.61).cos(),
            -amp * phase.cos(),
        ]);
        let (ia, ib, ic, id) = (
            f64::from(dt) * f64::from(c.x_axis.x),
            f64::from(dt) * f64::from(c.y_axis.x),
            f64::from(dt) * f64::from(c.x_axis.y),
            f64::from(dt) * f64::from(c.y_axis.y),
        );
        let m = increment_exp_f64(ia, ib, ic, id);
        f64_f = [
            m[0] * f64_f[0] + m[1] * f64_f[2],
            m[0] * f64_f[1] + m[1] * f64_f[3],
            m[2] * f64_f[0] + m[3] * f64_f[2],
            m[2] * f64_f[1] + m[3] * f64_f[3],
        ];
        f32_increment_only += f64::from(increment_exp_f32(dt * c).determinant()).ln();
        fixed_f = increment_exp_f32_rescaled(dt * c) * fixed_f;
        {
            let inc = increment_exp_f32(dt * c);
            let (m0, m1, m2, m3) = (
                f64::from(inc.x_axis.x),
                f64::from(inc.y_axis.x),
                f64::from(inc.x_axis.y),
                f64::from(inc.y_axis.y),
            );
            let (f0, f1, f2, f3) = (
                f64::from(mixed_f.x_axis.x),
                f64::from(mixed_f.y_axis.x),
                f64::from(mixed_f.x_axis.y),
                f64::from(mixed_f.y_axis.y),
            );
            mixed_f = Mat2::from_cols_array(&[
                (m0 * f0 + m1 * f2) as f32,
                (m2 * f0 + m3 * f2) as f32,
                (m0 * f1 + m1 * f3) as f32,
                (m2 * f1 + m3 * f3) as f32,
            ]);
        }
        {
            let inc = increment_exp_f32(dt * c);
            carried_ln_j += dt * (c.x_axis.x + c.y_axis.y);
            let target = carried_ln_j.exp();
            let product = inc * pinned_f;
            let det = product.determinant();
            pinned_f = if det > 0.0 {
                product * (target / det).sqrt()
            } else {
                product
            };
        }
        {
            let inc = increment_exp_f32(dt * c);
            // exp(dt tr C), the factor the increment already carries.
            carried_j *= (dt * (c.x_axis.x + c.y_axis.y)).exp();
            let product = inc * vol_f;
            let det = product.determinant();
            vol_f = if det > 0.0 {
                product * (carried_j / det).sqrt()
            } else {
                product
            };
        }
        let before = particles.deformation_gradient[0];
        particles.velocity_gradient[0] = c;
        material.update_particle(&mut particles.update_ctx(0), dt);
        let after = particles.deformation_gradient[0];
        // det(F_new) / det(F_old) is exactly det of the f32 increment.
        f32_increment_ln_det +=
            (f64::from(after.determinant()) / f64::from(before.determinant())).ln();
        exact += f64::from(dt) * f64::from(c.x_axis.x + c.y_axis.y);
        if step % (steps / 9).max(1) == 0 {
            let ln_det = f64::from(particles.deformation_gradient[0].determinant()).ln();
            println!(
                "step={step:7} ln(det F)={ln_det:+.3e} exact={exact:+.3e} gap={:+.3e} gap/step={:+.3e} f64_same_law={:+.3e} f32_total={:+.3e} f32_increment_matrix={:+.3e} rescaled_f32={:+.3e} f64_product_f32_storage={:+.3e} carried_log={:+.3e} carried_ratio={:+.3e}",
                ln_det - exact,
                (ln_det - exact) / step as f64,
                (f64_f[0] * f64_f[3] - f64_f[1] * f64_f[2]).ln() - exact,
                f32_increment_ln_det - exact,
                f32_increment_only - exact,
                f64::from(fixed_f.determinant()).ln() - exact,
                f64::from(mixed_f.determinant()).ln() - exact,
                f64::from(pinned_f.determinant()).ln() - exact,
                f64::from(vol_f.determinant()).ln() - exact
            );
        }
    }
}

/// The engine's own 2x2 closed-form exponential, in f64: same formula, same
/// branches, so the only difference from the shipped f32 one is precision.
fn increment_exp_f64(a: f64, b: f64, c: f64, d: f64) -> [f64; 4] {
    let half_trace = 0.5 * (a + d);
    let half_difference = 0.5 * (a - d);
    let delta_sq = half_difference * half_difference + b * c;
    let (even, odd) = if delta_sq.abs() < 1.0e-8 {
        let x2 = delta_sq * delta_sq;
        (
            1.0 + 0.5 * delta_sq + x2 / 24.0,
            1.0 + delta_sq / 6.0 + x2 / 120.0,
        )
    } else if delta_sq > 0.0 {
        let delta = delta_sq.sqrt();
        (delta.cosh(), delta.sinh() / delta)
    } else {
        let omega = (-delta_sq).sqrt();
        (omega.cos(), omega.sin() / omega)
    };
    let scale = half_trace.exp();
    [
        scale * (even + (a - half_trace) * odd),
        scale * (b * odd),
        scale * (c * odd),
        scale * (even + (d - half_trace) * odd),
    ]
}

/// The same formula again, this time in f32, so the determinant of the
/// increment ITSELF can be read -- the shipped helper is crate-private.
fn increment_exp_f32(m: Mat2) -> Mat2 {
    let (a, b, c, d) = (m.x_axis.x, m.y_axis.x, m.x_axis.y, m.y_axis.y);
    let half_trace = 0.5 * (a + d);
    let half_difference = 0.5 * (a - d);
    let delta_sq = half_difference * half_difference + b * c;
    let (even, odd) = if delta_sq.abs() < 1.0e-8 {
        let x2 = delta_sq * delta_sq;
        (
            1.0 + 0.5 * delta_sq + x2 / 24.0,
            1.0 + delta_sq / 6.0 + x2 / 120.0,
        )
    } else if delta_sq > 0.0 {
        let delta = delta_sq.sqrt();
        (delta.cosh(), delta.sinh() / delta)
    } else {
        let omega = (-delta_sq).sqrt();
        (omega.cos(), omega.sin() / omega)
    };
    let traceless = m - Mat2::from_diagonal(glam::Vec2::splat(half_trace));
    half_trace.exp() * (Mat2::IDENTITY * even + traceless * odd)
}

/// The candidate fix: the continuity equation fixes the increment's
/// determinant at exp(dt tr C) exactly, so rescale the f32 matrix onto it
/// instead of letting round-off decide the volume.
fn increment_exp_f32_rescaled(m: Mat2) -> Mat2 {
    let increment = increment_exp_f32(m);
    let exact = (m.x_axis.x + m.y_axis.y).exp();
    let det = increment.determinant();
    if det > 0.0 {
        increment * (exact / det).sqrt()
    } else {
        increment
    }
}
