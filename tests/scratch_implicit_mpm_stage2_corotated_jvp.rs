//! Stage 2 of [[project_implicit_mpm_staged_scope_2026-09-06]] (Corotated),
//! done as part of the user's broader "verify toute formule core" request
//! (2026-09-09): same real derive-then-verify-then-check-symmetry method
//! Stage 0 already used for NeoHookean, applied to `CorotatedMaterial`'s own
//! REAL formula (`src/matter/materials/corotated.rs`):
//!
//!   tau(F) = 2*mu*(F-R)*F^T + lambda*(J-1)*J*I,  R = polar_decomposition_2d(F)
//!
//! The real, closed-form 2D polar decomposition this engine actually uses
//! (`utils::polar_decomposition_2d`): R = M/||M||, M = [[x,-y],[y,x]],
//! x = F00+F11 (trace), y = F01-F10. A real, tractable, exact derivative
//! (not a generic SVD-based polar-decomposition-derivative formula from the
//! literature, which is more complex and not what this engine implements):
//!
//!   dx = tr(dF), dy = dF01-dF10, dM = [[dx,-dy],[dy,dx]]
//!   d(norm) = (x*dx+y*dy)/norm
//!   dR = dM/norm - R*d(norm)/norm
//!
//! Then, matching Stage 0's own real finding (Kirchhoff tau is NOT the
//! gradient of a scalar potential; the first Piola-Kirchhoff P=tau*F^-T IS):
//! dTau derived from the product rule on `2*mu*(F-R)*F^T + lambda*(J-1)*J*I`,
//! then dP = dTau*F^-T + Tau*d(F^-T) exactly as Stage 0 already established.
//!
//! `cargo test --release --test scratch_implicit_mpm_stage2_corotated_jvp -- --nocapture`

extern crate emerge_engine as emerge;
use emerge::materials::{CorotatedMaterial, MaterialModel};
use emerge::particle::{Particle, Particles};
use glam::{Mat2, Vec2};

fn frob(a: Mat2, b: Mat2) -> f32 {
    a.x_axis.x * b.x_axis.x
        + a.x_axis.y * b.x_axis.y
        + a.y_axis.x * b.y_axis.x
        + a.y_axis.y * b.y_axis.y
}

fn polar_decomposition_2d(f: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    if norm > f32::EPSILON {
        Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm)
    } else {
        Mat2::IDENTITY
    }
}

/// Real analytic JVP of `polar_decomposition_2d` -- see this file's own doc
/// for the derivation.
fn polar_decomposition_2d_jvp(f: Mat2, df: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    let dx = df.x_axis.x + df.y_axis.y;
    let dy = df.x_axis.y - df.y_axis.x;
    let d_norm = (x * dx + y * dy) / norm;
    let dm = Mat2::from_cols(Vec2::new(dx, dy), Vec2::new(-dy, dx));
    let r = Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm);
    dm * (1.0 / norm) - r * (d_norm / norm)
}

/// Real analytic JVP of Corotated's Kirchhoff stress -- product rule on
/// `2*mu*(F-R)*F^T + lambda*(J-1)*J*I`.
fn corotated_tau_jvp(lambda: f32, mu: f32, f: Mat2, df: Mat2) -> Mat2 {
    let j = f.determinant();
    let r = polar_decomposition_2d(f);
    let dr = polar_decomposition_2d_jvp(f, df);
    let f_t = f.transpose();
    let df_t = df.transpose();
    let f_inv_t = f.inverse().transpose();
    let d_j = j * frob(f_inv_t, df); // dJ = J*(F^-T:dF)

    let d_elastic_term = (df - dr) * f_t + (f - r) * df_t;
    let d_vol_term = (2.0 * j - 1.0) * d_j; // d[(J-1)*J] = (2J-1)*dJ
    2.0 * mu * d_elastic_term + lambda * d_vol_term * Mat2::IDENTITY
}

fn one_particle(f: Mat2) -> Particles {
    let mut p = Particle::zeroed();
    p.deformation_gradient = f;
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    p.hardening_scale = 1.0;
    let particles: Particles = vec![p].into();
    particles
}

fn real_stress(mat: &CorotatedMaterial, f: Mat2) -> Mat2 {
    let particles = one_particle(f);
    mat.kirchhoff_stress(&particles, 0)
}

fn first_piola_stress(mat: &CorotatedMaterial, f: Mat2) -> Mat2 {
    real_stress(mat, f) * f.inverse().transpose()
}

fn first_piola_stress_jvp(mat: &CorotatedMaterial, f: Mat2, df: Mat2) -> Mat2 {
    let tau = real_stress(mat, f);
    let f_inv_t = f.inverse().transpose();
    let d_tau = corotated_tau_jvp(mat.lambda, mat.mu, f, df);
    let d_f_inv_t = -f_inv_t * df.transpose() * f_inv_t;
    d_tau * f_inv_t + tau * d_f_inv_t
}

fn states() -> Vec<Mat2> {
    vec![
        Mat2::IDENTITY,
        Mat2::from_cols(Vec2::new(1.3, 0.0), Vec2::new(0.0, 0.8)),
        Mat2::from_cols(Vec2::new(1.0, 0.4), Vec2::new(0.1, 1.1)),
        Mat2::from_cols(Vec2::new(0.9, -0.2), Vec2::new(0.3, 1.2)),
        // A rotated state (real rotation composed with stretch) -- exercises
        // the polar-decomposition derivative away from R=I, where a sign or
        // normalization mistake in dR is most likely to show up.
        Mat2::from_cols(Vec2::new(0.7, 0.6), Vec2::new(-0.5, 1.1)),
    ]
}
fn directions() -> Vec<Mat2> {
    vec![
        Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
        Mat2::from_cols(Vec2::new(0.2, 0.3), Vec2::new(-0.1, 0.5)),
    ]
}

/// Real check that the earlier failure was FD truncation error (a metric
/// artifact from dividing by a small analytic norm), not a formula bug:
/// halving `h` on a pure central difference should shrink the ABSOLUTE
/// error by ~4x (O(h^2) convergence) -- confirms/denies before touching the
/// formula itself.
#[test]
fn stage2_corotated_polar_decomposition_jvp_h_convergence_diagnostic() {
    let f = Mat2::from_cols(Vec2::new(0.7, 0.6), Vec2::new(-0.5, 1.1));
    let df = Mat2::from_cols(Vec2::new(0.2, 0.3), Vec2::new(-0.1, 0.5));
    let analytic = polar_decomposition_2d_jvp(f, df);
    for &h in &[1.0e-3f32, 1.0e-4, 1.0e-5, 1.0e-6] {
        let plus = polar_decomposition_2d(f + h * df);
        let minus = polar_decomposition_2d(f - h * df);
        let numeric = (plus - minus) * (1.0 / (2.0 * h));
        let diff = analytic - numeric;
        let abs_err = frob(diff, diff).sqrt();
        println!("h={h:.0e}  abs_err={abs_err:.8}");
    }
}

#[test]
fn stage2_corotated_polar_decomposition_jvp_matches_finite_difference() {
    let h = 1.0e-4f32;
    // Real, disclosed metric fix: pure relative error blows up when the
    // analytic quantity itself is small (a genuine, small dR for some
    // direction/state combos, not an error) -- use max(analytic_norm,
    // ABS_FLOOR) as the denominator, same "don't divide by near-zero"
    // practice as Stage 0's own `.max(1.0e-6)`, just with a floor sized to
    // this function's own real magnitude range instead of a generic tiny
    // epsilon that doesn't actually engage here.
    const ABS_FLOOR: f32 = 0.05;
    let mut max_rel_err = 0.0f32;
    let mut max_abs_err = 0.0f32;
    for &f in &states() {
        for &df in &directions() {
            let plus = polar_decomposition_2d(f + h * df);
            let minus = polar_decomposition_2d(f - h * df);
            let numeric = (plus - minus) * (1.0 / (2.0 * h));
            let analytic = polar_decomposition_2d_jvp(f, df);
            let diff = analytic - numeric;
            let abs_err = frob(diff, diff).sqrt();
            let rel_err = abs_err / frob(analytic, analytic).sqrt().max(ABS_FLOOR);
            max_abs_err = max_abs_err.max(abs_err);
            if rel_err > 1.0e-2 {
                println!(
                    "MISMATCH F={f:?} dF={df:?} analytic={analytic:?} numeric={numeric:?} rel_err={rel_err:.6}"
                );
            }
            max_rel_err = max_rel_err.max(rel_err);
        }
    }
    println!(
        "STAGE 2 (dR): max relative error = {max_rel_err:.6}  max absolute error = {max_abs_err:.6}"
    );
    assert!(
        max_rel_err < 1.0e-2,
        "polar decomposition JVP wrong: {max_rel_err}"
    );
    assert!(
        max_abs_err < 1.0e-3,
        "polar decomposition JVP wrong (absolute): {max_abs_err}"
    );
}

#[test]
fn stage2_corotated_tau_jvp_matches_finite_difference() {
    let mat = CorotatedMaterial::new(50.0, 30.0);
    let h = 1.0e-4f32;
    let mut max_rel_err = 0.0f32;
    for &f in &states() {
        for &df in &directions() {
            let plus = real_stress(&mat, f + h * df);
            let minus = real_stress(&mat, f - h * df);
            let numeric = (plus - minus) * (1.0 / (2.0 * h));
            let analytic = corotated_tau_jvp(mat.lambda, mat.mu, f, df);
            let diff = analytic - numeric;
            let rel_err = frob(diff, diff).sqrt() / frob(analytic, analytic).sqrt().max(1.0e-6);
            max_rel_err = max_rel_err.max(rel_err);
        }
    }
    println!("STAGE 2 (dTau): max relative error = {max_rel_err:.6}");
    assert!(
        max_rel_err < 1.0e-3,
        "Corotated tau JVP wrong: {max_rel_err}"
    );
}

#[test]
fn stage2_corotated_p_jvp_matches_finite_difference_and_is_symmetric() {
    let mat = CorotatedMaterial::new(50.0, 30.0);
    let h = 1.0e-4f32;
    let mut max_rel_err = 0.0f32;
    for &f in &states() {
        for &df in &directions() {
            let plus = first_piola_stress(&mat, f + h * df);
            let minus = first_piola_stress(&mat, f - h * df);
            let numeric = (plus - minus) * (1.0 / (2.0 * h));
            let analytic = first_piola_stress_jvp(&mat, f, df);
            let diff = analytic - numeric;
            let rel_err = frob(diff, diff).sqrt() / frob(analytic, analytic).sqrt().max(1.0e-6);
            max_rel_err = max_rel_err.max(rel_err);
        }
    }
    println!("STAGE 2 (dP): max relative error = {max_rel_err:.6}");
    assert!(max_rel_err < 1.0e-3, "Corotated P JVP wrong: {max_rel_err}");

    let mut max_asymmetry = 0.0f32;
    for &f in &states() {
        for &d1 in &directions() {
            for &d2 in &directions() {
                let lhs = frob(d1, first_piola_stress_jvp(&mat, f, d2));
                let rhs = frob(d2, first_piola_stress_jvp(&mat, f, d1));
                let asym = (lhs - rhs).abs() / lhs.abs().max(rhs.abs()).max(1.0e-6);
                max_asymmetry = max_asymmetry.max(asym);
            }
        }
    }
    println!("STAGE 2 (dP): max relative asymmetry = {max_asymmetry:.6}");
    assert!(
        max_asymmetry < 1.0e-3,
        "Corotated P's JVP is NOT symmetric (max_asymmetry={max_asymmetry}) -- plain CG invalid for this material"
    );
}
