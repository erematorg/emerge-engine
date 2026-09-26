//! Real, cheap consolidation: `src/matter/materials/utils.rs::
//! corotated_elastic_stress` (verified via JVP/finite-difference/symmetry in
//! `scratch_implicit_mpm_stage2_corotated_jvp.rs`) is the SAME shared
//! function `DruckerPragerMaterial` (sand), `VonMisesMaterial`,
//! `RankineMaterial`, and `MuIRheologyMaterial` all call directly for their
//! own `kirchhoff_stress`'s elastic branch (`elastic_viscosity=0.0`,
//! confirmed by reading each file directly, not assumed) -- so the already-
//! verified JVP/symmetric-Hessian math for Corotated applies to all FIVE
//! materials' pure-elastic stress evaluation without a fresh per-material
//! derivation. This test confirms that equivalence holds for the REAL
//! materials (catches a wrong lambda/mu wiring or an accidental extra term,
//! which a pure formula re-derivation wouldn't catch).
//!
//! **Real, disclosed scope limit, not swept under the rug**: this covers
//! only the ELASTIC branch (stress as a function of F on its own). The
//! actual Stage 3 target (DruckerPrager sand's real-time problem) applies a
//! PLASTIC return-mapping to F first (`update_particle`) -- the derivative
//! of that composite map is a real, separate, harder question (Klar 2016's
//! own cited symmetry-breaking caveat, Simo & Taylor 1985's "consistent
//! tangent operator" is the standard real fix) and is NOT checked here.
//!
//! `cargo test --release --test scratch_implicit_mpm_stage2_shared_elastic_branch_check -- --nocapture`

extern crate emerge_engine as emerge;
use emerge::materials::{
    DruckerPragerMaterial, MaterialModel, MuIRheologyMaterial, RankineMaterial, VonMisesMaterial,
};
use emerge::particle::{Particle, Particles};
use glam::{Mat2, Vec2};

fn frob(a: Mat2, b: Mat2) -> f32 {
    a.x_axis.x * b.x_axis.x
        + a.x_axis.y * b.x_axis.y
        + a.y_axis.x * b.y_axis.x
        + a.y_axis.y * b.y_axis.y
}

/// Same real formula already verified in the Corotated Stage 2 file --
/// duplicated here (not imported, scratch tests don't share modules) so
/// this file stands alone as its own real check.
fn corotated_elastic_stress_reference(f: Mat2, lambda: f32, mu: f32) -> Mat2 {
    let j = f.determinant();
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    let r = if norm > f32::EPSILON {
        Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm)
    } else {
        Mat2::IDENTITY
    };
    2.0 * mu * (f - r) * f.transpose() + lambda * (j - 1.0) * j * Mat2::IDENTITY
}

fn one_particle(f: Mat2) -> Particles {
    let mut p = Particle::zeroed();
    p.deformation_gradient = f;
    p.mass = 1.0;
    p.initial_volume = 1.0;
    p.volume = 1.0;
    p.density = 1.0;
    p.hardening_scale = 1.0;
    p.plastic_volume_ratio = 1.0;
    let particles: Particles = vec![p].into();
    particles
}

fn states() -> Vec<Mat2> {
    vec![
        Mat2::IDENTITY,
        Mat2::from_cols(Vec2::new(1.3, 0.0), Vec2::new(0.0, 0.8)),
        Mat2::from_cols(Vec2::new(1.0, 0.4), Vec2::new(0.1, 1.1)),
        Mat2::from_cols(Vec2::new(0.7, 0.6), Vec2::new(-0.5, 1.1)),
    ]
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn drucker_prager_sand_elastic_branch_matches_shared_corotated_formula() {
    let mat = DruckerPragerMaterial::cohesionless(2.0e3, 0.3);
    assert_eq!(
        mat.elastic_viscosity, 0.0,
        "test assumes default viscosity=0"
    );
    for &f in &states() {
        let particles = one_particle(f);
        let real = mat.kirchhoff_stress(&particles, 0);
        let reference = corotated_elastic_stress_reference(f, mat.lambda, mat.mu);
        let diff = real - reference;
        assert!(
            frob(diff, diff).sqrt() < 1.0e-4,
            "F={f:?} real={real:?} reference={reference:?}"
        );
    }
    println!(
        "DruckerPrager sand: elastic-branch stress matches the shared, already-JVP-verified formula exactly"
    );
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn von_mises_elastic_branch_matches_shared_corotated_formula() {
    let mat = VonMisesMaterial::from_young_modulus(50.0, 0.3, 10.0);
    assert_eq!(
        mat.elastic_viscosity, 0.0,
        "test assumes default viscosity=0"
    );
    for &f in &states() {
        let particles = one_particle(f);
        let real = mat.kirchhoff_stress(&particles, 0);
        let reference = corotated_elastic_stress_reference(f, mat.lambda, mat.mu);
        let diff = real - reference;
        assert!(
            frob(diff, diff).sqrt() < 1.0e-4,
            "F={f:?} real={real:?} reference={reference:?}"
        );
    }
    println!(
        "VonMises: elastic-branch stress matches the shared, already-JVP-verified formula exactly"
    );
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn rankine_elastic_branch_matches_shared_corotated_formula() {
    let mat = RankineMaterial::stiff_brittle(50.0, 0.3);
    assert_eq!(
        mat.elastic_viscosity, 0.0,
        "test assumes default viscosity=0"
    );
    for &f in &states() {
        let particles = one_particle(f);
        let real = mat.kirchhoff_stress(&particles, 0);
        let reference = corotated_elastic_stress_reference(f, mat.lambda, mat.mu);
        let diff = real - reference;
        assert!(
            frob(diff, diff).sqrt() < 1.0e-4,
            "F={f:?} real={real:?} reference={reference:?}"
        );
    }
    println!(
        "Rankine: elastic-branch stress matches the shared, already-JVP-verified formula exactly"
    );
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn mui_rheology_elastic_branch_matches_shared_corotated_formula() {
    let mat = MuIRheologyMaterial::small_grain(50.0, 0.3);
    for &f in &states() {
        let particles = one_particle(f);
        let real = mat.kirchhoff_stress(&particles, 0);
        let reference = corotated_elastic_stress_reference(f, mat.lambda, mat.mu);
        let diff = real - reference;
        assert!(
            frob(diff, diff).sqrt() < 1.0e-4,
            "F={f:?} real={real:?} reference={reference:?}"
        );
    }
    println!(
        "MuIRheology: elastic-branch stress matches the shared, already-JVP-verified formula exactly"
    );
}
