//! Where does a flowing fluid's volume go?
//!
//! The slab sweep put the drift on flow rather than on the plastic
//! decomposition: a Newtonian fluid, which rebuilds no F, drifts as much
//! as a yield-stress one and more, at 3 to 8e-8 per substep. That is a
//! fraction of an ULP at J near one, so the suspect is arithmetic, not
//! physics.
//!
//! A fluid in this engine does not CARRY its volume ratio. Every substep
//! it re-derives it from its own already-isotropic F
//! (`old_j = det(F)`, `F = sqrt(J) I`), so J makes a square-root and
//! squaring round trip each time. Phase 1b measured exactly that pattern
//! as the worst of three options for the deformation gradient: re-reading
//! a determinant each step drifted 4.0e-3 where carrying the value drifted
//! 6e-8.
//!
//! Two cheap tests settle whether that is what this is, with no solver and
//! no grid: the same update in f64, and the same flow at half the step.
//! Rounding error per unit time DOUBLES when the step halves, because
//! there are twice as many of them. Truncation error HALVES.
//!
//! # What it found
//!
//! Swept over the size of the divergence, with the f64 replica beside it:
//!
//! ```text
//!   divergence   increment a step   f32 J      f64 J       drift a step
//!     20 /s          8e-4           1.000010   1.000008     1e-10
//!      0.2 /s        8e-6           0.999468   1.000000    -5.3e-9
//!      0.002 /s      8e-8           0.999658   1.000000    -3.4e-9
//!      0.002 /s      2e-8           1.000000   1.000000     0
//! ```
//!
//! At a realistic divergence the f64 replica is exact and the f32 one is
//! not, so this is arithmetic. At the smallest increment the f32 update
//! vanishes entirely and J freezes: below the resolution of a number near
//! one, an increment simply has nowhere to go.
//!
//! Halving the step halves the drift PER STEP (-5.32, -2.81, -1.03e-9),
//! so per unit time it is constant. That is neither plain rounding, which
//! would double, nor truncation, which would halve: it is a relative
//! error on each increment, the fraction f32 loses when a small number is
//! added to one. Carrying ln J in its own accumulator, the recipe phase
//! 1b measured for the deformation gradient, is what keeps that fraction.

extern crate emerge_engine as emerge;

use emerge::{MaterialModel, NewtonianFluidMaterial, Particle, Particles};
use glam::{Mat2, Vec2};

/// The engine's own fluid volume update, in f64: read J back from an
/// isotropic F, advance it by the continuity equation, store it back as
/// `sqrt(J) I`. Same steps, same order, only the precision differs.
fn f64_reference(f: &mut [f64; 2], div_v: f64, dt: f64) {
    // f holds the diagonal of an isotropic F, so det = f[0]*f[1].
    let old_j = f[0] * f[1];
    let j = old_j * (dt * div_v).exp();
    let s = j.sqrt();
    f[0] = s;
    f[1] = s;
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn a_zero_mean_divergence_must_leave_a_fluid_its_volume() {
    let amp: f32 = std::env::var("FLUID_AMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(20.0);
    let seconds: f32 = std::env::var("FLUID_SECONDS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4.0);

    println!("  a divergence of zero mean, amplitude {amp}/s, over {seconds} s of prescribed flow");
    println!("        dt      steps    f32 J       ln J        f64 J       drift per step");

    for dt in [4.0e-5f32, 2.0e-5, 1.0e-5] {
        let steps = (seconds / dt).round() as usize;
        // One oscillation per 200 steps at the coarsest dt, held at the same
        // PHYSICAL frequency as dt shrinks, so every run sees the same flow.
        let omega = std::f32::consts::TAU / (200.0 * 4.0e-5);

        let material = NewtonianFluidMaterial::new(1.0, 1.0e-3, 50.0, 7.0);
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.density = 1.0;
        material.init_particle(&mut p);
        let mut particles = Particles::from(vec![p]);
        let mut reference = [1.0f64, 1.0f64];

        for step in 1..=steps {
            let t = dt * step as f32;
            let div = amp * (omega * t).cos();
            // A pure dilation carries the divergence; the off-diagonal terms
            // would only add shear, which a fluid's own volume book ignores.
            particles.velocity_gradient[0] = Mat2::from_diagonal(Vec2::splat(div * 0.5));
            material.update_particle(&mut particles.update_ctx(0), dt);
            f64_reference(&mut reference, f64::from(div), f64::from(dt));
        }
        let j = particles.deformation_gradient[0].determinant();
        let j64 = reference[0] * reference[1];
        println!(
            "  {dt:>8.1e}  {steps:>8}  {j:>9.6}  {:>+10.3e}  {j64:>9.6}  {:>+14.3e}",
            f64::from(j).ln(),
            (f64::from(j) - 1.0) / steps as f64
        );
    }
}
