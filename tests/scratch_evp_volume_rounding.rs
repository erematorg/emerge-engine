//! Does the yield-stress law gain volume on its own?
//!
//! The slab sweep established that the drift needs motion (zero gravity
//! gives exactly 0.00000), is not about thin layers (a one-cell layer is
//! flat, a sixteen-cell one drifts most), and belongs to the law rather
//! than the transfer (the same slabs in NeoHookean drift ten to fifty
//! times less and change sign). What it could not isolate is WHICH part
//! of the elastoviscoplastic path does it, because the purely viscous
//! branch cannot run that scene at all.
//!
//! So this removes the solver entirely: one particle, a prescribed
//! velocity gradient of zero trace, driven straight through
//! `update_particle`. Volume can only change through the trace, so J must
//! come back. Anything it keeps is the law's own arithmetic.
extern crate emerge_engine as emerge;

use emerge::{
    BinghamFluidMaterial, BinghamProps, FromSI, MaterialModel, NeoHookeanMaterial, Particle,
    Particles, SimConfig,
};
use glam::{Mat2, Vec2};

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn a_prescribed_shear_cycle_must_not_change_volume() {
    let steps: usize = std::env::var("EVP_STEPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(200_000);
    let amp: f32 = std::env::var("EVP_AMP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let dt = 1.0e-4f32;
    let config = SimConfig::earth(64, 0.002, 0.002);
    let props = BinghamProps {
        rho_kg_m3: 1000.0,
        eta_pa_s: 0.5,
        bulk_modulus_pa: 78_480.0,
        yield_stress_pa: 2.0,
        shear_modulus_pa: 40.0,
    };
    let laws: Vec<(&str, Box<dyn MaterialModel>)> = vec![
        (
            "Bingham EVP",
            Box::new(BinghamFluidMaterial::from_physical(&props, &config)),
        ),
        (
            "NeoHookean",
            Box::new(NeoHookeanMaterial::from_young_modulus(2.0e5, 0.3)),
        ),
    ];

    // One oscillation per 200 steps, the order the slab scenes show.
    let omega = std::f32::consts::TAU / 200.0;
    for (name, law) in laws {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.density = 1.0;
        law.init_particle(&mut p);
        let mut particles = Particles::from(vec![p]);
        let start_j = particles.deformation_gradient[0].determinant();

        let mut exact = 0.0f64;
        for step in 1..=steps {
            let phase = omega * step as f32;
            // Pure shear plus a deviatoric stretch, both oscillating: the
            // trace is zero at every step, so the exact answer never moves.
            let c = Mat2::from_cols(
                Vec2::new(amp * phase.cos(), amp * (phase * 0.37).sin()),
                Vec2::new(amp * (phase * 0.61).cos(), -amp * phase.cos()),
            );
            particles.velocity_gradient[0] = c;
            law.update_particle(&mut particles.update_ctx(0), dt);
            exact += f64::from(dt) * f64::from(c.x_axis.x + c.y_axis.y);
            if step % (steps / 5).max(1) == 0 {
                let j = particles.deformation_gradient[0].determinant();
                println!(
                    "  {name:>12} step {step:>7}: J={j:.6}  ln J={:+.3e}  exact={exact:+.3e}",
                    f64::from(j).ln()
                );
            }
        }
        let end_j = particles.deformation_gradient[0].determinant();
        println!(
            "  {name:>12}: J went {start_j:.6} -> {end_j:.6} over {steps} steps of a zero-trace cycle"
        );
    }
}
