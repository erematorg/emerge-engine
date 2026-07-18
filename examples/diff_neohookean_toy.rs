//! Toy gradient-descent sanity check for `NeoHookeanMaterial::kirchhoff_stress_vjp`.
//!
//! Not a simulation demo -- this doesn't touch P2G/G2P/the solver at all. It's
//! a real, visible proof that the analytic adjoint is USEFUL for optimization,
//! not just numerically correct against finite differences (which the unit
//! tests in `src/matter/materials/elastic.rs` already establish).
//!
//! Task: given a target Kirchhoff stress tensor, use gradient descent on the
//! deformation gradient F (starting from the identity, i.e. undeformed) to
//! find an F that produces it. Loss L = 0.5 * ||tau(F) - tau_target||^2 (squared
//! Frobenius distance). dL/dtau = tau(F) - tau_target feeds directly into
//! kirchhoff_stress_vjp to get dL/dF, then F -= lr * dL/dF.
//!
//! This is the actual mechanism a real trainer would use per-particle, per-
//! substep, chained backward through many more steps (P2G/G2P/grid update,
//! not yet differentiable) -- this example isolates just the one piece that
//! IS differentiable right now and proves it drives real convergence.
//!
//! Run: cargo run --example diff_neohookean_toy

extern crate emerge_engine as emerge;

use emerge::materials::NeoHookeanMaterial;
use emerge::{MaterialModel, Particle, Particles};
use glam::{Mat2, Vec2};

fn particle_with_f(f: Mat2) -> Particles {
    let mut particles = Particles::default();
    particles.push(Particle {
        x: Vec2::ZERO,
        v: Vec2::ZERO,
        velocity_gradient: Mat2::ZERO,
        deformation_gradient: f,
        mass: 1.0,
        initial_volume: 1.0,
        volume: 1.0,
        density: 1.0,
        material_id: 0,
        plastic_volume_ratio: 1.0,
        hardening_scale: 1.0,
        friction_hardening: 0.0,
        log_volume_strain: 0.0,
        temperature: 0.0,
        scalar_field: 0.0,
        user_tag: 0,
        activation: 0.0,
        activation_dir: Vec2::ZERO,
        muscle_group_id: 0,
        contact_group: 0,
        sleeping: 0,
        pinned: 0,
        _pad: 0,
    });
    particles
}

fn frobenius_norm_sq(m: Mat2) -> f32 {
    m.x_axis.x * m.x_axis.x
        + m.x_axis.y * m.x_axis.y
        + m.y_axis.x * m.y_axis.x
        + m.y_axis.y * m.y_axis.y
}

fn main() {
    let mat = NeoHookeanMaterial::new(1000.0, 800.0);

    // A real, physically-reasonable target: the stress produced by a modest
    // stretch-plus-shear deformation, computed once and then "forgotten" --
    // gradient descent only ever sees tau_target, never the F that produced it.
    let hidden_f = Mat2::from_cols(Vec2::new(1.25, 0.15), Vec2::new(-0.08, 0.85));
    let tau_target = {
        let particles = particle_with_f(hidden_f);
        mat.kirchhoff_stress(&particles, 0)
    };

    println!("Target stress: {tau_target:?}");
    println!("(produced by a hidden F the optimizer never sees: {hidden_f:?})\n");

    // Start from the undeformed state -- deliberately far from the answer.
    let mut f = Mat2::IDENTITY;
    let lr = 1.0e-7;
    let steps = 2000;

    let mut losses = Vec::with_capacity(steps);

    for step in 0..steps {
        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        let residual = tau - tau_target;
        let loss = 0.5 * frobenius_norm_sq(residual);
        losses.push(loss);

        if step % 50 == 0 || step == steps - 1 {
            println!("step {step:4}  loss={loss:12.6}  F={f:?}");
        }

        // dL/dtau = residual (derivative of 0.5*||tau - target||^2 w.r.t. tau).
        let d_loss_d_f = mat.kirchhoff_stress_vjp(&particles, 0, residual);
        f -= d_loss_d_f * lr;
    }

    let initial_loss = losses[0];
    let final_loss = *losses.last().unwrap();
    println!("\nInitial loss: {initial_loss:.6}");
    println!("Final loss:   {final_loss:.6}");
    println!("Reduction:    {:.2}x", initial_loss / final_loss.max(1e-12));
    println!("\nRecovered F:  {f:?}");
    println!("Hidden F was: {hidden_f:?}");
    println!(
        "(these needn't match exactly -- NeoHookean stress depends on F only \
         through B=F*F^T, which is rotation-invariant, so multiple F's can \
         produce the identical stress. The optimizer found A valid answer, \
         not necessarily THE original one -- that's real physics, not a bug.)"
    );

    assert!(
        final_loss < initial_loss * 0.01,
        "gradient descent using kirchhoff_stress_vjp should drive loss down by \
         at least 100x over {steps} steps -- got only {:.2}x (initial={initial_loss:.6}, \
         final={final_loss:.6}). Either the gradient isn't pointing the right way, \
         or the learning rate/step count needs retuning.",
        initial_loss / final_loss.max(1e-12)
    );
    println!("\nConverged: the analytic adjoint is a real, usable descent direction.");
}
