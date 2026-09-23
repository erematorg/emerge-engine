//! Real, direct diagnostic: WHY does the poured pile never spread
//! laterally? Six independent, real fix attempts (post_event_relax_
//! threshold, switch_step-style damping timing, Pradhana v1/v2, MIBF, all
//! combinations) have now measured ZERO effect on the real production pour
//! scenario -- all converge to the same ~88.7-88.9deg non-toppling tower.
//! Rather than guess a seventh mechanism, this directly instruments the
//! REAL kinematics: does ANY particle ever develop meaningful LATERAL
//! (x-direction) velocity during a pour, or is the material moving purely
//! vertically the entire time (a real, direct, checkable fact, not an
//! assumption)? Identical geometry/config to the real, established pour
//! test.

extern crate emerge_engine as emerge;
use emerge::{DruckerPragerMaterial, FrictionBoundary, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

#[test]
#[ignore = "diagnostic, run explicitly with --release --ignored --nocapture"]
fn diag_lateral_motion_during_real_pour() {
    const POUR_GRID: usize = 128;
    const POUR_DT: f32 = 0.016;
    const POUR_FLOOR: f32 = 2.0;
    const N_POURS: usize = 20; // enough to see the pattern, cheaper than the full 45
    const STEPS_BETWEEN_POURS: usize = 15;

    let config = SimConfig {
        max_substeps_per_step: 64,
        apic_blend: 0.05,
        cundall_damping: 0.0,
        ..SimConfig::standard(POUR_GRID, POUR_DT, Vec2::new(0.0, -0.3))
    };
    let cx = POUR_GRID as f32 * 0.5;
    let sand = DruckerPragerMaterial::from_young_modulus(1.0e5, 0.2);

    let seed = SpawnRegion {
        spacing: 0.25,
        box_size: IVec2::new(4, 1),
        box_center: Vec2::new(cx, POUR_FLOOR + 0.5),
        material_id: 0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, seed)
        .with_default_material(Box::new(sand))
        .with_boundary(Box::new(FrictionBoundary::new(2, 0.7)));

    for i in 0..N_POURS {
        let xs_now = &solver.particles().x;
        let surface_y = xs_now
            .iter()
            .filter(|p| (p.x - cx).abs() < 4.0)
            .map(|p| p.y)
            .fold(POUR_FLOOR, f32::max);
        let batch = SpawnRegion {
            spacing: 0.25,
            box_size: IVec2::new(3, 1),
            box_center: Vec2::new(cx, surface_y + 2.0),
            material_id: 0,
            rng_seed: 200 + i as u32,
            position_jitter: 0.15,
            ..SpawnRegion::for_sim(solver.config())
        };
        let _ = solver.add_body(batch);

        // Real, direct per-substep instrumentation for this pour's own
        // window -- step ONE substep at a time (not step_n) so we can see
        // the real, immediate dynamics of THIS specific impact, not just
        // the state after it's already settled.
        let mut max_abs_vx_this_pour = 0.0f32;
        let mut max_abs_vy_this_pour = 0.0f32;
        for _ in 0..STEPS_BETWEEN_POURS {
            solver.step();
            let vs = &solver.particles().v;
            for v in vs.iter() {
                max_abs_vx_this_pour = max_abs_vx_this_pour.max(v.x.abs());
                max_abs_vy_this_pour = max_abs_vy_this_pour.max(v.y.abs());
            }
        }

        let xs = &solver.particles().x;
        let base_xs: Vec<f32> = xs
            .iter()
            .filter(|p| p.y < POUR_FLOOR + 1.5)
            .map(|p| p.x)
            .collect();
        let base_min = base_xs.iter().copied().fold(f32::INFINITY, f32::min);
        let base_max = base_xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let base_half_width = ((base_max - base_min) * 0.5).max(0.0);

        // Real, direct check: has ANY particle ever yielded at all?
        // `friction_hardening` (q) starts at the material's own real
        // neutral-rest value (`friction_residual/hardening_peak`) and only
        // moves away from it via a real plastic (shear-yield or
        // tension-cutoff) event -- if it's frozen at that same initial
        // value for every particle, NOTHING has ever crossed the yield
        // surface, regardless of any yield-adjacent fix.
        let q = &solver.particles().friction_hardening;
        let q_min = q.iter().copied().fold(f32::INFINITY, f32::min);
        let q_max = q.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        // Real, known neutral-rest value for `from_young_modulus`'s own
        // default hardening params (`friction_residual/hardening_peak` =
        // 10deg/9deg), matching `init_particle`'s own real formula --
        // NOT read off particle 0 (which could itself already have
        // yielded), a real, independently-computed reference value.
        const Q_NEUTRAL: f32 = 10.0 / 9.0;
        let n_yielded = q
            .iter()
            .filter(|&&v| (v - Q_NEUTRAL).abs() > 1.0e-4)
            .count();

        println!(
            "pour {i:2}: n={:5} surface_y={surface_y:7.2} base_half_width={base_half_width:5.2} \
             max|vx|={max_abs_vx_this_pour:8.5} max|vy|={max_abs_vy_this_pour:8.5} \
             q_range=[{q_min:.4},{q_max:.4}] n_yielded={n_yielded}",
            xs.len()
        );
    }

    // Real, final direct check on a handful of individual particles near
    // the surface and near the base -- their own real position history
    // isn't tracked here (would need per-particle IDs across add_body
    // calls, real future work if this diagnostic doesn't already answer
    // the question), but their CURRENT velocity/state is real and direct.
    let xs = &solver.particles().x;
    let vs = &solver.particles().v;
    let n = xs.len();
    println!("\nFinal state after {N_POURS} pours, {n} particles:");
    let mut sample: Vec<usize> = (0..n).step_by((n / 10).max(1)).collect();
    sample.truncate(10);
    for &i in &sample {
        println!(
            "  particle {i}: x=({:.3},{:.3}) v=({:.6},{:.6})",
            xs[i].x, xs[i].y, vs[i].x, vs[i].y
        );
    }
}
