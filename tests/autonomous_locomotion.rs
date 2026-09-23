//! Real, permanent proof of the tier-1 "autonomous creature" capability:
//! `Lnn` drives sustained real locomotion with ZERO human input polled
//! anywhere in the loop -- no keyboard, no per-frame steer decision, just
//! the controller's own continuous-time oscillator plus the same real
//! muscle/ratchet mechanism `basic_creature.rs` already uses interactively.
//!
//! `basic_creature.rs` itself gates muscle activation on `steer != 0.0` --
//! a deliberate DEMO-level choice (an ungated crawl looked identical to a
//! fake baked-in "always walks the same way from frame 1" idle state, see
//! that file's own 2026-07-13 fix), not an engine limitation. `Lnn::step`
//! has no coupling to any input source by construction (see its own module
//! doc: "emerge supplies the controller math; it has no opinion on when or
//! whether it runs") -- this test drives it with a FIXED bias baked in at
//! construction, then just steps controller + physics every frame, matching
//! the architecture boundary: emerge proves the CAPABILITY exists, LP
//! decides autonomously when/whether to invoke it.
extern crate emerge_engine as emerge;
use emerge::{
    Lnn, NeoHookeanMaterial, RatchetFrictionBoundary, SimConfig, Simulation, SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;

const GRID: usize = 96;
const DT: f32 = 0.1;
const MAT_BODY: u32 = 0;
const MUSCLE_GROUPS: u32 = 8;
const N_RINGS: usize = 2;
const N_PER_RING: usize = MUSCLE_GROUPS as usize / N_RINGS;
const RING_CROSS_COUPLING: f32 = 0.5;
const MUSCLE_AMPLITUDE: f32 = 0.9;
const CPG_BURN_IN_STEPS: usize = 600;
const FIBER_DIAG: f32 = 3.0;
// A fixed, real steer bias baked in at construction -- the ONLY "decision"
// made in this whole test, made once, before any stepping starts. No
// per-frame input of any kind after that.
const AUTONOMOUS_BIAS: f32 = 1.0;

fn make_cpg_biased(bias: f32) -> Lnn {
    let mut lnn = Lnn::coupled_traveling_wave(N_RINGS, N_PER_RING, 1.0, RING_CROSS_COUPLING);
    lnn.set_ring_bias(0, N_PER_RING, bias);
    lnn.set_ring_bias(1, N_PER_RING, -bias);
    for _ in 0..CPG_BURN_IN_STEPS {
        lnn.step(DT);
    }
    lnn
}

fn make_sim() -> (
    Simulation,
    std::ops::Range<usize>,
    Arc<RatchetFrictionBoundary>,
) {
    let mut mat = NeoHookeanMaterial::new(13.0, 26.0);
    mat.active_stress_coeff = 80.0;
    mat.viscosity = 150.0;
    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(48.0, 20.0);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(36, 4),
        box_center: body_center,
        material_id: MAT_BODY,
        ..SpawnRegion::for_sim(&config)
    };
    let ratchet = Arc::new(RatchetFrictionBoundary::new(4, 0.1, 0.95, Vec2::X));
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(Arc::clone(&ratchet)));

    let body_range = 0..solver.particles().len();
    let body_len = 36.0 * 0.5;
    let body_left = body_center.x - body_len / 2.0;
    {
        let particles = solver.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / body_len).clamp(0.0, 1.0);
            let group = ((t * MUSCLE_GROUPS as f32) as u32).min(MUSCLE_GROUPS - 1);
            particles.muscle_group_id[i] = group;
            let local_y = particles.x[i].y - body_center.y;
            let flip = if group % 2 == 1 { -1.0 } else { 1.0 };
            particles.activation_dir[i] = if local_y >= 0.0 {
                Vec2::new(-FIBER_DIAG * flip, 1.0).normalize()
            } else {
                Vec2::new(FIBER_DIAG * flip, 1.0).normalize()
            };
        }
    }
    (solver, body_range, ratchet)
}

#[test]
fn lnn_drives_sustained_locomotion_with_zero_human_input() {
    let (mut sim, body_range, ratchet) = make_sim();
    // The one, one-time "decision" -- made before the loop starts, not per
    // frame. Everything after this point reads NO external input source.
    let mut lnn = make_cpg_biased(AUTONOMOUS_BIAS);
    ratchet.set_easy_direction(Vec2::X);
    ratchet.set_friction(0.1, 0.95);

    let spawn_centroid = {
        let particles = sim.particles();
        let n = particles.len() as f32;
        body_range.clone().map(|i| particles.x[i]).sum::<Vec2>() / n
    };

    const STEPS: usize = 6000;
    let mut min_j_ever = f32::INFINITY;
    for step in 0..STEPS {
        // The entire "control loop": step the controller, read its own
        // output, apply it. No keyboard, no branching on any external state.
        lnn.step(DT);
        let activations: Vec<f32> = lnn.activations().collect();
        {
            let particles = sim.particles_mut();
            for i in body_range.clone() {
                let group = particles.muscle_group_id[i] as usize;
                particles.activation[i] = (MUSCLE_AMPLITUDE * activations[group]).clamp(0.0, 1.0);
            }
        }
        sim.step();

        let snap = sim.diagnostics_snapshot();
        min_j_ever = min_j_ever.min(snap.min_deformation_j);
        assert_eq!(
            snap.non_finite_particle_values, 0,
            "non-finite particle values at step {step}"
        );
        assert_eq!(
            snap.out_of_bounds_particles, 0,
            "particles left the grid at step {step}"
        );
    }

    let particles = sim.particles();
    let n = particles.len() as f32;
    let final_centroid = body_range.clone().map(|i| particles.x[i]).sum::<Vec2>() / n;
    let drift = final_centroid - spawn_centroid;

    println!(
        "autonomous run: {STEPS} steps, zero human input, net drift={drift:?}, min_j_ever={min_j_ever:.3}"
    );
    assert!(
        drift.x.abs() > 5.0,
        "expected real sustained autonomous locomotion (>5 units drift), got {:.2}",
        drift.x
    );
    assert!(
        min_j_ever > 0.05,
        "body deformation stayed physically healthy (min_j_ever={min_j_ever:.3})"
    );
}
