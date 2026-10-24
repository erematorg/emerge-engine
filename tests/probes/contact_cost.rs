//! Probe: wall time and substeps of the passive-settle contact scene from
//! `tests/physics_correctness.rs` (a snake body resting on Drucker-Prager
//! sand through multi-field contact), over its first 1000 steps. Run on two
//! commits to compare the cost of the contact code.

use emerge::{DruckerPragerMaterial, NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

#[test]
#[ignore = "probe: run with --ignored --nocapture"]
fn contact_scene_cost() {
    const GRID: usize = 128;
    const DT: f32 = 0.1;
    const MUSCLE_GROUPS: usize = 8;
    const STEPS: usize = 1000;

    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let terrain_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(100, 12),
        box_center: Vec2::new(64.0, 10.0),
        material_id: 0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, terrain_spawn)
        .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)));
    let terrain_count = sim.particles().len();

    let mut snake_mat = NeoHookeanMaterial::new(13.0, 26.0);
    snake_mat.active_stress_coeff = 80.0;
    snake_mat.viscosity = 150.0;
    let snake_mat_id = sim.register_material(Box::new(snake_mat));
    let body_center = Vec2::new(64.0, 20.0);
    let body_len = 36.0 * 0.5;
    let snake_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(36, 4),
        box_center: body_center,
        material_id: snake_mat_id.0,
        ..SpawnRegion::for_sim(sim.config())
    };
    let _ = sim.add_body(snake_spawn);
    let snake_range = terrain_count..sim.particles().len();

    let body_left = body_center.x - body_len / 2.0;
    {
        let particles = sim.particles_mut();
        for i in snake_range.clone() {
            particles.contact_group[i] = 1;
            let t = ((particles.x[i].x - body_left) / body_len).clamp(0.0, 1.0);
            let group = ((t * MUSCLE_GROUPS as f32) as u32).min(MUSCLE_GROUPS as u32 - 1);
            particles.muscle_group_id[i] = group;
            let local_y = particles.x[i].y - body_center.y;
            let flip = if group % 2 == 1 { -1.0 } else { 1.0 };
            particles.activation_dir[i] = if local_y >= 0.0 {
                Vec2::new(-3.0 * flip, 1.0).normalize()
            } else {
                Vec2::new(3.0 * flip, 1.0).normalize()
            };
        }
    }
    let grip = std::sync::Arc::new(emerge::DirectionalContactGrip::new(0.5, 0.5, Vec2::X));
    let mut sim = sim.with_contact_grip(grip);

    let mut substeps = 0usize;
    let started = std::time::Instant::now();
    for _ in 0..STEPS {
        sim.step();
        substeps += sim.diagnostics_snapshot().substeps_last_step;
    }
    let seconds = started.elapsed().as_secs_f64();
    println!(
        "{STEPS} steps: {seconds:.1} s, {substeps} substeps ({:.2} per step), {:.3} ms per substep, {} particles",
        substeps as f64 / STEPS as f64,
        1000.0 * seconds / substeps as f64,
        sim.particles().len()
    );
}
