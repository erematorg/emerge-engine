//! Probes on the load scene of `tests/rod_grid_coupling.rs` (a pinned
//! cantilever with a small particle block on its tip), kept for issue
//! #47: the loaded and unloaded tip over time, printed so the same scene
//! can be run on two versions, and the particles' position increments
//! against half the f32 spacing at their height. Probes, no criterion.
use emerge::rod::{RodMaterial, build_straight_rod};
use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};

fn cantilever_with_optional_particles(with_particles: bool, steps: usize) -> (Option<f32>, f32) {
    let config = SimConfig {
        grid_res: 48,
        dt: 0.02,
        gravity: Vec2::new(0.0, -0.3),
        max_substeps_per_step: 200,
        min_dt: 0.0005,
        ..SimConfig::default()
    };
    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    // span must stay small: self-weight tip deflection ~ L^4/EI, so a much longer span at
    // this stiffness folds nearly flat (not a coupling bug, just overload). 2.0
    // hand-checks to ~8% of span, a visible sag without collapse.
    let span = 2.0;
    let n_points = 12usize;
    let mut rod_points = build_straight_rod(
        Vec2::new(8.0, 24.0),
        Vec2::new(8.0 + span, 24.0),
        n_points,
        0.3,
        1.0,
    );
    rod_points.pinned[0] = 1;
    rod_points.pinned[1] = 1;
    let l0 = span / (n_points as f32 - 1.0);
    let point_mass = 0.3 * l0;
    let ea = 3.0e6 * 0.06 * 0.02;
    let ei = 3.0e6 * 0.06_f32.powi(3) * 0.02 / 12.0;
    let (axial_damping, bending_damping) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
    let material = RodMaterial::from_young_modulus_rectangular(
        3.0e6,
        0.06,
        0.02,
        axial_damping,
        bending_damping,
    );
    solver.add_rod(emerge::rod::Rod::new(rod_points, material));

    if with_particles {
        // The 0.05-cell sag margin below was set for a load of 1.0 per
        // particle. Particle mass is now derived from grid density (0.25 at
        // this spacing), which leaves only a 0.028-cell margin, so the load
        // is fixed explicitly rather than lowering the margin.
        let particle_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(1, 1),
            box_center: Vec2::new(9.0, 25.5),
            mass_override: Some(1.0),
            ..SpawnRegion::for_sim(solver.config())
        };
        let _ = solver.add_body(particle_spawn);
    }

    solver.step_n(steps);

    let rod_tip_y = solver.rods()[0].points.x.last().unwrap().y;
    let avg_particle_y = if with_particles {
        let particles = solver.particles();
        let n = particles.len();
        if n == 0 {
            None
        } else {
            Some(particles.iter().map(|p| p.x.y).sum::<f32>() / n as f32)
        }
    } else {
        None
    };
    (avg_particle_y, rod_tip_y)
}

#[test]
#[ignore = "scratch probe: the loaded tip and the particles over time"]
fn rod_load_over_time() {
    let mut last = f32::NAN;
    for steps in [2000usize, 4000, 8000, 12000, 16000, 24000] {
        let (avg, loaded) = cantilever_with_optional_particles(true, steps);
        println!(
            "steps {steps}: loaded tip {loaded:.4} (moved {:.4} since last), particles avg y {avg:?}",
            loaded - last
        );
        last = loaded;
    }
}

#[test]
#[ignore = "scratch A/B probe"]
fn rod_load_ab() {
    for steps in [4000usize, 8000] {
        let (_, alone) = cantilever_with_optional_particles(false, steps);
        let (avg, loaded) = cantilever_with_optional_particles(true, steps);
        println!(
            "steps {steps}: tip alone {alone:.4}, loaded {loaded:.4}, extra sag {:.4}, particles avg y {avg:?}",
            alone - loaded
        );
    }
}

fn loaded_solver() -> Simulation {
    let with_particles = true;
    let config = SimConfig {
        grid_res: 48,
        dt: 0.02,
        gravity: Vec2::new(0.0, -0.3),
        max_substeps_per_step: 200,
        min_dt: 0.0005,
        ..SimConfig::default()
    };
    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(NeoHookeanMaterial::new(20.0, 40.0)));

    // span must stay small: self-weight tip deflection ~ L^4/EI, so a much longer span at
    // this stiffness folds nearly flat (not a coupling bug, just overload). 2.0
    // hand-checks to ~8% of span, a visible sag without collapse.
    let span = 2.0;
    let n_points = 12usize;
    let mut rod_points = build_straight_rod(
        Vec2::new(8.0, 24.0),
        Vec2::new(8.0 + span, 24.0),
        n_points,
        0.3,
        1.0,
    );
    rod_points.pinned[0] = 1;
    rod_points.pinned[1] = 1;
    let l0 = span / (n_points as f32 - 1.0);
    let point_mass = 0.3 * l0;
    let ea = 3.0e6 * 0.06 * 0.02;
    let ei = 3.0e6 * 0.06_f32.powi(3) * 0.02 / 12.0;
    let (axial_damping, bending_damping) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
    let material = RodMaterial::from_young_modulus_rectangular(
        3.0e6,
        0.06,
        0.02,
        axial_damping,
        bending_damping,
    );
    solver.add_rod(emerge::rod::Rod::new(rod_points, material));

    if with_particles {
        // The 0.05-cell sag margin below was set for a load of 1.0 per
        // particle. Particle mass is now derived from grid density (0.25 at
        // this spacing), which leaves only a 0.028-cell margin, so the load
        // is fixed explicitly rather than lowering the margin.
        let particle_spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(1, 1),
            box_center: Vec2::new(9.0, 25.5),
            mass_override: Some(1.0),
            ..SpawnRegion::for_sim(solver.config())
        };
        let _ = solver.add_body(particle_spawn);
    }

    solver
}

#[test]
#[ignore = "scratch probe: are the particles' position increments absorbed in f32?"]
fn particle_increments_vs_ulp() {
    let mut solver = loaded_solver();
    solver.step_n(4000);
    let x0: Vec<Vec2> = solver.particles().iter().map(|p| p.x).collect();
    let steps = 100;
    solver.step_n(steps);
    // The frame's mean substep, not the last one (often a short remainder).
    let dt = solver.config().dt / solver.last_substeps().max(1) as f32;
    for (i, p) in solver.particles().iter().enumerate() {
        let a = p.x.y.abs();
        let half_ulp = 0.5 * (a.next_up() - a);
        println!(
            "particle {i}: y {:.6}, v ({:.3e}, {:.3e}) cells/s, mean substep {dt:.3e} s, |v_y| dt {:.3e}, half ulp of y {:.3e}, y moved {:.3e} over {steps} frames of {} substeps (v_y alone would move it {:.3e})",
            p.x.y,
            p.v.x,
            p.v.y,
            p.v.y.abs() * dt,
            half_ulp,
            p.x.y - x0[i].y,
            solver.last_substeps(),
            p.v.y * solver.config().dt * steps as f32
        );
    }
    let rod = &solver.rods()[0].points;
    println!("rod tip y {:.6}", rod.x[rod.x.len() - 1].y);
}
