//! Diagnostic-only (2026-09-11): the passing settled-pile correctness test
//! uses `lame_from_young(6e5, 0.3)` (grid-native units) with 64 particles;
//! the failing real `basic_sand`-scale measurement uses
//! `lame_from_si_physical_cfg(15e6, 0.3, 1600.0)` (real SI, converted) with
//! 1008 particles -- two variables changed at once (stiffness AND scene
//! size), in two different unit conventions. Isolates which actually
//! breaks Newton convergence by using the SAME two concrete (lambda, mu)
//! pairs each real test uses, varied independently against scene size.
//!
//! `cargo test --release --test scratch_implicit_stiffness_vs_scale_isolation -- --nocapture`

extern crate emerge_engine as emerge;
use emerge::materials::{DruckerPragerMaterial, lame_from_young};
use emerge::{SimConfig, Simulation, SlipBoundary, SpawnRegion};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const MAT_SAND: u32 = 1;

fn make_sim(lambda: f32, mu: f32, box_size: IVec2) -> Simulation {
    let config = SimConfig {
        boundary_thickness: 3,
        max_substeps_per_step: 3000,
        material_cfl_coefficient: 0.7,
        implicit_corotated_elastic: true,
        ..SimConfig::earth(GRID, 0.01, 0.016)
    };
    let mut m = DruckerPragerMaterial::new(lambda, mu);
    m.friction_angle = 30.0f32.to_radians();
    let mass_grid = (1600.0 / config.reference_density_kg_m3) * 0.5 * 0.5;
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size,
        box_center: Vec2::new(32.0, 20.0),
        material_id: MAT_SAND,
        position_jitter: 0.5,
        rng_seed: 11,
        mass_override: Some(mass_grid),
        ..SpawnRegion::for_sim(&config)
    };
    Simulation::new(config, spawn)
        .with_default_material(Box::new(m))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
}

fn probe(label: &str, lambda: f32, mu: f32, box_size: IVec2) {
    let mut sim = make_sim(lambda, mu, box_size);
    for _ in 0..60 {
        sim.step();
    }
    let n = sim.particles().len();
    println!("--- {label}: lambda={lambda:.3e} mu={mu:.3e} n={n} ---");
    unsafe {
        std::env::set_var("EMERGE_IMPLICIT_DIAG", "1");
    }
    for _ in 0..3 {
        sim.step();
    }
    unsafe {
        std::env::remove_var("EMERGE_IMPLICIT_DIAG");
    }
}

#[test]
#[ignore = "diagnostic probe kept for reruns, not part of the CI suite"]
fn isolate_stiffness_vs_particle_count() {
    let (soft_lambda, soft_mu) = lame_from_young(6.0e5, 0.3);
    let config = SimConfig::earth(GRID, 0.01, 0.016);
    let (real_lambda, real_mu) = config.lame_from_si_physical_cfg(15.0e6, 0.3, 1600.0);
    println!(
        "soft (correctness test): lambda={soft_lambda:.3e} mu={soft_mu:.3e}; \
         real (fps test): lambda={real_lambda:.3e} mu={real_mu:.3e}"
    );

    // Same softer stiffness the PASSING correctness test uses, but the
    // FAILING test's large real particle count.
    probe(
        "soft_stiffness_large_scene",
        soft_lambda,
        soft_mu,
        IVec2::new(18, 14),
    );
    // Same real stiffness the FAILING test uses, but the PASSING
    // correctness test's small scene.
    probe(
        "real_stiffness_small_scene",
        real_lambda,
        real_mu,
        IVec2::new(8, 8),
    );
    // Real stiffness AND the large real scene (reproduces the failure).
    probe(
        "real_stiffness_large_scene",
        real_lambda,
        real_mu,
        IVec2::new(18, 14),
    );
}
