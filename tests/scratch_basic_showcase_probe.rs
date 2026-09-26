//! TEMPORARY, not part of the real suite -- direct headless reproduction of
//! `examples/cpu/basic_showcase.rs`'s scene (no GUI, no arrow-key/mouse
//! interaction) to verify the sand terrain's real SI migration
//! (E=15 MPa/nu=0.3/rho=1600, same citation as basic_sand.rs) is stable
//! alongside the unmigrated elastic body and the already-real fluid, under
//! this scene's own deliberately-weak gravity.
extern crate emerge_engine as emerge;

use emerge::{
    DruckerPragerMaterial, MaterialModel, NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig,
    Simulation, SlipBoundary, SpawnRegion,
};
use glam::{IVec2, Vec2};

const GRID: usize = 64;
const DT: f32 = 0.1;
const ELASTIC_ID: u32 = 0;
const SAND_ID: u32 = 1;
const FLUID_ID: u32 = 2;
const SPACING: f32 = 0.7;

const SAND_YOUNG_MODULUS_PA: f32 = 15.0e6;
const SAND_POISSON_RATIO: f32 = 0.3;
const SAND_DENSITY_KG_M3: f32 = 1600.0;

fn make_sim(max_substeps_per_step: usize) -> Simulation {
    let config = SimConfig {
        min_dt: 0.005,
        max_substeps_per_step,
        recompute_density_each_step: true,
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let elastic = NeoHookeanMaterial::new(40.0, 80.0);
    let (sand_lambda, sand_mu) = config.lame_from_si_physical_cfg(
        SAND_YOUNG_MODULUS_PA,
        SAND_POISSON_RATIO,
        SAND_DENSITY_KG_M3,
    );
    let sand = DruckerPragerMaterial::new(sand_lambda, sand_mu);
    let fluid = NewtonianFluidMaterial::low_viscosity(0.1, 0.25);
    let sand_mass = (SAND_DENSITY_KG_M3 / config.reference_density_kg_m3) * SPACING * SPACING;

    let mut solver = Simulation::empty(config)
        .with_default_material(Box::new(elastic))
        .with_material(SAND_ID, Box::new(sand))
        .with_material(FLUID_ID, Box::new(fluid))
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(19.0, 9.0),
        material_id: SAND_ID,
        mass_override: Some(sand_mass),
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(22, 14),
        box_center: Vec2::new(45.0, 9.0),
        material_id: FLUID_ID,
        mass_override: Some(0.1 * SPACING * SPACING),
        ..SpawnRegion::for_sim(&config)
    });
    let _ = solver.add_body(SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(12, 12),
        box_center: Vec2::new(32.0, 46.0),
        material_id: ELASTIC_ID,
        ..SpawnRegion::for_sim(&config)
    });
    solver
}

fn run_probe(label: &str, max_substeps_per_step: usize, steps: u64) {
    let mut sim = make_sim(max_substeps_per_step);
    for step in 1..=steps {
        sim.step();
        if step.is_multiple_of(steps / 20) || step == 1 || step == steps {
            let snap = sim.diagnostics_snapshot();
            println!(
                "[{label}] step={step} t={:.2} sub={} J=[{:.4},{:.4}] cfl={:.4} \
                 non_finite={} mass_err={:.2e} time_dropped={:.4}",
                step as f32 * DT,
                snap.substeps_last_step,
                snap.min_deformation_j,
                snap.max_deformation_j,
                snap.cfl_number,
                snap.non_finite_particle_values,
                snap.relative_mass_error,
                snap.sim_time_dropped,
            );
            assert_eq!(
                snap.non_finite_particle_values, 0,
                "[{label}] NaN/Inf at step {step}"
            );
        }
    }
    let snap = sim.diagnostics_snapshot();
    assert!(
        snap.sim_time_dropped < 1.0e-6,
        "[{label}] sim_time_dropped={} -- max_substeps_per_step too low",
        snap.sim_time_dropped
    );
}

#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn basic_showcase_real_sand_stiffness_settles() {
    run_probe("substeps=3000", 3000, 300);
}

/// How much would a genuinely coarser grid (same real domain size, same real
/// sand pile size, fewer/bigger cells) actually buy us? `elastic_wave_dt`'s
/// dt bound scales linearly with `dx_meters` once lambda/mu come from
/// `lame_from_si_physical_cfg` (which divides by `rho*dx_meters^2`, so the
/// resulting wave speed scales as `1/dx_meters`, and `dt = cfl/c` scales as
/// `dx_meters`) -- this sweep confirms that scaling holds for the REAL
/// function, not a hand-derived guess, and reports the real substep count
/// at each candidate resolution. Doubling/tripling `dx_meters` alone (never
/// done in the real demo) would silently redefine what real-world size this
/// scene represents -- a genuine, honest resolution tradeoff also needs
/// `GRID` and every cell-unit `box_size`/`box_center` shrunk by the same
/// factor, so the real domain and the real pile size stay unchanged and the
/// demo visibly gets chunkier (fewer, bigger particles) as the honest price.
#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn basic_showcase_sand_substep_cost_vs_dx_meters_sweep() {
    let config = SimConfig::earth(GRID, 0.01, DT);
    for factor in [1.0f32, 1.5, 2.0, 3.0, 5.0, 8.0] {
        let dx = 0.01 * factor;
        let (lambda, mu) = lame_at_dx(
            &config,
            dx,
            SAND_YOUNG_MODULUS_PA,
            SAND_POISSON_RATIO,
            SAND_DENSITY_KG_M3,
        );
        let sand = DruckerPragerMaterial::new(lambda, mu);
        let density = SAND_DENSITY_KG_M3 / config.reference_density_kg_m3;
        let dt = sand.timestep_bound(
            density,
            1.0,
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        let substeps_needed = (config.dt / dt).ceil();
        println!(
            "dx_meters={dx:.4} (factor={factor:.1}x)  dt_bound={dt:.6e}  substeps_for_frame_dt={substeps_needed:.0}"
        );
    }
}

/// Real, published, PROVEN alternative to guessing a softer stiffness:
/// Haeri & Skonieczny 2022 (the SAME paper already cited for
/// SAND_YOUNG_MODULUS_PA=15MPa, arXiv:2111.01523) publish their own
/// "relaxed Young's modulus" variant at E=0.15 MPa (100x softer), used
/// explicitly "for significant computational efficiency," their Table 2's
/// own footnote calling it "yet acceptable accuracy." They report a real,
/// measured cost for it: 15.8% mean error on excavation forward force,
/// versus -0.5% for the real, validated 15 MPa case. This is not a guess --
/// it is the literal number the authors of our own citation already
/// published and used for exactly this performance reason.
#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn basic_showcase_sand_substep_cost_at_published_relaxed_modulus() {
    let config = SimConfig::earth(GRID, 0.01, DT);
    for (label, e_pa) in [
        (
            "validated (paper's own -0.5% error case)",
            SAND_YOUNG_MODULUS_PA,
        ),
        ("published relaxed (paper's own 15.8% error case)", 0.15e6),
    ] {
        let (lambda, mu) = lame_at_dx(&config, 0.01, e_pa, SAND_POISSON_RATIO, SAND_DENSITY_KG_M3);
        let sand = DruckerPragerMaterial::new(lambda, mu);
        let density = SAND_DENSITY_KG_M3 / config.reference_density_kg_m3;
        let dt = sand.timestep_bound(
            density,
            1.0,
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        let substeps_needed = (config.dt / dt).ceil();
        println!(
            "{label:50} E={e_pa:.3e}Pa  dt_bound={dt:.6e}  substeps_for_frame_dt={substeps_needed:.0}"
        );
    }
}

fn lame_at_dx(
    config: &SimConfig,
    dx_meters: f32,
    e_pa: f32,
    nu: f32,
    rho_kg_m3: f32,
) -> (f32, f32) {
    let mut cfg = *config;
    cfg.dx_meters = dx_meters;
    cfg.lame_from_si_physical_cfg(e_pa, nu, rho_kg_m3)
}

/// Root-cause which of the 3 materials in this scene actually drives the
/// ~3000-substep/frame cost measured live in `basic_showcase_gpu.rs`
/// (~0.5fps, worse than the isolated fluid stress test). Calls each
/// material's own real `timestep_bound` directly (same function the real
/// CFL scan uses every substep, `src/spacetime/solver/cfl.rs`) with real
/// spawned particle data and this scene's own real config, instead of
/// hand-deriving the SI-to-grid unit conversion by eye.
#[test]
#[ignore = "temporary manual probe, not a regression test"]
fn basic_showcase_substep_cost_breakdown_by_material() {
    let sim = make_sim(3000);
    let config = *sim.config();
    let particles = sim.particles();

    // Real spawn-time density per material (the spawn measures V0
    // means this is each material's own true rest density, before any
    // compression) -- read from the actual spawned scene, not assumed.
    let spawn_state_of = |mat_id: u32| {
        (0..particles.material_id.len())
            .find(|&i| particles.material_id[i] == mat_id)
            .map(|i| (particles.density[i], particles.hardening_scale[i]))
            .unwrap_or_else(|| panic!("no spawned particle for material {mat_id}"))
    };

    // Same real construction as `make_sim` -- `MaterialModel::timestep_bound`
    // is a pure function of the material's own parameters plus these scalars,
    // so calling it directly on freshly-built values (not through the
    // registry, which only exposes this as `pub(crate)`) is exact, not an
    // approximation.
    let (sand_lambda, sand_mu) = config.lame_from_si_physical_cfg(
        SAND_YOUNG_MODULUS_PA,
        SAND_POISSON_RATIO,
        SAND_DENSITY_KG_M3,
    );
    let cases: [(&str, u32, &dyn MaterialModel); 3] = [
        (
            "elastic (NeoHookean 40/80)",
            ELASTIC_ID,
            &NeoHookeanMaterial::new(40.0, 80.0),
        ),
        (
            "sand (Drucker-Prager, E=15MPa)",
            SAND_ID,
            &DruckerPragerMaterial::new(sand_lambda, sand_mu),
        ),
        (
            "fluid (Newtonian, low_viscosity)",
            FLUID_ID,
            &NewtonianFluidMaterial::low_viscosity(0.1, 0.25),
        ),
    ];
    for (label, mat_id, material) in cases {
        let (density, hardening_scale) = spawn_state_of(mat_id);
        let dt = material.timestep_bound(
            density,
            hardening_scale,
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        let substeps_needed = (config.dt / dt).ceil();
        println!(
            "{label:35} density={density:8.4}  dt_bound={dt:.6e}  substeps_for_frame_dt={substeps_needed:.0}"
        );
    }
}
