//! Micro-benchmarks for the emerge hot path.
//!
//! Groups:
//!   step_scaling          -- full solver.step() at varying particle counts (regression guard)
//!   mixed_materials       -- step() with sand + fluid + jelly simultaneously (LP workload)
//!   material_count_scaling -- step() at fixed particle count, varying distinct material count
//!   force_field_scaling   -- step() at fixed particle count, varying active force field count
//!   grid_resolution_scaling -- step() at fixed particle count, varying grid resolution
//!   sand_sheared          -- step() on pre-deformed sand (50 warm-up steps before measuring)
//!   p2g                   -- scatter_particles_to_grid in isolation
//!   g2p                   -- gather_grid_to_particles in isolation
//!   kirchhoff             -- kirchhoff_stress per material (NeoHookean / Sand / Fluid / Snow)
//!   update_particle       -- plasticity update per material
//!   grid_update           -- grid.update_velocities in isolation
//!   grains_step_scaling   -- full solver.step() with a GrainPopulation, varying grain count
//!   grains_contact_resolution -- GrainPopulation::resolve_contact_forces in isolation (the
//!                             O(n^2) all-pairs check, suspected bottleneck behind the demo's
//!                             observed low fps -- compare against grains_step_scaling to see
//!                             what fraction of total step cost it actually accounts for)
//!
//! GPU groups (feature = "gpu" only):
//!   gpu_sleep_wake_scaling -- step_frame() sleep on/off at varying particle counts
//!   gpu_step_scaling       -- full GpuSimulation::step_frame() at varying particle counts,
//!                             the direct GPU counterpart to step_scaling above (was a real
//!                             coverage gap -- no GPU-path benchmark existed before)
//!   gpu_sparse_grid_scaling -- fixed SMALL particle cluster, varying grid resolution -- the
//!                             regression guard for GPU sparse grid Phase 2 (grid_update.wgsl's
//!                             active-block dispatch): cost should stay roughly flat across
//!                             grid_res once compaction is working, not scale with grid_res²
//!
//!   cargo bench --bench scaling
//!   cargo bench --bench scaling -- mixed_materials   (single group)
//!   cargo bench --bench scaling --features gpu -- gpu_sparse_grid_scaling
//!
//! Reports: target/criterion/<group>/report/index.html

extern crate emerge_engine as emerge;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use emerge::grains::population::GrainPopulation;
use emerge::materials::granular::grain_contact_law::{ContactLawConfig, critical_timestep};
use emerge::particle::Grain;
use emerge::{
    BoundaryCondition, DruckerPragerMaterial, FrictionBoundary, Grid, MAX_MATERIAL_SLOTS,
    MaterialModel, MaterialRegistry, NeoHookeanMaterial, NewtonianFluidMaterial, Particles,
    RadialConfinementField, SimConfig, Simulation, SlipBoundary, SpawnRegion, StomakhinMaterial,
    ViscoelasticMaterial, build_particles, lame_from_young,
};
use glam::{IVec2, Vec2};

// â”€â”€ helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

const GRID: usize = 64;

fn base_config() -> SimConfig {
    SimConfig::standard(GRID, 0.1, Vec2::new(0.0, -0.3))
}

fn box_body(config: &SimConfig, side: i32) -> SpawnRegion {
    SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::splat(side),
        box_center: Vec2::splat(GRID as f32 * 0.5),
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(config)
    }
}

fn build_settled_sim(target: usize) -> Simulation {
    let config = base_config();
    let side = ((target as f32).sqrt() * 0.5).ceil() as i32;
    let (l, u) = lame_from_young(5.0e4, 0.3);
    let mut sim = Simulation::new(config, box_body(&config, side))
        .with_default_material(Box::new(NeoHookeanMaterial::new(l, u)))
        .with_boundary(Box::new(SlipBoundary::new(2)));
    sim.step_n(20);
    sim
}

/// Pre-built particles + grid + registry for transfer-level benches.
struct TransferFixture {
    particles: Particles,
    grid: Grid,
    registry: MaterialRegistry,
    config: SimConfig,
    n: usize,
}

impl TransferFixture {
    fn new(target: usize) -> Self {
        let config = base_config();
        let side = ((target as f32).sqrt() * 0.5).ceil() as i32;
        let raw = build_particles(&config, box_body(&config, side));
        let n = raw.len();
        let particles = Particles::from(raw);
        let grid = Grid::new(GRID);
        let (l, u) = lame_from_young(5.0e4, 0.3);
        let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(l, u)));
        Self {
            particles,
            grid,
            registry,
            config,
            n,
        }
    }

    fn fill_grid(&mut self) {
        self.grid.clear();
        emerge::transfer::scatter_particles_to_grid(
            &self.particles,
            &mut self.grid,
            &self.registry,
            self.config.dt,
            self.n,
        );
        self.grid
            .update_velocities(self.config.dt, self.config.gravity);
    }
}

// â”€â”€ step_scaling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn step_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("step_scaling");
    for &target in &[100usize, 500, 1000, 2500, 5000] {
        let mut sim = build_settled_sim(target);
        let n = sim.particles().len();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| sim.step());
        });
    }
    group.finish();
}

// â”€â”€ p2g â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn bench_p2g(c: &mut Criterion) {
    let mut group = c.benchmark_group("p2g");
    for &target in &[500usize, 2500, 5000] {
        let mut fx = TransferFixture::new(target);
        let n = fx.n;
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                fx.grid.clear();
                emerge::transfer::scatter_particles_to_grid(
                    &fx.particles,
                    &mut fx.grid,
                    &fx.registry,
                    fx.config.dt,
                    fx.n,
                );
            });
        });
    }
    group.finish();
}

// â”€â”€ g2p â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn bench_g2p(c: &mut Criterion) {
    let mut group = c.benchmark_group("g2p");
    for &target in &[500usize, 2500, 5000] {
        let mut fx = TransferFixture::new(target);
        fx.fill_grid();
        let n = fx.n;
        let dt = fx.config.dt;
        let boundaries: Vec<Box<dyn BoundaryCondition>> =
            vec![Box::new(SlipBoundary::new(fx.config.boundary_thickness))];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                emerge::transfer::gather_grid_to_particles(
                    &mut fx.particles,
                    &fx.grid,
                    dt,
                    fx.config.gravity,
                    &boundaries,
                    &fx.registry,
                    emerge::transfer::G2PParams {
                        apic_blend: 1.0,
                        active_count: fx.n,
                        pre_force_snapshot: None,
                        asflip_blend: 0.0,
                        boundary_thickness: fx.config.boundary_thickness,
                        nonlocal_fluidity: &[],
                        cosserat_curvature: &[],
                    },
                );
            });
        });
    }
    group.finish();
}

// â”€â”€ kirchhoff per-material â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn bench_kirchhoff(c: &mut Criterion) {
    let config = base_config();
    let raw = build_particles(&config, box_body(&config, 20));
    let particles = Particles::from(raw);
    let n = particles.len();

    let (l, u) = lame_from_young(5.0e4, 0.3);
    let neo = NeoHookeanMaterial::new(l, u);
    let sand = DruckerPragerMaterial::cohesionless(133.3, 0.333);
    let fluid = NewtonianFluidMaterial::low_viscosity(1000.0, 128_000.0);
    let snow = StomakhinMaterial::new(1389.0, 2083.0, 10.0, 0.02, 0.005, 0.6, 20.0);

    let mut group = c.benchmark_group("kirchhoff");
    group.bench_function("NeoHookean", |b| {
        b.iter(|| {
            for i in 0..n {
                criterion::black_box(neo.kirchhoff_stress(&particles, i));
            }
        })
    });
    group.bench_function("Sand", |b| {
        b.iter(|| {
            for i in 0..n {
                criterion::black_box(sand.kirchhoff_stress(&particles, i));
            }
        })
    });
    group.bench_function("Fluid", |b| {
        b.iter(|| {
            for i in 0..n {
                criterion::black_box(fluid.kirchhoff_stress(&particles, i));
            }
        })
    });
    group.bench_function("Snow", |b| {
        b.iter(|| {
            for i in 0..n {
                criterion::black_box(snow.kirchhoff_stress(&particles, i));
            }
        })
    });
    group.finish();
}

// â”€â”€ update_particle per-material â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn bench_update_particle(c: &mut Criterion) {
    let config = base_config();
    let dt = config.dt;

    let (l, u) = lame_from_young(5.0e4, 0.3);
    let neo = NeoHookeanMaterial::new(l, u);
    let sand = DruckerPragerMaterial::cohesionless(133.3, 0.333);
    let fluid = NewtonianFluidMaterial::low_viscosity(1000.0, 128_000.0);
    let snow = StomakhinMaterial::new(1389.0, 2083.0, 10.0, 0.02, 0.005, 0.6, 20.0);

    let mut group = c.benchmark_group("update_particle");

    macro_rules! bench_mat {
        ($name:expr, $mat:expr) => {{
            let raw = build_particles(&config, box_body(&config, 20));
            let n = raw.len();
            let mut ps = Particles::from(raw);
            group.bench_function($name, |b| {
                b.iter(|| {
                    for i in 0..n {
                        $mat.update_particle(&mut ps.update_ctx(i), dt);
                    }
                })
            });
        }};
    }
    bench_mat!("NeoHookean", neo);
    bench_mat!("Sand", sand);
    bench_mat!("Fluid", fluid);
    bench_mat!("Snow", snow);
    group.finish();
}

// â”€â”€ grid_update â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

fn bench_grid_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_update");
    for &target in &[500usize, 2500, 5000] {
        let mut fx = TransferFixture::new(target);
        fx.grid.clear();
        emerge::transfer::scatter_particles_to_grid(
            &fx.particles,
            &mut fx.grid,
            &fx.registry,
            fx.config.dt,
            fx.n,
        );
        let n = fx.n;
        let dt = fx.config.dt;
        let gravity = fx.config.gravity;
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| fx.grid.update_velocities(dt, gravity));
        });
    }
    group.finish();
}

// â”€â”€ mixed_materials â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
//
// LP workload: sand terrain + Newtonian water + viscoelastic creature bodies active in the same
// substep. Material dispatch branches across all three plasticity paths simultaneously.

const SAND_ID: u32 = 1;
const WATER_ID: u32 = 2;

fn build_mixed_sim(n_each: usize) -> Simulation {
    let config = base_config();
    let side = ((n_each as f32).sqrt() * 0.5).ceil() as i32;

    let jelly_spawn = SpawnRegion {
        box_size: IVec2::splat(side),
        box_center: Vec2::new(GRID as f32 * 0.3, GRID as f32 * 0.6),
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let sand_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::splat(side),
        box_center: Vec2::new(GRID as f32 * 0.5, GRID as f32 * 0.25),
        material_id: SAND_ID,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let water_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::splat(side),
        box_center: Vec2::new(GRID as f32 * 0.7, GRID as f32 * 0.6),
        material_id: WATER_ID,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };

    let mut sim = Simulation::empty(config)
        .with_default_material(Box::new(ViscoelasticMaterial::near_incompressible(
            5.0e4, 10.0,
        )))
        .with_material(
            SAND_ID,
            Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)),
        )
        .with_material(
            WATER_ID,
            Box::new(NewtonianFluidMaterial::low_viscosity(1000.0, 1.28e5)),
        )
        .with_boundary(Box::new(SlipBoundary::new(2)));

    let _ = sim.add_body(jelly_spawn);
    let _ = sim.add_body(sand_spawn);
    let _ = sim.add_body(water_spawn);
    sim.step_n(10);
    sim
}

fn bench_mixed_materials(c: &mut Criterion) {
    let mut group = c.benchmark_group("mixed_materials");
    for &n_each in &[100usize, 333, 666] {
        let mut sim = build_mixed_sim(n_each);
        let total = sim.particles().len();
        group.bench_with_input(BenchmarkId::from_parameter(total), &total, |b, _| {
            b.iter(|| sim.step());
        });
    }
    group.finish();
}

// ── material_count_scaling ───────────────────────────────────────────────────
//
// Fixed-size regions, varying number of distinct active materials (1..MAX_MATERIAL_SLOTS).
// Stresses per-particle material dispatch (registry lookup + kirchhoff_stress vtable call) as
// material diversity grows -- the axis LP pushes as it adds more constitutive models to one scene.

fn build_material_count_sim(k: usize) -> Simulation {
    let config = base_config();
    let side = 2i32; // 4x4 particles per region at spacing 0.5
    let grid_dim = (k as f32).sqrt().ceil() as usize;
    let spacing_cells = GRID as f32 / (grid_dim as f32 + 1.0);

    let spawns: Vec<SpawnRegion> = (0..k)
        .map(|i| {
            let col = i % grid_dim;
            let row = i / grid_dim;
            let center = Vec2::new(
                (col as f32 + 1.0) * spacing_cells,
                (row as f32 + 1.0) * spacing_cells,
            );
            SpawnRegion {
                spacing: 0.5,
                box_size: IVec2::splat(side),
                box_center: center,
                material_id: i as u32,
                precompute_initial_volumes: true,
                ..SpawnRegion::for_sim(&config)
            }
        })
        .collect();

    let (l, u) = lame_from_young(5.0e4, 0.3);
    let mut sim = Simulation::empty(config).with_boundary(Box::new(SlipBoundary::new(2)));
    for i in 0..k {
        sim = sim.with_material(i as u32, Box::new(NeoHookeanMaterial::new(l, u)));
    }
    for spawn in spawns {
        let _ = sim.add_body(spawn);
    }
    sim.step_n(5);
    sim
}

fn bench_material_count_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("material_count_scaling");
    for &k in &[1usize, 4, 8, 16, 32, MAX_MATERIAL_SLOTS] {
        let mut sim = build_material_count_sim(k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| sim.step());
        });
    }
    group.finish();
}

// ── force_field_scaling ──────────────────────────────────────────────────────
//
// Fixed particle count, varying number of active force fields (1..16, mirrors GPU
// MAX_FORCE_FIELDS). Each field is evaluated per-particle per-substep -- stresses the
// linear scan over `force_fields` in the post-step pass as field count grows.

fn build_force_field_sim(n: usize, k: usize) -> Simulation {
    let config = base_config();
    let side = ((n as f32).sqrt() * 0.5).ceil() as i32;
    let (l, u) = lame_from_young(5.0e4, 0.3);
    let mut sim = Simulation::new(config, box_body(&config, side))
        .with_default_material(Box::new(NeoHookeanMaterial::new(l, u)))
        .with_boundary(Box::new(SlipBoundary::new(2)));
    for i in 0..k {
        // Centers far outside the particle cluster -- fields contribute negligible force,
        // isolating dispatch overhead from confinement-induced dynamics.
        let center = Vec2::new(GRID as f32 * 2.0 + i as f32, GRID as f32 * 2.0);
        sim.add_force_field(Box::new(RadialConfinementField::new(center, 5.0, 100.0)));
    }
    sim
}

fn bench_force_field_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("force_field_scaling");
    for &k in &[1usize, 4, 8, 16] {
        let mut sim = build_force_field_sim(2000, k);
        group.bench_with_input(BenchmarkId::from_parameter(k), &k, |b, _| {
            b.iter(|| sim.step());
        });
    }
    group.finish();
}

// ── grid_resolution_scaling ───────────────────────────────────────────────────
//
// Fixed particle count, varying grid resolution (32/64/128/256). The CPU grid is sparse
// (HashMap keyed by touched cell index, src/grid/mod.rs) -- this should stay flat as
// grid_res grows, confirming cost tracks particle count, not domain size. The GPU grid is
// dense (grid_res² buffer, src/gpu/buffers.rs) and does NOT have this property -- this bench
// is the CPU-side baseline that motivates a sparse GPU grid for LP's planetary scale (roadmap).

fn build_grid_res_sim(grid_res: usize) -> Simulation {
    let config = SimConfig::standard(grid_res, 0.1, Vec2::new(0.0, -0.3));
    let side = 16i32; // fixed particle cluster size regardless of grid_res
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::splat(side),
        box_center: Vec2::splat(grid_res as f32 * 0.5),
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let (l, u) = lame_from_young(5.0e4, 0.3);
    let mut sim = Simulation::new(config, spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(l, u)))
        .with_boundary(Box::new(SlipBoundary::new(2)));
    sim.step_n(5);
    sim
}

fn bench_grid_resolution_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("grid_resolution_scaling");
    for &grid_res in &[32usize, 64, 128, 256] {
        let mut sim = build_grid_res_sim(grid_res);
        group.bench_with_input(BenchmarkId::from_parameter(grid_res), &grid_res, |b, _| {
            b.iter(|| sim.step());
        });
    }
    group.finish();
}

// ── sand_sheared ───────────────────────────────────────────────────────────
//
// step() on sand that has already undergone plastic deformation. Undeformed sand skips the
// yield-surface projection; this captures the cost of active return-mapping in real sims.

fn build_sheared_sand(n: usize) -> Simulation {
    let config = base_config();
    let side = ((n as f32).sqrt() * 0.5).ceil() as i32;
    let mut sim = Simulation::new(config, box_body(&config, side))
        .with_default_material(Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)))
        .with_boundary(Box::new(SlipBoundary::new(2)));
    sim.step_n(50);
    sim
}

fn bench_sand_sheared(c: &mut Criterion) {
    let mut group = c.benchmark_group("sand_sheared");
    for &target in &[500usize, 2500, 5000] {
        let mut sim = build_sheared_sand(target);
        let n = sim.particles().len();
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| sim.step());
        });
    }
    group.finish();
}

// ── gpu_sleep_wake_scaling ──────────────────────────────────────────────────
//
// Phase 1 GPU sleep/wake (flag-based, no compaction). Compares step_frame() cost for a
// large settled sand pile with sleep_threshold=0.0 (off, baseline -- every particle does
// full P2G/G2P/plasticity/force-field work every substep) vs a real threshold (most of
// the pile asleep, skipping that work). This is the actual measurement the plan's
// verification step requires before claiming a win -- same "measure before claiming a
// win, revert if it doesn't deliver" rule used for the CPU P2G/G2P parallelization
// attempts earlier in this project (two reverted, one kept).
#[cfg(feature = "gpu")]
mod gpu_benches {
    use super::*;
    use emerge::gpu::GpuSimulation;
    use pollster::block_on;

    fn build_gpu_pile(target: usize, sleep_threshold: f32) -> GpuSimulation {
        // base_config()'s fixed GRID=64 (tuned for the smaller CPU bench targets) is too
        // small here -- at target=20000, side ≈ 71 cells, bigger than the whole grid,
        // spawning a boundary-saturated mess from the start (chaotic for reasons unrelated
        // to sleep/wake). Scale grid_res to the target instead of reusing the CPU constant.
        let side = ((target as f32).sqrt() * 0.5).ceil() as i32;
        let grid_res = (side as usize + 32).max(64);
        let config = SimConfig {
            sleep_threshold,
            ..SimConfig::standard(grid_res, 0.1, Vec2::new(0.0, -0.3))
        };
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::splat(side),
            box_center: Vec2::splat(grid_res as f32 * 0.5),
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let particles = build_particles(&config, spawn);
        let registry =
            MaterialRegistry::with_default(Box::new(DruckerPragerMaterial::new(400.0, 200.0)));
        let mut sim = block_on(GpuSimulation::new(config, particles, registry));
        // Settle under gravity before measuring, until genuinely calm -- not a fixed step
        // count. A boxy DP sand spawn has to collapse and find its angle of repose, which
        // takes longer at larger scale; a fixed 300-step budget (the original guess) left
        // larger piles still actively flowing when criterion started measuring, so the
        // sleep_threshold=0.0 vs >0.0 comparison wasn't measuring the same physical state --
        // it was comparing "still avalanching" against itself, not "settled" against itself.
        // Use max speed directly (not the sleeping flag, which doesn't exist when threshold
        // is 0.0) so both variants settle by the same real criterion.
        let mut prev_max_v = f32::INFINITY;
        let mut stable_checks = 0;
        for _ in 0..400 {
            for _ in 0..20 {
                sim.step_frame();
            }
            sim.sync_particles_blocking();
            let max_v = sim
                .particles()
                .iter()
                .map(|p| p.v.length())
                .fold(0.0f32, f32::max);
            if max_v < 0.05 && prev_max_v < 0.05 {
                stable_checks += 1;
                if stable_checks >= 3 {
                    break;
                }
            } else {
                stable_checks = 0;
            }
            prev_max_v = max_v;
        }
        sim
    }

    pub fn bench_gpu_sleep_wake_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("gpu_sleep_wake_scaling");
        for &target in &[2_000usize, 8_000, 20_000] {
            let mut sim_off = build_gpu_pile(target, 0.0);
            group.bench_with_input(BenchmarkId::new("sleep_off", target), &target, |b, _| {
                b.iter(|| sim_off.step_frame())
            });
            // 0.05: same threshold validated in tests/gpu.rs's gpu_sleep_freezes_settled_particles
            // -- comfortably below genuine free-fall/jostle speed, catches real post-impact rest.
            let mut sim_on = build_gpu_pile(target, 0.05);
            group.bench_with_input(BenchmarkId::new("sleep_on", target), &target, |b, _| {
                b.iter(|| sim_on.step_frame())
            });
        }
        group.finish();
    }

    // ── gpu_step_scaling ─────────────────────────────────────────────────────
    //
    // GPU counterpart to the CPU `step_scaling` group -- was a real coverage gap (this
    // benchmark file had zero GPU-path benchmarks before), meaning no GPU perf work had a
    // regression guard or before/after evidence. Mirrors step_scaling's own particle-count
    // sweep exactly, just on GpuSimulation.

    fn build_gpu_settled_sim(target: usize) -> GpuSimulation {
        let side = ((target as f32).sqrt() * 0.5).ceil() as i32;
        let grid_res = (side as usize + 32).max(64);
        let config = SimConfig::standard(grid_res, 0.1, Vec2::new(0.0, -0.3));
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::splat(side),
            box_center: Vec2::splat(grid_res as f32 * 0.5),
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let particles = build_particles(&config, spawn);
        let (l, u) = lame_from_young(5.0e4, 0.3);
        let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(l, u)));
        let mut sim = block_on(GpuSimulation::new(config, particles, registry));
        for _ in 0..20 {
            sim.step_frame();
        }
        sim
    }

    pub fn bench_gpu_step_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("gpu_step_scaling");
        for &target in &[100usize, 500, 1000, 2500, 5000] {
            let mut sim = build_gpu_settled_sim(target);
            group.bench_with_input(BenchmarkId::from_parameter(target), &target, |b, _| {
                b.iter(|| sim.step_frame());
            });
        }
        group.finish();
    }

    // ── gpu_sparse_grid_scaling ──────────────────────────────────────────────
    //
    // The direct regression guard for GPU sparse grid Phase 2 (grid_update.wgsl's
    // active-block dispatch, src/systems/gpu/shaders/grid_update.wgsl): fixed, SMALL
    // particle cluster (side=16, same as the CPU grid_resolution_scaling's
    // build_grid_res_sim), varying ONLY grid resolution. Before Phase 2, grid_update
    // dispatched over the full dense grid_res × grid_res domain regardless of how much of
    // it was actually occupied -- cost should have scaled with grid_res² even though the
    // real workload (particle count) never changed. After Phase 2, cost should stay
    // roughly FLAT across grid_res once the active-block dispatch is doing its job, since
    // the occupied footprint (and therefore the number of active blocks visited) doesn't
    // grow with the surrounding empty grid. A group that still scales with grid_res² here
    // would mean Phase 2 regressed or never took effect -- the real, falsifiable claim
    // this benchmark exists to check, not just "it still passes tests."
    fn build_gpu_grid_res_sim(grid_res: usize) -> GpuSimulation {
        let config = SimConfig::standard(grid_res, 0.1, Vec2::new(0.0, -0.3));
        let side = 16i32; // fixed particle cluster size regardless of grid_res
        let spawn = SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::splat(side),
            box_center: Vec2::splat(grid_res as f32 * 0.5),
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        };
        let particles = build_particles(&config, spawn);
        let (l, u) = lame_from_young(5.0e4, 0.3);
        let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(l, u)));
        let mut sim = block_on(GpuSimulation::new(config, particles, registry));
        for _ in 0..5 {
            sim.step_frame();
        }
        sim
    }

    pub fn bench_gpu_sparse_grid_scaling(c: &mut Criterion) {
        let mut group = c.benchmark_group("gpu_sparse_grid_scaling");
        for &grid_res in &[32usize, 64, 128, 256] {
            let mut sim = build_gpu_grid_res_sim(grid_res);
            group.bench_with_input(BenchmarkId::from_parameter(grid_res), &grid_res, |b, _| {
                b.iter(|| sim.step_frame());
            });
        }
        group.finish();
    }
}

// ── grains_step_scaling / grains_contact_resolution ─────────────────────────
//
// Real perf measurement (2026-08-21), not a guess: `sand_repose_angle_gui.rs`'s
// own Grains mode was observed live at fps=22 with ~80 grains (25 grain-safe
// substeps/frame, see that file's own doc for why). `GrainPopulation::
// resolve_contact_forces` (src/spacetime/grains/population.rs) is an O(n^2)
// all-pairs check -- the prime suspect, unconfirmed until measured. Two
// groups: `grains_step_scaling` mirrors `step_scaling`'s own full-`step()`
// black-box regression guard; `grains_contact_resolution` isolates JUST the
// contact-resolution call (same real isolation discipline `bench_p2g`/
// `bench_g2p` already use for the ordinary-particle transfer layer) so the
// two can be directly compared -- if contact resolution dominates total step
// cost AND scales faster than the other group, that's real, direct evidence
// for (not just a hunch about) an O(n^2) neighbor-search bottleneck, the
// natural next target being the same spatial-hash approach ordinary MPM
// particles already get for free through the grid.

/// Matches `sand_repose_angle_gui.rs`'s own real, already-proven-stable-in-
/// a-live-`Simulation` `grain_contact_config()` exactly, not a fresh guess
/// -- that file's own doc explains why: `ContactLawConfig::dry_sand`'s
/// real-SI stiffness scale (1e5-ish) needs a punishingly fine dt once
/// genuinely grid-coupled, which is BELOW `SimConfig::default().min_dt`
/// (confirmed the hard way: an earlier version of this bench used
/// `dry_sand` directly and hit that exact validation panic). This deliberately
/// softer, real-time-tuned scale is what the demo actually runs.
fn grain_bench_contact_config() -> ContactLawConfig {
    let m_eff = 0.5; // GRAIN_MASS * 0.5, same convention as the demo's own
    const DAMPING_RATIO: f32 = 0.6;
    let critical_damping = |k: f32| 2.0 * (k * m_eff).sqrt() * DAMPING_RATIO;
    let normal_stiffness = 1.0e4;
    let tangential_stiffness = 0.8e4;
    let rolling_stiffness = 5.0e2;
    ContactLawConfig {
        normal_stiffness,
        tangential_stiffness,
        rolling_stiffness,
        normal_damping: critical_damping(normal_stiffness),
        tangential_damping: critical_damping(tangential_stiffness),
        rolling_damping: critical_damping(rolling_stiffness),
        friction: (35.0_f32).to_radians().tan(),
        rolling_friction: 0.2,
    }
}

/// Real jitter+polydispersity column (same non-optional convention every
/// other grain scene in this codebase uses -- an unjittered lattice has no
/// physical asymmetry to ever settle/topple, a real, previously-confirmed
/// false-flat-line trap, not decoration).
fn build_grain_column(n: usize, center_x: f32, base_y: f32) -> Vec<Grain> {
    const RADIUS: f32 = 1.0;
    const MASS: f32 = 1.0;
    let cols = (n as f32).sqrt().ceil() as usize;
    let spacing = 2.6 * RADIUS;
    let mut seed = 0xC0FF_EE11_u64;
    let mut next_f32 = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((seed >> 33) as f32) / (u32::MAX as f32)
    };
    (0..n)
        .map(|i| {
            let row = i / cols;
            let col = i % cols;
            let jx = (next_f32() - 0.5) * 0.3 * spacing;
            let jy = (next_f32() - 0.5) * 0.3 * spacing;
            let x = center_x + (col as f32 - cols as f32 * 0.5) * spacing + jx;
            let y = base_y + row as f32 * spacing + RADIUS + jy;
            let r = RADIUS * (0.9 + 0.2 * next_f32());
            Grain::new(Vec2::new(x, y), r, MASS * (r / RADIUS).powi(2))
        })
        .collect()
}

fn build_grains_sim(target: usize) -> Simulation {
    let cfg = grain_bench_contact_config();
    let m_eff = 0.5; // matches grain mass=1.0 * 0.5, same convention as critical_timestep's own doc
    let dt_crit = critical_timestep(m_eff, &cfg);
    let grain_safe_dt = (dt_crit * 0.2).min(0.02);
    let config = SimConfig {
        grid_res: 128,
        dt: grain_safe_dt,
        gravity: Vec2::new(0.0, -0.3),
        adaptive_timestep: true,
        boundary_thickness: 2,
        ..SimConfig::default()
    };
    let mut solver = Simulation::empty(config).with_boundary(Box::new(FrictionBoundary::new(
        config.boundary_thickness,
        0.6,
    )));
    let grains = build_grain_column(target, 64.0, 4.0);
    solver.add_grain_population(GrainPopulation::new(grains, cfg));
    // Settle briefly under gravity before measuring -- real, consistent with
    // `build_settled_sim`'s own "measure a real physical state, not the
    // initial spawn transient" convention.
    solver.step_n(50);
    solver
}

fn bench_grains_step_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("grains_step_scaling");
    for &target in &[20usize, 50, 100, 200, 400] {
        let mut sim = build_grains_sim(target);
        group.bench_with_input(BenchmarkId::from_parameter(target), &target, |b, _| {
            b.iter(|| sim.step());
        });
    }
    group.finish();
}

fn bench_grains_contact_resolution(c: &mut Criterion) {
    let mut group = c.benchmark_group("grains_contact_resolution");
    for &target in &[20usize, 50, 100, 200, 400] {
        let cfg = grain_bench_contact_config();
        let grains = build_grain_column(target, 64.0, 4.0);
        let mut pop = GrainPopulation::new(grains, cfg);
        // Settle standalone (no grid) first so contacts are representative
        // of a real packed configuration, not the initial loose spawn.
        for _ in 0..2000 {
            pop.step(Vec2::new(0.0, -0.3), 0.001);
        }
        let dt = 0.001;
        group.bench_with_input(BenchmarkId::from_parameter(target), &target, |b, _| {
            b.iter(|| pop.resolve_contact_forces(dt));
        });
    }
    group.finish();
}

// ── registry ──────────────────────────────────────────────────────────────

// KNOWN BROKEN (found 2026-08-21, not fixed, real and disclosed): running
// the full group below via `cargo bench --bench scaling` currently panics
// inside `bench_mixed_materials`'s own setup (`build_mixed_sim`'s `sim.
// step_n(10)`) -- "strict WC-MPM fluid could not advance the full requested
// dt ... a genuine CFL/retry instability, not a false alarm". NOT reproduced
// in any debug-mode test tonight (the whole `cargo test` suite is green,
// see [[grain_apic_and_boundary_default_stacking_bug_2026-08-20]]) -- this
// bench file compiles in the `bench` (release-like, optimized) profile,
// and criterion only skips a filtered-out benchmark's TIMED closure, not
// the setup code every `fn bench_xxx` runs unconditionally before it, so
// even `-- grains` doesn't avoid it. Plausibly a real release-only
// numerical-precision instability (different FP rounding/optimization than
// this whole project's debug-only convention has ever exercised), not yet
// investigated -- a genuinely separate issue from the grain perf work this
// session's own investigation was actually scoped to. `bench_g2p`'s call to
// `gather_grid_to_particles` was ALSO found and fixed here tonight (missing
// two args the function's own signature has long since gained -- this whole
// file had clearly not been run/maintained in a while).
criterion_group!(
    benches,
    step_scaling,
    bench_mixed_materials,
    bench_material_count_scaling,
    bench_force_field_scaling,
    bench_grid_resolution_scaling,
    bench_sand_sheared,
    bench_p2g,
    bench_g2p,
    bench_kirchhoff,
    bench_update_particle,
    bench_grid_update,
    bench_grains_step_scaling,
    bench_grains_contact_resolution,
);

#[cfg(feature = "gpu")]
criterion_group!(
    gpu_benches_group,
    gpu_benches::bench_gpu_sleep_wake_scaling,
    gpu_benches::bench_gpu_step_scaling,
    gpu_benches::bench_gpu_sparse_grid_scaling,
);

#[cfg(feature = "gpu")]
criterion_main!(benches, gpu_benches_group);
#[cfg(not(feature = "gpu"))]
criterion_main!(benches);
