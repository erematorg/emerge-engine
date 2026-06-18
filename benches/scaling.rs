//! Micro-benchmarks for the emerge hot path.
//!
//! Groups:
//!   step_scaling    -- full solver.step() at varying particle counts (regression guard)
//!   mixed_materials -- step() with sand + fluid + jelly simultaneously (LP workload)
//!   sand_sheared    -- step() on pre-deformed sand (50 warm-up steps before measuring)
//!   p2g             -- scatter_particles_to_grid in isolation
//!   g2p             -- gather_grid_to_particles in isolation
//!   kirchhoff       -- kirchhoff_stress per material (NeoHookean / Sand / Fluid / Snow)
//!   update_particle -- plasticity update per material
//!   grid_update     -- grid.update_velocities in isolation
//!
//!   cargo bench --bench scaling
//!   cargo bench --bench scaling -- mixed_materials   (single group)
//!
//! Reports: target/criterion/<group>/report/index.html

extern crate emerge_engine as emerge;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use emerge::{
    BoundaryCondition, DruckerPragerMaterial, Grid, MaterialModel, MaterialRegistry,
    NeoHookeanMaterial, NewtonianFluidMaterial, Particles, SimConfig, Simulation, SlipBoundary,
    SpawnRegion, StomakhinMaterial, ViscoelasticMaterial, build_particles, lame_from_young,
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
        let vel_limit = fx.config.grid_cell_size / dt;
        let boundaries: Vec<Box<dyn BoundaryCondition>> =
            vec![Box::new(SlipBoundary::new(fx.config.boundary_thickness))];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, _| {
            b.iter(|| {
                emerge::transfer::gather_grid_to_particles(
                    &mut fx.particles,
                    &fx.grid,
                    dt,
                    &boundaries,
                    &fx.registry,
                    emerge::transfer::G2PParams {
                        vel_limit,
                        apic_blend: 1.0,
                        active_count: fx.n,
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
                        $mat.update_particle(&mut ps, i, dt);
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
        .with_default_material(Box::new(ViscoelasticMaterial::near_incompressible(5.0e4, 10.0)))
        .with_material(SAND_ID, Box::new(DruckerPragerMaterial::cohesionless(133.3, 0.333)))
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

// â”€â”€ sand_sheared â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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

// â”€â”€ registry â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

criterion_group!(
    benches,
    step_scaling,
    bench_mixed_materials,
    bench_sand_sheared,
    bench_p2g,
    bench_g2p,
    bench_kirchhoff,
    bench_update_particle,
    bench_grid_update,
);
criterion_main!(benches);
