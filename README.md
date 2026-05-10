# emerge

Real-time continuum physics engine. MLS-MPM (Hu et al. 2018), pure Rust, wgpu GPU backend.

Built for [Life's Progress](https://github.com/M1thieu/LP) — a systemic simulation game.
Not a game engine: no ECS, no rendering, no game logic. Physics only.

## Materials

| Model | Use case |
|---|---|
| `NeoHookeanMaterial` | Soft solids, muscle, elastic bodies |
| `CorotatedMaterial` | Stiffer elastic baseline |
| `ViscoelasticMaterial` | Tissue, hydrogel, damped solids (Kelvin-Voigt) |
| `NewtonianFluidMaterial` | Water, thin fluids (Tait EOS + viscosity) |
| `BinghamFluidMaterial` | Viscoplastic fluids with yield stress |
| `SnowMaterial` | Compressible granular + SVD plasticity (Stomakhin 2013) |
| `SandMaterial` | Drucker-Prager return mapping + dilatancy (Klar 2016) |
| `SandMuIMaterial` | Rate-dependent granular friction µ(I)-rheology |
| `VonMisesMaterial` | J2 plasticity, linear isotropic hardening |
| `RankineMaterial` | Tensile cutoff + exponential softening (brittle fracture) |

## Quick start

```toml
[dependencies]
emerge = { path = "path/to/emerge" }
# GPU backend:
emerge = { path = "path/to/emerge", features = ["gpu"] }
```

```rust
use emerge::prelude::*;

const FLUID_ID: u32 = 1;

let config = SolverConfig::standard(64, 0.05, Vec2::NEG_Y);

// Default material (id=0): elastic jelly in the centre.
let elastic_spawn = SpawnConfig {
    spacing: 0.5,
    box_size: IVec2::new(12, 12),
    box_center: Vec2::new(24.0, 40.0),
    precompute_initial_volumes: true,
    ..SpawnConfig::for_solver(&config)
};

// Material 1: Newtonian fluid, spawned separately.
let fluid_spawn = SpawnConfig {
    spacing: 0.5,
    box_size: IVec2::new(12, 8),
    box_center: Vec2::new(40.0, 36.0),
    material_id: FLUID_ID,
    precompute_initial_volumes: true,
    ..SpawnConfig::for_solver(&config)
};

let mut solver = MpmSolver::new(config, elastic_spawn)
    .with_default_material(Box::new(NeoHookeanMaterial::new(400.0, 200.0)))
    .with_material(FLUID_ID, Box::new(NewtonianFluidMaterial::new(1000.0, 1e-3, 1e4, 7.0)))
    .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

solver.spawn_region(fluid_spawn);
solver.step_n(60);

let state = solver.material_state(FLUID_ID);
println!("fluid centroid: {:?}  avg_speed: {:.3}", state.centroid, state.avg_speed);
```

## Examples

```sh
# No dependencies — start here
cargo run --example headless

# Bevy visualisation
cargo run --example basic_sand    --features bevy_examples
cargo run --example basic_snow    --features bevy_examples
cargo run --example basic_fluids  --features bevy_examples
cargo run --example basic_jellies --features bevy_examples
cargo run --example basic_showcase --features bevy_examples

# GPU variants (wgpu compute — all plasticity on GPU)
cargo run --example basic_fluids_gpu  --features bevy_examples,gpu
cargo run --example basic_jellies_gpu --features bevy_examples,gpu
cargo run --example basic_sand_gpu    --features bevy_examples,gpu
cargo run --example basic_snow_gpu    --features bevy_examples,gpu
```

## GPU pipeline

Per substep: `grid_clear → p2g → grid_update → g2p → particles_update → force_fields`

All plasticity runs on GPU. No CPU roundtrip per substep.
Particle sort: identity permutation currently (radix sort — future work).

## Features

| Feature | Description |
|---|---|
| `gpu` | wgpu WGSL compute backend |
| `render` | Instanced particle debug draw (requires `gpu`) |
| `bevy_examples` | Bevy + egui examples |
| `experimental` | Acoustics, electromagnetics, information-theoretic measures |

## Build

```sh
cargo test          # 97 tests, physics correctness + integration
cargo check --features gpu,experimental
```

Debug mode only during development. Never `cargo clean` — full rebuild is 10–30 min.
