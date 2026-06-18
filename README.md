# emerge

[![crates.io](https://img.shields.io/crates/v/emerge-engine.svg)](https://crates.io/crates/emerge-engine)
[![docs.rs](https://docs.rs/emerge-engine/badge.svg)](https://docs.rs/emerge-engine)
[![license](https://img.shields.io/crates/l/emerge-engine.svg)](LICENSE-MIT)

Real-time continuum physics engine. MLS-MPM (Hu et al. 2018), pure Rust, optional wgpu GPU backend.

Built for [Life's Progress](https://github.com/M1thieu/LP), a systemic simulation game. Not a game engine: no ECS, no rendering, no game loop. Physics only.

```toml
[dependencies]
emerge = { package = "emerge-engine", version = "0.1" }
# with GPU compute:
emerge = { package = "emerge-engine", version = "0.1", features = ["gpu"] }
```

## Quick start

```rust
use emerge::prelude::*;

const WATER: u32 = 1;

let config = SimConfig::standard(64, 0.05, Vec2::NEG_Y);

let mut sim = Simulation::empty(config)
    .with_default_material(Box::new(NeoHookeanMaterial::new(400.0, 200.0)))
    .with_material(WATER, Box::new(NewtonianFluidMaterial::water(1000.0, 1e4)))
    .with_boundary(Box::new(SlipBoundary::new(2)));

sim.add_body(SpawnRegion {
    box_size: IVec2::new(12, 12),
    box_center: Vec2::new(24.0, 40.0),
    precompute_initial_volumes: true,
    ..SpawnRegion::for_sim(&config)
});

sim.add_body(SpawnRegion {
    box_size: IVec2::new(12, 8),
    box_center: Vec2::new(40.0, 36.0),
    material_id: WATER,
    precompute_initial_volumes: true,
    ..SpawnRegion::for_sim(&config)
});

sim.step_n(60);

let state = sim.region_state(Vec2::new(40.0, 36.0), 10.0);
println!("avg speed: {:.3}", state.avg_speed);
```

## Materials

| Model | Constitutive description |
|---|---|
| `NeoHookeanMaterial` | Hyperelastic, finite-strain (Green-Lagrange energy) |
| `CorotatedMaterial` | Corotated linear elasticity, stiff baseline |
| `ViscoelasticMaterial` | Kelvin-Voigt: elastic spring + viscous dashpot in parallel |
| `NewtonianFluidMaterial` | Tait EOS pressure + Newtonian viscosity |
| `BinghamFluidMaterial` | Tait EOS + viscoplastic yield stress (Bingham) |
| `StomakhinMaterial` | Corotated elastoplastic, SVD singular-value return mapping, Jp hardening (Stomakhin 2013) |
| `DruckerPragerMaterial` | Elastoplastic, Drucker-Prager cone yield surface, dilatancy (Klar 2016) |
| `MuIRheologyMaterial` | Elastoplastic, rate-dependent friction µ(I), dense granular flow |
| `VonMisesMaterial` | J2 plasticity, linear isotropic hardening |
| `RankineMaterial` | Tensile cutoff + exponential damage softening (brittle) |
| `NaccMaterial` | Non-Associated Cam-Clay, critical state soil mechanics |
| `GranularFluidMaterial` | Tait EOS + corotated deviatoric + SVD plasticity (fluid-granular) |

Surface tension is built into `NewtonianFluidMaterial` and `BinghamFluidMaterial` via `surface_tension_coeff`.

## Features

| Flag | Description |
|---|---|
| `gpu` | wgpu WGSL compute backend, all plasticity on GPU |
| `render` | Instanced particle debug renderer (requires `gpu`) |
| `experimental` | Acoustics, EM, information-theoretic measures |

## Examples

```sh
cargo run --example headless        # no feature flags, start here
cargo run --example basic_sand
cargo run --example basic_fluids
cargo run --example basic_jellies
cargo run --example basic_showcase  # three materials at once
cargo run --example basic_sand_gpu     --features gpu
cargo run --example basic_fluids_gpu   --features gpu
```

## Physics references

| Module | Paper |
|---|---|
| MLS-APIC transfer | Hu et al. 2018, *A Moving Least Squares Material Point Method* |
| NeoHookean / Corotated | Stomakhin et al. 2012, *Energetically Consistent Invertible Elasticity* |
| Snow | Stomakhin et al. 2013, *A Material Point Method for Snow Simulation* |
| Sand | Klar et al. 2016, *Drucker-Prager Elastoplasticity for Sand Animation* |
| µ(I)-rheology | Dunatunga & Kamrin 2015, *Continuum modelling and simulation of granular flow* |
| Surface tension | Stomakhin et al. 2014, *Augmented MPM for cloth and soft bodies* |
| N-body gravity | Barnes & Hut 1986, *A hierarchical O(N log N) force-calculation algorithm* |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
