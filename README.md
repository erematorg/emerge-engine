# emerge

[![crates.io](https://img.shields.io/crates/v/emerge-engine.svg)](https://crates.io/crates/emerge-engine)
[![docs.rs](https://docs.rs/emerge-engine/badge.svg)](https://docs.rs/emerge-engine)
[![license](https://img.shields.io/crates/l/emerge-engine.svg)](LICENSE-MIT)

An MLS-MPM continuum solver (Hu et al. 2018). Fluids, sand, snow, elastic and plastic solids — one particle-grid transfer for all of them. No rigid bodies, no separate fluid/cloth/soft-body systems bolted together. Pure Rust on the CPU path; an optional wgpu backend runs the whole pipeline on GPU.

Built for [Life's Progress](https://github.com/erematorg/LP). Not a game engine — no ECS, no game loop, no asset pipeline. It steps particles forward and answers queries about regions of space; everything else is up to the caller.

```toml
[dependencies]
emerge = { package = "emerge-engine", version = "0.1" }
# GPU compute, all plasticity included:
emerge = { package = "emerge-engine", version = "0.1", features = ["gpu"] }
```

## Quick start

```rust
use emerge::prelude::*;

const WATER: u32 = 1;

let config = SimConfig::standard(64, 0.05, Vec2::NEG_Y);

let mut sim = Simulation::empty(config)
    .with_default_material(Box::new(NeoHookeanMaterial::new(400.0, 200.0)))
    .with_material(WATER, Box::new(NewtonianFluidMaterial::low_viscosity(1000.0, 1e4)))
    .with_boundary(Box::new(SlipBoundary::new(2)));

let _ = sim.add_body(SpawnRegion {
    box_size: IVec2::new(12, 12),
    box_center: Vec2::new(24.0, 40.0),
    precompute_initial_volumes: true,
    ..SpawnRegion::for_sim(&config)
});

let _ = sim.add_body(SpawnRegion {
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

Fourteen constitutive models, grouped by real continuum state of matter — `matter/materials/{solid,liquid,gas,mixture}/` on disk mirrors this exactly, not an implementation-layer split:

| State | Models |
|---|---|
| **Solid** — elastic | `NeoHookeanMaterial` (finite-strain), `CorotatedMaterial` (stiffer, corotated-linear), `ViscoelasticMaterial` (Kelvin-Voigt) |
| **Solid** — granular | `StomakhinMaterial` (snow), `DruckerPragerMaterial` / `MuIRheologyMaterial` (two ways to get sand right) |
| **Solid** — plastic / failure | `VonMisesMaterial` (ductile), `RankineMaterial` (brittle, damage softening), `NaccMaterial` (Cam-Clay soil) |
| **Solid** — tension-only | `NoCompressionMaterial` (cables, membranes, tendons) |
| **Liquid** | `NewtonianFluidMaterial` (Tait EOS + viscosity), `BinghamFluidMaterial` (adds a yield stress — mud, not water) — both take `surface_tension_coeff` for free |
| **Gas** | `GasMaterial` (isentropic ideal-gas EOS, real adiabatic sound speed — CPU only, no GPU shader branch yet) |
| **Mixture** | `GranularFluidMaterial` — genuinely both at once (`τ = τ_EOS(liquid) + τ_corotated(solid)`, Dunatunga & Kamrin 2015), not a fifth state |

`plasma/` exists as a documented placeholder folder — real quantum/exotic states beyond these four are out of scope for a classical continuum engine. Each material cites its source paper in the doc comment — see [Physics references](#physics-references). Quick presets:

| Type | Key preset |
|---|---|
| `NeoHookeanMaterial` | `from_young_modulus(E, nu)` |
| `CorotatedMaterial` | `from_young_modulus(E, nu)` |
| `ViscoelasticMaterial` | `.near_incompressible()` `.moderately_compressible()` |
| `StomakhinMaterial` | `from_young_modulus(E, nu)` `.low_cohesion()` |
| `DruckerPragerMaterial` | `.cohesionless()` `.low_friction()` `.dilatant()` |
| `MuIRheologyMaterial` | `.small_grain()` `.dense_packed()` |
| `VonMisesMaterial` | `from_young_modulus(E, nu, yield_stress)` |
| `RankineMaterial` | `.stiff_brittle()` `.high_tensile()` |
| `NaccMaterial` | `.soft_clay(E, nu)` `.wet_soil(E, nu)` |
| `NoCompressionMaterial` | `FromSI<Elastic>` |
| `NewtonianFluidMaterial` | `.low_viscosity(density, stiffness)` |
| `BinghamFluidMaterial` | `.low_yield()` `.medium_yield()` `.high_yield()` |
| `GasMaterial` | `::air(rho_kg_m3, temperature_k, &config)` `::from_physical(...)` |
| `GranularFluidMaterial` | `.saturated_loam(E, nu)` `.cytoplasmic(E, nu)` |

## Rod solver

A second, narrower solver alongside the MPM materials above — for slender (length ≫ width) bodies like a blade of grass or a fishing line, where `EI` (bending stiffness) is a direct input instead of an emergent property of carved cross-section width. Shares the same grid every MPM material uses, so a rod and ordinary particles genuinely exchange momentum, not two solvers running side by side.

| Type / API | Meaning |
|---|---|
| `rod::Rod` | embedded in a `Simulation` — `points` (`RodPoints`), `material`, `wind_velocity`/`wind_drag_coeff`, `push_center`/`push_strength`/`push_radius` (read fresh every substep), `sleeping` |
| `rod::RodPoints` | own SoA — `x`/`v`/`mass`/`pinned`/`rest_edge_length`/`rest_curvature` |
| `rod::RodMaterial` | real `EA`/`EI` — `from_young_modulus_rectangular(E, width, thickness, axial_damping, bending_damping)`, `.critical_damping(l0, mass, ea, ei)` |
| `rod::build_straight_rod(start, end, n_points, linear_density, dx_meters)` | construct a straight `RodPoints` |
| `rod::rod_cfl_dt(&points, &material, safety)` | Gershgorin-bound CFL dt, folded into `choose_substep_dt` automatically |
| `Simulation::add_rod`/`with_rod`/`rods()`/`rods_mut()` | lifecycle, same fluent convention as `with_default_material` |

Real Euler/Greenhill self-buckling comes out of the same `EA`/`EI` for free — no separate stability model needed. See [Physics references](#physics-references) for the citation.

Same sleep/wake pattern as MPM particles, at rod granularity (a rod's points are elastically coupled, so it sleeps as one body): `SimConfig::rod_sleep_threshold` (0.0 = disabled). A settled rod skips scatter/gather/force integration *and* its own `rod_cfl_dt` term in the substep bound — the real fix for many-simultaneous-rods cost (a grass field), since that per-point CFL sum is otherwise paid every substep regardless of how still the rod looks. Wakes on push or on any new grid activity nearby (e.g. something landing on it).

## Force fields & boundaries

Ten force fields, six boundary conditions — mix and match, all optional, zero cost when unused.

| Group | Types |
|---|---|
| **External fields** | `GravityWellField` / `NBodyGravityField` (Barnes-Hut N-body), `CoulombField` / `UniformElectricField`, `BuoyancyField` |
| **Flow / drag** | `LinearDragField` (fixed target velocity — wind, current), `SpatialDragField` (spatially-varying flow) |
| **Confinement** | `RadialConfinementField`, `AabbConfinementField` |
| **Chemotaxis** | `ChemotaxisField` (gradient-following, Keller-Segel) |
| **Domain walls** | `SlipBoundary` (default), `FrictionBoundary`, `PredictiveBoundary` (tighter keep-out), `GripFrictionBoundary` (strain-rate-gated grip), `RatchetFrictionBoundary` (direction-dependent), `HeightmapBoundary` (terrain profile) |

## Particle fields

`Particle` is `repr(C)`, 128 bytes, GPU-uploadable — every field below round-trips through the GPU pipeline unchanged.

| Field | Type | Meaning |
|---|---|---|
| `x` | `Vec2` | position (grid coords) |
| `v` | `Vec2` | velocity |
| `velocity_gradient` | `Mat2` | APIC affine matrix C, ∂v/∂x |
| `deformation_gradient` | `Mat2` | F |
| `mass`, `initial_volume`, `volume`, `density` | `f32` | standard MPM state |
| `material_id` | `u32` | slot index into `MaterialRegistry` |
| `plastic_volume_ratio` | `f32` | Jₚ = det(Fₚ) |
| `hardening_scale` | `f32` | h = exp(ξ(1−Jₚ)) |
| `friction_hardening` | `f32` | shared plasticity scratch — DP `q` / Von Mises `κ` / Rankine damage / SandMuI µ(I). One particle runs one material, so one field safely serves all of them |
| `log_volume_strain` | `f32` | DP εᵥ |
| `temperature` | `f32` | used by thermal diffusion + phase rules |
| `user_tag` | `u32` | caller-owned (e.g. LP: creature/body ownership) — no engine meaning |
| `activation` | `f32` | [0,1] active-matter drive (muscle contraction) |
| `activation_dir` | `Vec2` | muscle fiber direction, material frame |
| `muscle_group_id` | `u32` | tags a subset of particles for independent activation control — same continuum, different control group |
| `contact_group` | `u32` | 0 = ordinary particle; nonzero = opts into multi-field frictional contact (Bardenhagen 2001 + Nairn/Hammerquist/Smith 2020). Zero-cost when unused |
| `sleeping` | `u32` | active/sleeping partition flag |
| `internal_pressure` | `f32` | pre-stress pressure (already SI-converted to grid units) |
| `pinned` | `u32` | real Dirichlet anchor — forces v=0, velocity_gradient=0 every substep in G2P |

## Core mechanisms

- **`MaterialModel::activation_scale()`** — scaling coefficient for activation-driven deviatoric stress. Muscle/active-matter hook. Default 0.0 (opt-in per material).
- **`MaterialModel::pressure_scale()`** — scaling coefficient for internal pre-stress. Turgor-pressure-style hook (any internally-pressurized body, not plant-specific). Default 0.0.
- **`MixturePhase`** (`SOLID`/`FLUID`) — two-phase mixture coupling role (Tampubolon et al. 2017, Darcy drag between interpenetrating granular/fluid phases).
- **`WithLatentHeat<M>` / `WithMixturePhase<M>` / `WithPreStress<M>`** — delegating wrapper structs that bolt one extra behavior onto any `MaterialModel` without rewriting it.
- **`add_phase_rule(Fn(&Particle) -> Option<u32>)`** — automatic material_id transition evaluated every substep (freezing, melting, evaporation).
- **`ScalarDiffusionField`** — generic diffusion field (heat, pheromone, nutrients, morphogen). Reaction-diffusion (Gray-Scott/Turing) ready via its `source` closure.
- **Sleep/wake** — flag-based active/sleeping partition, not memory compaction. Particles: `SimConfig::sleep_threshold`, per-particle swap into a sleeping tail. Rods: see [Rod solver](#rod-solver) above.
- **Adaptive substeps** — `Simulation::step()` always advances exactly `config.dt`, internally split into as many CFL-safe substeps as needed. Not a tuning knob — real physics.
- **`Lnn`** (`information::control::lnn`) — Liquid Time-constant Network CPG (Hasani et al. 2020), a standalone locomotion controller. Does not participate in the substep loop; writes into `activation`/`activation_dir` between steps.
- **`spacetime::diff`** — separate, hand-derived-adjoint forward+reverse MLS-MPM implementation for gradient-based offline controller training. Not used at real-time/play time.

## API reference

```rust
// Spawn a body later (e.g. a creature born mid-run)
let creature_id = sim.add_body(SpawnRegion::for_sim(&config));

// Carve a shape out of one continuous lattice (no seams, no detachment risk —
// never stack multiple separately-spawned SpawnRegions edge-to-edge)
sim.retain_particles(|p| /* keep predicate */ true);

// Phase transitions
sim.phase_transition(|p| p.temperature > 373.0, STEAM_ID);
sim.add_phase_rule(|p| if p.material_id == WATER && p.temperature < 273.0 { Some(ICE) } else { None });

// Neighbor queries
for idx in sim.particles_near(center, radius) { .. }
let n = sim.count_near(center, radius, FOOD_ID);

// Impulses
sim.apply_impulse(center, radius, force);
sim.apply_radial_impulse(center, radius, strength);

// Queries
sim.material_state(material_id) -> BodyState
sim.region_state(center, radius) -> BodyState
sim.particles() / particles_mut()
sim.diagnostics_snapshot() -> SimSnapshot   // min/max_deformation_j, total_kinetic_energy, max_pinned_particle_speed, ...
```

### Extension seams

| Seam | Trait | Applied |
|---|---|---|
| Constitutive response | `MaterialModel` / `ConstitutiveModel` + `PlasticityModel` | P2G stress |
| External body forces | `Field` | after G2P |
| Grid boundaries | `BoundaryCondition` | grid update |
| Multi-field contact | `Particle::contact_group` (opt-in, not a trait) | grid update |
| Scalar transport | `ScalarDiffusionField` | per substep |
| Phase change | phase rules (`Fn(&Particle) -> Option<u32>`) | per substep |
| Observation | `DiagnosticsRegistry` plugins | per step |

Every seam above accepts external implementations with no engine-internal access — see `tests/extensibility.rs` for a working proof (each seam implemented as a third-party consumer, against `emerge::prelude::*` alone).

## Features

- `gpu` — the whole pipeline (P2G, grid update, G2P, every plasticity model) as WGSL compute
- `render` — instanced particle debug renderer, on top of `gpu`
- `experimental` — acoustics, electromagnetics, information-theoretic measures (real, just not API-stable yet)

## Examples

```sh
cargo run --example headless                        # no feature flags, start here
cargo run --example basic_sand      --features render
cargo run --example basic_fluids    --features render
cargo run --example basic_snow      --features render
cargo run --example basic_jellies   --features render
cargo run --example basic_creature  --features render  # LNN-driven muscle locomotion
cargo run --example basic_showcase  --features render  # three materials at once
cargo run --example basic_sand_gpu  --features render
cargo run --example rod_blade_of_grass                       # rod solver, no window
cargo run --example rod_blade_of_grass_gui --features render # rod solver, live + pushable
```

Windowed examples (everything except `headless` and `validate_materials`) need `--features render` — they draw via wgpu/winit directly, no Bevy.

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
| Rod (Cosserat) | Bergou, Wardetzky, Robinson, Audoly & Grinspun 2008, *Discrete Elastic Rods* |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Licensed under either of [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE) at your option.
