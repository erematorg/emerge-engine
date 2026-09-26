# emerge

[![crates.io](https://img.shields.io/crates/v/emerge-engine.svg)](https://crates.io/crates/emerge-engine)
[![docs.rs](https://docs.rs/emerge-engine/badge.svg)](https://docs.rs/emerge-engine)
[![license](https://img.shields.io/crates/l/emerge-engine.svg)](LICENSE-MIT)

**A physics engine for matter that flows, piles up, bends and breaks.**

Water, sand, snow, mud, rubber, clay, gas. Not objects bouncing off each
other -- actual material, simulated as a continuum, the way a laboratory
would model it.

## What that means, concretely

Most game physics engines treat the world as rigid boxes and spheres that
collide. That works for a crate sliding down a ramp; it cannot give you a
wave breaking, a sandpile collapsing at its natural angle, or snow
compacting underfoot.

emerge takes the other approach. Matter is represented by particles carrying
mass, temperature and deformation, which exchange momentum through a
background grid every step. One single mechanism handles all of it -- there
is no separate fluid system, cloth system and soft-body system bolted
together. Water and sand differ only by which equation describes their
internal stress.

The method is the Material Point Method, specifically MLS-MPM (Hu et al.
2018), the same family of solver used in visual effects and in computational
geomechanics.

## Why it is built this way

**The physics is meant to be real, not merely convincing.** Every material
model comes from published literature and names its source in the code. Snow
follows Stomakhin et al. 2013, sand follows Drucker-Prager as formulated by
Klar et al. 2016, dense granular flow follows the mu(I) rheology of Jop,
Forterre and Pouliquen 2006.

**Constants are measured, not chosen to look nice.** Water's colour comes
from its real absorption spectrum -- 140 laboratory measurements from Pope &
Fry 1997 -- which is why it is nearly clear in a glass and deep blue at
thirty metres, without anyone tuning a colour. Thermal emission comes from
Planck's law read through the CIE 1931 standard observer, so a hot body goes
red, then orange, then white, as real hot bodies do.

**Claims are checked against numbers the project did not pick.** The engine
reproduces CIE standard illuminant A's chromaticity, Fresnel reflectance for
water and glass, and recovers Stefan-Boltzmann's law from an independent
integration of Planck's. Where something is approximated, it is written down
in [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) with its measurement.

The aim is a simulation faithful enough to be useful outside entertainment --
in research, in a museum, in a classroom -- not only inside a game.

## What it is not

Not a game engine. No entity system, no game loop, no asset pipeline, no
renderer beyond a debug view. It advances particles and answers questions
about regions of space; everything else belongs to the caller. It was built
to power [Life's Progress](https://github.com/erematorg/LP) and is usable
standalone by anything needing the same solver.

Pure Rust on the CPU path. An optional wgpu backend runs the entire pipeline
on the GPU.

## How the code is organised

Each top-level folder is a domain of physics, not a layer of software. You
can read any one of them without knowing the others; each has its own README.

| Folder | What lives there |
|---|---|
| [`matter/`](src/matter) | What things are made of -- the particle, and 17 material models with their measured constants |
| [`spacetime/`](src/spacetime) | How matter moves -- the solver itself: particle-to-grid transfer, grid update, grid-to-particle, plus a separate solver for slender bodies |
| [`forces/`](src/forces) | What pushes on matter -- gravity, drag, buoyancy, walls and friction |
| [`energy/`](src/energy) | Heat and light -- conduction, diffusion, radiation, and the optics that turn a spectrum into a colour |
| [`information/`](src/information) | Control and measurement -- a neural locomotion controller, information-theoretic measures |
| [`systems/`](src/systems) | The machinery, not the physics -- GPU compute, rendering, diagnostics |
| [`runtime/`](src/runtime) | Fixed-timestep stepping |

The rule the tree follows: a piece of code lives where its *concept* belongs,
never where it happens to be used. Beer-Lambert absorption is a law of energy
transport, so it sits in `energy/` even though only the renderer calls it.
Water's absorption spectrum is a property of water, so it sits in `matter/`.
`systems/` consumes physics and never owns any.

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

Thirteen constitutive models, grouped by what they're for:

| Group | Models |
|---|---|
| **Elastic solids** | `NeoHookeanMaterial` (finite-strain), `CorotatedMaterial` (stiffer, corotated-linear), `ViscoelasticMaterial` (Kelvin-Voigt) |
| **Fluids** | `NewtonianFluidMaterial` (Tait EOS + viscosity), `BinghamFluidMaterial` (adds a yield stress -- mud, not water) -- both take `surface_tension_coeff` for free |
| **Granular** | `StomakhinMaterial` (snow), `DruckerPragerMaterial` / `MuIRheologyMaterial` (two ways to get sand right), `GranularFluidMaterial` (granular suspensions) |
| **Plastic / failure** | `VonMisesMaterial` (ductile), `RankineMaterial` (brittle, damage softening), `NaccMaterial` (Cam-Clay soil) |
| **Tension-only** | `NoCompressionMaterial` (cables, membranes, tendons) |

Each cites its source paper in the doc comment -- see [Physics references](#physics-references). Quick presets:

| Type | Key preset |
|---|---|
| `NeoHookeanMaterial` | `from_young_modulus(E, nu)` |
| `CorotatedMaterial` | `from_young_modulus(E, nu)` |
| `ViscoelasticMaterial` | `.near_incompressible()` `.moderately_compressible()` |
| `NewtonianFluidMaterial` | `.low_viscosity(density, stiffness)` |
| `BinghamFluidMaterial` | `.low_yield()` `.medium_yield()` `.high_yield()` |
| `StomakhinMaterial` | `from_young_modulus(E, nu)` `.low_cohesion()` |
| `DruckerPragerMaterial` | `.cohesionless()` `.low_friction()` `.dilatant()` |
| `MuIRheologyMaterial` | `.small_grain()` `.dense_packed()` |
| `VonMisesMaterial` | `from_young_modulus(E, nu, yield_stress)` |
| `RankineMaterial` | `.stiff_brittle()` `.high_tensile()` |
| `NaccMaterial` | `.soft_clay(E, nu)` `.wet_soil(E, nu)` |
| `GranularFluidMaterial` | `.saturated_loam(E, nu)` `.cytoplasmic(E, nu)` |
| `NoCompressionMaterial` | `FromSI<Elastic>` |

## Rod solver

A second, narrower solver alongside the MPM materials above -- for slender (length ≫ width) bodies like a blade of grass or a fishing line, where `EI` (bending stiffness) is a direct input instead of an emergent property of carved cross-section width. Shares the same grid every MPM material uses, so a rod and ordinary particles genuinely exchange momentum, not two solvers running side by side.

| Type / API | Meaning |
|---|---|
| `rod::Rod` | embedded in a `Simulation` -- `points` (`RodPoints`), `material`, `wind_velocity`/`wind_drag_coeff`, `push_center`/`push_strength`/`push_radius` (read fresh every substep), `sleeping` |
| `rod::RodPoints` | own SoA -- `x`/`v`/`mass`/`pinned`/`rest_edge_length`/`rest_curvature` |
| `rod::RodMaterial` | real `EA`/`EI` -- `from_young_modulus_rectangular(E, width, thickness, axial_damping, bending_damping)`, `.critical_damping(l0, mass, ea, ei)` |
| `rod::build_straight_rod(start, end, n_points, linear_density, dx_meters)` | construct a straight `RodPoints` |
| `rod::rod_cfl_dt(&points, &material, safety)` | Gershgorin-bound CFL dt, folded into `choose_substep_dt` automatically |
| `Simulation::add_rod`/`with_rod`/`rods()`/`rods_mut()` | lifecycle, same fluent convention as `with_default_material` |

Real Euler/Greenhill self-buckling comes out of the same `EA`/`EI` for free -- no separate stability model needed. See [Physics references](#physics-references) for the citation.

Same sleep/wake pattern as MPM particles, at rod granularity (a rod's points are elastically coupled, so it sleeps as one body): `SimConfig::rod_sleep_threshold` (0.0 = disabled). A settled rod skips scatter/gather/force integration *and* its own `rod_cfl_dt` term in the substep bound -- the real fix for many-simultaneous-rods cost (a grass field), since that per-point CFL sum is otherwise paid every substep regardless of how still the rod looks. Wakes on push or on any new grid activity nearby (e.g. something landing on it).

## Force fields & boundaries

Ten force fields, six boundary conditions -- mix and match, all optional, zero cost when unused.

| Group | Types |
|---|---|
| **External fields** | `GravityWellField` / `NBodyGravityField` (Barnes-Hut N-body), `CoulombField` / `UniformElectricField`, `BuoyancyField` |
| **Flow / drag** | `LinearDragField` (fixed target velocity -- wind, current), `SpatialDragField` (spatially-varying flow) |
| **Confinement** | `RadialConfinementField`, `AabbConfinementField` |
| **Chemotaxis** | `ChemotaxisField` (gradient-following, Keller-Segel) |
| **Domain walls** | `SlipBoundary` (default), `FrictionBoundary`, `GripFrictionBoundary` (strain-rate-gated grip), `RatchetFrictionBoundary` (direction-dependent), `HeightmapBoundary` (terrain profile) |

## Particle fields

`Particle` is `repr(C)`, 128 bytes, GPU-uploadable -- every field below round-trips through the GPU pipeline unchanged.

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
| `friction_hardening` | `f32` | shared plasticity scratch -- DP `q` / Von Mises `κ` / Rankine damage / SandMuI µ(I). One particle runs one material, so one field safely serves all of them |
| `log_volume_strain` | `f32` | DP εᵥ |
| `temperature` | `f32` | used by thermal diffusion + phase rules |
| `user_tag` | `u32` | caller-owned (e.g. LP: creature/body ownership) -- no engine meaning |
| `activation` | `f32` | [0,1] active-matter drive (muscle contraction) |
| `activation_dir` | `Vec2` | muscle fiber direction, material frame |
| `muscle_group_id` | `u32` | tags a subset of particles for independent activation control -- same continuum, different control group |
| `contact_group` | `u32` | 0 = ordinary particle; nonzero = opts into multi-field frictional contact (Bardenhagen 2001 + Nairn/Hammerquist/Smith 2020). Zero-cost when unused |
| `sleeping` | `u32` | active/sleeping partition flag |
| `internal_pressure` | `f32` | pre-stress pressure (already SI-converted to grid units) |
| `pinned` | `u32` | real Dirichlet anchor -- forces v=0, velocity_gradient=0 every substep in G2P |

## Core mechanisms

- **`MaterialModel::activation_scale()`** -- scaling coefficient for activation-driven deviatoric stress. Muscle/active-matter hook. Default 0.0 (opt-in per material).
- **`MaterialModel::pressure_scale()`** -- scaling coefficient for internal pre-stress. Turgor-pressure-style hook (any internally-pressurized body, not plant-specific). Default 0.0.
- **`MixturePhase`** (`SOLID`/`FLUID`) -- two-phase mixture coupling role (Tampubolon et al. 2017, Darcy drag between interpenetrating granular/fluid phases).
- **`WithLatentHeat<M>` / `WithMixturePhase<M>` / `WithPreStress<M>`** -- delegating wrapper structs that bolt one extra behavior onto any `MaterialModel` without rewriting it.
- **`add_phase_rule(Fn(&Particle) -> Option<u32>)`** -- automatic material_id transition evaluated every substep (freezing, melting, evaporation).
- **`ScalarDiffusionField`** -- generic diffusion field (heat, pheromone, nutrients, morphogen). Reaction-diffusion (Gray-Scott/Turing) ready via its `source` closure.
- **Sleep/wake** -- flag-based active/sleeping partition, not memory compaction. Particles: `SimConfig::sleep_threshold`, per-particle swap into a sleeping tail. Rods: see [Rod solver](#rod-solver) above.
- **Adaptive substeps** -- `Simulation::step()` always advances exactly `config.dt`, internally split into as many CFL-safe substeps as needed. Not a tuning knob -- real physics.
- **`Lnn`** (`information::control::lnn`) -- Liquid Time-constant Network CPG (Hasani et al. 2020), a standalone locomotion controller. Does not participate in the substep loop; writes into `activation`/`activation_dir` between steps.
- **`spacetime::diff`** -- separate, hand-derived-adjoint forward+reverse MLS-MPM implementation for gradient-based offline controller training. Not used at real-time/play time.

## API reference

```rust
// Spawn a body later (e.g. a creature born mid-run)
let creature_id = sim.add_body(SpawnRegion::for_sim(&config));

// Carve a shape out of one continuous lattice (no seams, no detachment risk --
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

Every seam above accepts external implementations with no engine-internal access -- see `tests/extensibility.rs` for a working proof (each seam implemented as a third-party consumer, against `emerge::prelude::*` alone).

## Features

- `gpu` -- the whole pipeline (P2G, grid update, G2P, every plasticity model) as WGSL compute
- `render` -- instanced particle debug renderer, on top of `gpu`
- `experimental` -- acoustics, electromagnetics, information-theoretic measures (real, just not API-stable yet)

## Examples

```sh
cargo run --example headless                        # no feature flags, start here
cargo run --example basic_sand      --features render
cargo run --example basic_fluids    --features render
cargo run --example basic_snow      --features render
cargo run --example basic_jellies   --features render
cargo run --example basic_creature  --features render  # LNN-driven muscle locomotion
cargo run --example basic_showcase  --features render  # three materials at once
cargo run --example basic_sand_grid_gpu --features render
cargo run --example rod_blade_of_grass                    # rod solver, single blade, no window
cargo run --example rod_blade_and_root --features render  # rod solver, two-blade buckling comparison + root, live + pushable
```

Windowed examples (everything except `headless` and `validate_materials`) need `--features render` -- they draw via wgpu/winit directly, no Bevy.

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
