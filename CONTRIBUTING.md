# Contributing to emerge

emerge is a 2D MLS-MPM continuum physics engine in pure Rust, built for real-time simulation of fluids, granular materials, elastic solids, and active matter.

---

## Issues

Bug reports and questions are welcome via GitHub issues. This is a solo-maintained project — response time is best-effort, not guaranteed. Include a minimal repro (a failing test or example is ideal) where possible; it's the fastest path to a fix.

---

## Quick start

```sh
cargo check                        # verify it compiles
cargo test                         # run all unit + integration tests
cargo clippy -- -D warnings        # must be clean before any PR
cargo run --example headless       # smoke test, no feature flags needed
```

No `cargo clean`; incremental builds work fine. Debug mode only, never `--release` during development.

---

## Architecture

```
src/
  solver/        Simulation · SimConfig · SpawnRegion · spatial hash · query
  particle.rs    Particle (repr(C), 112 B, GPU-uploadable) · Particles (SoA)
  grid/          Grid · Cell · quadratic B-spline kernel
  transfer.rs    P2G scatter + G2P gather (MLS-APIC)
  boundary.rs    BoundaryCondition + Slip / Predictive / Friction / Heightmap
  materials/     MaterialModel trait · 12 constitutive models · SVD · registry
  fields/        Field trait · NBody / GravityWell / Coulomb / EM / Confinement
  control/       Lnn (neural locomotion controller)
  thermodynamics/ ThermalDiffusion · ScalarDiffusionField
  diagnostics/   plugin system · health · per-material stats
  runtime/       FixedStepController
  gpu/           [feature=gpu] GpuSimulation + WGSL shaders
  render/        [feature=render] instanced particle renderer
```

Feature flags: `gpu` | `render` (requires `gpu`) | `experimental`

---

## Adding a material model

A new material requires changes in four places:

### 1. `src/materials/<name>.rs`

Implement the `MaterialModel` trait:

All methods have default implementations (an elastic-only material can override just
`kirchhoff_stress`). The signatures below are exact — copy them, not the idea of them:

```rust
pub struct MyMaterial { /* parameters */ }

impl MaterialModel for MyMaterial {
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 { ... }
    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 { ... }
    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) { ... }
    // Seeds per-particle plastic state at spawn time — takes a single `Particle`,
    // not the `Particles` collection (called once per particle, before it's in the SoA).
    fn init_particle(&self, particle: &mut Particle) { ... }
    // CFL bound — reads the spawned particle's own state, not just dx/rho.
    fn timestep_bound(
        &self,
        particles: &Particles,
        i: usize,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 { ... }
    fn needs_cpu_update(&self) -> bool { false }
}
```

### 2. `src/materials/mod.rs`

Add a variant to `ConstitutiveModel`. The discriminant must be the next consecutive `u32`, and a matching compile-time ABI assertion is required:

```rust
#[repr(u32)]
pub enum ConstitutiveModel {
    // ... existing variants ...
    MyMaterial = 12,  // next available discriminant
}

// in the assert block:
assert!(ConstitutiveModel::MyMaterial as u32 == 12);
```

Re-export from `mod.rs` and add to `src/prelude.rs`.

### 3. `src/gpu/shaders/p2g.wgsl`

Add `case 12u` to the Kirchhoff stress `switch`. If the material is CPU-only, return zero stress and set `needs_cpu_update = true` in Rust.

### 4. `src/gpu/shaders/particles_update.wgsl`

Add `case 12u` to the plasticity update `switch`. CPU-only materials can leave this as a no-op.

---

## Code conventions

- **Zero warnings.** Fix the root cause; never `#[allow(...)]`.
- **Comments explain why, not what.** Good reasons: a physical invariant, a numerical workaround, a paper citation. Skip everything else.
- **Material names follow the constitutive model** (`NeoHookeanMaterial`, `GranularFluidMaterial`), not the phenomenon (`mud`, `water`, `rock`).
- **Use `SimConfig::standard()`** for real simulations. The bare `default()` has `project_invalid_state: false`, which allows J to go negative.
- **No game logic in the engine.** Policy decisions (splitting thresholds, adhesion rules, phase boundaries) belong in the caller.
- **Core stays pure Rust.** No Bevy or game-engine dependencies in `src/`. Optional integrations go behind feature flags.
- **YAGNI.** No abstractions beyond what the current codebase needs.

---

## Physics references

Before changing numerical constants or plasticity return-mapping, check the source paper:

| Module | Reference |
|---|---|
| MLS-APIC transfer | Hu et al. 2018, *A Moving Least Squares Material Point Method* |
| NeoHookean / Corotated | Stomakhin et al. 2012, *Energetically Consistent Invertible Elasticity* |
| Snow | Stomakhin et al. 2013, *A Material Point Method for Snow Simulation* |
| Sand (DP) | Klar et al. 2016, *Drucker-Prager Elastoplasticity for Sand Animation* |
| SandMuI (µ(I)) | Dunatunga & Kamrin 2015, *Continuum modelling and simulation of granular flow* |
| GranularFluid | Dunatunga & Kamrin 2015 (Tait EOS + corotated deviatoric) |
| Surface tension | Stomakhin et al. 2014, *Augmented MPM for cloth and soft bodies* (ψ=γ·J) |
| N-body gravity | Barnes & Hut 1986, *A hierarchical O(N log N) force-calculation algorithm* |
| Viscoelastic | Fung 1993, *Biomechanics: Mechanical Properties of Living Tissues* (Kelvin-Voigt) |

---

## Running tests

```sh
cargo test                             # all tests
cargo test --test physics_correctness  # physics-specific
cargo test --test accuracy             # quantitative accuracy (slow)
cargo test --features gpu              # GPU path (requires wgpu-compatible GPU)
```

`tests/accuracy.rs` documents known numerical gaps. Read the test comments before treating a failure as a bug.

---

## Benchmarks

```sh
cargo bench --bench scaling                    # all groups
cargo bench --bench scaling -- step_scaling    # single group
```

Reports go to `target/criterion/<group>/report/index.html`.

---

## Pull requests

- One logical change per PR.
- All clippy warnings resolved.
- New material models need at least one physics test in `tests/physics_correctness.rs` (conservation law, known limit case, or regression against reference data).
- If your material has `needs_cpu_update = true`, document it in the struct's doc comment.
