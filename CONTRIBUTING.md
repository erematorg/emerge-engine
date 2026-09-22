# Contributing to emerge

emerge is a 2D MLS-MPM continuum physics engine in pure Rust, built for real-time simulation of fluids, granular materials, elastic solids, and active matter.

---

## Issues

Bug reports and questions are welcome via GitHub issues. This is a solo-maintained project -- response time is best-effort, not guaranteed. Include a minimal repro (a failing test or example is ideal) where possible; it's the fastest path to a fix.

---

## Quick start

```sh
cargo check                        # verify it compiles
cargo test                         # run all unit + integration tests
cargo clippy -- -D warnings        # must be clean before any PR
cargo run --example headless       # smoke test, no feature flags needed
```

---

## Architecture

Domain-taxonomy layout -- each top-level directory mirrors one universe-scale domain,
not an implementation layer:

```
src/
  matter/            particle/ (Particle, repr(C) 128 B GPU-uploadable · Grain ·
                     RodPoints · Particles SoA)
    materials/        MaterialModel trait · registry · 14 material models ·
                      granular/ (sand, sand_mui, cosserat, grain_contact_law,
                      scale_contract -- grouped by active research thread)
  spacetime/          the actual solver
    solver/            Simulation · SimConfig · SpawnRegion · spatial hash ·
                       body_state (BodyState aggregation)
    grid/               Grid · Cell · ContactCell (multi-field contact) · kernel
    transfer/           P2G scatter + G2P gather (MLS-APIC)
    diff.rs             differentiable/gradient-trainable stepping
    rod/                Rod · RodMaterial · build_straight_rod ·
                        coupling.rs (scatter/gather to the shared Grid)
    grains/             DEM grain dynamics: population/coupling/oracle
                        (state lives in matter::particle::Grain)
  forces/             boundary/ (Slip / Heightmap / friction/
                      [Friction / GripFriction / RatchetFriction]) ·
                      fields/ (NBody / GravityWell / Coulomb / Confinement / cutoff) ·
                      electromagnetics.rs
  energy/             thermodynamics/ (ThermalDiffusion · ScalarDiffusionField) ·
                      acoustics/, electromagnetics.rs [feature=experimental]
  information/        control/ (Lnn neural locomotion controller) · measures/
  runtime/            FixedStepController
  systems/            gpu/ [feature=gpu] GpuSimulation + WGSL shaders ·
                      render/ [feature=render] instanced particle renderer ·
                      diagnostics/ plugin system · health · per-material stats
```

Every domain root is a `mod.rs` inside its own folder (`energy/mod.rs`, not a
sibling `energy.rs`) -- keeps `lib.rs` the only file at the true top level,
each domain visually self-contained in its own directory rather than spread
across loose sibling files.

Feature flags: `gpu` | `render` (requires `gpu`) | `experimental`

---

## Adding a material model

A new material requires changes in four places:

### 1. `src/matter/materials/<name>.rs`

Implement the `MaterialModel` trait:

All methods have default implementations (an elastic-only material can override just
`kirchhoff_stress`). The signatures below are exact, copied directly from the trait's
own current definition (`src/matter/materials/mod.rs`) -- copy them, not the idea of them,
and re-check against the trait itself before relying on this doc, since it's the kind
of thing that silently drifts:

```rust
pub struct MyMaterial { /* parameters */ }

impl MaterialModel for MyMaterial {
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 { ... }
    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 { ... }
    // Disjoint per-field borrows (`ParticleUpdateCtx`), not `&mut Particles, i` --
    // this shape is what lets G2P run every particle's update in parallel.
    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) { ... }
    // Seeds per-particle plastic state at spawn time -- takes a single `Particle`,
    // not the `Particles` collection (called once per particle, before it's in the SoA).
    fn init_particle(&self, particle: &mut Particle) { ... }
    // CFL bound -- plain scalars (not &Particles, i), so both the CPU (SoA) and
    // GPU (AoS) CFL scans can call it without either needing the other's layout.
    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 { ... }
    fn needs_cpu_update(&self) -> bool { false }
    // `Some(reason)` makes `GpuSimulation` refuse this material.
    fn gpu_unsupported_reason(&self) -> Option<&'static str> { None }
}
```

### 2. `src/matter/materials/mod.rs`

Add a variant to `ConstitutiveModel`. The discriminant must be the next consecutive `u32`, and a matching compile-time ABI assertion is required:

```rust
#[repr(u32)]
pub enum ConstitutiveModel {
    // ... existing variants ...
    MyMaterial = 14,  // next available discriminant (14 material models exist today)
}

// in the assert block:
assert!(ConstitutiveModel::MyMaterial as u32 == 14);
```

Re-export from `mod.rs` and add to `src/prelude.rs`.

### 3. `src/systems/gpu/shaders/p2g.wgsl`

Add `case 14u` to the Kirchhoff stress `switch`. If the material has no GPU law yet, leave the shaders alone and override `gpu_unsupported_reason` in Rust instead: `GpuSimulation` then refuses to start with it rather than running another law in its place.

### 4. `src/systems/gpu/shaders/particles_update.wgsl`

Add `case 14u` to the plasticity update `switch`. CPU-only materials can leave this as a no-op.

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
| Rankine | Rankine 1876 (original criterion); Wolper et al. 2019 (MPM brittle fracture) |
| NACC | Klar et al. 2016; sparkl `plasticity_nacc.rs` |
| Surface tension | Stomakhin et al. 2014, *Augmented MPM for cloth and soft bodies* (ψ=γ·J) |
| N-body gravity | Barnes & Hut 1986, *A hierarchical O(N log N) force-calculation algorithm* |
| Viscoelastic | Fung 1993, *Biomechanics: Mechanical Properties of Living Tissues* (Kelvin-Voigt) |
| Rod (`spacetime::rod`) | Bergou, Wardetzky, Robinson, Audoly, Grinspun 2008, *Discrete Elastic Rods* (SIGGRAPH); modern discrete form of Cosserat rod theory (Cosserat brothers, 1909) |

---

## Running tests

```sh
cargo test                             # all tests
cargo test --test physics_correctness  # physics-specific
cargo test --test accuracy             # quantitative accuracy (slow)
cargo test --features gpu              # GPU path (requires wgpu-compatible GPU)
```

`tests/accuracy.rs` documents known numerical gaps. Read the test comments before treating a failure as a bug.

**`--all-features --tests` (or `cargo test --all-features`) can crash the linker**
(`STATUS_STACK_BUFFER_OVERRUN` from `rustc.exe`/the linker, not an emerge bug) on
machines with limited memory -- building every integration test binary with
`gpu`+`render`+`experimental` all at once, fully parallel, exhausts a real resource.
Confirmed fix: cap build parallelism --

```sh
cargo test --all-features -j 2
```

`cargo test --features gpu` alone (no `render`/`experimental`) is unaffected; only
the full `--all-features` combination needs the `-j 2` cap.

### Tests the CI does not run

**GPU tests need real hardware.** Software adapters (lavapipe, D3D12 WARP) give
verdicts that differ from real GPUs, so CI does not judge the GPU path. Run these
by hand on a machine with a real GPU:

```sh
cargo test --profile quick --features gpu --test gpu -- --test-threads=1
cargo test --profile quick --features render --lib -- --ignored systems::render::tests systems::gpu::solver::device_lost_tests
cargo test --profile quick --features gpu --test solver -- --ignored gpu
```

**Slow tests run on demand.** Long-horizon correctness tests that would push a CI
shard past its 45-minute limit in the debug profile are marked
`#[ignore = "slow: ..."]`. The `slow tests` workflow (Actions tab, "Run workflow")
runs them in the quick profile; the exact list lives in
`.github/workflows/slow-tests.yml` and can be run locally the same way.

**Diagnostic probes** (`tests/scratch_*.rs`) are kept for reruns and are all
ignored; run one with `--ignored --nocapture` and its name.

---

## Benchmarks

```sh
cargo bench --bench scaling                    # all groups
cargo bench --bench scaling -- step_scaling    # single group
```

Reports go to `target/criterion/<group>/report/index.html`.

---

## Commits

The history is part of what this repository publishes: someone reading
`git log --oneline` should understand what happened without opening a diff.

**Subject** -- `type(scope): imperative summary`, following
[Conventional Commits 1.0.0](https://www.conventionalcommits.org/en/v1.0.0/):

- At most **7 words after the prefix** (`type(scope):` itself does not count).
- Whole line near 50 characters, imperative mood, no trailing period.
- Types: `feat`, `fix`, `perf`, `refactor`, `test`, `docs`, `chore`, `ci`, `style`.
- Scopes are areas: `gpu`, `render`, `engine`, `materials`, `solver`, `grains`, `diag`.

**Body** -- write one only when the commit carries what a diff cannot show: a
root cause, a measurement, or an alternative that was tried and rejected. A
one-line fix needs none. Wrap at 72 columns (Git's own convention, Pro Git
ch. 5.2).

- Put real numbers in, with what they were measured on: `336 -> 72us, dam
  break, 2912 particles`, not "significantly faster".
- Cite the source for anything taken from the literature.
- Record what was tried and did NOT work when it cost real time; that is what
  stops the next person repeating it.
- Short bullets over prose paragraphs.

Avoid restating the diff, self-justification, filler adjectives ("robust",
"comprehensive"), em-dashes, and `Co-Authored-By` trailers.

**Splitting** -- one commit per logical unit, including under deadline
pressure: a root-cause fix, a performance pass and a new feature are three
commits. When one file legitimately carries two of them, say so in the body
rather than merging the commits.

## Pull requests

- One logical change per PR.
- All clippy warnings resolved.
- New material models need at least one physics test in `tests/physics_correctness.rs` (conservation law, known limit case, or regression against reference data).
- If your material has `needs_cpu_update = true`, document it in the struct's doc comment.
