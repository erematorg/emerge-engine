# Matter

What things are made of: per-particle state and the constitutive models that turn strain into stress.

## Core API

- `Particle` (`particle/mod.rs`) — 128 bytes, `repr(C)`, GPU-uploadable via `bytemuck::Pod`. Position, velocity, APIC velocity gradient, deformation gradient, mass/volume/density, material ID, plasticity state (`plastic_volume_ratio`, `hardening_scale`, `friction_hardening`, `log_volume_strain`), temperature, `user_tag`, activation + muscle fields, `contact_group`, `sleeping`, `pinned`, `scalar_field`, `internal_pressure`.
- `Particles` (`particle/soa.rs`) — SoA container; the long-term storage form `Simulation` operates on.
- `MaterialModel` trait (`materials/`) — `kirchhoff_stress` + `update_particle`, one impl per constitutive law.
- 13 material models: `NeoHookeanMaterial`, `CorotatedMaterial`, `NewtonianFluidMaterial`, `BinghamFluidMaterial`, `StomakhinMaterial` (snow), `DruckerPragerMaterial` (sand), `MuIRheologyMaterial` (µ(I) dense granular flow), `VonMisesMaterial`, `RankineMaterial`, `ViscoelasticMaterial`, `NaccMaterial` (Cam-Clay), `GranularFluidMaterial`, `NoCompressionMaterial`. See root `README.md`'s materials table for real citations and preset constructors.
- `grain_contact_law` — Cundall & Strack 1979 / Luding 2008 / Ai et al. 2011 discrete-element contact force law (normal/tangential/rolling spring-dashpot, Coulomb + rolling-friction caps). Not a `MaterialModel` — a portable, self-contained function of local `GrainContactState` pairs, consumed by `spacetime::grains`.
- `cosserat` — micropolar (Cosserat) grain kinematics + constitutive relation, the root-cause fix candidate for sand's self-arrest/repose-angle gap.
- `MaterialRegistry`, `MaterialParams` (96-byte GPU-uploadable union), `registry.rs` — material-slot dispatch shared by CPU and GPU paths.

## Scope & Limits

- 2D only, throughout.
- Constitutive models bundle kinematics (strain reconstruction), kinetics (stress), and the material-specific law together per particle — matches how production MPM codebases structure this (kinematics/kinetics/constitutive-modeling are textbook-distinct in continuum mechanics, but tightly coupled per substep in practice).
- `grain_contact_law` and `cosserat` are portable/self-contained (no dependency on solver-side state) — that's why they live here rather than in `spacetime`, unlike e.g. `RodMaterial`, which reaches directly into `spacetime::rod`'s own `RodPoints` and stays there.

## Status

Production-stable core. All 13 material categories have at least one real dynamic test (see `matter/materials/*_tests.rs` and `tests/accuracy.rs`). `GranularFluidMaterial::consolidated_clay` preset has a documented, unresolved residual instability (5-10x improved, not fully settled) — unused in shipped scenes.
