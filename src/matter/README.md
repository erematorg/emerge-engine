# Matter

**What things are made of.**

Every bit of simulated matter is a particle carrying its own mass, temperature,
and how much it has been squashed or stretched. What makes water behave like
water and sand like sand is one thing only: the equation that says how hard it
pushes back when you deform it. That equation is the "material model", and
there are seventeen of them here, each taken from a published paper and saying
so in the code.

Also here: what each substance does to light and heat -- its measured
absorption spectrum, its heat capacity. Those are properties of the substance,
so they live with the substance, never in the renderer or the thermal solver.

In the language of the field: per-particle state and the constitutive models
that turn strain into stress.

## Core API

- `Particle` (`particle/mod.rs`) -- 128 bytes, `repr(C)`, GPU-uploadable via `bytemuck::Pod`. Position, velocity, APIC velocity gradient, deformation gradient, mass/volume/density, material ID, plasticity state (`plastic_volume_ratio`, `hardening_scale`, `friction_hardening`, `log_volume_strain`), temperature, `user_tag`, activation + muscle fields, `contact_group`, `sleeping`, `pinned`, `scalar_field`, `internal_pressure`.
- `Particles` (`particle/soa.rs`) -- SoA container; the long-term storage form `Simulation` operates on.
- `MaterialModel` trait (`materials/`) -- `kirchhoff_stress` + `update_particle`, one impl per constitutive law. A substance also declares what it is beyond mechanics: `optical_properties()` (measured absorption and scattering, `m^-1`), `specific_heat_j_kg_k()`, `luminous_emission_w_m3()`. Each defaults to "undeclared", so a material carries only what is actually known about it.
- `optical.rs` -- measured optical constants of real substances, with the band averaging that collapses a spectrum onto three channels. Pure water's absorption spectrum (Pope & Fry 1997, 140 points, 380-727 nm).
- 17 material models: `NeoHookeanMaterial`, `CorotatedMaterial`, `NewtonianFluidMaterial`, `BinghamFluidMaterial`, `StomakhinMaterial` (snow), `DruckerPragerMaterial` (sand), `MuIRheologyMaterial` (µ(I) dense granular flow), `VonMisesMaterial`, `RankineMaterial`, `ViscoelasticMaterial`, `NaccMaterial` (Cam-Clay), `GranularFluidMaterial`, `NoCompressionMaterial`. See root `README.md`'s materials table for real citations and preset constructors.
- `grain_contact_law` -- Cundall & Strack 1979 / Luding 2008 / Ai et al. 2011 discrete-element contact force law (normal/tangential/rolling spring-dashpot, Coulomb + rolling-friction caps). Not a `MaterialModel` -- a portable, self-contained function of local `GrainContactState` pairs, consumed by `spacetime::grains`.
- `cosserat` -- micropolar (Cosserat) grain kinematics + constitutive relation, the root-cause fix candidate for sand's self-arrest/repose-angle gap.
- `MaterialRegistry`, `MaterialParams` (112-byte GPU-uploadable union, carrying each substance's own `specific_heat_j_kg_k` alongside its mechanical parameters), `registry.rs` -- material-slot dispatch shared by CPU and GPU paths.

## Scope & Limits

- 2D only, throughout.
- Constitutive models bundle kinematics (strain reconstruction), kinetics (stress), and the material-specific law together per particle -- matches how production MPM codebases structure this (kinematics/kinetics/constitutive-modeling are textbook-distinct in continuum mechanics, but tightly coupled per substep in practice).
- `grain_contact_law` and `cosserat` are portable/self-contained (no dependency on solver-side state) -- that's why they live here rather than in `spacetime`, unlike e.g. `RodMaterial`, which reaches directly into `spacetime::rod`'s own `RodPoints` and stays there.

## Status

Production-stable core. All 17 material categories have at least one real dynamic test (see `matter/materials/*_tests.rs` and `tests/accuracy.rs`). `GranularFluidMaterial::consolidated_clay` preset has a documented, unresolved residual instability (5-10x improved, not fully settled) -- unused in shipped scenes.
