# Changelog

All notable changes to this project are documented here.

## 0.1.0 — unreleased

Initial release.

### Added
- MLS-MPM solver (P2G → grid update → G2P), adaptive substeps, CFL stability.
- 12 constitutive models: NeoHookean, Corotated, NewtonianFluid, BinghamFluid,
  Stomakhin (snow), DruckerPrager (sand), MuIRheology (µ(I) granular), VonMises,
  Rankine, Viscoelastic (Kelvin-Voigt), NACC, GranularFluid.
- Force fields: NBodyGravity (Barnes-Hut), GravityWell, Coulomb, UniformEM,
  AabbConfinement, RadialConfinement, Buoyancy, Chemotaxis.
- Boundary conditions: Slip, Predictive, Friction, Heightmap.
- Thermal diffusion (Fourier FD) and generic scalar reaction-diffusion fields.
- GPU compute backend (`gpu` feature): full P2G/G2P/grid pipeline, all plasticity
  except `NaccMaterial`, which remains CPU-only.
- Instanced particle renderer (`render` feature).
- Diagnostics plugin system, per-material stats, neighbor queries.
- LNN creature locomotion controller (`src/control/lnn.rs`).
- Experimental modules (`experimental` feature): acoustics, electromagnetics,
  information-theoretic measures — not part of the guaranteed public API.

### Known limitations
- No GPU particle sleep/wake — CPU solver supports it, GPU does not yet.
- No incremental particle-add API on `GpuSimulation` — adding particles to a
  running GPU simulation requires a full readback + rebuild + reupload.
- Sand (Drucker-Prager) repose angle undershoots target by a measured margin
  (~12° vs ~35° for dry sand) — documented in `tests/accuracy.rs`, not yet tuned.
- Phase transitions (`add_phase_rule`) swap `material_id` with no latent-heat
  energy cost.
- `RankineMaterial` and `MuIRheologyMaterial` have real GPU plasticity branches
  (`particles_update.wgsl`, model==7/8) but no GPU-specific test exercises them —
  implemented, not yet verified on that path.
