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
- GPU particle sleep/wake (Phase 1 — flag-based, no compaction): resting particles
  skip G2P gather, plasticity, and force fields (P2G scatter still runs for every
  particle — see below). Opt-in via `SimConfig::sleep_threshold > 0.0` (default
  `0.0`, off). Correctness verified (`tests/gpu.rs`:
  `gpu_sleep_freezes_settled_particles`, `gpu_sleep_wakes_on_nearby_activity`, 28/28
  repeat passes across this session). Measured
  (`cargo bench --bench scaling --features gpu -- gpu_sleep_wake_scaling`, two
  independent runs): consistently 3.6-3.7x faster at 2k particles, 32-55% faster at
  20k; the 8k case is borderline (21% faster / 11% slower across the two runs) —
  that specific box geometry sits right at the edge of fully settling within the
  bench's settle budget. Real, reproducible win at the scales that settle cleanly;
  not yet proven uniformly positive at every scale.
- Fixed four real bugs surfaced while validating sleep/wake against a live example
  (not just unit tests) — each confirmed by reading the actual code, not guessed:
  1. **GPU/CPU substep-count inconsistency**: GPU's adaptive substep estimate
     scanned the *total* particle count including sleeping ones, diluting the
     velocity statistics with a frozen population — unlike CPU's `Simulation::step()`,
     which correctly scans only its active partition. Fixed: scan only awake
     particles; when none are awake AND nothing could wake them (no pending
     impulse), there is genuinely nothing to resolve finely, so this no longer pays
     for unnecessary resolution either.
  2. **Lost contact support**: an earlier version of this fix made P2G skip
     sleeping particles entirely, which made them invisible to the grid — any awake
     particle resting on a sleeping one would find no support beneath it, causing
     permanent unresolvable jitter at every awake/asleep boundary and preventing
     piles from ever fully settling. Fixed: P2G scatters mass+stress for every
     particle regardless of sleep state (deterministic for a frozen particle — same
     contribution every substep); only the gather and integration passes skip
     sleeping particles. G2P's wake-check now reads grid *velocity*, not mass,
     since mass alone no longer distinguishes real activity from a calm neighbor.
  3. **Cold-start instant sleep**: a particle spawned at rest (v=0) satisfies any
     positive `sleep_threshold` on its very first substep, before gravity has
     accelerated it at all — the same problem every real physics engine solves
     with a sleep delay (Box2D, PhysX, Bullet all require sustained low velocity,
     never an instant single-frame check). Fixed with the simulation-level
     equivalent: sleep-scoring is disabled for the first 10 frames after
     construction. Without this, an entire scene could go to sleep before ever
     truly falling, then never wake (nothing left awake to trigger the neighbor-
     activity wake check) — observed directly in `basic_sand_gpu`, which stopped
     falling entirely before this fix.
  4. **`Particles::push()` silently hardcoded `sleeping=false`**, breaking the
     GPU→CPU readback path for materials needing CPU-side plasticity (e.g.
     `NaccMaterial`): every readback reset sleeping particles back to "awake" before
     the CPU update loop ran, so it silently re-ran full plasticity on particles
     that should have been skipped. Fixed to honor the real value; the CPU update
     loop now also explicitly skips sleeping particles.
  5. **`apply_impulses.wgsl` never woke sleeping particles** — a sleeping particle
     hit by an impulse got a real velocity written but stayed `sleeping=1`, so the
     velocity sat inert (position never integrates) until it happened to wake on
     its own, then suddenly resumed motion with a stale, surprising "pop." Fixed:
     a genuine disturbance now wakes it, same as everywhere else.
  All five validated against `tmp/sparkl`'s `adaptive_timestep_length`
  (`tmp/sparkl/src/dynamics/solver/timestep_estimator.rs`) and Fang et al. 2018
  "A Temporally Adaptive Material Point Method with Regional Time Stepping" (CGF)
  — activity transitions need fine time resolution, not a globally-diluted coarse
  one; a settled region still needs to provide structural support.
- GPU force-sleep/force-wake by tag (`GpuSimulation::sleep_tag`/`wake_tag`),
  mirroring the existing CPU `Simulation` API — a minimal hook for LP's future
  chunk-loading system (force-freeze/unfreeze a tagged group of particles by
  `user_tag`, independent of velocity). Correctness verified
  (`tests/gpu.rs::gpu_sleep_tag_force_sleeps_and_wakes`). One real bug found and
  fixed during validation: a same-substep wake-undo, where `step_frame()`'s
  internal multi-substep loop re-applied the natural velocity-based sleep check
  on substeps after the one that woke the particle, undoing the wake before it
  was ever observed — fixed by exempting a wake-tagged particle from natural
  re-sleep scoring for the whole frame, not just the substep where the flag
  flipped. Known, by-design limitation: force-sleeping a particle that's still
  genuinely fast doesn't reliably stick (P2G deliberately keeps scattering a
  sleeping particle's frozen momentum for support, so the particle sees its own
  residual momentum and immediately wakes itself back up) — not a problem for the
  intended use (freezing already-calm distant terrain), but a real gap outside it.
- Instanced particle renderer (`render` feature).
- Diagnostics plugin system, per-material stats, neighbor queries.
- LNN creature locomotion controller (`src/control/lnn.rs`).
- Experimental modules (`experimental` feature): acoustics, electromagnetics,
  information-theoretic measures — not part of the guaranteed public API.
- `DruckerPragerMaterial::cohesion` (default `0.0`, zero behavior change): a
  calibrated, non-pressure-dependent resistance floor for sand. Pressure-
  proportional friction alone vanishes in thin, fast-flowing granular layers
  regardless of the friction coefficient — confirmed by three independently
  configured friction models (plain DP, rate-dependent, q-gated) all producing
  identical excess runout. `cohesion = 5.0` cuts the Lajeunesse et al. 2004
  column-collapse runout ratio from 4.7x to 1.5x the empirical prediction
  (`tests/accuracy.rs::sand_column_collapse_runout_matches_lajeunesse_scaling`,
  permanent regression test) and, with the same value never re-tuned for this
  second scenario, improves a continuous-pour repose angle from 13° to ~19-22°.
- `ParticleMass` trait + `SpawnRegion::mass_from(&props, &config)` builder:
  computes `mass_override` from a physical-property struct (`Elastic`,
  `Elastoplastic`, `Viscoelastic`, `FluidGranular`, `Fluid`) using the region's
  own `spacing`, instead of requiring the caller to pass spacing twice (a real
  duplication bug class — already bit LP once). Purely additive; existing
  direct `.particle_mass(spacing, &config)` calls are unaffected.
- `MaterialModel::latent_heat()` (default `0.0`, CPU-only): `phase_transition`
  and `add_phase_rule` now debit `temperature -= latent_heat / heat_capacity`
  on transition when a thermal model is configured, closing the energy-
  conservation gap noted below. No-op without `with_thermal`/`set_thermal`, so
  existing phase-rule users see no behavior change
  (`tests/solver.rs::phase_transition_applies_latent_heat_energy_debit`,
  `phase_transition_skips_latent_heat_without_thermal_model`).
- `examples/stress_cfl_scan_50k.rs`: a live, vsync-paced example at the actual
  ~50k-particle / grid_res=320 LP target. Demonstrated a real, sustained
  60-66fps over 22,000+ frames after disabling an unused `readback_stride` for
  pure-rendering use and relaxing `material_cfl_coefficient` from the default
  `0.5` to `0.7` for that scene specifically (still inside the literature's
  normal 0.3-1.0 range) — dropped Drucker-Prager sand from 3 substeps to 2.
  Correctness re-verified explicitly under the relaxed coefficient
  (`tests/gpu.rs::gpu_relaxed_cfl_coefficient_stays_correct_50k_dpsand`).
  `SimConfig::material_cfl_coefficient`'s own default is unchanged at `0.5`.

### Known limitations
- GPU sleep/wake doesn't shrink the actual dispatch size (same thread count every
  substep), and P2G now scatters for every particle regardless of sleep state
  (required for sleeping particles to keep providing structural support to
  neighbors — see above) — the win comes from fewer substeps overall plus skipping
  G2P/plasticity/force-field work per sleeping thread, not from launching fewer
  threads or a cheaper P2G. A real, measured win at scales that fully settle; full
  particle compaction + indirect dispatch (Phase 2) would be the next lever if a
  bigger win is needed.
- No incremental particle-add API on `GpuSimulation` — adding particles to a
  running GPU simulation requires a full readback + rebuild + reupload.
- Sand (Drucker-Prager) repose angle still undershoots target after the
  `cohesion` fix above (~19-22° vs ~30-35° for dry sand) — three independently
  configured gating mechanisms all converge to the same ceiling, treated as
  evidence the remaining gap needs a structurally different fix (e.g. a
  properly-rotated geostatic initialization), not further tuning. Documented
  in `tests/accuracy.rs`.
- Real GPU compute budget at LP's actual ~100-150k particle scene target is
  still 2.4-4.6x over a 60fps frame budget
  (`tests/gpu.rs::gpu_particle_count_lp_budget_0_1_0_scene`); only the ~50k
  point is verified live at 60+fps (see `stress_cfl_scan_50k.rs` above).
- `add_phase_rule`'s rules accumulate plastic-friction state (e.g. Drucker-
  Prager's `q`) that, by design (critical-state soil mechanics), rarely
  reaches exactly zero even once a pile looks settled — GPU sleep/wake
  therefore rarely engages on granular terrain specifically. Not a defect;
  open as a possible reason to use a different sleep heuristic for granular
  materials (e.g. thresholding the rate of change of `q` rather than raw
  velocity).
