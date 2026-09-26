//! Adaptive-timestep (CFL) selection -- split out of `step.rs` (was ~85 of that
//! file's ~730 lines). A distinct concern from the substep pipeline itself:
//! picking how big a step is safe, not advancing the simulation by one.
//! `choose_substep_dt`/`cfl_bound`/`affine_cfl_speed_contribution` are reused
//! by the GPU solver's own CFL scan (`systems::gpu::solver::step`), which is
//! why the latter two stay `pub(crate)` and re-exported from `solver/mod.rs`.

#[cfg(feature = "gpu")]
use glam::Mat2;
use glam::Vec2;
use rayon::prelude::*;

use super::{MaterialRegistry, SimConfig};
use crate::grains::population::GrainPopulation;
use crate::particle::Particles;
use crate::rod::{Rod, rod_cfl_dt};

// choose_substep_dt: picks the largest CFL-safe dt ≤ max_dt.
// Called inside step()'s substep loop -- max_dt is the remaining frame time.
// pub(crate) so the GPU solver can reuse this without duplicating CFL logic.
/// The simulated bodies this CFL scan reads. Bundled with `SubstepBounds`
/// below so `choose_substep_dt` needs no
/// `#[allow(clippy::too_many_arguments)]` -- fixing the cause (ten loose
/// parameters that always travel together from the same `Simulation`)
/// rather than silencing the lint, same pattern as `G2PParams`/
/// `ProjectInputs` elsewhere in this codebase. Pure regrouping: every field
/// is passed through unchanged, no logic or numeric value is altered.
pub(crate) struct SubstepScene<'a> {
    pub particles: &'a Particles,
    pub active_count: usize,
    pub materials: &'a MaterialRegistry,
    pub rods: &'a [Rod],
    pub grain_populations: &'a [GrainPopulation],
}

/// The externally-supplied timestep bounds and lagged measurements this CFL
/// scan folds in alongside the bodies' own bounds. See `SubstepScene`.
pub(crate) struct SubstepBounds {
    /// Remaining frame time -- the hard upper bound on the returned dt.
    pub max_dt: f32,
    pub granular_fluidity_dt_bound: Option<f32>,
    pub thermal_dt_bound: Option<f32>,
    // Real max particle speed from the PREVIOUS call to this function
    // (one-substep-lagged -- see `Simulation::last_max_particle_speed`'s own
    // doc). Used ONLY by the near-wall gate's Mach-relative compression
    // threshold (`SimConfig::fluid_near_wall_compression_mach_margin`) --
    // THIS call's own max_speed isn't known yet at the point the gate needs
    // it (it's still being folded), so the previous substep's value is the
    // freshest real data available, same "react at the next sync point"
    // pattern this codebase's GPU batch CFL scan already uses.
    pub last_max_speed: f32,
}

pub(crate) fn choose_substep_dt(
    config: &SimConfig,
    scene: SubstepScene<'_>,
    bounds: SubstepBounds,
) -> (f32, f32) {
    let SubstepScene {
        particles,
        active_count,
        materials,
        rods,
        grain_populations,
    } = scene;
    let SubstepBounds {
        max_dt,
        granular_fluidity_dt_bound,
        thermal_dt_bound,
        last_max_speed,
    } = bounds;
    if !config.adaptive_timestep {
        return (max_dt.min(config.dt), 0.0);
    }
    // Single pass for both velocity CFL and material timestep bound.
    // Parallelized (2026-08-07, real measured win: this scan was ~8ms/frame
    // running on a SINGLE core, the other 7 idle) -- pure `(f32, f32)`
    // fold/reduce, no allocation per chunk at all, so none of the same-night
    // dense-buffer failure mode applies (see `transfer::p2g::
    // scatter_particles_to_grid`'s own doc for that story). `with_min_len`
    // matches P2G/G2P's own real fix, same reasoning: rayon's default
    // chunking splits far more/smaller tasks than a naive one-per-core
    // assumption.
    let min_len = (active_count / (rayon::current_num_threads() * 2)).max(1);
    let (max_speed, min_mat_dt, near_wall_gravity_scale) = (0..active_count)
        .into_par_iter()
        .with_min_len(min_len)
        .fold(
            || (0.0f32, max_dt, 1.0f32),
            |(mut max_speed, mut min_mat_dt, mut near_wall_scale), i| {
                // Real, small (2026-08-14) solver-core tightening: this loop
                // body used to call `affine_cfl_speed_contribution` and
                // `deformation_gradient_cfl_bound` separately below, each
                // independently recomputing the SAME Frobenius norm of
                // `velocity_gradient[i]` -- two sqrt where one suffices.
                // Computed once here and reused by both; bit-identical
                // result (same formula, same operands), not an approximation.
                // Left the two `pub(crate)` functions themselves untouched
                // (their own doc: also called from the GPU CFL scan's CPU
                // side) so this stays a local, self-contained change with no
                // GPU-path blast radius.
                let grad_norm = (particles.velocity_gradient[i].x_axis.length_squared()
                    + particles.velocity_gradient[i].y_axis.length_squared())
                .sqrt();

                let mut s = particles.v[i].length();
                if config.cfl_include_affine_speed {
                    s += grad_norm * AFFINE_CFL_STENCIL_CORNER_DISTANCE * config.grid_cell_size;
                }
                max_speed = max_speed.max(s);
                // Proactive near-wall tightening for strict fluids (see
                // `SimConfig::fluid_near_wall_cfl_scale`'s own doc) -- `1.0`
                // (default) makes this branch's division a no-op, so every
                // scene that never opts in pays nothing extra beyond the
                // branch check itself. Gated on ACTUAL compression too, but
                // (2026-08-11) relative to THIS material's own acoustic
                // stiffness when it has one, not a fixed absolute percentage
                // -- see `fluid_near_wall_compression_mach_margin`'s own doc
                // for the real WCSPH-literature grounding (Ma²≈Δρ) and why an
                // absolute threshold self-defeats for a deliberately
                // softened EOS. Falls back to the old absolute
                // `fluid_near_wall_compression_threshold` for a material with
                // no acoustic term at all (e.g. `eos_stiffness=0.0`
                // pressure-projection fluids), unchanged from before.
                let material_cfl = if config.fluid_near_wall_cfl_scale != 1.0
                    && materials.owns_deformation_volume_state(particles.material_id[i])
                    && is_near_wall(particles.x[i], config.grid_res, config.boundary_thickness)
                    && {
                        let j = particles.volume[i] / particles.initial_volume[i];
                        let threshold = match materials.rest_acoustic_c2(particles.material_id[i]) {
                            Some(c2_rest) if c2_rest > f32::EPSILON => {
                                let mach = last_max_speed / c2_rest.sqrt();
                                (mach * mach) * config.fluid_near_wall_compression_mach_margin
                            }
                            _ => config.fluid_near_wall_compression_threshold,
                        };
                        (j - 1.0).abs() > threshold
                    } {
                    config.material_cfl_coefficient / config.fluid_near_wall_cfl_scale
                } else {
                    config.material_cfl_coefficient
                };
                let mdt = materials.timestep_bound(
                    particles.material_id[i],
                    particles.density[i],
                    particles.hardening_scale[i],
                    config.grid_cell_size,
                    material_cfl,
                    config.viscous_timestep_coefficient,
                );
                if mdt.is_finite() && mdt > 0.0 {
                    min_mat_dt = min_mat_dt.min(mdt);
                }
                // Real, standard hydrocode stability correction for von
                // Neumann-Richtmyer artificial (shock) viscosity (Wilkins 1980,
                // "Calculation of Elastic-Plastic Flow," Methods in
                // Computational Physics; Benson 1992, "Computational methods in
                // Lagrangian and Eulerian hydrocodes," Comput. Methods Appl.
                // Mech. Engrg. 99) -- found live 2026-08-28 debugging a real,
                // severe steam fps collapse (`phase_states_gui.rs`): both
                // `IdealGasMaterial` and `NewtonianFluidMaterial` feed
                // `von_neumann_richtmyer_q`'s real, dimensionally-correct
                // (fixed 2026-08-13, see that fix's own comment for the
                // identical "J pinned at clamp, fps collapses" signature this
                // closes the rest of) quadratic term
                // (`c0_quadratic*h^2*div_v^2`) -- real added numerical
                // stiffness that GROWS with the current compression rate. But
                // `timestep_bound`'s own trait signature only carries
                // density/hardening/cell_width, never live `div_v`, so this
                // term was completely invisible to the CFL scan: nothing ever
                // shrank dt in response to it, live-measured
                // `|trace(velocity_gradient)|` climbing past 5000/s with no
                // corresponding tightening. Standard practice augments the
                // acoustic sound speed with the viscosity's own contribution,
                // `c_eff = c_sound + 2*c0_quadratic*h*|div_v|`, then bounds dt
                // the same way the plain acoustic term already does.
                // `grad_norm` (Frobenius norm of the velocity gradient, already
                // computed above for the deformation-rate bound) is a real,
                // conservative proxy for `|div_v|` -- it upper-bounds any
                // single directional derivative including the trace/
                // divergence, so this errs toward MORE caution, never less.
                // Gated to strict-fluid materials with a real acoustic term
                // (`owns_deformation_volume_state` + `rest_acoustic_c2`) --
                // the same real population `von_neumann_richtmyer_q` is ever
                // invoked for; an elastic solid's own (shear-based) acoustic
                // term has nothing to do with this mechanism and must not be
                // tightened by it. `eos_power` doubles as
                // `von_neumann_richtmyer_q`'s own `weak_shock_gamma` argument
                // for BOTH materials that call it (confirmed: gas.rs passes
                // `adiabatic_index`, fluid.rs passes its own Tait `eos_power`
                // -- same convention, same field in `MaterialParams`).
                if let Some(shock_dt) = shock_viscosity_dt_bound(
                    grad_norm,
                    materials.owns_deformation_volume_state(particles.material_id[i]),
                    materials.rest_acoustic_c2(particles.material_id[i]),
                    materials.get(particles.material_id[i]).params().eos_power,
                    config.grid_cell_size,
                    material_cfl,
                ) {
                    min_mat_dt = min_mat_dt.min(shock_dt);
                }
                // Real, derived "single-particle instability" bound for strict-fluid
                // materials (Sun, Shinar & Schroeder 2020, "Effective time step
                // restrictions for explicit MPM simulation," SCA 2020, Section 4.5)
                // -- found live 2026-08-28 chasing the still-open steam divergence
                // (see project memory): when a particle becomes isolated (few or no
                // neighbors sharing its local grid nodes -- exactly what a rising,
                // buoyancy-driven steam particle does as it spreads into
                // previously-empty upper cells), the grid velocity there is driven
                // by that ONE particle's own pressure force, which feeds back into
                // its own next-substep J -- a real fixed-point iteration on J that
                // can diverge if the timestep doesn't respect it. The paper derives
                // this bound assuming the WORST case (`tr(H)` at its own proven
                // upper bound `K*d/dx^2`), so it's a valid universal restriction for
                // every strict-fluid particle, not conditional on detecting
                // isolation directly -- and it's the authors' own relaxed form
                // (their stricter Eq. 6 forces J<=1 outright; this one instead
                // allows overshoot but bounds it from diverging, "relaxed by up to
                // a factor of 2 near Jp~=1," and is the version they report using
                // for their own fluid results). `K=6` is the paper's own derived
                // constant for quadratic B-splines (this engine's own kernel,
                // `spacetime::grid::kernel::quadratic_weights`); `d=2` for this 2D
                // engine. Continuous at J=1 from both sides by construction
                // (verified by hand: both branches evaluate to
                // `dx*sqrt(2*rest_density/(K*d))` there), a real internal-
                // consistency check on the derivation, not just trust in the source.
                // Real constitutive stiffness lambda = rho0*c0^2 (the Tait/
                // ideal-gas EOS tangent bulk modulus at rest, J=1) -- REQUIRED
                // by Sun, Shinar & Schroeder's own derivation (their own
                // `lambda` term); see `single_particle_instability_dt_bound`'s
                // own doc for the full citation. Honest, disclosed limitation:
                // this uses the REST-state tangent stiffness (correct near
                // J=1), not a full nonlinear-Tait worst-case bound over the
                // whole admissible J range.
                if materials.owns_deformation_volume_state(particles.material_id[i]) {
                    let rest_density = materials
                        .get(particles.material_id[i])
                        .params()
                        .rest_density;
                    let j = particles.volume[i] / particles.initial_volume[i];
                    if let Some(single_particle_dt) = single_particle_instability_dt_bound(
                        true,
                        rest_density,
                        j,
                        materials.rest_acoustic_c2(particles.material_id[i]),
                        config.grid_cell_size,
                    ) {
                        min_mat_dt = min_mat_dt.min(single_particle_dt);
                    }
                }
                // Real, live density-AND-temperature-aware acoustic term
                // (found live via a direct A/B on `phase_states_gui.rs`'s
                // own sustained-heating steam scene: divergence escalating
                // into the thousands, `last_substeps` climbing toward its
                // own cap, fps collapsing) -- see `MaterialModel::
                // acoustic_c2_at`'s own doc for the real, previously-
                // disclosed-but-unclosed gap: `timestep_bound` alone can
                // only ever see a material's fixed, construction-time
                // acoustic stiffness, never a particle's own LIVE state --
                // which climbs continuously under real active heating
                // (`IdealGasMaterial`'s own `c^2` linear in `T`), or shifts
                // with BOTH density and temperature jointly (a real
                // temperature-coupled cavitation EOS's own mixture band and
                // C^1 patches, which `T` alone cannot resolve -- external
                // review's own explicit point). Same established pattern as
                // the shock-viscosity/single-particle-instability terms
                // above: a real, separate CFL term, not a `timestep_bound`
                // signature change (every other material's own
                // `acoustic_c2_at` defaults down through `acoustic_c2_at_
                // temperature`/`rest_acoustic_c2`, so this is a no-op for
                // anything that doesn't override one of those). Calls the
                // most general tier (`acoustic_c2_at_particle`, not
                // `acoustic_c2_at` directly) so a material needing a real
                // per-particle scalar beyond density/temperature (e.g.
                // `BoilingMixtureMaterial`'s own mass quality) is reachable
                // too -- every other material's own default just forwards
                // straight through to `acoustic_c2_at`, unchanged.
                if let Some(c2_live) =
                    materials.acoustic_c2_at_particle(particles.material_id[i], particles, i)
                    && c2_live.is_finite()
                    && c2_live > f32::EPSILON
                {
                    let live_temp_dt = material_cfl * config.grid_cell_size / c2_live.sqrt();
                    if live_temp_dt.is_finite() && live_temp_dt > 0.0 {
                        min_mat_dt = min_mat_dt.min(live_temp_dt);
                    }
                }
                // The deformation update is a local ODE in its own right.  Bound the
                // dimensionless velocity-gradient increment even when affine velocity
                // contribution is disabled for the advection CFL; otherwise an Euler
                // F update can invert within a nominally velocity-safe substep.
                // Reuses `grad_norm` computed above instead of calling
                // `deformation_gradient_cfl_bound` (which would recompute the
                // identical norm) -- same formula as that function's own body,
                // bit-identical result.
                let deformation_dt =
                    deformation_gradient_ode_dt_bound(grad_norm, config.cfl_coefficient);
                if deformation_dt.is_finite() && deformation_dt > 0.0 {
                    min_mat_dt = min_mat_dt.min(deformation_dt);
                }
                // Real, PREDICTIVE (not reactive) near-wall tightening for a
                // strict fluid with `eos_stiffness=0` -- see MEMORY.md's
                // fluid-recovery notes, Round 9, for why this is needed:
                // `fluid_near_wall_cfl_scale`'s ORIGINAL tightening (above,
                // dividing `material_cfl`) only affects the ACOUSTIC bound
                // (`c2` in `NewtonianFluidMaterial::timestep_bound`), which
                // is IDENTICALLY ZERO once `eos_stiffness=0` -- confirmed
                // live, bit-for-bit identical results at scale=5 vs scale=20
                // vs disabled entirely, since the lever has nothing left to
                // act on. The gravity-CFL bound below (module-level, folded
                // in after this loop) is the one bound that's actually still
                // ACTIVE and PREDICTIVE for an eos-less fluid at rest -- so
                // THIS is the real lever to tighten, not the dead acoustic
                // one. No compression gate here on purpose (unlike the
                // acoustic version above): a compression-based gate is
                // reactive by definition (needs `J` to have ALREADY drifted
                // away from 1 to fire), which is exactly what fails at the
                // critical first substep, before anything has moved yet.
                if config.fluid_near_wall_cfl_scale != 1.0
                    && materials.owns_deformation_volume_state(particles.material_id[i])
                    && is_near_wall(particles.x[i], config.grid_res, config.boundary_thickness)
                {
                    near_wall_scale = near_wall_scale.max(config.fluid_near_wall_cfl_scale);
                }
                (max_speed, min_mat_dt, near_wall_scale)
            },
        )
        .reduce(
            || (0.0f32, max_dt, 1.0f32),
            |(ms1, md1, nw1), (ms2, md2, nw2)| (ms1.max(ms2), md1.min(md2), nw1.max(nw2)),
        );
    let mut min_mat_dt = min_mat_dt;
    let mut max_speed = max_speed;
    // Grains aren't scanned by the particle loop above (separate storage,
    // same reason rods below aren't either) -- fold in their own advection
    // speed the same way, so a spinning grain can never silently escape the
    // adaptive substep logic. Real, found-not-guessed need (2026-08-20):
    // `scatter_grains_to_grid` now scatters a grain's TRUE rigid-body
    // rotational velocity field (`v_com + spin*perp(r)`, see that
    // function's own doc), which can put a FAR larger velocity on the grid
    // than the grain's own `v.length()` once `spin` is large -- exactly the
    // same real effect ordinary particles' own affine `velocity_gradient`
    // term already gets a CFL contribution for
    // (`AFFINE_CFL_STENCIL_CORNER_DISTANCE`, reused here unchanged: same
    // 3x3 kernel, same max corner offset, same reasoning). Before this,
    // a scene relying on `adaptive_timestep` still had zero protection once
    // grain spin grew large -- confirmed directly: a real column-collapse
    // isolation test exploded (domain-spanning positions, spin up to ~10.6)
    // once real spin started feeding the grid, at a dt sized only for the
    // grains' own DEM contact stiffness, blind to this term entirely.
    for population in grain_populations {
        for grain in &population.grains {
            let s = grain.v.length()
                + grain.spin.abs() * AFFINE_CFL_STENCIL_CORNER_DISTANCE * config.grid_cell_size;
            max_speed = max_speed.max(s);
        }
    }
    // Rods aren't scanned by the particle loop above (separate SoA) -- fold
    // in their own CFL bound the same way a stiff material would clamp
    // min_mat_dt, so a rod going unstable can never silently escape the
    // adaptive substep logic (the exact bug class this whole rod effort
    // started from: something CFL never knew about). Skipped for sleeping
    // rods -- this is the real cost fix for many simultaneous rods (a grass
    // field): `rod_cfl_dt` is a per-point Gershgorin sum over every stiffness
    // term touching it, paid EVERY substep for EVERY rod before this; a
    // settled rod contributing nothing to the min anyway (its own dt bound
    // stays constant while frozen) has no reason to keep paying for it.
    for rod in rods {
        // An implicit-integration rod is advanced ONCE per `step()` call, entirely
        // outside this substep loop (see `Simulation::step`'s own implicit-rod
        // pass) -- its stability no longer depends on this shared adaptive dt at
        // all (the whole point of implicit integration: unconditionally stable
        // regardless of the rod's own stiffness), so it correctly contributes
        // nothing here, same spirit as a sleeping rod contributing nothing while
        // frozen.
        if rod.sleeping || rod.use_implicit_integration {
            continue;
        }
        let rod_dt = rod_cfl_dt(&rod.points, &rod.material, config.rod_cfl_coefficient);
        if rod_dt.is_finite() && rod_dt > 0.0 {
            min_mat_dt = min_mat_dt.min(rod_dt);
        }
    }
    // Nonlocal Granular Fluidity's own real, quoted Von Neumann stability
    // bound (`GranularFluidityConfig::stability_dt`, Haeri & Skonieczny
    // 2022). `None` (every scene without a configured `GranularFluidityField`)
    // leaves this exactly as it always was.
    if let Some(bound) = granular_fluidity_dt_bound
        && bound.is_finite()
        && bound > 0.0
    {
        min_mat_dt = min_mat_dt.min(bound);
    }
    // `ThermalDiffusion`'s own explicit-diffusion stability bound
    // (`ThermalConfig::stability_dt`) -- normally many orders of magnitude
    // larger than MPM's own CFL (real thermal diffusivity is tiny), so this
    // is a no-op for any correctly-configured scene. It only bites on a
    // real, already-reproduced misconfiguration (passing `grid_cell_size`
    // instead of `dx_meters`, see that field's own doc) -- folding it in
    // turns that from a silent runaway into an automatically clamped,
    // still-correct substep, same precedent as NGF above.
    if let Some(bound) = thermal_dt_bound
        && bound.is_finite()
        && bound > 0.0
    {
        min_mat_dt = min_mat_dt.min(bound);
    }
    // Real, standard "additional stability condition" for explicit
    // integration under a body force (Bridson, "Fluid Simulation for
    // Computer Graphics" ch. 3; Foster & Fedkiw 2001) -- gravity alone can
    // move a particle more than one cell per substep even at REST (zero
    // velocity, zero material stress), which none of the bounds above catch
    // (they all key off existing velocity/stress/velocity-gradient, all
    // zero at t=0). Derived the same way the velocity CFL above already is
    // (a substep shouldn't let a particle gain more than
    // `cfl_coefficient*cell_width` of *implied* motion): a constant
    // acceleration g reaches speed g*dt after time dt, so bounding
    // `g*dt <= cfl_coefficient*cell_width/dt` gives `dt <=
    // sqrt(cfl_coefficient*cell_width/g)`. Normally dominated by a much
    // tighter material bound (e.g. a stiff EOS's acoustic term) and never
    // the binding constraint -- confirmed live as a real, previously-latent
    // gap once a strict fluid's `eos_stiffness` is set to 0.0 for pressure-
    // projection incompressibility (see `SimConfig::fluid_pressure_
    // iterations`'s own doc): with no per-material bound left at rest, a
    // scene's very first substep took the full frame dt in one step, and a
    // single ~0.1s free-fall substep at real Earth gravity is enormous for
    // an explicit MPM update (Δv ≈ 98 grid-units/s in one substep on a
    // 64-cell domain at dx_meters=0.01) -- root cause, not the projection
    // itself, which was correctly reacting to state a too-large substep had
    // already made extreme.
    let g = config.gravity.length();
    if g > f32::EPSILON {
        // `near_wall_gravity_scale` (computed in the fold above) is >1.0
        // only when at least one strict-fluid particle is both near a wall
        // and `fluid_near_wall_cfl_scale != 1.0` -- 1.0 (no particle
        // qualifies, or the feature is off) makes this an exact no-op,
        // identical to the plain gravity bound below it replaces.
        let gravity_dt =
            (config.cfl_coefficient * config.grid_cell_size / (g * near_wall_gravity_scale)).sqrt();
        if gravity_dt.is_finite() && gravity_dt > 0.0 {
            min_mat_dt = min_mat_dt.min(gravity_dt);
        }
    }
    (cfl_bound(config, max_speed, min_mat_dt, max_dt), max_speed)
}

/// TEMPORARY diagnostic (2026-08-28): identifies which real CFL term is
/// actually binding `min_mat_dt` for the single worst (most constraining)
/// particle of a given material -- found needed live-debugging the
/// still-open steam divergence, after three independent, correctly-
/// implemented, sourced CFL tightenings (shock-viscosity augmentation,
/// single-particle instability, a since-reverted retry-bound experiment)
/// each showed ZERO measurable effect on the actual bug. Three real
/// negative results in a row means the next step is answering directly
/// which term is deciding, not guessing a fourth blind.
///
/// Deliberately sequential and separate from the production fold above,
/// not a refactor of it: this is diagnostic-only, not performance-
/// sensitive, and reuses the exact same formulas already proven correct in
/// that fold (copied, not re-derived, so there's no risk of the two
/// drifting apart) -- touching the hot, already-hardened, parallel path
/// itself carries real regression risk this session can't afford to
/// re-verify from scratch this late. Remove once the investigation
/// concludes; see project memory for the full context.
pub(crate) fn diagnose_worst_particle_cfl_term(
    config: &SimConfig,
    particles: &Particles,
    active_count: usize,
    materials: &MaterialRegistry,
    material_filter: Option<u32>,
) -> Option<(usize, &'static str, f32)> {
    let mut worst_dt = f32::INFINITY;
    let mut worst_i = None;
    let mut worst_term = "none";
    for i in 0..active_count {
        if let Some(filter) = material_filter
            && particles.material_id[i] != filter
        {
            continue;
        }
        let grad_norm = (particles.velocity_gradient[i].x_axis.length_squared()
            + particles.velocity_gradient[i].y_axis.length_squared())
        .sqrt();

        let mdt = materials.timestep_bound(
            particles.material_id[i],
            particles.density[i],
            particles.hardening_scale[i],
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        if mdt.is_finite() && mdt > 0.0 && mdt < worst_dt {
            worst_dt = mdt;
            worst_i = Some(i);
            worst_term = "material_timestep_bound(acoustic/viscous)";
        }

        if grad_norm.is_finite()
            && grad_norm > f32::EPSILON
            && materials.owns_deformation_volume_state(particles.material_id[i])
            && let Some(c2_rest) = materials.rest_acoustic_c2(particles.material_id[i])
            && c2_rest.is_finite()
            && c2_rest > f32::EPSILON
        {
            let weak_shock_gamma = materials.get(particles.material_id[i]).params().eos_power;
            if weak_shock_gamma.is_finite() && weak_shock_gamma > 0.0 {
                let c0_quadratic = (weak_shock_gamma + 1.0) * 0.25;
                let c_eff = c2_rest.sqrt() + 2.0 * c0_quadratic * config.grid_cell_size * grad_norm;
                if c_eff.is_finite() && c_eff > f32::EPSILON {
                    let shock_dt = config.material_cfl_coefficient * config.grid_cell_size / c_eff;
                    if shock_dt.is_finite() && shock_dt > 0.0 && shock_dt < worst_dt {
                        worst_dt = shock_dt;
                        worst_i = Some(i);
                        worst_term = "shock_viscosity_augmented_acoustic";
                    }
                }
            }
        }

        if materials.owns_deformation_volume_state(particles.material_id[i]) {
            let rest_density = materials
                .get(particles.material_id[i])
                .params()
                .rest_density;
            let j = particles.volume[i] / particles.initial_volume[i];
            // Same real constitutive-stiffness fix as `choose_substep_dt`'s
            // own copy of this bound -- see that copy's own comment for the
            // full story (missing `lambda=rho0*c0^2` term, found 2026-08-29).
            // This diagnostic deliberately COPIES the production formula
            // rather than sharing a helper (see this function's own
            // top-level doc) -- so this copy must be kept in sync by hand
            // whenever the production formula changes, exactly as it just
            // was here.
            if let Some(c2_rest) = materials.rest_acoustic_c2(particles.material_id[i])
                && rest_density.is_finite()
                && rest_density > 0.0
                && j.is_finite()
                && j > 0.0
                && c2_rest.is_finite()
                && c2_rest > f32::EPSILON
            {
                const QUADRATIC_SPLINE_K: f32 = 6.0;
                const DIMENSION_D: f32 = 2.0;
                let kd = QUADRATIC_SPLINE_K * DIMENSION_D;
                let lambda = rest_density * c2_rest;
                let kd_lambda = kd * lambda;
                let single_particle_dt = if j <= 1.0 {
                    (config.grid_cell_size / (2.0 - j)) * (2.0 * rest_density / kd_lambda).sqrt()
                } else {
                    config.grid_cell_size
                        * (rest_density * (j + 1.0) / (j * j * j * kd_lambda)).sqrt()
                };
                if single_particle_dt.is_finite()
                    && single_particle_dt > 0.0
                    && single_particle_dt < worst_dt
                {
                    worst_dt = single_particle_dt;
                    worst_i = Some(i);
                    worst_term = "single_particle_instability";
                }
            }
        }

        // Real, live density-AND-temperature-aware acoustic term -- same
        // production formula `choose_substep_dt`'s own copy adds, see that
        // copy's own doc for the full account (this diagnostic must be
        // kept in sync by hand, per this function's own top-level doc).
        if let Some(c2_live) =
            materials.acoustic_c2_at_particle(particles.material_id[i], particles, i)
            && c2_live.is_finite()
            && c2_live > f32::EPSILON
        {
            let live_temp_dt =
                config.material_cfl_coefficient * config.grid_cell_size / c2_live.sqrt();
            if live_temp_dt.is_finite() && live_temp_dt > 0.0 && live_temp_dt < worst_dt {
                worst_dt = live_temp_dt;
                worst_i = Some(i);
                worst_term = "live_temperature_acoustic";
            }
        }

        let deformation_coefficient = config.cfl_coefficient.min(0.5);
        let deformation_dt = if grad_norm.is_finite() && grad_norm > f32::EPSILON {
            deformation_coefficient / grad_norm
        } else {
            f32::INFINITY
        };
        if deformation_dt.is_finite() && deformation_dt > 0.0 && deformation_dt < worst_dt {
            worst_dt = deformation_dt;
            worst_i = Some(i);
            worst_term = "deformation_gradient_rate";
        }

        let mut s = particles.v[i].length();
        if config.cfl_include_affine_speed {
            s += grad_norm * AFFINE_CFL_STENCIL_CORNER_DISTANCE * config.grid_cell_size;
        }
        if s > f32::EPSILON {
            let velocity_dt = config.cfl_coefficient * config.grid_cell_size / s;
            if velocity_dt.is_finite() && velocity_dt > 0.0 && velocity_dt < worst_dt {
                worst_dt = velocity_dt;
                worst_i = Some(i);
                worst_term = "velocity_cfl(incl. affine)";
            }
        }
    }
    worst_i.map(|i| (i, worst_term, worst_dt))
}

/// Shared CFL formula: clamps dt to advection + material bounds.
/// Called by both SoA and AoS scan paths after computing their respective max values.
pub(crate) fn cfl_bound(config: &SimConfig, max_speed: f32, min_mat_dt: f32, max_dt: f32) -> f32 {
    assert!(
        max_dt.is_finite() && max_dt > 0.0,
        "CFL maximum timestep must be finite and positive"
    );
    let mut dt = max_dt;
    if max_speed > f32::EPSILON {
        dt = dt.min(config.cfl_coefficient * config.grid_cell_size / max_speed);
    }
    if min_mat_dt.is_finite() && min_mat_dt > 0.0 {
        dt = dt.min(min_mat_dt);
    }
    assert!(
        dt.is_finite() && dt > 0.0,
        "CFL produced a non-positive timestep; inspect material state and parameters"
    );
    // `min_dt` is intentionally not a floor.  Raising a material/acoustic CFL
    // upper bound changes the PDE integration; callers must substep, defer, or
    // report inability to meet their work budget instead.
    dt
}

/// Whether `x` sits within `boundary_thickness` cells of any wall -- the SAME
/// zone `apply_slip_wall_velocity` already treats specially, not a new margin.
/// Used by `SimConfig::fluid_near_wall_cfl_scale`'s proactive tightening.
pub(crate) fn is_near_wall(x: Vec2, grid_res: usize, boundary_thickness: usize) -> bool {
    let t = boundary_thickness as f32;
    let hi = grid_res as f32 - t;
    x.x < t || x.x > hi || x.y < t || x.y > hi
}

// The APIC affine matrix C encodes the local velocity gradient.
// The farthest point in the quadratic B-spline 3×3 stencil is at 1.5 cells per axis,
// so its corner distance is 1.5*√2 cells -- the effective maximum affine speed contribution.
// Hoisted to module scope (2026-08-14, was local to `affine_cfl_speed_contribution`)
// so `choose_substep_dt`'s own fold can share it too, without duplicating the
// magic number, when it inlines this same formula against a pre-shared
// Frobenius norm -- see that call site's own comment.
pub(crate) const AFFINE_CFL_STENCIL_CORNER_DISTANCE: f32 = 1.5 * std::f32::consts::SQRT_2;

// Only called from `systems::gpu::solver::step`'s own substep loop since
// 2026-08-14: the CPU solver's own `choose_substep_dt` used to call this
// directly too, but now inlines the same math against a norm it computes
// once and shares (see that fold's own comment) rather than recomputing it
// separately. Real, disclosed feature-gate (matches the existing convention
// already on this function's own re-export in `solver::mod`, not a new
// one) -- without it, a build without `gpu` correctly has no caller.
#[cfg(feature = "gpu")]
pub(crate) fn affine_cfl_speed_contribution(c: &Mat2, cell_width: f32) -> f32 {
    let grad_norm = (c.x_axis.length_squared() + c.y_axis.length_squared()).sqrt();
    grad_norm * AFFINE_CFL_STENCIL_CORNER_DISTANCE * cell_width
}

/// The F/J update is a local ODE in its own right (`dF/dt = C*F`); a large
/// velocity gradient can invert F within a nominally velocity-safe substep
/// even when the affine speed contribution above is disabled for the
/// advection CFL. `grad_norm` is the Frobenius norm of the particle's own
/// `velocity_gradient` -- a caller that already computed it for another CFL
/// term (advection speed, shock viscosity) should pass that same value
/// rather than recomputing it. Real, standard practice (bounds the
/// dimensionless per-substep velocity-gradient increment, `cfl_coefficient`
/// capped at 0.5 as the safety margin), unconditional on material type --
/// every particle integrates its own F this way.
pub(crate) fn deformation_gradient_ode_dt_bound(grad_norm: f32, cfl_coefficient: f32) -> f32 {
    if grad_norm.is_finite() && grad_norm > f32::EPSILON {
        cfl_coefficient.min(0.5) / grad_norm
    } else {
        f32::INFINITY
    }
}

/// Real, standard hydrocode stability correction for von Neumann-Richtmyer
/// artificial (shock) viscosity (Wilkins 1980, "Calculation of Elastic-
/// Plastic Flow," Methods in Computational Physics; Benson 1992,
/// "Computational methods in Lagrangian and Eulerian hydrocodes," Comput.
/// Methods Appl. Mech. Engrg. 99): augments the acoustic sound speed with
/// the shock-viscosity's own contribution, `c_eff = c_sound +
/// 2*c0_quadratic*h*|div_v|`, so a violent local compression event tightens
/// dt even before it has driven J far from 1. `grad_norm` (Frobenius norm of
/// the velocity gradient) is a real, conservative proxy for `|div_v|` -- it
/// upper-bounds any single directional derivative including the trace/
/// divergence, so this errs toward MORE caution, never less. `None` when
/// the material has no real acoustic term (not a strict fluid) or an input
/// is degenerate -- the same population `von_neumann_richtmyer_q` is ever
/// invoked for.
pub(crate) fn shock_viscosity_dt_bound(
    grad_norm: f32,
    owns_deformation_volume_state: bool,
    rest_acoustic_c2: Option<f32>,
    eos_power: f32,
    grid_cell_size: f32,
    material_cfl: f32,
) -> Option<f32> {
    if !(grad_norm.is_finite() && grad_norm > f32::EPSILON && owns_deformation_volume_state) {
        return None;
    }
    let c2_rest = rest_acoustic_c2?;
    if !(c2_rest.is_finite() && c2_rest > f32::EPSILON && eos_power.is_finite() && eos_power > 0.0)
    {
        return None;
    }
    let c0_quadratic = (eos_power + 1.0) * 0.25;
    let c_eff = c2_rest.sqrt() + 2.0 * c0_quadratic * grid_cell_size * grad_norm;
    if !(c_eff.is_finite() && c_eff > f32::EPSILON) {
        return None;
    }
    let shock_dt = material_cfl * grid_cell_size / c_eff;
    (shock_dt.is_finite() && shock_dt > 0.0).then_some(shock_dt)
}

/// Real, derived "single-particle instability" bound (Sun, Shinar &
/// Schroeder 2020, "Effective time step restrictions for explicit MPM
/// simulation," SCA 2020, Section 4.5): when a particle becomes isolated,
/// the grid velocity there is driven by that ONE particle's own pressure
/// force, which feeds back into its own next-substep J -- a fixed-point
/// iteration on J that can diverge if the timestep doesn't respect it. Valid
/// for every strict-fluid particle unconditionally (the paper derives it
/// from `tr(H)`'s own proven worst-case upper bound, not from detecting
/// isolation directly) -- the authors' own relaxed form (their stricter Eq.
/// 6 forces J<=1 outright; this one instead allows overshoot but bounds it
/// from diverging). `K=6` is the paper's own derived constant for quadratic
/// B-splines (this engine's own kernel); `d=2` for this 2D engine.
/// Continuous at J=1 from both sides by construction. `None` when the
/// material has no real acoustic term or an input is degenerate.
pub(crate) fn single_particle_instability_dt_bound(
    owns_deformation_volume_state: bool,
    rest_density: f32,
    j: f32,
    rest_acoustic_c2: Option<f32>,
    grid_cell_size: f32,
) -> Option<f32> {
    if !owns_deformation_volume_state {
        return None;
    }
    let c2_rest = rest_acoustic_c2?;
    if !(rest_density.is_finite()
        && rest_density > 0.0
        && j.is_finite()
        && j > 0.0
        && c2_rest.is_finite()
        && c2_rest > f32::EPSILON)
    {
        return None;
    }
    const QUADRATIC_SPLINE_K: f32 = 6.0;
    const DIMENSION_D: f32 = 2.0;
    let kd_lambda = QUADRATIC_SPLINE_K * DIMENSION_D * rest_density * c2_rest;
    let dt = if j <= 1.0 {
        (grid_cell_size / (2.0 - j)) * (2.0 * rest_density / kd_lambda).sqrt()
    } else {
        grid_cell_size * (rest_density * (j + 1.0) / (j * j * j * kd_lambda)).sqrt()
    };
    (dt.is_finite() && dt > 0.0).then_some(dt)
}

#[cfg(test)]
mod tests {
    use glam::{Mat2, Vec2};

    use super::{SubstepBounds, SubstepScene, cfl_bound, choose_substep_dt};
    use crate::materials::{MaterialRegistry, NewtonianFluidMaterial};
    use crate::particle::Particles;
    use crate::solver::SimConfig;

    #[test]
    fn min_dt_never_raises_a_cfl_upper_bound() {
        let mut config = SimConfig::standard(16, 1.0, Vec2::ZERO);
        config.grid_cell_size = 1.0;
        config.cfl_coefficient = 0.5;
        config.min_dt = 0.1;

        let dt = cfl_bound(&config, 100.0, 0.01, 1.0);

        assert!((dt - 0.005).abs() < 1.0e-7);
        assert!(dt < config.min_dt);
    }

    // Real, one-particle scene for the near-wall Mach-relative gate
    // (2026-08-11): a strict-fluid particle sitting near a wall, with a
    // known, real Tait EOS (eos_stiffness=100, eos_power=7, rest_density=1
    // -> c2_rest=700, c_s_rest=sqrt(700)~=26.46) and a deliberate 10%
    // compression (J=0.9), so `|J-1|=0.1` is a fixed, known probe value.
    fn near_wall_fluid_scene(eos_stiffness: f32) -> (SimConfig, Particles, MaterialRegistry) {
        let mut config = SimConfig::standard(16, 1.0, Vec2::ZERO);
        config.grid_cell_size = 1.0;
        config.cfl_coefficient = 0.9;
        config.material_cfl_coefficient = 0.1;
        config.fluid_near_wall_cfl_scale = 20.0;
        config.fluid_near_wall_compression_mach_margin = 2.0;
        config.fluid_near_wall_compression_threshold = 0.01;

        let rest_density = 1.0;
        let material = NewtonianFluidMaterial::new(rest_density, 1.0e-3, eos_stiffness, 7.0);
        let materials = MaterialRegistry::with_default(Box::new(material));

        let j = 0.9; // 10% compression, well above both the old 1% absolute
        // threshold AND (at low last_max_speed) the new Mach-relative one.
        let mut particles = Particles::new();
        particles.x.push(Vec2::new(0.5, 8.0)); // x=0.5 < boundary_thickness=2 -> near wall
        particles.v.push(Vec2::ZERO);
        particles.velocity_gradient.push(Mat2::ZERO);
        particles.deformation_gradient.push(Mat2::IDENTITY);
        particles.mass.push(1.0);
        particles.initial_volume.push(1.0);
        particles.volume.push(j);
        particles.density.push(rest_density / j);
        particles.material_id.push(0);
        particles.plastic_volume_ratio.push(1.0);
        particles.hardening_scale.push(1.0);
        particles.friction_hardening.push(0.0);
        particles.log_volume_strain.push(0.0);
        particles.temperature.push(0.0);
        particles.user_tag.push(0);
        particles.activation.push(0.0);
        particles.activation_dir.push(Vec2::ZERO);
        particles.muscle_group_id.push(0);
        particles.contact_group.push(0);
        particles.pinned.push(0);
        particles.scalar_field.push(0.0);
        particles.internal_pressure.push(0.0);
        particles.sleeping.push(false);

        (config, particles, materials)
    }

    #[test]
    fn near_wall_gate_relaxes_when_measured_speed_predicts_this_much_compression() {
        let (config, particles, materials) = near_wall_fluid_scene(100.0);

        // c_s_rest = sqrt(700) ~= 26.46. At last_max_speed=1.0, Mach~=0.038,
        // Mach^2*margin ~= 0.0029 -- far below the real |J-1|=0.1 probe, so
        // the gate SHOULD fire (near-wall 20x tightening applies).
        let (dt_low_speed, _) = choose_substep_dt(
            &config,
            SubstepScene {
                particles: &particles,
                active_count: 1,
                materials: &materials,
                rods: &[],
                grain_populations: &[],
            },
            SubstepBounds {
                max_dt: 1.0,
                granular_fluidity_dt_bound: None,
                thermal_dt_bound: None,
                last_max_speed: 1.0,
            },
        );

        // At last_max_speed=20.0 (Mach~=0.756, close to the material's own
        // c_s_rest -- a genuinely fast flow), Mach^2*margin ~= 1.14 -- ABOVE
        // the real |J-1|=0.1 probe, so the SAME compression is now within
        // what this flow speed already predicts as normal, and the gate
        // should NOT fire (no 20x tightening).
        let (dt_high_speed, _) = choose_substep_dt(
            &config,
            SubstepScene {
                particles: &particles,
                active_count: 1,
                materials: &materials,
                rods: &[],
                grain_populations: &[],
            },
            SubstepBounds {
                max_dt: 1.0,
                granular_fluidity_dt_bound: None,
                thermal_dt_bound: None,
                last_max_speed: 20.0,
            },
        );

        assert!(
            dt_high_speed > dt_low_speed * 5.0,
            "expected the near-wall gate to relax (larger dt) once the measured \
             flow speed already predicts this much compression as normal -- \
             got dt_low_speed={dt_low_speed}, dt_high_speed={dt_high_speed}"
        );
    }

    #[test]
    fn near_wall_gate_falls_back_to_absolute_threshold_with_no_acoustic_term() {
        // eos_stiffness=0.0 -- the real `fluid_pressure_projection_gui.rs`
        // case (`rest_acoustic_c2()` must return `None` here). The gate must
        // then use `fluid_near_wall_compression_threshold` (0.01) regardless
        // of `last_max_speed`, unchanged from before this session's fix.
        let (config, particles, materials) = near_wall_fluid_scene(0.0);
        assert!(materials.get(0).rest_acoustic_c2().is_none());

        let (dt_low_speed, _) = choose_substep_dt(
            &config,
            SubstepScene {
                particles: &particles,
                active_count: 1,
                materials: &materials,
                rods: &[],
                grain_populations: &[],
            },
            SubstepBounds {
                max_dt: 1.0,
                granular_fluidity_dt_bound: None,
                thermal_dt_bound: None,
                last_max_speed: 1.0,
            },
        );
        let (dt_high_speed, _) = choose_substep_dt(
            &config,
            SubstepScene {
                particles: &particles,
                active_count: 1,
                materials: &materials,
                rods: &[],
                grain_populations: &[],
            },
            SubstepBounds {
                max_dt: 1.0,
                granular_fluidity_dt_bound: None,
                thermal_dt_bound: None,
                last_max_speed: 500.0,
            },
        );

        // With eos_stiffness=0 the acoustic term is dead (`timestep_bound`
        // filters its non-finite/zero contribution) regardless of the
        // near-wall scale ever being applied to it -- so both calls should
        // land on the SAME dt (gravity/velocity bound only), proving
        // `last_max_speed` has zero effect on this fallback path.
        assert!(
            (dt_low_speed - dt_high_speed).abs() < 1.0e-6,
            "fallback path must be independent of last_max_speed: \
             dt_low_speed={dt_low_speed}, dt_high_speed={dt_high_speed}"
        );
    }

    /// Real, closed-form check on the single-particle-instability bound's
    /// missing constitutive-stiffness term (found 2026-08-29, both by
    /// independent deep research and by reading this
    /// formula directly -- before this fix the formula had no stiffness
    /// term at all, so `sqrt(density)` alone is not dimensionally a time).
    /// At J=1, K=6 (this engine's own quadratic-B-spline constant), d=2:
    /// the `j<=1.0` branch's `(dx/(2-j))*sqrt(2*rho0/(kd*lambda))` reduces
    /// to `dx*sqrt(2/(12*rho0*c0^2/rho0))` = `dx*sqrt(1/(6*c0^2))`
    /// = `sqrt(1/6)*dx/c0` -- an exact, hand-derivable identity, not a
    /// tuned/fitted expectation.
    #[test]
    fn single_particle_instability_bound_matches_closed_form_at_j_equals_one() {
        let mut config = SimConfig::standard(16, 1.0, Vec2::ZERO);
        config.grid_cell_size = 2.0; // dx=2.0, deliberately != 1.0 so the
        // test can't pass by accident if dx were silently dropped from the
        // formula.

        let rest_density = 3.0;
        let eos_stiffness = 50.0;
        let eos_power = 7.0;
        // c2_rest = eos_stiffness*eos_power/rest_density (real Tait
        // rest-state acoustic speed squared, see `rest_acoustic_c2`'s own
        // doc) -- this test's c0 is whatever that derivation gives, not an
        // independently chosen value, so the check stays tied to the real
        // material rather than a coincidence.
        let material = NewtonianFluidMaterial::new(rest_density, 1.0e-3, eos_stiffness, eos_power);
        let materials = MaterialRegistry::with_default(Box::new(material));
        let c0 = (eos_stiffness * eos_power / rest_density).sqrt();

        let mut particles = Particles::new();
        particles.x.push(Vec2::new(8.0, 8.0)); // far from any wall
        particles.v.push(Vec2::ZERO);
        particles.velocity_gradient.push(Mat2::ZERO); // zero grad_norm -- keeps
        // the shock-viscosity/deformation-gradient terms inert so only the
        // single-particle-instability term can bind.
        particles.deformation_gradient.push(Mat2::IDENTITY); // J=1.0 exactly
        particles.mass.push(1.0);
        particles.initial_volume.push(1.0);
        particles.volume.push(1.0);
        particles.density.push(rest_density);
        particles.material_id.push(0);
        particles.plastic_volume_ratio.push(1.0);
        particles.hardening_scale.push(1.0);
        particles.friction_hardening.push(0.0);
        particles.log_volume_strain.push(0.0);
        particles.temperature.push(0.0);
        particles.user_tag.push(0);
        particles.activation.push(0.0);
        particles.activation_dir.push(Vec2::ZERO);
        particles.muscle_group_id.push(0);
        particles.contact_group.push(0);
        particles.pinned.push(0);
        particles.scalar_field.push(0.0);
        particles.internal_pressure.push(0.0);
        particles.sleeping.push(false);

        let (idx, term, dt) =
            super::diagnose_worst_particle_cfl_term(&config, &particles, 1, &materials, None)
                .expect("a single strict-fluid particle at rest must report a binding term");

        assert_eq!(idx, 0);
        assert_eq!(
            term, "single_particle_instability",
            "at J=1 with zero velocity gradient, no other term should bind tighter"
        );

        let expected_dt = (1.0_f32 / 6.0).sqrt() * config.grid_cell_size / c0;
        assert!(
            (dt - expected_dt).abs() / expected_dt < 1.0e-4,
            "single-particle-instability dt must match the closed-form \
             sqrt(1/6)*dx/c0 at J=1: expected={expected_dt}, got={dt}"
        );
    }
}
