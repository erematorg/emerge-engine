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
    // Max particle speed from the previous call (one substep lagged, see
    // `Simulation::last_max_particle_speed`), used only by the near-wall
    // gate's Mach-relative threshold
    // (`SimConfig::fluid_near_wall_compression_mach_margin`): this call's
    // own max speed is still being folded when the gate needs it.
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
        last_max_speed,
    } = bounds;
    if !config.adaptive_timestep {
        return (max_dt.min(config.dt), 0.0);
    }
    // Single parallel pass for both the velocity CFL and the material bound
    // (single-threaded it measured ~8 ms per frame): a pure `(f32, f32)`
    // fold/reduce with no per-chunk allocation. `with_min_len` as in P2G/G2P,
    // since rayon's default chunking makes many more, smaller tasks than
    // one per core.
    let min_len = (active_count / (rayon::current_num_threads() * 2)).max(1);
    let (max_speed, min_mat_dt, near_wall_gravity_scale) = (0..active_count)
        .into_par_iter()
        .with_min_len(min_len)
        .fold(
            || (0.0f32, max_dt, 1.0f32),
            |(mut max_speed, mut min_mat_dt, mut near_wall_scale), i| {
                // Frobenius norm of `velocity_gradient[i]`, computed once and
                // shared by the affine and deformation-rate bounds below
                // (`affine_cfl_speed_contribution`/
                // `deformation_gradient_cfl_bound` each compute it themselves
                // for the GPU CFL scan's CPU side).
                let grad_norm = (particles.velocity_gradient[i].x_axis.length_squared()
                    + particles.velocity_gradient[i].y_axis.length_squared())
                .sqrt();

                let mut s = particles.v[i].length();
                if config.cfl_include_affine_speed {
                    s += grad_norm * AFFINE_CFL_STENCIL_CORNER_DISTANCE * config.grid_cell_size;
                }
                max_speed = max_speed.max(s);
                // Proactive near-wall tightening for strict fluids (see
                // `SimConfig::fluid_near_wall_cfl_scale`); the default `1.0`
                // makes the division a no-op. Gated on actual compression,
                // relative to the material's acoustic stiffness when it has
                // one (see `fluid_near_wall_compression_mach_margin`, Ma² ≈ Δρ
                // in WCSPH; an absolute threshold defeats itself for a
                // softened EOS), else on the absolute
                // `fluid_near_wall_compression_threshold` (e.g.
                // `eos_stiffness=0.0` projection fluids).
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
                // Hydrocode stability correction for von Neumann-Richtmyer
                // shock viscosity (Wilkins 1980, "Calculation of
                // Elastic-Plastic Flow", Methods in Computational Physics;
                // Benson 1992, "Computational methods in Lagrangian and
                // Eulerian hydrocodes", Comput. Methods Appl. Mech. Engrg. 99):
                // the quadratic term `c0_quadratic*h^2*div_v^2` of
                // `von_neumann_richtmyer_q` (used by `IdealGasMaterial` and
                // `NewtonianFluidMaterial`) adds stiffness that grows with the
                // compression rate, and `timestep_bound` never sees `div_v`
                // (measured: `|trace(velocity_gradient)|` past 5000/s with no
                // tightening). Standard practice: `c_eff = c_sound +
                // 2*c0_quadratic*h*|div_v|`. `grad_norm` bounds `|div_v|` from
                // above, so this errs toward caution. Only strict fluids with
                // an acoustic term (`owns_deformation_volume_state` +
                // `rest_acoustic_c2`), the population that calls
                // `von_neumann_richtmyer_q`; `eos_power` is its
                // `weak_shock_gamma` (gas.rs passes `adiabatic_index`,
                // fluid.rs its Tait `eos_power`).
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
                // Single-particle instability bound for strict fluids (Sun,
                // Shinar & Schroeder 2020, "Effective time step restrictions
                // for explicit MPM simulation", SCA 2020, Section 4.5): an
                // isolated particle (a rising steam particle spreading into
                // empty cells) drives its grid nodes alone, so its pressure
                // feeds back into its own next J, a fixed-point iteration
                // that diverges if dt is too large. Derived for the worst
                // case (`tr(H)` at its bound `K*d/dx^2`), so it holds for every
                // strict-fluid particle without detecting isolation. This is
                // the paper's relaxed form, the one it uses for fluids (its
                // Eq. 6 forces J <= 1; this bounds the overshoot, "relaxed by up
                // to a factor of 2 near Jp~=1"). `K = 6` for quadratic
                // B-splines (`spacetime::grid::kernel::quadratic_weights`),
                // `d = 2`. Both branches give `dx*sqrt(2*rest_density/(K*d))`
                // at J = 1.
                // The constitutive stiffness lambda = rho0*c0^2 (the Tait or
                // ideal-gas tangent bulk modulus at rest) is the paper's own
                // `lambda` (see `single_particle_instability_dt_bound`). It is
                // the rest-state tangent, right near J = 1, not a worst case
                // over the whole admissible J range.
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
                // Acoustic term aware of density and temperature together (see
                // `MaterialModel::acoustic_c2_at`): `timestep_bound` only sees a
                // material's construction-time stiffness, while a heated
                // `IdealGasMaterial`'s `c^2` grows linearly with `T` and a
                // temperature-coupled cavitation EOS shifts with density and
                // temperature jointly (measured on `phase_states_gui.rs`'s
                // heated steam: divergence in the thousands, substeps at their
                // cap). A separate term like the two above, no signature
                // change: materials that override nothing fall back through
                // `acoustic_c2_at_temperature`/`rest_acoustic_c2`. Calls the
                // most general tier, `acoustic_c2_at_particle`, so a
                // per-particle scalar (`BoilingMixtureMaterial`'s quality) is
                // reachable too.
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
                // Predictive near-wall tightening for a strict fluid with
                // `eos_stiffness=0`: `fluid_near_wall_cfl_scale`'s tightening
                // above only affects the acoustic bound, which is zero for
                // such a fluid (bit-for-bit identical at scale 5, 20 or off).
                // The gravity CFL bound (folded in after this loop) is the one
                // still active and predictive at rest, so it is the lever. No
                // compression gate: a gate needs J to have drifted already,
                // which fails at the first substep.
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
    // Grains are not scanned by the particle loop above (separate storage,
    // like rods below), so their advection speed is folded in here.
    // `scatter_grains_to_grid` scatters the rigid-body velocity field
    // (`v_com + spin*perp(r)`), which can put far more than `v.length()` on
    // the grid once `spin` is large, the effect particles' affine
    // `velocity_gradient` term covers (`AFFINE_CFL_STENCIL_CORNER_DISTANCE`,
    // same 3x3 kernel). Without it a column collapse exploded (spin up to
    // ~10.6) at a dt sized only for the DEM contact stiffness.
    for population in grain_populations {
        for grain in &population.grains {
            let s = grain.v.length()
                + grain.spin.abs() * AFFINE_CFL_STENCIL_CORNER_DISTANCE * config.grid_cell_size;
            max_speed = max_speed.max(s);
        }
    }
    // Rods aren't scanned by the particle loop above (separate SoA). An
    // explicit rod sub-cycles its own forces within its own stable step
    // inside the substep (`rod::advance_rod`), so a free rod's stiffness no
    // longer bounds the substep; only its speed does, the same way the
    // grains' speed does above. A rod touching other matter still bounds
    // it with its own stable step, as the grid exchanges momentum only once
    // per substep (`Rod::touching_other_matter`). Sleeping and implicit rods
    // (advanced once per `step()`, outside this loop) contribute nothing.
    for rod in rods {
        if rod.sleeping || rod.use_implicit_integration {
            continue;
        }
        for v in &rod.points.v {
            max_speed = max_speed.max(v.length());
        }
        if rod.touching_other_matter {
            let rod_dt = rod_cfl_dt(&rod.points, &rod.material, config.material_cfl_coefficient);
            if rod_dt.is_finite() && rod_dt > 0.0 {
                min_mat_dt = min_mat_dt.min(rod_dt);
            }
        }
    }
    // Nonlocal Granular Fluidity's own quoted Von Neumann stability
    // bound (`GranularFluidityConfig::stability_dt`, Haeri & Skonieczny
    // 2022). `None` (every scene without a configured `GranularFluidityField`)
    // leaves this exactly as it always was.
    if let Some(bound) = granular_fluidity_dt_bound
        && bound.is_finite()
        && bound > 0.0
    {
        min_mat_dt = min_mat_dt.min(bound);
    }
    // Body-force stability condition for explicit integration (Bridson,
    // "Fluid Simulation for Computer Graphics" ch. 3; Foster & Fedkiw 2001):
    // gravity can move a particle more than a cell per substep from rest,
    // which none of the bounds above see (they key off velocity, stress or
    // velocity gradient, all zero at t = 0). Bounding `g*dt <=
    // cfl_coefficient*cell_width/dt` gives `dt <=
    // sqrt(cfl_coefficient*cell_width/g)`. Usually dominated by a material
    // bound; it binds when nothing else does, e.g. a strict fluid with
    // `eos_stiffness = 0.0` under pressure projection (see
    // `SimConfig::fluid_pressure_iterations`), whose first substep otherwise
    // took the whole frame (~98 cells/s of free fall in one substep at 1 cm
    // cells).
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

/// Diagnostic: which CFL term binds `min_mat_dt` for the most constraining
/// particle of a material, built for the steam divergence after three sourced
/// CFL tightenings (shock-viscosity augmentation, single-particle
/// instability, a reverted retry bound) each changed nothing.
///
/// Sequential and separate from the production fold, calling the same bound
/// helpers (`shock_viscosity_dt_bound`, `single_particle_instability_dt_bound`,
/// `deformation_gradient_ode_dt_bound`) at `material_cfl_coefficient`, without
/// the near-wall scaling; only the live acoustic term is an inline copy.
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

        let owns_volume = materials.owns_deformation_volume_state(particles.material_id[i]);
        let rest_acoustic_c2 = materials.rest_acoustic_c2(particles.material_id[i]);
        let params = materials.get(particles.material_id[i]).params();
        if let Some(shock_dt) = shock_viscosity_dt_bound(
            grad_norm,
            owns_volume,
            rest_acoustic_c2,
            params.eos_power,
            config.grid_cell_size,
            config.material_cfl_coefficient,
        ) && shock_dt < worst_dt
        {
            worst_dt = shock_dt;
            worst_i = Some(i);
            worst_term = "shock_viscosity_augmented_acoustic";
        }

        let j = particles.volume[i] / particles.initial_volume[i];
        if let Some(single_particle_dt) = single_particle_instability_dt_bound(
            owns_volume,
            params.rest_density,
            j,
            rest_acoustic_c2,
            config.grid_cell_size,
        ) && single_particle_dt < worst_dt
        {
            worst_dt = single_particle_dt;
            worst_i = Some(i);
            worst_term = "single_particle_instability";
        }

        // Live density- and temperature-aware acoustic term, a copy of
        // `choose_substep_dt`'s (see there); kept in sync by hand.
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

        let deformation_dt = deformation_gradient_ode_dt_bound(grad_norm, config.cfl_coefficient);
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
// Module scope so `choose_substep_dt`'s fold shares it.
pub(crate) const AFFINE_CFL_STENCIL_CORNER_DISTANCE: f32 = 1.5 * std::f32::consts::SQRT_2;

// Only called from `systems::gpu::solver::step`'s substep loop: the CPU
// `choose_substep_dt` inlines the same math against a shared norm. Gated on
// `gpu` like its re-export in `solver::mod`, since a build without it has no
// caller.
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
/// rather than recomputing it. Standard practice (bounds the
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

/// Hydrocode stability correction for von Neumann-Richtmyer shock viscosity
/// (Wilkins 1980, "Calculation of Elastic-Plastic Flow", Methods in
/// Computational Physics; Benson 1992, "Computational methods in Lagrangian
/// and Eulerian hydrocodes", Comput. Methods Appl. Mech. Engrg. 99): augments
/// the sound speed with the shock viscosity's contribution,
/// `c_eff = c_sound + 2*c0_quadratic*h*|div_v|`, so a violent compression tightens dt before J
/// has moved far from 1. `grad_norm` (Frobenius norm of the velocity
/// gradient) bounds `|div_v|` from above, so this errs toward caution. `None`
/// when the material has no acoustic term (not a strict fluid) or an input
/// is degenerate.
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

/// Single-particle instability bound (Sun, Shinar & Schroeder 2020,
/// "Effective time step restrictions for explicit MPM simulation", SCA 2020,
/// Section 4.5): an isolated particle drives its grid nodes alone, so its
/// pressure feeds back into its own next J, a fixed-point iteration that
/// diverges if dt is too large. Holds for every strict-fluid particle (it is
/// derived from `tr(H)`'s worst-case bound, not from detecting isolation);
/// the paper's relaxed form (its Eq. 6 forces J <= 1; this bounds the
/// overshoot). `K = 6` for quadratic B-splines, `d = 2`. Continuous at J = 1.
/// `None` when the material has no acoustic term or an input is degenerate.
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

    // One strict-fluid particle near a wall for the near-wall Mach-relative
    // gate, with a known Tait EOS (eos_stiffness=100, eos_power=7,
    // rest_density=1 -> c2_rest=700, c_s_rest=sqrt(700)~=26.46) and a 10%
    // compression (J=0.9), so `|J-1| = 0.1` is a known probe value.
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
                last_max_speed: 1.0,
            },
        );

        // At last_max_speed=20.0 (Mach~=0.756, close to the material's own
        // c_s_rest -- a fast flow), Mach^2*margin ~= 1.14 -- ABOVE
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
        // eos_stiffness=0.0, the `fluid_pressure_projection_gui.rs` case
        // (`rest_acoustic_c2()` returns `None`): the gate uses
        // `fluid_near_wall_compression_threshold` (0.01) whatever
        // `last_max_speed` is.
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

    /// Closed-form check of the single-particle bound's stiffness term
    /// (without it `sqrt(density)` alone is not a time). At J=1, K=6, d=2 the
    /// `j<=1.0` branch `(dx/(2-j))*sqrt(2*rho0/(kd*lambda))` reduces to
    /// `dx*sqrt(2/(12*rho0*c0^2/rho0))` = `sqrt(1/6)*dx/c0`, an exact identity.
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
