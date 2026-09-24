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
use crate::particle::Particles;
use crate::rod::{Rod, RodNetwork, network_cfl_dt, rod_cfl_dt};

// choose_substep_dt: picks the largest CFL-safe dt ≤ max_dt.
// Called inside step()'s substep loop — max_dt is the remaining frame time.
// pub(crate) so the GPU solver can reuse this without duplicating CFL logic.
#[allow(clippy::too_many_arguments)]
pub(crate) fn choose_substep_dt(
    config: &SimConfig,
    particles: &Particles,
    active_count: usize,
    materials: &MaterialRegistry,
    rods: &[Rod],
    rod_networks: &[RodNetwork],
    max_dt: f32,
    granular_fluidity_dt_bound: Option<f32>,
    thermal_dt_bound: Option<f32>,
    // Real max particle speed from the PREVIOUS call to this function
    // (one-substep-lagged -- see `Simulation::last_max_particle_speed`'s own
    // doc). Used ONLY by the near-wall gate's Mach-relative compression
    // threshold (`SimConfig::fluid_near_wall_compression_mach_margin`) --
    // THIS call's own max_speed isn't known yet at the point the gate needs
    // it (it's still being folded), so the previous substep's value is the
    // freshest real data available, same "react at the next sync point"
    // pattern this codebase's GPU batch CFL scan already uses.
    last_max_speed: f32,
) -> (f32, f32) {
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
                // The deformation update is a local ODE in its own right.  Bound the
                // dimensionless velocity-gradient increment even when affine velocity
                // contribution is disabled for the advection CFL; otherwise an Euler
                // F update can invert within a nominally velocity-safe substep.
                // Reuses `grad_norm` computed above instead of calling
                // `deformation_gradient_cfl_bound` (which would recompute the
                // identical norm) -- same formula as that function's own body,
                // bit-identical result.
                let deformation_coefficient = config.cfl_coefficient.min(0.5);
                let deformation_dt = if grad_norm.is_finite() && grad_norm > f32::EPSILON {
                    deformation_coefficient / grad_norm
                } else {
                    f32::INFINITY
                };
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
    // Same real fold as rods above, same reason: a network going unstable
    // must never silently escape the adaptive substep logic. Real Gershgorin
    // CFL bound (`network_cfl_dt`), same `rod_cfl_coefficient` a single rod
    // uses -- the underlying stiffness/damping physics is identical, only
    // the topology iteration differs (see `network.rs`'s own doc).
    for network in rod_networks {
        let net_dt = network_cfl_dt(network, config.rod_cfl_coefficient);
        if net_dt.is_finite() && net_dt > 0.0 {
            min_mat_dt = min_mat_dt.min(net_dt);
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
fn is_near_wall(x: Vec2, grid_res: usize, boundary_thickness: usize) -> bool {
    let t = boundary_thickness as f32;
    let hi = grid_res as f32 - t;
    x.x < t || x.x > hi || x.y < t || x.y > hi
}

// The APIC affine matrix C encodes the local velocity gradient.
// The farthest point in the quadratic B-spline 3×3 stencil is at 1.5 cells per axis,
// so its corner distance is 1.5*√2 cells — the effective maximum affine speed contribution.
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

#[cfg(test)]
mod tests {
    use glam::{Mat2, Vec2};

    use super::{cfl_bound, choose_substep_dt};
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
            &particles,
            1,
            &materials,
            &[],
            &[],
            1.0,
            None,
            None,
            1.0,
        );

        // At last_max_speed=20.0 (Mach~=0.756, close to the material's own
        // c_s_rest -- a genuinely fast flow), Mach^2*margin ~= 1.14 -- ABOVE
        // the real |J-1|=0.1 probe, so the SAME compression is now within
        // what this flow speed already predicts as normal, and the gate
        // should NOT fire (no 20x tightening).
        let (dt_high_speed, _) = choose_substep_dt(
            &config,
            &particles,
            1,
            &materials,
            &[],
            &[],
            1.0,
            None,
            None,
            20.0,
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
            &particles,
            1,
            &materials,
            &[],
            &[],
            1.0,
            None,
            None,
            1.0,
        );
        let (dt_high_speed, _) = choose_substep_dt(
            &config,
            &particles,
            1,
            &materials,
            &[],
            &[],
            1.0,
            None,
            None,
            500.0,
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
}
