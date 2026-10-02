//! The adaptive-substep physics step: CFL timestep selection, P2G, grid update,
//! G2P, force fields, thermal/scalar diffusion, phase rules, and sleep scoring.
//!
//! Split out of `solver/mod.rs` (was 1536 lines, doing 5-6 jobs in one file) --
//! this is the one piece that's purely "advance the simulation by one step,"
//! distinct from construction, queries, and particle-lifecycle management that
//! live alongside `Simulation` in the parent module. `do_substep`'s own body
//! stays a single ordering-sensitive sequence on purpose (see its inline
//! comments for why each phase must run where it does) -- only the two
//! self-contained pieces split further, into sibling files:
//! adaptive-timestep selection (`cfl.rs`) and per-substep NaN/invalid-state
//! guards (`projection.rs`).

use glam::Vec2;

use super::Simulation;
use super::cfl::choose_substep_dt;
use super::projection::{
    apply_boundary_conditions_to_grid, assert_owned_deformation_state,
    assert_owned_deformation_state_j_range_deferred, project_particle_state_to_admissible,
};
use crate::grains::coupling::{
    apply_grain_contact_forces, gather_grid_to_grains, scatter_grains_to_grid,
};
use crate::grains::micro_rotation::{GrainMicroRotationConfig, couple_grain_spin_to_local_average};
use crate::rod::{
    RodForceParams, RodImplicitStepParams, advance_rod, apply_bending_plasticity,
    apply_gravitropism, apply_growth, apply_phototropism, apply_secondary_growth,
    gather_grid_to_rod, rod_touches_grid_mass, scatter_rod_to_grid, step_rod_implicit,
};
use crate::transfer::{
    G2PParams, gather_contact_point_cloud, gather_grid_to_particles, scatter_particles_to_grid,
    scatter_particles_to_grid_sorted, spatial_sort_order,
};

impl Simulation {
    /// Strict WC-MPM liquids have a deliberately narrow supported coupling
    /// domain.  These features alter particle velocity/state through a
    /// separate numerical/contact model rather than the liquid momentum and
    /// constitutive equations, so silently combining them would make a result
    /// look like a fluid solution when it is not one.
    fn assert_strict_fluid_mode_is_supported(&self) {
        let mut has_strict_fluid = false;
        for i in 0..self.particles.len() {
            let material = self.materials.get(self.particles.material_id[i]);
            if !material.owns_deformation_volume_state() {
                // Pressure-projection incompressibility (`grid::pressure`) has
                // no per-cell fluid-fraction tracking yet -- see
                // `SimConfig::fluid_pressure_iterations`'s doc -- so it is
                // only correct when EVERY particle on the shared grid is a
                // strict fluid. A non-fluid particle here means it isn't.
                assert!(
                    self.config.fluid_pressure_iterations == 0,
                    "fluid_pressure_iterations > 0 requires every particle in the scene to be a strict WC-MPM fluid (particle {i} is not); mixed fluid+solid scenes aren't supported by this projection yet"
                );
                continue;
            }
            has_strict_fluid = true;
            assert!(
                self.particles.contact_group[i] == 0,
                "strict WC-MPM fluid particle {i} cannot use multi-field contact; use a fluid--solid boundary/coupling model"
            );
            assert!(
                self.particles.pinned[i] == 0,
                "strict WC-MPM fluid particle {i} cannot be pinned; use a geometric wall boundary instead"
            );
            assert!(
                !self.particles.sleeping[i],
                "strict WC-MPM fluid particle {i} cannot be sleeping; sleeping removes momentum evolution from the PDE"
            );
            assert!(
                material.mixture_phase().is_none(),
                "strict WC-MPM fluid particle {i} cannot use porous-mixture coupling; that requires a separate multiphase PDE"
            );
        }
        if has_strict_fluid {
            assert!(
                self.boundaries
                    .iter()
                    .all(|boundary| boundary.is_strict_wc_mpm_fluid_compatible()),
                "strict WC-MPM fluid cannot use a boundary with an undeclared post-G2P particle mutation; use a compatible geometric wall/traction discretisation"
            );
            assert!(
                self.config.apic_blend == 1.0,
                "strict WC-MPM fluid requires apic_blend = 1: attenuating the gathered velocity gradient changes the continuity equation"
            );
            assert!(
                self.config.asflip_blend <= 0.0,
                "strict WC-MPM fluid cannot use ASFLIP: its blend/compression switch is a transfer heuristic, not part of this liquid PDE"
            );
            assert!(
                self.config.cundall_damping <= 0.0,
                "strict WC-MPM fluid cannot use Cundall damping; model a declared viscous stress or drag force instead"
            );
            assert!(
                self.config.sleep_threshold <= 0.0,
                "strict WC-MPM fluid cannot sleep; sleeping removes momentum evolution from the PDE"
            );
        }
    }

    /// One MLS-MPM timestep: particle→grid→particle cycle.
    /// The grid is temporary scratch -- only particles hold long-term material memory.
    pub fn step(&mut self) {
        self.assert_strict_fluid_mode_is_supported();
        // Adaptive substep loop: step() always advances exactly config.dt of simulation time,
        // but uses smaller sub-steps when CFL requires it (stiff materials, high velocities).
        // Without this loop, the FixedStepController accounts for config.dt per call but the
        // simulation only advances sub_dt -- causing it to run orders of magnitude too slowly.
        let step_start = std::time::Instant::now();
        self.substep_index_in_frame = 0;
        let mut remaining = self.config.dt;
        let mut substeps_taken = 0;
        self.last_j_projection_count = 0;
        self.last_timing = crate::diagnostics::StepTiming::default();
        // Computed ONCE here, reused by every substep below -- see
        // `cached_spatial_sort_order`'s doc for why (measured:
        // recomputing per substep cost more than it saved).
        if self.config.spatial_sort_enabled {
            self.cached_spatial_sort_order =
                spatial_sort_order(&self.particles, self.active_count, self.grid.resolution());
        }

        // Implicit-integration rods (Baraff & Witkin 1998, see
        // `rod::implicit` doc): advanced ONCE per `step()` at the full
        // frame dt, outside the substep loop. Gravity/wind/push stay inside
        // this solve, not on the shared grid -- `Grid::apply_gravity` is a
        // plain explicit `v += g*dt`, only safe elsewhere because every other
        // body has its own CFL ceiling keeping dt small; an implicit rod has
        // none, so that path is unconditionally unstable at the full frame dt.
        // Not grid-coupled yet -- see the struct field's doc.
        for rod in &mut self.rods {
            if !rod.use_implicit_integration {
                continue;
            }
            if rod.sleeping {
                // Wake on push/wind HERE, before the skip below: the grid-touch
                // wake check runs only after this loop, so waking there alone
                // would miss the same frame a push or wind is first applied.
                // Must check wind_velocity too, not just push_strength -- wind
                // never touches the grid, so it's the only way a sleeping rod
                // can wake from wind alone.
                let has_external_force = (rod.push_strength > 0.0 && rod.push_center.is_some())
                    || rod.wind_velocity.length_squared() > 0.0;
                if has_external_force {
                    rod.sleeping = false;
                    rod.below_threshold_time = 0.0;
                } else {
                    continue;
                }
            }
            // Splitting frame `dt` into smaller implicit substeps reduces backward
            // Euler's numerical damping, letting the rod's tuned damping ratio show
            // as visible sway instead of a smooth glide. Default 1 = prior behavior.
            let substeps = rod.implicit_substeps.max(1);
            let sub_dt = self.config.dt / substeps as f32;
            for _ in 0..substeps {
                step_rod_implicit(
                    &mut rod.points,
                    &rod.material,
                    RodImplicitStepParams {
                        gravity: self.config.gravity,
                        wind_velocity: rod.wind_velocity,
                        wind_drag_coeff: rod.wind_drag_coeff,
                        push_center: rod.push_center,
                        push_strength: rod.push_strength,
                        push_radius: rod.push_radius,
                        dx_meters: self.config.dx_meters,
                        dt: sub_dt,
                    },
                );
            }
            if let Some(gravitropism) = &rod.gravitropism {
                apply_gravitropism(
                    &mut rod.points,
                    gravitropism,
                    self.config.gravity,
                    &self.grid,
                    self.config.dt,
                );
            }
            if let Some(phototropism) = &rod.phototropism {
                apply_phototropism(
                    &mut rod.points,
                    phototropism,
                    self.config.light_dir,
                    &self.grid,
                    self.config.dt,
                );
            }
            if let Some(growth) = &mut rod.growth {
                apply_growth(
                    &mut rod.points,
                    growth,
                    &self.grid,
                    self.config.light_dir,
                    self.config.dx_meters,
                    self.config.dt,
                );
            }
            // Gated on the rod still being over-critical (same Greenhill buckling
            // gate as gravitropism/phototropism above, opposite direction):
            // unconditional secondary growth keeps stiffening the rod past the
            // point it's mechanically needed, making it progressively less
            // responsive to later pushes.
            if let Some(secondary_growth) = &rod.secondary_growth {
                let gravity_si = self.config.gravity.length() * self.config.dx_meters;
                if rod.buckling_warning(gravity_si).is_some() {
                    apply_secondary_growth(
                        &mut rod.points,
                        secondary_growth,
                        self.config.dx_meters,
                        self.config.dt,
                    );
                }
            }
            // Elastic-perfectly-plastic bending (see `plasticity`), applied
            // after any biological reshaping above, so yield acts on top of
            // this step's tropism and growth.
            if let Some(plasticity) = &rod.plasticity {
                apply_bending_plasticity(&mut rod.points, plasticity, self.config.dx_meters);
            }
        }
        // Opt-in implicit big step (see `implicit_corotated`): one Newton-CG
        // solve at the full frame `dt` instead of the CFL-limited loop below.
        // Only when the whole active scene qualifies and the solve converges;
        // otherwise the explicit loop runs unchanged.
        if remaining > 0.0 && self.try_implicit_corotated_substep(remaining) {
            substeps_taken = 1;
            self.last_step_dt = remaining;
            remaining = 0.0;
        }
        while remaining > 0.0 && substeps_taken < self.config.max_substeps_per_step {
            // Cap sub-step at remaining time so we don't overshoot the configured frame dt.
            let t_cfl = std::time::Instant::now();
            let (sub_dt, measured_max_speed) = choose_substep_dt(
                &self.config,
                crate::solver::cfl::SubstepScene {
                    particles: &self.particles,
                    active_count: self.active_count,
                    materials: &self.materials,
                    rods: &self.rods,
                    grain_populations: &self.grain_populations,
                },
                crate::solver::cfl::SubstepBounds {
                    max_dt: remaining,
                    granular_fluidity_dt_bound: self
                        .granular_fluidity
                        .as_ref()
                        .map(|f| f.config.stability_dt(self.config.dx_meters)),
                    last_max_speed: self.last_max_particle_speed,
                },
            );
            // One-substep-lagged, real (not estimated): feeds the near-wall
            // gate's Mach-relative threshold on the NEXT call -- see
            // `choose_substep_dt`'s own `last_max_speed` param doc.
            self.last_max_particle_speed = measured_max_speed;
            self.last_timing.cfl_us += t_cfl.elapsed().as_micros() as u64;
            // Diagnostic, see `diagnose_worst_particle_cfl_term`: opt-in
            // through `EMERGE_CFL_DIAGNOSE` (`all` scans every particle,
            // `<material_id>` one material) in a `research-diagnostics`
            // build, compiled out otherwise (see `research_switch`).
            if let Some(spec) = crate::diagnostics::research_switch("EMERGE_CFL_DIAGNOSE") {
                let filter = if spec.eq_ignore_ascii_case("all") {
                    None
                } else {
                    spec.parse::<u32>().ok()
                };
                if let Some((i, term, dt)) = crate::solver::cfl::diagnose_worst_particle_cfl_term(
                    &self.config,
                    &self.particles,
                    self.active_count,
                    &self.materials,
                    filter,
                ) {
                    println!(
                        "[cfl-diagnose] worst particle={i} material={} binding_term={term} dt={dt:.6} (chosen sub_dt={sub_dt:.6})",
                        self.particles.material_id[i]
                    );
                }
            }
            // Diagnostic, see `transfer::diagnose_particle_node_material_
            // sources`: opt-in through `EMERGE_TRACK_PARTICLE_NODES` in a
            // `research-diagnostics` build. Tests whether a tracked
            // particle's runaway velocity comes from a neighbour of another
            // material sharing its P2G/G2P nodes, which same-material
            // neighbour counts cannot see.
            if let Some(spec) = crate::diagnostics::research_switch("EMERGE_TRACK_PARTICLE_NODES")
                && let Ok(tracked_index) = spec.parse::<usize>()
                && tracked_index < self.active_count
            {
                let breakdowns = crate::transfer::diagnose_particle_node_material_sources(
                    &self.particles,
                    &self.materials,
                    sub_dt,
                    self.config.grid_res,
                    self.active_count,
                    tracked_index,
                );
                for b in &breakdowns {
                    let mut sources = String::new();
                    for s in &b.sources {
                        sources.push_str(&format!(
                            " mat{}[mass={:.4} adv=({:.3},{:.3}) stress=({:.4},{:.4})]",
                            s.material_id,
                            s.mass,
                            s.advective_momentum.x,
                            s.advective_momentum.y,
                            s.stress_momentum.x,
                            s.stress_momentum.y
                        ));
                    }
                    println!(
                        "[node-track p={tracked_index}] cell=({},{}) w={:.4} real_v=({:.3},{:.3}) real_mass={:.4} |{sources}",
                        b.cell_pos.x,
                        b.cell_pos.y,
                        b.tracked_particle_weight,
                        b.real_velocity.x,
                        b.real_velocity.y,
                        b.real_mass,
                    );
                }
            }
            // Sticky fine-substep hold (`fluid_sticky_fine_dt`'s doc) -- caps
            // the ordinary CFL result while a recent retry's hold is still active,
            // so a sustained near-wall compression event doesn't relax back to a
            // too-coarse dt on the very next substep. A no-op read (`None`) for
            // every scene that never enables `fluid_step_retry_enabled`.
            let sub_dt = match self.fluid_sticky_fine_dt {
                Some((held_dt, _)) => sub_dt.min(held_dt),
                None => sub_dt,
            };
            assert!(
                sub_dt.is_finite() && sub_dt > 0.0 && remaining - sub_dt < remaining,
                "adaptive timestep cannot advance the requested simulation time; state requires a smaller representable timestep"
            );
            let actual_dt = self.do_substep_with_retry(sub_dt);
            // The retry loop leaves only its final attempt in `pending`; fold
            // it into the public diagnostic report here, after acceptance.
            #[cfg(any(test, feature = "research-diagnostics"))]
            if let Some(diagnostic) = &mut self.boundary_impulse_diagnostic {
                diagnostic.accept_pending();
            }
            remaining -= actual_dt;
            self.last_step_dt = actual_dt;
            substeps_taken += 1;
            // Diagnostic, see `transfer::diagnose_particle_divergence_
            // decomposition` and `pending_divergence_diagnostic`. Read after
            // `do_substep_with_retry` returns: each retry attempt overwrites
            // the field, so it holds the accepted attempt and its `dt`.
            if let Some((
                tracked_index,
                trace_translation,
                trace_affine,
                trace_stress,
                trace_final,
                dt_used,
            )) = self.pending_divergence_diagnostic.take()
            {
                println!(
                    "[divergence-decomp p={tracked_index}] tr(C) translation={trace_translation:.6} \
                     affine={trace_affine:.6} stress={trace_stress:.6} pre_total={:.6} \
                     final={trace_final:.6} delta_final_minus_pre={:.6} dt_used={dt_used:.6}",
                    trace_translation + trace_affine + trace_stress,
                    trace_final - (trace_translation + trace_affine + trace_stress),
                );
            }
            // Decay the hold by one substep, regardless of whether THIS substep
            // needed a fresh retry -- see the field's doc for why persistence
            // across several substeps (not just the one that triggered it) is the
            // real fix.
            if let Some((held_dt, remaining_holds)) = self.fluid_sticky_fine_dt {
                self.fluid_sticky_fine_dt = if remaining_holds > 1 {
                    Some((held_dt, remaining_holds - 1))
                } else {
                    None
                };
            }
        }
        self.last_substeps = substeps_taken;
        // `max_substeps_per_step` is a per-frame work budget: a CFL collapse
        // must not make one step() call take seconds. If it runs out before
        // `remaining` reaches zero, the simulation time not advanced is
        // reported instead of discarded silently or looped for.
        self.last_sim_time_dropped = remaining.max(0.0);
        // A strict fluid panics instead of dropping time: strict WC-MPM
        // forbids hidden corner cuts (like `check_j_range`/
        // `assert_owned_deformation_state`), and advancing less than the
        // requested dt would break the conservation it promises. Other
        // materials tolerate a reported drop. Still bounded, not an
        // unbounded loop.
        if self.last_sim_time_dropped > 0.0 && self.materials.any_owns_deformation_volume_state() {
            panic!(
                "strict WC-MPM fluid could not advance the full requested dt within \
                 max_substeps_per_step={} ({} of {} simulated time units dropped) -- a genuine \
                 CFL/retry instability, not a false alarm: raise max_substeps_per_step, fix the \
                 underlying instability (relaxation, near-wall CFL, retry threshold), or reduce \
                 the material's stiffness, rather than accepting a silently short-advanced step",
                self.config.max_substeps_per_step, self.last_sim_time_dropped, self.config.dt
            );
        }
        // Lazy: only mark the spatial hash stale here (rebuilding it every
        // step measured 16.4% of a step); the first query after the step
        // (`particles_near`/`count_near`/`particles_knn`/`region_state`)
        // rebuilds it through `ensure_spatial_hash_fresh`.
        // Diffusion operators, applied ONCE for all the time this step
        // advanced -- see `do_substep`'s own note for the stability
        // derivation. Runs after the substep loop so temperature/scalar
        // fields see this step's final particle state.
        let t_diff = std::time::Instant::now();
        let diffusion_dt = std::mem::take(&mut self.pending_diffusion_dt);
        if diffusion_dt > 0.0 {
            // Each operator sub-cycles to this fraction of its own stable
            // step (`material_cfl_coefficient`'s definition), whatever time
            // it is handed.
            let fraction = self.config.material_cfl_coefficient;
            if let Some(thermal) = &mut self.thermal {
                thermal.stability_fraction = fraction;
                thermal.apply(
                    &mut self.particles,
                    diffusion_dt,
                    self.config.slice_thickness_m,
                );
            }
            for field in &mut self.scalar_fields {
                field.stability_fraction = fraction;
                field.apply(&mut self.particles, diffusion_dt, &self.materials);
            }
        }
        self.last_timing.thermal_us += t_diff.elapsed().as_micros() as u64;

        let t_hash = std::time::Instant::now();
        self.spatial_hash_dirty.set(true);
        self.last_timing.spatial_hash_us = t_hash.elapsed().as_micros() as u64;
        self.last_timing.total_us = step_start.elapsed().as_micros() as u64;
        self.frame_index = self.frame_index.saturating_add(1);
    }

    /// Wraps `do_substep` with the preflight/retry check for strict WC-MPM
    /// fluids -- see `SimConfig::fluid_step_retry_enabled`'s doc for the full
    /// derivation and the empirical evidence this is a genuine, convergent
    /// stability limit (not a hidden clamp masking a bug). Returns the dt actually
    /// committed, which may be smaller than requested if retries fired -- callers
    /// must advance `remaining` by the RETURNED value, not the original `sub_dt`.
    ///
    /// Fast path (a plain `self.do_substep(sub_dt)` call) whenever the feature is
    /// disabled (the default) or no registered material owns strict
    /// deformation/volume state.
    fn do_substep_with_retry(&mut self, requested_dt: f32) -> f32 {
        if !self.config.fluid_step_retry_enabled
            || !self.materials.any_owns_deformation_volume_state()
        {
            self.do_substep(requested_dt);
            return requested_dt;
        }
        // Engineering safety cap on retry attempts (same category as
        // `max_substeps_per_step` itself -- bounds worst-case cost, not a physics
        // parameter). 16 halvings reaches ~65000x finer than the original
        // CFL-chosen dt.
        const FLUID_STEP_RETRY_LIMIT: u32 = 16;
        // How many FURTHER substeps hold the fine dt once a retry fires -- see
        // `fluid_sticky_fine_dt`'s doc for why a one-off retry alone doesn't
        // work. An engineering constant (how long to keep paying the extra cost
        // after the LAST sign of trouble), not a physics parameter.
        const STICKY_HOLD_SUBSTEPS: u32 = 20;
        let admissible_ln_j_change = self.config.fluid_step_retry_threshold;
        let mut sub_dt = requested_dt;
        for attempt in 0..=FLUID_STEP_RETRY_LIMIT {
            let t_snapshot = std::time::Instant::now();
            let snapshot = self.particles.clone();
            self.last_timing.retry_snapshot_us += t_snapshot.elapsed().as_micros() as u64;
            self.do_substep(sub_dt);
            let mut worst_ln_j_change = 0.0f32;
            // `worst_ln_j_change` misses a substep that blows up a particle's
            // velocity before its J catches up (measured: kinetic energy from
            // ~19 to 33 million between two substeps with J flat at 3.212).
            // The pressure projection acts on the grid after
            // `choose_substep_dt` has committed to a dt from the previous
            // state, so a particle can move further than the CFL allows in one
            // substep. This checks that reactively, with `cfl_bound`'s
            // velocity term (`cfl_coefficient*grid_cell_size/max_speed`).
            let mut worst_speed = 0.0f32;
            // Also check the absolute band: many small, same-signed changes,
            // none inadmissible relative to the previous substep, can walk J
            // past `assert_owned_deformation_state`'s [j_min, j_max] over a
            // frame's ~150 substeps, so retry would never engage. Leaving
            // the band triggers a retry at a finer dt (held through
            // `fluid_sticky_fine_dt`), and if `FLUID_STEP_RETRY_LIMIT` runs
            // out, the exhaustion backstop below clamps J to `config.j_max`.
            let mut worst_j_out_of_bounds = false;
            for i in 0..self.active_count {
                if !self
                    .materials
                    .get(self.particles.material_id[i])
                    .owns_deformation_volume_state()
                {
                    continue;
                }
                let old_j = snapshot.volume[i] / snapshot.initial_volume[i];
                let new_j = self.particles.volume[i] / self.particles.initial_volume[i];
                let ln_j_change = (new_j / old_j).ln().abs();
                worst_ln_j_change = worst_ln_j_change.max(ln_j_change);
                worst_speed = worst_speed.max(self.particles.v[i].length());
                // Not checked against the material's own `volume_ratio_min/
                // max`: `new_j` is post-`update_particle`, which already clamps
                // to those bounds, so such a check could never fire.
                if new_j < self.config.j_min || new_j > self.config.j_max {
                    worst_j_out_of_bounds = true;
                }
            }
            let cfl_safe_speed = self.config.cfl_coefficient * self.config.grid_cell_size / sub_dt;
            let admissible = worst_ln_j_change <= admissible_ln_j_change
                && worst_speed <= cfl_safe_speed
                && !worst_j_out_of_bounds;
            if admissible || attempt == FLUID_STEP_RETRY_LIMIT {
                // Backstop when retries are exhausted with `worst_ln_j_change`
                // still above threshold: a strict fluid's pre-P2G pass only
                // asserts (`assert_owned_deformation_state`), trusting this
                // loop, and an extreme but finite state (J = 0.004, v =
                // 800,000+) passes that assert, reaches the next P2G and
                // drives its CFL scan to a sub-ULP dt. So the material-
                // agnostic `project_particle_state_to_admissible` (already used
                // for other materials via `project_invalid_state`) runs here,
                // only on exhaustion.
                if attempt == FLUID_STEP_RETRY_LIMIT && !admissible {
                    // `project_particle_state_to_admissible` only catches
                    // non-finite velocity; after an exhausted retry it reached
                    // ~9.5e6 cells/s. Cap it where `cfl_bound`'s velocity term
                    // (`cfl_coefficient*grid_cell_size/max_speed`) still
                    // returns at least `min_dt`, which needs the
                    // `cfl_coefficient` factor: without it the next frame got
                    // `cfl_coefficient*min_dt`, below `min_dt`.
                    let max_representable_speed = self.config.cfl_coefficient
                        * self.config.grid_cell_size
                        / self.config.min_dt;
                    for i in 0..self.active_count {
                        if self
                            .materials
                            .get(self.particles.material_id[i])
                            .owns_deformation_volume_state()
                        {
                            project_particle_state_to_admissible(
                                &mut self.particles,
                                i,
                                &self.config,
                            );
                            let speed = self.particles.v[i].length();
                            if speed > max_representable_speed {
                                self.particles.v[i] *= max_representable_speed / speed;
                            }
                        }
                    }
                }
                // Only a retry that found an admissible finer dt on its own
                // earns the sticky hold. On exhaustion the state was repaired
                // by the backstop above and deserves a fresh CFL scan, not the
                // failed sub_dt (16 halvings down, near 2^-16 of the request,
                // small enough to fail the next frame's `remaining - sub_dt <
                // remaining` representability check).
                if admissible && sub_dt < requested_dt {
                    // At least one halving was needed to reach an admissible
                    // state -- hold this fine dt for subsequent substeps too,
                    // not just this one (refreshes/extends an existing hold).
                    //
                    // Floored at `min_dt`: admissibility gets easier as
                    // `sub_dt` shrinks (`cfl_safe_speed` grows without bound,
                    // `worst_ln_j_change` shrinks), so a violent substep can
                    // succeed far below `min_dt`, and holding that for
                    // `STICKY_HOLD_SUBSTEPS` (possibly into the next frame)
                    // poisoned later CFL scans. The floor only limits the
                    // hold; `sub_dt.min(held_dt)` still lets a fresh scan go
                    // below `min_dt` when the state needs it.
                    self.fluid_sticky_fine_dt =
                        Some((sub_dt.max(self.config.min_dt), STICKY_HOLD_SUBSTEPS));
                }
                return sub_dt;
            }
            self.particles = snapshot;
            sub_dt *= 0.5;
        }
        sub_dt
    }

    fn do_substep(&mut self, sub_dt: f32) {
        // Project invalid particle state before it can corrupt the grid scatter.
        // Running pre-P2G (not post) means a bad particle from a previous substep is
        // fixed before its momentum enters the grid -- no NaN cascade possible.
        let t_pre = std::time::Instant::now();
        // The two branches below are NOT the same kind of work, and only one
        // of them has to run every substep:
        //
        // * `project_particle_state_to_admissible` MUTATES -- it repairs a bad
        //   particle before its momentum can enter the grid scatter, which is
        //   exactly the "no NaN cascade possible" guarantee this pre-P2G
        //   placement exists for. Stays per-substep, unconditionally.
        // * `assert_owned_deformation_state` only VALIDATES -- it panics on an
        //   inconsistent strict-fluid state and changes nothing otherwise. The
        //   per-substep safety for those materials is enforced inside their
        //   own `update_particle` (the J admissibility assert), so this scan
        //   is a second opinion. Running it once per FRAME still catches any
        //   corruption within that frame, just at the frame boundary.
        //
        // Measured: this scan was `project_us` ~2900 us of a ~29000 us step
        // (10%), and in a fluid-only scene every particle takes the assert
        // branch.
        let validate_owned_state = self.substep_index_in_frame == 0;
        for i in 0..self.active_count {
            let material = self.materials.get(self.particles.material_id[i]);
            if material.owns_deformation_volume_state() {
                if validate_owned_state {
                    assert_owned_deformation_state(&self.particles, i, &self.config);
                }
            } else if self.config.project_invalid_state
                && project_particle_state_to_admissible(&mut self.particles, i, &self.config)
            {
                self.last_j_projection_count += 1;
            }
        }
        self.last_timing.project_us += t_pre.elapsed().as_micros() as u64;

        // ── P2G ──────────────────────────────────────────────────────────────
        let t0 = std::time::Instant::now();
        self.grid.clear();
        if self.config.spatial_sort_enabled {
            scatter_particles_to_grid_sorted(
                &self.particles,
                &mut self.grid,
                &self.materials,
                sub_dt,
                self.active_count,
                &self.cached_spatial_sort_order,
            );
        } else {
            scatter_particles_to_grid(
                &self.particles,
                &mut self.grid,
                &self.materials,
                sub_dt,
                self.active_count,
            );
        }
        // Second particle pass for the contact-normal point cloud (see
        // `gather_contact_point_cloud` doc) -- must run after the above, since
        // contact-active nodes aren't fully known until every grip particle's mass
        // has been scattered. No-op when `contact_group` is unused anywhere.
        gather_contact_point_cloud(
            &self.particles,
            &mut self.grid,
            &self.materials,
            self.active_count,
        );
        // Rod -> grid scatter, same P2G pass, same shared `Grid` -- BEFORE the
        // wake pass below so a rod touching settled sand/fluid wakes it with
        // zero new code (the wake scan just sees active cells the rod itself
        // created). No-op for every scene that never calls add_rod/with_rod.
        // Sleeping rods skip this entirely (see `Rod::sleeping` doc) -- they
        // neither scatter mass/momentum nor self-trigger their own wake check
        // below; they're woken only by external activity.
        //
        // Grain -> grid scatter, same shared `Grid`, same convention as rods
        // below -- see `grains::coupling`'s doc. No-op for every scene
        // that never calls `add_grain_population`. Before the rods, so each
        // rod can tell whether it touches other matter.
        for population in &self.grain_populations {
            scatter_grains_to_grid(population, &mut self.grid);
        }
        for rod in &mut self.rods {
            if !rod.sleeping && !rod.use_implicit_integration {
                rod.touching_other_matter = rod_touches_grid_mass(&rod.points, &self.grid);
                scatter_rod_to_grid(&rod.points, &mut self.grid);
            }
        }
        self.last_timing.p2g_us += t0.elapsed().as_micros() as u64;

        // Structural-boundary ledger (research diagnostic). Reconstructs the same
        // particle P2G terms with this retry attempt's exact `sub_dt`; the
        // outer loop accepts only the final attempt's pending ledger.
        #[cfg(any(test, feature = "research-diagnostics"))]
        if self.boundary_impulse_diagnostic.is_some() {
            let components = crate::transfer::diagnose_grid_p2g_components(
                &self.particles,
                &self.materials,
                sub_dt,
                self.config.grid_res,
                self.active_count,
            );
            let ledger = super::boundary_diagnostics::begin_ledger(self, sub_dt, &components);
            let Some(diagnostic) = &mut self.boundary_impulse_diagnostic else {
                unreachable!("just checked is_some() above")
            };
            diagnostic.pending = Some(ledger);
        }

        // Diagnostic, `EMERGE_TRACK_BOUNDARY_BIAS` (see
        // `transfer::diagnose_particle_divergence_decomposition`): right
        // after the P2G scatter, with the `sub_dt` it used. Stores the
        // pre-grid-update decomposition; `trace_final`/`dt_used` are filled
        // in after G2P below.
        if let Some(spec) = crate::diagnostics::research_switch("EMERGE_TRACK_BOUNDARY_BIAS")
            && let Ok(tracked_index) = spec.parse::<usize>()
            && tracked_index < self.active_count
        {
            let (trace_translation, trace_affine, trace_stress) =
                crate::transfer::diagnose_particle_divergence_decomposition(
                    &self.particles,
                    &self.materials,
                    sub_dt,
                    self.active_count,
                    tracked_index,
                    self.config.apic_blend,
                );
            self.pending_divergence_diagnostic = Some((
                tracked_index,
                trace_translation,
                trace_affine,
                trace_stress,
                0.0,
                sub_dt,
            ));
        }

        // Wake any sleeping particle whose kernel overlaps a MEANINGFULLY active
        // grid cell. This propagates activity from moving regions into
        // neighbouring sleeping ones without a separate O(N) scan -- we only
        // visit the sleeping partition.
        //
        // Must gate on the neighbour's actual velocity, not just `cell_is_active`
        // (has ANY mass, regardless of speed): a body that scatters into the grid
        // every substep but never itself goes to sleep (e.g. a rod gated awake by
        // ongoing `Growth`, see `Rod::is_growing`) would otherwise count as
        // permanent "activity" for every neighbour touching its cells, even once
        // its own residual speed is tiny -- causing spurious sleep/wake cycling.
        // Requiring the neighbour's actual velocity (momentum/mass -- `cell.momentum`
        // is still RAW scattered momentum at this point in the substep, before
        // `update_velocities` normalizes it) to exceed THIS body's own sleep
        // threshold gives the same hysteresis a "settled" body already assumes:
        // something merely present but equally quiescent shouldn't wake it up.
        let wake_speed_sq = self.config.sleep_threshold * self.config.sleep_threshold;
        if self.active_count < self.particles.len() {
            let total = self.particles.len();
            self.scratch_indices.clear();
            for i in self.active_count..total {
                let x = self.particles.x[i];
                let base = crate::grid::kernel::quadratic_weights(x).base_cell;
                'outer: for gx in 0i32..3 {
                    for gy in 0i32..3 {
                        let cell = base + glam::IVec2::new(gx - 1, gy - 1);
                        let mass = self.grid.mass_at(cell);
                        if mass <= 0.0 {
                            continue;
                        }
                        let speed_sq = (self.grid.velocity_at(cell) / mass).length_squared();
                        if speed_sq > wake_speed_sq {
                            self.scratch_indices.push(i);
                            break 'outer;
                        }
                    }
                }
            }
            // Index directly -- wake_particle doesn't touch scratch_indices, capacity preserved.
            for j in 0..self.scratch_indices.len() {
                let i = self.scratch_indices[j];
                self.wake_particle(i);
            }
        }

        // Same wake test, rod granularity: a sleeping rod's own (frozen) points
        // didn't scatter above, so any overlap found here comes from genuinely
        // external activity (another body's P2G, or another awake rod) -- the
        // particle wake pass's no-self-trigger property. Also wakes unconditionally
        // on an active push, since `push_strength > 0` is a direct move request the
        // grid-activity test can't see yet (nothing has touched the grid near it).
        // Requires actual velocity over rod_sleep_threshold, not merely "some mass
        // present" -- otherwise a permanently-active neighbour (e.g. a growing root
        // that never sleeps) keeps waking every sleeping rod touching its cells.
        // Must also check wind_velocity, not just push_strength: wind is
        // rod-internal and never touches the grid, so it's the only way a sleeping
        // rod can wake from wind alone (same as the implicit-rod path above).
        let rod_wake_speed_sq = self.config.rod_sleep_threshold * self.config.rod_sleep_threshold;
        for rod in &mut self.rods {
            if !rod.sleeping {
                continue;
            }
            let has_push = rod.push_strength > 0.0 && rod.push_center.is_some();
            let has_wind = rod.wind_velocity.length_squared() > 0.0;
            let touched = has_push
                || has_wind
                || rod.points.x.iter().any(|&x| {
                    let base = crate::grid::kernel::quadratic_weights(x).base_cell;
                    (0i32..3).any(|gx| {
                        (0i32..3).any(|gy| {
                            let cell = base + glam::IVec2::new(gx - 1, gy - 1);
                            let mass = self.grid.mass_at(cell);
                            if mass <= 0.0 {
                                return false;
                            }
                            let speed_sq = (self.grid.velocity_at(cell) / mass).length_squared();
                            speed_sq > rod_wake_speed_sq
                        })
                    })
                });
            if touched {
                rod.sleeping = false;
                rod.below_threshold_time = 0.0;
            }
        }

        // ── Grid update ───────────────────────────────────────────────────────
        let t1 = std::time::Instant::now();
        // ASFLIP (SimConfig::asflip_blend, Fei et al. 2021) needs the grid's velocity
        // right after P2G's own momentum normalization -- before THIS substep's gravity,
        // boundary conditions, or contact resolution modify it -- to compute G2P's FLIP
        // residual. Cundall damping (SimConfig::cundall_damping) needs the exact same
        // pre-force reference point -- see `Grid::apply_cundall_damping`'s doc --
        // so both features share one snapshot. Taking it only when either feature is
        // enabled keeps every other scene on the exact original single-call path (zero
        // cost, zero behavior change).
        let pre_force_snapshot =
            if self.config.asflip_blend > 0.0 || self.config.cundall_damping > 0.0 {
                self.grid.normalize_velocities();
                self.grid.normalize_friction();
                // A prescribed particle anchor is an essential boundary on
                // the grid velocity field. Enforce it before the pre-force
                // snapshot so FLIP/Cundall never treat forbidden anchor
                // motion as a previous velocity.
                self.grid.apply_pinned_node_constraints();
                let snapshot = self.grid.snapshot_velocities();
                self.grid.apply_gravity(sub_dt, self.config.gravity);
                Some(snapshot)
            } else {
                self.grid.update_velocities(sub_dt, self.config.gravity);
                None
            };
        let grid_res = self.grid.resolution();
        #[cfg(any(test, feature = "research-diagnostics"))]
        if let Some(diagnostic) = &mut self.boundary_impulse_diagnostic
            && let Some(ledger) = &mut diagnostic.pending
        {
            super::boundary_diagnostics::capture_before_wall(&self.grid, ledger);
        }
        for boundary in &self.boundaries {
            apply_boundary_conditions_to_grid(&mut self.grid, grid_res, boundary.as_ref());
        }
        // First law at a rubbing wall: the kinetic energy Coulomb friction
        // just removed becomes heat in the matter that rubbed, instead of
        // vanishing. Gated on a node having actually dissipated something,
        // so a frictionless scene pays one boolean.
        if self.grid.has_friction_heat() {
            crate::spacetime::transfer::gather_friction_heat_to_particles(
                &self.grid,
                &mut self.particles,
                &self.materials,
                self.config.dx_meters,
                &mut self.friction_heat_debt,
            );
        }
        #[cfg(any(test, feature = "research-diagnostics"))]
        if let Some(diagnostic) = &mut self.boundary_impulse_diagnostic
            && let Some(ledger) = &mut diagnostic.pending
        {
            super::boundary_diagnostics::apply_experimental_lower_wall(
                &mut self.grid,
                grid_res,
                self.config.boundary_thickness,
                diagnostic.mode,
                ledger,
            );
            ledger.grid_momentum_after_wall = self.grid.velocity_field_momentum_sum();
            ledger.wall_impulse =
                ledger.grid_momentum_after_wall - ledger.grid_momentum_before_wall;
            ledger.grid_momentum_after_wall_f64 = self.grid.velocity_field_momentum_sum_f64();
            ledger.wall_impulse_f64 =
                ledger.grid_momentum_after_wall_f64 - ledger.grid_momentum_before_wall_f64;
        }
        // Multi-field frictional contact (Bardenhagen 2001). It is rejected for
        // strict WC-MPM liquids above; ordinary solid/contact scenes retain this
        // separate constraint solve. No-op when no particle uses contact groups.
        self.grid.resolve_contact(
            sub_dt,
            self.config.gravity,
            self.config.contact_friction,
            self.config.grid_cell_size,
            self.config.material_cfl_coefficient,
            self.contact_grip.as_deref(),
        );
        #[cfg(any(test, feature = "research-diagnostics"))]
        if let Some(diagnostic) = &mut self.boundary_impulse_diagnostic
            && let Some(ledger) = &mut diagnostic.pending
        {
            let after_contact = self.grid.velocity_field_momentum_sum();
            ledger.contact_impulse = after_contact - ledger.grid_momentum_after_wall;
            let after_contact_f64 = self.grid.velocity_field_momentum_sum_f64();
            ledger.contact_impulse_f64 = after_contact_f64 - ledger.grid_momentum_after_wall_f64;
        }
        // Two-phase mixture coupling (Tampubolon et al. 2017). Strict WC-MPM
        // liquids reject this separate porous-medium model above. No-op when unused -- see
        // `Grid::resolve_mixture_coupling` doc.
        self.grid.resolve_mixture_coupling(
            sub_dt,
            self.config.gravity,
            self.config.mixture_drag_coefficient,
            self.config.grid_cell_size,
            self.config.mixture_pressure_iterations,
        );
        // Strict (single-phase) fluid incompressibility pressure projection
        // (`grid::pressure`, see `SimConfig::fluid_pressure_iterations`). Runs
        // after boundary/contact/mixture resolution, so it corrects the
        // post-gravity, post-wall velocity field, and before G2P. No-op at
        // `fluid_pressure_iterations == 0`, the default.
        //
        // Called `fluid_pressure_iterations` times in a row: outer corrector
        // passes, as in PISO/SIMPLE-family solvers. One projection is a
        // first-order splitting of a violent state; re-measuring the residual
        // divergence and correcting again gets much closer to divergence-free.
        // A single exact solve (Gauss-Seidel gave the same answer) was the true
        // solution for an already extreme input, so the fix is to keep the
        // input from getting that extreme, which repeated correction does.
        let t_pressure = std::time::Instant::now();
        for _ in 0..self.config.fluid_pressure_iterations {
            self.grid
                .project_fluid_incompressibility(self.config.grid_cell_size, 1);
            // The projection's gradient correction
            // (`Grid::project_fluid_incompressibility`) writes `cell.momentum`
            // without knowing the wall, and pushed velocity back through a
            // wall already satisfied by `apply_boundary_conditions_to_grid`.
            // Measured: a particle resting on the floor read a sustained
            // positive velocity-gradient trace for over 100 frames, J from ~1
            // to >1,000,000, while `grad_p` stayed under 50 and `C` under 200
            // (5, 10 or 30 GS sweeps made no difference). The boundary pass
            // reruns after every corrector iteration, so the wall has the
            // last word each time.
            for boundary in &self.boundaries {
                apply_boundary_conditions_to_grid(&mut self.grid, grid_res, boundary.as_ref());
            }
        }
        self.last_timing.pressure_us += t_pressure.elapsed().as_micros() as u64;
        // Cundall damping (see the snapshot comment above) -- applied LAST, after
        // gravity/boundary/contact/mixture have all had their say, so it damps the
        // real NET result of everything this substep, not just one contributor.
        if self.config.cundall_damping > 0.0
            && let Some(snapshot) = &pre_force_snapshot
        {
            self.grid
                .apply_cundall_damping(snapshot, self.config.cundall_damping);
        }
        // Last word before G2P: forces, contact, pressure projection and
        // damping may all have modified these nodes since the first anchor
        // application. Overwriting them here supplies the actual Dirichlet
        // reaction that the old post-G2P-only particle reset could not.
        self.grid.apply_pinned_node_constraints();
        #[cfg(any(test, feature = "research-diagnostics"))]
        if let Some(diagnostic) = &mut self.boundary_impulse_diagnostic
            && let Some(ledger) = &mut diagnostic.pending
        {
            let before_g2p = self.grid.velocity_field_momentum_sum();
            ledger.other_grid_impulse =
                before_g2p - ledger.grid_momentum_after_wall - ledger.contact_impulse;
            ledger.grid_momentum_before_g2p_f64 = self.grid.velocity_field_momentum_sum_f64();
            ledger.other_grid_impulse_f64 = ledger.grid_momentum_before_g2p_f64
                - ledger.grid_momentum_after_wall_f64
                - ledger.contact_impulse_f64;
        }
        #[cfg(any(test, feature = "research-diagnostics"))]
        if self.boundary_impulse_diagnostic.is_some() {
            let mass_closure = super::boundary_diagnostics::measure_g2p_mass_closure(self);
            let Some(diagnostic) = &mut self.boundary_impulse_diagnostic else {
                unreachable!("just checked is_some() above")
            };
            if let Some(ledger) = &mut diagnostic.pending {
                ledger.g2p_mass_closure = mass_closure;
            }
        }
        self.last_timing.grid_update_us += t1.elapsed().as_micros() as u64;

        // ── G2P ──────────────────────────────────────────────────────────────
        let t2 = std::time::Instant::now();
        let g_len = self.active_count.min(self.granular_fluidity_g.len());
        let cosserat_len = self.active_count.min(self.cosserat_curvature.len());
        gather_grid_to_particles(
            &mut self.particles,
            &self.grid,
            sub_dt,
            self.config.gravity,
            &self.boundaries,
            &self.materials,
            G2PParams {
                apic_blend: self.config.apic_blend,
                active_count: self.active_count,
                asflip_blend: self.config.asflip_blend,
                boundary_thickness: self.config.boundary_thickness,
                // If only `cundall_damping` is enabled (asflip_blend still 0.0),
                // G2P still takes the `Some` branch and computes the pre-force
                // gather: harmless (asflip_blend = 0.0 zeroes its contribution)
                // but not free. One snapshot for both features.
                pre_force_snapshot: pre_force_snapshot.as_ref(),
                // Computed at the END of the PREVIOUS substep, by the
                // granular-fluidity pass alongside thermal/scalar diffusion
                // below -- same one-substep-lag convention those already
                // use. Empty when no `GranularFluidityField` is configured
                // for this scene (every existing scene) -- every read falls
                // back to 0.0, `ParticleUpdateCtx::nonlocal_fluidity`'s own
                // real-rest-state default.
                nonlocal_fluidity: &self.granular_fluidity_g[..g_len],
                // Same one-substep-lag convention, computed by the Cosserat
                // pass alongside granular fluidity below. Empty when no
                // `CosseratField` is configured -- every read falls back to
                // `Vec2::ZERO`, `ParticleUpdateCtx::cosserat_curvature`'s own
                // real-rest-state default.
                cosserat_curvature: &self.cosserat_curvature[..cosserat_len],
            },
        );
        #[cfg(any(test, feature = "research-diagnostics"))]
        if self.boundary_impulse_diagnostic.is_some() {
            let particle_momentum_end = super::boundary_diagnostics::particle_momentum(self);
            let particle_momentum_end_f64 =
                super::boundary_diagnostics::particle_momentum_f64(self);
            let Some(diagnostic) = &mut self.boundary_impulse_diagnostic else {
                unreachable!("just checked is_some() above")
            };
            if let Some(ledger) = &mut diagnostic.pending {
                ledger.particle_momentum_end = particle_momentum_end;
                ledger.residual = ledger.particle_momentum_end
                    - ledger.particle_momentum_start
                    - ledger.gravity_impulse
                    - ledger.wall_impulse
                    - ledger.contact_impulse
                    - ledger.other_grid_impulse;
                ledger.particle_momentum_end_f64 = particle_momentum_end_f64;
                ledger.residual_f64 = ledger.particle_momentum_end_f64
                    - ledger.particle_momentum_start_f64
                    - ledger.gravity_impulse_f64
                    - ledger.wall_impulse_f64
                    - ledger.contact_impulse_f64
                    - ledger.other_grid_impulse_f64;
                ledger.g2p_transfer_residual_f64 =
                    ledger.particle_momentum_end_f64 - ledger.grid_momentum_before_g2p_f64;
                ledger.g2p_unexplained_residual_f64 =
                    ledger.g2p_transfer_residual_f64 - ledger.g2p_mass_closure.delta_p_sum;
            }
        }
        // Diagnostic (see the P2G-side insertion above and
        // `pending_divergence_diagnostic`): fills in the final `tr(C)` G2P
        // just wrote onto the tracked particle.
        if let Some((tracked_index, trace_translation, trace_affine, trace_stress, _, dt_used)) =
            self.pending_divergence_diagnostic
        {
            let c_final = self.particles.velocity_gradient[tracked_index];
            let trace_final = c_final.x_axis.x + c_final.y_axis.y;
            self.pending_divergence_diagnostic = Some((
                tracked_index,
                trace_translation,
                trace_affine,
                trace_stress,
                trace_final,
                dt_used,
            ));
        }
        // Grid -> rod gather (this rod's own G2P) and the rod's motion over
        // the substep: what the grid did to the rod (gravity, the exchange
        // with particles) is spread over the substep, and the rod advances
        // in its own sub-steps within its own stable step, with its own
        // internal, wind and push forces (`coupling::advance_rod`). Gravity is
        // not applied again. A stiff rod stays coupled without shrinking the
        // substep.
        for rod in &mut self.rods {
            if !rod.sleeping && !rod.use_implicit_integration {
                let external = gather_grid_to_rod(&rod.points, &self.grid, sub_dt);
                advance_rod(
                    &mut rod.points,
                    &rod.material,
                    RodForceParams {
                        wind_velocity: rod.wind_velocity,
                        wind_drag_coeff: rod.wind_drag_coeff,
                        push_center: rod.push_center,
                        push_strength: rod.push_strength,
                        push_radius: rod.push_radius,
                        dx_meters: self.config.dx_meters,
                        dt: sub_dt,
                        stability_fraction: self.config.material_cfl_coefficient,
                    },
                    &external,
                );
            }
        }
        // Grid -> grain gather: velocity only, no position advance (unlike
        // rods above), see `grains::coupling::gather_grid_to_grains` for why
        // contact is resolved before integration. APIC, through the same
        // `apic_blend` as particles: pure PIC froze a jittered column collapse
        // near its lattice (Jiang et al. 2015's "PIC causes sand to clump
        // together"), pure FLIP was noisy and dt-unstable. `asflip_blend`/
        // `pre_force_snapshot` are shared with `gather_grid_to_particles`;
        // `asflip_blend` defaults to 0.0.
        for population in &mut self.grain_populations {
            gather_grid_to_grains(
                population,
                &self.grid,
                self.config.apic_blend,
                self.config.asflip_blend,
                pre_force_snapshot.as_ref(),
            );
        }
        self.last_timing.g2p_us += t2.elapsed().as_micros() as u64;

        // ── Force fields ──────────────────────────────────────────────────────
        // External body force fields: v += dt × acceleration(p) per particle.
        // Applied after G2P so each field sees the fully gathered particle state.
        // No velocity clamp follows this pass: the next adaptive substep is
        // selected from the actual field-updated state. A nonphysical force
        // must not be disguised as numerical stabilization.
        // prepare() is called first so stateful fields (e.g. Barnes-Hut tree) can
        // rebuild their internal state from the current particle snapshot.
        if !self.force_fields.is_empty() {
            let t3 = std::time::Instant::now();
            let mut fields = std::mem::take(&mut self.force_fields);
            for (_, field) in &mut fields {
                field.prepare(&self.particles);
            }
            for i in 0..self.active_count {
                // Dirichlet/kinematic anchor (`Particle::pinned`): must stay at v=0,
                // matching G2P's own unconditional pinned branch just before this pass.
                // Force fields ran AFTER G2P with no pinned check, silently un-zeroing
                // pinned particles' velocity every substep -- P2G then scatters that as
                // real momentum next substep (`scatter_particles_to_grid` doesn't special-
                // case pinned particles either, since a pinned particle's mass/stress
                // SHOULD still be felt by neighbors, just not its velocity). A supposedly-
                // fixed anchor was quietly injecting wind-driven momentum into the grid
                // every substep -- a confirmed root cause of long-horizon energy
                // injection at every pinned+force-field composition, not just this scene.
                if self.particles.pinned[i] != 0 {
                    continue;
                }
                let mut dv = Vec2::ZERO;
                for (_, field) in &fields {
                    dv += field.acceleration(&self.particles, i);
                }
                self.particles.v[i] += sub_dt * dv;
            }
            self.force_fields = fields;
            self.last_timing.fields_us += t3.elapsed().as_micros() as u64;
        }

        // A fluid is deliberately never repaired by the generic projection
        // code.  Validate again after G2P and externally supplied forces so an
        // invalid force/state is reported at the causal substep rather than
        // contaminating the next P2G scatter.
        //
        // J-range check deferred to `do_substep_with_retry`'s own loop when
        // retry is enabled (see `assert_owned_deformation_state_j_range_
        // deferred`'s doc): that loop is the only caller of `do_substep`
        // in that configuration (the retry guard at the top of `do_substep_
        // with_retry` falls back to a plain, undeferred `do_substep` call
        // otherwise), and it already re-derives this exact bound to decide
        // whether to retry at a finer dt or apply its exhaustion backstop --
        // panicking here first made that decision unreachable.
        for i in 0..self.active_count {
            if self
                .materials
                .get(self.particles.material_id[i])
                .owns_deformation_volume_state()
            {
                if self.config.fluid_step_retry_enabled {
                    assert_owned_deformation_state_j_range_deferred(
                        &self.particles,
                        i,
                        &self.config,
                    );
                } else {
                    assert_owned_deformation_state(&self.particles, i, &self.config);
                }
            }
        }

        // ── Grain contact forces ────────────────────────────────────────────────
        // Corrects velocity AND advances position -- unlike rod internal
        // forces below (position already advanced in their own gather),
        // matching the proven standalone `GrainPopulation::step`'s own
        // order. Gravity NOT reapplied here (already received via the
        // shared grid-update step) -- see
        // `grains::coupling::apply_grain_contact_forces`'s doc.
        for population in &mut self.grain_populations {
            apply_grain_contact_forces(
                population,
                sub_dt,
                &self.boundaries,
                grid_res,
                &self.grid,
                self.config.material_cfl_coefficient,
            );
        }
        // Bounded grid-mediated rotational coupling between nearby grains
        // (see `grains::micro_rotation`), replacing a scatter of `spin` into
        // the momentum grid that was exact for one grain but leaked energy
        // without bound once many spinning grains shared nodes.
        //
        // Stable by construction (checked at 1.0 and 1000x on a replay of a
        // near-instability), but it does not change the grid-coupled column
        // collapse of `tests/grains_grid_coupling.rs`: swept from 1.0 to
        // 1000.0, the final spread ratio is bit-for-bit identical. That pile
        // locks into a static frictional equilibrium within the first ~20%
        // of the run; why the grid-coupled pile is more stable than the same
        // contact law standalone is not root-caused (#28).
        const GRAIN_MICRO_ROTATION_COUPLING_MODULUS: f32 = 1.0;
        for population in &mut self.grain_populations {
            couple_grain_spin_to_local_average(
                population,
                &GrainMicroRotationConfig {
                    coupling_modulus: GRAIN_MICRO_ROTATION_COUPLING_MODULUS,
                },
                sub_dt,
            );
        }

        // ── Rod tropisms, growth and plasticity ─────────────────────────────────
        // The explicit rod already moved with its forces at the gather above.
        // No-op for every scene with no rods.
        //
        for rod in &mut self.rods {
            if rod.sleeping || rod.use_implicit_integration {
                continue;
            }
            // Root gravitropism (Porat, Rivière, Meroz 2024 -- see
            // `rod::gravitropism`): evolves the tip's rest_curvature toward
            // gravity alignment. No-op for rods that do not opt in.
            if let Some(gravitropism) = &rod.gravitropism {
                apply_gravitropism(
                    &mut rod.points,
                    gravitropism,
                    self.config.gravity,
                    &self.grid,
                    sub_dt,
                );
            }
            // Phototropism (Cholodny & Went auxin asymmetry -- see
            // `rod::gravitropism`'s "Phototropism reuses the SAME core").
            // No-op for rods that do not opt in.
            if let Some(phototropism) = &rod.phototropism {
                apply_phototropism(
                    &mut rod.points,
                    phototropism,
                    self.config.light_dir,
                    &self.grid,
                    sub_dt,
                );
            }
            // Elongation growth (Verhulst 1838 logistic law -- see
            // `rod::growth`). No-op for rods that do not opt in.
            if let Some(growth) = &mut rod.growth {
                apply_growth(
                    &mut rod.points,
                    growth,
                    &self.grid,
                    self.config.light_dir,
                    self.config.dx_meters,
                    sub_dt,
                );
            }
            // Stress-driven secondary growth (Jaffe 1973, Mattheck & Kübler
            // 1995 -- see `rod::secondary_growth`). No-op for rods that do not
            // opt in. Gated on the rod still being over-critical, so it stops
            // stiffening once no longer needed (see the other call site).
            if let Some(secondary_growth) = &rod.secondary_growth {
                let gravity_si = self.config.gravity.length() * self.config.dx_meters;
                if rod.buckling_warning(gravity_si).is_some() {
                    apply_secondary_growth(
                        &mut rod.points,
                        secondary_growth,
                        self.config.dx_meters,
                        sub_dt,
                    );
                }
            }
            // Elastic-perfectly-plastic bending (see `rod::plasticity`), after
            // this substep's biological reshaping, as in the implicit branch.
            if let Some(plasticity) = &rod.plasticity {
                apply_bending_plasticity(&mut rod.points, plasticity, self.config.dx_meters);
            }
        }

        // ── Thermal / scalar diffusion ────────────────────────────────────────
        let t4 = std::time::Instant::now();
        // Thermal / scalar diffusion are SEPARATE operators from the momentum
        // solve, so they are accumulated here and applied ONCE per `step()`
        // with the total advanced time (see `flush_diffusion_operators`),
        // each sub-cycling to its own stable step. This is ordinary operator
        // splitting at each operator's own stable rate, not an approximation
        // introduced for speed.
        //
        // Quantified for this engine's own water config (`conductivity 0.6`,
        // `rho 1000`, `c_p 4182`, `dx 0.01`): `alpha_grid = 0.0014 1/s`, so
        // `ThermalConfig::stability_dt = 1/(4*alpha) = 174 SECONDS`. The
        // acoustic CFL drives `sub_dt` to ~0.0055 s, i.e. diffusion was being
        // sub-cycled ~31,000x more often than its own stability requires;
        // even a full 0.1 s frame sits 1743x inside the limit. Live-measured
        // cost of that waste: `thermal_us` was ~7500 us of a ~34000 us step
        // (22%, the second-largest phase).
        //
        // Their stable steps are no longer folded into `choose_substep_dt`:
        // the mechanics substep never bounded this once-per-step update, so
        // the fold slowed the mechanics and protected nothing
        // (`tests/subsystem_time_steps.rs`, gate 1).
        self.pending_diffusion_dt += sub_dt;
        self.substep_index_in_frame = self.substep_index_in_frame.saturating_add(1);
        // Nonlocal Granular Fluidity (see `energy::thermodynamics::
        // granular_fluidity` module doc) -- same one-substep-lag placement
        // as thermal/scalar diffusion above: computed here from THIS
        // substep's just-updated particle stress state, read by next
        // substep's G2P via `G2PParams::nonlocal_fluidity`.
        if let Some(field) = &mut self.granular_fluidity {
            if self.granular_fluidity_g.len() < self.active_count {
                self.granular_fluidity_g.resize(self.active_count, 0.0);
            }
            field.apply(
                &self.particles,
                sub_dt,
                self.config.dx_meters,
                &mut self.granular_fluidity_g[..self.active_count],
            );
        }
        // Cosserat micro-rotation field (see `energy::thermodynamics::
        // cosserat_field` module doc) -- same one-substep-lag placement as
        // granular fluidity above: computed here from THIS substep's just-
        // updated particle velocity gradient, read by next substep's G2P via
        // `G2PParams::cosserat_curvature`.
        if let Some(field) = &mut self.cosserat {
            if self.cosserat_curvature.len() < self.active_count {
                self.cosserat_curvature
                    .resize(self.active_count, Vec2::ZERO);
            }
            if self.cosserat_omega.len() < self.active_count {
                self.cosserat_omega.resize(self.active_count, 0.0);
            }
            field.apply(
                &self.particles,
                sub_dt,
                self.config.dx_meters,
                macro_spin_from_velocity_gradient,
                &mut self.cosserat_omega[..self.active_count],
                &mut self.cosserat_curvature[..self.active_count],
            );
        }
        self.last_timing.thermal_us += t4.elapsed().as_micros() as u64;

        // ── Phase rules + sleep scoring ───────────────────────────────────────
        let t5 = std::time::Instant::now();
        // Per-substep by default (`add_phase_rule`'s documented contract). A scene
        // whose rules are thermodynamic -- which cannot change within a frame,
        // since diffusion advances once per `step()` -- can opt into
        // once-per-step via `SimConfig::phase_rules_once_per_step` and skip
        // ~17 redundant O(N) scans per frame. See that field's doc.
        let evaluate_phase_rules =
            !self.config.phase_rules_once_per_step || self.substep_index_in_frame == 0;
        if !self.phase_rules.is_empty() && evaluate_phase_rules {
            let rules = std::mem::take(&mut self.phase_rules);
            for i in 0..self.active_count {
                let p = self.particles.get(i);
                for rule in &rules {
                    if let Some(new_id) = rule(&p) {
                        // Shared with `Simulation::phase_transition` -- see
                        // `apply_phase_transition`'s doc (`solver::
                        // particles`) for the elastic-reference
                        // rebaseline this applies (the fix for a genuine
                        // fluid->solid "spring" artifact) and the
                        // material-specific-state reset it also performs.
                        self.apply_phase_transition(i, new_id);
                        break;
                    }
                }
            }
            self.phase_rules = rules;
        }
        let threshold = self.config.sleep_threshold;
        if threshold > 0.0 {
            let threshold_sq = threshold * threshold;
            self.scratch_indices.clear();
            self.scratch_indices
                .extend((0..self.active_count).filter(|&i| {
                    self.particles.activation[i] == 0.0
                        && self.particles.v[i].length_squared() < threshold_sq
                }));
            // Descending order: sleep_particle swaps i↔last_active (high end of active zone).
            // Processing high-to-low ensures each displacement lands in already-processed
            // positions, so no sleeping candidate is accidentally skipped.
            self.scratch_indices.sort_unstable_by(|a, b| b.cmp(a));
            for j in 0..self.scratch_indices.len() {
                self.sleep_particle(self.scratch_indices[j]);
            }
        }
        // Rod sleep scoring: same threshold-crossing test as particles above,
        // but scored over the WHOLE rod (max point speed) since points are
        // elastically coupled -- one point can't sleep while its neighbor
        // keeps swinging. Never sleeps mid-push (`push_strength > 0`), since
        // that's a live interaction the caller is actively driving.
        //
        // Sleep needs a sustained duration below threshold, not one instant:
        // a new rod has `v = 0` before gravity acts, would sleep on its first
        // substep, and later receive the deferred fall as one velocity spike
        // (Box2D's `b2_timeToSleep = 0.5s`, Bullet and PhysX do the same).
        //
        // The window scales with the rod's own natural period: velocity also
        // dips near zero at every swing peak, and a fixed 0.5 s froze a soft
        // blade (period ~0.53 s) several cells off vertical. Three periods
        // (`ROD_SLEEP_SETTLE_PERIODS`) clears any single peak's dwell,
        // clamped to `[ROD_SLEEP_SETTLE_MIN_SECONDS,
        // ROD_SLEEP_SETTLE_MAX_SECONDS]` so a very stiff rod keeps the
        // anti-instant-sleep floor and a very soft one does not wait forever.
        const ROD_SLEEP_SETTLE_PERIODS: f32 = 3.0;
        const ROD_SLEEP_SETTLE_MIN_SECONDS: f32 = 0.3;
        const ROD_SLEEP_SETTLE_MAX_SECONDS: f32 = 8.0;
        let rod_threshold = self.config.rod_sleep_threshold;
        if rod_threshold > 0.0 {
            let threshold_sq = rod_threshold * rod_threshold;
            for rod in &mut self.rods {
                if rod.sleeping
                    || rod.push_strength > 0.0
                    || rod.is_growing()
                    || rod.is_correcting_gravitropically(self.config.gravity, &self.grid)
                    || rod.is_correcting_phototropically(self.config.light_dir, &self.grid)
                {
                    rod.below_threshold_time = 0.0;
                    continue;
                }
                let max_speed_sq = rod
                    .points
                    .v
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.length_squared()));
                if max_speed_sq < threshold_sq {
                    rod.below_threshold_time += sub_dt;
                    let period =
                        crate::rod::RodMaterial::fundamental_period_s(&rod.points, rod.material.ei);
                    let settle_seconds = (period * ROD_SLEEP_SETTLE_PERIODS)
                        .clamp(ROD_SLEEP_SETTLE_MIN_SECONDS, ROD_SLEEP_SETTLE_MAX_SECONDS);
                    if rod.below_threshold_time >= settle_seconds {
                        rod.sleeping = true;
                    }
                } else {
                    rod.below_threshold_time = 0.0;
                }
            }
        }
        self.last_timing.phase_sleep_us += t5.elapsed().as_micros() as u64;
    }

    pub const fn effective_dt(&self) -> f32 {
        self.last_step_dt
    }

    pub const fn last_substeps(&self) -> usize {
        self.last_substeps
    }

    /// Mean duration of the last step's substeps: the simulated time it
    /// advanced over the substeps it took, 0.0 before any step. The
    /// explicit CFL bounds how far matter moves in one substep, so this is
    /// the interval over which a particle's motion stays within about a
    /// cell, what the surface view's velocity stretch measures motion over
    /// (`SurfaceReconstructionSource::dt`). The frame `dt` spans many
    /// substeps and is not that interval.
    pub fn mean_substep_dt(&self) -> f32 {
        if self.last_substeps == 0 {
            return 0.0;
        }
        (self.config.dt - self.last_sim_time_dropped).max(0.0) / self.last_substeps as f32
    }

    pub fn step_n(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

/// Macro-spin (the antisymmetric velocity-gradient part, vorticity) of a
/// particle: `0.5*(dvy/dx - dvx/dy)`, with `l.x_axis.y` = dvy/dx and
/// `l.y_axis.x` = dvx/dy as in `sand.rs`'s `strain_rate_norm`. A plain `fn`
/// so it coerces to `CosseratField::apply`'s fn-pointer parameter.
fn macro_spin_from_velocity_gradient(p: &crate::particle::Particle) -> f32 {
    let l = p.velocity_gradient;
    0.5 * (l.x_axis.y - l.y_axis.x)
}

// apply_boundary_conditions_to_grid, project_particle_state_to_admissible: projection.rs
// choose_substep_dt, cfl_bound, affine_cfl_speed_contribution: cfl.rs
