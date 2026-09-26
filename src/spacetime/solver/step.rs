//! The adaptive-substep physics step: CFL timestep selection, P2G, grid update,
//! G2P, force fields, thermal/scalar diffusion, phase rules, and sleep scoring.
//!
//! Split out of `solver/mod.rs` (was 1536 lines, doing 5-6 jobs in one file) --
//! this is the one piece that's purely "advance the simulation by one step,"
//! distinct from construction, queries, and particle-lifecycle management that
//! live alongside `Simulation` in the parent module. `do_substep`'s own body
//! stays a single ordering-sensitive sequence on purpose (see its inline
//! comments for why each phase must run where it does) -- only the two
//! genuinely self-contained pieces split further, into sibling files:
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
    RodForceParams, RodImplicitStepParams, apply_bending_plasticity, apply_gravitropism,
    apply_growth, apply_phototropism, apply_rod_internal_and_wind_forces, apply_secondary_growth,
    gather_grid_to_rod, scatter_rod_to_grid, step_rod_implicit,
};
use crate::solver::density::estimate_particle_volumes;
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
                // `SimConfig::fluid_pressure_iterations`'s own doc -- so it is
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
        self.last_vel_clamp_count = 0;
        self.last_j_projection_count = 0;
        self.last_timing = crate::diagnostics::StepTiming::default();
        // Computed ONCE here, reused by every substep below -- see
        // `cached_spatial_sort_order`'s own doc for why (real, measured:
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
        // Not grid-coupled yet -- see the struct field's own doc.
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
            // Real elastic-perfectly-plastic bending (see `plasticity`
            // module doc) -- applied AFTER any biological reshaping above,
            // so mechanical yield acts on top of whatever active tropism/
            // growth already did this step, not instead of it.
            if let Some(plasticity) = &rod.plasticity {
                apply_bending_plasticity(&mut rod.points, plasticity, self.config.dx_meters);
            }
        }
        // Real, opt-in implicit big-step (see `implicit_corotated`'s own
        // module doc): one Newton-CG solve at the FULL frame `dt` instead
        // of the CFL-limited loop below. Only engages when the whole active
        // scene qualifies AND the solve actually converges -- otherwise
        // falls through to the exact same explicit loop every other scene
        // already runs, byte-identical to before this existed.
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
                    thermal_dt_bound: self.thermal.as_ref().map(|t| t.config.stability_dt()),
                    last_max_speed: self.last_max_particle_speed,
                },
            );
            // One-substep-lagged, real (not estimated): feeds the near-wall
            // gate's Mach-relative threshold on the NEXT call -- see
            // `choose_substep_dt`'s own `last_max_speed` param doc.
            self.last_max_particle_speed = measured_max_speed;
            self.last_timing.cfl_us += t_cfl.elapsed().as_micros() as u64;
            // TEMPORARY diagnostic (2026-08-28), see `diagnose_worst_particle_
            // cfl_term`'s own doc -- opt-in via env var so every existing scene
            // pays nothing. `EMERGE_CFL_DIAGNOSE=all` scans every particle;
            // `EMERGE_CFL_DIAGNOSE=<material_id>` filters to one material.
            if let Ok(spec) = std::env::var("EMERGE_CFL_DIAGNOSE") {
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
            // TEMPORARY diagnostic (2026-08-29), see `transfer::diagnose_
            // particle_node_material_sources`'s own doc -- opt-in via env
            // var so every existing scene pays nothing. Built to test
            // whether a tracked particle's runaway velocity (found live in
            // `phase_states_gui.rs`'s Moon-gravity run, particle 15) comes
            // from a neighboring particle of a DIFFERENT material sharing
            // its P2G/G2P support nodes -- not visible from `same_material_
            // neighbors`-style checks, which only ever count the tracked
            // particle's own material.
            if let Ok(spec) = std::env::var("EMERGE_TRACK_PARTICLE_NODES")
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
            // Sticky fine-substep hold (`fluid_sticky_fine_dt`'s own doc) -- caps
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
            if let Some(diagnostic) = &mut self.boundary_impulse_diagnostic {
                diagnostic.accept_pending();
            }
            remaining -= actual_dt;
            self.last_step_dt = actual_dt;
            substeps_taken += 1;
            // TEMPORARY diagnostic (2026-08-30), see `transfer::diagnose_
            // particle_divergence_decomposition`'s own doc and
            // `pending_divergence_diagnostic`'s own doc for why this reads
            // AFTER `do_substep_with_retry` returns: `do_substep` (called
            // once per retry attempt, real dt each time) overwrites this
            // field every call, so what's here now reflects only the LAST
            // (i.e. ACCEPTED) attempt -- a real, disclosed fix for a real
            // methodological gap in this diagnostic's first version
            // (`diagnose_particle_boundary_divergence_bias`,
            // removed): that one reconstructed its "before" state from a
            // separate scatter fed whatever `dt` the OUTER caller happened
            // to have, before `do_substep_with_retry` had a chance to
            // settle on a different `actual_dt`.
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
            // needed a fresh retry -- see the field's own doc for why persistence
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
        // Honest accounting: `max_substeps_per_step` is a real per-frame work
        // budget again (a runaway CFL collapse must not be free to make a
        // single step() call take seconds). If the budget runs out before
        // `remaining` reaches zero, report exactly how much simulation time
        // this call did not advance instead of either discarding it silently
        // (pre-2026-08-06 behavior) or looping unbounded to always finish it.
        self.last_sim_time_dropped = remaining.max(0.0);
        // Real, deliberate choice (2026-08-10), not a coin flip: a silently
        // dropped strict-fluid step is exactly the class of hidden
        // corner-cut "strict" WC-MPM mode exists to forbid (same real
        // philosophy as `check_j_range`/`assert_owned_deformation_state`
        // just above -- report loudly rather than silently accept a wrong
        // state). Ordinary materials tolerate an honestly-tracked drop (the
        // whole point of the accounting above); a strict fluid does not --
        // its own physical model has no notion of "close enough," so
        // silently advancing less than the requested dt would silently
        // break the mass/momentum conservation this mode's own strictness
        // promises. NOT a revert to unbounded looping (the real cost-DoS
        // concern the "budget again" fix above addresses stays valid) --
        // fails loud and immediately instead, with an actionable message,
        // the same real tradeoff every other strict-fluid safety check in
        // this codebase already makes.
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
        // Lazy: just mark stale here, don't do the rebuild work every frame
        // regardless of whether a query will ever consume it before the next
        // step -- see `spatial_hash`'s own doc on `Simulation` for the real,
        // measured cost this was (16.4% of a step, 2026-08-03). The first
        // query method called after this (`particles_near`/`count_near`/
        // `particles_knn`/`region_state`) does the real rebuild, lazily, via
        // `ensure_spatial_hash_fresh`.
        // Diffusion operators, applied ONCE for all the time this step
        // advanced -- see `do_substep`'s own note for the stability
        // derivation. Runs after the substep loop so temperature/scalar
        // fields see this step's final particle state.
        let t_diff = std::time::Instant::now();
        let diffusion_dt = std::mem::take(&mut self.pending_diffusion_dt);
        if diffusion_dt > 0.0 {
            if let Some(thermal) = &mut self.thermal {
                thermal.apply(&mut self.particles, diffusion_dt);
            }
            for field in &mut self.scalar_fields {
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

    /// Wraps `do_substep` with the real preflight/retry check for strict WC-MPM
    /// fluids -- see `SimConfig::fluid_step_retry_enabled`'s own doc for the full
    /// derivation and the empirical evidence this is a genuine, convergent
    /// stability limit (not a hidden clamp masking a bug). Returns the dt actually
    /// committed, which may be smaller than requested if retries fired -- callers
    /// must advance `remaining` by the RETURNED value, not the original `sub_dt`.
    ///
    /// No-op fast path (a plain `self.do_substep(sub_dt)` call, byte-identical to
    /// before this existed) whenever the feature is disabled (the default) or no
    /// registered material owns strict deformation/volume state -- every existing
    /// scene is completely unaffected.
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
        // `fluid_sticky_fine_dt`'s own doc for why a one-off retry alone doesn't
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
            // Real gap, found 2026-08-09 alongside the retry-exhaustion fix below:
            // `worst_ln_j_change` alone is blind to a substep that blows up a
            // particle's VELOCITY without its J having caught up yet in the SAME
            // substep (measured live: `ke` spiking from ~19 to 33 MILLION between
            // two consecutive substeps while `J` stayed flat at 3.212 throughout --
            // the pressure-projection correction can kick a particle's velocity far
            // past what its own CFL-safe motion should have been THIS substep,
            // since it's applied to the grid AFTER `choose_substep_dt` already
            // committed to a dt based on the PREVIOUS substep's state). Same
            // physical reasoning as `cfl_bound`'s own velocity term
            // (`cfl_coefficient*grid_cell_size/max_speed`), just checked reactively
            // here instead of predicted in advance -- a substep that moved a
            // particle further than that in ONE step already violated the CFL it
            // was supposed to satisfy, regardless of what J says.
            let mut worst_speed = 0.0f32;
            // Real gap, found 2026-08-09 debugging a hard panic in
            // `assert_owned_deformation_state`'s own J-range check ([j_min,
            // j_max]) on a scene where no single substep ever tripped
            // `admissible_ln_j_change`: this loop only ever compared J's
            // RELATIVE change against the immediately preceding substep --
            // many individually-small, consistently-signed changes (the
            // exact "many small changes" compounding-bias class already
            // named in `fluid_step_retry_threshold`'s own doc) can walk J
            // past the assert's absolute [j_min, j_max] band over the course
            // of a frame's ~150 substeps without any single step ever
            // looking inadmissible, so retry never engages and the
            // exhaustion backstop below never gets a chance to clamp it.
            // Checking the ABSOLUTE bound here too closes that gap: once a
            // substep's own J leaves the admissible band, retrying at a
            // finer dt (and holding it via `fluid_sticky_fine_dt`) slows the
            // drift's rate, and if it still can't recover within
            // `FLUID_STEP_RETRY_LIMIT`, the existing exhaustion backstop
            // (`project_particle_state_to_admissible`, already clamps J to
            // `config.j_max`) now actually gets invoked instead of the state
            // silently sailing through to the next substep's hard assert.
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
                // Tried tightening this to the material's own (usually
                // stricter) `volume_ratio_min/max` on 2026-08-28 while
                // debugging a real steam fps collapse -- reverted: `new_j`
                // here is the POST-`update_particle` value, and that function
                // already unconditionally clamps to those exact bounds before
                // this ever runs, so a check against the SAME bounds can
                // structurally never fire (confirmed live: zero effect on the
                // actual collapse, just added a `.params()` call per particle
                // per retry attempt for nothing). Real fix for that failure
                // mode still open -- see project memory.
                if new_j < self.config.j_min || new_j > self.config.j_max {
                    worst_j_out_of_bounds = true;
                }
            }
            let cfl_safe_speed = self.config.cfl_coefficient * self.config.grid_cell_size / sub_dt;
            let admissible = worst_ln_j_change <= admissible_ln_j_change
                && worst_speed <= cfl_safe_speed
                && !worst_j_out_of_bounds;
            if admissible || attempt == FLUID_STEP_RETRY_LIMIT {
                // Real gap, found 2026-08-09 via a delayed crash one frame after a
                // violent first wall impact (see project memory's
                // pressure_projection_wall_leak_bug_fixed entry): the retry loop
                // above can exhaust `FLUID_STEP_RETRY_LIMIT` with `worst_ln_j_change`
                // STILL above threshold and, until this fix, silently returned the
                // still-corrupted particle state anyway -- nothing else ever clamped
                // it, because `do_substep`'s own pre-P2G pass routes any material
                // that `owns_deformation_volume_state()` (every strict fluid) to an
                // ASSERT-only path (`assert_owned_deformation_state`), trusting THIS
                // retry loop to keep state admissible. That trust had no backstop:
                // an extreme-but-finite J/velocity (e.g. J=0.004, v=800,000+ measured
                // live) passes the assert fine, gets scattered into next frame's P2G,
                // and poisons its CFL scan into demanding a sub-ULP dt -- the actual
                // observed crash. `project_particle_state_to_admissible` already
                // exists, is material-agnostic (operates on raw particle fields, no
                // `owns_deformation_volume_state` check inside it), and is already
                // trusted for every NON-fluid material via `project_invalid_state` --
                // reusing it here for the one case that currently has no backstop at
                // all, only when retries were truly exhausted (not on the common
                // "converged within budget" path, so this never fires for a healthy
                // scene). General fix, not scene-specific: applies to any strict
                // fluid material, any wall, any geometry.
                if attempt == FLUID_STEP_RETRY_LIMIT && !admissible {
                    // `project_particle_state_to_admissible` only catches NON-FINITE
                    // velocity, not finite-but-absurd velocity (measured live: up to
                    // ~9.5e6 grid-units/s after a retry-exhausted substep, still
                    // "finite" by IEEE754's definition but far beyond anything the
                    // solver's own timestep floor can ever represent). A real,
                    // derived ceiling: `choose_substep_dt`'s own velocity-CFL term
                    // is `dt = cfl_coefficient*grid_cell_size/max_speed` (`cfl.rs::
                    // cfl_bound`) -- solving for the max_speed that keeps that
                    // formula's OWN result at or above `min_dt` requires the
                    // `cfl_coefficient` factor here too. Real bug, found
                    // 2026-08-09: the previous version omitted it (`grid_cell_size/
                    // min_dt` alone), so a velocity clamped to exactly that ceiling
                    // still produced `cfl_coefficient*min_dt` next frame -- SMALLER
                    // than `min_dt` itself by construction whenever
                    // `cfl_coefficient<1.0` (every real config), which is exactly
                    // what was still poisoning the NEXT frame's CFL scan into a
                    // sub-ULP dt even after this clamp fired.
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
                // Real bug, found 2026-08-09 via direct instrumentation of a
                // sub-ULP-dt crash a full frame after the triggering event:
                // this used to fire on EXHAUSTION too (`attempt ==
                // FLUID_STEP_RETRY_LIMIT && !admissible`), holding the
                // FAILED sub_dt -- one that 16 halvings still couldn't make
                // admissible -- as a ceiling on the next `STICKY_HOLD_
                // SUBSTEPS` substeps. That's backwards: exhaustion means "no
                // dt this loop tried was enough," so the state was instead
                // force-corrected by the backstop just above; the fresh,
                // now-admissible state deserves a FRESH CFL scan next
                // substep, not a stale, arbitrarily tiny floor inherited from
                // the attempt that just failed (measured live: an exhausted
                // sub_dt near 2^-16 of the original request, small enough to
                // fail the very next frame's `remaining - sub_dt < remaining`
                // representability check even though the corrected state
                // itself was perfectly reasonable). Only the genuinely
                // successful path -- retry found an admissible finer dt on
                // its own -- earns the sticky hold now.
                if admissible && sub_dt < requested_dt {
                    // At least one halving was needed to reach an admissible
                    // state -- hold this fine dt for subsequent substeps too,
                    // not just this one (refreshes/extends an existing hold).
                    //
                    // Real gap, found 2026-08-09 alongside the exhaustion-path
                    // fix just above: even the genuinely-`admissible` path can
                    // land on a vanishingly small `sub_dt` -- `admissible`
                    // itself gets EASIER to satisfy as `sub_dt` shrinks
                    // (`cfl_safe_speed = cfl_coefficient*grid_cell_size/
                    // sub_dt` grows without bound, and `worst_ln_j_change`
                    // over a near-zero step trivially shrinks too), so a
                    // sufficiently violent substep can walk all the way down
                    // through several halvings to something far below
                    // `min_dt` and still report success, not exhaustion.
                    // Holding THAT floorless value for `STICKY_HOLD_SUBSTEPS`
                    // more substeps (crossing into the next frame if the
                    // hold outlives `step()`'s own substep budget) is what
                    // was still poisoning a later, otherwise-healthy CFL scan
                    // into an unrepresentable sub-`min_dt` result. `min_dt`
                    // is the solver's own configured floor for exactly this
                    // situation elsewhere (see the exhaustion backstop's
                    // `max_representable_speed`, derived from the same
                    // constant) -- flooring the HOLD here (an artificial,
                    // engineering-only ceiling, not a physics quantity) at
                    // `min_dt` doesn't touch the real per-substep CFL scan's
                    // own freedom to legitimately go below `min_dt` again
                    // later if the state genuinely still demands it (`sub_dt.
                    // min(held_dt)` only ever tightens, never loosens, so a
                    // smaller FRESH result still wins).
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
        //   real per-substep safety for those materials is enforced inside
        //   their own `update_particle` (the J admissibility assert), so this
        //   scan is a redundant second opinion. Running it once per FRAME
        //   still catches any corruption within that frame, just at the frame
        //   boundary instead of mid-substep.
        //
        // Live-measured: this scan was `project_us` ~2900 us of a ~29000 us
        // step (10%), and in a fluid-only scene every particle takes the
        // assert branch.
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

        // Optional kernel density gather for materials that use it.  Strict
        // WC-MPM fluids still scatter mass for this diagnostic field, but retain
        // their constitutive rho=rho0/J and V=V0J state.
        let t_density = std::time::Instant::now();
        if self.config.recompute_density_each_step || self.materials.any_needs_density_recompute() {
            estimate_particle_volumes(
                &mut self.particles,
                &mut self.grid,
                Some(&self.materials),
                self.active_count,
                false,
            );
        }
        self.last_timing.density_us += t_density.elapsed().as_micros() as u64;

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
        gather_contact_point_cloud(&self.particles, &mut self.grid, self.active_count);
        // Rod -> grid scatter, same P2G pass, same shared `Grid` -- BEFORE the
        // wake pass below so a rod touching settled sand/fluid wakes it with
        // zero new code (the wake scan just sees active cells the rod itself
        // created). No-op for every scene that never calls add_rod/with_rod.
        // Sleeping rods skip this entirely (see `Rod::sleeping` doc) -- they
        // neither scatter mass/momentum nor self-trigger their own wake check
        // below; they're woken only by genuinely external activity.
        //
        for rod in &self.rods {
            if !rod.sleeping && !rod.use_implicit_integration {
                scatter_rod_to_grid(&rod.points, &mut self.grid);
            }
        }
        // Grain -> grid scatter, same shared `Grid`, same convention as rods
        // above -- see `grains::coupling`'s own doc. No-op for every scene
        // that never calls `add_grain_population`.
        for population in &self.grain_populations {
            scatter_grains_to_grid(population, &mut self.grid);
        }
        self.last_timing.p2g_us += t0.elapsed().as_micros() as u64;

        // TEMPORARY structural-boundary diagnostic. Reconstructs the same
        // particle P2G terms with this retry attempt's exact `sub_dt`; the
        // outer loop accepts only the final attempt's pending ledger.
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

        // TEMPORARY diagnostic (2026-08-30), `EMERGE_TRACK_BOUNDARY_BIAS`,
        // see `transfer::diagnose_particle_divergence_decomposition`'s own
        // doc. Deliberately placed HERE, right after the real P2G scatter,
        // using the SAME `sub_dt` that scatter just used -- no separate
        // reconstruction, no possible retry-`dt` mismatch. Stores only the
        // pre-grid-update decomposition for now; `trace_final`/`dt_used`
        // get filled in after G2P runs, below.
        if let Ok(spec) = std::env::var("EMERGE_TRACK_BOUNDARY_BIAS")
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
        // pre-force reference point -- see `Grid::apply_cundall_damping`'s own doc --
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
                // motion as a real previous velocity.
                self.grid.apply_pinned_node_constraints();
                let snapshot = self.grid.snapshot_velocities();
                self.grid.apply_gravity(sub_dt, self.config.gravity);
                Some(snapshot)
            } else {
                self.grid.update_velocities(sub_dt, self.config.gravity);
                None
            };
        let grid_res = self.grid.resolution();
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
            self.contact_grip.as_deref(),
        );
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
        // (`grid::pressure`, see `SimConfig::fluid_pressure_iterations`'s own
        // doc). Runs after boundary/contact/mixture resolution so it corrects
        // the real post-gravity/post-wall velocity field, and before G2P so
        // particles gather the corrected field. No-op when
        // `fluid_pressure_iterations == 0`, the default -- every existing
        // scene is unaffected.
        //
        // Called `fluid_pressure_iterations` times in a row, not once --
        // real, standard technique (outer corrector passes, the same
        // principle PISO/SIMPLE-family incompressible-flow solvers use: one
        // projection is a first-order splitting of a genuinely violent
        // state, re-measuring the residual divergence AFTER a correction and
        // correcting again converges much closer to a true divergence-free
        // field than a single pass can, especially right after a violent
        // impact the first substep(s) haven't had a chance to smooth yet --
        // see MEMORY.md's fluid-recovery notes, Round 9, for why a single
        // exact solve was confirmed (not guessed: Gauss-Seidel, which has no
        // spectral artifacts at all, converged to the SAME extreme answer)
        // to be the true solution of the underlying equation for an already-
        // extreme input, not a solver artifact -- the fix has to reduce how
        // extreme that input is allowed to get, which repeated correction
        // within the same substep does directly.
        let t_pressure = std::time::Instant::now();
        for _ in 0..self.config.fluid_pressure_iterations {
            self.grid
                .project_fluid_incompressibility(self.config.grid_cell_size, 1);
            // Real, confirmed bug (2026-08-09): the projection's own gradient
            // correction (`Grid::project_fluid_incompressibility`) writes
            // directly to `cell.momentum` with no awareness of the wall --
            // it can (and, measured, does) push velocity back through a wall
            // whose zero-normal-velocity condition was already satisfied by
            // the `apply_boundary_conditions_to_grid` call above. G2P then
            // gathers that un-reclamped field. Symptom, root-caused via
            // direct per-substep instrumentation (not guessed): a particle
            // resting against the floor showed a SUSTAINED (not spiking,
            // never reversing) positive velocity-gradient trace every
            // substep for over 100 frames, compounding `J` from ~1 to
            // >1,000,000 -- while `grad_p` itself never exceeded 50 and the
            // gathered `C` matrix never exceeded 200 in any single substep,
            // ruling out a solver-accuracy/spike explanation (GS sweep count
            // 5 vs 10 vs 30 made zero measurable difference). Re-applying the
            // same boundary pass after every corrector iteration -- not just
            // once at the end -- keeps the wall as the LAST word on grid
            // velocity for every one of the `fluid_pressure_iterations`
            // passes, matching the loop's own multi-pass-corrector
            // rationale above.
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
        self.last_vel_clamp_count += gather_grid_to_particles(
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
                // Real, honest, minor shared cost: if only `cundall_damping` is enabled
                // (asflip_blend still 0.0), G2P still takes the `Some` branch and computes
                // the extra pre-force stencil gather -- harmless (asflip_blend=0.0 zeroes
                // its own contribution exactly) but not free. Reusing one snapshot for both
                // features beats duplicating the mechanism; this is the real tradeoff.
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
        // TEMPORARY diagnostic (2026-08-30), see the P2G-side insertion
        // above and `pending_divergence_diagnostic`'s own doc -- fills in
        // the REAL, final `tr(C)` G2P just wrote onto the tracked
        // particle, now that G2P has actually run this same substep.
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
        // Grid -> rod gather (this rod's own G2P): pulls velocity (gravity
        // already baked in via the shared grid-update step above) AND
        // advances `rod.points.x`, mirroring `gather_grid_to_particles`'s own
        // position-advection contract exactly (see `coupling::gather_grid_to_rod`'s
        // doc) so rod force integration below only ever touches velocity,
        // matching how particle force fields never touch `particles.x` either.
        for rod in &mut self.rods {
            if !rod.sleeping && !rod.use_implicit_integration {
                gather_grid_to_rod(&mut rod.points, &self.grid, sub_dt);
            }
        }
        // Grid -> grain gather -- velocity only, does NOT advance position
        // (unlike rods above): see `grains::coupling::gather_grid_to_grains`'s
        // own doc for why grains need contact resolved before integration.
        // Real APIC (2026-08-20), reusing the SAME `apic_blend` config knob
        // ordinary particles already use (no new parameter) -- confirmed
        // necessity for grains specifically: pure PIC held a real, jittered
        // column-collapse frozen near its initial lattice shape (matches
        // Jiang et al. 2015's own documented "PIC causes sand to clump
        // together" finding); an interim pure-FLIP fix worked but showed
        // real, literature-predicted noise/dt-instability. APIC is the
        // literature's own real resolution to both problems at once.
        // Also reuses `asflip_blend`/`pre_force_snapshot` (2026-08-20) --
        // the SAME real hybrid ordinary particles already get from
        // `gather_grid_to_particles`, no new config surface. `asflip_blend`
        // defaults to 0.0 (every existing scene), so this is a zero-cost,
        // zero-behavior-change extension until a scene explicitly opts in.
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
                // every substep -- a real, confirmed root cause of long-horizon energy
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
        // deferred`'s own doc): that loop is the only caller of `do_substep`
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
        // `grains::coupling::apply_grain_contact_forces`'s own doc.
        for population in &mut self.grain_populations {
            apply_grain_contact_forces(population, sub_dt, &self.boundaries, grid_res, &self.grid);
        }
        // Real, bounded grid-mediated rotational coupling between nearby
        // grains -- see `grains::micro_rotation`'s own doc for the full
        // story: replaces an earlier, reverted attempt that scattered
        // `spin` directly into the shared momentum grid (exact for an
        // isolated grain, but a confirmed unbounded energy leak once many
        // spinning grains share overlapping grid nodes).
        //
        // Real, honest, disclosed status (2026-08-20): proven SAFE
        // (unconditionally stable by construction, confirmed directly at
        // both 1.0 and 1000x that value against a real, deterministic
        // replay of a genuine near-instability capture) but NOT proven
        // sufficient on its own to fix the real column-collapse isolation
        // test's own frozen-lattice result (`tests/grains_grid_coupling.rs`)
        // -- swept 1.0 to 1000.0, bit-for-bit IDENTICAL final spread ratio
        // at every value. That test's own geometry dump shows the pile
        // reaching a genuinely static, friction-locked mechanical
        // equilibrium within the first ~20% of the run and simply staying
        // there -- current leading hypothesis is that the grid coupling
        // path itself gives the pile MORE stability than the same real
        // contact_law physics provides standalone (not proven, not yet
        // root-caused -- see memory for the full, real investigation
        // history). Kept enabled at a modest value: real, safe, physically
        // motivated (Cosserat-family rotational coupling), zero known
        // downside -- just not, on its own, the fix for the frozen-column
        // question above.
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

        // ── Rod internal + wind forces ──────────────────────────────────────────
        // Runs where particle force fields just ran, on the SAME real convention:
        // velocity-only (position already advanced in the gather above), so a
        // rod's own stretch/bend/damping + wind drag land exactly like an
        // ordinary force field would. Gravity is NOT reapplied here -- the rod
        // already received it via the shared grid-update step, same mechanism
        // ordinary particles use. No-op for every scene with no rods.
        //
        for rod in &mut self.rods {
            if rod.sleeping || rod.use_implicit_integration {
                continue;
            }
            apply_rod_internal_and_wind_forces(
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
                },
            );
            // Real root gravitropism (Porat, Rivière, Meroz 2024 -- see
            // `rod::gravitropism` module doc): evolves the tip's own
            // rest_curvature toward gravity-alignment. No-op for every rod
            // that doesn't opt in (plain stems/blades don't grow toward
            // gravity).
            if let Some(gravitropism) = &rod.gravitropism {
                apply_gravitropism(
                    &mut rod.points,
                    gravitropism,
                    self.config.gravity,
                    &self.grid,
                    sub_dt,
                );
            }
            // Real phototropism (Cholodny & Went auxin-asymmetry theory --
            // see `rod::gravitropism` module doc's own "Phototropism reuses
            // the SAME core" section). No-op for every rod that doesn't
            // opt in.
            if let Some(phototropism) = &rod.phototropism {
                apply_phototropism(
                    &mut rod.points,
                    phototropism,
                    self.config.light_dir,
                    &self.grid,
                    sub_dt,
                );
            }
            // Real elongation growth (Verhulst 1838 logistic law -- see
            // `rod::growth` module doc). No-op for every rod that doesn't
            // opt in.
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
            // Real stress-driven secondary growth (Jaffe 1973, Mattheck &
            // Kübler 1995 -- see `rod::secondary_growth` module doc). No-op
            // for every rod that doesn't opt in. Gated on the rod STILL
            // being over-critical -- see the other call site's own doc for
            // the real, measured bug this fixes (unbounded stiffening long
            // past the point it was actually needed).
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
            // Real elastic-perfectly-plastic bending (see `rod::plasticity`
            // module doc) -- same ordering rationale as the implicit branch's
            // own call site: mechanical yield applies on top of whatever
            // biological reshaping already happened this substep.
            if let Some(plasticity) = &rod.plasticity {
                apply_bending_plasticity(&mut rod.points, plasticity, self.config.dx_meters);
            }
        }

        // ── Thermal / scalar diffusion ────────────────────────────────────────
        let t4 = std::time::Instant::now();
        // Thermal / scalar diffusion are SEPARATE operators from the momentum
        // solve, with their own -- far laxer -- explicit stability limit, so
        // they are accumulated here and applied ONCE per `step()` with the
        // total advanced time (see `flush_diffusion_operators`). This is
        // ordinary operator splitting at each operator's own stable rate, not
        // an approximation introduced for speed.
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
        // `stability_dt` is still folded into `choose_substep_dt`, so a scene
        // whose diffusion genuinely IS the bottleneck still clamps the whole
        // substep and this stays correct for it too.
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
        // Per-substep by default (CLAUDE.md's documented contract). A scene
        // whose rules are thermodynamic -- which cannot change within a frame,
        // since diffusion advances once per `step()` -- can opt into
        // once-per-step via `SimConfig::phase_rules_once_per_step` and skip
        // ~17 redundant O(N) scans per frame. See that field's own doc.
        let evaluate_phase_rules =
            !self.config.phase_rules_once_per_step || self.substep_index_in_frame == 0;
        if !self.phase_rules.is_empty() && evaluate_phase_rules {
            let rules = std::mem::take(&mut self.phase_rules);
            for i in 0..self.active_count {
                let p = self.particles.get(i);
                for rule in &rules {
                    if let Some(new_id) = rule(&p) {
                        // Shared with `Simulation::phase_transition` -- see
                        // `apply_phase_transition`'s own doc (`solver::
                        // particles`) for the real elastic-reference
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
        // Must sleep on a sustained duration below threshold, not the instant
        // `max_speed_sq < threshold_sq`: a freshly-constructed rod trivially
        // satisfies that (`v = Vec2::ZERO` at birth) before gravity/grid coupling
        // gets a chance to act within one tiny substep, so it could fall asleep on
        // its very first substep, then skip its own gravity entirely while
        // "asleep" until external activity woke it -- receiving the entire
        // deferred gravitational transient at once as an unphysical velocity
        // spike. Every major real-time physics engine (Box2D's documented
        // `b2_timeToSleep = 0.5s`, Bullet, PhysX) requires staying below threshold
        // for a minimum duration, not one instant, for exactly this reason.
        //
        // Real root-cause fix (user-reported "never settles straight",
        // headlessly confirmed): a FIXED settle-duration (this used to be a
        // single constant, 0.5s) is wrong for ANY rod whose own natural
        // period is comparable to or longer than that fixed window. A
        // rod's velocity genuinely dips near zero at every swing peak, not
        // just at true rest -- if the fixed window is short enough relative
        // to the period, the sustained-below-threshold requirement can
        // complete DURING a single slow peak of a still-large-amplitude
        // swing, freezing the rod there at a real, wrong, off-rest position.
        // A soft demo blade (period ~0.53s) froze
        // several cells from true vertical rest at a fixed 0.5s window, and
        // even bumping that fixed constant up only shifts the same failure
        // to an even slower rod -- the real fix is SCALING the window to
        // each rod's OWN period, not picking a bigger universal constant.
        // `ROD_SLEEP_SETTLE_PERIODS=3.0`: three full natural periods of
        // sustained quiet is real headroom past any single swing peak's own
        // dwell time, for any rod's own stiffness/mass. Clamped to
        // `[ROD_SLEEP_SETTLE_MIN_SECONDS, ROD_SLEEP_SETTLE_MAX_SECONDS]`:
        // the floor preserves the original anti-instant-sleep protection
        // above for a very stiff/fast rod (three periods of a very fast rod
        // could be under a millisecond); the ceiling keeps an extremely
        // soft/slow rod from waiting an impractically long real time.
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

    pub fn step_n(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

/// Real macro-spin (antisymmetric velocity-gradient component, the ordinary
/// vorticity) for a particle: `0.5*(dvy/dx - dvx/dy)`. Same field-access
/// convention `sand.rs`'s own `strain_rate_norm` computation already uses
/// for the symmetric part -- `l.x_axis.y` = dvy/dx, `l.y_axis.x` = dvx/dy.
/// Plain `fn` (not a closure) so it coerces to `CosseratField::apply`'s
/// fn-pointer parameter, matching `GranularFluidityField::pressure_and_ratio`'s
/// own caller-supplied convention.
fn macro_spin_from_velocity_gradient(p: &crate::particle::Particle) -> f32 {
    let l = p.velocity_gradient;
    0.5 * (l.x_axis.y - l.y_axis.x)
}

// apply_boundary_conditions_to_grid, project_particle_state_to_admissible: projection.rs
// choose_substep_dt, cfl_bound, affine_cfl_speed_contribution: cfl.rs
