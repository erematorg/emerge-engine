//! The adaptive-substep physics step: CFL timestep selection, P2G, grid update,
//! G2P, force fields, thermal/scalar diffusion, phase rules, and sleep scoring.
//!
//! Split out of `solver/mod.rs` (was 1536 lines, doing 5-6 jobs in one file) --
//! this is the one piece that's purely "advance the simulation by one step,"
//! distinct from construction, queries, and particle-lifecycle management that
//! live alongside `Simulation` in the parent module.

use glam::{Mat2, Vec2};

use super::{MaterialRegistry, SimConfig, Simulation};
use crate::boundary::BoundaryCondition;
use crate::grid::Grid;
use crate::particle::Particles;
use crate::solver::density::estimate_particle_volumes;
use crate::transfer::{
    G2PParams, gather_contact_point_cloud, gather_grid_to_particles, scatter_particles_to_grid,
};

impl Simulation {
    /// One MLS-MPM timestep: particle→grid→particle cycle.
    /// The grid is temporary scratch — only particles hold long-term material memory.
    pub fn step(&mut self) {
        // Adaptive substep loop: step() always advances exactly config.dt of simulation time,
        // but uses smaller sub-steps when CFL requires it (stiff materials, high velocities).
        // Without this loop, the FixedStepController accounts for config.dt per call but the
        // simulation only advances sub_dt — causing it to run orders of magnitude too slowly.
        let step_start = std::time::Instant::now();
        let mut remaining = self.config.dt;
        let mut substeps_taken = 0;
        self.last_vel_clamp_count = 0;
        self.last_j_projection_count = 0;
        self.last_timing = crate::diagnostics::StepTiming::default();
        while remaining > f32::EPSILON && substeps_taken < self.config.max_substeps_per_step {
            // Cap sub-step at remaining time so we don't overshoot the configured frame dt.
            let t_cfl = std::time::Instant::now();
            let sub_dt = choose_substep_dt(
                &self.config,
                &self.particles,
                self.active_count,
                &self.materials,
                remaining,
            );
            self.last_timing.cfl_us += t_cfl.elapsed().as_micros() as u64;
            self.do_substep(sub_dt);
            remaining -= sub_dt;
            self.last_step_dt = sub_dt;
            substeps_taken += 1;
        }
        self.last_substeps = substeps_taken;
        self.last_sim_time_dropped = remaining.max(0.0);
        // Rebuild once per step, not per substep — LP queries happen between step() calls,
        // never mid-substep, so one rebuild after the loop is sufficient and correct.
        let t_hash = std::time::Instant::now();
        self.spatial_hash
            .rebuild(&self.particles.x, self.active_count);
        self.last_timing.spatial_hash_us = t_hash.elapsed().as_micros() as u64;
        self.last_timing.total_us = step_start.elapsed().as_micros() as u64;
        self.frame_index = self.frame_index.saturating_add(1);
    }

    fn do_substep(&mut self, sub_dt: f32) {
        // Project invalid particle state before it can corrupt the grid scatter.
        // Running pre-P2G (not post) means a bad particle from a previous substep is
        // fixed before its momentum enters the grid — no NaN cascade possible.
        let t_pre = std::time::Instant::now();
        if self.config.project_invalid_state {
            for i in 0..self.active_count {
                if project_particle_state_to_admissible(&mut self.particles, i, &self.config) {
                    self.last_j_projection_count += 1;
                }
            }
        }
        self.last_timing.project_us += t_pre.elapsed().as_micros() as u64;

        // Density recompute: fluid EOS materials need current ρ each substep (pressure = f(ρ)).
        // Auto-enabled when any registered material declares needs_density_recompute=true.
        // Manual override via config.recompute_density_each_step for edge cases.
        let t_density = std::time::Instant::now();
        if self.config.recompute_density_each_step || self.materials.any_needs_density_recompute() {
            estimate_particle_volumes(
                &mut self.particles,
                &mut self.grid,
                self.active_count,
                false,
            );
        }
        self.last_timing.density_us += t_density.elapsed().as_micros() as u64;

        // ── P2G ──────────────────────────────────────────────────────────────
        let t0 = std::time::Instant::now();
        self.grid.clear();
        scatter_particles_to_grid(
            &self.particles,
            &mut self.grid,
            &self.materials,
            sub_dt,
            self.active_count,
        );
        // Second particle pass for the contact-normal point cloud (see
        // `gather_contact_point_cloud` doc) -- must run after the above, since
        // contact-active nodes aren't fully known until every grip particle's mass
        // has been scattered. No-op when `contact_group` is unused anywhere.
        gather_contact_point_cloud(&self.particles, &mut self.grid, self.active_count);
        self.last_timing.p2g_us += t0.elapsed().as_micros() as u64;

        // Wake any sleeping particle whose kernel overlaps an active grid cell.
        // This propagates activity from moving regions into neighbouring sleeping ones
        // without a separate O(N) scan — we only visit the sleeping partition.
        if self.active_count < self.particles.len() {
            let total = self.particles.len();
            self.scratch_indices.clear();
            for i in self.active_count..total {
                let x = self.particles.x[i];
                let base = crate::grid::kernel::quadratic_weights(x).base_cell;
                'outer: for gx in 0i32..3 {
                    for gy in 0i32..3 {
                        let cell = base + glam::IVec2::new(gx - 1, gy - 1);
                        if self.grid.cell_is_active(cell) {
                            self.scratch_indices.push(i);
                            break 'outer;
                        }
                    }
                }
            }
            // Index directly — wake_particle doesn't touch scratch_indices, capacity preserved.
            for j in 0..self.scratch_indices.len() {
                let i = self.scratch_indices[j];
                self.wake_particle(i);
            }
        }

        // ── Grid update ───────────────────────────────────────────────────────
        let t1 = std::time::Instant::now();
        // ASFLIP (SimConfig::asflip_blend, Fei et al. 2021) needs the grid's velocity
        // right after P2G's own momentum normalization -- before THIS substep's gravity,
        // boundary conditions, or contact resolution modify it -- to compute G2P's FLIP
        // residual. Snapshotting only when the feature is enabled keeps every other scene
        // on the exact original single-call path (zero cost, zero behavior change).
        let asflip_snapshot = if self.config.asflip_blend > 0.0 {
            self.grid.normalize_velocities();
            let snapshot = self.grid.snapshot_velocities();
            self.grid.apply_gravity(sub_dt, self.config.gravity);
            Some(snapshot)
        } else {
            self.grid.update_velocities(sub_dt, self.config.gravity);
            None
        };
        let grid_res = self.grid.resolution();
        for boundary in &self.boundaries {
            apply_boundary_conditions_to_grid(&mut self.grid, grid_res, boundary.as_ref());
        }
        // Clamp grid velocity before G2P — bounds both v_p and C_p at the source.
        // Post-G2P clamping misses C_p: large C_p → F = (I + dt·C)·F blows up → J→0.
        let vel_limit = self.config.grid_cell_size / sub_dt;
        {
            for cell in self.grid.active_cells_mut() {
                if cell.mass > 0.0 {
                    let spd = cell.momentum.length();
                    if spd > vel_limit {
                        cell.momentum *= vel_limit / spd;
                    }
                }
            }
        }
        // Multi-field frictional contact (Bardenhagen 2001) — AFTER the clamp above, so
        // the grip/rest split is resolved against an already-safe total, and passed
        // `vel_limit` to apply the SAME clamp to the grip field's own raw velocity and
        // to both resolved outputs (a tiny-mass grip node could otherwise carry a huge
        // raw velocity even when the total is fine). No-op (no dirty contact cells) for
        // every scene that never sets `Particle::contact_group` — see
        // `Grid::resolve_contact` doc.
        self.grid.resolve_contact(
            sub_dt,
            self.config.gravity,
            self.config.contact_friction,
            vel_limit,
            self.config.grid_cell_size,
            self.contact_grip.as_deref(),
        );
        // Two-phase mixture coupling (Tampubolon et al. 2017) — same "after the
        // clamp, no-op when unused" positioning as contact above. No-op (no dirty
        // mixture cells) for every scene that never uses `WithMixturePhase` — see
        // `Grid::resolve_mixture_coupling` doc.
        self.grid.resolve_mixture_coupling(
            sub_dt,
            self.config.gravity,
            self.config.mixture_drag_coefficient,
            self.config.grid_cell_size,
            self.config.mixture_pressure_iterations,
        );
        self.last_timing.grid_update_us += t1.elapsed().as_micros() as u64;

        // ── G2P ──────────────────────────────────────────────────────────────
        let t2 = std::time::Instant::now();
        self.last_vel_clamp_count += gather_grid_to_particles(
            &mut self.particles,
            &self.grid,
            sub_dt,
            &self.boundaries,
            &self.materials,
            G2PParams {
                vel_limit: self.config.grid_cell_size / sub_dt,
                apic_blend: self.config.apic_blend,
                active_count: self.active_count,
                asflip_blend: self.config.asflip_blend,
                pre_force_snapshot: asflip_snapshot.as_ref(),
            },
        );
        self.last_timing.g2p_us += t2.elapsed().as_micros() as u64;

        // ── Force fields ──────────────────────────────────────────────────────
        // External body force fields: v += dt × acceleration(p) per particle.
        // Applied after G2P so each field sees the fully gathered particle state.
        // A post-field velocity clamp (same limit as G2P) prevents large impulses from
        // leaving particles with >1 cell/substep velocity that P2G then scatters as extreme
        // momentum — the clamp re-asserts the CFL contract after external perturbation.
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
            // Re-clamp velocity after force fields — large external impulses (explosions,
            // creature bursts, planetary impacts) must not enter P2G with >1 cell/substep.
            let vel_limit = self.config.grid_cell_size / sub_dt;
            for i in 0..self.active_count {
                if self.particles.pinned[i] != 0 {
                    continue;
                }
                let spd = self.particles.v[i].length();
                if spd > vel_limit {
                    self.particles.v[i] *= vel_limit / spd;
                }
            }
            self.last_timing.fields_us += t3.elapsed().as_micros() as u64;
        }

        // ── Thermal / scalar diffusion ────────────────────────────────────────
        let t4 = std::time::Instant::now();
        if let Some(thermal) = &mut self.thermal {
            thermal.apply(&mut self.particles, sub_dt);
        }
        for field in &mut self.scalar_fields {
            field.apply(&mut self.particles, sub_dt);
        }
        self.last_timing.thermal_us += t4.elapsed().as_micros() as u64;

        // ── Phase rules + sleep scoring ───────────────────────────────────────
        let t5 = std::time::Instant::now();
        if !self.phase_rules.is_empty() {
            let rules = std::mem::take(&mut self.phase_rules);
            let heat_capacity = self.thermal.as_ref().map(|t| t.config.heat_capacity);
            for i in 0..self.active_count {
                let p = self.particles.get(i);
                for rule in &rules {
                    if let Some(new_id) = rule(&p) {
                        self.particles.material_id[i] = new_id;
                        let latent_heat = self.materials.get(new_id).latent_heat();
                        if let (true, Some(cp)) = (latent_heat != 0.0, heat_capacity) {
                            self.particles.temperature[i] -= latent_heat / cp;
                        }
                        // See `Simulation::phase_transition`'s doc: reset
                        // material-specific plastic state to the new material's
                        // own defaults instead of silently inheriting the old
                        // material's stale values under a different meaning.
                        let mut p = self.particles.get(i);
                        self.materials.get(new_id).init_particle(&mut p);
                        self.particles.set(i, p);
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
        self.last_timing.phase_sleep_us += t5.elapsed().as_micros() as u64;
    }

    pub fn effective_dt(&self) -> f32 {
        self.last_step_dt
    }

    pub fn last_substeps(&self) -> usize {
        self.last_substeps
    }

    pub fn step_n(&mut self, steps: usize) {
        for _ in 0..steps {
            self.step();
        }
    }
}

fn apply_boundary_conditions_to_grid(
    grid: &mut Grid,
    grid_res: usize,
    boundary: &dyn BoundaryCondition,
) {
    for (i, cell) in grid.active_cells_with_index_mut() {
        if cell.mass > 0.0 {
            boundary.apply_to_grid_velocity(i, grid_res, &mut cell.momentum);
        }
    }
}

/// Returns `true` if any field was corrected (state was invalid/non-finite).
fn project_particle_state_to_admissible(
    particles: &mut Particles,
    i: usize,
    config: &SimConfig,
) -> bool {
    let mut projected = false;
    let min = config.boundary_thickness.saturating_sub(1) as f32;
    let max = config.grid_res.saturating_sub(config.boundary_thickness) as f32;
    let domain_center = Vec2::splat((min + max) * 0.5);

    if !particles.x[i].is_finite() {
        particles.x[i] = domain_center;
        projected = true;
    } else {
        particles.x[i] = particles.x[i].clamp(Vec2::splat(min), Vec2::splat(max));
    }

    if !particles.v[i].is_finite() {
        particles.v[i] = Vec2::ZERO;
        projected = true;
    }
    if !particles.velocity_gradient[i].x_axis.is_finite()
        || !particles.velocity_gradient[i].y_axis.is_finite()
    {
        particles.velocity_gradient[i] = Mat2::ZERO;
        projected = true;
    }

    let f = particles.deformation_gradient[i];
    if !f.x_axis.is_finite()
        || !f.y_axis.is_finite()
        || f.determinant() <= config.projection_min_deformation_j
    {
        particles.deformation_gradient[i] = Mat2::IDENTITY;
        projected = true;
    } else {
        let j = f.determinant();
        if j > config.j_max {
            particles.deformation_gradient[i] *= (config.j_max / j).sqrt();
            projected = true;
        }
    }

    if !particles.plastic_volume_ratio[i].is_finite() || particles.plastic_volume_ratio[i] <= 0.0 {
        particles.plastic_volume_ratio[i] = 1.0;
        projected = true;
    }
    if !particles.hardening_scale[i].is_finite() || particles.hardening_scale[i] <= 0.0 {
        particles.hardening_scale[i] = 1.0;
        projected = true;
    }
    if !particles.friction_hardening[i].is_finite() {
        particles.friction_hardening[i] = 0.0;
        projected = true;
    }
    if !particles.log_volume_strain[i].is_finite() {
        particles.log_volume_strain[i] = 0.0;
        projected = true;
    }

    if !particles.mass[i].is_finite() || particles.mass[i] <= 0.0 {
        particles.mass[i] = config.particle_mass;
        projected = true;
    }
    if !particles.initial_volume[i].is_finite() || particles.initial_volume[i] <= 0.0 {
        particles.initial_volume[i] = config
            .default_initial_volume
            .max(config.projection_min_volume);
        projected = true;
    }
    if !particles.volume[i].is_finite() || particles.volume[i] <= 0.0 {
        particles.volume[i] = particles.initial_volume[i].max(config.projection_min_volume);
        projected = true;
    }
    if !particles.density[i].is_finite() || particles.density[i] <= 0.0 {
        particles.density[i] =
            (particles.mass[i] / particles.volume[i]).max(config.projection_min_density);
        projected = true;
    } else {
        particles.density[i] = particles.density[i].max(config.projection_min_density);
    }
    projected
}

// choose_substep_dt: picks the largest CFL-safe dt ≤ max_dt.
// Called inside step()'s substep loop — max_dt is the remaining frame time.
// pub(crate) so the GPU solver can reuse this without duplicating CFL logic.
pub(crate) fn choose_substep_dt(
    config: &SimConfig,
    particles: &Particles,
    active_count: usize,
    materials: &MaterialRegistry,
    max_dt: f32,
) -> f32 {
    if !config.adaptive_timestep {
        return max_dt.min(config.dt);
    }
    // Single pass for both velocity CFL and material timestep bound.
    let mut max_speed = 0.0f32;
    let mut min_mat_dt = max_dt;
    for i in 0..active_count {
        let mut s = particles.v[i].length();
        if config.cfl_include_affine_speed {
            s += affine_cfl_speed_contribution(
                &particles.velocity_gradient[i],
                config.grid_cell_size,
            );
        }
        max_speed = max_speed.max(s);
        let mdt = materials.get(particles.material_id[i]).timestep_bound(
            particles.density[i],
            particles.hardening_scale[i],
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        if mdt.is_finite() && mdt > 0.0 {
            min_mat_dt = min_mat_dt.min(mdt);
        }
    }
    cfl_bound(config, max_speed, min_mat_dt, max_dt)
}

/// Shared CFL formula: clamps dt to advection + material bounds.
/// Called by both SoA and AoS scan paths after computing their respective max values.
pub(crate) fn cfl_bound(config: &SimConfig, max_speed: f32, min_mat_dt: f32, max_dt: f32) -> f32 {
    let mut dt = max_dt;
    if max_speed > f32::EPSILON {
        dt = dt.min(config.cfl_coefficient * config.grid_cell_size / max_speed);
    }
    dt = dt.min(min_mat_dt);
    dt.clamp(config.min_dt.min(max_dt), max_dt)
}

pub(crate) fn affine_cfl_speed_contribution(c: &Mat2, cell_width: f32) -> f32 {
    // The APIC affine matrix C encodes the local velocity gradient.
    // The farthest point in the quadratic B-spline 3×3 stencil is at 1.5 cells per axis,
    // so its corner distance is 1.5*√2 cells — the effective maximum affine speed contribution.
    const STENCIL_CORNER_DISTANCE: f32 = 1.5 * std::f32::consts::SQRT_2;
    let grad_norm = (c.x_axis.length_squared() + c.y_axis.length_squared()).sqrt();
    grad_norm * STENCIL_CORNER_DISTANCE * cell_width
}
