pub mod config;
pub mod cutoff;
pub mod density;
pub mod handle;
pub mod query;

pub use config::{SolverConfig, SpawnConfig};
pub use cutoff::smooth_cutoff;
pub use density::compute_density_grid;
pub use handle::{MaterialHandle, ParticleGroup};
pub use query::{MaterialState, material_state_of, region_state_of};

use std::collections::BTreeMap;

use glam::{Mat2, Vec2};


use crate::diagnostics::{MpmSnapshot, collect_mpm_snapshot};
use crate::{
    boundary::{BoundaryCondition, SlipBoundary},
    fields::ForceField,
    materials::{MaterialModel, FallbackMaterial},
    materials::registry::MaterialRegistry,
    solver::density::{estimate_initial_particle_volumes, estimate_particle_density_and_volume},
    transfer::{gather_grid_to_particles, scatter_particles_to_grid},
};
use crate::{
    grid::Grid,
    particle::{Particle, Particles},
};
use crate::thermodynamics::{ScalarDiffusionField, ThermalDiffusion};

pub struct MpmSolver {
    config: SolverConfig,
    particles: Particles,
    grid: Grid,
    materials: MaterialRegistry,
    boundaries: Vec<Box<dyn BoundaryCondition>>,
    force_fields: Vec<(String, Box<dyn ForceField>)>,
    thermal: Option<ThermalDiffusion>,
    /// Scalar diffusion fields (pheromone, nutrients, morphogen) — run automatically each substep.
    scalar_fields: Vec<ScalarDiffusionField>,
    frame_index: u64,
    last_step_dt: f32,
    last_substeps: usize,
    last_vel_clamp_count: usize,
    last_j_projection_count: usize,
    last_sim_time_dropped: f32,
    /// Automatic phase transition rules, evaluated every substep.
    phase_rules: Vec<Box<dyn Fn(&Particle) -> Option<u32> + Send + Sync>>,
}

impl MpmSolver {
    pub fn new(config: SolverConfig, spawn: SpawnConfig) -> Self {
        config.validate();
        spawn.validate_for_solver(&config);


        let mut rng = LcgRng::new(spawn.rng_seed);
        let mut particles = Particles::from(initialize_particles(&config, spawn, &mut rng));
        let mut grid = Grid::new(config.grid_res);
        if spawn.precompute_initial_volumes {
            estimate_initial_particle_volumes(&mut particles, &mut grid);
        }
        let materials = MaterialRegistry::with_default(Box::new(FallbackMaterial));
        let default_boundary: Box<dyn BoundaryCondition> =
            Box::new(SlipBoundary::new(config.boundary_thickness));
        Self {
            config,
            particles,
            grid,
            materials,
            boundaries: vec![default_boundary],
            force_fields: Vec::new(),
            thermal: None,
            scalar_fields: Vec::new(),
            frame_index: 0,
            last_step_dt: config.dt,
            last_substeps: 0,
            last_vel_clamp_count: 0,
            last_j_projection_count: 0,
            last_sim_time_dropped: 0.0,
            phase_rules: Vec::new(),
        }
    }

    pub fn with_default_material(mut self, material: Box<dyn MaterialModel>) -> Self {
        self.set_default_material(material);
        self.reinit_all_particle_state();
        self
    }

    /// Append a boundary condition (builder). Multiple boundaries are applied in order.
    pub fn with_boundary(mut self, boundary: Box<dyn BoundaryCondition>) -> Self {
        self.add_boundary_condition(boundary);
        self
    }

    /// Append an anonymous force field (auto-named "force_field_N").
    pub fn with_force_field(mut self, field: Box<dyn ForceField>) -> Self {
        self.add_force_field(field);
        self
    }

    /// Append a named force field — name can be used later to remove or replace it.
    pub fn with_named_force_field(
        mut self,
        name: impl Into<String>,
        field: Box<dyn ForceField>,
    ) -> Self {
        self.add_named_force_field(name, field);
        self
    }

    pub fn with_thermal(mut self, thermal: ThermalDiffusion) -> Self {
        self.thermal = Some(thermal);
        self
    }

    pub fn set_thermal(&mut self, thermal: ThermalDiffusion) {
        self.thermal = Some(thermal);
    }

    /// Register a material and return its typed `MaterialHandle`.
    ///
    /// Preferred over `with_material(id, mat)` — handle is type-safe, auto-allocates ID.
    /// ```rust,no_run
    /// # use emerge::solver::MpmSolver;
    /// # use emerge::{SolverConfig, SpawnConfig, NewtonianFluidMaterial};
    /// # let config = SolverConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
    /// # let mut solver = MpmSolver::new(config, SpawnConfig::default());
    /// let water = solver.register_material(Box::new(NewtonianFluidMaterial::water(1000.0, 1e4)));
    /// // use water.id() in SpawnConfig or phase_transition
    /// ```
    pub fn register_material(&mut self, material: Box<dyn MaterialModel>) -> MaterialHandle {
        let id = self.materials.next_id();
        self.materials.insert(id, material);
        MaterialHandle(id)
    }

    /// Builder variant of `register_material` — chains with other `.with_*` calls.
    /// Note: returns `(Self, MaterialHandle)` so the handle is accessible.
    pub fn with_registered_material(
        mut self,
        material: Box<dyn MaterialModel>,
    ) -> (Self, MaterialHandle) {
        let handle = self.register_material(material);
        (self, handle)
    }

    pub fn with_material(mut self, material_id: u32, material: Box<dyn MaterialModel>) -> Self {
        self.set_material(material_id, material);
        self
    }

    pub fn with_particle_materials_by_position<F>(mut self, material_for: F) -> Self
    where
        F: FnMut(Vec2) -> u32,
    {
        self.assign_particle_materials_by_position(material_for);
        self
    }

    pub fn config(&self) -> &SolverConfig {
        &self.config
    }

    pub fn particles(&self) -> &Particles {
        &self.particles
    }

    /// Direct mutable access to all particles.
    ///
    /// **CFL WARNING:** velocity changes made here bypass the solver's CFL clamp.
    /// Any velocity written must satisfy `|v| ≤ grid_cell_size / current_sub_dt` or the
    /// next P2G scatter will inject extreme momentum → J→0 → deformation collapse.
    /// For gameplay impulses use `apply_impulse` / `apply_radial_impulse` instead.
    /// Safe uses: writing non-velocity fields (temperature, activation, user_tag, material_id).
    pub fn particles_mut(&mut self) -> &mut Particles {
        &mut self.particles
    }

    pub fn assign_particle_materials_by_position<F>(&mut self, mut material_for: F)
    where
        F: FnMut(Vec2) -> u32,
    {
        for i in self.particles.indices() {
            let new_id = material_for(self.particles.x[i]);
            self.particles.material_id[i] = new_id;
        }
        self.reinit_all_particle_state();
    }

    /// Re-run `init_particle` on every particle using its current material_id.
    ///
    /// Call after bulk material reassignment (e.g. `assign_particle_materials_by_position`)
    /// or after `with_default_material` when the first spawn happened before material
    /// registration. Materials that don't override `init_particle` are a no-op.
    pub fn reinit_all_particle_state(&mut self) {
        for i in 0..self.particles.len() {
            let mut p = self.particles.get(i);
            self.materials.get(p.material_id).init_particle(&mut p);
            self.particles.set(i, p);
        }
    }

    pub fn material_particle_counts(&self) -> BTreeMap<u32, usize> {
        let mut counts = BTreeMap::new();
        for &id in &self.particles.material_id {
            *counts.entry(id).or_insert(0) += 1;
        }
        counts
    }

    pub fn set_default_material(&mut self, material: Box<dyn MaterialModel>) {
        self.materials.set_default(material);
    }

    pub fn set_material(&mut self, material_id: u32, material: Box<dyn MaterialModel>) {
        self.materials.insert(material_id, material);
    }

    /// Replace all boundary conditions with one (backwards-compat).
    pub fn set_boundary_condition(&mut self, boundary: Box<dyn BoundaryCondition>) {
        self.boundaries.clear();
        self.boundaries.push(boundary);
    }

    /// Append an additional boundary condition (stacks with existing ones).
    pub fn add_boundary_condition(&mut self, boundary: Box<dyn BoundaryCondition>) {
        self.boundaries.push(boundary);
    }

    /// Remove all boundary conditions.
    pub fn clear_boundaries(&mut self) {
        self.boundaries.clear();
    }

    /// Append an anonymous force field (auto-named "force_field_N").
    pub fn add_force_field(&mut self, field: Box<dyn ForceField>) {
        let name = format!("force_field_{}", self.force_fields.len());
        self.force_fields.push((name, field));
    }

    /// Append a named force field.
    pub fn add_named_force_field(&mut self, name: impl Into<String>, field: Box<dyn ForceField>) {
        self.force_fields.push((name.into(), field));
    }

    /// Remove the first force field with this name. Returns true if found and removed.
    pub fn remove_force_field(&mut self, name: &str) -> bool {
        if let Some(pos) = self.force_fields.iter().position(|(n, _)| n == name) {
            self.force_fields.remove(pos);
            true
        } else {
            false
        }
    }

    /// Remove all force fields.
    pub fn clear_force_fields(&mut self) {
        self.force_fields.clear();
    }

    /// Names of all currently active force fields, in application order.
    pub fn force_field_names(&self) -> Vec<&str> {
        self.force_fields.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn gravity(&self) -> Vec2 {
        self.config.gravity
    }

    pub fn set_gravity(&mut self, gravity: Vec2) {
        self.config.gravity = gravity;
    }

    pub fn diagnostics_snapshot(&self) -> MpmSnapshot {
        let mut snap = collect_mpm_snapshot(
            self.frame_index,
            &self.particles,
            &self.grid,
            &self.config,
            self.last_step_dt,
            self.last_substeps,
        );
        snap.vel_clamp_count = self.last_vel_clamp_count;
        snap.j_projection_count = self.last_j_projection_count;
        snap.sim_time_dropped = self.last_sim_time_dropped;
        snap
    }

    // ── Query & Transition API ────────────────────────────────────────────────

    /// Aggregate state for all particles of a given material.
    pub fn material_state(&self, material_id: u32) -> MaterialState {
        material_state_of(&self.particles, material_id)
    }

    /// Aggregate state for all particles within `radius` grid-cells of `center`.
    pub fn region_state(&self, center: Vec2, radius: f32) -> MaterialState {
        region_state_of(&self.particles, center, radius)
    }

    /// Iterate over particles within `radius` grid-cells of `center`.
    ///
    /// Yields `(index, &Particle)` so LP can read state and use the index
    /// to apply targeted impulses or phase transitions by index.
    ///
    /// O(N) — fine for organism-scale queries (hundreds to low thousands of particles).
    /// For planetary-scale queries, use `region_state` aggregates instead.
    ///
    /// # Example — cell adhesion sensing
    /// ```rust,ignore
    /// for (idx, neighbor) in solver.particles_near(cell.x, sensing_radius) {
    ///     if neighbor.material_id == FOOD_ID {
    ///         solver.particles_mut()[idx].user_tag = CONSUMED;
    ///     }
    /// }
    /// ```
    pub fn particles_near(
        &self,
        center: Vec2,
        radius: f32,
    ) -> impl Iterator<Item = (usize, Particle)> + '_ {
        let r2 = radius * radius;
        self.particles.indices().filter_map(move |i| {
            if (self.particles.x[i] - center).length_squared() <= r2 {
                Some((i, self.particles.get(i)))
            } else {
                None
            }
        })
    }

    /// Count particles of a given material within `radius` of `center`. O(N).
    pub fn count_near(&self, center: Vec2, radius: f32, material_id: u32) -> usize {
        let r2 = radius * radius;
        self.particles.indices().filter(|&i| {
            self.particles.material_id[i] == material_id
                && (self.particles.x[i] - center).length_squared() <= r2
        }).count()
    }

    /// Switch material for every particle where `predicate` returns true.
    ///
    /// After a transition involving fluid materials, call `recompute_initial_volumes()`
    /// if density has shifted significantly.
    pub fn phase_transition<F>(&mut self, predicate: F, new_material_id: u32)
    where
        F: Fn(&Particle) -> bool,
    {
        assert!(
            self.materials.is_registered(new_material_id),
            "phase_transition: material_id {new_material_id} is not registered — \
             call solver.with_material({new_material_id}, ...) first"
        );
        for i in 0..self.particles.len() {
            let p = self.particles.get(i);
            if predicate(&p) {
                self.particles.material_id[i] = new_material_id;
            }
        }
    }

    /// Register an automatic phase transition rule, evaluated every substep.
    ///
    /// `rule` receives a particle and returns `Some(new_material_id)` to transition it,
    /// or `None` to leave it unchanged. Rules are checked in registration order;
    /// first match wins for each particle.
    ///
    /// All `new_material_id` values returned by the rule must be pre-registered via
    /// `solver.with_material(id, ...)` before any step is taken.
    ///
    /// # Examples
    /// ```rust,ignore
    /// // Water freezes below 273 K
    /// solver.add_phase_rule(|p| {
    ///     if p.material_id == WATER_ID && p.temperature < 273.0 { Some(ICE_ID) } else { None }
    /// });
    /// // Rock melts above 1500 K
    /// solver.add_phase_rule(|p| {
    ///     if p.material_id == ROCK_ID && p.temperature > 1500.0 { Some(LAVA_ID) } else { None }
    /// });
    /// ```
    pub fn add_phase_rule<F>(&mut self, rule: F)
    where
        F: Fn(&Particle) -> Option<u32> + Send + Sync + 'static,
    {
        self.phase_rules.push(Box::new(rule));
    }

    /// Builder-style variant of `add_phase_rule`.
    pub fn with_phase_rule<F>(mut self, rule: F) -> Self
    where
        F: Fn(&Particle) -> Option<u32> + Send + Sync + 'static,
    {
        self.add_phase_rule(rule);
        self
    }

    /// Apply a velocity delta to all particles within `radius` of `center`, with linear falloff.
    /// `force` units: grid-cell/s (instantaneous velocity change).
    /// Result is clamped to the solver's CFL velocity limit so LP impulses can't break stability.
    ///
    /// **This is the safe API for external impulses.** Always prefer this over `particles_mut()`
    /// for any gameplay-driven velocity change — direct mutation bypasses the CFL clamp and can
    /// collapse deformation gradients (J→0) under large forces.
    ///
    /// KNOWN OPEN ISSUE: the CFL clamp here uses `min_dt`, which is a conservative bound.
    /// Under adaptive substeps the actual sub_dt may be larger, making the clamp overly
    /// permissive. True safety requires clamping to `current_sub_dt` at the moment of application,
    /// but `apply_impulse` is called between solver steps where `current_sub_dt` is unknown.
    /// Options under research: (a) grid-velocity projection post-P2G, (b) semi-implicit
    /// integration, (c) energy-bounded impulse splitting across substeps. See fields/mod.rs.
    pub fn apply_impulse(&mut self, center: Vec2, radius: f32, force: Vec2) {
        let vel_limit = self.config.grid_cell_size / self.config.min_dt;
        let r2 = radius * radius;
        for i in 0..self.particles.len() {
            let d = self.particles.x[i] - center;
            let dist2 = d.length_squared();
            if dist2 <= r2 && dist2 > 1e-8 {
                let falloff = 1.0 - (dist2 / r2).sqrt();
                self.particles.v[i] += force * falloff;
                let spd = self.particles.v[i].length();
                if spd > vel_limit {
                    self.particles.v[i] *= vel_limit / spd;
                }
            }
        }
    }

    /// Apply an outward radial velocity delta to particles within `radius`, with linear falloff.
    /// Result is clamped to the solver's CFL velocity limit.
    pub fn apply_radial_impulse(&mut self, center: Vec2, radius: f32, strength: f32) {
        let vel_limit = self.config.grid_cell_size / self.config.min_dt;
        let r2 = radius * radius;
        for i in 0..self.particles.len() {
            let d = self.particles.x[i] - center;
            let dist2 = d.length_squared();
            if dist2 <= r2 && dist2 > 1e-8 {
                let dist = dist2.sqrt();
                let falloff = 1.0 - dist / radius;
                self.particles.v[i] += (d / dist) * strength * falloff;
                let spd = self.particles.v[i].length();
                if spd > vel_limit {
                    self.particles.v[i] *= vel_limit / spd;
                }
            }
        }
    }

    /// One MLS-MPM timestep: particle→grid→particle cycle.
    /// The grid is temporary scratch — only particles hold long-term material memory.
    pub fn step(&mut self) {
        // Adaptive substep loop: step() always advances exactly config.dt of simulation time,
        // but uses smaller sub-steps when CFL requires it (stiff materials, high velocities).
        // Without this loop, the FixedStepController accounts for config.dt per call but the
        // simulation only advances sub_dt — causing it to run orders of magnitude too slowly.
        let mut remaining = self.config.dt;
        let mut substeps_taken = 0;
        self.last_vel_clamp_count = 0;
        self.last_j_projection_count = 0;
        while remaining > f32::EPSILON && substeps_taken < self.config.max_substeps_per_step {
            // Cap sub-step at remaining time so we don't overshoot the configured frame dt.
            let sub_dt =
                choose_substep_dt(&self.config, &self.particles, &self.materials, remaining);
            self.do_substep(sub_dt);
            remaining -= sub_dt;
            self.last_step_dt = sub_dt;
            substeps_taken += 1;
        }
        self.last_substeps = substeps_taken;
        self.last_sim_time_dropped = remaining.max(0.0);
        self.frame_index = self.frame_index.saturating_add(1);
    }

    fn do_substep(&mut self, sub_dt: f32) {
        // Project invalid particle state before it can corrupt the grid scatter.
        // Running pre-P2G (not post) means a bad particle from a previous substep is
        // fixed before its momentum enters the grid — no NaN cascade possible.
        if self.config.project_invalid_state {
            for i in 0..self.particles.len() {
                let mut p = self.particles.get(i);
                if project_particle_state_to_admissible(&mut p, &self.config) {
                    self.last_j_projection_count += 1;
                }
                self.particles.set(i, p);
            }
        }

        // Density recompute: fluid EOS materials need current ρ each substep (pressure = f(ρ)).
        // Auto-enabled when any registered material declares needs_density_recompute=true.
        // Manual override via config.recompute_density_each_step for edge cases.
        if self.config.recompute_density_each_step || self.materials.any_needs_density_recompute() {
            estimate_particle_density_and_volume(&mut self.particles, &mut self.grid);
        }

        self.grid.clear();
        scatter_particles_to_grid(
            &self.particles,
            &mut self.grid,
            &self.materials,
            sub_dt,
        );

        // Normalize accumulated momentum to velocity, then apply gravity and wall constraints.
        self.grid.update_velocities(sub_dt, self.config.gravity);
        let grid_res = self.grid.resolution();
        for boundary in &self.boundaries {
            apply_boundary_conditions_to_grid(&mut self.grid, grid_res, boundary.as_ref());
        }

        // Grid velocity projection — clamp every cell to CFL-safe speed before G2P.
        //
        // This is the root fix for large-force instability. G2P gathers both v_p and the
        // APIC affine matrix C_p from grid velocities. The post-G2P particle velocity clamp
        // only covers v_p — C_p is unclamped. Large C_p → F = (I + dt·C)·F blows up → J→0.
        // Clamping here bounds both v_p and C_p at the source, regardless of how the velocity
        // got there (gravity, force fields applied previous step, or impulses between steps).
        //
        // Trade-off: cells with zero mass have no physical velocity — skip them (momentum=0).
        // Only mass-bearing cells can produce valid C_p contributions anyway.
        //
        // This is Tier 1 of the large-force stability plan (see apply_impulse doc).
        // Tier 2 (force-triggered substepping) will further reduce energy loss for large
        // one-shot impulses. Tier 3 (affine projection stabilizer, 2025) would make this
        // unnecessary, but is deferred until LP stress-tests demand it.
        {
            let vel_limit = self.config.grid_cell_size / sub_dt;
            for cell in self.grid.active_cells_mut() {
                if cell.mass > 0.0 {
                    let spd = cell.momentum.length();
                    if spd > vel_limit {
                        cell.momentum *= vel_limit / spd;
                    }
                }
            }
        }

        self.last_vel_clamp_count += gather_grid_to_particles(
            &mut self.particles,
            &self.grid,
            sub_dt,
            &self.boundaries,
            &self.materials,
            self.config.grid_cell_size / sub_dt,
            self.config.apic_blend,
        );

        // External body force fields: v += dt × acceleration(p) per particle.
        // Applied after G2P so each field sees the fully gathered particle state.
        // A post-field velocity clamp (same limit as G2P) prevents large impulses from
        // leaving particles with >1 cell/substep velocity that P2G then scatters as extreme
        // momentum — the clamp re-asserts the CFL contract after external perturbation.
        // prepare() is called first so stateful fields (e.g. Barnes-Hut tree) can
        // rebuild their internal state from the current particle snapshot.
        if !self.force_fields.is_empty() {
            let mut fields = std::mem::take(&mut self.force_fields);
            for (_, field) in &mut fields {
                field.prepare(&self.particles);
            }
            for i in 0..self.particles.len() {
                let p = self.particles.get(i);
                let mut dv = Vec2::ZERO;
                for (_, field) in &fields {
                    dv += field.acceleration(&p);
                }
                self.particles.v[i] += sub_dt * dv;
            }
            self.force_fields = fields;

            // Re-clamp velocity after force fields — large external impulses (explosions,
            // creature bursts, planetary impacts) must not enter P2G with >1 cell/substep.
            let vel_limit = self.config.grid_cell_size / sub_dt;
            for i in 0..self.particles.len() {
                let spd = self.particles.v[i].length();
                if spd > vel_limit {
                    self.particles.v[i] *= vel_limit / spd;
                }
            }
        }

        // Thermal diffusion — grid-based Fourier heat exchange between particles.
        if let Some(thermal) = &mut self.thermal {
            thermal.apply(&mut self.particles, sub_dt);
        }

        // Scalar diffusion fields (pheromone, nutrients, morphogen) — auto-managed.
        for field in &mut self.scalar_fields {
            field.apply(&mut self.particles, sub_dt);
        }

        // Automatic phase transitions — evaluate registered rules, first match wins.
        if !self.phase_rules.is_empty() {
            let rules = std::mem::take(&mut self.phase_rules);
            for i in 0..self.particles.len() {
                let p = self.particles.get(i);
                for rule in &rules {
                    if let Some(new_id) = rule(&p) {
                        self.particles.material_id[i] = new_id;
                        break;
                    }
                }
            }
            self.phase_rules = rules;
        }

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

    pub fn recompute_initial_volumes(&mut self) {
        estimate_initial_particle_volumes(&mut self.particles, &mut self.grid);
    }

    /// Remove all particles where `predicate` returns true. Returns count removed.
    ///
    /// Uses stable retain (preserves order). O(N).
    ///
    /// **Important:** any `ParticleGroup` index ranges are invalidated after removal —
    /// rebuild groups via `spawn_group` or `particles_with_tag` afterward.
    /// LP pattern: tag particles to remove with a sentinel `user_tag`, then call
    /// `solver.remove_particles(|p| p.user_tag == DEAD)`.
    pub fn remove_particles<F: Fn(&Particle) -> bool>(&mut self, predicate: F) -> usize {
        let before = self.particles.len();
        self.particles.retain(|p| !predicate(p));
        before - self.particles.len()
    }

    /// Iterate all particles with the given `user_tag`. O(N).
    ///
    /// Use this to rebuild a `ParticleGroup` after `remove_particles` shifts indices,
    /// or to query all particles belonging to a creature / region by ownership tag.
    pub fn particles_with_tag(&self, tag: u32) -> impl Iterator<Item = (usize, Particle)> + '_ {
        self.particles.indices().filter_map(move |i| {
            if self.particles.user_tag[i] == tag {
                Some((i, self.particles.get(i)))
            } else {
                None
            }
        })
    }

    /// Collect all particles into a `Vec<Particle>` (for diagnostics or GPU upload).
    pub fn collect_particles(&self) -> Vec<Particle> {
        self.particles.to_vec()
    }

    /// Spawn an additional region of particles and append them to the existing simulation.
    ///
    /// Returns the index range `start..end` into `self.particles()` for the newly added
    /// particles. Store this range to track ownership (e.g. map a body ID → particle slice).
    ///
    /// Recomputes initial volumes for the full particle set after spawning — new particles'
    /// densities depend on existing neighbours, so this is O(N_total) per call.
    /// Avoid calling `spawn_region` in a hot loop at large N.
    ///
    /// ```ignore
    /// let fluid_range = solver.spawn_region(SpawnConfig { box_center: fluid_center, .. });
    /// let sand_range  = solver.spawn_region(SpawnConfig { box_center: sand_center,  .. });
    /// ```
    #[must_use = "returns the particle index range for the spawned region — store it to track ownership"]
    pub fn spawn_region(&mut self, spawn: SpawnConfig) -> std::ops::Range<usize> {
        let start = self.particles.len();
        spawn.validate_for_solver(&self.config);
        debug_assert!(
            self.materials.is_registered(spawn.material_id),
            "spawn_region: material_id {} is not registered — call solver.with_material({}, ...) first",
            spawn.material_id, spawn.material_id
        );
        let mut rng = LcgRng::new(spawn.rng_seed);
        let new_particles = initialize_particles(&self.config, spawn, &mut rng);
        for p in new_particles {
            self.particles.push(p);
        }
        // Let the material seed its own per-particle plastic state (e.g. sand friction accumulator).
        let mat_id = spawn.material_id;
        if self.materials.is_registered(mat_id) {
            for i in start..self.particles.len() {
                let mut p = self.particles.get(i);
                self.materials.get(mat_id).init_particle(&mut p);
                self.particles.set(i, p);
            }
        }
        // Volume estimation is neighbour-weighted (P2G mass scatter → per-particle density),
        // so it must run over all particles — new particles' densities depend on existing neighbours.
        // This is O(N_total) per call; avoid calling spawn_region in a hot loop at large N.
        estimate_initial_particle_volumes(&mut self.particles, &mut self.grid);
        start..self.particles.len()
    }

    /// Spawn particles and return a typed `ParticleGroup` handle.
    ///
    /// Preferred ergonomic API over `spawn_region` — the group tracks the index
    /// range and provides bulk operations (set_activation, apply_impulse, state).
    ///
    /// ```rust,no_run
    /// # use emerge::solver::MpmSolver;
    /// # use emerge::{SolverConfig, SpawnConfig};
    /// # let config = SolverConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
    /// # let mut solver = MpmSolver::new(config, SpawnConfig::default());
    /// let creature = solver.spawn_group(SpawnConfig { ..Default::default() });
    /// creature.set_activation(&mut solver.particles_mut(), 1.0);
    /// let centroid = creature.centroid(solver.particles());
    /// ```
    #[must_use = "store the ParticleGroup to track this region's particles"]
    pub fn spawn_group(&mut self, spawn: SpawnConfig) -> ParticleGroup {
        ParticleGroup::new(self.spawn_region(spawn))
    }

    /// Spawn a named group — label appears in diagnostics output.
    #[must_use = "store the ParticleGroup to track this region's particles"]
    pub fn spawn_named_group(&mut self, spawn: SpawnConfig, label: impl Into<String>) -> ParticleGroup {
        ParticleGroup::named(self.spawn_region(spawn), label)
    }

    /// Attach a scalar diffusion field (pheromone, nutrients, morphogen).
    ///
    /// Attached fields are applied automatically every substep — LP does not need
    /// to call `field.apply()` manually.
    ///
    /// ```rust,no_run
    /// # use emerge::solver::MpmSolver;
    /// # use emerge::{SolverConfig, SpawnConfig, ScalarDiffusionConfig, ScalarDiffusionField};
    /// # let config = SolverConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
    /// # let mut solver = MpmSolver::new(config, SpawnConfig::default());
    /// let pheromone = ScalarDiffusionField::new(
    ///     ScalarDiffusionConfig { diffusivity: 0.5, decay_rate: 0.1, ambient: 0.0 },
    ///     |p| p.temperature,
    ///     |p, d| p.temperature += d,
    ///     64,
    /// );
    /// solver.attach_scalar_field(pheromone);
    /// // No manual field.apply() needed — runs automatically in solver.step()
    /// ```
    pub fn attach_scalar_field(&mut self, field: ScalarDiffusionField) {
        self.scalar_fields.push(field);
    }

    /// Builder variant of `attach_scalar_field`.
    pub fn with_scalar_field(mut self, field: ScalarDiffusionField) -> Self {
        self.attach_scalar_field(field);
        self
    }
}

pub(crate) fn initialize_particles(
    config: &SolverConfig,
    spawn: SpawnConfig,
    rng: &mut LcgRng,
) -> Vec<Particle> {
    use crate::solver::config::SpawnShape;
    let mut particles = Vec::new();
    let half = spawn.box_size.as_vec2() * 0.5;
    let min = spawn.box_center - half;
    let max = spawn.box_center + half;

    let mut i = min.x;
    while i < max.x {
        let mut j = min.y;
        while j < max.y {
            let pos = Vec2::new(i, j);

            // Apply shape mask — skip particles outside the disk if disk shape is active.
            let inside = match spawn.shape {
                SpawnShape::Box => true,
                SpawnShape::Disk { radius } => (pos - spawn.box_center).length() <= radius,
            };

            if inside {
                let jitter_mag = spawn.position_jitter * spawn.spacing;
                let jx = (rng.next_f32() - 0.5) * 2.0 * jitter_mag;
                let jy = (rng.next_f32() - 0.5) * 2.0 * jitter_mag;
                let jittered_pos = pos + Vec2::new(jx, jy);
                let random = Vec2::new(rng.next_f32(), rng.next_f32());
                let velocity = (random - Vec2::splat(0.5)) * spawn.initial_velocity_scale;
                particles.push(Particle {
                    x: jittered_pos,
                    v: velocity,
                    velocity_gradient: Mat2::ZERO,
                    deformation_gradient: spawn.initial_deformation_gradient,
                    mass: config.particle_mass,
                    initial_volume: config.default_initial_volume,
                    volume: config.default_initial_volume,
                    density: config.particle_mass / config.default_initial_volume,
                    material_id: spawn.material_id,
                    plastic_volume_ratio: 1.0,
                    hardening_scale: 1.0,
                    friction_hardening: 0.0,
                    log_volume_strain: 0.0,
                    temperature: 0.0,
                    user_tag: 0,
                    activation: 0.0,
                    activation_dir: Vec2::ZERO,
                    muscle_group_id: 0,
                    _pad: 0,
                });
            }

            j += spawn.spacing;
        }
        i += spawn.spacing;
    }

    particles
}

#[derive(Debug)]
pub(crate) struct LcgRng {
    state: u32,
}

impl LcgRng {
    pub(crate) fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next_u32(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        self.next_u32() as f32 / (u32::MAX as f32 + 1.0)
    }
}

// choose_substep_dt: picks the largest CFL-safe dt ≤ max_dt.
// Called inside step()'s substep loop — max_dt is the remaining frame time.
// pub(crate) so the GPU solver can reuse this without duplicating CFL logic.
pub(crate) fn choose_substep_dt(
    config: &SolverConfig,
    particles: &Particles,
    materials: &MaterialRegistry,
    max_dt: f32,
) -> f32 {
    if !config.adaptive_timestep {
        return max_dt.min(config.dt);
    }
    let mut max_speed = 0.0f32;
    for i in 0..particles.len() {
        let mut s = particles.v[i].length();
        if config.cfl_include_affine_speed {
            s += affine_cfl_speed_contribution(&particles.velocity_gradient[i], config.grid_cell_size);
        }
        max_speed = max_speed.max(s);
    }
    let mut min_mat_dt = max_dt;
    for i in 0..particles.len() {
        let p = particles.get(i);
        let mdt = materials.get(p.material_id).timestep_bound(
            &p, config.grid_cell_size, config.material_cfl_coefficient, config.viscous_timestep_coefficient,
        );
        if mdt.is_finite() && mdt > 0.0 { min_mat_dt = min_mat_dt.min(mdt); }
    }
    cfl_bound(config, max_speed, min_mat_dt, max_dt)
}

/// CFL scan over a flat AoS `&[Particle]` slice — used by the GPU path which keeps
/// particles as `Vec<Particle>` to avoid SoA conversion overhead.
#[cfg(feature = "gpu")]
pub(crate) fn choose_substep_dt_flat(
    config: &SolverConfig,
    particles: &[Particle],
    materials: &MaterialRegistry,
    max_dt: f32,
) -> f32 {
    if !config.adaptive_timestep {
        return max_dt.min(config.dt);
    }
    let mut max_speed = 0.0f32;
    for p in particles {
        let mut s = p.v.length();
        if config.cfl_include_affine_speed {
            s += affine_cfl_speed_contribution(&p.velocity_gradient, config.grid_cell_size);
        }
        max_speed = max_speed.max(s);
    }
    let mut min_mat_dt = max_dt;
    for p in particles {
        let mdt = materials.get(p.material_id).timestep_bound(
            p, config.grid_cell_size, config.material_cfl_coefficient, config.viscous_timestep_coefficient,
        );
        if mdt.is_finite() && mdt > 0.0 { min_mat_dt = min_mat_dt.min(mdt); }
    }
    cfl_bound(config, max_speed, min_mat_dt, max_dt)
}

/// Shared CFL formula: clamps dt to advection + material bounds.
/// Called by both SoA and AoS scan paths after computing their respective max values.
fn cfl_bound(config: &SolverConfig, max_speed: f32, min_mat_dt: f32, max_dt: f32) -> f32 {
    let mut dt = max_dt;
    if max_speed > f32::EPSILON {
        dt = dt.min(config.cfl_coefficient * config.grid_cell_size / max_speed);
    }
    dt = dt.min(min_mat_dt);
    dt.clamp(config.min_dt.min(max_dt), max_dt)
}

fn affine_cfl_speed_contribution(c: &Mat2, cell_width: f32) -> f32 {
    // The APIC affine matrix C encodes the local velocity gradient.
    // The farthest point in the quadratic B-spline 3×3 stencil is at 1.5 cells per axis,
    // so its corner distance is 1.5*√2 cells — the effective maximum affine speed contribution.
    const STENCIL_CORNER_DISTANCE: f32 = 1.5 * std::f32::consts::SQRT_2;
    let grad_norm = (c.x_axis.length_squared() + c.y_axis.length_squared()).sqrt();
    grad_norm * STENCIL_CORNER_DISTANCE * cell_width
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
fn project_particle_state_to_admissible(particle: &mut Particle, config: &SolverConfig) -> bool {
    let mut projected = false;
    let min = config.boundary_thickness.saturating_sub(1) as f32;
    let max = config.grid_res.saturating_sub(config.boundary_thickness) as f32;
    let domain_center = Vec2::splat((min + max) * 0.5);

    if !particle.x.is_finite() {
        particle.x = domain_center;
        projected = true;
    } else {
        particle.x = particle.x.clamp(Vec2::splat(min), Vec2::splat(max));
    }

    if !particle.v.is_finite() {
        particle.v = Vec2::ZERO;
        projected = true;
    }
    if !particle.velocity_gradient.x_axis.is_finite()
        || !particle.velocity_gradient.y_axis.is_finite()
    {
        particle.velocity_gradient = Mat2::ZERO;
        projected = true;
    }

    if !particle.deformation_gradient.x_axis.is_finite()
        || !particle.deformation_gradient.y_axis.is_finite()
        || particle.deformation_gradient.determinant() <= config.projection_min_deformation_j
    {
        particle.deformation_gradient = Mat2::IDENTITY;
        projected = true;
    } else {
        let j = particle.deformation_gradient.determinant();
        if j > config.j_max {
            particle.deformation_gradient *= (config.j_max / j).sqrt();
            projected = true;
        }
    }

    if !particle.plastic_volume_ratio.is_finite() || particle.plastic_volume_ratio <= 0.0 {
        particle.plastic_volume_ratio = 1.0;
        projected = true;
    }
    if !particle.hardening_scale.is_finite() || particle.hardening_scale <= 0.0 {
        particle.hardening_scale = 1.0;
        projected = true;
    }
    if !particle.friction_hardening.is_finite() {
        particle.friction_hardening = 0.0;
        projected = true;
    }
    if !particle.log_volume_strain.is_finite() {
        particle.log_volume_strain = 0.0;
        projected = true;
    }

    if !particle.mass.is_finite() || particle.mass <= 0.0 {
        particle.mass = config.particle_mass;
        projected = true;
    }
    if !particle.initial_volume.is_finite() || particle.initial_volume <= 0.0 {
        particle.initial_volume = config
            .default_initial_volume
            .max(config.projection_min_volume);
        projected = true;
    }
    if !particle.volume.is_finite() || particle.volume <= 0.0 {
        particle.volume = particle.initial_volume.max(config.projection_min_volume);
        projected = true;
    }
    if !particle.density.is_finite() || particle.density <= 0.0 {
        particle.density = (particle.mass / particle.volume).max(config.projection_min_density);
        projected = true;
    } else {
        particle.density = particle.density.max(config.projection_min_density);
    }
    projected
}
