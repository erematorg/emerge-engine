pub mod config;
pub mod cutoff;
pub mod density;
pub mod handle;
pub mod query;
pub mod spatial_hash;

pub use config::{SolverConfig, SpawnConfig};
pub use cutoff::smooth_cutoff;
pub use density::compute_density_grid;
pub use handle::{MaterialHandle, ParticleGroup};
pub use query::{MaterialState, material_state_of, region_state_of};

use std::collections::{BTreeMap, HashMap, HashSet};

use spatial_hash::SpatialHash;

use glam::{Mat2, Vec2};

use crate::diagnostics::{MpmSnapshot, collect_mpm_snapshot};
use crate::thermodynamics::{ScalarDiffusionField, ThermalDiffusion};
use crate::{
    boundary::{BoundaryCondition, SlipBoundary},
    fields::ForceField,
    materials::registry::MaterialRegistry,
    materials::{FallbackMaterial, MaterialModel},
    solver::density::estimate_particle_volumes,
    transfer::{gather_grid_to_particles, scatter_particles_to_grid},
};
use crate::{
    grid::Grid,
    particle::{Particle, Particles},
};

pub struct MpmSolver {
    config: SolverConfig,
    particles: Particles,
    /// Partition boundary: particles[0..active_count] are active, [active_count..N] sleeping.
    /// P2G / G2P only visit [0..active_count]. Maintained by sleep_particle / wake_particle.
    active_count: usize,
    /// Maps user_tag → physical indices of all particles with that tag.
    /// HashSet gives O(1) insert/remove on every sleep/wake swap.
    tag_index: HashMap<u32, HashSet<usize>>,
    /// Monotonically increasing counter — next tag issued by spawn_group.
    next_tag: u32,
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
    /// Spatial hash over active particles — rebuilt each substep after G2P.
    /// Turns O(N) radius queries into O(candidates_in_neighborhood).
    spatial_hash: SpatialHash,
}

impl MpmSolver {
    /// Create an empty solver with no particles. Use `spawn_region` to add particles.
    pub fn empty(config: SolverConfig) -> Self {
        config.validate();
        let materials = MaterialRegistry::with_default(Box::new(FallbackMaterial));
        let default_boundary: Box<dyn BoundaryCondition> =
            Box::new(SlipBoundary::new(config.boundary_thickness));
        Self {
            config,
            particles: Particles::default(),
            active_count: 0,
            tag_index: HashMap::new(),
            next_tag: 1,
            grid: Grid::new(config.grid_res),
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
            spatial_hash: SpatialHash::new(config.grid_cell_size),
        }
    }

    pub fn new(config: SolverConfig, spawn: SpawnConfig) -> Self {
        config.validate();
        spawn.validate_for_solver(&config);

        let mut rng = LcgRng::new(spawn.rng_seed);
        let mut particles = Particles::from(initialize_particles(&config, spawn, &mut rng));
        let mut grid = Grid::new(config.grid_res);
        if spawn.precompute_initial_volumes {
            let n = particles.len();
            estimate_particle_volumes(&mut particles, &mut grid, n, true);
        }
        let materials = MaterialRegistry::with_default(Box::new(FallbackMaterial));
        let default_boundary: Box<dyn BoundaryCondition> =
            Box::new(SlipBoundary::new(config.boundary_thickness));
        let active_count = particles.len();
        let mut tag_index: HashMap<u32, HashSet<usize>> = HashMap::new();
        if active_count > 0 {
            // Initial particles carry user_tag=0; register them so group ops work.
            tag_index.insert(0, (0..active_count).collect());
        }
        let mut solver = Self {
            config,
            particles,
            active_count,
            tag_index,
            next_tag: 1,
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
            spatial_hash: SpatialHash::new(config.grid_cell_size),
        };
        solver.spatial_hash.rebuild(&solver.particles.x, solver.active_count);
        solver
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

    /// Remove particles where `pred` returns `false`, keeping `active_count` and
    /// tag index in sync. Use instead of `particles_mut().retain()` directly.
    pub fn retain_particles<F: Fn(&Particle) -> bool>(&mut self, pred: F) {
        self.particles.retain(pred);
        let new_len = self.particles.len();
        self.active_count = new_len;
        // Rebuild tag index from scratch — indices shift after retain.
        self.tag_index.clear();
        for i in 0..new_len {
            self.tag_index
                .entry(self.particles.user_tag[i])
                .or_default()
                .insert(i);
        }
        self.spatial_hash.rebuild(
            &self.particles.x,
            self.active_count,
        );
    }

    /// Split particles where `pred` returns `true` into two daughter particles.
    ///
    /// Each daughter gets half the mass and half the volume of the parent, with all
    /// other fields (v, F, C, damage, tag, material_id, temperature…) copied identically.
    /// Daughters are offset ±`offset_cells` cells along the principal stretch direction
    /// of the deformation gradient (eigenvector of FᵀF corresponding to max eigenvalue).
    ///
    /// Conservation: total mass and volume are preserved (halved per daughter × 2).
    ///
    /// # LP usage
    /// ```rust,ignore
    /// // Split Rankine particles whose damage exceeds 0.8:
    /// solver.split_particles_where(|p| p.friction_hardening > 0.8, 0.3);
    /// // Split snow particles compressed past 60% volume:
    /// solver.split_particles_where(|p| p.plastic_volume_ratio < 0.6, 0.3);
    /// ```
    pub fn split_particles_where<F: Fn(&Particle) -> bool>(
        &mut self,
        pred: F,
        offset_cells: f32,
    ) {
        // max 200 new particles per call — prevents exponential blow-up if condition
        // stays true across multiple frames. Caller should make condition one-shot
        // (e.g. check a damage field that gets set after split).
        const MAX_NEW: usize = 200;
        let active = self.active_count;
        let mut new_particles: Vec<Particle> = Vec::new();

        for i in 0..active {
            if new_particles.len() >= MAX_NEW {
                break;
            }
            let p = self.particles.get(i);
            if !pred(&p) {
                continue;
            }

            // Principal stretch direction from FᵀF.
            let ftf = p.deformation_gradient.transpose() * p.deformation_gradient;
            let trace = ftf.x_axis.x + ftf.y_axis.y;
            let diff = ftf.x_axis.x - ftf.y_axis.y;
            let disc = (diff * diff * 0.25 + ftf.x_axis.y * ftf.y_axis.x).max(0.0);
            let lambda_max = trace * 0.5 + disc.sqrt();
            let dir = if disc > 1e-8 {
                let v = Vec2::new(lambda_max - ftf.x_axis.x, -ftf.x_axis.y);
                let n = v.length();
                if n > 1e-6 { v / n } else { Vec2::X }
            } else {
                Vec2::X
            };

            // Halve mass and volume on parent in-place.
            self.particles.mass[i] *= 0.5;
            self.particles.initial_volume[i] *= 0.5;
            self.particles.volume[i] *= 0.5;
            self.particles.x[i] -= dir * offset_cells;
            // Mark as split so pred can exclude it on future calls.
            self.particles.friction_hardening[i] = f32::MAX;

            // Daughter: clone parent state, offset in opposite direction.
            // Give parent/child diverging velocities so fragments visibly separate.
            let speed = self.particles.v[i].length().min(20.0);
            self.particles.v[i] -= dir * speed * 0.3;
            let mut child = self.particles.get(i);
            child.x = p.x + dir * offset_cells;
            child.v = p.v + dir * speed * 0.3;
            child.friction_hardening = 0.0;
            new_particles.push(child);
        }

        // Append daughters as active particles.
        for child in new_particles {
            let idx = self.particles.len();
            self.particles.push(child);
            self.tag_index
                .entry(child.user_tag)
                .or_default()
                .insert(idx);
        }
        self.active_count = self.particles.len();
        self.spatial_hash
            .rebuild(&self.particles.x, self.active_count);
    }

    /// Apply pairwise adhesion between particles of the same `tag`.
    ///
    /// For each active particle with the given tag, neighboring same-tag particles
    /// within `radius` cells exert an attraction that nudges velocities toward each other.
    /// `strength` is a velocity correction per cell of separation per call — tune to taste.
    ///
    /// Call once per frame after `step_n()`. Does not add energy to the system when
    /// particles are already touching (force fades to zero at contact).
    ///
    /// # LP usage
    /// ```rust,ignore
    /// // Keep creature body particles together (tag = creature id).
    /// solver.apply_particle_adhesion(creature_tag, 2.0, 0.05);
    /// ```
    pub fn apply_particle_adhesion(&mut self, tag: u32, radius: f32, strength: f32) {
        // Collect positions of all same-tag active particles.
        let positions: Vec<(usize, glam::Vec2)> = (0..self.active_count)
            .filter(|&i| self.particles.user_tag[i] == tag)
            .map(|i| (i, self.particles.x[i]))
            .collect();

        for &(i, xi) in &positions {
            let mut delta_v = glam::Vec2::ZERO;
            for &idx in self.spatial_hash.query(xi, radius).collect::<Vec<_>>().iter() {
                if idx == i || self.particles.user_tag[idx] != tag {
                    continue;
                }
                let xj = self.particles.x[idx];
                let diff = xj - xi;
                let dist = diff.length();
                if dist < 1e-4 || dist > radius {
                    continue;
                }
                // Linear fade: full at contact, zero at radius.
                let fade = 1.0 - dist / radius;
                delta_v += diff / dist * strength * fade;
            }
            self.particles.v[i] += delta_v;
        }
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
        snap.active_count = self.active_count;
        snap.sleeping_count = self.particles.len().saturating_sub(self.active_count);
        snap
    }

    // ── Tag-based group API ───────────────────────────────────────────────────

    /// Aggregate physics state for all particles with `tag`. O(group_size).
    pub fn group_state(&self, tag: u32) -> MaterialState {
        let mut s = MaterialState::default();
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                s.accumulate(
                    self.particles.x[i],
                    self.particles.v[i].length(),
                    self.particles.plastic_volume_ratio[i],
                    self.particles.deformation_gradient[i].determinant(),
                    self.particles.density[i],
                );
            }
        }
        s.finalize();
        s
    }

    /// Center of mass for all particles with `tag`. O(group_size).
    pub fn group_centroid(&self, tag: u32) -> glam::Vec2 {
        let indices = match self.tag_index.get(&tag) {
            Some(s) if !s.is_empty() => s,
            _ => return glam::Vec2::ZERO,
        };
        let sum: glam::Vec2 = indices.iter().map(|&i| self.particles.x[i]).sum();
        sum / indices.len() as f32
    }

    /// Number of particles with `tag`. O(1).
    pub fn group_count(&self, tag: u32) -> usize {
        self.tag_index.get(&tag).map_or(0, |v| v.len())
    }

    /// Set `activation` uniformly on all particles with `tag`. O(group_size).
    pub fn set_group_activation(&mut self, tag: u32, value: f32) {
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                self.particles.activation[i] = value.clamp(0.0, 1.0);
            }
        }
    }

    /// Set `activation` per particle using a spatial function. O(group_size).
    pub fn set_group_activation_fn(&mut self, tag: u32, f: impl Fn(glam::Vec2) -> f32) {
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                self.particles.activation[i] = f(self.particles.x[i]).clamp(0.0, 1.0);
            }
        }
    }

    /// Set `temperature` uniformly on all particles with `tag`. O(group_size).
    pub fn set_group_temperature(&mut self, tag: u32, value: f32) {
        if let Some(indices) = self.tag_index.get(&tag) {
            for &i in indices {
                self.particles.temperature[i] = value;
            }
        }
    }

    /// Apply a velocity impulse to all particles with `tag`, with optional distance falloff. O(group_size).
    pub fn apply_group_impulse(&mut self, tag: u32, impulse: glam::Vec2, falloff_center: Option<glam::Vec2>) {
        let indices: Vec<usize> = match self.tag_index.get(&tag) {
            Some(s) => s.iter().copied().collect(),
            None => return,
        };
        for i in indices {
            let scale = match falloff_center {
                None => 1.0,
                Some(c) => (1.0 - (self.particles.x[i] - c).length() * 0.1).max(0.0),
            };
            self.particles.v[i] += impulse * scale;
        }
    }

    // ── Query & Transition API ────────────────────────────────────────────────

    /// Aggregate state for all particles of a given material.
    pub fn material_state(&self, material_id: u32) -> MaterialState {
        material_state_of(&self.particles, material_id)
    }

    /// Aggregate state for all particles within `radius` grid-cells of `center`.
    pub fn region_state(&self, center: Vec2, radius: f32) -> MaterialState {
        let r2 = radius * radius;
        let mut s = query::MaterialState::default();
        for i in self.spatial_hash.query(center, radius) {
            if (self.particles.x[i] - center).length_squared() <= r2 {
                s.accumulate(
                    self.particles.x[i],
                    self.particles.v[i].length(),
                    self.particles.plastic_volume_ratio[i],
                    self.particles.deformation_gradient[i].determinant(),
                    self.particles.density[i],
                );
            }
        }
        s.finalize();
        s
    }

    /// Iterate indices of active particles within `radius` grid-cells of `center`.
    ///
    /// Returns indices only — read particle data via `solver.particles().x[i]` etc.
    /// O(candidates) via spatial hash, not O(N).
    pub fn particles_near(
        &self,
        center: Vec2,
        radius: f32,
    ) -> impl Iterator<Item = usize> + '_ {
        let r2 = radius * radius;
        self.spatial_hash
            .query(center, radius)
            .filter(move |&i| (self.particles.x[i] - center).length_squared() <= r2)
    }

    /// Count active particles of a given material within `radius` of `center`.
    /// O(candidates) via spatial hash, not O(N).
    pub fn count_near(&self, center: Vec2, radius: f32, material_id: u32) -> usize {
        let r2 = radius * radius;
        self.spatial_hash
            .query(center, radius)
            .filter(|&i| {
                self.particles.material_id[i] == material_id
                    && (self.particles.x[i] - center).length_squared() <= r2
            })
            .count()
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
        let mut to_wake = Vec::new();
        for i in 0..self.particles.len() {
            let d = self.particles.x[i] - center;
            let dist2 = d.length_squared();
            if dist2 <= r2 && dist2 > 1e-8 {
                if self.particles.sleeping[i] {
                    to_wake.push(i);
                }
                let falloff = 1.0 - (dist2 / r2).sqrt();
                self.particles.v[i] += force * falloff;
                let spd = self.particles.v[i].length();
                if spd > vel_limit {
                    self.particles.v[i] *= vel_limit / spd;
                }
            }
        }
        for i in to_wake {
            self.wake_particle(i);
        }
    }

    /// Apply an outward radial velocity delta to particles within `radius`, with linear falloff.
    /// Result is clamped to the solver's CFL velocity limit.
    pub fn apply_radial_impulse(&mut self, center: Vec2, radius: f32, strength: f32) {
        let vel_limit = self.config.grid_cell_size / self.config.min_dt;
        let r2 = radius * radius;
        let mut to_wake = Vec::new();
        for i in 0..self.particles.len() {
            let d = self.particles.x[i] - center;
            let dist2 = d.length_squared();
            if dist2 <= r2 && dist2 > 1e-8 {
                if self.particles.sleeping[i] {
                    to_wake.push(i);
                }
                let dist = dist2.sqrt();
                let falloff = 1.0 - dist / radius;
                self.particles.v[i] += (d / dist) * strength * falloff;
                let spd = self.particles.v[i].length();
                if spd > vel_limit {
                    self.particles.v[i] *= vel_limit / spd;
                }
            }
        }
        for i in to_wake {
            self.wake_particle(i);
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
                choose_substep_dt(&self.config, &self.particles, self.active_count, &self.materials, remaining);
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
            for i in 0..self.active_count {
                if project_particle_state_to_admissible(&mut self.particles, i, &self.config) {
                    self.last_j_projection_count += 1;
                }
            }
        }

        // Density recompute: fluid EOS materials need current ρ each substep (pressure = f(ρ)).
        // Auto-enabled when any registered material declares needs_density_recompute=true.
        // Manual override via config.recompute_density_each_step for edge cases.
        if self.config.recompute_density_each_step || self.materials.any_needs_density_recompute() {
            estimate_particle_volumes(&mut self.particles, &mut self.grid, self.active_count, false);
        }

        self.grid.clear();
        scatter_particles_to_grid(&self.particles, &mut self.grid, &self.materials, sub_dt, self.active_count);

        // Wake any sleeping particle whose kernel overlaps an active grid cell.
        // This propagates activity from moving regions into neighbouring sleeping ones
        // without a separate O(N) scan — we only visit the sleeping partition.
        if self.active_count < self.particles.len() {
            let total = self.particles.len();
            let mut to_wake = Vec::new();
            for i in self.active_count..total {
                let x = self.particles.x[i];
                let base = crate::grid::kernel::quadratic_weights(x).base_cell;
                'outer: for gx in 0i32..3 {
                    for gy in 0i32..3 {
                        let cell = base + glam::IVec2::new(gx - 1, gy - 1);
                        if self.grid.cell_is_active(cell) {
                            to_wake.push(i);
                            break 'outer;
                        }
                    }
                }
            }
            for i in to_wake {
                self.wake_particle(i);
            }
        }

        // Normalize accumulated momentum to velocity, then apply gravity and wall constraints.
        self.grid.update_velocities(sub_dt, self.config.gravity);
        let grid_res = self.grid.resolution();
        for boundary in &self.boundaries {
            apply_boundary_conditions_to_grid(&mut self.grid, grid_res, boundary.as_ref());
        }

        // Clamp grid velocity before G2P — bounds both v_p and C_p at the source.
        // Post-G2P clamping misses C_p: large C_p → F = (I + dt·C)·F blows up → J→0.
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
            self.active_count,
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
            for i in 0..self.active_count {
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
            for i in 0..self.active_count {
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

        // Rebuild spatial hash over the active partition after all position updates.
        // O(active_count). Subsequent particles_near / count_near / region_state calls
        // within this frame use the hash rather than a full scan.
        self.spatial_hash.rebuild(&self.particles.x, self.active_count);

        // Sleep evaluation: put slow, passive active particles to sleep.
        // Creatures (activation > 0) are never slept — they drive locomotion.
        let threshold = self.config.sleep_threshold;
        if threshold > 0.0 {
            let threshold_sq = threshold * threshold;
            let mut to_sleep: Vec<usize> = (0..self.active_count)
                .filter(|&i| {
                    self.particles.activation[i] == 0.0
                        && self.particles.v[i].length_squared() < threshold_sq
                })
                .collect();
            // Descending order: sleep_particle swaps i↔last_active (high end of active zone).
            // Processing high-to-low ensures each displacement lands in already-processed
            // positions, so no sleeping candidate is accidentally skipped.
            to_sleep.sort_unstable_by(|a, b| b.cmp(a));
            for i in to_sleep {
                self.sleep_particle(i);
            }
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
        estimate_particle_volumes(&mut self.particles, &mut self.grid, self.active_count, true);
    }

    /// Remove all particles where `predicate` returns true. Returns count removed.
    ///
    /// Uses stable retain (preserves order). O(N).
    ///
    /// LP pattern: tag particles with a sentinel before the step, then call
    /// `solver.remove_particles(|p| p.user_tag == DEAD)`. The tag-based group API
    /// remains valid after removal — tag_index is rebuilt internally.
    pub fn remove_particles<F: Fn(&Particle) -> bool>(&mut self, predicate: F) -> usize {
        let before = self.particles.len();
        self.particles.retain(|p| !predicate(p));
        let removed = before - self.particles.len();
        if removed > 0 {
            // retain() compacted the array — all physical indices in tag_index are stale.
            // Rebuild from scratch and re-establish the sleep partition.
            self.tag_index.clear();
            // Re-partition: move all sleeping particles to the back.
            let n = self.particles.len();
            let mut write = 0usize;
            for read in 0..n {
                if !self.particles.sleeping[read] {
                    if write != read {
                        self.particles.swap(write, read);
                    }
                    write += 1;
                }
            }
            self.active_count = write;
            // Rebuild tag_index over the freshly-partitioned array.
            for i in 0..n {
                self.tag_index
                    .entry(self.particles.user_tag[i])
                    .or_default()
                    .insert(i);
            }
        }
        removed
    }

    /// Iterate physical indices of all particles with `tag`. O(group_size) via tag_index.
    ///
    /// Returns indices only — read particle data via `solver.particles().x[i]` etc.
    /// This avoids cloning 112B per particle on every call.
    pub fn particles_with_tag(&self, tag: u32) -> impl Iterator<Item = usize> + '_ {
        self.tag_index
            .get(&tag)
            .into_iter()
            .flat_map(|s| s.iter())
            .copied()
    }

    // ── Sleep / wake ─────────────────────────────────────────────────────────

    /// Put particle at physical index `i` to sleep.
    ///
    /// Swaps it with the last active particle, decrementing `active_count`.
    /// Updates `tag_index` for both affected particles.
    /// No-op if already sleeping.
    fn sleep_particle(&mut self, i: usize) {
        if self.particles.sleeping[i] || self.active_count == 0 {
            return;
        }
        self.particles.sleeping[i] = true;
        let last_active = self.active_count - 1;
        if i != last_active {
            let tag_i = self.particles.user_tag[i];
            let tag_j = self.particles.user_tag[last_active];
            // Same-tag swap: both indices stay in the same set — no update needed.
            // Different-tag swap: each particle moves to the other's former position.
            if tag_i != tag_j {
                Self::tag_index_replace(&mut self.tag_index, tag_i, i, last_active);
                Self::tag_index_replace(&mut self.tag_index, tag_j, last_active, i);
            }
            self.particles.swap(i, last_active);
        }
        self.active_count -= 1;
    }

    /// Wake particle at physical index `i`.
    ///
    /// Swaps it with the first sleeping particle, incrementing `active_count`.
    /// Updates `tag_index` for both affected particles.
    /// No-op if already awake.
    fn wake_particle(&mut self, i: usize) {
        if !self.particles.sleeping[i] {
            return;
        }
        self.particles.sleeping[i] = false;
        let first_sleeping = self.active_count;
        if i != first_sleeping {
            let tag_i = self.particles.user_tag[i];
            let tag_j = self.particles.user_tag[first_sleeping];
            if tag_i != tag_j {
                Self::tag_index_replace(&mut self.tag_index, tag_i, i, first_sleeping);
                Self::tag_index_replace(&mut self.tag_index, tag_j, first_sleeping, i);
            }
            self.particles.swap(i, first_sleeping);
        }
        self.active_count += 1;
    }

    /// Wake all particles belonging to `tag`. O(group_size · log group_size).
    pub fn wake_tag(&mut self, tag: u32) {
        // Snapshot sleeping indices, then wake ascending.
        // wake_particle(i) swaps i↔first_sleeping (low end of sleeping zone).
        // Ascending order: each displacement lands at an already-processed lower position,
        // so no sleeping tag particle is silently displaced to an unvisited index.
        let mut to_wake: Vec<usize> = self.tag_index
            .get(&tag)
            .map(|s| s.iter().filter(|&&i| self.particles.sleeping[i]).copied().collect())
            .unwrap_or_default();
        to_wake.sort_unstable();
        for i in to_wake {
            self.wake_particle(i);
        }
    }

    /// Sleep all particles belonging to `tag`. O(group_size · log group_size).
    pub fn sleep_tag(&mut self, tag: u32) {
        // Snapshot active indices, then sleep descending.
        // sleep_particle(i) swaps i↔last_active (high end of active zone).
        // Descending order: each displacement lands at an already-processed higher position.
        let mut to_sleep: Vec<usize> = self.tag_index
            .get(&tag)
            .map(|s| s.iter().filter(|&&i| !self.particles.sleeping[i]).copied().collect())
            .unwrap_or_default();
        to_sleep.sort_unstable_by(|a, b| b.cmp(a));
        for i in to_sleep {
            self.sleep_particle(i);
        }
    }

    /// Number of currently active (non-sleeping) particles.
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    fn tag_index_replace(
        tag_index: &mut HashMap<u32, HashSet<usize>>,
        tag: u32,
        old_idx: usize,
        new_idx: usize,
    ) {
        if let Some(s) = tag_index.get_mut(&tag) {
            s.remove(&old_idx);
            s.insert(new_idx);
        }
    }

    /// Collect all particles into a `Vec<Particle>` (for diagnostics or GPU upload).
    pub fn collect_particles(&self) -> Vec<Particle> {
        self.particles.to_vec()
    }

    /// Spawn particles, tag them, and return the stable tag.
    ///
    /// The returned `u32` tag is the only stable identity for this group — physical
    /// indices change whenever particles sleep or wake. Pass it to `group_state`,
    /// `set_group_activation`, `group_centroid`, etc.
    ///
    /// All particles in the region are stamped with `user_tag = tag` (overrides
    /// any `user_tag` set in the SpawnConfig).
    ///
    /// ```rust,no_run
    /// # use emerge::solver::MpmSolver;
    /// # use emerge::{SolverConfig, SpawnConfig};
    /// # let config = SolverConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
    /// # let mut solver = MpmSolver::empty(config);
    /// let creature = solver.spawn_group(SpawnConfig::for_solver(&config));
    /// solver.set_group_activation(creature, 1.0);
    /// let centroid = solver.group_centroid(creature);
    /// ```
    #[must_use = "store the tag — it is the only stable identity for this group"]
    pub fn spawn_group(&mut self, spawn: SpawnConfig) -> u32 {
        let tag = self.next_tag;
        self.next_tag += 1;

        let old_active = self.active_count;
        let old_len = self.particles.len();
        // sleeping zone is [old_active..old_len] — new particles must land before it.

        spawn.validate_for_solver(&self.config);
        debug_assert!(
            self.materials.is_registered(spawn.material_id),
            "spawn_group: material_id {} is not registered",
            spawn.material_id,
        );
        let mut rng = LcgRng::new(spawn.rng_seed);
        let new_particles = initialize_particles(&self.config, spawn, &mut rng);
        for p in new_particles {
            self.particles.push(p);
        }
        let new_len = self.particles.len();
        let new_count = new_len - old_len;
        let sleeping_count = old_len - old_active;

        // Stamp tag and init material plastic state (new particles still at [old_len..new_len]).
        let mat_id = spawn.material_id;
        for i in old_len..new_len {
            self.particles.user_tag[i] = tag;
            if self.materials.is_registered(mat_id) {
                let mut p = self.particles.get(i);
                self.materials.get(mat_id).init_particle(&mut p);
                self.particles.set(i, p);
            }
        }

        // If sleeping particles sit between the active zone and the new particles, rotate new
        // particles before the sleeping zone so the partition invariant is maintained:
        //   before: [0..old_active] active | [old_active..old_len] sleeping | [old_len..new_len] new
        //   after:  [0..old_active] active | [old_active..old_active+new_count] new | [...] sleeping
        if sleeping_count > 0 {
            self.particles.rotate_range(old_active, old_len, new_len);
            // Sleeping particles moved from [old_active+k] → [old_active+new_count+k].
            // Update tag_index for each displaced sleeping particle.
            for k in 0..sleeping_count {
                let old_pos = old_active + k;
                let new_pos = old_active + new_count + k;
                let t = self.particles.user_tag[new_pos];
                Self::tag_index_replace(&mut self.tag_index, t, old_pos, new_pos);
            }
        }

        // New particles are at [old_active..old_active+new_count].
        let group_start = old_active;
        let group_end = old_active + new_count;
        self.tag_index.insert(tag, (group_start..group_end).collect::<HashSet<usize>>());
        self.active_count = group_end;

        // Scatter only particles in the spawn region + 3-cell margin.
        // O(active_count) scan but O(local × stencil) grid work — fast for sparse spawns.
        density::estimate_particle_volumes_local(
            &mut self.particles,
            &mut self.grid,
            self.active_count,
            group_start,
            true,
        );
        self.spatial_hash.rebuild(&self.particles.x, self.active_count);
        tag
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
            particles,
            i,
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
    let soa = Particles::from(particles.to_vec());
    let mut min_mat_dt = max_dt;
    for i in 0..soa.len() {
        let mdt = materials.get(soa.material_id[i]).timestep_bound(
            &soa,
            i,
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
fn project_particle_state_to_admissible(
    particles: &mut Particles,
    i: usize,
    config: &SolverConfig,
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
