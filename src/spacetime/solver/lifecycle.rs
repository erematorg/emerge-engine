//! Construction, builders, and registration: `Simulation::empty`/`new`, the
//! `.with_*` builder chain, material/boundary/force-field CRUD, and basic
//! particle-buffer accessors.
//!
//! Split out of `solver/mod.rs` -- everything here sets up or reconfigures
//! the simulation, as opposed to advancing it (`solver::step`) or reading
//! aggregate state from it (`solver::queries`).

use std::collections::{BTreeMap, HashMap, HashSet};

use glam::Vec2;

use super::spatial_hash::SpatialHash;
use super::{LcgRng, MaterialHandle, SimConfig, Simulation, SpawnRegion, initialize_particles};
use crate::boundary::{BoundaryCondition, SlipBoundary};
use crate::fields::Field;
use crate::grid::Grid;
use crate::materials::registry::MaterialRegistry;
use crate::materials::{FallbackMaterial, MaterialModel};
use crate::particle::{Particle, Particles};
use crate::solver::density::estimate_particle_volumes;
use crate::thermodynamics::{ThermalConfig, ThermalDiffusion};

impl Simulation {
    /// Create an empty solver with no particles. Use `spawn_region` to add particles.
    pub fn empty(config: SimConfig) -> Self {
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
            contact_grip: None,
            force_fields: Vec::new(),
            thermal: None,
            scalar_fields: Vec::new(),
            frame_index: 0,
            last_step_dt: config.dt,
            last_substeps: 0,
            last_vel_clamp_count: 0,
            last_j_projection_count: 0,
            last_sim_time_dropped: 0.0,
            last_timing: crate::diagnostics::StepTiming::default(),
            phase_rules: Vec::new(),
            spatial_hash: SpatialHash::new(config.grid_cell_size),
            scratch_indices: Vec::new(),
        }
    }

    pub fn new(config: SimConfig, spawn: SpawnRegion) -> Self {
        config.validate();
        spawn.validate_for_sim(&config);

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
            contact_grip: None,
            force_fields: Vec::new(),
            thermal: None,
            scalar_fields: Vec::new(),
            frame_index: 0,
            last_step_dt: config.dt,
            last_substeps: 0,
            last_vel_clamp_count: 0,
            last_j_projection_count: 0,
            last_sim_time_dropped: 0.0,
            last_timing: crate::diagnostics::StepTiming::default(),
            phase_rules: Vec::new(),
            spatial_hash: SpatialHash::new(config.grid_cell_size),
            scratch_indices: Vec::new(),
        };
        solver
            .spatial_hash
            .rebuild(&solver.particles.x, solver.active_count);
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

    /// Set directional (setae-style) friction for the multi-field contact "grip"
    /// field — see `DirectionalContactGrip`'s doc. Takes an `Arc` so the same
    /// instance can be shared with external code (player/AI input) for live
    /// steering, matching `RatchetFrictionBoundary`'s own established pattern.
    /// Only affects particles with `contact_group != 0`; a scene that never sets
    /// that field is completely unaffected whether or not this is set.
    pub fn with_contact_grip(
        mut self,
        grip: std::sync::Arc<crate::grid::DirectionalContactGrip>,
    ) -> Self {
        self.contact_grip = Some(grip);
        self
    }

    /// Append an anonymous force field (auto-named "force_field_N").
    pub fn with_force_field(mut self, field: Box<dyn Field>) -> Self {
        self.add_force_field(field);
        self
    }

    /// Append a named force field — name can be used later to remove or replace it.
    pub fn with_named_force_field(
        mut self,
        name: impl Into<String>,
        field: Box<dyn Field>,
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

    /// Mutable access to the attached thermal model's config, if any (`None` when no
    /// `with_thermal`/`set_thermal` was ever called). The real, minimal hook for a
    /// scene/LP-driven day-night or seasonal cycle: mutate `.ambient` each frame from a
    /// time-varying function (e.g. a sinusoid) BEFORE calling `step()` — `ThermalDiffusion
    /// ::apply` already runs automatically every substep and reads `config.ambient` fresh
    /// each time via the existing Newton-cooling term (`dT/dt = -k_c*(T-ambient)`), so no
    /// new physics is needed, just this accessor to reach the config from outside.
    pub fn thermal_config_mut(&mut self) -> Option<&mut ThermalConfig> {
        self.thermal.as_mut().map(|t| &mut t.config)
    }

    /// Register a material and return its typed `MaterialHandle`.
    ///
    /// Preferred over `with_material(id, mat)` — handle is type-safe, auto-allocates ID.
    /// ```rust,no_run
    /// # extern crate emerge_engine as emerge;
    /// # use emerge::solver::Simulation;
    /// # use emerge::{SimConfig, SpawnRegion, NewtonianFluidMaterial};
    /// # let config = SimConfig::standard(64, 0.05, glam::Vec2::NEG_Y);
    /// # let mut solver = Simulation::new(config, SpawnRegion::default());
    /// let water = solver.register_material(Box::new(NewtonianFluidMaterial::low_viscosity(1000.0, 1e4)));
    /// // use water.id() in SpawnRegion or phase_transition
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

    pub fn config(&self) -> &SimConfig {
        &self.config
    }

    pub fn particles(&self) -> &Particles {
        &self.particles
    }

    /// Direct read-only access to the background grid -- lets a CPU-simulated scene's
    /// renderer sample the solver's own mass field (e.g. for grid-volume rendering,
    /// mirroring what GPU scenes get via `GpuSimulation::grid_buffer()`) without
    /// duplicating the solver's own P2G-computed density.
    pub fn grid(&self) -> &Grid {
        &self.grid
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
        self.spatial_hash
            .rebuild(&self.particles.x, self.active_count);
    }

    /// Splits active particles matching `should_split` into two half-mass/half-volume
    /// children, jittered apart by `jitter` (grid units) so they don't start exactly
    /// overlapping — an un-jittered split would put both children at the literal same
    /// position, the same lattice-symmetry failure mode found and fixed for spawn lattices
    /// earlier this session ("combed" sand). Every other field (velocity, deformation
    /// gradient, material_id, temperature, etc.) is inherited unchanged from the parent;
    /// only mass/volume/position differ, and children always wake up (a freshly-fractured
    /// piece has no reason to start asleep). Sleeping particles are left untouched, never
    /// split. CPU-only (`Simulation`, not `GpuSimulation`) — splitting requires growing the
    /// particle buffer, which the GPU path's fixed-size buffers don't support; not attempted
    /// here, real future work if needed.
    ///
    /// LP use case: pass a predicate checking `p.material_id == BONE && p.friction_hardening`
    /// against a damage threshold (Rankine's `friction_hardening` field IS its damage
    /// variable) to turn accumulated fracture damage into actual visible breakage instead of
    /// an invisible internal number.
    pub fn split_particles<F: Fn(&Particle) -> bool>(&mut self, should_split: F, jitter: f32) {
        let mut rng = LcgRng::new(0xC0FF_EE11);
        let n = self.particles.len();
        let mut new_particles = Particles::from(Vec::with_capacity(n));
        let mut new_active_count = 0usize;
        for i in 0..self.active_count {
            let p = self.particles.get(i);
            if should_split(&p) {
                for _ in 0..2 {
                    let mut child = p;
                    child.mass *= 0.5;
                    child.initial_volume *= 0.5;
                    child.volume *= 0.5;
                    let jx = (rng.next_f32() - 0.5) * 2.0 * jitter;
                    let jy = (rng.next_f32() - 0.5) * 2.0 * jitter;
                    child.x += Vec2::new(jx, jy);
                    child.sleeping = 0;
                    new_particles.push(child);
                    new_active_count += 1;
                }
            } else {
                new_particles.push(p);
                new_active_count += 1;
            }
        }
        for i in self.active_count..n {
            new_particles.push(self.particles.get(i));
        }
        self.particles = new_particles;
        self.active_count = new_active_count;
        self.tag_index.clear();
        for i in 0..self.particles.len() {
            self.tag_index
                .entry(self.particles.user_tag[i])
                .or_default()
                .insert(i);
        }
        self.spatial_hash
            .rebuild(&self.particles.x, self.active_count);
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
    pub fn add_force_field(&mut self, field: Box<dyn Field>) {
        let name = format!("force_field_{}", self.force_fields.len());
        self.force_fields.push((name, field));
    }

    /// Append a named force field.
    pub fn add_named_force_field(&mut self, name: impl Into<String>, field: Box<dyn Field>) {
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
}
