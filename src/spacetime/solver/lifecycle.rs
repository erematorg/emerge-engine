//! Construction, builders, and registration: `Simulation::empty`/`new`, the
//! `.with_*` builder chain, material/boundary/force-field CRUD, and basic
//! particle-buffer accessors.
//!
//! Split out of `solver/mod.rs` -- everything here sets up or reconfigures
//! the simulation, as opposed to advancing it (`solver::step`) or reading
//! aggregate state from it (`solver::queries`).

use std::cell::{Cell, RefCell};
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
            pending_diffusion_dt: 0.0,
            substep_index_in_frame: 0,
            granular_fluidity: None,
            granular_fluidity_g: Vec::new(),
            cosserat: None,
            cosserat_omega: Vec::new(),
            cosserat_curvature: Vec::new(),
            frame_index: 0,
            fluid_sticky_fine_dt: None,
            last_max_particle_speed: 0.0,
            last_step_dt: config.dt,
            last_substeps: 0,
            last_vel_clamp_count: 0,
            last_j_projection_count: 0,
            last_sim_time_dropped: 0.0,
            last_timing: crate::diagnostics::StepTiming::default(),
            cached_spatial_sort_order: Vec::new(),
            phase_rules: Vec::new(),
            spatial_hash: RefCell::new(SpatialHash::new(config.grid_cell_size)),
            spatial_hash_dirty: Cell::new(false),
            scratch_indices: Vec::new(),
            rods: Vec::new(),
            grain_populations: Vec::new(),
            rod_networks: Vec::new(),
            stage_ops: Vec::new(),
            coupled_bodies: Vec::new(),
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
            // No MaterialRegistry exists yet at this point in construction (built
            // just below) -- harmless: write_initial=true never reaches the
            // material-aware clamp, see density.rs's own doc comment.
            estimate_particle_volumes(&mut particles, &mut grid, None, n, true);
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
        let solver = Self {
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
            pending_diffusion_dt: 0.0,
            substep_index_in_frame: 0,
            granular_fluidity: None,
            granular_fluidity_g: Vec::new(),
            cosserat: None,
            cosserat_omega: Vec::new(),
            cosserat_curvature: Vec::new(),
            frame_index: 0,
            fluid_sticky_fine_dt: None,
            last_max_particle_speed: 0.0,
            last_step_dt: config.dt,
            last_substeps: 0,
            last_vel_clamp_count: 0,
            last_j_projection_count: 0,
            last_sim_time_dropped: 0.0,
            last_timing: crate::diagnostics::StepTiming::default(),
            cached_spatial_sort_order: Vec::new(),
            phase_rules: Vec::new(),
            spatial_hash: RefCell::new(SpatialHash::new(config.grid_cell_size)),
            spatial_hash_dirty: Cell::new(false),
            scratch_indices: Vec::new(),
            rods: Vec::new(),
            grain_populations: Vec::new(),
            rod_networks: Vec::new(),
            stage_ops: Vec::new(),
            coupled_bodies: Vec::new(),
        };
        solver
            .spatial_hash
            .borrow_mut()
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

    /// Attach a Nonlocal Granular Fluidity field (see `energy::
    /// thermodynamics::granular_fluidity` module doc). `None` (never
    /// calling this) is the default, zero-cost, byte-identical to every
    /// existing scene -- same convention `with_thermal` already has.
    ///
    /// Its explicit von Neumann stability bound is folded directly into each
    /// adaptive substep by `choose_substep_dt`. `SimConfig::min_dt` is never
    /// allowed to raise that upper bound, so attaching this field does not
    /// retune unrelated solver settings or trade stability for throughput.
    pub fn with_granular_fluidity(
        mut self,
        field: crate::thermodynamics::GranularFluidityField,
    ) -> Self {
        self.granular_fluidity = Some(field);
        self
    }

    /// Attach a Cosserat micro-rotation field (see `energy::thermodynamics::
    /// cosserat_field` module doc). `None` (never calling this) is the
    /// default, zero-cost, byte-identical to every existing scene -- same
    /// convention `with_granular_fluidity` already has.
    pub fn with_cosserat_field(mut self, field: crate::thermodynamics::CosseratField) -> Self {
        self.cosserat = Some(field);
        self
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

    /// Read-only access to the attached `GranularFluidityField`, if any --
    /// same "`None` unless opted in" convention as `thermal_config_mut`.
    pub const fn granular_fluidity(&self) -> Option<&crate::thermodynamics::GranularFluidityField> {
        self.granular_fluidity.as_ref()
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

    pub const fn config(&self) -> &SimConfig {
        &self.config
    }

    pub const fn particles(&self) -> &Particles {
        &self.particles
    }

    /// Diagnostic-only read access to the gathered Cosserat micro-curvature
    /// buffer -- lets tests measure whether the coupling is actually
    /// producing nonzero curvature, instead of only inferring it indirectly.
    /// Empty when no `CosseratField` is configured for this scene.
    pub fn cosserat_curvature(&self) -> &[glam::Vec2] {
        &self.cosserat_curvature
    }

    /// Direct read-only access to the background grid -- lets a CPU-simulated scene's
    /// renderer sample the solver's own mass field (e.g. for grid-volume rendering,
    /// mirroring what GPU scenes get via `GpuSimulation::grid_buffer()`) without
    /// duplicating the solver's own P2G-computed density.
    pub const fn grid(&self) -> &Grid {
        &self.grid
    }

    /// Direct mutable access to all particles.
    ///
    /// **State warning:** velocity changes made here are used exactly. The
    /// next substep recomputes its CFL bound from the altered state; callers
    /// must keep values finite and use physically meaningful forcing.
    /// Safe uses: writing non-velocity fields (temperature, activation, user_tag, material_id).
    pub const fn particles_mut(&mut self) -> &mut Particles {
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
            .borrow_mut()
            .rebuild(&self.particles.x, self.active_count);
        self.spatial_hash_dirty.set(false);
    }

    /// Splits active particles matching `should_split` into two half-mass/half-volume
    /// children, jittered apart by `jitter` (grid units) so they don't start exactly
    /// overlapping — an un-jittered split would put both children at the literal same
    /// position, the same lattice-symmetry failure mode ("combed" sand) that spawn
    /// lattices need jitter to avoid. Every other field (velocity, deformation gradient,
    /// material_id, temperature, etc.) is inherited unchanged from the parent; only
    /// mass/volume/position differ, and children always wake up (a freshly-fractured
    /// piece has no reason to start asleep). Sleeping particles are left untouched, never
    /// split. CPU-only (`Simulation`, not `GpuSimulation`) — splitting requires growing the
    /// particle buffer, which the GPU path's fixed-size buffers don't support; not
    /// attempted here, future work if needed.
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
            .borrow_mut()
            .rebuild(&self.particles.x, self.active_count);
        self.spatial_hash_dirty.set(false);
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

    pub const fn gravity(&self) -> Vec2 {
        self.config.gravity
    }

    pub const fn set_gravity(&mut self, gravity: Vec2) {
        self.config.gravity = gravity;
    }

    /// Live-tunable Cundall (1982) non-viscous damping coefficient, same
    /// precedent as `set_gravity` -- lets a caller phase-gate it (e.g. off
    /// while material is actively falling/impacting, on once it should
    /// relax toward equilibrium) instead of one constant value for a
    /// scene's entire run.
    pub const fn set_cundall_damping(&mut self, damping: f32) {
        self.config.cundall_damping = damping;
    }

    /// Live-tunable APIC/FLIP blend, same phase-gating precedent as
    /// `set_cundall_damping` -- lets a caller run the violent/dynamic part
    /// of a collapse at the scene's own default blend (real toppling
    /// energy preserved), then switch to the proven quasi-static holding
    /// value (0.05) once the material has actually settled, instead of one
    /// constant blend fighting both phases at once.
    pub const fn set_apic_blend(&mut self, blend: f32) {
        self.config.apic_blend = blend;
    }

    /// Append a rod, returning its index into `rods()`/`rods_mut()`. A rod's
    /// `points.x` must already be in this simulation's grid-cell coordinate
    /// space (same convention as `Particle::x`) — build it with
    /// `rod::build_straight_rod(start, end, n, linear_density, config.dx_meters)`
    /// so `start`/`end` (grid-cell units) and the resulting rest lengths
    /// (real meters) both land in the right space for `scatter_rod_to_grid`/
    /// `gather_grid_to_rod` to interoperate with ordinary particles.
    ///
    /// Real Euler/Greenhill self-weight buckling check happens HERE, not as
    /// something each example has to remember to call (2026-07-27: a live
    /// GUI session found blade B swinging wide and slow under a push and it
    /// read as "broken" until traced back to real, disclosed buckling
    /// physics -- the check existed but only one example was actually
    /// calling it). `self.config.gravity` is already real, whatever this
    /// simulation was configured with (grid units, `g_si = g_grid *
    /// dx_meters` per `gravity_to_grid`'s own convention) -- not hardcoded to
    /// Earth's 9.81, so this holds for any configured gravity.
    pub fn add_rod(&mut self, rod: crate::rod::Rod) -> usize {
        let gravity_m_s2 = self.config.gravity.length() * self.config.dx_meters;
        if let Some(warning) = rod.buckling_warning(gravity_m_s2) {
            eprintln!("[emerge::rod] {warning}");
        }
        self.rods.push(rod);
        self.rods.len() - 1
    }

    /// Builder variant of `add_rod`.
    pub fn with_rod(mut self, rod: crate::rod::Rod) -> Self {
        self.add_rod(rod);
        self
    }

    pub fn rods(&self) -> &[crate::rod::Rod] {
        &self.rods
    }

    pub fn rods_mut(&mut self) -> &mut [crate::rod::Rod] {
        &mut self.rods
    }

    /// Adds a discrete-element grain population (`spacetime::grains`),
    /// returning its index. Mirrors `add_rod` exactly. See `grain_populations`'s
    /// own doc on `Simulation` for real scope (no automatic oracle yet --
    /// this is an explicit, caller-decided population, same as a rod).
    pub fn add_grain_population(
        &mut self,
        population: crate::grains::population::GrainPopulation,
    ) -> usize {
        self.grain_populations.push(population);
        self.grain_populations.len() - 1
    }

    /// Builder variant of `add_grain_population`.
    pub fn with_grain_population(
        mut self,
        population: crate::grains::population::GrainPopulation,
    ) -> Self {
        self.add_grain_population(population);
        self
    }

    pub fn grain_populations(&self) -> &[crate::grains::population::GrainPopulation] {
        &self.grain_populations
    }

    pub fn grain_populations_mut(&mut self) -> &mut [crate::grains::population::GrainPopulation] {
        &mut self.grain_populations
    }

    /// Adds a branching rod/root network (`spacetime::rod::network`),
    /// returning its index. Mirrors `add_rod`/`add_grain_population` exactly.
    pub fn add_rod_network(&mut self, network: crate::rod::RodNetwork) -> usize {
        self.rod_networks.push(network);
        self.rod_networks.len() - 1
    }

    /// Builder variant of `add_rod_network`.
    pub fn with_rod_network(mut self, network: crate::rod::RodNetwork) -> Self {
        self.add_rod_network(network);
        self
    }

    pub fn rod_networks(&self) -> &[crate::rod::RodNetwork] {
        &self.rod_networks
    }

    pub fn rod_networks_mut(&mut self) -> &mut [crate::rod::RodNetwork] {
        &mut self.rod_networks
    }
}

#[cfg(test)]
mod add_rod_buckling_check_tests {
    use super::*;
    use crate::rod::{Rod, RodMaterial, build_straight_rod};

    /// Real regression guard for the 2026-07-27 fix: the buckling check used
    /// to be an opt-in print each example had to remember to call (only one
    /// of three rod-using examples actually did) -- now `add_rod` itself
    /// checks every rod against its own real Greenhill critical height using
    /// THIS simulation's own configured gravity, so no example can silently
    /// add an unstable rod without at least a real, printed warning. This
    /// test only confirms `add_rod` keeps working correctly (returns the
    /// right index, `rods()` reflects it) whether or not the rod happens to
    /// be over its own critical height -- the warning CONTENT itself is
    /// already covered by `rod::root_cause_fixes_tests::
    /// buckling_warning_matches_expected_critical_height`.
    fn make_rod(young_modulus: f32, height_m: f32, dx_meters: f32) -> Rod {
        let start = Vec2::new(9.0, 4.0);
        let end = Vec2::new(start.x, start.y + height_m / dx_meters);
        let points = build_straight_rod(start, end, 20, 0.01, dx_meters);
        let ea = young_modulus * 0.003 * 0.001;
        let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;
        Rod::new(points, RodMaterial::new(ea, ei, 0.0, 0.0))
    }

    #[test]
    fn add_rod_still_registers_correctly_when_over_its_own_critical_height() {
        let mut sim = Simulation::empty(SimConfig::earth(32, 0.01, 0.02));
        // E=5e6, height=0.10m -- the real, confirmed-over-critical blade B case.
        let idx = sim.add_rod(make_rod(5.0e6, 0.10, 0.01));
        assert_eq!(idx, 0);
        assert_eq!(sim.rods().len(), 1);
    }

    #[test]
    fn add_rod_still_registers_correctly_when_safely_under_its_own_critical_height() {
        let mut sim = Simulation::empty(SimConfig::earth(32, 0.01, 0.02));
        // E=1e7, height=0.10m -- the real, confirmed-safe blade A case.
        let idx = sim.add_rod(make_rod(1.0e7, 0.10, 0.01));
        assert_eq!(idx, 0);
        assert_eq!(sim.rods().len(), 1);
    }
}
