use std::collections::BTreeMap;

use glam::{Mat2, Vec2};

use crate::diagnostics::{MpmSnapshot, collect_mpm_snapshot};
use crate::solver::{
    BoundaryCondition, MaterialModel, MaterialRegistry, SlipBoundary,
    density::{estimate_initial_particle_volumes, estimate_particle_density_and_volume},
    materials::FallbackMaterial,
    transfer::{gather_grid_to_particles, scatter_particles_to_grid},
};
use crate::{
    solver::config::{SolverConfig, SpawnConfig},
    state::{
        grid::{Cell, Grid},
        particle::Particle,
    },
};

pub struct MpmSolver {
    config: SolverConfig,
    particles: Vec<Particle>,
    grid: Grid,
    materials: MaterialRegistry,
    boundary: Box<dyn BoundaryCondition>,
    frame_index: u64,
    last_step_dt: f32,
    last_substeps: usize,
}

impl MpmSolver {
    pub fn new(config: SolverConfig, spawn: SpawnConfig) -> Self {
        config.validate();
        spawn.validate_for_solver(&config);

        let mut rng = LcgRng::new(spawn.rng_seed);
        let mut particles = initialize_particles(&config, spawn, &mut rng);
        let mut grid = Grid::new(config.grid_res);
        if spawn.precompute_initial_volumes {
            estimate_initial_particle_volumes(&mut particles, &mut grid);
        }
        let materials = MaterialRegistry::with_default(Box::new(FallbackMaterial));
        let boundary = Box::new(SlipBoundary::new(config.boundary_thickness));
        Self {
            config,
            particles,
            grid,
            materials,
            boundary,
            frame_index: 0,
            last_step_dt: config.dt,
            last_substeps: 0,
        }
    }

    pub fn with_default_material(mut self, material: Box<dyn MaterialModel>) -> Self {
        self.set_default_material(material);
        self
    }

    pub fn with_boundary(mut self, boundary: Box<dyn BoundaryCondition>) -> Self {
        self.set_boundary_condition(boundary);
        self
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

    pub fn particles(&self) -> &[Particle] {
        &self.particles
    }

    pub fn particles_mut(&mut self) -> &mut Vec<Particle> {
        &mut self.particles
    }

    pub fn assign_particle_materials_by_position<F>(&mut self, mut material_for: F)
    where
        F: FnMut(Vec2) -> u32,
    {
        for particle in &mut self.particles {
            particle.material_id = material_for(particle.x);
        }
    }

    pub fn material_particle_counts(&self) -> BTreeMap<u32, usize> {
        let mut counts = BTreeMap::new();
        for particle in &self.particles {
            *counts.entry(particle.material_id).or_insert(0) += 1;
        }
        counts
    }

    pub fn set_default_material(&mut self, material: Box<dyn MaterialModel>) {
        self.materials.set_default(material);
    }

    pub fn set_material(&mut self, material_id: u32, material: Box<dyn MaterialModel>) {
        self.materials.insert(material_id, material);
    }

    pub fn set_boundary_condition(&mut self, boundary: Box<dyn BoundaryCondition>) {
        self.boundary = boundary;
    }

    pub fn gravity(&self) -> f32 {
        self.config.gravity
    }

    pub fn set_gravity(&mut self, gravity: f32) {
        self.config.gravity = gravity;
    }

    pub fn diagnostics_snapshot(&self) -> MpmSnapshot {
        collect_mpm_snapshot(
            self.frame_index,
            &self.particles,
            &self.grid,
            &self.config,
            self.last_step_dt,
            self.last_substeps,
        )
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
        while remaining > f32::EPSILON && substeps_taken < self.config.max_substeps_per_step {
            // Cap sub-step at remaining time so we don't overshoot the configured frame dt.
            let sub_dt = choose_substep_dt(&self.config, &self.particles, &self.materials, remaining);
            self.do_substep(sub_dt);
            remaining -= sub_dt;
            self.last_step_dt = sub_dt;
            substeps_taken += 1;
        }
        self.last_substeps = substeps_taken;
        self.frame_index = self.frame_index.saturating_add(1);
    }

    fn do_substep(&mut self, sub_dt: f32) {
        // Density from the current grid is more accurate than the analytical estimate kept from last step.
        if self.config.recompute_density_each_step {
            estimate_particle_density_and_volume(&mut self.particles, &mut self.grid);
        }

        self.grid.clear();
        scatter_particles_to_grid(
            &self.particles,
            &mut self.grid,
            &self.materials,
            sub_dt,
            self.config.mls_d_inverse,
        );

        // Normalize accumulated momentum to velocity, then apply gravity and wall constraints.
        self.grid.update_velocities(sub_dt, self.config.gravity);
        let grid_res = self.grid.resolution();
        apply_boundary_conditions_to_grid(self.grid.cells_mut(), grid_res, self.boundary.as_ref());

        gather_grid_to_particles(
            &mut self.particles,
            &self.grid,
            sub_dt,
            self.boundary.as_ref(),
            &self.materials,
            self.config.mls_d_inverse,
        );

        // Explicit integration drifts under large stresses — reset before NaN propagates.
        if self.config.project_invalid_state {
            for particle in &mut self.particles {
                project_particle_state_to_admissible(particle, &self.config);
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
        estimate_initial_particle_volumes(&mut self.particles, &mut self.grid);
    }
}

fn initialize_particles(
    config: &SolverConfig,
    spawn: SpawnConfig,
    rng: &mut LcgRng,
) -> Vec<Particle> {
    let mut particles = Vec::new();
    let half = spawn.box_size.as_vec2() * 0.5;
    let min = spawn.box_center - half;
    let max = spawn.box_center + half;

    let mut i = min.x;
    while i < max.x {
        let mut j = min.y;
        while j < max.y {
            let random = Vec2::new(rng.next_f32(), rng.next_f32());
            let velocity = (random + spawn.initial_velocity_offset) * spawn.initial_velocity_scale;

            particles.push(Particle {
                x: Vec2::new(i, j),
                v: velocity,
                affine: Mat2::ZERO,
                deformation_gradient: spawn.initial_deformation_gradient,
                mass: config.particle_mass,
                initial_volume: config.default_initial_volume,
                volume: config.default_initial_volume,
                density: config.particle_mass / config.default_initial_volume,
                material_id: 0,
                plastic_jacobian: 1.0,
                elastic_hardening: 1.0,
                plastic_hardening: 0.0,
                log_vol_gain: 0.0,
            });
            j += spawn.spacing;
        }
        i += spawn.spacing;
    }

    particles
}

#[derive(Debug)]
struct LcgRng {
    state: u32,
}

impl LcgRng {
    fn new(seed: u32) -> Self {
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
fn choose_substep_dt(
    config: &SolverConfig,
    particles: &[Particle],
    materials: &MaterialRegistry,
    max_dt: f32,
) -> f32 {
    if !config.adaptive_timestep {
        return max_dt.min(config.dt);
    }

    let max_advection_speed = particles
        .iter()
        .map(|p| {
            let mut effective_speed = p.v.length();
            if config.cfl_include_affine_speed {
                effective_speed += affine_cfl_speed_contribution(&p.affine, config.grid_cell_size);
            }
            effective_speed
        })
        .fold(0.0f32, f32::max);

    let mut dt_bound = max_dt;
    if max_advection_speed > f32::EPSILON {
        dt_bound =
            dt_bound.min(config.cfl_coefficient * config.grid_cell_size / max_advection_speed);
    }

    // Material wave speeds (elastic, acoustic) impose tighter bounds than advection alone.
    for particle in particles {
        let material = materials.get(particle.material_id);
        let material_dt = material.timestep_bound(
            particle,
            config.grid_cell_size,
            config.material_cfl_coefficient,
            config.viscous_timestep_coefficient,
        );
        if material_dt.is_finite() && material_dt > 0.0 {
            dt_bound = dt_bound.min(material_dt);
        }
    }

    // min_dt may exceed max_dt when remaining frame time is tiny — take the smaller bound.
    dt_bound.clamp(config.min_dt.min(max_dt), max_dt)
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
    cells: &mut [Cell],
    grid_res: usize,
    boundary: &dyn BoundaryCondition,
) {
    for (i, cell) in cells.iter_mut().enumerate() {
        if cell.mass > 0.0 {
            boundary.apply_to_grid_velocity(i, grid_res, &mut cell.momentum);
        }
    }
}

fn project_particle_state_to_admissible(particle: &mut Particle, config: &SolverConfig) {
    let min = config.boundary_thickness.saturating_sub(1) as f32;
    let max = config.grid_res.saturating_sub(config.boundary_thickness) as f32;
    let domain_center = Vec2::splat((min + max) * 0.5);

    if !particle.x.is_finite() {
        particle.x = domain_center;
    } else {
        particle.x = particle.x.clamp(Vec2::splat(min), Vec2::splat(max));
    }

    if !particle.v.is_finite() {
        particle.v = Vec2::ZERO;
    }
    if !particle.affine.x_axis.is_finite() || !particle.affine.y_axis.is_finite() {
        particle.affine = Mat2::ZERO;
    }

    if !particle.deformation_gradient.x_axis.is_finite()
        || !particle.deformation_gradient.y_axis.is_finite()
        || particle.deformation_gradient.determinant() <= config.projection_min_deformation_j
    {
        particle.deformation_gradient = Mat2::IDENTITY;
    }

    if !particle.plastic_jacobian.is_finite() || particle.plastic_jacobian <= 0.0 {
        particle.plastic_jacobian = 1.0;
    }
    if !particle.elastic_hardening.is_finite() || particle.elastic_hardening <= 0.0 {
        particle.elastic_hardening = 1.0;
    }
    if !particle.plastic_hardening.is_finite() {
        particle.plastic_hardening = 0.0;
    }
    if !particle.log_vol_gain.is_finite() {
        particle.log_vol_gain = 0.0;
    }

    if !particle.mass.is_finite() || particle.mass <= 0.0 {
        particle.mass = config.particle_mass;
    }
    if !particle.initial_volume.is_finite() || particle.initial_volume <= 0.0 {
        particle.initial_volume = config
            .default_initial_volume
            .max(config.projection_min_volume);
    }
    if !particle.volume.is_finite() || particle.volume <= 0.0 {
        particle.volume = particle.initial_volume.max(config.projection_min_volume);
    }
    if !particle.density.is_finite() || particle.density <= 0.0 {
        particle.density = (particle.mass / particle.volume).max(config.projection_min_density);
    } else {
        particle.density = particle.density.max(config.projection_min_density);
    }
}
