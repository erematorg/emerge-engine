use glam::{IVec2, Mat2, Vec2};

/// Parameters that control the physics solver and its runtime behavior.
#[derive(Clone, Copy, Debug)]
pub struct SolverConfig {
    pub grid_res: usize,
    pub grid_cell_size: f32,
    pub dt: f32,
    pub adaptive_timestep: bool,
    pub cfl_include_affine_speed: bool,
    pub cfl_coefficient: f32,
    pub material_cfl_coefficient: f32,
    pub viscous_timestep_coefficient: f32,
    pub min_dt: f32,
    pub project_invalid_state: bool,
    pub projection_min_density: f32,
    pub projection_min_volume: f32,
    pub projection_min_deformation_j: f32,
    pub gravity: f32,
    pub boundary_thickness: usize,
    pub default_initial_volume: f32,
    pub recompute_density_each_step: bool,
    pub particle_mass: f32,
    // APIC/MLS quadratic-kernel D^{-1} coefficient. Standard value is 4.0 in grid units.
    pub mls_d_inverse: f32,
    /// Maximum substeps the adaptive loop may run per step() call.
    /// Prevents stiff materials or fast particles from making a single step() unboundedly expensive.
    /// 64 covers snow at lambda=38889 (c_P≈197, ~50 substeps) with headroom.
    pub max_substeps_per_step: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            grid_res: 64,
            grid_cell_size: 1.0,
            dt: 1.0,
            adaptive_timestep: false,
            cfl_include_affine_speed: true,
            cfl_coefficient: 0.9,
            material_cfl_coefficient: 0.5,
            viscous_timestep_coefficient: 0.5,
            min_dt: 1.0e-3,
            project_invalid_state: false,
            projection_min_density: 1.0e-6,
            projection_min_volume: 1.0e-6,
            projection_min_deformation_j: 1.0e-6,
            gravity: -0.05,
            boundary_thickness: 2,
            default_initial_volume: 1.0,
            recompute_density_each_step: false,
            particle_mass: 1.0,
            mls_d_inverse: 4.0,
            max_substeps_per_step: 64,
        }
    }
}

impl SolverConfig {
    /// Validate solver-side numerical and domain constraints.
    pub fn validate(&self) {
        assert!(self.grid_res >= 4, "grid_res must be >= 4");
        assert!(self.grid_cell_size > 0.0, "grid_cell_size must be positive");
        assert!(self.dt > 0.0, "dt must be positive");
        assert!(
            self.cfl_coefficient > 0.0,
            "cfl_coefficient must be positive"
        );
        assert!(
            self.material_cfl_coefficient > 0.0,
            "material_cfl_coefficient must be positive"
        );
        assert!(
            self.viscous_timestep_coefficient > 0.0,
            "viscous_timestep_coefficient must be positive"
        );
        assert!(self.min_dt > 0.0, "min_dt must be positive");
        assert!(self.min_dt <= self.dt, "min_dt must be <= dt");
        assert!(
            self.projection_min_density > 0.0,
            "projection_min_density must be positive"
        );
        assert!(
            self.projection_min_volume > 0.0,
            "projection_min_volume must be positive"
        );
        assert!(
            self.projection_min_deformation_j > 0.0,
            "projection_min_deformation_j must be positive"
        );
        assert!(self.particle_mass > 0.0, "particle_mass must be positive");
        assert!(self.mls_d_inverse > 0.0, "mls_d_inverse must be positive");
        assert!(self.max_substeps_per_step > 0, "max_substeps_per_step must be > 0");
        assert!(
            self.default_initial_volume > 0.0,
            "default_initial_volume must be positive"
        );
        assert!(
            self.boundary_thickness > 0 && self.boundary_thickness < self.grid_res - 1,
            "boundary_thickness must be in [1, grid_res-2]"
        );
    }
}

/// Initial particle layout — consumed once at solver construction, not needed afterward.
#[derive(Clone, Copy, Debug)]
pub struct SpawnConfig {
    pub spacing: f32,
    pub box_size: IVec2,
    pub box_center: Vec2,
    pub initial_deformation_gradient: Mat2,
    pub precompute_initial_volumes: bool,
    pub initial_velocity_offset: Vec2,
    pub initial_velocity_scale: f32,
    pub rng_seed: u32,
}

impl Default for SpawnConfig {
    fn default() -> Self {
        Self {
            spacing: 1.0,
            box_size: IVec2::new(16, 16),
            box_center: Vec2::splat(32.0),
            initial_deformation_gradient: Mat2::IDENTITY,
            precompute_initial_volumes: false,
            initial_velocity_offset: Vec2::new(-0.5, 2.25),
            initial_velocity_scale: 0.5,
            rng_seed: 1,
        }
    }
}

impl SpawnConfig {
    /// Build a spawn config centered in the solver domain.
    ///
    /// This is the safest default when `grid_res` changes, because the spawn box
    /// stays anchored to the current simulation domain instead of assuming 64x64.
    pub fn for_solver(solver: &SolverConfig) -> Self {
        Self {
            box_center: Vec2::splat(solver.grid_res as f32 * 0.5),
            ..Self::default()
        }
    }

    /// Validate spawn-side constraints relative to the solver domain.
    pub fn validate_for_solver(&self, solver: &SolverConfig) {
        assert!(self.spacing > 0.0, "spacing must be positive");
        assert!(self.box_size.x > 0, "box_size.x must be positive");
        assert!(self.box_size.y > 0, "box_size.y must be positive");

        let min = self.box_center - self.box_size.as_vec2() * 0.5;
        let max = self.box_center + self.box_size.as_vec2() * 0.5;
        let domain_min = solver.boundary_thickness.saturating_sub(1) as f32;
        let domain_max = solver.grid_res.saturating_sub(solver.boundary_thickness) as f32;

        assert!(
            min.x >= domain_min
                && min.y >= domain_min
                && max.x <= domain_max
                && max.y <= domain_max,
            "spawn box must stay inside the simulation domain"
        );
    }
}
