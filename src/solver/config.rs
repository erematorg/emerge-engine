use glam::{IVec2, Mat2, Vec2};

/// Shape mask applied to the particle grid during spawning.
///
/// The grid always iterates the bounding box defined by `SpawnConfig::box_size`.
/// `SpawnShape::Disk` discards particles whose grid position falls outside the
/// circle, producing a disk-shaped region with the same spacing and jitter.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SpawnShape {
    /// Fill the entire axis-aligned bounding box (default).
    Box,
    /// Fill a disk of `radius` grid-cells centered on `box_center`.
    ///
    /// Set `box_size` large enough to contain the disk — a square of side
    /// `2 * radius` is exactly right, e.g. `IVec2::splat((2.0 * radius) as i32 + 1)`.
    Disk { radius: f32 },
}

/// D⁻¹ = 4.0 for the quadratic B-spline MLS-MPM kernel (always).
/// Not a tunable parameter — hardcoded from Hu 2018 Table 1.
pub(crate) const KERNEL_D_INVERSE: f32 = 4.0;

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
    /// Gravitational acceleration in grid-coordinate units/s².
    /// Use `Vec2::new(x, y)` for angled or planetary gravity. Typical: `Vec2::new(0.0, -9.81)`.
    pub gravity: Vec2,
    pub boundary_thickness: usize,
    pub default_initial_volume: f32,
    pub recompute_density_each_step: bool,
    pub particle_mass: f32,
    /// Maximum substeps the adaptive loop may run per step() call.
    /// Prevents stiff materials or fast particles from making a single step() unboundedly expensive.
    /// 64 covers snow at lambda=38889 (c_P≈197, ~50 substeps) with headroom.
    pub max_substeps_per_step: usize,
    /// APIC affine-matrix blend [0, 1].
    /// 1.0 = full APIC (angular-momentum-conserving, taichi default).
    /// 0.0 = pure PIC (maximum numerical dissipation, fastest settling).
    /// Intermediate values blend between the two — equivalent to taichi's `apic_damping`.
    /// Tune down for fluids that need to damp out; keep at 1.0 for elastic solids.
    pub apic_blend: f32,
    /// Upper bound on volumetric expansion J = det(F).
    /// Particles that expand beyond this are rescaled back. No physical material expands
    /// this many times its initial volume without fracturing or flowing first.
    /// Default 50.0. Set higher for extreme-deformation sims (explosions, impacts).
    pub j_max: f32,

    // ── Physical unit scaling ──────────────────────────────────────────────────
    // Default 1.0 = simulation units (no scaling). Set these to enable SI-calibrated materials.
    // Use `lame_from_si` / `gravity_to_grid` in `materials::utils` to convert SI values.

    /// Physical length of one grid cell in meters. Default 1.0 (grid units).
    ///
    /// Example: if the simulation domain is 64 cells representing 0.64 m, set `dx_meters = 0.01`.
    pub dx_meters: f32,
    /// Physical duration of one simulation time unit in seconds. Default 1.0.
    ///
    /// Typically set to match `config.dt` in physical seconds.
    /// Gravity: `gravity = Vec2::new(0.0, -9.81) * dt_seconds^2 / dx_meters`.
    pub dt_seconds: f32,
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
            gravity: Vec2::new(0.0, -0.05),
            boundary_thickness: 2,
            default_initial_volume: 1.0,
            recompute_density_each_step: false,
            particle_mass: 1.0,
            max_substeps_per_step: 64,
            apic_blend: 1.0,
            j_max: 50.0,
            dx_meters: 1.0,
            dt_seconds: 1.0,
        }
    }
}

impl SolverConfig {
    /// Simulation-ready defaults for interactive examples.
    ///
    /// Enables adaptive timestepping and state projection — the two settings that are almost
    /// always desired in practice but are off in `Default` for backward compatibility.
    pub fn standard(grid_res: usize, dt: f32, gravity: Vec2) -> Self {
        Self {
            grid_res,
            dt,
            gravity,
            adaptive_timestep: true,
            project_invalid_state: true,
            // Fluid EOS requires per-step density from grid mass gather (two-pass equivalent).
            // Without this, fluid density is static from initialization → EOS pressure = 0 always.
            // Reference: incremental_mpm two-pass P2G, basic_fluids.rs explicit setting.
            recompute_density_each_step: true,
            ..Self::default()
        }
    }

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
        assert!(
            self.max_substeps_per_step > 0,
            "max_substeps_per_step must be > 0"
        );
        assert!(
            self.default_initial_volume > 0.0,
            "default_initial_volume must be positive"
        );
        assert!(self.j_max > 1.0, "j_max must be > 1.0");
        assert!((0.0..=1.0).contains(&self.apic_blend), "apic_blend must be in [0, 1]");
        assert!(
            self.boundary_thickness > 0 && self.boundary_thickness < self.grid_res - 1,
            "boundary_thickness must be in [1, grid_res-2]"
        );
    }
}

/// Initial particle layout — consumed once at spawn, not needed afterward.
///
/// Build via fluent methods on `SpawnConfig::for_solver`:
/// ```rust,no_run
/// # use emerge::{SolverConfig, SpawnConfig};
/// # use glam::Vec2;
/// # let config = SolverConfig::standard(64, 0.05, Vec2::NEG_Y * 0.3);
/// let spawn = SpawnConfig::for_solver(&config)
///     .at(Vec2::new(32.0, 40.0))
///     .disk(12.0)            // circle instead of box
///     .spacing(0.5)
///     .material(1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SpawnConfig {
    pub spacing: f32,
    pub box_size: IVec2,
    pub box_center: Vec2,
    pub shape: SpawnShape,
    pub initial_deformation_gradient: Mat2,
    pub precompute_initial_volumes: bool,
    /// Randomized initial speed. Each particle gets a random velocity in [−scale/2, +scale/2]².
    /// 0.0 = at rest (default). Small values (0.1–1.0) add visual variety.
    pub initial_velocity_scale: f32,
    /// Randomized position offset per particle, as a fraction of `spacing`.
    /// 0.0 = perfect lattice. 0.2 is a good default for granular materials (sand, snow)
    /// to break lattice symmetry and prevent artificially regular pile formation.
    pub position_jitter: f32,
    pub rng_seed: u32,
    /// Material for all particles in this region (default 0).
    pub material_id: u32,
}

impl Default for SpawnConfig {
    fn default() -> Self {
        Self {
            spacing: 1.0,
            box_size: IVec2::new(16, 16),
            box_center: Vec2::splat(32.0),
            shape: SpawnShape::Box,
            initial_deformation_gradient: Mat2::IDENTITY,
            precompute_initial_volumes: false,
            initial_velocity_scale: 0.0,
            position_jitter: 0.0,
            rng_seed: 1,
            material_id: 0,
        }
    }
}

impl SpawnConfig {
    /// Starting point for fluent spawn configuration, centered in the solver domain.
    ///
    /// The center tracks `grid_res` so examples remain correct when you change resolution.
    pub fn for_solver(solver: &SolverConfig) -> Self {
        Self {
            box_center: Vec2::splat(solver.grid_res as f32 * 0.5),
            ..Self::default()
        }
    }

    // ── Fluent builder methods ─────────────────────────────────────────────────

    /// Set the center of the spawn region in grid coordinates.
    pub fn at(mut self, center: Vec2) -> Self {
        self.box_center = center;
        self
    }

    /// Set the bounding box size in grid cells (used for box shape and disk bounding box).
    pub fn box_of(mut self, size: IVec2) -> Self {
        self.box_size = size;
        self
    }

    /// Spawn a disk of radius `r` grid-cells centered on `box_center`.
    ///
    /// Also sets `box_size` to the smallest square that contains the disk.
    /// Adjust `box_size` manually if you need a non-square bounding box.
    pub fn disk(mut self, r: f32) -> Self {
        self.shape = SpawnShape::Disk { radius: r };
        let side = (2.0 * r).ceil() as i32 + 1;
        self.box_size = IVec2::splat(side);
        self
    }

    /// Particle lattice spacing in grid cells.
    pub fn spacing(mut self, s: f32) -> Self {
        self.spacing = s;
        self
    }

    /// Material ID for all particles in this region.
    pub fn material(mut self, id: u32) -> Self {
        self.material_id = id;
        self
    }

    /// Run a P2G density pass after spawning to compute physically accurate initial volumes.
    ///
    /// Use for elastic solids and dense granular materials where incorrect initial density
    /// would cause a pressure spike on the first substep. Costs one extra P2G pass at spawn.
    pub fn precompute_volumes(mut self) -> Self {
        self.precompute_initial_volumes = true;
        self
    }

    /// Initial speed randomization magnitude (0 = all particles at rest).
    pub fn velocity_scale(mut self, scale: f32) -> Self {
        self.initial_velocity_scale = scale;
        self
    }

    /// Position jitter magnitude, as a fraction of `spacing`.
    ///
    /// 0.0 = perfect lattice (default). 0.2 is a good default for granular materials
    /// (sand, snow) to break lattice symmetry and prevent artificially regular piles.
    pub fn jitter(mut self, scale: f32) -> Self {
        self.position_jitter = scale;
        self
    }

    /// Seed for jitter and initial velocity RNG.
    pub fn rng_seed(mut self, seed: u32) -> Self {
        self.rng_seed = seed;
        self
    }

    /// Validate spawn-side constraints relative to the solver domain.
    pub fn validate_for_solver(&self, solver: &SolverConfig) {
        assert!(self.spacing > 0.0, "spacing must be positive");
        assert!(self.box_size.x > 0, "box_size.x must be positive");
        assert!(self.box_size.y > 0, "box_size.y must be positive");

        let half = self.box_size.as_vec2() * 0.5;
        let min = self.box_center - half;
        let max = self.box_center + half;
        // Spawn must stay strictly inside the boundary zone.
        let domain_min = solver.boundary_thickness as f32;
        let domain_max = solver.grid_res.saturating_sub(solver.boundary_thickness) as f32;

        assert!(
            min.x >= domain_min
                && min.y >= domain_min
                && max.x <= domain_max
                && max.y <= domain_max,
            "spawn region must stay inside the simulation domain \
             (boundary_thickness={}, grid_res={}): box [{:.1},{:.1}]–[{:.1},{:.1}]",
            solver.boundary_thickness, solver.grid_res, min.x, min.y, max.x, max.y
        );
    }
}
