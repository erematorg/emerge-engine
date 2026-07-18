use glam::{IVec2, Mat2, Vec2};

/// Shape mask applied to the particle grid during spawning.
///
/// The grid always iterates the bounding box defined by `SpawnRegion::box_size`.
/// `SpawnShape::Disk` discards particles whose grid position falls outside the
/// circle, producing a disk-shaped region with the same spacing and jitter.
#[non_exhaustive]
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
pub struct SimConfig {
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
    /// Speed below which a passive (activation == 0) particle becomes eligible for sleep.
    /// 0.0 = sleep disabled. Typical: 0.01–0.05 grid-cells/s.
    /// Sleeping particles skip P2G and G2P entirely; woken by neighbouring active cells.
    pub sleep_threshold: f32,
    /// Coulomb friction coefficient for multi-field contact between a `contact_group != 0`
    /// particle and everything else (Bardenhagen 2001 — see `Particle::contact_group` doc).
    /// Only has any effect at all when at least one particle actually sets a nonzero
    /// `contact_group`; otherwise `Grid::resolve_contact` never has anything to resolve,
    /// regardless of this value. 0.0 = frictionless (normal no-penetration only, free
    /// tangential slip). Real dry-material Coulomb coefficients are typically 0.3-0.9.
    pub contact_friction: f32,
    /// ASFLIP blend factor [0, 1] (Fei, Guo, Wu, Huang, Gao 2021, "Revisiting Integration in
    /// the Material Point Method: A Scheme for Easier Separation and Less Dissipation", ACM
    /// TOG 40(4)). Reintroduces a FLIP-style velocity/position correction on top of ordinary
    /// APIC, letting granular/debris material separate crisply instead of smearing together.
    /// 0.0 = disabled — byte-identical to plain APIC, the default for every existing scene/
    /// test. ~0.97 matches the paper's own reference implementation (`nepluno/pyasflip`).
    /// Costs nothing when 0.0: no grid-velocity snapshot is taken, G2P takes the exact
    /// original code path.
    pub asflip_blend: f32,

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

impl Default for SimConfig {
    /// Safe production defaults: adaptive timestepping on, state projection on.
    /// Use [`SimConfig::standard`] or [`SimConfig::earth`] in practice — they set the
    /// important physical parameters (grid_res, dt, gravity) from arguments.
    fn default() -> Self {
        Self {
            grid_res: 64,
            grid_cell_size: 1.0,
            dt: 1.0,
            adaptive_timestep: true,
            cfl_include_affine_speed: true,
            cfl_coefficient: 0.9,
            material_cfl_coefficient: 0.5,
            viscous_timestep_coefficient: 0.5,
            min_dt: 1.0e-3,
            project_invalid_state: true,
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
            sleep_threshold: 0.0,
            contact_friction: 0.5,
            asflip_blend: 0.0,
            dx_meters: 1.0,
            dt_seconds: 1.0,
        }
    }
}

impl SimConfig {
    /// Simulation-ready config: sets the three physical parameters that differ per sim.
    ///
    /// Inherits safe defaults from `Default` (adaptive timestepping, state projection on).
    pub fn standard(grid_res: usize, dt: f32, gravity: Vec2) -> Self {
        Self {
            grid_res,
            dt,
            gravity,
            ..Self::default()
        }
    }

    /// Stripped-down config with adaptive timestepping and state projection disabled.
    ///
    /// Use only for: unit tests that need exact deterministic substeps, benchmarks
    /// where you want to measure a fixed workload, or comparing against an external reference.
    /// Never use for real simulations — J can go negative and NaN-cascade.
    pub fn unsafe_defaults() -> Self {
        Self {
            adaptive_timestep: false,
            project_invalid_state: false,
            ..Self::default()
        }
    }

    /// Earth-scale simulation preset.
    ///
    /// Derives gravity and unit scaling from real physical constants so that
    /// material parameters passed via `lame_from_si` produce correct behaviour.
    ///
    /// # Arguments
    /// * `grid_res`    — number of cells per side
    /// * `cell_m`      — physical size of one grid cell in metres (e.g. `0.01` for 1 cm)
    /// * `dt`          — frame time step in simulation seconds (e.g. `0.05`)
    ///
    /// # Derived values
    /// `gravity_solver = 9.81 / cell_m` cells/s² (downward, −Y).
    ///
    /// # Example
    /// ```rust,no_run
    /// # extern crate emerge_engine as emerge;
    /// # use emerge::SimConfig;
    /// // 64-cell domain, 1 cm/cell → g = 981 cells/s²
    /// let config = SimConfig::earth(64, 0.01, 0.05);
    /// ```
    pub fn earth(grid_res: usize, cell_m: f32, dt: f32) -> Self {
        // g [cells/s²] = 9.81 [m/s²] / cell_m [m/cell]
        // Derived from v += gravity * sub_dt where sub_dt is in real seconds.
        let g_solver = 9.81 / cell_m;
        Self {
            dx_meters: cell_m,
            dt_seconds: dt,
            ..Self::standard(grid_res, dt, Vec2::new(0.0, -g_solver))
        }
    }

    // ── SI conversion helpers ─────────────────────────────────────────────────

    /// Convert SI Young's modulus (Pa) + Poisson ratio to grid-unit Lamé parameters.
    ///
    /// Equivalent to `lame_from_si(e_pa, nu, rho, self.dx_meters, self.dt_seconds)`.
    /// Requires `earth()` or explicit `dx_meters`/`dt_seconds` to be meaningful.
    pub fn lame_from_si_cfg(&self, e_pa: f32, nu: f32, rho_kg_m3: f32) -> (f32, f32) {
        crate::materials::lame_from_si(e_pa, nu, rho_kg_m3, self.dx_meters, self.dt_seconds)
    }

    /// Convert SI stress or pressure (Pa) to grid units.
    ///
    /// Use for: yield stress, tensile strength, eos_stiffness, surface tension.
    /// Scale: `p_grid = p_SI · dt² / (ρ · dx²)`
    pub fn stress_from_si(&self, pa: f32, rho_kg_m3: f32) -> f32 {
        pa * self.dt_seconds * self.dt_seconds / (rho_kg_m3 * self.dx_meters * self.dx_meters)
    }

    /// Convert SI dynamic viscosity (Pa·s) to grid units.
    ///
    /// Viscosity multiplies the velocity gradient (units: 1/step in grid space), so its
    /// non-dimensionalization has one extra factor of dt versus stress:
    /// `η_grid = η_SI · ρ · dx² / dt³`
    pub fn visc_from_si(&self, eta_pa_s: f32, rho_kg_m3: f32) -> f32 {
        eta_pa_s * rho_kg_m3 * self.dx_meters * self.dx_meters
            / (self.dt_seconds * self.dt_seconds * self.dt_seconds)
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
            self.contact_friction >= 0.0,
            "contact_friction must be non-negative"
        );
        assert!(
            self.max_substeps_per_step > 0,
            "max_substeps_per_step must be > 0"
        );
        assert!(
            self.default_initial_volume > 0.0,
            "default_initial_volume must be positive"
        );
        assert!(self.j_max > 1.0, "j_max must be > 1.0");
        assert!(
            (0.0..=1.0).contains(&self.apic_blend),
            "apic_blend must be in [0, 1]"
        );
        assert!(
            self.boundary_thickness > 0 && self.boundary_thickness < self.grid_res - 1,
            "boundary_thickness must be in [1, grid_res-2]"
        );
    }
}

/// Initial particle layout — consumed once at spawn, not needed afterward.
///
/// Build via fluent methods on `SpawnRegion::for_sim`:
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{SimConfig, SpawnRegion};
/// # use glam::Vec2;
/// # let config = SimConfig::standard(64, 0.05, Vec2::NEG_Y * 0.3);
/// let spawn = SpawnRegion::for_sim(&config)
///     .at(Vec2::new(32.0, 40.0))
///     .disk(12.0)            // circle instead of box
///     .spacing(0.5)
///     .material(1);
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SpawnRegion {
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
    /// Per-region particle mass override (grid units). `None` (default) falls back to
    /// `SimConfig::particle_mass` — the single global value used when every material in a
    /// scene has the same real density. Set this explicitly when spawning multiple materials
    /// with different `rho_kg_m3` in the same simulation: `SimConfig::particle_mass` is one
    /// value shared by the whole `Simulation`, so without a per-region override every
    /// material's particles get identical mass regardless of their specified density —
    /// stiffness differs correctly (via Lamé/EOS conversion) but inertia does not.
    /// Compute as `rho_kg_m3 * (spacing * dx_meters).powi(2)` for a 2D areal-density particle.
    /// `.mass_from(&props, &config)` computes and sets this from a physical-property struct
    /// using this region's own `spacing` — prefer it over `.mass()` to avoid passing spacing
    /// twice (a real duplication risk).
    pub mass_override: Option<f32>,
}

impl Default for SpawnRegion {
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
            mass_override: None,
        }
    }
}

impl SpawnRegion {
    /// Starting point for fluent spawn configuration, centered in the solver domain.
    ///
    /// The center tracks `grid_res` so examples remain correct when you change resolution.
    pub fn for_sim(solver: &SimConfig) -> Self {
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

    /// Per-region particle mass override (grid units), for scenes mixing materials with
    /// different real densities. See the field doc on `mass_override` for the SI formula.
    pub fn mass(mut self, particle_mass: f32) -> Self {
        self.mass_override = Some(particle_mass);
        self
    }

    /// Like `.mass()`, but computes the value from a physical-property struct and
    /// THIS region's own `spacing` (already set via `.spacing()` or the `spacing`
    /// field) — avoids passing spacing twice, a real duplication risk (see
    /// `mass_override`'s field doc; LP hit a related sync bug from this exact
    /// pattern, fixed 2026-06-22).
    pub fn mass_from(mut self, props: &impl crate::ParticleMass, config: &SimConfig) -> Self {
        self.mass_override = Some(props.particle_mass(self.spacing, config));
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

    /// Non-panicking check: would this region fit entirely inside `solver`'s
    /// domain (same boundary math `validate_for_sim` asserts on)? For callers
    /// building a `SpawnRegion` from live/interactive input (mouse position,
    /// a creature's current location) where going out of bounds is a normal,
    /// expected outcome to skip gracefully -- not a programmer error to crash
    /// on. `validate_for_sim` stays a hard assert for the scripted/startup
    /// spawn path, where an out-of-bounds region really is a real bug worth
    /// catching loudly; this is the same check, exposed so interactive
    /// callers aren't forced to hand-derive the margin math themselves (that
    /// duplication is exactly how a real off-by-one crash slipped into
    /// `material_sandbox_gpu`'s paint tool).
    pub fn fits_in_sim(&self, solver: &SimConfig) -> bool {
        if self.spacing <= 0.0 || self.box_size.x <= 0 || self.box_size.y <= 0 {
            return false;
        }
        let half = self.box_size.as_vec2() * 0.5;
        let min = self.box_center - half;
        let max = self.box_center + half;
        let domain_min = solver.boundary_thickness as f32;
        let domain_max = solver.grid_res.saturating_sub(solver.boundary_thickness) as f32;
        min.x >= domain_min && min.y >= domain_min && max.x <= domain_max && max.y <= domain_max
    }

    /// Validate spawn-side constraints relative to the solver domain.
    pub fn validate_for_sim(&self, solver: &SimConfig) {
        assert!(self.spacing > 0.0, "spacing must be positive");
        assert!(self.box_size.x > 0, "box_size.x must be positive");
        assert!(self.box_size.y > 0, "box_size.y must be positive");

        let half = self.box_size.as_vec2() * 0.5;
        let min = self.box_center - half;
        let max = self.box_center + half;

        assert!(
            self.fits_in_sim(solver),
            "spawn region must stay inside the simulation domain \
             (boundary_thickness={}, grid_res={}): box [{:.1},{:.1}]–[{:.1},{:.1}]",
            solver.boundary_thickness,
            solver.grid_res,
            min.x,
            min.y,
            max.x,
            max.y
        );
    }
}

#[cfg(test)]
mod fits_in_sim_tests {
    use super::*;

    fn config() -> SimConfig {
        SimConfig::standard(64, 0.05, glam::Vec2::NEG_Y)
    }

    #[test]
    fn region_well_inside_domain_fits() {
        let region = SpawnRegion {
            spacing: 0.5,
            box_size: glam::IVec2::new(6, 6),
            box_center: glam::Vec2::new(32.0, 32.0),
            ..SpawnRegion::for_sim(&config())
        };
        assert!(region.fits_in_sim(&config()));
    }

    #[test]
    fn region_crossing_the_boundary_does_not_fit() {
        // Exact repro of the material_sandbox_gpu panic: box_size=6 centered
        // near the domain's right edge overruns the boundary by 0.5 units.
        let region = SpawnRegion {
            spacing: 0.5,
            box_size: glam::IVec2::new(6, 6),
            box_center: glam::Vec2::new(59.5, 19.8),
            ..SpawnRegion::for_sim(&config())
        };
        assert!(!region.fits_in_sim(&config()));
    }

    #[test]
    fn region_exactly_on_the_boundary_fits() {
        // grid_res=64, boundary_thickness default -- confirm the check is
        // inclusive (>=/<=) at the exact edge, not off-by-one in either
        // direction.
        let c = config();
        let half = 3.0;
        let edge = c.grid_res as f32 - c.boundary_thickness as f32 - half;
        let region = SpawnRegion {
            spacing: 0.5,
            box_size: glam::IVec2::new(6, 6),
            box_center: glam::Vec2::new(edge, edge),
            ..SpawnRegion::for_sim(&c)
        };
        assert!(region.fits_in_sim(&c));
    }

    #[test]
    fn fits_in_sim_and_validate_for_sim_agree() {
        // The two must never disagree -- validate_for_sim delegates to
        // fits_in_sim internally specifically to prevent them drifting apart.
        let c = config();
        let bad = SpawnRegion {
            spacing: 0.5,
            box_size: glam::IVec2::new(6, 6),
            box_center: glam::Vec2::new(59.5, 19.8),
            ..SpawnRegion::for_sim(&c)
        };
        assert!(!bad.fits_in_sim(&c));
        let result = std::panic::catch_unwind(|| bad.validate_for_sim(&c));
        assert!(result.is_err(), "validate_for_sim should have panicked");
    }
}
