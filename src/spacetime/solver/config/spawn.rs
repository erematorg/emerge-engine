//! `SpawnRegion` -- initial particle layout, consumed once at spawn -- split
//! out of `config.rs` (was its "Initial particle layout" section, ~270 of the
//! file's ~596 lines). Kept as its own module because it's a genuinely
//! different concern from `SimConfig`: a fluent one-shot builder for WHERE and
//! HOW to spawn particles, not the solver's own long-lived numerical/physical
//! parameters.

use glam::{IVec2, Mat2, Vec2};

use super::SimConfig;

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
    /// Set `box_size` large enough to contain the disk -- a square of side
    /// `2 * radius` is exactly right, e.g. `IVec2::splat((2.0 * radius) as i32 + 1)`.
    Disk { radius: f32 },
}

/// Initial particle layout -- consumed once at spawn, not needed afterward.
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
    /// Per-region particle mass override (GRID units). `None` (default) derives
    /// it as `SimConfig::grid_density * spacing^2`, which is correct for any
    /// single-material scene and adapts automatically when the region is
    /// refined.
    ///
    /// Set it only to give one region a real density CONTRAST against another:
    /// `(rho_kg_m3 / config.reference_density_kg_m3) * spacing^2`, with every
    /// material's stress converted using that same `reference_density_kg_m3`.
    /// Without it, mixed-density regions differ correctly in stiffness (via the
    /// Lamé/EOS conversion) but not in inertia.
    ///
    /// Note the units: this is NOT the SI kilogram mass
    /// `rho_kg_m3 * (spacing * dx_meters)^2` that `ParticleMass::particle_mass`
    /// returns. The two differ by exactly `rho * dx_meters^2` -- the same
    /// factor `lame_from_si_physical` divides stress by -- so passing an SI
    /// mass in here directly is a unit mismatch. `.mass_from(&props, &config)`
    /// applies that conversion for you and uses this region's own `spacing`;
    /// prefer it over `.mass()`.
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
    pub const fn at(mut self, center: Vec2) -> Self {
        self.box_center = center;
        self
    }

    /// Set the bounding box size in grid cells (used for box shape and disk bounding box).
    pub const fn box_of(mut self, size: IVec2) -> Self {
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
    pub const fn spacing(mut self, s: f32) -> Self {
        self.spacing = s;
        self
    }

    /// Material ID for all particles in this region.
    pub const fn material(mut self, id: u32) -> Self {
        self.material_id = id;
        self
    }

    /// Per-region particle mass override, in GRID units, for scenes mixing
    /// materials with different real densities. See the field doc on
    /// `mass_override` -- an SI kilogram mass passed here is a unit mismatch.
    pub const fn mass(mut self, grid_mass: f32) -> Self {
        self.mass_override = Some(grid_mass);
        self
    }

    /// Like `.mass()`, but derives the value from a physical-property struct and
    /// THIS region's own `spacing` (already set via `.spacing()` or the `spacing`
    /// field) -- avoids passing spacing twice, a real duplication risk (see
    /// `mass_override`'s field doc; LP hit a sync bug from this exact pattern).
    ///
    /// `ParticleMass::particle_mass` reports real SI kilograms, so this divides
    /// by `reference_density_kg_m3 * dx_meters^2` to land in the grid units the
    /// solver actually integrates -- the same conversion
    /// `lame_from_si_physical` applies to stress. What survives the division is
    /// the density RATIO: a region of water at `spacing` gets exactly
    /// `spacing^2`, and denser materials get proportionally more.
    pub fn mass_from(mut self, props: &impl crate::ParticleMass, config: &SimConfig) -> Self {
        let si_kg = props.particle_mass(self.spacing, config);
        let to_grid = 1.0 / (config.reference_density_kg_m3 * config.dx_meters * config.dx_meters);
        self.mass_override = Some(si_kg * to_grid);
        self
    }

    /// Run a P2G density pass after spawning to compute physically accurate initial volumes.
    ///
    /// Use for elastic solids and dense granular materials where incorrect initial density
    /// would cause a pressure spike on the first substep. Costs one extra P2G pass at spawn.
    pub const fn precompute_volumes(mut self) -> Self {
        self.precompute_initial_volumes = true;
        self
    }

    /// Initial speed randomization magnitude (0 = all particles at rest).
    pub const fn velocity_scale(mut self, scale: f32) -> Self {
        self.initial_velocity_scale = scale;
        self
    }

    /// Position jitter magnitude, as a fraction of `spacing`.
    ///
    /// 0.0 = perfect lattice (default). 0.2 is a good default for granular materials
    /// (sand, snow) to break lattice symmetry and prevent artificially regular piles.
    pub const fn jitter(mut self, scale: f32) -> Self {
        self.position_jitter = scale;
        self
    }

    /// Seed for jitter and initial velocity RNG.
    pub const fn rng_seed(mut self, seed: u32) -> Self {
        self.rng_seed = seed;
        self
    }

    /// Non-panicking check: would this region fit entirely inside `solver`'s
    /// domain (same boundary math `validate_for_sim` asserts on)? For callers
    /// building a `SpawnRegion` from live/interactive input (mouse position,
    /// a creature's current location) where going out of bounds is a normal,
    /// expected outcome to skip gracefully -- not a programmer error to crash
    /// on. `validate_for_sim` stays a hard assert for the scripted/startup
    /// spawn path, where an out-of-bounds region is a programmer error worth
    /// catching loudly; this is the same check, exposed so interactive
    /// callers aren't forced to hand-derive the margin math themselves.
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
