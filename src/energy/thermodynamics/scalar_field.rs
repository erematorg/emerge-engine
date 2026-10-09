//! Generic scalar diffusion-advection field, grid-coupled to MPM particles.
//!
//! Implements ∂φ/∂t = D·∇²φ − λ·φ + S  (diffusion + first-order decay + sources)
//! where φ is any per-particle scalar, read/written via function pointers.
//!
//! # Algorithm (per call) -- identical to ThermalDiffusion
//! 1. **Source** -- optional: inject S(p)·dt into each particle before scattering
//! 2. **P2G** -- scatter mass-weighted φ to the grid
//! 3. **Normalize** -- grid_φ = Σ(w·m·φ) / Σ(w·m); empty cells = ambient
//! 4. **Laplacian FD** -- explicit Euler, φ_new = φ + dt·D·∇²φ; a call
//!    longer than `stability_fraction` of the stable step runs 1 to 6 in
//!    several equal passes
//! 5. **Decay** -- φ_new *= exp(−λ·dt), the exact solution
//! 6. **G2P** -- gather Δφ back to particles
//!
//! # Use cases
//! - **Heat** -- same as `ThermalDiffusion`, D = k/(ρ·cₚ·dx²)
//! - **Chemical / pheromone** -- high decay_rate (seconds to minutes half-life)
//! - **Nutrient / oxygen** -- low decay_rate, sourced by terrain particles
//! - **Signal / pressure wave** -- high diffusivity, zero decay
//!
//! # Fn pointer API
//! `get` and `set` are plain function pointers (not closures) so the field
//! is `Send + Sync` and can be stored without lifetime annotation.
//! `set` receives the **delta** (Δφ), not the new absolute value -- this
//! preserves per-particle state not captured by the grid (sparse regions, edges).

use glam::IVec2;

use crate::{
    grid::kernel::quadratic_weights,
    materials::{MaterialModel, registry::MaterialRegistry},
    particle::{Particle, Particles},
    solver::config::DEFAULT_MATERIAL_CFL_COEFFICIENT,
};

/// A diffusing, decaying scalar field grid-coupled to MPM particles.
///
/// # Example -- pheromone field
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// # use emerge::{ScalarDiffusionConfig, ScalarDiffusionField};
/// # use emerge::particle::Particle;
/// // Pheromone stored in particle.temperature; evaporates in ~10s.
/// let field = ScalarDiffusionField::new(
///     ScalarDiffusionConfig {
///         diffusivity: 0.5,   // spreads ~0.5 cells²/s
///         decay_rate:  0.1,   // 10s half-life
///         ambient:     0.0,
///     },
///     |p: &Particle| p.temperature,
///     |p: &mut Particle, delta: f32| p.temperature += delta,
///     64,
/// );
/// ```
/// Signature for `ScalarDiffusionField::source` -- factored into its own
/// alias (clippy's own complexity threshold, not just cosmetic) once the
/// resolved-material parameter joined the particle/phi pair. See `source`'s
/// doc for what each argument is for.
pub type ScalarFieldSource = fn(&Particle, f32, &dyn MaterialModel) -> f32;

pub struct ScalarDiffusionField {
    pub config: ScalarDiffusionConfig,
    /// Read the scalar value φ from a particle.
    pub get: fn(&Particle) -> f32,
    /// Apply a delta Δφ to a particle (called during G2P).
    pub set: fn(&mut Particle, f32),
    /// Optional per-particle source term in φ/s.
    /// Each substep: φ_particle += source(p, φ, material) · dt before P2G.
    ///
    /// Second argument is the current φ value of the particle -- enables
    /// nonlinear (reaction-diffusion) sources, e.g. Gray-Scott: `−u·v²`.
    /// Third argument is this particle's own resolved material -- lets a
    /// source classify by REAL PHYSICAL CONDITIONS (e.g.
    /// `material.owns_deformation_volume_state()` for "behaves like a
    /// strict fluid") instead of checking `material_id` against a specific,
    /// named identity. A material becomes a source because it genuinely
    /// satisfies real conditions, not because the engine was told in
    /// advance what it is -- the same property-driven principle every
    /// material's own construction already follows (`ElasticProps`,
    /// `FluidProps`, etc. -- see `matter::materials::physical_props`).
    /// Use for fire emitting heat, creatures emitting pheromone, Turing patterns, etc.
    pub source: Option<ScalarFieldSource>,

    /// PIC/FLIP-style transfer blend (Zhu & Bridson 2005; production fluid
    /// solvers commonly use ~0.95 FLIP/0.05 PIC, see Bridson, *Fluid
    /// Simulation for Computer Graphics*). `apic_blend` applies the same idea
    /// to velocity transfer.
    ///
    /// `1.0` (default) = pure delta transfer ("FLIP-like"): a particle's
    /// stored value plus the grid-computed change, preserving per-particle
    /// heterogeneity.
    /// `0.0` = pure absolute transfer ("PIC-like"): a particle simply takes
    /// the local grid average, discarding its own prior value -- damped,
    /// stable, no nullspace noise.
    ///
    /// A passive reader (e.g. sand) co-located with a persistent source (e.g.
    /// water) can receive the same grid delta as the source every substep
    /// with nothing of its own to counterbalance it, FLIP's documented
    /// nullspace-noise failure. A lower blend gives up some of "keep my own
    /// value" for the stability a passive participant needs.
    pub blend: f32,

    /// Fraction of the explicit diffusion step's stability limit each pass
    /// of `apply` uses: the definition of
    /// `SimConfig::material_cfl_coefficient`, which `Simulation` sets before
    /// each application.
    pub stability_fraction: f32,

    grid_res: usize,
    grid_mass: Vec<f32>, // Σ(w · mass)          -- cleared each step
    grid_norm: Vec<f32>, // φ_grid (pre-Laplacian) -- needed for G2P delta
    grid_work: Vec<f32>, // dual-use: P2G scatter buffer, then Laplacian output
                         // Note: grid_work is reused between P2G and Laplacian to avoid a 4th allocation.
                         // P2G phase:       grid_work = Σ(w · mass · φ)
                         // After normalize: grid_work = post-Laplacian φ  (φ_old data discarded)
                         // G2P reads:       (grid_work − grid_norm) = Δφ
}

/// Configuration for a scalar diffusion field.
#[derive(Clone, Debug, Default)]
pub struct ScalarDiffusionConfig {
    /// Diffusivity D in grid-units²/s.
    ///
    /// Controls how fast the scalar spreads spatially.
    /// - Heat in water (dx=1m): D ≈ 1.4e-7
    /// - Pheromone in air (dx=1m): D ≈ 0.2
    /// - Fast signal (dx=1m): D ≈ 5.0
    pub diffusivity: f32,

    /// First-order decay rate λ in 1/s.
    ///
    /// φ decreases as φ·e^(−λ·t). Half-life = ln(2)/λ.
    /// - 0.0 = conserved (heat, oxygen in closed system)
    /// - 0.07 = ~10s half-life (short-range pheromone)
    /// - 0.001 = ~700s half-life (persistent nutrient)
    pub decay_rate: f32,

    /// Value assigned to empty grid cells (no particle mass) and domain boundaries.
    ///
    /// Acts as a Dirichlet boundary condition at walls and vacuum regions.
    /// For pheromones: 0.0. For ambient temperature: background temperature.
    pub ambient: f32,
}

impl ScalarDiffusionField {
    /// Create a new scalar diffusion field.
    ///
    /// `get` reads the scalar from a particle; `set` adds a delta to it.
    /// `grid_res` must match the MPM solver's grid resolution.
    pub fn new(
        config: ScalarDiffusionConfig,
        get: fn(&Particle) -> f32,
        set: fn(&mut Particle, f32),
        grid_res: usize,
    ) -> Self {
        let n = grid_res * grid_res;
        Self {
            config,
            get,
            set,
            source: None,
            blend: 1.0,
            stability_fraction: DEFAULT_MATERIAL_CFL_COEFFICIENT,
            grid_res,
            grid_mass: vec![0.0; n],
            grid_norm: vec![0.0; n],
            grid_work: vec![0.0; n],
        }
    }

    /// Convenience constructor: field operates on `particle.temperature`.
    ///
    /// Equivalent to `ThermalDiffusion` but with the generic API.
    pub fn for_temperature(config: ScalarDiffusionConfig, grid_res: usize) -> Self {
        Self::new(
            config,
            |p| p.temperature,
            |p, delta| p.temperature += delta,
            grid_res,
        )
    }

    /// Read-only view of the post-step scalar field on the grid.
    ///
    /// Layout: `phi[x * grid_res + y]`.  Valid after the first call to `apply()`.
    /// Use with `ChemotaxisField::sync_from` to drive gradient-following forces.
    pub fn current_phi(&self) -> &[f32] {
        &self.grid_work
    }

    /// Grid resolution this field was created with.
    pub const fn grid_res(&self) -> usize {
        self.grid_res
    }

    /// Longest time one explicit diffusion step stays stable,
    /// `dx^2 / (4 D)` with `dx` one grid cell (`stencil::FIVE_POINT_
    /// STABILITY_LIMIT`); `None` without diffusion. The decay has no limit:
    /// it is applied by its exact solution.
    pub fn stable_step_limit(&self) -> Option<f32> {
        (self.config.diffusivity > 0.0)
            .then(|| super::stencil::FIVE_POINT_STABILITY_LIMIT / self.config.diffusivity)
    }

    /// Advances the field by `dt`, of any length, in as many equal passes
    /// as keep each diffusion step at `stability_fraction` of its stable
    /// step; each pass goes the whole way, particles to grid, one step,
    /// grid to particles. Splitting only the grid step, with one transfer
    /// back for the whole time, smoothed the whole change through the
    /// transfer kernels at once, so the result depended on how long `dt`
    /// was (`tests/subsystem_time_steps.rs`, gate 1). `Simulation` calls it
    /// once per `step()` with the time the step advanced; with an ordinary
    /// diffusivity that is one pass.
    pub fn apply(&mut self, particles: &mut Particles, dt: f32, materials: &MaterialRegistry) {
        let passes =
            super::stencil::stable_sub_steps(self.config.diffusivity * dt, self.stability_fraction);
        let pass_dt = dt / passes as f32;
        for _ in 0..passes {
            self.apply_pass(particles, pass_dt, materials);
        }
    }

    fn apply_pass(&mut self, particles: &mut Particles, sub_dt: f32, materials: &MaterialRegistry) {
        let n = self.grid_res * self.grid_res;
        let res = self.grid_res as i32;

        // --- Source injection: φ += S(p, φ, material)·dt before scattering ---
        if let Some(src) = self.source {
            for pi in 0..particles.len() {
                let mut p = particles.get(pi);
                let phi = (self.get)(&p);
                let material = materials.get(p.material_id);
                let inject = src(&p, phi, material) * sub_dt;
                (self.set)(&mut p, inject);
                particles.set(pi, p);
            }
        }

        // --- Clear scratch (grid_work = P2G scatter buffer, grid_mass = weights) ---
        for i in 0..n {
            self.grid_work[i] = 0.0;
            self.grid_mass[i] = 0.0;
        }

        // --- P2G: scatter mass-weighted φ into grid_work ---
        for pi in 0..particles.len() {
            let p = particles.get(pi);
            let phi = (self.get)(&p);
            let w = quadratic_weights(p.x);
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    let mw = weight * p.mass;
                    self.grid_work[idx] += mw * phi;
                    self.grid_mass[idx] += mw;
                }
            }
        }

        // --- Normalize: grid_norm = φ_grid or ambient where empty ---
        // grid_work (P2G scatter) is now discarded and reused for Laplacian output.
        for i in 0..n {
            self.grid_norm[i] = if self.grid_mass[i] > 1e-10 {
                self.grid_work[i] / self.grid_mass[i]
            } else {
                self.config.ambient
            };
        }

        // --- Laplacian FD + decay: explicit Euler, output into grid_work ---
        // grid_norm = φ_old (read-only from here). grid_work = φ_new (write).
        // `apply` keeps `sub_dt` within the stable step; one update past it
        // turned the profile to noise while keeping its variance and total
        // exact (`tests/subsystem_time_steps.rs`, gate 1).
        super::stencil::laplacian_step(
            &self.grid_norm,
            &mut self.grid_work,
            self.grid_res,
            self.config.diffusivity * sub_dt,
            self.config.ambient,
        );
        // Decay pulls toward zero (not ambient -- a deliberate
        // difference from ThermalDiffusion's Newton cooling, see module doc),
        // by the exact solution of `dphi/dt = -lambda phi`. The linear form
        // `1 - lambda dt` it replaces went negative past `lambda dt = 1`.
        let decay_factor = (-self.config.decay_rate * sub_dt).exp();
        if decay_factor != 1.0 {
            for v in self.grid_work.iter_mut() {
                *v *= decay_factor;
            }
        }

        // --- G2P: gather back to particles, PIC/FLIP-blended (see `blend`'s doc) ---
        // grid_work = φ_new, grid_norm = φ_old.
        for pi in 0..particles.len() {
            let p_ref = particles.get(pi);
            let w = quadratic_weights(p_ref.x);
            let mut new_local = 0.0f32; // interpolated φ_new at this particle's position
            let mut old_local = 0.0f32; // interpolated φ_old at this particle's position
            let mut w_sum = 0.0f32;

            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    new_local += weight * self.grid_work[idx];
                    old_local += weight * self.grid_norm[idx];
                    w_sum += weight;
                }
            }

            if w_sum > 1e-10 {
                let mut p = particles.get(pi);
                let new_local = new_local / w_sum;
                let old_local = old_local / w_sum;
                // FLIP-like: preserve the particle's own prior value, apply only
                // the grid-computed change. PIC-like: replace with the local
                // grid average outright, discarding the particle's own value.
                let flip_delta = new_local - old_local;
                let pic_delta = new_local - (self.get)(&p);
                let blended = self.blend * flip_delta + (1.0 - self.blend) * pic_delta;
                (self.set)(&mut p, blended);
                particles.set(pi, p);
            }
        }
    }
}

#[cfg(test)]
mod pic_flip_blend_tests {
    use super::*;
    use crate::materials::registry::MaterialRegistry;
    use crate::materials::{DruckerPragerMaterial, NewtonianFluidMaterial};
    use glam::Vec2;

    /// `blend` at its hardest case: sand and water exactly co-located (every
    /// pair at one position), where a passive reader gets the source's grid
    /// delta every substep with nothing to counterbalance it (see `blend`).
    /// At a PIC-leaning blend, the passive material must still pick up
    /// positive saturation instead of drifting negative.
    #[test]
    fn low_blend_gives_stable_positive_transfer_even_at_exact_colocation() {
        let mut registry = MaterialRegistry::with_default(Box::new(
            DruckerPragerMaterial::cohesionless(1.0e5, 0.2),
        ));
        registry.insert(
            1,
            Box::new(NewtonianFluidMaterial::low_viscosity(4.0, 10.0)),
        );

        // A small block, not a single point: two particles at one exact
        // position give the Laplacian no gradient to act on.
        let mut raw = Vec::new();
        for bx in 0..4 {
            for by in 0..4 {
                raw.push(Particle {
                    x: Vec2::new(6.0 + bx as f32, 6.0 + by as f32),
                    mass: 1.0,
                    initial_volume: 1.0,
                    volume: 1.0,
                    density: 1.0,
                    material_id: 0,
                    ..Particle::zeroed()
                });
                raw.push(Particle {
                    x: Vec2::new(6.0 + bx as f32, 6.0 + by as f32),
                    mass: 1.0,
                    initial_volume: 1.0,
                    volume: 1.0,
                    density: 1.0,
                    material_id: 1,
                    ..Particle::zeroed()
                });
            }
        }
        let mut particles = Particles::from(raw);

        let mut field = ScalarDiffusionField::new(
            ScalarDiffusionConfig {
                diffusivity: 0.5,
                decay_rate: 0.0,
                ambient: 0.0,
            },
            |p| p.scalar_field,
            |p, delta| p.scalar_field += delta,
            16,
        );
        fn src(_p: &Particle, phi: f32, material: &dyn MaterialModel) -> f32 {
            if material.owns_deformation_volume_state() && phi < 1.0 {
                4.0
            } else {
                0.0
            }
        }
        field.source = Some(src);
        field.blend = 0.3;

        for _ in 0..10 {
            field.apply(&mut particles, 0.1, &registry);
        }

        let sand_sum: f32 = (0..particles.len())
            .filter(|&i| particles.material_id[i] == 0)
            .map(|i| particles.scalar_field[i])
            .sum();
        let water_sum: f32 = (0..particles.len())
            .filter(|&i| particles.material_id[i] == 1)
            .map(|i| particles.scalar_field[i])
            .sum();
        println!("DIAG: sand_sum={sand_sum} water_sum={water_sum}");
        assert!(
            sand_sum > 0.0,
            "direct apply() must transfer real, positive phi to co-located non-fluid particles"
        );
    }
}
