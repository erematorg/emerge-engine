//! Generic scalar diffusion-advection field, grid-coupled to MPM particles.
//!
//! Implements ∂φ/∂t = D·∇²φ − λ·φ + S  (diffusion + first-order decay + sources)
//! where φ is any per-particle scalar, read/written via function pointers.
//!
//! # Algorithm (per substep) -- identical to ThermalDiffusion
//! 1. **Source** -- optional: inject S(p)·dt into each particle before scattering
//! 2. **P2G** -- scatter mass-weighted φ to the grid
//! 3. **Normalize** -- grid_φ = Σ(w·m·φ) / Σ(w·m); empty cells = ambient
//! 4. **Laplacian FD** -- explicit Euler: φ_new = φ + dt·D·∇²φ
//! 5. **Decay** -- φ_new *= exp(−λ·dt)  (or equivalently φ_new += −λ·φ·dt for small λ·dt)
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
/// own doc for what each argument is for.
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

    /// PIC/FLIP-style transfer blend, real and established (Zhu & Bridson
    /// 2005; standard in production fluid solvers, commonly ~0.95 FLIP/0.05
    /// PIC -- see Bridson, *Fluid Simulation for Computer Graphics*, already
    /// cited elsewhere in this engine; this engine's own `apic_blend`
    /// applies the identical idea to velocity transfer already).
    ///
    /// `1.0` (default) = pure delta transfer ("FLIP-like"): a particle's own
    /// stored value plus the grid-computed change, preserving per-particle
    /// heterogeneity -- exactly this field's original, only behavior, so
    /// every existing heat/pheromone/nutrient scene is byte-identical.
    /// `0.0` = pure absolute transfer ("PIC-like"): a particle simply takes
    /// the local grid average, discarding its own prior value -- damped,
    /// stable, no nullspace noise.
    ///
    /// Real motivating case: a passive reader (e.g. sand) co-located with an
    /// aggressive, persistent source (e.g. water) can accumulate the SAME
    /// grid-computed delta the source itself gets every substep, with
    /// nothing of its own to counterbalance it -- the exact nullspace-noise
    /// failure FLIP is documented to have. A lower blend trades some of the
    /// "preserve my own value" property for the stability a passive
    /// participant actually needs.
    pub blend: f32,

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

    /// Apply one substep of diffusion to the particle set.
    ///
    /// Call once per MPM substep, after force fields.
    pub fn apply(&mut self, particles: &mut Particles, sub_dt: f32, materials: &MaterialRegistry) {
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
        let d_dt = self.config.diffusivity * sub_dt;
        super::stencil::laplacian_step(
            &self.grid_norm,
            &mut self.grid_work,
            self.grid_res,
            d_dt,
            self.config.ambient,
        );
        // Decay pulls toward zero (not ambient -- a real, deliberate
        // difference from ThermalDiffusion's Newton cooling, see module doc).
        let decay_factor = 1.0 - self.config.decay_rate * sub_dt;
        if decay_factor != 1.0 {
            for v in self.grid_work.iter_mut() {
                *v *= decay_factor;
            }
        }

        // --- G2P: gather back to particles, PIC/FLIP-blended (see `blend`'s own doc) ---
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

    /// Real regression test for `blend` itself, at the hardest case on
    /// purpose: sand and water EXACTLY co-located (every pair sharing one
    /// position), not just spatially nearby -- the worst-case scenario for
    /// FLIP's nullspace-noise failure (see `blend`'s own doc), since a
    /// passive reader here gets the identical grid delta the source itself
    /// gets, every substep, with nothing of its own to counterbalance it.
    /// At a real PIC-leaning blend, the passive material must still pick up
    /// genuine positive saturation instead of drifting negative.
    #[test]
    fn low_blend_gives_stable_positive_transfer_even_at_exact_colocation() {
        let mut registry = MaterialRegistry::with_default(Box::new(
            DruckerPragerMaterial::cohesionless(1.0e5, 0.2),
        ));
        registry.insert(
            1,
            Box::new(NewtonianFluidMaterial::low_viscosity(4.0, 10.0)),
        );

        // Real spatial spread (a small block, not a singular point) -- two
        // particles sharing one exact position is a degenerate edge case
        // (no meaningful gradient for the Laplacian to act on), not a
        // realistic test of spatial diffusion between two bodies.
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
