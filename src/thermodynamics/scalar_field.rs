//! Generic scalar diffusion-advection field, grid-coupled to MPM particles.
//!
//! Implements ∂φ/∂t = D·∇²φ − λ·φ + S  (diffusion + first-order decay + sources)
//! where φ is any per-particle scalar, read/written via function pointers.
//!
//! # Algorithm (per substep) — identical to ThermalDiffusion
//! 1. **Source** — optional: inject S(p)·dt into each particle before scattering
//! 2. **P2G** — scatter mass-weighted φ to the grid
//! 3. **Normalize** — grid_φ = Σ(w·m·φ) / Σ(w·m); empty cells = ambient
//! 4. **Laplacian FD** — explicit Euler: φ_new = φ + dt·D·∇²φ
//! 5. **Decay** — φ_new *= exp(−λ·dt)  (or equivalently φ_new += −λ·φ·dt for small λ·dt)
//! 6. **G2P** — gather Δφ back to particles
//!
//! # Use cases
//! - **Heat** — same as `ThermalDiffusion`, D = k/(ρ·cₚ·dx²)
//! - **Chemical / pheromone** — high decay_rate (seconds to minutes half-life)
//! - **Nutrient / oxygen** — low decay_rate, sourced by terrain particles
//! - **Signal / pressure wave** — high diffusivity, zero decay
//!
//! # Fn pointer API
//! `get` and `set` are plain function pointers (not closures) so the field
//! is `Send + Sync` and can be stored without lifetime annotation.
//! `set` receives the **delta** (Δφ), not the new absolute value — this
//! preserves per-particle state not captured by the grid (sparse regions, edges).

use glam::IVec2;

use crate::{
    grid::kernel::quadratic_weights,
    particle::{Particle, Particles},
};

/// A diffusing, decaying scalar field grid-coupled to MPM particles.
///
/// # Example — pheromone field
/// ```rust,no_run
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
pub struct ScalarDiffusionField {
    pub config: ScalarDiffusionConfig,
    /// Read the scalar value φ from a particle.
    pub get: fn(&Particle) -> f32,
    /// Apply a delta Δφ to a particle (called during G2P).
    pub set: fn(&mut Particle, f32),
    /// Optional per-particle source term in φ/s.
    /// Each substep: φ_particle += source(p, φ) · dt before P2G.
    ///
    /// Second argument is the current φ value of the particle — enables
    /// nonlinear (reaction-diffusion) sources, e.g. Gray-Scott: `−u·v²`.
    /// Use for fire emitting heat, creatures emitting pheromone, Turing patterns, etc.
    pub source: Option<fn(&Particle, f32) -> f32>,

    grid_res: usize,
    grid_mass: Vec<f32>, // Σ(w · mass)          — cleared each step
    grid_norm: Vec<f32>, // φ_grid (pre-Laplacian) — needed for G2P delta
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
    pub fn grid_res(&self) -> usize {
        self.grid_res
    }

    /// Apply one substep of diffusion to the particle set.
    ///
    /// Call once per MPM substep, after force fields.
    pub fn apply(&mut self, particles: &mut Particles, sub_dt: f32) {
        let n = self.grid_res * self.grid_res;
        let res = self.grid_res as i32;

        // --- Source injection: φ += S(p)·dt before scattering ---
        if let Some(src) = self.source {
            for pi in 0..particles.len() {
                let mut p = particles.get(pi);
                let phi = (self.get)(&p);
                let inject = src(&p, phi) * sub_dt;
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
        // Layout: column-major (x * res + y), matching mechanics grid.
        // ∇²φ ≈ φ[x-1,y] + φ[x+1,y] + φ[x,y-1] + φ[x,y+1] − 4·φ[x,y]
        let d_dt = self.config.diffusivity * sub_dt;
        let decay_factor = 1.0 - self.config.decay_rate * sub_dt;
        for x in 0..self.grid_res {
            for y in 0..self.grid_res {
                let c = x * self.grid_res + y;
                let phi_c = self.grid_norm[c];
                let phi_xm = if x > 0 {
                    self.grid_norm[c - self.grid_res]
                } else {
                    self.config.ambient
                };
                let phi_xp = if x + 1 < self.grid_res {
                    self.grid_norm[c + self.grid_res]
                } else {
                    self.config.ambient
                };
                let phi_ym = if y > 0 {
                    self.grid_norm[c - 1]
                } else {
                    self.config.ambient
                };
                let phi_yp = if y + 1 < self.grid_res {
                    self.grid_norm[c + 1]
                } else {
                    self.config.ambient
                };
                let laplacian = phi_xm + phi_xp + phi_ym + phi_yp - 4.0 * phi_c;
                self.grid_work[c] = (phi_c + d_dt * laplacian) * decay_factor;
            }
        }

        // --- G2P: gather Δφ = (φ_new − φ_old) back to particles ---
        // Scatter delta, not absolute — preserves per-particle state in sparse/edge regions.
        // grid_work = φ_new, grid_norm = φ_old.
        for pi in 0..particles.len() {
            let p_ref = particles.get(pi);
            let w = quadratic_weights(p_ref.x);
            let mut delta = 0.0f32;
            let mut w_sum = 0.0f32;

            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    delta += weight * (self.grid_work[idx] - self.grid_norm[idx]);
                    w_sum += weight;
                }
            }

            if w_sum > 1e-10 {
                let mut p = particles.get(pi);
                (self.set)(&mut p, delta / w_sum);
                particles.set(pi, p);
            }
        }
    }
}
