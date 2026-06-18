//! Grid-based Fourier heat diffusion for MPM particles.
//!
//! Implements ∂T/∂t = α·∇²T (Fourier's law) where α = k / (ρ·c_p).
//!
//! # Algorithm (per substep)
//! 1. **P2G** — scatter particle temperatures (mass-weighted) to a temporary grid
//! 2. **Normalize** — grid_temp = grid_heat / grid_mass (mass-weighted average)
//! 3. **Laplacian** — explicit Euler FD: T_new = T + α·dt·∇²T
//! 4. **G2P** — gather temperature delta back to particles
//!
//! Uses the same quadratic B-spline kernel as MPM transfer for consistency.
//!
//! # CFL note
//! Thermal CFL limit: dt_thermal ≤ dx² / (4α).
//! For typical materials (water α≈1.4e-7 m²/s, dx=0.1m) this is ~18000s —
//! orders of magnitude larger than MPM's wave-speed CFL (~0.002s).
//! Thermal CFL is never the bottleneck; no separate substep needed.

use glam::IVec2;

use crate::{grid::kernel::quadratic_weights, particle::Particles};

/// Configuration for grid-based thermal diffusion.
#[derive(Clone, Debug, Default)]
pub struct ThermalConfig {
    /// Thermal conductivity k in W/(m·K).
    ///
    /// Reference values (approximate):
    /// - Air:   0.025 W/(m·K)
    /// - Water: 0.6   W/(m·K)
    /// - Rock:  2.0   W/(m·K)
    /// - Steel: 50    W/(m·K)
    pub conductivity: f32,

    /// Specific heat capacity c_p in J/(kg·K).
    ///
    /// Reference values (approximate):
    /// - Air:   1005 J/(kg·K)
    /// - Water: 4182 J/(kg·K)
    /// - Rock:  840  J/(kg·K)
    /// - Steel: 490  J/(kg·K)
    pub heat_capacity: f32,

    /// Ambient/boundary temperature in K (or simulation-unit temperature).
    ///
    /// Grid cells with no particle mass (empty cells) are held at this temperature.
    /// Boundary cells equilibrate toward this value.
    pub ambient: f32,

    /// Grid cell physical size in meters (matches `SimConfig::grid_cell_size`).
    ///
    /// Used to convert conductivity/capacity into grid-unit diffusivity.
    pub grid_cell_size: f32,

    /// Newton cooling rate k_c in 1/s: dT/dt = −k_c·(T − ambient).
    ///
    /// Models convective or radiative heat loss to the environment.
    /// 0.0 = no cooling (default, adiabatic walls).
    pub cooling_rate: f32,
}

impl ThermalConfig {
    /// Thermal diffusivity α = k / (c_p · dx²) in grid-units²/s.
    ///
    /// Folding dx² in keeps the Laplacian formula dimensionless over grid indices.
    #[inline]
    pub fn alpha_grid(&self) -> f32 {
        // α = k / (c_p · dx²): units = m²/s / m² = 1/s (frequency in grid coords)
        self.conductivity / (self.heat_capacity * self.grid_cell_size * self.grid_cell_size)
    }
}

/// Grid-based Fourier heat diffusion.
///
/// Add to `Simulation` via `solver.with_thermal(ThermalDiffusion::new(config, grid_res))`.
/// Applied once per MPM substep, after force fields, before state projection.
pub struct ThermalDiffusion {
    pub config: ThermalConfig,
    grid_res: usize,
    // Preallocated scratch buffers — no per-substep heap allocation.
    grid_work: Vec<f32>, // dual-use: P2G scatter (Σ w·m·T), then Laplacian output (T_new)
    grid_mass: Vec<f32>, // Σ (w · mass) per cell
    grid_temp: Vec<f32>, // normalized T_old — needed for G2P delta (T_new − T_old)
}

impl ThermalDiffusion {
    pub fn new(config: ThermalConfig, grid_res: usize) -> Self {
        let n = grid_res * grid_res;
        Self {
            config,
            grid_res,
            grid_work: vec![0.0; n],
            grid_mass: vec![0.0; n],
            grid_temp: vec![0.0; n],
        }
    }

    /// Apply one thermal substep. Call from `Simulation::do_substep` after force fields.
    ///
    /// `sub_dt`: substep duration in seconds.
    pub fn apply(&mut self, particles: &mut Particles, sub_dt: f32) {
        let n = self.grid_res * self.grid_res;
        let res = self.grid_res as i32;

        // --- Clear scratch (grid_work = P2G scatter, grid_mass = weights) ---
        for i in 0..n {
            self.grid_work[i] = 0.0;
            self.grid_mass[i] = 0.0;
        }

        // --- P2G: scatter mass-weighted temperature into grid_work ---
        for pi in 0..particles.len() {
            let x = particles.x[pi];
            let mass = particles.mass[pi];
            let temperature = particles.temperature[pi];
            let w = quadratic_weights(x);
            for gx in 0i32..3 {
                for gy in 0i32..3 {
                    let weight = w.wx[gx as usize] * w.wy[gy as usize];
                    let cell = w.base_cell + IVec2::new(gx - 1, gy - 1);
                    if cell.x < 0 || cell.y < 0 || cell.x >= res || cell.y >= res {
                        continue;
                    }
                    let idx = (cell.x * res + cell.y) as usize;
                    let mw = weight * mass;
                    self.grid_work[idx] += mw * temperature;
                    self.grid_mass[idx] += mw;
                }
            }
        }

        // --- Normalize: grid_temp = T_old; empty cells = ambient ---
        // grid_work (P2G scatter) is discarded and reused for Laplacian output below.
        for i in 0..n {
            self.grid_temp[i] = if self.grid_mass[i] > 1e-10 {
                self.grid_work[i] / self.grid_mass[i]
            } else {
                self.config.ambient
            };
        }

        // --- Laplacian: explicit Euler FD, output into grid_work ---
        // grid_temp = T_old (read-only). grid_work = T_new (write).
        // Layout: column-major, idx = x * res + y — matches mechanics grid.
        // ∇²T ≈ T[x-1,y] + T[x+1,y] + T[x,y-1] + T[x,y+1] − 4·T[x,y]
        let alpha_dt = self.config.alpha_grid() * sub_dt;
        for x in 0..self.grid_res {
            for y in 0..self.grid_res {
                let c = x * self.grid_res + y;
                let t_c = self.grid_temp[c];
                let t_xm = if x > 0 {
                    self.grid_temp[c - self.grid_res]
                } else {
                    self.config.ambient
                };
                let t_xp = if x + 1 < self.grid_res {
                    self.grid_temp[c + self.grid_res]
                } else {
                    self.config.ambient
                };
                let t_ym = if y > 0 {
                    self.grid_temp[c - 1]
                } else {
                    self.config.ambient
                };
                let t_yp = if y + 1 < self.grid_res {
                    self.grid_temp[c + 1]
                } else {
                    self.config.ambient
                };
                let laplacian = t_xm + t_xp + t_ym + t_yp - 4.0 * t_c;
                self.grid_work[c] = t_c + alpha_dt * laplacian;
            }
        }

        // --- G2P: gather Δ T = (T_new − T_old) back to particles ---
        // Delta scatter preserves per-particle state at sparse/edge regions.
        // grid_work = T_new, grid_temp = T_old.
        for pi in 0..particles.len() {
            let x = particles.x[pi];
            let w = quadratic_weights(x);
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
                    delta += weight * (self.grid_work[idx] - self.grid_temp[idx]);
                    w_sum += weight;
                }
            }

            if w_sum > 1e-10 {
                particles.temperature[pi] += delta / w_sum;
            }
        }

        // Newton cooling: dT/dt = −k_c·(T − T_ambient).
        // Explicit Euler: T_new = T + sub_dt·(−k_c)·(T − ambient)
        //               = T·(1 − k_c·sub_dt) + k_c·sub_dt·ambient
        if self.config.cooling_rate > 0.0 {
            let decay = self.config.cooling_rate * sub_dt;
            let ambient = self.config.ambient;
            for pi in 0..particles.len() {
                particles.temperature[pi] += decay * (ambient - particles.temperature[pi]);
            }
        }
    }
}
