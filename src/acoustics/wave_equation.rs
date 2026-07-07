//! 2D finite-difference wave equation solver.
//!
//! Solves ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²) using explicit Euler time integration.
//!
//! # Applications in LP
//! - Pressure waves through terrain (seismic, explosions)
//! - Sound propagation in fluid/gas particles
//! - Electromagnetic wave fields (year 2)
//!
//! # Stability
//! Check `is_stable()` before running — the Courant condition c·dt·√(1/dx² + 1/dy²) ≤ 1
//! must hold or the simulation will diverge (Courant, Friedrichs & Lewy 1928,
//! "Über die partiellen Differenzengleichungen der mathematischen Physik" —
//! the original real derivation of this stability bound for explicit FD wave
//! schemes, still the standard reference cited for it today).
//!
//! # Reference
//! Ported from `crates/energy/src/waves/wave_equation.rs`.
//! Cache-blocking (32×32 tiles) retained for performance.

/// 2D wave equation solver on a rectangular grid.
///
/// Grid layout: row-major `u[y * nx + x]`.
#[derive(Debug, Clone)]
pub struct WaveEquation2D {
    /// Grid width (x dimension).
    pub nx: usize,
    /// Grid height (y dimension).
    pub ny: usize,
    /// Wave speed c (grid units / second).
    pub c: f32,
    /// Grid spacing in x.
    pub dx: f32,
    /// Grid spacing in y.
    pub dy: f32,
    /// Time step.
    pub dt: f32,
    /// Current field u(t).
    pub u_current: Vec<f32>,
    /// Previous field u(t-dt).
    pub u_previous: Vec<f32>,
    // Pre-computed Courant coefficients.
    cx: f32,
    cy: f32,
}

impl WaveEquation2D {
    /// Create a new solver, zero-initialized.
    pub fn new(nx: usize, ny: usize, c: f32, dx: f32, dy: f32, dt: f32) -> Self {
        Self {
            nx,
            ny,
            c,
            dx,
            dy,
            dt,
            u_current: vec![0.0; nx * ny],
            u_previous: vec![0.0; nx * ny],
            cx: (c * dt / dx).powi(2),
            cy: (c * dt / dy).powi(2),
        }
    }

    /// Returns `true` if the Courant condition is satisfied (stable integration).
    pub fn is_stable(&self) -> bool {
        let courant = self.c * self.dt * ((1.0 / self.dx).powi(2) + (1.0 / self.dy).powi(2)).sqrt();
        courant <= 1.0
    }

    /// Set initial displacement u(0) = u(-dt) = `u0`.
    pub fn set_initial_displacement(&mut self, u0: Vec<f32>) {
        assert_eq!(u0.len(), self.nx * self.ny);
        self.u_previous = u0.clone();
        self.u_current = u0;
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.nx + x
    }

    /// Advance one time step. Boundary conditions: u = 0 (Dirichlet).
    pub fn step(&mut self) {
        let mut u_next = vec![0.0_f32; self.nx * self.ny];
        let cx = self.cx;
        let cy = self.cy;

        // Interior points — cache-blocked for better L1 usage.
        const TILE: usize = 32;
        let rows = self.ny.saturating_sub(2);
        let cols = self.nx.saturating_sub(2);
        if rows > 0 && cols > 0 {
            for jt in (0..rows).step_by(TILE) {
                for it in (0..cols).step_by(TILE) {
                    for j in jt..(jt + TILE).min(rows) {
                        let y = j + 1;
                        for i in it..(it + TILE).min(cols) {
                            let x = i + 1;
                            let cur = self.u_current[self.idx(x, y)];
                            let lap_x = self.u_current[self.idx(x + 1, y)] - 2.0 * cur
                                + self.u_current[self.idx(x - 1, y)];
                            let lap_y = self.u_current[self.idx(x, y + 1)] - 2.0 * cur
                                + self.u_current[self.idx(x, y - 1)];
                            u_next[self.idx(x, y)] = 2.0 * cur - self.u_previous[self.idx(x, y)]
                                + cx * lap_x
                                + cy * lap_y;
                        }
                    }
                }
            }
        }

        // Dirichlet boundary (u = 0) — already zero from vec! initialisation.
        self.u_previous = std::mem::replace(&mut self.u_current, u_next);
    }

    /// Sample the current field at grid point (x, y).
    pub fn get(&self, x: usize, y: usize) -> f32 {
        self.u_current[self.idx(x, y)]
    }

    /// Set a value in the current field (e.g. to inject a wave source).
    pub fn set(&mut self, x: usize, y: usize, value: f32) {
        let idx = self.idx(x, y);
        self.u_current[idx] = value;
    }
}
