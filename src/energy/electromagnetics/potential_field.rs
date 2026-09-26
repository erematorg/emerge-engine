//! Real electric potential field on a 2D grid, solved by relaxing Laplace's
//! equation (Gauss's law with zero charge density in air, ∇²φ = 0) via
//! Jacobi iteration -- the same numerical family `ScalarDiffusionField`'s
//! own diffusion already uses (Laplace is that same PDE's zero-source,
//! steady-state limit: `∂φ/∂t = D∇²φ` settles to `∇²φ = 0`), but a pure
//! GRID field with no particle coupling -- the potential between a cloud
//! and the ground doesn't live on any particle, unlike temperature or
//! moisture.
//!
//! Generic, not lightning-specific: this is a real, reusable steady-state
//! potential-field solver, usable for any problem needing one (a first real
//! application is dielectric-breakdown / lightning-leader growth, see
//! `super::leader`, since the leader grows toward the strongest local
//! field, `-∇φ`).
//!
//! Dirichlet boundaries top/bottom (fixed potential each -- e.g. cloud=0,
//! ground=1), Neumann (zero-gradient, copy-nearest) on the left/right sides
//! -- an open lateral domain. Same convention the real reference
//! implementation this was checked against uses
//! (github.com/diluuuu10/triggered-discharge, cloned in `tmp/` -- Jacobi
//! relaxation for the potential field, confirmed via that repo's own code).

use glam::Vec2;

#[derive(Clone)]
pub struct ElectricPotentialField {
    width: usize,
    height: usize,
    phi: Vec<f32>,
    phi_next: Vec<f32>,
    /// Real, generalized Dirichlet overlay: `Some(v)` pins a cell at
    /// potential `v` every relaxation sweep, `None` leaves it free. The
    /// top/bottom rows start pinned this way at construction -- but this is
    /// the SAME mechanism a growing conductor (a dielectric-breakdown
    /// leader channel) uses to extend the boundary as it grows (see `pin`'s
    /// own doc): a real conductor is an equipotential, so once a cell joins
    /// the channel, the field around it must be solved as if THAT cell were
    /// also a fixed-potential electrode, which is the actual physical
    /// mechanism (field concentration ahead of a growing conductive tip)
    /// that makes a leader grow roughly toward its own origin's field
    /// direction instead of uniformly at random.
    fixed: Vec<Option<f32>>,
}

impl ElectricPotentialField {
    /// Real initial guess: linear interpolation from `top_value` to
    /// `bottom_value`. Not required for correctness -- Jacobi relaxation
    /// converges from any starting state, including all-zero -- but a
    /// starting guess already close to the true solution converges in far
    /// fewer iterations than starting flat, the same real reason a good
    /// initial guess matters for any iterative solver.
    pub fn new(width: usize, height: usize, top_value: f32, bottom_value: f32) -> Self {
        assert!(
            width > 0 && height > 0,
            "ElectricPotentialField: grid dimensions must be positive"
        );
        let denom = (height.max(2) - 1) as f32;
        let mut phi = vec![0.0_f32; width * height];
        let mut fixed = vec![None; width * height];
        for (y, row) in phi.chunks_mut(width).enumerate() {
            let t = y as f32 / denom;
            let v = top_value + (bottom_value - top_value) * t;
            row.fill(v);
        }
        for x in 0..width {
            fixed[x] = Some(top_value);
            fixed[(height - 1) * width + x] = Some(bottom_value);
        }
        let phi_next = phi.clone();
        Self {
            width,
            height,
            phi,
            phi_next,
            fixed,
        }
    }

    /// Pin a single cell at a fixed potential, extending the Dirichlet
    /// boundary to include it -- the real mechanism a growing dielectric-
    /// breakdown leader uses to make itself part of the conducting
    /// boundary once it joins the channel (see the `fixed` field's own
    /// doc). Takes effect starting from the NEXT `relax_step` call.
    pub fn pin(&mut self, x: usize, y: usize, value: f32) {
        let i = self.idx(x, y);
        self.fixed[i] = Some(value);
        self.phi[i] = value;
    }

    #[inline]
    fn idx(&self, x: usize, y: usize) -> usize {
        y * self.width + x
    }

    pub fn width(&self) -> usize {
        self.width
    }
    pub fn height(&self) -> usize {
        self.height
    }

    /// One Jacobi relaxation sweep: interior cells become the average of
    /// their 4 neighbors (the real discretization of `∇²φ=0` -- see
    /// `ScalarDiffusionField`'s own doc for the identical Laplacian finite-
    /// difference form, just without the diffusion coefficient/dt scaling
    /// since this solves the steady state directly rather than stepping
    /// toward it in time). Top/bottom rows stay pinned at the Dirichlet
    /// boundary values every sweep. Left/right columns use their own
    /// nearest interior neighbor twice in the average, the standard finite-
    /// difference form of a zero-gradient (Neumann) boundary.
    pub fn relax_step(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let i = self.idx(x, y);
                if let Some(v) = self.fixed[i] {
                    self.phi_next[i] = v;
                    continue;
                }
                let left = if x == 0 {
                    self.phi[self.idx(x + 1, y)]
                } else {
                    self.phi[self.idx(x - 1, y)]
                };
                let right = if x == self.width - 1 {
                    self.phi[self.idx(x - 1, y)]
                } else {
                    self.phi[self.idx(x + 1, y)]
                };
                // Up/down: with pinned interior cells now possible (a
                // channel cell may sit at y=0 or y=height-1's own row is
                // already fully fixed, so this only matters for a channel
                // reaching an interior row), the vertical neighbors always
                // exist for 0<y<height-1; a channel cell can only be
                // interior, never on the fixed top/bottom rows themselves.
                let up = self.phi[self.idx(x, y - 1)];
                let down = self.phi[self.idx(x, y + 1)];
                self.phi_next[i] = 0.25 * (left + right + up + down);
            }
        }
        std::mem::swap(&mut self.phi, &mut self.phi_next);
    }

    /// Repeated relaxation -- see `relax_step`'s own doc. Real convergence
    /// rate for Jacobi iteration on a Laplace grid is `O(N^2)` sweeps for an
    /// `N`-cell-tall domain (no acceleration here, e.g. no red-black
    /// Gauss-Seidel or multigrid -- a real, disclosed simplification,
    /// matching the reference implementation's own plain Jacobi approach).
    pub fn relax_n(&mut self, iterations: usize) {
        for _ in 0..iterations {
            self.relax_step();
        }
    }

    pub fn phi_at(&self, x: usize, y: usize) -> f32 {
        self.phi[self.idx(x, y)]
    }

    /// Real electric field `E = -∇φ`, central difference (one-sided at
    /// domain edges). This is the actual physics the ball-on-a-hill
    /// analogy describes: the field points DOWNHILL (toward lower
    /// potential), with magnitude equal to the slope.
    pub fn field_at(&self, x: usize, y: usize) -> Vec2 {
        let xl = x.saturating_sub(1);
        let xr = (x + 1).min(self.width - 1);
        let yu = y.saturating_sub(1);
        let yd = (y + 1).min(self.height - 1);
        let dx = (xr - xl).max(1) as f32;
        let dy = (yd - yu).max(1) as f32;
        let dphidx = (self.phi_at(xr, y) - self.phi_at(xl, y)) / dx;
        let dphidy = (self.phi_at(x, yd) - self.phi_at(x, yu)) / dy;
        Vec2::new(-dphidx, -dphidy)
    }
}

#[cfg(test)]
mod electric_potential_field_tests {
    use super::*;

    /// Real, closed-form check: with uniform Dirichlet top/bottom and
    /// Neumann sides, and no x-varying feature anywhere in the domain, the
    /// EXACT analytic solution to Laplace's equation has no x-dependence at
    /// all -- it's the 1D linear interpolation `new()` already initializes
    /// to. That means this starting state is already the fixed point of
    /// the Jacobi iteration: relaxing it further should change nothing
    /// (within float tolerance), confirming `relax_step` doesn't introduce
    /// a bug that perturbs an already-correct state.
    #[test]
    fn uniform_case_is_a_fixed_point_of_relaxation() {
        let mut field = ElectricPotentialField::new(16, 32, 0.0, 1.0);
        let before: Vec<f32> = (0..field.height()).map(|y| field.phi_at(8, y)).collect();
        field.relax_n(50);
        for (y, &b) in before.iter().enumerate() {
            let after = field.phi_at(8, y);
            assert!(
                (after - b).abs() < 1.0e-5,
                "uniform linear-gradient state should be a fixed point of Jacobi \
                 relaxation (no x-variation anywhere to disturb it), but row {y} moved \
                 from {b} to {after}"
            );
        }
    }

    /// Real convergence check, starting from a WRONG initial guess (flat
    /// zero, not the linear interpolation `new()` normally starts from):
    /// after enough relaxation, the field must still converge to the same
    /// real analytic solution -- a uniform field pointing from high to low
    /// potential, magnitude `(bottom-top)/(height-1)`, zero in x. This is
    /// the textbook parallel-plate capacitor field solution.
    #[test]
    fn converges_to_uniform_field_from_a_flat_start() {
        const WIDTH: usize = 16;
        const HEIGHT: usize = 40;
        const TOP: f32 = 0.0;
        const BOTTOM: f32 = 1.0;
        let mut field = ElectricPotentialField::new(WIDTH, HEIGHT, TOP, BOTTOM);
        // Deliberately wipe the smart initial guess back to flat zero
        // (except the Dirichlet boundaries, which relax_step re-pins every
        // sweep regardless) to test real convergence, not just confirm the
        // constructor's own initial guess was already right.
        for y in 1..HEIGHT - 1 {
            for x in 0..WIDTH {
                let i = y * WIDTH + x;
                field.phi[i] = 0.0;
                field.phi_next[i] = 0.0;
            }
        }
        // O(N^2) plain-Jacobi convergence -- see relax_n's own doc.
        field.relax_n(HEIGHT * HEIGHT * 4);

        let expected_field_y = -(BOTTOM - TOP) / (HEIGHT as f32 - 1.0);
        for y in 1..HEIGHT - 1 {
            for x in 1..WIDTH - 1 {
                let e = field.field_at(x, y);
                assert!(
                    (e.y - expected_field_y).abs() < 0.01,
                    "expected uniform E.y={expected_field_y:.5} at ({x},{y}), got {:.5}",
                    e.y
                );
                assert!(
                    e.x.abs() < 0.01,
                    "expected zero E.x (no x-variation in a uniform parallel-plate \
                     setup) at ({x},{y}), got {:.5}",
                    e.x
                );
            }
        }
    }
}
