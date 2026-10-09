//! Storage for the staggered grid: one scalar array per sample lattice.

use glam::Vec2;

/// Size of a MAC grid: `nx` by `ny` cells of width `dx`, cell `(0, 0)`
/// at the origin of the engine's grid coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MacLayout {
    pub nx: usize,
    pub ny: usize,
    pub dx: f32,
}

impl MacLayout {
    pub fn new(nx: usize, ny: usize, dx: f32) -> Self {
        assert!(nx > 0 && ny > 0, "a MAC grid needs at least one cell");
        assert!(dx > 0.0, "cell width must be positive");
        Self { nx, ny, dx }
    }

    pub fn cell_centre(&self, i: usize, j: usize) -> Vec2 {
        Vec2::new(i as f32 + 0.5, j as f32 + 0.5) * self.dx
    }

    /// Where `u` of face `(i, j)` sits, on the left side of cell `(i, j)`.
    pub fn u_position(&self, i: usize, j: usize) -> Vec2 {
        Vec2::new(i as f32, j as f32 + 0.5) * self.dx
    }

    /// Where `v` of face `(i, j)` sits, on the bottom side of cell `(i, j)`.
    pub fn v_position(&self, i: usize, j: usize) -> Vec2 {
        Vec2::new(i as f32 + 0.5, j as f32) * self.dx
    }

    pub fn cells(&self, value: f32) -> Field2 {
        Field2::new(self.nx, self.ny, value)
    }

    pub fn corners(&self, value: f32) -> Field2 {
        Field2::new(self.nx + 1, self.ny + 1, value)
    }
}

/// A scalar per sample of one lattice, `i` running fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct Field2 {
    ni: usize,
    nj: usize,
    data: Vec<f32>,
}

impl Field2 {
    pub fn new(ni: usize, nj: usize, value: f32) -> Self {
        Self {
            ni,
            nj,
            data: vec![value; ni * nj],
        }
    }

    pub fn ni(&self) -> usize {
        self.ni
    }

    pub fn nj(&self) -> usize {
        self.nj
    }

    #[inline]
    pub fn index(&self, i: usize, j: usize) -> usize {
        debug_assert!(i < self.ni && j < self.nj);
        i + self.ni * j
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.data[self.index(i, j)]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        let k = self.index(i, j);
        self.data[k] = value;
    }

    #[inline]
    pub fn add(&mut self, i: usize, j: usize, value: f32) {
        let k = self.index(i, j);
        self.data[k] += value;
    }

    /// `None` outside the lattice.
    #[inline]
    pub fn get_signed(&self, i: i32, j: i32) -> Option<f32> {
        (i >= 0 && j >= 0 && (i as usize) < self.ni && (j as usize) < self.nj)
            .then(|| self.get(i as usize, j as usize))
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Bilinear value at `at`, in this lattice's index units, clamped to
    /// the lattice.
    pub fn bilinear(&self, at: Vec2) -> f32 {
        let (i, j, fx, fy) = self.cell_of(at);
        let (a, b) = (self.get(i, j), self.get(i + 1, j));
        let (c, d) = (self.get(i, j + 1), self.get(i + 1, j + 1));
        let bottom = a + (b - a) * fx;
        let top = c + (d - c) * fx;
        bottom + (top - bottom) * fy
    }

    /// Gradient of the bilinear interpolant at `at`, per index unit.
    pub fn bilinear_gradient(&self, at: Vec2) -> Vec2 {
        let (i, j, fx, fy) = self.cell_of(at);
        let (a, b) = (self.get(i, j), self.get(i + 1, j));
        let (c, d) = (self.get(i, j + 1), self.get(i + 1, j + 1));
        Vec2::new(
            (b - a) * (1.0 - fy) + (d - c) * fy,
            (c - a) * (1.0 - fx) + (d - b) * fx,
        )
    }

    fn cell_of(&self, at: Vec2) -> (usize, usize, f32, f32) {
        assert!(
            self.ni >= 2 && self.nj >= 2,
            "bilinear needs two samples per axis"
        );
        let x = at.x.clamp(0.0, (self.ni - 1) as f32);
        let y = at.y.clamp(0.0, (self.nj - 1) as f32);
        let i = (x.floor() as usize).min(self.ni - 2);
        let j = (y.floor() as usize).min(self.nj - 2);
        (i, j, x - i as f32, y - j as f32)
    }
}

/// Velocity on the faces: `u` on the `(nx + 1) x ny` faces normal to x,
/// `v` on the `nx x (ny + 1)` faces normal to y.
#[derive(Clone, Debug, PartialEq)]
pub struct MacVelocity {
    pub u: Field2,
    pub v: Field2,
}

impl MacVelocity {
    pub fn zeros(layout: &MacLayout) -> Self {
        Self {
            u: Field2::new(layout.nx + 1, layout.ny, 0.0),
            v: Field2::new(layout.nx, layout.ny + 1, 0.0),
        }
    }

    /// Velocity at `position` (grid coordinates), each component
    /// interpolated bilinearly on its own lattice.
    pub fn at(&self, layout: &MacLayout, position: Vec2) -> Vec2 {
        let p = position / layout.dx;
        Vec2::new(
            self.u.bilinear(Vec2::new(p.x, p.y - 0.5)),
            self.v.bilinear(Vec2::new(p.x - 0.5, p.y)),
        )
    }
}

/// One flag per face, indexed like the matching `MacVelocity` lattice.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceFlags {
    pub u: Vec<bool>,
    pub v: Vec<bool>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bilinear_reproduces_a_linear_field_and_its_gradient() {
        let mut f = Field2::new(4, 3, 0.0);
        for j in 0..3 {
            for i in 0..4 {
                f.set(i, j, 2.0 * i as f32 - 3.0 * j as f32 + 1.0);
            }
        }
        let at = Vec2::new(1.3, 0.6);
        assert!((f.bilinear(at) - (2.0 * 1.3 - 3.0 * 0.6 + 1.0)).abs() < 1e-5);
        assert!((f.bilinear_gradient(at) - Vec2::new(2.0, -3.0)).length() < 1e-5);
    }

    #[test]
    fn velocity_at_reads_each_component_on_its_own_lattice() {
        let layout = MacLayout::new(4, 4, 1.0);
        let mut vel = MacVelocity::zeros(&layout);
        for j in 0..4 {
            for i in 0..5 {
                let x = layout.u_position(i, j).x;
                vel.u.set(i, j, x);
            }
        }
        for j in 0..5 {
            for i in 0..4 {
                let y = layout.v_position(i, j).y;
                vel.v.set(i, j, -2.0 * y);
            }
        }
        let at = Vec2::new(1.7, 2.2);
        assert!((vel.at(&layout, at) - Vec2::new(1.7, -4.4)).length() < 1e-5);
    }
}
