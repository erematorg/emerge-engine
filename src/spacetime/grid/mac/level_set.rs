//! Where the liquid is: a signed distance at the cell centres, negative in
//! the liquid, built from the particles alone.
//!
//! Zhu and Bridson, *Animating Sand as a Fluid*, section 5: the exact
//! distance of one particle, `|x - x0| - r0`, generalised by replacing the
//! particle with a kernel-weighted average of its neighbours' positions
//! and radii (their eqs. 6 to 10, kernel `k(s) = max(0, (1 - s^2)^3)`).
//! Their stated weakness: in concave regions the averaged position can
//! fall outside the liquid and leave a small spurious blob.

use glam::Vec2;

use super::field::{Field2, MacLayout};

/// The two lengths of the surface construction, in grid units.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceSettings {
    /// Radius given to every particle.
    pub radius: f32,
    /// Neighbourhood the average runs over.
    pub kernel_radius: f32,
}

impl SurfaceSettings {
    /// Zhu and Bridson's choice: every radius equal to the particle
    /// spacing, and a neighbourhood of twice the spacing.
    pub fn from_spacing(spacing: f32) -> Self {
        Self {
            radius: spacing,
            kernel_radius: 2.0 * spacing,
        }
    }
}

/// Signed distance to the liquid at every cell centre.
///
/// Where no particle lies within the kernel the average is undefined; the
/// value there is the single-particle formula for the nearest particle
/// within two kernel radii, the limit the average tends to as neighbours
/// thin out, and positive (air) beyond that.
pub fn liquid_phi(layout: &MacLayout, x: &[Vec2], surface: &SurfaceSettings) -> Field2 {
    let (nx, ny, dx) = (layout.nx, layout.ny, layout.dx);
    let bins = Bins::new(layout, x);
    let reach_len = 2.0 * surface.kernel_radius;
    let reach = (reach_len / dx).ceil() as i32 + 1;
    let r2_kernel = surface.kernel_radius * surface.kernel_radius;
    let mut phi = layout.cells(0.0);
    for j in 0..ny {
        for i in 0..nx {
            let centre = layout.cell_centre(i, j);
            let mut sum_w = 0.0f32;
            let mut sum_x = Vec2::ZERO;
            let mut nearest2 = f32::INFINITY;
            bins.for_each_near(i as i32, j as i32, reach, |p| {
                let d2 = (x[p] - centre).length_squared();
                nearest2 = nearest2.min(d2);
                if d2 < r2_kernel {
                    let k = (1.0 - d2 / r2_kernel).powi(3);
                    sum_w += k;
                    sum_x += k * x[p];
                }
            });
            let value = if sum_w > 0.0 {
                (centre - sum_x / sum_w).length() - surface.radius
            } else if nearest2 <= reach_len * reach_len {
                nearest2.sqrt() - surface.radius
            } else {
                reach_len - surface.radius
            };
            phi.set(i, j, value);
        }
    }
    phi
}

/// Particles sorted by the cell they sit in.
struct Bins {
    nx: i32,
    ny: i32,
    start: Vec<u32>,
    order: Vec<u32>,
}

impl Bins {
    fn new(layout: &MacLayout, x: &[Vec2]) -> Self {
        let (nx, ny) = (layout.nx as i32, layout.ny as i32);
        let cell = |q: Vec2| -> Option<usize> {
            let c = (q / layout.dx).floor().as_ivec2();
            (c.x >= 0 && c.y >= 0 && c.x < nx && c.y < ny).then(|| (c.x + nx * c.y) as usize)
        };
        let n = (nx * ny) as usize;
        let mut count = vec![0u32; n + 1];
        for &q in x {
            if let Some(k) = cell(q) {
                count[k + 1] += 1;
            }
        }
        for k in 0..n {
            count[k + 1] += count[k];
        }
        let start = count.clone();
        let mut fill = count;
        let mut order = vec![0u32; start[n] as usize];
        for (p, &q) in x.iter().enumerate() {
            if let Some(k) = cell(q) {
                order[fill[k] as usize] = p as u32;
                fill[k] += 1;
            }
        }
        Self {
            nx,
            ny,
            start,
            order,
        }
    }

    fn for_each_near(&self, i: i32, j: i32, reach: i32, mut f: impl FnMut(usize)) {
        for cj in (j - reach).max(0)..=(j + reach).min(self.ny - 1) {
            for ci in (i - reach).max(0)..=(i + reach).min(self.nx - 1) {
                let k = (ci + self.nx * cj) as usize;
                for &p in &self.order[self.start[k] as usize..self.start[k + 1] as usize] {
                    f(p as usize);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_particle_gives_its_own_distance() {
        let layout = MacLayout::new(16, 16, 1.0);
        let surface = SurfaceSettings::from_spacing(0.5);
        let x = [Vec2::new(8.2, 7.9)];
        let phi = liquid_phi(&layout, &x, &surface);
        // Inside the kernel, then between one and two kernel radii.
        for (i, j) in [(8, 7), (9, 8), (7, 6)] {
            let d = (layout.cell_centre(i, j) - x[0]).length();
            assert!((phi.get(i, j) - (d - 0.5)).abs() < 1e-5, "cell ({i},{j})");
        }
        // Far away it is only air.
        assert!(phi.get(12, 11) > 0.0);
    }

    /// A flat lattice seeded as Zhu and Bridson 4.2.1 do (the two rows
    /// within one cell of the surface moved to the spacing below it) puts
    /// the zero crossing within half a cell of the true surface, and every
    /// cell centre well inside is liquid.
    #[test]
    fn a_flat_lattice_puts_the_surface_near_its_true_height() {
        let layout = MacLayout::new(20, 20, 1.0);
        let s = 0.5;
        let surface = SurfaceSettings::from_spacing(s);
        let h = 10.0;
        let mut x = Vec::new();
        for i in 0..40 {
            let px = i as f32 * s + 0.25;
            x.push(Vec2::new(px, h - s));
            x.push(Vec2::new(px, h - s));
            let mut y = h - 1.25;
            while y > 0.0 {
                x.push(Vec2::new(px, y));
                y -= s;
            }
        }
        let phi = liquid_phi(&layout, &x, &surface);
        let column = 10;
        for j in 0..9 {
            assert!(phi.get(column, j) < 0.0, "cell {j} should be liquid");
        }
        let (below, above) = (phi.get(column, 9), phi.get(column, 10));
        assert!(below < 0.0 && above >= 0.0);
        let crossing = 9.5 + below / (below - above);
        assert!(
            (crossing - h).abs() < 0.5,
            "surface read at {crossing}, true height {h}"
        );
    }
}
