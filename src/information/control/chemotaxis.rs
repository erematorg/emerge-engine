//! Chemotaxis: sensing a chemical gradient and the drift it calls for.
//!
//! Keller & Segel 1970, "Initiation of slime mold aggregation viewed as an
//! instability", population balance
//!
//!   d(rho)/dt = div(D grad(rho) - chi rho grad(phi))
//!
//! The chemotactic term is a flux `chi rho grad(phi)`: a population of density
//! `rho` drifting at the VELOCITY `v_c = chi grad(phi)`. It is not a force and
//! not an acceleration. An organism realises that drift with its own muscles
//! (the active stress of its material, in `matter`), so this module only
//! senses: it reports the local gradient and the drift velocity a controller
//! steers toward. It applies nothing to matter.
//!
//! The gradient is read with the derivative of the same quadratic B-spline
//! that scattered `phi` onto the grid (`grid::kernel`), so it is evaluated at
//! the organism's own position, with grid node `i` at `i + 0.5`.

use glam::Vec2;

use crate::grid::kernel::{axis_weights, axis_weights_derivative, quadratic_weights};
use crate::thermodynamics::ScalarDiffusionField;

/// A snapshot of a scalar concentration field `phi` (pheromone, nutrient,
/// morphogen) and the chemotactic sensitivity `chi` of the organism reading it.
#[derive(Debug, Clone)]
pub struct ChemotaxisSensor {
    /// Grid resolution, the same as the `ScalarDiffusionField` it reads.
    grid_res: usize,
    /// Snapshot of `phi` on the grid. Layout: `phi[x * grid_res + y]`.
    phi: Vec<f32>,
    /// Chemotactic sensitivity `chi`, in (cells/s) per (phi unit / cell):
    /// the drift speed per unit gradient. Positive climbs the gradient
    /// (attraction), negative descends it (repulsion).
    pub sensitivity: f32,
}

impl ChemotaxisSensor {
    pub fn new(grid_res: usize, sensitivity: f32) -> Self {
        Self {
            grid_res,
            phi: vec![0.0; grid_res * grid_res],
            sensitivity,
        }
    }

    /// Copy the current concentration from `source`. Call after `source.apply()`.
    pub fn sync_from(&mut self, source: &ScalarDiffusionField) {
        let src = source.current_phi();
        let n = self.grid_res * self.grid_res;
        debug_assert_eq!(src.len(), n, "ChemotaxisSensor grid_res mismatch");
        self.phi[..n].copy_from_slice(&src[..n]);
    }

    /// `grad(phi)` at `position` (grid units), from the derivative of the
    /// quadratic B-spline weights over the 3x3 stencil. Nodes outside the grid
    /// contribute nothing, as in the scatter that built `phi`.
    pub fn gradient_at(&self, position: Vec2) -> Vec2 {
        let w = quadratic_weights(position);
        let d = position - w.base_cell.as_vec2() - Vec2::splat(0.5);
        let dwx = axis_weights_derivative(d.x);
        let dwy = axis_weights_derivative(d.y);
        let wx = axis_weights(d.x);
        let wy = axis_weights(d.y);
        let res = self.grid_res as i32;
        let mut grad = Vec2::ZERO;
        for gx in 0..3 {
            for gy in 0..3 {
                let cx = w.base_cell.x + gx as i32 - 1;
                let cy = w.base_cell.y + gy as i32 - 1;
                if cx < 0 || cy < 0 || cx >= res || cy >= res {
                    continue;
                }
                let phi = self.phi[(cx * res + cy) as usize];
                grad.x += phi * dwx[gx] * wy[gy];
                grad.y += phi * wx[gx] * dwy[gy];
            }
        }
        grad
    }

    /// Keller-Segel drift velocity `chi grad(phi)` at `position`, in cells/s.
    pub fn drift_velocity(&self, position: Vec2) -> Vec2 {
        self.sensitivity * self.gradient_at(position)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RES: usize = 16;

    /// `phi` sampled at the node positions `i + 0.5` from `f(x, y)`.
    fn sensor_from(f: impl Fn(f32, f32) -> f32) -> ChemotaxisSensor {
        let mut sensor = ChemotaxisSensor::new(RES, 1.0);
        for x in 0..RES {
            for y in 0..RES {
                sensor.phi[x * RES + y] = f(x as f32 + 0.5, y as f32 + 0.5);
            }
        }
        sensor
    }

    /// A linear field is reproduced exactly by the quadratic B-spline away from
    /// the edges, so its gradient is exact at any position, not only at nodes.
    #[test]
    fn linear_field_gradient_is_exact_between_nodes() {
        let sensor = sensor_from(|x, y| 3.0 * x - 2.0 * y);
        for p in [
            Vec2::new(7.3, 8.9),
            Vec2::new(5.0, 5.0),
            Vec2::new(9.71, 6.02),
        ] {
            let g = sensor.gradient_at(p);
            assert!(
                (g - Vec2::new(3.0, -2.0)).length() < 1e-4,
                "gradient at {p} should be (3, -2), got {g}"
            );
        }
    }

    #[test]
    fn drift_velocity_is_sensitivity_times_gradient() {
        let mut sensor = sensor_from(|x, _| x);
        sensor.sensitivity = -0.5;
        let v = sensor.drift_velocity(Vec2::new(8.2, 8.2));
        assert!((v - Vec2::new(-0.5, 0.0)).length() < 1e-4, "got {v}");
    }

    #[test]
    fn sync_from_accepts_a_field_of_the_same_resolution() {
        use crate::particle::Particle;
        use crate::thermodynamics::ScalarDiffusionConfig;
        let scalar = ScalarDiffusionField::new(
            ScalarDiffusionConfig::default(),
            |p: &Particle| p.temperature,
            |p: &mut Particle, d: f32| p.temperature += d,
            RES,
        );
        let mut sensor = ChemotaxisSensor::new(RES, 1.0);
        sensor.sync_from(&scalar);
    }
}
