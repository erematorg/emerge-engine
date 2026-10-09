//! The pressure equations and the pressure update.
//!
//! One equation per cell whose centre is liquid and which has an open
//! face (Bridson and Muller-Fischer 4.3, with the face weights of 4.5.2):
//! the weighted divergence of the updated velocity is zero. A neighbour in
//! the air takes the ghost pressure that puts `p = 0` where the surface
//! cuts the line between the two centres, which adds `w dt / (dx^2 theta)`
//! to the diagonal instead of an off-diagonal term (their 4.5.1, eqs. 4.35
//! to 4.37). The same `theta` divides the pressure difference in the
//! update, so the two stay one consistent operator.
//!
//! Cells inside a solid but next to an open face are liquid in the level
//! set the caller passes (the solid distance is folded in, as `apic2d`
//! does); their pressures are unknowns like any other, which is how the
//! variational form holds `u . n = 0` at a wall (the notes, after 4.47).

use super::field::{FaceFlags, Field2, MacLayout, MacVelocity};
use super::solid::{FaceWeights, fraction_inside};

/// The five-point system, stored per cell as the notes store it: the
/// diagonal and the couplings to the `+i` and `+j` neighbours (the other
/// two are the neighbours' own). Symmetric positive (semi-)definite.
#[derive(Clone, Debug, PartialEq)]
pub struct PressureSystem {
    pub nx: usize,
    pub ny: usize,
    /// Cell has a pressure unknown.
    pub active: Vec<bool>,
    pub diag: Vec<f32>,
    /// Coupling of cell `(i, j)` to `(i + 1, j)`; zero unless both are
    /// active.
    pub plus_i: Vec<f32>,
    /// Coupling of cell `(i, j)` to `(i, j + 1)`.
    pub plus_j: Vec<f32>,
    /// Minus the weighted divergence of the velocity before the update, in
    /// 1/s.
    pub rhs: Vec<f32>,
    /// Cell centre in the air: where a neighbour's pressure is zero. What
    /// is neither active nor air is solid. Read only by the multigrid
    /// preconditioner, which coarsens these three kinds of cell.
    pub air: Vec<bool>,
    /// `dt / dx^2`, the coefficient of a fully open face.
    pub scale: f32,
    /// `1 / (c^2 dt)` added to every active diagonal by
    /// `add_compressibility`, zero for an incompressible liquid. The
    /// multigrid preconditioner adds it on its coarse levels too.
    pub compressibility: f32,
}

impl PressureSystem {
    pub fn unknowns(&self) -> usize {
        self.active.iter().filter(|&&a| a).count()
    }
}

/// Where the material is, as the equations and the update read it.
#[derive(Clone, Copy)]
pub struct Surface<'a> {
    /// Negative at the centre of a cell with a pressure unknown, the
    /// solids folded in; the free surface cuts the line to a neighbour at
    /// or above zero where this crosses zero (ghost fluid).
    pub phi: &'a Field2,
    /// `ProjectionSettings::theta_floor`.
    pub theta_floor: f32,
}

/// Builds the equations for `dt` from the velocity before the update, the
/// open fraction of the faces and where the material is.
pub fn assemble(
    layout: &MacLayout,
    dt: f32,
    vel: &MacVelocity,
    weights: &FaceWeights,
    surface: Surface<'_>,
) -> PressureSystem {
    let Surface {
        phi: liquid_phi,
        theta_floor,
    } = surface;
    let (nx, ny, dx) = (layout.nx, layout.ny, layout.dx);
    let n = nx * ny;
    let mut sys = PressureSystem {
        nx,
        ny,
        active: vec![false; n],
        diag: vec![0.0; n],
        plus_i: vec![0.0; n],
        plus_j: vec![0.0; n],
        rhs: vec![0.0; n],
        air: vec![false; n],
        scale: dt / (dx * dx),
        compressibility: 0.0,
    };
    let scale = sys.scale;
    for j in 0..ny {
        for i in 0..nx {
            let centre = liquid_phi.get(i, j);
            sys.air[i + nx * j] = centre >= 0.0;
            let faces = [
                // (open fraction, neighbour, face velocity, sign in div)
                (
                    weights.u.get(i + 1, j),
                    (i + 1, j),
                    vel.u.get(i + 1, j),
                    1.0,
                ),
                (
                    weights.u.get(i, j),
                    (i.wrapping_sub(1), j),
                    vel.u.get(i, j),
                    -1.0,
                ),
                (
                    weights.v.get(i, j + 1),
                    (i, j + 1),
                    vel.v.get(i, j + 1),
                    1.0,
                ),
                (
                    weights.v.get(i, j),
                    (i, j.wrapping_sub(1)),
                    vel.v.get(i, j),
                    -1.0,
                ),
            ];
            if centre >= 0.0 || faces.iter().all(|f| f.0 == 0.0) {
                continue;
            }
            let c = i + nx * j;
            sys.active[c] = true;
            for (k, &(w, (ni, nj), face_vel, sign)) in faces.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                // An open face always has a cell on both sides: the grid's
                // edge faces are closed.
                let neighbour = liquid_phi.get(ni, nj);
                let term = w * scale;
                if neighbour < 0.0 {
                    sys.diag[c] += term;
                    match k {
                        0 => sys.plus_i[c] = -term,
                        2 => sys.plus_j[c] = -term,
                        _ => {}
                    }
                } else {
                    let theta = fraction_inside(centre, neighbour).max(theta_floor);
                    sys.diag[c] += term / theta;
                }
                sys.rhs[c] -= sign * w * face_vel / dx;
            }
        }
    }
    sys
}

/// Turns the incompressible system into the compressible one of
/// Stomakhin, Schroeder, Jiang, Chai, Teran and Selle 2014 (eqs. 14 to 18):
/// `dp/dt = -K div v`, taken implicitly with the pressure update `v = v* -
/// dt grad q`, gives `div v* + A q = (q_n - q) / (c^2 dt)` in this
/// system's units (`q = p / rho`, `c^2 = K / rho`, `A q = -dt lap q`), so
/// every active row gains `1 / (c^2 dt)` on its diagonal and `q_n / (c^2
/// dt)` on its right-hand side. `q_n` is the cell's pressure before the
/// step. The system stays symmetric positive definite, and as `c` grows it
/// becomes the incompressible one.
///
/// Only cells of the material: `material[c]` false for a cell whose centre
/// lies in a solid. Those cells are unknowns only so that the wall's open
/// face carries no flux (module doc); made compressible, they let the
/// liquid flow into the wall, which the first gate run of this step
/// counted: 34 particles through the floor of the column in 5 s.
pub fn add_compressibility(
    sys: &mut PressureSystem,
    dt: f32,
    sound_speed2: f32,
    q_before: &[f32],
    material: &[bool],
) {
    let term = 1.0 / (sound_speed2 * dt);
    sys.compressibility = term;
    for c in 0..sys.active.len() {
        if sys.active[c] && material[c] {
            sys.diag[c] += term;
            sys.rhs[c] += term * q_before[c];
        }
    }
}

/// Subtracts `dt grad p` from every open face with liquid on at least one
/// side and marks it valid; closed faces are zeroed and, like faces with
/// air on both sides, left invalid for the extrapolation to fill.
pub fn apply_pressure(
    layout: &MacLayout,
    dt: f32,
    pressure: &[f32],
    weights: &FaceWeights,
    surface: Surface<'_>,
    vel: &mut MacVelocity,
) -> FaceFlags {
    let Surface {
        phi: liquid_phi,
        theta_floor,
    } = surface;
    let (nx, ny, dx) = (layout.nx, layout.ny, layout.dx);
    let p = |i: usize, j: usize| pressure[i + nx * j];
    let update = |phi_a: f32, phi_b: f32, p_a: f32, p_b: f32| -> Option<f32> {
        if phi_a >= 0.0 && phi_b >= 0.0 {
            return None;
        }
        let theta = if phi_a < 0.0 && phi_b < 0.0 {
            1.0
        } else {
            fraction_inside(phi_a, phi_b).max(theta_floor)
        };
        Some(dt * (p_b - p_a) / (dx * theta))
    };
    let mut valid = FaceFlags {
        u: vec![false; (nx + 1) * ny],
        v: vec![false; nx * (ny + 1)],
    };
    for j in 0..ny {
        for i in 0..=nx {
            let k = vel.u.index(i, j);
            if weights.u.get(i, j) == 0.0 {
                vel.u.set(i, j, 0.0);
                continue;
            }
            let (a, b) = ((i - 1, j), (i, j));
            let phi = (liquid_phi.get(a.0, a.1), liquid_phi.get(b.0, b.1));
            if let Some(delta) = update(phi.0, phi.1, p(a.0, a.1), p(b.0, b.1)) {
                vel.u.add(i, j, -delta);
                valid.u[k] = true;
            }
        }
    }
    for j in 0..=ny {
        for i in 0..nx {
            let k = vel.v.index(i, j);
            if weights.v.get(i, j) == 0.0 {
                vel.v.set(i, j, 0.0);
                continue;
            }
            let (a, b) = ((i, j - 1), (i, j));
            let phi = (liquid_phi.get(a.0, a.1), liquid_phi.get(b.0, b.1));
            if let Some(delta) = update(phi.0, phi.1, p(a.0, a.1), p(b.0, b.1)) {
                vel.v.add(i, j, -delta);
                valid.v[k] = true;
            }
        }
    }
    valid
}

/// Weighted divergence `(sum of w u . n) / dx` of every liquid cell, zero
/// elsewhere: what the equations drive to zero.
pub fn divergence(
    layout: &MacLayout,
    vel: &MacVelocity,
    weights: &FaceWeights,
    liquid_phi: &Field2,
) -> Field2 {
    let mut div = layout.cells(0.0);
    for j in 0..layout.ny {
        for i in 0..layout.nx {
            if liquid_phi.get(i, j) >= 0.0 {
                continue;
            }
            let flux = weights.u.get(i + 1, j) * vel.u.get(i + 1, j)
                - weights.u.get(i, j) * vel.u.get(i, j)
                + weights.v.get(i, j + 1) * vel.v.get(i, j + 1)
                - weights.v.get(i, j) * vel.v.get(i, j);
            div.set(i, j, flux / layout.dx);
        }
    }
    div
}

#[cfg(test)]
mod tests {
    /// The level set alone, uniform density.
    fn uniform(phi: &Field2) -> Surface<'_> {
        Surface {
            phi,
            theta_floor: THETA_FLOOR,
        }
    }

    use super::super::pcg::{SolverSettings, solve};
    use super::super::solid::{box_container, face_weights, sample_centres, sample_corners};
    use super::*;
    use glam::Vec2;

    const THETA_FLOOR: f32 = 0.01;

    /// A tank filled to a flat height with every face open, a level set
    /// that is exactly the distance to that surface, the solid folded in.
    fn tank(fill: f32) -> (MacLayout, FaceWeights, Field2) {
        let layout = MacLayout::new(12, 16, 1.0);
        let solid = box_container(Vec2::new(2.0, 2.0), Vec2::new(10.0, 15.0));
        let weights = face_weights(&layout, &sample_corners(&layout, solid));
        let solid_centres = sample_centres(&layout, solid);
        let mut phi = layout.cells(0.0);
        for j in 0..layout.ny {
            for i in 0..layout.nx {
                let liquid = layout.cell_centre(i, j).y - fill;
                phi.set(i, j, liquid.min(solid_centres.get(i, j)));
            }
        }
        (layout, weights, phi)
    }

    fn project(
        layout: &MacLayout,
        dt: f32,
        vel: &mut MacVelocity,
        weights: &FaceWeights,
        phi: &Field2,
    ) -> Vec<f32> {
        let sys = assemble(layout, dt, vel, weights, uniform(phi));
        let settings = SolverSettings {
            relative_tolerance: 1e-6,
            absolute_tolerance: 1e-5,
            max_iterations: 500,
            ..SolverSettings::default()
        };
        let solution = solve(&sys, &settings);
        assert!(solution.converged, "solve did not converge");
        apply_pressure(layout, dt, &solution.pressure, weights, uniform(phi), vel);
        solution.pressure
    }

    /// Bridson and Muller-Fischer, end of 4.5.2: fluid at rest under
    /// gravity with a flat surface is handled exactly, the hydrostatic
    /// pressure cancels the step's `g dt` and leaves zero velocity.
    #[test]
    fn hydrostatic_pressure_cancels_gravity_exactly() {
        let fill = 9.3;
        let (layout, weights, phi) = tank(fill);
        let (g, dt) = (981.0, 1.0 / 60.0);
        let mut vel = MacVelocity::zeros(&layout);
        for value in vel.v.data_mut() {
            *value = -g * dt;
        }
        let pressure = project(&layout, dt, &mut vel, &weights, &phi);
        for j in 0..layout.ny {
            for i in 0..=layout.nx {
                if weights.u.get(i, j) > 0.0 {
                    assert!(vel.u.get(i, j).abs() < 1e-3, "u({i},{j})");
                }
            }
        }
        for j in 0..=layout.ny {
            for i in 0..layout.nx {
                let wet =
                    j > 0 && (phi.get(i, j - 1) < 0.0 || (j < layout.ny && phi.get(i, j) < 0.0));
                if weights.v.get(i, j) > 0.0 && wet {
                    let v = vel.v.get(i, j);
                    assert!(v.abs() < 1e-2 * g * dt, "v({i},{j}) = {v}");
                }
            }
        }
        // p = g (h - y) at the fluid centres, in kinematic units.
        for j in 2..9 {
            let y = layout.cell_centre(5, j).y;
            let p = pressure[5 + layout.nx * j];
            assert!((p - g * (fill - y)).abs() < 1e-3 * g, "p at y={y}: {p}");
        }
    }

    /// The update is a projection: once applied, a second one changes
    /// nothing, and the weighted divergence of every liquid cell is gone.
    #[test]
    fn projecting_twice_changes_nothing() {
        let (layout, weights, phi) = tank(11.6);
        let dt = 0.01;
        let mut vel = MacVelocity::zeros(&layout);
        let mut seed = 12345u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed as f32 / u32::MAX as f32 - 0.5
        };
        for value in vel.u.data_mut() {
            *value = 40.0 * next();
        }
        for value in vel.v.data_mut() {
            *value = 40.0 * next();
        }
        project(&layout, dt, &mut vel, &weights, &phi);
        let div = divergence(&layout, &vel, &weights, &phi);
        let solid = box_container(Vec2::new(2.0, 2.0), Vec2::new(10.0, 15.0));
        let solid_centres = sample_centres(&layout, solid);
        for j in 0..layout.ny {
            for i in 0..layout.nx {
                if solid_centres.get(i, j) > 0.0 {
                    assert!(div.get(i, j).abs() < 1e-3, "div({i},{j})");
                }
            }
        }
        let once = vel.clone();
        project(&layout, dt, &mut vel, &weights, &phi);
        for (a, b) in once.u.data().iter().zip(vel.u.data()) {
            assert!((a - b).abs() < 1e-3);
        }
        for (a, b) in once.v.data().iter().zip(vel.v.data()) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    #[test]
    fn the_matrix_is_symmetric() {
        let (layout, weights, phi) = tank(8.4);
        let vel = MacVelocity::zeros(&layout);
        let sys = assemble(&layout, 0.02, &vel, &weights, uniform(&phi));
        for j in 0..layout.ny {
            for i in 0..layout.nx {
                let c = i + layout.nx * j;
                if sys.plus_i[c] != 0.0 {
                    assert!(sys.active[c] && sys.active[c + 1]);
                }
                if sys.plus_j[c] != 0.0 {
                    assert!(sys.active[c] && sys.active[c + layout.nx]);
                }
                if sys.active[c] {
                    assert!(sys.diag[c] > 0.0);
                }
            }
        }
    }
}
