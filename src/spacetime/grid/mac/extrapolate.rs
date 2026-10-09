//! After the update only the faces touching liquid hold a projected
//! velocity. The gather reads a little beyond them. A face there that the
//! particles reached keeps its own velocity, with the pressure correction
//! of the faces next to it carried over (`keep_reached_faces`); the rest
//! take the velocity extended outward (Bridson and Muller-Fischer 6.3),
//! layer by layer as the average of the valid neighbours (`apic2d`'s
//! `extrapolate`).
//!
//! Faces closed by a solid then take the mirror image of the flow across
//! the wall: the velocity at the point as far outside the wall as the face
//! is inside it, with its component along the wall's normal reversed and
//! its component along the wall kept. `apic2d`'s `constrain_velocity`
//! keeps only the part along the wall instead; with that, the gather's
//! quadratic kernel still reads an eighth of the next face's normal
//! velocity at the wall itself, and in the first gate run every one of 474
//! wall crossings started within 0.154 cell of a wall. The mirror makes the
//! normal velocity read at a still, flat wall exactly zero, and since the
//! mirror image of a divergence-free flow is divergence free, the cells
//! inside the wall keep zero divergence too. Derived here from those two
//! conditions; the sources read push particles back out of the wall after
//! they move instead (Zhu and Bridson 4.2.5, `apic2d`), which the gates do
//! not allow. The solid supplies the image (`solid::box_container_image`):
//! in a corner it is the reflection across both walls.

use glam::Vec2;

use super::field::{FaceFlags, Field2, MacLayout, MacVelocity};
use super::solid::{FaceWeights, fraction_inside};
use super::transfer::FaceMass;

/// Extends `field` over `layers` rings of invalid samples; each new sample
/// is the mean of its valid neighbours from the ring before.
pub fn extrapolate(field: &mut Field2, valid: &mut [bool], layers: u32) {
    let (ni, nj) = (field.ni() as i32, field.nj() as i32);
    for _ in 0..layers {
        let mut filled = Vec::new();
        for j in 0..nj {
            for i in 0..ni {
                let k = field.index(i as usize, j as usize);
                if valid[k] {
                    continue;
                }
                let mut sum = 0.0;
                let mut count = 0;
                for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (a, b) = (i + di, j + dj);
                    if a < 0 || b < 0 || a >= ni || b >= nj {
                        continue;
                    }
                    let n = field.index(a as usize, b as usize);
                    if valid[n] {
                        sum += field.data()[n];
                        count += 1;
                    }
                }
                if count > 0 {
                    filled.push((k, sum / count as f32));
                }
            }
        }
        if filled.is_empty() {
            return;
        }
        for (k, value) in filled {
            field.data_mut()[k] = value;
            valid[k] = true;
        }
    }
}

/// Extrapolates both components.
pub fn extrapolate_velocity(vel: &mut MacVelocity, valid: &mut FaceFlags, layers: u32) {
    extrapolate(&mut vel.u, &mut valid.u, layers);
    extrapolate(&mut vel.v, &mut valid.v, layers);
}

/// The pressure continued past the free surface, per cell: the solved
/// pressure in the material, and in the air within `layers` cells of it
/// the ghost fluid's own straight line (Bridson and Muller-Fischer 4.5.1)
/// carried on: along each grid axis from the last material cell `a`
/// through the point where the surface cuts the line (`theta` of the way
/// to the next centre, `p = 0` there), so the `k`-th air cell reads `p_a
/// (1 - k / theta)`, averaged over the axes that reach material. Cells in
/// a solid are never a source: their unknowns only hold `u . n = 0` at the
/// wall (`pressure` module doc).
pub struct GhostPressure {
    nx: usize,
    value: Vec<Option<f32>>,
    /// `dt / dx`, so a face's correction is `-scale (p_b - p_a)`.
    scale: f32,
}

/// Builds `GhostPressure` from the level set with the solids folded in,
/// the solved pressure, which cells have their centre outside every solid
/// (`open`), and the ghost fluid's floor on `theta`.
pub fn ghost_pressure(
    layout: &MacLayout,
    dt: f32,
    phi: &Field2,
    pressure: &[f32],
    open: &[bool],
    (theta_floor, layers): (f32, u32),
) -> GhostPressure {
    let (nx, ny) = (layout.nx as i32, layout.ny as i32);
    let at = |i: i32, j: i32| (i + nx * j) as usize;
    let inside = |i: i32, j: i32| i >= 0 && j >= 0 && i < nx && j < ny;
    let material =
        |i: i32, j: i32| inside(i, j) && open[at(i, j)] && phi.get(i as usize, j as usize) < 0.0;
    let air =
        |i: i32, j: i32| inside(i, j) && open[at(i, j)] && phi.get(i as usize, j as usize) >= 0.0;
    let mut value = vec![None; (nx * ny) as usize];
    for j in 0..ny {
        for i in 0..nx {
            if material(i, j) {
                value[at(i, j)] = Some(pressure[at(i, j)]);
                continue;
            }
            if !air(i, j) {
                continue;
            }
            let (mut sum, mut count) = (0.0f32, 0u32);
            for (di, dj) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                // Walk towards the material along this axis, through air.
                for k in 1..=layers as i32 {
                    let (ai, aj) = (i + di * k, j + dj * k);
                    if material(ai, aj) {
                        let (bi, bj) = (ai - di, aj - dj);
                        let theta = fraction_inside(
                            phi.get(ai as usize, aj as usize),
                            phi.get(bi as usize, bj as usize),
                        )
                        .max(theta_floor);
                        sum += pressure[at(ai, aj)] * (1.0 - k as f32 / theta);
                        count += 1;
                        break;
                    }
                    if !air(ai, aj) {
                        break;
                    }
                }
            }
            if count > 0 {
                value[at(i, j)] = Some(sum / count as f32);
            }
        }
    }
    GhostPressure {
        nx: layout.nx,
        value,
        scale: dt / layout.dx,
    }
}

impl GhostPressure {
    /// The correction `-dt (p_b - p_a) / dx` across a face between cells
    /// `a` and `b`, when both have a pressure.
    fn correction(&self, a: (usize, usize), b: (usize, usize)) -> Option<f32> {
        let p = |(i, j): (usize, usize)| self.value[i + self.nx * j];
        Some(-self.scale * (p(b)? - p(a)?))
    }
}

/// Every open face the particles reached (`mass > 0`) outside the
/// projected ones keeps the velocity it held before the projection,
/// `before`, plus the correction `-dt grad p` of the pressure continued
/// past the surface (`GhostPressure`); a face beyond that continuation
/// gets none, as a particle alone in the air feels no pressure. Such faces
/// become valid.
///
/// Why not the velocity extension alone: it is constant along the normal,
/// so it replaces the velocity the particles gave these faces with their
/// neighbours' and loses its gradient across the surface. A rigid rotation
/// read at the edge of a block then shows a strain rate: measured at a
/// block spinning at 2 rad/s, the edge particle's `du/dy` fell from the
/// exact -2 after the transfer to 0.003 after the extension, which an
/// elastic solid turns into stress. Keeping the face's own velocity without
/// any correction is wrong the other way: at a resting surface the pressure
/// gradient balances gravity up to the surface itself, where `p` vanishes
/// but its gradient does not; without it a liquid column's surface
/// particles moved up to 5.8 cells. A first version carried the
/// neighbouring faces' corrections over instead of the pressure: at a
/// solid's corner against a wall it copied the wall's impact reaction onto
/// faces that do not bear it, and the corner particle's velocity gradient
/// jumped from 85 to 712 1/s in one substep, then NaN. The continued
/// pressure is exact for a pressure linear across the surface (the resting
/// column) and gives each face the gradient of its own line. Derived here
/// from the ghost fluid's construction; no source read gives it.
pub fn keep_reached_faces(
    vel: &mut MacVelocity,
    before: &MacVelocity,
    valid: &mut FaceFlags,
    mass: &FaceMass,
    weights: &FaceWeights,
    ghost: &GhostPressure,
) {
    for j in 0..vel.u.nj() {
        for i in 0..vel.u.ni() {
            let k = vel.u.index(i, j);
            if valid.u[k] || mass.u.get(i, j) <= 0.0 || weights.u.get(i, j) <= 0.0 {
                continue;
            }
            let extra = ghost.correction((i - 1, j), (i, j)).unwrap_or(0.0);
            vel.u.set(i, j, before.u.get(i, j) + extra);
            valid.u[k] = true;
        }
    }
    for j in 0..vel.v.nj() {
        for i in 0..vel.v.ni() {
            let k = vel.v.index(i, j);
            if valid.v[k] || mass.v.get(i, j) <= 0.0 || weights.v.get(i, j) <= 0.0 {
                continue;
            }
            let extra = ghost.correction((i, j - 1), (i, j)).unwrap_or(0.0);
            vel.v.set(i, j, before.v.get(i, j) + extra);
            valid.v[k] = true;
        }
    }
}

/// On every closed face, the mirror image of the flow across the wall
/// (module doc): `image` gives, for a point in the solid, the point it
/// mirrors and the sign of each velocity component there. On the grid's
/// own edge, the edge is the wall and the component normal to it is zero.
pub fn constrain_to_solids(
    layout: &MacLayout,
    vel: &mut MacVelocity,
    weights: &FaceWeights,
    image: &dyn Fn(Vec2) -> (Vec2, Vec2),
) {
    let before = vel.clone();
    let mirror = |position: Vec2| -> Vec2 {
        let (at, sign) = image(position);
        before.at(layout, at) * sign
    };
    for j in 0..layout.ny {
        for i in 0..=layout.nx {
            if weights.u.get(i, j) > 0.0 {
                continue;
            }
            let value = if i == 0 || i == layout.nx {
                0.0
            } else {
                mirror(layout.u_position(i, j)).x
            };
            vel.u.set(i, j, value);
        }
    }
    for j in 0..=layout.ny {
        for i in 0..layout.nx {
            if weights.v.get(i, j) > 0.0 {
                continue;
            }
            let value = if j == 0 || j == layout.ny {
                0.0
            } else {
                mirror(layout.v_position(i, j)).y
            };
            vel.v.set(i, j, value);
        }
    }
}

/// The adjoint of `constrain_to_solids`: every closed face's value, times
/// the sign of its component in the image (`signed`), is added to the
/// faces around the point it mirrors, with the bilinear weights the mirror
/// reads them with, and the closed face is cleared. The mirror stands for
/// an image body across the wall; the gather reads that body's velocity,
/// so the forces it would exert, and the mass it would lend the faces near
/// the wall, are what the particles' own forces and masses on the closed
/// faces become under the same reflection. Dropping them, as the faces are
/// then overwritten, makes the gather and the forces no longer adjoint
/// next to a wall: in step A2 an elastic block hitting a wall read a
/// velocity gradient of 500 to 1800 1/s in its first layer, sheared 9 to 1
/// and blew up, while two identical blocks hitting each other with no wall
/// near bounced apart. Derived here from the image construction; no
/// source read gives it.
pub fn fold_onto_images(
    layout: &MacLayout,
    field: &mut MacVelocity,
    weights: &FaceWeights,
    image: &dyn Fn(Vec2) -> (Vec2, Vec2),
    signed: bool,
) {
    let spread = |lattice: &mut Field2, at: Vec2, value: f32| {
        let (i0, j0) = (at.x.floor(), at.y.floor());
        let (fx, fy) = (at.x - i0, at.y - j0);
        for (di, dj, w) in [
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ] {
            let (i, j) = (i0 as i32 + di, j0 as i32 + dj);
            if w > 0.0
                && i >= 0
                && j >= 0
                && (i as usize) < lattice.ni()
                && (j as usize) < lattice.nj()
            {
                lattice.add(i as usize, j as usize, w * value);
            }
        }
    };
    for j in 0..layout.ny {
        for i in 1..layout.nx {
            let value = field.u.get(i, j);
            if weights.u.get(i, j) > 0.0 || value == 0.0 {
                continue;
            }
            let (at, sign) = image(layout.u_position(i, j));
            let p = at / layout.dx;
            field.u.set(i, j, 0.0);
            let factor = if signed { sign.x } else { 1.0 };
            spread(&mut field.u, Vec2::new(p.x, p.y - 0.5), factor * value);
        }
    }
    for j in 1..layout.ny {
        for i in 0..layout.nx {
            let value = field.v.get(i, j);
            if weights.v.get(i, j) > 0.0 || value == 0.0 {
                continue;
            }
            let (at, sign) = image(layout.v_position(i, j));
            let p = at / layout.dx;
            field.v.set(i, j, 0.0);
            let factor = if signed { sign.y } else { 1.0 };
            spread(&mut field.v, Vec2::new(p.x - 0.5, p.y), factor * value);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::solid::{box_container, box_container_image, face_weights, sample_corners};
    use super::super::transfer::faces_to_particles;
    use super::*;

    #[test]
    fn valid_samples_stay_and_rings_fill_outward() {
        let mut f = Field2::new(5, 1, 0.0);
        let mut valid = vec![false; 5];
        f.set(2, 0, 3.0);
        valid[2] = true;
        extrapolate(&mut f, &mut valid, 1);
        assert_eq!(f.data(), &[0.0, 3.0, 3.0, 3.0, 0.0]);
        assert_eq!(valid, vec![false, true, true, true, false]);
        extrapolate(&mut f, &mut valid, 1);
        assert_eq!(f.data(), &[3.0; 5]);
    }

    #[test]
    fn a_closed_face_mirrors_the_flow_across_its_wall() {
        let layout = MacLayout::new(8, 8, 1.0);
        let (min, max) = (Vec2::new(2.0, 2.0), Vec2::new(6.0, 7.0));
        let corners = sample_corners(&layout, box_container(min, max));
        let weights = face_weights(&layout, &corners);
        let image = box_container_image(min, max);
        let mut vel = MacVelocity::zeros(&layout);
        for value in vel.u.data_mut() {
            *value = 1.0;
        }
        for value in vel.v.data_mut() {
            *value = -2.0;
        }
        constrain_to_solids(&layout, &mut vel, &weights, &image);
        // One cell inside the left wall: the flow one cell into the tank,
        // its normal part reversed.
        assert!((vel.u.get(1, 4) + 1.0).abs() < 1e-6);
        // Inside the floor: v (normal) reversed, u (along it) kept.
        assert!((vel.v.get(4, 1) - 2.0).abs() < 1e-6);
        assert!((vel.u.get(4, 1) - 1.0).abs() < 1e-6);
        // Open faces are untouched.
        assert_eq!(vel.u.get(4, 4), 1.0);
        assert_eq!(vel.v.get(4, 4), -2.0);
    }

    /// With the mirror, a particle's normal velocity at a flat wall is zero
    /// and grows linearly away from it: it cannot be carried through.
    #[test]
    fn the_normal_velocity_read_at_a_wall_is_zero() {
        let layout = MacLayout::new(16, 16, 1.0);
        let (min, max) = (Vec2::new(4.0, 2.0), Vec2::new(14.0, 14.0));
        let corners = sample_corners(&layout, box_container(min, max));
        let weights = face_weights(&layout, &corners);
        let image = box_container_image(min, max);
        let mut vel = MacVelocity::zeros(&layout);
        let flow = -3.0;
        for j in 0..16 {
            for i in 5..14 {
                vel.u.set(i, j, flow);
            }
        }
        // The solve leaves the face on the wall at zero.
        constrain_to_solids(&layout, &mut vel, &weights, &image);
        for d in [0.0f32, 0.1, 0.25, 0.4] {
            let x = [Vec2::new(4.0 + d, 8.3)];
            let mut v = [Vec2::ZERO];
            let mut c = [glam::Mat2::ZERO];
            faces_to_particles(&layout, &vel, &x, &mut v, &mut c);
            assert!((v[0].x - d * flow).abs() < 1e-5, "d={d}: {}", v[0].x);
        }
    }

    /// Above a resting column the continued pressure is the hydrostatic
    /// line itself: a face the particles reached two cells into the air
    /// takes exactly the correction of the surface face below it.
    #[test]
    fn the_continued_pressure_carries_the_hydrostatic_gradient() {
        let layout = MacLayout::new(6, 10, 1.0);
        let (g, h, dt) = (9.0f32, 5.3f32, 0.01f32);
        let mut phi = layout.cells(0.0);
        let mut pressure = vec![0.0f32; 60];
        for j in 0..10 {
            for i in 0..6 {
                let y = j as f32 + 0.5;
                phi.set(i, j, y - h);
                if y < h {
                    pressure[i + 6 * j] = g * (h - y);
                }
            }
        }
        let open = vec![true; 60];
        let ghost = ghost_pressure(&layout, dt, &phi, &pressure, &open, (0.01, 2));
        // The surface face (between rows 4 and 5) and the two above it.
        let surface = ghost.correction((2, 4), (2, 5)).expect("surface face");
        assert!((surface - g * dt).abs() < 1e-4, "{surface}");
        for j in 6..=6 {
            let above = ghost.correction((2, j - 1), (2, j)).expect("air face");
            assert!(
                (above - surface).abs() < 1e-4,
                "row {j}: {above} vs {surface}"
            );
        }
        // Beyond the continuation: nothing.
        assert!(ghost.correction((2, 7), (2, 8)).is_none());
    }
}
