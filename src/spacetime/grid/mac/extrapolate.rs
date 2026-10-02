//! After the update only the faces touching liquid hold a projected
//! velocity. The gather reads a little beyond them, so the velocity is
//! extended outward first (Bridson and Muller-Fischer 6.3), here layer by
//! layer as the average of the valid neighbours (`apic2d`'s
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
//! not allow.

use glam::Vec2;

use super::field::{FaceFlags, Field2, MacLayout, MacVelocity};
use super::solid::FaceWeights;

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

/// On every closed face, the mirror image of the flow across the wall
/// (module doc). The solid's distance at the face gives the depth, its
/// gradient the normal. On the grid's own edge, the edge is the wall and
/// the component normal to it is zero.
pub fn constrain_to_solids(
    layout: &MacLayout,
    vel: &mut MacVelocity,
    weights: &FaceWeights,
    corners: &Field2,
) {
    let before = vel.clone();
    let mirror = |position: Vec2| -> Vec2 {
        let at = position / layout.dx;
        let n = corners.bilinear_gradient(at);
        if n.length_squared() == 0.0 {
            return Vec2::ZERO;
        }
        let n = n.normalize();
        let depth = (-corners.bilinear(at)).max(0.0);
        let outside = before.at(layout, position + 2.0 * depth * n);
        outside - 2.0 * outside.dot(n) * n
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

#[cfg(test)]
mod tests {
    use super::super::solid::{box_container, face_weights, sample_corners};
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
        let solid = box_container(Vec2::new(2.0, 2.0), Vec2::new(6.0, 7.0));
        let corners = sample_corners(&layout, solid);
        let weights = face_weights(&layout, &corners);
        let mut vel = MacVelocity::zeros(&layout);
        for value in vel.u.data_mut() {
            *value = 1.0;
        }
        for value in vel.v.data_mut() {
            *value = -2.0;
        }
        constrain_to_solids(&layout, &mut vel, &weights, &corners);
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
        let solid = box_container(Vec2::new(4.0, 2.0), Vec2::new(14.0, 14.0));
        let corners = sample_corners(&layout, solid);
        let weights = face_weights(&layout, &corners);
        let mut vel = MacVelocity::zeros(&layout);
        let flow = -3.0;
        for j in 0..16 {
            for i in 5..14 {
                vel.u.set(i, j, flow);
            }
        }
        // The solve leaves the face on the wall at zero.
        constrain_to_solids(&layout, &mut vel, &weights, &corners);
        for d in [0.0f32, 0.1, 0.25, 0.4] {
            let x = [Vec2::new(4.0 + d, 8.3)];
            let mut v = [Vec2::ZERO];
            let mut c = [glam::Mat2::ZERO];
            faces_to_particles(&layout, &vel, &x, &mut v, &mut c);
            assert!((v[0].x - d * flow).abs() < 1e-5, "d={d}: {}", v[0].x);
        }
    }
}
