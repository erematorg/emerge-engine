//! APIC transfers between particles and the MAC faces. Each velocity
//! component lives on its own lattice, as in `apic2d`, with a kernel that
//! is quadratic along the component's own direction and linear across it:
//! `u` quadratic in x and linear in y, `v` the other way round.
//!
//! Why not quadratic both ways, as `apic2d` does: the gradient of that
//! interpolant is not divergence free even when every cell of the grid is,
//! and J integrates the divergence the particle reads. Measured in the
//! first gate run, deep in the liquid, it read 1000 to 3000 times the
//! divergence the grid keeps after the solve. With this pair of degrees
//! the derivative of the quadratic B-spline is the difference of two
//! linear ones, so `du/dx + dv/dy` at any point is the cells' own
//! divergence interpolated bilinearly between cell centres: zero wherever
//! the solve made it zero. Derived here; no source read gives it.
//!
//! A particle's affine matrix `C` follows the engine's convention: the
//! velocity near the particle is `v + C (q - x)`. So row 0 of `C` is the
//! gradient of `u` and row 1 the gradient of `v`. The gather sets it to
//! the exact gradient of what the particle reads, so `tr C` is that
//! divergence.

use glam::{Mat2, Vec2};

use super::field::{Field2, MacLayout, MacVelocity};

/// Sum of the kernel-weighted particle mass on every face, the same shapes
/// as `MacVelocity`. A face with zero mass received nothing.
#[derive(Clone, Debug, PartialEq)]
pub struct FaceMass {
    pub u: Field2,
    pub v: Field2,
}

/// Kernel weights along one axis for samples at `(k + offset) dx`: the
/// first sample index, the weights, their derivatives with respect to the
/// particle's coordinate, and how many samples are used.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Axis {
    pub base: i32,
    pub w: [f32; 3],
    pub dw: [f32; 3],
    pub len: usize,
}

/// The engine's quadratic B-spline (`grid::kernel::axis_weights`).
#[inline]
pub(crate) fn quadratic_axis(coord: f32, offset: f32, dx: f32) -> Axis {
    let s = coord / dx - offset;
    let base = (s - 0.5).floor();
    let f = s - base;
    Axis {
        base: base as i32,
        w: [
            0.5 * (1.5 - f).powi(2),
            0.75 - (f - 1.0).powi(2),
            0.5 * (f - 0.5).powi(2),
        ],
        dw: [-(1.5 - f) / dx, -2.0 * (f - 1.0) / dx, (f - 0.5) / dx],
        len: 3,
    }
}

/// The linear hat.
#[inline]
pub(crate) fn linear_axis(coord: f32, offset: f32, dx: f32) -> Axis {
    let s = coord / dx - offset;
    let base = s.floor();
    let f = s - base;
    Axis {
        base: base as i32,
        w: [1.0 - f, f, 0.0],
        dw: [-1.0 / dx, 1.0 / dx, 0.0],
        len: 2,
    }
}

/// Offsets of the `u` and `v` lattices, in cells.
const U_OFFSET: Vec2 = Vec2::new(0.0, 0.5);
const V_OFFSET: Vec2 = Vec2::new(0.5, 0.0);

fn u_axes(xp: Vec2, dx: f32) -> (Axis, Axis) {
    (
        quadratic_axis(xp.x, U_OFFSET.x, dx),
        linear_axis(xp.y, U_OFFSET.y, dx),
    )
}

fn v_axes(xp: Vec2, dx: f32) -> (Axis, Axis) {
    (
        linear_axis(xp.x, V_OFFSET.x, dx),
        quadratic_axis(xp.y, V_OFFSET.y, dx),
    )
}

/// Particle to faces: each face takes the mass-weighted average of the
/// particles' affine velocity at the face.
pub fn particles_to_faces(
    layout: &MacLayout,
    x: &[Vec2],
    v: &[Vec2],
    c: &[Mat2],
    mass: &[f32],
) -> (MacVelocity, FaceMass) {
    let mut vel = MacVelocity::zeros(layout);
    let mut face_mass = FaceMass {
        u: vel.u.clone(),
        v: vel.v.clone(),
    };
    let dx = layout.dx;
    for p in 0..x.len() {
        let particle = (x[p], mass[p]);
        let target = (&mut vel.u, &mut face_mass.u);
        let affine = (v[p].x, c[p].row(0));
        scatter(target, dx, U_OFFSET, u_axes(x[p], dx), particle, affine);
        let target = (&mut vel.v, &mut face_mass.v);
        let affine = (v[p].y, c[p].row(1));
        scatter(target, dx, V_OFFSET, v_axes(x[p], dx), particle, affine);
    }
    for (field, weight) in [(&mut vel.u, &face_mass.u), (&mut vel.v, &face_mass.v)] {
        for (value, &m) in field.data_mut().iter_mut().zip(weight.data()) {
            *value = if m > 0.0 { *value / m } else { 0.0 };
        }
    }
    (vel, face_mass)
}

/// Adds one particle's mass and affine momentum, `(value, gradient)` of the
/// component this lattice carries, to `(momentum, face_mass)`.
fn scatter(
    (momentum, face_mass): (&mut Field2, &mut Field2),
    dx: f32,
    offset: Vec2,
    (ax, ay): (Axis, Axis),
    (xp, m): (Vec2, f32),
    (component, gradient): (f32, Vec2),
) {
    for a in 0..ax.len {
        let i = ax.base + a as i32;
        if i < 0 || i as usize >= momentum.ni() {
            continue;
        }
        for b in 0..ay.len {
            let j = ay.base + b as i32;
            if j < 0 || j as usize >= momentum.nj() {
                continue;
            }
            let face = (Vec2::new(i as f32, j as f32) + offset) * dx;
            let w = ax.w[a] * ay.w[b] * m;
            momentum.add(
                i as usize,
                j as usize,
                w * (component + gradient.dot(face - xp)),
            );
            face_mass.add(i as usize, j as usize, w);
        }
    }
}

/// Faces to particles: each particle takes the kernel-weighted velocity of
/// the faces around it, and `C` the exact gradient of that interpolant.
pub fn faces_to_particles(
    layout: &MacLayout,
    vel: &MacVelocity,
    x: &[Vec2],
    v: &mut [Vec2],
    c: &mut [Mat2],
) {
    for p in 0..x.len() {
        let (u, grad_u) = gather(&vel.u, u_axes(x[p], layout.dx));
        let (w, grad_w) = gather(&vel.v, v_axes(x[p], layout.dx));
        v[p] = Vec2::new(u, w);
        c[p] = Mat2::from_cols(Vec2::new(grad_u.x, grad_w.x), Vec2::new(grad_u.y, grad_w.y));
    }
}

fn gather(field: &Field2, (ax, ay): (Axis, Axis)) -> (f32, Vec2) {
    let mut value = 0.0;
    let mut grad = Vec2::ZERO;
    for a in 0..ax.len {
        for b in 0..ay.len {
            let (i, j) = (ax.base + a as i32, ay.base + b as i32);
            let Some(f) = field.get_signed(i, j) else {
                continue;
            };
            value += ax.w[a] * ay.w[b] * f;
            grad += Vec2::new(ax.dw[a] * ay.w[b], ax.w[a] * ay.dw[b]) * f;
        }
    }
    (value, grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grid::kernel::axis_weights;

    #[test]
    fn the_quadratic_axis_is_the_engines_kernel() {
        // The engine's nodes sit at cell centres: offset 0.5.
        for &x in &[3.0f32, 3.2, 3.5, 3.9] {
            let axis = quadratic_axis(x, 0.5, 1.0);
            let d = x - x.floor() - 0.5;
            assert_eq!(axis.base, x.floor() as i32 - 1);
            let engine = axis_weights(d);
            for (k, (ours, theirs)) in axis.w.iter().zip(engine).enumerate() {
                assert!((ours - theirs).abs() < 1e-6, "x={x} k={k}");
            }
        }
    }

    #[test]
    fn both_axes_sum_to_one_have_no_first_moment_and_the_right_slope() {
        for &x in &[0.6f32, 1.0, 1.25, 2.49, 7.77] {
            for &offset in &[0.0f32, 0.5] {
                for axis_of in [quadratic_axis, linear_axis] {
                    let axis = axis_of(x, offset, 1.0);
                    let at = |k: usize| (axis.base + k as i32) as f32 + offset;
                    let sum: f32 = axis.w[..axis.len].iter().sum();
                    let moment: f32 = (0..axis.len).map(|k| axis.w[k] * (at(k) - x)).sum();
                    // Derivatives: of the constant 1 is 0, of the sample
                    // position is 1.
                    let d_sum: f32 = axis.dw[..axis.len].iter().sum();
                    let d_linear: f32 = (0..axis.len).map(|k| axis.dw[k] * at(k)).sum();
                    assert!((sum - 1.0).abs() < 1e-6);
                    assert!(moment.abs() < 1e-5);
                    assert!(d_sum.abs() < 1e-5);
                    assert!((d_linear - 1.0).abs() < 1e-5);
                }
            }
        }
    }

    /// What the gather reads has the cells' own divergence, interpolated
    /// bilinearly between cell centres, for any face velocity.
    #[test]
    fn the_divergence_read_is_the_cells_divergence() {
        let layout = MacLayout::new(12, 12, 1.0);
        let mut vel = MacVelocity::zeros(&layout);
        let mut seed = 99u32;
        let mut next = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            seed as f32 / u32::MAX as f32 - 0.5
        };
        for value in vel.u.data_mut().iter_mut().chain(vel.v.data_mut()) {
            *value = 10.0 * next();
        }
        let div = |i: usize, j: usize| {
            vel.u.get(i + 1, j) - vel.u.get(i, j) + vel.v.get(i, j + 1) - vel.v.get(i, j)
        };
        let x: Vec<Vec2> = (0..40)
            .map(|k| {
                let k = k as f32;
                Vec2::new(
                    3.0 + 6.0 * (k * 0.37).fract(),
                    3.0 + 6.0 * (k * 0.61).fract(),
                )
            })
            .collect();
        let mut v = vec![Vec2::ZERO; x.len()];
        let mut c = vec![Mat2::ZERO; x.len()];
        faces_to_particles(&layout, &vel, &x, &mut v, &mut c);
        for p in 0..x.len() {
            let t = x[p] - Vec2::splat(0.5);
            let (i, j) = (t.x.floor() as usize, t.y.floor() as usize);
            let (fx, fy) = (t.x.fract(), t.y.fract());
            let expected = div(i, j) * (1.0 - fx) * (1.0 - fy)
                + div(i + 1, j) * fx * (1.0 - fy)
                + div(i, j + 1) * (1.0 - fx) * fy
                + div(i + 1, j + 1) * fx * fy;
            let read = c[p].x_axis.x + c[p].y_axis.y;
            assert!(
                (read - expected).abs() < 1e-4,
                "at {}: {read} vs {expected}",
                x[p]
            );
        }
    }

    /// APIC carries an affine field exactly through a round trip, which is
    /// the property the MAC gather has to keep from the nodal one.
    #[test]
    fn an_affine_field_survives_a_round_trip() {
        let layout = MacLayout::new(16, 16, 1.0);
        let grad = Mat2::from_cols(Vec2::new(0.3, -0.2), Vec2::new(0.7, -0.3));
        let v0 = Vec2::new(1.5, -2.0);
        let centre = Vec2::splat(8.0);
        let field = |q: Vec2| v0 + grad * (q - centre);
        let mut x = Vec::new();
        for j in 0..24 {
            for i in 0..24 {
                x.push(Vec2::new(
                    2.0 + i as f32 * 0.5 + 0.25,
                    2.0 + j as f32 * 0.5 + 0.25,
                ));
            }
        }
        let v: Vec<Vec2> = x.iter().map(|&q| field(q)).collect();
        let c = vec![grad; x.len()];
        let mass = vec![0.25; x.len()];
        let (vel, _) = particles_to_faces(&layout, &x, &v, &c, &mass);
        let mut v_back = vec![Vec2::ZERO; x.len()];
        let mut c_back = vec![Mat2::ZERO; x.len()];
        faces_to_particles(&layout, &vel, &x, &mut v_back, &mut c_back);
        // Particles whose whole stencil sat on faces that received mass.
        for p in 0..x.len() {
            let q = x[p];
            if !(4.0..12.0).contains(&q.x) || !(4.0..12.0).contains(&q.y) {
                continue;
            }
            assert!((v_back[p] - field(q)).length() < 1e-4, "v at {q}");
            let err = (c_back[p] - grad).to_cols_array();
            assert!(err.iter().all(|e| e.abs() < 1e-4), "C at {q}");
        }
    }

    #[test]
    fn momentum_on_the_faces_equals_the_particles_momentum() {
        let layout = MacLayout::new(12, 12, 1.0);
        let x = vec![
            Vec2::new(4.3, 5.1),
            Vec2::new(6.8, 7.4),
            Vec2::new(5.5, 5.5),
        ];
        let v = vec![
            Vec2::new(1.0, -2.0),
            Vec2::new(-0.5, 0.25),
            Vec2::new(3.0, 1.0),
        ];
        let c = vec![Mat2::ZERO; 3];
        let mass = vec![1.0, 2.0, 0.5];
        let (vel, face_mass) = particles_to_faces(&layout, &x, &v, &c, &mass);
        let sum = |f: &Field2, m: &Field2| -> f32 {
            f.data().iter().zip(m.data()).map(|(a, b)| a * b).sum()
        };
        let particle: Vec2 = (0..3).map(|p| v[p] * mass[p]).sum();
        assert!((sum(&vel.u, &face_mass.u) - particle.x).abs() < 1e-5);
        assert!((sum(&vel.v, &face_mass.v) - particle.y).abs() < 1e-5);
    }
}
