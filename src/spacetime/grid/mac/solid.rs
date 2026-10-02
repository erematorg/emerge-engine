//! Solid walls as a signed distance, negative inside the solid, and the
//! face weights they give: the open fraction of each face, from the wall's
//! real position rather than a cell label (Batty, Bertails and Bridson
//! 2007; `apic2d`'s `compute_weights`).
//!
//! Walls here are still: the pressure equations below carry no solid
//! velocity term.

use glam::Vec2;

use super::field::{Field2, MacLayout};

/// Fraction of the segment between two samples where the signed distance
/// is negative, assuming it varies linearly between them.
pub fn fraction_inside(phi_a: f32, phi_b: f32) -> f32 {
    match (phi_a < 0.0, phi_b < 0.0) {
        (true, true) => 1.0,
        (true, false) => phi_a / (phi_a - phi_b),
        (false, true) => phi_b / (phi_b - phi_a),
        (false, false) => 0.0,
    }
}

/// A solid's signed distance sampled at the cell corners.
pub fn sample_corners(layout: &MacLayout, solid: impl Fn(Vec2) -> f32) -> Field2 {
    let mut f = layout.corners(0.0);
    for j in 0..=layout.ny {
        for i in 0..=layout.nx {
            f.set(i, j, solid(Vec2::new(i as f32, j as f32) * layout.dx));
        }
    }
    f
}

/// A solid's signed distance sampled at the cell centres.
pub fn sample_centres(layout: &MacLayout, solid: impl Fn(Vec2) -> f32) -> Field2 {
    let mut f = layout.cells(0.0);
    for j in 0..layout.ny {
        for i in 0..layout.nx {
            f.set(i, j, solid(layout.cell_centre(i, j)));
        }
    }
    f
}

/// Open fraction of every face, in [0, 1].
#[derive(Clone, Debug, PartialEq)]
pub struct FaceWeights {
    pub u: Field2,
    pub v: Field2,
}

/// Open fraction of each face from the solid distance at its two ends.
/// The faces on the grid's own edge are closed: the edge is a wall.
pub fn face_weights(layout: &MacLayout, corners: &Field2) -> FaceWeights {
    let (nx, ny) = (layout.nx, layout.ny);
    let mut u = Field2::new(nx + 1, ny, 0.0);
    for j in 0..ny {
        for i in 1..nx {
            let open = 1.0 - fraction_inside(corners.get(i, j + 1), corners.get(i, j));
            u.set(i, j, open.clamp(0.0, 1.0));
        }
    }
    let mut v = Field2::new(nx, ny + 1, 0.0);
    for j in 1..ny {
        for i in 0..nx {
            let open = 1.0 - fraction_inside(corners.get(i + 1, j), corners.get(i, j));
            v.set(i, j, open.clamp(0.0, 1.0));
        }
    }
    FaceWeights { u, v }
}

/// A closed box container: positive inside, negative in its walls, the
/// distance to the nearest wall either way.
pub fn box_container(min: Vec2, max: Vec2) -> impl Fn(Vec2) -> f32 + Copy {
    move |q: Vec2| {
        let excess = (min - q).max(q - max);
        let outside = excess.max(Vec2::ZERO).length();
        let inside = excess.x.max(excess.y).min(0.0);
        -(outside + inside)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fraction_inside_follows_the_linear_crossing() {
        assert_eq!(fraction_inside(-1.0, -2.0), 1.0);
        assert_eq!(fraction_inside(1.0, 2.0), 0.0);
        assert!((fraction_inside(-1.0, 3.0) - 0.25).abs() < 1e-6);
        assert!((fraction_inside(3.0, -1.0) - 0.25).abs() < 1e-6);
    }

    #[test]
    fn the_box_container_measures_distance_to_its_nearest_wall() {
        let solid = box_container(Vec2::new(2.0, 2.0), Vec2::new(10.0, 6.0));
        assert!((solid(Vec2::new(3.0, 4.0)) - 1.0).abs() < 1e-6);
        assert!((solid(Vec2::new(6.0, 5.5)) - 0.5).abs() < 1e-6);
        assert!((solid(Vec2::new(1.0, 4.0)) + 1.0).abs() < 1e-6);
        assert!((solid(Vec2::new(11.0, 7.0)) + 2f32.sqrt()).abs() < 1e-6);
    }

    /// A wall that does not sit on a grid line leaves the faces it cuts
    /// partly open, by the fraction the wall leaves.
    #[test]
    fn a_wall_between_grid_lines_gives_partial_weights() {
        let layout = MacLayout::new(8, 8, 1.0);
        // Floor at y = 2.3: v faces on y = 2 are in the floor; u faces of
        // row 2 (from y = 2 to 3) are open over 0.7 of their length.
        let floor = |q: Vec2| q.y - 2.3;
        let weights = face_weights(&layout, &sample_corners(&layout, floor));
        assert!((weights.u.get(4, 2) - 0.7).abs() < 1e-5);
        assert_eq!(weights.u.get(4, 1), 0.0);
        assert_eq!(weights.u.get(4, 3), 1.0);
        assert_eq!(weights.v.get(4, 2), 0.0);
        assert_eq!(weights.v.get(4, 3), 1.0);
    }

    #[test]
    fn the_grid_edge_is_closed() {
        let layout = MacLayout::new(4, 3, 1.0);
        let open = |_: Vec2| 1.0;
        let weights = face_weights(&layout, &sample_corners(&layout, open));
        for j in 0..3 {
            assert_eq!(weights.u.get(0, j), 0.0);
            assert_eq!(weights.u.get(4, j), 0.0);
            assert_eq!(weights.u.get(2, j), 1.0);
        }
        for i in 0..4 {
            assert_eq!(weights.v.get(i, 0), 0.0);
            assert_eq!(weights.v.get(i, 3), 0.0);
        }
    }
}
