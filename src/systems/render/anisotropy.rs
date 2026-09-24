//! Anisotropic surface kernels (Yu & Turk 2013, "Reconstructing Surfaces of
//! Particle-Based Fluids Using Anisotropic Kernels").
//!
//! ## Why this exists
//!
//! `curvature_flow.wgsl`'s splat is already anisotropic: it transforms each
//! candidate cell's offset through a per-particle 2x2 matrix before evaluating
//! the isotropic B-spline kernel. Until now that matrix came from the
//! particle's own `deformation_gradient` (composed with a velocity-stretch
//! term). That works for a solid, but it degenerates for a fluid, and the
//! reason is constitutive rather than incidental: a liquid carries no shear
//! memory, so its `F` stays isotropic (`sqrt(J) * I`) by definition, and the
//! velocity-stretch term is `1 + |v|*dt / kernel_radius`, which is ~1 for any
//! CFL-limited step. The product collapses to a scaled identity, every
//! particle splats the same symmetric blob, and blobs sitting on a regular
//! spawn lattice interfere -- the visible regular crosshatch/"white noise"
//! over fluid surfaces.
//!
//! Yu & Turk's answer is to derive the kernel's shape from the *neighborhood*
//! instead of from the material state: a weighted PCA of nearby particle
//! positions. On a flat sheet the neighbors lie along the surface, so the
//! kernel flattens along it -- flat regions render flat, thin sheets stay thin,
//! and neighboring kernels stop being identical, which is what removes the
//! lattice interference. The same construction is what NVIDIA Flex exposes as
//! its fluid-surface anisotropy (`NvFlexGetAnisotropy`, with `anisotropyScale`
//! /`anisotropyMin`/`anisotropyMax`): a per-particle oriented ellipsoid fitted
//! to the local particle distribution.
//!
//! ## Scale independence
//!
//! The matrix this module returns is normalized to `det == 1` -- a pure
//! *shape*, carrying orientation and aspect ratio but no size. Size continues
//! to come from the existing kernel radius and from `F`'s own `J`, exactly as
//! before. Two consequences, both wanted:
//!
//! - It composes multiplicatively with the existing `motion_stretch * F`
//!   without changing the splat's total footprint area, so the mass the splat
//!   deposits (and the volume-preserving correction built on top of it) is
//!   untouched.
//! - Because a covariance of positions scales as (length)^2 and the
//!   normalization divides that scale straight back out, the result is
//!   invariant to the absolute particle spacing. The same code is correct at
//!   ant scale and at landscape scale with no retuning -- the property this
//!   was chosen for.
//!
//! It is also material-agnostic: it reads positions only, so sand, water,
//! snow and tissue all get the same treatment with no per-material branch.
//!
//! CPU-first per this project's own rule ("CPU correctness first, GPU port
//! second"): the math lives here, unit-tested against the cases that actually
//! matter (flat sheet, isotropic bulk, sparse splash), so the GPU port is a
//! translation of verified code rather than eigendecomposition debugged
//! through a shader.

use glam::{Mat2, Vec2};

/// Tunables for [`kernel_shape_from_neighbors`].
///
/// Every field is a real, stated quantity rather than a scene-fitted fudge --
/// see each one's own doc for where its default comes from.
#[derive(Clone, Copy, Debug)]
pub struct AnisotropyParams {
    /// Neighbor search radius, in grid cells.
    ///
    /// Yu & Turk use twice the smoothing radius. The splat's own B-spline
    /// half-width is 1.5 cells (`BSPLINE_OUTER_LIMIT`), so the matching value
    /// here is 3.0. Expressed in grid cells rather than metres so it tracks
    /// `grid_cell_size` automatically.
    pub support_radius_cells: f32,
    /// Upper bound on the kernel's axis ratio (Yu & Turk's `k_r`, 4 in the
    /// paper; the same role as Flex's `anisotropyMin`/`anisotropyMax` clamp).
    ///
    /// Without it a near-degenerate neighborhood -- a particle with two
    /// collinear neighbors, say -- yields an arbitrarily thin kernel, which
    /// renders as a sliver rather than a surface.
    pub max_axis_ratio: f32,
    /// Below this many neighbors the fit is not trustworthy and the kernel
    /// falls back to isotropic (Yu & Turk's `N_eps` spherical fallback, which
    /// exists so isolated spray does not get a wild shape from one or two
    /// samples).
    ///
    /// The paper's 25 is a 3D number and does not port. Derived for 2D
    /// instead: a non-degenerate 2x2 covariance needs 3 non-collinear
    /// samples, so 4 is that minimum plus one sample of redundancy.
    pub min_neighbors: usize,
}

impl Default for AnisotropyParams {
    fn default() -> Self {
        Self {
            support_radius_cells: 3.0,
            max_axis_ratio: 4.0,
            min_neighbors: 4,
        }
    }
}

/// Yu & Turk's isotropic neighbor weighting (their eq. 8): `1 - (r/R)^3`
/// inside the support radius, 0 outside.
///
/// Chosen by the paper over a plain B-spline because it falls to zero with
/// zero slope at `r = R`, so a neighbor drifting across the support boundary
/// does not step-change the fitted covariance -- temporal stability, which
/// matters here for exactly the same reason the density field already needs
/// band hysteresis.
fn neighbor_weight(distance: f32, support_radius: f32) -> f32 {
    if distance >= support_radius || support_radius <= 0.0 {
        return 0.0;
    }
    let t = distance / support_radius;
    1.0 - t * t * t
}

/// Closed-form eigendecomposition of a symmetric 2x2 matrix
/// `[[a, b], [b, c]]`, returning `(lambda_major, lambda_minor, major_axis)`
/// with `lambda_major >= lambda_minor` and `major_axis` unit length.
///
/// 2D is the reason this technique is cheap here at all: the 3D case needs a
/// real iterative symmetric eigensolver (the cost that makes people hesitate
/// over Yu & Turk), whereas the 2x2 case is a quadratic root plus a
/// normalization.
fn symmetric_eigen_2x2(a: f32, b: f32, c: f32) -> (f32, f32, Vec2) {
    let half_trace = 0.5 * (a + c);
    // Discriminant of the characteristic polynomial, written as the
    // half-difference form so it cannot go negative through cancellation the
    // way `trace^2/4 - det` can for a nearly-degenerate matrix.
    let half_diff = 0.5 * (a - c);
    let disc = (half_diff * half_diff + b * b).max(0.0).sqrt();
    let lambda_major = half_trace + disc;
    let lambda_minor = half_trace - disc;

    // For [[a,b],[b,c]] an eigenvector of lambda is (lambda - c, b). When `b`
    // vanishes the matrix is already diagonal and that expression degenerates,
    // so pick the axis belonging to the larger diagonal entry directly.
    let axis = if b.abs() > 1.0e-12 {
        Vec2::new(lambda_major - c, b)
    } else if a >= c {
        Vec2::X
    } else {
        Vec2::Y
    };
    let len = axis.length();
    let axis = if len > 1.0e-12 { axis / len } else { Vec2::X };
    (lambda_major, lambda_minor, axis)
}

/// Fits an area-preserving kernel shape matrix to one particle's neighborhood.
///
/// `center` is the particle's own position and `neighbors` the positions of
/// the particles around it, both in grid coordinates; `center` itself may be
/// included or not, it changes nothing beyond one extra sample at zero
/// distance. Returns a symmetric matrix with `det == 1` whose major axis lies
/// along the direction the neighbors spread -- i.e. along a free surface
/// rather than across it -- and exactly [`Mat2::IDENTITY`] whenever the
/// neighborhood is too sparse or too isotropic to say anything.
///
/// Composes with the existing splat matrix as `motion_stretch * F * shape`.
pub fn kernel_shape_from_neighbors(
    center: Vec2,
    neighbors: &[Vec2],
    params: &AnisotropyParams,
) -> Mat2 {
    let r = params.support_radius_cells;

    // Weighted mean of the neighborhood. This is deliberately *not* assumed to
    // equal `center`: in the bulk it nearly does, but at a free surface the
    // neighborhood centroid sits measurably inward, and that offset is
    // precisely the signal that a surface is there. Assuming it away would
    // discard the case the whole technique exists to handle.
    let mut weight_sum = 0.0f32;
    let mut mean = Vec2::ZERO;
    let mut counted = 0usize;
    for &n in neighbors {
        let offset = n - center;
        let w = neighbor_weight(offset.length(), r);
        if w <= 0.0 {
            continue;
        }
        weight_sum += w;
        mean += w * offset;
        counted += 1;
    }
    if counted < params.min_neighbors || weight_sum <= 1.0e-12 {
        return Mat2::IDENTITY;
    }
    mean /= weight_sum;

    // Weighted covariance about that mean (Yu & Turk eq. 9). Accumulated in
    // offsets relative to `center` rather than in absolute grid coordinates:
    // the second moment of a coordinate of order 64 is of order 4096, and
    // subtracting the squared mean from it loses most of an f32's precision to
    // cancellation, whereas offsets stay of order the particle spacing.
    let mut cxx = 0.0f32;
    let mut cxy = 0.0f32;
    let mut cyy = 0.0f32;
    for &n in neighbors {
        let offset = n - center;
        let w = neighbor_weight(offset.length(), r);
        if w <= 0.0 {
            continue;
        }
        let d = offset - mean;
        cxx += w * d.x * d.x;
        cxy += w * d.x * d.y;
        cyy += w * d.y * d.y;
    }
    cxx /= weight_sum;
    cxy /= weight_sum;
    cyy /= weight_sum;

    let (lambda_major, lambda_minor, major_axis) = symmetric_eigen_2x2(cxx, cxy, cyy);
    if !lambda_major.is_finite() || lambda_major <= 1.0e-12 {
        return Mat2::IDENTITY;
    }

    // Kernel extent along an axis goes as the standard deviation, not the
    // variance -- the eigenvalues of a covariance are squared lengths, and the
    // splat matrix is applied to lengths.
    let major = lambda_major.max(0.0).sqrt();
    let minor = lambda_minor.max(0.0).sqrt();

    // Clamp the aspect ratio before normalizing, so a degenerate fit becomes a
    // bounded ellipse rather than a sliver.
    let ratio_limit = params.max_axis_ratio.max(1.0);
    let minor = minor.max(major / ratio_limit);
    if major <= 1.0e-12 {
        return Mat2::IDENTITY;
    }

    // Normalize to unit determinant: strip the size, keep orientation and
    // aspect. This is what makes the result independent of particle spacing
    // and keeps the splat's footprint area (and therefore its deposited mass)
    // exactly what it was before this term existed.
    let norm = (major * minor).sqrt();
    if norm <= 1.0e-12 {
        return Mat2::IDENTITY;
    }
    let s_major = major / norm;
    let s_minor = minor / norm;

    // Reassemble R * diag(s_major, s_minor) * R^T from the major axis.
    let e0 = major_axis;
    let e1 = Vec2::new(-e0.y, e0.x);
    Mat2::from_cols(
        s_major * e0.x * e0 + s_minor * e1.x * e1,
        s_major * e0.y * e0 + s_minor * e1.y * e1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn det(m: Mat2) -> f32 {
        m.x_axis.x * m.y_axis.y - m.y_axis.x * m.x_axis.y
    }

    /// A particle with too few neighbors -- isolated spray -- must fall back to
    /// isotropic rather than take a wild shape from one or two samples.
    #[test]
    fn sparse_neighborhood_falls_back_to_isotropic() {
        let params = AnisotropyParams::default();
        let neighbors = [Vec2::new(0.5, 0.0), Vec2::new(-0.5, 0.0)];
        let shape = kernel_shape_from_neighbors(Vec2::ZERO, &neighbors, &params);
        assert_eq!(shape, Mat2::IDENTITY);
    }

    /// Neighbors spread evenly in all directions carry no directional
    /// information, so the fitted kernel must come back (very near) isotropic.
    /// This is the bulk-interior case, and it is what guarantees the change is
    /// a no-op away from surfaces.
    #[test]
    fn isotropic_neighborhood_stays_isotropic() {
        let params = AnisotropyParams::default();
        let neighbors: Vec<Vec2> = (0..12)
            .map(|k| {
                let a = std::f32::consts::TAU * k as f32 / 12.0;
                Vec2::new(a.cos(), a.sin())
            })
            .collect();
        let shape = kernel_shape_from_neighbors(Vec2::ZERO, &neighbors, &params);
        for (got, want) in shape
            .to_cols_array()
            .iter()
            .zip(Mat2::IDENTITY.to_cols_array().iter())
        {
            assert!(
                (got - want).abs() < 1.0e-3,
                "expected near-identity for an isotropic neighborhood, got {shape:?}"
            );
        }
    }

    /// The case the whole technique exists for: neighbors lying along a flat
    /// sheet must elongate the kernel ALONG the sheet (here, x) and thin it
    /// across, so flat surfaces render flat instead of as a row of blobs.
    #[test]
    fn flat_sheet_elongates_along_the_surface() {
        let params = AnisotropyParams::default();
        // A horizontal band: wide spread in x, a single row in y.
        let neighbors: Vec<Vec2> = (-4..=4).map(|k| Vec2::new(k as f32 * 0.5, 0.0)).collect();
        let shape = kernel_shape_from_neighbors(Vec2::ZERO, &neighbors, &params);

        let x_extent = shape.x_axis.length();
        let y_extent = shape.y_axis.length();
        assert!(
            x_extent > y_extent * 2.0,
            "expected the kernel to stretch along the sheet (x) and thin across it (y), \
             got x_extent={x_extent}, y_extent={y_extent}"
        );
    }

    /// Area preservation is what lets this compose with `F` without disturbing
    /// the splat's deposited mass or the volume-preserving correction built on
    /// it. It must hold for any neighborhood, including degenerate ones that
    /// hit the aspect-ratio clamp.
    #[test]
    fn shape_matrix_always_has_unit_determinant() {
        let params = AnisotropyParams::default();
        let cases: Vec<Vec<Vec2>> = vec![
            (-4..=4).map(|k| Vec2::new(k as f32 * 0.5, 0.0)).collect(),
            (-4..=4).map(|k| Vec2::new(0.0, k as f32 * 0.5)).collect(),
            (-4..=4)
                .map(|k| Vec2::new(k as f32 * 0.4, k as f32 * 0.4))
                .collect(),
            (0..12)
                .map(|k| {
                    let a = std::f32::consts::TAU * k as f32 / 12.0;
                    Vec2::new(a.cos(), a.sin())
                })
                .collect(),
        ];
        for neighbors in cases {
            let shape = kernel_shape_from_neighbors(Vec2::ZERO, &neighbors, &params);
            assert!(
                (det(shape) - 1.0).abs() < 1.0e-3,
                "determinant must stay 1 (area-preserving), got {} for {shape:?}",
                det(shape)
            );
        }
    }

    /// The aspect-ratio clamp must actually bind: a perfectly collinear
    /// neighborhood would otherwise fit an infinitely thin kernel.
    #[test]
    fn aspect_ratio_clamp_bounds_a_degenerate_fit() {
        let params = AnisotropyParams::default();
        let neighbors: Vec<Vec2> = (-4..=4).map(|k| Vec2::new(k as f32 * 0.5, 0.0)).collect();
        let shape = kernel_shape_from_neighbors(Vec2::ZERO, &neighbors, &params);
        let (major, minor, _) = symmetric_eigen_2x2(shape.x_axis.x, shape.x_axis.y, shape.y_axis.y);
        let ratio = major / minor.max(1.0e-6);
        assert!(
            ratio <= params.max_axis_ratio + 1.0e-3,
            "aspect ratio {ratio} exceeded the configured clamp {}",
            params.max_axis_ratio
        );
    }

    /// Rotating the neighborhood must rotate the fitted kernel by the same
    /// amount -- the fit may not depend on how the scene happens to sit
    /// relative to the grid axes, or surfaces would render differently
    /// depending on their orientation.
    #[test]
    fn fit_is_rotation_equivariant() {
        let params = AnisotropyParams::default();
        let flat: Vec<Vec2> = (-4..=4).map(|k| Vec2::new(k as f32 * 0.5, 0.0)).collect();
        let angle = 0.7f32;
        let (s, c) = angle.sin_cos();
        let rot = Mat2::from_cols(Vec2::new(c, s), Vec2::new(-s, c));
        let rotated: Vec<Vec2> = flat.iter().map(|&p| rot * p).collect();

        let shape = kernel_shape_from_neighbors(Vec2::ZERO, &flat, &params);
        let shape_rotated = kernel_shape_from_neighbors(Vec2::ZERO, &rotated, &params);
        let expected = rot * shape * rot.transpose();

        for (got, want) in shape_rotated
            .to_cols_array()
            .iter()
            .zip(expected.to_cols_array().iter())
        {
            assert!(
                (got - want).abs() < 1.0e-3,
                "expected rotation equivariance: got {shape_rotated:?}, \
                 expected {expected:?} (unrotated {shape:?})"
            );
        }
    }

    /// A free surface offsets the neighborhood centroid inward. The fit must
    /// use that real weighted mean rather than assuming the particle sits at
    /// its own neighborhood's centre, since that offset is the surface signal.
    #[test]
    fn surface_particle_fits_along_its_own_surface() {
        let params = AnisotropyParams::default();
        // Particle at the top of a block: neighbors to the sides and below,
        // none above -- the real configuration at a free surface.
        let mut neighbors = Vec::new();
        for gx in -3..=3 {
            for gy in -3..=0 {
                if gx == 0 && gy == 0 {
                    continue;
                }
                neighbors.push(Vec2::new(gx as f32 * 0.5, gy as f32 * 0.5));
            }
        }
        let shape = kernel_shape_from_neighbors(Vec2::ZERO, &neighbors, &params);
        assert!(
            shape.x_axis.length() > shape.y_axis.length(),
            "a particle on a horizontal free surface should stretch along it, got {shape:?}"
        );
        assert!(
            (det(shape) - 1.0).abs() < 1.0e-3,
            "surface fit must stay area-preserving, got det={}",
            det(shape)
        );
    }
}
