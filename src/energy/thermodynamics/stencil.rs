//! Shared 5-point explicit-Euler Laplacian diffusion stencil.
//!
//! The one piece of math genuinely identical between [`super::diffusion`]
//! and [`super::scalar_field`] -- both scatter a particle scalar to the grid,
//! run this stencil, then gather the delta back. What differs between them
//! (direct-field access vs. runtime fn-pointer access, decay-to-zero vs.
//! Newton-cooling-to-ambient) is real, not accidental duplication -- see
//! each module's own docs. Only the stencil itself was hand-copied.

/// Applies one explicit-Euler diffusion step: `grid_out[c] = grid_in[c] +
/// diffusivity_dt * laplacian(grid_in, c)`.
///
/// Off-grid neighbors (domain edges) are treated as `ambient` -- a Dirichlet
/// boundary condition. Column-major layout: `idx = x * grid_res + y`,
/// matching the mechanics grid.
pub(crate) fn laplacian_step(
    grid_in: &[f32],
    grid_out: &mut [f32],
    grid_res: usize,
    diffusivity_dt: f32,
    ambient: f32,
) {
    for x in 0..grid_res {
        for y in 0..grid_res {
            let c = x * grid_res + y;
            let t_c = grid_in[c];
            let t_xm = if x > 0 {
                grid_in[c - grid_res]
            } else {
                ambient
            };
            let t_xp = if x + 1 < grid_res {
                grid_in[c + grid_res]
            } else {
                ambient
            };
            let t_ym = if y > 0 { grid_in[c - 1] } else { ambient };
            let t_yp = if y + 1 < grid_res {
                grid_in[c + 1]
            } else {
                ambient
            };
            let laplacian = t_xm + t_xp + t_ym + t_yp - 4.0 * t_c;
            grid_out[c] = t_c + diffusivity_dt * laplacian;
        }
    }
}
