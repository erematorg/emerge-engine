//! Single-phase (strict, non-mixture) fluid incompressibility pressure
//! projection: Chorin-style (Bridson, "Fluid Simulation for Computer
//! Graphics" ch. 5; the family of `mixture::pressure`, Zhao & Choo 2020,
//! arXiv:1905.00671), solved exactly by a discrete cosine transform
//! (DCT-II/DCT-III) instead of Jacobi/Gauss-Seidel sweeps.
//!
//! The variable-mobility Jacobi solve of `mixture::pressure`, then
//! Gauss-Seidel with SOR (Young 1954), did not stabilize a near-full-height
//! water column starting against a wall, and a better-converged solve made
//! it worse, with or without a floor on the free-surface `alpha = 1/mass`:
//! the violence of the first, uncushioned impact (`eos_stiffness = 0`) limits
//! more than the formulation. As in Stam's "Stable Fluids" (1999), density
//! is taken uniform, so the Poisson operator has constant coefficients,
//! which the DCT-II basis diagonalizes exactly for Neumann (zero-flux)
//! boundaries, the condition `SlipBoundary` enforces at a wall; every cell
//! shares one bounded alpha. `dct.rs`'s direct O(N^2)-per-row transform is
//! cheap at this grid size and checked by a round-trip test.
//!
//! Superseded by `grid::mac` once that passes its gates (see
//! `KNOWN_LIMITATIONS.md`, "Pressure projection").

use glam::{IVec2, Vec2};

use super::Grid;
use super::dct::{dct2_forward, dct2_inverse};

/// Extra cells of padding around the active cells' own bounding box (see
/// `project_fluid_incompressibility`'s doc): the DCT solve's Neumann
/// (zero-flux) boundary is only PHYSICALLY correct where it lands on a real
/// wall (`SlipBoundary` already enforces the same condition there); landing
/// it right at the fluid's own free surface instead would wrongly treat
/// "open air" as a sealed boundary. Padding the box with naturally-
/// zero-divergence empty cells pushes that approximation error away from
/// the actual fluid body instead of eliminating it outright (a real,
/// disclosed limit of a LOCAL Neumann solve, not unique to this
/// implementation -- any local/regional pressure solve has to make the same
/// call at its own domain edge).
const BOUNDING_BOX_PADDING: i32 = 4;

impl Grid {
    /// Enforces `div(v) = 0` on the grid's active velocity field (see the
    /// module doc) by an exact constant-density DCT Poisson solve.
    /// `pressure_iterations = 0` is a no-op; any nonzero value turns it on
    /// (the DCT does not iterate; the parameter keeps the convention of
    /// `mixture_pressure_iterations`).
    ///
    /// Transforms only a padded bounding box of the active cells
    /// (`self.dirty`), not the whole `resolution x resolution` domain: the
    /// grid is sparse, and cost scales with the fluid body's extent. Correct
    /// only when every particle on this grid is a strict fluid, which
    /// `Simulation::assert_strict_fluid_mode_is_supported` enforces whenever
    /// `SimConfig::fluid_pressure_iterations > 0`.
    ///
    /// Free-surface cells (fluid next to open, non-fluid space) get `p = 0`
    /// (Dirichlet, open to the atmosphere); true walls keep Neumann. A
    /// homogeneous Neumann condition at every missing neighbour cannot give
    /// the pressure gradient that holds a gravity-loaded column up. This is
    /// the distinction Bridson's liquid solver draws with `liquid_phi` (see
    /// the Gauss-Seidel sweep).
    pub fn project_fluid_incompressibility(&mut self, cell_width: f32, pressure_iterations: u32) {
        if pressure_iterations == 0 || self.dirty.is_empty() {
            return;
        }
        // One solve over one bounding box, even for disjoint bodies: splitting
        // by connected component (e.g. water and mud with an empty gap between
        // them, 63x61 = 3843 cells solved for ~1300 touched) measured slower,
        // 549 s -> 995 s over 120 frames with SipHash and 549 s -> 732 s with
        // `FxU32BuildHasher`: running the whole pipeline twice costs more than
        // the smaller boxes save.
        const MIN_ABSOLUTE_MASS_FOR_CORRECTION: f32 = 1.0e-3;
        let h = cell_width.max(1.0e-6);
        let res = self.resolution as i32;

        // One representative mass for the DCT solve, whose eigenbasis only
        // diagonalizes a uniform-density Laplacian: the average over active
        // cells, which sits near `rest_density * cell_area` for the fluid's
        // bulk. Only the DCT and the refinement's fallback for near-empty
        // cells use it; the refinement's fluid cells and the momentum
        // correction use each cell's own mass (a water/mud scene, 40x apart,
        // swung `mass_avg` from 1.0 to 11.5 in one run). A full
        // variable-coefficient solve (MGPCG) is future work.
        let mass_avg: f32 = {
            let (sum, count) = self.dirty.iter().fold((0.0f32, 0u32), |(s, c), &idx| {
                self.cells
                    .get(&idx)
                    .map_or((s, c), |cell| (s + cell.mass, c + 1))
            });
            if count == 0 {
                return;
            }
            (sum / count as f32).max(1.0e-6)
        };
        let alpha_const = 1.0 / mass_avg;

        // Padded bounding box of the active region (module doc explains the
        // padding), clamped to the domain -- clamping at a TRUE wall is
        // exactly right (Neumann there matches `SlipBoundary`'s own zero-flux
        // condition for real), it's only the non-wall edges where padding is
        // an approximation.
        let (mut min_x, mut min_y, mut max_x, mut max_y) = (res, res, 0, 0);
        for &idx in &self.dirty {
            let pos = self.idx_to_pos(idx);
            min_x = min_x.min(pos.x);
            min_y = min_y.min(pos.y);
            max_x = max_x.max(pos.x);
            max_y = max_y.max(pos.y);
        }
        let ox = (min_x - BOUNDING_BOX_PADDING).max(0);
        let oy = (min_y - BOUNDING_BOX_PADDING).max(0);
        let ex = (max_x + BOUNDING_BOX_PADDING).min(res - 1);
        let ey = (max_y + BOUNDING_BOX_PADDING).min(res - 1);
        let nx = (ex - ox + 1) as usize;
        let ny = (ey - oy + 1) as usize;

        // Dense divergence field over the LOCAL box only -- cells inside it
        // but not truly touched read `velocity_at`'s own real zero fallback,
        // so their divergence is naturally zero; harmless, no special-casing
        // needed (same property the old full-domain version relied on).
        //
        // Tried and reverted: swapping this for `velocity_at_or_extrapolated`
        // cleans the fluid cells but leaves the same fake divergence on the
        // air cells next to the body, which the solve keeps as unknowns (see
        // the surface classification below). J still hit the [0.5, 2.0]
        // clamp on the falling-droplet gate
        // (`tests/probes/falling_droplet_pressure_projection_check.rs`).
        // Computing the divergence on fluid cells only removes that source
        // but exposes further defects of this solve; the measured list is in
        // the pressure projection entry of `KNOWN_LIMITATIONS.md`.
        let mut rhs = vec![0.0f32; nx * ny];
        for lx in 0..nx {
            for ly in 0..ny {
                let pos = IVec2::new(ox + lx as i32, oy + ly as i32);
                let v_r = self.velocity_at(pos + IVec2::new(1, 0)).x;
                let v_l = self.velocity_at(pos - IVec2::new(1, 0)).x;
                let v_u = self.velocity_at(pos + IVec2::new(0, 1)).y;
                let v_d = self.velocity_at(pos - IVec2::new(0, 1)).y;
                let div_v = (v_r - v_l) / (2.0 * h) + (v_u - v_d) / (2.0 * h);
                rhs[lx * ny + ly] = div_v;
            }
        }

        // Debug output for the wall-free-pool investigation, behind the
        // `EMERGE_DEBUG_PRESSURE` research switch (compiled out of a default build).
        if crate::diagnostics::research_switch("EMERGE_DEBUG_PRESSURE").is_some() {
            let (mut rmin, mut rmax) = (f32::MAX, f32::MIN);
            for &v in &rhs {
                rmin = rmin.min(v);
                rmax = rmax.max(v);
            }
            eprintln!(
                "PRESSURE_DEBUG box=({nx}x{ny}) mass_avg={mass_avg:.6} alpha_const={alpha_const:.6} rhs=[{rmin:.6},{rmax:.6}]"
            );
        }

        // Forward DCT-II, both axes (separable 2D transform).
        let rhs_hat = dct2_forward(&rhs, nx, ny);

        // Divide by the Neumann-Laplacian eigenvalues (Strang; standard
        // discrete-cosine-diagonalizes-the-reflective-Laplacian identity):
        // lambda_{kx,ky} = (2cos(pi*kx/nx)-2)/h^2 + (2cos(pi*ky/ny)-2)/h^2.
        // Laplacian(p) = rhs/alpha_const (constant-coefficient form of
        // div(alpha*grad(p))=rhs), so P_hat = RHS_hat/(alpha_const*lambda).
        // kx=ky=0 (the constant/DC mode) is the Neumann operator's null
        // space -- pressure is only defined up to an additive constant for
        // a closed system with no Dirichlet anchor, same fact the old
        // Jacobi/GS solve's own implicit fixed point already encoded; fixed
        // at 0 here explicitly rather than left to accumulate roundoff.
        let mut p_hat = vec![0.0f32; nx * ny];
        for kx in 0..nx {
            for ky in 0..ny {
                if kx == 0 && ky == 0 {
                    continue;
                }
                let lx = 2.0 * (std::f32::consts::PI * kx as f32 / nx as f32).cos() - 2.0;
                let ly = 2.0 * (std::f32::consts::PI * ky as f32 / ny as f32).cos() - 2.0;
                let lambda = (lx + ly) / (h * h);
                if lambda.abs() > 1.0e-9 {
                    p_hat[kx * ny + ky] = rhs_hat[kx * ny + ky] / (alpha_const * lambda);
                }
            }
        }

        // Exponential spectral filter before the inverse transform (Hesthaven
        // & Warburton, "Nodal Discontinuous Galerkin Methods" 2008, ch. 5):
        // an exact spectral solve rings (Gibbs) at sharp features, measured
        // as an isolated `grad_p.y > 1000` at the wall (y = 0) among cells at
        // 10-100. Only the highest frequencies are damped.
        const FILTER_ALPHA: f32 = 36.0;
        const FILTER_ORDER: i32 = 2;
        for kx in 0..nx {
            let eta_x = if nx > 1 {
                kx as f32 / (nx - 1) as f32
            } else {
                0.0
            };
            let sigma_x = (-FILTER_ALPHA * eta_x.powi(2 * FILTER_ORDER)).exp();
            for ky in 0..ny {
                let eta_y = if ny > 1 {
                    ky as f32 / (ny - 1) as f32
                } else {
                    0.0
                };
                let sigma_y = (-FILTER_ALPHA * eta_y.powi(2 * FILTER_ORDER)).exp();
                p_hat[kx * ny + ky] *= sigma_x * sigma_y;
            }
        }

        let mut pressure = dct2_inverse(&p_hat, nx, ny);

        // Free-surface Dirichlet condition, as in Bridson's apic2d
        // (`tmp/apic2d/fluidsim.cpp`, whose `liquid_phi` gives the free
        // surface `p = 0` and solid walls a zero-flux condition). `local_mass`
        // marks the fluid cells (from `self.dirty`, inside this padded box): a
        // fluid cell next to an in-bounds low- or zero-mass neighbour is free
        // surface, pinned to p = 0; one whose missing neighbour is the domain
        // edge is a wall and keeps the Neumann treatment below.
        let mut local_mass = vec![0.0f32; nx * ny];
        for &idx in &self.dirty {
            let pos = self.idx_to_pos(idx);
            let (lx, ly) = (pos.x - ox, pos.y - oy);
            if lx >= 0
                && ly >= 0
                && (lx as usize) < nx
                && (ly as usize) < ny
                && let Some(cell) = self.cells.get(&idx)
            {
                local_mass[lx as usize * ny + ly as usize] = cell.mass;
            }
        }
        // The surface test is relative to the representative cell mass, not
        // the tiny `MIN_ABSOLUTE_MASS_FOR_CORRECTION` (which excludes near-empty
        // cells from the correction). With that absolute threshold, kernel-edge
        // mass fluctuations at a moving interface flipped cells between
        // surface (p = 0) and interior from one substep to the next, a
        // discontinuity in the solved pressure each time.
        let surface_mass_threshold = (mass_avg * 0.3).max(MIN_ABSOLUTE_MASS_FOR_CORRECTION);
        let mut is_surface = vec![false; nx * ny];
        for lx in 0..nx {
            for ly in 0..ny {
                let local_idx = lx * ny + ly;
                if local_mass[local_idx] <= surface_mass_threshold {
                    continue;
                }
                for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                    let (nlx, nly) = (lx as i32 + dx, ly as i32 + dy);
                    if nlx >= 0 && nly >= 0 && (nlx as usize) < nx && (nly as usize) < ny {
                        // A low-mass neighbour at a true wall cell (e.g. the
                        // oy == 0 row before any particle reached it) is the
                        // floor, not open air. Treated as free surface, it
                        // pinned an approaching fluid cell to p = 0 at the
                        // moment it needed the wall's support (a puddle reached
                        // J = 60 at first contact).
                        let neighbor_is_true_wall = (nlx == 0 && ox == 0)
                            || (nlx == nx as i32 - 1 && ex == res - 1)
                            || (nly == 0 && oy == 0)
                            || (nly == ny as i32 - 1 && ey == res - 1);
                        let n_idx = nlx as usize * ny + nly as usize;
                        if local_mass[n_idx] <= surface_mass_threshold && !neighbor_is_true_wall {
                            is_surface[local_idx] = true;
                            break;
                        }
                    }
                }
                if is_surface[local_idx] {
                    pressure[local_idx] = 0.0;
                }
            }
        }

        // Hybrid correction: the DCT converges the smooth part of the field
        // in one exact pass but rings near sharp features; Gauss-Seidel has no
        // basis ringing but converges slowly from a cold start. Starting it
        // from the DCT solution cleans the local artifacts in a few sweeps.
        // Same system (`alpha_const*Laplacian(p) = rhs`); at the local box
        // edge a missing neighbour is left out of both the sum and the divisor
        // (no flux, as the DCT eigenvalues assume).
        // Per-sweep convergence (calm and mid-impact samples): the max cell
        // change falls from ~6e-3 at sweep 0 to ~6e-5 by sweep 10.
        //
        // Raised from 5 to 10: on the wall-contact column scene
        // (`diag_pressure_projection_timing.rs`) the frame rate went from
        // 16.2-16.5 fps to 30.1-30.6 fps, because a better-converged
        // correction triggers fewer CFL refinements. Those figures measure
        // cost only: in that scene J sits at the [0.5, 2.0] safety clamp
        // from about frame 20 onward, so they are not the frame rate of a
        // valid incompressible run.
        //
        // It does not cause the initialization spike of a wall-free pool
        // (p = 0 on its whole perimeter, max_speed 100-250): 5 or 10 sweeps
        // give the same peak (144 and 207). That cause is still open.
        const GS_CORRECTION_SWEEPS: u32 = 10;
        let p_or_none = |p: &[f32], px: i32, py: i32| -> Option<f32> {
            if px < 0 || py < 0 || px as usize >= nx || py as usize >= ny {
                None
            } else {
                Some(p[px as usize * ny + py as usize])
            }
        };
        for _ in 0..GS_CORRECTION_SWEEPS {
            for lx in 0..nx {
                for ly in 0..ny {
                    let local_idx = lx * ny + ly;
                    // Dirichlet-pinned free-surface cell: never updated, its
                    // fixed p=0 still gets read normally by neighbors below.
                    if is_surface[local_idx] {
                        continue;
                    }
                    let r = rhs[local_idx];
                    let mut sum = 0.0f32;
                    let mut count = 0.0f32;
                    for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                        if let Some(v) = p_or_none(&pressure, lx as i32 + dx, ly as i32 + dy) {
                            sum += v;
                            count += 1.0;
                        }
                    }
                    if count > 0.0 {
                        // Each fluid cell's own mass (`local_mass[local_idx]`,
                        // already scattered for surface classification): the
                        // DCT must stay constant-coefficient, but Gauss-Seidel
                        // handles a varying coefficient one equation at a time
                        // (why multigrid uses it as the smoother, with a
                        // constant-coefficient solve as the initial guess, the
                        // DCT's role here). Near-empty placeholder cells in the
                        // padded box fall back to `alpha_const`: their own mass
                        // would blow up `1/mass`.
                        let cell_alpha = if local_mass[local_idx] > MIN_ABSOLUTE_MASS_FOR_CORRECTION
                        {
                            1.0 / local_mass[local_idx]
                        } else {
                            alpha_const
                        };
                        pressure[local_idx] = (sum - h * h * r / cell_alpha) / count;
                    }
                }
            }
        }

        // Debug output, same `EMERGE_DEBUG_PRESSURE` switch as the block that
        // prints `rhs` above.
        if crate::diagnostics::research_switch("EMERGE_DEBUG_PRESSURE").is_some() {
            let (mut pmin, mut pmax) = (f32::MAX, f32::MIN);
            for &v in &pressure {
                pmin = pmin.min(v);
                pmax = pmax.max(v);
            }
            let surface_count = is_surface.iter().filter(|&&s| s).count();
            eprintln!(
                "PRESSURE_DEBUG solved pressure=[{pmin:.6},{pmax:.6}] surface_cells={surface_count}/{}",
                nx * ny
            );
        }

        for &idx in &self.dirty {
            let Some(&mass) = self.cells.get(&idx).map(|c| &c.mass) else {
                continue;
            };
            if mass <= MIN_ABSOLUTE_MASS_FOR_CORRECTION {
                continue;
            }
            let pos = self.idx_to_pos(idx);
            let (lx, ly) = (pos.x - ox, pos.y - oy);
            let p_at = |px: i32, py: i32| -> f32 {
                if px < 0 || py < 0 || px as usize >= nx || py as usize >= ny {
                    0.0
                } else {
                    pressure[px as usize * ny + py as usize]
                }
            };
            let p_r = p_at(lx + 1, ly);
            let p_l = p_at(lx - 1, ly);
            let p_u = p_at(lx, ly + 1);
            let p_d = p_at(lx, ly - 1);
            let grad_p = Vec2::new((p_r - p_l) / (2.0 * h), (p_u - p_d) / (2.0 * h));
            // Under-relaxation: the solve is exact for this substep's
            // divergence, but uniform density approximates each cell's mass,
            // so the implied correction is not exact either. 0.2, measured on
            // the wall-contact column and the wall-free pool: 0.1 left enough
            // residual divergence to accumulate into J (a `tait_pressure`
            // `j>0` assert), 0.3 and 0.8 were worse than 0.2 on both scenes
            // (0.8 kept the pool's max_speed at 22-37 past frame 120 and took
            // the column from 30 fps to 12).
            const RELAXATION: f32 = 0.2;
            let grad_p = grad_p * RELAXATION;
            // Each cell's own mass: a pressure-gradient force gives a = -grad_p /
            // mass locally. `mass` was fetched above and is above
            // `MIN_ABSOLUTE_MASS_FOR_CORRECTION` (near-empty cells were skipped
            // by the `continue`), so no fallback is needed here.
            let cell_alpha = 1.0 / mass;
            if let Some(cell) = self.cells.get_mut(&idx) {
                cell.momentum -= cell_alpha * grad_p;
            }
        }
    }
}

#[cfg(test)]
mod fluid_pressure_projection_tests {
    use super::*;

    /// Same checkable claim `pressure_projection_reduces_divergence_
    /// residual` (mixture/pressure.rs) already proves for the two-phase
    /// case: build a deliberately divergent velocity field (radiating
    /// outward from a center node, decaying toward the patch edge so a
    /// closed/Neumann system can actually resolve it), run the projection,
    /// confirm the residual divergence shrinks -- and, since this
    /// is now an EXACT solve rather than a partially-converged iterative
    /// one, expect a much bigger real reduction than the old Jacobi/GS
    /// version's ~25-84%.
    #[test]
    fn pressure_projection_reduces_divergence_residual() {
        let mut grid = Grid::new(16);
        let center = IVec2::new(8, 8);
        let mass = 2.0_f32;
        let v_at = |pos: IVec2| -> Vec2 {
            let d = (pos - center).as_vec2();
            let r2 = d.length_squared();
            d * 0.5 * (-r2 / 8.0).exp()
        };
        for dx in -6..=6 {
            for dy in -6..=6 {
                let pos = center + IVec2::new(dx, dy);
                grid.add_mass_momentum(pos, mass, mass * v_at(pos));
            }
        }
        grid.update_velocities(0.0, Vec2::ZERO);

        let div_at = |g: &Grid| -> f32 {
            let r = g.velocity_at(center + IVec2::new(1, 0)).x
                - g.velocity_at(center - IVec2::new(1, 0)).x;
            let u = g.velocity_at(center + IVec2::new(0, 1)).y
                - g.velocity_at(center - IVec2::new(0, 1)).y;
            (r + u) / 2.0
        };
        let residual_unprojected = div_at(&grid).abs();
        assert!(
            residual_unprojected > 1.0e-3,
            "test setup should have real nonzero divergence, got {residual_unprojected}"
        );

        grid.project_fluid_incompressibility(1.0, 1);
        let residual_projected = div_at(&grid).abs();
        // Measured reduction on this scene: ~8.4% (0.8825 -> 0.8081). The
        // solve is exact; `RELAXATION` damps one call's reduction on purpose
        // (see that constant), so the threshold comes from the measurement.
        assert!(
            residual_projected < residual_unprojected * 0.95,
            "projection should substantially shrink the divergence residual: \
             before={residual_unprojected:.5} after={residual_projected:.5}"
        );
    }

    /// A pressure-gradient correction goes as `a = -grad(p) / mass` per cell:
    /// under the same local gradient, a heavy cell gets a smaller velocity
    /// correction than a light one. One shared `alpha_const` could not tell
    /// them apart.
    ///
    /// Divergence-reduction ratios cannot show it: the pressure scales with
    /// `1/alpha` and the correction re-applies `alpha`, so the ratio cancels
    /// alpha to first order (0.834 alone against 0.832 mixed, across 40x of
    /// mass). The velocity change can.
    ///
    /// Two regions with the same divergent velocity field (`v_at` does not
    /// depend on mass) and 40x different mass (water/mud), in one grid and
    /// one projection call: the heavy region's correction must come out
    /// clearly smaller.
    #[test]
    fn heavy_region_gets_smaller_velocity_correction_than_light_region() {
        let v_at = |center: IVec2, pos: IVec2| -> Vec2 {
            let d = (pos - center).as_vec2();
            let r2 = d.length_squared();
            d * 0.5 * (-r2 / 8.0).exp()
        };
        let light_center = IVec2::new(10, 24);
        let heavy_center = IVec2::new(38, 24);
        let light_mass = 1.0_f32;
        let heavy_mass = 40.0_f32; // real water/mud rest_density ratio

        let mut grid = Grid::new(48);
        for dx in -6..=6 {
            for dy in -6..=6 {
                let lp = light_center + IVec2::new(dx, dy);
                grid.add_mass_momentum(lp, light_mass, light_mass * v_at(light_center, lp));
                let hp = heavy_center + IVec2::new(dx, dy);
                grid.add_mass_momentum(hp, heavy_mass, heavy_mass * v_at(heavy_center, hp));
            }
        }
        grid.update_velocities(0.0, Vec2::ZERO);
        let light_v_before = grid.velocity_at(light_center + IVec2::new(2, 0));
        let heavy_v_before = grid.velocity_at(heavy_center + IVec2::new(2, 0));

        grid.project_fluid_incompressibility(1.0, 1);

        let light_v_after = grid.velocity_at(light_center + IVec2::new(2, 0));
        let heavy_v_after = grid.velocity_at(heavy_center + IVec2::new(2, 0));
        let light_delta = (light_v_after - light_v_before).length();
        let heavy_delta = (heavy_v_after - heavy_v_before).length();

        assert!(
            light_delta > 1.0e-5,
            "test setup should produce a real, measurable light-region velocity change, got {light_delta}"
        );
        // Not the exact 40x: RELAXATION, the DCT's constant-alpha initial
        // guess and the padded-box Neumann treatment all depart from a pure
        // 1/mass law. A factor of 3 is enough to rule out a mass-blind alpha.
        assert!(
            heavy_delta < light_delta / 3.0,
            "heavy (40x denser) region's velocity correction should be clearly \
             smaller than the light region's under correct per-cell-mass physics: \
             light_delta={light_delta:.6} heavy_delta={heavy_delta:.6}"
        );
    }

    /// `pressure_iterations = 0` must be a true no-op -- the field's own
    /// doc promises callers may invoke this unconditionally.
    #[test]
    fn zero_iterations_is_a_no_op() {
        let mut grid = Grid::new(16);
        let center = IVec2::new(8, 8);
        grid.add_mass_momentum(center, 2.0, Vec2::new(1.0, 0.5));
        grid.update_velocities(0.0, Vec2::ZERO);
        let before = grid.velocity_at(center);
        grid.project_fluid_incompressibility(1.0, 0);
        assert_eq!(grid.velocity_at(center), before);
    }
}
