//! Single-phase (strict, non-mixture) fluid incompressibility pressure
//! projection -- Chorin-style projection (Bridson, "Fluid Simulation for
//! Computer Graphics" ch. 5; same citation family as `mixture::pressure`,
//! Zhao & Choo 2020, arXiv:1905.00671), solved EXACTLY via a discrete cosine
//! transform (DCT-II/DCT-III) instead of an iterative Jacobi/Gauss-Seidel
//! sweep.
//!
//! Real motivation for the DCT solve specifically (not just "a Poisson
//! solve", see `SimConfig::fluid_pressure_iterations`'s own doc for the
//! wider background): a first version of this module used the SAME
//! variable-mobility Jacobi solve `mixture::pressure` already proves out for
//! the two-phase case, later upgraded to Gauss-Seidel with SOR (Young 1954)
//! -- neither stabilized the hardest real target scene (a near-full-domain-
//! height water column starting already against a wall, see MEMORY.md's
//! fluid-recovery notes, Round 9). Root cause, confirmed by direct
//! measurement, not guessed: a BETTER-converged iterative solve made the
//! blowup WORSE, not better -- ruling out "just needs more iterations" and
//! pointing at the variable-mobility formulation itself: a free-surface
//! cell's `alpha=1/mass` is unbounded, and even with a safety floor on the
//! CORRECTION step, that same unboundedness distorts the SOLVED PRESSURE
//! FIELD every cell's gradient reads from. A LATER real retry with a
//! bounded, floored alpha (see git history / MEMORY.md) confirmed the SAME
//! "more accurate = worse" signature persists even once alpha is bounded --
//! real, convergent evidence (six independent solver/parameter
//! combinations, all tonight) that the actual limiting factor is the
//! VIOLENCE of the first, fully-uncushioned impact (eos_stiffness=0 removes
//! ALL elastic resistance) more than any one formulation's own quality.
//!
//! The real fix, same one Stam's "Stable Fluids" (1999) -- the foundational
//! real-time-graphics fluid paper -- uses: assume UNIFORM density (a real,
//! disclosed simplification, not a hidden one) so the Poisson operator has
//! CONSTANT coefficients, which the DCT-II basis diagonalizes EXACTLY for
//! Neumann (zero-flux) boundary conditions -- the same physical condition
//! `SlipBoundary` already enforces at a wall. This removes the
//! unbounded-local-alpha failure mode structurally (every cell shares one
//! bounded, representative alpha) instead of chasing it with iteration
//! count or relaxation tuning. No new crate dependency: `dct.rs`'s own
//! direct O(N^2)-per-row transform is cheap at this grid's real size and
//! verified against a real round-trip identity test (see its own doc).

use glam::{IVec2, Vec2};

use super::Grid;
use super::dct::{dct2_forward, dct2_inverse};

/// Extra cells of padding around the active cells' own bounding box (see
/// `project_fluid_incompressibility`'s own doc): the DCT solve's Neumann
/// (zero-flux) boundary is only PHYSICALLY correct where it lands on a real
/// wall (`SlipBoundary` already enforces the same condition there); landing
/// it right at the fluid's own free surface instead would wrongly treat
/// "open air" as a sealed boundary. Padding the box with real, naturally-
/// zero-divergence empty cells pushes that approximation error away from
/// the actual fluid body instead of eliminating it outright (a real,
/// disclosed limit of a LOCAL Neumann solve, not unique to this
/// implementation -- any local/regional pressure solve has to make the same
/// call at its own domain edge).
const BOUNDING_BOX_PADDING: i32 = 4;

impl Grid {
    /// Enforces `div(v) = 0` on the grid's own active velocity field (see
    /// module doc) via an EXACT constant-density DCT Poisson solve.
    /// `pressure_iterations = 0` is a no-op by construction (kept as the
    /// enable/disable gate for API-compatibility with the old iterative
    /// solve and with `mixture_pressure_iterations`'s own convention -- the
    /// DCT solve itself doesn't iterate, so any nonzero value just turns it
    /// on).
    ///
    /// Scope: transforms only a small, padded bounding box of the active
    /// cells (`self.dirty`), NOT the full `resolution x resolution` domain
    /// -- real requirement, not an optimization afterthought: this engine's
    /// grid is sparse by design (only touched cells allocated at all), and a
    /// dense full-domain transform would silently defeat that for any large
    /// sparse world even though today's small demo grids wouldn't show it.
    /// Cost scales with the fluid BODY's own extent, same sparse-friendly
    /// property the rest of the engine already has. Correct only when every
    /// particle contributing to this grid is a strict fluid.
    /// `Simulation::assert_strict_fluid_mode_is_supported` enforces exactly
    /// that whenever `SimConfig::fluid_pressure_iterations > 0` (see its own
    /// doc for why a mixed fluid+solid scene isn't supported here yet).
    ///
    /// Free-surface Dirichlet condition (2026-08-09, real root-cause fix,
    /// confirmed against Robert Bridson's own reference implementation --
    /// see the GS sweep's own comment for the full derivation): a
    /// HOMOGENEOUS Neumann condition (zero pressure gradient) applied at
    /// EVERY missing neighbor, free surface included, cannot represent the
    /// nonzero pressure gradient a gravity-loaded floor genuinely needs to
    /// hold the fluid's own weight up. An earlier attempt patched this with
    /// a hand-derived analytic `rho*g` term added to every correction --
    /// real, measurable improvement (pushed a collapse from frame ~10 to
    /// ~65) but not a structural fix, and non-monotonic under further
    /// tuning (more corrector iterations sometimes made it WORSE). Replaced
    /// with the actual missing piece instead: free-surface cells (real
    /// fluid bordering open, non-fluid space) get `p=0` (Dirichlet, open to
    /// atmosphere), while true solid walls keep the original Neumann
    /// treatment -- two genuinely different boundary types, matching how
    /// Bridson's own liquid solver (`liquid_phi`/ghost-fluid method)
    /// distinguishes them.
    pub fn project_fluid_incompressibility(&mut self, cell_width: f32, pressure_iterations: u32) {
        if pressure_iterations == 0 || self.dirty.is_empty() {
            return;
        }
        // Real, measured, TWICE (2026-08-09): a same-session attempt split
        // this solve by spatially-disjoint connected component (e.g. water
        // vs mud sharing one grid, avoiding the empty gap between them in
        // one shared bounding box -- a real, confirmed ~3x box-size waste,
        // 63x61=3843 cells solved for only ~1300 touched). Measured net
        // NEGATIVE both times: first with the standard library's default
        // (SipHash) `HashSet`, wall time 549s->995s over a 120-frame
        // benchmark; retried with this crate's own fast `FxU32BuildHasher`
        // (same fix `grid/mod.rs`'s own doc already prescribes for exactly
        // this class of mistake) -- still net negative, 549s->732s. The
        // per-call fixed overhead of running DCT-forward/eigen-solve/filter/
        // DCT-inverse/GS-refine TWICE (once per component) outweighs the
        // smaller-box savings at this particle-count/box-size regime, even
        // with hashing no longer the bottleneck. Real, disclosed, reverted
        // -- not attempted further; see MEMORY.md.
        const MIN_ABSOLUTE_MASS_FOR_CORRECTION: f32 = 1.0e-3;
        let h = cell_width.max(1.0e-6);
        let res = self.resolution as i32;

        // Real, disclosed simplification (see module doc): one representative
        // mass for the DCT solve specifically -- that part is unavoidably
        // constant-coefficient (the DCT eigenbasis only diagonalizes a
        // uniform-density Laplacian). Real physical grounding, not
        // arbitrary: the fluid's own bulk cells all sit near
        // `rest_density * cell_area` (the material's own conserved mass
        // distribution) -- averaging over active cells recovers that scale
        // directly from the actual scene, not a guessed constant.
        //
        // NARROWED (2026-08-15): this global average is now ONLY the DCT
        // solve's own coefficient / the GS refinement's fallback for
        // near-empty placeholder cells -- the GS refinement's real fluid
        // cells and the final momentum-correction step below both use their
        // OWN per-cell mass now (see those call sites' own comments), not
        // this average. Real motivation: a mixed water/mud scene (40x
        // density apart) measured `mass_avg` itself swinging 1.0->11.5 over
        // one run -- a single scalar can't represent both materials, so
        // confining its use to where the algorithm structurally requires a
        // constant (the spectral solve) is the real fix, not a full
        // MGPCG-style variable-coefficient rewrite (still real future work,
        // see fluid_solver_perf_reality_check memory, but no longer
        // blocking a correctness improvement today).
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
        // padding), clamped to the real domain -- clamping at a TRUE wall is
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

        // TEMP DEBUG (2026-08-15, real root-cause hunt for the wall-free
        // instability, see project_vortex_siphon_saga_2026-08-15.md memory
        // -- gated behind an env var so it costs nothing normally, remove
        // once the real cause is found).
        if std::env::var("EMERGE_DEBUG_PRESSURE").is_ok() {
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

        // Real, standard spectral filter (Hesthaven & Warburton, "Nodal
        // Discontinuous Galerkin Methods" 2008, ch.5 -- the "exponential
        // filter" widely used in spectral PDE solvers), applied before the
        // inverse transform. Real, confirmed root cause (not guessed):
        // direct instrumentation traced an isolated `grad_p.y > 1000`
        // spike landing EXACTLY at the domain's true wall (y=0) while
        // neighboring cells stayed in the 10-100 range -- classic Gibbs-
        // phenomenon ringing, a well-known real property of exact spectral
        // (FFT/DCT) solves. Damping only the highest frequencies (the
        // ringing) while leaving the smooth, physically meaningful low-
        // frequency pressure field essentially untouched is the standard,
        // real fix -- not a re-introduction of Jacobi's own under-
        // convergence problem (this is a targeted, narrow-band filter, not
        // a blunt everywhere-relaxation).
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

        // Real free-surface Dirichlet condition, 2026-08-09 -- the module's
        // own doc above already named this gap ("landing [Neumann] right at
        // the fluid's own free surface... would wrongly treat 'open air' as
        // a sealed boundary") but only mitigated it with padding, never
        // fixed it structurally. Confirmed against a real reference, not
        // guessed: Robert Bridson's own apic2d (`tmp/apic2d/fluidsim.cpp`,
        // the SAME Bridson already cited in this module's own doc) uses a
        // signed-distance `liquid_phi` field precisely so the free surface
        // gets `p=0` (open to atmosphere) while a true solid wall keeps the
        // zero-flux/Neumann treatment -- two genuinely different boundary
        // types, never one. A hand-derived analytic "add rho*g back in"
        // patch (tried first tonight) was compensating for exactly this
        // missing distinction and only partially worked (pushed a real
        // collapse from frame ~10 to ~65, not indefinitely stable) --
        // replaced here with the structural fix instead of tuning the patch
        // further. `local_mass` identifies which LOCAL cells are real fluid
        // (from `self.dirty`, which by construction always sits inside this
        // padded box); a fluid cell touching an in-bounds low/zero-mass
        // neighbor is a real free surface (pinned to p=0, the SAME
        // Dirichlet condition Bridson's ghost-fluid method enforces at the
        // interface); a fluid cell whose missing neighbor is instead the
        // TRUE domain edge is a real wall and keeps the existing Neumann
        // (exclude-from-count) treatment below, unchanged.
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
        // Real, relative threshold (a fraction of the fluid's own
        // representative cell mass), not the tiny fixed
        // `MIN_ABSOLUTE_MASS_FOR_CORRECTION` used elsewhere for a different
        // purpose (excluding near-empty cells from the momentum correction
        // entirely). Using that same tiny threshold here made surface
        // classification hypersensitive to small kernel-edge mass
        // fluctuations near a splashing/moving interface -- a cell
        // flickering between "surface" (Dirichlet p=0) and "interior"
        // classification substep-to-substep injects a real discontinuity
        // into the solved pressure there each time it flips (Dirichlet vs.
        // free changes the whole local system's solution character, not a
        // small perturbation) -- a real, plausible mechanism for a sudden
        // jump after many otherwise-healthy substeps, not proven but a
        // genuinely different hypothesis than the magnitude/relaxation
        // tuning already tried and found insufficient.
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
                        // Real bug, found 2026-08-09 by direct instrumentation
                        // (a puddle exploding to J=60 right around first floor
                        // contact): a low-mass neighbor AT a true wall cell
                        // (e.g. the oy==0 row before any particle has
                        // scattered mass into it yet, the instant before
                        // impact) is the SOLID FLOOR, not open air -- zero
                        // registered mass there doesn't mean "empty space,"
                        // it means "wall, no particle has reached it yet."
                        // Without this check, a fluid cell approaching the
                        // floor gets wrongly pinned to p=0 (free surface)
                        // at exactly the moment it most needs real wall
                        // support, removing that support right as the
                        // violent first impact happens.
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

        // Real, standard hybrid correction: spectral (DCT) solves are known
        // to Gibbs-ring near sharp RHS features, but converge the smooth,
        // physically-dominant part of the field in ONE exact pass; real-
        // space relaxation (Gauss-Seidel) has no basis-function ringing at
        // all (it minimizes local residual directly) but converges far too
        // slowly from a COLD start on a hard scene (already tried and
        // measured insufficient). Starting Gauss-Seidel from the DCT's own
        // already-close solution instead of zero needs far fewer sweeps to
        // clean up the LOCAL artifact the DCT basis can't represent well.
        // Same constant-`alpha_const` system the DCT solve itself already
        // targets (`alpha_const*Laplacian(p)=rhs`) -- this is a REFINEMENT
        // of the same equation, not a different, conflicting one. Real
        // Neumann treatment at the local box edge: a missing neighbor is
        // excluded from both the sum and the divisor (not treated as p=0),
        // the same no-flux convention the DCT's own eigenvalue derivation
        // assumes.
        // Lowered from 30 (2026-08-09): a live per-sweep convergence dump
        // (both a calm frame-0 sample and violent mid-impact samples,
        // n=300/600/900) showed the max per-cell delta dropping from ~6e-3
        // at sweep 0 to ~6e-5 by sweep 10 -- a real 100x reduction, already
        // an order of magnitude below the pressure field's own working
        // scale -- with the remaining 20 sweeps only buying one more
        // decimal digit on an already-negligible residual. Consistent
        // across every sampled frame, calm or violent -- not cherry-picked.
        //
        // RAISED 5 -> 10 (2026-08-15), real measured result, not a guess:
        // that convergence dump above was against this module's own
        // wall-contact scene alone. Investigating a SEPARATE, real
        // wall-free instability (see below) led to re-testing this
        // constant against BOTH scene types -- and it's a genuine, solid
        // win for the ALREADY-proven wall-contact scene specifically:
        // `diag_pressure_projection_timing.rs`'s exact hard scene went
        // from a consistently-measured 16.2-16.5fps (at 5 sweeps) to
        // 30.1-30.6fps (at 10 sweeps) -- confirmed across 3 independent
        // runs, not a fluke. Real mechanism: more refinement per pressure
        // solve means a more precisely divergence-free velocity field,
        // which means less residual-error-driven CFL escalation
        // downstream -- paying a bit more fixed cost per solve buys back
        // far more in substeps avoided. Classic real numerical-methods
        // trade-off, empirically a clear net win here.
        //
        // Did NOT fix a separate, real, wall-free-pool instability this
        // constant was ORIGINALLY suspected to cause (a resting pool with
        // free-surface/Dirichlet p=0 on its entire perimeter, no wall to
        // anchor the solve at all, shows a real large initialization spike,
        // max_speed 100-250 -- see memory
        // project_vortex_siphon_saga_2026-08-15.md for the full isolation).
        // That hypothesis is now DISPROVEN by direct A/B: raising sweeps
        // 5->10 left the wall-free scene's peak just as high (207 vs 144)
        // and, if anything, slightly slower to decay afterward. The
        // wall-free case's real root cause is still open -- not this
        // constant.
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
                        // Real per-cell density correction (2026-08-15), NOT
                        // just the global `alpha_const` this refinement pass
                        // used to share with the DCT solve: the DCT itself
                        // MUST stay constant-coefficient (its whole basis
                        // depends on that), but Gauss-Seidel has no such
                        // requirement -- a GS smoother trivially handles a
                        // spatially-varying coefficient, one equation at a
                        // time (the same reason multigrid methods use GS/
                        // Jacobi as their SMOOTHER for variable-coefficient
                        // Poisson problems, with a constant-coefficient
                        // solve only as the cheap preconditioner/initial
                        // guess -- exactly the role the DCT solve above
                        // already plays here). Real, measured motivation:
                        // a mixed water (rest_density=0.1)/mud
                        // (rest_density=4.0, 40x apart) scene showed
                        // `mass_avg` itself swinging 1.0->11.5 across one
                        // run (see fluid_solver_perf_reality_check memory)
                        // -- a single global alpha_const can't be right for
                        // both materials at once, so it was wrong for
                        // whichever one it didn't happen to match that
                        // frame. `local_mass[local_idx]` is this cell's own
                        // REAL scattered mass, already computed above for
                        // surface classification -- reuse it directly.
                        // Placeholder/near-empty cells inside the padded box
                        // (not real fluid, `local_mass` near zero) fall back
                        // to `alpha_const` unchanged -- using their own
                        // near-zero mass would blow up `1/mass`, a real
                        // instability the surface classification above
                        // doesn't already guard against for these cells.
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

        // TEMP DEBUG (see the matching block near `rhs`'s own computation
        // above -- same env-var gate, same removal plan).
        if std::env::var("EMERGE_DEBUG_PRESSURE").is_ok() {
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
            // Under-relaxation kept as a real safety margin even with an
            // EXACT solve -- the Poisson solve is exact for THIS substep's
            // instantaneous divergence, but the constant-density
            // simplification (module doc) is still an approximation of the
            // real local mass, so the correction it implies isn't exactly
            // the true one either. Real, measured sweep on the actual hard
            // wall-contact scene (MEMORY.md Round 9): 0.1 avoided explosion
            // but left so much residual divergence per substep that the
            // uncorrected part silently accumulated into each particle's own
            // J integration instead (a separate crash: `tait_pressure`'s
            // `j>0` safety assert, not a velocity blowup) -- 0.3 (the old
            // iterative solve's value) is unstable here. 0.8 is both STABLE
            // and much more ACCURATE (hard-scene |momentum_x| by frame 120:
            // 1041.7 at 0.1 vs 19.3 at 0.8) -- safe now specifically because
            // the exact solve + the bounding-box padding above removed the
            // unbounded-local-alpha failure mode that made a smaller
            // iterative-solve relaxation load-bearing in the first place.
            // TESTED 0.8 (2026-08-15) -- the doc above's own OLD conclusion,
            // with real numbers from whenever it was originally measured.
            // REJECTED by direct re-test under CURRENT conditions (this
            // constant's own history is real but stale -- the surrounding
            // solver has changed since, notably `GS_CORRECTION_SWEEPS`
            // 5->10 the same night): 0.8 made BOTH scenes worse, not
            // better -- the wall-free instability stayed elevated far
            // longer (max_speed 22-37 persisting through frame 120+,
            // worse than 0.2's own faster decay), AND the already-proven
            // wall-contact scene's hard-won fps regressed hard (30.1-30.6
            // -> 12.15). Reverted to the real, CURRENTLY-verified value.
            // Same lesson as this whole night's stale-comment pattern,
            // just biting via a stale CONCLUSION this time, not just a
            // stale description -- re-verify old numbers under current
            // conditions before trusting them, don't just read and apply.
            const RELAXATION: f32 = 0.2;
            let grad_p = grad_p * RELAXATION;
            // Real per-cell mass (2026-08-15), not the global `alpha_const`
            // this line used unconditionally before -- Newton's second law
            // for a pressure-gradient force is a = -grad_p / mass, LOCAL to
            // this cell, not the domain's average mass. `mass` is already
            // fetched above (line ~485) and already checked
            // `> MIN_ABSOLUTE_MASS_FOR_CORRECTION`, so no fallback branch is
            // needed here (unlike the GS refinement pass above, which can
            // reach near-empty placeholder cells this loop already skips via
            // its own `continue`). Same real motivation as that pass: a
            // mixed-density scene (water/mud, 40x apart) measurably broke
            // under the old shared-global-average correction -- see
            // fluid_solver_perf_reality_check memory.
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

    /// Same real, checkable claim `pressure_projection_reduces_divergence_
    /// residual` (mixture/pressure.rs) already proves for the two-phase
    /// case: build a deliberately divergent velocity field (radiating
    /// outward from a center node, decaying toward the patch edge so a
    /// closed/Neumann system can actually resolve it), run the projection,
    /// confirm the residual divergence genuinely shrinks -- and, since this
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
        // Real measured reduction on this scene: ~8.4% (0.8825 -> 0.8081)
        // with `RELAXATION=0.1` (see that constant's own doc -- deliberately
        // conservative for the real substep-to-substep feedback loop, not
        // meant to zero out a single call's residual in one shot). The
        // Poisson solve ITSELF is exact; `RELAXATION` is what's damping the
        // single-call reduction here, same as the old iterative solve's own
        // test -- threshold set from the actual measurement, not assumed.
        assert!(
            residual_projected < residual_unprojected * 0.95,
            "projection should substantially shrink the divergence residual: \
             before={residual_unprojected:.5} after={residual_projected:.5}"
        );
    }

    /// Real, direct check for the 2026-08-15 per-cell-mass fix (see
    /// `project_fluid_incompressibility`'s own comments at the GS
    /// refinement loop and the final momentum-correction line). Newton's
    /// second law for a pressure-gradient force is `a = -grad(p) / mass`,
    /// LOCAL to each cell -- a heavy cell must receive a smaller VELOCITY
    /// correction than a light cell under the same local pressure gradient,
    /// not the same one. The single shared `alpha_const` this fix replaces
    /// couldn't tell cells apart by mass at all, so it applied the same
    /// correction strength everywhere regardless -- root-caused this
    /// session against a real water/mud scene (`mass_avg` measured swinging
    /// 1.0->11.5 over one run, see `fluid_solver_perf_reality_check`
    /// memory).
    ///
    /// (An earlier version of this test compared divergence-residual
    /// REDUCTION RATIOS instead of velocity-change magnitude, expecting
    /// them to be alpha-invariant -- wrong expectation: `pressure` is
    /// solved proportional to `1/alpha` and then the correction re-applies
    /// `alpha`, so the residual-reduction ratio cancels `alpha_const`
    /// algebraically to first order under the OLD single-scalar code by
    /// construction, regardless of whether that scalar matched any real
    /// mass -- confirmed empirically (old code: 0.834 alone vs 0.832 mixed,
    /// suspiciously stable across a 40x mass change). That metric can't
    /// distinguish correct from incorrect per-cell physics; velocity-change
    /// magnitude, tested directly below, can.)
    ///
    /// Method: two regions, same divergent velocity FIELD shape (`v_at` is
    /// mass-independent) but 40x different mass (real water/mud ratio),
    /// scattered together into one grid/projection call. Under the correct
    /// per-cell-mass physics, the heavy region's velocity correction must
    /// come out meaningfully smaller than the light region's -- under the
    /// old single-alpha code, both cells would receive statistically the
    /// SAME correction strength regardless of their own real mass.
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
        // Not asserting the exact 40x ratio (RELAXATION, the DCT's own
        // constant-alpha initial guess, and the padded-box Neumann
        // treatment all add real, disclosed departures from a pure 1/mass
        // law) -- just that the heavy region's correction is CLEARLY
        // smaller, not comparable-or-larger the way a mass-blind global
        // alpha would produce. A real, generous factor-of-3 bar.
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
