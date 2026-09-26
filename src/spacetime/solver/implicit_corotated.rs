//! Opt-in implicit (Newton-CG) grid-velocity update for scenes built
//! entirely from the shared Corotated elastic branch (`MaterialModel::
//! corotated_lame_params`) -- DruckerPrager (sand), Corotated, VonMises,
//! Rankine, DruckerPragerMuI. Real, disclosed strategy (Klar 2016 operator
//! split): solve the velocity field treating every particle as ordinary
//! Corotated elasticity via Newton-CG at the FULL frame `dt` (one solve
//! instead of thousands of CFL-limited explicit substeps), then apply each
//! particle's own REAL, unmodified plastic return-mapping
//! (`MaterialModel::update_particle`, via the existing `gather_grid_to_
//! particles` G2P pass -- no reimplementation) once at the end.
//!
//! Real measured motivation (2026-09-10, `tests/scratch_implicit_mpm_
//! stage3_drucker_prager_multi_particle.rs`, a STANDALONE synthetic
//! Newton-CG benchmark scaled to basic_sand's own real ~2016-particle
//! count, NOT this file's real production wiring): 10.3x wall-clock
//! speedup at that scale, real operator-split plastic-correction error
//! under 1% at far coarser correction frequency than used here (1
//! correction per big-step here, vs. 100 corrections/big-step in that error
//! measurement). That number predates every correctness fix landed since
//! (Kirchhoff-vs-Piola, mass-normalized tolerance, wall-frozen DOFs,
//! eigenvalue-clamp SPD projection) and, per the REAL scale-dependent
//! limitation disclosed below, does not currently hold for this module's
//! actual production wiring at `basic_sand`'s real scene size.
//!
//! Deliberately narrow v1 scope, not a silent partial application:
//! `eligible` below requires the WHOLE active scene to qualify (every
//! particle's material on the shared elastic branch, no rods/grains/
//! contact/mixture/ASFLIP/Cundall/pressure-projection/pinned particles/
//! sleeping particles). Any scene that doesn't qualify falls back to the
//! normal explicit substep loop,
//! byte-identical to before this module existed -- turning `SimConfig::
//! implicit_corotated_elastic` on for an unsupported scene is always safe,
//! just inert. Convergence itself is ALSO checked before committing: a
//! Newton-CG solve that fails to reach its relative tolerance leaves
//! `self.particles`/`self.grid` untouched and reports ineligible, so the
//! caller's normal substep loop runs instead -- this is a real, working
//! fallback, not merely a static scope guard.
//!
//! **STATUS AS OF 2026-09-12 -- CORRECT BUT INERT AT REAL SCALE, AND THE
//! REASON IS NOW PROVEN, NOT OPEN. Read this before trusting anything above
//! about "verified working": this module is correct and matches the
//! explicit baseline at the correctness suite's SMALL scale (256
//! particles), but at `basic_sand.rs`'s ACTUAL production scale (~1008
//! particles, real E=15MPa DruckerPrager) it still does NOT converge --
//! every measured production frame falls back to the normal explicit
//! substep loop. The two numbers are not a contradiction (see "the scale
//! gap" below for why), but they describe genuinely different outcomes and
//! must not be conflated: this module has never yet delivered a real
//! performance win on the scene it was built to speed up, and per the
//! root-cause analysis below (points 7-14), delivering one is not simply a
//! matter of continuing to tune this solver -- see "why this closes the
//! investigation" for what a real fix would actually require.**
//!
//! **Small-scale correctness (still real, still verified):** an already-
//! settled or slowly-consolidating pile at the correctness suite's scale
//! (256 particles) matches the explicit baseline closely (`tests/
//! implicit_corotated_substep.rs`'s `implicit_matches_explicit_for_an_
//! already_settled_pile`, 0.0000 grid-cell drift). A violent impact
//! (dropped from height) at that same small scale currently drifts ~3.05
//! grid cells from the explicit baseline over 20 frames (`violent_impact_
//! diverges_more_than_settled_pile_a_real_disclosed_limitation`) -- down
//! from ~9.7 before wall-adjacent-DOF freezing was added, and briefly
//! 0.0000 with an intermediate trust-region design, but back up to ~3.05
//! after later fixes (Gauss-Newton merit switch, Jacobi preconditioning)
//! changed the solver's exact trajectory again; still comfortably under
//! that test's own tolerance, not re-investigated further tonight.
//!
//! **The scale gap -- basic_sand's real production scene still does not
//! converge, and this is the actual, currently open problem:** at 1008
//! particles / real E=15MPa, Newton reduces the (correctly-measured, see
//! below) residual by a real, substantial amount before stalling short of
//! `RELATIVE_TOLERANCE`, never reaching convergence within one frame
//! (`tests/scratch_implicit_corotated_real_fps_measurement.rs` with
//! `EMERGE_IMPLICIT_DIAG=1`). This session's real, sequential findings,
//! roughly in the order they were found (each one real and kept, even
//! though the final problem remains open):
//!
//! 1. Root-caused via controlled isolation (`tests/scratch_implicit_
//!    stiffness_vs_scale_isolation.rs`) that the failure tracks problem
//!    SIZE (DOF count/domain extent), not material stiffness: the
//!    correctness suite's softer modulus fails just as fast at the LARGE
//!    scene, while real E=15MPa at the SMALL scene makes real progress for
//!    several iterations before eventually failing.
//! 2. Proved (via a real dense Cholesky direct solve, `direct_solve` --
//!    kept, `#[cfg(test)]`, see its own doc) that this was NEVER a linear-
//!    solver-quality problem: an EXACT solve of the same linearized system
//!    hit the identical wall a "fully converged" CG did.
//! 3. Found and fixed a real bug in the trust-region model construction
//!    (`model_residual`/`model_jacobian_vector_product`, both kept
//!    `#[cfg(test)]`): the true production `residual` (Kirchhoff stress,
//!    spatial gradient, exact exponential-map `F`) is NOT the literal
//!    calculus gradient of any simple `Psi(F_new(v))` energy under a
//!    multiplicative `F` update -- the correct model gradient needs Piola
//!    stress paired with `F_n^T*grad`, scaled by `dt`. Verified by hand on
//!    a minimal single-particle case (`direct_solve_end_to_end_with_real_
//!    material_parameters` and the `trust_region_consistency_tests`
//!    module), not just derived.
//! 4. Found and fixed a real structural bug in the convergence CRITERION
//!    itself (`free_mass_normalized_norm`'s own doc has the full
//!    quantitative story): 31% of grid DOFs are `wall_frozen` and their
//!    residual contribution can never be reduced by this free-DOF-only
//!    solve, so a naive frozen+free residual norm made `RELATIVE_
//!    TOLERANCE` mathematically unreachable regardless of solver quality.
//! 5. Found and fixed a real scale-mismatch bug: using `model_residual`
//!    (Piola/`dt`-scaled) as Steihaug-CG's own gradient corrupted the
//!    trust region's radius calibration relative to what the REAL
//!    residual's landscape needed (confirmed via a direct probe: stepping
//!    along the real residual's own negative-gradient direction by a tiny
//!    `eps` DID improve it, in a window the old radius never explored).
//!    Fixed by switching `newton_solve` to a Gauss-Newton trust region
//!    (Nocedal & Wright 2nd ed. ch.10; Moré 1978) using the REAL `residual`
//!    directly as the gradient, `model_jacobian_vector_product` only as an
//!    approximate curvature operator, and `0.5*||residual||^2` (not
//!    energy) as the ratio-test merit. Result: residual reduction went
//!    from 0% (immediate stall) to 97.7%.
//! 6. Added Jacobi preconditioning to `steihaug_cg` (`jacobi_diagonal`,
//!    same lumped mass+stiffness diagonal this module's earlier, now-
//!    removed CG path used) after confirming the unpreconditioned trust
//!    region was itself the limiter. Result: 97.7% -> 99.6% reduction.
//!    Confirmed NOT an iteration-budget problem: raising `MAX_CG_ITERS`
//!    from 150 to 400 (a separate constant from `MAX_NEWTON_ITERS` after
//!    this same investigation) reproduced the identical stall bit-for-bit.
//!
//! **Root cause now IDENTIFIED AND PROVEN (2026-09-12), remedy PROVEN NOT
//! PERFORMANCE-VIABLE -- this investigation is closed, not abandoned
//! mid-hypothesis.** Continuing past the 99.6% stall above, a further
//! session ruled out every remaining solver-quality explanation with real,
//! run experiments before finding the true cause:
//!
//! 7. Found and fixed a real (if ultimately non-causal) bug: `steihaug_cg`
//!    never projected its own operator output (`model_jacobian_vector_
//!    product`) back onto the free-DOF subspace each iteration -- only
//!    `newton_solve` zeroed the FINAL `p` once, after the fact. A node
//!    sharing a particle with a `wall_frozen` neighbor gets a real, nonzero
//!    force response even when the search direction `d` is exactly zero
//!    there, and that leak was silently corrupting `r`/the preconditioner/
//!    `beta` from the second CG iteration onward (confirmed live:
//!    `hd_frozen_norm` up to ~465 with `d_frozen_norm` pinned at exactly
//!    0.0 after adding the per-iteration projection -- matching Ziran's own
//!    real production practice, `tmp/ref_ziran_implicit_mpm.md`). Kept as
//!    a real, permanent, unconditional correctness fix -- but the SAME
//!    real `basic_sand` scene stalled at a bit-for-bit identical residual
//!    (317) before and after, ruling this out as the cause of the stall.
//! 8. Directly measured whether `model_jacobian_vector_product`'s
//!    approximation (built from the LINEAR `model_deformed_f`, Piola
//!    stress) was simply too different from the TRUE residual's own
//!    Jacobian (built from the exponential-map `deformed_f`, Kirchhoff
//!    stress) for Steihaug-CG's inexact-Newton theory to hold. It genuinely
//!    was: a finite-difference JVP of the real `residual` (`real_residual_
//!    jvp_fd`, kept `#[cfg(test)]`) showed the approximation UNDERESTIMATES
//!    the true local sensitivity by ~62-70x in the free residual's own
//!    direction (`cos_similarity`~0.9999 -- same direction, wildly
//!    different scale). Swapping the EXACT finite-difference Jacobian in as
//!    `steihaug_cg`'s own curvature, though, still stalled (r_norm floor
//!    moved from 317 to 276, same collapse signature) -- ruling this out
//!    too, even though the underestimate itself was real.
//! 9. Measured the true Jacobian's own symmetry (a genuine energy gradient
//!    would guarantee it; `residual`'s Kirchhoff/spatial-gradient
//!    convention has no such guarantee, and Stage 0 already proved raw
//!    Kirchhoff's own derivative is asymmetric): relative asymmetry came
//!    back small (0.1%-6% across several probes) -- real, but far too
//!    small to explain a total stall on its own. Ruled out.
//! 10. Measured whether the stuck residual was concentrated on nodes
//!     sharing a particle with a `wall_frozen` neighbor -- the leading
//!     hypothesis carried over from the previous session. It was NOT:
//!     142 wall-adjacent free nodes carried r_norm~140-188 while 175
//!     purely-interior free nodes carried a COMPARABLE OR LARGER
//!     ~204-240 -- no concentration near the wall. This specific,
//!     previously-unproven hypothesis is now empirically falsified, not
//!     just superseded.
//! 11. With every solver-quality and boundary-coupling explanation ruled
//!     out by direct measurement, tested the one thing left: does the SAME
//!     exact configuration (particles, `v_n`, `wall_frozen`, unchanged)
//!     converge if Newton only has to cover a FRACTION of the full frame
//!     `dt`? This is real, standard nonlinear-FEM practice (load/time
//!     stepping) for exactly this failure signature (a genuine local
//!     minimum of the Gauss-Newton merit under one enormous step) -- and it
//!     is the real, confirmed answer: a `dt`-divisor sweep on the identical
//!     stuck state converged cleanly (99.76% reduction) at `dt/400`-
//!     `dt/500`, and reproducibly failed at `dt/300` and every larger
//!     fraction tested (2, 5, 10, 20, 50, 100, 150, 200, 300). The stall is
//!     real, provable, single-giant-Newton-step Gauss-Newton stagnation --
//!     confirmed positively (a fix exists and works), not just by
//!     elimination.
//!
//! **Follow-up, same session: why does a near-identical SYNTHETIC benchmark
//! (`stage3_dp_multi_particle_real_wall_clock_speedup_vs_real_explicit`,
//! N=2025, 10.3x speedup) converge fine while this REAL N=1008 production
//! scene does not, given fewer real particles should if anything be
//! easier?** Three further real, measured, disclosed findings, still not
//! fully resolved:
//!
//! 12. The synthetic benchmark has NO concept of a wall/frozen DOF
//!     anywhere -- a free-floating elastic system under uniform force.
//!     Directly tested: does the SAME real stuck configuration converge if
//!     EVERY node is treated as free (no `wall_frozen` at all)? It does
//!     NOT -- ruling out the mere STRUCTURAL PRESENCE of Dirichlet DOFs
//!     (distinct from finding 10's "not concentrated near the wall" --
//!     this tests removing the constraint entirely, a stronger claim, also
//!     falsified).
//! 13. The synthetic benchmark also uses ONE identical, mildly-deformed
//!     `f_n` (`det~1.02`) for every particle -- never a real, heterogeneous,
//!     history-dependent settled-pile state. Measured the real scene's own
//!     `f_n` statistics first rather than assuming: `min_j=0.9994,
//!     max_j=1.0000` -- i.e. this real settled pile is barely deformed at
//!     all yet, actually CLOSER to identity and less variable than the
//!     synthetic benchmark's own uniform value. Directly tested forcing
//!     every particle to `f_n=IDENTITY` (real `wall_frozen` kept intact):
//!     still did NOT converge. Both F_n heterogeneity and F_n magnitude are
//!     ruled out.
//! 14. Measured the real per-node mass distribution and found a genuine,
//!     real structural fact: at least one grid node (a quadratic-kernel
//!     stencil corner touched by only one particle's near-zero edge
//!     weight) carries essentially ZERO accumulated mass alongside nodes
//!     carrying O(1) mass -- an astronomically ill-conditioned mass
//!     distribution the synthetic benchmark's regular lattice may never
//!     produce. This is a real, plausible contributor to numerical
//!     ill-conditioning (and a real, independent explanation for why the
//!     EXACT Cholesky solve in finding 2 hit the identical wall -- a
//!     near-singular mass matrix defeats any linear solver equally,
//!     regardless of technique). Directly tested: freezing every node
//!     with mass below `1e-4` (6 of 462 nodes) still did NOT converge --
//!     so this real, measured pathology is not SUFFICIENT on its own
//!     either, though it has not been ruled out as a contributing factor
//!     (only as a sufficient standalone fix).
//!
//! **Honest state of the real-vs-synthetic discrepancy: still open.**
//! Seven real hypotheses (findings 7-14, seven, not the six that closed
//! the dt-stepping question) have each been measured and found wanting --
//! this is a genuinely harder discrepancy than any single mechanism found
//! so far explains, not evidence the search was sloppy. A future session
//! picking this up should look for a factor combining several of these
//! (e.g. near-zero-mass nodes SPECIFICALLY where they also sit near
//! `wall_frozen` nodes) rather than another single-variable isolation.
//!
//! **Why this closes tonight's investigation instead of opening a path
//! forward:**
//! the fix that provably works is not performance-viable at this scale.
//! Converging needs ~350-400 implicit sub-steps per frame (vs. ~2263 raw
//! explicit substeps today -- only a ~6x reduction in STEP COUNT), and one
//! real Newton-CG solve at this scale measured ~9.5-10.5ms wall-clock --
//! projecting to ~4.2-4.7 SECONDS per frame, roughly 25-30x SLOWER than the
//! ~145-180ms explicit baseline this module exists to beat. (This
//! projection assumes each sub-step costs about the same as the one
//! measured -- a real, disclosed simplification, not a claim every one of
//! 400 sub-steps was individually timed; the margin against the explicit
//! baseline is wide enough that this simplification cannot flip the
//! conclusion.) Load-stepping fine enough to converge and coarse enough to
//! win would need a per-solve cost roughly 2 orders of magnitude below
//! what this Newton-CG implementation currently achieves -- a different,
//! much larger undertaking (e.g. a real multigrid preconditioner, or
//! abandoning per-frame Newton-CG for a fundamentally cheaper scheme) than
//! anything scoped so far, not a tuning pass on the current design.
//!
//! **Practical consequence, unchanged in kind, now understood in full:**
//! `implicit_corotated_eligible` and `newton_solve`'s own convergence check
//! both trigger the real, safe explicit fallback whenever the full-frame
//! solve doesn't converge, so `SimConfig::implicit_corotated_elastic`
//! remains SAFE to enable on any scene -- it is INERT (falls back every
//! frame, no crash, no wrong physics) at `basic_sand`'s real production
//! scale, and now for a proven, understood reason rather than an open
//! question. It has not delivered, and per the analysis above is not
//! expected to deliver without substantially different solver machinery, a
//! real measured speedup on that scene.
//!
//! **Diagnostic infrastructure kept, not deleted, despite being unused in
//! production right now** (each gated `#[cfg(test)]`, each with its own
//! doc explaining why it's still worth having on hand for whatever
//! investigation comes next): the dense Cholesky direct-solve chain
//! (`direct_solve`, `assemble_free_dof_system`, `particle_stiffness_block`,
//! `cholesky_solve`, `free_dof_map`, `grad_basis`, `build_particle_
//! matrices`) that proved this was never a linear-solver problem; the
//! eigenvalue-clamp PSD projection chain in `materials::utils`
//! (`corotated_kirchhoff_dtau_dl_psd_matrix`, `spd_project_symmetric_4x4`,
//! `jacobi_eigen_symmetric_4x4`, `corotated_elastic_energy_density`) that
//! proved the projection method was never the bottleneck either;
//! `model_residual`/`total_energy`, the verified-consistent energy
//! formulation `model_jacobian_vector_product` (still real production
//! code) is checked against; and `real_residual_jvp_fd`, the exact-but-
//! expensive finite-difference oracle that proved the approximate
//! curvature's own error was real but non-causal.

use glam::{IVec2, Mat2, Vec2};

use super::Simulation;
use super::projection::apply_boundary_conditions_to_grid;
use crate::grid::kernel::{axis_weights_derivative, quadratic_weights};
#[cfg(test)]
use crate::materials::utils::{
    corotated_elastic_energy_density, corotated_kirchhoff_dtau_dl_psd_matrix,
};
use crate::materials::utils::{corotated_elastic_stress, corotated_elastic_stress_jvp};
use crate::transfer::{G2PParams, gather_grid_to_particles};

/// Real max Newton iteration count and relative tolerance -- same values
/// `stage3_dp_multi_particle_real_wall_clock_speedup_vs_real_explicit`
/// used at basic_sand's own real scale, not guessed.
const MAX_NEWTON_ITERS: usize = 150;
/// `steihaug_cg`'s own, SEPARATE iteration budget -- real fix (2026-09-11):
/// sharing `MAX_NEWTON_ITERS` between the OUTER Newton loop (expensive:
/// real residual/energy evaluation, `min_deformed_j`, per retry) and the
/// INNER CG solve (cheap: matrix-free, now Jacobi-preconditioned) starved
/// CG at a real, densely-coupled `basic_sand`-scale problem -- confirmed
/// live, the preconditioned solve reduced the real residual 99.6% (from
/// ~8e4 to ~317) but stalled short of `RELATIVE_TOLERANCE`, needing more
/// CG iterations to finish, not a different algorithm.
const MAX_CG_ITERS: usize = 400;
const RELATIVE_TOLERANCE: f32 = 1.0e-3;
/// Line-search admissibility floor on `det(F)` -- see `ImplicitProblem::
/// min_deformed_j`'s own doc. Comfortably above `corotated_elastic_
/// stress`'s `MIN_J=1e-6` hard-zero clamp (that discontinuity is exactly
/// what a residual-only acceptance test can be fooled by), while still
/// permissive enough to allow real, large compaction under a genuinely
/// stiff sand pile's own weight.
const MIN_ADMISSIBLE_J: f32 = 0.1;

/// One particle's 9-node quadratic-kernel stencil: node position, weight,
/// and the EXACT analytic kernel gradient (`axis_weights_derivative`) --
/// deliberately NOT the `weight*cell_dist*KERNEL_D_INVERSE` MLS-MPM
/// quadrature approximation (Hu et al. 2018) the engine's own explicit
/// P2G/G2P pipeline uses.
///
/// Real fix (2026-09-11), reversing an earlier same-week fix that turned
/// out to be backwards: a prior version of this file used the MLS-MPM
/// approximation here specifically to match what `gather_grid_to_
/// particles` independently re-derives for `particles.deformation_
/// gradient`/`velocity_gradient` every frame. That mismatch is real, but
/// per a direct read of two independent real implicit-MPM codebases --
/// `tmp/ziran2020` (Chenfanfu Jiang's group, the SAME lineage that
/// published Klar 2016's own sand model and the MLS-MPM paper this
/// approximation comes from) and `tmp/GeoTaichi` (an independent
/// geotechnical MPM/DEM framework with its own real implicit
/// Drucker-Prager solver) -- it is the WRONG mismatch to close. Both
/// build their Newton/CG force assembly (residual AND its
/// differential/Hessian-vector-product) from the exact analytic shape-
/// function gradient unconditionally, even in code that supports the MLS
/// approximation elsewhere for explicit kinematics --
/// `ziran2020/Lib/Ziran/Sim/MpmSimulationBase.cpp` asserts `mls_mpm`
/// requires `symplectic==true` (explicit) and every real implicit demo in
/// that repo explicitly sets `mls_mpm=false`; its own force assembly
/// (`Lib/MPM/Force/MpmForceBase.cpp::rasterizeForceToTVStack`/
/// `FBasedMpmForceHelper.cpp::computeStressDifferential`) is built on
/// `BSplineWeights`' real analytic derivative, never `cell_dist`.
/// `GeoTaichi`'s `NewtonIteration.py::assemble_element_local_stiffness_2D`
/// takes `scene.element.dshape_fn` (the exact gradient) regardless of
/// which velocity-gradient reconstruction its OWN kinematics use
/// elsewhere. The likely mechanism: the MLS-MPM approximation is a
/// quadrature scheme whose consistency guarantee is per-particle, in the
/// specific explicit single-step context it was derived for -- summing it
/// across MULTIPLE particles at a shared node and demanding Newton drive
/// THAT SUM to a stationary point via a Hessian built the same way does
/// not inherit the same guarantee, which would explain exactly why an
/// isolated particle (nothing to be inconsistent WITH) matched the
/// explicit baseline throughout prior testing while a densely-packed pile
/// (many particles' approximations summed at shared nodes) diverged from
/// frame one regardless of which other formula got fixed.
fn build_stencil(pos: Vec2) -> [(IVec2, f32, Vec2); 9] {
    let w = quadratic_weights(pos);
    let dx = axis_weights_derivative(pos.x - w.base_cell.x as f32 - 0.5);
    let dy = axis_weights_derivative(pos.y - w.base_cell.y as f32 - 0.5);
    let mut nodes = [(IVec2::ZERO, 0.0, Vec2::ZERO); 9];
    for gy in 0..3 {
        for gx in 0..3 {
            let weight = w.wx[gx] * w.wy[gy];
            let grad = Vec2::new(dx[gx] * w.wy[gy], w.wx[gx] * dy[gy]);
            let cell_pos = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            nodes[gy * 3 + gx] = (cell_pos, weight, grad);
        }
    }
    nodes
}

struct ImplicitParticle {
    lambda: f32,
    mu: f32,
    f_n: Mat2,
    v0: f32,
    /// (dof index into the shared node arrays below, weight, spatial gradient)
    entries: [(usize, f32, Vec2); 9],
}

/// The assembled Newton-CG problem for one big implicit step -- every
/// active particle sharing one dense set of touched grid DOFs.
struct ImplicitProblem {
    particles: Vec<ImplicitParticle>,
    node_pos: Vec<IVec2>,
    node_mass: Vec<f32>,
    /// Pre-force (mass-only-scattered, gravity-free) node velocity -- the
    /// residual's `v_n` reference state, exactly `Grid::normalize_
    /// velocities`'s own output before any force is applied.
    v_n: Vec<Vec2>,
    ext_force: Vec<Vec2>,
    dt: f32,
    /// `true` for a node close enough to a domain wall (within
    /// `SimConfig::boundary_thickness`) that a real boundary condition
    /// will act on it. Real fix (2026-09-11): a real, densely-packed sand
    /// pile resting on a floor needs the wall's reaction force in
    /// CONTINUOUS balance against gravity, every substep -- something a
    /// single free elastic Newton solve across the WHOLE frame, with the
    /// wall applied as a one-shot correction only after convergence, can
    /// never represent (confirmed live: an 8-particle clump matched the
    /// explicit baseline closely right up until it touched a wall, then
    /// diverged sharply -- explicit's real bounce came from thousands of
    /// tiny "apply gravity, clip once" cycles through the same contact,
    /// which a single big step cannot reproduce). These DOFs are held
    /// fixed at `v_n` throughout the free Newton search (never perturbed
    /// by `delta`) -- the SAME real "essential boundary condition lives on
    /// grid DOFs" principle `Particle::pinned`/`Grid::pinned_nodes`
    /// already use elsewhere in this engine -- letting Newton solve ONLY
    /// the genuinely free interior DOFs; the real, unmodified boundary
    /// condition (`apply_boundary_conditions_to_grid`) still runs
    /// afterward exactly as before, on top of whatever these nodes end up
    /// at.
    wall_frozen: Vec<bool>,
}

impl ImplicitProblem {
    fn velocity_gradient(entries: &[(usize, f32, Vec2)], v: &[Vec2]) -> Mat2 {
        let mut g = Mat2::ZERO;
        for &(idx, _w, grad) in entries {
            g += Mat2::from_cols(v[idx] * grad.x, v[idx] * grad.y);
        }
        g
    }

    /// Real, exact closed-form `exp(dt*grad_v)*F_n` (`deformation_increment_
    /// exp`, already real production code used elsewhere for the same
    /// reason -- see its own doc), NOT the naive linear `(I+dt*grad_v)*F_n`
    /// this file used originally. Real, necessary fix (2026-09-10): a
    /// settled `DruckerPragerMaterial` particle's real rest-state `F_n` is a
    /// PURE ROTATION (correctly zero elastic stress -- corotated elasticity
    /// is rotation-invariant by construction), not Identity, once its
    /// plastic return-mapping has relaxed all elastic strain away. The
    /// linear approximation is only accurate for SMALL `dt*grad_v`
    /// regardless of `F_n`, but composing it onto an ALREADY substantially
    /// rotated `F_n` (confirmed live: ~34 degrees, `tests/scratch_implicit_
    /// corotated_wiring_diagnostic.rs`'s `diag_properly_isolated_
    /// equilibrium_maintenance`) is NOT exactly orthogonal to first order
    /// the way the true rotation composition is -- that gap manufactures
    /// spurious "strain" (and hence spurious stress) that is pure
    /// linearization error, not real physics. Newton then had something
    /// genuine to chase in the wrong direction: reducing a residual built
    /// from an artifact rather than the true equilibrium, which is exactly
    /// what let a real, properly-isolated (explicit-settled, then switched
    /// to implicit from that identical verified-good state) sand pile drift
    /// several grid cells from the explicit baseline within a handful of
    /// frames even though each individual Newton solve "converged."
    ///
    /// Deliberately NOT also switching `jacobian_vector_product`'s `df`
    /// formula to the exponential map's own (much harder) Fréchet
    /// derivative: `newton_solve`'s line search always re-evaluates
    /// acceptance against THIS (exact) residual before taking a step, so an
    /// approximate/quasi-Newton search direction only affects convergence
    /// speed and robustness, never what the solve converges TO.
    fn deformed_f(p: &ImplicitParticle, v: &[Vec2], dt: f32) -> Mat2 {
        let grad_v = Self::velocity_gradient(&p.entries, v);
        crate::materials::utils::deformation_increment_exp(dt * grad_v) * p.f_n
    }

    /// The LINEAR (forward-Euler-style) `(I+dt*grad_v)*F_n` `deformed_f`'s
    /// own doc deliberately moved away from for the REAL residual (exact
    /// exponential map fixes real spurious-strain error at large
    /// pre-existing rotation, see that doc). Used ONLY to build a
    /// self-consistent (energy, gradient, Hessian) triple for `newton_
    /// solve`'s trust-region MODEL -- `model_residual` is THIS linear F's
    /// exact gradient and `model_jacobian_vector_product` is its exact
    /// Hessian-vector product (both verified by real FD tests, not
    /// assumed: `trust_region_consistency_tests`), unlike the true
    /// exponential-map `residual`, which is NOT the gradient of any energy
    /// built the same way (the matrix exponential's own Fréchet derivative
    /// would be needed for that, real, substantial extra math this file
    /// deliberately avoids -- confirmed necessary by a real, failed FD
    /// check attempting to skip it). The trust-region model only decides
    /// SEARCH DIRECTION and STEP SIZE; `newton_solve`'s actual convergence
    /// test and final acceptance always use the REAL `residual`/`min_
    /// deformed_j`, exactly the same "approximate model, exact acceptance"
    /// split this module's own JVP doc already used for the (now-removed)
    /// CG search direction.
    fn model_deformed_f(p: &ImplicitParticle, v: &[Vec2], dt: f32) -> Mat2 {
        let grad_v = Self::velocity_gradient(&p.entries, v);
        (Mat2::IDENTITY + dt * grad_v) * p.f_n
    }

    /// Smallest `det(F)` across every particle at trial velocity field `v`.
    /// `deformed_f` is the LINEAR (forward-Euler-style) approximation
    /// `(I+dt*grad_v)*F_n`, only valid for small `dt*grad_v` -- a large
    /// Newton trial step can push it past `corotated_elastic_stress`'s own
    /// `j <= MIN_J` hard clamp (stress pinned to exactly zero there). That
    /// clamp is a discontinuous cliff in the residual: crossing it makes
    /// `|r|` look like it dropped a lot (a large stress term vanished), so
    /// an acceptance test based on `|r|` alone will happily walk a particle
    /// INTO that degenerate, zero-elastic-support regime and call it
    /// progress -- confirmed live (2026-09-10): a real, densely-packed,
    /// already-settled sand pile's particles picked up near-free-fall
    /// velocity within the FIRST implicit frame after this exact mechanism,
    /// `tests/scratch_implicit_corotated_wiring_diagnostic.rs`'s
    /// `diag_settled_pile_first_forked_step_velocity_field`. `newton_solve`
    /// below additionally requires this to stay comfortably above `MIN_J`
    /// before accepting a step, closing that acceptance-criterion gap.
    fn min_deformed_j(&self, v: &[Vec2]) -> f32 {
        self.particles
            .iter()
            .map(|p| Self::deformed_f(p, v, self.dt).determinant())
            .fold(f32::INFINITY, f32::min)
    }

    fn residual(&self, v: &[Vec2]) -> Vec<Vec2> {
        let n = self.node_pos.len();
        let mut r: Vec<Vec2> = (0..n)
            .map(|i| self.node_mass[i] * (v[i] - self.v_n[i]) / self.dt - self.ext_force[i])
            .collect();
        for p in &self.particles {
            let f_new = Self::deformed_f(p, v, self.dt);
            // Kirchhoff stress + reference volume + SPATIAL gradient -- the
            // SAME pairing `spacetime::transfer::p2g::scatter_one_into` uses
            // (`stress_volume` defaults to `initial_volume`, paired directly
            // with Kirchhoff `combined_kirchhoff_stress` and a spatial
            // `cell_dist`/grad_w term, never a First-Piola/material-gradient
            // conversion). An earlier version of this file used First-Piola
            // stress (`tau * F^-T`) here, which introduces a spurious extra
            // `F^-T` factor when paired with a SPATIAL gradient instead of
            // the material gradient it actually requires -- caught by
            // `tests/scratch_implicit_corotated_wiring_diagnostic.rs`'s
            // real multi-frame trajectory comparison (a barely-moving,
            // already-settled pile diverged 8+ grid cells from the explicit
            // baseline in ONE frame with that bug in place), not by any
            // finite-difference self-consistency check -- those only verify
            // the JVP matches ITS OWN residual formula, not that the
            // formula matches this engine's real force convention.
            let stress = corotated_elastic_stress(f_new, p.lambda, p.mu);
            for &(idx, _w, grad) in &p.entries {
                r[idx] += p.v0 * (stress * grad);
            }
        }
        r
    }

    /// The trust-region MODEL's own residual -- `total_energy`'s exact
    /// gradient, built on `model_deformed_f` (linear, not the exact
    /// exponential map -- see that function's own doc for why the true
    /// `residual` cannot make this same claim without the matrix
    /// exponential's own Fréchet derivative).
    ///
    /// Real, necessary fix over a first attempt: naively reusing
    /// `residual`'s own `stress*grad` (Kirchhoff tau, no `dt` factor) form
    /// here is WRONG -- confirmed by a real, numerically hand-verified
    /// minimal case (`debug_minimal_single_entry_case`, single particle,
    /// single stencil entry, `F_n=I`): the true FD gradient of `total_
    /// energy` was 13.918 while `tau*grad` gave 487.0, off by >30x, while
    /// separately verified `dF_new/dv` (matches its own closed-form FD
    /// check to 0.16%) and the material-level `Psi`-vs-Piola relationship
    /// (already FD-verified in `materials::utils`) were each individually
    /// correct. The actual chain rule through `F_new(v)=(I+dt*grad_v)*F_n`
    /// (worked by hand via the trace identity `frob(P,(e⊗grad)*F_n) =
    /// e·(P*F_n^T*grad)`, then confirmed numerically) gives `dt*V0*Piola*
    /// (F_n^T*grad)` for the elastic term -- Piola (not Kirchhoff), paired
    /// with `F_n^T*grad` (not the raw spatial `grad`), scaled by `dt`
    /// (from the `dt*grad_v` inside `model_deformed_f`). `residual`'s own
    /// `tau*grad` (Kirchhoff, spatial gradient, no `dt`) is the REAL,
    /// separately-validated production MPM force convention (matches
    /// `spacetime::transfer::p2g` exactly) -- it is simply NOT the literal
    /// calculus gradient of a `Psi(F_new(v))`-type energy under a
    /// multiplicative F update, a real, structural MPM fact rather than a
    /// bug in either formula.
    ///
    /// Test-only (2026-09-11): `newton_solve` no longer uses this as
    /// Steihaug-CG's gradient `g` (a real, measured scale mismatch between
    /// this Piola/`dt`-scaled quantity and the real `residual` corrupted
    /// the trust region's own radius calibration -- see `newton_solve`'s
    /// own doc). Kept, not deleted: it is the verified oracle proving
    /// `model_jacobian_vector_product` (still real production code, used
    /// as Steihaug-CG's approximate curvature) is a genuine, principled
    /// Hessian-vector product of a real energy's gradient, not an
    /// arbitrary formula -- exactly the kind of hard-won diagnostic tool
    /// worth keeping callable for whatever the next real fix turns out to
    /// need, not just archived in test-only assertions.
    #[cfg(test)]
    fn model_residual(&self, v: &[Vec2]) -> Vec<Vec2> {
        let n = self.node_pos.len();
        let mut r: Vec<Vec2> = (0..n)
            .map(|i| self.node_mass[i] * (v[i] - self.v_n[i]) / self.dt - self.ext_force[i])
            .collect();
        for p in &self.particles {
            let f_new = Self::model_deformed_f(p, v, self.dt);
            let tau = corotated_elastic_stress(f_new, p.lambda, p.mu);
            // Matches `corotated_elastic_stress`'s own `J<=MIN_J` zero
            // convention: `tau=ZERO` there already, but `f_new.inverse()`
            // is ill-conditioned/NaN-prone at that same degeneracy, so
            // guard explicitly rather than let `0 * inverse` silently
            // produce NaN.
            let piola = if tau == Mat2::ZERO {
                Mat2::ZERO
            } else {
                tau * f_new.inverse().transpose()
            };
            let f_n_t = p.f_n.transpose();
            for &(idx, _w, grad) in &p.entries {
                r[idx] += self.dt * p.v0 * (piola * (f_n_t * grad));
            }
        }
        r
    }

    /// Total scalar MODEL objective for one implicit big-step: `0.5*m*(v-
    /// v_n)^2/dt - ext_force.v + sum_particles(V0*Psi(F_new_linear(v)))`
    /// -- the standard "optimization time integration" formulation of
    /// implicit Euler MPM (e.g. Gast et al. 2015 "Optimization Integrator
    /// for Large Time Steps"; also how Klar 2016's own implicit sand solve
    /// is framed), built on `model_deformed_f` (see that function's own
    /// doc for why, not the real `deformed_f`). Needed by `newton_solve`'s
    /// trust-region ratio test (Nocedal & Wright ch.4): predicted-vs-
    /// actual reduction needs a real scalar value, not just a gradient.
    ///
    /// Test-only (2026-09-11): `newton_solve`'s ratio test now measures
    /// `0.5*||residual||^2` (Gauss-Newton merit, Moré 1978) instead of
    /// this energy -- see `newton_solve`'s own doc for the real scale-
    /// mismatch that motivated the switch. Kept as `model_residual`'s own
    /// verified-gradient oracle (`model_residual_matches_energy_gradient_
    /// in_a_hand_verifiable_minimal_case`), not deleted.
    #[cfg(test)]
    fn total_energy(&self, v: &[Vec2]) -> f32 {
        let mut e = 0.0f32;
        for (i, &vi) in v.iter().enumerate() {
            let dv = vi - self.v_n[i];
            e += 0.5 * self.node_mass[i] * dv.length_squared() / self.dt;
            e -= self.ext_force[i].dot(vi);
        }
        for p in &self.particles {
            let f_new = Self::model_deformed_f(p, v, self.dt);
            e += p.v0 * corotated_elastic_energy_density(f_new, p.lambda, p.mu);
        }
        e
    }

    /// `model_residual`'s own exact Hessian-vector product -- `corotated_
    /// elastic_stress_jvp` applied to `dF = dt*d_grad_v*F_n`, EXACT here
    /// (not an approximation) because `model_deformed_f` is linear in `v`,
    /// so its own derivative in any direction `dv` is exactly `dt*grad_v
    /// (dv)*F_n` with no higher-order/Fréchet terms to drop. Used by
    /// `steihaug_cg`'s trust-region subproblem, which -- unlike the
    /// eigenvalue-clamped operator `corotated_kirchhoff_dtau_dl_psd_
    /// matrix` built for the (now-removed) direct/CG solve -- is DESIGNED
    /// to detect and handle negative curvature on its own (terminating
    /// exactly on the trust-region boundary), so it needs the TRUE model
    /// Hessian, not a PSD projection of it (see this module's own doc,
    /// "remaining limitation #2", for why a fixed PSD projection was
    /// confirmed to blind Newton to real curvature it still needed after
    /// the first accepted step).
    ///
    /// `model_residual`'s elastic term is `dt*V0*Piola*(F_n^T*grad)` with
    /// `Piola = tau*F_new^-T` -- differentiating that product rule (`Piola`
    /// depends on `F_new` through BOTH `tau` and the inverse) gives `d(
    /// Piola) = d(tau)*F_new^-T - Piola*dF^T*F_new^-T` (`d(F^-T) = -F^-T*
    /// dF^T*F^-T`, the standard matrix-inverse derivative identity), where
    /// `d(tau) = corotated_elastic_stress_jvp(F_new, dF, lambda, mu)`.
    fn model_jacobian_vector_product(&self, v: &[Vec2], dv: &[Vec2]) -> Vec<Vec2> {
        let n = self.node_pos.len();
        let mut dr: Vec<Vec2> = (0..n)
            .map(|i| self.node_mass[i] * dv[i] / self.dt)
            .collect();
        for p in &self.particles {
            let f_new = Self::model_deformed_f(p, v, self.dt);
            let tau = corotated_elastic_stress(f_new, p.lambda, p.mu);
            if tau == Mat2::ZERO {
                continue;
            }
            let f_inv_t = f_new.inverse().transpose();
            let piola = tau * f_inv_t;
            let d_grad_v = Self::velocity_gradient(&p.entries, dv);
            let df = self.dt * d_grad_v * p.f_n;
            let d_tau = corotated_elastic_stress_jvp(f_new, df, p.lambda, p.mu);
            let d_piola = d_tau * f_inv_t - piola * df.transpose() * f_inv_t;
            let f_n_t = p.f_n.transpose();
            for &(idx, _w, grad) in &p.entries {
                dr[idx] += self.dt * p.v0 * (d_piola * (f_n_t * grad));
            }
        }
        dr
    }

    /// Test-only (2026-09-12): a finite-difference Jacobian-vector product
    /// of the REAL, exact-exponential-map `residual` -- built to test
    /// whether `model_jacobian_vector_product`'s approximation (linear
    /// `model_deformed_f`, Piola/`F_n^T*grad`/`dt`-scaled) was close enough
    /// to the TRUE curvature for Steihaug-CG's inexact-Newton theory to
    /// hold, at `basic_sand`'s real production scale. Normalizes `dv` to
    /// unit length before perturbing (matching the h-convergence lesson
    /// from this project's own JVP verification history: `h=1e-2` on a
    /// UNIT direction stays clear of f32 catastrophic cancellation, unlike
    /// a fixed absolute `h` applied to a `dv` of unknown magnitude), then
    /// rescales the result back by `dv`'s real norm (valid because a true
    /// JVP is linear in `dv`).
    ///
    /// Real, measured finding (2026-09-12): at the exact point `newton_
    /// solve` stalls, `model_jacobian_vector_product` underestimates this
    /// EXACT Jacobian's magnitude by ~62-70x in the direction of the free
    /// residual (`cos_similarity` ~0.9999 at `h=1e-2` -- same DIRECTION,
    /// wildly different SCALE). A real, substantial finding on its own --
    /// but swapping this exact (if 2-evaluations-per-call expensive) JVP
    /// in as `steihaug_cg`'s own curvature did NOT fix the stall either
    /// (r_norm floor moved from ~317 to ~276, same collapse signature) --
    /// ruling out the approximation as the root cause too. See this
    /// module's own top-of-file doc for the real root cause this
    /// elimination process led to. Kept `#[cfg(test)]`, not deleted: the
    /// real, verified oracle proving `model_jacobian_vector_product`'s
    /// own approximation error, on hand for whatever a future faster/
    /// better curvature construction needs to be checked against.
    #[cfg(test)]
    fn real_residual_jvp_fd(&self, v: &[Vec2], dv: &[Vec2]) -> Vec<Vec2> {
        let n = v.len();
        let dv_norm = Self::plain_norm(dv);
        if dv_norm < 1.0e-20 {
            return vec![Vec2::ZERO; n];
        }
        let dv_unit: Vec<Vec2> = dv.iter().map(|x| *x / dv_norm).collect();
        const H: f32 = 1.0e-2;
        let v_plus: Vec<Vec2> = (0..n).map(|i| v[i] + H * dv_unit[i]).collect();
        let v_minus: Vec<Vec2> = (0..n).map(|i| v[i] - H * dv_unit[i]).collect();
        let r_plus = self.residual(&v_plus);
        let r_minus = self.residual(&v_minus);
        (0..n)
            .map(|i| dv_norm * (r_plus[i] - r_minus[i]) / (2.0 * H))
            .collect()
    }

    /// Unweighted L2 norm -- the trust-region radius's own natural measure
    /// (geometric distance in velocity-step space), deliberately NOT the
    /// mass-normalized norm `newton_solve`'s convergence tolerance uses
    /// (that anchors CONVERGENCE to a physical residual scale; this
    /// anchors how far the trust region trusts its local quadratic model,
    /// a different, purely geometric question).
    fn plain_norm(v: &[Vec2]) -> f32 {
        v.iter().map(|x| x.length_squared()).sum::<f32>().sqrt()
    }

    fn plain_dot(a: &[Vec2], b: &[Vec2]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x.dot(*y)).sum()
    }

    /// Solves for `tau>=0` such that `||p+tau*d||=radius` (Steihaug's own
    /// boundary-crossing formula, Nocedal & Wright "Numerical
    /// Optimization" 2nd ed., eq. 4.9/4.16) -- the positive root of the
    /// quadratic `||d||^2*tau^2 + 2*(p.d)*tau + (||p||^2-radius^2) = 0`.
    fn trust_region_boundary_tau(p: &[Vec2], d: &[Vec2], radius: f32) -> f32 {
        let dd = Self::plain_dot(d, d).max(1.0e-20);
        let pd = Self::plain_dot(p, d);
        let pp = Self::plain_dot(p, p);
        let c = pp - radius * radius;
        let disc = (pd * pd - dd * c).max(0.0);
        (-pd + disc.sqrt()) / dd
    }

    /// Diagonal (Jacobi) preconditioner -- same lumped-mass-plus-lumped-
    /// stiffness approximation this module's own earlier (now-removed) CG
    /// path used (`m/dt` inertia term plus `dt*V0*(lambda+2*mu)*|grad|^2`
    /// per touching particle), reused here as `steihaug_cg`'s
    /// preconditioner. Only needs to be a REASONABLE diagonal estimate of
    /// `model_jacobian_vector_product`'s own diagonal, not exact -- a
    /// preconditioner only affects CG's convergence RATE, never what it
    /// converges to (that's `steihaug_cg`'s own real, exact recurrence).
    fn jacobi_diagonal(&self) -> Vec<Vec2> {
        let mut diag: Vec<Vec2> = self
            .node_mass
            .iter()
            .map(|&m| Vec2::splat(m / self.dt))
            .collect();
        for p in &self.particles {
            let modulus = p.lambda + 2.0 * p.mu;
            for &(idx, _w, grad) in &p.entries {
                diag[idx] += Vec2::splat(self.dt * p.v0 * modulus * grad.length_squared());
            }
        }
        diag
    }

    /// Preconditioned Steihaug-Toint truncated CG (Nocedal & Wright 2nd
    /// ed., Algorithm 4.3 plus its own "Preconditioning" section) --
    /// approximately minimizes the trust-region model `m(p) = g.p +
    /// 0.5*p.H.p` subject to `||p||<=radius`, using `model_jacobian_
    /// vector_product` matrix-free. Its real, standard advantage over the
    /// (removed) eigenvalue-clamped direct solve: it detects negative
    /// curvature ON ITS OWN (terminating exactly on the trust-region
    /// boundary along the current search direction) instead of needing
    /// the operator pre-projected to PSD -- a fixed PSD projection was
    /// confirmed (this module's own doc, "remaining limitation #2") to
    /// blind Newton to real curvature it still needed after its first
    /// accepted step; Steihaug-CG never needs that projection because it
    /// handles indefiniteness structurally.
    ///
    /// Real fix (2026-09-11): preconditions the CG RECURRENCE (`jacobi_
    /// diagonal`, standard preconditioned-CG `r^T*y` inner products for
    /// `alpha`/`beta`) to fix a real, measured stall -- confirmed live, a
    /// real `basic_sand`-scale settled pile (1008 particles, real
    /// E=15MPa) reduced its residual 97.7% (from ~1e5 to ~2.3e3) via the
    /// unpreconditioned Gauss-Newton trust region, then stalled short of
    /// `RELATIVE_TOLERANCE`, the classic symptom of a genuinely stiff,
    /// poorly-conditioned system exhausting CG's iteration budget before
    /// finishing, not a wrong direction. Deliberately does NOT also switch
    /// the trust-region BOUNDARY check to the preconditioner's own `M`-
    /// norm (the textbook-complete preconditioned-Steihaug form) -- that
    /// would reintroduce exactly the kind of radius/scale-mismatch risk
    /// this module's own `newton_solve` doc already found and fixed once
    /// (switching from `model_residual`'s mismatched scale to the real
    /// residual's own); preconditioning only the search-direction
    /// generation, while keeping the boundary check in the SAME plain
    /// Euclidean norm the real residual's own scale is calibrated to, is a
    /// real, common, defensible simplification that keeps that fix intact.
    ///
    /// Real fix (2026-09-12): projects `hd` back onto the free-DOF subspace
    /// EVERY iteration, not just the final `p` once at the end (`newton_
    /// solve` already zeroed `p[i]` for `wall_frozen` `i`, but only after
    /// this loop returned). `model_jacobian_vector_product` operates on the
    /// full node set per particle stencil -- it has no notion of
    /// `wall_frozen` -- so a purely-free-DOF search direction `d` (`d[i]=0`
    /// for every frozen `i`, true by construction on the first iteration
    /// since `g` is already zeroed there) can still produce a NONZERO `hd`
    /// at a frozen node whenever that node shares a particle with a moving
    /// free neighbor -- a real, physically meaningful reaction force, but
    /// one this free-DOF-only solve has no business injecting back into its
    /// own recurrence. Left unprojected, that leak flows straight into
    /// `r_next[frozen]`, then `precondition(r_next)`, then `ry_next`/`beta`,
    /// then `d` itself picks up nonzero frozen components from the SECOND
    /// iteration onward -- silently corrupting the Krylov subspace this
    /// algorithm is supposed to build for the free-free reduced system
    /// `H_ff*p_f=-g_f`, the standard "Dirichlet-DOF projection every
    /// iteration" practice (confirmed as real production precedent in
    /// `tmp/ziran2020`'s own preconditioned CG, distilled in this project's
    /// own `tmp/ref_ziran_implicit_mpm.md`).
    ///
    /// Real, measured outcome of this specific fix (2026-09-12): a genuine
    /// leak was confirmed (`hd_frozen_norm` up to ~465 at `basic_sand`'s
    /// real production scale, `d_frozen_norm` correctly pinned at exactly
    /// 0.0 after this projection, proving it structurally prevents any
    /// contamination). This is kept unconditionally as a real correctness
    /// fix regardless -- but it did NOT close `basic_sand`'s own long-
    /// standing ~99.6%-then-stall gap (bit-for-bit identical stall before
    /// and after). See this module's own top-of-file doc for the real
    /// root cause that investigation went on to find.
    fn steihaug_cg(&self, v: &[Vec2], g: &[Vec2], radius: f32, tol: f32) -> Vec<Vec2> {
        let n = self.node_pos.len();
        let diag = self.jacobi_diagonal();
        let precondition = |r: &[Vec2]| -> Vec<Vec2> {
            (0..n)
                .map(|i| {
                    Vec2::new(
                        r[i].x / diag[i].x.abs().max(1.0e-8),
                        r[i].y / diag[i].y.abs().max(1.0e-8),
                    )
                })
                .collect()
        };

        let mut p = vec![Vec2::ZERO; n];
        let mut r = g.to_vec();
        if Self::plain_norm(&r) < tol {
            return p;
        }
        let y0 = precondition(&r);
        let mut d: Vec<Vec2> = y0.iter().map(|x| -*x).collect();
        let mut ry = Self::plain_dot(&r, &y0);

        for _ in 0..MAX_CG_ITERS {
            let mut hd = self.model_jacobian_vector_product(v, &d);
            for (i, &frozen) in self.wall_frozen.iter().enumerate() {
                if frozen {
                    hd[i] = Vec2::ZERO;
                }
            }
            let dhd = Self::plain_dot(&d, &hd);
            if dhd <= 1.0e-12 {
                let tau = Self::trust_region_boundary_tau(&p, &d, radius);
                return (0..n).map(|i| p[i] + tau * d[i]).collect();
            }
            let alpha = ry / dhd;
            let p_next: Vec<Vec2> = (0..n).map(|i| p[i] + alpha * d[i]).collect();
            if Self::plain_norm(&p_next) >= radius {
                let tau = Self::trust_region_boundary_tau(&p, &d, radius);
                return (0..n).map(|i| p[i] + tau * d[i]).collect();
            }
            let r_next: Vec<Vec2> = (0..n).map(|i| r[i] + alpha * hd[i]).collect();
            if Self::plain_norm(&r_next) < tol {
                return p_next;
            }
            let y_next = precondition(&r_next);
            let ry_next = Self::plain_dot(&r_next, &y_next);
            let beta = ry_next / ry;
            d = (0..n).map(|i| -y_next[i] + beta * d[i]).collect();
            p = p_next;
            r = r_next;
            ry = ry_next;
        }
        p
    }

    /// Build the per-particle Gershgorin-PSD 4x4 Hessian ONCE for a given
    /// trial velocity field `v` (see `corotated_kirchhoff_dtau_dl_psd_
    /// matrix`'s own doc for why this is the expensive part -- 4 JVP
    /// evaluations per particle). Real fix (2026-09-11): `v` is fixed for
    /// the whole duration of one `conjugate_gradient_solve` call, so this
    /// only needs to run once per Newton iteration, not once per CG
    /// iteration -- confirmed live as the actual cause of a real production
    /// regression: `basic_sand`-scale (1008 particles, real E=15MPa) ran
    /// the implicit path at 0.73x the explicit path's speed (SLOWER, not
    /// faster) purely from rebuilding this matrix from scratch on every one
    /// of up to 50 CG iterations x up to 150 Newton iterations, despite
    /// depending only on `v`.
    ///
    /// Test-only (2026-09-11): this function and the rest of the direct-
    /// solve chain it feeds (`free_dof_map`, `grad_basis`, `particle_
    /// stiffness_block`, `assemble_free_dof_system`, `cholesky_solve`,
    /// `direct_solve`) are no longer production code -- `newton_solve` now
    /// uses `steihaug_cg` (matrix-free, handles indefinite curvature
    /// structurally, no PSD projection needed). Kept callable, not
    /// deleted: this chain is the real, test-verified oracle that PROVED
    /// the original convergence failure was not a linear-solver problem
    /// (an exact Cholesky solve of this SAME eigenvalue-clamped system hit
    /// the identical wall a "fully converged" CG did) -- exactly the kind
    /// of diagnostic tool worth keeping on hand for the next investigation
    /// into `basic_sand`'s still-unresolved real-scale convergence gap
    /// (see this module's own top-of-file doc), not archived away.
    #[cfg(test)]
    fn build_particle_matrices(&self, v: &[Vec2]) -> Vec<[[f32; 4]; 4]> {
        self.particles
            .iter()
            .map(|p| {
                let f_new = Self::deformed_f(p, v, self.dt);
                corotated_kirchhoff_dtau_dl_psd_matrix(f_new, p.lambda, p.mu, self.dt, p.f_n)
            })
            .collect()
    }

    /// Maps each full grid-node index to its row/column in the dense
    /// free-DOF-only system (`None` for a `wall_frozen` node, which is held
    /// fixed and excluded from the system entirely rather than solved for
    /// and zeroed afterward -- a real, smaller, better-posed subsystem, not
    /// just a masking step).
    ///
    /// Test-only -- see `build_particle_matrices`'s own doc for why this
    /// whole chain is kept callable rather than deleted.
    #[cfg(test)]
    fn free_dof_map(wall_frozen: &[bool]) -> Vec<Option<usize>> {
        let mut next = 0usize;
        wall_frozen
            .iter()
            .map(|&frozen| {
                if frozen {
                    None
                } else {
                    let idx = next;
                    next += 1;
                    Some(idx)
                }
            })
            .collect()
    }

    /// `L`'s 4-vector (see `apply_dtau_dl_psd_matrix`'s vectorization order:
    /// `[L00, L10, L01, L11]`) is linear in a single stencil node's `dv`:
    /// `velocity_gradient` builds `L = sum_j dv_j ⊗ grad_j` (column `0` is
    /// `dv_j*grad_j.x`, column `1` is `dv_j*grad_j.y`), so node `j`'s own
    /// contribution to that 4-vector is exactly `B_j * dv_j` for this 4x2
    /// matrix.
    ///
    /// Test-only -- see `build_particle_matrices`'s own doc.
    #[cfg(test)]
    fn grad_basis(g: Vec2) -> [[f32; 2]; 4] {
        [[g.x, 0.0], [0.0, g.x], [g.y, 0.0], [0.0, g.y]]
    }

    /// Real closed-form per-particle-pair 2x2 stiffness block, derived
    /// mechanically from the SAME linear map `jacobian_vector_product`
    /// already applies matrix-free (see this module's own doc for the full
    /// derivation and why a direct solve, not a better-preconditioned CG,
    /// is the right tool at this problem's real DOF count): `dr_i = v0 *
    /// (B_i^T * M * B_j) * dv_j` for stencil nodes `i`, `j` of the SAME
    /// particle, reusing the already eigenvalue-clamped 4x4 `matrix`
    /// (`corotated_kirchhoff_dtau_dl_psd_matrix`) unchanged -- no new
    /// physics, purely the bilinear form's own explicit matrix, standard
    /// FEM/MPM local-stiffness assembly (e.g. `GeoTaichi`'s own
    /// `assemble_element_local_stiffness_2D`). Verified via a real
    /// self-consistency test (`assembled_operator_matches_matrix_free_
    /// jacobian_vector_product`) against the matrix-free path this
    /// mirrors, not assumed correct from the derivation alone.
    ///
    /// Test-only -- see `build_particle_matrices`'s own doc.
    #[cfg(test)]
    fn particle_stiffness_block(
        matrix: &[[f32; 4]; 4],
        v0: f32,
        grad_i: Vec2,
        grad_j: Vec2,
    ) -> [[f32; 2]; 2] {
        let b_i = Self::grad_basis(grad_i);
        let b_j = Self::grad_basis(grad_j);
        let mut tmp = [[0.0f32; 2]; 4];
        for (r, tmp_row) in tmp.iter_mut().enumerate() {
            for (c, tmp_rc) in tmp_row.iter_mut().enumerate() {
                *tmp_rc = (0..4).map(|k| matrix[r][k] * b_j[k][c]).sum();
            }
        }
        let mut block = [[0.0f32; 2]; 2];
        for (r, block_row) in block.iter_mut().enumerate() {
            for (c, block_rc) in block_row.iter_mut().enumerate() {
                *block_rc = v0 * (0..4).map(|k| b_i[k][r] * tmp[k][c]).sum::<f32>();
            }
        }
        block
    }

    /// Assembles the EXACT SAME linear operator `jacobian_vector_product`
    /// applies matrix-free, as an explicit dense SPD matrix over only the
    /// free (non-`wall_frozen`) DOFs. `dim = 2*n_free`; `a` is row-major
    /// `dim x dim`. Direct assembly + factorization, not a better-
    /// preconditioned iterative solve, is the standard tool at this
    /// problem's real scale (a few hundred to ~1000 free DOFs -- Golub &
    /// Van Loan; Saad, "Iterative Methods for Sparse Linear Systems": the
    /// regime iterative solvers exist for is much larger than this).
    ///
    /// Test-only -- see `build_particle_matrices`'s own doc.
    #[cfg(test)]
    fn assemble_free_dof_system(
        &self,
        matrices: &[[[f32; 4]; 4]],
        free_of: &[Option<usize>],
        n_free: usize,
        rhs: &[Vec2],
    ) -> (Vec<f32>, Vec<f32>) {
        let dim = 2 * n_free;
        let mut a = vec![0.0f32; dim * dim];
        let mut b = vec![0.0f32; dim];

        for (full_idx, &maybe_free) in free_of.iter().enumerate() {
            if let Some(free_idx) = maybe_free {
                let m_over_dt = self.node_mass[full_idx] / self.dt;
                a[(2 * free_idx) * dim + 2 * free_idx] += m_over_dt;
                a[(2 * free_idx + 1) * dim + 2 * free_idx + 1] += m_over_dt;
                b[2 * free_idx] = rhs[full_idx].x;
                b[2 * free_idx + 1] = rhs[full_idx].y;
            }
        }

        // A frozen node's `dv` is always zero by construction (never
        // perturbed by the free search), so any pair involving one
        // contributes nothing to the free system -- simply dropped, the
        // same physical statement `newton_solve`'s old post-hoc
        // `delta[i]=0` masking made, just built into the system itself.
        for (p, matrix) in self.particles.iter().zip(matrices) {
            for &(i, _wi, gi) in &p.entries {
                let Some(fi) = free_of[i] else { continue };
                for &(j, _wj, gj) in &p.entries {
                    let Some(fj) = free_of[j] else { continue };
                    let block = Self::particle_stiffness_block(matrix, p.v0, gi, gj);
                    for (r, block_row) in block.iter().enumerate() {
                        for (c, &block_rc) in block_row.iter().enumerate() {
                            a[(2 * fi + r) * dim + (2 * fj + c)] += block_rc;
                        }
                    }
                }
            }
        }
        (a, b)
    }

    /// Dense Cholesky factorization + solve for a symmetric positive-
    /// definite system (Golub & Van Loan, "Matrix Computations", 4th ed.,
    /// algorithm 4.2.1) -- `a` row-major `dim x dim`. Returns `None` if a
    /// diagonal pivot is non-positive: the assembled matrix should always
    /// be SPD (strictly positive mass/dt diagonal plus a sum of
    /// eigenvalue-clamped-PSD per-particle blocks), so this is a checked
    /// invariant, not an expected failure mode -- treated the same as any
    /// other Newton failure (fall back to the normal explicit substep).
    ///
    /// Test-only -- see `build_particle_matrices`'s own doc.
    #[cfg(test)]
    fn cholesky_solve(a: &[f32], dim: usize, b: &[f32]) -> Option<Vec<f32>> {
        let mut l = vec![0.0f32; dim * dim];
        for i in 0..dim {
            for j in 0..=i {
                let mut sum = a[i * dim + j];
                for k in 0..j {
                    sum -= l[i * dim + k] * l[j * dim + k];
                }
                if i == j {
                    if sum <= 0.0 {
                        return None;
                    }
                    l[i * dim + j] = sum.sqrt();
                } else {
                    l[i * dim + j] = sum / l[j * dim + j];
                }
            }
        }
        let mut y = vec![0.0f32; dim];
        for i in 0..dim {
            let mut sum = b[i];
            for k in 0..i {
                sum -= l[i * dim + k] * y[k];
            }
            y[i] = sum / l[i * dim + i];
        }
        let mut x = vec![0.0f32; dim];
        for i in (0..dim).rev() {
            let mut sum = y[i];
            for k in (i + 1)..dim {
                sum -= l[k * dim + i] * x[k];
            }
            x[i] = sum / l[i * dim + i];
        }
        Some(x)
    }

    /// Direct solve replacing `conjugate_gradient_solve` (see this module's
    /// own doc, "remaining limitation #2 -- scale", for why an iterative
    /// solve is the wrong tool at this problem's real DOF count): assembles
    /// the dense free-DOF system once and factors it once, instead of
    /// iterating a matrix-free operator. Returns a full-length `delta`
    /// (zero at every `wall_frozen` node, by construction). `None` only if
    /// the assembled matrix fails the SPD invariant (checked, not expected).
    ///
    /// Test-only (2026-09-11) -- superseded in production by `steihaug_cg`
    /// (matrix-free, no PSD projection needed). Kept callable, not
    /// deleted: this is the real, test-verified proof (`direct_solve_
    /// actually_solves_the_assembled_system`) that a real production
    /// regression (`basic_sand`-scale non-convergence) was NEVER a linear-
    /// solver problem -- an EXACT Cholesky solve of this same eigenvalue-
    /// clamped system hit the identical wall a "fully converged" CG did --
    /// a diagnostic worth having on hand for the next investigation into
    /// this module's still-unresolved real-scale convergence gap (see this
    /// module's own top-of-file doc), not archived away.
    #[cfg(test)]
    fn direct_solve(&self, v: &[Vec2], neg_r: &[Vec2]) -> Option<Vec<Vec2>> {
        let n = self.node_pos.len();
        let free_of = Self::free_dof_map(&self.wall_frozen);
        let n_free = free_of.iter().filter(|f| f.is_some()).count();
        let mut delta = vec![Vec2::ZERO; n];
        if n_free == 0 {
            return Some(delta);
        }
        let matrices = self.build_particle_matrices(v);
        let (a, b) = self.assemble_free_dof_system(&matrices, &free_of, n_free, neg_r);
        let x = Self::cholesky_solve(&a, 2 * n_free, &b)?;
        for (full_idx, &maybe_free) in free_of.iter().enumerate() {
            if let Some(fi) = maybe_free {
                delta[full_idx] = Vec2::new(x[2 * fi], x[2 * fi + 1]);
            }
        }
        Some(delta)
    }

    /// Per-node MASS-NORMALIZED residual norm (`sum(|r_i|^2 / mass_i)`),
    /// NOT the plain global L2 norm (`sum(|r_i|^2)`) -- real, cited fix
    /// (2026-09-11) for a genuine failure mode confirmed present in this
    /// exact solver: `ziran2020` (Chenfanfu Jiang's group, `Lib/Ziran/Sim/
    /// BackwardEuler.h::BackwardEulerLagrangianForceObjective::
    /// computeNorm`) divides every node's squared residual by that node's
    /// own mass before summing, specifically because an UNWEIGHTED global
    /// sum lets a low-mass/low-support node's residual go unnoticed: EVERY
    /// term in that node's residual (`m*(v-vn)/dt`, `m*g`, and its
    /// internal elastic support) scales down together with its tiny mass,
    /// so the residual can look small in absolute terms for almost ANY
    /// trial velocity there -- including pure free-fall, which trivially
    /// nearly satisfies `m*(v-vn)/dt - m*g ~= 0` regardless of whether the
    /// (also tiny) elastic term is actually correct. A global sum
    /// dominated by well-supported interior nodes can report a genuine
    /// 1000x overall reduction while such a node sits at a "near-zero
    /// residual but physically wrong" point the whole time -- exactly the
    /// symptom confirmed live in `tests/scratch_implicit_corotated_
    /// wiring_diagnostic.rs` (Newton reporting real, verified convergence
    /// while some particles retained near-free-fall velocity). `HOT`
    /// (`Projects/multigrid/MultigridSimulation.h::computeCharacteristicNorm`,
    /// same Ziran lineage) independently confirms the same principle from
    /// the other direction: its absolute tolerance is built from the
    /// material's own stiffness scale specifically so the stopping test is
    /// material/mesh independent, not anchored to gravity or the initial
    /// residual the way this file's tolerance was before this fix.
    ///
    /// Real, root-cause fix (2026-09-11) ADDED on top of the above,
    /// excluding `wall_frozen` nodes from the sum -- a structural
    /// convergence-criterion bug, not a search-algorithm one: `wall_
    /// frozen` nodes are held fixed at `v_n` throughout the ENTIRE free
    /// Newton search (see that field's own doc -- their real force
    /// balance is resolved separately, afterward, by `apply_boundary_
    /// conditions_to_grid`), so their own residual contribution can NEVER
    /// be reduced by anything this solve does. Confirmed live at `basic_
    /// sand`'s real production scale (1008 particles): 145 of 462 grid
    /// DOFs (31%) are wall_frozen, and their own residual norm (~5.5-5.8e4)
    /// is COMPARABLE to the free DOFs' own (~5.2-6.3e4) -- meaning a plain
    /// frozen+free mass-normalized sum has a hard, structural floor around
    /// the frozen contribution alone, making `RELATIVE_TOLERANCE=1e-3` (a
    /// 99.9% reduction target) mathematically UNREACHABLE regardless of
    /// solver quality. This is the real explanation for why THREE
    /// completely different search algorithms (CG, direct Cholesky solve,
    /// Steihaug-CG trust region) all independently hit the identical wall
    /// at this exact scale before this fix: none of them were ever solving
    /// an achievable problem, because the CONVERGENCE MEASURE itself
    /// included a quantity none of them could touch. `newton_solve`'s
    /// outer convergence check and its trust-region acceptance gate both
    /// use this free-only norm; the full (frozen-inclusive) `residual` is
    /// still what's added back to the grid on success (`wall_frozen` nodes
    /// keep their real, current momentum-balance contribution -- only the
    /// CONVERGENCE JUDGMENT excludes them, not the physics). Real,
    /// disclosed limitation this fix does NOT close on its own: even after
    /// it, `basic_sand`'s real production scale still does not converge
    /// (stalls around ~99.6% reduction, not the required 99.9% -- see this
    /// module's own top-of-file doc for the current, unresolved state).
    fn free_mass_normalized_norm(&self, r: &[Vec2]) -> f32 {
        self.node_mass
            .iter()
            .zip(r)
            .zip(&self.wall_frozen)
            .filter(|&(_, &frozen)| !frozen)
            .map(|((&m, ri), _)| ri.length_squared() / m.max(1.0e-6))
            .sum::<f32>()
            .sqrt()
    }

    /// Trust-region Newton (Nocedal & Wright 2nd ed., ch.4; Steihaug-CG
    /// subproblem solve, Algorithm 4.3) -- real fix (2026-09-11) replacing
    /// this module's earlier backtracking-line-search Newton. Root cause
    /// that motivated the switch: a real, densely-packed `basic_sand`-
    /// scale settled pile (1008 particles, real E=15MPa) made ONE genuine
    /// Newton step (21% residual reduction, confirmed via `EMERGE_
    /// IMPLICIT_LINESEARCH_DIAG=1` trace) then STALLED completely -- every
    /// subsequent iteration's full-to-tiny backtrack found no real
    /// improvement at ANY step scale, because backtracking can only RESCALE
    /// a single fixed Newton direction, never change it; when that
    /// direction stops correlating with real improvement (expected once
    /// the true, possibly-indefinite Hessian diverges from whatever
    /// approximation produced the direction), no amount of rescaling
    /// recovers. A trust region fixes this structurally: on rejection it
    /// SHRINKS the region and RE-SOLVES the subproblem, which can return a
    /// genuinely different (more steepest-descent-like) direction, not
    /// just a smaller step along the same one.
    ///
    /// Uses the SEPARATE, verified-self-consistent model triple (`model_
    /// residual`=`total_energy`'s exact gradient, `model_jacobian_vector_
    /// product`=its exact Hessian-vector product -- see `model_deformed_f`
    /// and `model_residual`'s own docs for why these differ from the real
    /// `residual`/`deformed_f`, and `trust_region_consistency_tests` for
    /// the real, run verification) ONLY to pick a direction and size the
    /// trust region. Acceptance is ALWAYS gated on the REAL `residual`
    /// (exact exponential map) actually improving -- the model's own ratio
    /// test only fine-tunes how much to grow the region after a real
    /// success, never overrides a real failure. Returns `None` if it fails
    /// to converge within `MAX_NEWTON_ITERS` or the trust region collapses
    /// without ever helping the real residual -- callers must treat that
    /// as "run the normal explicit substep instead," never a partial
    /// result.
    ///
    /// Real absolute-vs-relative lesson (2026-09-10, still applies):
    /// anchoring the outer convergence tolerance to the LARGER of
    /// `r0_norm` and `ext_force`'s own magnitude avoids demanding an
    /// impossible absolute residual when a genuinely at-rest particle's
    /// `r0_norm` starts tiny by definition.
    fn newton_solve(&self, v_init: &[Vec2]) -> Option<Vec<Vec2>> {
        let n = self.node_pos.len();
        let mut v = v_init.to_vec();
        let mut r = self.residual(&v);
        let mut r_norm = self.free_mass_normalized_norm(&r);
        let r0_norm = r_norm.max(1.0e-12);
        let ext_force_norm = self.free_mass_normalized_norm(&self.ext_force);
        let scale = r0_norm.max(ext_force_norm).max(1.0e-12);

        const RADIUS_MAX: f32 = 1.0e8;
        const RADIUS_MIN: f32 = 1.0e-12;
        const MAX_RADIUS_RETRIES: usize = 80;
        let diag = std::env::var("EMERGE_IMPLICIT_DIAG").is_ok();

        // Gauss-Newton trust region for a NONLINEAR EQUATION SOLVE
        // (`residual(v)=0`, Nocedal & Wright 2nd ed. ch.10; Moré 1978's
        // classic trust-region Levenberg-Marquardt paper), NOT an energy-
        // minimization trust region -- real fix (2026-09-11) replacing an
        // energy-based design that, while internally provably consistent
        // (`model_residual` IS `total_energy`'s exact gradient, verified
        // by `trust_region_consistency_tests`), used `model_residual` as
        // Steihaug-CG's gradient `g` -- a DIFFERENT vector from the real
        // `residual` (Piola-based, `dt`-scaled, paired with `F_n^T*grad`,
        // vs. `residual`'s Kirchhoff-based, unscaled, spatial-`grad`
        // convention -- `model_deformed_f`'s own doc explains why they
        // must differ). That scale/direction mismatch corrupted the trust
        // region's own radius calibration: a real diagnostic probe
        // (stepping along the REAL residual's own negative-gradient
        // direction by a tiny `eps`) confirmed a genuinely improving
        // direction DOES exist from `v_n` at `basic_sand`'s real
        // production scale (1008 particles, real E=15MPa) -- `eps=1e-8,
        // 1e-10, 1e-12` all improved the real residual, `eps=1e-6`
        // already overshot -- but the OLD design's radius, derived from
        // `model_residual`'s own (differently-scaled) magnitude, never
        // actually explored that specific window. Using the REAL
        // `residual` directly as `g` (the model's approximate Hessian-
        // vector product, `model_jacobian_vector_product`, still informs
        // Steihaug-CG's curvature/step-size choice -- an approximate
        // Jacobian for a quasi-Newton search direction is real, standard
        // practice, same "approximate model, exact acceptance" split this
        // module has used throughout) and measuring predicted/actual
        // reduction in `0.5*||r||^2` (the standard Gauss-Newton merit for
        // nonlinear equations, not a potentially-mismatched energy)
        // removes the scale-mismatch risk structurally, not by tuning a
        // constant.
        let mut radius = Self::plain_norm(&r).max(1.0e-8);

        for _ in 0..MAX_NEWTON_ITERS {
            if r_norm / scale < RELATIVE_TOLERANCE {
                return Some(v);
            }

            let r_free: Vec<Vec2> = (0..n)
                .map(|i| {
                    if self.wall_frozen[i] {
                        Vec2::ZERO
                    } else {
                        r[i]
                    }
                })
                .collect();
            let r_free_plain_norm = Self::plain_norm(&r_free);
            if r_free_plain_norm < 1.0e-10 {
                if diag {
                    eprintln!(
                        "[newton_diag] real residual vanished on free DOFs (r_free_plain_norm={r_free_plain_norm:.4e}) while r_norm={r_norm:.4e}"
                    );
                }
                return None;
            }
            let cg_tol = (r_free_plain_norm * 1.0e-1).max(1.0e-12);
            let half_r_free_sq = 0.5 * Self::plain_dot(&r_free, &r_free);

            let mut accepted_this_iter = false;
            for _retry in 0..MAX_RADIUS_RETRIES {
                let mut p = self.steihaug_cg(&v, &r_free, radius, cg_tol);
                for (i, &frozen) in self.wall_frozen.iter().enumerate() {
                    if frozen {
                        p[i] = Vec2::ZERO;
                    }
                }

                let v_trial: Vec<Vec2> = (0..n).map(|i| v[i] + p[i]).collect();
                let trial_j = self.min_deformed_j(&v_trial);

                // Predicted reduction in 0.5*||r||^2 under the LINEAR
                // model r_model = r_free + H*p (Gauss-Newton).
                let hp = self.model_jacobian_vector_product(&v, &p);
                let r_model: Vec<Vec2> = (0..n).map(|i| r_free[i] + hp[i]).collect();
                let predicted_reduction =
                    half_r_free_sq - 0.5 * Self::plain_dot(&r_model, &r_model);

                let real_accept = if trial_j >= MIN_ADMISSIBLE_J {
                    let r_trial = self.residual(&v_trial);
                    let r_trial_norm = self.free_mass_normalized_norm(&r_trial);
                    let r_trial_free: Vec<Vec2> = (0..n)
                        .map(|i| {
                            if self.wall_frozen[i] {
                                Vec2::ZERO
                            } else {
                                r_trial[i]
                            }
                        })
                        .collect();
                    let actual_reduction =
                        half_r_free_sq - 0.5 * Self::plain_dot(&r_trial_free, &r_trial_free);
                    let rho = if predicted_reduction > 1.0e-20 {
                        actual_reduction / predicted_reduction
                    } else {
                        -1.0
                    };
                    if diag {
                        eprintln!(
                            "[trust_region_diag] radius={radius:.4e} rho={rho:.4e} r_norm={r_norm:.4e} r_trial_norm={r_trial_norm:.4e} j={trial_j:.4e}"
                        );
                    }
                    if r_trial_norm < r_norm {
                        v = v_trial;
                        r = r_trial;
                        r_norm = r_trial_norm;
                        Some(rho)
                    } else {
                        None
                    }
                } else {
                    None
                };

                let p_norm = Self::plain_norm(&p);
                if let Some(rho) = real_accept {
                    if rho > 0.75 && p_norm >= radius * 0.99 {
                        radius = (radius * 2.0).min(RADIUS_MAX);
                    } else if rho < 0.25 {
                        radius *= 0.5;
                    }
                    accepted_this_iter = true;
                    break;
                } else {
                    radius *= 0.25;
                    if radius < RADIUS_MIN {
                        break;
                    }
                }
            }

            if !accepted_this_iter {
                if diag {
                    eprintln!(
                        "[newton_diag] trust region collapsed (radius={radius:.4e}) without improving the real residual (r_norm={r_norm:.4e})"
                    );
                }
                return None;
            }
        }
        (r_norm / scale < RELATIVE_TOLERANCE).then_some(v)
    }
}

impl Simulation {
    /// `true` when every active particle's material qualifies for the
    /// shared Corotated elastic branch AND no feature this v1 doesn't model
    /// yet is active in the scene -- see this module's own doc for the
    /// full, disclosed list.
    fn implicit_corotated_eligible(&self) -> bool {
        if !self.config.implicit_corotated_elastic {
            return false;
        }
        if !self.rods.is_empty()
            || !self.grain_populations.is_empty()
            || self.granular_fluidity.is_some()
            || self.cosserat.is_some()
        {
            return false;
        }
        if self.config.asflip_blend != 0.0
            || self.config.cundall_damping != 0.0
            || self.config.mixture_pressure_iterations != 0
            || self.config.fluid_pressure_iterations != 0
            || self.config.sleep_threshold != 0.0
        {
            return false;
        }
        if self.active_count == 0 || self.active_count != self.particles.len() {
            return false;
        }
        for i in 0..self.active_count {
            if self.particles.contact_group[i] != 0 || self.particles.pinned[i] != 0 {
                return false;
            }
            let material = self.materials.get(self.particles.material_id[i]);
            if material.corotated_lame_params().is_none() {
                return false;
            }
        }
        true
    }

    /// Real assembly + solve for one implicit big-step. Returns `false`
    /// (and touches nothing) whenever the scene is ineligible OR the solve
    /// fails to converge -- see this module's own doc for why both are
    /// real, safe fallback triggers, not just a static scope guard.
    pub(crate) fn try_implicit_corotated_substep(&mut self, dt: f32) -> bool {
        if !self.implicit_corotated_eligible() {
            return false;
        }

        // ── Mass-only P2G: same real APIC mass/momentum accumulation
        // `scatter_particles_to_grid` uses, minus its force term -- the
        // force this substep applies comes from the Newton-CG residual's
        // own `ext_force`/internal-stress terms below, not from baking a
        // single-F-linearized force into the scatter the way the explicit
        // path does (see `spacetime::transfer::p2g`'s own doc for that
        // convention -- this is deliberately NOT that).
        self.grid.clear();
        for i in 0..self.active_count {
            let x = self.particles.x[i];
            let mass_i = self.particles.mass[i];
            let v_i = self.particles.v[i];
            let c_i = self.particles.velocity_gradient[i];
            for &(cell_pos, weight, _grad) in &build_stencil(x) {
                if weight == 0.0 {
                    continue;
                }
                let cell_dist = cell_pos.as_vec2() - x + Vec2::splat(0.5);
                let momentum = weight * mass_i * (v_i + c_i * cell_dist);
                self.grid
                    .add_mass_momentum(cell_pos, weight * mass_i, momentum);
            }
        }
        self.grid.normalize_velocities();

        // Dense DOF assembly: one entry per touched node, real O(1) hash
        // dedup on grid coordinate (same technique the engine's own sparse
        // `Grid` uses internally) instead of a linear scan.
        let mut node_lookup: std::collections::HashMap<IVec2, usize> =
            std::collections::HashMap::with_capacity(self.active_count * 2);
        let mut node_pos: Vec<IVec2> = Vec::with_capacity(self.active_count * 2);
        let mut particles = Vec::with_capacity(self.active_count);
        for i in 0..self.active_count {
            let material = self.materials.get(self.particles.material_id[i]);
            let Some((lambda, mu)) = material.corotated_lame_params() else {
                unreachable!("eligibility check above already verified every material qualifies")
            };
            let stencil = build_stencil(self.particles.x[i]);
            let mut entries = [(0usize, 0.0f32, Vec2::ZERO); 9];
            for (k, &(abs_node, weight, grad)) in stencil.iter().enumerate() {
                let dof = *node_lookup.entry(abs_node).or_insert_with(|| {
                    node_pos.push(abs_node);
                    node_pos.len() - 1
                });
                entries[k] = (dof, weight, grad);
            }
            particles.push(ImplicitParticle {
                lambda,
                mu,
                f_n: self.particles.deformation_gradient[i],
                v0: self.particles.initial_volume[i],
                entries,
            });
        }

        let n = node_pos.len();
        let mut node_mass = vec![0.0f32; n];
        let mut v_n = vec![Vec2::ZERO; n];
        for (dof, &pos) in node_pos.iter().enumerate() {
            node_mass[dof] = self.grid.mass_at(pos);
            v_n[dof] = self.grid.velocity_at(pos);
        }
        let gravity = self.config.gravity;
        let ext_force: Vec<Vec2> = node_mass.iter().map(|&m| m * gravity).collect();

        let grid_res_i32 = self.config.grid_res as i32;
        let thickness = self.config.boundary_thickness as i32;
        let wall_frozen: Vec<bool> = node_pos
            .iter()
            .map(|p| {
                p.x < thickness
                    || p.y < thickness
                    || p.x >= grid_res_i32 - thickness
                    || p.y >= grid_res_i32 - thickness
            })
            .collect();

        let problem = ImplicitProblem {
            particles,
            node_pos: node_pos.clone(),
            node_mass,
            v_n: v_n.clone(),
            ext_force,
            dt,
            wall_frozen,
        };

        // TEMPORARY diagnostic (2026-09-10/11), same opt-in-via-env-var
        // convention as `EMERGE_CFL_DIAGNOSE` elsewhere in this solver --
        // zero cost for every scene that doesn't set it. Real, resolved
        // investigation this was built for (kept for future debugging, not
        // stale): a densely-packed, already-settled `DruckerPragerMaterial`
        // sand pile used to diverge several grid cells from the explicit
        // baseline within a handful of frames even though each Newton
        // solve reported genuine convergence -- root-caused to wall-
        // adjacent grid DOFs being perturbed by the free search and
        // corrected only once, after the fact (see `ImplicitProblem::
        // wall_frozen`'s own doc for the real fix and `tests/
        // implicit_corotated_substep.rs`'s `implicit_matches_explicit_
        // for_an_already_settled_pile` for the passing regression test).
        // Real, disclosed, remaining limitation NOT covered by that fix: a
        // VIOLENT impact still diverges more than a settled pile does --
        // see this file's own top-of-module doc and `violent_impact_
        // diverges_more_than_settled_pile_a_real_disclosed_limitation`.
        let diag = std::env::var("EMERGE_IMPLICIT_DIAG").is_ok();
        if diag {
            let r0_norm = problem
                .residual(&v_n)
                .iter()
                .map(|x| x.length_squared())
                .sum::<f32>()
                .sqrt();
            let n_frozen = problem.wall_frozen.iter().filter(|&&f| f).count();
            let r_full = problem.residual(&v_n);
            let r_frozen_norm: f32 = (0..n)
                .filter(|&i| problem.wall_frozen[i])
                .map(|i| r_full[i].length_squared())
                .sum::<f32>()
                .sqrt();
            let r_free_norm: f32 = (0..n)
                .filter(|&i| !problem.wall_frozen[i])
                .map(|i| r_full[i].length_squared())
                .sum::<f32>()
                .sqrt();
            eprintln!(
                "[implicit_diag] n_particles={} n_dofs={n} r0_norm={r0_norm:.6e} n_frozen={n_frozen} r_frozen_norm={r_frozen_norm:.4e} r_free_norm={r_free_norm:.4e}",
                problem.particles.len(),
            );
            // Trivial-direction probe: does the most obvious possible
            // step (follow the free-DOF residual's own negative gradient
            // by a tiny epsilon, i.e. a pure gradient-descent nudge) help
            // AT ALL? If not even this trivial direction improves the
            // real residual, the problem isn't which search algorithm
            // picks the direction.
            let r0_free_norm = problem.free_mass_normalized_norm(&r_full);
            for &eps in &[1.0e-6f32, 1.0e-8, 1.0e-10, 1.0e-12] {
                let v_trial: Vec<Vec2> = (0..n)
                    .map(|i| {
                        if problem.wall_frozen[i] {
                            v_n[i]
                        } else {
                            v_n[i] - eps * r_full[i]
                        }
                    })
                    .collect();
                let r_trial_norm = problem.free_mass_normalized_norm(&problem.residual(&v_trial));
                eprintln!(
                    "[gradient_probe] eps={eps:.1e} r0_free_norm={r0_free_norm:.6e} r_trial_norm={r_trial_norm:.6e} improved={}",
                    r_trial_norm < r0_free_norm
                );
            }
        }
        let Some(v_solved) = problem.newton_solve(&v_n) else {
            if diag {
                eprintln!("[implicit_diag] newton_solve did NOT converge -- fallback to explicit");
            }
            // Convergence failure -- leave `self.grid` as scratch state (it
            // gets `clear()`-ed again unconditionally at the top of the
            // normal explicit substep this falls back to) and touch no
            // particle state at all.
            return false;
        };
        if diag {
            let r_final = problem.residual(&v_solved);
            let r_final_norm: f32 = r_final
                .iter()
                .map(|x| x.length_squared())
                .sum::<f32>()
                .sqrt();
            let max_speed = v_solved.iter().map(|v| v.length()).fold(0.0f32, f32::max);
            eprintln!(
                "[implicit_diag] converged: r_final_norm={r_final_norm:.6e} max_node_speed={max_speed:.4}"
            );
        }

        // Write the solved field back into the (already-normalized) grid,
        // then run the SAME boundary-condition pass the explicit path uses
        // -- `add_mass_momentum(pos, 0.0, delta)` on an already-touched,
        // already-normalized cell adds directly to `cell.momentum` (real
        // velocity at this point, not mass-weighted), so this is an exact
        // "set velocity to v_solved" for every dof, not an approximation.
        for (dof, &pos) in node_pos.iter().enumerate() {
            self.grid
                .add_mass_momentum(pos, 0.0, v_solved[dof] - v_n[dof]);
        }
        let grid_res = self.grid.resolution();
        for boundary in &self.boundaries {
            apply_boundary_conditions_to_grid(&mut self.grid, grid_res, boundary.as_ref());
        }

        // Real, unmodified G2P: gathers velocity/position/APIC-C from the
        // (implicit-solved, boundary-corrected) grid AND applies each
        // particle's own real plastic return-mapping
        // (`MaterialModel::update_particle`) exactly as the explicit path
        // does -- see this module's own doc for why that single call here
        // IS the Klar 2016 operator-split plastic correction, not a
        // separate reimplementation of it.
        self.last_vel_clamp_count += gather_grid_to_particles(
            &mut self.particles,
            &self.grid,
            dt,
            gravity,
            &self.boundaries,
            &self.materials,
            G2PParams {
                apic_blend: self.config.apic_blend,
                active_count: self.active_count,
                asflip_blend: 0.0,
                pre_force_snapshot: None,
                nonlocal_fluidity: &[],
                cosserat_curvature: &[],
                boundary_thickness: self.config.boundary_thickness,
            },
        );
        true
    }
}

#[cfg(test)]
mod direct_solve_consistency_tests {
    use super::*;
    use crate::materials::utils::apply_dtau_dl_psd_matrix;

    /// Real self-consistency check for `particle_stiffness_block`'s
    /// closed-form derivation (this module's own doc, on `particle_
    /// stiffness_block` and `assemble_free_dof_system`, has the full
    /// derivation). Direct assembly replaced the matrix-free Newton-CG
    /// path entirely (2026-09-11) -- this test proves the block formula
    /// `dr_i = v0 * (B_i^T * M * B_j) * dv_j`, summed over every pair of a
    /// particle's stencil nodes, EXACTLY reproduces what the removed
    /// matrix-free path computed (`v0 * apply_dtau_dl_psd_matrix(M,
    /// velocity_gradient(entries, dv)) * grad_i`, reconstructed here
    /// inline for comparison since the original method no longer exists in
    /// production) -- a real, run, empirical check, not an unverified
    /// derivation claim.
    #[test]
    fn particle_stiffness_block_matches_matrix_free_jvp_formula() {
        let entries: [(usize, f32, Vec2); 9] = [
            (0, 0.2, Vec2::new(0.3, -0.1)),
            (1, 0.15, Vec2::new(-0.2, 0.25)),
            (2, 0.1, Vec2::new(0.1, 0.4)),
            (0, 0.05, Vec2::new(-0.15, 0.05)),
            (1, 0.2, Vec2::new(0.35, -0.3)),
            (2, 0.1, Vec2::new(-0.05, -0.2)),
            (0, 0.1, Vec2::new(0.2, 0.2)),
            (1, 0.05, Vec2::new(-0.1, -0.1)),
            (2, 0.05, Vec2::new(0.05, -0.05)),
        ];
        let v0 = 0.037f32;
        // An arbitrary (not necessarily symmetric or PSD) 4x4 matrix --
        // this is a pure linear-algebra identity check, not a physics
        // check, so the matrix's own structure is irrelevant to what's
        // being verified.
        let matrix = [
            [12.0, 1.5, -0.5, 0.3],
            [1.5, 9.0, 0.2, -0.4],
            [-0.5, 0.2, 7.0, 1.1],
            [0.3, -0.4, 1.1, 10.0],
        ];
        let n = 3;
        let dv = vec![
            Vec2::new(0.4, -0.2),
            Vec2::new(-0.1, 0.3),
            Vec2::new(0.05, 0.15),
        ];

        let d_grad_v = ImplicitProblem::velocity_gradient(&entries, &dv);
        let dp = apply_dtau_dl_psd_matrix(&matrix, d_grad_v);
        let mut reference = vec![Vec2::ZERO; n];
        for &(idx, _w, grad) in &entries {
            reference[idx] += v0 * (dp * grad);
        }

        let mut assembled = vec![Vec2::ZERO; n];
        for &(i, _wi, gi) in &entries {
            for &(j, _wj, gj) in &entries {
                let block = ImplicitProblem::particle_stiffness_block(&matrix, v0, gi, gj);
                let dv_j = dv[j];
                assembled[i] += Vec2::new(
                    block[0][0] * dv_j.x + block[0][1] * dv_j.y,
                    block[1][0] * dv_j.x + block[1][1] * dv_j.y,
                );
            }
        }

        for i in 0..n {
            let diff = (reference[i] - assembled[i]).length();
            let scale = reference[i].length().max(1.0);
            assert!(
                diff / scale < 1.0e-4,
                "block assembly diverges from the matrix-free JVP formula at node {i}: reference={:?} assembled={:?}",
                reference[i],
                assembled[i]
            );
        }
    }

    /// Real end-to-end check that `assemble_free_dof_system` + `cholesky_
    /// solve` actually solves the linear system it claims to: for a random
    /// SPD-guaranteed local matrix and a small synthetic 2-particle,
    /// shared-DOF problem, `A*x` (recomputed via the SAME matrix-free
    /// formula this file used before tonight's rewrite) must equal `b` to
    /// float precision.
    #[test]
    fn direct_solve_actually_solves_the_assembled_system() {
        let entries_a: [(usize, f32, Vec2); 9] = [
            (0, 0.2, Vec2::new(0.4, 0.1)),
            (1, 0.15, Vec2::new(-0.1, 0.3)),
            (2, 0.1, Vec2::new(0.2, -0.2)),
            (0, 0.05, Vec2::new(-0.1, 0.05)),
            (1, 0.2, Vec2::new(0.3, -0.1)),
            (2, 0.1, Vec2::new(-0.05, 0.15)),
            (0, 0.1, Vec2::new(0.15, 0.2)),
            (1, 0.05, Vec2::new(-0.05, -0.1)),
            (2, 0.05, Vec2::new(0.1, -0.05)),
        ];
        let entries_b: [(usize, f32, Vec2); 9] = [
            (1, 0.2, Vec2::new(0.2, -0.15)),
            (2, 0.15, Vec2::new(-0.3, 0.1)),
            (3, 0.1, Vec2::new(0.1, 0.25)),
            (1, 0.05, Vec2::new(-0.2, 0.1)),
            (2, 0.2, Vec2::new(0.25, -0.05)),
            (3, 0.1, Vec2::new(-0.1, -0.2)),
            (1, 0.1, Vec2::new(0.1, 0.1)),
            (2, 0.05, Vec2::new(-0.15, -0.1)),
            (3, 0.05, Vec2::new(0.05, 0.15)),
        ];
        // Diagonally dominant with positive diagonal -> genuinely SPD, so
        // this exercises a real solve, not a degenerate one.
        let matrix = [
            [20.0, 1.0, -0.5, 0.2],
            [1.0, 18.0, 0.3, -0.4],
            [-0.5, 0.3, 22.0, 0.6],
            [0.2, -0.4, 0.6, 19.0],
        ];
        let node_mass = vec![0.4f32, 0.6, 0.5, 0.3];
        let dt = 0.016f32;
        let n = node_mass.len();

        let problem = ImplicitProblem {
            particles: vec![
                ImplicitParticle {
                    lambda: 0.0,
                    mu: 0.0,
                    f_n: Mat2::IDENTITY,
                    v0: 0.041,
                    entries: entries_a,
                },
                ImplicitParticle {
                    lambda: 0.0,
                    mu: 0.0,
                    f_n: Mat2::IDENTITY,
                    v0: 0.033,
                    entries: entries_b,
                },
            ],
            node_pos: (0..n).map(|_| IVec2::ZERO).collect(),
            node_mass: node_mass.clone(),
            v_n: vec![Vec2::ZERO; n],
            ext_force: vec![Vec2::ZERO; n],
            dt,
            wall_frozen: vec![false, false, false, false],
        };

        // Matrix-free reference `A*dv` for an arbitrary `dv`, using the SAME
        // per-particle matrix for both particles (real particles would each
        // get their own via `build_particle_matrices`, but this test only
        // needs ONE fixed, known-SPD matrix to exercise the assembly path).
        let apply_reference = |dv: &[Vec2]| -> Vec<Vec2> {
            let mut out: Vec<Vec2> = (0..n).map(|i| node_mass[i] * dv[i] / dt).collect();
            for p in &problem.particles {
                let d_grad_v = ImplicitProblem::velocity_gradient(&p.entries, dv);
                let dp = apply_dtau_dl_psd_matrix(&matrix, d_grad_v);
                for &(idx, _w, grad) in &p.entries {
                    out[idx] += p.v0 * (dp * grad);
                }
            }
            out
        };

        let x_expected = [
            Vec2::new(0.11, -0.07),
            Vec2::new(-0.05, 0.09),
            Vec2::new(0.06, 0.02),
            Vec2::new(-0.03, -0.04),
        ];
        let b = apply_reference(&x_expected);

        let matrices = vec![matrix, matrix];
        let free_of = ImplicitProblem::free_dof_map(&problem.wall_frozen);
        let n_free = free_of.iter().filter(|f| f.is_some()).count();
        let (a, assembled_b) = problem.assemble_free_dof_system(&matrices, &free_of, n_free, &b);
        let x = ImplicitProblem::cholesky_solve(&a, 2 * n_free, &assembled_b).expect(
            "assembled matrix must be SPD (positive mass/dt diagonal + arbitrary SPD blocks)",
        );

        for i in 0..n_free {
            let got = Vec2::new(x[2 * i], x[2 * i + 1]);
            let want = x_expected[i];
            let diff = (got - want).length();
            assert!(
                diff < 1.0e-3,
                "direct_solve did not recover the known solution at free dof {i}: got={got:?} want={want:?}"
            );
        }
    }

    /// Real end-to-end exercise of `direct_solve` ITSELF (not just its
    /// sub-components `assemble_free_dof_system`/`cholesky_solve`, which
    /// the test above already covers with a hand-fabricated matrix) --
    /// with REAL, nonzero `lambda`/`mu`/non-identity `f_n`, so `build_
    /// particle_matrices` computes a genuine eigenvalue-clamped operator
    /// from the actual corotated stress JVP, not a fixture. Verifies
    /// `direct_solve`'s own returned `delta` actually solves the SAME
    /// system `assemble_free_dof_system` would build from `build_
    /// particle_matrices(v)` -- the real, closed loop this "kept as a
    /// diagnostic oracle" function needs to still demonstrably work.
    #[test]
    fn direct_solve_end_to_end_with_real_material_parameters() {
        let entries: [(usize, f32, Vec2); 9] = [
            (0, 0.2, Vec2::new(0.4, 0.1)),
            (1, 0.15, Vec2::new(-0.1, 0.3)),
            (2, 0.1, Vec2::new(0.2, -0.2)),
            (0, 0.05, Vec2::new(-0.1, 0.05)),
            (1, 0.2, Vec2::new(0.3, -0.1)),
            (2, 0.1, Vec2::new(-0.05, 0.15)),
            (0, 0.1, Vec2::new(0.15, 0.2)),
            (1, 0.05, Vec2::new(-0.05, -0.1)),
            (2, 0.05, Vec2::new(0.1, -0.05)),
        ];
        let problem = ImplicitProblem {
            particles: vec![ImplicitParticle {
                lambda: 3.0e5,
                mu: 2.0e5,
                f_n: Mat2::from_cols(Vec2::new(1.05, 0.04), Vec2::new(-0.03, 0.97)),
                v0: 0.041,
                entries,
            }],
            node_pos: vec![IVec2::ZERO, IVec2::ZERO, IVec2::ZERO],
            node_mass: vec![0.4, 0.6, 0.5],
            v_n: vec![Vec2::ZERO; 3],
            ext_force: vec![Vec2::new(0.0, -0.02); 3],
            dt: 0.016,
            wall_frozen: vec![false, false, false],
        };
        let v = problem.v_n.clone();
        let r = problem.residual(&v);
        let neg_r: Vec<Vec2> = r.iter().map(|x| -*x).collect();

        let delta = problem
            .direct_solve(&v, &neg_r)
            .expect("a real corotated system at these parameters must stay SPD");

        // Re-derive the SAME assembled system independently and confirm
        // `delta` actually solves it (`A*delta == b` on the free DOFs).
        let matrices = problem.build_particle_matrices(&v);
        let free_of = ImplicitProblem::free_dof_map(&problem.wall_frozen);
        let n_free = free_of.iter().filter(|f| f.is_some()).count();
        let (a, b) = problem.assemble_free_dof_system(&matrices, &free_of, n_free, &neg_r);
        let mut a_delta = vec![0.0f32; 2 * n_free];
        for (i, &maybe_free) in free_of.iter().enumerate() {
            if let Some(fi) = maybe_free {
                for j in 0..(2 * n_free) {
                    a_delta[j] += a[j * (2 * n_free) + 2 * fi] * delta[i].x
                        + a[j * (2 * n_free) + 2 * fi + 1] * delta[i].y;
                }
            }
        }
        for j in 0..(2 * n_free) {
            let diff = (a_delta[j] - b[j]).abs();
            let scale = b[j].abs().max(1.0);
            assert!(
                diff / scale < 1.0e-2,
                "direct_solve's own delta does not solve A*delta=b at row {j}: a_delta={:.4e} b={:.4e}",
                a_delta[j],
                b[j]
            );
        }
    }
}

#[cfg(test)]
mod trust_region_consistency_tests {
    use super::*;

    /// A real, non-trivial 2-particle problem (3 shared nodes, non-
    /// identity `f_n` so the exponential-map subtlety this module's own
    /// `deformed_f` doc flags is actually exercised, not sidestepped by
    /// testing only at the trivial `F_n=I` state) -- shared by both FD
    /// checks below.
    fn synthetic_problem() -> ImplicitProblem {
        let entries_a: [(usize, f32, Vec2); 9] = [
            (0, 0.2, Vec2::new(0.4, 0.1)),
            (1, 0.15, Vec2::new(-0.1, 0.3)),
            (2, 0.1, Vec2::new(0.2, -0.2)),
            (0, 0.05, Vec2::new(-0.1, 0.05)),
            (1, 0.2, Vec2::new(0.3, -0.1)),
            (2, 0.1, Vec2::new(-0.05, 0.15)),
            (0, 0.1, Vec2::new(0.15, 0.2)),
            (1, 0.05, Vec2::new(-0.05, -0.1)),
            (2, 0.05, Vec2::new(0.1, -0.05)),
        ];
        let entries_b: [(usize, f32, Vec2); 9] = [
            (1, 0.2, Vec2::new(0.2, -0.15)),
            (2, 0.15, Vec2::new(-0.3, 0.1)),
            (0, 0.1, Vec2::new(0.1, 0.25)),
            (1, 0.05, Vec2::new(-0.2, 0.1)),
            (2, 0.2, Vec2::new(0.25, -0.05)),
            (0, 0.1, Vec2::new(-0.1, -0.2)),
            (1, 0.1, Vec2::new(0.1, 0.1)),
            (2, 0.05, Vec2::new(-0.15, -0.1)),
            (0, 0.05, Vec2::new(0.05, 0.15)),
        ];
        let n = 3;
        ImplicitProblem {
            particles: vec![
                ImplicitParticle {
                    lambda: 3.0e5,
                    mu: 2.0e5,
                    f_n: Mat2::from_cols(Vec2::new(1.05, 0.04), Vec2::new(-0.03, 0.97)),
                    v0: 0.041,
                    entries: entries_a,
                },
                ImplicitParticle {
                    lambda: 2.5e5,
                    mu: 1.8e5,
                    f_n: Mat2::from_cols(Vec2::new(0.98, -0.02), Vec2::new(0.05, 1.02)),
                    v0: 0.033,
                    entries: entries_b,
                },
            ],
            node_pos: (0..n).map(|_| IVec2::ZERO).collect(),
            node_mass: vec![0.4, 0.6, 0.5],
            v_n: vec![
                Vec2::new(0.1, -0.05),
                Vec2::new(-0.02, 0.03),
                Vec2::new(0.04, 0.01),
            ],
            ext_force: vec![
                Vec2::new(0.0, -0.02),
                Vec2::new(0.0, -0.03),
                Vec2::new(0.0, -0.025),
            ],
            dt: 0.016,
            wall_frozen: vec![false, false, false],
        }
    }

    /// Minimal hand-verifiable case (single particle, single active
    /// stencil entry, `F_n=I`, one node) that root-caused `model_
    /// residual`'s real formula bug (2026-09-11): the true FD gradient of
    /// `total_energy` at this exact state is `13.9177` -- `residual`'s own
    /// `tau*grad` convention gives `487.03` (>30x off, confirmed wrong for
    /// THIS purpose), while `inertia (6.25) + dt*Piola*(F_n^T*grad)
    /// (7.68)` gives `13.93`, matching to 0.09%. Kept as a real, cheap,
    /// exactly-reproducible regression guard for that fix, not just a
    /// diagnostic artifact.
    #[test]
    fn model_residual_matches_energy_gradient_in_a_hand_verifiable_minimal_case() {
        let entries: [(usize, f32, Vec2); 9] = [
            (0, 1.0, Vec2::new(1.0, 0.0)),
            (0, 0.0, Vec2::ZERO),
            (0, 0.0, Vec2::ZERO),
            (0, 0.0, Vec2::ZERO),
            (0, 0.0, Vec2::ZERO),
            (0, 0.0, Vec2::ZERO),
            (0, 0.0, Vec2::ZERO),
            (0, 0.0, Vec2::ZERO),
            (0, 0.0, Vec2::ZERO),
        ];
        let problem = ImplicitProblem {
            particles: vec![ImplicitParticle {
                lambda: 1.0e5,
                mu: 1.0e5,
                f_n: Mat2::IDENTITY,
                v0: 1.0,
                entries,
            }],
            node_pos: vec![IVec2::ZERO],
            node_mass: vec![1.0],
            v_n: vec![Vec2::ZERO],
            ext_force: vec![Vec2::ZERO],
            dt: 0.016,
            wall_frozen: vec![false],
        };
        let v = vec![Vec2::new(0.1, 0.05)];
        let h = 1.0e-2f32;

        let mut v_plus = v.clone();
        v_plus[0] += h * Vec2::X;
        let mut v_minus = v.clone();
        v_minus[0] -= h * Vec2::X;
        let numeric = (problem.total_energy(&v_plus) - problem.total_energy(&v_minus)) / (2.0 * h);
        let analytic = problem.model_residual(&v)[0].x;

        println!("numeric={numeric} analytic={analytic}");
        let rel_err = (numeric - analytic).abs() / numeric.abs().max(1.0);
        assert!(
            rel_err < 5.0e-3,
            "model_residual doesn't match total_energy's gradient in the minimal case: numeric={numeric} analytic={analytic} rel_err={rel_err}"
        );
    }

    /// Real, run verification (not an assumed derivation) that `model_
    /// residual` is `total_energy`'s own gradient -- central finite
    /// difference at `h=1e-2`, NOT this file's/`materials::utils`'s usual
    /// `h=1e-3` convention. Real, measured reason, not an arbitrary choice:
    /// a genuine h-convergence sweep (`h=1e-2,1e-3,1e-4` -> max_rel_err
    /// `0.30%, 2.17%, 8.22%`) shows error GROWING as `h` shrinks -- the
    /// exact opposite of what a real formula bug would produce (which
    /// would stay roughly constant or shrink toward some genuine residual
    /// as `h`->0), and the exact textbook signature of f32 catastrophic
    /// cancellation in `total_energy`'s own central difference (summing
    /// large `O(lambda)~2.5-3e5`-scale terms into ONE scalar before
    /// differencing is far more cancellation-prone than this codebase's
    /// usual matrix-valued JVP checks, which difference component-wise).
    /// `h=1e-2` is the point in that sweep where truncation error still
    /// dominates over float noise. This is the SAME "smaller h is WORSE"
    /// lesson `materials::utils`'s own corotated JVP tests already
    /// document, applied at a coarser `h` because THIS check differences a
    /// scalar built from a stiffer-scale sum, not a stress matrix.
    ///
    /// This is the load-bearing assumption behind the trust-region ratio
    /// test (`newton_solve`): if it didn't hold, `predicted_reduction`
    /// (built from `model_residual`/`model_jacobian_vector_product`) and
    /// `actual_reduction` (built from `total_energy`) would be comparing
    /// two different, inconsistent quantities. Deliberately checks `model_
    /// residual` (linear `model_deformed_f`), NOT the real exponential-map
    /// `residual` -- a real, failed first attempt at this exact test
    /// (max_rel_err=1.28, 128%) proved the true `residual` is NOT this
    /// energy's gradient (the matrix exponential's own Fréchet derivative
    /// would be needed for that); a second real, failed attempt
    /// (max_rel_err=1.24, still ~124%) additionally found `model_residual`
    /// itself needed a genuine formula fix -- Piola (not Kirchhoff) paired
    /// with `F_n^T*grad` (not the raw spatial `grad`), scaled by `dt` --
    /// confirmed via a hand-verifiable minimal case
    /// (`debug_minimal_single_entry_case`, single particle/entry/node,
    /// `F_n=I`, matches to 0.09%) before trusting it here.
    #[test]
    fn energy_gradient_matches_model_residual_via_finite_difference() {
        let problem = synthetic_problem();
        let n = problem.node_pos.len();
        let v = vec![
            Vec2::new(0.12, -0.04),
            Vec2::new(-0.01, 0.05),
            Vec2::new(0.06, 0.02),
        ];
        let analytic = problem.model_residual(&v);
        let h = 1.0e-2f32;
        let mut max_rel_err = 0.0f32;
        for i in 0..n {
            for &axis in &[Vec2::X, Vec2::Y] {
                let mut v_plus = v.clone();
                let mut v_minus = v.clone();
                v_plus[i] += h * axis;
                v_minus[i] -= h * axis;
                let numeric =
                    (problem.total_energy(&v_plus) - problem.total_energy(&v_minus)) / (2.0 * h);
                let analytic_component = analytic[i].dot(axis);
                let rel_err =
                    (analytic_component - numeric).abs() / analytic_component.abs().max(1.0);
                max_rel_err = max_rel_err.max(rel_err);
            }
        }
        println!("energy_gradient_matches_model_residual: max_rel_err={max_rel_err:.6}");
        assert!(
            max_rel_err < 5.0e-3,
            "model_residual is not total_energy's gradient: max_rel_err={max_rel_err}"
        );
    }

    /// Real, run verification that `model_jacobian_vector_product` is
    /// `model_residual`'s own derivative (the Hessian `steihaug_cg`
    /// needs) -- central finite difference of `model_residual` itself
    /// along a probe direction, same h=1e-3 convention.
    #[test]
    fn model_jvp_matches_finite_difference_of_model_residual() {
        let problem = synthetic_problem();
        let n = problem.node_pos.len();
        let v = vec![
            Vec2::new(0.12, -0.04),
            Vec2::new(-0.01, 0.05),
            Vec2::new(0.06, 0.02),
        ];
        let directions: Vec<Vec<Vec2>> = vec![
            vec![Vec2::new(1.0, 0.0), Vec2::ZERO, Vec2::ZERO],
            vec![Vec2::ZERO, Vec2::new(0.0, 1.0), Vec2::ZERO],
            vec![
                Vec2::new(0.3, -0.2),
                Vec2::new(-0.1, 0.4),
                Vec2::new(0.2, 0.1),
            ],
        ];
        let h = 1.0e-3f32;
        let mut max_rel_err = 0.0f32;
        for dv in &directions {
            let v_plus: Vec<Vec2> = (0..n).map(|i| v[i] + h * dv[i]).collect();
            let v_minus: Vec<Vec2> = (0..n).map(|i| v[i] - h * dv[i]).collect();
            let r_plus = problem.model_residual(&v_plus);
            let r_minus = problem.model_residual(&v_minus);
            let numeric: Vec<Vec2> = (0..n)
                .map(|i| (r_plus[i] - r_minus[i]) / (2.0 * h))
                .collect();
            let analytic = problem.model_jacobian_vector_product(&v, dv);
            for i in 0..n {
                let diff = (analytic[i] - numeric[i]).length();
                let rel_err = diff / analytic[i].length().max(1.0);
                max_rel_err = max_rel_err.max(rel_err);
            }
        }
        println!(
            "model_jvp_matches_finite_difference_of_model_residual: max_rel_err={max_rel_err:.6}"
        );
        assert!(
            max_rel_err < 5.0e-3,
            "model_jacobian_vector_product is not model_residual's derivative: max_rel_err={max_rel_err}"
        );
    }

    /// Real correctness check for `real_residual_jvp_fd` itself (kept
    /// `#[cfg(test)]`, see that function's own doc) -- confirms its own
    /// unit-normalize/rescale convention gives the same answer as an
    /// independently-written, non-normalized central difference of the
    /// REAL `residual`, on `synthetic_problem`'s own small hand-built
    /// system. Real, measured note: this only agrees at the SAME `h=1e-2`
    /// the helper itself uses -- a first attempt at `h=1e-3` disagreed by
    /// 19%, not a bug (confirmed a pure step-size effect, not a formula
    /// error: `residual`'s exponential-map nonlinearity has its own
    /// truncation-error behavior, genuinely different from `model_
    /// residual`'s linear one at this problem's stiffness). This function
    /// was the decisive tool that ruled out `model_jacobian_vector_
    /// product`'s own approximation as `basic_sand`'s real production
    /// stall (see `implicit_corotated`'s own top-of-file doc, "the real,
    /// proven root cause") -- kept correct and callable for whatever
    /// future curvature construction needs checking against the TRUE
    /// residual's own Jacobian next.
    #[test]
    fn real_residual_jvp_fd_matches_a_naive_non_normalized_central_difference() {
        let problem = synthetic_problem();
        let n = problem.node_pos.len();
        let v = vec![
            Vec2::new(0.12, -0.04),
            Vec2::new(-0.01, 0.05),
            Vec2::new(0.06, 0.02),
        ];
        let dv = vec![
            Vec2::new(0.3, -0.2),
            Vec2::new(0.1, 0.4),
            Vec2::new(-0.2, 0.1),
        ];
        let via_helper = problem.real_residual_jvp_fd(&v, &dv);

        let h = 1.0e-2f32;
        let v_plus: Vec<Vec2> = (0..n).map(|i| v[i] + h * dv[i]).collect();
        let v_minus: Vec<Vec2> = (0..n).map(|i| v[i] - h * dv[i]).collect();
        let r_plus = problem.residual(&v_plus);
        let r_minus = problem.residual(&v_minus);
        let naive: Vec<Vec2> = (0..n)
            .map(|i| (r_plus[i] - r_minus[i]) / (2.0 * h))
            .collect();

        let mut max_rel_err = 0.0f32;
        for i in 0..n {
            let diff = (via_helper[i] - naive[i]).length();
            let rel_err = diff / naive[i].length().max(1.0);
            max_rel_err = max_rel_err.max(rel_err);
        }
        println!("real_residual_jvp_fd_matches_naive: max_rel_err={max_rel_err:.6}");
        assert!(
            max_rel_err < 5.0e-2,
            "real_residual_jvp_fd disagrees with a naive non-normalized central difference: max_rel_err={max_rel_err}"
        );
    }
}
