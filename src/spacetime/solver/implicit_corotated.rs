//! Opt-in implicit (Newton-CG) grid-velocity update for scenes built
//! entirely from the shared Corotated elastic branch (`MaterialModel::
//! corotated_lame_params`): DruckerPrager (sand), Corotated, VonMises,
//! Rankine, DruckerPragerMuI. Klar 2016 operator split: solve the velocity
//! field as Corotated elasticity by Newton-CG at the full frame `dt`, then
//! apply each particle's own plastic return mapping
//! (`MaterialModel::update_particle`, through the ordinary G2P pass) once.
//!
//! `eligible` requires the whole active scene to qualify (every material on
//! the shared elastic branch; no rods, grains, contact, mixture, ASFLIP,
//! Cundall damping, pressure projection, pinned or sleeping particles). A
//! solve that does not reach its relative tolerance leaves `self.particles`/
//! `self.grid` untouched. Either way the explicit substep loop runs instead,
//! so `SimConfig::implicit_corotated_elastic` is always safe to turn on.
//!
//! # Status: correct at small scale, inert at `basic_sand` scale
//!
//! At 256 particles a settled pile matches the explicit baseline (0.0000
//! cells, `tests/implicit_corotated_substep.rs`), and a dropped pile drifts
//! ~3.05 cells over 20 frames, within that test's tolerance. At
//! `basic_sand.rs`'s scale (1008 particles, E = 15 MPa DruckerPrager) the
//! full-frame solve stalls at ~99.6% residual reduction, short of
//! `RELATIVE_TOLERANCE`, so every frame falls back to explicit.
//!
//! The cause is single-giant-step Gauss-Newton stagnation: the same stuck
//! state converges (99.76%) when Newton covers `dt/400` to `dt/500` and
//! fails at `dt/300` and every larger fraction. Load stepping that fine
//! needs ~350-400 solves per frame at ~10 ms each, ~4.2-4.7 s per frame
//! against ~145-180 ms explicit: 25-30x slower. A speedup would need a
//! per-solve cost about two orders of magnitude lower (e.g. multigrid
//! preconditioning), a different solver, not a tuning pass.
//!
//! Ruled out by measurement on the stuck state: linear-solver quality (an
//! exact dense Cholesky solve stalls identically); the approximate curvature
//! (it underestimates the true Jacobian 62-70x, but the exact FD Jacobian
//! stalls too); Jacobian asymmetry (0.1-6%); residual concentrated at the
//! wall (it is not); the presence of frozen wall DOFs (freeing all still
//! stalls); heterogeneous or large `F_n` (forcing `F_n = I` still stalls);
//! near-zero-mass nodes alone (freezing the 6 below 1e-4 still stalls). Why a
//! synthetic 2025-particle benchmark without walls converges while this
//! scene does not is still open; a combination (near-zero-mass nodes next to
//! frozen ones) is the next candidate.
//!
//! Kept `#[cfg(test)]` as oracles for that work: the dense direct-solve
//! chain (`direct_solve`, `assemble_free_dof_system`,
//! `particle_stiffness_block`, `cholesky_solve`, `free_dof_map`,
//! `grad_basis`, `build_particle_matrices`), the eigenvalue-clamp PSD
//! projection chain in `materials::utils`
//! (`corotated_kirchhoff_dtau_dl_psd_matrix`, `spd_project_symmetric_4x4`,
//! `jacobi_eigen_symmetric_4x4`, `corotated_elastic_energy_density`), the
//! energy formulation `model_residual`/`total_energy` that
//! `model_jacobian_vector_product` is checked against, and
//! `real_residual_jvp_fd`, the exact finite-difference Jacobian.

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

/// Max Newton iterations and relative tolerance: the values of the probe
/// `stage3_dp_multi_particle_real_wall_clock_speedup_vs_real_explicit` at
/// basic_sand's scale.
const MAX_NEWTON_ITERS: usize = 150;
/// Newton converges once the residual falls below this fraction of
/// `max(r0_norm, |ext_force|)` (see `newton_solve`).
const RELATIVE_TOLERANCE: f32 = 1.0e-3;
/// `steihaug_cg`'s own iteration budget, separate from `MAX_NEWTON_ITERS`:
/// the outer Newton loop is expensive, the inner matrix-free CG cheap, and
/// sharing one budget starved CG on basic_sand-scale problems.
const MAX_CG_ITERS: usize = 400;
/// Line-search admissibility floor on `det(F)` -- see `ImplicitProblem::
/// min_deformed_j`'s doc. Comfortably above `corotated_elastic_
/// stress`'s `MIN_J=1e-6` hard-zero clamp (that discontinuity is exactly
/// what a residual-only acceptance test can be fooled by), while still
/// permissive enough to allow large compaction under a genuinely
/// stiff sand pile's own weight.
const MIN_ADMISSIBLE_J: f32 = 0.1;

/// One particle's 9-node quadratic-kernel stencil: node position, weight,
/// and the exact analytic kernel gradient (`axis_weights_derivative`), not
/// the `weight*cell_dist*KERNEL_D_INVERSE` MLS-MPM quadrature (Hu et al.
/// 2018) the explicit P2G/G2P uses.
///
/// Implicit MPM codes assemble the residual and its Hessian-vector product
/// from the exact shape-function gradient even where they use MLS for
/// explicit kinematics: `ziran2020` requires `symplectic==true` for
/// `mls_mpm` and every implicit demo sets `mls_mpm=false`, its force
/// assembly (`MpmForceBase.cpp::rasterizeForceToTVStack`,
/// `FBasedMpmForceHelper.cpp::computeStressDifferential`) uses
/// `BSplineWeights`' analytic derivative; GeoTaichi's
/// `NewtonIteration.py::assemble_element_local_stiffness_2D` uses
/// `dshape_fn`. MLS consistency is per particle in one explicit step;
/// summed over many particles at shared nodes it is not, which fits an
/// isolated particle matching the explicit baseline while a packed pile
/// diverged from frame one.
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
    /// `true` for a node within `SimConfig::boundary_thickness` of a wall.
    /// A pile on a floor needs the wall reaction in continuous balance with
    /// gravity; one free solve with the wall applied afterwards cannot
    /// represent that (an 8-particle clump matched explicit until it touched
    /// a wall, then diverged). These DOFs stay at `v_n` through the free
    /// search, the essential-boundary principle of `Particle::pinned`/
    /// `Grid::pinned_nodes`; `apply_boundary_conditions_to_grid` still runs
    /// afterwards.
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

    /// Exact closed-form `exp(dt*grad_v)*F_n`, applied as
    /// `F_n + (exp(dt*grad_v) - I)*F_n` (`advance_deformation_gradient`, whose
    /// doc has why the increment is never formed as one plus a small
    /// number), not the linear `(I+dt*grad_v)*F_n`. A settled DruckerPrager particle's
    /// rest `F_n` is a pure rotation (zero corotated stress), ~34 degrees in
    /// `diag_properly_isolated_equilibrium_maintenance`; composing the linear
    /// increment onto it is not orthogonal to first order, and the spurious
    /// strain it makes gave Newton an artifact to chase (a settled pile
    /// drifted several cells although every solve converged).
    ///
    /// `jacobian_vector_product` keeps the linear `df`, not the exponential
    /// map's Fréchet derivative: the line search always re-checks this exact
    /// residual, so the approximate direction only affects convergence speed,
    /// never what the solve converges to.
    fn deformed_f(p: &ImplicitParticle, v: &[Vec2], dt: f32) -> Mat2 {
        let grad_v = Self::velocity_gradient(&p.entries, v);
        crate::materials::utils::advance_deformation_gradient(p.f_n, dt * grad_v)
    }

    /// The LINEAR (forward-Euler-style) `(I+dt*grad_v)*F_n` `deformed_f`'s
    /// doc deliberately moved away from for the residual (exact
    /// exponential map fixes real spurious-strain error at large
    /// pre-existing rotation, see that doc). Used ONLY to build a
    /// self-consistent (energy, gradient, Hessian) triple for `newton_
    /// solve`'s trust-region MODEL -- `model_residual` is THIS linear F's
    /// exact gradient and `model_jacobian_vector_product` is its exact
    /// Hessian-vector product (both verified by real FD tests, not
    /// assumed: `trust_region_consistency_tests`), unlike the true
    /// exponential-map `residual`, which is NOT the gradient of any energy
    /// built the same way (the matrix exponential's own Fréchet derivative
    /// would be needed for that, substantial extra math this file
    /// deliberately avoids -- confirmed necessary by a failed FD
    /// check attempting to skip it). The trust-region model only decides
    /// SEARCH DIRECTION and STEP SIZE; `newton_solve`'s actual convergence
    /// test and final acceptance always use the `residual`/`min_
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
    /// clamp is a cliff in the residual: crossing it makes `|r|` drop (a
    /// large stress term vanished), so an `|r|`-only acceptance test walks a
    /// particle into that zero-support regime and calls it progress
    /// (`diag_settled_pile_first_forked_step_velocity_field`: a settled pile
    /// reached near free-fall speed in the first implicit frame).
    /// `newton_solve` also requires this to stay above `MIN_J` before
    /// accepting a step.
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
            // conversion). First-Piola stress (`tau * F^-T`) here would add a
            // spurious `F^-T` factor, since it pairs with the material
            // gradient, not a spatial one. With it, a settled pile diverged 8+
            // grid cells from the explicit baseline in one frame
            // (`tests/probes/implicit_corotated_wiring_diagnostic.rs`'s
            // multi-frame trajectory comparison). Finite-difference checks
            // cannot catch this: they verify the JVP against its own residual
            // formula, not the formula against this engine's force convention.
            let stress = corotated_elastic_stress(f_new, p.lambda, p.mu);
            for &(idx, _w, grad) in &p.entries {
                r[idx] += p.v0 * (stress * grad);
            }
        }
        r
    }

    /// The trust-region model's residual: `total_energy`'s exact gradient,
    /// on the linear `model_deformed_f` (the true `residual` would need the
    /// matrix exponential's Fréchet derivative to make that claim).
    ///
    /// Through `F_new(v) = (I+dt*grad_v)*F_n` and the identity
    /// `frob(P,(e⊗grad)*F_n) = e·(P*F_n^T*grad)`, the elastic term is
    /// `dt*V0*Piola*(F_n^T*grad)`: Piola, not Kirchhoff, paired with
    /// `F_n^T*grad`, scaled by `dt` (`debug_minimal_single_entry_case`: FD
    /// 13.918 against 487.0 for `tau*grad`). `residual`'s `tau*grad` is the
    /// production MPM force (as in `spacetime::transfer::p2g`); it is simply
    /// not the gradient of a `Psi(F_new(v))` energy under a multiplicative
    /// update.
    ///
    /// Test-only: `newton_solve` uses the `residual` as its gradient
    /// (see its doc); this stays as the oracle showing
    /// `model_jacobian_vector_product` is the Hessian-vector product of a
    /// real energy's gradient.
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
    /// -- the "optimization time integration" form of implicit Euler MPM
    /// (Gast et al. 2015 "Optimization Integrator for Large Time Steps"; Klar
    /// 2016's implicit sand solve is framed the same way), on
    /// `model_deformed_f`.
    ///
    /// Test-only: `newton_solve`'s ratio test measures `0.5*||residual||^2`
    /// (Gauss-Newton merit, Moré 1978). Kept as `model_residual`'s gradient
    /// oracle (`model_residual_matches_energy_gradient_in_a_hand_verifiable_
    /// minimal_case`).
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
    /// Hessian, not a PSD projection of it (see this module's doc,
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

    /// Test-only: finite-difference Jacobian-vector product of the true,
    /// exponential-map `residual`. Normalizes `dv` to unit length before
    /// perturbing (`h = 1e-2` on a unit direction stays clear of f32
    /// cancellation), then rescales by `dv`'s norm (a JVP is linear in `dv`).
    ///
    /// At the point `newton_solve` stalls, `model_jacobian_vector_product`
    /// underestimates this Jacobian 62-70x along the free residual
    /// (`cos_similarity` ~0.9999), yet using this exact JVP as `steihaug_cg`'s
    /// curvature still stalls (r_norm floor 317 -> 276), so the approximation
    /// is not the cause (see the module doc).
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
    /// converges to (that's `steihaug_cg`'s own exact recurrence).
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
    /// ed., Algorithm 4.3 and its "Preconditioning" section): approximately
    /// minimizes `m(p) = g.p + 0.5*p.H.p` subject to `||p|| <= radius`, with
    /// `model_jacobian_vector_product` matrix-free. It detects negative
    /// curvature itself (stopping on the trust-region boundary), so the
    /// operator needs no PSD projection, which blinded Newton to curvature it
    /// still needed after its first step.
    ///
    /// Jacobi preconditioning (`jacobi_diagonal`, preconditioned `r^T*y`
    /// inner products for `alpha`/`beta`) took the basic_sand-scale
    /// reduction from 97.7% to 99.6%. Only the search direction is
    /// preconditioned; the boundary check stays in the Euclidean norm the
    /// real residual's scale is calibrated to, since an `M`-norm boundary
    /// would reopen the radius/scale mismatch `newton_solve` avoids.
    ///
    /// `hd` is projected onto the free-DOF subspace every iteration, not only
    /// the final `p`: `model_jacobian_vector_product` knows nothing of
    /// `wall_frozen`, so a free-only `d` still gives a nonzero `hd` at a
    /// frozen node sharing a particle with a moving neighbour, which would
    /// leak through `r_next`, the preconditioner and `beta` into `d` from the
    /// second iteration on. Projecting every iteration is standard Dirichlet
    /// practice (ziran2020's preconditioned CG, `tmp/ref_ziran_implicit_
    /// mpm.md`). The leak was real (`hd_frozen_norm` up to ~465) but not the
    /// cause of the stall.
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
    /// matrix` for why this is the expensive part -- 4 JVP evaluations per
    /// particle). `v` is fixed for one `conjugate_gradient_solve`, so this
    /// runs once per Newton iteration; rebuilding it per CG iteration made
    /// the implicit path 0.73x the explicit speed at basic_sand scale.
    ///
    /// Test-only, with the rest of the direct-solve chain it feeds
    /// (`free_dof_map`, `grad_basis`, `particle_stiffness_block`,
    /// `assemble_free_dof_system`, `cholesky_solve`, `direct_solve`):
    /// production uses `steihaug_cg`. Kept as the oracle showing the stall is
    /// not a linear-solver problem (see the module doc).
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
    /// and zeroed afterward -- a smaller, better-posed subsystem, not
    /// just a masking step).
    ///
    /// Test-only -- see `build_particle_matrices`'s doc for why this
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
    /// Test-only -- see `build_particle_matrices`'s doc.
    #[cfg(test)]
    fn grad_basis(g: Vec2) -> [[f32; 2]; 4] {
        [[g.x, 0.0], [0.0, g.x], [g.y, 0.0], [0.0, g.y]]
    }

    /// Closed-form per-particle-pair 2x2 stiffness block, the explicit matrix
    /// of the linear map `jacobian_vector_product` applies matrix-free:
    /// `dr_i = v0 * (B_i^T * M * B_j) * dv_j` for stencil nodes `i`, `j` of one
    /// particle, with the eigenvalue-clamped 4x4 `matrix`
    /// (`corotated_kirchhoff_dtau_dl_psd_matrix`). Standard local-stiffness
    /// assembly (as in GeoTaichi's `assemble_element_local_stiffness_2D`),
    /// checked by `assembled_operator_matches_matrix_free_jacobian_vector_
    /// product`.
    ///
    /// Test-only -- see `build_particle_matrices`.
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
    /// Test-only -- see `build_particle_matrices`'s doc.
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
    /// Test-only -- see `build_particle_matrices`'s doc.
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

    /// Direct solve: assembles the dense free-DOF system once and factors it
    /// once. Returns a full-length `delta` (zero at every `wall_frozen` node).
    /// `None` only if the assembled matrix fails the SPD invariant.
    ///
    /// Test-only: production uses `steihaug_cg`. Kept as the proof
    /// (`direct_solve_actually_solves_the_assembled_system`) that the
    /// basic_sand-scale stall is not a linear-solver problem.
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

    /// Per-node mass-normalized residual norm, `sum(|r_i|^2 / mass_i)`, over
    /// free nodes only.
    ///
    /// Mass-normalized as in ziran2020 (`BackwardEuler.h::
    /// BackwardEulerLagrangianForceObjective::computeNorm`): every term of a
    /// low-mass node's residual (`m*(v-vn)/dt`, `m*g`, its elastic support)
    /// shrinks with its mass, so in a plain sum it looks converged at almost
    /// any velocity, free fall included, while interior nodes dominate
    /// (measured: Newton reported convergence with particles still near free
    /// fall). HOT (`MultigridSimulation.h::computeCharacteristicNorm`) builds
    /// its tolerance from the material stiffness for the same reason.
    ///
    /// `wall_frozen` nodes are excluded: they stay at `v_n` through the whole
    /// search, so their residual cannot be reduced. At basic_sand scale they
    /// are 145 of 462 DOFs with a residual (~5.5-5.8e4) comparable to the
    /// free ones (~5.2-6.3e4), which made `RELATIVE_TOLERANCE = 1e-3`
    /// unreachable for any solver. Only the convergence judgment excludes
    /// them; the full `residual` is still what goes back to the grid.
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
    /// subproblem, Algorithm 4.3). Backtracking can only rescale one Newton
    /// direction; at basic_sand scale one step gave 21% and then no scale
    /// helped. On rejection a trust region shrinks and re-solves, which can
    /// return a different, more steepest-descent-like direction.
    ///
    /// The model (`model_jacobian_vector_product`, see `model_deformed_f`
    /// and `model_residual`, checked by `trust_region_consistency_tests`)
    /// only picks a direction and sizes the region; acceptance is always
    /// gated on the exponential-map `residual` improving. Returns `None`
    /// if it does not converge within `MAX_NEWTON_ITERS` or the region
    /// collapses without helping: callers then run the explicit substep.
    ///
    /// The outer tolerance is anchored to the larger of `r0_norm` and
    /// `ext_force`'s magnitude, since a particle at rest starts with a tiny
    /// `r0_norm`.
    fn newton_solve(&self, v_init: &[Vec2]) -> Option<Vec<Vec2>> {
        let n = self.node_pos.len();
        let mut v = v_init.to_vec();
        let mut r = self.residual(&v);
        let mut r_norm = self.free_mass_normalized_norm(&r);
        let r0_norm = r_norm.max(1.0e-12);
        let ext_force_norm = self.free_mass_normalized_norm(&self.ext_force);
        let scale = r0_norm.max(ext_force_norm).max(1.0e-12);

        // Trust-region radius cap and give-up floor: 1e-12 is the low end of
        // the step window measured to improve the residual (see below); 1e8
        // only keeps repeated doubling finite.
        const RADIUS_MAX: f32 = 1.0e8;
        const RADIUS_MIN: f32 = 1.0e-12;
        const MAX_RADIUS_RETRIES: usize = 80;
        let diag = crate::diagnostics::research_switch("EMERGE_IMPLICIT_DIAG").is_some();

        // Gauss-Newton trust region for the nonlinear equation
        // `residual(v) = 0` (Nocedal & Wright 2nd ed. ch.10; Moré 1978), not
        // an energy minimization. The real `residual` is the gradient `g`;
        // `model_jacobian_vector_product` only supplies approximate
        // curvature; predicted/actual reduction is measured in
        // `0.5*||r||^2`. Using `model_residual` (Piola, `dt`-scaled) as `g`
        // mis-scaled the radius: stepping along the residual's negative
        // gradient by eps = 1e-8 to 1e-12 improved it (1e-6 overshot), a window
        // the old radius never reached.
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
    /// yet is active in the scene -- see this module's doc for the
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

    /// Assembly and solve for one implicit big step. Returns `false` (and
    /// touches nothing) when the scene is ineligible or the solve does not
    /// converge; see the module doc.
    pub(crate) fn try_implicit_corotated_substep(&mut self, dt: f32) -> bool {
        if !self.implicit_corotated_eligible() {
            return false;
        }

        // ── Mass-only P2G: same real APIC mass/momentum accumulation
        // `scatter_particles_to_grid` uses, minus its force term -- the
        // force this substep applies comes from the Newton-CG residual's
        // own `ext_force`/internal-stress terms below, not from baking a
        // single-F-linearized force into the scatter the way the explicit
        // path does (see `spacetime::transfer::p2g`'s doc for that
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

        // Diagnostic, opt-in through the `EMERGE_IMPLICIT_DIAG` research switch.
        // Built for a settled DruckerPrager pile drifting from the explicit
        // baseline although each solve converged, fixed by
        // `ImplicitProblem::wall_frozen` (regression test
        // `implicit_matches_explicit_for_an_already_settled_pile` in
        // `tests/implicit_corotated_substep.rs`). A violent impact still
        // diverges more (`violent_impact_diverges_more_than_settled_pile_a_
        // real_disclosed_limitation`).
        let diag = crate::diagnostics::research_switch("EMERGE_IMPLICIT_DIAG").is_some();
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

        // Unmodified G2P: gathers velocity, position and APIC C from the
        // solved, boundary-corrected grid and applies each particle's plastic
        // return mapping (`MaterialModel::update_particle`) as the explicit
        // path does; this call is Klar 2016's operator-split plastic step.
        gather_grid_to_particles(
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

    /// Self-consistency of `particle_stiffness_block`: summed over every pair
    /// of a particle's stencil nodes, `dr_i = v0 * (B_i^T * M * B_j) * dv_j`
    /// reproduces the matrix-free product `v0 * apply_dtau_dl_psd_matrix(M,
    /// velocity_gradient(entries, dv)) * grad_i`, rebuilt inline here.
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

    /// `assemble_free_dof_system` + `cholesky_solve` solve the system they
    /// claim to: for a random SPD local matrix on a small 2-particle,
    /// shared-DOF problem, `A*x` (by the matrix-free formula) equals `b` to
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
        // this exercises a solve, not a degenerate one.
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

    /// `direct_solve` end to end with nonzero `lambda`/`mu` and non-identity
    /// `f_n`, so `build_particle_matrices` builds an eigenvalue-clamped
    /// operator from the corotated stress JVP: its `delta` solves the system
    /// `assemble_free_dof_system` builds from `build_particle_matrices(v)`.
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

    /// A non-trivial 2-particle problem (3 shared nodes, non-
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

    /// Minimal hand-verifiable case (single particle, single active stencil
    /// entry, `F_n = I`, one node): the FD gradient of `total_energy` is
    /// `13.9177`; `residual`'s `tau*grad` gives `487.03`, while
    /// `inertia (6.25) + dt*Piola*(F_n^T*grad) (7.68)` gives `13.93`, within 0.09%. Guards
    /// `model_residual`'s formula.
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

    /// `model_residual` is `total_energy`'s gradient, by central finite
    /// difference at `h = 1e-2`: an h sweep (1e-2, 1e-3, 1e-4 -> 0.30%, 2.17%,
    /// 8.22%) grows as `h` shrinks, the signature of f32 cancellation in a
    /// scalar summing `O(lambda) ~ 2.5-3e5` terms, not of a formula error.
    ///
    /// The trust-region ratio test relies on it: `predicted_reduction` (from
    /// `model_residual`/`model_jacobian_vector_product`) and
    /// `actual_reduction` (from `total_energy`) must describe one quantity.
    /// The true exponential-map `residual` is not this energy's gradient
    /// (max_rel_err 1.28); see `debug_minimal_single_entry_case` for the
    /// Piola/`F_n^T*grad`/`dt` form.
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

    /// `model_jacobian_vector_product` is `model_residual`'s derivative (the
    /// Hessian `steihaug_cg` needs), by central finite difference along a
    /// probe direction, h = 1e-3.
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

    /// `real_residual_jvp_fd`'s unit-normalize/rescale convention matches an
    /// independent, non-normalized central difference of the `residual`
    /// on `synthetic_problem`. They agree at the helper's `h = 1e-2`; at
    /// `h = 1e-3` they differ by 19%, a step-size effect of the exponential
    /// map's nonlinearity, not a formula error.
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
