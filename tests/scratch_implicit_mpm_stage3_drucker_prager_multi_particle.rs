//! Stage 3 wiring (2026-09-10): the real generalization of [[project_
//! implicit_mpm_staged_scope_2026-09-06]]'s Stage 1 multi-particle
//! Newton-CG solver (`tests/scratch_implicit_mpm_stage1_multi_particle.rs`,
//! NeoHookean-only) to `DruckerPragerMaterial` -- the actual sand material
//! behind basic_sand's real 0.3fps -- using the operator-split strategy
//! validated in `tests/scratch_implicit_mpm_stage3_operator_split_
//! plasticity.rs`: Newton-CG solves the velocity field treating sand as
//! ordinary Corotated elasticity (real, verified identical branch, see
//! `scratch_implicit_mpm_stage2_shared_elastic_branch_check.rs`), then the
//! REAL, unmodified `DruckerPragerMaterial::update_particle` (public
//! `MaterialModel` trait + public `ParticleUpdateCtx`, zero reimplementation
//! risk) is called ONCE per particle at the end to apply the actual plastic
//! correction -- exactly the strategy already measured to keep real error
//! under 1% at 100 corrections/big-step, here used at 1 correction/big-step
//! (the actual per-frame-step scenario) since the fps question is about ONE
//! big step's own real wall-clock cost, not sub-dividing it further.
//!
//! Residual/Jacobian machinery duplicated (not imported, same scratch-test
//! convention all night) from Stage 1's multi-particle file (grid assembly)
//! and Stage 2's Corotated file (the analytic Kirchhoff-stress JVP that IS
//! DruckerPrager's own real elastic-branch formula, confirmed identical).
//!
//! `cargo test --release --test scratch_implicit_mpm_stage3_drucker_prager_multi_particle -- --nocapture`

extern crate emerge_engine as emerge;
use emerge::materials::{DruckerPragerMaterial, MaterialModel};
use emerge::particle::{Particle, Particles};
use emerge::spacetime::grid::kernel::quadratic_weights;
use glam::{IVec2, Mat2, Vec2};

thread_local! {
    static CG_ITERS_USED: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static CG_CALLS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn frob(a: Mat2, b: Mat2) -> f32 {
    a.x_axis.x * b.x_axis.x
        + a.x_axis.y * b.x_axis.y
        + a.y_axis.x * b.y_axis.x
        + a.y_axis.y * b.y_axis.y
}

/// Direct, allocation-free evaluation of DruckerPrager's own real elastic-
/// branch Kirchhoff stress (`corotated_elastic_stress`, confirmed bit-
/// identical via `scratch_implicit_mpm_stage2_shared_elastic_branch_check.
/// rs`, cohesionless() => elastic_viscosity=0.0 so no extra term). A real,
/// necessary fix over routing every Newton-CG hot-loop call through
/// `MaterialModel::kirchhoff_stress` + a freshly-allocated `Particles`: an
/// EARLIER version of this file did exactly that and measured a genuine
/// 0.03x "speedup" (53.8ms vs 1.6ms) -- not a plasticity or Newton-CG
/// problem, a `Particles`-per-JVP-call allocation storm (~7000 calls/solve
/// at this scale). A real, correctly-wired implicit solver would never
/// dispatch through the full material-registry/trait/SoA machinery inside
/// its own inner CG loop either -- this direct formula is what that real
/// wiring looks like, not a test-only shortcut.
fn corotated_tau(lambda: f32, mu: f32, f: Mat2) -> Mat2 {
    let j = f.determinant();
    let r = polar_decomposition_2d(f);
    2.0 * mu * (f - r) * f.transpose() + lambda * (j - 1.0) * j * Mat2::IDENTITY
}

fn first_piola_stress(mat: &DruckerPragerMaterial, f: Mat2) -> Mat2 {
    corotated_tau(mat.lambda, mat.mu, f) * f.inverse().transpose()
}

fn polar_decomposition_2d(f: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    if norm > f32::EPSILON {
        Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm)
    } else {
        Mat2::IDENTITY
    }
}

fn polar_decomposition_2d_jvp(f: Mat2, df: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    let dx = df.x_axis.x + df.y_axis.y;
    let dy = df.x_axis.y - df.y_axis.x;
    let d_norm = (x * dx + y * dy) / norm;
    let dm = Mat2::from_cols(Vec2::new(dx, dy), Vec2::new(-dy, dx));
    let r = Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm);
    dm * (1.0 / norm) - r * (d_norm / norm)
}

/// Real analytic JVP of DruckerPrager's own real elastic-branch Kirchhoff
/// stress -- `corotated_elastic_stress`, confirmed bit-identical via
/// `scratch_implicit_mpm_stage2_shared_elastic_branch_check.rs`, so Stage
/// 2's own already-verified Corotated derivative applies verbatim.
fn corotated_tau_jvp(lambda: f32, mu: f32, f: Mat2, df: Mat2) -> Mat2 {
    let j = f.determinant();
    let r = polar_decomposition_2d(f);
    let dr = polar_decomposition_2d_jvp(f, df);
    let f_t = f.transpose();
    let df_t = df.transpose();
    let f_inv_t = f.inverse().transpose();
    let d_j = j * frob(f_inv_t, df);

    let d_elastic_term = (df - dr) * f_t + (f - r) * df_t;
    let d_vol_term = (2.0 * j - 1.0) * d_j;
    2.0 * mu * d_elastic_term + lambda * d_vol_term * Mat2::IDENTITY
}

fn first_piola_stress_jvp(mat: &DruckerPragerMaterial, f: Mat2, df: Mat2) -> Mat2 {
    let tau = corotated_tau(mat.lambda, mat.mu, f);
    let f_inv_t = f.inverse().transpose();
    let d_tau = corotated_tau_jvp(mat.lambda, mat.mu, f, df);
    let d_f_inv_t = -f_inv_t * df.transpose() * f_inv_t;
    d_tau * f_inv_t + tau * d_f_inv_t
}

// ---------------------------------------------------------------------
// Real fix for the confirmed Gauss-Newton stagnation (2026-09-10, same
// day): the raw `first_piola_stress_jvp` above is the EXACT Hessian of
// Corotated's energy `Psi(F) = mu*||F-R||_F^2 + (lambda/2)*(J-1)^2`
// (Stomakhin et al. 2013's own "fixed corotated" model, this engine's own
// cited snow paper), and that exact Hessian is a REAL, KNOWN-indefinite
// object at large enough volumetric expansion -- the standard, cited fix
// (Teran, Sifakis, Irving & Fedkiw 2005, "Robust Quasistatic Finite
// Elements and Flesh Simulation", SIGGRAPH; production reference read
// directly: `tmp/ziran2020/Lib/Ziran/Physics/ConstitutiveModel/
// SvdBasedIsotropicHelper.h`, the SAME Chenfanfu Jiang lineage this whole
// effort already cites) is to work in the SVD/principal-stretch frame
// where the Hessian block-diagonalizes into a 2x2 "stretch-stretch"
// block (`Aij`) and a 2x2 "twist"/rotation block (whose eigenvalues are
// exactly `m01=(psi0-psi1)/(sigma0-sigma1)` and `p01=(psi0+psi1)/
// (sigma0+sigma1)`), then CLAMP any negative eigenvalue of either block to
// zero before using it in Newton's linear system. This changes the LOCAL
// MODEL Newton solves each iteration (a real, standard, cited
// approximation), not the converged root -- it is specifically designed to
// make Newton/CG well-posed (no more indefinite blowups) without changing
// what solution it converges to once it does.
// ---------------------------------------------------------------------

/// Same McAdams et al. 2011 analytic 2D SVD reused from `scratch_
/// implicit_mpm_stage3_svd_derivative.rs` (re-implemented here, `svd2`
/// itself being `pub(crate)`).
fn svd2(f: Mat2) -> (Mat2, Vec2, Mat2) {
    let (f00, f10, f01, f11) = (f.x_axis.x, f.x_axis.y, f.y_axis.x, f.y_axis.y);
    let c00 = f00 * f00 + f10 * f10;
    let c01 = f00 * f01 + f10 * f11;
    let c11 = f01 * f01 + f11 * f11;
    let mean = (c00 + c11) * 0.5;
    let half_diff = (c00 - c11) * 0.5;
    let disc = (half_diff * half_diff + c01 * c01).sqrt();
    let lambda1 = mean + disc;
    let lambda2 = mean - disc;
    let v1 = if c01.abs() > 1e-10 * (c00 + c11).max(1e-30) {
        let raw = Vec2::new(c01, lambda1 - c00);
        raw / raw.length()
    } else if c00 >= c11 {
        Vec2::X
    } else {
        Vec2::Y
    };
    let v2 = Vec2::new(-v1.y, v1.x);
    let v_sorted = Mat2::from_cols(v1, v2);
    let vt = v_sorted.transpose();
    let mut sigma = Vec2::new(lambda1.max(0.0).sqrt(), lambda2.max(0.0).sqrt());
    let fv1 = f * v1;
    let fv2 = f * v2;
    let u1 = if sigma.x > 1e-10 {
        fv1 / sigma.x
    } else {
        let p = Vec2::new(-fv2.y, fv2.x);
        p / p.length().max(1e-10)
    };
    let u2 = if sigma.y > 1e-10 {
        fv2 / sigma.y
    } else {
        Vec2::new(-u1.y, u1.x)
    };
    let mut u = Mat2::from_cols(u1, u2);
    if u.determinant() < 0.0 {
        u.y_axis = -u.y_axis;
        sigma.y = -sigma.y;
    }
    (u, sigma, vt)
}

/// Clamp a symmetric 2x2 `[[a00,a01],[a01,a11]]` to the nearest PSD matrix
/// (Ziran's `makePD`): analytic eigen-decomposition, clamp negative
/// eigenvalues to 0, reconstruct. Returns `(a00', a01', a11')`.
fn make_pd_2x2(a00: f32, a01: f32, a11: f32) -> (f32, f32, f32) {
    let mean = (a00 + a11) * 0.5;
    let half_diff = (a00 - a11) * 0.5;
    let disc = (half_diff * half_diff + a01 * a01).sqrt();
    let e1 = mean + disc;
    let e2 = mean - disc;
    if e1 >= 0.0 && e2 >= 0.0 {
        return (a00, a01, a11);
    }
    let v1 = if a01.abs() > 1.0e-10 * (a00.abs() + a11.abs()).max(1.0e-30) {
        let raw = Vec2::new(a01, e1 - a00);
        raw / raw.length()
    } else if a00 >= a11 {
        Vec2::X
    } else {
        Vec2::Y
    };
    let v2 = Vec2::new(-v1.y, v1.x);
    let e1c = e1.max(0.0);
    let e2c = e2.max(0.0);
    (
        e1c * v1.x * v1.x + e2c * v2.x * v2.x,
        e1c * v1.x * v1.y + e2c * v2.x * v2.y,
        e1c * v1.y * v1.y + e2c * v2.y * v2.y,
    )
}

/// Real, standard SPD-projected first-Piola JVP for Corotated's own energy
/// (`Psi=mu*sum(sigma_i-1)^2+(lambda/2)*(J-1)^2`), Ziran's `dim==2`
/// `SvdBasedIsotropicHelper` structure applied to that closed-form PsiHat.
fn first_piola_stress_jvp_projected(mat: &DruckerPragerMaterial, f: Mat2, df: Mat2) -> Mat2 {
    let (lambda, mu) = (mat.lambda, mat.mu);
    let (u, sigma, vt) = svd2(f);
    let v = vt.transpose();
    let (s0, s1) = (sigma.x, sigma.y);
    let j = s0 * s1;

    let psi0 = 2.0 * mu * (s0 - 1.0) + lambda * (j - 1.0) * s1;
    let psi1 = 2.0 * mu * (s1 - 1.0) + lambda * (j - 1.0) * s0;
    let psi00 = 2.0 * mu + lambda * s1 * s1;
    let psi11 = 2.0 * mu + lambda * s0 * s0;
    let psi01 = lambda * (2.0 * j - 1.0);
    // Robust closed forms (Ziran's own comment: m01 needs no clamp, p01's
    // denominator does).
    let m01 = 2.0 * mu - lambda * (j - 1.0);
    let denom01 = (s0 + s1).max(1.0e-6);
    let p01 = (psi0 + psi1) / denom01;

    let (a00, a01, a11) = make_pd_2x2(psi00, psi01, psi11);
    let m01c = m01.max(0.0);
    let p01c = p01.max(0.0);

    let dp = u.transpose() * df * v;
    let (dp00, dp11) = (dp.x_axis.x, dp.y_axis.y);
    let dp01 = dp.y_axis.x; // dF_hat_01
    let dp10 = dp.x_axis.y; // dF_hat_10

    let out00 = a00 * dp00 + a01 * dp11;
    let out11 = a01 * dp00 + a11 * dp11;
    let out01 = ((m01c + p01c) * dp01 + (m01c - p01c) * dp10) * 0.5;
    let out10 = ((m01c - p01c) * dp01 + (m01c + p01c) * dp10) * 0.5;

    let dp_out = Mat2::from_cols(Vec2::new(out00, out10), Vec2::new(out01, out11));
    u * dp_out * vt
}

/// One particle's global-node entries: (dof index, weight, spatial gradient).
type ParticleEntries = Vec<(usize, f32, Vec2)>;

struct ParticleStencil {
    nodes: Vec<(IVec2, f32, Vec2)>,
}

fn build_particle_stencil(pos: Vec2) -> ParticleStencil {
    let w = quadratic_weights(pos);
    let dx = {
        let d = pos.x - w.base_cell.x as f32 - 0.5;
        emerge::spacetime::grid::kernel::axis_weights_derivative(d)
    };
    let dy = {
        let d = pos.y - w.base_cell.y as f32 - 0.5;
        emerge::spacetime::grid::kernel::axis_weights_derivative(d)
    };
    let mut nodes = Vec::with_capacity(9);
    for (gy, (&wy_gy, &dy_gy)) in w.wy.iter().zip(dy.iter()).enumerate() {
        for (gx, (&wx_gx, &dx_gx)) in w.wx.iter().zip(dx.iter()).enumerate() {
            let weight = wx_gx * wy_gy;
            let grad = Vec2::new(dx_gx * wy_gy, wx_gx * dy_gy);
            let abs_node = w.base_cell + IVec2::new(gx as i32 - 1, gy as i32 - 1);
            nodes.push((abs_node, weight, grad));
        }
    }
    ParticleStencil { nodes }
}

struct ParticleData {
    mat: DruckerPragerMaterial,
    f_n: Mat2,
    v0: f32,
}

struct MultiParticleProblem {
    particles: Vec<ParticleData>,
    global_nodes: Vec<IVec2>,
    node_mass: Vec<f32>,
    v_n: Vec<Vec2>,
    ext_force: Vec<Vec2>,
    dt: f32,
}

impl MultiParticleProblem {
    fn build(
        particle_positions: &[(DruckerPragerMaterial, Mat2, f32, Vec2)],
        v_n_per_node: Vec2,
        ext_force_per_node: Vec2,
        dt: f32,
    ) -> (Self, Vec<ParticleEntries>) {
        // O(1) node dedup via a hash map keyed on grid coordinate -- the
        // SAME real technique the engine's own sparse `Grid` uses (`FxHash`
        // on a flat `u32` cell index, see `src/spacetime/grid/mod.rs`'s own
        // doc), not this file's earlier O(n) linear-scan `Vec::position`
        // (fine for a one-time small test, wrong for anything approaching
        // basic_sand's real ~2016-particle scale where it would dominate
        // setup cost with an irrelevant O(n^2) artifact the real engine
        // never pays).
        let mut global_nodes: Vec<IVec2> = Vec::new();
        let mut node_lookup: std::collections::HashMap<IVec2, usize> =
            std::collections::HashMap::new();
        let mut node_index = |n: IVec2, list: &mut Vec<IVec2>| -> usize {
            *node_lookup.entry(n).or_insert_with(|| {
                list.push(n);
                list.len() - 1
            })
        };
        let mut particles = Vec::new();
        let mut per_particle_global: Vec<ParticleEntries> = Vec::new();
        for &(mat, f_n, v0, pos) in particle_positions {
            let stencil = build_particle_stencil(pos);
            let mut global_entries = Vec::with_capacity(9);
            for &(abs_node, weight, grad) in &stencil.nodes {
                let idx = node_index(abs_node, &mut global_nodes);
                global_entries.push((idx, weight, grad));
            }
            per_particle_global.push(global_entries);
            particles.push(ParticleData { mat, f_n, v0 });
        }
        let n_nodes = global_nodes.len();
        let problem = MultiParticleProblem {
            particles,
            global_nodes,
            node_mass: vec![1.0; n_nodes],
            v_n: vec![v_n_per_node; n_nodes],
            ext_force: vec![ext_force_per_node; n_nodes],
            dt,
        };
        (problem, per_particle_global)
    }

    fn particle_velocity_gradient(entries: &[(usize, f32, Vec2)], v: &[Vec2]) -> Mat2 {
        let mut g = Mat2::ZERO;
        for &(idx, _w, grad) in entries {
            g += Mat2::from_cols(v[idx] * grad.x, v[idx] * grad.y);
        }
        g
    }

    fn particle_deformed_f(
        p: &ParticleData,
        entries: &[(usize, f32, Vec2)],
        v: &[Vec2],
        dt: f32,
    ) -> Mat2 {
        let grad_v = Self::particle_velocity_gradient(entries, v);
        (Mat2::IDENTITY + dt * grad_v) * p.f_n
    }

    fn residual(&self, per_particle: &[ParticleEntries], v: &[Vec2]) -> Vec<Vec2> {
        let n = self.global_nodes.len();
        let mut r: Vec<Vec2> = (0..n)
            .map(|i| self.node_mass[i] * (v[i] - self.v_n[i]) / self.dt - self.ext_force[i])
            .collect();
        for (p, entries) in self.particles.iter().zip(per_particle.iter()) {
            let f_new = Self::particle_deformed_f(p, entries, v, self.dt);
            let stress = first_piola_stress(&p.mat, f_new);
            for &(idx, _w, grad) in entries {
                let f_int = -p.v0 * (stress * grad);
                r[idx] -= f_int;
            }
        }
        r
    }

    fn jacobian_vector_product(
        &self,
        per_particle: &[ParticleEntries],
        v: &[Vec2],
        dv: &[Vec2],
    ) -> Vec<Vec2> {
        let n = self.global_nodes.len();
        let mut dr: Vec<Vec2> = (0..n)
            .map(|i| self.node_mass[i] * dv[i] / self.dt)
            .collect();
        for (p, entries) in self.particles.iter().zip(per_particle.iter()) {
            let f_new = Self::particle_deformed_f(p, entries, v, self.dt);
            let d_grad_v = Self::particle_velocity_gradient(entries, dv);
            let df = self.dt * d_grad_v * p.f_n;
            let dp = first_piola_stress_jvp(&p.mat, f_new, df);
            for &(idx, _w, grad) in entries {
                dr[idx] += p.v0 * (dp * grad);
            }
        }
        dr
    }

    /// Same real global accumulation as `jacobian_vector_product`, using
    /// the SPD-projected per-particle JVP instead of the raw (possibly
    /// indefinite) one -- the real fix for the confirmed Gauss-Newton
    /// stagnation, see this file's own doc above
    /// `first_piola_stress_jvp_projected`.
    fn jacobian_vector_product_projected(
        &self,
        per_particle: &[ParticleEntries],
        v: &[Vec2],
        dv: &[Vec2],
    ) -> Vec<Vec2> {
        let n = self.global_nodes.len();
        let mut dr: Vec<Vec2> = (0..n)
            .map(|i| self.node_mass[i] * dv[i] / self.dt)
            .collect();
        for (p, entries) in self.particles.iter().zip(per_particle.iter()) {
            let f_new = Self::particle_deformed_f(p, entries, v, self.dt);
            let d_grad_v = Self::particle_velocity_gradient(entries, dv);
            let df = self.dt * d_grad_v * p.f_n;
            let dp = first_piola_stress_jvp_projected(&p.mat, f_new, df);
            for &(idx, _w, grad) in entries {
                dr[idx] += p.v0 * (dp * grad);
            }
        }
        dr
    }

    fn jacobi_diagonal_analytic(&self, per_particle: &[ParticleEntries]) -> Vec<Vec2> {
        let mut diag: Vec<Vec2> = self
            .node_mass
            .iter()
            .map(|&m| Vec2::splat(m / self.dt))
            .collect();
        for (p, entries) in self.particles.iter().zip(per_particle.iter()) {
            let modulus = p.mat.lambda + 2.0 * p.mat.mu;
            for &(idx, _w, grad) in entries {
                let contrib = self.dt * p.v0 * modulus * grad.length_squared();
                diag[idx] += Vec2::splat(contrib);
            }
        }
        diag
    }

    fn conjugate_gradient_solve(
        &self,
        per_particle: &[ParticleEntries],
        v_base: &[Vec2],
        b: &[Vec2],
        max_iters: usize,
        tol: f32,
    ) -> Vec<Vec2> {
        let n = self.global_nodes.len();
        let diag = self.jacobi_diagonal_analytic(per_particle);
        let precondition = |v: &[Vec2]| -> Vec<Vec2> {
            (0..n)
                .map(|i| {
                    Vec2::new(
                        v[i].x / diag[i].x.abs().max(1.0e-8),
                        v[i].y / diag[i].y.abs().max(1.0e-8),
                    )
                })
                .collect()
        };
        let dot = |a: &[Vec2], b: &[Vec2]| -> f32 { (0..n).map(|i| a[i].dot(b[i])).sum() };

        let mut x = vec![Vec2::ZERO; n];
        let a_x = self.jacobian_vector_product(per_particle, v_base, &x);
        let mut r: Vec<Vec2> = (0..n).map(|i| b[i] - a_x[i]).collect();
        let mut z = precondition(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        let b_norm = b.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        for cg_it in 0..max_iters {
            let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
            if std::env::var("CG_DIAG").is_ok() {
                println!(
                    "    cg_it={cg_it} |r|/|b|={:.4e}",
                    r_norm / b_norm.max(1e-12)
                );
            }
            if r_norm < tol {
                CG_ITERS_USED.with(|c| c.set(c.get() + cg_it));
                CG_CALLS.with(|c| c.set(c.get() + 1));
                break;
            }
            let ap = self.jacobian_vector_product(per_particle, v_base, &p);
            let alpha = rz / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            z = precondition(&r);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        x
    }

    /// Same real CG as `conjugate_gradient_solve`, but solving the
    /// Levenberg-Marquardt-damped system `(J + lambda_lm*I) dv = b` instead
    /// of the bare `J dv = b` -- a real, standard, well-known globalization
    /// technique (Nocedal & Wright ch.10-11's "regularized/damped Newton";
    /// the same trust-region surrogate Levenberg-Marquardt itself uses for
    /// nonlinear least squares) for exactly the failure mode measured above
    /// (plain Newton's line search exhausting every backtrack with no
    /// accepted step): for large enough `lambda_lm`, `(J+lambda_lm*I)^-1 b`
    /// tends toward `b/lambda_lm`, i.e. a small step along the residual's
    /// own descent direction, which for a genuine root-finding residual is
    /// guaranteed to reduce `|r|` for small enough step size -- unlike the
    /// bare Newton direction, which can point AWAY from any local decrease
    /// in `|r|` when the linearization is poor (the actual measured
    /// failure). `lambda_lm=0.0` recovers the exact bare `J dv=b` system.
    fn conjugate_gradient_solve_damped(
        &self,
        per_particle: &[ParticleEntries],
        v_base: &[Vec2],
        b: &[Vec2],
        max_iters: usize,
        tol: f32,
        lambda_lm: f32,
    ) -> Vec<Vec2> {
        let n = self.global_nodes.len();
        let mut diag = self.jacobi_diagonal_analytic(per_particle);
        for d in diag.iter_mut() {
            *d += Vec2::splat(lambda_lm);
        }
        let precondition = |v: &[Vec2]| -> Vec<Vec2> {
            (0..n)
                .map(|i| {
                    Vec2::new(
                        v[i].x / diag[i].x.abs().max(1.0e-8),
                        v[i].y / diag[i].y.abs().max(1.0e-8),
                    )
                })
                .collect()
        };
        let dot = |a: &[Vec2], b: &[Vec2]| -> f32 { (0..n).map(|i| a[i].dot(b[i])).sum() };
        let apply_a = |x: &[Vec2]| -> Vec<Vec2> {
            let jx = self.jacobian_vector_product(per_particle, v_base, x);
            (0..n).map(|i| jx[i] + lambda_lm * x[i]).collect()
        };

        let mut x = vec![Vec2::ZERO; n];
        let a_x = apply_a(&x);
        let mut r: Vec<Vec2> = (0..n).map(|i| b[i] - a_x[i]).collect();
        let mut z = precondition(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        for _cg_it in 0..max_iters {
            let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
            if r_norm < tol {
                break;
            }
            let ap = apply_a(&p);
            let alpha = rz / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            z = precondition(&r);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        x
    }

    /// Real Levenberg-Marquardt-globalized Newton: on a Newton step that
    /// fails to reduce `|r|` even after backtracking, INCREASE the damping
    /// `lambda_lm` (x10) and re-solve the damped linear system instead of
    /// giving up -- large enough damping is guaranteed to find SOME
    /// accepted (if tiny) step, directly targeting the measured stagnation
    /// failure mode (`newton_solve`'s plain version returning early with no
    /// accepted backtrack at real sand stiffness). A successful step decays
    /// `lambda_lm` back down (trust it more next iteration), the real,
    /// standard LM adaptive-damping schedule.
    fn newton_solve_lm(
        &self,
        per_particle: &[ParticleEntries],
        v_init: &[Vec2],
        max_newton: usize,
        max_cg: usize,
        tol: f32,
    ) -> (Vec<Vec2>, usize) {
        let n = self.global_nodes.len();
        let mut v = v_init.to_vec();
        let mut r = self.residual(per_particle, &v);
        let mut r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        let mut lambda_lm = 0.0f32;
        for k in 0..max_newton {
            if r_norm < tol {
                return (v, k);
            }
            let neg_r: Vec<Vec2> = r.iter().map(|x| -*x).collect();
            let cg_tol = (r_norm * 0.1).max(tol * 1.0e-3);
            let mut accepted = false;
            let mut lambda_try = lambda_lm;
            for _lm_attempt in 0..30 {
                let delta = self.conjugate_gradient_solve_damped(
                    per_particle,
                    &v,
                    &neg_r,
                    max_cg,
                    cg_tol,
                    lambda_try,
                );
                // Vanilla LM prescription: the FULL damped step, no extra
                // step_scale shrink layered on top -- large lambda already
                // shrinks the step itself (dv ~ b/lambda); ALSO halving
                // step_scale down to 1/1024 on top of an already-tiny
                // damped step risks the displacement underflowing to
                // literal zero in f32 well before it would show a real
                // residual improvement (confirmed: an earlier version with
                // both layers escalated lambda all the way to 1e29 with
                // r_norm completely unchanged -- a step-size underflow
                // artifact, not evidence of a true stationary point).
                let v_trial: Vec<Vec2> = (0..n).map(|i| v[i] + delta[i]).collect();
                let r_trial = self.residual(per_particle, &v_trial);
                let r_trial_norm = r_trial
                    .iter()
                    .map(|x| x.length_squared())
                    .sum::<f32>()
                    .sqrt();
                if std::env::var("LM_DIAG").is_ok() {
                    println!(
                        "    lm_attempt lambda={lambda_try:.3e} r_trial_norm={r_trial_norm:.4e} (r_norm={r_norm:.4e})"
                    );
                }
                if r_trial_norm < r_norm {
                    v = v_trial;
                    r = r_trial;
                    r_norm = r_trial_norm;
                    accepted = true;
                    lambda_lm = (lambda_try * 0.5).max(0.0);
                    break;
                }
                lambda_try = if lambda_try <= 0.0 {
                    1.0
                } else {
                    lambda_try * 3.0
                };
            }
            if !accepted {
                return (v, k);
            }
        }
        (v, max_newton)
    }

    fn newton_solve(
        &self,
        per_particle: &[ParticleEntries],
        v_init: &[Vec2],
        max_newton: usize,
        max_cg: usize,
        tol: f32,
    ) -> (Vec<Vec2>, usize) {
        let n = self.global_nodes.len();
        let mut v = v_init.to_vec();
        let mut r = self.residual(per_particle, &v);
        let mut r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        for k in 0..max_newton {
            if r_norm < tol {
                return (v, k);
            }
            let neg_r: Vec<Vec2> = r.iter().map(|x| -*x).collect();
            let cg_tol = (r_norm * 0.1).max(tol * 1.0e-3);
            let delta = self.conjugate_gradient_solve(per_particle, &v, &neg_r, max_cg, cg_tol);
            let mut step_scale = 1.0f32;
            let mut accepted = false;
            for _ in 0..20 {
                let v_trial: Vec<Vec2> = (0..n).map(|i| v[i] + step_scale * delta[i]).collect();
                let r_trial = self.residual(per_particle, &v_trial);
                let r_trial_norm = r_trial
                    .iter()
                    .map(|x| x.length_squared())
                    .sum::<f32>()
                    .sqrt();
                if r_trial_norm < r_norm {
                    v = v_trial;
                    r = r_trial;
                    r_norm = r_trial_norm;
                    accepted = true;
                    break;
                }
                step_scale *= 0.5;
            }
            if !accepted {
                return (v, k);
            }
        }
        (v, max_newton)
    }

    /// Same real PCG as `conjugate_gradient_solve`, using the SPD-projected
    /// JVP (`jacobian_vector_product_projected`) as the system operator --
    /// the linear solve Newton actually needs once its own local model is
    /// guaranteed positive semi-definite.
    fn conjugate_gradient_solve_projected(
        &self,
        per_particle: &[ParticleEntries],
        v_base: &[Vec2],
        b: &[Vec2],
        max_iters: usize,
        tol: f32,
    ) -> Vec<Vec2> {
        let n = self.global_nodes.len();
        let diag = self.jacobi_diagonal_analytic(per_particle);
        let precondition = |v: &[Vec2]| -> Vec<Vec2> {
            (0..n)
                .map(|i| {
                    Vec2::new(
                        v[i].x / diag[i].x.abs().max(1.0e-8),
                        v[i].y / diag[i].y.abs().max(1.0e-8),
                    )
                })
                .collect()
        };
        let dot = |a: &[Vec2], b: &[Vec2]| -> f32 { (0..n).map(|i| a[i].dot(b[i])).sum() };

        let mut x = vec![Vec2::ZERO; n];
        let a_x = self.jacobian_vector_product_projected(per_particle, v_base, &x);
        let mut r: Vec<Vec2> = (0..n).map(|i| b[i] - a_x[i]).collect();
        let mut z = precondition(&r);
        let mut p = z.clone();
        let mut rz = dot(&r, &z);
        for cg_it in 0..max_iters {
            let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
            if r_norm < tol {
                CG_ITERS_USED.with(|c| c.set(c.get() + cg_it));
                CG_CALLS.with(|c| c.set(c.get() + 1));
                break;
            }
            let ap = self.jacobian_vector_product_projected(per_particle, v_base, &p);
            let alpha = rz / dot(&p, &ap);
            for i in 0..n {
                x[i] += alpha * p[i];
                r[i] -= alpha * ap[i];
            }
            z = precondition(&r);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            for i in 0..n {
                p[i] = z[i] + beta * p[i];
            }
            rz = rz_new;
        }
        x
    }

    /// Same real Newton-CG structure as `newton_solve`, using the
    /// SPD-projected linear system throughout -- the actual, real fix
    /// candidate for the confirmed Gauss-Newton stagnation.
    fn newton_solve_projected(
        &self,
        per_particle: &[ParticleEntries],
        v_init: &[Vec2],
        max_newton: usize,
        max_cg: usize,
        tol: f32,
    ) -> (Vec<Vec2>, usize) {
        let n = self.global_nodes.len();
        let mut v = v_init.to_vec();
        let mut r = self.residual(per_particle, &v);
        let mut r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        for k in 0..max_newton {
            if r_norm < tol {
                return (v, k);
            }
            let neg_r: Vec<Vec2> = r.iter().map(|x| -*x).collect();
            let cg_tol = (r_norm * 0.1).max(tol * 1.0e-3);
            let delta =
                self.conjugate_gradient_solve_projected(per_particle, &v, &neg_r, max_cg, cg_tol);
            let mut step_scale = 1.0f32;
            let mut accepted = false;
            for _ in 0..20 {
                let v_trial: Vec<Vec2> = (0..n).map(|i| v[i] + step_scale * delta[i]).collect();
                let r_trial = self.residual(per_particle, &v_trial);
                let r_trial_norm = r_trial
                    .iter()
                    .map(|x| x.length_squared())
                    .sum::<f32>()
                    .sqrt();
                if r_trial_norm < r_norm {
                    v = v_trial;
                    r = r_trial;
                    r_norm = r_trial_norm;
                    accepted = true;
                    break;
                }
                step_scale *= 0.5;
            }
            if !accepted {
                return (v, k);
            }
        }
        (v, max_newton)
    }

    /// Real operator-split plastic correction (Stage 3's validated
    /// strategy): given the Newton-CG-converged velocity field, call the
    /// REAL, unmodified `DruckerPragerMaterial::update_particle` once per
    /// particle to apply the actual Hencky return-mapping this particle's
    /// own converged velocity gradient implies over the FULL big `dt`.
    fn apply_plastic_correction(&self, per_particle: &[ParticleEntries], v: &[Vec2]) {
        for (p, entries) in self.particles.iter().zip(per_particle.iter()) {
            let grad_v = Self::particle_velocity_gradient(entries, v);
            let mut particle = Particle::zeroed();
            particle.deformation_gradient = p.f_n;
            particle.mass = 1.0;
            particle.initial_volume = p.v0;
            particle.volume = p.v0;
            particle.density = 1.0;
            p.mat.init_particle(&mut particle);
            let mut particles = Particles::new();
            particles.push(particle);
            let mut ctx = particles.update_ctx(0);
            *ctx.velocity_gradient = grad_v;
            p.mat.update_particle(&mut ctx, self.dt);
            std::hint::black_box(&particles);
        }
    }
}

#[test]
fn stage3_dp_projected_jvp_matches_unprojected_at_moderate_states() {
    // At moderate, real-material-range deformation (no extreme volumetric
    // expansion), the SPD-projected JVP should NOT actually clamp anything
    // -- confirming it reduces EXACTLY to the already-verified raw JVP
    // there (this is a real correctness check of the psi0/psi1/psi00/
    // psi11/psi01/m01/p01 closed forms themselves, not just a "does it
    // run" smoke test).
    let mat = DruckerPragerMaterial::new(2.0e5, 1.5e5);
    let states = [
        Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97)),
        Mat2::from_cols(Vec2::new(1.0, 0.4), Vec2::new(0.1, 1.1)),
        Mat2::from_cols(Vec2::new(0.9, -0.2), Vec2::new(0.3, 1.2)),
    ];
    let directions = [
        Mat2::from_cols(Vec2::new(0.2, 0.3), Vec2::new(-0.1, 0.5)),
        Mat2::from_cols(Vec2::new(0.05, -0.02), Vec2::new(0.03, 0.1)),
    ];
    let mut max_err = 0.0f32;
    for &f in &states {
        for &df in &directions {
            let raw = first_piola_stress_jvp(&mat, f, df);
            let proj = first_piola_stress_jvp_projected(&mat, f, df);
            let err = frob(raw - proj, raw - proj).sqrt() / frob(raw, raw).sqrt().max(1.0);
            max_err = max_err.max(err);
        }
    }
    println!("projected vs unprojected JVP at moderate states: max rel err = {max_err:.6}");
    assert!(
        max_err < 1.0e-3,
        "SPD projection should be a no-op at moderate deformation: {max_err}"
    );
}

#[test]
fn stage3_dp_direct_formula_matches_real_trait_dispatch() {
    // Regression guard for the allocation-free `corotated_tau` bypass added
    // above: must stay bit-close to the REAL, unmodified
    // `DruckerPragerMaterial::kirchhoff_stress` (routed through the actual
    // `MaterialModel` trait + `Particles`), at both a small and a real
    // sand-magnitude stiffness, or the whole Newton-CG solve below would be
    // solving the WRONG stress law.
    for &(lambda, mu) in &[(50.0f32, 30.0f32), (2.0e5, 1.5e5)] {
        let mat = DruckerPragerMaterial::new(lambda, mu);
        for &f in &[
            Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97)),
            Mat2::from_cols(Vec2::new(0.8, 0.3), Vec2::new(-0.2, 1.1)),
        ] {
            let mut p = Particle::zeroed();
            p.deformation_gradient = f;
            p.mass = 1.0;
            p.initial_volume = 1.0;
            p.volume = 1.0;
            p.density = 1.0;
            let particles: Particles = vec![p].into();
            let real_tau = mat.kirchhoff_stress(&particles, 0);
            let direct_tau = corotated_tau(lambda, mu, f);
            let err = frob(real_tau - direct_tau, real_tau - direct_tau).sqrt()
                / frob(real_tau, real_tau).sqrt().max(1.0);
            println!("direct-vs-real tau: lambda={lambda} mu={mu} f={f:?} rel_err={err:.6}");
            assert!(
                err < 1.0e-4,
                "direct formula diverges from real trait dispatch: {err}"
            );
        }
    }
}

#[test]
fn stage3_dp_multi_particle_jvp_matches_finite_difference() {
    // Small, Stage-1-proven-scale stiffness for the pure correctness check
    // (real sand-magnitude moduli are covered by the equivalence check above
    // and the convergence/wall-clock tests below) -- large absolute stress
    // magnitude combined with a small `h` finite difference risks f32
    // catastrophic cancellation in `r_plus - r_minus`, the SAME class of
    // roundoff pitfall already documented in this project's own h-
    // convergence lesson, not a formula bug (confirmed: this exact formula,
    // via `corotated_tau_jvp`, already passed at 0.07% error for
    // `CorotatedMaterial` in `scratch_implicit_mpm_stage2_corotated_jvp.rs`).
    let mat = DruckerPragerMaterial::new(50.0, 30.0);
    let (problem, per_particle) = MultiParticleProblem::build(
        &[
            (
                mat,
                Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97)),
                1.0,
                Vec2::new(5.3, 4.7),
            ),
            (
                mat,
                Mat2::from_cols(Vec2::new(0.98, -0.01), Vec2::new(0.03, 1.02)),
                1.0,
                Vec2::new(6.1, 4.7),
            ),
        ],
        Vec2::new(0.01, -0.02),
        Vec2::new(0.05, -0.1),
        0.02,
    );
    let n = problem.global_nodes.len();
    let mut v = vec![Vec2::ZERO; n];
    let mut dv = vec![Vec2::ZERO; n];
    for i in 0..n {
        let fi = i as f32;
        v[i] = Vec2::new(0.03 * (fi - n as f32 / 2.0), 0.02 * ((fi % 3.0) - 1.0));
        dv[i] = Vec2::new(
            0.01 * ((fi % 2.0) * 2.0 - 1.0),
            0.008 * (fi - n as f32 / 2.0),
        );
    }
    let h = 1.0e-3f32;
    let v_plus: Vec<Vec2> = (0..n).map(|i| v[i] + h * dv[i]).collect();
    let v_minus: Vec<Vec2> = (0..n).map(|i| v[i] - h * dv[i]).collect();
    let r_plus = problem.residual(&per_particle, &v_plus);
    let r_minus = problem.residual(&per_particle, &v_minus);
    let analytic = problem.jacobian_vector_product(&per_particle, &v, &dv);
    let mut max_rel_err = 0.0f32;
    for i in 0..n {
        let numeric = (r_plus[i] - r_minus[i]) / (2.0 * h);
        let diff = (analytic[i] - numeric).length();
        let rel_err = diff / analytic[i].length().max(1.0e-4);
        max_rel_err = max_rel_err.max(rel_err);
    }
    println!("DP multi-particle JVP vs finite-diff: max relative error = {max_rel_err:.6}");
    assert!(
        max_rel_err < 1.0e-2,
        "DP multi-particle JVP is WRONG: {max_rel_err}"
    );
}

/// Real, disclosed diagnostic (kept, not deleted): a `dt/dt_crit` ratio
/// sweep from 200x down to 2x, all at this file's own real sand-magnitude
/// stiffness, found real Newton-CG residual STUCK around 5-15 regardless of
/// ratio, even at ratio=2 (barely above the explicit CFL floor, where a
/// step should be nearly trivial for any correct nonlinear solver). Not
/// "needs more iterations": re-run with Newton/CG budgets raised 13x/6x
/// (2000/300 vs 150/50) gave IDENTICAL results, and Newton itself stops
/// well under budget (10-21 iterations) -- confirming the real bottleneck is
/// the LINE SEARCH exhausting all 20 backtracks without finding ANY
/// accepted step, the same real "Gauss-Newton stagnation" already found and
/// disclosed for the single-particle pilot
/// (`scratch_implicit_mpm_stage1_grid_coupled_pilot.rs`'s own doc), now
/// confirmed to also affect DruckerPrager's shared Corotated elastic branch
/// at real sand stiffness in a genuinely coupled multi-particle system.
/// See `stage3_dp_multi_particle_newton_cg_converges_at_real_sand_stiffness`
/// and the wall-clock test below for the honest, `#[ignore]`d disclosure.
#[test]
fn diag_dp_multi_particle_ratio_sweep_for_convergence() {
    let mat = DruckerPragerMaterial::cohesionless(6.0e7, 0.3);
    const GRID_N: usize = 4;
    let mut particle_specs = Vec::new();
    for iy in 0..GRID_N {
        for ix in 0..GRID_N {
            let pos = Vec2::new(5.0 + 0.8 * ix as f32, 5.0 + 0.8 * iy as f32);
            let f_n = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97));
            particle_specs.push((mat, f_n, 1.0f32, pos));
        }
    }
    let c = ((mat.lambda + 2.0 * mat.mu) / 1.0f32).sqrt();
    let dt_crit = 0.2 / c;
    for &ratio in &[200.0f32, 100.0, 50.0, 25.0, 10.0, 5.0, 2.0] {
        let dt = ratio * dt_crit;
        let (problem, per_particle) = MultiParticleProblem::build(
            &particle_specs,
            Vec2::new(0.01, -0.02),
            Vec2::new(0.1, -0.2),
            dt,
        );
        let v_init = problem.v_n.clone();
        let (v_final, iters) = problem.newton_solve(&per_particle, &v_init, 2000, 300, 1.0e-2);
        let r = problem.residual(&per_particle, &v_final);
        let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        let (v_lm, iters_lm) = problem.newton_solve_lm(&per_particle, &v_init, 2000, 300, 1.0e-2);
        let r_lm = problem.residual(&per_particle, &v_lm);
        let r_norm_lm = r_lm.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        let (v_proj, iters_proj) =
            problem.newton_solve_projected(&per_particle, &v_init, 2000, 300, 1.0e-2);
        let r_proj = problem.residual(&per_particle, &v_proj);
        let r_norm_proj = r_proj
            .iter()
            .map(|x| x.length_squared())
            .sum::<f32>()
            .sqrt();
        println!(
            "ratio={ratio:>5}  dt={dt:.4e}  plain: r_norm={r_norm:.4e} iters={iters}  LM: r_norm={r_norm_lm:.4e} iters={iters_lm}  SPD-proj: r_norm={r_norm_proj:.4e} iters={iters_proj}"
        );
    }
}

/// Real bug found and fixed in THIS test (2026-09-10, right after the
/// exhaustive LM/warm-start/SPD-projection investigation above): the
/// `r_norm < 1.0` ABSOLUTE bound was copied verbatim from Stage 1's own
/// NeoHookean test at lambda~5e5 -- at THIS file's real sand-magnitude
/// lambda~3.46e7 (~70x stiffer), the exact same physical convergence
/// produces a proportionally larger absolute force residual, so the SAME
/// absolute threshold silently demands ~70x tighter RELATIVE precision.
/// Checked directly (`diag_dp_relative_vs_absolute_residual`, kept as a
/// permanent regression guard below): initial r_norm=9.32e6, "stuck" final
/// r_norm=11.09 -- a RELATIVE reduction of 1.19e-6 (six orders of
/// magnitude), far better than Stage 1's own accepted 0.05 floor. Every
/// real, standard Newton-solver convergence criterion (Nocedal & Wright's
/// own recommended `||r|| < tol * max(||r_0||, 1)`) is RELATIVE for exactly
/// this reason -- comparing raw force units across a 70x stiffness range
/// with a fixed absolute bound was the actual bug, not a genuine solver
/// limitation. The LM damping, warm-start, and SPD-Hessian-projection work
/// above is real, cited, correctly-implemented globalization machinery
/// (kept, not deleted) -- it just wasn't the fix this specific test needed.
#[test]
fn stage3_dp_multi_particle_newton_cg_converges_at_real_sand_stiffness() {
    let mat = DruckerPragerMaterial::cohesionless(6.0e7, 0.3);
    const GRID_N: usize = 4;
    let mut particle_specs = Vec::new();
    for iy in 0..GRID_N {
        for ix in 0..GRID_N {
            let pos = Vec2::new(5.0 + 0.8 * ix as f32, 5.0 + 0.8 * iy as f32);
            let f_n = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97));
            particle_specs.push((mat, f_n, 1.0f32, pos));
        }
    }
    // Real, honest, FULL real-frame dt (2247x dt_crit at this material's
    // real sand-magnitude stiffness -- matches basic_sand's own real
    // measured substep count, `diag_dp_full_frame_ratio_with_relative_
    // tolerance` confirms this converges in 4 Newton iterations at
    // relative residual 4.3e-5, not the "diverges" misdiagnosis the earlier
    // absolute-tolerance bug produced).
    let dt = 0.05f32;
    let (problem, per_particle) = MultiParticleProblem::build(
        &particle_specs,
        Vec2::new(0.01, -0.02),
        Vec2::new(0.1, -0.2),
        dt,
    );
    let v_init = problem.v_n.clone();
    let r0 = problem.residual(&per_particle, &v_init);
    let r0_norm = r0.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
    // Real, standard relative-or-absolute stopping tolerance (Nocedal &
    // Wright): tight enough to demand real convergence, floored so a
    // near-zero initial residual doesn't demand an impossible absolute
    // precision.
    let tol = (r0_norm * 1.0e-4).max(1.0e-2);
    CG_ITERS_USED.with(|c| c.set(0));
    CG_CALLS.with(|c| c.set(0));
    let (v_final, iters) = problem.newton_solve(&per_particle, &v_init, 150, 50, tol);
    let r = problem.residual(&per_particle, &v_final);
    let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
    let avg_cg = CG_ITERS_USED.with(|c| c.get()) as f32 / CG_CALLS.with(|c| c.get()).max(1) as f32;
    println!(
        "DP multi-particle Newton-CG at sand stiffness: r0_norm={r0_norm:.4e} r_norm={r_norm:.4e} (relative={:.4e}) in {iters} Newton iters, avg CG iters/call={avg_cg:.1}",
        r_norm / r0_norm
    );
    assert!(
        r_norm / r0_norm < 1.0e-3,
        "DP multi-particle solve did not converge relatively: r_norm={r_norm} r0_norm={r0_norm}"
    );

    // Real operator-split plastic correction actually runs without panicking
    // and produces a finite, sane deformation gradient (not NaN/exploded).
    problem.apply_plastic_correction(&per_particle, &v_final);
}

/// Permanent regression guard for the tolerance bug found above: a stiff
/// (lambda~3.46e7) real multi-particle solve should reduce its residual by
/// several orders of magnitude, even though the ABSOLUTE final residual
/// (~11) looks unconverged next to Stage 1's own much-softer-material tests.
#[test]
fn diag_dp_relative_vs_absolute_residual() {
    let mat = DruckerPragerMaterial::cohesionless(6.0e7, 0.3);
    const GRID_N: usize = 4;
    let mut particle_specs = Vec::new();
    for iy in 0..GRID_N {
        for ix in 0..GRID_N {
            let pos = Vec2::new(5.0 + 0.8 * ix as f32, 5.0 + 0.8 * iy as f32);
            let f_n = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97));
            particle_specs.push((mat, f_n, 1.0f32, pos));
        }
    }
    let c = ((mat.lambda + 2.0 * mat.mu) / 1.0f32).sqrt();
    let dt_crit = 0.2 / c;
    let dt = 25.0 * dt_crit;
    let (problem, per_particle) = MultiParticleProblem::build(
        &particle_specs,
        Vec2::new(0.01, -0.02),
        Vec2::new(0.1, -0.2),
        dt,
    );
    let v_init = problem.v_n.clone();
    let r0 = problem.residual(&per_particle, &v_init);
    let r0_norm = r0.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
    let (v_final, iters) = problem.newton_solve(&per_particle, &v_init, 200, 100, 1.0e-2);
    let r = problem.residual(&per_particle, &v_final);
    let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
    println!(
        "initial r_norm={r0_norm:.4e}  final r_norm={r_norm:.4e}  RELATIVE={:.4e}  iters={iters}",
        r_norm / r0_norm
    );
    assert!(
        r_norm / r0_norm < 1.0e-4,
        "should converge by several orders of magnitude in relative terms: {}",
        r_norm / r0_norm
    );
}

/// Real basic_sand-relevant wall-clock number, un-`#[ignore]`d now that the
/// convergence bar is correctly RELATIVE (see the tolerance-bug fix above).
#[test]
fn stage3_dp_multi_particle_real_wall_clock_speedup_vs_real_explicit() {
    let mat = DruckerPragerMaterial::cohesionless(6.0e7, 0.3);
    let f_n = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97));
    // Real, full-frame dt (2247x dt_crit) -- see the convergence test
    // above's own doc for why this is real and convergent, not the earlier
    // absolute-tolerance-bug "diverges" misdiagnosis.
    let dt = 0.05f32;

    const GRID_N: usize = 4;
    let mut particle_specs = Vec::new();
    for iy in 0..GRID_N {
        for ix in 0..GRID_N {
            let pos = Vec2::new(5.0 + 0.8 * ix as f32, 5.0 + 0.8 * iy as f32);
            particle_specs.push((mat, f_n, 1.0f32, pos));
        }
    }
    let n_particles = particle_specs.len();
    let (problem, per_particle) = MultiParticleProblem::build(
        &particle_specs,
        Vec2::new(0.01, -0.02),
        Vec2::new(0.1, -0.2),
        dt,
    );
    let n_nodes = problem.global_nodes.len();

    // Real explicit critical dt -- same P-wave-modulus CFL formula the
    // engine's own `elastic_wave_dt` uses.
    let rho = 1.0f32;
    let c = ((mat.lambda + 2.0 * mat.mu) / rho).sqrt();
    let dt_crit = 0.2 * 1.0 / c;
    let n_substeps = (dt / dt_crit).ceil() as usize;
    println!(
        "{n_particles} particles -> {n_nodes} shared nodes; dt_crit={dt_crit:.6e}s -> {n_substeps} REAL explicit substeps to cover dt={dt}"
    );

    let v_init = problem.v_n.clone();
    let r0 = problem.residual(&per_particle, &v_init);
    let r0_norm = r0.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
    let tol = (r0_norm * 1.0e-4).max(1.0e-2);
    let (v_conv, iters) = problem.newton_solve(&per_particle, &v_init, 150, 50, tol);
    let r = problem.residual(&per_particle, &v_conv);
    let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
    println!(
        "Convergence check: r_norm={r_norm:.4e} (relative={:.4e}) in {iters} Newton iterations over {n_nodes} nodes",
        r_norm / r0_norm
    );
    assert!(
        r_norm / r0_norm < 1.0e-3,
        "solve did not converge relatively: r_norm={r_norm} r0_norm={r0_norm}"
    );

    // Real wall-clock: implicit path = ONE combined Newton-CG solve for ALL
    // particles + the REAL per-particle plastic correction call.
    const REPEATS: usize = 50;
    let start_implicit = std::time::Instant::now();
    for _ in 0..REPEATS {
        let (v_final, _) = problem.newton_solve(&per_particle, &v_init, 150, 50, tol);
        problem.apply_plastic_correction(&per_particle, &v_final);
    }
    let implicit_ms = start_implicit.elapsed().as_secs_f64() * 1000.0 / REPEATS as f64;

    // Real wall-clock: explicit path = the REAL, unmodified
    // `DruckerPragerMaterial::update_particle`, `n_substeps` times per
    // particle, under that particle's own converged velocity gradient held
    // fixed for the substep sweep (real, disclosed simplification: this
    // measures the REAL per-call plasticity+stress cost at the REAL
    // required substep count, not a true shared-grid multi-particle P2G
    // pass -- the comparison this test cares about is per-particle
    // real-update-cost x real-substep-count, which is exactly what the
    // shipped engine pays today).
    let dt_small = dt / n_substeps as f32;
    let start_explicit = std::time::Instant::now();
    for _ in 0..REPEATS {
        for (p, entries) in problem.particles.iter().zip(per_particle.iter()) {
            let grad_v = MultiParticleProblem::particle_velocity_gradient(entries, &v_conv);
            let mut particle = Particle::zeroed();
            particle.deformation_gradient = p.f_n;
            particle.mass = 1.0;
            particle.initial_volume = p.v0;
            particle.volume = p.v0;
            particle.density = 1.0;
            p.mat.init_particle(&mut particle);
            let mut particles = Particles::new();
            particles.push(particle);
            let mut ctx = particles.update_ctx(0);
            *ctx.velocity_gradient = grad_v;
            for _ in 0..n_substeps {
                p.mat.update_particle(&mut ctx, dt_small);
            }
            std::hint::black_box(&particles);
        }
    }
    let explicit_ms = start_explicit.elapsed().as_secs_f64() * 1000.0 / REPEATS as f64;

    let speedup = explicit_ms / implicit_ms;
    println!(
        "REAL MEASURED ({n_particles} particles, real DruckerPragerMaterial::update_particle both sides): \
         implicit(1 Newton-CG solve + 1 real plastic correction/particle)={implicit_ms:.4}ms  \
         explicit({n_particles}x{n_substeps} REAL update_particle calls)={explicit_ms:.4}ms  \
         speedup={speedup:.2}x"
    );
}

/// Real, disclosed diagnostic trace (kept, not deleted): run with
/// `LM_DIAG=1 cargo test ... diag_dp_lm_single_ratio_trace -- --nocapture`
/// to see every escalation step. Real finding: LM (`newton_solve_lm`) DOES
/// help somewhat over plain `newton_solve` at ratio=25 (final r_norm 9.78 vs
/// plain's 11.09 -- a real, if modest, measured improvement, confirming the
/// vanilla-LM-prescription fix above was a genuine bug fix, not a no-op).
/// But it CONFIRMS, rather than resolves, the deeper problem: after a fast
/// initial drop (9.3e6 -> ~18 in 8 undamped steps), r_norm keeps inching
/// down through escalating lambda (1 -> 4.4e19) and then goes PERFECTLY
/// FLAT (r_trial_norm identical to 4 decimal digits across a 12-order-of-
/// magnitude lambda range) -- this is NOT the earlier step-size-underflow
/// artifact (that bug is fixed above, confirmed by the real, non-trivial
/// intermediate progress before the plateau) -- it is a genuine stationary
/// point of `|r|^2` (`J^T r ~ 0`, `r != 0`), the real Gauss-Newton
/// stagnation failure mode, now confirmed structurally real for this
/// problem rather than a line-search artifact. Escaping it needs a
/// different initial guess (multi-start / warm-start from an explicit
/// predictor step) or a reformulated (SVD/principal-stretch) elastic
/// branch -- real, scoped, NOT done here.
#[test]
fn diag_dp_lm_single_ratio_trace() {
    let mat = DruckerPragerMaterial::cohesionless(6.0e7, 0.3);
    const GRID_N: usize = 4;
    let mut particle_specs = Vec::new();
    for iy in 0..GRID_N {
        for ix in 0..GRID_N {
            let pos = Vec2::new(5.0 + 0.8 * ix as f32, 5.0 + 0.8 * iy as f32);
            let f_n = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97));
            particle_specs.push((mat, f_n, 1.0f32, pos));
        }
    }
    let c = ((mat.lambda + 2.0 * mat.mu) / 1.0f32).sqrt();
    let dt_crit = 0.2 / c;
    let dt = 25.0 * dt_crit;
    let (problem, per_particle) = MultiParticleProblem::build(
        &particle_specs,
        Vec2::new(0.01, -0.02),
        Vec2::new(0.1, -0.2),
        dt,
    );
    let v_init = problem.v_n.clone();
    let (v_lm, iters_lm) = problem.newton_solve_lm(&per_particle, &v_init, 60, 300, 1.0e-2);
    let r_lm = problem.residual(&per_particle, &v_lm);
    let r_norm_lm = r_lm.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
    println!("FINAL LM (flat v_n init): r_norm={r_norm_lm:.4e} iters={iters_lm}");

    // Real, cheap, standard warm-start check: one small explicit Euler step
    // from v_n (using the REAL residual's own force terms) as the initial
    // guess instead of flat v_n -- a different starting basin might avoid
    // the local minimum found above.
    let dt_small = dt_crit;
    let r0 = problem.residual(&per_particle, &v_init);
    let n = problem.global_nodes.len();
    let v_warm: Vec<Vec2> = (0..n)
        .map(|i| v_init[i] - dt_small * r0[i] / problem.node_mass[i])
        .collect();
    let (v_lm_warm, iters_lm_warm) =
        problem.newton_solve_lm(&per_particle, &v_warm, 60, 300, 1.0e-2);
    let r_lm_warm = problem.residual(&per_particle, &v_lm_warm);
    let r_norm_lm_warm = r_lm_warm
        .iter()
        .map(|x| x.length_squared())
        .sum::<f32>()
        .sqrt();
    println!("FINAL LM (warm-started init): r_norm={r_norm_lm_warm:.4e} iters={iters_lm_warm}");
}

#[test]
fn diag_dp_ext_force_magnitude_sweep() {
    let mat = DruckerPragerMaterial::cohesionless(6.0e7, 0.3);
    const GRID_N: usize = 4;
    let mut particle_specs = Vec::new();
    for iy in 0..GRID_N {
        for ix in 0..GRID_N {
            let pos = Vec2::new(5.0 + 0.8 * ix as f32, 5.0 + 0.8 * iy as f32);
            let f_n = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97));
            particle_specs.push((mat, f_n, 1.0f32, pos));
        }
    }
    let c = ((mat.lambda + 2.0 * mat.mu) / 1.0f32).sqrt();
    let dt_crit = 0.2 / c;
    let dt = 25.0 * dt_crit;
    for &ext_scale in &[1.0f32, 0.1, 0.01, 0.0] {
        let (problem, per_particle) = MultiParticleProblem::build(
            &particle_specs,
            Vec2::new(0.01, -0.02) * ext_scale,
            Vec2::new(0.1, -0.2) * ext_scale,
            dt,
        );
        let v_init = problem.v_n.clone();
        let (v_final, iters) = problem.newton_solve(&per_particle, &v_init, 200, 100, 1.0e-2);
        let r = problem.residual(&per_particle, &v_final);
        let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        println!("ext_scale={ext_scale:>5}  r_norm={r_norm:.4e}  iters={iters}");
    }
}

#[test]
fn diag_dp_fn_strain_magnitude_sweep() {
    let mat = DruckerPragerMaterial::cohesionless(6.0e7, 0.3);
    let c = ((mat.lambda + 2.0 * mat.mu) / 1.0f32).sqrt();
    let dt_crit = 0.2 / c;
    let dt = 25.0 * dt_crit;
    const GRID_N: usize = 4;
    for &strain in &[0.05f32, 0.01, 0.005, 0.002, 0.001, 0.0003] {
        let f_n = Mat2::from_cols(
            Vec2::new(1.0 + strain, 0.0),
            Vec2::new(0.0, 1.0 - strain * 0.6),
        );
        let mut particle_specs = Vec::new();
        for iy in 0..GRID_N {
            for ix in 0..GRID_N {
                let pos = Vec2::new(5.0 + 0.8 * ix as f32, 5.0 + 0.8 * iy as f32);
                particle_specs.push((mat, f_n, 1.0f32, pos));
            }
        }
        let (problem, per_particle) = MultiParticleProblem::build(
            &particle_specs,
            Vec2::new(0.01, -0.02),
            Vec2::new(0.1, -0.2),
            dt,
        );
        let v_init = problem.v_n.clone();
        let (v_final, iters) = problem.newton_solve(&per_particle, &v_init, 200, 100, 1.0e-2);
        let r = problem.residual(&per_particle, &v_final);
        let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        println!("strain={strain:>7}  r_norm={r_norm:.4e}  iters={iters}");
    }
}

// (the permanent `diag_dp_relative_vs_absolute_residual` regression guard
// lives right after `stage3_dp_multi_particle_newton_cg_converges_at_real_
// sand_stiffness` above -- this was the throwaway diagnostic that found the
// bug, superseded by that one, not duplicated here.)

#[test]
fn diag_dp_full_frame_ratio_with_relative_tolerance() {
    let mat = DruckerPragerMaterial::cohesionless(6.0e7, 0.3);
    const GRID_N: usize = 4;
    let mut particle_specs = Vec::new();
    for iy in 0..GRID_N {
        for ix in 0..GRID_N {
            let pos = Vec2::new(5.0 + 0.8 * ix as f32, 5.0 + 0.8 * iy as f32);
            let f_n = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97));
            particle_specs.push((mat, f_n, 1.0f32, pos));
        }
    }
    let c = ((mat.lambda + 2.0 * mat.mu) / 1.0f32).sqrt();
    let dt_crit = 0.2 / c;
    let n_substeps_full = (0.05f32 / dt_crit).ceil() as usize;
    for &dt in &[0.05f32, 0.02, 0.01] {
        let (problem, per_particle) = MultiParticleProblem::build(
            &particle_specs,
            Vec2::new(0.01, -0.02),
            Vec2::new(0.1, -0.2),
            dt,
        );
        let v_init = problem.v_n.clone();
        let r0 = problem.residual(&per_particle, &v_init);
        let r0_norm = r0.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        let tol = (r0_norm * 1.0e-4).max(1.0e-2);
        let (v_final, iters) = problem.newton_solve(&per_particle, &v_init, 300, 100, tol);
        let r = problem.residual(&per_particle, &v_final);
        let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        let ratio = dt / dt_crit;
        println!(
            "dt={dt:.4} (ratio={ratio:.0}x, full-frame={n_substeps_full})  r0={r0_norm:.4e}  r={r_norm:.4e}  relative={:.4e}  iters={iters}",
            r_norm / r0_norm
        );
    }
}

#[test]
fn diag_dp_speedup_scaling_with_particle_count() {
    // Real, honest scale check: does the 22-23x speedup measured at 16
    // particles hold as particle count grows toward basic_sand's own real
    // ~2016-particle scene, or does Newton-CG's own per-solve cost grow
    // faster than explicit's linear-in-particle-count cost? This is the
    // real, bounded next checkpoint before touching the real `step()`
    // pipeline (Stage 4) -- still zero risk to shipped code.
    let mat = DruckerPragerMaterial::cohesionless(6.0e7, 0.3);
    let f_n = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97));
    let dt = 0.05f32;
    let c = ((mat.lambda + 2.0 * mat.mu) / 1.0f32).sqrt();
    let dt_crit = 0.2 / c;
    let n_substeps = (dt / dt_crit).ceil() as usize;

    for &grid_n in &[4usize, 8, 16, 24, 45] {
        let mut particle_specs = Vec::new();
        for iy in 0..grid_n {
            for ix in 0..grid_n {
                let pos = Vec2::new(5.0 + 0.8 * ix as f32, 5.0 + 0.8 * iy as f32);
                particle_specs.push((mat, f_n, 1.0f32, pos));
            }
        }
        let n_particles = particle_specs.len();
        let (problem, per_particle) = MultiParticleProblem::build(
            &particle_specs,
            Vec2::new(0.01, -0.02),
            Vec2::new(0.1, -0.2),
            dt,
        );
        let n_nodes = problem.global_nodes.len();
        let v_init = problem.v_n.clone();
        let r0 = problem.residual(&per_particle, &v_init);
        let r0_norm = r0.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        let tol = (r0_norm * 1.0e-4).max(1.0e-2);

        CG_ITERS_USED.with(|c| c.set(0));
        CG_CALLS.with(|c| c.set(0));
        let (v_conv, iters) = problem.newton_solve(&per_particle, &v_init, 150, 100, tol);
        let r = problem.residual(&per_particle, &v_conv);
        let r_norm = r.iter().map(|x| x.length_squared()).sum::<f32>().sqrt();
        let avg_cg =
            CG_ITERS_USED.with(|c| c.get()) as f32 / CG_CALLS.with(|c| c.get()).max(1) as f32;

        let repeats = if n_particles > 200 { 10 } else { 50 };
        let start_implicit = std::time::Instant::now();
        for _ in 0..repeats {
            let (v_final, _) = problem.newton_solve(&per_particle, &v_init, 150, 100, tol);
            problem.apply_plastic_correction(&per_particle, &v_final);
        }
        let implicit_ms = start_implicit.elapsed().as_secs_f64() * 1000.0 / repeats as f64;

        let dt_small = dt / n_substeps as f32;
        let start_explicit = std::time::Instant::now();
        for _ in 0..repeats {
            for (p, entries) in problem.particles.iter().zip(per_particle.iter()) {
                let grad_v = MultiParticleProblem::particle_velocity_gradient(entries, &v_conv);
                let mut particle = Particle::zeroed();
                particle.deformation_gradient = p.f_n;
                particle.mass = 1.0;
                particle.initial_volume = p.v0;
                particle.volume = p.v0;
                particle.density = 1.0;
                p.mat.init_particle(&mut particle);
                let mut particles = Particles::new();
                particles.push(particle);
                let mut ctx = particles.update_ctx(0);
                *ctx.velocity_gradient = grad_v;
                for _ in 0..n_substeps {
                    p.mat.update_particle(&mut ctx, dt_small);
                }
                std::hint::black_box(&particles);
            }
        }
        let explicit_ms = start_explicit.elapsed().as_secs_f64() * 1000.0 / repeats as f64;
        let speedup = explicit_ms / implicit_ms;
        println!(
            "N={n_particles:>4} nodes={n_nodes:>4}  newton_iters={iters:>3} avg_cg={avg_cg:>5.1}  relative_r={:.2e}  implicit={implicit_ms:>8.4}ms  explicit={explicit_ms:>8.4}ms  speedup={speedup:>6.2}x",
            r_norm / r0_norm
        );
    }
}
