use crate::materials::svd::svd2;
use glam::{Mat2, Vec2};

/// `x.powf(exp)` using exact integer exponentiation-by-squaring (`powi`)
/// when `exp` is a representable integer -- identical result, no
/// transcendental call. Every Tait-EOS exponent this engine ships (7.0,
/// Cole 1948) is a clean integer; this is a pure implementation-cost win in
/// a hot per-particle, per-substep path (kirchhoff_stress + timestep_bound
/// for every fluid material), not a physics change. Falls back to `powf`
/// exactly for any exponent that isn't a clean integer, so behavior for a
/// non-standard EOS power is unchanged.
#[inline]
pub(crate) fn fast_pow(x: f32, exp: f32) -> f32 {
    if exp.fract() == 0.0 && exp.abs() < 32.0 {
        x.powi(exp as i32)
    } else {
        x.powf(exp)
    }
}

/// Floor applied to singular values before taking log -- prevents ln(0).
/// All material `update_particle` implementations clamp σᵢ above this value.
pub(crate) const LOG_CLAMP: f32 = 1e-10;

/// Floor applied by legacy solid/plastic constitutive laws before using
/// `det(F)=J` in a singular expression. Strict WC-MPM liquid state does not
/// use this floor: it keeps positive `J` through its exponential continuity
/// update and reports an inadmissible state instead of clamping it.
pub(crate) const MIN_J: f32 = 1e-6;

/// Below this `|delta^2|` (a stretch or rotation under one radian per
/// substep, every substep the CFL bounds allow) the even and odd factors of
/// the exponential come from their Taylor series to `x^5`. The first
/// omitted terms, `x^6 / 12!` and `x^6 / 13!`, are then under 5e-9 of the
/// leading ones, below f32 resolution.
///
/// The series is used this far, rather than only near zero, because the
/// GPU copy cannot rely on its built-ins for small arguments: WGSL (W3C,
/// section 15.7.4 Floating Point Accuracy) gives `sinh` only the accuracy
/// of `(exp(x) - exp(-x)) * 0.5` and allows `sin` an absolute error of
/// 2^-11 on [-pi, pi], either of which can be the whole of a substep's
/// value. Both solvers use the same series so they agree.
const INCREMENT_SERIES_LIMIT: f32 = 1.0;

/// `exp(A) - I` for a 2x2 `A`, never formed as one plus a small number:
/// the exact deformation-gradient increment for a velocity gradient held
/// constant over one substep, minus the identity.
///
/// Continuum kinematics gives `dF/dt = L F`.  The exact update for constant
/// `L` is therefore `F(t+dt) = exp(dt L) F(t)`.  The commonly used forward-
/// Euler approximation `(I + dt L) F` has a systematic ratchet: two equal
/// and opposite diagonal rates multiply to `(1+a)(1-a)=1-a^2`, so a
/// perfectly reversible oscillation loses volume every cycle.  That error is
/// especially visible in a tension-only material because its slack direction
/// has no constitutive restoring force.
///
/// This uses the closed-form exponential of a real 2x2 matrix obtained from
/// Cayley-Hamilton.  The trigonometric branch covers complex-conjugate
/// eigenvalues (including rigid rotation); the hyperbolic branch covers real
/// eigenvalues.  Series expansions remove the removable singularity at zero.
///
/// A substep's `exp(dt L)` is within a few parts in 1e4 of the identity.
/// An f32 near 1 is spaced 1.19e-7 above it and 5.96e-8 below it, so
/// writing the increment as `1 + small` rounds the small part onto steps of
/// two different sizes, and a factor that should be just above or just
/// below 1 is quantized differently in each direction. Multiplied into an
/// `F` whose entries straddle 1 (a loaded body: 1.0027 across, 0.9899 along
/// the load), the rounding stops averaging out and pushes one way, every
/// substep. Measured on a resting self-weight column (E = 1e5 Pa, earth
/// gravity, 240 particles), with each particle's `F` re-integrated in f64
/// from the solver's own velocity gradients: the former update (this
/// increment formed as `1 + small`, then rescaled onto a carried volume)
/// drifted the mean `F_xx` by -3.6e-4 and `F_yy` by +3.0e-4 in 10 s, so `F`
/// read the body as less deformed than it was, while this form, applied as
/// `F + (exp(dt L) - I) F`, stayed within 5e-7 of the f64 integral.
/// Equilibrium holds `F` at the load, so in the solver that drift showed up
/// as the body's shape creeping instead (shorter and wider, without end, at
/// a rate proportional to the number of substeps).
///
/// Every term here stays small: `exp_m1` for the trace part and the
/// half-angle form `cosh x - 1 = 2 sinh^2(x / 2)` for the even part.
#[inline]
fn deformation_increment_exp_minus_identity(dt_velocity_gradient: Mat2) -> Mat2 {
    // glam is column-major: [[a,b],[c,d]] is stored as columns (a,c),(b,d).
    let a = dt_velocity_gradient.x_axis.x;
    let b = dt_velocity_gradient.y_axis.x;
    let c = dt_velocity_gradient.x_axis.y;
    let d = dt_velocity_gradient.y_axis.y;
    let half_trace = 0.5 * (a + d);
    let half_difference = 0.5 * (a - d);
    let delta_sq = half_difference * half_difference + b * c;

    // cosh(sqrt(x)) - 1 and sinh(sqrt(x))/sqrt(x), continued analytically
    // through x = 0 (cos and sin of sqrt(-x) for x < 0).
    let (even_minus_one, odd) = if delta_sq.abs() < INCREMENT_SERIES_LIMIT {
        let x = delta_sq;
        (
            x * (1.0 / 2.0
                + x * (1.0 / 24.0 + x * (1.0 / 720.0 + x * (1.0 / 40320.0 + x / 3628800.0)))),
            1.0 + x
                * (1.0 / 6.0
                    + x * (1.0 / 120.0
                        + x * (1.0 / 5040.0 + x * (1.0 / 362880.0 + x / 39916800.0)))),
        )
    } else if delta_sq > 0.0 {
        let delta = delta_sq.sqrt();
        let half = (0.5 * delta).sinh();
        (2.0 * half * half, delta.sinh() / delta)
    } else {
        let omega = (-delta_sq).sqrt();
        let half = (0.5 * omega).sin();
        (-2.0 * half * half, omega.sin() / omega)
    };

    // exp(A) = e^h (cosh I + odd (A - h I)) with h = tr(A) / 2; writing
    // s = e^h - 1 and e = cosh - 1 gives
    // exp(A) - I = (e + s (1 + e)) I + (1 + s) odd (A - h I).
    let scale_minus_one = half_trace.exp_m1();
    let traceless = dt_velocity_gradient - Mat2::from_diagonal(Vec2::splat(half_trace));
    Mat2::from_diagonal(Vec2::splat(
        even_minus_one + scale_minus_one * (1.0 + even_minus_one),
    )) + traceless * ((1.0 + scale_minus_one) * odd)
}

/// One substep of the continuity equation for a material that owns its
/// own volume, carried in the log so a small increment is not absorbed.
///
/// The scalar twin of `advance_deformation_gradient`, for the same
/// reason. Reading `J` back from an isotropic `F` each substep and
/// multiplying, `J_new = det(F) * exp(dt div v)`, loses precision: near one, an
/// f32 has a resolution of about 1.2e-7, while a calm flow's own
/// increment is a thousandth of that, so each step loses a fixed
/// FRACTION of its increment to absorption and the smallest increments
/// vanish outright. Measured with no solver and no grid
/// (`tests/probes/fluid_j_rounding.rs`), against an f64 replica of the
/// same update that holds J at exactly 1.000000: f32 walked to 0.999468
/// at a divergence of 0.2 per second, and at the finest increment it
/// froze completely, 0.000 drift, J stuck.
///
/// Adding `dt div v` to `ln J` instead keeps the increment: near J = 1
/// the logarithm is near zero, where f32 resolution is not 1.2e-7 but
/// vanishingly small. The clamp is applied in the same place, in the log,
/// so a bounded material stays bounded.
///
/// Returns the carried logarithm and the `J` it means.
#[inline]
pub(crate) fn advance_log_volume_ratio(
    carried_log_j: f32,
    dt_div_v: f32,
    min_j: f32,
    max_j: f32,
) -> (f32, f32) {
    let advanced = carried_log_j + dt_div_v;
    let clamped = advanced.clamp(min_j.max(f32::MIN_POSITIVE).ln(), max_j.ln());
    (clamped, clamped.exp())
}

/// One substep of `dF/dt = L F`, as `F + (exp(dt L) - I) F`.
///
/// Applying the increment as a correction to `F`, rather than multiplying
/// by a matrix that sits on the f32 grid around 1, keeps each rounding
/// relative to the small part (see `deformation_increment_exp_minus_identity`),
/// so what rounding is left averages out instead of pushing one way.
///
/// The form this replaces multiplied by `exp(dt L)` and then rescaled the
/// product every substep onto a volume ratio carried beside `F`, to undo
/// the steady loss of determinant that product had. The rescale factor
/// was itself within a few ULP of 1, so it fed the same one-way rounding
/// back into the shape, and once the carried ratio and `det(F)` drifted
/// more than 1e-3 apart it stopped rescaling for good. Measured on one
/// particle driven by a prescribed oscillation, no solver
/// (`tests/probes/f_rounding_horizon.rs`, ROUND_FXX=1.0027
/// ROUND_FYY=0.9899, ROUND_AMP=0.4, 900 000 steps of 2.94e-4 s), error
/// against the f64 integral: the rescaled form held `ln det F` to -1.2e-7
/// but put -5.6e-3 into `ln(F_xx / F_yy)`; this form leaves +7.9e-5 and
/// +6.3e-5, about what an unbiased walk of f32 roundings reaches in that
/// many steps. On the anchored tension-only body of
/// `tests/probes/no_compression_drift_horizon.rs` (one 4.37 ms substep per
/// step), where shape error is what turned into volume, `max |J - 1|` after
/// 900 000 substeps goes from 0.131 to 0.0011.
///
/// The volume ratio is `det(F)`, as in any MPM: with no bias left to
/// cancel, nothing needs a second copy kept in step with it.
#[inline]
pub(crate) fn advance_deformation_gradient(f_old: Mat2, dt_velocity_gradient: Mat2) -> Mat2 {
    f_old + deformation_increment_exp_minus_identity(dt_velocity_gradient) * f_old
}

/// Floor on Rankine's exponentially-softened effective tensile strength, as a
/// fraction of the virgin `tensile_strength`. Without this floor, `t_eff` decays
/// toward zero as damage grows, so ANY sustained cyclic stress eventually exceeds
/// it every step by a growing margin -- an unbounded damage ratchet with no
/// resting state. Real quasi-brittle/ductile damage models retain nonzero residual capacity
/// after yield rather than decaying to zero (Lemaitre & Chaboche, "Mechanics of
/// Solid Materials," 1990 -- continuum damage mechanics caps effective
/// stiffness/strength at a small nonzero residual specifically to keep the
/// damage variable bounded under sustained loading). 5% is a conservative
/// residual: still lets damage climb substantially before saturating, but
/// guarantees a stress level cyclic loading can eventually stay under.
pub(crate) const RANKINE_MIN_RESIDUAL_TENSILE_FRACTION: f32 = 0.05;

/// Damage value at which `t_eff` has already reached its residual floor -- i.e.
/// `tensile_strength * exp(-softening_rate * d) == tensile_strength * RESIDUAL_FRACTION`,
/// solved for d. Past this point, MORE damage would not lower `t_eff` any further
/// (it's already floored), so there is nothing left for the number to usefully
/// track: sustained cyclic loading that exceeds even the residual strength would
/// otherwise still accumulate damage forever, linearly, once the floor was hit --
/// the floor alone stops the exponential runaway but not an unbounded linear
/// climb. Clamping `prior_damage` at this saturation point is a no-op on future
/// stress behavior (t_eff there is identical to t_eff at any higher damage value)
/// and gives the accumulator a rupture point instead of an arbitrary cutoff.
/// `softening_rate <= 0` means no softening at all (hard cutoff, doc'd on
/// `RankineMaterial::softening_rate`) -- saturation point is +infinity, i.e. no cap.
#[inline]
pub(crate) fn rankine_damage_saturation_point(softening_rate: f32) -> f32 {
    if softening_rate <= 0.0 {
        f32::INFINITY
    } else {
        -RANKINE_MIN_RESIDUAL_TENSILE_FRACTION.ln() / softening_rate
    }
}

/// Compute Hencky (logarithmic) strains from SVD singular values.
///
/// ε_i = ln(|σ_i|), clamped above 1e-10 to avoid ln(0).
/// The absolute value preserves Hencky strain magnitudes when `svd2()` encodes
/// an inversion via a signed second singular value.
/// Used identically by VonMisesMaterial, DruckerPragerMaterial, RankineMaterial.
#[inline]
pub(crate) fn hencky_strains(sigma: Vec2) -> Vec2 {
    let sigma = sigma.abs().max(Vec2::splat(LOG_CLAMP));
    Vec2::new(sigma.x.ln(), sigma.y.ln())
}

/// Reconstruct a 2×2 deformation gradient from SVD factors and (possibly updated) singular values.
///
/// F = U · diag(sigma) · Vᵀ
#[inline]
pub(crate) fn reconstruct_f(u: Mat2, sigma: Vec2, vt: Mat2) -> Mat2 {
    u * Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y)) * vt
}

/// Reconstruct a full symmetric Kirchhoff stress tensor from principal (Hencky-basis)
/// stresses and the LEFT singular vectors of `F`'s SVD (`F = U·Σ·Vᵀ`).
///
/// τ = U · diag(τ_principal) · Uᵀ -- the standard result for an isotropic hyperelastic
/// material: Kirchhoff/Cauchy stress is coaxial with the left stretch tensor's
/// eigenvectors (U), not V (see e.g. Bonet & Wood, "Nonlinear Continuum Mechanics for
/// Finite Element Analysis"). Distinct from `reconstruct_f`, which rebuilds F itself
/// (U on the left, Vᵀ on the right) for PLASTIC/irreversible return-mapping materials
/// (Rankine, VonMises) that permanently alter the deformation gradient -- this helper
/// is for REVERSIBLE materials whose principal stress response is asymmetric (e.g. a
/// no-compression/tension-only law) but that never modify F, only its own stress
/// output for the CURRENT F.
#[inline]
pub(crate) fn reconstruct_stress_from_principal(u: Mat2, tau_principal: Vec2) -> Mat2 {
    u * Mat2::from_diagonal(tau_principal) * u.transpose()
}

/// Convert 2D principal Kirchhoff stresses back to Hencky strains (inverse of corotated elastic).
///
/// For corotated/Hencky elastic: τᵢ = (2µ+λ)·εᵢ + λ·ε_j  →  system inversion.
/// Inverse: ε = A⁻¹·τ where det(A) = 4µ(µ+λ).
#[inline]
pub(crate) fn stress_to_hencky(tau: Vec2, lambda: f32, mu: f32) -> Vec2 {
    let det = 4.0 * mu * (mu + lambda);
    let a = 2.0 * mu + lambda;
    Vec2::new(
        (a * tau.x - lambda * tau.y) / det,
        (a * tau.y - lambda * tau.x) / det,
    )
}

/// Rankine (maximum principal stress) tensile-damage estimate -- the same failure
/// criterion `RankineMaterial` uses internally (max principal Kirchhoff stress vs.
/// an exponentially-softening tensile threshold), exposed as a standalone read-only
/// analysis function so ANY material can track real structural damage without
/// adopting Rankine's full constitutive/return-mapping model as its own stress
/// response. Does not modify `deformation_gradient` -- purely observational, safe
/// to call alongside a material's own (unrelated) `update_particle`, e.g. a
/// muscle-actuated `NeoHookeanMaterial` that still needs a damage/health
/// signal without giving up its own stress model.
///
/// `lambda`/`mu` should be the SAME Lamé parameters the calling material already
/// uses for its own elastic response -- this reads the strain state via the
/// same Hencky-strain path Rankine's own `update_particle` computes from, just
/// without writing the projected state back into `F`.
pub fn rankine_damage_estimate(
    deformation_gradient: Mat2,
    lambda: f32,
    mu: f32,
    tensile_strength: f32,
    softening_rate: f32,
    prior_damage: f32,
) -> f32 {
    let (_, sigma, _) = svd2(deformation_gradient);
    let eps = hencky_strains(sigma);
    let a = 2.0 * mu + lambda;
    let tau = Vec2::new(a * eps.x + lambda * eps.y, lambda * eps.x + a * eps.y);

    let t_eff = (tensile_strength * (-softening_rate * prior_damage).exp())
        .max(tensile_strength * RANKINE_MIN_RESIDUAL_TENSILE_FRACTION);
    let tau_proj = Vec2::new(tau.x.min(t_eff), tau.y.min(t_eff));
    if tau_proj == tau {
        return prior_damage; // within the tensile limit, no new damage
    }

    let eps_proj = stress_to_hencky(tau_proj, lambda, mu);
    (prior_damage + (eps - eps_proj).length()).min(rankine_damage_saturation_point(softening_rate))
}

/// 2D polar decomposition: returns the rotation R such that F = R·S.
///
/// Uses the analytical formula for 2×2 matrices (no SVD needed):
///   x = F₀₀+F₁₁, y = F₁₀−F₀₁, norm = √(x²+y²), R = `[[x,−y],[y,x]]`/norm
/// Returns Mat2::IDENTITY when F is near-singular (norm ≤ ε).
/// Used identically by CorotatedMaterial, StomakhinMaterial, VonMisesMaterial, DruckerPragerMaterial.
pub fn polar_decomposition_2d(f: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    if norm > f32::EPSILON {
        Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm)
    } else {
        Mat2::IDENTITY
    }
}

/// Corotated-elastic Kirchhoff stress: `2*mu*(F-R)*F^T + lambda*(J-1)*J*I`,
/// the standard hyperelastic form (Stomakhin et al. 2013, the same
/// corotated model this engine's snow material already cites). Every
/// plastic material that returns to this SAME elastic branch after its own
/// return-mapping (DruckerPragerMaterial, VonMisesMaterial, RankineMaterial,
/// MuIRheologyMaterial) used to hand-duplicate this exact formula in its own
/// `kirchhoff_stress` -- each material's plasticity lives entirely in its
/// own `update_particle`, this function owns only the shared elastic stress
/// evaluated on the (already plastically corrected) current F.
#[inline]
pub(crate) fn corotated_elastic_stress(f: Mat2, lambda: f32, mu: f32) -> Mat2 {
    let j = f.determinant();
    if j <= MIN_J {
        return Mat2::ZERO;
    }
    let r = polar_decomposition_2d(f);
    2.0 * mu * (f - r) * f.transpose() + lambda * (j - 1.0) * j * Mat2::IDENTITY
}

/// Corotated elastic energy density `Psi(F) = mu*||F-R||_F^2 +
/// 0.5*lambda*(J-1)^2` (Stomakhin et al. 2013). `corotated_elastic_stress` is
/// its First-Piola gradient times `F^T`: `dPsi/dF = 2*mu*(F-R) +
/// lambda*(J-1)*J*F^-T`, and `(dPsi/dF)*F^T = 2*mu*(F-R)*F^T +
/// lambda*(J-1)*J*I`. Zero at `J <= MIN_J`, like the stress ("no force").
///
/// Test-only: the oracle `model_residual`'s gradient is checked against in
/// `spacetime::solver::implicit_corotated`, whose trust region measures a
/// Gauss-Newton `0.5*||residual||^2` merit instead (see `newton_solve`).
#[cfg(test)]
#[inline]
pub(crate) fn corotated_elastic_energy_density(f: Mat2, lambda: f32, mu: f32) -> f32 {
    let j = f.determinant();
    if j <= MIN_J {
        return 0.0;
    }
    let r = polar_decomposition_2d(f);
    let dev = f - r;
    mu * frob(dev, dev) + 0.5 * lambda * (j - 1.0) * (j - 1.0)
}

/// Analytic Jacobian-vector product of `polar_decomposition_2d`: given `F`
/// and a perturbation direction `dF`, returns `dR`. Closed form for this
/// engine's `R = M/norm`, `M = [[x,-y],[y,x]]`, `x = tr(F)`, `y = F01-F10`
/// (the generic SVD-based polar derivative does not match it):
///   dx=tr(dF), dy=dF01-dF10, dM=[[dx,-dy],[dy,dx]]
///   d(norm)=(x*dx+y*dy)/norm,  dR=dM/norm - R*d(norm)/norm
/// Checked by central finite differences
/// (`polar_decomposition_jvp_matches_finite_difference`): max relative error
/// 0.06% at h = 1e-3. Smaller h is worse here (f32 cancellation on this
/// normalized quantity).
///
/// Used by `corotated_elastic_stress_jvp` below, which
/// `spacetime::solver::implicit_corotated` calls to build its local Hessian.
#[inline]
pub(crate) fn polar_decomposition_2d_jvp(f: Mat2, df: Mat2) -> Mat2 {
    let x = f.x_axis.x + f.y_axis.y;
    let y = f.x_axis.y - f.y_axis.x;
    let norm = (x * x + y * y).sqrt();
    if norm <= f32::EPSILON {
        return Mat2::ZERO;
    }
    let dx = df.x_axis.x + df.y_axis.y;
    let dy = df.x_axis.y - df.y_axis.x;
    let d_norm = (x * dx + y * dy) / norm;
    let dm = Mat2::from_cols(Vec2::new(dx, dy), Vec2::new(-dy, dx));
    let r = Mat2::from_cols(Vec2::new(x, y) / norm, Vec2::new(-y, x) / norm);
    dm * (1.0 / norm) - r * (d_norm / norm)
}

/// Analytic Jacobian-vector product of `corotated_elastic_stress`'s
/// Kirchhoff stress w.r.t. `F`: given `F` and a perturbation `dF`, returns
/// `dTau`. Product rule on `2*mu*(F-R)*F^T + lambda*(J-1)*J*I` using
/// `polar_decomposition_2d_jvp` above and `dJ=J*(F^-T:dF)`. The
/// Hessian-vector-product building block of an implicit (Newton-CG) solve for
/// every material on this elastic branch (DruckerPrager, VonMises, Rankine,
/// MuIRheology, Corotated), see `spacetime::solver::implicit_corotated`.
/// Checked by central finite differences
/// (`corotated_stress_jvp_matches_finite_difference`): max relative error
/// 0.23% at sand-magnitude stiffness.
///
/// Zero at `j <= MIN_J`, where the stress is pinned at zero: a practical
/// simplification at a hard clamp, not a claim about the continuous
/// derivative.
#[inline]
pub(crate) fn corotated_elastic_stress_jvp(f: Mat2, df: Mat2, lambda: f32, mu: f32) -> Mat2 {
    let j = f.determinant();
    if j <= MIN_J {
        return Mat2::ZERO;
    }
    let r = polar_decomposition_2d(f);
    let dr = polar_decomposition_2d_jvp(f, df);
    let f_t = f.transpose();
    let df_t = df.transpose();
    let f_inv_t = f.inverse().transpose();
    let d_j = j * frob(f_inv_t, df); // dJ = J*(F^-T:dF)

    let d_elastic_term = (df - dr) * f_t + (f - r) * df_t;
    let d_vol_term = (2.0 * j - 1.0) * d_j; // d[(J-1)*J] = (2J-1)*dJ
    2.0 * mu * d_elastic_term + lambda * d_vol_term * Mat2::IDENTITY
}

/// A PSD approximation of `d(tau)/d(L)` (the Kirchhoff stress derivative
/// with respect to a velocity-gradient perturbation `dl`, through `dF =
/// dt*dl*f_n`), applied to `dl`.
///
/// Teran et al. 2005's SPD construction proves `dP:dF >= 0` for First-Piola
/// stress with a material-space perturbation. This engine's force pairs
/// Kirchhoff stress with the spatial kernel gradient (as `tmp/ziran2020` and
/// `tmp/GeoTaichi` do), and `tau = P*F^T` as a function of `L` inherits no such
/// guarantee: a Piola-projected operator converted by the product rule still
/// let the residual grow across CG iterations, which a PSD system cannot do.
///
/// So the local 4x4 operator (`L` and `tau` both have 4 components in 2D) is
/// built from four calls to the exact `corotated_elastic_stress_jvp`,
/// symmetrized (only the symmetric part enters the quadratic form), then
/// eigendecomposed (`jacobi_eigen_symmetric_4x4`, Golub & Van Loan's cyclic
/// Jacobi) with negative eigenvalues clamped to zero: the nearest SPD matrix
/// in Frobenius norm (Higham 1988), Teran's clamping principle without his
/// `dP/dF`-specific closed form. A Gershgorin diagonal shift, also
/// sufficient for PSD, swamps the curvature at production stiffness
/// (`basic_sand`'s E = 15 MPa): the line search rejected every step.
///
/// Split in two: [`corotated_kirchhoff_dtau_dl_psd_matrix`] builds the matrix
/// once per trial `v` (it does not depend on `dl`); [`apply_dtau_dl_psd_matrix`]
/// applies it for a 4x4 mat-vec, instead of rebuilding it every CG iteration.
///
/// Test-only: `implicit_corotated` uses `steihaug_cg` (Nocedal & Wright),
/// which handles indefinite curvature and needs no PSD projection. Kept, with
/// `corotated_kirchhoff_dtau_dl_psd_tests`, in case a PSD operator is needed
/// again.
#[cfg(test)]
pub(crate) fn corotated_kirchhoff_dtau_dl_psd_matrix(
    f: Mat2,
    lambda: f32,
    mu: f32,
    dt: f32,
    f_n: Mat2,
) -> [[f32; 4]; 4] {
    let basis = [
        Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
        Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
    ];
    let mut h = [[0.0f32; 4]; 4];
    for (col, &e) in basis.iter().enumerate() {
        let df = dt * e * f_n;
        let d_tau = corotated_elastic_stress_jvp(f, df, lambda, mu);
        let d_tau_vec = [
            d_tau.x_axis.x,
            d_tau.x_axis.y,
            d_tau.y_axis.x,
            d_tau.y_axis.y,
        ];
        for (row, &val) in d_tau_vec.iter().enumerate() {
            h[row][col] = val;
        }
    }
    let mut sym = [[0.0f32; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            sym[i][j] = 0.5 * (h[i][j] + h[j][i]);
        }
    }
    spd_project_symmetric_4x4(sym)
}

/// Eigenvalue-clamping SPD projection (Teran, Sifakis, Irving & Fedkiw
/// 2005's own principle, applied generally): eigendecompose the given real
/// symmetric matrix, clamp every negative eigenvalue to zero, reconstruct.
/// Produces the nearest SPD matrix in Frobenius norm (Higham 1988) --
/// unlike a Gershgorin diagonal shift, it never touches an eigenvalue that
/// was already non-negative, so it cannot dilute real positive curvature
/// regardless of how stiff the underlying material is (see this function's
/// caller for the measured failure this replaces).
///
/// Test-only -- see `corotated_kirchhoff_dtau_dl_psd_matrix`'s doc.
#[cfg(test)]
fn spd_project_symmetric_4x4(sym: [[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let (eigenvalues, v) = jacobi_eigen_symmetric_4x4(sym);
    let clamped = eigenvalues.map(|lambda| lambda.max(0.0));
    let mut out = [[0.0f32; 4]; 4];
    for (i, out_row) in out.iter_mut().enumerate() {
        for (j, out_ij) in out_row.iter_mut().enumerate() {
            *out_ij = (0..4).map(|k| v[i][k] * clamped[k] * v[j][k]).sum();
        }
    }
    out
}

/// Classical cyclic Jacobi eigenvalue algorithm for a real symmetric
/// matrix (Golub & Van Loan, "Matrix Computations", 4th ed., section
/// 8.4.3) -- standard and numerically robust at this fixed 4x4 scale
/// (used here only for the local per-particle Kirchhoff/spatial-gradient
/// operator, never a global assembly). Returns eigenvalues and the matching
/// eigenvector matrix (columns), i.e. `sym == v * diag(eigenvalues) * v^T`.
///
/// Test-only -- see `corotated_kirchhoff_dtau_dl_psd_matrix`'s doc.
#[cfg(test)]
fn jacobi_eigen_symmetric_4x4(mut a: [[f32; 4]; 4]) -> ([f32; 4], [[f32; 4]; 4]) {
    let mut v = [[0.0f32; 4]; 4];
    for (i, row) in v.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    for _sweep in 0..50 {
        let off: f32 = (0..4)
            .flat_map(|p| ((p + 1)..4).map(move |q| (p, q)))
            .map(|(p, q)| a[p][q] * a[p][q])
            .sum();
        if off < 1.0e-20 {
            break;
        }
        for p in 0..4 {
            for q in (p + 1)..4 {
                let a_pq = a[p][q];
                if a_pq.abs() < 1.0e-12 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * a_pq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                let tau = s / (1.0 + c);
                let a_pp = a[p][p];
                let a_qq = a[q][q];
                a[p][p] = a_pp - t * a_pq;
                a[q][q] = a_qq + t * a_pq;
                a[p][q] = 0.0;
                a[q][p] = 0.0;
                let mut updated_p = [0.0f32; 4];
                let mut updated_q = [0.0f32; 4];
                for (i, row) in a.iter().enumerate() {
                    if i != p && i != q {
                        let a_ip = row[p];
                        let a_iq = row[q];
                        updated_p[i] = a_ip - s * (a_iq + tau * a_ip);
                        updated_q[i] = a_iq + s * (a_ip - tau * a_iq);
                    }
                }
                // `a` stays symmetric: each updated off-diagonal entry
                // writes into both its row and its mirrored column.
                for (i, (&up, &uq)) in updated_p.iter().zip(&updated_q).enumerate() {
                    if i != p && i != q {
                        a[i][p] = up;
                        a[p][i] = up;
                        a[i][q] = uq;
                        a[q][i] = uq;
                    }
                }
                for row in v.iter_mut() {
                    let v_ip = row[p];
                    let v_iq = row[q];
                    row[p] = v_ip - s * (v_iq + tau * v_ip);
                    row[q] = v_iq + s * (v_ip - tau * v_iq);
                }
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2], a[3][3]], v)
}

/// Application step for [`corotated_kirchhoff_dtau_dl_psd_matrix`]'s output:
/// a 4x4 mat-vec, no JVP evaluations. Test-only, for
/// [`corotated_kirchhoff_dtau_dl_psd`]'s direct comparison.
#[cfg(test)]
#[inline]
pub(crate) fn apply_dtau_dl_psd_matrix(matrix: &[[f32; 4]; 4], dl: Mat2) -> Mat2 {
    let dl_vec = [dl.x_axis.x, dl.x_axis.y, dl.y_axis.x, dl.y_axis.y];
    let mut out = [0.0f32; 4];
    for (i, row) in matrix.iter().enumerate() {
        out[i] = row.iter().zip(&dl_vec).map(|(a, b)| a * b).sum();
    }
    Mat2::from_cols(Vec2::new(out[0], out[1]), Vec2::new(out[2], out[3]))
}

/// Convenience wrapper matching the original single-call API (build +
/// apply in one shot) -- kept for the test suite's own direct-comparison
/// checks; production code (`spacetime::solver::implicit_corotated`) uses
/// the split build/apply pair instead to avoid rebuilding the matrix once
/// per CG iteration.
#[cfg(test)]
pub(crate) fn corotated_kirchhoff_dtau_dl_psd(
    f: Mat2,
    lambda: f32,
    mu: f32,
    dt: f32,
    f_n: Mat2,
    dl: Mat2,
) -> Mat2 {
    let matrix = corotated_kirchhoff_dtau_dl_psd_matrix(f, lambda, mu, dt, f_n);
    apply_dtau_dl_psd_matrix(&matrix, dl)
}

/// Frobenius inner product `sum_ij a_ij*b_ij` -- used by the plain JVP
/// above, the Gershgorin-PSD construction, and this module's own
/// relative-error checks.
#[inline]
pub(crate) fn frob(a: Mat2, b: Mat2) -> f32 {
    a.x_axis.x * b.x_axis.x
        + a.x_axis.y * b.x_axis.y
        + a.y_axis.x * b.y_axis.x
        + a.y_axis.y * b.y_axis.y
}

/// Fixed-point iteration to a self-consistent (closest-point-projection)
/// plastic multiplier -- real numerical rigor per Simo & Taylor 1985
/// ("Consistent tangent operators for rate-independent elastoplasticity,"
/// CMAME 48:101-118) and Simo & Hughes, *Computational Inelasticity* (1998),
/// the standard reference on return-mapping consistency. Generic across
/// EVERY plastic material with a hardening-dependent yield surface (DP's
/// friction-angle hardening, VonMises' yield-stress evolution, Rankine's
/// damage softening, NACC's consolidation state, mu(I)'s own friction law) --
/// each material supplies its OWN yield equation via `yield_at`, this
/// function owns only the shared iteration/convergence logic, so adding
/// self-consistency to another material never means re-deriving or
/// re-implementing this loop, just plugging in that material's own closure.
///
/// `initial_gamma`: the single-pass (pre-step) plastic multiplier, used as
/// the starting guess -- real materials without self-consistency already
/// compute this value, so callers get it for free.
/// `hardening_state`: the pre-step internal variable (q, damage, etc.).
/// `yield_at`: given a CANDIDATE end-of-step hardening state
/// (`hardening_state + candidate_gamma`), returns the yield function's value
/// there -- the only material-specific piece.
/// 8 iterations is real headroom, not a tuned number: every hardening
/// law in this engine is a bounded, smooth, saturating function of its own
/// internal variable, making this a contraction that converges to float
/// precision in 2-3 passes in practice (confirmed on `DruckerPragerMaterial`,
/// the first real adopter).
pub fn self_consistent_plastic_multiplier(
    initial_gamma: f32,
    hardening_state: f32,
    mut yield_at: impl FnMut(f32) -> f32,
) -> f32 {
    let mut gamma = initial_gamma;
    for _ in 0..8 {
        let gamma_next = yield_at(hardening_state + gamma.max(0.0));
        let converged = (gamma_next - gamma).abs() < 1.0e-6;
        gamma = gamma_next;
        if converged {
            break;
        }
    }
    gamma
}

/// Generic 1D elastic-perfectly-plastic return mapping: clamps a trial value
/// to an interval around a permanent offset and moves the excess into the
/// offset. The scalar sibling of the tensor radial return (e.g.
/// `VonMisesMaterial::update_particle`'s `dev * (effective_yield/elastic_dev)`):
/// same elastic trial, yield check and return mapping, an interval clamp
/// instead of a norm rescale. Used by `rod::plasticity`'s bending curvature;
/// any scalar-state plasticity (a damage variable, an axial yield) can use it.
///
/// `trial`: the fully-elastic candidate value (e.g. current curvature).
/// `permanent`: the current permanent/plastic offset (e.g. rest curvature).
/// `limit`: the positive elastic half-width of the interval.
pub fn scalar_return_map(trial: f32, permanent: f32, limit: f32) -> f32 {
    let elastic = trial - permanent;
    if elastic > limit {
        permanent + (elastic - limit)
    } else if elastic < -limit {
        permanent + (elastic + limit)
    } else {
        permanent
    }
}

/// CFL timestep bound from elastic longitudinal wave speed c_P = √((λ+2µ)·h / ρ).
///
/// `hardening` = 1.0 for materials without hardening (elastic, sand, von Mises).
/// `hardening` = particle.hardening_scale for corotated/snow (stiffness grows on compression).
/// Returns f32::INFINITY when the material has zero or negative stiffness.
pub fn elastic_wave_dt(
    lambda: f32,
    mu: f32,
    hardening: f32,
    density: f32,
    min_density: f32,
    cell_width: f32,
    material_cfl: f32,
) -> f32 {
    let rho = density.max(min_density);
    let modulus = ((lambda + 2.0 * mu) * hardening).max(0.0);
    if modulus <= f32::EPSILON {
        return f32::INFINITY;
    }
    let c = (modulus / rho).sqrt();
    if c <= f32::EPSILON {
        return f32::INFINITY;
    }
    material_cfl * cell_width / c
}

/// Convert Young's modulus E and Poisson's ratio ν to Lamé parameters (λ, µ).
///
/// Valid for ν ∈ (−1, 0.5), E > 0. Matches the API used by sparkl and wgsparkl:
///   `ElasticCoefficients::from_young_modulus(E, nu)`
///
/// # Canonical values (from published MPM papers)
///
/// | Material       | E           | ν    | Source                     |
/// |----------------|-------------|------|----------------------------|
/// | sand (demo)    | 1.0×10⁵     | 0.20 | Klar 2016, sparkl basic2   |
/// | snow           | 1.4×10⁵     | 0.20 | Stomakhin 2013, MPM2D ref  |
/// | soft elastic   | 5.0×10⁶     | 0.20 | wgsparkl elasticity2       |
/// | soft tissue    | 1.0×10³     | 0.45 | typical MPM bio            |
///
/// Note: these are in whatever units your grid uses (not necessarily SI).
/// At emerge's default `grid_cell_size = 1.0`, use values that give
/// `sqrt((λ+2µ)/ρ) ≈ 10–60 cells/s` for interactive framerates.
pub fn lame_from_young(young_modulus: f32, poisson_ratio: f32) -> (f32, f32) {
    debug_assert!(young_modulus > 0.0, "Young's modulus must be positive");
    debug_assert!(
        poisson_ratio > -1.0 && poisson_ratio < 0.5,
        "Poisson's ratio must be in (-1, 0.5)"
    );
    let lambda =
        young_modulus * poisson_ratio / ((1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio));
    let mu = young_modulus / (2.0 * (1.0 + poisson_ratio));
    (lambda, mu)
}

/// SI Young's modulus and Poisson's ratio to grid Lamé parameters:
/// `lambda_SI / (rho dx^2)`, the squared elastic wave speed in cells/s, the
/// same at every timestep.
pub fn lame_from_si(
    young_modulus_pa: f32,
    poisson_ratio: f32,
    rest_density_kg_m3: f32,
    dx_meters: f32,
) -> (f32, f32) {
    let (lambda_si, mu_si) = lame_from_young(young_modulus_pa, poisson_ratio);
    let scale = 1.0 / (rest_density_kg_m3 * dx_meters * dx_meters);
    (lambda_si * scale, mu_si * scale)
}

/// Convert SI gravity (m/s²) to solver units (grid cells / s²).
///
/// In the solver `v += gravity * sub_dt` where sub_dt is in real seconds,
/// so gravity must be in [cells/s²] = g_SI / dx_meters.
///
/// # Example -- Earth gravity at 1 cm/cell
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::gravity_to_grid;
/// use glam::Vec2;
/// let g = gravity_to_grid(Vec2::new(0.0, -9.81), 0.01);
/// // g ≈ Vec2::new(0.0, -981.0) cells/s²
/// ```
pub fn gravity_to_grid(g_si: glam::Vec2, dx_meters: f32) -> glam::Vec2 {
    g_si / dx_meters
}

/// Von Neumann & Richtmyer 1950 (LA-671) artificial bulk viscosity, EOS-
/// agnostic core: `q = rho*(c0*h^2*(div v)^2 - c1*h*c_sound*div v)`, gated
/// to compression (`div v < 0`), where shocks form. `c0 = (gamma+1)/4` is
/// the Kurapatenko 1967 weak-shock coefficient; `c1 = 1.0` (Landshoff) is
/// the standard linear term. Each EOS supplies its own `c_sound` (its
/// `dp/drho` at the current state) and `gamma` (an ideal gas's adiabatic
/// index, or a Tait liquid's `eos_power` standing in, see
/// `fluid::artificial_bulk_viscosity`); this function owns only the shared
/// form, so the ideal gas reuses it without fake Tait parameters.
#[inline]
pub(crate) fn von_neumann_richtmyer_q(
    rest_density: f32,
    j: f32,
    div_v: f32,
    grid_cell_size: f32,
    c_sound: f32,
    weak_shock_gamma: f32,
) -> f32 {
    if div_v.is_nan() || div_v >= 0.0 {
        return 0.0;
    }
    let c0_quadratic = (weak_shock_gamma + 1.0) * 0.25;
    const C1_LINEAR: f32 = 1.0;
    let rho = rest_density / j;
    let h = grid_cell_size;
    let quadratic = c0_quadratic * h * h * div_v * div_v;
    let linear = C1_LINEAR * h * c_sound * div_v;
    let q = rho * (quadratic - linear);
    if q.is_finite() { q } else { 0.0 }
}

/// Single-sphere Stokes drag rate (Stokes 1851) for
/// `LinearDragField::drag_coefficient` (1/s, the solver's own time unit):
/// `F = 6 pi mu r v` gives `dv/dt = -(6 pi mu r / m) v`.
///
/// Only valid at low Reynolds number (`Re = rho_air v 2r / mu <~ 1`), e.g.
/// fine dust or sand grains in a gentle wind. A 1 cm ball at 1 m/s sits at
/// `Re ~ 2000-3000`, where drag is quadratic (form drag), not this.
///
/// # Example -- fine dust grain in air (r=50 micron, m=6.5e-10 kg, 20 C air)
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::materials::stokes_drag_rate_from_si;
/// let k = stokes_drag_rate_from_si(5.0e-5, 6.5e-10, 1.81e-5);
/// # let _ = k;
/// ```
pub fn stokes_drag_rate_from_si(radius_m: f32, mass_kg: f32, dynamic_viscosity_pa_s: f32) -> f32 {
    6.0 * std::f32::consts::PI * dynamic_viscosity_pa_s * radius_m / mass_kg
}

#[cfg(test)]
mod rankine_damage_estimate_tests {
    use super::*;

    #[test]
    fn no_damage_within_tensile_limit() {
        let f = Mat2::from_cols(Vec2::new(1.01, 0.0), Vec2::new(0.0, 1.0));
        let damage = rankine_damage_estimate(f, 1000.0, 1000.0, 1.0e6, 1.0, 0.0);
        assert_eq!(
            damage, 0.0,
            "tiny strain must stay under a huge tensile threshold"
        );
    }

    #[test]
    fn damage_accumulates_past_tensile_limit() {
        let f = Mat2::from_cols(Vec2::new(1.5, 0.0), Vec2::new(0.0, 1.0));
        let damage = rankine_damage_estimate(f, 1000.0, 1000.0, 10.0, 1.0, 0.0);
        assert!(
            damage > 0.0,
            "large tensile strain must accumulate real damage"
        );
    }

    #[test]
    fn damage_never_decreases_across_repeated_overload() {
        let f = Mat2::from_cols(Vec2::new(1.5, 0.0), Vec2::new(0.0, 1.0));
        let mut damage = 0.0;
        for _ in 0..5 {
            let next = rankine_damage_estimate(f, 1000.0, 1000.0, 10.0, 1.0, damage);
            assert!(
                next >= damage,
                "damage must be monotonically non-decreasing"
            );
            damage = next;
        }
        assert!(damage > 0.0);
    }

    #[test]
    fn does_not_mutate_deformation_gradient() {
        // Purely observational -- caller's F is untouched, function only reads it.
        let f = Mat2::from_cols(Vec2::new(1.5, 0.0), Vec2::new(0.0, 1.0));
        let f_before = f;
        let _ = rankine_damage_estimate(f, 1000.0, 1000.0, 10.0, 1.0, 0.0);
        assert_eq!(f, f_before);
    }
}

#[cfg(test)]
mod deformation_increment_tests {
    use super::*;

    fn deformation_increment_exp(a: Mat2) -> Mat2 {
        Mat2::IDENTITY + deformation_increment_exp_minus_identity(a)
    }

    fn matrix_error(a: Mat2, b: Mat2) -> f32 {
        (a.x_axis - b.x_axis).length() + (a.y_axis - b.y_axis).length()
    }

    #[test]
    fn opposite_rates_are_exactly_reversible_without_volume_ratchet() {
        let a = Mat2::from_diagonal(Vec2::new(0.02, -0.01));
        let forward = deformation_increment_exp(a);
        let backward = deformation_increment_exp(-a);
        let round_trip = backward * forward;
        assert!(
            matrix_error(round_trip, Mat2::IDENTITY) < 1.0e-6,
            "exp(-A)exp(A) must be identity: {round_trip:?}"
        );
    }

    #[test]
    fn rigid_rotation_preserves_volume_and_matches_angle() {
        let angle = 0.37;
        let spin = Mat2::from_cols(Vec2::new(0.0, angle), Vec2::new(-angle, 0.0));
        let increment = deformation_increment_exp(spin);
        let expected = Mat2::from_angle(angle);
        assert!(
            matrix_error(increment, expected) < 1.0e-6,
            "matrix exponential of planar spin must be the matching rotation: \
             expected={expected:?} got={increment:?}"
        );
        assert!((increment.determinant() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn determinant_obeys_exact_continuity_identity() {
        let a = Mat2::from_cols(Vec2::new(0.11, -0.07), Vec2::new(0.13, -0.03));
        let increment = deformation_increment_exp(a);
        let expected_det = (a.x_axis.x + a.y_axis.y).exp();
        assert!(
            (increment.determinant() - expected_det).abs() < 2.0e-6,
            "det(exp(A)) must equal exp(trace(A)): expected={expected_det} got={}",
            increment.determinant()
        );
    }

    /// The same closed form in f64, row-major `[m00, m01, m10, m11]`.
    fn exp_f64(a: f64, b: f64, c: f64, d: f64) -> [f64; 4] {
        let half_trace = 0.5 * (a + d);
        let half_difference = 0.5 * (a - d);
        let delta_sq = half_difference * half_difference + b * c;
        let (even, odd) = if delta_sq == 0.0 {
            (1.0, 1.0)
        } else if delta_sq > 0.0 {
            let delta = delta_sq.sqrt();
            (delta.cosh(), delta.sinh() / delta)
        } else {
            let omega = (-delta_sq).sqrt();
            (omega.cos(), omega.sin() / omega)
        };
        let scale = half_trace.exp();
        [
            scale * (even + (a - half_trace) * odd),
            scale * b * odd,
            scale * c * odd,
            scale * (even + (d - half_trace) * odd),
        ]
    }

    /// 200 000 substeps of an oscillating velocity gradient applied to a
    /// loaded `F` (1.0027 across, 0.9899 along the load, entries on both
    /// sides of 1), against the same exponential integrated in f64. The
    /// former update (the increment formed as `1 + small`, then rescaled
    /// onto a carried volume) ended this at -4.7e-3 (0.04 / s), -1.4e-3
    /// (0.4 / s) and -4.6e-4 (4 / s) in `ln(F_xx / F_yy)`, and -2.5e-3 in
    /// `ln det F` at 4 / s: a one-way drift. What this form leaves is the
    /// size of an unbiased walk of f32 roundings over that many steps.
    #[test]
    fn repeated_substeps_do_not_bias_a_loaded_shape_or_volume() {
        let dt = 2.94e-4_f32;
        let omega = std::f32::consts::TAU / 200.0;
        for amplitude in [0.04_f32, 0.4, 4.0] {
            let mut f = Mat2::from_diagonal(Vec2::new(1.0027, 0.9899));
            let mut exact = [f64::from(f.x_axis.x), 0.0, 0.0, f64::from(f.y_axis.y)];
            for step in 1..=200_000 {
                let phase = omega * step as f32;
                let c = Mat2::from_cols_array(&[
                    amplitude * phase.cos(),
                    amplitude * (phase * 0.37).sin(),
                    amplitude * (phase * 0.61).cos(),
                    -amplitude * phase.cos() + 0.3 * amplitude * (phase * 1.3).sin(),
                ]);
                let dc = dt * c;
                let m = exp_f64(
                    f64::from(dc.x_axis.x),
                    f64::from(dc.y_axis.x),
                    f64::from(dc.x_axis.y),
                    f64::from(dc.y_axis.y),
                );
                exact = [
                    m[0] * exact[0] + m[1] * exact[2],
                    m[0] * exact[1] + m[1] * exact[3],
                    m[2] * exact[0] + m[3] * exact[2],
                    m[2] * exact[1] + m[3] * exact[3],
                ];
                f = advance_deformation_gradient(f, dc);
            }
            let det_error = (f64::from(f.x_axis.x) * f64::from(f.y_axis.y)
                - f64::from(f.y_axis.x) * f64::from(f.x_axis.y))
            .ln()
                - (exact[0] * exact[3] - exact[1] * exact[2]).ln();
            let shape_error =
                (f64::from(f.x_axis.x) / f64::from(f.y_axis.y)).ln() - (exact[0] / exact[3]).ln();
            println!(
                "amplitude {amplitude}/s: ln det error {det_error:+.2e}, ln(F_xx/F_yy) error {shape_error:+.2e}"
            );
            assert!(
                det_error.abs() < 3.0e-4 && shape_error.abs() < 3.0e-4,
                "200 000 substeps at {amplitude}/s must not drift F one way: \
                 ln det error {det_error:+.2e}, ln(F_xx/F_yy) error {shape_error:+.2e}"
            );
        }
    }
}

#[cfg(test)]
mod corotated_elastic_stress_jvp_tests {
    use super::*;

    fn frob_pub(a: Mat2, b: Mat2) -> f32 {
        a.x_axis.x * b.x_axis.x
            + a.x_axis.y * b.x_axis.y
            + a.y_axis.x * b.y_axis.x
            + a.y_axis.y * b.y_axis.y
    }

    fn states() -> Vec<Mat2> {
        vec![
            Mat2::IDENTITY,
            Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97)),
            Mat2::from_cols(Vec2::new(1.0, 0.4), Vec2::new(0.1, 1.1)),
            Mat2::from_cols(Vec2::new(0.9, -0.2), Vec2::new(0.3, 1.2)),
            Mat2::from_cols(Vec2::new(0.7, 0.6), Vec2::new(-0.5, 1.1)),
        ]
    }
    fn directions() -> Vec<Mat2> {
        vec![
            Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
            Mat2::from_cols(Vec2::new(0.2, 0.3), Vec2::new(-0.1, 0.5)),
        ]
    }

    /// Central finite difference at h = 1e-3, not smaller: f32 cancellation on
    /// `polar_decomposition_2d`'s normalized quantity makes smaller h worse.
    #[test]
    fn matches_finite_difference_at_moderate_and_real_sand_stiffness() {
        let h = 1.0e-3f32;
        for &(lambda, mu) in &[(50.0f32, 30.0f32), (2.0e7, 1.5e7)] {
            let mut max_rel_err = 0.0f32;
            for &f in &states() {
                for &df in &directions() {
                    let plus = corotated_elastic_stress(f + h * df, lambda, mu);
                    let minus = corotated_elastic_stress(f - h * df, lambda, mu);
                    let numeric = (plus - minus) * (1.0 / (2.0 * h));
                    let analytic = corotated_elastic_stress_jvp(f, df, lambda, mu);
                    let diff = analytic - numeric;
                    let rel_err =
                        frob_pub(diff, diff).sqrt() / frob_pub(analytic, analytic).sqrt().max(1.0);
                    max_rel_err = max_rel_err.max(rel_err);
                }
            }
            println!(
                "corotated_elastic_stress_jvp: lambda={lambda} mu={mu} max_rel_err={max_rel_err:.6}"
            );
            assert!(
                max_rel_err < 5.0e-3,
                "JVP wrong at lambda={lambda} mu={mu}: {max_rel_err}"
            );
        }
    }

    #[test]
    fn zero_below_min_j_floor_matching_the_stress_itself() {
        // A near-singular F that pins `corotated_elastic_stress` at
        // Mat2::ZERO -- the JVP must match that pinned-constant behavior.
        let f = Mat2::from_cols(Vec2::new(1.0e-8, 0.0), Vec2::new(0.0, 1.0e-8));
        let df = Mat2::from_cols(Vec2::new(0.1, 0.0), Vec2::new(0.0, 0.1));
        assert_eq!(corotated_elastic_stress(f, 100.0, 50.0), Mat2::ZERO);
        assert_eq!(corotated_elastic_stress_jvp(f, df, 100.0, 50.0), Mat2::ZERO);
    }

    /// `corotated_elastic_energy_density` is the potential
    /// `corotated_elastic_stress` is the gradient of, so the trust-region
    /// ratio test solves the problem the other tests trust. Central finite
    /// difference at h = 1e-3 (smaller h is worse here); First-Piola comes from
    /// the Kirchhoff function via `P = tau*F^-T` rather than a second copy of
    /// the formula.
    #[test]
    fn energy_density_gradient_matches_finite_difference_of_the_stress() {
        let h = 1.0e-3f32;
        for &(lambda, mu) in &[(50.0f32, 30.0f32), (2.0e7, 1.5e7)] {
            let mut max_rel_err = 0.0f32;
            for &f in &states() {
                // `Mat2::IDENTITY` is a critical point of `Psi`
                // (F=R, J=1 -> both terms are exactly zero, the analytic
                // global minimum) -- its analytic gradient is exactly
                // zero, which makes central-difference ill-conditioned
                // there by construction: `f(x+h)`/`f(x-h)` are both
                // dominated by the (identical, non-cancelling) O(h^2)
                // curvature term, so `(plus-minus)/(2h)` recovers pure
                // O(h^2)*f'''/6 truncation noise, not a discrepancy
                // (confirmed: at lambda=2e7, this alone produced a raw FD
                // value of ~1.8 against an analytic 0 -- but EVERY other
                // state, all deformed with nonzero analytic
                // gradients, passed at the same h and stiffness with
                // max_rel_err=2.45e-4). Standard, documented FD-checking
                // practice (e.g. PyTorch's `gradcheck`) explicitly skips
                // exactly this scenario -- checking a derivative AT a
                // critical point -- rather than loosening the tolerance
                // for every other, well-conditioned case too.
                if f == Mat2::IDENTITY {
                    continue;
                }
                for &df in &directions() {
                    let plus = corotated_elastic_energy_density(f + h * df, lambda, mu);
                    let minus = corotated_elastic_energy_density(f - h * df, lambda, mu);
                    let numeric = (plus - minus) / (2.0 * h);
                    let tau = corotated_elastic_stress(f, lambda, mu);
                    let piola = tau * f.inverse().transpose();
                    let analytic = frob_pub(piola, df);
                    let rel_err = (analytic - numeric).abs() / analytic.abs().max(1.0);
                    max_rel_err = max_rel_err.max(rel_err);
                }
            }
            println!(
                "corotated_elastic_energy_density: lambda={lambda} mu={mu} max_rel_err={max_rel_err:.6}"
            );
            assert!(
                max_rel_err < 5.0e-3,
                "energy gradient doesn't match the stress at lambda={lambda} mu={mu}: {max_rel_err}"
            );
        }
    }
}

#[cfg(test)]
mod corotated_kirchhoff_dtau_dl_psd_tests {
    use super::*;

    fn states() -> Vec<Mat2> {
        vec![
            Mat2::IDENTITY,
            Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(-0.01, 0.97)),
            Mat2::from_cols(Vec2::new(1.0, 0.1), Vec2::new(-0.05, 1.05)),
            // Large volumetric expansion -- the regime the plain
            // (unprojected) Hessian is known-indefinite in.
            Mat2::from_cols(Vec2::new(3.0, 0.0), Vec2::new(0.0, 3.0)),
            Mat2::from_cols(Vec2::new(0.7, 0.6), Vec2::new(-0.5, 1.1)),
        ]
    }

    fn directions() -> Vec<Mat2> {
        vec![
            Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(0.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(1.0, 0.0)),
            Mat2::from_cols(Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0)),
            Mat2::from_cols(Vec2::new(0.2, 0.1), Vec2::new(-0.1, 0.3)),
            Mat2::from_cols(Vec2::new(-0.1, 0.2), Vec2::new(0.15, -0.2)),
        ]
    }

    /// The actual property this function exists for: `frob(H(dl), dl) >=
    /// 0` for EVERY direction `dl`, at EVERY state including the ones the
    /// plain (unprojected) Hessian is known-indefinite at -- this is what
    /// guarantees CG can never see a negative-curvature direction from
    /// this operator, unlike the plain JVP or a Piola-space projection
    /// (which does not survive the Kirchhoff/spatial-gradient conversion:
    /// CG in `spacetime::solver::implicit_corotated` saw negative curvature
    /// with it).
    #[test]
    fn quadratic_form_is_never_negative() {
        let lambda = 6.0e7f32;
        let mu = 4.0e7f32;
        let dt = 0.016f32;
        for &f in &states() {
            for &f_n in &states() {
                for &dl in &directions() {
                    let h_dl = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, dt, f_n, dl);
                    let q = frob(h_dl, dl);
                    assert!(
                        q >= -1.0e-3 * frob(dl, dl).max(1.0),
                        "quadratic form went negative: f={f:?} f_n={f_n:?} dl={dl:?} q={q}"
                    );
                }
            }
        }
    }

    #[test]
    fn matches_plain_jvp_via_chain_rule_at_moderate_already_psd_states() {
        // At small/moderate deformation with no pre-existing rotation, the
        // real local Hessian's SYMMETRIC PART is already close to PSD (no
        // real Gershgorin shift needed), so this should stay close to the
        // exact chain-rule JVP. NOT bit-identical even here, by design:
        // symmetrizing intentionally discards `dTau/dL`'s antisymmetric
        // part (which contributes nothing to the quadratic form PSD-ness
        // is actually about), so a few-percent difference from the
        // raw (non-symmetrized) exact JVP is expected, not a bug -- this
        // check is only about ruling out AGGRESSIVE over-damping at a
        // benign state, not exact reproduction.
        let lambda = 2.0e5f32;
        let mu = 1.5e5f32;
        let dt = 0.016f32;
        let f = Mat2::from_cols(Vec2::new(1.02, 0.01), Vec2::new(-0.01, 0.98));
        let f_n = Mat2::IDENTITY;
        for &dl in &directions() {
            let via_psd = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, dt, f_n, dl);
            let df = dt * dl * f_n;
            let exact = corotated_elastic_stress_jvp(f, df, lambda, mu);
            let diff = via_psd - exact;
            let rel_err = frob(diff, diff).sqrt() / frob(exact, exact).sqrt().max(1.0);
            assert!(
                rel_err < 5.0e-2,
                "should stay close to the exact JVP at a moderate, benign state (no aggressive over-damping): dl={dl:?} rel_err={rel_err}"
            );
        }
    }

    /// At production stiffness (`basic_sand`'s E = 15 MPa DruckerPrager, lambda
    /// and mu of this order) and a benign, mostly PSD state, the projection
    /// keeps the curvature close to intact: it tracks the exact JVP's
    /// quadratic form about as closely as
    /// `matches_plain_jvp_via_chain_rule_at_moderate_already_psd_states` does
    /// at low stiffness. A Gershgorin shift large enough for diagonal
    /// dominance here swamped the curvature (every line-search step
    /// rejected over 30 frames).
    #[test]
    fn eigenvalue_clamp_preserves_curvature_at_real_production_stiffness() {
        let lambda = 6.0e7f32;
        let mu = 4.0e7f32;
        let dt = 0.016f32;
        let f = Mat2::from_cols(Vec2::new(1.02, 0.01), Vec2::new(-0.01, 0.98));
        let f_n = Mat2::IDENTITY;
        for &dl in &directions() {
            let via_psd = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, dt, f_n, dl);
            let df = dt * dl * f_n;
            let exact = corotated_elastic_stress_jvp(f, df, lambda, mu);
            let q_psd = frob(via_psd, dl);
            let q_exact = frob(exact, dl);
            assert!(
                q_psd > 0.5 * q_exact,
                "eigenvalue clamp over-damped real positive curvature at production stiffness: dl={dl:?} q_psd={q_psd} q_exact={q_exact}"
            );
        }
    }

    #[test]
    fn stays_finite_at_a_large_volumetric_expansion() {
        let lambda = 6.0e7f32;
        let mu = 4.0e7f32;
        let f = Mat2::from_cols(Vec2::new(3.0, 0.0), Vec2::new(0.0, 3.0));
        let f_n = Mat2::IDENTITY;
        let dl = Mat2::from_cols(Vec2::new(0.5, 0.2), Vec2::new(-0.3, 0.4));
        let out = corotated_kirchhoff_dtau_dl_psd(f, lambda, mu, 0.016, f_n, dl);
        assert!(
            out.x_axis.is_finite() && out.y_axis.is_finite(),
            "must stay finite even in the indefinite regime: {out:?}"
        );
    }
}
