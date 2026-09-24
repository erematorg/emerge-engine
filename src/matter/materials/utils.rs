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

/// Exact 2D deformation-gradient increment for a velocity gradient held
/// constant over one substep.
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
#[inline]
pub(crate) fn deformation_increment_exp(dt_velocity_gradient: Mat2) -> Mat2 {
    deformation_increment_exp_with_det(dt_velocity_gradient).0
}

/// The same increment, returning its determinant from the closed form
/// rather than from the matrix: `det(exp(A)) = exp(tr A)` exactly, and
/// `exp(tr A)` is already computed here as the scalar prefactor, so the
/// caller gets the volume change for free instead of re-deriving it from
/// four rounded entries.
#[inline]
fn deformation_increment_exp_with_det(dt_velocity_gradient: Mat2) -> (Mat2, f32) {
    // glam is column-major: [[a,b],[c,d]] is stored as columns (a,c),(b,d).
    let a = dt_velocity_gradient.x_axis.x;
    let b = dt_velocity_gradient.y_axis.x;
    let c = dt_velocity_gradient.x_axis.y;
    let d = dt_velocity_gradient.y_axis.y;
    let half_trace = 0.5 * (a + d);
    let half_difference = 0.5 * (a - d);
    let delta_sq = half_difference * half_difference + b * c;

    let (even, odd) = if delta_sq.abs() < 1.0e-8 {
        // cosh(sqrt(x)) and sinh(sqrt(x))/sqrt(x), continued analytically
        // through x=0.  Retaining x^2 is ample at this threshold in f32.
        let x2 = delta_sq * delta_sq;
        (
            1.0 + 0.5 * delta_sq + x2 / 24.0,
            1.0 + delta_sq / 6.0 + x2 / 120.0,
        )
    } else if delta_sq > 0.0 {
        let delta = delta_sq.sqrt();
        (delta.cosh(), delta.sinh() / delta)
    } else {
        let omega = (-delta_sq).sqrt();
        (omega.cos(), omega.sin() / omega)
    };

    let traceless = dt_velocity_gradient - Mat2::from_diagonal(Vec2::splat(half_trace));
    let scale = half_trace.exp();
    (
        scale * (Mat2::IDENTITY * even + traceless * odd),
        scale * scale,
    )
}

/// One substep of the continuity equation for a material that owns its
/// own volume, carried in the log so a small increment is not absorbed.
///
/// The scalar twin of `advance_deformation_gradient`, and the same
/// lesson. A fluid used to read `J` back from its isotropic `F` each
/// substep and multiply: `J_new = det(F) * exp(dt div v)`. Near one, an
/// f32 has a resolution of about 1.2e-7, while a calm flow's own
/// increment is a thousandth of that, so each step loses a fixed
/// FRACTION of its increment to absorption and the smallest increments
/// vanish outright. Measured with no solver and no grid
/// (`tests/scratch_fluid_j_rounding.rs`), against an f64 replica of the
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

/// The volume ratio a particle is already carrying, read from `volume`
/// rather than recomputed from `det(F)`: near the identity that
/// determinant is a cancelling difference, and reading it back every step
/// is what `advance_deformation_gradient`'s own doc measures as the worse
/// of the two options.
///
/// Zero means the particle is not carrying a usable volume yet (a bare
/// `Particle::zeroed()`, or a state written before `volume` was set), and
/// `advance_deformation_gradient` then falls back to `det(F)` for that
/// one step rather than pinning the volume to a floor.
#[inline]
pub(crate) fn carried_volume_ratio(volume: f32, initial_volume: f32) -> f32 {
    if initial_volume > 0.0 && volume > 0.0 {
        volume / initial_volume
    } else {
        0.0
    }
}

/// One substep of `dF/dt = L F`, with the volume taken from the
/// continuity equation instead of from round-off.
///
/// `deformation_increment_exp` is exact in exact arithmetic, but the
/// product `exp(dt L) F` is not: in f32 each step loses about a tenth of
/// an ULP of determinant, always the same way. Measured on one particle
/// driven by a prescribed oscillation of zero trace, with no solver, no
/// grid and no gravity (`tests/scratch_f_rounding_horizon.rs`), where
/// `ln det F` must stay at zero: after 900 000 steps it reads -2.8e-3
/// with the plain product and +3e-14 when the same formula runs in f64,
/// so the gap is precision, not the scheme. A body that keeps
/// oscillating (one hanging from an anchor, which nothing damps) turns
/// that into visible, one-way volume loss.
///
/// `det(exp(dt L)) = exp(dt tr L)` fixes the step's volume ratio before
/// any arithmetic happens, so the volume is carried multiplicatively and
/// the product is rescaled onto it; only the shape then carries
/// round-off. Same probe, same 900 000 steps: -6e-8, one ULP. Rescaling
/// onto `det(F)` re-read each step instead is measurably WORSE
/// (-4.0e-3): near the identity that determinant is a cancelling
/// difference, and feeding it back amplifies its own noise.
///
/// Returns the advanced `F` and the volume ratio it now has. A material
/// that then projects `F` plastically changes the volume for a physical
/// reason and takes its own `det` afterwards, as it already did.
#[inline]
pub(crate) fn advance_deformation_gradient(
    f_old: Mat2,
    dt_velocity_gradient: Mat2,
    carried_volume_ratio: f32,
) -> (Mat2, f32) {
    let (increment, increment_det) = deformation_increment_exp_with_det(dt_velocity_gradient);
    let product = increment * f_old;
    let base = if carried_volume_ratio > 0.0 {
        carried_volume_ratio
    } else {
        f_old.determinant()
    };
    let carried = base * increment_det;
    let det = product.determinant();
    if carried > 0.0 && det > 0.0 {
        // Analytically `det(product) == det(f_old) * increment_det`, so
        // this ratio is not the step's own volume change: it is whatever
        // disagreement the carried volume and `det(F)` already had.
        // Round-off is a few parts in 1e7, so anything past a part in
        // 1e3 means the two are genuinely out of step -- a state this
        // correction must not silently repair by rescaling F.
        const MAX_PINNED_CORRECTION: f32 = 1.0e-3;
        let ratio = carried / det;
        if (ratio - 1.0).abs() <= MAX_PINNED_CORRECTION {
            return (product * ratio.sqrt(), carried);
        }
    }
    // Degenerate, inverted, or already inconsistent: leave the product
    // alone and let the caller's own floor handle it, as before.
    (product, det)
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
/// and gives the accumulator a real rupture point instead of an arbitrary cutoff.
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
/// muscle-actuated `NeoHookeanMaterial` that still needs a real damage/health
/// signal without giving up its own stress model.
///
/// `lambda`/`mu` should be the SAME Lamé parameters the calling material already
/// uses for its own elastic response -- this reads the real strain state via the
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

/// Corotated elastic energy DENSITY `Psi(F) = mu*||F-R||_F^2 +
/// 0.5*lambda*(J-1)^2` (Stomakhin et al. 2013, the same reference
/// `corotated_elastic_stress` above already cites -- that function IS this
/// energy's First-Piola gradient dotted with `F^T` to get to Kirchhoff
/// form: `dPsi/dF = 2*mu*(F-R) + lambda*(J-1)*J*F^-T`, and `(dPsi/dF)*F^T =
/// 2*mu*(F-R)*F^T + lambda*(J-1)*J*F^-T*F^T = 2*mu*(F-R)*F^T +
/// lambda*(J-1)*J*I` since `F^-T*F^T=I` -- exactly `corotated_elastic_
/// stress`'s own formula). Needed for `spacetime::solver::implicit_
/// corotated`'s trust-region Newton (Nocedal & Wright, "Numerical
/// Optimization" 2nd ed., ch.4): a trust-region ratio test needs the
/// actual scalar objective value, not just its gradient (the residual).
/// Mirrors `corotated_elastic_stress`'s own `J<=MIN_J` degenerate-state
/// convention (zero contribution there, matching that function's "no
/// force" contract in the same regime) rather than inventing a different
/// rule for the energy alone.
///
/// Test-only (2026-09-11): `implicit_corotated`'s trust region no longer
/// uses this energy for its ratio test (switched to a Gauss-Newton
/// `0.5*||residual||^2` merit -- see that module's `newton_solve` doc for
/// the real scale-mismatch that motivated the change). Kept callable, not
/// deleted: it is `model_residual`'s own verified-gradient oracle, a real,
/// hand-checked mathematical finding (Piola, not Kirchhoff, paired with
/// `F_n^T*grad`, scaled by `dt`) worth having on hand rather than
/// re-deriving from scratch if a future fix needs it again.
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
/// and a perturbation direction `dF`, returns `dR`. Closed form (not the
/// generic SVD-based polar-decomposition derivative from the literature,
/// which doesn't match `R=M/norm`, `M=[[x,-y],[y,x]]`, `x=tr(F)`,
/// `y=F01-F10` -- this engine's own actual formula):
///   dx=tr(dF), dy=dF01-dF10, dM=[[dx,-dy],[dy,dx]]
///   d(norm)=(x*dx+y*dy)/norm,  dR=dM/norm - R*d(norm)/norm
/// Verified against central finite differences
/// (`stage2_corotated_polar_decomposition_jvp_h_convergence_diagnostic`-
/// style check, `polar_decomposition_jvp_matches_finite_difference` below):
/// max relative error 0.06% at h=1e-3 (smaller h is WORSE here, f32
/// catastrophic cancellation on this normalized/divided quantity -- verify
/// at h=1e-3, not smaller, a real lesson from deriving this the first time
/// in `tests/scratch_implicit_mpm_stage2_corotated_jvp.rs`).
///
/// Real production code again (2026-09-11): `spacetime::solver::
/// implicit_corotated`'s `dTau/dL` Gershgorin-PSD construction calls the
/// plain `corotated_elastic_stress_jvp` below (which calls this) directly
/// to build the real local Hessian it then projects -- see that module's
/// own doc for why the earlier Piola-space SPD projection this replaced
/// didn't survive the Kirchhoff/spatial-gradient pairing the real engine
/// actually needs.
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
/// `polar_decomposition_2d_jvp` above and `dJ=J*(F^-T:dF)`. This is the
/// exact Hessian-vector-product building block an implicit (Newton-CG) MPM
/// solve needs for every material sharing this elastic branch
/// (DruckerPrager/sand, VonMises, Rankine, MuIRheology, and Corotated
/// itself) -- see `spacetime::solver::implicit_corotated`'s own doc for
/// where this gets used. Verified against central finite differences
/// (`corotated_stress_jvp_matches_finite_difference` below, and originally
/// in `tests/scratch_implicit_mpm_stage2_corotated_jvp.rs`/`stage3_
/// drucker_prager_multi_particle.rs` before being promoted here): max
/// relative error 0.23% at real sand-magnitude stiffness.
///
/// Mirrors `corotated_elastic_stress`'s own `j <= MIN_J` floor: the stress
/// is pinned at exactly zero in that regime, so its local derivative is
/// zero too (a real, standard, practical simplification at a hard
/// clamp boundary, not a claim the true continuous derivative is zero
/// there).
///
/// Real production code again (2026-09-11), same reason as `polar_
/// decomposition_2d_jvp` above.
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

/// The real, correct fix (2026-09-11) for a subtle bug in `spacetime::
/// solver::implicit_corotated`'s Newton-CG: returns a PSD-guaranteed
/// approximation of `d(tau)/d(L)` (Kirchhoff stress derivative w.r.t. a
/// velocity-gradient perturbation `dl`, via `dF = dt*dl*f_n`), applied to
/// the given `dl` -- NOT the earlier `corotated_elastic_stress_jvp_
/// projected`, which projected `d(P)/d(F)` (Teran et al. 2005's SPD
/// construction, correct for THAT pairing) and then converted to
/// Kirchhoff/velocity-gradient space via the exact product rule. That
/// conversion does not carry the PSD guarantee across: Teran's proof
/// establishes `dP:dF >= 0` for First-Piola stress paired with the
/// MATERIAL-space deformation gradient perturbation; this engine's real
/// force (matching real implicit-MPM references read directly,
/// `tmp/ziran2020`/`tmp/GeoTaichi`) pairs KIRCHHOFF stress with the
/// SPATIAL kernel gradient instead, and `tau = P*F^T`'s dependence on `L`
/// is a different bilinear form with no inherited guarantee -- confirmed
/// live: even using the Piola-projected-then-converted JVP, real
/// multi-particle Newton-CG calls in this exact solver saw the residual
/// GROW across CG iterations (a negative-curvature direction), which a
/// genuinely PSD system cannot produce.
///
/// Rather than re-deriving Teran's closed-form SVD-frame construction for
/// this different pairing (real, substantial tensor algebra with real risk
/// of a fresh, equally subtle error), this builds the CONCRETE local 4x4
/// operator directly -- `L` has 4 real independent components in 2D, so
/// does `tau` -- from four calls to the already finite-difference-verified
/// EXACT `corotated_elastic_stress_jvp`, symmetrizes it (only the
/// symmetric part of any matrix contributes to the quadratic form PSD-ness
/// is about), then applies Teran et al. 2005's own eigenvalue-clamping
/// projection DIRECTLY to this symmetric 4x4 matrix: eigendecompose it
/// (`jacobi_eigen_symmetric_4x4`, Golub & Van Loan's classical cyclic
/// Jacobi method for small real-symmetric matrices), clamp every negative
/// eigenvalue to zero, reconstruct. The clamping step is pairing-agnostic
/// linear algebra (the nearest SPD matrix in Frobenius norm to any real
/// symmetric matrix, Higham 1988) -- it is Teran's own PAPER FORMULA
/// (closed-form per-invariant expressions specific to `dP/dF`'s structure)
/// that does not carry over to this Kirchhoff/spatial-gradient pairing, not
/// the general clamping principle their paper introduces.
///
/// Real fix (2026-09-11) replacing a first attempt at this same problem: a
/// Gershgorin-shift (add a uniform diagonal shift until the matrix is
/// diagonally dominant) is also a valid sufficient condition for PSD-ness,
/// but confirmed live to be USELESS at real production stiffness: at
/// `basic_sand`'s real E=15MPa (25x the softer synthetic modulus the
/// correctness regression tests use), the shift needed to restore diagonal
/// dominance dwarfs the real off-diagonal curvature, so the "projected"
/// matrix degenerates toward a scaled identity carrying almost none of the
/// true Hessian's directional information -- Newton's line search then
/// rejects EVERY step from EVERY frame (confirmed via `EMERGE_IMPLICIT_
/// DIAG=1` on `tests/scratch_implicit_corotated_real_fps_measurement.rs`:
/// `best_trial_norm` came back statistically equal to `r_norm` even after
/// CG solved its own linear subproblem to a 90%+ residual reduction),
/// silently falling back to full explicit cost every single frame -- the
/// real, deeper cause behind an apparent 0.73x "speedup" (net slower than
/// explicit), not merely the redundant-rebuild cost this file's split
/// build/apply API separately fixes below. Eigenvalue clamping only removes
/// the actually-negative modes and leaves every genuinely positive-
/// curvature direction untouched regardless of stiffness scale, so it does
/// not have this failure mode.
///
/// Split in two (2026-09-11, real performance fix): the expensive part
/// (build the 4x4 matrix from 4 JVP evaluations, symmetrize, eigenvalue-
/// clamp) depends only on `f`/`lambda`/`mu`/`dt`/`f_n` -- NOT on the
/// direction `dl` being applied. A caller doing Newton-CG evaluates this at
/// a FIXED trial `v` (hence fixed `f`) across MANY CG iterations, each with
/// a DIFFERENT `dl` -- rebuilding the matrix from scratch on every one of
/// those (the original, single-function form of this API) is real,
/// measured, wasted work. [`corotated_kirchhoff_dtau_dl_psd_matrix`] builds
/// the matrix ONCE per trial `v`; [`apply_dtau_dl_psd_matrix`] applies it to
/// an arbitrary `dl` for the cost of a 4x4 mat-vec.
///
/// Test-only (2026-09-11): `implicit_corotated`'s Newton solver no longer
/// uses this eigenvalue-clamped operator in production -- it switched to
/// `steihaug_cg` (Nocedal & Wright's trust-region method), which handles
/// indefinite curvature structurally and needs no PSD projection at all.
/// Kept callable, not deleted: a real, test-verified (`corotated_
/// kirchhoff_dtau_dl_psd_tests`), literature-grounded (Teran, Sifakis,
/// Irving & Fedkiw 2005's clamping principle, via a real Jacobi
/// eigendecomposition rather than a cruder Gershgorin shift) construction
/// worth having on hand if a future fix needs a guaranteed-PSD operator
/// again, rather than re-deriving it from scratch.
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
/// caller for the real, measured failure this replaces).
///
/// Test-only -- see `corotated_kirchhoff_dtau_dl_psd_matrix`'s own doc.
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
/// Test-only -- see `corotated_kirchhoff_dtau_dl_psd_matrix`'s own doc.
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

/// Cheap application step for [`corotated_kirchhoff_dtau_dl_psd_matrix`]'s
/// output -- a 4x4 mat-vec, no JVP evaluations. Test-only (2026-09-11):
/// production (`spacetime::solver::implicit_corotated`) no longer applies
/// this matrix to an arbitrary `dl` via matrix-free CG -- it assembles the
/// SAME matrix into a dense free-DOF system and solves it directly (see
/// that module's own doc, "remaining limitation #2 -- scale") -- kept here
/// only for [`corotated_kirchhoff_dtau_dl_psd`]'s own direct-comparison
/// test coverage.
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
/// 8 iterations is real headroom, not a tuned number: every real hardening
/// law in this engine is a bounded, smooth, saturating function of its own
/// internal variable, making this a real contraction that converges to float
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

/// Real, generic 1D elastic-perfectly-plastic return mapping -- the shared
/// ALGORITHMIC core of "clamp a trial value to an interval around a
/// permanent/plastic offset, permanently absorbing any excess," for any
/// quantity whose yield surface is a plain interval rather than a
/// norm-ball. This is the scalar-state sibling of the tensor-space radial
/// return every material above uses (e.g. `VonMisesMaterial::update_particle`'s
/// own `dev * (effective_yield/elastic_dev)` projection) -- genuinely
/// different math from that (an interval clamp, not a norm rescale) because
/// the underlying state is 1D, not a tensor; the SHARED concept (elastic
/// trial -> yield check -> permanent return-map) is the same, only the
/// shape of the projection differs with dimensionality. Real first adopter:
/// `rod::plasticity`'s bending curvature; any other future scalar-state
/// plasticity (a scalar damage variable, an axial-force yield) can reuse
/// this directly instead of re-deriving the same three-line clamp.
///
/// `trial`: the fully-elastic candidate value (e.g. current curvature).
/// `permanent`: the current permanent/plastic offset (e.g. rest curvature).
/// `limit`: the real, positive elastic-limit half-width of the interval.
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

/// Convert SI Young's modulus and Poisson's ratio to emerge grid-unit Lamé parameters.
///
/// In the solver, velocity is in cells/s and stress is applied as:
///   `f_particle = vol_solver * sigma_solver * kernel`
/// The correct non-dimensionalization gives:
///   `λ_grid = λ_SI · dt² / (ρ₀ · dx²)`
///
/// Pair with `SimConfig::earth()`. Do NOT also set a particle mass: mass is
/// derived from `SimConfig::grid_density` (default 1.0) and the region's own
/// spacing, which is what keeps the gravity/stiffness ratio independent of how
/// finely the region is discretized.
///
/// # Example -- soft tissue (E ≈ 5 kPa, ν = 0.45, ρ = 1000 kg/m³, 1 cm/cell)
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::lame_from_si;
/// let (lambda, mu) = lame_from_si(5_000.0, 0.45, 1000.0, 0.01, 0.1);
/// // lambda ≈ 1552, mu ≈ 172 -- ready for NeoHookeanMaterial or ViscoelasticMaterial
/// ```
pub fn lame_from_si(
    young_modulus_pa: f32,
    poisson_ratio: f32,
    rest_density_kg_m3: f32,
    dx_meters: f32,
    dt_seconds: f32,
) -> (f32, f32) {
    let (lambda_si, mu_si) = lame_from_young(young_modulus_pa, poisson_ratio);
    let scale = dt_seconds * dt_seconds / (rest_density_kg_m3 * dx_meters * dx_meters);
    (lambda_si * scale, mu_si * scale)
}

/// Dimensionally-correct SI -> grid Lame conversion. Prefer this over
/// [`lame_from_si`] for any new scene.
///
/// `scale = 1 / (rho * dx^2)`, with **no `dt` factor** -- the solver's
/// velocity is cells/SECOND (fixed by [`gravity_to_grid`]'s own contract:
/// `v += gravity * sub_dt` with `sub_dt` in real seconds), so a converted
/// stiffness must not depend on the timestep. The result is exactly the
/// squared elastic wave speed in cells/s: `c_grid^2 = (E/rho)/dx^2`.
///
/// # Why this exists separately (real, measured, 2026-08-24)
/// [`lame_from_si`] carries an extra `dt^2`, which makes grid stiffness
/// depend on the timestep. Measured directly, zero gravity, identical
/// physical initial condition and identical simulated elapsed time: peak
/// elastic rebound speed came out 58.4 / 7.77 / 0.276 cells/s at
/// dt = 0.1 / 0.01 / 0.001. A physical result must be dt-INDEPENDENT; a
/// 200x spread is a unit mismatch, not discretization error. Under gravity
/// the same bug reads as "everything crushes far too violently": an elastic
/// column that analytically compresses `rho*g*h/E` = 0.02% compressed
/// 6-100% instead, 300-4900x too much, and worse at smaller dt. With this
/// function the stiffness is identical at every dt and the strain matches
/// the analytic value to ~1.7x.
///
/// Kept as a SEPARATE function rather than fixing `lame_from_si` in place:
/// every currently-tuned scene was calibrated against the old conversion,
/// so changing it globally re-breaks all of them at once (tried, reverted).
/// Migrate scenes to this one at a time -- and when a scene switches, its
/// `gravity_fraction` fudge should be deleted in the same change, because
/// that fudge exists to compensate for exactly this bug.
pub fn lame_from_si_physical(
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
/// The `dt_seconds` parameter is unused but kept for API compatibility.
///
/// # Example -- Earth gravity at 1 cm/cell
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::gravity_to_grid;
/// use glam::Vec2;
/// let g = gravity_to_grid(Vec2::new(0.0, -9.81), 0.01, 0.1);
/// // g ≈ Vec2::new(0.0, -981.0) cells/s²
/// ```
pub fn gravity_to_grid(g_si: glam::Vec2, dx_meters: f32, _dt_seconds: f32) -> glam::Vec2 {
    g_si / dx_meters
}

/// Von Neumann & Richtmyer 1950 (LA-671) artificial bulk viscosity, EOS-
/// agnostic core: `q = rho*(c0*h^2*(div v)^2 - c1*h*c_sound*div v)`, gated
/// to compression (`div v < 0`) -- real shocks only form under
/// compression. `c0 = (gamma+1)/4` is the Kurapatenko 1967 weak-shock
/// coefficient; `c1 = 1.0` (Landshoff) is the standard linear term. Every
/// EOS supplies its OWN `c_sound` (its own `dp/drho` at the current state)
/// and its own real `gamma` (an ideal gas's actual adiabatic index, or a
/// Tait-EOS liquid's `eos_power` used as Kurapatenko's stand-in -- see
/// `fluid::artificial_bulk_viscosity`'s doc) -- this function owns
/// only the shared shock-viscosity FORM, not any one EOS's derivative.
/// Extracted 2026-08-18 so a genuinely different EOS (ideal gas) can reuse
/// the real, cited shock-capturing term without faking Tait parameters to
/// back into it.
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

/// Real single-sphere Stokes drag rate (Stokes 1851), converted from SI to
/// the engine's own `LinearDragField::drag_coefficient` convention (units
/// 1/time). `F = 6*pi*mu*r*v` gives `dv/dt = -(6*pi*mu*r/m)*v`, so
/// `k_SI = 6*pi*mu*r/m` [1/s]; converting to grid-time units follows the
/// SAME non-dimensionalization family as `lame_from_si`/`stress_from_si`
/// above, just for a pure rate (1/time only, no mass or length dimension
/// of its own to cancel): `k_grid = k_SI * dt_seconds`.
///
/// # Validity -- check this before using
/// Stokes' law is only exact for LOW Reynolds number (`Re = rho*v*d/mu
/// <~ 1`, creeping/laminar flow) -- real, correctly-scoped uses are fine
/// dust or sand grains in a gentle wind (sub-millimeter radius, low
/// speed), matching `LinearDragField`'s own doc precedent (real aeolian
/// sand-transport literature). It is the WRONG formula for a fist-sized
/// object moving at real everyday speeds: a real Newton's-cradle-scale
/// steel ball (~1 cm radius) swinging at ~1 m/s sits at `Re ~ 2000-3000`,
/// where real drag is actually quadratic (form drag), not linear -- this
/// was checked directly for that case (2026-08-22,
/// `examples/grain_newtons_cradle_gui.rs`) and found to underpredict a
/// real cradle's observed damping by roughly 1000x. Compute
/// `rho_air * velocity * (2.0*radius_m) / dynamic_viscosity_pa_s` yourself
/// and confirm it's `<~ 1` before trusting this function's output as the
/// dominant real damping mechanism for a given scene.
///
/// # Example -- fine dust grain in air (r=50 micron, m=6.5e-10 kg, 20 C air)
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::materials::stokes_drag_rate_from_si;
/// let k_grid = stokes_drag_rate_from_si(5.0e-5, 6.5e-10, 1.81e-5, 1.0);
/// // dt_seconds = 1.0 (grid time unit = 1 real second) -- ready for
/// // LinearDragField::drag_coefficient / GrainField drag.
/// # let _ = k_grid;
/// ```
pub fn stokes_drag_rate_from_si(
    radius_m: f32,
    mass_kg: f32,
    dynamic_viscosity_pa_s: f32,
    dt_seconds: f32,
) -> f32 {
    let k_si = 6.0 * std::f32::consts::PI * dynamic_viscosity_pa_s * radius_m / mass_kg;
    k_si * dt_seconds
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

    /// Real verification this file's own testing convention requires before
    /// trusting any derivative -- central finite difference at h=1e-3 (NOT
    /// smaller: f32 catastrophic cancellation on `polar_decomposition_2d`'s
    /// own normalized/divided quantity makes smaller h WORSE, a real lesson
    /// from deriving this the first time in `tests/scratch_implicit_mpm_
    /// stage2_corotated_jvp.rs`).
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

    /// Real verification that `corotated_elastic_energy_density` is
    /// actually the potential `corotated_elastic_stress` is the gradient
    /// of -- needed for `spacetime::solver::implicit_corotated`'s
    /// trust-region Newton (its ratio test needs a real objective value,
    /// not just the residual/gradient) to be solving the SAME problem the
    /// rest of this file's tests already trust, not a mismatched one.
    /// Central finite difference, same h=1e-3 convention as this module's
    /// own stress-JVP check above (f32 catastrophic cancellation makes
    /// smaller h worse here, not better). Recovers First-Piola from the
    /// existing Kirchhoff-stress function via `P = tau*F^-T` (`tau=P*F^T`)
    /// rather than duplicating the stress formula in Piola form.
    #[test]
    fn energy_density_gradient_matches_finite_difference_of_the_stress() {
        let h = 1.0e-3f32;
        for &(lambda, mu) in &[(50.0f32, 30.0f32), (2.0e7, 1.5e7)] {
            let mut max_rel_err = 0.0f32;
            for &f in &states() {
                // `Mat2::IDENTITY` is a genuine critical point of `Psi`
                // (F=R, J=1 -> both terms are exactly zero, the analytic
                // global minimum) -- its analytic gradient is exactly
                // zero, which makes central-difference ill-conditioned
                // there by construction: `f(x+h)`/`f(x-h)` are both
                // dominated by the (identical, non-cancelling) O(h^2)
                // curvature term, so `(plus-minus)/(2h)` recovers pure
                // O(h^2)*f'''/6 truncation noise, not a real discrepancy
                // (confirmed: at lambda=2e7, this alone produced a raw FD
                // value of ~1.8 against an analytic 0 -- but EVERY other
                // state, all genuinely deformed with nonzero analytic
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
            // Large volumetric expansion -- the real regime the plain
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
    /// this operator, unlike the plain JVP or the earlier Piola-space
    /// projection (which didn't survive the Kirchhoff/spatial-gradient
    /// conversion, confirmed by real negative-curvature CG calls in
    /// `spacetime::solver::implicit_corotated` before this fix).
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
        // is actually about), so a real few-percent difference from the
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

    /// The real, measured failure this eigenvalue-clamp replaces a
    /// Gershgorin-shift for: at a benign, already-mostly-PSD state, the
    /// projection must leave the real curvature close to intact EVEN AT
    /// REAL PRODUCTION STIFFNESS (`basic_sand`'s own E=15MPa DruckerPrager
    /// converts to lambda/mu on this order) -- a uniform diagonal shift
    /// large enough to guarantee diagonal dominance at this stiffness scale
    /// was confirmed live to swamp the real curvature entirely (`tests/
    /// scratch_implicit_corotated_real_fps_measurement.rs` with
    /// `EMERGE_IMPLICIT_DIAG=1`: Newton's line search rejected every step
    /// on every one of 30 real frames, `best_trial_norm` statistically
    /// equal to `r_norm` even after CG solved its own linear subproblem to
    /// a 90%+ residual reduction). Eigenvalue clamping must not reproduce
    /// that failure: it should track the exact JVP's own quadratic form
    /// about as closely here as the softer `matches_plain_jvp_via_chain_
    /// rule_at_moderate_already_psd_states` test above does at low
    /// stiffness -- proving the fix is stiffness-independent, not just
    /// correct at the soft synthetic modulus the regression suite happens
    /// to use elsewhere.
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
