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

/// Floor applied to singular values before taking log — prevents ln(0).
/// All material `update_particle` implementations clamp σᵢ above this value.
pub(crate) const LOG_CLAMP: f32 = 1e-10;

/// Floor applied by legacy solid/plastic constitutive laws before using
/// `det(F)=J` in a singular expression. Strict WC-MPM liquid state does not
/// use this floor: it keeps positive `J` through its exponential continuity
/// update and reports an inadmissible state instead of clamping it.
pub(crate) const MIN_J: f32 = 1e-6;

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
/// τ = U · diag(τ_principal) · Uᵀ — the standard result for an isotropic hyperelastic
/// material: Kirchhoff/Cauchy stress is coaxial with the left stretch tensor's
/// eigenvectors (U), not V (see e.g. Bonet & Wood, "Nonlinear Continuum Mechanics for
/// Finite Element Analysis"). Distinct from `reconstruct_f`, which rebuilds F itself
/// (U on the left, Vᵀ on the right) for PLASTIC/irreversible return-mapping materials
/// (Rankine, VonMises) that permanently alter the deformation gradient — this helper
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
/// Pair with `SimConfig::earth()` and set `config.particle_mass =
/// rest_density_kg_m3 * (spacing * dx_meters).powi(2)` for a fully IRL-calibrated sim.
///
/// # Example — soft tissue (E ≈ 5 kPa, ν = 0.45, ρ = 1000 kg/m³, 1 cm/cell)
/// ```rust,no_run
/// # extern crate emerge_engine as emerge;
/// use emerge::lame_from_si;
/// let (lambda, mu) = lame_from_si(5_000.0, 0.45, 1000.0, 0.01, 0.1);
/// // lambda ≈ 1552, mu ≈ 172 — ready for NeoHookeanMaterial or ViscoelasticMaterial
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

/// Von Neumann & Richtmyer 1950 (LA-671) artificial bulk viscosity, EOS-
/// agnostic core: `q = rho*(c0*h^2*(div v)^2 - c1*h*c_sound*div v)`, gated
/// to compression (`div v < 0`) -- real shocks only form under
/// compression. `c0 = (gamma+1)/4` is the Kurapatenko 1967 weak-shock
/// coefficient; `c1 = 1.0` (Landshoff) is the standard linear term. Every
/// EOS supplies its OWN `c_sound` (its own `dp/drho` at the current state)
/// and its own real `gamma` (an ideal gas's actual adiabatic index, or a
/// Tait-EOS liquid's `eos_power` used as Kurapatenko's stand-in -- see
/// `liquid::fluid::artificial_bulk_viscosity`'s doc) -- this function owns
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

/// Convert SI gravity (m/s²) to solver units (grid cells / s²).
///
/// In the solver `v += gravity * sub_dt` where sub_dt is in real seconds,
/// so gravity must be in [cells/s²] = g_SI / dx_meters.
/// The `dt_seconds` parameter is unused but kept for API compatibility.
///
/// # Example — Earth gravity at 1 cm/cell
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
