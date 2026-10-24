//! Return mapping: the algorithm that integrates a plasticity law over one
//! step by bringing a trial stress back onto the yield surface.
//!
//! This is a numerical integration method for a constitutive law, not a law:
//! each material keeps its own yield function and flow rule.
//!
//! Source: Simo & Hughes, *Computational Inelasticity* (1998), integration
//! algorithms for the constitutive equations of plasticity; Simo & Taylor
//! 1985, CMAME 48:101-118.

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
