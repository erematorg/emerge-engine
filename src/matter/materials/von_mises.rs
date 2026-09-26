use glam::{Mat2, Vec2};

use crate::materials::physical_props::{DuctileProps, FromSI, scale_lame, scale_stress};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    LOG_CLAMP, MIN_J, advance_deformation_gradient, carried_volume_ratio, corotated_elastic_stress,
    elastic_wave_dt, hencky_strains, lame_from_young, reconstruct_f,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{ParticleUpdateCtx, Particles};

/// Von Mises elastoplastic material: J2 plasticity with optional linear isotropic hardening.
///
/// Elastic response: corotated linear elastic (τ = 2µ(F−R)Fᵀ + λ(J−1)J·I).
/// Yield criterion: 2µ·|dev(ε)| ≤ σ_Y(κ), where ε is the Hencky (log) strain and
///   σ_Y(κ) = yield_stress + hardening_modulus·κ  (κ = accumulated equivalent plastic strain).
/// Return mapping: scale deviatoric log-strain back to the yield surface; volumetric
/// strain is preserved exactly (incompressible plastic flow assumption).
///
/// `hardening_modulus = 0.0` → perfect plasticity (original behaviour, backward compatible).
/// `hardening_modulus > 0` → linear isotropic hardening (metals, biological tissue stiffening).
///
/// Suitable for: lava flows, ductile metals, clay, soft rock under shear, biological tissue.
///
/// GPU note: `yield_stress` is stored in `hardening_exponent` in `MaterialParams` (union layout).
/// `hardening_modulus` is stored in `MaterialParams::hardening_modulus`.
/// `κ` is accumulated into `Particle::friction_hardening` each substep.
#[derive(Debug, Clone, Copy)]
pub struct VonMisesMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Initial yield stress σ_Y₀ in simulation stress units (same scale as λ/µ).
    /// Flow begins when 2µ|dev(ε)| > σ_Y₀ + H·κ.
    pub yield_stress: f32,
    /// Linear isotropic hardening modulus H.
    /// σ_Y(κ) = yield_stress + H·κ. Set 0.0 for perfect plasticity (default).
    pub hardening_modulus: f32,
    /// Real Kelvin-Voigt viscous damping on the deviatoric elastic strain
    /// rate (SI Pa.s, converted with the SAME convention `lambda`/`mu` used
    /// -- see `rankine::q_factor_elastic_viscosity_pa_s`'s own doc for the
    /// pairing rule and the real regression it documents) -- same
    /// mechanism, same formula, as
    /// `RankineMaterial::elastic_viscosity` / `DruckerPragerMaterial::elastic_viscosity`.
    /// Zero cost, zero behavior change at `0.0` (default, matching every
    /// other material using this same mechanism).
    ///
    /// Only damps the ELASTIC response below yield -- real solids are never
    /// purely elastic even before plastic flow begins (internal friction
    /// measurably dissipates energy in every real material, reported as a
    /// seismic/ultrasonic quality factor Q -- see
    /// `q_factor_elastic_viscosity_pa_s`). Confirmed live 2026-08-29: this
    /// is a real, structural gap -- `VonMisesMaterial` had NO damping
    /// mechanism of any kind before this field existed (found while
    /// root-causing sustained post-impact bouncing on `RankineMaterial::ice()`,
    /// same class of missing dissipation, different material).
    pub elastic_viscosity: f32,
}

impl VonMisesMaterial {
    /// Construct from Young's modulus E, Poisson's ratio ν, and yield stress σ_Y.
    ///
    /// Typical values for lava/clay: E = 5e4–1e5, ν = 0.3–0.4, σ_Y = 1e2–1e3.
    /// **Grid units, NOT real Pascals** (real disclosure added 2026-09-05,
    /// same finding as `NeoHookeanMaterial::from_young_modulus`'s own doc):
    /// calls [`lame_from_young`] directly, never touches `dx_meters`/
    /// density. For a real, correctly SI-to-grid-converted material, build
    /// an [`Elastoplastic`](crate::materials::Elastoplastic) with
    /// `model: PlasticityModel::Ductile { yield_stress_pa }` and call its
    /// `.material(&config)` (real dispatch, see that method's own doc) --
    /// `Self::from_physical` exists but its `DuctileProps` input type is
    /// crate-internal, not constructible from outside.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32, yield_stress: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, yield_stress)
    }

    /// Perfect plasticity (no hardening).
    pub const fn new(lambda: f32, mu: f32, yield_stress: f32) -> Self {
        Self {
            lambda,
            mu,
            yield_stress,
            hardening_modulus: 0.0,
            elastic_viscosity: 0.0,
        }
    }

    /// Linear isotropic hardening. `hardening_modulus` > 0 makes the material stiffen
    /// as it deforms plastically -- yield stress grows as `yield_stress + H·κ`.
    pub fn with_hardening(lambda: f32, mu: f32, yield_stress: f32, hardening_modulus: f32) -> Self {
        assert!(hardening_modulus >= 0.0, "hardening_modulus must be ≥ 0");
        Self {
            lambda,
            mu,
            yield_stress,
            hardening_modulus,
            elastic_viscosity: 0.0,
        }
    }

    /// Soft ductile: yield_stress = E/200. Low yield-to-stiffness ratio. Remoulded clay regime.
    pub fn soft_ductile(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::from_young_modulus(young_modulus, poisson_ratio, young_modulus * 0.005)
    }

    /// The yield surface after `kappa` of accumulated equivalent plastic
    /// strain: `yield_stress + hardening_modulus * kappa`.
    pub fn yield_surface(&self, kappa: f32) -> f32 {
        self.yield_stress + self.hardening_modulus * kappa
    }

    /// What the yield criterion tests, for stretches `sigma` (the singular
    /// values of F): `2 mu |dev(eps)|`, eps the Hencky strain, returned with
    /// the deviator itself and the trace the return mapping needs.
    fn deviatoric_state(&self, sigma: Vec2) -> (Vec2, f32, f32) {
        let eps = hencky_strains(sigma);
        let tr = eps.x + eps.y;
        let dev = eps - Vec2::splat(tr * 0.5);
        (dev, tr, 2.0 * self.mu * dev.length())
    }

    /// Where particle `i` sits against its own yield surface: the quantity
    /// the criterion tests over `yield_surface` of its own `kappa`, both
    /// computed by the code the return mapping runs. Below 1 the particle is
    /// elastic; a particle flowing plastically has been returned onto its
    /// surface and reads 1. How far it has hardened is `kappa` itself, in
    /// `friction_hardening`.
    pub fn yield_ratio(&self, particles: &Particles, i: usize) -> f32 {
        self.yield_ratio_of(
            particles.deformation_gradient[i],
            particles.friction_hardening[i],
        )
    }

    /// `yield_ratio` for one particle's deformation gradient and `kappa`,
    /// for callers that hold a `Particle` rather than the store, such as a
    /// `DiagnosticsPlugin`.
    pub fn yield_ratio_of(&self, deformation_gradient: Mat2, kappa: f32) -> f32 {
        let (_, sigma, _) = svd2(deformation_gradient);
        let (_, _, measure) = self.deviatoric_state(sigma);
        measure / self.yield_surface(kappa).max(f32::MIN_POSITIVE)
    }
}

impl FromSI<DuctileProps> for VonMisesMaterial {
    fn from_physical(props: &DuctileProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        let yield_stress = scale_stress(props.yield_stress_pa, props.elastic.rho_kg_m3, config);
        Self::new(lambda, mu, yield_stress)
    }
}

impl MaterialModel for VonMisesMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::VonMises
    }

    fn corotated_lame_params(&self) -> Option<(f32, f32)> {
        if self.elastic_viscosity == 0.0 {
            Some((self.lambda, self.mu))
        } else {
            None
        }
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let elastic =
            corotated_elastic_stress(particles.deformation_gradient[i], self.lambda, self.mu);
        if self.elastic_viscosity == 0.0 {
            return elastic;
        }
        // Same Kelvin-Voigt dashpot formula as `RankineMaterial`/
        // `DruckerPragerMaterial`/`ViscoelasticMaterial`: tau_v = eta*D_dev
        // (NOT 2*eta*D_dev -- see `RankineMaterial::kirchhoff_stress`'s own
        // doc for why), D the symmetric part of the APIC velocity gradient.
        let c = particles.velocity_gradient[i];
        let sym = c + c.transpose();
        let d = sym * 0.5;
        let trace = d.x_axis.x + d.y_axis.y;
        let d_dev = d - Mat2::from_diagonal(Vec2::splat(trace * 0.5));
        elastic + self.elastic_viscosity * d_dev
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // Controlled A/B experiment (external review, 2026-09-03): swapped
        // from forward-Euler `(I+dt*C)*F` to the exact matrix exponential
        // `exp(dt*C)*F` (see `deformation_increment_exp`'s own doc for the
        // real O(dt^2) volumetric-ratchet mechanism this removes). Nothing
        // else in this function changed -- the plastic return-mapping below
        // still operates on whatever F_trial it's handed, so this isolates
        // the kinematic integration as the ONLY variable, matching the
        // basic_vonmises.rs live-drift investigation this is testing against.
        let (f_trial, _) = advance_deformation_gradient(
            *ctx.deformation_gradient,
            dt * *ctx.velocity_gradient,
            carried_volume_ratio(*ctx.volume, ctx.initial_volume),
        );
        let (u, sigma, vt) = svd2(f_trial);

        let (dev, tr, elastic_dev) = self.deviatoric_state(sigma);
        let dev_norm = dev.length();

        let kappa = *ctx.friction_hardening;
        let effective_yield = self.yield_surface(kappa);

        let sigma_new = if elastic_dev > effective_yield && dev_norm > LOG_CLAMP {
            let denom = 2.0 * self.mu + self.hardening_modulus;
            let gamma = if denom > f32::EPSILON {
                (elastic_dev - effective_yield) / denom
            } else {
                0.0
            };
            *ctx.friction_hardening = kappa + gamma;
            // Real, disclosed regression fix (2026-09-02, external review):
            // the radial-return consistency condition requires projecting
            // onto the UPDATED yield surface (after this step's own
            // hardening increment), not the trial-state limit computed
            // BEFORE it -- Simo & Taylor's own associative J2 return
            // mapping. The old code divided by `effective_yield` (pre-
            // hardening), which silently reproduces the OLD surface every
            // single step: worked counterexample (mu=3000, yield_stress=100,
            // hardening_modulus=500, elastic_dev=150) gives gamma=0.007692,
            // real post-hardening limit=103.846, but the old code returned
            // exactly 100. `new_effective_yield = effective_yield +
            // hardening_modulus*gamma` is the exact, real value the
            // projected stress must land on.
            let new_effective_yield = effective_yield + self.hardening_modulus * gamma;
            let eps_proj = dev * (new_effective_yield / elastic_dev) + Vec2::splat(tr * 0.5);
            Vec2::new(eps_proj.x.exp(), eps_proj.y.exp())
        } else {
            sigma
        };

        *ctx.deformation_gradient = reconstruct_f(u, sigma_new, vt);
        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::VonMises as u32,
            lambda: self.lambda,
            mu: self.mu,
            hardening_exponent: self.yield_stress,
            hardening_modulus: self.hardening_modulus,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let elastic_dt = elastic_wave_dt(
            self.lambda,
            self.mu,
            1.0,
            density,
            MIN_J,
            cell_width,
            material_cfl,
        );
        // Same explicit-viscous-diffusion stability bound `RankineMaterial`/
        // `DruckerPragerMaterial` already use for their own Kelvin-Voigt term --
        // without this, `elastic_viscosity` adds real stiffness the substep
        // selector never sees.
        let viscous_dt = if self.elastic_viscosity > 0.0 {
            let density = density.max(1.0e-6);
            let kinematic = self.elastic_viscosity / density;
            if kinematic > f32::EPSILON {
                viscous_cfl * cell_width * cell_width / kinematic
            } else {
                f32::INFINITY
            }
        } else {
            f32::INFINITY
        };
        elastic_dt.min(viscous_dt)
    }
}

#[cfg(test)]
mod marginal_yield_tests {
    use super::*;
    use crate::Particle;

    /// Isolates whether `update_particle`'s return mapping matches the material's
    /// OWN documented yield criterion (`2*mu*|dev(eps)| <= yield_stress`) exactly,
    /// bypassing MPM's grid/transfer pipeline entirely -- same pattern as
    /// `sand.rs::marginal_yield_tests` for verifying a plasticity return-mapping
    /// against its own analytical yield surface.
    fn run_one_step(mat: &VonMisesMaterial, sigma: Vec2, kappa: f32) -> (Vec2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::from_cols(Vec2::new(sigma.x, 0.0), Vec2::new(0.0, sigma.y));
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = kappa;
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles.update_ctx(0), 1.0);
        let f = particles.deformation_gradient[0];
        (
            Vec2::new(f.x_axis.x, f.y_axis.y),
            particles.friction_hardening[0],
        )
    }

    fn run_rate_step(
        mat: &VonMisesMaterial,
        f: Mat2,
        velocity_gradient: Mat2,
        dt: f32,
        kappa: f32,
    ) -> (Mat2, f32) {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.velocity_gradient = velocity_gradient;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.friction_hardening = kappa;
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles.update_ctx(0), dt);
        (
            particles.deformation_gradient[0],
            particles.friction_hardening[0],
        )
    }

    fn elastic_deviatoric_stress(mat: &VonMisesMaterial, f: Mat2) -> f32 {
        let (_, sigma, _) = svd2(f);
        let eps = hencky_strains(sigma);
        let dev = eps - Vec2::splat((eps.x + eps.y) * 0.5);
        2.0 * mat.mu * dev.length()
    }

    /// Maps a TARGET `dev_norm` (the code's own L2-norm convention,
    /// `dev.length()` for `dev=(eps1-tr/2, eps2-tr/2)`) to the per-component
    /// magnitude `d` needed so that constructing `eps=(tr/2+d, tr/2-d)`
    /// actually produces that `dev_norm` exactly: `dev=(d,-d)`, so
    /// `dev.length() = d*sqrt(2)`, i.e. `d = dev_norm/sqrt(2)`.
    fn per_component_d_for_target_dev_norm(target_dev_norm: f32) -> f32 {
        target_dev_norm / std::f32::consts::SQRT_2
    }

    /// `yield_ratio` reads a state against its own surface: 0.6 for a state
    /// at 60 percent of the threshold, and 1 once a state beyond it has
    /// been returned, onto the surface its own hardening moved.
    #[test]
    fn yield_ratio_is_below_one_inside_and_one_after_a_return() {
        let mat = VonMisesMaterial::with_hardening(2000.0, 3000.0, 100.0, 500.0);
        let at = |fraction: f32| {
            let d =
                per_component_d_for_target_dev_norm(fraction * mat.yield_stress / (2.0 * mat.mu));
            Vec2::new(d.exp(), (-d).exp())
        };
        let read = |sigma: Vec2, kappa: f32| {
            let mut p = Particle::zeroed();
            p.deformation_gradient = Mat2::from_diagonal(sigma);
            p.friction_hardening = kappa;
            p.mass = 1.0;
            p.initial_volume = 1.0;
            mat.yield_ratio(&Particles::from(vec![p]), 0)
        };
        let inside = read(at(0.6), 0.0);
        assert!((inside - 0.6).abs() < 1.0e-4, "inside reads {inside}");

        let (sigma_after, kappa_after) = run_one_step(&mat, at(1.5), 0.0);
        assert!(kappa_after > 0.0, "a state beyond yield must harden");
        let returned = read(sigma_after, kappa_after);
        assert!(
            (returned - 1.0).abs() < 1.0e-3,
            "returned state reads {returned}"
        );
    }

    #[test]
    fn marginal_state_at_yield_stress_does_not_yield() {
        let mat = VonMisesMaterial::new(2000.0, 3000.0, 100.0);
        // dev_norm exactly AT the yield threshold: 2*mu*dev_norm = yield_stress
        // => dev_norm = yield_stress / (2*mu). Comfortably inside (99% of it).
        let target_dev_norm = 0.99 * mat.yield_stress / (2.0 * mat.mu);
        let d = per_component_d_for_target_dev_norm(target_dev_norm);
        let trace = 0.0; // pure deviatoric, no volumetric strain
        let eps1 = trace * 0.5 + d;
        let eps2 = trace * 0.5 - d;
        let sigma = Vec2::new(eps1.exp(), eps2.exp());

        let (sigma_after, kappa_after) = run_one_step(&mat, sigma, 0.0);
        assert!(
            (sigma_after - sigma).length() < 1.0e-4,
            "state inside the yield surface should stay elastic (no change): \
             sigma={sigma:?} sigma_after={sigma_after:?}"
        );
        assert_eq!(
            kappa_after, 0.0,
            "kappa must not accumulate on an elastic step"
        );
    }

    #[test]
    fn marginal_state_beyond_yield_stress_projects_exactly_to_the_yield_surface() {
        let mat = VonMisesMaterial::new(2000.0, 3000.0, 100.0);
        // Comfortably OUTSIDE: 150% of the yield threshold.
        let target_dev_norm = 1.5 * mat.yield_stress / (2.0 * mat.mu);
        let d = per_component_d_for_target_dev_norm(target_dev_norm);
        let trace = 0.4; // nonzero volumetric strain -- must be preserved exactly
        let eps1 = trace * 0.5 + d;
        let eps2 = trace * 0.5 - d;
        let sigma = Vec2::new(eps1.exp(), eps2.exp());

        let (sigma_after, kappa_after) = run_one_step(&mat, sigma, 0.0);

        // Real, exact analytical claim: the projected state's dev_norm must equal
        // EXACTLY yield_stress/(2*mu) (perfect plasticity, no hardening here) --
        // not just "less than before." Computed the SAME way the material's own
        // code does (L2 norm of the deviatoric vector), not a per-component value.
        let eps_after = crate::materials::utils::hencky_strains(sigma_after);
        let tr_after = eps_after.x + eps_after.y;
        let dev_after_vec = eps_after - Vec2::splat(tr_after * 0.5);
        let dev_after = dev_after_vec.length();
        let expected_dev = mat.yield_stress / (2.0 * mat.mu);
        assert!(
            (dev_after - expected_dev).abs() < 1.0e-4,
            "projected deviatoric strain should land EXACTLY on the yield surface \
             (dev_norm = yield_stress/(2*mu) = {expected_dev:.6}), got {dev_after:.6}"
        );

        // Volumetric (trace) strain must be preserved exactly -- incompressible
        // plastic flow assumption, a real documented claim of this material.
        assert!(
            (tr_after - trace).abs() < 1.0e-4,
            "plastic flow must preserve volumetric strain exactly: expected trace={trace}, \
             got {tr_after}"
        );

        assert!(kappa_after > 0.0, "kappa must accumulate on a plastic step");
    }

    #[test]
    fn hardening_raises_the_effective_yield_surface() {
        // With hardening_modulus > 0, a state that would yield at kappa=0 should
        // require LESS additional plastic strain once kappa has already
        // accumulated (softer transition) -- real, checkable monotonic claim.
        let mat = VonMisesMaterial::with_hardening(2000.0, 3000.0, 100.0, 500.0);
        let target_dev_norm = 1.5 * mat.yield_stress / (2.0 * mat.mu);
        let d = per_component_d_for_target_dev_norm(target_dev_norm);
        let sigma = Vec2::new(d.exp(), (-d).exp());

        let (_, kappa_from_zero) = run_one_step(&mat, sigma, 0.0);
        let (_, kappa_from_existing) = run_one_step(&mat, sigma, 1.0);

        // Effective yield stress is HIGHER when kappa is already 1.0 (hardening),
        // so the SAME trial state should trigger a SMALLER incremental gamma.
        let gamma_from_zero = kappa_from_zero - 0.0;
        let gamma_from_existing = kappa_from_existing - 1.0;
        assert!(
            gamma_from_existing < gamma_from_zero,
            "hardening should shrink the plastic strain increment for the same trial \
             state once kappa has already accumulated: gamma(kappa=0)={gamma_from_zero:.6} \
             gamma(kappa=1)={gamma_from_existing:.6}"
        );
    }

    /// Real regression guard (2026-09-02, external review): the two tests
    /// above cannot catch a real bug that shipped here -- neither checks
    /// the projected stress against the real, ANALYTICAL post-hardening
    /// yield surface with `hardening_modulus > 0`
    /// (`marginal_state_beyond_yield_stress_projects_exactly_to_the_yield_
    /// surface` uses `hardening_modulus=0`, where the bug is invisible;
    /// `hardening_raises_the_effective_yield_surface` only checks
    /// monotonicity of `kappa`, never the final stress value). The real
    /// bug: `update_particle` computed `gamma` correctly and updated
    /// `kappa_new = kappa + gamma` correctly, but then projected onto
    /// `effective_yield` (the PRE-hardening limit) instead of
    /// `effective_yield + hardening_modulus*gamma` (the real, consistent
    /// POST-hardening limit) -- radial-return consistency requires the
    /// latter (Simo & Taylor's own associative J2 return mapping).
    ///
    /// Exact worked case (independently hand-derived, not just re-deriving
    /// what the code itself computes): mu=3000, yield_stress=100,
    /// hardening_modulus=500, a trial state giving `elastic_dev=150` ->
    /// `gamma=(150-100)/(6000+500)=0.0076923...`, real post-hardening
    /// limit `=100+500*0.0076923=103.846...`. The pre-fix code returned
    /// exactly 100 (the untouched pre-hardening limit) here.
    #[test]
    fn hardened_projection_lands_exactly_on_the_real_post_hardening_surface() {
        let mat = VonMisesMaterial::with_hardening(2000.0, 3000.0, 100.0, 500.0);
        let target_dev_norm = 150.0 / (2.0 * mat.mu); // elastic_dev = 150 exactly
        let d = per_component_d_for_target_dev_norm(target_dev_norm);
        let sigma = Vec2::new(d.exp(), (-d).exp());

        let (sigma_after, kappa_after) = run_one_step(&mat, sigma, 0.0);

        let gamma = kappa_after; // kappa started at 0.0
        let expected_post_hardening_limit = mat.yield_stress + mat.hardening_modulus * gamma;
        assert!(
            (expected_post_hardening_limit - 103.846).abs() < 0.01,
            "test setup sanity: hand-derived limit should match the real \
             worked case (~103.846), got {expected_post_hardening_limit}"
        );

        let eps_after = crate::materials::utils::hencky_strains(sigma_after);
        let tr_after = eps_after.x + eps_after.y;
        let dev_after = (eps_after - Vec2::splat(tr_after * 0.5)).length();
        let measured_limit = 2.0 * mat.mu * dev_after;
        assert!(
            (measured_limit - expected_post_hardening_limit).abs() < 1.0e-2,
            "projected stress must land EXACTLY on the REAL post-hardening \
             yield surface ({expected_post_hardening_limit:.3}), not the \
             pre-hardening one (100.0, the real bug this guards against): \
             got {measured_limit:.3}"
        );
    }

    #[test]
    fn exponential_trial_preserves_rigid_rotation_without_false_yield() {
        let mat = VonMisesMaterial::new(2000.0, 3000.0, 100.0);
        let dt = 0.25;
        let omega = 0.8;
        let spin = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));
        let (f_after, kappa_after) = run_rate_step(&mat, Mat2::IDENTITY, spin, dt, 0.0);
        let expected = Mat2::from_angle(omega * dt);

        let error = (f_after.x_axis - expected.x_axis)
            .abs()
            .max_element()
            .max((f_after.y_axis - expected.y_axis).abs().max_element());
        assert!(
            error < 1.0e-6,
            "rigid spin must integrate to a rotation: {f_after:?}"
        );
        assert!((f_after.determinant() - 1.0).abs() < 1.0e-6);
        assert_eq!(kappa_after, 0.0, "rigid rotation must not trigger J2 flow");
    }

    #[test]
    fn opposite_subyield_rates_are_reversible_and_do_not_harden() {
        let mat = VonMisesMaterial::new(2000.0, 3000.0, 100.0);
        let dt = 0.5;
        let rate = Mat2::from_diagonal(Vec2::new(0.01, -0.01));
        let (f_loaded, kappa_loaded) = run_rate_step(&mat, Mat2::IDENTITY, rate, dt, 0.0);
        assert!(elastic_deviatoric_stress(&mat, f_loaded) < mat.yield_stress);
        assert_eq!(kappa_loaded, 0.0);

        let (f_unloaded, kappa_unloaded) = run_rate_step(&mat, f_loaded, -rate, dt, kappa_loaded);
        let error = (f_unloaded.x_axis - Mat2::IDENTITY.x_axis)
            .abs()
            .max_element()
            .max(
                (f_unloaded.y_axis - Mat2::IDENTITY.y_axis)
                    .abs()
                    .max_element(),
            );
        assert!(
            error < 1.0e-6,
            "opposite elastic rates must return F to identity: {f_unloaded:?}"
        );
        assert_eq!(kappa_unloaded, 0.0);
    }

    #[test]
    fn nonzero_rate_hardening_projection_lands_on_updated_surface_and_preserves_volume() {
        let mat = VonMisesMaterial::with_hardening(2000.0, 3000.0, 100.0, 500.0);
        let target_dev_norm = 150.0 / (2.0 * mat.mu);
        let d = per_component_d_for_target_dev_norm(target_dev_norm);
        let trace = 0.2;
        let log_increment = Mat2::from_diagonal(Vec2::new(trace * 0.5 + d, trace * 0.5 - d));
        let trial_det = trace.exp();

        let (f_after, kappa_after) = run_rate_step(&mat, Mat2::IDENTITY, log_increment, 1.0, 0.0);
        let measured_limit = elastic_deviatoric_stress(&mat, f_after);
        let expected_limit = mat.yield_stress + mat.hardening_modulus * kappa_after;
        assert!((measured_limit - expected_limit).abs() < 1.0e-2);
        assert!(
            (f_after.determinant() - trial_det).abs() < 1.0e-5,
            "isochoric J2 return must preserve the exponential trial volume"
        );
    }

    #[test]
    fn yield_unload_reload_keeps_kappa_monotone_and_elastic_during_unload() {
        let mat = VonMisesMaterial::with_hardening(2000.0, 3000.0, 100.0, 500.0);
        let load = Mat2::from_diagonal(Vec2::new(0.04, -0.04));
        let (f_yielded, kappa_yielded) = run_rate_step(&mat, Mat2::IDENTITY, load, 1.0, 0.0);
        assert!(kappa_yielded > 0.0);

        let unload = Mat2::from_diagonal(Vec2::new(-0.01, 0.01));
        let (f_unloaded, kappa_unloaded) =
            run_rate_step(&mat, f_yielded, unload, 1.0, kappa_yielded);
        assert_eq!(
            kappa_unloaded, kappa_yielded,
            "an elastic unloading step must not accumulate plastic strain"
        );
        assert!(
            elastic_deviatoric_stress(&mat, f_unloaded)
                <= mat.yield_stress + mat.hardening_modulus * kappa_unloaded + 1.0e-3
        );

        let (f_reloaded, kappa_reloaded) =
            run_rate_step(&mat, f_unloaded, load, 1.0, kappa_unloaded);
        assert!(kappa_reloaded >= kappa_unloaded, "kappa must be monotone");
        let updated_surface = mat.yield_stress + mat.hardening_modulus * kappa_reloaded;
        assert!(
            (elastic_deviatoric_stress(&mat, f_reloaded) - updated_surface).abs() < 1.0e-2,
            "reloaded state must return to the updated hardening surface"
        );
    }
}

#[cfg(test)]
mod elastic_viscosity_tests {
    use super::*;
    use crate::Particle;

    /// Same audit-closing test `RankineMaterial`/`CorotatedMaterial`/
    /// `NaccMaterial` all carry for their own copy of this identical
    /// mechanism (see `elastic_viscosity`'s own doc): `kirchhoff_stress`
    /// must actually respond to the particle's velocity gradient when
    /// `elastic_viscosity > 0.0`, not just carry the field.
    #[test]
    fn nonzero_elastic_viscosity_adds_a_real_viscous_stress_term() {
        let elastic_only = VonMisesMaterial::new(2000.0, 3000.0, 100.0);
        let mut damped = elastic_only;
        damped.elastic_viscosity = 50.0;

        let mut p = Particle::zeroed();
        p.deformation_gradient = Mat2::IDENTITY;
        p.velocity_gradient = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0));
        let particles = Particles::from(vec![p]);

        let tau_elastic = elastic_only.kirchhoff_stress(&particles, 0);
        let tau_damped = damped.kirchhoff_stress(&particles, 0);

        let diff = tau_damped - tau_elastic;
        let max_abs = diff
            .x_axis
            .abs()
            .max_element()
            .max(diff.y_axis.abs().max_element());
        assert!(
            max_abs > 1.0e-6,
            "nonzero elastic_viscosity under a real velocity gradient must \
             change the Kirchhoff stress: elastic={tau_elastic:?} damped={tau_damped:?}"
        );
    }

    /// `elastic_viscosity == 0.0` (every preset's default) must leave
    /// `kirchhoff_stress` completely blind to the velocity gradient -- a
    /// real regression guard against the early-return branch getting
    /// "simplified" away into an unconditional `elastic + 0.0*d_dev`.
    #[test]
    fn zero_elastic_viscosity_ignores_the_velocity_gradient() {
        let mat = VonMisesMaterial::new(2000.0, 3000.0, 100.0);
        assert_eq!(mat.elastic_viscosity, 0.0);

        let mut p_rest = Particle::zeroed();
        p_rest.deformation_gradient = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(0.01, 0.97));
        let mut p_shearing = p_rest;
        p_shearing.velocity_gradient = Mat2::from_cols(Vec2::new(0.3, -0.1), Vec2::new(0.2, 0.4));
        let particles = Particles::from(vec![p_rest, p_shearing]);

        let tau_rest = mat.kirchhoff_stress(&particles, 0);
        let tau_shearing = mat.kirchhoff_stress(&particles, 1);
        assert_eq!(
            tau_rest, tau_shearing,
            "elastic_viscosity=0.0 must be bit-identical regardless of velocity_gradient"
        );
    }
}
