use glam::Mat2;

use crate::materials::physical_props::{Elastic, FromSI, scale_lame};
use crate::materials::utils::{MIN_J, deformation_increment_exp, elastic_wave_dt, lame_from_young};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams, polar_decomposition_2d};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Corotated linear elasticity.
///
/// Kirchhoff stress: τ = 2µ·h·(F−R)·Fᵀ + λ·h·(J−1)·J·I
/// R is the rotation from 2D polar decomposition (analytical -- no SVD needed in 2D).
/// h = particle.hardening_scale (1.0 baseline; snow plasticity scales this up on compression).
/// Reference: Stomakhin et al. 2013, eq. (5)–(8). Used as the elastic base for snow.
/// Also the elastic component of Drucker-Prager (Klar et al. 2016).
#[derive(Debug, Clone, Copy)]
pub struct CorotatedMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Thermal modulus scale: µ_eff = µ·h·(1 + thermal_expansion·T), same for λ.
    /// Negative = thermal softening (typical). 0.0 = isothermal (default).
    pub thermal_expansion: f32,
    /// Active stress coefficient for muscle/motile-cell behaviour (same semantics as NeoHookean).
    /// τ_total = τ_elastic + activation × coeff × F·(n₀⊗n₀)·Fᵀ  (fiber-directional contraction).
    /// 0.0 = passive (default). Tune to be on the order of µ for visible locomotion.
    pub active_stress_coeff: f32,
    /// Clamp J ≥ j_min before evaluating the volumetric term, same convention
    /// `ViscoelasticMaterial`/`NeoHookeanMaterial` use (default 0.01) -- see
    /// `NeoHookeanMaterial::j_min`'s own doc. Clamp, don't hard-zero below
    /// `MIN_J`, or the volumetric restoring force can't act exactly when it's
    /// needed most. Honest, disclosed limitation: this model's volumetric
    /// term (`λ·(J−1)·J`, a bounded polynomial, not a diverging log-barrier)
    /// is inherently weaker against extreme compression than NeoHookean's
    /// (Stomakhin et al. 2013's simplified elastic base for snow/DP
    /// plasticity supplies its OWN separate compression_limit clamps; this
    /// material alone, e.g. `basic_jellies_gpu`, doesn't have that net) --
    /// the clamp removes the permanent-zero-stress trap, it doesn't give
    /// this model NeoHookean's stronger barrier.
    pub j_min: f32,
    /// Real Kelvin-Voigt viscous damping on the deviatoric elastic strain
    /// rate (SI Pa.s, converted with the SAME convention `lambda`/`mu` used
    /// -- see `rankine::q_factor_elastic_viscosity_pa_s`'s own doc for the
    /// pairing rule and the real regression it documents) -- same
    /// mechanism, same formula, as
    /// `RankineMaterial::elastic_viscosity` / `DruckerPragerMaterial::elastic_viscosity`.
    /// Zero cost, zero behavior change at `0.0` (default, matching every
    /// other material using this same mechanism).
    ///
    /// Without this, a pure elastic solid has NO energy dissipation at all
    /// -- real solids are never purely elastic (internal friction from
    /// dislocation motion and grain-boundary sliding measurably dissipates
    /// energy in every real material, reported as a seismic/ultrasonic
    /// quality factor Q -- see `q_factor_elastic_viscosity_pa_s`). Confirmed
    /// live 2026-08-29: this is a real, structural gap -- `CorotatedMaterial`
    /// had NO damping mechanism of any kind before this field existed, so
    /// any solid built from it rings elastically forever under repeated
    /// impact (found while root-causing sustained post-impact bouncing on
    /// `RankineMaterial::ice()`, which shares this same corotated elastic
    /// base -- see `project_rankine_damping_and_fluid_condensation_
    /// continuity_fixed`).
    pub elastic_viscosity: f32,
}

impl CorotatedMaterial {
    /// Construct directly from grid-native Lame parameters -- NOT SI
    /// Pascals (see [`Self::from_young_modulus`] for the common gotcha and
    /// the real SI conversion path).
    pub const fn new(lambda: f32, mu: f32) -> Self {
        Self {
            lambda,
            mu,
            thermal_expansion: 0.0,
            active_stress_coeff: 0.0,
            j_min: 0.01,
            elastic_viscosity: 0.0,
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν -- **grid
    /// units, NOT real Pascals** (real disclosure added 2026-09-05, same
    /// finding as `NeoHookeanMaterial::from_young_modulus`'s own doc): calls
    /// [`lame_from_young`] directly, never touches `dx_meters`/density.
    /// For a real, correctly SI-to-grid-converted material use
    /// [`Self::from_physical`] (needs a `&SimConfig` and real `rho_kg_m3`).
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }
}

impl FromSI<Elastic> for CorotatedMaterial {
    fn from_physical(props: &Elastic, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(props.e_pa, props.nu, props.rho_kg_m3, config);
        Self::new(lambda, mu)
    }
}

impl MaterialModel for CorotatedMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Corotated
    }

    fn init_particle(&self, particle: &mut Particle) {
        particle.hardening_scale = 1.0;
        particle.plastic_volume_ratio = 1.0;
    }

    fn corotated_lame_params(&self) -> Option<(f32, f32)> {
        // `hardening_scale`/`temperature` modifiers below fold into
        // mu_eff/lambda_eff only when thermal_expansion != 0 or
        // hardening_scale != 1 (never true for a passive Corotated particle
        // -- `init_particle` sets it to 1.0 and nothing ever touches it
        // again, no plastic return-mapping on this material). Real fix
        // (2026-09-11): this used to say the SOLVER'S eligibility check
        // also verified `hardening_scale == 1.0` per particle -- removed
        // from there, it was a blanket, all-materials check based on an
        // assumption that field means the same thing everywhere. It
        // doesn't: `DruckerPragerMaterial` legitimately repurposes
        // `hardening_scale` as strain-rate edge-detection memory (see its
        // own `update_particle`), so a blanket `== 1.0` gate silently
        // rejected the implicit solver for nearly every real, non-static
        // sand particle. Correctness for THIS material still holds without
        // that external gate -- `hardening_scale` genuinely never leaves
        // 1.0 here, for the reason stated above.
        if self.elastic_viscosity == 0.0
            && self.thermal_expansion == 0.0
            && self.active_stress_coeff == 0.0
        {
            Some((self.lambda, self.mu))
        } else {
            None
        }
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        // Clamp, don't zero -- see `j_min`'s own doc.
        let j = f.determinant().max(self.j_min);

        let r = polar_decomposition_2d(f);

        let h = particles.hardening_scale[i];
        let t_scale = 1.0 + self.thermal_expansion * particles.temperature[i];
        let mu_eff = self.mu * h * t_scale;
        let lambda_eff = self.lambda * h * t_scale;

        let f_t = f.transpose();
        let elastic = 2.0 * mu_eff * (f - r) * f_t + lambda_eff * (j - 1.0) * j * Mat2::IDENTITY;
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
        let d_dev = d - Mat2::from_diagonal(glam::Vec2::splat(trace * 0.5));
        elastic + self.elastic_viscosity * d_dev
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // Exact matrix exponential, not forward Euler -- see
        // `deformation_increment_exp`'s own doc for the real O(dt^2)
        // volumetric ratchet this removes (found+fixed 2026-09,
        // NoCompression/VonMises, now rolled out here on the same basis:
        // every tensor-F material shares the identical exposure).
        let fp_new = deformation_increment_exp(dt * *ctx.velocity_gradient);
        *ctx.deformation_gradient = fp_new * *ctx.deformation_gradient;
        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn activation_scale(&self) -> f32 {
        self.active_stress_coeff
    }

    fn pressure_scale(&self) -> f32 {
        1.0
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Corotated as u32,
            lambda: self.lambda,
            mu: self.mu,
            thermal_expansion: self.thermal_expansion,
            active_stress_coeff: self.active_stress_coeff,
            // Real, per-material stress-computation floor -- see `j_min`'s own
            // doc. NOT shared with Snow/Sand's own reuse of this same GPU
            // param slot for their real, different plastic-clamp range --
            // Corotated doesn't populate it for anything else.
            volume_ratio_min: self.j_min,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let elastic_dt = elastic_wave_dt(
            self.lambda,
            self.mu,
            hardening_scale,
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
mod small_strain_linear_elasticity_tests {
    use super::*;
    use glam::Vec2;

    /// **Small-strain limit must recover exact linear elasticity (Hooke's law).**
    ///
    /// `CorotatedMaterial` had zero test comparing it to any measured/analytical
    /// result (confirmed via a full test-file audit, 2026-07-07). For a
    /// SYMMETRIC small strain F=I+delta*E (E symmetric, no rotation), the polar
    /// decomposition's rotation R is EXACTLY I (not just to leading order --
    /// `polar_decomposition_2d`'s own formula gives y=F10-F01=delta(E10-E01)=0
    /// exactly when E is symmetric, so R=I for any delta, not an approximation).
    /// This makes the leading-order linearization of tau=2*mu*(F-R)*F^T +
    /// lambda*(J-1)*J*I reduce to EXACTLY tau = 2*mu*eps + lambda*tr(eps)*I
    /// (eps=delta*E) -- the textbook linear-elasticity form directly, no
    /// plane-strain k=lambda+mu correction needed (unlike NeoHookean's
    /// vol-dev split form -- Corotated already IS linear elasticity, just
    /// extended to finite rotation).
    fn particle_with_f(f: Mat2) -> Particles {
        let mut particles = Particles::default();
        particles.push(Particle {
            x: Vec2::ZERO,
            v: Vec2::ZERO,
            velocity_gradient: Mat2::ZERO,
            deformation_gradient: f,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening: 0.0,
            log_volume_strain: 0.0,
            temperature: 0.0,
            scalar_field: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            internal_pressure: 0.0,
        });
        particles
    }

    fn linear_elastic_prediction(lambda: f32, mu: f32, eps: Mat2) -> Mat2 {
        let tr_eps = eps.x_axis.x + eps.y_axis.y;
        Mat2::from_diagonal(Vec2::splat(lambda * tr_eps)) + 2.0 * mu * eps
    }

    #[test]
    fn small_uniaxial_strain_matches_hookes_law() {
        let lambda = 1000.0;
        let mu = 800.0;
        let mat = CorotatedMaterial::new(lambda, mu);

        let delta = 1.0e-4_f32;
        let e = Mat2::from_diagonal(Vec2::new(1.0, 0.0));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        let predicted = linear_elastic_prediction(lambda, mu, delta * e);

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain Corotated stress should match linear elasticity to O(delta^2): \
             predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    #[test]
    fn small_shear_strain_matches_hookes_law() {
        let lambda = 500.0;
        let mu = 1200.0;
        let mat = CorotatedMaterial::new(lambda, mu);

        let delta = 1.0e-4_f32;
        let e = Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0));
        let f = Mat2::IDENTITY + delta * e;

        let particles = particle_with_f(f);
        let tau = mat.kirchhoff_stress(&particles, 0);
        let predicted = linear_elastic_prediction(lambda, mu, delta * e);

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        let scale = (predicted.x_axis.length_squared() + predicted.y_axis.length_squared()).sqrt();
        assert!(
            err / scale < 1.0e-3,
            "small-strain Corotated shear stress should match linear elasticity to O(delta^2): \
             predicted={predicted:?} actual={tau:?} relative_err={:.2e}",
            err / scale
        );
    }

    /// Real, exact property (not an approximation): for symmetric F, the
    /// rotation R from polar decomposition must be EXACTLY identity, at any
    /// strain magnitude (not just small) -- this is what makes Corotated
    /// reduce cleanly to linear elasticity for pure-strain (no-rotation)
    /// deformations. Checked directly, independent of the Hooke's-law tests
    /// above.
    #[test]
    fn symmetric_deformation_gradient_has_exact_identity_rotation() {
        let e = Mat2::from_cols(Vec2::new(0.3, -0.15), Vec2::new(-0.15, 0.6)); // symmetric, large
        let f = Mat2::IDENTITY + e;
        let r = polar_decomposition_2d(f);
        assert!(
            (r - Mat2::IDENTITY).x_axis.length() < 1.0e-6
                && (r - Mat2::IDENTITY).y_axis.length() < 1.0e-6,
            "symmetric F must give EXACTLY R=I, got {r:?}"
        );
    }
}

#[cfg(test)]
mod elastic_viscosity_tests {
    use super::*;
    use glam::Vec2;

    /// Same audit-closing test `RankineMaterial`/`VonMisesMaterial`/
    /// `NaccMaterial` all carry for their own copy of this identical
    /// mechanism (see `elastic_viscosity`'s own doc): `kirchhoff_stress`
    /// must actually respond to the particle's velocity gradient when
    /// `elastic_viscosity > 0.0`, not just carry the field.
    #[test]
    fn nonzero_elastic_viscosity_adds_a_real_viscous_stress_term() {
        let elastic_only = CorotatedMaterial::new(2000.0, 3000.0);
        let mut damped = elastic_only;
        damped.elastic_viscosity = 50.0;

        let mut particles = Particles::default();
        particles.push(Particle {
            x: Vec2::ZERO,
            v: Vec2::ZERO,
            velocity_gradient: Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)),
            deformation_gradient: Mat2::IDENTITY,
            mass: 1.0,
            initial_volume: 1.0,
            volume: 1.0,
            density: 1.0,
            material_id: 0,
            plastic_volume_ratio: 1.0,
            hardening_scale: 1.0,
            friction_hardening: 0.0,
            log_volume_strain: 0.0,
            temperature: 0.0,
            scalar_field: 0.0,
            user_tag: 0,
            activation: 0.0,
            activation_dir: Vec2::ZERO,
            muscle_group_id: 0,
            contact_group: 0,
            sleeping: 0,
            pinned: 0,
            internal_pressure: 0.0,
        });

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
    /// "simplified" away into an unconditional `elastic + 0.0*d_dev`,
    /// which would silently propagate a NaN/Inf `d_dev` (e.g. from a
    /// sleeping or freshly-spawned particle's own garbage velocity
    /// gradient) into every material using this mechanism at its default.
    #[test]
    fn zero_elastic_viscosity_ignores_the_velocity_gradient() {
        let mat = CorotatedMaterial::new(2000.0, 3000.0);
        assert_eq!(mat.elastic_viscosity, 0.0);

        fn particle_with(f: Mat2, c: Mat2) -> Particles {
            let mut particles = Particles::default();
            particles.push(Particle {
                x: Vec2::ZERO,
                v: Vec2::ZERO,
                velocity_gradient: c,
                deformation_gradient: f,
                mass: 1.0,
                initial_volume: 1.0,
                volume: 1.0,
                density: 1.0,
                material_id: 0,
                plastic_volume_ratio: 1.0,
                hardening_scale: 1.0,
                friction_hardening: 0.0,
                log_volume_strain: 0.0,
                temperature: 0.0,
                scalar_field: 0.0,
                user_tag: 0,
                activation: 0.0,
                activation_dir: Vec2::ZERO,
                muscle_group_id: 0,
                contact_group: 0,
                sleeping: 0,
                pinned: 0,
                internal_pressure: 0.0,
            });
            particles
        }

        let f = Mat2::from_cols(Vec2::new(1.05, 0.02), Vec2::new(0.01, 0.97));
        let particles_at_rest = particle_with(f, Mat2::ZERO);
        let particles_shearing = particle_with(
            f,
            Mat2::from_cols(Vec2::new(0.3, -0.1), Vec2::new(0.2, 0.4)),
        );

        let tau_rest = mat.kirchhoff_stress(&particles_at_rest, 0);
        let tau_shearing = mat.kirchhoff_stress(&particles_shearing, 0);
        assert_eq!(
            tau_rest, tau_shearing,
            "elastic_viscosity=0.0 must be bit-identical regardless of velocity_gradient"
        );
    }
}

/// Real regression guard for the 2026-09 kinematic-integrator rollout
/// (`deformation_increment_exp` replacing forward Euler across every
/// tensor-F material -- see that function's own doc for the O(dt^2)
/// volumetric-ratchet mechanism it removes, first found+fixed on
/// `NoCompressionMaterial`/`VonMisesMaterial`). Tests the actual real
/// call site (`update_particle`), not just the isolated helper (already
/// covered by its own 3 tests in `utils.rs`) -- proves this material is
/// really wired to the exponential integrator, not still silently on Euler.
#[cfg(test)]
mod kinematic_integrator_tests {
    use super::*;
    use glam::Vec2;

    fn particle_with_f(f: Mat2) -> Particle {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.density = 1.0;
        p
    }

    /// The real, direct test of the O(dt^2) ratchet Euler had: apply a
    /// velocity gradient for one substep, then its exact opposite for the
    /// same substep -- a perfectly reversible round trip. Under the OLD
    /// Euler integration this would NOT return exactly to the starting F
    /// (that was the whole bug); under the exact exponential it must,
    /// to floating-point precision.
    #[test]
    fn opposite_velocity_gradients_cancel_exactly_no_volumetric_ratchet() {
        let mat = CorotatedMaterial::new(1000.0, 800.0);
        let p = particle_with_f(Mat2::IDENTITY);
        let mut particles = Particles::from(vec![p]);
        let dt = 0.05;
        let c = Mat2::from_cols(Vec2::new(0.3, -0.15), Vec2::new(0.1, -0.3));

        *particles.update_ctx(0).velocity_gradient = c;
        mat.update_particle(&mut particles.update_ctx(0), dt);
        *particles.update_ctx(0).velocity_gradient = -c;
        mat.update_particle(&mut particles.update_ctx(0), dt);

        let f_after = particles.deformation_gradient[0];
        let err = (f_after.x_axis - Mat2::IDENTITY.x_axis).length()
            + (f_after.y_axis - Mat2::IDENTITY.y_axis).length();
        assert!(
            err < 1.0e-5,
            "a reversible round trip must return exactly to the starting F, \
             not lose/gain volume: f_after={f_after:?} err={err}"
        );
    }

    /// A pure rigid rotation (antisymmetric velocity_gradient) must leave J
    /// exactly at 1.0 -- Euler's own failure mode was a spurious
    /// O(omega^2*dt^2) dilation under exactly this kind of loading.
    #[test]
    fn rigid_rotation_preserves_volume_exactly() {
        let mat = CorotatedMaterial::new(1000.0, 800.0);
        let p = particle_with_f(Mat2::IDENTITY);
        let mut particles = Particles::from(vec![p]);
        let spin = Mat2::from_cols(Vec2::new(0.0, 0.4), Vec2::new(-0.4, 0.0));

        *particles.update_ctx(0).velocity_gradient = spin;
        for _ in 0..50 {
            mat.update_particle(&mut particles.update_ctx(0), 0.02);
        }

        let j = particles.deformation_gradient[0].determinant();
        assert!(
            (j - 1.0).abs() < 1.0e-4,
            "sustained rigid rotation must not drift volume: J={j}"
        );
    }
}
