use glam::{Mat2, Vec2};

use crate::materials::physical_props::{Elastic, FromSI, scale_lame};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    MIN_J, deformation_increment_exp, elastic_wave_dt, hencky_strains,
    reconstruct_stress_from_principal,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::Particles;

/// No-compression (tension-only) elastic material -- the continuum-mechanics dual of
/// no-tension masonry theory, a real, established treatment for cables, membranes,
/// tendons, and spider silk (unified variational framework for no-tension/no-
/// compression solids; the classic special case, tension-field theory for wrinkling
/// membranes under compression, describes exactly this: "an ideal membrane... can
/// sustain only tensile loads and offers no resistance to in-plane compression,
/// instead forming wrinkles").
///
/// Elastic response: the relaxed isotropic Hencky energy in principal-strain
/// space. In a taut region this is the same Hencky law used by
/// `RankineMaterial`/`VonMisesMaterial`. In a wrinkled region the transverse
/// strain is minimized out subject to zero transverse stress, leaving a real
/// uniaxial tensile stress independent of any additional slack contraction.
/// If both principal logarithmic strains are non-positive the region is slack
/// and carries no stress. This is the tension-field construction of Pipkin and
/// Steigmann, rather than component-wise clipping of an already-coupled stress
/// (which is not the gradient of a relaxed energy and can incorrectly erase a
/// load-bearing tension when the transverse direction contracts).
///
/// Fully REVERSIBLE, unlike every plasticity model in this engine: a particle that
/// goes slack under compression regains full tensile stiffness immediately once
/// stretched back past zero -- no permanent damage/hardening state. Its own
/// `update_particle` does real elastic F-integration
/// (`F_new=exp(dt*C)*F_old`, the exact constant-rate solution of `dF/dt=C*F`)
/// with volume/density sync, but no plastic return-mapping on
/// top -- "reversible" means no PERMANENT state, not that F itself never updates.
/// (An earlier version of this doc claimed F updated via some separate "ordinary
/// G2P integration" needing no override here at all -- that was wrong, and a real
/// bug: with no override, F stayed bit-for-bit IDENTITY forever in every dynamic
/// scene, confirmed live. `kirchhoff_stress` was always correct GIVEN a real F;
/// F itself just never became one.) This is a genuinely different kind of material
/// from Rankine/VonMises/Sand: an asymmetric nonlinear ELASTIC law, not an
/// irreversible return-mapping plasticity law -- belongs alongside `Elastic`/
/// `Viscoelastic` in the property taxonomy, not `PlasticityModel`.
///
/// CPU-only for now -- a real, disclosed scope limit (this engine's own "CPU
/// correctness first, GPU port second" rule): a WGSL SVD-based stress branch is real,
/// separate follow-up work, not silently skipped.
///
/// Suitable for: spider silk/webs, tendons/ligaments, membranes (wings, fins, drum
/// skins, inflatable structures -- pairs naturally with `Particle::internal_pressure`
/// for a pressurized membrane), climbing-plant tendrils -- any structure that only
/// carries tension.
#[derive(Debug, Clone, Copy)]
pub struct NoCompressionMaterial {
    pub lambda: f32,
    pub mu: f32,
}

impl NoCompressionMaterial {
    /// Construct directly from grid-native Lame parameters -- NOT SI
    /// Pascals. Prefer the [`FromSI`] impl on this type (via `Elastic`
    /// properties) for a real SI-to-grid conversion.
    pub const fn new(lambda: f32, mu: f32) -> Self {
        Self { lambda, mu }
    }

    /// Principal Kirchhoff stress from the relaxed Hencky energy.
    /// `eps.x >= eps.y` because `svd2` returns singular values in descending
    /// magnitude (and the kinematic exponential preserves positive J).
    #[inline]
    fn relaxed_principal_stress(&self, eps: Vec2) -> Vec2 {
        if eps.x <= 0.0 {
            return Vec2::ZERO;
        }

        let a = 2.0 * self.mu + self.lambda;
        let free_transverse_strain = -self.lambda * eps.x / a;
        if eps.y <= free_transverse_strain {
            // Minimize W=mu*|eps|^2 + lambda/2*tr(eps)^2 over the free
            // transverse strain. The resulting derivative dW_rel/d eps_x
            // is the uniaxial tangent below; tau_y is exactly zero.
            let uniaxial_tangent = 4.0 * self.mu * (self.mu + self.lambda) / a;
            Vec2::new(uniaxial_tangent * eps.x, 0.0)
        } else {
            Vec2::new(
                a * eps.x + self.lambda * eps.y,
                self.lambda * eps.x + a * eps.y,
            )
        }
    }
}

impl FromSI<Elastic> for NoCompressionMaterial {
    fn from_physical(props: &Elastic, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(props.e_pa, props.nu, props.rho_kg_m3, config);
        Self::new(lambda, mu)
    }
}

impl MaterialModel for NoCompressionMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::NoCompression
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let f = particles.deformation_gradient[i];
        let j = f.determinant();
        if j <= MIN_J {
            return Mat2::ZERO;
        }
        let (u, sigma, _vt) = svd2(f);
        let eps = hencky_strains(sigma);
        reconstruct_stress_from_principal(u, self.relaxed_principal_stress(eps))
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
    }

    /// Real regression fix (external review, found live building the first
    /// real interactive example for this material): this struct's own doc
    /// used to claim `deformation_gradient` "advances through the ordinary
    /// G2P integration, same as `NeoHookeanMaterial`/`CorotatedMaterial`",
    /// as the reason it didn't need its own `update_particle` override.
    /// That claim was simply wrong -- `NeoHookeanMaterial::update_particle`
    /// (elastic.rs) and `CorotatedMaterial::update_particle` (corotated.rs)
    /// BOTH perform the real `F_new = (I+dt*C)*F_old` integration
    /// THEMSELVES; there is no separate, generic mechanism that does it for
    /// materials which skip the override. Confirmed directly, live: with no
    /// override, `deformation_gradient` stayed bit-for-bit `Mat2::IDENTITY`
    /// forever, for every particle, in a real dynamic scene (a block
    /// falling under real gravity) -- the material's own `kirchhoff_stress`
    /// is correct GIVEN a real F (its own closed-form test proves this),
    /// but F itself never updated to reflect any real motion at all, so a
    /// NoCompressionMaterial body has been unable to register real strain
    /// in ANY dynamic simulation, ever, undetected until now because no
    /// interactive/dynamic example exercised it before. This material now
    /// uses the exact constant-rate exponential update rather
    /// than forward Euler.  Forward Euler is not reversible even for equal
    /// and opposite rates: `(1+a)(1-a)=1-a^2`, producing a deterministic
    /// volume-loss ratchet in precisely the slack directions where this
    /// material has no restoring stress.  The exponential is the exact
    /// solution of the continuum kinematic equation `dF/dt=C*F`; it is the
    /// physical correction, not an admissibility clamp.
    fn update_particle(&self, ctx: &mut crate::particle::ParticleUpdateCtx, dt: f32) {
        let f_new =
            deformation_increment_exp(dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;
        *ctx.deformation_gradient = f_new;
        let j = f_new.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::NoCompression as u32,
            lambda: self.lambda,
            mu: self.mu,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        _viscous_cfl: f32,
    ) -> f32 {
        elastic_wave_dt(
            self.lambda,
            self.mu,
            1.0,
            density,
            MIN_J,
            cell_width,
            material_cfl,
        )
    }
}

#[cfg(test)]
mod tension_compression_tests {
    use super::*;
    use crate::Particle;

    fn particle_with_f(f: Mat2) -> Particle {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = 1.0;
        p.density = 1.0;
        p
    }

    /// Real, checkable asymmetry: pure uniaxial STRETCH must produce the full
    /// underlying elastic stiffness (same as an ordinary elastic material), pure
    /// uniaxial COMPRESSION of the same magnitude must produce exactly zero stress
    /// on that axis (goes slack) -- the textbook no-compression signature, not a
    /// vibes check.
    #[test]
    fn stretch_gives_full_stiffness_compression_gives_zero() {
        let mat = NoCompressionMaterial::new(100.0, 200.0);

        let stretched = particle_with_f(Mat2::from_diagonal(Vec2::new(1.2, 1.0)));
        let soa_stretch = Particles::from(vec![stretched]);
        let tau_stretch = mat.kirchhoff_stress(&soa_stretch, 0);
        assert!(
            tau_stretch.x_axis.x > 1.0,
            "stretching (J>1 along x) must produce real positive (tensile) stress: {tau_stretch:?}"
        );

        let compressed = particle_with_f(Mat2::from_diagonal(Vec2::new(0.8, 1.0)));
        let soa_compress = Particles::from(vec![compressed]);
        let tau_compress = mat.kirchhoff_stress(&soa_compress, 0);
        assert!(
            tau_compress.x_axis.x.abs() < 1e-5,
            "compressing (J<1 along x) must produce exactly zero stress (slack), not resist: {tau_compress:?}"
        );
    }

    /// Tension-field theory's defining wrinkled-state property: once the
    /// transverse direction is slack, further transverse contraction forms
    /// more unresolved wrinkle amplitude; it must not reduce the tension in
    /// the load-carrying direction. Component-wise clipping of the coupled
    /// Lamé stress failed this invariant and could make a hanging strip lose
    /// all vertical support as its horizontal F drifted.
    #[test]
    fn transverse_slack_contraction_does_not_erase_carrying_tension() {
        let mat = NoCompressionMaterial::new(100.0, 200.0);
        let axial_log_strain = 0.1;
        let moderately_wrinkled = Vec2::new(axial_log_strain, -0.1);
        let deeply_wrinkled = Vec2::new(axial_log_strain, -2.0);
        let tau_a = mat.relaxed_principal_stress(moderately_wrinkled);
        let tau_b = mat.relaxed_principal_stress(deeply_wrinkled);
        assert!(tau_a.x > 0.0 && tau_a.y == 0.0);
        assert_eq!(
            tau_a, tau_b,
            "additional contraction in a zero-stress wrinkle direction must not \
             erase or change the orthogonal carrying tension"
        );
    }

    #[test]
    fn relaxed_stress_is_continuous_at_taut_wrinkled_junction() {
        let mat = NoCompressionMaterial::new(100.0, 200.0);
        let eps_x = 0.2;
        let junction_y = -mat.lambda * eps_x / (2.0 * mat.mu + mat.lambda);
        let at = mat.relaxed_principal_stress(Vec2::new(eps_x, junction_y));
        let just_taut = mat.relaxed_principal_stress(Vec2::new(eps_x, junction_y + 1.0e-6));
        assert!(
            (at - just_taut).length() < 1.0e-3,
            "relaxed energy derivative must join the taut law continuously: \
             wrinkled={at:?} taut={just_taut:?}"
        );
    }

    /// Real regression guard (external review): `update_particle` used to be
    /// the trait default (a true no-op), on the mistaken belief that F
    /// updated some other, generic way -- it did not, so F was frozen at
    /// IDENTITY forever in real dynamic use. Confirms zero velocity_gradient
    /// still leaves F genuinely unchanged (a real invariant, not just an
    /// accident of the old no-op), and that `needs_cpu_update()` stays
    /// false (this is a pure elastic law, no CPU-only plastic pass needed,
    /// same as NeoHookean/Corotated).
    #[test]
    fn update_particle_is_a_no_op_only_under_zero_velocity_gradient() {
        let mat = NoCompressionMaterial::new(100.0, 200.0);
        assert!(
            !mat.needs_cpu_update(),
            "no-compression is a pure stress law (no plastic state) -- should not need a CPU-only update pass, same as NeoHookean/Corotated"
        );

        let f_before = Mat2::from_diagonal(Vec2::new(0.7, 1.3));
        let p = particle_with_f(f_before);
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles.update_ctx(0), 1.0);
        assert_eq!(
            particles.deformation_gradient[0], f_before,
            "with zero velocity_gradient, F_new=(I+dt*0)*F_old=F_old exactly -- \
             real invariant, not the old (wrong) universal no-op"
        );
    }

    /// Real regression guard (external review, the actual bug): with a
    /// REAL, nonzero velocity_gradient, `update_particle` must genuinely
    /// integrate the kinematic equation. Before this fix, F stayed at
    /// IDENTITY here too, silently, in every real simulation using this
    /// material. Unlike the legacy forward-Euler solid paths, this regression
    /// checks the exact constant-rate solution so it cannot encode their
    /// volume-ratchet error as the expected result.
    #[test]
    fn update_particle_integrates_exact_constant_rate_elastic_strain() {
        let mat = NoCompressionMaterial::new(100.0, 200.0);
        let p = particle_with_f(Mat2::IDENTITY);
        let mut particles = Particles::from(vec![p]);
        let c = Mat2::from_diagonal(Vec2::new(0.1, -0.05));
        *particles.update_ctx(0).velocity_gradient = c;
        let dt = 0.02;
        mat.update_particle(&mut particles.update_ctx(0), dt);

        let expected =
            Mat2::from_diagonal(Vec2::new((dt * c.x_axis.x).exp(), (dt * c.y_axis.y).exp()));
        let got = particles.deformation_gradient[0];
        assert!(
            (got.x_axis - expected.x_axis).length() < 1.0e-6
                && (got.y_axis - expected.y_axis).length() < 1.0e-6,
            "F must integrate the exact exp(dt*C)*F_old solution: expected={expected:?} got={got:?}"
        );

        let expected_j = expected.determinant();
        let expected_volume = (particles.initial_volume[0] * expected_j).max(1.0e-6);
        assert!(
            (particles.volume[0] - expected_volume).abs() < 1.0e-6,
            "volume must sync from the new J: expected={expected_volume} got={}",
            particles.volume[0]
        );
    }

    /// Real reversibility check, now that F genuinely updates: compressing
    /// then restretching back to the SAME F must land on the SAME stress a
    /// fresh particle at that F would give -- no hysteresis, no lingering
    /// memory, matching this material's own "fully reversible" claim. This
    /// specifically could not have been checked meaningfully against the
    /// old no-op version (F never moved at all, so there was nothing to
    /// return FROM).
    #[test]
    fn compress_then_restretch_matches_a_fresh_particle_at_the_same_f() {
        let mat = NoCompressionMaterial::new(100.0, 200.0);
        let p = particle_with_f(Mat2::IDENTITY);
        let mut particles = Particles::from(vec![p]);
        let dt = 0.01;

        // Compress (negative C), then apply the exact opposite C for the
        // same duration -- net elongation should cancel, landing back
        // exactly at IDENTITY.
        *particles.update_ctx(0).velocity_gradient = Mat2::from_diagonal(Vec2::new(-2.0, 0.0));
        mat.update_particle(&mut particles.update_ctx(0), dt);
        *particles.update_ctx(0).velocity_gradient = Mat2::from_diagonal(Vec2::new(2.0, 0.0));
        mat.update_particle(&mut particles.update_ctx(0), dt);

        let round_tripped_f = particles.deformation_gradient[0];
        let diff = (round_tripped_f.x_axis - Mat2::IDENTITY.x_axis).length()
            + (round_tripped_f.y_axis - Mat2::IDENTITY.y_axis).length();
        assert!(
            diff < 1.0e-6,
            "equal opposite constant rates must return F itself to identity, not merely \
             hide a residual compression behind zero stress: F={round_tripped_f:?}"
        );
    }
}
