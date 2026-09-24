use glam::{Mat2, Vec2};

use crate::materials::physical_props::{Elastic, FromSI, scale_lame};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    MIN_J, elastic_wave_dt, hencky_strains, reconstruct_stress_from_principal,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::Particles;

/// No-compression (tension-only) elastic material — the continuum-mechanics dual of
/// no-tension masonry theory, a real, established treatment for cables, membranes,
/// tendons, and spider silk (unified variational framework for no-tension/no-
/// compression solids; the classic special case, tension-field theory for wrinkling
/// membranes under compression, describes exactly this: "an ideal membrane... can
/// sustain only tensile loads and offers no resistance to in-plane compression,
/// instead forming wrinkles").
///
/// Elastic response: isotropic Hencky (log-strain) elasticity in principal stress
/// space — the SAME formula `RankineMaterial`/`VonMisesMaterial` use for their own
/// passive elastic term — but any COMPRESSIVE (negative) principal Kirchhoff stress
/// is clamped to zero instead of being resisted, so the material offers zero
/// resistance to being pushed together along that axis.
///
/// Fully REVERSIBLE, unlike every plasticity model in this engine: a particle that
/// goes slack under compression regains full tensile stiffness immediately once
/// stretched back past zero — no permanent damage/hardening state, and
/// `deformation_gradient` is never modified by this material directly (no
/// `update_particle` override; F advances through the ordinary G2P integration, same
/// as `NeoHookeanMaterial`/`CorotatedMaterial`). This is a genuinely different kind
/// of material from Rankine/VonMises/Sand: an asymmetric nonlinear ELASTIC law, not
/// an irreversible return-mapping plasticity law — belongs alongside `Elastic`/
/// `Viscoelastic` in the property taxonomy, not `PlasticityModel`.
///
/// CPU-only for now — a real, disclosed scope limit (this engine's own "CPU
/// correctness first, GPU port second" rule): a WGSL SVD-based stress branch is real,
/// separate follow-up work, not silently skipped.
///
/// Suitable for: spider silk/webs, tendons/ligaments, membranes (wings, fins, drum
/// skins, inflatable structures — pairs naturally with `Particle::internal_pressure`
/// for a pressurized membrane), climbing-plant tendrils — any structure that only
/// carries tension.
#[derive(Debug, Clone, Copy)]
pub struct NoCompressionMaterial {
    pub lambda: f32,
    pub mu: f32,
}

impl NoCompressionMaterial {
    pub const fn new(lambda: f32, mu: f32) -> Self {
        Self { lambda, mu }
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
        let a = 2.0 * self.mu + self.lambda;
        let tau = Vec2::new(
            a * eps.x + self.lambda * eps.y,
            self.lambda * eps.x + a * eps.y,
        );
        // No-compression: any negative (compressive) principal stress goes slack.
        let tau_clamped = Vec2::new(tau.x.max(0.0), tau.y.max(0.0));
        reconstruct_stress_from_principal(u, tau_clamped)
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.initial_volume[i]
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
    /// on that axis (goes slack) — the textbook no-compression signature, not a
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

    /// Reversibility: no permanent state, unlike every plasticity model in this
    /// engine (Rankine/VonMises/Sand all override `update_particle` to permanently
    /// alter `deformation_gradient` via return-mapping). `update_particle` must be a
    /// true no-op (the trait default) so a compressed-then-restretched particle
    /// regains full stiffness immediately, with no damage/hardening memory.
    #[test]
    fn update_particle_is_a_true_no_op() {
        let mat = NoCompressionMaterial::new(100.0, 200.0);
        assert!(
            !mat.needs_cpu_update(),
            "no-compression is a pure stress law (no plastic state) -- should not need a CPU-only update pass, same as NeoHookean/Corotated"
        );

        let f_before = Mat2::from_diagonal(Vec2::new(0.7, 1.3)); // deliberately in the compressive regime on x
        let p = particle_with_f(f_before);
        let mut particles = Particles::from(vec![p]);
        mat.update_particle(&mut particles.update_ctx(0), 1.0);
        assert_eq!(
            particles.deformation_gradient[0], f_before,
            "update_particle must leave deformation_gradient completely untouched (trait default no-op)"
        );
    }
}
