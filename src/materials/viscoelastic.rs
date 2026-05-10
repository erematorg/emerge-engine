use glam::{Mat2, Vec2};

use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::materials::utils::{MIN_J, elastic_wave_dt, lame_from_young};
use crate::particle::Particle;

/// Kelvin-Voigt viscoelastic solid.
///
/// Stress = elastic (NeoHookean) + viscous (Newtonian dashpot in parallel).
///   τ = τ_elastic(F) + η · D_dev
/// where D = (C + Cᵀ)/2 is the symmetric strain-rate from the APIC velocity gradient.
///
/// This is the Kelvin-Voigt model: spring and dashpot in **parallel**.
/// The material deforms elastically AND dissipates energy simultaneously.
/// Creep under constant stress eventually stops (unlike Maxwell).
///
/// # Physical regime
/// - Biological soft tissue (tendon, cartilage, ligament): E=1–100 MPa, ν≈0.45, η=1–100 Pa·s
/// - Cell cytoskeleton: E=1–10 kPa, η=0.01–0.1 Pa·s
/// - Gelatin / hydrogel: E=0.1–10 kPa, η=0.01–10 Pa·s
/// - Rubber dampers: E=1–10 MPa, η=100–10000 Pa·s
///
/// # Why Kelvin-Voigt, not Maxwell?
/// Maxwell (spring + dashpot in series) has stress relaxation to zero — models fluids/polymers.
/// KV (parallel) has strain creep that stops — models biological solids correctly.
/// Biological tissues (Fung 1993) are better approximated by KV at the MPM particle scale.
/// Maxwell requires storing Fₑ per particle (new Mat2 field); KV uses only existing `velocity_gradient`.
///
/// # Reference
/// Kelvin-Voigt: Christensen 1982, "Theory of Viscoelasticity".
/// NeoHookean base: Bonet & Wood 2008, §6.4.
/// MPM viscoelastic: Stomakhin et al. 2014, §3 (foam); Fang et al. 2019 (MPM-DEM).
#[derive(Debug, Clone, Copy)]
pub struct ViscoelasticMaterial {
    pub lambda: f32,
    pub mu: f32,
    /// Kelvin-Voigt viscosity η (Pa·s in SI, sim-units²/s at emerge scale).
    /// Larger = more dissipation, stiffer rate response.
    pub viscosity: f32,
    /// Clamp J ∈ [j_min, 1/j_min] to prevent stress explosion on extreme compression.
    pub j_min: f32,
    /// Active stress coefficient for motile-cell / amoeba behaviour.
    /// τ_total = τ_elastic + τ_viscous + activation × coeff × I  (isotropic contractile pressure).
    /// 0.0 = passive (default). Tune to order of µ for visible deformation.
    pub active_stress_coeff: f32,
}

impl ViscoelasticMaterial {
    pub fn new(lambda: f32, mu: f32, viscosity: f32) -> Self {
        Self { lambda, mu, viscosity, j_min: 0.01, active_stress_coeff: 0.0 }
    }

    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32, viscosity: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, viscosity)
    }

    /// Biological soft tissue (tendon, cartilage): stiff, nearly incompressible, highly damped.
    /// E=50 kPa, ν=0.45, η=10 Pa·s.
    pub fn soft_tissue() -> Self {
        Self::from_young_modulus(5.0e4, 0.45, 10.0)
    }

    /// Cell body / cytoskeleton: very soft, moderate damping.
    /// E=5 kPa, ν=0.40, η=0.05 Pa·s.
    pub fn cell_body() -> Self {
        Self::from_young_modulus(5.0e3, 0.40, 0.05)
    }

    /// Hydrogel / biological scaffold: soft, lightly damped.
    /// E=1 kPa, ν=0.45, η=0.01 Pa·s.
    pub fn hydrogel() -> Self {
        Self::from_young_modulus(1.0e3, 0.45, 0.01)
    }

    /// Dense rubber-like material (e.g. intervertebral disc): stiff, highly damped.
    /// E=5 MPa, ν=0.45, η=1000 Pa·s.
    pub fn rubber_damper() -> Self {
        Self::from_young_modulus(5.0e6, 0.45, 1000.0)
    }
}

impl MaterialModel for ViscoelasticMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Viscoelastic
    }

    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let f = particle.deformation_gradient;
        let j = (f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x)
            .max(self.j_min)
            .min(1.0 / self.j_min);

        // NeoHookean elastic: τ = µ·(F·Fᵀ − I) + λ·ln(J)·I
        let b = f * f.transpose(); // Left Cauchy-Green tensor
        let elastic = self.mu * (b - Mat2::IDENTITY)
            + Mat2::from_diagonal(Vec2::splat(self.lambda * j.ln()));

        // Kelvin-Voigt viscous dashpot: τ_v = η · D_dev
        // D = (C + Cᵀ)/2 (symmetric strain rate from APIC velocity gradient)
        let c = particle.velocity_gradient;
        let sym = c + c.transpose();
        let d = sym * 0.5;
        let trace = d.x_axis.x + d.y_axis.y;
        let d_dev = d - Mat2::from_diagonal(Vec2::splat(trace * 0.5));
        let viscous = self.viscosity * d_dev;

        // Active stress: isotropic contractile pressure proportional to activation.
        let active = if self.active_stress_coeff != 0.0 {
            particle.activation * self.active_stress_coeff * Mat2::IDENTITY
        } else {
            Mat2::ZERO
        };

        elastic + viscous + active
    }

    fn update_particle(&self, particle: &mut Particle, dt: f32) {
        let fp_new = Mat2::IDENTITY + dt * particle.velocity_gradient;
        particle.deformation_gradient = fp_new * particle.deformation_gradient;
        let j = particle.deformation_gradient.determinant().max(MIN_J);
        particle.sync_volume_and_density(j);
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.initial_volume
    }

    fn activation_scale(&self) -> f32 {
        self.active_stress_coeff
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Viscoelastic as u32,
            lambda: self.lambda,
            mu: self.mu,
            dynamic_viscosity: self.viscosity,
            active_stress_coeff: self.active_stress_coeff,
            volume_ratio_min: self.j_min, // GPU uses this to clamp J in ln(J) stress term
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        particle: &Particle,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let elastic_dt = elastic_wave_dt(
            self.lambda, self.mu, 1.0,
            particle.density, 1.0e-6, cell_width, material_cfl,
        );
        let viscous_dt = if self.viscosity > 0.0 {
            let density = particle.density.max(1.0e-6);
            let kinematic = self.viscosity / density;
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
