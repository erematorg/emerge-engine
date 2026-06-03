use glam::{Mat2, Vec2};

use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::Particles;

/// Bingham viscoplastic fluid.
///
/// Below yield stress τ₀: rigid plug (no deviatoric flow).
/// Above yield stress τ₀: Newtonian with apparent viscosity η_app = τ₀/γ̇ + η.
///
/// Stress decomposition: σ = −p·I + τ_deviatoric
/// Pressure: Tait EOS — p = k·((ρ/ρ₀)^γ − 1), same as NewtonianFluid.
/// Deviatoric:
///   γ̇ = √(2·D:D)   (scalar shear rate, D = symmetric velocity gradient)
///   τ = (τ₀/γ̇ + η)·D_dev   if γ̇ > critical_shear_rate, else 0
///
/// Reference: Bingham 1916. MPM formulation: GeoTaichi BinghamModel (Taichi lang).
///
/// # Natural phenomena
/// - Mud / wet clay: τ₀ = 50–500 Pa, η = 0.1–5 Pa·s
/// - Lava (basaltic): τ₀ = 100–2000 Pa, η = 10–10000 Pa·s
/// - Biological cytoplasm: τ₀ ≈ 0.5–5 Pa, η ≈ 0.005–0.05 Pa·s
/// - Dense biological fluids (mucus, blood clot): τ₀ = 1–50 Pa, η = 0.01–1 Pa·s
#[derive(Debug, Clone, Copy)]
pub struct BinghamFluidMaterial {
    pub rest_density: f32,
    /// Dynamic viscosity η (Pa·s) — slope of stress-rate curve above yield.
    pub dynamic_viscosity: f32,
    /// Tait EOS stiffness k (pressure scale factor).
    pub eos_stiffness: f32,
    /// Tait EOS exponent γ (7 for water-like, 1 for linear).
    pub eos_power: f32,
    /// Yield stress τ₀ — shear stress required to initiate flow.
    /// Below this, deviatoric stress is zero (plug flow).
    pub yield_stress: f32,
    /// Minimum shear rate to avoid τ₀/γ̇ singularity.
    /// Particles below this rate are treated as rigid. Default: 1e-4.
    pub critical_shear_rate: f32,
    pub pressure_floor: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Surface tension coefficient γ — adds γ·J·I to Kirchhoff stress.
    /// See `NewtonianFluidMaterial::surface_tension_coeff` for details.
    pub surface_tension_coeff: f32,
    /// Per-step velocity decay: v *= (1 − settling_damping · dt).
    /// Same as `NewtonianFluidMaterial::settling_damping`. 0.0 = off.
    pub settling_damping: f32,
}

impl BinghamFluidMaterial {
    pub fn new(
        rest_density: f32,
        dynamic_viscosity: f32,
        eos_stiffness: f32,
        eos_power: f32,
        yield_stress: f32,
    ) -> Self {
        Self {
            rest_density,
            dynamic_viscosity,
            eos_stiffness,
            eos_power,
            yield_stress,
            critical_shear_rate: 1.0e-4,
            pressure_floor: 0.0,
            min_density: 1.0e-6,
            min_volume: 1.0e-6,
            surface_tension_coeff: 0.0,
            settling_damping: 0.0,
        }
    }

    /// Wet mud: flows under sustained stress, holds shape under small loads.
    /// ρ = 1500 kg/m³, η = 0.5 Pa·s, τ₀ = 100 Pa.
    pub fn mud() -> Self {
        Self::new(1500.0, 0.5, 1.0e4, 7.0, 100.0)
    }

    /// Basaltic lava: extremely viscous, high yield stress.
    /// ρ = 2700 kg/m³, η = 500 Pa·s, τ₀ = 1000 Pa.
    pub fn lava() -> Self {
        Self::new(2700.0, 500.0, 1.0e5, 7.0, 1000.0)
    }

    /// Biological cytoplasm: very low yield, near-Newtonian.
    /// ρ = 1050 kg/m³, η = 0.01 Pa·s, τ₀ = 1 Pa.
    pub fn cytoplasm() -> Self {
        Self::new(1050.0, 0.01, 5.0e2, 7.0, 1.0)
    }

    /// Dense biological fluid (mucus, blood clot).
    /// ρ = 1060 kg/m³, η = 0.1 Pa·s, τ₀ = 10 Pa.
    pub fn mucus() -> Self {
        Self::new(1060.0, 0.1, 1.0e3, 7.0, 10.0)
    }

    /// Compute deviatoric Bingham stress from the APIC velocity gradient C.
    ///
    /// D = (C + Cᵀ)/2 (symmetric strain rate)
    /// γ̇ = √(2·D:D)   (scalar shear rate)
    /// Below yield: returns zero matrix (plug flow).
    /// Above yield: returns (τ₀/γ̇ + η)·D_dev.
    fn deviatoric_stress(&self, c: Mat2) -> Mat2 {
        // Symmetric strain rate D = (C + Cᵀ) / 2
        let sym = c + c.transpose();
        let d = sym * 0.5;

        // Deviatoric: remove isotropic part
        let trace = d.x_axis.x + d.y_axis.y;
        let d_dev = d - Mat2::from_diagonal(Vec2::splat(trace * 0.5));

        // Scalar shear rate γ̇ = √(2·D:D) — Frobenius norm of full D, scaled
        // 2D: D:D = D_xx² + D_yy² + 2·D_xy·D_yx (for symmetric D, 2·D_xy²)
        let d_xx = d.x_axis.x;
        let d_yy = d.y_axis.y;
        let d_xy = d.x_axis.y; // = d.y_axis.x for symmetric D
        let d_sq = d_xx * d_xx + d_yy * d_yy + 2.0 * d_xy * d_xy;
        let shear_rate = (2.0 * d_sq).sqrt();

        if shear_rate < self.critical_shear_rate {
            return Mat2::ZERO;
        }

        // Apparent viscosity: Bingham formula η_app = τ₀/γ̇ + η
        let eta_app = self.yield_stress / shear_rate + self.dynamic_viscosity;
        let tau = d_dev * eta_app;

        // Consistency check: if ||τ||_F < τ₀/√2, particle is in plug regime → zero.
        // (||τ||² for symmetric 2D tensor = τ_xx² + τ_yy² + 2·τ_xy²)
        let tau_xx = tau.x_axis.x;
        let tau_yy = tau.y_axis.y;
        let tau_xy = tau.x_axis.y;
        let tau_sq = tau_xx * tau_xx + tau_yy * tau_yy + 2.0 * tau_xy * tau_xy;
        let tau_yield_sq = self.yield_stress * self.yield_stress * 0.5; // (τ₀/√2)²
        if tau_sq < tau_yield_sq {
            return Mat2::ZERO;
        }

        tau
    }
}

impl MaterialModel for BinghamFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        // Pressure from Tait EOS (same as NewtonianFluid)
        let density = particles.density[i].max(self.min_density);
        let pressure = (self.eos_stiffness
            * ((density / self.rest_density).powf(self.eos_power) - 1.0))
            .max(self.pressure_floor);

        let hydrostatic = Mat2::from_diagonal(Vec2::splat(-pressure));
        let deviatoric = self.deviatoric_stress(particles.velocity_gradient[i]);

        // Surface tension: τ += γ·J·I
        let surface = if self.surface_tension_coeff != 0.0 {
            let f = particles.deformation_gradient[i];
            let j = f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x;
            Mat2::from_diagonal(Vec2::splat(self.surface_tension_coeff * j))
        } else {
            Mat2::ZERO
        };

        hydrostatic + deviatoric + surface
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    fn update_particle(&self, particles: &mut Particles, i: usize, dt: f32) {
        let j = particles.deformation_gradient[i]
            .determinant()
            .clamp(0.5, 2.0);
        let s = j.sqrt();
        particles.deformation_gradient[i] =
            glam::Mat2::from_cols(glam::Vec2::new(s, 0.0), glam::Vec2::new(0.0, s));
        if self.settling_damping > 0.0 {
            particles.v[i] *= 1.0 - (self.settling_damping * dt).min(0.5);
        }
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            dynamic_viscosity: self.dynamic_viscosity,
            compression_limit: self.yield_stress,
            volume_ratio_max: 2.0,
            pressure_floor: self.pressure_floor,
            surface_tension_coeff: self.surface_tension_coeff,
            dp_h0: self.settling_damping,
            ..Default::default()
        }
    }

    fn timestep_bound(
        &self,
        particles: &Particles,
        i: usize,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let density = particles.density[i].max(self.min_density);
        let ratio = (density / self.rest_density.max(self.min_density)).max(1.0e-6);

        let mut dt_bound = f32::INFINITY;

        // Acoustic bound from EOS
        let c2 = self.eos_stiffness * self.eos_power * ratio.powf(self.eos_power - 1.0)
            / self.rest_density.max(self.min_density);
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        // Viscous diffusion bound — apparent viscosity is at least dynamic_viscosity
        if self.dynamic_viscosity > 0.0 {
            let kinematic_viscosity = self.dynamic_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }

        dt_bound
    }

    fn needs_density_recompute(&self) -> bool {
        true
    }
}
