use glam::{Mat2, Vec2};

use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::Particles;

/// Weakly-compressible Newtonian fluid (Tait EOS + deviatoric viscosity).
/// Refs: Becker & Teschner 2007 (WCSPH), Hu et al. 2018 (MLS-MPM).
#[derive(Debug, Clone, Copy)]
pub struct NewtonianFluidMaterial {
    pub rest_density: f32,
    pub dynamic_viscosity: f32,
    pub eos_stiffness: f32,
    pub eos_power: f32,
    pub pressure_floor: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Thermal thinning: µ_eff = dynamic_viscosity · exp(−thermal_viscosity_coeff · T).
    /// 0.0 = isothermal. Positive values make the fluid flow easier when hot.
    pub thermal_viscosity_coeff: f32,
    /// Bulk viscosity ζ (second viscosity, Pa·s in physical units).
    ///
    /// Adds τ += ζ·(∇·v)·I to Kirchhoff stress — damps compression waves (acoustic damping).
    /// Physical: Navier-Stokes second viscosity, distinct from shear viscosity µ.
    /// Stokes assumption (ζ=0) holds for dilute ideal gases; real liquids have ζ > 0.
    /// For water: ζ ≈ 3e-3 Pa·s (Dukhin & Goetz 2009). In simulation units set to
    /// ~0.5–5× dynamic_viscosity. 0.0 = no acoustic damping.
    pub bulk_viscosity: f32,
    /// Surface tension coefficient γ (N/m in physical units).
    ///
    /// Adds isotropic Kirchhoff stress τ += γ·J·I — continuum surface energy ψ = γ·J.
    /// Reference: Ziran 2020, `SurfaceTension.h` (Chenfanfu Jiang group).
    ///
    /// **Limitation**: curvature-free. Young-Laplace gives Δp = γ·κ (interface curvature κ),
    /// but MPM particles carry no interface normal. This term resists volumetric compression
    /// isotropically — sufficient for cohesion/droplet stability, not for curvature-driven
    /// flow (e.g. Rayleigh-Plateau instability). 0.0 = disabled.
    pub surface_tension_coeff: f32,
    /// Per-step velocity decay: v *= (1 − settling_damping · dt).
    ///
    /// Damps residual sloshing and slow plastic creep without affecting fast flow.
    /// 0.0 = off (default). 0.05–0.2 for water, 0.1–0.5 for mud/viscous fluids.
    /// Implemented in the GPU shader via the `dp_h0` slot (unused for fluids).
    pub settling_damping: f32,
}

impl NewtonianFluidMaterial {
    pub fn new(
        rest_density: f32,
        dynamic_viscosity: f32,
        eos_stiffness: f32,
        eos_power: f32,
    ) -> Self {
        Self {
            rest_density,
            dynamic_viscosity,
            eos_stiffness,
            eos_power,
            pressure_floor: -0.1,
            min_density: 1.0e-6,
            min_volume: 1.0e-6,
            thermal_viscosity_coeff: 0.0,
            bulk_viscosity: 0.0,
            surface_tension_coeff: 0.0,
            settling_damping: 0.0,
        }
    }

    /// Preset: water-like fluid at the given rest density.
    ///
    /// Parameters: γ=7 (Tait EOS exponent for water), µ≈1e-3 Pa·s (physical water viscosity
    /// at 20°C). `eos_stiffness` controls incompressibility — higher = stiffer; 1e4 works
    /// well at emerge's default grid scale. Reference: Becker & Teschner 2007 §4.
    pub fn water(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 1.0e-3, eos_stiffness, 7.0)
    }
}

impl MaterialModel for NewtonianFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        // Clamp density both ways: min prevents div-by-zero at low PPC,
        // max (2×ρ₀) prevents pressure spikes when particles are over-compressed on impact.
        let density = particles.density[i]
            .max(self.min_density)
            .min(self.rest_density * 2.0);
        let pressure = (self.eos_stiffness
            * ((density / self.rest_density).powf(self.eos_power) - 1.0))
            .max(self.pressure_floor);

        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure));

        let eff_viscosity = if self.thermal_viscosity_coeff > 0.0 {
            self.dynamic_viscosity
                * (-self.thermal_viscosity_coeff * particles.temperature[i]).exp()
        } else {
            self.dynamic_viscosity
        };
        let c = particles.velocity_gradient[i];
        let sym_strain = c + c.transpose();
        let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y; // = 2·tr(D) = 2·∇·v
        let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
        stress += eff_viscosity * strain_dev;

        // Bulk viscosity ζ: τ += ζ·(∇·v)·I — damps longitudinal/acoustic waves.
        // ∇·v ≈ div_v/2 (div_v here is trace of sym_strain = C+Cᵀ = 2D, so ∇·v = div_v/2).
        if self.bulk_viscosity > 0.0 {
            stress += Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5));
        }

        if self.surface_tension_coeff != 0.0 {
            let f = particles.deformation_gradient[i];
            let j = f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x;
            stress += Mat2::from_diagonal(Vec2::splat(self.surface_tension_coeff * j));
        }

        stress
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
            thermal_viscosity_coeff: self.thermal_viscosity_coeff,
            // Free-surface J cap: GPU clamps det(F) to [J_MIN, volume_ratio_max].
            // 2.0 = realistic free-surface density (half rest_density with no restoring EOS force).
            volume_ratio_max: 2.0,
            pressure_floor: self.pressure_floor,
            bulk_viscosity: self.bulk_viscosity,
            surface_tension_coeff: self.surface_tension_coeff,
            dp_h0: self.settling_damping, // fluid repurposes dp_h0 for settling damping (DP unused)
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
        const MIN_DENSITY_RATIO: f32 = 1.0e-6;
        let density = particles.density[i].max(self.min_density);
        let ratio = (density / self.rest_density.max(self.min_density)).max(MIN_DENSITY_RATIO);

        let mut dt_bound = f32::INFINITY;

        // Acoustic timestep bound from EOS derivative dp/drho.
        let c2 = self.eos_stiffness * self.eos_power * ratio.powf(self.eos_power - 1.0)
            / self.rest_density.max(self.min_density);
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        // Viscous diffusion bound for explicit integration.
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
