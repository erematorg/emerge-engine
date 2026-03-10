use glam::{Mat2, Vec2};

use crate::solver::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::state::particle::Particle;

/// Weakly-compressible Newtonian fluid.
///
/// Pressure: Tait equation of state — p = k·((ρ/ρ₀)^γ − 1).
///   Reference: Monaghan 1994 (SPH), Becker & Teschner 2007 (WCSPH), γ≈7 for water.
/// Viscosity: deviatoric Newtonian stress — τ_visc = η·dev(Ċ + Ċᵀ).
///   C (the APIC affine matrix) approximates the local velocity gradient.
/// Coupled to MLS-MPM transfer: Hu et al. 2018, §4.
#[derive(Debug, Clone, Copy)]
pub struct NewtonianFluidMaterial {
    pub rest_density: f32,
    pub dynamic_viscosity: f32,
    pub eos_stiffness: f32,
    pub eos_power: f32,
    pub pressure_floor: f32,
    pub min_density: f32,
    pub min_volume: f32,
    pub velocity_damping: f32,
    pub affine_damping: f32,
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
            velocity_damping: 1.0,
            affine_damping: 1.0,
        }
    }
}

impl MaterialModel for NewtonianFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn kirchhoff_stress(&self, particle: &Particle) -> Mat2 {
        let density = particle.density.max(self.min_density);
        let pressure = (self.eos_stiffness
            * ((density / self.rest_density).powf(self.eos_power) - 1.0))
            .max(self.pressure_floor);

        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure));

        // Viscous stress acts on deviatoric strain rate only — shear resistance, not bulk.
        // Full strain would add spurious bulk viscosity; real Newtonian fluids have none.
        let sym_strain = particle.affine + particle.affine.transpose();
        let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(sym_strain.x_axis.x + sym_strain.y_axis.y) * 0.5);
        stress += self.dynamic_viscosity * strain_dev;
        stress
    }

    fn stress_volume(&self, particle: &Particle) -> f32 {
        particle.volume.max(self.min_volume)
    }

    fn update_particle(&self, particle: &mut Particle, _dt: f32) {
        particle.v *= self.velocity_damping;
        particle.affine *= self.affine_damping;
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Fluid as u32,
            rest_density: self.rest_density,
            eos_stiffness: self.eos_stiffness,
            eos_power: self.eos_power,
            dynamic_viscosity: self.dynamic_viscosity,
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
        const MIN_DENSITY_RATIO: f32 = 1.0e-6;
        let density = particle.density.max(self.min_density);
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
}
