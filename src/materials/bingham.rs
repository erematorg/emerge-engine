use glam::{Mat2, Vec2};

use crate::materials::physical_props::{BinghamProps, FromSI, scale_stress, scale_visc};
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

    /// High yield stress, low viscosity: τ₀=100 Pa, η=0.5 Pa·s. Wet mud regime.
    pub fn high_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 0.5, eos_stiffness, 7.0, 100.0)
    }

    /// High yield stress, high viscosity: τ₀=1000 Pa, η=500 Pa·s. Basaltic lava regime.
    pub fn viscous_high_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 500.0, eos_stiffness, 7.0, 1000.0)
    }

    /// Low yield stress, low viscosity: τ₀=1 Pa, η=0.01 Pa·s. Biological cytoplasm regime.
    pub fn low_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 0.01, eos_stiffness, 7.0, 1.0)
    }

    /// Medium yield stress, medium viscosity: τ₀=10 Pa, η=0.1 Pa·s. Dense biological fluid regime.
    pub fn medium_yield(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 0.1, eos_stiffness, 7.0, 10.0)
    }

    /// Compute deviatoric Bingham stress from the APIC velocity gradient C.
    ///
    /// D = (C + Cᵀ)/2 (symmetric strain rate)
    /// γ̇ = √(2·D_dev:D_dev) (scalar shear rate — deviatoric only: a yield criterion
    /// must not respond to pure volumetric expansion/compression, which isn't shear)
    /// Below yield: returns zero matrix (plug flow).
    /// Above yield: returns (τ₀/γ̇ + η)·D_dev.
    fn deviatoric_stress(&self, c: Mat2) -> Mat2 {
        // Symmetric strain rate D = (C + Cᵀ) / 2
        let sym = c + c.transpose();
        let d = sym * 0.5;

        // Deviatoric: remove isotropic part
        let trace = d.x_axis.x + d.y_axis.y;
        let d_dev = d - Mat2::from_diagonal(Vec2::splat(trace * 0.5));

        // Scalar shear rate γ̇ = √(2·D_dev:D_dev) — Frobenius norm of deviatoric D, scaled.
        let d_xx = d_dev.x_axis.x;
        let d_yy = d_dev.y_axis.y;
        let d_xy = d_dev.x_axis.y; // = d_dev.y_axis.x for symmetric D
        let d_sq = d_xx * d_xx + d_yy * d_yy + 2.0 * d_xy * d_xy;
        let shear_rate = (2.0 * d_sq).sqrt();

        if shear_rate < self.critical_shear_rate {
            return Mat2::ZERO;
        }

        // Apparent viscosity: Bingham formula η_app = τ₀/γ̇ + η
        let eta_app = self.yield_stress / shear_rate + self.dynamic_viscosity;
        d_dev * eta_app
    }
}

impl FromSI<BinghamProps> for BinghamFluidMaterial {
    fn from_physical(props: &BinghamProps, config: &crate::SimConfig) -> Self {
        // Tait EOS polytropic exponent -- Cole 1948, "Underwater Explosions"; standard
        // in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994). Applies to the
        // volumetric/EOS part of a Bingham fluid same as any other weakly-compressible
        // liquid; the yield-stress physics (tau0 below) is separate and unaffected.
        const GAMMA: f32 = 7.0;
        let visc = scale_visc(props.eta_pa_s, props.rho_kg_m3, config);
        let tau0 = scale_stress(props.yield_stress_pa, props.rho_kg_m3, config);
        let eos = scale_stress(props.bulk_modulus_pa / GAMMA, props.rho_kg_m3, config);
        // See `NewtonianFluidMaterial::from_physical`'s fix doc (2026-07-07) -- rest_density
        // must match `particles.density[i]`'s real units, not an extra `/dt_seconds^2`.
        let rho_grid = props.rho_kg_m3 * config.dx_meters * config.dx_meters;
        Self::new(rho_grid, visc, eos, GAMMA, tau0)
    }
}

impl MaterialModel for BinghamFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        // Pressure from Tait EOS (same as NewtonianFluid). Clamp density both
        // ways, matching `NewtonianFluidMaterial::kirchhoff_stress` exactly:
        // min prevents div-by-zero at low PPC, max (2x rho0) limits how far
        // the EOS pressure response saturates under impact overcompression.
        // FIXED 2026-07-07 (real, found via audit): this material was missing
        // the upper clamp entirely despite sharing the identical Tait EOS
        // formula with NewtonianFluidMaterial (which has it) -- an unbounded
        // EOS pressure spike under violent compression is the same real risk
        // here. Matches NewtonianFluid's 2x (see that file's doc for the real
        // A/B-tested reason 2x, not a looser value, is correct).
        let density = particles.density[i]
            .max(self.min_density)
            .min(self.rest_density * 2.0);
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
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let density = density.max(self.min_density);
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

#[cfg(test)]
mod analytical_validation_tests {
    use super::*;

    /// **Deviatoric stress must match the Bingham formula exactly** (Bingham
    /// 1916: below yield, rigid plug/zero stress; above yield, Newtonian with
    /// apparent viscosity `tau0/shear_rate + eta`). `BinghamFluidMaterial` had
    /// zero test comparing its deviatoric response to this analytical formula
    /// directly (only whole-simulation stability checks existed).
    #[test]
    fn below_critical_shear_rate_gives_zero_deviatoric_stress() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        // A tiny, sub-critical shear rate: pure shear C with a small magnitude.
        let c = Mat2::from_cols(Vec2::new(0.0, 1.0e-6), Vec2::new(1.0e-6, 0.0));
        let tau = mat.deviatoric_stress(c);
        assert_eq!(
            tau,
            Mat2::ZERO,
            "sub-critical shear rate must give exactly zero deviatoric stress (rigid plug)"
        );
    }

    #[test]
    fn above_yield_matches_bingham_formula_exactly() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        // Pure shear strain rate: C = [[0, g], [g, 0]] gives D=C (already symmetric),
        // D_dev=D (already traceless), d_xx=d_yy=0, d_xy=g, d_sq=2*g^2,
        // shear_rate=sqrt(2*2*g^2)=2*g (real, hand-derivable from the formula).
        let g = 5.0_f32;
        let c = Mat2::from_cols(Vec2::new(0.0, g), Vec2::new(g, 0.0));
        let tau = mat.deviatoric_stress(c);

        let shear_rate = 2.0 * g;
        let eta_app = mat.yield_stress / shear_rate + mat.dynamic_viscosity;
        let d_dev = Mat2::from_cols(Vec2::new(0.0, g), Vec2::new(g, 0.0)); // D_dev = D here
        let predicted = d_dev * eta_app;

        let diff = tau - predicted;
        let err = (diff.x_axis.length_squared() + diff.y_axis.length_squared()).sqrt();
        assert!(
            err < 1.0e-3,
            "above-yield deviatoric stress should match tau0/gamma_dot+eta exactly: \
             predicted={predicted:?} actual={tau:?}"
        );
    }

    /// Real, checkable monotonic claim: apparent viscosity (and thus deviatoric
    /// stress magnitude at a FIXED shear rate) must DECREASE as shear rate
    /// increases -- shear-thinning behavior intrinsic to the Bingham model
    /// (tau0/gamma_dot term shrinks as gamma_dot grows), not an assumption.
    #[test]
    fn apparent_viscosity_decreases_as_shear_rate_increases() {
        let mat = BinghamFluidMaterial::new(1000.0, 0.5, 5000.0, 7.0, 100.0);
        let tau_slow =
            mat.deviatoric_stress(Mat2::from_cols(Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0)));
        let tau_fast =
            mat.deviatoric_stress(Mat2::from_cols(Vec2::new(0.0, 10.0), Vec2::new(10.0, 0.0)));

        // Stress DOES grow with shear rate overall (more strain rate -> more
        // stress), but the EFFECTIVE viscosity (stress/shear_rate) must shrink --
        // check the ratio, not the raw magnitude.
        let eff_visc_slow = tau_slow.x_axis.y / 2.0; // shear_rate=2*g=2 here
        let eff_visc_fast = tau_fast.x_axis.y / 20.0; // shear_rate=2*g=20 here
        assert!(
            eff_visc_fast < eff_visc_slow,
            "apparent viscosity must decrease as shear rate increases (shear-thinning): \
             slow={eff_visc_slow:.4} fast={eff_visc_fast:.4}"
        );
    }
}
