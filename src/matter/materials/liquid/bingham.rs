use glam::{Mat2, Vec2};

use crate::materials::physical_props::{BinghamProps, FromSI};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Bingham viscoplastic fluid.
///
/// Below yield stress τ₀: rigid plug (no deviatoric flow).
/// Above yield stress τ₀: Newtonian with apparent viscosity η_app = τ₀/γ̇ + η.
///
/// Stress decomposition: σ = −p·I + τ_deviatoric
/// Pressure: Tait EOS — p = k·((ρ/ρ₀)^γ − 1), same as NewtonianFluid.
/// Deviatoric:
///   γ̇ = √(2·D_dev:D_dev)   (scalar shear rate, D_dev = deviatoric part of D)
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
    /// Lower bound on the Tait EOS pressure; negative values permit limited
    /// isotropic tension. Default: 0.0 (no tensile pressure).
    ///
    /// HONEST DISCLOSURE (audit 2026-08-15): 0.0 is a free numerical parameter,
    /// chosen empirically for this engine, not a physical consequence of Bingham
    /// rheology and not a literature-calibrated value for mud or another yield-
    /// stress material. The classical model separates total stress as
    /// `sigma_total = -p I + sigma_dev` and applies the Bingham yield condition
    /// to `sigma_dev` (Roquet & Saramito, J. Non-Newtonian Fluid Mech. 155,
    /// 2008, Eqs. 1-2, doi:10.1016/j.jnnfm.2007.12.008); it does not prescribe a
    /// tensile-pressure cutoff. Unlike `NewtonianFluidMaterial`'s `-0.1`, this
    /// default does not trace to the `tmp/sparkl` or `tmp/incremental_mpm`
    /// precedents. Keep it labelled as engine calibration unless tensile data
    /// for a specific viscoplastic material supplies a real value -- the same
    /// honesty convention used for `FORAGING_RECOVERY_RATE` elsewhere.
    pub pressure_floor: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Surface tension coefficient γ — adds γ·J·I to Kirchhoff stress.
    /// See `NewtonianFluidMaterial::surface_tension_coeff` for details.
    pub surface_tension_coeff: f32,
    /// Per-step velocity decay: v *= (1 − settling_damping · dt).
    /// Same as `NewtonianFluidMaterial::settling_damping`. 0.0 = off.
    pub settling_damping: f32,
    /// Physical second (bulk) viscosity ζ, same real term as
    /// `NewtonianFluidMaterial::bulk_viscosity` (see that field's own doc --
    /// Litovitz & Davis, ~3x shear viscosity for water-like liquids). NOT
    /// part of the original pre-`cac544b` file this struct was otherwise
    /// restored from (2026-08-13) -- added back here because it's a real,
    /// separately-sourced fix from later the same investigation
    /// (2026-08-12, acoustic/volumetric oscillation damping), not a
    /// guard-rail bandaid, and dropping it would lose real value.
    pub bulk_viscosity: f32,
}

impl BinghamFluidMaterial {
    pub const fn new(
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
            bulk_viscosity: 0.0,
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
        // Real, confirmed regression (2026-08-17), same class/cause as
        // `NewtonianFluidMaterial::from_physical`'s -- see that method's own
        // doc for the full derivation and regression history. Pressure,
        // yield stress, and viscosity all stay raw SI; only density
        // converts.
        let visc = props.eta_pa_s;
        let tau0 = props.yield_stress_pa;
        let eos = props.bulk_modulus_pa / GAMMA;
        let rho_grid = props.rho_kg_m3 * config.dx_meters * config.dx_meters;
        Self::new(rho_grid, visc, eos, GAMMA, tau0)
    }
}

impl MaterialModel for BinghamFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    // TEMPORARY, explicitly disclosed restoration (2026-08-13) -- same
    // reasoning as `NewtonianFluidMaterial::init_particle`, see that
    // method's own doc for the full live-confirmed root cause.
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density / j;
    }

    /// Same restoration and reasoning as
    /// `NewtonianFluidMaterial::rest_acoustic_c2` -- this material shares the
    /// identical Tait EOS, so it needs the identical rest sound speed.
    fn rest_acoustic_c2(&self) -> Option<f32> {
        if self.eos_stiffness > 0.0 && self.rest_density > 0.0 {
            Some(self.eos_stiffness * self.eos_power / self.rest_density)
        } else {
            None
        }
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        // Pressure from Tait EOS (same as NewtonianFluid). Clamp density both
        // ways, matching `NewtonianFluidMaterial::kirchhoff_stress` exactly:
        // min prevents div-by-zero at low PPC, max (2x rho0) limits how far
        // the EOS pressure response saturates under impact overcompression.
        // Matches `NewtonianFluidMaterial::kirchhoff_stress`'s clamp exactly --
        // same Tait EOS, same unbounded-pressure-spike risk under violent
        // compression; see that file's doc for why 2x specifically.
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

        // Bulk viscosity: τ += ζ·div(v)·I -- see field's own doc.
        let bulk = if self.bulk_viscosity > 0.0 {
            let gradient = particles.velocity_gradient[i];
            let div_v = gradient.x_axis.x + gradient.y_axis.y;
            Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5))
        } else {
            Mat2::ZERO
        };

        // Artificial (shock) viscosity -- same real PDE term, same citation,
        // as `NewtonianFluidMaterial::kirchhoff_stress` (see
        // `artificial_bulk_viscosity`'s own doc). This material shares the
        // identical Tait EOS, so it needs the identical shock-capturing term;
        // it was lost from the CPU path by the same wholesale revert.
        let gradient = particles.velocity_gradient[i];
        let div_v_true = gradient.x_axis.x + gradient.y_axis.y;
        let j_now = crate::materials::liquid::fluid::volume_j(
            particles.initial_volume[i],
            particles.volume[i],
            "BinghamFluidMaterial",
        );
        let q = crate::materials::liquid::fluid::artificial_bulk_viscosity(
            self.eos_stiffness,
            self.eos_power,
            self.rest_density,
            j_now,
            div_v_true,
            1.0,
        );
        let shock = Mat2::from_diagonal(Vec2::splat(-q));

        hydrostatic + deviatoric + surface + bulk + shock
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // REAL BUG, found+fixed for real 2026-08-13: this material's comment
        // already said it "carries the identical dead-EOS-pressure bug" that
        // `NewtonianFluidMaterial::update_particle` was fixed for 2026-08-06
        // -- but the fix itself was never actually ported here, only noted
        // as still-outstanding. Live-confirmed consequence tonight: mud's
        // `deformation_gradient` updates normally (J drifts, e.g. 0.9989)
        // but `volume`/`density` never move from their spawn values (stuck
        // at exactly V/V0=1 forever) -- an internally inconsistent state
        // that trips `assert_owned_deformation_state`
        // ("det(F)=0.9989268, V/V0=1"). This is the user's own live
        // observation exactly: water settles correctly, mud does not.
        let f_trial = (Mat2::IDENTITY + dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;
        let j = f_trial.determinant().clamp(0.5, 2.0);
        let s = j.sqrt();
        *ctx.deformation_gradient =
            glam::Mat2::from_cols(glam::Vec2::new(s, 0.0), glam::Vec2::new(0.0, s));
        if self.settling_damping > 0.0 {
            *ctx.v *= 1.0 - (self.settling_damping * dt).min(0.5);
        }
        let density = (self.rest_density / j)
            .max(self.min_density)
            .min(self.rest_density * 2.0);
        *ctx.density = density;
        *ctx.volume = (ctx.mass / density).max(1.0e-9);
    }

    // TEMPORARY, explicitly disclosed restoration (2026-08-13) -- same
    // reasoning as `NewtonianFluidMaterial::owns_deformation_volume_state`,
    // see that method's own doc for the full live-confirmed root cause.
    fn owns_deformation_volume_state(&self) -> bool {
        true
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
            dp_h0: self.settling_damping,
            bulk_viscosity: self.bulk_viscosity,
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
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

    /// `false` -- same reasoning and same 2026-08-13 fix as
    /// `NewtonianFluidMaterial::needs_density_recompute`; see that method's
    /// own doc. This material owns `rho = rho0/J` through `init_particle` /
    /// `update_particle` and declares it via `owns_deformation_volume_state`,
    /// so the kernel-density gather was computed for it and then discarded.
    fn needs_density_recompute(&self) -> bool {
        false
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
