use glam::{Mat2, Vec2};

pub mod damage;

use crate::materials::lame_parameters::lame_from_young;
use crate::materials::physical_props::{BrittleProps, FromSI};
use crate::materials::solid::corotated::stress::corotated_elastic_stress;
use crate::materials::solid::rankine::damage::RANKINE_MIN_RESIDUAL_TENSILE_FRACTION;
use crate::materials::solid::rankine::damage::rankine_damage_saturation_point;
use crate::materials::solid::rankine::damage::stress_to_hencky;
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::matter::kinematics::deformation_gradient::advance_deformation_gradient;
use crate::matter::kinematics::hencky_strain::hencky_strains;
use crate::matter::kinematics::volume_ratio::MIN_J;
use crate::particle::{ParticleUpdateCtx, Particles};
use crate::spacetime::solver::cfl::elastic_wave_dt;
use crate::tensor_algebra::svd::reconstruct_f;
use crate::tensor_algebra::svd::svd2;

/// Kelvin-Voigt viscosity (SI Pa.s) for `RankineMaterial::elastic_viscosity`,
/// from a material's measured seismic or ultrasonic quality factor Q, the way
/// solid-earth and ice physics report internal damping (soil uses a damping
/// ratio instead, `granular::sand::small_strain_elastic_viscosity_pa_s`; same
/// equivalent-viscous conversion).
///
/// `zeta = 1/(2*Q)` (Aki & Richards, "Quantitative Seismology," 2002). The
/// textbook `eta = 2*zeta*G/omega` assumes `sigma = 2G*eps + 2*eta*D`; this
/// engine's `kirchhoff_stress` applies `eta*D_dev` (see `ViscoelasticMaterial`'s
/// `viscous_term_matches_eta_times_deviatoric_strain_rate_exactly`), so here
/// `eta = 2*G/(Q*omega)`, which reproduces the cited Q exactly
/// (`measured_q_factor_matches_target_after_the_conversion_fix`); the
/// textbook form gives twice the Q.
///
/// `reference_frequency_hz` must be the frequency of the cited Q measurement:
/// Q depends on frequency in polycrystalline solids (grain-boundary friction
/// loss ~linear, scattering loss ~quartic), so the conversion is exact only
/// there, like `small_strain_elastic_viscosity_pa_s` at its 1 Hz.
///
/// Returns SI Pa.s. `kirchhoff_stress` adds `elastic_viscosity * d_dev` into
/// the stress tensor built from `lambda`/`mu`, so convert it the same way:
/// raw `lame_from_young`/`from_young_modulus` values take it raw;
/// density-normalized `lame_from_si` values take
/// `SimConfig::visc_from_si(eta, rho)` (as `examples/cpu/
/// sand_water_saturation.rs`). `RankineMaterial::ice` builds raw values, so
/// its call sites (`phase_states_gui.rs`, `phase_states_headless.rs`) assign
/// the result raw; converting it divided the damping by ~917 there.
pub fn q_factor_elastic_viscosity_pa_s(
    shear_modulus_pa: f32,
    quality_factor: f32,
    reference_frequency_hz: f32,
) -> f32 {
    debug_assert!(
        quality_factor > 0.0,
        "quality factor {quality_factor} must be positive -- Q<=0 is not a real material"
    );
    let omega = 2.0 * std::f32::consts::PI * reference_frequency_hz;
    2.0 * shear_modulus_pa / (quality_factor * omega)
}

/// P-wave quality factor of cold polycrystalline ice. Bentley & Kohnen
/// (1976), "Seismic refraction measurements of internal friction in
/// Antarctic ice," Journal of Geophysical Research 81(9):1519-1526, measured
/// Q_P ~= 715 at 136 Hz (Byrd Station, ~-28C, 100-500 m depth); Peters et al.
/// (2012), "Seismic attenuation in glacial ice: A proxy for englacial
/// temperature," JGR Earth Surface, report Q_P ~ 500-1700 for cold Antarctic
/// ice at the same site. A typical value near the low end of both.
///
/// Limitation: Q drops sharply toward the melting point (grain-boundary
/// sliding and premelting dominate); Peters et al. 2012 report Q_P ~ 65 for
/// temperate ice near 0C (Athabasca Glacier). This matches `ice()`'s -10C
/// reference, so ice near 0C dissipates faster than it implies.
pub const ICE_QUALITY_FACTOR_Q: f32 = 700.0;

/// The real measurement frequency Bentley & Kohnen (1976) used -- see
/// `ICE_QUALITY_FACTOR_Q`'s doc. Pair the two together, never `Q`
/// alone, at a different reference frequency.
pub const ICE_Q_REFERENCE_FREQUENCY_HZ: f32 = 136.0;

/// Rankine (maximum principal stress) elastoplastic material -- brittle tensile failure.
///
/// Elastic response: corotated linear elastic (same as DruckerPragerMaterial / VonMisesMaterial).
/// Yield criterion: max(τ₁, τ₂) ≤ σ_t_eff, where τᵢ are principal Kirchhoff stresses and
///   σ_t_eff = max(tensile_strength · exp(−softening_rate · damage), tensile_strength · 5%)
///   (exponential softening, floored at a small residual so damage saturates under
///   sustained loading instead of ratcheting forever -- see `RANKINE_MIN_RESIDUAL_TENSILE_FRACTION`).
///
/// Return mapping: when a principal stress exceeds σ_t_eff, it is projected back to the
/// tensile cutoff surface; the remaining stress component is unaffected (1D projection).
/// Biaxial tension (both τ₁ > σ_t AND τ₂ > σ_t) projects at the corner -- both set to σ_t.
///
/// Damage accumulates in `Particle::friction_hardening` (repurposed as damage), bounded
/// above by `rankine_damage_saturation_point(softening_rate)` -- the point past which
/// `t_eff` is already at its residual floor, so further stress cannot lower it any
/// more and additional accumulation would be pure bookkeeping, not physical (see
/// `RANKINE_MIN_RESIDUAL_TENSILE_FRACTION` doc). `softening_rate <= 0` (hard cutoff,
/// no softening) has no saturation point -- damage can grow unbounded in that case,
/// same as before.
/// Softening reduces effective tensile strength exponentially toward a small residual.
///
/// Suitable for: brittle rock, bone, eggshell, chitin, ice with fracture.
/// At zero softening rate: perfect tensile cutoff (material can never exceed σ_t).
///
/// References: Rankine 1876 (original criterion); Wolper et al. 2019 (MPM brittle fracture);
/// sparkl `RankinePlasticity` (Rust open-source reference, Apache-2.0).
#[derive(Debug, Clone, Copy)]
pub struct RankineMaterial {
    /// Rest density in grid units (`rho / reference density`, the same
    /// scale `SpawnRegion::mass_from` gives mass), when built from SI
    /// properties; `None` for grid-unit constructors. Contact reads it to
    /// size a particle's undeformed domain, see
    /// [`MaterialModel::rest_density`].
    pub rest_density: Option<f32>,
    pub lambda: f32,
    pub mu: f32,
    /// Maximum tensile principal Kirchhoff stress (compressive stress is unlimited).
    /// Rock: 1e3–1e4. Bone: 1e4–1e5. Ice: 1e2–1e3.
    pub tensile_strength: f32,
    /// Exponential softening rate. 0.0 = no softening (hard cutoff).
    /// Positive values reduce σ_t as damage accumulates.
    /// Typical: 0.5–5.0 -- higher = more brittle (strength collapses fast after first crack).
    pub softening_rate: f32,
    /// Kelvin-Voigt viscous damping on the deviatoric elastic strain rate
    /// (SI Pa.s, converted like `lambda`/`mu`: raw after `lame_from_young`,
    /// `SimConfig::visc_from_si` after `lame_from_si`, see
    /// `q_factor_elastic_viscosity_pa_s`), the formula of
    /// `DruckerPragerMaterial::elastic_viscosity`. `0.0` (every preset's
    /// default) = off.
    ///
    /// Without it an elastic-plus-brittle-fracture model dissipates nothing
    /// below the fracture threshold, while real solids lose energy to
    /// internal friction (measured as Q): `ice()` at 0.0 bounces almost
    /// elastically on any sub-fracture impact, and a borderline impact looks
    /// like a bounce with partial fracture.
    pub elastic_viscosity: f32,
    /// Measured optical coefficients (absorption and reduced scattering,
    /// `m^-1`) the caller declares for what this material represents, or
    /// `None`. Filled from `matter::materials::optical`'s datasets
    /// (`pure_ice`, `snow`, ...), never chosen by the model: see
    /// `NewtonianFluidMaterial::optics` for why this is not a substance flag.
    pub optics: Option<crate::energy::radiation::OpticalCoefficientsSi>,
    /// Specific heat capacity `c_p`, J/(kg*K). 0 (the default) means
    /// undeclared -- see `MaterialModel::specific_heat_j_kg_k`. Declared by
    /// the caller like `optics`, from `matter::materials::heat_capacity`
    /// (`ICE_IH_AT_MELTING_POINT_J_KG_K` for `ice()`), never by a preset.
    pub specific_heat_j_kg_k: f32,
}

impl RankineMaterial {
    /// Construct directly from grid-native Lame parameters and tensile
    /// strength -- NOT SI Pascals (see [`Self::from_young_modulus`] for the
    /// common gotcha and the real SI conversion path).
    pub const fn new(lambda: f32, mu: f32, tensile_strength: f32, softening_rate: f32) -> Self {
        Self {
            rest_density: None,
            lambda,
            mu,
            tensile_strength,
            softening_rate,
            elastic_viscosity: 0.0,
            optics: None,
            specific_heat_j_kg_k: 0.0,
        }
    }

    /// **Grid units, not pascals**: calls [`lame_from_young`] directly and
    /// never touches `dx_meters` or density. For an SI material use
    /// [`Self::from_physical`] (needs a `&SimConfig` and `rho_kg_m3`).
    pub fn from_young_modulus(
        young_modulus: f32,
        poisson_ratio: f32,
        tensile_strength: f32,
        softening_rate: f32,
    ) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu, tensile_strength, softening_rate)
    }

    /// Brittle rock regime: tensile strength as a fraction of the caller's
    /// `young_modulus`, not an absolute number, which would only mean rock at one E
    /// (a fixed tensile=500 gives an 18-50% tensile/E ratio at the moduli the tests
    /// pass). Brittle rock's tensile-to-modulus ratio is ~2-3e-4: granite/basalt,
    /// E~50 GPa, tensile strength~10-15 MPa (Goodman 1989, "Introduction to Rock
    /// Mechanics"). Fast softening_rate=2.0: brittle failure propagates quickly.
    pub fn stiff_brittle(young_modulus: f32, poisson_ratio: f32) -> Self {
        const ROCK_TENSILE_TO_MODULUS_RATIO: f32 = 2.5e-4;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * ROCK_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Bone regime: tensile strength as a fraction of `young_modulus`, as in
    /// `stiff_brittle`. Cortical bone tolerates a much higher tensile-to-modulus
    /// ratio than rock (a tougher composite): E~15-20 GPa, tensile strength~100-150
    /// MPa, ratio ~7e-3 (Currey 2002, "Bones: Structure and Mechanics"). Slower
    /// softening_rate=1.0: bone fails less abruptly than rock.
    pub fn high_tensile(young_modulus: f32, poisson_ratio: f32) -> Self {
        const BONE_TENSILE_TO_MODULUS_RATIO: f32 = 7.0e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * BONE_TENSILE_TO_MODULUS_RATIO,
            1.0,
        )
    }

    /// Sandstone regime: sedimentary clastic rock, tensile strength as a fraction of
    /// `young_modulus` as in `stiff_brittle`. E range 11.3-40 GPa (avg ~19.9 GPa),
    /// tensile strength 19.17-65.66 MPa (Xu 2016, "Characterization of Rock
    /// Mechanical Properties Using Lab Tests and Numerical Interpretation Model of
    /// Well Logs"), a wide spread from cementation and porosity. Representative pick
    /// near the lower/typical end of both ranges (E~20 GPa, tensile~20 MPa).
    /// softening_rate=2.0 as `stiff_brittle`: brittle failure, no cited reason to
    /// differ.
    pub fn sandstone(young_modulus: f32, poisson_ratio: f32) -> Self {
        const SANDSTONE_TENSILE_TO_MODULUS_RATIO: f32 = 1.0e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * SANDSTONE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Limestone regime: sedimentary chemical rock. E range 4.6-12 GPa, tensile
    /// strength 18.00-38.76 MPa (same Xu 2016 source as `sandstone`): softer than
    /// sandstone and stronger in tension per unit modulus. Representative pick
    /// E~8 GPa, tensile~25 MPa.
    pub fn limestone(young_modulus: f32, poisson_ratio: f32) -> Self {
        const LIMESTONE_TENSILE_TO_MODULUS_RATIO: f32 = 3.1e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * LIMESTONE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Shale regime: sedimentary clastic, fissile/foliated. E range 15-36.9 GPa
    /// (avg ~27 GPa, foliated), tensile strength ~168 MPa average across foliation
    /// (same Xu 2016 source). Limitation: shale is strongly anisotropic (it splits
    /// far more easily along bedding planes than across them; the source notes that
    /// laminated shale shows lower values, without a number). This preset is
    /// isotropic (the yield surface has no per-particle orientation field), so it
    /// represents the across-foliation (stronger) direction; bedding-plane weakness
    /// needs an anisotropic yield surface, which is not built.
    pub fn shale(young_modulus: f32, poisson_ratio: f32) -> Self {
        const SHALE_TENSILE_TO_MODULUS_RATIO: f32 = 6.2e-3;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * SHALE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Ice regime, with the same strength-to-modulus-ratio pattern as the rock
    /// presets. Ice belongs here, not in `StomakhinMaterial` (snow): snow's law
    /// is compaction hardening (crushing air pockets between crystals), which
    /// solid, non-porous ice does not have; ice cracks rather than crushes, the
    /// failure mode this material models for rock and bone. E for
    /// polycrystalline ice is 9.0-11.2 GPa at -10C (random polycrystals ~9.0
    /// GPa at -5C; granular polycrystalline ice 9.3 GPa at 263 K); 9.0 GPa is
    /// the typical low end. Tensile strength 0.7-3.1 MPa from -10C to -20C
    /// (Petrovic 2003, "Review: mechanical properties of ice and snow," J.
    /// Materials Science 38), ~1 MPa as an engineering estimate: a ratio ~1.1e-4,
    /// the order of rock's 2.5e-4. softening_rate=2.0 as for the rocks: ice
    /// fractures abruptly, not like bone.
    ///
    /// Explicit MPM must resolve `c = sqrt(E/rho)`, ~3130 m/s here, close to
    /// ice's measured bar longitudinal speed, 3163 m/s (Northwood 1947,
    /// "Propagation of Elastic Waves in Ice"; deep-ice P waves, a different
    /// mode, run 3410-3878 m/s). Scenes need a correspondingly fine substep,
    /// as for `stiff_brittle`/`sandstone`/`shale`; the stiffness is not
    /// reduced.
    ///
    /// `elastic_viscosity` stays `0.0`: presets are functions of
    /// `young_modulus` alone (as `DruckerPragerMaterial`'s, see
    /// `examples/cpu/sand_water_saturation.rs`). A scene under gravity or
    /// impacts sets it, raw SI Pa.s, no `SimConfig` conversion (see
    /// `q_factor_elastic_viscosity_pa_s`):
    /// ```ignore
    /// let g = young_modulus / (2.0 * (1.0 + poisson_ratio));
    /// let eta_pa_s = q_factor_elastic_viscosity_pa_s(g, ICE_QUALITY_FACTOR_Q, ICE_Q_REFERENCE_FREQUENCY_HZ);
    /// RankineMaterial { elastic_viscosity: eta_pa_s, ..RankineMaterial::ice(young_modulus, poisson_ratio) }
    /// ```
    pub fn ice(young_modulus: f32, poisson_ratio: f32) -> Self {
        const ICE_TENSILE_TO_MODULUS_RATIO: f32 = 1.1e-4;
        Self::from_young_modulus(
            young_modulus,
            poisson_ratio,
            young_modulus * ICE_TENSILE_TO_MODULUS_RATIO,
            2.0,
        )
    }

    /// Effective tensile strength after damage softening. Floored at a small
    /// residual fraction of virgin strength -- see `RANKINE_MIN_RESIDUAL_TENSILE_FRACTION`
    /// doc for why an unfloored exponential decay is an unbounded damage ratchet.
    #[inline]
    fn tensile_strength_eff(&self, damage: f32) -> f32 {
        (self.tensile_strength * (-self.softening_rate * damage).exp())
            .max(self.tensile_strength * RANKINE_MIN_RESIDUAL_TENSILE_FRACTION)
    }
}

impl FromSI<BrittleProps> for RankineMaterial {
    fn from_physical(props: &BrittleProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = config.lame_from_si(props.elastic.e_pa, props.elastic.nu);
        let ts = config.stress_from_si(props.tensile_strength_pa);
        Self {
            rest_density: Some(props.elastic.rho_kg_m3 / config.reference_density_kg_m3),
            ..Self::new(lambda, mu, ts, props.softening_rate)
        }
    }
}

impl RankineMaterial {
    /// Rankine return mapping in 2D principal stress space.
    ///
    /// Returns (projected_tau, yielded) -- `yielded` is true if any projection occurred.
    #[inline]
    fn project_stress(&self, tau: Vec2, t_eff: f32) -> (Vec2, bool) {
        let t1 = tau.x > t_eff;
        let t2 = tau.y > t_eff;
        match (t1, t2) {
            (false, false) => (tau, false),
            (true, false) => (Vec2::new(t_eff, tau.y), true),
            (false, true) => (Vec2::new(tau.x, t_eff), true),
            (true, true) => (Vec2::splat(t_eff), true), // biaxial tension corner return
        }
    }
}

impl MaterialModel for RankineMaterial {
    /// Whatever the caller measured, verbatim (see the `optics` field).
    fn optical_properties(&self) -> Option<crate::energy::radiation::OpticalCoefficientsSi> {
        self.optics
    }

    fn specific_heat_j_kg_k(&self) -> f32 {
        self.specific_heat_j_kg_k
    }

    fn rest_density(&self) -> Option<f32> {
        self.rest_density
    }

    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Rankine
    }

    /// Corotated elastic Kirchhoff stress plus a Kelvin-Voigt viscous term
    /// on the deviatoric strain rate -- see `elastic_viscosity`'s doc.
    /// Zero when `elastic_viscosity == 0.0`. `tau_v = eta*D_dev`, without a
    /// factor of 2, the convention of `DruckerPragerMaterial` and
    /// `ViscoelasticMaterial` (locked by
    /// `viscous_term_matches_eta_times_deviatoric_strain_rate_exactly`); the
    /// factor lives in `q_factor_elastic_viscosity_pa_s`'s conversion.
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
        let f_trial =
            advance_deformation_gradient(*ctx.deformation_gradient, dt * *ctx.velocity_gradient);
        let (u, sigma, vt) = svd2(f_trial);

        let eps = hencky_strains(sigma);

        let a = 2.0 * self.mu + self.lambda;
        let tau = Vec2::new(
            a * eps.x + self.lambda * eps.y,
            self.lambda * eps.x + a * eps.y,
        );

        let damage = *ctx.friction_hardening;
        let t_eff = self.tensile_strength_eff(damage);

        let (tau_proj, yielded) = self.project_stress(tau, t_eff);

        let sigma_new = if yielded {
            let eps_proj = stress_to_hencky(tau_proj, self.lambda, self.mu);
            let eps_trial = stress_to_hencky(tau, self.lambda, self.mu);
            *ctx.friction_hardening = (damage + (eps_trial - eps_proj).length())
                .min(rankine_damage_saturation_point(self.softening_rate));
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
            model: ConstitutiveModel::Rankine as u32,
            // Contact sizes particles from it (see `MaterialModel::rest_density`);
            // 0 = unknown. No solid stress branch reads this slot.
            rest_density: self.rest_density.unwrap_or(0.0),
            lambda: self.lambda,
            mu: self.mu,
            // tensile_strength → hardening_exponent slot (union layout, CPU-only plasticity)
            hardening_exponent: self.tensile_strength,
            // softening_rate → hardening_modulus slot
            hardening_modulus: self.softening_rate,
            specific_heat_j_kg_k: self.specific_heat_j_kg_k,
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
        // The explicit viscous-diffusion bound of
        // `DruckerPragerMaterial::timestep_bound` for the Kelvin-Voigt term:
        // without it the substep selector does not see `elastic_viscosity`'s
        // stiffness.
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

    fn needs_cpu_update(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests;
