use glam::{Mat2, Vec2};

use crate::energy::thermodynamics::ideal_gas::{
    AIR_ADIABATIC_INDEX, AIR_SPECIFIC_GAS_CONSTANT_J_KG_K, ideal_gas_sound_speed_from_temperature,
};
use crate::materials::utils::von_neumann_richtmyer_q;
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Compressible ideal-gas material: isentropic (adiabatic) EOS
/// `p = p0·(ρ/ρ0)^γ` where `p0 = ρ0·R·T` (real ideal gas law evaluated at
/// the particle's own reference state), real adiabatic sound speed
/// `c = √(γRT)`, von Neumann-Richtmyer shock viscosity for compression
/// events. Genuinely different EOS shape from `NewtonianFluidMaterial`'s
/// Tait law: `p0` is a real physical rest pressure (not an empirically-fit
/// stiffness) and there's NO rest-pressure offset -- `p → 0` as `ρ → 0`,
/// exactly (see `energy::thermodynamics::ideal_gas`'s own
/// `pressure_vanishes_with_density` test).
///
/// Real, disclosed history (2026-08-18): the FIRST version of this
/// material used the naive isothermal law `p=ρRT` at fixed T directly --
/// internally inconsistent with the ADIABATIC sound speed it already used
/// for CFL/shock viscosity, and NOT self-limiting under expansion
/// (`p·V=nRT` stays exactly constant for a fixed-T ideal gas, so the
/// outward force never throttles down). A live demo (`examples/
/// basic_gas.rs`) caught this as a genuine runaway (avg J climbing to
/// 8-11 within under 2 simulated seconds from a 3:1 initial pressure
/// ratio) -- not a tuning problem. The isentropic form fixes it: MPM
/// substeps run far too fast for heat conduction to equalize (the SAME
/// real justification the adiabatic sound speed already relied on), and
/// `p·V` now falls off as `V^(1-γ)` under expansion, strictly decreasing
/// for γ>1.
///
/// Real, disclosed coupling: `p0` reads `Particles::temperature` directly,
/// so a `ThermalDiffusion` attached to the same scene still genuinely
/// shifts gas pressure as it heats/cools particles over its own slower
/// (conductive) timescale -- the fast, per-substep mechanical response to
/// compression/expansion is what changed, not the (slow) thermal coupling.
/// Without a `ThermalDiffusion`, temperature stays at whatever
/// `init_particle` seeds it to (`reference_temperature_k`).
///
/// CPU only today -- `p2g.wgsl`/`particles_update.wgsl` have no
/// `case 13u` branch, so a `Gas`-modeled particle on the GPU path silently
/// falls through those shaders' `default: { return mat2x2<f32>(); }` arm
/// (zero stress). Real, disclosed limitation, not a hidden gap -- see
/// `ConstitutiveModel::Gas`'s own doc. Matches this engine's own standing
/// rule: CPU correctness first, GPU port second.
#[derive(Debug, Clone, Copy)]
pub struct GasMaterial {
    /// Reference density ρ₀ (grid units, `rho_SI * dx_meters²`).
    pub rest_density: f32,
    /// Dynamic viscosity µ (Pa·s, raw SI -- passes through unconverted,
    /// same regression-fixed convention `NewtonianFluidMaterial` uses).
    /// Real air value ≈1.81e-5 Pa·s; 0.0 = inviscid (Euler gas dynamics).
    pub dynamic_viscosity: f32,
    /// Specific gas constant R, GRID-scaled (`R_SI / dx_meters²`) -- see
    /// `from_physical`'s own doc for the full derivation of why R itself
    /// (not just density) needs this factor, unlike Tait's ratio-based EOS.
    pub specific_gas_constant: f32,
    /// Real adiabatic index γ = Cp/Cv (air/diatomic: 1.4, `AIR_ADIABATIC_INDEX`;
    /// monatomic: 5/3; triatomic: ~1.3). Used both for the real adiabatic
    /// sound speed AND as the shock-viscosity weak-shock coefficient's own
    /// gamma (`von_neumann_richtmyer_q`'s `weak_shock_gamma`) -- for this
    /// material that substitution is exact, not a stand-in: γ here IS the
    /// real thermodynamic adiabatic index Kurapatenko's own coefficient is
    /// defined in terms of.
    pub adiabatic_index: f32,
    /// Temperature (Kelvin) seeded onto every particle at spawn
    /// (`init_particle`) and used as the fixed reference state for
    /// `timestep_bound`/`rest_acoustic_c2` (see `timestep_bound`'s own
    /// doc for the real, disclosed limitation this implies under strong
    /// active heating).
    pub reference_temperature_k: f32,
    pub min_density: f32,
    pub min_volume: f32,
    /// Lower bound on `J = V/V0` -- unlike a weakly-compressible liquid, a
    /// real gas can compress far below half its rest volume, so this is
    /// deliberately much wider than `NewtonianFluidMaterial`'s pinned
    /// `[0.5, 2.0]`. NOT tuned against a real impact/shock test scene the
    /// way fluid's own bounds are (no such scene exists for gas yet) --
    /// first real cut, disclosed as provisional.
    pub volume_ratio_min: f32,
    /// Upper bound on `J`. See `volume_ratio_min`'s own doc.
    pub volume_ratio_max: f32,
}

impl GasMaterial {
    pub const fn new(
        rest_density: f32,
        dynamic_viscosity: f32,
        specific_gas_constant: f32,
        adiabatic_index: f32,
        reference_temperature_k: f32,
    ) -> Self {
        Self {
            rest_density,
            dynamic_viscosity,
            specific_gas_constant,
            adiabatic_index,
            reference_temperature_k,
            min_density: 1.0e-6,
            min_volume: 1.0e-6,
            volume_ratio_min: 0.05,
            volume_ratio_max: 20.0,
        }
    }

    /// Real dry air at a given SI density/temperature. Real, standard
    /// constants: `AIR_SPECIFIC_GAS_CONSTANT_J_KG_K`/`AIR_ADIABATIC_INDEX`
    /// (already verified against the real ~343 m/s reference speed of
    /// sound in `energy::thermodynamics::ideal_gas`'s own tests), dynamic
    /// viscosity 1.81e-5 Pa·s (air at ~20°C, Sutherland's law reference).
    pub fn air(rho_kg_m3: f32, temperature_k: f32, config: &crate::SimConfig) -> Self {
        Self::from_physical(
            rho_kg_m3,
            1.81e-5,
            AIR_SPECIFIC_GAS_CONSTANT_J_KG_K,
            AIR_ADIABATIC_INDEX,
            temperature_k,
            config,
        )
    }

    /// General ideal-gas constructor from real SI properties.
    ///
    /// `specific_gas_constant_j_kg_k` -- R for the specific gas (air:
    /// 287.05, `AIR_SPECIFIC_GAS_CONSTANT_J_KG_K`). `adiabatic_index` --
    /// real γ=Cp/Cv (air/diatomic: 1.4; monatomic: 5/3; triatomic: ~1.3).
    ///
    /// Grid-scaling derivation (mirrors `NewtonianFluidMaterial::
    /// from_physical`'s own doc and the real regression it fixed, extended
    /// here for the ideal gas law's different functional form): solver
    /// time is already real seconds and positions are grid cells
    /// (`x_grid = x_SI/dx`), so pressure must stay raw SI
    /// (`p_grid == p_SI`) and only density converts
    /// (`rho_grid = rho_SI*dx²`), exactly as that fix established. Tait's
    /// EOS uses a density RATIO (`ρ/ρ₀`), which is scale-invariant under
    /// that conversion for free -- both numerator and denominator carry
    /// the same `dx²` factor, which cancels. The ideal gas law is LINEAR
    /// in density (`p=ρRT`, no ratio), so nothing cancels automatically:
    /// `R` itself must absorb the compensating `1/dx²` for `p` to come out
    /// raw SI: `R_grid = R_SI/dx²` gives
    /// `ρ_grid·R_grid·T = ρ_SI·dx²·(R_SI/dx²)·T = ρ_SI·R_SI·T = p_SI`.
    /// The SAME `R_grid` also gives the correct GRID-unit (cells/s)
    /// adiabatic sound speed for the CFL/shock terms:
    /// `c_grid² = γ·R_grid·T = (γ·R_SI·T)/dx² = c_SI²/dx²`, i.e.
    /// `c_grid = c_SI/dx` -- the same `v_grid = v_SI/dx` convention every
    /// other velocity in this engine already uses (e.g. `gravity_to_grid`).
    pub fn from_physical(
        rho_kg_m3: f32,
        eta_pa_s: f32,
        specific_gas_constant_j_kg_k: f32,
        adiabatic_index: f32,
        temperature_k: f32,
        config: &crate::SimConfig,
    ) -> Self {
        assert!(
            config.dx_meters.is_finite() && config.dx_meters > 0.0,
            "GasMaterial::from_physical requires a positive dx_meters"
        );
        let dx2 = config.dx_meters * config.dx_meters;
        let rho_grid = rho_kg_m3 * dx2;
        let r_grid = specific_gas_constant_j_kg_k / dx2;
        Self::new(rho_grid, eta_pa_s, r_grid, adiabatic_index, temperature_k)
    }
}

impl MaterialModel for GasMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Gas
    }

    /// Seeds the exact analytical rest state, same contract
    /// `NewtonianFluidMaterial::init_particle` establishes (`V0=m/ρ0`,
    /// `ρ=ρ0/J`) -- plus `temperature`, which a strict fluid never needs
    /// but this EOS genuinely depends on (`p=ρRT`).
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density / j;
        particle.temperature = self.reference_temperature_k;
    }

    /// Real fix, found live 2026-08-18 debugging `examples/basic_steam.rs`
    /// (water boiling into gas -- see `MaterialModel::
    /// init_particle_from_transition`'s own doc for the full mechanism):
    /// `init_particle`'s own `mass/rest_density` formula is exactly right
    /// for a FRESH spawn, but wrong for a TRANSITION from a material with
    /// a dramatically different rest density (e.g. water->steam, a real
    /// ~1700x ratio) -- it would make the particle's claimed volume jump
    /// that same ~1700x in a single instant, injecting a real but wildly
    /// under-resolved force spike (confirmed live as a real crash cause).
    ///
    /// Correct fix: keep the reference volume TRUE (`mass/rest_density`,
    /// matching exactly what `kirchhoff_stress`/`update_particle` already
    /// assume every substep), and instead give the particle a STARTING
    /// deformation gradient reflecting how compressed it genuinely is
    /// relative to that true reference -- a real, physically honest
    /// picture (freshly-formed gas still occupying roughly its old, much
    /// smaller prior footprint IS heavily compressed relative to this
    /// material's own rest state), clamped to this material's own
    /// `[volume_ratio_min, volume_ratio_max]`, the SAME bounds every
    /// subsequent substep already enforces -- so the starting state is
    /// consistent with the ongoing dynamics from frame one, not a
    /// separate, inconsistent value that later dynamics silently
    /// overwrite (a first, wrong fix attempt -- capping `initial_volume`
    /// directly -- learned this the hard way: `update_particle` recomputes
    /// volume fresh from `mass*j/rest_density` every substep, never
    /// reading `initial_volume` again, so a capped `initial_volume` alone
    /// only held for one substep before becoming permanently inconsistent
    /// with the real volume update.
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        let true_initial_volume = particle.mass / self.rest_density;
        let prior_volume = particle.volume.max(1.0e-9);
        let j = (prior_volume / true_initial_volume)
            .clamp(self.volume_ratio_min, self.volume_ratio_max);
        let s = j.sqrt();
        particle.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        particle.initial_volume = true_initial_volume;
        particle.volume = true_initial_volume * j;
        particle.density = particle.mass / particle.volume.max(1.0e-9);
        particle.temperature = self.reference_temperature_k;
    }

    /// `c² = γ·R·T` evaluated at the reference state (`J=1`,
    /// `T=reference_temperature_k`) -- the same formula `timestep_bound`
    /// evaluates, not a second derivation. See that method's own doc for
    /// the real, disclosed limitation this fixed-reference-T evaluation
    /// implies under active heating.
    fn rest_acoustic_c2(&self) -> Option<f32> {
        if self.specific_gas_constant > 0.0 && self.reference_temperature_k > 0.0 {
            Some(self.adiabatic_index * self.specific_gas_constant * self.reference_temperature_k)
        } else {
            None
        }
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        let density = (self.rest_density / j).max(self.min_density);
        let temperature = particles.temperature[i].max(0.0);

        // Isentropic (adiabatic) pressure law, NOT the naive isothermal
        // `p=ρRT` at fixed T this used until 2026-08-18. Real, found via a
        // live-crashing demo (basic_gas.rs): MPM substeps operate on
        // timescales far too short for heat conduction to equalize --
        // exactly the same real justification `ideal_gas_sound_speed`'s
        // own doc already gives for using the ADIABATIC sound speed
        // (`c=√(γRT)`), not the isothermal one. Using the isothermal
        // pressure law together with the adiabatic sound speed was a real,
        // internal inconsistency: the CFL/shock-viscosity terms assumed a
        // stiffer, self-limiting adiabatic response while the actual
        // restoring force was the softer isothermal one, which never
        // throttles down under expansion (`p·V = nRT` stays EXACTLY
        // constant as a fixed-T ideal gas expands, so the P2G force
        // contribution never decays) -- a real energy-injecting mismatch,
        // not a tuning problem, and the direct cause of the observed
        // runaway (avg J climbing to 8-11 in well under 2 simulated
        // seconds from an initial 3:1 pressure ratio).
        //
        // Standard compressible-flow result (isentropic relation):
        // combining `p=ρRT` with the adiabatic relation `T/T0=(ρ/ρ0)^(γ-1)`
        // gives `p = p0·(ρ/ρ0)^γ`, where `p0=ρ0·R·T` is the real rest
        // pressure at the particle's CURRENT temperature (still real
        // thermal coupling -- a `ThermalDiffusion` shifts `p0` exactly as
        // `p=ρRT` would). This form self-limits under expansion (`p·V`
        // now falls off as `V^(1-γ)`, strictly decreasing for γ>1) and its
        // own `dp/dρ` at `ρ=ρ0` is exactly `γRT` -- the SAME formula
        // `rest_acoustic_c2`/`timestep_bound` already compute, now
        // internally consistent rather than assumed.
        let p0 = self.rest_density * self.specific_gas_constant * temperature;
        let pressure = (p0
            * crate::materials::utils::fast_pow(density / self.rest_density, self.adiabatic_index))
        .max(0.0); // physically required floor: ρ,T >= 0 => p >= 0, nothing to configure

        let mut stress = Mat2::from_diagonal(Vec2::splat(-pressure));

        let c = particles.velocity_gradient[i];
        let sym_strain = c + c.transpose();
        let div_v = sym_strain.x_axis.x + sym_strain.y_axis.y; // = 2·∇·v

        if self.dynamic_viscosity > 0.0 {
            let strain_dev = sym_strain - Mat2::from_diagonal(Vec2::splat(div_v * 0.5));
            stress += self.dynamic_viscosity * strain_dev;
        }

        // Artificial (shock) viscosity -- von Neumann & Richtmyer 1950,
        // reusing the SAME shared q-formula `NewtonianFluidMaterial` uses
        // (`materials::utils::von_neumann_richtmyer_q`), fed this
        // material's own real adiabatic sound speed and real γ rather
        // than a Tait-EOS-derived stand-in (see that function's own doc).
        let c_sound = ideal_gas_sound_speed_from_temperature(
            self.specific_gas_constant,
            self.adiabatic_index,
            temperature,
        );
        let q = von_neumann_richtmyer_q(
            self.rest_density,
            j,
            0.5 * div_v,
            1.0,
            c_sound,
            self.adiabatic_index,
        );
        stress += Mat2::from_diagonal(Vec2::splat(-q));

        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        let f_trial = (Mat2::IDENTITY + dt * *ctx.velocity_gradient) * *ctx.deformation_gradient;
        let j = f_trial
            .determinant()
            .clamp(self.volume_ratio_min, self.volume_ratio_max);
        let s = j.sqrt();
        *ctx.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        let density = (self.rest_density / j).max(self.min_density);
        *ctx.density = density;
        *ctx.volume = (ctx.mass / density).max(1.0e-9);
    }

    fn owns_deformation_volume_state(&self) -> bool {
        true
    }

    fn needs_density_recompute(&self) -> bool {
        false
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::Gas as u32,
            rest_density: self.rest_density,
            // Repurposed slots -- GPU has no `case 13u` branch to read
            // these yet (see this struct's own top-of-file doc); kept
            // filled so the CPU-side struct is complete and ready for
            // when that branch lands, same convention `params()` already
            // uses for every other material's union-layout fields.
            eos_stiffness: self.specific_gas_constant,
            eos_power: self.adiabatic_index,
            dynamic_viscosity: self.dynamic_viscosity,
            volume_ratio_min: self.volume_ratio_min,
            volume_ratio_max: self.volume_ratio_max,
            owns_deformation_volume_state: self.owns_deformation_volume_state() as u32,
            ..Default::default()
        }
    }

    /// Real, disclosed limitation: this trait method's signature carries
    /// only `density`, not per-particle temperature, so the acoustic bound
    /// uses `reference_temperature_k` rather than the particle's actual
    /// current `T`. Exact when no `ThermalDiffusion` is attached (the gas
    /// then stays isothermal at that reference forever); a real,
    /// disclosed under-estimate risk if a strong heat source pushes local
    /// T well above reference (sound speed scales as `√T`, so the true
    /// CFL bound would be tighter than this one computes). Needs a
    /// temperature-aware `timestep_bound` signature extension to close
    /// fully -- same disclosed-gap class as `artificial_bulk_viscosity`'s
    /// own CFL-feedback note in `liquid::fluid`.
    fn timestep_bound(
        &self,
        density: f32,
        _hardening_scale: f32,
        cell_width: f32,
        material_cfl: f32,
        viscous_cfl: f32,
    ) -> f32 {
        let mut dt_bound = f32::INFINITY;

        let c2 = self.adiabatic_index * self.specific_gas_constant * self.reference_temperature_k;
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        if self.dynamic_viscosity > 0.0 {
            let density = density.max(self.min_density);
            let kinematic_viscosity = self.dynamic_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }

        dt_bound
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SimConfig;
    use crate::particle::Particles;
    use glam::Vec2;

    /// `dx_meters = 1.0` collapses grid<->SI scaling to identity, so this
    /// test can compare `kirchhoff_stress`'s output directly against the
    /// real, textbook air reference `energy::thermodynamics::ideal_gas`'s
    /// own test already verifies (~101,325 Pa at ρ=1.204 kg/m³, T=293.15K).
    fn unit_dx_config() -> SimConfig {
        SimConfig::standard(64, 0.05, Vec2::NEG_Y * 0.3)
    }

    fn one_particle_at_rest(mat: &GasMaterial) -> Particles {
        let mut p = Particle::zeroed();
        p.mass = mat.rest_density; // V0 = mass/rest_density = 1.0
        p.deformation_gradient = Mat2::IDENTITY;
        mat.init_particle(&mut p);
        Particles::from(vec![p])
    }

    #[test]
    fn rest_pressure_matches_real_ideal_gas_law_reference() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let mat = GasMaterial::air(1.204, 293.15, &config);
        let particles = one_particle_at_rest(&mat);

        let stress = mat.kirchhoff_stress(&particles, 0);
        let pressure = -stress.x_axis.x;

        assert!(
            (pressure - 101_325.0).abs() / 101_325.0 < 0.01,
            "GasMaterial's own stress should reproduce the real ideal gas \
             pressure at real air density/temperature: got {pressure:.1} Pa"
        );
    }

    #[test]
    fn pressure_vanishes_as_density_vanishes() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let mat = GasMaterial::air(1.204, 293.15, &config);
        let mut particles = one_particle_at_rest(&mat);
        // Expand hugely (J >> 1) -> density -> 0.
        particles.deformation_gradient[0] = Mat2::from_diagonal(Vec2::splat(1000.0));

        let stress = mat.kirchhoff_stress(&particles, 0);
        let pressure = -stress.x_axis.x;
        assert!(
            pressure < 1.0,
            "pressure should vanish toward zero as density does (no Tait-style \
             rest-pressure offset): got {pressure}"
        );
    }

    #[test]
    fn hotter_gas_has_higher_pressure_at_the_same_density() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let cold = GasMaterial::air(1.204, 250.0, &config);
        let hot = GasMaterial::air(1.204, 400.0, &config);

        let p_cold = -cold
            .kirchhoff_stress(&one_particle_at_rest(&cold), 0)
            .x_axis
            .x;
        let p_hot = -hot
            .kirchhoff_stress(&one_particle_at_rest(&hot), 0)
            .x_axis
            .x;

        assert!(
            p_hot > p_cold,
            "real p=rhoRT must increase with temperature at fixed density: \
             p_cold={p_cold:.1} p_hot={p_hot:.1}"
        );
    }

    #[test]
    fn compression_increases_pressure_at_fixed_temperature() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let mat = GasMaterial::air(1.204, 293.15, &config);
        let mut particles = one_particle_at_rest(&mat);
        particles.deformation_gradient[0] = Mat2::from_diagonal(Vec2::splat(0.5)); // J=0.25, denser

        let stress = mat.kirchhoff_stress(&particles, 0);
        let pressure = -stress.x_axis.x;
        let rest_pressure = -mat
            .kirchhoff_stress(&one_particle_at_rest(&mat), 0)
            .x_axis
            .x;

        assert!(
            pressure > rest_pressure,
            "compressing a real ideal gas at fixed T must raise its pressure: \
             rest={rest_pressure:.1} compressed={pressure:.1}"
        );
    }

    #[test]
    fn rest_acoustic_c2_matches_timestep_bound_derivation() {
        let mut config = unit_dx_config();
        config.dx_meters = 1.0;
        let mat = GasMaterial::air(1.204, 293.15, &config);
        let c2 = mat
            .rest_acoustic_c2()
            .expect("air has a real acoustic term");
        let expected = AIR_ADIABATIC_INDEX * mat.specific_gas_constant * 293.15;
        assert!(
            (c2 - expected).abs() / expected < 1.0e-4,
            "rest_acoustic_c2 must match the same formula timestep_bound uses: \
             got {c2}, expected {expected}"
        );
    }

    #[test]
    fn constitutive_model_is_gas() {
        let mat = GasMaterial::new(0.1, 0.0, 287.05, 1.4, 293.15);
        assert_eq!(mat.constitutive_model(), ConstitutiveModel::Gas);
        assert_eq!(mat.constitutive_model() as u32, 13);
    }
}
