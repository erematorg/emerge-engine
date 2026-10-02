use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, NewtonianFluid, scale_stress, scale_visc};
use crate::materials::utils::{advance_log_volume_ratio, von_neumann_richtmyer_q};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Current volume ratio `J = V/V0` from the material's own conserved volume
/// state (not a kernel-density gather, which is free-surface biased).
#[inline]
pub(crate) fn volume_j(initial_volume: f32, volume: f32, material_name: &str) -> f32 {
    assert!(
        initial_volume.is_finite() && initial_volume > 0.0 && volume.is_finite() && volume > 0.0,
        "{material_name}: fluid reference/current volume must be finite and positive"
    );
    volume / initial_volume
}

/// Artificial (shock) viscosity `q`, added to pressure as `-q*I`.
///
/// von Neumann & Richtmyer 1950 (LA-671) quadratic term + Landshoff linear
/// term -- the standard shock-capturing pair for Lagrangian hydrocodes, and
/// the same combined form used for MPM specifically (Wang et al., "Portable,
/// Massively Parallel Implementation of a Material Point Method for
/// Compressible Flows", arXiv:2404.17057, eq. 4). Gated to compression
/// (`div(v) < 0`) so it vanishes identically wherever the flow is smooth --
/// shocks only form under compression.
///
/// `c0` (quadratic) is `(gamma+1)/4`, the weak-shock limit of the fundamental
/// derivative Kurapatenko 1967 ties it to, not the strong-shock `(gamma+1)/2`:
/// the strong-shock value made a crash worse while its contribution was not
/// yet in the CFL bound (Bate et al. 1995's `c_eff = c_sound +
/// 2*c0*h*|div(v)|`). That CFL term now exists (`shock_viscosity_dt_bound` in
/// `cfl.rs`); the strong-shock value has not been re-measured with it.
///
/// `c1 = 1.0` (Landshoff) is the standard theoretical value.
pub(crate) fn artificial_bulk_viscosity(
    eos_stiffness: f32,
    eos_power: f32,
    rest_density: f32,
    j: f32,
    div_v: f32,
    grid_cell_size: f32,
) -> f32 {
    // Both terms carry rho exactly once, as in Wang et al. arXiv:2404.17057
    // eq. 4 and `tmp/GeoTaichi`'s `MaterialModel.py::artifical_viscosity` (the
    // shared `von_neumann_richtmyer_q` below). With rho^2 in the quadratic
    // term and none in the linear one, q was ~8x too large at grid rho ~0.1
    // (max_speed 11 -> 130, J pinned at 2.0, 45 -> 12 fps); now q ~63 against
    // an EOS pressure scale of ~94.
    //
    // Tait EOS's own `c_sound` (this material's `dp/drho`), passed in since
    // `von_neumann_richtmyer_q` owns only the shared form. `eos_power` stands
    // in for Kurapatenko's weak-shock gamma (see `von_neumann_richtmyer_q`).
    let density_ratio = 1.0 / j;
    let c2 = eos_stiffness
        * eos_power
        * crate::materials::utils::fast_pow(density_ratio, eos_power - 1.0)
        / rest_density;
    let c_sound = c2.max(0.0).sqrt();
    von_neumann_richtmyer_q(rest_density, j, div_v, grid_cell_size, c_sound, eos_power)
}

/// Weakly-compressible Newtonian fluid (Tait EOS + deviatoric viscosity).
/// Refs: Becker & Teschner 2007 (WCSPH), Hu et al. 2018 (MLS-MPM).
#[derive(Debug, Clone, Copy)]
pub struct NewtonianFluidMaterial {
    pub rest_density: f32,
    pub dynamic_viscosity: f32,
    pub eos_stiffness: f32,
    pub eos_power: f32,
    /// Floor on the Tait EOS pressure: limits negative (tensile) pressure at
    /// a free surface, where a real fluid cavitates rather than sustaining
    /// tension. `tmp/sparkl`'s `MonaghanSphEos::max_neg_pressure` clamps
    /// `Ktait0*((rho/rho0)^gamma - 1)` at the same `-0.1`, and
    /// `tmp/incremental_mpm`'s MLS-MPM fluid uses the same constant ("i
    /// clamped it as a bit of a hack"). Bounds the pressure only, not the
    /// drift of `density`/`volume`.
    pub pressure_floor: f32,
    /// Specific heat capacity `c_p`, J/(kg*K). 0 (the default) means
    /// undeclared -- see `MaterialModel::specific_heat_j_kg_k`. Liquid water
    /// is 4182 at 25 C (CRC Handbook); set it the same way `bulk_viscosity`
    /// and `pressure_floor` are set, after construction.
    pub specific_heat_j_kg_k: f32,
    /// Measured optical coefficients for this material, or `None` when the
    /// caller has not supplied any.
    ///
    /// Deliberately NOT a substance flag. A material model never decides
    /// that it "is water" because its numbers happen to look like water's;
    /// it carries whatever measured absorption and scattering the caller
    /// declares, and a name for the result is a label applied on top, never
    /// a branch inside the physics. `matter::materials::optical` holds the
    /// measured datasets to fill this with (`pure_water`, `dry_quartz_sand`,
    /// ...); anything else measured is equally valid here.
    ///
    /// `None` is the honest default: no spectrum was measured, so the
    /// renderer is told nothing rather than being handed an invented one.
    pub optics: Option<crate::energy::radiation::OpticalCoefficientsSi>,
    pub min_density: f32,
    pub min_volume: f32,
    /// Thermal thinning: µ_eff = dynamic_viscosity · exp(−thermal_viscosity_coeff · T).
    /// 0.0 = isothermal. Positive values make the fluid flow easier when hot.
    pub thermal_viscosity_coeff: f32,
    /// Bulk viscosity ζ (second viscosity, Pa·s in physical units).
    ///
    /// Adds τ += ζ·(∇·v)·I to Kirchhoff stress -- damps compression waves (acoustic damping).
    /// Physical: Navier-Stokes second viscosity, distinct from shear viscosity µ.
    /// Stokes assumption (ζ=0) holds for dilute ideal gases; real liquids have ζ > 0.
    /// For water: ζ ≈ 3e-3 Pa·s (Dukhin & Goetz 2009). In simulation units set to
    /// ~0.5–5× dynamic_viscosity. 0.0 = no acoustic damping.
    pub bulk_viscosity: f32,
    /// Surface tension coefficient γ (N/m in physical units).
    ///
    /// Adds isotropic Kirchhoff stress τ += γ·J·I -- continuum surface energy ψ = γ·J.
    /// Reference: Ziran 2020, `SurfaceTension.h` (Chenfanfu Jiang group).
    ///
    /// **Limitation**: curvature-free. Young-Laplace gives Δp = γ·κ (interface curvature κ),
    /// but MPM particles carry no interface normal. This term resists volumetric compression
    /// isotropically -- sufficient for cohesion/droplet stability, not for curvature-driven
    /// flow (e.g. Rayleigh-Plateau instability). 0.0 = disabled.
    pub surface_tension_coeff: f32,
}

impl NewtonianFluidMaterial {
    /// Construct directly from grid-native parameters -- NOT SI units.
    /// Prefer [`Self::low_viscosity`] for a real-water preset, or the
    /// [`FromSI`] impl on this type (via `Fluid` properties) for a real
    /// SI-to-grid conversion from measured density/viscosity/stiffness.
    pub const fn new(
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
            specific_heat_j_kg_k: 0.0,
            optics: None,
            min_density: 1.0e-6,
            min_volume: 1.0e-6,
            thermal_viscosity_coeff: 0.0,
            bulk_viscosity: 0.0,
            surface_tension_coeff: 0.0,
        }
    }

    /// Low-viscosity preset: γ=7, µ=1e-3 Pa·s. Corresponds to water at 20°C.
    ///
    /// `eos_stiffness` controls incompressibility -- higher = stiffer; 1e4 works
    /// well at emerge's default grid scale. Reference: Becker & Teschner 2007 §4.
    pub fn low_viscosity(rest_density: f32, eos_stiffness: f32) -> Self {
        Self::new(rest_density, 1.0e-3, eos_stiffness, 7.0)
    }

    /// Weakly-compressible variant: caps sound speed at `c_ref_m_s`.
    ///
    /// Use `c_ref_m_s = 10 * v_max_m_s` (WCSPH rule) to limit compressibility to ~1%.
    /// `rho_kg_m3` and `eta_pa_s` are the fluid's SI density and viscosity.
    pub fn weakly_compressible(
        rho_kg_m3: f32,
        eta_pa_s: f32,
        c_ref_m_s: f32,
        config: &crate::SimConfig,
    ) -> Self {
        // Tait EOS polytropic exponent for water -- Cole 1948, "Underwater Explosions"
        // (the original real-fluid measurement this exponent is drawn from); used
        // identically in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994;
        // Becker & Teschner 2007, already cited elsewhere in this project).
        //
        // Stress and viscosity stay raw SI, as in `IdealGasMaterial::
        // from_physical`: solver time is seconds and positions are cells, so
        // only density converts (through `dx^2`). Routed through the old
        // `scale_stress`/`dt_seconds` conversion, this scene's water EOS was
        // ~4.4 million times too soft and its viscosity ~296 million times
        // too large.
        const GAMMA: f32 = 7.0;
        assert!(
            config.dx_meters.is_finite() && config.dx_meters > 0.0,
            "weakly_compressible requires a positive dx_meters"
        );
        let rho_grid = rho_kg_m3 * config.dx_meters * config.dx_meters;
        let tait_b_pa = rho_kg_m3 * c_ref_m_s * c_ref_m_s / GAMMA;
        let mut material = Self::new(rho_grid, eta_pa_s, tait_b_pa, GAMMA);
        // `Self::new`'s `pressure_floor: -0.1` is an unconverted grid-unit
        // constant. This constructor keeps stress raw SI (see above), so the
        // cavitation pressure stays raw SI Pa too; `scale_stress` would
        // convert it twice.
        material.pressure_floor = -100_000.0; // real dissolved-gas cavitation onset, Pa gauge
        material
    }
}

impl FromSI<NewtonianFluid> for NewtonianFluidMaterial {
    /// `rest_density` defaults to `props.rho_kg_m3`. Caller should adjust if
    /// particle mass/volume don't match the SI density.
    fn from_physical(props: &NewtonianFluid, config: &crate::SimConfig) -> Self {
        // Tait EOS polytropic exponent for water -- Cole 1948, "Underwater Explosions";
        // standard in SPH/MPM weakly-compressible fluid solvers (Monaghan 1994).
        const GAMMA: f32 = 7.0;
        let visc = scale_visc(props.eta_pa_s, props.rho_kg_m3, config);
        let eos = scale_stress(props.bulk_modulus_pa / GAMMA, props.rho_kg_m3, config);
        // rest_density must be in the SAME units `particles.density[i]` actually
        // comes out in. A particle spawned at `spacing` carries
        // `grid_density * spacing^2` of mass in a `spacing^2` cell area, so the
        // density the solver measures is a RATIO against the scene's reference
        // density -- exactly 1.0 for a fluid at that reference. Getting this
        // wrong is not a small error: it makes the fluid believe it is spawned
        // pre-compressed, and the Tait EOS answers a `rho/rho_0` of 4 with a
        // pressure spike no CFL substep can bound.
        //
        // Do not reintroduce a `dx_meters^2` (or `/dt_seconds^2`) factor here.
        // Those pin a real fluid's EOS pressure at its floor regardless of
        // actual depth or compression -- see `SimConfig::grid_density`.
        let rho_grid = props.rho_kg_m3 / config.reference_density_kg_m3;
        let mut material = Self::new(rho_grid, visc, eos, GAMMA);
        // `Self::new`'s `pressure_floor: -0.1` is an unconverted grid-unit
        // constant. Water's cavitation onset in practice (dissolved-gas
        // nucleation, the engineering figure, not the higher degassed lab
        // value) is ~-100,000 Pa gauge, converted through the same
        // `scale_stress`/`stress_from_si` as `eos`, at this material's own
        // `props.rho_kg_m3`.
        const REAL_CAVITATION_PRESSURE_PA: f32 = -100_000.0;
        material.pressure_floor =
            scale_stress(REAL_CAVITATION_PRESSURE_PA, props.rho_kg_m3, config);
        material
    }
}

impl MaterialModel for NewtonianFluidMaterial {
    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::Fluid
    }

    // Required: the spawn contract lets materials that own their volume and
    // density state set them here, overriding the spawn's kernel-density
    // estimate, which is a fine default for materials without an exact
    // initial state but wrong for a strict fluid, which has one: V0 =
    // mass/rest_density exactly. Without this override the kernel estimate
    // set `volume`/`density` while F stayed at identity (J=1), an
    // inconsistent state `assert_owned_deformation_state` caught ("det(F)=1,
    // V/V0=0.528").
    fn init_particle(&self, particle: &mut Particle) {
        let j = particle.deformation_gradient.determinant();
        particle.initial_volume = particle.mass / self.rest_density;
        particle.volume = particle.initial_volume * j;
        particle.density = self.rest_density / j;
    }

    /// The condensing counterpart of `IdealGasMaterial::init_particle_from_
    /// transition`. The default (delegating to `init_particle`) throws away
    /// `Simulation::apply_phase_transition`'s rebaseline (`initial_volume` =
    /// the particle's current volume as steam) and sets water's rest volume at
    /// `F = IDENTITY`, several times smaller than the footprint the particle
    /// still occupies next to neighbours that are still steam. The Tait EOS
    /// answers that fake overcompression with a pressure spike that blew a
    /// cooling gas apart.
    ///
    /// As for the gas: the reference volume stays `mass/rest_density` (what
    /// `kirchhoff_stress`/`update_particle` assume), and the particle starts
    /// with a deformation gradient giving its compression against it,
    /// clamped to the `[0.5, 2.0]` `update_particle` enforces every substep
    /// (load-bearing, see that method).
    fn init_particle_from_transition(&self, particle: &mut Particle) {
        let true_initial_volume = particle.mass / self.rest_density;
        let prior_volume = particle.volume.max(1.0e-9);
        let j = (prior_volume / true_initial_volume).clamp(0.5, 2.0);
        let s = j.sqrt();
        particle.deformation_gradient = Mat2::from_cols(Vec2::new(s, 0.0), Vec2::new(0.0, s));
        particle.initial_volume = true_initial_volume;
        particle.volume = true_initial_volume * j;
        particle.density = self.rest_density / j;
    }

    /// Rest-state acoustic speed squared, `c^2 = B*gamma/rho0` (Tait EOS
    /// evaluated at `J = 1`).
    ///
    /// Required: without it the near-wall CFL gate (`cfl.rs`) cannot compute
    /// a Mach number, falls back to its fixed absolute threshold and fires
    /// the same at every flow speed
    /// (`near_wall_gate_relaxes_when_measured_speed_predicts_this_much_compression`).
    fn rest_acoustic_c2(&self) -> Option<f32> {
        if self.eos_stiffness > 0.0 && self.rest_density > 0.0 {
            Some(self.eos_stiffness * self.eos_power / self.rest_density)
        } else {
            None
        }
    }

    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        // Density from F's determinant (rho = rest_density / J), not the
        // grid-gathered `particles.density[i]`: the GPU fluid path (`p2g.wgsl`'s
        // case 1u) and `GranularFluidMaterial` use this formula, and the
        // grid-mass density lags one substep (P2G -> grid -> G2P) while J is
        // current.
        //
        // Clamp density both ways: min prevents div-by-zero, max (2x rho0)
        // limits how far the EOS pressure response saturates under impact
        // overcompression. Keep this at 2x, not looser --
        // `fluid_spreads_more_than_elastic_under_gravity` (tests/accuracy.rs)
        // needs it (a looser clamp stops the fluid spreading at all).
        let j = particles.deformation_gradient[i].determinant().max(1.0e-6);
        let density = (self.rest_density / j)
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

        // Bulk viscosity ζ: τ += ζ·(∇·v)·I -- damps longitudinal/acoustic waves.
        // ∇·v ≈ div_v/2 (div_v here is trace of sym_strain = C+Cᵀ = 2D, so ∇·v = div_v/2).
        if self.bulk_viscosity > 0.0 {
            stress += Mat2::from_diagonal(Vec2::splat(self.bulk_viscosity * div_v * 0.5));
        }

        if self.surface_tension_coeff != 0.0 {
            let f = particles.deformation_gradient[i];
            let j = f.x_axis.x * f.y_axis.y - f.x_axis.y * f.y_axis.x;
            stress += Mat2::from_diagonal(Vec2::splat(self.surface_tension_coeff * j));
        }

        // Artificial (shock) viscosity, a PDE term: von Neumann & Richtmyer
        // 1950 (LA-671) quadratic plus Landshoff linear, the standard
        // shock-capturing pair, gated to compression (`div(v) < 0`) so it
        // vanishes in smooth flow. `tmp/GeoTaichi` has the same form
        // (`MaterialModel.py::artifical_viscosity`) and enables it in its
        // Newtonian dam-break example (`cL: 1.0, cQ: 2`). `p2g.wgsl`'s fluid
        // branch carries a port of this function's current form; no test pins
        // the two together yet.
        // Reuses `j` (this function's own `det(F)`, computed above) rather
        // than recomputing `volume/initial_volume`. They are the SAME
        // quantity for a strict fluid -- `assert_owned_deformation_state`
        // exists precisely to enforce that they agree (to 2e-4) -- so this is
        // bit-equivalent, not an approximation. Saves one call with four
        // asserts and a division per particle per substep (~52k/frame at this
        // demo's 2912 particles x 18 substeps).
        let q = artificial_bulk_viscosity(
            self.eos_stiffness,
            self.eos_power,
            self.rest_density,
            j,
            // `div_v` above is tr(C + C^T) = 2*div(v); this term wants the
            // true divergence.
            0.5 * div_v,
            // `kirchhoff_stress`'s trait signature carries no grid_cell_size;
            // every scene in this engine uses 1.0 (exact today, not an
            // approximation) -- same disclosed limitation the GPU copy has.
            1.0,
        );
        stress += Mat2::from_diagonal(Vec2::splat(-q));

        stress
    }

    fn stress_volume(&self, particles: &Particles, i: usize) -> f32 {
        particles.volume[i].max(self.min_volume)
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // J advances by the continuity equation's exact solution for constant
        // `C` over the substep, `J_{n+1} = J_n * exp(dt*div(v))`. `det(I + dt*C)`
        // is not rotation-invariant: for a rigid rotation `C = [[0,-w],[w,0]]`
        // (div(v) = 0) it gives `1 + dt^2*w^2`, an expansion every substep that
        // isotropization then keeps (measured: a monotonic `detF` max drift
        // that continued after the flow had come to rest). The exponential is
        // exactly 1 for any rotation and matches `det(I+dt*C)` to first order
        // for real compression or expansion.
        // `old_j` recovers the fluid's own scalar state from its ALREADY-
        // isotropic F (`s*I`, `det=s^2`) -- exactly this material's `j`
        // from the previous substep, the same equivalence
        // `assert_owned_deformation_state` already enforces elsewhere in
        // this file (bit-equivalent to `volume/initial_volume` to 2e-4).
        let old_j = ctx.deformation_gradient.determinant();
        let div_v = ctx.velocity_gradient.x_axis.x + ctx.velocity_gradient.y_axis.y;
        // The [0.5, 2.0] clamp is load-bearing: a 6x6 block dropped 20 units
        // onto a rigid floor at eos_stiffness=50
        // (`fluid_impact_shows_real_free_surface_splash_separation`) hits both
        // bounds exactly over 250 steps.
        // `old_j` above is only the fallback: the carried logarithm is the
        // real state, because reading J back from F and multiplying loses a
        // fraction of every small increment (see
        // `advance_log_volume_ratio`'s doc for the measurement).
        let carried = if *ctx.log_volume_strain != 0.0 || old_j == 1.0 {
            *ctx.log_volume_strain
        } else {
            old_j.max(1.0e-9).ln()
        };
        let (log_j, j) = advance_log_volume_ratio(carried, dt * div_v, 0.5, 2.0);
        *ctx.log_volume_strain = log_j;
        let s = j.sqrt();
        *ctx.deformation_gradient =
            glam::Mat2::from_cols(glam::Vec2::new(s, 0.0), glam::Vec2::new(0.0, s));
        // `stress_volume`/`timestep_bound` read `particles.volume`/`density`,
        // so they are set here every substep from the same bounded formula
        // `stress()` uses (`(rest_density/j).max(min_density).min(2*rest_density)`),
        // as every plastic solid does (`sand.rs`'s `*ctx.density = ctx.mass /
        // v`). Left to `estimate_particle_volumes`'s kernel estimate, which has
        // no ceiling on compaction, the per-substep noise in a settling scene
        // only ratcheted up and stiffened the CFL bound over time
        // (`mixture_sand_water.rs`'s min_dt clamp).
        let density = (self.rest_density / j)
            .max(self.min_density)
            .min(self.rest_density * 2.0);
        *ctx.density = density;
        *ctx.volume = (ctx.mass / density).max(1.0e-9);
    }

    // Required: this material owns its volume and density (see
    // `update_particle`), so no kernel gather may overwrite them. With the
    // default `false`, `basic_fluids_gpu.rs`'s water froze from frame 1 (v~0,
    // J = 1.000), most likely because the kernel-density recompute broke
    // `strict_fluid_state_is_admissible`'s consistency check in
    // `particles_update.wgsl` every substep, whose early return skipped the
    // fluid's state update.
    fn owns_deformation_volume_state(&self) -> bool {
        true
    }

    fn specific_heat_j_kg_k(&self) -> f32 {
        self.specific_heat_j_kg_k
    }

    /// Tait inverted. The state equation is
    /// `p = k * ((rho/rho_0)^gamma - 1)`, so the density in equilibrium at
    /// pressure `p` is `rho = rho_0 * (1 + p/k)^(1/gamma)`, and since MPM
    /// carries density as `rho = rho_0 / J`, the volume ratio is
    /// `J = (1 + p/k)^(-1/gamma)`.
    ///
    /// Returns `None` for a non-positive pressure: a free surface is
    /// already at `J = 1` and needs no correction, and a negative pressure
    /// here would mean tension, which this state equation does not model.
    fn hydrostatic_volume_ratio(&self, pressure: f32) -> Option<f32> {
        if !pressure.is_finite() || pressure <= 0.0 || self.eos_stiffness <= 0.0 {
            return None;
        }
        Some((1.0 + pressure / self.eos_stiffness).powf(-1.0 / self.eos_power))
    }

    /// Whatever the caller measured, verbatim. See the `optics` field: this
    /// model reports coefficients, it does not identify a substance.
    fn optical_properties(&self) -> Option<crate::energy::radiation::OpticalCoefficientsSi> {
        self.optics
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
            specific_heat_j_kg_k: self.specific_heat_j_kg_k,
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
        // Floor on density / rest_density: keeps `ratio.powf(eos_power - 1.0)`
        // finite for `eos_power < 1` at a vanishing density.
        const MIN_DENSITY_RATIO: f32 = 1.0e-6;
        let density = density.max(self.min_density);
        let ratio = (density / self.rest_density.max(self.min_density)).max(MIN_DENSITY_RATIO);

        let mut dt_bound = f32::INFINITY;

        // Acoustic timestep bound from EOS derivative dp/drho.
        let c2 = self.eos_stiffness * self.eos_power * ratio.powf(self.eos_power - 1.0)
            / self.rest_density.max(self.min_density);
        if c2.is_finite() && c2 > f32::EPSILON {
            dt_bound = dt_bound.min(material_cfl * cell_width / c2.sqrt());
        }

        // Viscous diffusion bound for explicit integration, over
        // dynamic_viscosity and bulk_viscosity together: bulk viscosity's
        // explicit damping term needs a CFL bound too, since an explicit
        // damping term whose dt*viscosity/mass ratio is too large injects
        // energy instead of removing it (see
        // `GranularFluidMaterial::timestep_bound`). Both terms multiply the
        // same velocity-gradient-derived stress, so their linear sum is a
        // conservative bound.
        let combined_viscosity = self.dynamic_viscosity + self.bulk_viscosity.max(0.0);
        if combined_viscosity > 0.0 {
            let kinematic_viscosity = combined_viscosity / density;
            if kinematic_viscosity > f32::EPSILON {
                dt_bound =
                    dt_bound.min(viscous_cfl * cell_width * cell_width / kinematic_viscosity);
            }
        }

        dt_bound
    }
}

#[cfg(test)]
mod si_construction_tests {
    use super::*;

    /// `weakly_compressible` keeps stress and viscosity in raw SI (as
    /// `IdealGasMaterial::from_physical` does) and converts only density
    /// through `dx^2`, not through `scale_stress`/`scale_visc`.
    #[test]
    fn si_constructor_preserves_pressure_and_viscosity_units() {
        let cfg = crate::SimConfig::earth(32, 0.01, 0.1);
        let material = NewtonianFluidMaterial::weakly_compressible(1000.0, 1.0e-3, 20.0, &cfg);
        assert!((material.rest_density - 0.1).abs() < 1.0e-7);
        assert!((material.dynamic_viscosity - 1.0e-3).abs() < 1.0e-9);
        assert!((material.eos_stiffness - (1000.0 * 20.0 * 20.0 / 7.0)).abs() < 1.0e-3);
    }

    /// Both SI construction paths carry a converted cavitation pressure
    /// (~-100,000 Pa gauge, dissolved-gas nucleation onset) by default instead
    /// of `Self::new`'s unconverted grid-unit `-0.1`, each in its own unit
    /// convention (`weakly_compressible` raw SI; `from_physical` through
    /// `scale_stress`, like its `eos_stiffness`).
    #[test]
    fn si_constructors_convert_pressure_floor_not_just_stiffness() {
        let cfg = crate::SimConfig::earth(32, 0.01, 0.1);

        let wc = NewtonianFluidMaterial::weakly_compressible(1000.0, 1.0e-3, 20.0, &cfg);
        assert!(
            (wc.pressure_floor - (-100_000.0)).abs() < 1.0e-3,
            "weakly_compressible: pressure_floor={} -- expected raw SI -100000.0 Pa, \
             not the unconverted default -0.1",
            wc.pressure_floor
        );

        let props = NewtonianFluid {
            rho_kg_m3: 1000.0,
            eta_pa_s: 1.0e-3,
            bulk_modulus_pa: 1000.0 * 20.0 * 20.0,
        };
        let si = NewtonianFluidMaterial::from_physical(&props, &cfg);
        let expected = cfg.stress_from_si(-100_000.0, props.rho_kg_m3);
        assert!(
            (si.pressure_floor - expected).abs() < 1.0e-6,
            "from_physical: pressure_floor={} -- expected {expected} (real cavitation \
             pressure through the same scale_stress conversion eos_stiffness uses), not \
             the unconverted default -0.1",
            si.pressure_floor
        );
        assert_ne!(
            si.pressure_floor, -0.1,
            "from_physical must not silently keep Self::new's raw grid-unit default"
        );
    }
}

#[cfg(test)]
mod volume_integration_tests {
    use super::*;
    use crate::particle::{Particle, Particles};

    fn particle_with_f(f: Mat2) -> Particles {
        let mut p = Particle::zeroed();
        p.deformation_gradient = f;
        p.mass = 1.0;
        p.initial_volume = 1.0;
        p.volume = f.determinant();
        p.density = 1.0;
        Particles::from(vec![p])
    }

    /// A rigid rotation carries zero divergence (`tr(C) = 0`) and leaves
    /// `J = det(F)` exactly unchanged. `det((I+dt*C)*F_old)` gives `1 +
    /// dt^2*omega^2` for this `C`, a positive expansion every substep; the
    /// exponential `J_new = J_old*exp(dt*div(v))` is exactly 1 for any rotation.
    #[test]
    fn rigid_rotation_leaves_j_exactly_unchanged() {
        let mat = NewtonianFluidMaterial::new(1.0, 0.0, 100.0, 7.0);
        let mut particles = particle_with_f(Mat2::IDENTITY);
        let dt = 0.01;
        let omega = 5.0_f32; // deliberately large -- the old bug scales as omega^2
        {
            let mut ctx = particles.update_ctx(0);
            *ctx.velocity_gradient = Mat2::from_cols(Vec2::new(0.0, omega), Vec2::new(-omega, 0.0));
            for _ in 0..500 {
                mat.update_particle(&mut ctx, dt);
            }
        }
        let j = particles.deformation_gradient[0].determinant();
        assert!(
            (j - 1.0).abs() < 1.0e-5,
            "500 substeps of pure rotation (omega={omega}) must leave J exactly at 1.0, \
             got {j} -- the old det(I+dt*C) bug would give a real, measurable expansion here"
        );
    }

    /// A constant, uniform dilation (`C = k*I`, `div(v) = 2k` in 2D)
    /// integrates to exactly the continuity equation's solution,
    /// `J_new = J_old * exp(N*dt*2k)`, over `N` substeps of constant `C`, not
    /// the per-step `det(I+dt*C)`, which only agrees to first order.
    #[test]
    fn constant_dilation_matches_exact_exponential_solution() {
        let mat = NewtonianFluidMaterial::new(1.0, 0.0, 100.0, 7.0);
        let mut particles = particle_with_f(Mat2::IDENTITY);
        let dt = 0.001;
        let k = 2.0_f32;
        const N: i32 = 50;
        {
            let mut ctx = particles.update_ctx(0);
            *ctx.velocity_gradient = Mat2::from_diagonal(Vec2::splat(k));
            for _ in 0..N {
                mat.update_particle(&mut ctx, dt);
            }
        }
        let j = particles.deformation_gradient[0].determinant();
        let expected = (N as f32 * dt * 2.0 * k).exp();
        assert!(
            (j - expected).abs() / expected < 1.0e-4,
            "constant dilation over {N} substeps must match J_old*exp(N*dt*2k) exactly \
             (continuity equation's own solution for constant C) -- got {j}, expected {expected}"
        );
    }
}

#[cfg(test)]
mod transition_continuity_tests {
    use super::*;

    /// The real bug this override fixes: condensing from a much-LESS-dense
    /// prior material (e.g. steam, same mass spread over a much larger real
    /// volume) must NOT reset straight to `deformation_gradient=IDENTITY`
    /// (`j=1`, the old default-`init_particle`-fallback behavior) -- it must
    /// clamp to the SAME real `[0.5, 2.0]` bound `update_particle` already
    /// enforces every substep, landing at the upper bound here since the
    /// real ratio (6.0) is far outside it.
    #[test]
    fn condensing_from_a_much_larger_prior_volume_clamps_to_the_upper_compression_bound() {
        let water = NewtonianFluidMaterial::new(1.0, 0.0, 5.0, 7.0);
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.volume = 6.0; // real prior (steam) volume: 6x this material's true rest volume
        water.init_particle_from_transition(&mut p);

        let true_initial_volume = p.mass / water.rest_density;
        assert_eq!(p.initial_volume, true_initial_volume);
        let j = p.deformation_gradient.determinant();
        assert!(
            (j - 2.0).abs() < 1.0e-4,
            "ratio 6.0 is far outside [0.5, 2.0], must clamp to the upper bound 2.0, got {j}"
        );
        assert!(
            (p.volume - true_initial_volume * 2.0).abs() < 1.0e-4,
            "volume must be true_initial_volume * clamped j, not an instant jump to \
             true_initial_volume alone: got {}",
            p.volume
        );
    }

    /// Same mechanism, opposite direction: transitioning from a much-MORE-
    /// dense prior material clamps to the lower compression bound instead
    /// of silently allowing an unbounded compression spike.
    #[test]
    fn transitioning_from_a_much_smaller_prior_volume_clamps_to_the_lower_compression_bound() {
        let water = NewtonianFluidMaterial::new(1.0, 0.0, 5.0, 7.0);
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.volume = 0.1; // real prior volume: far denser than water's own rest state
        water.init_particle_from_transition(&mut p);

        let j = p.deformation_gradient.determinant();
        assert!(
            (j - 0.5).abs() < 1.0e-4,
            "ratio 0.1 is far outside [0.5, 2.0], must clamp to the lower bound 0.5, got {j}"
        );
    }

    /// Regression parity: a prior material with the SAME real rest density
    /// (prior volume already equal to this material's true rest volume)
    /// must land at j=1 exactly -- no artificial jump introduced where none
    /// is physically warranted.
    #[test]
    fn transitioning_from_an_already_matching_volume_introduces_no_artificial_jump() {
        let water = NewtonianFluidMaterial::new(1.0, 0.0, 5.0, 7.0);
        let mut p = Particle::zeroed();
        p.mass = 1.0;
        p.volume = 1.0; // already equal to mass/rest_density
        water.init_particle_from_transition(&mut p);

        let j = p.deformation_gradient.determinant();
        assert!(
            (j - 1.0).abs() < 1.0e-5,
            "no real density mismatch should mean no artificial jump: got j={j}"
        );
    }
}
