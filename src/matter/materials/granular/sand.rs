use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, GranularProps, scale_lame};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    LOG_CLAMP, MIN_J, advance_deformation_gradient, corotated_elastic_stress, elastic_wave_dt,
    lame_from_young, self_consistent_plastic_multiplier,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Dry medium-sand grain diameter (standard soil classification), the value
/// Haeri & Skonieczny 2022 (CMAME, arXiv:2111.01523) calibrate their
/// excavation nonlocal-granular-fluidity case against. `scale_contract`
/// uses it to check that a scene's `dx_meters` lies in this material's
/// continuum window (see that module for the formula).
pub const GRAIN_DIAMETER_M: f32 = 0.3e-3;

/// Surface tension of water at 20 C, `N/m`: 72.74 mN/m, IAPWS R1-76(2014),
/// "Revised Release on Surface Tension of Ordinary Water Substance", Table
/// 1, which its equation `B * tau^mu * (1 - b * tau)` (`tau = 1 - T/Tc`)
/// reproduces. Was 0.072, the value at 25 C (71.97 mN/m in the same table).
pub const WATER_SURFACE_TENSION_N_M: f32 = 0.07274;

/// Capillary cohesion stress (SI Pa) of wet sand from grain-scale physics,
/// so it follows grain diameter and porosity instead of a flat
/// `saturation_cohesion_coeff`.
///
/// Two mechanisms, chained:
/// 1. Capillary bridge force between two grains in contact (Lian, Thornton
///    & Adams 1993, "A theoretical study of the liquid bridge forces
///    between two rigid spherical bodies," J. Colloid Interface Sci.
///    161:138-147): `F_c = 2*pi*R*gamma*cos(theta)`.
/// 2. Continuum tensile stress from that force (Rumpf 1962):
///    `sigma_T = (1-porosity)/porosity * F_c / (4*R^2)`, with Rumpf's
///    `porosity * coordination_number ~= pi`, which removes the need for a
///    measured coordination number (as in Hornbaker et al. 1997; Halsey &
///    Levine 1998).
///
/// `porosity`: void fraction `e/(1+e)`, from the same cohesionless-soil
/// void-ratio data `min_volume_jacobian` uses -- 0.355 (dense, e_min=0.55)
/// to 0.479 (loose, e_max=0.92). `contact_angle_deg`: water on clean quartz
/// is close to 0 (fully wetting).
///
/// Returns SI pascals. `cohesion_bonus_pa` adds it into the same stress space
/// as `lambda`/`mu`, so convert it the way they were: raw `lame_from_young`
/// values take it raw; `SimConfig::lame_from_si` values take
/// `SimConfig::stress_from_si` (as `examples/cpu/sand_water_saturation.rs`
/// does throughout).
pub fn capillary_cohesion_stress_pa(
    grain_diameter_m: f32,
    porosity: f32,
    contact_angle_deg: f32,
) -> f32 {
    let r = grain_diameter_m * 0.5;
    let f_c = 2.0
        * std::f32::consts::PI
        * r
        * WATER_SURFACE_TENSION_N_M
        * contact_angle_deg.to_radians().cos();
    (1.0 - porosity) / porosity * f_c / (4.0 * r * r)
}

/// Small-strain Kelvin-Voigt viscosity (SI Pa.s) for `elastic_viscosity`,
/// from sand's measured damping: two sources bracket a small-strain damping
/// ratio `zeta` of 0.5%-2% for clean sand (Seed & Idriss 1970, "Soil Moduli
/// and Damping Factors for Dynamic Response Analyses"; Darendeli 2001 PhD
/// dissertation, `D_min` for clean sand). `damping_ratio` must lie in that
/// range. Where to sit in it is a cost trade-off: this term's viscous CFL
/// bound (`DruckerPragerMaterial::timestep_bound`) took the
/// `sand_water_saturation` scene from 31 to 56 substeps per step at 1%,
/// measured before the factor 2 below, so expect more now.
///
/// Equivalent viscous damping at the 1 Hz reference (`omega = 2*pi rad/s`)
/// of Seed & Idriss's resonant-column tests. The textbook
/// `eta = 2*zeta*G/omega` assumes `sigma = 2G*eps + 2*eta*D`; this engine's
/// `kirchhoff_stress` applies `eta*D_dev` without the 2 (see
/// `ViscoelasticMaterial`'s test), so here `eta = 4*zeta*G/omega`, as for
/// ice (`RankineMaterial::q_factor_elastic_viscosity_pa_s`, checked by hand
/// and by a cyclic-oscillation test).
///
/// `shear_modulus_pa`: SI shear modulus (`E / (2*(1+nu))`) of the same
/// material, so a stiffer sand gets proportionally more damping, as `zeta`
/// is defined relative to `G`.
///
/// Returns SI Pa.s; convert it the way the caller's `lambda`/`mu` were (see
/// `capillary_cohesion_stress_pa`).
pub fn small_strain_elastic_viscosity_pa_s(shear_modulus_pa: f32, damping_ratio: f32) -> f32 {
    debug_assert!(
        (0.005..=0.02).contains(&damping_ratio),
        "damping_ratio {damping_ratio} outside the cited Seed & Idriss / Darendeli range for \
         clean sand (0.5%-2%) -- not a real small-strain sand value"
    );
    const REFERENCE_OMEGA: f32 = 2.0 * std::f32::consts::PI;
    4.0 * damping_ratio * shear_modulus_pa / REFERENCE_OMEGA
}

// Test-only counters of how often the NGF rate-limiter cap (`project()`'s
// `gamma.min(gamma_rate_limited)`) binds during a collapse.
#[cfg(test)]
static NGF_CAP_TOTAL_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static NGF_CAP_BINDING_COUNT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
#[cfg(test)]
static NGF_CAP_SEVERITY_SUM_X1E6: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

/// Drucker-Prager elastoplastic sand. Ref: Klar et al. 2016.
#[derive(Debug, Clone, Copy)]
pub struct DruckerPragerMaterial {
    /// Rest density in grid units (`rho / reference density`, the same
    /// scale `SpawnRegion::mass_from` gives mass), when built from SI
    /// properties; `None` for grid-unit constructors. Contact reads it to
    /// size a particle's undeformed domain, see
    /// [`MaterialModel::rest_density`].
    pub rest_density: Option<f32>,
    pub lambda: f32,
    pub mu: f32,
    /// φ₀: Initial friction angle (radians). Dry sand ≈ 35° = 0.611 rad. (Klar 2016 h₀)
    pub friction_angle: f32,
    /// φ₁: Friction hardening sensitivity -- slope of φ(q) near q=0. (Klar 2016 h₁)
    pub hardening_peak: f32,
    /// φ₂: Hardening decay rate -- exponential falloff coefficient. (Klar 2016 h₂)
    pub hardening_decay: f32,
    /// φ_r: Residual friction angle (radians). ≈ 10° = 0.175 rad. (Klar 2016 h₃)
    pub friction_residual: f32,
    /// Volume correction factor. 1.0 = full sparkl correction, 0.0 = none.
    pub volume_correction: f32,
    /// Reynolds dilatancy angle ψ (radians). Dense sand ≈ 10–15°.
    ///
    /// When ψ > 0, plastic shear increments drive volumetric expansion:
    /// δεᵥᵖ = sin(ψ) · dq. Physical for dense/compacted sand;
    /// set to 0 for loose/loose-packed sand.
    pub dilatancy_angle: f32,
    /// Yield-surface floor, independent of confining pressure (Pa-equivalent, same
    /// units as `lambda`/`mu`). 0.0 = true cohesionless Mohr-Coulomb (real dry sand,
    /// the Klar 2016 default).
    ///
    /// NOT a claim that dry sand has real cohesion -- it doesn't. This compensates for
    /// a measured continuum-MPM-resolution artifact: pressure-proportional
    /// friction (`alpha * trace`) vanishes in thin, fast-flowing layers where local
    /// confining pressure is near zero, regardless of the friction angle -- confirmed
    /// by three different friction coefficients (DP 35°, µ(I) 20.9-32.8°, µ(I)
    /// 35-40°) all producing IDENTICAL excess runout (~4.7x the Lajeunesse et al. 2004
    /// empirical scaling law for this aspect ratio -- see
    /// `sand_column_collapse_runout_matches_lajeunesse_scaling`). Real grain-scale
    /// effects (interlocking, local rearrangement) give actual sand a baseline
    /// resistance in thin layers that point-wise continuum MPM at this resolution
    /// doesn't capture. Calibrate against that benchmark, not against a literature
    /// "sand cohesion" value (which is ~0 and would be the wrong justification).
    pub cohesion: f32,
    /// The Drucker-Prager cone yield surface, BY CONSTRUCTION in the published model
    /// (Klar 2016, verified identical in sparkl/wgsparkl), only ever trims DEVIATORIC
    /// (shear) strain -- `project()`'s Case III preserves `trace(eps)` exactly. A
    /// near-hydrostatic impact (mostly compression, little shear) is judged "elastic"
    /// essentially always, regardless of how hard the impact is, because `gamma` stays
    /// negative -- nothing in the published model caps pure volumetric compression.
    /// The real closure for exactly this gap is a CAP surface (DiMaggio & Sandler
    /// 1971, "Material Model for Granular Soils," J. Eng. Mech. Div. ASCE; formalized
    /// in Resende & Martin 1985, "Formulation of Drucker-Prager Cap Model,"
    /// J. Eng. Mech. 111(7)) -- a moving surface closing off the open end of the
    /// Coulomb-Mohr cone at high confining pressure, so compression alone is also
    /// bounded by the material's own packing limit.
    ///
    /// Same mechanism as `StomakhinMaterial`'s `min_plastic_jacobian`: a hard floor on
    /// the STORED singular values' product (the actual `deformation_gradient` written
    /// back), applied AFTER the shear-yield projection so friction/cohesion physics
    /// stay unaffected -- only engages when volumetric compression alone would exceed
    /// sand's own packing limit. `update_particle`'s comment (at the point of use)
    /// records that this already goes beyond a naive clamp: the excess velocity along
    /// the compressed axis is zeroed too, an inelastic (dissipative) event, not an
    /// elastic rebound off the floor. A simplified, single-J-threshold stand-in for
    /// DiMaggio-Sandler's full elliptical (p, q) cap, not a literal transcription --
    /// disclosed as such, matching this file's own convention for the Cosserat term.
    ///
    /// 0.807, not Snow's 0.6 (which has no source for sand). Derived
    /// from measured void-ratio limits for cohesionless soils: `(1+e_min)/(1+e_max)`
    /// with the database mean `e_min=0.55, e_max=0.92` gives `1.55/1.92 = 0.8073`, a
    /// ~19.3% maximum volumetric strain. At 0.6 (40% compression), an ordinary
    /// push essentially never reached the floor, leaving compression purely elastic --
    /// which is what read as sand "springing back" no matter how hard it's disturbed.
    /// The rescale itself floors each axis individually first -- see
    /// `update_particle`'s comment at the point of use for why (a pure
    /// product-rescale can't recover an axis already at zero under an extreme impact).
    pub min_volume_jacobian: f32,
    /// Compaction hardening: extra friction angle (radians) per unit of net
    /// volumetric COMPACTION at the moment of yielding (`project`'s own `trace`,
    /// ln of the current effective volume ratio vs the particle's initial state --
    /// negative trace = real net volume loss, i.e. densified). 0.0 (default) = zero
    /// coupling between density and friction,
    /// byte-identical to Klar 2016's own DP model. Correctly-directed physics
    /// (denser packing -> higher friction resistance/interlocking is Bolton 1986's
    /// established relative-density-to-friction-angle relation -- the same paper
    /// already cited for the repose-angle target) -- but this coefficient is
    /// a simplified linear proportionality, NOT a claim of Bolton's own
    /// precise empirical dilatancy-index formula. Single-phase (dry) compaction only
    /// -- real wet/saturated consolidation (Terzaghi effective stress, pore-pressure-
    /// gated densification) is a distinct phenomenon needing the mixture-coupling
    /// system, not this field.
    pub compaction_sensitivity: f32,
    /// Couple this material's yield check to a `GranularFluidityField`'s
    /// gathered `g` (see `energy::thermodynamics::granular_fluidity` module
    /// doc for the cited PDE). `false` (default) = byte-identical to
    /// every existing behavior; every current constructor/preset builds via
    /// `..Self::new(...)`, so this cannot change anything unless explicitly
    /// set. See `project()`'s doc for exactly how `g` modulates the
    /// return mapping when enabled.
    pub ngf_enabled: bool,
    /// Extra friction angle (radians) required to INITIATE yielding while
    /// the material isn't currently straining, on top of `friction_angle`'s
    /// own q-hardening. 0.0 (default) = off.
    ///
    /// An undisturbed sand pile is stable anywhere between the angle of
    /// repose (where flow arrests) and a higher maximum angle of stability
    /// (where flow starts) (Bagnold 1954; Jaeger, Nagel & Behringer 1996,
    /// "Granular solids, liquids, and gases," Rev. Mod. Phys. 68; the same
    /// two angles underlie the Bak-Tang-Wiesenfeld 1987 sandpile toppling
    /// rule). `friction_angle` stays the flowing threshold used by the
    /// return mapping; `friction_angle + static_friction_boost` only decides
    /// whether yielding starts while the material is not straining (see
    /// `rest_rate_scale`). The hardening law brings `phi(q)` back to
    /// `friction_angle` for large q whatever the history, so a marginally
    /// yielded particle sits on the yield surface with no margin against
    /// noise: dynamically formed piles crept under heavy damping while
    /// hand-placed ones held at the same angle.
    pub static_friction_boost: f32,
    /// Strain-rate scale (1/time, same units as `velocity_gradient`) over
    /// which `static_friction_boost` decays as the material starts actively
    /// straining: boost is multiplied by `exp(-strain_rate/rest_rate_scale)`,
    /// so it is fully active at rest and fades once flow is underway. A
    /// calibration knob tuned against this engine's substep dynamics, not a
    /// literature value (like `cohesion`). Irrelevant when
    /// `static_friction_boost == 0.0`.
    pub rest_rate_scale: f32,
    /// Opt-in relaxation rate (1/time) of a particle's stored deviatoric
    /// (shear) log-strain, the elastic state in `deformation_gradient`. 0.0
    /// (default) = off.
    ///
    /// The return mapping leaves a just-yielded particle exactly on the yield
    /// surface. A violently collapsed pile ends with its particles riding the
    /// surface at the pressure and hardening of their last yield, not the
    /// lower stress their settled position needs, so grid-transfer noise in
    /// `velocity_gradient` keeps nudging them over and the return mapping
    /// keeps clamping them back, never below. A pre-shaped particle starts at
    /// `deformation_gradient = IDENTITY` and only builds the shear its
    /// geometry needs, below yield, and never yields.
    ///
    /// Granular and clay soils relax stress slowly under sustained sub-yield
    /// shear (secondary consolidation), distinct from Perzyna/mu(I) rate
    /// dependence (how fast yielding starts). Same form as
    /// `ViscoelasticMaterial`'s Kelvin-Voigt relaxation, on DP's elastic
    /// predictor: `dev(t+dt) = dev(t) * exp(-elastic_relaxation_rate *
    /// rest_factor * dt)`, gated by the `rest_factor` of `rest_rate_scale`,
    /// so it never relaxes the elastic support of active flow.
    pub elastic_relaxation_rate: f32,
    /// Opt-in relaxation rate (1/time) of the plastic memory
    /// (`friction_hardening` q and `log_volume_strain`) toward the values
    /// `init_particle` sets (`friction_residual/hardening_peak` and `0.0`).
    /// 0.0 (default) = off.
    ///
    /// Targets what a long-horizon run of a creeping pile showed
    /// (`diag_j_and_plastic_memory_drift_long_horizon`): the elastic state at
    /// rest (|J-1| median and p90 at 0.0 from step 3000 to 25000) while q
    /// stayed far from its baseline (median |q-baseline| 0.696 -> 0.723) and
    /// `log_volume_strain` grew a tail (p90 0.017 -> 0.043). Resetting both
    /// with F (`diag_collapsed_pile_after_full_tensor_state_reset`) froze the
    /// pile.
    ///
    /// It does not arrest creep (`diag_hardening_relaxation_calibration_sweep`
    /// in `tests/accuracy.rs`): the baseline creeps 24.8 -> 16.2 degrees from
    /// step 7500 to 26500, rate 0.01 gives 24.0 -> 15.8, rate 0.1 gives
    /// 22.4 -> 14.8. A continuous relaxation removes the hardening that
    /// supports the settled shape too. For creep use
    /// `post_event_relax_threshold` (edge-triggered, one time). Kept as opt-in
    /// physics: plastic memory fading over time is a reasonable thing to
    /// model elsewhere.
    pub hardening_relaxation_rate: f32,
    /// Opt-in edge-triggered elastic-strain reset: fires once when a
    /// particle's strain rate falls from above this threshold to below it
    /// (it was straining, it just went quiet). 0.0 (default) = off.
    ///
    /// The principle of Cundall 1982's kinetic damping (the paper cited for
    /// `cundall_damping`), which resets state at each detected kinetic-energy
    /// peak to reach static equilibrium: detect an event, act once. Applied
    /// per particle, on a strain-rate falling edge, because parts of a pile go
    /// quiet at different times. `resetDeformation()` (F -> IDENTITY) is a
    /// method of Stomakhin/Jiang's `ziran2020`; neither it nor Cundall wires
    /// it to a per-particle trigger.
    ///
    /// A one-time reset of `deformation_gradient` alone reproduces the frozen
    /// plateau (29.6 degrees, `diag_collapsed_pile_after_deformation_gradient_
    /// only_reset`), while the continuous mechanisms (static/kinetic
    /// hysteresis, `elastic_relaxation_rate` slow and fast,
    /// `hardening_relaxation_rate`) all failed: continuous suppression
    /// removes the ability to hold any shear stress.
    ///
    /// Uses `Particle::hardening_scale` as edge memory (`1.0 +
    /// previous_substep_strain_rate_norm`, offset by 1.0 so it stays positive
    /// for `projection.rs`'s `<= 0.0` safety net and reads 1.0 at rest like
    /// other materials). `DruckerPragerMaterial` does not otherwise use that
    /// field (its `timestep_bound` ignores `_hardening_scale`), and the
    /// 128-byte `Particle` has no spare room for a new one.
    pub post_event_relax_threshold: f32,
    /// Cosserat/micropolar grain-scale rolling-resistance coupling (de Borst,
    /// Sabet & Hageman 2022, "Non-associated Cosserat plasticity", IJMS
    /// 230:107535, open access). The repose angle is set by rolling friction,
    /// grain size and container geometry, not damping, E, nu or restitution,
    /// and a scalar sliding-friction model has no rolling at all. 0.0
    /// (default) = off.
    ///
    /// Adapted, not the paper's formula: their generalized J2 = a1*(sT:s) +
    /// a2*(s:s) + a3*(mT:m)/l^2 assumes a possibly asymmetric stress, while
    /// this material works in SVD principal-stretch space (`dev`, `dev_norm`),
    /// symmetric by construction. The couple-stress magnitude is added as a
    /// strengthening term on the yield threshold `cohesion` occupies,
    /// converted to strain-space units by the same `/(2*mu)`. The coupling's
    /// existence and the elastic relation giving `m` are cited; its
    /// integration point is this SVD formulation's.
    pub cosserat_modulus_pa: f32,
    /// Internal length scale `l` of the coupling above. With the literal
    /// grain diameter (`GRAIN_DIAMETER_M`, 0.3 mm) `l^2 ~ 9e-8 m^2` crushes the
    /// couple-stress term to ~1e-8 of the yield check's other terms: a ~1 cm
    /// cell cannot resolve rotation gradients at grain scale (dry-sand shear
    /// bands are 10-20 grain diameters, a few mm). Set it to the scene's
    /// `dx_meters`, what the grid resolves, a simulation-scale calibration
    /// like the NGF `EFFECTIVE_GRAIN_DIAMETER_M = 0.008`, not a grain size.
    pub cosserat_length_scale_m: f32,
    /// Apparent cohesion (Pa-equivalent, same units as `cohesion`) per unit
    /// of a coupled `Particle::scalar_field` read as saturation (see
    /// `cohesion_bonus_pa`). 0.0 (default) = off; inert unless a scene wires a
    /// `ScalarDiffusionField` to `scalar_field` and sets this.
    ///
    /// Separate from `cohesion`, which compensates an MPM resolution artifact
    /// and is calibrated against the Lajeunesse runout; this one is cited
    /// capillary cohesion. Merging them would spoil that calibration, so
    /// they are added side by side at the yield check, like
    /// `couple_stress_term` next to `cohesion_term`.
    pub saturation_cohesion_coeff: f32,
    /// Saturation degree (0-1) at which apparent capillary cohesion peaks
    /// (see `cohesion_bonus_pa`). The pendular-regime peak lies at low
    /// saturation (Hornbaker et al. 1997; Halsey & Levine 1998; Scheel et al.
    /// 2008, "Morphological clues to wet granular pile stability," Nat.
    /// Mater.); 0.3 (default) is an estimate in that range, a calibration
    /// knob like `cosserat_length_scale_m`. Irrelevant when
    /// `saturation_cohesion_coeff == 0.0`.
    pub pendular_regime_ceiling: f32,
    /// Kelvin-Voigt viscous damping on the ELASTIC (sub-yield) response --
    /// same mechanism `ViscoelasticMaterial::kirchhoff_stress` already
    /// implements (Christensen 1982, "Theory of Viscoelasticity"; MPM
    /// usage: Stomakhin et al. 2014 Sec.3). `tau_v = viscosity * D_dev`,
    /// `D` the symmetric strain-rate from the APIC velocity gradient --
    /// depends on STRAIN RATE, not velocity itself, so a rigid or
    /// free-falling body (D=0, no internal deformation) is completely
    /// unaffected. This is the structural reason it is safe where
    /// grid-level Cundall damping (opposes velocity CHANGE, i.e. force,
    /// including gravity's own) is not.
    ///
    /// The Drucker-Prager cone (Klar 2016) only trims plastic strain past
    /// yield; below it this material is a perfect elastic spring, and a firm
    /// push on a settled pile rang for 500+ substeps. 0.0 (default) = off.
    pub elastic_viscosity: f32,
    /// Opt-in volumetric-correction limiter, after Tampubolon, Gast, Klar, Fu,
    /// Teran, Jiang & Museth 2017 (SIGGRAPH/ACM TOG 36:4, "Multi-species
    /// simulation of porous sand and water mixtures"; the flag takes the first
    /// author's name, Andre Pradhana Tampubolon), re-derived against Blatny &
    /// Gaume 2025's explicit adaptation (`tmp/matter/src/simulation/
    /// plasticity.cpp`), since the original targets an implicit solve.
    /// `false` (default) = off.
    ///
    /// `project`'s tension-cutoff branch (`trace > 0.0`, or a zero deviator
    /// found outside the cone;
    /// "Case III" in this file is the shear-yield cone branch) returns the
    /// deformation to `sigma = (1,1)`, the textbook cone-apex return (the same
    /// as `sparkl`'s `plasticity_drucker_prager.rs`), every time it fires,
    /// with no memory. Repeated firings (poured batches overshooting into
    /// expansion on impact) each keep a small volume gain: a slowly poured
    /// pile gained ~2.25 cells of height per batch
    /// (`sand_pile_built_by_slow_pour_tracking_real_surface_height`).
    ///
    /// One give-back per compaction cycle (Blatny's four-branch model does
    /// not map onto this two-branch material): the first firing since the
    /// particle was last non-yielding (`project` returned `None` or took the
    /// shear branch) applies the full apex return; later firings, until a
    /// non-yielding step clears the flag, are no-ops
    /// (`ProjectedBranch::DebtBlocked`, trial state unchanged, no volume or
    /// hardening update). The persistent flag is
    /// `Particles::eps_pl_vol_pradhana`.
    pub use_pradhana: bool,
}

/// Bundled inputs for `DruckerPragerMaterial::project` -- grew past clippy's
/// `too_many_arguments` threshold (7) once `cosserat_curvature` joined the
/// existing NGF/rate-hardening inputs, so the loose parameters were folded
/// into a struct here rather than silencing the lint. Private, single call
/// site (`update_particle`) -- not a public API, just a local grouping.
struct ProjectInputs {
    sigma: Vec2,
    log_volume_strain: f32,
    q: f32,
    dt: f32,
    nonlocal_fluidity: f32,
    strain_rate_norm: f32,
    cosserat_curvature: Vec2,
    /// Apparent cohesion (Pa-equivalent) from `cohesion_bonus_pa`; 0.0 when
    /// `saturation_cohesion_coeff == 0.0`.
    cohesion_bonus_pa: f32,
    /// See `Particles::eps_pl_vol_pradhana`'s doc. 0.0 when
    /// `use_pradhana == false` (every existing preset/scene).
    eps_pl_vol_pradhana: f32,
}

/// Outcome of one `DruckerPragerMaterial::project` call, driving
/// `eps_pl_vol_pradhana`'s one-give-back flag (see `use_pradhana`):
/// `TensionCutoff` sets it; `DebtBlocked` leaves it set (suppressed, still
/// used up); an elastic step or the shear-yield branch ("Case III") clears
/// it, ending the compaction cycle.
enum ProjectedBranch {
    TensionCutoff,
    /// The tension-cutoff condition fired again, but `eps_pl_vol_pradhana`
    /// was already set from an earlier firing this same compaction cycle --
    /// no correction applied (`update_particle` receives the trial `sigma`
    /// unchanged, `dq=0.0`, a verified no-op on `log_volume_strain`/
    /// `friction_hardening`, see this branch's own construction site).
    DebtBlocked,
    ShearYield,
}

impl DruckerPragerMaterial {
    /// Construct with Lamé parameters and default Klar 2016 friction-angle hardening.
    ///
    /// Use [`from_young_modulus`](Self::from_young_modulus) if you prefer E/ν inputs.
    pub const fn new(lambda: f32, mu: f32) -> Self {
        Self {
            rest_density: None,
            lambda,
            mu,
            friction_angle: 35.0_f32.to_radians(),
            hardening_peak: 9.0_f32.to_radians(),
            hardening_decay: 0.2,
            friction_residual: 10.0_f32.to_radians(),
            volume_correction: 1.0,
            dilatancy_angle: 0.0,
            cohesion: 0.0,
            min_volume_jacobian: 0.807,
            compaction_sensitivity: 0.0,
            ngf_enabled: false,
            static_friction_boost: 0.0,
            rest_rate_scale: 1.0,
            elastic_relaxation_rate: 0.0,
            hardening_relaxation_rate: 0.0,
            post_event_relax_threshold: 0.0,
            cosserat_modulus_pa: 0.0,
            cosserat_length_scale_m: GRAIN_DIAMETER_M,
            saturation_cohesion_coeff: 0.0,
            pendular_regime_ceiling: 0.3,
            elastic_viscosity: 0.0,
            use_pradhana: false,
        }
    }

    /// Construct from Young's modulus E and Poisson's ratio ν.
    ///
    /// Matches sparkl/wgsparkl API: `DruckerPragerPlasticity::new(E, nu)`.
    /// Canonical demo value (sparkl basic2): E = 1e5, ν = 0.2.
    /// **Grid units, not pascals**: calls [`lame_from_young`] directly and
    /// never touches `dx_meters` or density. For an SI material build an
    /// [`Elastoplastic`](crate::materials::Elastoplastic) with
    /// `model: PlasticityModel::Granular { friction_angle_deg, dilatancy_angle_deg }`
    /// and call its `.material(&config)`; `Self::from_physical` takes a
    /// crate-internal `GranularProps`.
    pub fn from_young_modulus(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self::new(lambda, mu)
    }

    /// Cohesionless: φ=35°, no dilatancy. Klar 2016 defaults. Dry sand regime.
    pub fn cohesionless(young_modulus: f32, poisson_ratio: f32) -> Self {
        Self::from_young_modulus(young_modulus, poisson_ratio)
    }

    /// Low friction: φ=25°, weaker hardening. Loose silty soil regime.
    pub fn low_friction(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            friction_angle: 25.0_f32.to_radians(),
            hardening_peak: 4.0_f32.to_radians(),
            hardening_decay: 0.1,
            friction_residual: 5.0_f32.to_radians(),
            ..Self::new(lambda, mu)
        }
    }

    /// Dilatant: φ=38°, ψ=12° Reynolds dilatancy. Dense compacted sand regime.
    pub fn dilatant(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            friction_angle: 38.0_f32.to_radians(),
            dilatancy_angle: 12.0_f32.to_radians(),
            ..Self::new(lambda, mu)
        }
    }

    /// Gravel: φ=42°, ψ=8° dilatancy. Gravel's internal friction angle spans
    /// ~30-48°, dense well-graded gravel above 45° (ScienceDirect's "Friction
    /// Angle" overview, Geoengineer.org's "Angle of Internal Friction");
    /// 42° is in the dense band, the register of `dilatant()`'s 38° for sand.
    /// Larger, more angular grains interlock and dilate more than sand; 8° is
    /// an estimate between `cohesionless` (0°) and `dilatant` (12°), not a
    /// measured gravel dilatancy.
    pub fn gravel(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            friction_angle: 42.0_f32.to_radians(),
            dilatancy_angle: 8.0_f32.to_radians(),
            ..Self::new(lambda, mu)
        }
    }

    /// For a loose cohesionless pile the theoretical angle of repose equals
    /// the internal friction angle (Coulomb 1776; Lambe & Whitman, *Soil
    /// Mechanics*, 1969), so this returns `friction_angle` in degrees, for a
    /// demo panel, a test or LP's material authoring. Exact only for the pure
    /// cohesionless case: a nonzero `cohesion` (a numerical compensation
    /// here) or `dilatancy_angle` moves a pile's angle away from it.
    pub const fn predicted_repose_angle_deg(&self) -> f32 {
        self.friction_angle.to_degrees()
    }

    /// Friction coefficient α(q) derived from friction angle φ(q).
    /// φ(q) = friction_angle + compaction_boost + (hardening_peak·q − friction_residual)·exp(−hardening_decay·q)
    /// α(q) = √(2/3) · 2·sin(φ) / (3 − sin(φ))
    ///
    /// `compaction_boost = compaction_sensitivity * max(0, -trace_ln_volume_ratio)` --
    /// `trace_ln_volume_ratio` is `project`'s own `trace` (ln of the current net
    /// volume ratio vs the particle's initial state, instantaneous + accumulated
    /// history combined) at the exact moment of yielding -- see
    /// `compaction_sensitivity`'s doc. Zero when the field is at its 0.0
    /// default, so this is byte-identical to the original q-only formula unless
    /// opted in.
    fn alpha(&self, q: f32, trace_ln_volume_ratio: f32) -> f32 {
        self.alpha_with_phi_delta(q, trace_ln_volume_ratio, 0.0)
    }

    /// The current Mohr-Coulomb friction ANGLE phi(q) itself -- the
    /// shared quantity `alpha_with_phi_delta` converts into the DP-cone
    /// coefficient below, and `current_friction_coefficient` converts into
    /// an ordinary Coulomb wall coefficient (`tan(phi)`, NOT `alpha`,
    /// which is a different, DP-cone-specific number for the same angle --
    /// see that method's doc). Extracted so both conversions share one
    /// real formula instead of drifting apart.
    fn phi(&self, q: f32, trace_ln_volume_ratio: f32, phi_delta: f32) -> f32 {
        let compaction_boost = self.compaction_sensitivity * (-trace_ln_volume_ratio).max(0.0);
        self.friction_angle
            + phi_delta
            + compaction_boost
            + (self.hardening_peak * q - self.friction_residual) * (-self.hardening_decay * q).exp()
    }

    /// Same formula as `alpha`, with an extra additive friction-angle term
    /// (`phi_delta`) -- used by `project`'s static/kinetic onset check (see
    /// `static_friction_boost`'s doc). `phi_delta=0.0` makes this
    /// byte-identical to `alpha`.
    fn alpha_with_phi_delta(&self, q: f32, trace_ln_volume_ratio: f32, phi_delta: f32) -> f32 {
        let s = self.phi(q, trace_ln_volume_ratio, phi_delta).sin();
        (2.0_f32 / 3.0).sqrt() * (2.0 * s) / (3.0 - s)
    }

    /// Drucker-Prager return mapping in log-strain (Hencky) space.
    ///
    /// Returns `Some((projected_sigma, delta_q))` if projection occurred (plastic step),
    /// `None` if the trial state is inside the yield surface (elastic step).
    ///
    /// Self-consistent (closest-point) return mapping: `alpha` is evaluated at
    /// the end-of-step hardening `q + gamma`, not the pre-step `q` (Simo & Taylor
    /// 1985, "Consistent tangent operators for rate-independent
    /// elastoplasticity," CMAME 48:101-118; Simo & Hughes, *Computational
    /// Inelasticity*, 1998). `sparkl::DruckerPragerPlasticity::project_
    /// deformation_gradient` and `wgsparkl::models::drucker_prager::project_
    /// deformation_gradient` use the cheaper pre-step `q`. `alpha` depends on
    /// `q + gamma` and `gamma` on `alpha`, so this is solved by fixed-point
    /// iteration (`phi(q)` is bounded and smooth). `q`, the accumulated
    /// plastic shear strain, keeps growing slowly under sustained load even
    /// in a settled pile, as in critical-state soil mechanics (friction
    /// relaxing from peak toward residual).
    ///
    /// # Nonlocal Granular Fluidity coupling (`ngf_enabled`)
    /// Kamrin & Henann's coupling (arXiv:1408.5205 eq. 4) is rate-explicit:
    /// `γ̇ = g·μ`, plastic flow at a finite rate set by the fluidity `g`. This
    /// return mapping is rate-independent (full relaxation onto the surface
    /// every step). The bridge is Perzyna (1966) viscoplastic regularization,
    /// not Haeri & Skonieczny 2022's formulation (a full rate-explicit
    /// hyperelastic scheme): cap the plastic multiplier applied this step at
    /// `g·μ·dt`, so fluidity throttles how fast a point flows without moving
    /// the yield surface. `μ` comes from the trial quantities below, with the
    /// formula of `MuIRheologyMaterial::update_particle` (`sand_mui.rs`):
    /// `p_trial = -(lambda+mu)*trace`, `q_trial = sqrt(2)*mu*dev_norm`,
    /// `μ = q_trial/p_trial = sqrt(2)*dev_norm/(-ratio*trace)`.
    fn project(&self, inputs: ProjectInputs) -> Option<(Vec2, f32, ProjectedBranch)> {
        let ProjectInputs {
            sigma,
            log_volume_strain,
            q,
            dt,
            nonlocal_fluidity,
            strain_rate_norm,
            cosserat_curvature,
            cohesion_bonus_pa,
            eps_pl_vol_pradhana,
        } = inputs;
        let sigma = sigma.abs().max(Vec2::splat(LOG_CLAMP));
        // Hencky (logarithmic) strain, shifted by the accumulated volumetric offset.
        let eps = Vec2::new(
            sigma.x.ln() + log_volume_strain * 0.5,
            sigma.y.ln() + log_volume_strain * 0.5,
        );
        let trace = eps.x + eps.y;
        let dev = eps - Vec2::splat(trace * 0.5);
        let dev_norm = dev.length();

        // Tension cutoff: expansion projects to identity (σ = 1), the cone's tip.
        // dq = dev_norm only -- friction hardening is driven by shear, not volumetric expansion.
        // Using eps.length() here would include the log_volume_strain offset and cause
        // unbounded q growth in static/settled sand. The cone-apex return itself is
        // the textbook one (as in `sparkl`); its repeated firing is what
        // `use_pradhana` limits.
        //
        // A zero deviator under compression is NOT sent to the tip here.
        // Klar et al. 2016 (sec. 7.1) test their Case I (`gamma <= 0`: inside
        // the cone, returned unchanged) before Case II (the tip, for a zero
        // deviator or expansion), and a purely isotropic compression sits on
        // the cone's axis, inside it. Testing `dev_norm == 0.0` first, as
        // `sparkl` does, made such a particle stress-free and booked its
        // compression as plastic `log_volume_strain`; a zero deviator is only
        // sent to the tip below, once the cone check has found it outside.
        if trace > 0.0 {
            // Pradhana limiter (`use_pradhana`): a later firing within the same
            // compaction cycle (`eps_pl_vol_pradhana` already set) passes the
            // trial state through unchanged, no update of `log_volume_strain`/
            // `friction_hardening` (see `update_particle`'s `DebtBlocked`).
            if self.use_pradhana && eps_pl_vol_pradhana > 0.0 {
                return Some((sigma, 0.0, ProjectedBranch::DebtBlocked));
            }
            return Some((Vec2::ONE, dev_norm, ProjectedBranch::TensionCutoff));
        }

        // Yield function: γ = |dev_ε| + ratio · tr · α − cohesion/(2µ).
        // Klar 2016 eq. 25, d=2: (d·λ + 2µ)/(2µ) = (2λ+2µ)/(2µ) = (λ+µ)/µ.
        // Verified against sparkl DruckerPragerPlasticity::project and wgsparkl drucker_prager.wgsl.
        // The cohesion term shifts the yield threshold by a pressure-INDEPENDENT amount --
        // converting stress-space Mohr-Coulomb cohesion c (||dev(sigma)|| <= alpha*p + c)
        // into this strain-space equation via dev(sigma) = 2*mu*dev(eps) gives the c/(2*mu)
        // divisor below. See `cohesion`'s doc comment for why this exists.
        let ratio = (self.lambda + self.mu) / self.mu;
        // `trace` (already ln(current effective volume ratio) relative to the initial
        // state, folding in BOTH the instantaneous trial state and accumulated
        // `log_volume_strain` history) is only reached here once already confirmed
        // <= 0 above -- i.e. real net compaction, at the exact moment of yielding.
        // A far more universally-responsive compaction signal than
        // `log_volume_strain` alone, which Case III's shear-only projection keeps
        // nearly invariant by construction for non-dilatant sand (dilatancy_angle=0).
        let cohesion_term = self.cohesion / (2.0 * self.mu);
        // Apparent capillary cohesion (see `cohesion_bonus_pa`), converted to
        // strain space like `cohesion_term` and kept as its own term.
        let saturation_cohesion_term = cohesion_bonus_pa / (2.0 * self.mu);

        // Cosserat rolling-resistance strengthening (see `cosserat_modulus_pa`
        // for the citation and the adapted integration). Zero when
        // `cosserat_modulus_pa == 0.0`.
        let couple_stress_term = if self.cosserat_modulus_pa != 0.0 {
            let m = crate::materials::granular::cosserat::elastic_couple_stress_2d(
                cosserat_curvature,
                self.cosserat_modulus_pa,
                self.cosserat_length_scale_m,
            );
            m.length() / (2.0 * self.mu)
        } else {
            0.0
        };

        // Static/kinetic onset check (see `static_friction_boost`'s doc).
        // Skipped entirely (zero cost, zero behavior change) when the field
        // is at its 0.0 default -- every existing preset/constructor. When
        // enabled: a HIGHER, boosted friction angle decides whether yielding
        // starts at all while the material isn't currently straining; the
        // ACTUAL projection below always uses the ordinary, unboosted,
        // already-proven `alpha(q)` -- extra resistance only gates the
        // ONSET of flow, never its sustained rate, matching real Coulomb
        // static-vs-kinetic behavior (more force to start sliding than to
        // keep something already sliding moving).
        if self.static_friction_boost != 0.0 {
            let rest_factor = if self.rest_rate_scale > 0.0 {
                (-strain_rate_norm / self.rest_rate_scale).exp()
            } else {
                0.0
            };
            let phi_delta = self.static_friction_boost * rest_factor;
            let gamma_onset_check = self_consistent_plastic_multiplier(
                dev_norm + ratio * trace * self.alpha_with_phi_delta(q, trace, phi_delta)
                    - cohesion_term
                    - saturation_cohesion_term
                    - couple_stress_term,
                q,
                |q_trial| {
                    dev_norm + ratio * trace * self.alpha_with_phi_delta(q_trial, trace, phi_delta)
                        - cohesion_term
                        - saturation_cohesion_term
                        - couple_stress_term
                },
            );
            if gamma_onset_check <= 0.0 {
                return None; // Boosted-at-rest threshold not crossed -- stays elastic.
            }
        }

        // Self-consistency: `alpha(q + gamma)` depends on gamma, and gamma depends
        // on alpha -- shared iteration logic lives in `self_consistent_plastic_
        // multiplier` (see its doc for the citation and why it's a
        // generic, cross-material solver, not DP-specific), this closure supplies
        // only DP's own yield equation. Single-pass (pre-step-q) value seeds the
        // initial guess.
        let initial_gamma = dev_norm + ratio * trace * self.alpha(q, trace)
            - cohesion_term
            - saturation_cohesion_term
            - couple_stress_term;
        let gamma = self_consistent_plastic_multiplier(initial_gamma, q, |q_trial| {
            dev_norm + ratio * trace * self.alpha(q_trial, trace)
                - cohesion_term
                - saturation_cohesion_term
                - couple_stress_term
        });

        if gamma <= 0.0 {
            return None; // Inside yield surface -- elastic step.
        }

        // NGF rate limiter (see this function's doc). `-ratio*trace > 0` here
        // (trace <= 0 above, ratio > 0), so `mu_ratio` is well defined.
        //
        // `dev_norm/(-ratio*trace)` alone is a strain-space ratio, not the true
        // stress ratio q_trial/p_trial (which needs
        // `sqrt(2)*mu*dev_norm / p_trial`) -- omitting the `self.mu` factor
        // understates `mu_ratio` by ~3600x at this scene's SI-to-grid
        // scaling, making `gamma_rate_limited` (and every collapse this
        // coupling is meant to permit) 3600x too small.
        let gamma = if self.ngf_enabled {
            let mu_ratio =
                std::f32::consts::SQRT_2 * self.mu * dev_norm / (-ratio * trace).max(1e-9);
            let gamma_rate_limited = (nonlocal_fluidity * mu_ratio * dt).max(0.0);
            #[cfg(test)]
            {
                NGF_CAP_TOTAL_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if gamma_rate_limited < gamma {
                    NGF_CAP_BINDING_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // Sum of the RATIO the cap forces gamma down to, so the
                    // average severity is recoverable (not just a binary
                    // bound/not-bound count).
                    let sev = (gamma_rate_limited / gamma.max(1e-12)).clamp(0.0, 1.0);
                    NGF_CAP_SEVERITY_SUM_X1E6
                        .fetch_add((sev * 1.0e6) as u64, std::sync::atomic::Ordering::Relaxed);
                }
            }
            gamma.min(gamma_rate_limited)
        } else {
            gamma
        };
        if gamma <= 0.0 {
            return None; // Yielded, but NGF's local fluidity hasn't built up
            // enough yet to permit real flow this step -- an elastic step
            // for now, not a bug (the whole point of a finite-rate coupling).
        }

        // Klar et al. 2016's Case II for a zero deviator outside the cone: no
        // direction to project along, so the tip. Under compression (all that
        // reaches here) this needs a negative friction term, so no preset takes it.
        if dev_norm == 0.0 {
            if self.use_pradhana && eps_pl_vol_pradhana > 0.0 {
                return Some((sigma, 0.0, ProjectedBranch::DebtBlocked));
            }
            return Some((Vec2::ONE, 0.0, ProjectedBranch::TensionCutoff));
        }

        // Project onto yield surface in log-strain space, then exponentiate.
        let h = eps - gamma * (dev / dev_norm);
        Some((
            Vec2::new(h.x.exp(), h.y.exp()),
            gamma,
            ProjectedBranch::ShearYield,
        ))
    }
}

impl FromSI<GranularProps> for DruckerPragerMaterial {
    fn from_physical(props: &GranularProps, config: &crate::SimConfig) -> Self {
        let (lambda, mu) = scale_lame(
            props.elastic.e_pa,
            props.elastic.nu,
            props.elastic.rho_kg_m3,
            config,
        );
        Self {
            friction_angle: props.friction_angle_deg.to_radians(),
            dilatancy_angle: props.dilatancy_angle_deg.to_radians(),
            rest_density: Some(props.elastic.rho_kg_m3 / config.reference_density_kg_m3),
            ..Self::new(lambda, mu)
        }
    }
}

impl MaterialModel for DruckerPragerMaterial {
    fn rest_density(&self) -> Option<f32> {
        self.rest_density
    }

    fn constitutive_model(&self) -> ConstitutiveModel {
        ConstitutiveModel::DruckerPrager
    }

    fn corotated_lame_params(&self) -> Option<(f32, f32)> {
        if self.elastic_viscosity == 0.0 {
            Some((self.lambda, self.mu))
        } else {
            None
        }
    }

    /// Current Coulomb wall-friction coefficient `tan(phi(q, trace))` (see
    /// `MaterialModel::current_friction_coefficient`; MIBF, Blatny & Gaume
    /// 2025). `tan(phi)`, not `alpha(q, trace)`: `alpha` is the Drucker-Prager
    /// cone's coefficient (`sqrt(2/3)*2*sin(phi)/(3-sin(phi))`), while
    /// `apply_coulomb_wall` uses the Mohr-Coulomb `mu = tan(phi)`
    /// (`friction_impulse = mu * normal_speed`). At `phi = 35` degrees `alpha =
    /// 0.386` and `tan(phi) = 0.700`, which is `FrictionBoundary`'s default
    /// 0.7.
    ///
    /// `friction_hardening` stores `q`; `phi` also takes `trace`, which is not
    /// stored per particle. Its only use is `compaction_boost =
    /// compaction_sensitivity * max(0, -trace)`, exactly zero when
    /// `compaction_sensitivity == 0.0` (every preset), so `self.phi(q, 0.0,
    /// 0.0).tan()` is exact there. Otherwise this returns `None` (the boundary
    /// keeps its own coefficient) rather than pay an SVD per particle for a
    /// feature nothing uses.
    fn current_friction_coefficient(&self, particles: &Particles, i: usize) -> Option<f32> {
        if self.compaction_sensitivity != 0.0 {
            return None;
        }
        Some(self.phi(particles.friction_hardening[i], 0.0, 0.0).tan())
    }

    /// Corotated elastic Kirchhoff stress: τ = 2µ(F−R)Fᵀ + λ(J−1)J·I
    /// R is the rotation from 2D polar decomposition of F, plus a
    /// Kelvin-Voigt viscous term on the deviatoric strain rate -- see
    /// `elastic_viscosity`'s doc. Zero cost, zero behavior change when
    /// `elastic_viscosity == 0.0` (every existing preset/scene).
    fn kirchhoff_stress(&self, particles: &Particles, i: usize) -> Mat2 {
        let elastic =
            corotated_elastic_stress(particles.deformation_gradient[i], self.lambda, self.mu);
        if self.elastic_viscosity == 0.0 {
            return elastic;
        }
        // Same formula as `ViscoelasticMaterial::kirchhoff_stress`'s own
        // Kelvin-Voigt dashpot: tau_v = eta * D_dev, D the symmetric part
        // of the APIC velocity gradient.
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

    fn init_particle(&self, particle: &mut Particle) {
        // q=0 gives φ = h0 − h3 = 25° (too weak). The neutral point where
        // φ(q) = h0 exactly is q = h3/h1. Matches sparkl's plastic_hardening=1.0
        // default (which gives φ ≈ 34.2°). At q = h3/h1 the hardening term = 0.
        particle.friction_hardening = if self.hardening_peak > 0.0 {
            self.friction_residual / self.hardening_peak
        } else {
            0.0
        };
    }

    fn update_particle(&self, ctx: &mut ParticleUpdateCtx, dt: f32) {
        // Deviatoric strain-rate norm (Frobenius) from this substep's APIC
        // velocity_gradient, computed only when used; the "at rest" signal of
        // `static_friction_boost`, `elastic_relaxation_rate`,
        // `hardening_relaxation_rate` and `post_event_relax_threshold`.
        let strain_rate_norm = if self.static_friction_boost != 0.0
            || self.elastic_relaxation_rate != 0.0
            || self.hardening_relaxation_rate != 0.0
            || self.post_event_relax_threshold != 0.0
        {
            let l = *ctx.velocity_gradient;
            let dxx = l.x_axis.x;
            let dyy = l.y_axis.y;
            let dxy = 0.5 * (l.x_axis.y + l.y_axis.x);
            let half_trace = (dxx + dyy) * 0.5;
            let dev_xx = dxx - half_trace;
            let dev_yy = dyy - half_trace;
            (dev_xx * dev_xx + dev_yy * dev_yy + 2.0 * dxy * dxy).sqrt()
        } else {
            0.0
        };

        // Edge-triggered elastic-strain reset (`post_event_relax_threshold`):
        // on the falling edge (above the threshold last substep, below now), F
        // goes to IDENTITY before this substep's trial strain is computed from
        // it, the one-time reset the ablation test found sufficient.
        if self.post_event_relax_threshold > 0.0 {
            let prev_strain_rate_norm = (*ctx.hardening_scale - 1.0).max(0.0);
            let was_straining = prev_strain_rate_norm > self.post_event_relax_threshold;
            let is_straining = strain_rate_norm > self.post_event_relax_threshold;
            if was_straining && !is_straining {
                *ctx.deformation_gradient = Mat2::IDENTITY;
            }
            *ctx.hardening_scale = 1.0 + strain_rate_norm;
        }

        // Exact constant-C integration prevents forward Euler's O(dt^2)
        // volume ratchet from being misclassified as permanent granular
        // compaction by the Hencky return mapping and its history variables.
        let f_trial =
            advance_deformation_gradient(*ctx.deformation_gradient, dt * *ctx.velocity_gradient);

        let (u, sigma, vt) = svd2(f_trial);
        let projected = self.project(ProjectInputs {
            sigma,
            log_volume_strain: *ctx.log_volume_strain,
            q: *ctx.friction_hardening,
            dt,
            nonlocal_fluidity: ctx.nonlocal_fluidity,
            strain_rate_norm,
            cosserat_curvature: ctx.cosserat_curvature,
            cohesion_bonus_pa: self.cohesion_bonus_pa(ctx.scalar_field),
            eps_pl_vol_pradhana: if self.use_pradhana {
                *ctx.eps_pl_vol_pradhana
            } else {
                0.0
            },
        });
        if self.use_pradhana {
            // One give-back per compaction cycle (see `use_pradhana`):
            // `TensionCutoff` uses the flag; `DebtBlocked` leaves it used (a
            // suppressed firing is not a non-yielding step, or every other
            // step would get a free correction); an elastic step or ordinary
            // shear yield ("Case III") clears it for the next cycle.
            *ctx.eps_pl_vol_pradhana = match &projected {
                Some((_, _, ProjectedBranch::TensionCutoff)) => 1.0,
                Some((_, _, ProjectedBranch::DebtBlocked)) => *ctx.eps_pl_vol_pradhana,
                Some((_, _, ProjectedBranch::ShearYield)) | None => 0.0,
            };
        }
        let new_sigma = if let Some((proj_sigma, dq, _)) = projected {
            let sigma_abs = sigma.abs().max(Vec2::splat(LOG_CLAMP));
            let prev_det = sigma_abs.x * sigma_abs.y;
            let new_det = proj_sigma.x * proj_sigma.y;
            let diff = new_det - prev_det;
            let corrected_det = if diff > 0.0 {
                new_det
            } else {
                prev_det + diff * self.volume_correction
            };

            *ctx.log_volume_strain += prev_det.ln() - corrected_det.ln();
            // Pradhana cap (see `use_pradhana`): a tension-cutoff firing can
            // leave a net positive `log_volume_strain` that a later shear-yield
            // step turns into permanent volume (`new_det = prev_det *
            // exp(log_volume_strain)` at the default `volume_correction = 1.0`).
            // With `dilatancy_angle == 0.0` (every current preset) nothing should
            // push net volumetric history positive, so tension cutoff may cancel
            // existing compaction but never leave the particle more expanded than
            // its reference. Applied before dilatancy's own, unclamped, positive
            // contribution below, which a dilatant preset keeps.
            if self.use_pradhana
                && let Some((_, _, ProjectedBranch::TensionCutoff)) = &projected
            {
                *ctx.log_volume_strain = ctx.log_volume_strain.min(0.0);
            }
            let q_max = 5.0 / self.hardening_decay.max(1e-6);
            *ctx.friction_hardening = (*ctx.friction_hardening + dq).min(q_max);
            if self.dilatancy_angle > 0.0 {
                *ctx.log_volume_strain += self.dilatancy_angle.sin() * dq;
            }
            proj_sigma
        } else {
            sigma
        };

        // Opt-in plastic-memory relaxation (`hardening_relaxation_rate`):
        // decays `friction_hardening`/`log_volume_strain` toward their
        // baseline while at rest, yielded or not. Skipped at the 0.0 default.
        if self.hardening_relaxation_rate > 0.0 {
            let rest_factor = if self.rest_rate_scale > 0.0 {
                (-strain_rate_norm / self.rest_rate_scale).exp()
            } else {
                0.0
            };
            if rest_factor > 1.0e-6 {
                let decay = (-self.hardening_relaxation_rate * rest_factor * dt).exp();
                let q_baseline = if self.hardening_peak > 0.0 {
                    self.friction_residual / self.hardening_peak
                } else {
                    0.0
                };
                *ctx.friction_hardening =
                    q_baseline + (*ctx.friction_hardening - q_baseline) * decay;
                *ctx.log_volume_strain *= decay;
            }
        }

        // Opt-in stress relaxation (`elastic_relaxation_rate`): decays the
        // stored deviatoric log-strain toward zero while at rest, yielded or
        // not (a settled particle keeps the shear its last yield left).
        // Skipped at the 0.0 default.
        let new_sigma = if self.elastic_relaxation_rate > 0.0 {
            let rest_factor = if self.rest_rate_scale > 0.0 {
                (-strain_rate_norm / self.rest_rate_scale).exp()
            } else {
                0.0
            };
            if rest_factor > 1.0e-6 {
                let sigma_abs = new_sigma.abs().max(Vec2::splat(LOG_CLAMP));
                let eps = Vec2::new(
                    sigma_abs.x.ln() + *ctx.log_volume_strain * 0.5,
                    sigma_abs.y.ln() + *ctx.log_volume_strain * 0.5,
                );
                let trace = eps.x + eps.y;
                let dev = eps - Vec2::splat(trace * 0.5);
                let dev_norm = dev.length();
                if dev_norm > 1.0e-9 {
                    let decay = (-self.elastic_relaxation_rate * rest_factor * dt).exp();
                    let new_eps = Vec2::splat(trace * 0.5) + dev * decay;
                    Vec2::new(
                        (new_eps.x - *ctx.log_volume_strain * 0.5).exp(),
                        (new_eps.y - *ctx.log_volume_strain * 0.5).exp(),
                    )
                } else {
                    new_sigma
                }
            } else {
                new_sigma
            }
        } else {
            new_sigma
        };

        // Volumetric floor -- see `min_volume_jacobian`'s doc. Applied AFTER the
        // shear-yield projection above and regardless of whether that projection
        // fired (a near-hydrostatic impact is judged "elastic" by the cone above and
        // never reaches it), so friction/cohesion physics are untouched. Uniform
        // rescale (not a per-axis clamp like Snow's) preserves the deviatoric shape
        // the yield projection already chose -- only overall volume is corrected.
        //
        // Take magnitudes FIRST: this engine's `svd2` does NOT guarantee non-negative
        // singular values like textbook SVD -- it keeps U a proper rotation by encoding
        // a reflection as sigma.y going NEGATIVE instead (see svd2's
        // `if u.determinant() < 0.0 { ...; sigma.y = -sigma.y }`). An already-inverted
        // state is exactly the "exceeded sand's packing limit" case this floor exists
        // for, just approached from the other side -- handles "too compressed" and
        // "already inverted" with one uniform rule instead of two different guards.
        let mut new_sigma = new_sigma.abs();
        // Floor each AXIS individually before the product-based rescale below:
        // under a hard enough impact, one singular value can collapse to exactly
        // (or within float noise of) zero on its own axis. The rescale below
        // multiplies both axes by the SAME scalar to bring their PRODUCT up to
        // `min_volume_jacobian`, which cannot recover an axis that's already at
        // zero (0 * any finite scalar is still 0) -- direct instrumentation showed
        // `new_sigma=(26.07, 0.0)`, `j_new=0` exactly, cascading across 46
        // substeps in the failing scenario, collapsing `min_j_terrain` from its
        // correct 0.6 plateau to exactly 0.0. A small per-axis floor, applied
        // before the rescale, guarantees the rescale always has two genuinely
        // nonzero numbers to work with -- for any real (non-degenerate) input
        // this floor never engages, since ordinary singular values sit far above
        // it.
        const MIN_AXIS: f32 = 1.0e-3;
        let sigma_before_floor = new_sigma.max(Vec2::splat(MIN_AXIS));
        new_sigma = sigma_before_floor;
        let j_new = new_sigma.x * new_sigma.y;
        if j_new < self.min_volume_jacobian {
            let rescale = (self.min_volume_jacobian / j_new.max(1e-6)).sqrt();
            new_sigma *= rescale;

            // The floor also stops the velocity: rewriting only the stored F
            // left the velocity driving the same forbidden compression every
            // substep, and with shear yield suppressed (strong Cosserat
            // coupling) `max_particle_speed` ran from 9.8 to 731 m/s over 20
            // steps (`diag_cosserat_high_alpha_collapse_trace`). Hitting a
            // packing limit is inelastic. The rescale is isotropic (both
            // singular values alike), so there is no per-axis direction to
            // damp; the particle comes to a dead stop (a uniform partial
            // damping only brought 731 down to 215).
            let _ = sigma_before_floor;
            let v_local_damped = Vec2::ZERO;
            #[cfg(any(test, feature = "research-diagnostics"))]
            {
                let v_before = *ctx.v;
                let v_after = u * v_local_damped;
                if crate::diagnostics::research_switch("EMERGE_DIAG_FLOOR_FIX").is_some() {
                    println!(
                        "  [floor-fix] v_before={v_before:?} v_after={v_after:?} rescale={rescale:.4}"
                    );
                }
            }
            *ctx.v = u * v_local_damped;
        }

        let sigma_mat = Mat2::from_cols(Vec2::new(new_sigma.x, 0.0), Vec2::new(0.0, new_sigma.y));
        *ctx.deformation_gradient = u * sigma_mat * vt;

        let j = ctx.deformation_gradient.determinant().max(MIN_J);
        let v = (ctx.initial_volume * j).max(1.0e-6);
        *ctx.volume = v;
        *ctx.density = ctx.mass / v;
    }

    /// Apparent cohesion from capillary bridges between grains, the
    /// "sandcastle effect": dry sand has none, a little interstitial liquid
    /// forms menisci that resist shear (Hornbaker, Albert, Barabasi &
    /// Schiffer 1997, "What keeps sandcastles standing," Nature 387:765;
    /// Halsey & Levine 1998, "How Sandcastles Fall," PRL 80:3141), so damp
    /// sand holds a steeper slope than dry or fully saturated sand (the
    /// bridges merge at saturation, which this does not model, see
    /// `pendular_regime_ceiling`).
    ///
    /// `scalar_field` is read as saturation degree Sr in [0,1], whatever
    /// `ScalarDiffusionField` a scene wires to it.
    ///
    /// A linear rise through the pendular regime capped at
    /// `pendular_regime_ceiling`, not any paper's formula; the post-peak
    /// decline through the funicular, capillary and slurry regimes is left
    /// out. 0.0 when `saturation_cohesion_coeff == 0.0`.
    fn cohesion_bonus_pa(&self, scalar_field: f32) -> f32 {
        if self.saturation_cohesion_coeff == 0.0 || self.pendular_regime_ceiling <= 0.0 {
            return 0.0;
        }
        let saturation = scalar_field.clamp(0.0, 1.0);
        self.saturation_cohesion_coeff * (saturation.min(self.pendular_regime_ceiling))
            / self.pendular_regime_ceiling
    }

    fn params(&self) -> MaterialParams {
        MaterialParams {
            model: ConstitutiveModel::DruckerPrager as u32,
            // Contact sizes particles from it (see `MaterialModel::rest_density`);
            // 0 = unknown. No solid stress branch reads this slot.
            rest_density: self.rest_density.unwrap_or(0.0),
            lambda: self.lambda,
            mu: self.mu,
            dp_h0: self.friction_angle,
            dp_h1: self.hardening_peak,
            dp_h2: self.hardening_decay,
            dp_h3: self.friction_residual,
            // compression_limit repurposed for DP: stores dilatancy angle ψ (radians).
            // Snow uses compression_limit for its singular-value clamp (model 4 only).
            compression_limit: self.dilatancy_angle,
            // stretch_limit repurposed for DP: stores the cohesion floor (Pa-equivalent).
            // Not read by the GPU's model==5u branch for any other purpose.
            stretch_limit: self.cohesion,
            volume_ratio_min: self.min_volume_jacobian,
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
        // `ViscoelasticMaterial::timestep_bound` for the Kelvin-Voigt term:
        // without it the substep selector does not see `elastic_viscosity`'s
        // stiffness (at 1000, peak speed went from 12 to 365 instead of
        // damping).
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
}

// Test suite split into two files -- was ~1630 of this file's ~2640 lines. Core,
// always-relevant correctness tests (presets, marginal-yield-surface derivation,
// scale-contract integration) live in `sand_tests.rs`; the much larger, distinct
// Nonlocal Granular Fluidity research investigation's own diagnostics/verification
// tests live in `sand_ngf_tests.rs` -- kept separate since it's a genuinely
// different research topic, not force-merged just because both touch sand. Same
// split reasoning as `elastic.rs` -> `elastic/elastic_tests.rs`.
#[cfg(test)]
mod sand_ngf_tests;
#[cfg(test)]
mod sand_tests;
