use glam::{Mat2, Vec2};

use crate::materials::physical_props::{FromSI, GranularProps, scale_lame};
use crate::materials::svd::svd2;
use crate::materials::utils::{
    LOG_CLAMP, MIN_J, advance_deformation_gradient, carried_volume_ratio, corotated_elastic_stress,
    elastic_wave_dt, lame_from_young, self_consistent_plastic_multiplier,
};
use crate::materials::{ConstitutiveModel, MaterialModel, MaterialParams};
use crate::particle::{Particle, ParticleUpdateCtx, Particles};

/// Real, cited dry-sand grain diameter (medium sand, standard soil-
/// classification convention) -- matches the value Haeri & Skonieczny 2022
/// (CMAME, arXiv:2111.01523) calibrate their Excavation nonlocal-granular-
/// fluidity case against. Used by `scale_contract` to check whether a
/// scene's `dx_meters` is inside the real, physically valid continuum
/// window for this material -- see that module's own doc for the formula.
pub const GRAIN_DIAMETER_M: f32 = 0.3e-3;

/// Real surface tension of water at 20 C -- standard physical constant
/// (CRC Handbook of Chemistry and Physics).
pub const WATER_SURFACE_TENSION_N_M: f32 = 0.072;

/// Real capillary cohesion stress (SI Pa) for wet sand, derived from
/// grain-scale physics -- NOT a hand-picked magnitude. Replaces a flat
/// `saturation_cohesion_coeff` constant with a formula: change the grain
/// diameter or porosity and the cohesion recalculates correctly, instead of
/// needing to be re-tuned by feel.
///
/// Two real, cited mechanisms, chained:
/// 1. Capillary bridge FORCE between two grains in contact (Lian, Thornton
///    & Adams 1993, "A theoretical study of the liquid bridge forces
///    between two rigid spherical bodies," J. Colloid Interface Sci.
///    161:138-147): `F_c = 2*pi*R*gamma*cos(theta)`.
/// 2. Continuum tensile/cohesive stress from that grain-scale force (Rumpf
///    1962): `sigma_T = (1-porosity)/porosity * F_c / (4*R^2)`, using
///    Rumpf's own simplifying assumption `porosity * coordination_number
///    ~= pi` -- eliminates needing an independently measured coordination
///    number, standard in the granular-cohesion literature this file
///    already cites (Hornbaker et al. 1997; Halsey & Levine 1998).
///
/// `porosity`: void fraction `e/(1+e)`, from the SAME cohesionless-soil
/// void-ratio database `min_volume_jacobian` already uses -- 0.355 (dense,
/// e_min=0.55) to 0.479 (loose, e_max=0.92). `contact_angle_deg`: water on
/// clean quartz is close to 0 (fully wetting).
///
/// Returns real SI pascals. Must be converted with the SAME convention the
/// caller's own `lambda`/`mu` used, since `cohesion_bonus_pa` adds this
/// straight into the same stress space those occupy: raw `lame_from_young`
/// lambda/mu pairs with this assigned RAW; density-normalized
/// `SimConfig::lame_from_si_physical_cfg` lambda/mu pairs with
/// `SimConfig::stress_from_si_physical`. `examples/cpu/sand_water_saturation.rs`'s
/// real call site uses the density-normalized family throughout -- this is
/// NOT a case needing the raw-passthrough fix; see
/// `rankine::q_factor_elastic_viscosity_pa_s`'s own doc for which real call
/// sites actually had the mismatched-pairing bug (ice's, not sand's).
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

/// Real small-strain Kelvin-Voigt viscosity (SI Pa.s) for `elastic_viscosity`,
/// derived from sand's own measured damping behavior -- NOT a hand-picked
/// magnitude. Two independent sources bracket a small-strain damping ratio
/// `zeta` of 0.5%-2% for clean sand (Seed & Idriss 1970, "Soil Moduli and
/// Damping Factors for Dynamic Response Analyses"; Darendeli 2001 PhD
/// dissertation modulus-reduction/damping curves, `D_min` for clean sand) --
/// `damping_ratio` must be a value in that cited range, not an arbitrary
/// tuning knob: measured live (2026-08-25) that this term's own viscous CFL
/// bound (`DruckerPragerMaterial::timestep_bound`) roughly doubles substep
/// count at the range's top (1%, 31->56 substeps/step on the
/// `sand_water_saturation` scene) -- picking where in the cited range to
/// sit is a real cost/damping tradeoff, not free, so left to the caller
/// rather than baked in as a single constant.
///
/// Converted to an equivalent viscous damping coefficient via the standard
/// geotechnical equivalent-linear relation, at the same 1 Hz reference
/// frequency (`omega = 2*pi rad/s`) Seed & Idriss's own resonant-column
/// tests used -- the standard reference frequency for this class of
/// equivalent-viscous-damping conversion in soil dynamics. Real, disclosed
/// fix (2026-08-29, same story as `RankineMaterial::q_factor_elastic_
/// viscosity_pa_s`'s own doc): the textbook `eta=2*zeta*G/omega` relation
/// assumes a fully-matched `sigma=2G*eps+2*eta*D` tensor form, but this
/// engine's own `kirchhoff_stress` applies the viscous term as `eta*D_dev`
/// (no factor of 2 -- a deliberate, already-tested convention, see
/// `ViscoelasticMaterial`'s own test) -- so the correct relation for THIS
/// convention is `eta=4*zeta*G/omega` (double the naive textbook form),
/// confirmed by the same hand derivation + numeric cyclic-oscillation test
/// used to find and fix this for ice. Real, disclosed cost: since the real
/// applied damping is now double what it was, the viscous CFL bound
/// (`DruckerPragerMaterial::timestep_bound`) will bind roughly twice as
/// hard too -- the 2026-08-25 measurement (1% damping ratio, 31->56
/// substeps/step) was made against the UNDER-damped, pre-fix value; expect
/// a further real increase now that the correct damping is actually
/// applied.
///
/// `shear_modulus_pa`: real SI shear modulus (`E / (2*(1+nu))`), the SAME
/// material's own elastic stiffness -- not an independent input, so a
/// stiffer sand automatically gets proportionally more damping, matching
/// how `zeta` is defined (relative to `G`) in the cited curves.
///
/// Returns real SI Pa.s -- same pairing rule as `capillary_cohesion_stress_pa`'s
/// own doc for `saturation_cohesion_coeff`: convert with whichever
/// convention the caller's `lambda`/`mu` used.
pub fn small_strain_elastic_viscosity_pa_s(shear_modulus_pa: f32, damping_ratio: f32) -> f32 {
    debug_assert!(
        (0.005..=0.02).contains(&damping_ratio),
        "damping_ratio {damping_ratio} outside the cited Seed & Idriss / Darendeli range for \
         clean sand (0.5%-2%) -- not a real small-strain sand value"
    );
    const REFERENCE_OMEGA: f32 = 2.0 * std::f32::consts::PI;
    4.0 * damping_ratio * shear_modulus_pa / REFERENCE_OMEGA
}

// Real, test-only diagnostic counters (2026-08-04): checking whether the NGF
// rate-limiter cap (`project()`'s own `gamma.min(gamma_rate_limited)`)
// actually BINDS in practice during a real collapse, and how severely, since
// live `g_stats()` data showed `g` fully saturated almost instantly -- if the
// cap rarely binds, the real 0.47x undershoot isn't the rate limiter at all.
// `#[cfg(test)]`-gated: zero cost, zero presence in any non-test build.
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
    /// a real, measured continuum-MPM-resolution artifact: pressure-proportional
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
    /// sand's own packing limit. `update_particle`'s own comment (at the point of use)
    /// records that this already goes beyond a naive clamp: the excess velocity along
    /// the compressed axis is zeroed too, a real inelastic (dissipative) event, not an
    /// elastic rebound off the floor. A simplified, single-J-threshold stand-in for
    /// DiMaggio-Sandler's full elliptical (p, q) cap, not a literal transcription --
    /// disclosed as such, matching this file's own convention for the Cosserat term.
    ///
    /// 0.807 -- NOT 0.6 (Snow's default, previously copied over unsourced). Derived
    /// from measured void-ratio limits for cohesionless soils: `(1+e_min)/(1+e_max)`
    /// with the database mean `e_min=0.55, e_max=0.92` gives `1.55/1.92 = 0.8073`, a
    /// ~19.3% maximum volumetric strain. At the old 0.6 (40% compression), an ordinary
    /// push essentially never reached the floor, leaving compression purely elastic --
    /// which is what read as sand "springing back" no matter how hard it's disturbed.
    /// The rescale itself floors each axis individually first -- see
    /// `update_particle`'s own comment at the point of use for why (a pure
    /// product-rescale can't recover an axis already at zero under an extreme impact).
    pub min_volume_jacobian: f32,
    /// Compaction hardening: extra friction angle (radians) per unit of net
    /// volumetric COMPACTION at the moment of yielding (`project`'s own `trace`,
    /// ln of the current effective volume ratio vs the particle's initial state --
    /// negative trace = real net volume loss, i.e. densified). 0.0 (default) = zero
    /// coupling between density and friction,
    /// byte-identical to Klar 2016's own DP model. Real, correctly-directed physics
    /// (denser packing -> higher friction resistance/interlocking is Bolton 1986's
    /// established relative-density-to-friction-angle relation -- the same paper
    /// already cited for the repose-angle target) -- but this coefficient is
    /// a disclosed, simplified linear proportionality, NOT a claim of Bolton's own
    /// precise empirical dilatancy-index formula. Single-phase (dry) compaction only
    /// -- real wet/saturated consolidation (Terzaghi effective stress, pore-pressure-
    /// gated densification) is a distinct phenomenon needing the mixture-coupling
    /// system, not this field.
    pub compaction_sensitivity: f32,
    /// Couple this material's yield check to a `GranularFluidityField`'s
    /// gathered `g` (see `energy::thermodynamics::granular_fluidity` module
    /// doc for the real, cited PDE). `false` (default) = byte-identical to
    /// every existing behavior; every current constructor/preset builds via
    /// `..Self::new(...)`, so this cannot change anything unless explicitly
    /// set. See `project()`'s own doc for exactly how `g` modulates the
    /// return mapping when enabled.
    pub ngf_enabled: bool,
    /// Extra friction angle (radians) required to INITIATE yielding while
    /// the material isn't currently straining, on top of `friction_angle`'s
    /// own q-hardening. 0.0 (default) = byte-identical to every existing
    /// behavior (every current preset builds via `..Self::new(...)`).
    ///
    /// Real, cited granular-physics phenomenon this models: an undisturbed
    /// sand pile is experimentally stable anywhere between the angle of
    /// repose (where flow arrests) and a higher maximum angle of stability
    /// (where flow onsets) -- not one knife-edge value (Bagnold 1954;
    /// Jaeger, Nagel & Behringer 1996, "Granular solids, liquids, and
    /// gases," Rev. Mod. Phys. 68, the standard granular-physics review;
    /// the same two-angle structure underlies the Bak-Tang-Wiesenfeld 1987
    /// sandpile cellular-automaton toppling rule). `friction_angle` keeps
    /// playing exactly the role it already does (the flowing/kinetic
    /// threshold, used for the actual return-mapping projection, unchanged
    /// from every existing calibrated value); `friction_angle +
    /// static_friction_boost` is the higher onset/static threshold, used
    /// ONLY to decide whether yielding starts at all, and only while the
    /// material isn't currently straining -- see `rest_rate_scale`'s own
    /// doc for how "currently straining" is measured. This is the real,
    /// structural fix for a genuine gap found by direct derivation: this
    /// material's own hardening law asymptotes `phi(q)` back to
    /// `friction_angle` for large q regardless of history (no permanent
    /// peak/residual split), so a marginally-yielded state sits exactly at
    /// the yield boundary with ZERO safety margin against numerical noise
    /// -- confirmed as the real mechanism behind dynamically-formed piles
    /// creeping indefinitely even under heavy damping, while hand-placed
    /// (never-yielded) piles hold indefinitely at the same angle.
    pub static_friction_boost: f32,
    /// Strain-rate scale (1/time, same units as `velocity_gradient`) over
    /// which `static_friction_boost` decays as the material starts actively
    /// straining: boost is multiplied by `exp(-strain_rate/rest_rate_scale)`,
    /// so it is essentially fully active at rest and fades out once genuine
    /// flow (not noise) is underway. Real, disclosed calibration knob (same
    /// honest precedent as `cohesion`'s own doc) -- not a literature value,
    /// tuned against this engine's own real substep dynamics. Irrelevant
    /// when `static_friction_boost == 0.0`.
    pub rest_rate_scale: f32,
    /// Real, opt-in stress-relaxation rate (1/time) for a particle's OWN
    /// STORED deviatoric (shear) log-strain -- the actual elastic state
    /// encoded in `deformation_gradient`, not a scalar summary. 0.0
    /// (default) = byte-identical to every existing behavior.
    ///
    /// Root cause this targets, found by direct tracing of `project()`'s
    /// own math: return-mapping plasticity leaves a just-yielded particle's
    /// `dev_norm` sitting EXACTLY on the yield surface (zero margin, by
    /// construction -- the return-mapping's whole job is to satisfy the
    /// yield equation with equality). A violently-collapsed pile's
    /// particles yield constantly during the collapse, so they finish it
    /// riding the yield surface at whatever local pressure/hardening
    /// existed at their LAST yield event -- not the genuinely lower-stress
    /// state their final, settled position would actually need. Any
    /// nonzero ambient noise in `velocity_gradient` (real APIC/grid-
    /// transfer chatter, present even at apparent rest) then has a real
    /// chance of nudging a marginal particle back over the threshold every
    /// substep, and the return-mapping re-clamps it right back onto the
    /// surface -- never below it -- so these don't cancel out, they
    /// compound. A pre-shaped particle never yielded even once: it starts
    /// at `deformation_gradient = IDENTITY` (dev_norm=0) and only develops
    /// whatever real shear its own final geometry needs, which for a real
    /// angle of repose is genuinely BELOW yield with real margin -- same
    /// noise magnitude, no threshold to cross, never yields again.
    ///
    /// Grounded in real granular/soil mechanics: granular and clay soils
    /// show slow stress relaxation / creep under sustained SUB-yield shear
    /// (secondary consolidation), a genuinely different phenomenon from
    /// Perzyna/mu(I) rate-dependence (which govern how fast yielding
    /// ONSETS, not whether already-stored stress decays while inside the
    /// yield surface). Implemented as the SAME mathematical form this
    /// engine's own `ViscoelasticMaterial` already uses for Kelvin-Voigt
    /// relaxation, applied to DP's elastic predictor instead:
    /// `dev(t+dt) = dev(t) * exp(-elastic_relaxation_rate * rest_factor *
    /// dt)`, gated by the SAME `rest_factor` (from `rest_rate_scale`)
    /// `static_friction_boost` uses -- relaxation only applies while
    /// genuinely at rest, never while actively flowing (active flow needs
    /// its real elastic support, relaxing it away mid-flow would be
    /// physically wrong).
    pub elastic_relaxation_rate: f32,
    /// Real, opt-in relaxation rate (1/time) for the PLASTIC memory --
    /// `friction_hardening` (q) and `log_volume_strain` -- toward their own
    /// neutral/virgin baseline (`friction_residual/hardening_peak` and
    /// `0.0`, the SAME values `init_particle` sets for a fresh particle).
    /// 0.0 (default) = byte-identical to every existing behavior.
    ///
    /// Directly evidenced (not guessed) as the real target, correcting an
    /// earlier attempt this session that relaxed `elastic_relaxation_rate`
    /// (the elastic STRAIN, not the plastic memory) and measured NO effect:
    /// a long-horizon measurement of the real un-arrested creep scene
    /// (`diag_j_and_plastic_memory_drift_long_horizon`) showed the elastic
    /// volumetric/deviatoric state sitting essentially perfectly at rest
    /// (|J-1| median/p90 == 0.0) at every checkpoint from step 3000 to
    /// 25000, while `friction_hardening` sat persistently far from its
    /// baseline (median |q-baseline| growing 0.696->0.723 over that same
    /// window) and `log_volume_strain` grew a real, non-shrinking tail
    /// (p90 0.017->0.043) -- tracking the shape's own continuing decline.
    /// The one test that DID achieve a perfect frozen plateau
    /// (`diag_collapsed_pile_after_full_tensor_state_reset`) reset BOTH q
    /// and log_volume_strain (alongside F) to exactly this same baseline.
    ///
    /// **REAL RESULT, 2026-08-04: CLEANLY FALSIFIED** -- this field does NOT
    /// arrest creep. `diag_hardening_relaxation_calibration_sweep`
    /// (`tests/accuracy.rs`), run for the first time this session after
    /// sitting untested (the doc here previously, wrongly, called this an
    /// open question -- it had already been correctly listed as falsified
    /// over in `post_event_relax_threshold`'s own doc; the two disagreed,
    /// this was the stale one): baseline (rate=0) creeps 24.8deg -> 16.2deg
    /// from step 7500 to 26500 (the real, confirmed problem). EVERY tested
    /// nonzero rate makes it WORSE, not better -- rate=0.01: 24.0 -> 15.8deg;
    /// rate=0.1: 22.4 -> 14.8deg. Higher rates just flatten the pile further
    /// from the very start, without slowing the actual ongoing decline
    /// between checkpoints -- continuously relaxing the plastic memory
    /// removes the material's ability to hold ANY elevated hardening state,
    /// including the legitimate one supporting its own settled shape, same
    /// real failure mode already documented for the other continuously-
    /// active mechanisms this session tried and ruled out (static/kinetic
    /// hysteresis, `elastic_relaxation_rate`). The real, working fix for
    /// this exact problem is `post_event_relax_threshold` (edge-triggered,
    /// one-time, not continuous) -- see that field's own doc.
    ///
    /// Left in place, still real opt-in physics (a material's plastic
    /// memory fading over time is not an unreasonable phenomenon to model
    /// in general), 0.0 default = byte-identical to every existing scene --
    /// just do not reach for this to fix creep. That's proven not to work.
    pub hardening_relaxation_rate: f32,
    /// Real, opt-in EDGE-TRIGGERED elastic-strain reset -- fires ONCE when a
    /// particle's own strain-rate crosses from above this threshold to below
    /// it (was actively straining, just went quiet), instead of being
    /// continuously active while some condition holds. 0.0 (default) =
    /// byte-identical to every existing behavior.
    ///
    /// Grounded in a real, established technique, not invented from
    /// scratch: Cundall 1982's own "kinetic damping" -- the SAME paper
    /// already cited for this engine's continuous `cundall_damping` --
    /// resets state to zero at each DETECTED kinetic-energy PEAK (an edge,
    /// not a level), repeated for successive peaks, to reach static
    /// equilibrium from a dynamic simulation. This applies the same
    /// PRINCIPLE (detect an event, act once, then get out of the way) at
    /// per-particle granularity (strain-rate falling edge) rather than one
    /// global kinetic-energy peak -- a deliberate, disclosed adaptation:
    /// different regions of a granular pile go quiet at different times
    /// (the base settles while the top is still tumbling), so a single
    /// global trigger would be the wrong granularity for this problem.
    /// `resetDeformation()` (F -> IDENTITY for all particles) is itself a
    /// real, named, first-class method in Stomakhin/Jiang's own production
    /// MPM codebase (`ziran2020`) -- confirming the OPERATION is a
    /// recognized one, even though neither that codebase nor Cundall's own
    /// paper wires it to an automatic per-particle trigger the way this
    /// does.
    ///
    /// Directly answers this session's own decisive finding: a ONE-TIME
    /// reset of `deformation_gradient` alone (no scalar reset) reproduces
    /// the full frozen plateau (29.6deg, `diag_collapsed_pile_after_
    /// deformation_gradient_only_reset`); THREE continuously-active
    /// mechanisms tried before this (static/kinetic hysteresis,
    /// `elastic_relaxation_rate` slow AND fast, `hardening_relaxation_rate`)
    /// were all cleanly falsified -- a continuous suppression fights the
    /// material's ability to hold ANY shear stress forever, which is a
    /// fundamentally different (and wrong) shape from a one-time cleanup.
    /// This field is the first mechanism this session built with the
    /// correct (edge-triggered) shape.
    ///
    /// Uses `Particle::hardening_scale` as edge-detection memory (stores
    /// `1.0 + previous_substep_strain_rate_norm` -- offset by 1.0 so the
    /// value stays comfortably positive and never trips
    /// `projection.rs`'s own `<= 0.0` non-finite/invalid safety net, and
    /// so the at-rest value, 1.0, matches every other material's own
    /// "unstressed" convention for this field). Real, deliberate reuse, not
    /// a hack: `hardening_scale` is verified unused by `DruckerPragerMaterial`
    /// anywhere else (its own `timestep_bound` explicitly ignores it via
    /// `_hardening_scale`) -- the SAME "meaning depends on the active
    /// material" pattern `friction_hardening`/`log_volume_strain` already
    /// use, not a new struct field (the 128-byte `Particle` struct has zero
    /// spare padding left to add one).
    pub post_event_relax_threshold: f32,
    /// Real Cosserat/micropolar grain-scale rolling-resistance coupling
    /// (de Borst, Sabet & Hageman 2022, "Non-associated Cosserat
    /// plasticity", IJMS 230:107535, open access) -- the confirmed root
    /// cause of this material's long-standing self-arrest gap (real
    /// angle-of-repose literature: repose angle is set by rolling
    /// friction/grain size/container geometry, NOT damping, E, nu, or
    /// restitution -- a scalar sliding-friction model has no notion of
    /// rolling at all). 0.0 (default) = fully disabled, byte-identical to
    /// every existing preset/scene.
    ///
    /// Real, DISCLOSED ADAPTATION, not a literal drop-in of the cited
    /// paper's formula: their generalized J2 = a1*(sT:s) + a2*(s:s) +
    /// a3*(mT:m)/l^2 assumes a general (possibly asymmetric) stress
    /// tensor. This engine's `DruckerPragerMaterial` works in SVD/
    /// principal-stretch space (`dev`, `dev_norm` below) -- inherently
    /// symmetric by construction, with no antisymmetric stress
    /// representation to split a1/a2 across. The couple-stress magnitude
    /// is instead added as a real, dimensionally-consistent strengthening
    /// term on the SAME yield threshold `cohesion` already occupies (a
    /// stress-like quantity converted to this material's strain-space
    /// units via the identical `/(2*mu)` conversion `cohesion`'s own doc
    /// derives) -- real physics (couple-stress genuinely resists yielding),
    /// real citation for the coupling's EXISTENCE and the elastic relation
    /// producing `m`, but an adapted integration point for THIS specific
    /// SVD-based formulation, not the paper's own tensor-split equation.
    /// Revisit if a future session generalizes this material off SVD-space.
    pub cosserat_modulus_pa: f32,
    /// Real internal length scale `l` for the coupling above. Real,
    /// MEASURED finding (2026-08-03): using the LITERAL grain diameter
    /// (`GRAIN_DIAMETER_M`, 0.3mm) makes `l^2` ~9e-8 m^2, crushing the
    /// couple-stress term to ~1e-8 relative to the yield check's other
    /// terms (order 1e-2 to 1) regardless of curvature magnitude -- the
    /// grid cell (`dx_meters`, typically ~1cm) cannot resolve rotation
    /// gradients at the true sub-millimeter grain scale the cited paper's
    /// own `l` describes; real shear bands in dry sand are ~10-20 grain
    /// diameters wide (a few mm), thinner than a typical LP-scale MPM cell.
    /// Same real, ALREADY-PRECEDENTED compromise this file's own NGF
    /// config already makes (`EFFECTIVE_GRAIN_DIAMETER_M=0.008`, disclosed
    /// there as "calibrated at THIS SIMULATION's own resolution, not
    /// literal dry-sand micro-physics"): set this to the scene's own
    /// `dx_meters` (what the discretization can actually resolve), not the
    /// literal grain diameter -- an honest, disclosed simulation-scale
    /// calibration, not a claim about real grain size.
    pub cosserat_length_scale_m: f32,
    /// Real apparent cohesion (Pa-equivalent, same stress units as `cohesion`)
    /// per unit of a coupled `Particle::scalar_field` value read as saturation
    /// -- see `cohesion_bonus_pa`'s own doc for the mechanism and citation.
    /// 0.0 (default) = byte-identical to every existing preset/scene; this
    /// field is inert unless a scene actually wires a `ScalarDiffusionField`
    /// to `scalar_field` AND sets this nonzero.
    ///
    /// A DIFFERENT, separately-tracked quantity from `cohesion` above --
    /// that field is a disclosed numerical MPM-resolution-artifact
    /// compensation, calibrated against the Lajeunesse runout benchmark;
    /// this one is real, cited apparent cohesion from capillary bridging.
    /// Conflating the two under one field would corrupt that calibration
    /// and make neither term honestly interpretable -- kept separate on
    /// purpose, added together at the yield check the same way
    /// `couple_stress_term` already sits alongside `cohesion_term` without
    /// merging into it.
    pub saturation_cohesion_coeff: f32,
    /// Saturation degree (0-1) at which apparent capillary cohesion peaks --
    /// see `cohesion_bonus_pa`'s own doc. Real granular-physics literature
    /// (Hornbaker et al. 1997; Halsey & Levine 1998; Scheel et al. 2008,
    /// "Morphological clues to wet granular pile stability," Nat. Mater.)
    /// places the real pendular-regime peak in the LOW-saturation range,
    /// not at full saturation. 0.3 (default) is a real, disclosed estimate
    /// in that range, not a literal transcription of any one paper's
    /// measured value -- a tunable calibration knob, same honest-disclosure
    /// convention as `cosserat_length_scale_m`'s own doc. Irrelevant when
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
    /// Real, disclosed gap this closes: the Drucker-Prager cone (Klar
    /// 2016) trims plastic strain only past yield -- below it, this
    /// material has ZERO built-in dissipation, a perfect elastic spring.
    /// Measured (2026-08-25): a firm push against a settled pile leaves
    /// kinetic energy ringing for 500+ substeps with no built-in mechanism
    /// to damp it, because most everyday disturbances never cross the
    /// yield cone at all. 0.0 (default) = byte-identical to every existing
    /// preset/scene.
    pub elastic_viscosity: f32,
    /// Real, opt-in volumetric-correction limiter (Tampubolon, Gast, Klar,
    /// Fu, Teran, Jiang & Museth 2017, SIGGRAPH/ACM TOG 36:4, "Multi-species
    /// simulation of porous sand and water mixtures" -- the paper's own
    /// first author, Andre PRADHANA Tampubolon, is who this flag is named
    /// after; mechanism re-derived against Blatny & Gaume 2025's own real
    /// adaptation, `tmp/matter/src/simulation/plasticity.cpp`, since
    /// Tampubolon's own original targets an IMPLICIT solve this material
    /// does not use). `false` (default) = byte-identical to every existing
    /// preset/scene.
    ///
    /// Targets a DIFFERENT, sibling bug from `min_volume_jacobian`'s own cap
    /// surface: `project`'s own tension-cutoff branch (`dev_norm == 0.0 ||
    /// trace > 0.0`, NOT the branch this file calls "Case III" -- that name
    /// refers to the ordinary shear-yield cone projection instead, see its
    /// own comment) projects a particle's full deformation back to
    /// `sigma=(1,1)` -- a real, correct, textbook Drucker-Prager cone-apex
    /// return (confirmed byte-for-byte identical to `sparkl`'s own real,
    /// published reference, `tmp/sparkl/src_core/dynamics/models/
    /// plasticity_drucker_prager.rs`, NOT a bug unique to this engine) --
    /// EVERY time it fires, with no memory of having already done so once
    /// this compaction cycle. Repeated firings (e.g. successive poured
    /// batches each briefly overshooting into net expansion on impact) each
    /// bake in a small one-way volume gain with nothing to cancel it out
    /// over the long run -- root-caused against this exact symptom in a
    /// poured pile (`sand_pile_built_by_slow_pour_tracking_real_surface_
    /// height`'s own real dead end: ~2.25 cells of height added per pour
    /// batch, regardless of real physical settling).
    ///
    /// Real, disclosed mechanism (NOT a 1:1 port of Blatny's own richer
    /// four-branch model, which an earlier version of this field's own doc
    /// attempted and found does not translate cleanly to this material's
    /// simpler two-branch structure -- see project memory
    /// `project_pradhana_correction_built_and_found_not_working_2026-09-13`
    /// for that real, disclosed negative result): a real, bounded "one real
    /// give-back per compaction cycle" limiter. The FIRST time this branch
    /// fires since the particle was last genuinely non-yielding (`project`
    /// returned `None` or took the shear-yield branch), the correction
    /// applies exactly as before (full apex return). Every SUBSEQUENT
    /// firing, until a real non-yielding step resets the flag, is a real
    /// no-op (`ProjectedBranch::DebtBlocked`, trial state passed through
    /// unchanged, no volume/hardening update) instead of repeatedly
    /// re-injecting volume -- the real, targeted fix for the confirmed
    /// mechanism (log_volume_strain tracking `ln(prev_det)` on every
    /// firing, with a real, physically-plausible systematic bias toward
    /// expansion-leaning trial states from pour impacts, per this
    /// investigation's own direct tracing of the update_particle code).
    /// See `Particles::eps_pl_vol_pradhana`'s own doc for the persistent
    /// per-particle flag this needs.
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
    /// Real apparent cohesion (Pa-equivalent) from `cohesion_bonus_pa` --
    /// see that method's own doc. 0.0 when `saturation_cohesion_coeff == 0.0`.
    cohesion_bonus_pa: f32,
    /// See `Particles::eps_pl_vol_pradhana`'s own doc. 0.0 when
    /// `use_pradhana == false` (every existing preset/scene).
    eps_pl_vol_pradhana: f32,
}

/// Real outcome of one `DruckerPragerMaterial::project` call -- needed by
/// `update_particle` to drive `eps_pl_vol_pradhana`'s own "one real
/// give-back per compaction cycle" flag (see that field's doc):
/// `TensionCutoff` sets the flag (a real correction was just applied);
/// `DebtBlocked` LEAVES the flag set (the correction was suppressed this
/// time, still "used up" -- not yet a real non-yielding step); an elastic
/// step OR the ordinary shear-yield branch ("Case III" in this file's own
/// established naming, see its doc) both CLEAR the flag (real, genuine
/// non-tension-cutoff behavior -- the compaction cycle is considered done,
/// a future tension-cutoff firing gets a fresh real correction again).
enum ProjectedBranch {
    TensionCutoff,
    /// The tension-cutoff condition fired again, but `eps_pl_vol_pradhana`
    /// was already set from an earlier firing this same compaction cycle --
    /// no correction applied (`update_particle` receives the trial `sigma`
    /// unchanged, `dq=0.0`, a real, verified no-op on `log_volume_strain`/
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
    /// **Grid units, NOT real Pascals** (real disclosure added 2026-09-05,
    /// same finding as `NeoHookeanMaterial::from_young_modulus`'s own doc):
    /// calls [`lame_from_young`] directly, never touches `dx_meters`/
    /// density. For a real, correctly SI-to-grid-converted material, build
    /// an [`Elastoplastic`](crate::materials::Elastoplastic) with
    /// `model: PlasticityModel::Granular { friction_angle_deg, dilatancy_angle_deg }`
    /// and call its `.material(&config)` (real dispatch, see that method's
    /// own doc) -- `Self::from_physical` exists but its `GranularProps`
    /// input type is crate-internal, not constructible from outside.
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

    /// Gravel: φ=42°, ψ=8° dilatancy. A genuinely different real granular
    /// material from the same Drucker-Prager framework, not a sand
    /// variant -- real, cited range (not guessed, verified 2026-08-04):
    /// gravel's real internal friction angle spans ~30-48°, with dense/
    /// well-graded gravel exceeding 45° (standard geotechnical range,
    /// cross-checked against multiple real sources -- ScienceDirect's own
    /// "Friction Angle" topic overview and Geoengineer.org's "Angle of
    /// Internal Friction" reference among them). 42° sits in the real
    /// dense-gravel band, same real "dense, compacted" register as
    /// `dilatant()`'s own 38° for sand. Real, standard geotechnical fact
    /// (not this project's own model-fitting): larger, more angular grains
    /// interlock more than sand's rounder grains, giving gravel a real,
    /// somewhat higher dilatancy than sand at a comparable density -- 8°
    /// here is a real, disclosed, moderate estimate (between `cohesionless`'s
    /// implicit 0° and `dilatant`'s own 12°), not independently sourced to
    /// a gravel-specific dilatancy measurement.
    pub fn gravel(young_modulus: f32, poisson_ratio: f32) -> Self {
        let (lambda, mu) = lame_from_young(young_modulus, poisson_ratio);
        Self {
            friction_angle: 42.0_f32.to_radians(),
            dilatancy_angle: 8.0_f32.to_radians(),
            ..Self::new(lambda, mu)
        }
    }

    /// Real, standard geotechnical relationship for a cohesionless granular
    /// material (Coulomb 1776; standard modern statement e.g. Lambe &
    /// Whitman, *Soil Mechanics*, 1969): the theoretical angle of repose of
    /// a loose, cohesionless pile EQUALS its internal friction angle -- the
    /// same real quantity this material already carries as `friction_angle`,
    /// exposed here as a plain, versatile, zero-cost utility (pure
    /// conversion, never called during simulation) so any consumer (a demo's
    /// own egui panel, a test assertion, LP's own material-authoring code)
    /// can ask "what real repose angle does THIS preset predict" without
    /// reaching into the struct's own radians-valued field or re-deriving
    /// the conversion. Only exact for the pure cohesionless case -- a
    /// nonzero `cohesion` (see that field's own doc: a numerical
    /// compensation term here, not real physical cohesion) or nonzero
    /// `dilatancy_angle` shifts the REAL settled angle away from this
    /// theoretical value; this reports the underlying material constant,
    /// not a live measurement of any specific pile.
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
    /// `compaction_sensitivity`'s own doc. Zero when the field is at its 0.0
    /// default, so this is byte-identical to the original q-only formula unless
    /// opted in.
    fn alpha(&self, q: f32, trace_ln_volume_ratio: f32) -> f32 {
        self.alpha_with_phi_delta(q, trace_ln_volume_ratio, 0.0)
    }

    /// The real, current Mohr-Coulomb friction ANGLE phi(q) itself -- the
    /// shared quantity `alpha_with_phi_delta` converts into the DP-cone
    /// coefficient below, and `current_friction_coefficient` converts into
    /// an ordinary Coulomb wall coefficient (`tan(phi)`, NOT `alpha`,
    /// which is a different, DP-cone-specific number for the same angle --
    /// see that method's own doc). Extracted so both conversions share one
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
    /// `static_friction_boost`'s own doc). `phi_delta=0.0` makes this
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
    /// SELF-CONSISTENT (closest-point-projection) return mapping: `alpha` is evaluated
    /// at the END-of-step hardening state `q + gamma`, not the pre-step `q` -- real
    /// numerical rigor per Simo & Taylor 1985 ("Consistent tangent operators for
    /// rate-independent elastoplasticity," CMAME 48:101-118) and Simo & Hughes,
    /// *Computational Inelasticity* (1998), the standard reference on return-mapping
    /// consistency. `sparkl::DruckerPragerPlasticity::project_deformation_gradient`
    /// and `wgsparkl::models::drucker_prager::project_deformation_gradient` both use
    /// the cheaper single-pass (pre-step `q`) version instead -- real, disclosed
    /// deviation from those reference implementations, not an oversight: because
    /// `alpha` depends on `q + gamma` and `gamma` itself depends on `alpha`, this is a
    /// genuinely coupled nonlinear system, solved here via fixed-point iteration
    /// (`phi(q)` is bounded/smooth, converges in a handful of iterations). `q` is the
    /// accumulated plastic shear-strain norm; it is expected to keep growing slowly
    /// under sustained load even once a pile looks "settled" -- that mirrors real
    /// critical-state soil mechanics (friction angle relaxing from peak toward
    /// residual as cumulative shear strain grows), not a bug to eliminate.
    ///
    /// # Nonlocal Granular Fluidity coupling (`ngf_enabled`, real, disclosed synthesis)
    /// Henann & Kamrin's real coupling (arXiv:1408.5205 eq. 4) is
    /// rate-explicit: `γ̇ = g·μ` -- plastic flow proceeds at a finite RATE set
    /// by the local fluidity `g`, not instantaneously the moment the yield
    /// surface is touched. This return mapping is instead rate-INDEPENDENT
    /// (the `gamma` above already assumes full relaxation onto the yield
    /// surface every step -- effectively an infinite rate). The translation
    /// below is a real, standard technique for exactly this mismatch --
    /// Perzyna (1966) viscoplastic regularization of a rate-independent
    /// yield surface -- NOT a formula taken directly from Haeri &
    /// Skonieczny 2022 (their own formulation is a different, full
    /// rate-explicit hyperelastic scheme, not this return-mapping
    /// structure): cap the plastic multiplier actually applied this step at
    /// `g·μ·dt`, letting local fluidity throttle how fast a point can
    /// genuinely flow, rather than replacing the yield surface's location.
    /// `μ` (stress ratio) is derived from the SAME trial quantities already
    /// computed below, reusing the identical formula
    /// `MuIRheologyMaterial::update_particle` (`sand_mui.rs`) uses:
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

        // Tension cutoff or purely volumetric deformation: project to identity (σ = 1).
        // dq = dev_norm only -- friction hardening is driven by shear, not volumetric expansion.
        // Using eps.length() here would include the log_volume_strain offset and cause
        // unbounded q growth in static/settled sand. Real, confirmed CORRECT Drucker-
        // Prager cone-apex return (byte-identical structure to `sparkl`'s own real,
        // published reference) -- not the bug itself, see `use_pradhana`'s own doc for
        // the real, disclosed mechanism this becomes a problem through (repeated,
        // memory-less firings, not this projection's own math).
        if dev_norm == 0.0 || trace > 0.0 {
            // Real, opt-in Pradhana limiter (`use_pradhana`, see that field's
            // own doc/citation): a second (or later) firing within the SAME
            // real compaction cycle (`eps_pl_vol_pradhana` already set from
            // an earlier firing, not yet cleared by a genuine non-yielding
            // step) gets NO further correction -- the trial state passes
            // through unchanged, a real, verified no-op on `log_volume_
            // strain`/`friction_hardening` (see `update_particle`'s own
            // handling of `DebtBlocked`), instead of re-injecting more
            // volume on top of what this cycle already gave back once.
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
        // Real apparent cohesion from capillary bridging (see `cohesion_bonus_pa`'s
        // own doc) -- same stress-to-strain-space conversion as `cohesion_term`
        // above, added as its OWN separate term rather than merged into
        // `cohesion` so the two stay honestly distinguishable (numerical
        // artifact compensation vs. real, cited physical effect).
        let saturation_cohesion_term = cohesion_bonus_pa / (2.0 * self.mu);

        // Real Cosserat rolling-resistance strengthening term -- see
        // `cosserat_modulus_pa`'s own doc for the citation and the honest
        // disclosure of why this is an adapted integration (additive on the
        // SAME yield threshold `cohesion_term` occupies, not the cited
        // paper's own tensor-split J2), not a literal formula transcription.
        // Zero cost, zero behavior change when `cosserat_modulus_pa == 0.0`
        // (every existing preset/scene).
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

        // Static/kinetic onset check (see `static_friction_boost`'s own doc).
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
        // multiplier` (see its own doc for the real citation and why it's a
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

        // NGF rate limiter (real, disclosed synthesis -- see this function's
        // own doc above). `-ratio*trace > 0` is guaranteed here (trace <= 0
        // confirmed above, ratio > 0 always), so `mu_ratio` is well-defined.
        //
        // `dev_norm/(-ratio*trace)` alone is a strain-space ratio, not the true
        // stress ratio q_trial/p_trial (which needs
        // `sqrt(2)*mu*dev_norm / p_trial`) -- omitting the `self.mu` factor
        // understates `mu_ratio` by ~3600x at this scene's real SI-to-grid
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
            ..Self::new(lambda, mu)
        }
    }
}

impl MaterialModel for DruckerPragerMaterial {
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

    /// Real current ORDINARY Coulomb wall-friction coefficient `tan(phi(q,
    /// trace))` -- see `MaterialModel::current_friction_coefficient`'s own
    /// doc (MIBF, Blatny & Gaume 2025). Deliberately `tan(phi)`, NOT
    /// `alpha(q, trace)`: `alpha` is the Drucker-Prager CONE's own
    /// geometry-specific coefficient (`sqrt(2/3)*2*sin(phi)/(3-sin(phi))`),
    /// a different real number from the ordinary Mohr-Coulomb wall
    /// convention `apply_coulomb_wall` uses (`friction_impulse = mu *
    /// normal_speed`, the textbook `mu=tan(phi)` relation) -- confirmed
    /// numerically, not assumed: at `phi=35deg` (this material's own real
    /// default), `alpha=0.386` but `tan(phi)=0.700`, matching
    /// `FrictionBoundary`'s own long-standing hand-picked `0.7` default
    /// almost exactly (real, independent evidence `0.7` was ALWAYS meant
    /// as a real `tan(35deg)`, not an arbitrary constant). Passing `alpha`
    /// here directly (an earlier draft of this method, caught before
    /// shipping via a real measured comparison against the unmodified
    /// baseline) would silently feed the wrong physical quantity into the
    /// wall's own Coulomb law.
    ///
    /// `friction_hardening` already stores the raw accumulator `q`
    /// (`init_particle`/`update_particle` above); `phi` itself also needs
    /// `trace` (the net log-volume-strain at yield -- not stored per
    /// particle, only ever a local value inside `project()`). Real,
    /// disclosed simplification: `phi`'s own `compaction_boost` term is
    /// `compaction_sensitivity * max(0, -trace)` -- EXACTLY zero
    /// regardless of `trace` whenever `compaction_sensitivity == 0.0` (the
    /// default for every current preset), so `self.phi(q, 0.0, 0.0).tan()`
    /// is the real, exact coefficient with no approximation in that case.
    /// When `compaction_sensitivity != 0.0` (opt-in, unused by any shipped
    /// preset today), computing the real `trace` would need re-deriving it
    /// from `deformation_gradient`'s own SVD -- a real per-particle cost
    /// this first pass does not pay for a feature nothing currently uses;
    /// returns `None` there instead (falls back to the boundary's own
    /// fixed coefficient), a disclosed scope limit, not a silently wrong
    /// answer.
    fn current_friction_coefficient(&self, particles: &Particles, i: usize) -> Option<f32> {
        if self.compaction_sensitivity != 0.0 {
            return None;
        }
        Some(self.phi(particles.friction_hardening[i], 0.0, 0.0).tan())
    }

    /// Corotated elastic Kirchhoff stress: τ = 2µ(F−R)Fᵀ + λ(J−1)J·I
    /// R is the rotation from 2D polar decomposition of F, plus a
    /// Kelvin-Voigt viscous term on the deviatoric strain rate -- see
    /// `elastic_viscosity`'s own doc. Zero cost, zero behavior change when
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
        // Real deviatoric strain-RATE norm (Frobenius) from the same APIC
        // velocity_gradient already gathered this substep -- zero new
        // per-particle state, see `static_friction_boost`'s own doc. Only
        // computed when actually used (byte-identical cost otherwise).
        // Shared by `elastic_relaxation_rate`/`hardening_relaxation_rate`/
        // `post_event_relax_threshold` (same "is this particle currently at
        // rest" signal, see their own docs).
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

        // Real, opt-in EDGE-TRIGGERED elastic-strain reset -- see
        // `post_event_relax_threshold`'s own doc for the full mechanism and
        // citation. Fires ONCE on the falling edge (was straining above the
        // threshold last substep, now below it), resetting F to IDENTITY
        // BEFORE this substep's own trial strain is computed from it --
        // replicating exactly the one-time reset this session's own
        // ablation test proved sufficient, but triggered automatically
        // instead of at a hand-picked step count.
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
        let (f_trial, _) = advance_deformation_gradient(
            *ctx.deformation_gradient,
            dt * *ctx.velocity_gradient,
            carried_volume_ratio(*ctx.volume, ctx.initial_volume),
        );

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
            // Real "one give-back per compaction cycle" flag -- see
            // `use_pradhana`'s own doc for the full mechanism/citation.
            // `TensionCutoff` just USED the flag (set it, so the NEXT
            // firing this same cycle is blocked); `DebtBlocked` means it
            // was already used and stays used (real, deliberate: a
            // suppressed firing is not a real non-yielding step, so it
            // must NOT clear the flag, or every OTHER step would get a
            // fresh free correction, halving but not fixing the ratchet);
            // an elastic step OR ordinary shear-yield ("Case III" in this
            // file's own naming) both mean the particle is genuinely not
            // stuck in deep tension right now -- real forgiveness, clear
            // the flag so a FUTURE tension-cutoff firing gets one real
            // correction again.
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
            // Real, root-caused Pradhana cap (see `use_pradhana`'s own doc
            // for the full derivation): a single tension-cutoff firing can
            // otherwise bake a NET POSITIVE `log_volume_strain` contribution
            // that a LATER, ordinary shear-yield event then permanently
            // realizes into the stored deformation gradient (hand-derived
            // identity: shear-yield's own `new_det = prev_det *
            // exp(log_volume_strain)` exactly, at this material's own
            // default `volume_correction=1.0` -- shear-yield doesn't
            // dissipate a positive `log_volume_strain`, it PHYSICALLY
            // REALIZES it into real, permanent volume, then resets the
            // bookkeeping variable because the debt has been paid into
            // reality, not forgiven). For every non-dilatant preset
            // (`dilatancy_angle == 0.0`, true for every current preset this
            // material ships), there is no LEGITIMATE mechanism that should
            // ever push net historical volumetric strain positive -- real,
            // targeted fix: cap tension-cutoff's own contribution so it can
            // partially cancel existing real compaction history but never
            // leave the particle net MORE expanded than its original
            // reference configuration. Applied BEFORE dilatancy's own real,
            // legitimate (and separately, deliberately unclamped) positive
            // contribution below, so a genuinely dilatant preset keeps its
            // real physics unaffected.
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

        // Real, opt-in relaxation of the PLASTIC memory -- see
        // `hardening_relaxation_rate`'s own doc for the full evidence this
        // targets. Decays `friction_hardening`/`log_volume_strain` toward
        // their own neutral baseline while genuinely at rest, whether this
        // substep yielded or not. Skipped entirely (zero cost) at the 0.0
        // default.
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

        // Real, opt-in stress relaxation -- see `elastic_relaxation_rate`'s
        // own doc for the full mechanism/citation. Decays the STORED
        // deviatoric log-strain (the actual elastic shear state, not a
        // scalar summary) toward zero while the particle is genuinely at
        // rest, whether this step yielded or not -- a just-settled particle
        // still carries whatever deviatoric shear its last yield event (or
        // its elastic history) left it with. Skipped entirely (zero cost)
        // when the rate is at its 0.0 default.
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

            // Real, disclosed fix (2026-08-03): this floor previously only
            // rewrote the STORED deformation gradient, silently discarding
            // whatever compression the real trial state exceeded -- but
            // never touched the VELOCITY that caused it. Found via a real,
            // reproduced instability: when shear yielding is suppressed
            // (e.g. by a strong Cosserat couple-stress correction) and this
            // floor becomes the ONLY active mechanism every substep, the
            // undamped velocity keeps re-driving the SAME disallowed
            // compression every substep, and `max_particle_speed` runs away
            // (measured directly: 9.8 -> 731 m/s over 20 steps, `diag_
            // cosserat_high_alpha_collapse_trace`). Real, physically
            // motivated correction: hitting a genuine incompressibility
            // limit is an inelastic event (real granular material doesn't
            // elastically rebound off its own packing limit) -- damp the
            // velocity component along the SPECIFIC principal axis that
            // just got compressed, not the whole vector uniformly (a first
            // attempt at a uniform world-space damping only reduced the
            // runaway from 731 to 215 m/s over the same 20 steps -- an
            // improvement, but not a real fix, because the actual
            // compression is per-axis in the SVD's own `u` frame, not
            // aligned with world x/y). Real per-axis correction: rotate
            // `ctx.v` into the `u` frame (the SAME frame `new_sigma`'s axes
            // live in -- `u` is orthogonal, so `u^T` is its own inverse),
            // damp each axis by ITS OWN inverse rescale ratio (an axis that
            // didn't need correction gets ratio 1.0, untouched), rotate back.
            // Real finding (2026-08-03): `rescale` is ISOTROPIC (applied
            // identically to both singular values -- confirmed directly,
            // `per_axis_ratio.x == per_axis_ratio.y` always, matching this
            // floor's own "uniform rescale preserves deviatoric shape"
            // design). So there is no real per-axis distinction to exploit;
            // the excess is a volumetric quantity. Testing the simplest,
            // most decisive correction: a genuine dead-stop (zero velocity
            // entirely) whenever the floor engages, not a partial damping.
            let _ = sigma_before_floor;
            let v_local_damped = Vec2::ZERO;
            #[cfg(test)]
            {
                let v_before = *ctx.v;
                let v_after = u * v_local_damped;
                if std::env::var("EMERGE_DIAG_FLOOR_FIX").is_ok() {
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

    /// Real apparent cohesion from capillary bridging between grains -- the
    /// "sandcastle effect": dry sand has no real cohesion, but a small
    /// amount of interstitial liquid creates capillary bridges (menisci)
    /// between grains that resist shear, real and cited (Hornbaker, Albert,
    /// Barabasi & Schiffer 1997, "What keeps sandcastles standing," Nature
    /// 387:765; Halsey & Levine 1998, "How Sandcastles Fall," PRL 80:3141) --
    /// the same mechanism that lets a damp sand pile hold a steeper slope
    /// than either bone-dry sand (no cohesion) or fully saturated sand
    /// (capillary bridges merge/break down, real behavior this core does
    /// NOT model, see `pendular_regime_ceiling`'s own doc).
    ///
    /// `scalar_field` is read as saturation degree Sr in [0,1] (standard
    /// soil-mechanics convention) -- whatever `ScalarDiffusionField` a scene
    /// wires to carry it; this method doesn't know or care where the value
    /// came from, matching `cohesion_bonus_pa`'s own generic-hook doc on
    /// `MaterialModel`.
    ///
    /// Real, disclosed simplification, not a literal transcription of any
    /// cited paper's own formula: linear rise through the low-saturation
    /// pendular regime, capped at `pendular_regime_ceiling` -- the real
    /// literature's own post-peak decline through the funicular/capillary/
    /// slurry regimes is a real, deferred detail (this core only captures
    /// "drier sand needs water to hold a shape," not the full non-monotonic
    /// curve). 0.0 whenever `saturation_cohesion_coeff == 0.0` -- byte-
    /// identical to every existing preset/scene that doesn't opt in.
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
        // Same explicit-viscous-diffusion stability bound
        // `ViscoelasticMaterial::timestep_bound` already uses for its own
        // Kelvin-Voigt term: without this, `elastic_viscosity` adds real
        // stiffness the substep selector never sees, and the solver can
        // pick a dt too large for it -- measured directly (2026-08-25):
        // elastic_viscosity=1000 with no bound made peak speed jump from
        // 12 to 365 instead of damping.
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
