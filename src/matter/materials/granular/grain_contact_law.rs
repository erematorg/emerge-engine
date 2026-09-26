//! Real, cited discrete-element contact force law for genuine grain-scale
//! rolling resistance -- the piece every rate-dependent mechanism already
//! tried for sand's repose-angle problem (Cundall damping, KE-peak
//! switches, Cosserat curvature coupling) structurally cannot provide,
//! because they all fade to zero at rest. This can: it is elastic-plastic
//! with memory (accumulated spring displacement, not instantaneous rate),
//! the same "trial-elastic + yield-check + return-mapping" pattern already
//! used throughout this codebase for material plasticity (`DruckerPragerMaterial`,
//! `VonMisesMaterial`, `RankineMaterial`) -- just applied to a contact pair
//! instead of a stress tensor.
//!
//! Real sources, cross-checked against a real shipped implementation
//! (`tmp/GeoTaichi/src/dem/contact/LinearRolling.py`, itself citing Luding
//! 2008) before writing any code here, not guessed:
//! - Cundall & Strack 1979, "A discrete numerical model for granular
//!   assemblies," Geotechnique 29(1):47-65 -- the foundational linear
//!   spring-dashpot normal + Coulomb-capped tangential spring model.
//! - Luding 2008, "Introduction to discrete element methods," European
//!   Journal of Environmental and Civil Engineering 12:7-8, 785-826 -- the
//!   standard real critical-timestep (Rayleigh-type) stability bound.
//! - Ai, Chen, Rotter & Ooi 2011, "Assessment of rolling resistance models
//!   in discrete element simulations," Powder Technology 206(3):269-282 --
//!   surveyed real rolling-resistance formulations against measured repose
//!   angles; the elastic-plastic spring-dashpot (EPSD) rolling model (real
//!   memory via an accumulated relative-rotation spring, Coulomb-like yield
//!   cap) is their recommended class for holding a genuine STATIC angle, as
//!   opposed to viscous/"directional constant" rolling models (which share
//!   Cosserat's own rate-only failure mode, already ruled out this session).
//!
//! Real fix, 2026-08-03: the tangential spring is tracked as a real 2D
//! vector, re-projected onto the CURRENT tangent plane every step (removing
//! any component that has drifted onto the normal axis as the contact
//! normal itself rotates between steps) -- standard, correct DEM practice
//! for contacts whose normal isn't fixed frame-to-frame. An earlier version
//! tracked this as a bare scalar along "the current tangent direction,"
//! silently assuming the normal changes slowly -- real, confirmed WRONG for
//! the general two-mutually-free-bodies case: a direct kinetic-energy
//! invariant test (`population::tests::two_free_grains_never_gain_kinetic_
//! energy_without_an_external_driver`) measured genuine, dt-independent,
//! unbounded energy growth (violates conservation with zero external
//! driver) traced to exactly this simplification, not a sign error in the
//! force law itself (confirmed separately: a one-pinned-body sliding test
//! converges correctly, matching the derived physics exactly). Rolling
//! stays a scalar -- 2D relative rotation is intrinsically direction-free,
//! unlike tangential displacement, which lives in the (rotating) tangent
//! plane.

use glam::Vec2;

/// Per-contact-pair persistent elastic memory -- the actual mechanism that
/// lets this model hold a genuine STATIC moment/force at zero relative
/// motion (unlike every rate-only mechanism already tried). Reset to zero
/// when a contact breaks (overlap <= 0) and reformed fresh if the same pair
/// touches again later -- real DEM convention, a broken contact has no
/// memory of its prior elastic state.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ContactSpring {
    /// Accumulated tangential (sliding) elastic displacement, as a real 2D
    /// vector -- re-projected onto the current tangent plane each step (see
    /// module doc) so it stays valid as the contact normal rotates.
    pub tangential: Vec2,
    /// Accumulated relative-rotation ("rolling") elastic displacement.
    pub rolling: f32,
}

/// Real material/contact parameters (Luding 2008 / Cundall & Strack 1979 /
/// Ai et al. 2011). All stiffnesses in force-per-length (normal/tangential)
/// or torque-per-angle (rolling) grid units; frictions are dimensionless
/// coefficients, same convention as `DruckerPragerMaterial::friction_angle`'s
/// own `tan(phi)` usage elsewhere in this codebase.
#[derive(Clone, Copy, Debug)]
pub struct ContactLawConfig {
    pub normal_stiffness: f32,
    pub tangential_stiffness: f32,
    pub rolling_stiffness: f32,
    /// Normal dashpot damping coefficient (Cundall & Strack 1979's own
    /// `c_n`). 0.0 = perfectly elastic normal contact (real, valid choice --
    /// a coefficient of restitution of 1).
    pub normal_damping: f32,
    /// Tangential dashpot damping coefficient -- real, standard DEM
    /// practice (every real reference implementation cross-checked this
    /// session has a SEPARATE tangential/shear damping alongside normal
    /// damping, e.g. GeoTaichi's own `ndratio`/`sdratio` pair). Damps the
    /// tangential relative velocity `v_t` (which includes the rotational
    /// contribution -- see `resolve_contact_pair`'s own doc), the actual
    /// dissipation mechanism for the tangential+rotational subsystem.
    /// Real, confirmed necessity, not a guess: an earlier version of this
    /// config omitted this entirely, and a real column-collapse test
    /// (`tests/grains_repose_angle.rs`) showed genuine unbounded energy
    /// growth (measured runout 12x the real predicted value and still
    /// growing at 40,000 steps) once the tangential/rotational coupling
    /// was otherwise correctly wired -- an undamped oscillatory subsystem
    /// integrated explicitly is a well-known real source of numerical
    /// energy injection, not a sign the underlying force law is wrong.
    pub tangential_damping: f32,
    /// Rolling dashpot damping coefficient -- the real, missing piece
    /// identified once the rolling-torque sign fix (see `resolve_contact_pair`'s
    /// own doc, 2026-08-03) took the 8-grain column-collapse test from a
    /// 31.3x/negative-center_y explosion to a near-exact 1.045x match, but
    /// left the FULL 80-grain column still growing (12.1x -> 3.9x: a real
    /// improvement, not a full fix). Same real precedent as `tangential_damping`
    /// -- GeoTaichi's own model has an independent `rdratio` alongside
    /// `ndratio`/`sdratio`, a THIRD separate damping channel, not a guess.
    /// Without it, the rolling spring is a purely elastic-plastic oscillator:
    /// correctly restoring now that the sign is fixed, but undamped, so many
    /// simultaneous rolling contacts across a real pile can still slowly pump
    /// energy in via the same explicit-integration mechanism `tangential_damping`'s
    /// own doc already explains for the tangential channel.
    pub rolling_damping: f32,
    /// Sliding Coulomb friction coefficient (same role as `DruckerPragerMaterial`'s
    /// `tan(friction_angle)`, just for a contact pair instead of a material point).
    pub friction: f32,
    /// Rolling friction coefficient (Ai et al. 2011's own real survey table:
    /// dimensionless, real range ~0.001-0.3 depending on grain angularity).
    pub rolling_friction: f32,
}

/// Real, standard DEM Rayleigh-type critical-timestep stability bound
/// (Luding 2008): the largest stable explicit timestep for a linear
/// spring-dashpot contact of reduced mass `m_eff` and stiffness `k`. Same
/// role as this codebase's own `rod_cfl_dt` -- a second, localized
/// stability constraint distinct from the MPM CFL bound, meant to be
/// folded into the adaptive substep chooser, not left documentation-only.
pub fn critical_timestep(m_eff: f32, config: &ContactLawConfig) -> f32 {
    let k_max = config
        .normal_stiffness
        .max(config.tangential_stiffness)
        .max(config.rolling_stiffness);
    (m_eff / k_max).sqrt()
}

/// Real, conservative critical-timestep bound for the Hertzian model
/// (`HertzianContactConfig`, below). GeoTaichi's own formula
/// (`calcu_critical_timestep` in `HertzMindlinModel.py`) needs density +
/// Poisson's ratio, inputs this engine's grains don't carry (mass + radius
/// only, no material-density concept). Simpler, real, defensible choice
/// used here instead: evaluate the SAME Rayleigh-type bound
/// `critical_timestep` above already uses, but at a real, conservative
/// WORST-CASE contact state (`kn`/`ks` evaluated at
/// `WORST_CASE_OVERLAP_FRACTION * radius` overlap) rather than at the true
/// instantaneous, overlap-dependent stiffness (which would require knowing
/// the overlap in advance). Genuinely conservative, not a guess: Hertzian
/// contact area only GROWS with overlap, so evaluating at a real, plausible
/// worst-case penetration overestimates the stiffness (and thus
/// underestimates the safe dt) rather than the other way around.
pub fn critical_timestep_hertzian(m_eff: f32, radius: f32, config: &HertzianContactConfig) -> f32 {
    const WORST_CASE_OVERLAP_FRACTION: f32 = 0.1;
    let worst_case_overlap = WORST_CASE_OVERLAP_FRACTION * radius;
    let r_eff = radius * 0.5; // conservative: same-radius pair, r_eff = r/2
    let contact_area_radius = (worst_case_overlap * r_eff).sqrt();
    let kn = 2.0 * config.effective_young_modulus * contact_area_radius;
    let ks = 8.0 * config.effective_shear_modulus * contact_area_radius;
    let k_max = kn.max(ks).max(config.rolling_stiffness);
    (m_eff / k_max).sqrt()
}

impl ContactLawConfig {
    /// Real, cited, disclosed preset for dry sand-like granular material --
    /// derives every stiffness/damping value from real physical inputs via
    /// the standard DEM formulas this module's own citations already use
    /// (Cundall & Strack 1979 linear-spring calibration `kn ~ E*r`, a real
    /// critical-damping ratio applied per-channel), instead of each CALLER
    /// hand-rolling its own literal numbers. Real, confirmed problem this
    /// replaces (2026-08-19): `examples/sand_repose_angle_gui.rs` and
    /// `tests/grains_repose_angle.rs` each independently hardcoded a
    /// DIFFERENT stiffness scale, and the test file's own damping used an
    /// unexplained literal `2.01` standing in for `m_eff` where an 8x-off
    /// value (real m_eff for these grains is ~0.25kg, not 2.01) meant the
    /// documented "60% critical" damping was actually running at ~170% of
    /// TRUE critical (confirmed by comparing against this SAME module's own
    /// already-established `m_eff = grain_mass*0.5` convention, already
    /// used correctly by `critical_timestep` above) -- a real, silent
    /// inconsistency, not a deliberate choice. Fixed here by taking
    /// `m_eff_kg` as a real, explicit, non-hidden parameter.
    ///
    /// `rolling_friction` is deliberately a REAL, PER-MATERIAL input, not a
    /// value baked into this preset -- direct portability measurement
    /// (2026-08-19, `tests/grains_repose_angle.rs`'s own
    /// `diag_portability_across_friction_angle`) confirmed a calibrated
    /// rolling-friction value is NOT a universal constant across sliding
    /// friction angles: holding it fixed while sweeping sliding friction
    /// 25-45deg gave real, non-monotonic deviations up to ~30%, exactly
    /// matching Ai et al. 2011's own real finding that rolling friction
    /// depends on grain angularity/shape -- a real, independent material
    /// property, not derivable from sliding friction. Pick a real value
    /// from Ai et al. 2011's own cited survey range (0.001-0.3) for the
    /// ACTUAL material being modeled, don't reuse another material's
    /// calibrated value unexamined.
    pub fn dry_sand(
        young_modulus_pa: f32,
        grain_radius_m: f32,
        m_eff_kg: f32,
        friction_angle_deg: f32,
        rolling_friction: f32,
    ) -> Self {
        let kn = young_modulus_pa * grain_radius_m;
        // HONEST DISCLOSURE: unlike `kn` above (Cundall & Strack 1979), the 0.8 and
        // 0.1 ratios below have no individual literature source -- a tangential/
        // rolling stiffness genuinely below the normal stiffness is physically
        // expected (shear/rolling contact compliance is always softer than direct
        // normal compression), but these two specific ratios are a reasonable,
        // undisclosed-until-now engineering choice, not a measured or cited value.
        let kt = 0.8 * kn;
        let kr = kn * grain_radius_m * grain_radius_m * 0.1;
        // Real ~60%-critical damping ratio, applied per-channel to each
        // channel's OWN stiffness (real dry sand grains are genuinely lossy
        // colliders -- real coefficient of restitution for sand is commonly
        // cited around 0.5 or lower, most of a collision's kinetic energy
        // converting to heat/sound/micro-plastic deformation, not an
        // elastic bounce).
        const DAMPING_RATIO: f32 = 0.6;
        let critical_damping = |k: f32| 2.0 * (k * m_eff_kg).sqrt() * DAMPING_RATIO;
        Self {
            normal_stiffness: kn,
            tangential_stiffness: kt,
            rolling_stiffness: kr,
            normal_damping: critical_damping(kn),
            tangential_damping: critical_damping(kt),
            rolling_damping: critical_damping(kr),
            friction: friction_angle_deg.to_radians().tan(),
            rolling_friction,
        }
    }
}

/// Real, cited Hertzian (nonlinear) contact model -- Johnson 1985 "Contact
/// Mechanics" elastic-sphere theory, formulas cross-checked against a real
/// shipped implementation (`tmp/GeoTaichi/src/physics_model/contact_model/
/// HertzMindlinModel.py`) before writing any code here. Unlike
/// `ContactLawConfig`'s linear spring (Cundall & Strack 1979, constant
/// stiffness), Hertzian contact stiffness GROWS with the contact patch
/// (`kn ~ sqrt(overlap)`), so normal force grows as `overlap^1.5`, not
/// `overlap^1` -- the real behavior of two smooth elastic spheres (steel,
/// glass), not granular material.
///
/// Found necessary live 2026-08-21: `grain_newtons_cradle_gui.rs`'s own
/// chain-collision demo showed a genuine, measured limitation of the
/// LINEAR model for this specific scenario -- a constant-stiffness
/// spring's compression pulse doesn't sharpen/localize through a touching
/// chain the way a real Hertzian contact's does (real, published
/// granular-chain physics: Nesterenko 2001, "Dynamics of Heterogeneous
/// Materials", solitary-wave propagation in chains of Hertzian spheres),
/// so momentum measurably lingered at middle grains instead of passing
/// cleanly through, confirmed via a direct diagnostic
/// (`tests/grains_grid_coupling.rs::diag_newtons_cradle_first_collision_
/// immediate_aftermath`) before any of this was written.
///
/// A real, additive, GENERAL-PURPOSE engine capability, not a demo-only
/// hack -- any future scene needing smooth hard-body contact (not just
/// this cradle) can use it via `GrainPopulation::new_hertzian`; it does
/// not replace `ContactLawConfig`, which stays the right, unchanged choice
/// for granular/sand material everywhere else in this codebase.
///
/// Rolling resistance stays the SAME elastic-plastic EPSD spring
/// (`rolling_stiffness`/`rolling_damping`/`rolling_friction`, Ai et al.
/// 2011) `resolve_contact_pair` already uses -- GeoTaichi's own Hertzian
/// rolling term is a plain Coulomb cap with no memory, but this engine's
/// own EPSD model is what made real STATIC repose-angle behavior work
/// earlier this session, and it's an independent, orthogonal concern from
/// normal/tangential pulse propagation (the actual cradle problem).
#[derive(Clone, Copy, Debug)]
pub struct HertzianContactConfig {
    /// Effective Young's modulus for the CONTACT PAIR (real formula: for
    /// two bodies of modulus E1,E2 and Poisson ratio nu1,nu2, `1/E_eff =
    /// (1-nu1^2)/E1 + (1-nu2^2)/E2` -- simplified here to a single real,
    /// per-material input, same "effective," not per-body, convention
    /// GeoTaichi's own `YoungModulus` field uses). Real formula, STYLIZED
    /// magnitude: real steel is ~200 GPa, which would force a punishingly
    /// fine dt at this engine's own grid-unit scale (same real tradeoff
    /// already disclosed in `grain_rolling_closeup_gui.rs`'s own
    /// `grain_contact_config` doc) -- pick a value at the SAME order of
    /// magnitude as `ContactLawConfig::normal_stiffness` elsewhere in this
    /// codebase, not a real SI number.
    pub effective_young_modulus: f32,
    /// Effective shear modulus, same real/stylized convention as above.
    pub effective_shear_modulus: f32,
    /// Coefficient of restitution (real, physical, dimensionless -- 1.0 =
    /// perfectly elastic, 0.0 = perfectly inelastic). Set this to the real
    /// physical value directly (e.g. 0.95 for steel-on-steel); the resolve
    /// functions internally convert it via `hertzian_damping_coefficient`
    /// before it ever reaches the Tsuji, Tanaka & Ishida 1992 damping
    /// formula -- unlike the linear model, which takes damping
    /// COEFFICIENTS directly, this field is the real restitution value
    /// itself, not pre-transformed. (Real, confirmed bug fixed 2026-08-21:
    /// an earlier version used this raw value directly in that formula,
    /// ~58x too large for e=0.95 -- see `hertzian_damping_coefficient`'s
    /// own doc for the full story.)
    pub restitution: f32,
    /// Sliding Coulomb friction coefficient, same role as
    /// `ContactLawConfig::friction`.
    pub friction: f32,
    /// Rolling-resistance spring stiffness/damping/friction -- same real
    /// EPSD model and units as `ContactLawConfig`'s own fields, kept as
    /// explicit, independent material inputs (not derived from
    /// `effective_young_modulus`/`effective_shear_modulus` -- no real
    /// citation covers deriving rolling resistance from elastic moduli,
    /// same "independent material property" precedent this file's own
    /// `dry_sand()` doc already establishes for `rolling_friction`).
    pub rolling_stiffness: f32,
    pub rolling_damping: f32,
    pub rolling_friction: f32,
}

/// One grain's contact-relevant state -- position, velocity, angular
/// velocity (2D planar spin, a scalar: this engine is 2D throughout, so
/// there is no 3D angular-velocity vector to track), radius, and mass.
#[derive(Clone, Copy, Debug)]
pub struct GrainContactState {
    pub x: Vec2,
    pub v: Vec2,
    pub spin: f32,
    pub radius: f32,
    pub mass: f32,
}

/// Resolved contact outputs for one pair, this substep. Forces/moment are
/// given as acting ON grain `j` (the second argument to `resolve_contact_pair`)
/// -- by Newton's third law, grain `i` feels the exact negation, which the
/// caller applies directly rather than this function computing it twice.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ContactResolution {
    /// Normal force magnitude (always >= 0, purely repulsive -- no adhesion),
    /// acting along the unit normal from `i` to `j`.
    pub normal_force: f32,
    /// Tangential (sliding-friction) force vector acting on `j`.
    pub tangential_force: Vec2,
    /// Rolling-resistance moment acting on `j` (equal and opposite on `i`,
    /// same convention as the linear forces -- real physics: a rolling
    /// contact moment always acts as a genuine action-reaction pair, exactly
    /// like a linear contact force, per Ai et al. 2011's own formulation).
    pub rolling_moment: f32,
    /// Real, separate torque source from the tangential force itself acting
    /// at the true contact point -- offset from `i`'s own center by `i.radius`
    /// along `n`, and from `j`'s own center by `-j.radius` along `n` -- NOT
    /// the same thing as `rolling_moment` above (which resists relative SPIN
    /// directly, independent of geometry). This is the real, standard rigid-
    /// body mechanics result (torque = r x F) for a force applied away from
    /// a body's own center of mass: this term is the actual mechanism by
    /// which real friction induces rolling from pure sliding contact at all.
    /// Torque on `i` uses `i.radius`, torque on `j` uses `j.radius` -- same
    /// underlying tangential force, different moment arms, returned
    /// separately since the caller applies each to a different body.
    pub friction_torque_on_i: f32,
    pub friction_torque_on_j: f32,
}

/// Resolves one grain-grain contact pair for one substep, given `dt` and
/// the pair's persistent elastic spring state (updated in place). Returns
/// `None` (springs reset to zero) when the grains are not actually
/// overlapping -- a broken contact has no memory, real DEM convention.
///
/// Real force law (see module doc for full citations):
/// - Normal: `F_n = kn*overlap - c_n*v_n`, clamped to `>= 0` (repulsive only).
/// - Tangential: elastic trial `-ks*spring`, Coulomb-capped at `mu*F_n`,
///   spring plastically rescaled on cap (same return-mapping pattern as
///   this codebase's own material plasticity).
/// - Rolling: elastic trial `-kr*spring`, capped at `mu_r*r_eff*F_n`, same
///   plastic correction on cap.
///
/// Rolling-resistance spring shared by every contact model (linear AND
/// Hertzian) and every geometry (grain-grain AND grain-wall) -- the real
/// Ai et al. 2011 elastic-plastic EPSD spring is byte-identical across all
/// four `resolve_*` functions below, only the caller-derived kinematics
/// (`omega_rel`, `r_eff`) and the already-resolved `normal_force` differ.
///
/// Real sign fix, 2026-08-03: `spring.rolling` (call it R) is exactly the
/// relative-rotation coordinate R = integral(omega_rel dt) = theta_i -
/// theta_j -- a genuine torsional-spring coordinate between the two
/// bodies' own rotation angles, same role as the tangential spring but
/// for the ROTATIONAL dof. `resolve_contact_forces` (population.rs)
/// documents and applies `rolling_moment` with the SAME convention as the
/// linear forces: "acting on j, equal and opposite on i"
/// (`torques[j] += rolling_moment; torques[i] -= rolling_moment;`). For a
/// torsional spring potential U(R) = 0.5*kr*R^2, the physically correct
/// generalized force (real Lagrangian mechanics, Q = -dU/dtheta) on that
/// convention is Q_j = -kr*R*(dR/dtheta_j) = -kr*R*(-1) = +kr*R -- i.e.
/// `rolling_moment` itself must carry a PLUS sign, not minus. The
/// previous `-kr*R` was exactly backwards (it's the formula for Q_i, not
/// Q_j, applied at j's callsite) -- confirmed empirically, not just by
/// derivation: instrumenting a single grain resting on a huge tilted
/// pinned floor (tests/grains_repose_angle.rs's
/// `diag_instrumented_single_step_breakdown`) showed `omega_rel`/`spin_i`
/// growing MONOTONICALLY (never oscillating back toward zero, the
/// opposite of what a real restoring torsional spring does) and, once
/// the Coulomb-like cap engaged, a torque that stayed pinned in a
/// constant, growth-REINFORCING direction forever instead of opposing
/// continued spin-up -- the textbook signature of positive feedback from
/// a flipped restoring-force sign, not a stiff-but-stable oscillator.
/// This is the real root cause of the long-standing column-collapse
/// divergence too (any real pile has grains resting at off-axis angles,
/// which is exactly the code path a perfectly-vertical stack never
/// exercises).
/// The three rolling-resistance coefficients, which always travel together
/// from a contact config. Bundled so `resolve_rolling_spring` stays under
/// clippy's argument threshold by FIXING the cause (too many loose
/// parameters) rather than silencing the lint -- same pattern
/// `ProjectInputs`/`G2PParams` already use elsewhere in this codebase.
#[derive(Clone, Copy)]
struct RollingParams {
    stiffness: f32,
    damping: f32,
    friction: f32,
}

fn resolve_rolling_spring(
    spring: &mut ContactSpring,
    omega_rel: f32,
    dt: f32,
    rolling: RollingParams,
    r_eff: f32,
    normal_force: f32,
) -> f32 {
    let (rolling_stiffness, rolling_damping, rolling_friction) =
        (rolling.stiffness, rolling.damping, rolling.friction);
    spring.rolling += omega_rel * dt;
    // Real dashpot damping added alongside the elastic term (2026-08-03,
    // same real necessity as `tangential_damping`'s own doc): a positive
    // `omega_rel` contributes a positive moment here (matching the fixed
    // elastic sign above), so it reinforces -- not opposes -- the spring's
    // own restoring action, genuinely dissipating relative-rotation energy
    // rather than just storing/returning it elastically.
    let trial_mr = rolling_stiffness * spring.rolling + rolling_damping * omega_rel;
    let max_mr = rolling_friction * r_eff * normal_force;
    if trial_mr.abs() > max_mr {
        let clamped = trial_mr.signum() * max_mr;
        spring.rolling = clamped / rolling_stiffness;
        clamped
    } else {
        trial_mr
    }
}

/// Real, shared linear-model (Cundall & Strack 1979) contact core --
/// `resolve_contact_pair` and `resolve_wall_contact` differ only in how
/// they derive `overlap`/`n`/`t`/`v_n`/`v_t`/`omega_rel`/`r_eff` (two free
/// bodies vs. one grain against a fixed wall); once derived, the actual
/// spring/Coulomb resolution is byte-identical, so it lives here once.
/// Returns `(normal_force, tangential_force_vec, rolling_moment, ft_scalar)`
/// -- callers turn `ft_scalar` into their own torque distribution (two-body
/// action-reaction vs. one-sided wall torque).
/// The contact-pair kinematics both core resolvers consume, bundled so
/// neither needs `#[allow(clippy::too_many_arguments)]` -- fixing the cause
/// (too many loose parameters that always travel together) rather than
/// silencing the lint, matching `ProjectInputs`/`G2PParams` elsewhere in
/// this codebase. Every field is derived from the same contact geometry and
/// relative-velocity computation at the call site.
#[derive(Clone, Copy)]
struct ContactKinematics {
    overlap: f32,
    n: Vec2,
    t: Vec2,
    v_n: f32,
    v_t: f32,
    omega_rel: f32,
    r_eff: f32,
}

fn resolve_contact_core_linear(
    kin: ContactKinematics,
    spring: &mut ContactSpring,
    config: &ContactLawConfig,
    dt: f32,
) -> (f32, Vec2, f32, f32) {
    let ContactKinematics {
        overlap,
        n,
        t,
        v_n,
        v_t,
        omega_rel,
        r_eff,
    } = kin;
    // Normal: linear spring-dashpot (Cundall & Strack 1979), repulsive only.
    let normal_force = (config.normal_stiffness * overlap - config.normal_damping * v_n).max(0.0);

    // Tangential: elastic-plastic Coulomb spring + real dashpot damping
    // (see `tangential_damping`'s own doc -- the actual dissipation
    // mechanism for the tangential/rotational subsystem; without it this
    // is undamped and explicit integration genuinely injects energy into
    // it over many contact cycles).
    //
    // Real tangent-plane rotation correction (see module doc, 2026-08-03
    // fix): reproject the spring onto the CURRENT tangent plane before
    // adding this step's increment, discarding whatever normal-direction
    // component has drifted in as `n` itself rotated since the spring was
    // last updated -- without this, a real, confirmed, dt-independent
    // energy-conservation violation occurs whenever the contact normal
    // changes direction over time (the general two-mutually-free-bodies
    // case; a one-body-fixed contact's normal barely rotates, which is why
    // that case tested fine in isolation).
    spring.tangential -= n * spring.tangential.dot(n);
    spring.tangential += v_t * t * dt;
    let trial_ft_vec =
        -config.tangential_stiffness * spring.tangential - config.tangential_damping * v_t * t;
    let max_ft = config.friction * normal_force;
    let trial_ft_mag = trial_ft_vec.length();
    let tangential_force_vec = if trial_ft_mag > max_ft {
        let clamped = trial_ft_vec * (max_ft / trial_ft_mag.max(1.0e-12));
        spring.tangential = -clamped / config.tangential_stiffness;
        clamped
    } else {
        trial_ft_vec
    };
    let ft_scalar = tangential_force_vec.dot(t);

    let rolling_moment = resolve_rolling_spring(
        spring,
        omega_rel,
        dt,
        RollingParams {
            stiffness: config.rolling_stiffness,
            damping: config.rolling_damping,
            friction: config.rolling_friction,
        },
        r_eff,
        normal_force,
    );

    (
        normal_force,
        tangential_force_vec,
        rolling_moment,
        ft_scalar,
    )
}

pub fn resolve_contact_pair(
    i: &GrainContactState,
    j: &GrainContactState,
    spring: &mut ContactSpring,
    config: &ContactLawConfig,
    dt: f32,
) -> Option<ContactResolution> {
    let d = j.x - i.x;
    let dist = d.length();
    if dist <= 1e-12 {
        // Degenerate (coincident centers) -- no well-defined normal; treat
        // as no contact rather than dividing by zero.
        *spring = ContactSpring::default();
        return None;
    }
    let overlap = i.radius + j.radius - dist;
    if overlap <= 0.0 {
        *spring = ContactSpring::default();
        return None;
    }
    let n = d / dist;
    let t = Vec2::new(-n.y, n.x);

    let r_eff = (i.radius * j.radius) / (i.radius + j.radius);

    let v_rel = j.v - i.v;
    let v_n = v_rel.dot(n);
    // Tangential slip velocity at the contact point -- real, derived
    // formula (surface velocity of each grain at the shared contact point,
    // including its own spin contribution): see module doc's derivation
    // reference (Zhu et al. 2007-style standard 2D DEM contact-point
    // kinematics). Sign convention verified against the physical
    // "two equal-radius grains rolling on each other without slipping
    // requires opposite-sign spin" check in this module's own tests.
    let v_t = v_rel.dot(t) - (i.radius * i.spin + j.radius * j.spin);
    let omega_rel = i.spin - j.spin;

    let (normal_force, tangential_force_vec, rolling_moment, ft_scalar) =
        resolve_contact_core_linear(
            ContactKinematics {
                overlap,
                n,
                t,
                v_n,
                v_t,
                omega_rel,
                r_eff,
            },
            spring,
            config,
            dt,
        );

    // Real torque from the tangential force acting at the true contact
    // point (offset from each center by its own radius along n): derived
    // via torque = r x F in 2D (cross(a,b) = a.x*b.y - a.y*b.x). Contact
    // point relative to i's center is +i.radius*n; relative to j's center
    // is -j.radius*n. Only the tangential component contributes (the
    // normal component is parallel to the offset vector, cross product
    // zero) -- verified: cross(n, t) = 1 exactly since t is n rotated 90
    // degrees, so both simplify to -radius * ft_scalar.
    let friction_torque_on_i = -i.radius * ft_scalar;
    let friction_torque_on_j = -j.radius * ft_scalar;

    Some(ContactResolution {
        normal_force,
        tangential_force: tangential_force_vec,
        rolling_moment,
        friction_torque_on_i,
        friction_torque_on_j,
    })
}

/// Resolves one grain-vs-WALL contact for one substep -- the real, missing
/// piece found live 2026-08-21: a grain resting on the ground had NO
/// mechanism anywhere to start rolling from rest, only grain-grain contact
/// (`resolve_contact_pair` above) ever produced torque. This is the exact
/// same real physics, specialized for a fixed wall (infinite effective
/// mass, zero velocity, zero spin) instead of deriving it via an extreme
/// "huge radius" numeric hack (which would lose real precision in f32 --
/// `r_eff = grain.radius` is the correct ANALYTIC limit as the wall's own
/// radius goes to infinity in `r_eff = i.radius*j.radius/(i.radius+j.radius)`,
/// used directly rather than approximated).
///
/// `normal` points AWAY from the wall surface (toward the grain, same
/// convention `BoundaryCondition::grain_contact` returns); `overlap` is how
/// far the grain's own surface has penetrated the wall (`grain.radius -
/// distance_to_surface`). Returns `None` (spring reset) when not actually
/// overlapping, same "broken contact has no memory" convention as the
/// grain-grain version.
pub fn resolve_wall_contact(
    grain: &GrainContactState,
    normal: Vec2,
    overlap: f32,
    spring: &mut ContactSpring,
    config: &ContactLawConfig,
    dt: f32,
) -> Option<ContactResolution> {
    if overlap <= 0.0 {
        *spring = ContactSpring::default();
        return None;
    }
    let n = normal;
    let t = Vec2::new(-n.y, n.x);
    let r_eff = grain.radius; // analytic limit as the wall's own radius -> infinity

    let v_rel = grain.v; // wall velocity is zero
    let v_n = v_rel.dot(n);
    let v_t = v_rel.dot(t) - grain.radius * grain.spin;
    let omega_rel = -grain.spin; // wall spin is zero

    let (normal_force, tangential_force_vec, rolling_moment, ft_scalar) =
        resolve_contact_core_linear(
            ContactKinematics {
                overlap,
                n,
                t,
                v_n,
                v_t,
                omega_rel,
                r_eff,
            },
            spring,
            config,
            dt,
        );

    // Same real torque = r x F mechanism as `resolve_contact_pair`'s own
    // `friction_torque_on_j`, using the grain's own radius as the moment
    // arm -- the actual mechanism by which static/kinetic friction at the
    // contact point induces real rolling from rest, not just resisting
    // existing slip.
    let friction_torque = -grain.radius * ft_scalar;

    Some(ContactResolution {
        normal_force,
        tangential_force: tangential_force_vec,
        rolling_moment,
        friction_torque_on_i: 0.0,
        friction_torque_on_j: friction_torque,
    })
}

/// Hertzian (nonlinear) counterpart to `resolve_contact_pair` -- see
/// `HertzianContactConfig`'s own doc for why this exists and its real
/// citations. Structurally identical to the linear version (same spring-
/// state types, same elastic-plastic Coulomb tangential/rolling structure,
/// same tangent-plane reprojection convention) -- only the NORMAL and
/// TANGENTIAL stiffness/damping terms change, per the real Hertz-Mindlin
/// formulas (`tmp/GeoTaichi/src/physics_model/contact_model/
/// HertzMindlinModel.py`, adapted to this engine's own `overlap > 0` sign
/// convention, the opposite of GeoTaichi's own `gapn < 0`). Real, disclosed
/// adaptation: GeoTaichi's own tangential damping only applies in the
/// sub-yield branch (added AFTER the Coulomb check); this folds it into
/// the trial force BEFORE the check instead, matching THIS engine's own
/// existing `resolve_contact_pair` structure exactly (consistency with the
/// rest of this file, not a different algorithm).
///
/// Real, cited conversion from the physical coefficient of restitution `e`
/// to the actual damping COEFFICIENT the Tsuji, Tanaka & Ishida 1992
/// formula needs -- found live 2026-08-21, a real, confirmed bug: an
/// earlier version of `resolve_contact_pair_hertzian`/`resolve_wall_
/// contact_hertzian` used the raw `e` (e.g. 0.95) directly in the formula
/// `-1.8257 * e * v * sqrt(k*m_eff)`, but GeoTaichi's own `HertzMindlin.py::
/// add_surface_property` does NOT use the raw input that way -- it
/// OVERWRITES its own `restitution` variable with this exact transform
/// (`-log(e)/sqrt(pi^2+log(e)^2)`) before ever using it in that formula.
/// Using the raw 0.95 directly was a ~58x too-large damping coefficient
/// (0.95 vs the correctly-transformed ~0.0163), confirmed via a direct,
/// isolated single-pair-collision test measuring the ACTUAL post-collision
/// velocity split against the real, standard 1D restitution formula:
/// `v0' = (1-e)/2 * v0`, `v1' = (1+e)/2 * v0`. The bug showed an effective
/// restitution of ~0.14 for a specified e=0.95, i.e. the collision was
/// behaving far more energy-absorbing (closer to perfectly inelastic) than
/// intended, while total momentum still conserved exactly (the bug was in
/// the SPLIT, not the conservation law itself).
fn hertzian_damping_coefficient(restitution: f32) -> f32 {
    if restitution < 1.0e-6 {
        return 0.0;
    }
    let ln_e = restitution.ln();
    -ln_e / (std::f32::consts::PI * std::f32::consts::PI + ln_e * ln_e).sqrt()
}

/// Real, shared Hertzian-model contact core -- same real relationship to
/// `resolve_contact_pair_hertzian`/`resolve_wall_contact_hertzian` that
/// `resolve_contact_core_linear` has to the linear pair: the two callers
/// differ only in how they derive `overlap`/`n`/`t`/`v_n`/`v_t`/
/// `omega_rel`/`r_eff`/`m_eff`, the actual Hertz-Mindlin + Tsuji-damping
/// resolution is byte-identical once those are known.
fn resolve_contact_core_hertzian(
    kin: ContactKinematics,
    m_eff: f32,
    spring: &mut ContactSpring,
    config: &HertzianContactConfig,
    dt: f32,
) -> (f32, Vec2, f32, f32) {
    let ContactKinematics {
        overlap,
        n,
        t,
        v_n,
        v_t,
        omega_rel,
        r_eff,
    } = kin;
    // Real Hertzian contact-patch-dependent stiffness -- grows with
    // overlap, unlike the linear model's constant `normal_stiffness`.
    let contact_area_radius = (overlap * r_eff).sqrt();
    let kn = 2.0 * config.effective_young_modulus * contact_area_radius;
    let ks = 8.0 * config.effective_shear_modulus * contact_area_radius;
    let damping_coeff = hertzian_damping_coefficient(config.restitution);

    // Real Hertzian normal force + Tsuji, Tanaka & Ishida 1992 nonlinear
    // damping (the "1.8257" constant is that paper's own real, cited
    // value, not a guess; `damping_coeff` is `hertzian_damping_coefficient`'s
    // own real transform of the physical restitution -- see its own doc for
    // why this is NOT the raw restitution value).
    let normal_force =
        ((2.0 / 3.0) * kn * overlap - 1.8257 * damping_coeff * v_n * (kn * m_eff).sqrt()).max(0.0);

    spring.tangential -= n * spring.tangential.dot(n);
    spring.tangential += v_t * t * dt;
    let trial_ft_vec =
        -ks * spring.tangential - 1.8257 * damping_coeff * v_t * (ks * m_eff).sqrt() * t;
    let max_ft = config.friction * normal_force;
    let trial_ft_mag = trial_ft_vec.length();
    let tangential_force_vec = if trial_ft_mag > max_ft {
        let clamped = trial_ft_vec * (max_ft / trial_ft_mag.max(1.0e-12));
        spring.tangential = -clamped / ks.max(1.0e-6);
        clamped
    } else {
        trial_ft_vec
    };
    let ft_scalar = tangential_force_vec.dot(t);

    // Rolling: SAME real EPSD spring as the linear model -- see
    // `HertzianContactConfig`'s own doc for why this stays unchanged.
    let rolling_moment = resolve_rolling_spring(
        spring,
        omega_rel,
        dt,
        RollingParams {
            stiffness: config.rolling_stiffness,
            damping: config.rolling_damping,
            friction: config.rolling_friction,
        },
        r_eff,
        normal_force,
    );

    (
        normal_force,
        tangential_force_vec,
        rolling_moment,
        ft_scalar,
    )
}

pub fn resolve_contact_pair_hertzian(
    i: &GrainContactState,
    j: &GrainContactState,
    spring: &mut ContactSpring,
    config: &HertzianContactConfig,
    dt: f32,
) -> Option<ContactResolution> {
    let d = j.x - i.x;
    let dist = d.length();
    if dist <= 1e-12 {
        *spring = ContactSpring::default();
        return None;
    }
    let overlap = i.radius + j.radius - dist;
    if overlap <= 0.0 {
        *spring = ContactSpring::default();
        return None;
    }
    let n = d / dist;
    let t = Vec2::new(-n.y, n.x);

    let r_eff = (i.radius * j.radius) / (i.radius + j.radius);
    let m_eff = (i.mass * j.mass) / (i.mass + j.mass);

    let v_rel = j.v - i.v;
    let v_n = v_rel.dot(n);
    let v_t = v_rel.dot(t) - (i.radius * i.spin + j.radius * j.spin);
    let omega_rel = i.spin - j.spin;

    let (normal_force, tangential_force_vec, rolling_moment, ft_scalar) =
        resolve_contact_core_hertzian(
            ContactKinematics {
                overlap,
                n,
                t,
                v_n,
                v_t,
                omega_rel,
                r_eff,
            },
            m_eff,
            spring,
            config,
            dt,
        );

    let friction_torque_on_i = -i.radius * ft_scalar;
    let friction_torque_on_j = -j.radius * ft_scalar;

    Some(ContactResolution {
        normal_force,
        tangential_force: tangential_force_vec,
        rolling_moment,
        friction_torque_on_i,
        friction_torque_on_j,
    })
}

/// Hertzian (nonlinear) counterpart to `resolve_wall_contact` -- see
/// `HertzianContactConfig`'s and `resolve_contact_pair_hertzian`'s own doc
/// for why this exists and its real citations. Wall = infinite effective
/// mass would make `m_eff` blow up in the Hertzian damping formulas
/// (`sqrt(kn * m_eff)` diverging), so this uses `grain.mass` directly as
/// `m_eff` -- the real analytic limit of the two-body reduced mass
/// `(m1*m2)/(m1+m2)` as the wall's own mass goes to infinity, same
/// "analytic limit, not an approximation" precedent `resolve_wall_contact`'s
/// own `r_eff = grain.radius` doc already establishes for the radius term.
pub fn resolve_wall_contact_hertzian(
    grain: &GrainContactState,
    normal: Vec2,
    overlap: f32,
    spring: &mut ContactSpring,
    config: &HertzianContactConfig,
    dt: f32,
) -> Option<ContactResolution> {
    if overlap <= 0.0 {
        *spring = ContactSpring::default();
        return None;
    }
    let n = normal;
    let t = Vec2::new(-n.y, n.x);
    let r_eff = grain.radius; // analytic limit as the wall's own radius -> infinity
    let m_eff = grain.mass; // analytic limit as the wall's own mass -> infinity

    let v_rel = grain.v; // wall velocity is zero
    let v_n = v_rel.dot(n);
    let v_t = v_rel.dot(t) - grain.radius * grain.spin;
    let omega_rel = -grain.spin; // wall spin is zero

    let (normal_force, tangential_force_vec, rolling_moment, ft_scalar) =
        resolve_contact_core_hertzian(
            ContactKinematics {
                overlap,
                n,
                t,
                v_n,
                v_t,
                omega_rel,
                r_eff,
            },
            m_eff,
            spring,
            config,
            dt,
        );

    let friction_torque = -grain.radius * ft_scalar;

    Some(ContactResolution {
        normal_force,
        tangential_force: tangential_force_vec,
        rolling_moment,
        friction_torque_on_i: 0.0,
        friction_torque_on_j: friction_torque,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ContactLawConfig {
        ContactLawConfig {
            normal_stiffness: 1000.0,
            tangential_stiffness: 800.0,
            rolling_stiffness: 50.0,
            normal_damping: 0.0,
            tangential_damping: 0.0,
            rolling_damping: 0.0,
            friction: 0.5,
            rolling_friction: 0.1,
        }
    }

    fn grain(x: Vec2, v: Vec2, spin: f32) -> GrainContactState {
        GrainContactState {
            x,
            v,
            spin,
            radius: 1.0,
            mass: 1.0,
        }
    }

    #[test]
    fn no_overlap_is_no_contact() {
        let i = grain(Vec2::ZERO, Vec2::ZERO, 0.0);
        let j = grain(Vec2::new(3.0, 0.0), Vec2::ZERO, 0.0);
        let mut spring = ContactSpring::default();
        let result = resolve_contact_pair(&i, &j, &mut spring, &config(), 0.01);
        assert!(result.is_none());
        assert_eq!(spring, ContactSpring::default());
    }

    #[test]
    fn normal_force_matches_linear_spring_at_rest() {
        // radii sum to 2.0, centers 1.5 apart -> overlap = 0.5.
        let i = grain(Vec2::ZERO, Vec2::ZERO, 0.0);
        let j = grain(Vec2::new(1.5, 0.0), Vec2::ZERO, 0.0);
        let mut spring = ContactSpring::default();
        let cfg = config();
        let result = resolve_contact_pair(&i, &j, &mut spring, &cfg, 0.01).unwrap();
        let expected = cfg.normal_stiffness * 0.5;
        assert!(
            (result.normal_force - expected).abs() < 1e-4,
            "F_n={} expected={}",
            result.normal_force,
            expected
        );
        // No relative motion at all -> zero tangential/rolling.
        assert!(result.tangential_force.length() < 1e-6);
        assert!(result.rolling_moment.abs() < 1e-6);
    }

    #[test]
    fn normal_force_never_negative_even_when_separating_fast() {
        let i = grain(Vec2::ZERO, Vec2::new(-100.0, 0.0), 0.0);
        let j = grain(Vec2::new(1.9, 0.0), Vec2::new(100.0, 0.0), 0.0);
        let mut spring = ContactSpring::default();
        let mut cfg = config();
        cfg.normal_damping = 10.0;
        let result = resolve_contact_pair(&i, &j, &mut spring, &cfg, 0.001).unwrap();
        assert!(result.normal_force >= 0.0, "F_n={}", result.normal_force);
    }

    #[test]
    fn equal_radius_grains_spinning_oppositely_have_zero_tangential_slip() {
        // Physical check: two equal-radius grains touching with zero
        // translational relative velocity roll on each other WITHOUT
        // slipping only when they spin in OPPOSITE senses (like meshing
        // gears must counter-rotate to roll smoothly) -- omega_i = -omega_j.
        let i = grain(Vec2::ZERO, Vec2::ZERO, 2.0);
        let j = grain(Vec2::new(1.5, 0.0), Vec2::ZERO, -2.0);
        let mut spring = ContactSpring::default();
        let cfg = config();
        // Single substep: trial tangential spring should stay at zero
        // since v_t (the rate of accumulation) is zero this whole time.
        let result = resolve_contact_pair(&i, &j, &mut spring, &cfg, 0.01).unwrap();
        assert!(
            result.tangential_force.length() < 1e-5,
            "expected ~zero tangential force for no-slip counter-rotation, got {:?}",
            result.tangential_force
        );
    }

    #[test]
    fn same_sense_spin_produces_nonzero_slip_and_rolling_response() {
        // Same physical setup, but BOTH grains spin the SAME sense (like two
        // gears jammed together, not meshing) -- real slip must appear.
        let i = grain(Vec2::ZERO, Vec2::ZERO, 2.0);
        let j = grain(Vec2::new(1.5, 0.0), Vec2::ZERO, 2.0);
        let mut spring = ContactSpring::default();
        let cfg = config();
        let result = resolve_contact_pair(&i, &j, &mut spring, &cfg, 0.01).unwrap();
        assert!(
            result.tangential_force.length() > 1e-6,
            "expected real slip-driven tangential force for same-sense spin"
        );
    }

    #[test]
    fn tangential_spring_stays_elastic_under_coulomb_cap() {
        let i = grain(Vec2::ZERO, Vec2::ZERO, 0.0);
        let j = grain(Vec2::new(1.5, 0.0), Vec2::new(0.0, 0.001), 0.0);
        let mut spring = ContactSpring::default();
        let cfg = config();
        let result = resolve_contact_pair(&i, &j, &mut spring, &cfg, 0.01).unwrap();
        let expected_ft = -cfg.tangential_stiffness * spring.tangential;
        // Tiny slip velocity -> trial force should be comfortably under the
        // Coulomb cap, so the elastic (uncapped) formula applies exactly.
        assert!(result.tangential_force.length() < cfg.friction * result.normal_force);
        assert!((result.tangential_force - expected_ft).length() < 1e-4);
    }

    #[test]
    fn tangential_force_caps_at_coulomb_limit_and_rescales_spring() {
        let i = grain(Vec2::ZERO, Vec2::ZERO, 0.0);
        // Large sustained slip velocity to drive the trial force well past
        // the Coulomb cap.
        let j = grain(Vec2::new(1.5, 0.0), Vec2::new(0.0, 50.0), 0.0);
        let mut spring = ContactSpring::default();
        let cfg = config();
        let mut result = None;
        for _ in 0..20 {
            result = resolve_contact_pair(&i, &j, &mut spring, &cfg, 0.01);
        }
        let result = result.unwrap();
        let max_ft = cfg.friction * result.normal_force;
        assert!(
            (result.tangential_force.length() - max_ft).abs() < 1e-3,
            "expected force pinned at Coulomb cap {max_ft}, got {}",
            result.tangential_force.length()
        );
        // Return-mapping invariant: the spring, re-evaluated through the
        // elastic law, must reproduce exactly the capped force (the same
        // "spring rescaled to match the yielded force" check already used
        // for this codebase's own material plasticity).
        let reconstructed = -cfg.tangential_stiffness * spring.tangential;
        assert!((reconstructed.length() - max_ft).abs() < 1e-3);
    }

    #[test]
    fn rolling_moment_caps_at_its_own_coulomb_like_limit() {
        let i = grain(Vec2::ZERO, Vec2::ZERO, 50.0);
        let j = grain(Vec2::new(1.5, 0.0), Vec2::ZERO, -50.0);
        // Note: opposite spins here mean zero SLIP, but omega_rel =
        // i.spin - j.spin = 100.0 is large -- rolling resistance responds
        // to relative SPIN directly, a genuinely separate channel from
        // tangential slip (see module doc).
        let mut spring = ContactSpring::default();
        let cfg = config();
        let mut result = None;
        for _ in 0..50 {
            result = resolve_contact_pair(&i, &j, &mut spring, &cfg, 0.01);
        }
        let result = result.unwrap();
        let max_mr = cfg.rolling_friction * 0.5 * result.normal_force; // r_eff = 1*1/(1+1) = 0.5
        assert!(
            (result.rolling_moment.abs() - max_mr).abs() < 1e-3,
            "expected rolling moment pinned at cap {max_mr}, got {}",
            result.rolling_moment.abs()
        );
    }

    #[test]
    fn rolling_moment_holds_static_nonzero_value_at_exactly_zero_rate() {
        // The whole point of this model over every rate-dependent mechanism
        // already ruled out: once a real elastic rolling spring has wound
        // up (from real deformation history), it must hold a genuine
        // nonzero moment even when the CURRENT relative spin rate is
        // exactly zero -- unlike Cosserat's curvature coupling, which is
        // strictly proportional to instantaneous curvature and vanishes the
        // instant motion stops.
        let cfg = config();
        let mut spring = ContactSpring {
            tangential: Vec2::ZERO,
            rolling: 0.05,
        };
        let i = grain(Vec2::ZERO, Vec2::ZERO, 0.0);
        let j = grain(Vec2::new(1.5, 0.0), Vec2::ZERO, 0.0); // omega_rel = 0 exactly
        let result = resolve_contact_pair(&i, &j, &mut spring, &cfg, 0.01).unwrap();
        assert!(
            result.rolling_moment.abs() > 1e-6,
            "expected a real nonzero static moment from wound-up spring history at zero rate, got {}",
            result.rolling_moment
        );
    }

    #[test]
    fn critical_timestep_matches_rayleigh_formula() {
        let cfg = ContactLawConfig {
            normal_stiffness: 400.0,
            tangential_stiffness: 100.0,
            rolling_stiffness: 25.0,
            normal_damping: 0.0,
            tangential_damping: 0.0,
            rolling_damping: 0.0,
            friction: 0.5,
            rolling_friction: 0.1,
        };
        let m_eff = 2.0;
        let dt_crit = critical_timestep(m_eff, &cfg);
        let expected = (m_eff / 400.0_f32).sqrt();
        assert!((dt_crit - expected).abs() < 1e-6);
    }
}
