//! Discrete-element contact force law with grain-scale rolling resistance.
//! Every rate-dependent mechanism tried for sand's repose angle (Cundall
//! damping, KE-peak switches, Cosserat curvature coupling) fades to zero at
//! rest; this one does not: it is elastic-plastic with memory (an
//! accumulated spring displacement), the trial-elastic / yield-check /
//! return-mapping pattern of the material plasticity
//! (`DruckerPragerMaterial`, `VonMisesMaterial`, `RankineMaterial`) applied
//! to a contact pair.
//!
//! Sources, cross-checked against GeoTaichi's `dem/contact/LinearRolling.py`
//! (which cites Luding 2008):
//! - Cundall & Strack 1979, "A discrete numerical model for granular
//!   assemblies," Geotechnique 29(1):47-65 -- the linear spring-dashpot
//!   normal and Coulomb-capped tangential spring.
//! - Luding 2008, "Introduction to discrete element methods," European
//!   Journal of Environmental and Civil Engineering 12:7-8, 785-826 -- the
//!   critical-timestep (Rayleigh-type) stability bound.
//! - Ai, Chen, Rotter & Ooi 2011, "Assessment of rolling resistance models
//!   in discrete element simulations," Powder Technology 206(3):269-282 --
//!   compared rolling-resistance models against measured repose angles; the
//!   elastic-plastic spring-dashpot (EPSD) model (an accumulated
//!   relative-rotation spring with a Coulomb-like cap) is their recommended
//!   class for a static angle, unlike viscous or "directional constant"
//!   models, which are rate-only like Cosserat.
//!
//! The tangential spring is a 2D vector, reprojected onto the current
//! tangent plane every step to drop the component that drifts onto the
//! normal as the normal rotates, standard DEM practice. As a scalar along
//! "the current tangent", two free grains gained kinetic energy without
//! bound (`population::tests::two_free_grains_never_gain_kinetic_energy_
//! without_an_external_driver`). Rolling stays a scalar: 2D relative
//! rotation has no direction.

use glam::Vec2;

use super::disc_contact::{self, ContactSide, DiscContactConfig};

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

/// Linear contact parameters (Luding 2008 / Cundall & Strack 1979 / Ai et al.
/// 2011). Stiffnesses in force per length (normal/tangential) or torque per
/// angle (rolling), grid units; frictions are dimensionless coefficients, the
/// `tan(phi)` of `DruckerPragerMaterial::friction_angle`.
#[derive(Clone, Copy, Debug)]
pub struct ContactLawConfig {
    pub normal_stiffness: f32,
    pub tangential_stiffness: f32,
    pub rolling_stiffness: f32,
    /// Normal dashpot damping coefficient (Cundall & Strack 1979's own
    /// `c_n`). 0.0 = perfectly elastic normal contact (valid choice --
    /// a coefficient of restitution of 1).
    pub normal_damping: f32,
    /// Tangential dashpot damping coefficient, a channel separate from the
    /// normal one as in GeoTaichi (`ndratio`/`sdratio`). Damps the
    /// tangential relative velocity `v_t` (rotation included, see
    /// `resolve_contact_pair`). Without it the tangential/rotational
    /// subsystem is an undamped oscillator integrated explicitly, and a
    /// column collapse (`tests/grains_repose_angle.rs`) ran out to 12x the
    /// predicted spread and kept growing.
    pub tangential_damping: f32,
    /// Rolling dashpot damping coefficient, a third channel as GeoTaichi's
    /// `rdratio`. Without it the rolling spring is an undamped
    /// elastic-plastic oscillator, and many rolling contacts in a pile pump
    /// energy in through explicit integration (the 80-grain column collapse
    /// still grew, 12.1x -> 3.9x, after the rolling sign fix).
    pub rolling_damping: f32,
    /// Sliding Coulomb friction coefficient (same role as `DruckerPragerMaterial`'s
    /// `tan(friction_angle)`, just for a contact pair instead of a material point).
    pub friction: f32,
    /// Rolling friction coefficient (Ai et al. 2011's own real survey table:
    /// dimensionless, real range ~0.001-0.3 depending on grain angularity).
    pub rolling_friction: f32,
}

/// Rayleigh-type critical timestep (Luding 2008): the largest stable
/// explicit step for a linear spring-dashpot contact of reduced mass `m_eff`
/// and stiffness `k`. A localized stability constraint next to the MPM CFL,
/// like `rod_cfl_dt`.
pub fn critical_timestep(m_eff: f32, config: &ContactLawConfig) -> f32 {
    let k_max = config
        .normal_stiffness
        .max(config.tangential_stiffness)
        .max(config.rolling_stiffness);
    (m_eff / k_max).sqrt()
}

/// Conservative critical timestep for the Hertzian model
/// (`HertzianContactConfig`). GeoTaichi's `calcu_critical_timestep`
/// (`HertzMindlinModel.py`) needs density and Poisson's ratio, which these
/// grains do not carry. Instead this evaluates `critical_timestep`'s bound
/// at a worst-case overlap (`WORST_CASE_OVERLAP_FRACTION * radius`): Hertzian
/// stiffness grows with overlap, so a plausible worst-case penetration
/// overestimates it and underestimates the safe dt.
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
    /// Dry sand-like preset: stiffness and damping derived from physical
    /// inputs by the standard DEM formulas (Cundall & Strack 1979 `kn ~ E*r`,
    /// a critical-damping ratio per channel), with `m_eff_kg` explicit (a
    /// hand-written `m_eff` 8x too large once ran a documented 60% of
    /// critical damping at ~170%).
    ///
    /// `rolling_friction` is a per-material input: holding it fixed while
    /// sweeping sliding friction 25-45 degrees gave non-monotonic deviations up
    /// to ~30% (`diag_portability_across_friction_angle`), consistent with Ai
    /// et al. 2011, where it depends on grain angularity. Pick it from their
    /// surveyed range (0.001-0.3) for the material modelled.
    pub fn dry_sand(
        young_modulus_pa: f32,
        grain_radius_m: f32,
        m_eff_kg: f32,
        friction_angle_deg: f32,
        rolling_friction: f32,
    ) -> Self {
        let kn = young_modulus_pa * grain_radius_m;
        // Unlike `kn` above (Cundall & Strack 1979), the 0.8 and 0.1 ratios below
        // have no literature source: tangential and rolling stiffness below the
        // normal stiffness is physically expected (shear/rolling contact
        // compliance is softer than direct normal compression), but these two
        // ratios are an engineering choice, not a measured or cited value.
        let kt = 0.8 * kn;
        let kr = kn * grain_radius_m * grain_radius_m * 0.1;
        // 60% of critical damping per channel, on each channel's own
        // stiffness. For the normal linear spring-dashpot this is a
        // restitution coefficient e = exp(-pi*zeta/sqrt(1-zeta^2)) ~= 0.09:
        // strongly dissipative, an engineering choice, not a measured sand
        // value.
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

/// Hertzian (nonlinear) contact model: Johnson 1985 "Contact Mechanics"
/// elastic spheres, cross-checked against GeoTaichi's
/// `physics_model/contact_model/HertzMindlinModel.py`. Unlike the linear
/// spring of `ContactLawConfig` (Cundall & Strack 1979), stiffness grows
/// with the contact patch (`kn ~ sqrt(overlap)`), so normal force goes as
/// `overlap^1.5`: two smooth elastic spheres (steel, glass), not granular
/// material.
///
/// A chain of linear contacts does not sharpen a compression pulse the way
/// Hertzian chains do (Nesterenko 2001, "Dynamics of Heterogeneous
/// Materials", solitary waves in Hertzian chains), so in a Newton's cradle
/// momentum lingered at the middle balls
/// (`diag_newtons_cradle_first_collision_immediate_aftermath`). Available
/// through `GrainPopulation::new_hertzian`; `ContactLawConfig` stays the
/// choice for granular material.
///
/// Rolling resistance is the same EPSD spring (Ai et al. 2011) as
/// `resolve_contact_pair`, not GeoTaichi's memoryless Coulomb cap: the
/// static repose angle needs its memory.
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
    /// Coefficient of restitution (dimensionless: 1.0 perfectly elastic, 0.0
    /// perfectly inelastic), the physical value itself (e.g. 0.95 for steel
    /// on steel). The resolve functions convert it through
    /// `hertzian_damping_coefficient` before the Tsuji, Tanaka & Ishida 1992
    /// damping formula; the linear model instead takes damping coefficients.
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
    /// contact moment always acts as a action-reaction pair, exactly
    /// like a linear contact force, per Ai et al. 2011's own formulation).
    pub rolling_moment: f32,
    /// Torque from the tangential force acting at the contact point, offset
    /// from `i`'s centre by `i.radius` along `n` and from `j`'s by `-j.radius`
    /// (torque = r x F): how friction turns sliding into rolling. Separate
    /// from `rolling_moment`, which resists relative spin. One tangential
    /// force, two moment arms, so the two torques are returned separately.
    pub friction_torque_on_i: f32,
    pub friction_torque_on_j: f32,
}

/// The three rolling-resistance coefficients, which always travel together
/// from a contact config.
#[derive(Clone, Copy)]
struct RollingParams {
    stiffness: f32,
    damping: f32,
    friction: f32,
}

/// Rolling-resistance spring shared by every contact model (linear,
/// Hertzian, 2D disc) and geometry (grain-grain, grain-wall): the Ai et al.
/// 2011 EPSD spring, elastic trial `kr*R` capped at `mu_r*r_eff*F_n` with
/// plastic correction on the cap; only the caller's `omega_rel`, `r_eff` and
/// `normal_force` differ.
///
/// `spring.rolling` is the relative-rotation coordinate R = integral(omega_rel
/// dt) = theta_i - theta_j. `rolling_moment` is applied to j and opposite to
/// i (`torques[j] += rolling_moment; torques[i] -= rolling_moment;` in
/// `resolve_contact_forces`); for U(R) = 0.5*kr*R^2 the generalized force on
/// j is Q_j = -kr*R*(dR/dtheta_j) = +kr*R, so the moment carries a plus sign.
/// With `-kr*R` (Q_i applied at j) a grain on a tilted pinned floor spun up
/// monotonically (`diag_instrumented_single_step_breakdown`), and piles,
/// whose grains rest at off-axis angles, diverged.
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
    // Dashpot on the relative rotation, same sign as the elastic term above
    // (a positive `omega_rel` gives a positive moment), so it dissipates
    // relative-rotation energy.
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

/// The contact-pair kinematics both core resolvers consume, derived from the
/// contact geometry and relative velocity at the call site.
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

/// Shared linear-model (Cundall & Strack 1979) contact core:
/// `resolve_contact_pair` and `resolve_wall_contact` differ only in how they
/// derive `overlap`/`n`/`t`/`v_n`/`v_t`/`omega_rel`/`r_eff` (two free bodies
/// vs. one grain against a fixed wall). Returns `(normal_force,
/// tangential_force_vec, rolling_moment, ft_scalar)`; callers turn
/// `ft_scalar` into their own torques.
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

    // Tangential: elastic-plastic Coulomb spring plus dashpot (see
    // `tangential_damping`). The spring is reprojected onto the current
    // tangent plane before this step's increment (see the module doc).
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

/// Resolves one grain-grain contact pair for one substep, given `dt` and the
/// pair's persistent spring state (updated in place). Returns `None` (springs
/// reset) when the grains do not overlap: a broken contact has no memory.
///
/// Force law (see the module doc for citations):
/// - Normal: `F_n = kn*overlap - c_n*v_n`, clamped to `>= 0` (repulsive only).
/// - Tangential: elastic trial `-ks*spring`, Coulomb-capped at `mu*F_n`,
///   spring rescaled on the cap (the return mapping of material plasticity).
/// - Rolling: see `resolve_rolling_spring`.
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
    // Tangential slip velocity at the contact point -- derived
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

    // Torque of the tangential force at the contact point, torque = r x F in
    // 2D (cross(a,b) = a.x*b.y - a.y*b.x). The contact point is +i.radius*n
    // from i's centre and -j.radius*n from j's; only the tangential part
    // contributes, and cross(n, t) = 1 (t is n rotated 90 degrees), so both
    // reduce to -radius * ft_scalar.
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

/// Resolves one grain-vs-wall contact for one substep, so a grain resting on
/// the ground can start rolling (otherwise only grain-grain contact gives
/// torque). The grain-grain physics against a fixed wall (infinite mass, no
/// velocity or spin), with `r_eff = grain.radius`, the analytic limit of
/// `i.radius*j.radius/(i.radius+j.radius)` as the wall's radius goes to
/// infinity, used directly rather than approximated with a huge radius.
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

/// Physical coefficient of restitution `e` to the damping coefficient the
/// Tsuji, Tanaka & Ishida 1992 formula `-1.8257 * c * v * sqrt(k*m_eff)`
/// takes, `c = -ln(e)/sqrt(pi^2 + ln(e)^2)`, the transform GeoTaichi's
/// `HertzMindlin.py::add_surface_property` applies before using it. The raw
/// `e` (0.95) in place of `c` (~0.0163) is ~58x too much damping: a single
/// pair collision then split its momentum as for e ~0.14 against the 1D
/// rule `v0' = (1-e)/2 v0`, `v1' = (1+e)/2 v0`.
fn hertzian_damping_coefficient(restitution: f32) -> f32 {
    if restitution < 1.0e-6 {
        return 0.0;
    }
    let ln_e = restitution.ln();
    -ln_e / (std::f32::consts::PI * std::f32::consts::PI + ln_e * ln_e).sqrt()
}

/// Shared Hertzian contact core: as for the linear pair, the two callers
/// differ only in how they derive `overlap`/`n`/`t`/`v_n`/`v_t`/
/// `omega_rel`/`r_eff`/`m_eff`; the Hertz-Mindlin and Tsuji-damping
/// resolution is the same once those are known.
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
    // Hertzian stiffness grows with overlap, unlike the linear model's
    // constant `normal_stiffness`.
    let contact_area_radius = (overlap * r_eff).sqrt();
    let kn = 2.0 * config.effective_young_modulus * contact_area_radius;
    let ks = 8.0 * config.effective_shear_modulus * contact_area_radius;
    let damping_coeff = hertzian_damping_coefficient(config.restitution);

    // Hertzian normal force with Tsuji, Tanaka & Ishida 1992 nonlinear
    // damping (1.8257 is that paper's constant; `damping_coeff` is the
    // transform of the restitution, see `hertzian_damping_coefficient`).
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
    // `HertzianContactConfig`'s doc for why this stays unchanged.
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

/// Hertzian (nonlinear) counterpart of `resolve_contact_pair` (see
/// `HertzianContactConfig`): the same spring state, elastic-plastic
/// tangential and rolling structure and tangent-plane reprojection; only the
/// normal and tangential stiffness and damping follow Hertz-Mindlin
/// (GeoTaichi's `HertzMindlinModel.py`, with this engine's `overlap > 0`
/// sign instead of its `gapn < 0`). The tangential damping is part of the
/// trial force before the Coulomb check, as in `resolve_contact_pair`, where
/// GeoTaichi adds it only in the sub-yield branch.
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
/// `HertzianContactConfig`'s and `resolve_contact_pair_hertzian`'s doc
/// for why this exists and its citations. Wall = infinite effective
/// mass would make `m_eff` blow up in the Hertzian damping formulas
/// (`sqrt(kn * m_eff)` diverging), so this uses `grain.mass` directly as
/// `m_eff` -- the analytic limit of the two-body reduced mass
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

/// The 2D grain contract's contact core (`DiscContactConfig`): the line
/// contact's force and tangent stiffness `k` at this overlap, a dashpot
/// `2 zeta sqrt(k m)` on the normal and tangential channels, a tangential
/// spring of stiffness `k` times `tangential_ratio` capped by Coulomb, and
/// the shared rolling spring. Same outputs as the other cores.
fn resolve_contact_core_disc(
    kin: ContactKinematics,
    sides: (ContactSide, ContactSide),
    m_eff: f32,
    spring: &mut ContactSpring,
    config: &DiscContactConfig,
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
    let (load, kn) = disc_contact::force_and_stiffness(overlap, sides.0, sides.1);
    let zeta = config.damping_ratio;
    let normal_force = (load - 2.0 * zeta * (kn * m_eff).sqrt() * v_n).max(0.0);

    let ks = kn * config.tangential_ratio();
    spring.tangential -= n * spring.tangential.dot(n);
    spring.tangential += v_t * t * dt;
    let trial_ft_vec = -ks * spring.tangential - 2.0 * zeta * (ks * m_eff).sqrt() * v_t * t;
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

/// The 2D grain contract's counterpart to `resolve_contact_pair`: two discs
/// of the population's material, each compressing by its own term of the
/// line contact (see `disc_contact`).
pub fn resolve_contact_pair_disc(
    i: &GrainContactState,
    j: &GrainContactState,
    spring: &mut ContactSpring,
    config: &DiscContactConfig,
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
    let side = |radius: f32| ContactSide::Disc {
        radius,
        elastic: config.elastic,
    };

    let (normal_force, tangential_force_vec, rolling_moment, ft_scalar) = resolve_contact_core_disc(
        ContactKinematics {
            overlap,
            n,
            t,
            v_n,
            v_t,
            omega_rel,
            r_eff,
        },
        (side(i.radius), side(j.radius)),
        m_eff,
        spring,
        config,
        dt,
    );

    Some(ContactResolution {
        normal_force,
        tangential_force: tangential_force_vec,
        rolling_moment,
        friction_torque_on_i: -i.radius * ft_scalar,
        friction_torque_on_j: -j.radius * ft_scalar,
    })
}

/// The 2D grain contract's counterpart to `resolve_wall_contact`: a disc
/// against a rigid flat wall, which compresses by nothing (see
/// `disc_contact`). Wall mass and radius at their infinite limits, as in
/// `resolve_wall_contact_hertzian`.
pub fn resolve_wall_contact_disc(
    grain: &GrainContactState,
    normal: Vec2,
    overlap: f32,
    spring: &mut ContactSpring,
    config: &DiscContactConfig,
    dt: f32,
) -> Option<ContactResolution> {
    if overlap <= 0.0 {
        *spring = ContactSpring::default();
        return None;
    }
    let n = normal;
    let t = Vec2::new(-n.y, n.x);
    let v_n = grain.v.dot(n);
    let v_t = grain.v.dot(t) - grain.radius * grain.spin;

    let (normal_force, tangential_force_vec, rolling_moment, ft_scalar) = resolve_contact_core_disc(
        ContactKinematics {
            overlap,
            n,
            t,
            v_n,
            v_t,
            omega_rel: -grain.spin,
            r_eff: grain.radius,
        },
        (
            ContactSide::Disc {
                radius: grain.radius,
                elastic: config.elastic,
            },
            ContactSide::RigidWall,
        ),
        grain.mass,
        spring,
        config,
        dt,
    );

    Some(ContactResolution {
        normal_force,
        tangential_force: tangential_force_vec,
        rolling_moment,
        friction_torque_on_i: 0.0,
        friction_torque_on_j: -grain.radius * ft_scalar,
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
        // to relative SPIN directly, a separate channel from
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
        // already ruled out: once an elastic rolling spring has wound
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
