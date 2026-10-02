//! Normal contact of elastic discs in 2D: the force per unit depth between
//! two long cylinders pressed along their length (plane strain), or a
//! cylinder and a rigid wall.
//!
//! A 2D grain is a disc of unit depth, so its contact is a line contact, not
//! the point contact of two spheres: the compression grows with the load
//! times a logarithm, not as `load^(2/3)`, and a sphere's `E r` stiffness
//! does not apply.
//!
//! # The law
//!
//! Gerl and Zippelius, "Coefficient of restitution for elastic disks",
//! arXiv:cond-mat/9808258, eq. 55: a disc of radius `R` loaded by a contact
//! force `P` per unit depth, balanced by a body force (its weight, or its
//! own inertia when it is being stopped), compresses by
//!
//! ```text
//! delta = P / (pi E) * (2 ln(4R/a) - 1 - nu)
//! ```
//!
//! where `a` is the Hertz half-width of the contact, `a^2 = 4 P R* / (pi E*)`
//! (eq. 56 is this with `a` substituted, eq. 58 its small-load inversion
//! `P ~ pi E delta / ln(4R/delta)`). Their `E`, `nu` are those of 2D plane
//! stress (`eps_xx = (sigma_xx - nu sigma_yy) / E`, their eq. 54); a disc of
//! unit depth in this plane-strain engine takes `E' = E / (1 - nu^2)` and
//! `nu' = nu / (1 - nu)`.
//!
//! Norden, "On the compression of a cylinder in contact with a plane
//! surface", NBSIR 73-243 (NBS, 1973), eqs. 70 to 78, gives the same
//! logarithmic compression per contact, `2 (P/L) (1 - nu^2)/(pi E)
//! [C + ln(2R/b)]` with `b` the same half-width: the same coefficient of the
//! logarithm, with a constant `C` (0.33 to 0.41 depending on the author)
//! that belongs to his loading, a cylinder squeezed between two flat
//! anvils, not to a grain stopped by its own inertia.
//!
//! Two bodies in contact each compress by their own term, with the common
//! half-width from `1/R* = 1/R1 + 1/R2` and `1/E* = 1/E1' + 1/E2'`; a rigid
//! wall compresses by nothing and has an infinite radius. Two equal discs
//! share the half-width of one disc on a wall, so they approach by twice as
//! much under the same load.
//!
//! The theory holds while `a` is small next to `R`. Past the load where the
//! compression's slope has fallen to `sum_i 1 / (pi E_i')` (on a wall, `a`
//! about `0.8 R` and an overlap about `0.3 R`), the force is continued
//! linearly at that stiffness: a declared continuation that keeps the force
//! finite and monotonic, far outside the theory.

/// The elastic constants of one body in a contact, in the engine's own
/// stress units (see `SimConfig::stress_from_si`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DiscElastic {
    /// Young's modulus.
    pub young_modulus: f32,
    /// Poisson's ratio.
    pub poisson_ratio: f32,
}

impl DiscElastic {
    /// A material's constants in the engine's units: `E / (rho_ref dx^2)`,
    /// the conversion every material's stress takes in a scene whose masses
    /// come from `SpawnRegion::mass_from` (see
    /// `SimConfig::reference_density_kg_m3`), so that a grain's force and
    /// its mass from `Grain::from_si` agree.
    pub fn from_si(elastic: &crate::Elastic, config: &crate::SimConfig) -> Self {
        Self {
            young_modulus: config.stress_from_si(elastic.e_pa, config.reference_density_kg_m3),
            poisson_ratio: elastic.nu,
        }
    }

    /// Plane-strain modulus `E / (1 - nu^2)`.
    pub fn plane_strain_modulus(&self) -> f32 {
        self.young_modulus / (1.0 - self.poisson_ratio * self.poisson_ratio)
    }

    /// Plane-strain Poisson's ratio `nu / (1 - nu)`.
    pub fn plane_strain_poisson(&self) -> f32 {
        self.poisson_ratio / (1.0 - self.poisson_ratio)
    }
}

/// One side of a contact: a disc of `radius` made of `elastic`, or a rigid
/// flat wall.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ContactSide {
    Disc { radius: f32, elastic: DiscElastic },
    RigidWall,
}

/// The two sides' combined constants, computed once per contact: the
/// compression is `delta(P) = P (t - s ln P)`, with `s` and `t` below.
#[derive(Clone, Copy, Debug)]
struct Compliance {
    /// `sum_i 1 / (pi E_i')` over the deformable sides.
    s: f32,
    /// `sum_i (ln(16 R_i^2 / c) - 1 - nu_i') / (pi E_i')`, where
    /// `a^2 = c P`.
    t: f32,
}

impl Compliance {
    fn new(first: ContactSide, second: ContactSide) -> Option<Self> {
        let discs = [first, second].into_iter().filter_map(|side| match side {
            ContactSide::Disc { radius, elastic } => Some((radius, elastic)),
            ContactSide::RigidWall => None,
        });
        // 1/R* and 1/E* over the discs; a rigid wall adds nothing to either.
        let (mut inv_r, mut inv_e) = (0.0f32, 0.0f32);
        for (radius, elastic) in discs.clone() {
            inv_r += 1.0 / radius;
            inv_e += 1.0 / elastic.plane_strain_modulus();
        }
        if inv_r <= 0.0 || inv_e <= 0.0 {
            return None;
        }
        // a^2 = 4 P R* / (pi E*) = c P.
        let c = 4.0 * inv_e / (std::f32::consts::PI * inv_r);
        let (mut s, mut t) = (0.0f32, 0.0f32);
        for (radius, elastic) in discs {
            let lambda = 1.0 / (std::f32::consts::PI * elastic.plane_strain_modulus());
            s += lambda;
            t +=
                lambda * ((16.0 * radius * radius / c).ln() - 1.0 - elastic.plane_strain_poisson());
        }
        Some(Self { s, t })
    }

    /// The compression under `load`, eq. 55 summed over the sides.
    fn delta(&self, load: f32) -> f32 {
        load * (self.t - self.s * load.ln())
    }

    /// `d delta / d P`.
    fn slope(&self, load: f32) -> f32 {
        self.t - self.s * load.ln() - self.s
    }

    /// The load past which the force is continued linearly: where the
    /// slope has fallen to `s` (see the module doc).
    fn load_limit(&self) -> f32 {
        (self.t / self.s - 2.0).exp()
    }
}

/// The approach of two bodies under a contact force `load` per unit depth:
/// Gerl and Zippelius eq. 55, each side's term summed (see the module doc).
/// 0 for no load or no deformable side.
pub fn approach(load: f32, first: ContactSide, second: ContactSide) -> f32 {
    match Compliance::new(first, second) {
        Some(compliance) if load > 0.0 => {
            let limit = compliance.load_limit();
            if load <= limit {
                compliance.delta(load)
            } else {
                compliance.delta(limit) + (load - limit) * compliance.s
            }
        }
        _ => 0.0,
    }
}

/// The contact force per unit depth that makes two bodies approach by
/// `overlap`, and its stiffness `dP / d overlap`: `approach` inverted by
/// Newton's method from eq. 58's small-load estimate. `(0, 0)` without
/// overlap or without a deformable side.
pub fn force_and_stiffness(overlap: f32, first: ContactSide, second: ContactSide) -> (f32, f32) {
    let Some(compliance) = Compliance::new(first, second) else {
        return (0.0, 0.0);
    };
    if overlap <= 0.0 {
        return (0.0, 0.0);
    }
    let limit = compliance.load_limit();
    let delta_limit = compliance.delta(limit);
    if overlap >= delta_limit {
        return (
            limit + (overlap - delta_limit) / compliance.s,
            1.0 / compliance.s,
        );
    }
    // Eq. 58's estimate, one fixed-point step from `P = overlap / s`, then
    // Newton on `delta(P) = overlap`, kept inside `(0, limit]` where
    // `delta` rises.
    let mut load = overlap / (compliance.t - compliance.s * (overlap / compliance.s).ln());
    if !(load > 0.0 && load <= limit) {
        load = 0.5 * limit;
    }
    for _ in 0..8 {
        let step = (compliance.delta(load) - overlap) / compliance.slope(load);
        let next = (load - step).clamp(0.5 * load, limit);
        let converged = (next - load).abs() <= 1.0e-7 * load;
        load = next;
        if converged {
            break;
        }
    }
    (load, 1.0 / compliance.slope(load))
}

/// The 2D grain contract's contact between grains, and between a grain and
/// a wall: `force_and_stiffness`'s line contact in the normal direction, a
/// dashpot on its tangent stiffness, Coulomb friction on a tangential
/// spring, and the rolling-resistance spring every grain contact model
/// shares (Ai et al. 2011).
#[derive(Clone, Copy, Debug)]
pub struct DiscContactConfig {
    /// The grains' elastic constants (every grain of a population is one
    /// material, as in the other contact models).
    pub elastic: DiscElastic,
    /// Damping ratio `zeta = c / (2 sqrt(k m))` of the normal and tangential
    /// dashpots, on the contact's tangent stiffness `k` and reduced mass `m`:
    /// set by `new` from the physical restitution (see
    /// `damping_ratio_for_restitution`).
    pub damping_ratio: f32,
    /// Sliding Coulomb friction coefficient.
    pub friction: f32,
    /// Rolling-resistance spring, as `HertzianContactConfig`'s own: an
    /// independent material property, in the engine's units.
    pub rolling_stiffness: f32,
    pub rolling_damping: f32,
    pub rolling_friction: f32,
}

/// The coefficient of restitution of a linear spring-dashpot of damping
/// ratio `zeta` whose force is kept repulsive (`max(0, F)`), so that the
/// collision ends when the force vanishes, before the overlap does:
/// Schwager and Poschel, "Coefficient of restitution and linear-dashpot
/// model revisited", Granular Matter 9 (2007) 465-469, eq. 21, with
/// `beta / omega_0 = zeta`. The first two branches of eq. 21 are one
/// `atan2`. The common `exp(-pi zeta / sqrt(1 - zeta^2))` is their eq. 15,
/// which lets the force turn attractive at the end of the contact.
pub fn restitution_for_damping_ratio(zeta: f64) -> f64 {
    if zeta <= 0.0 {
        return 1.0;
    }
    if zeta < 1.0 {
        let root = (1.0 - zeta * zeta).sqrt();
        let omega_tc = std::f64::consts::PI - (2.0 * zeta * root).atan2(1.0 - 2.0 * zeta * zeta);
        (-zeta / root * omega_tc).exp()
    } else if zeta > 1.0 {
        let big_omega = (zeta * zeta - 1.0).sqrt();
        (-zeta / big_omega * ((zeta + big_omega) / (zeta - big_omega)).ln()).exp()
    } else {
        // Both branches' limit at critical damping: `omega t_c` tends to
        // `2 sqrt(1 - zeta^2)`, so the exponent to `-2`.
        (-2.0f64).exp()
    }
}

/// The damping ratio whose restitution (`restitution_for_damping_ratio`) is
/// `restitution`: that relation inverted by bisection, as it falls
/// monotonically from 1 at `zeta = 0`. Exact for a linear spring; the line
/// contact is linear up to its logarithm, so for it a declared
/// approximation, measured within a few percent
/// (`disc_contract_restitution_table`). Not Tsuji's 1.8257 constant, which
/// belongs to the Hertz sphere's `3/2` power.
pub fn damping_ratio_for_restitution(restitution: f32) -> f32 {
    let target = f64::from(restitution.clamp(0.0, 1.0));
    if target >= 1.0 {
        return 0.0;
    }
    let (mut low, mut high) = (0.0f64, 1.0f64);
    while restitution_for_damping_ratio(high) > target && high < 1.0e6 {
        high *= 2.0;
    }
    for _ in 0..100 {
        let mid = 0.5 * (low + high);
        if restitution_for_damping_ratio(mid) > target {
            low = mid;
        } else {
            high = mid;
        }
    }
    (0.5 * (low + high)) as f32
}

impl DiscContactConfig {
    /// `elastic` grains that part with `restitution` of their approach
    /// speed (the physical value, 1 elastic, 0 fully inelastic), with
    /// `friction` and the rolling spring.
    pub fn new(
        elastic: DiscElastic,
        restitution: f32,
        friction: f32,
        rolling_stiffness: f32,
        rolling_damping: f32,
        rolling_friction: f32,
    ) -> Self {
        Self {
            elastic,
            damping_ratio: damping_ratio_for_restitution(restitution),
            friction,
            rolling_stiffness,
            rolling_damping,
            rolling_friction,
        }
    }

    /// Tangential over normal contact stiffness, `2 (1 - nu) / (2 - nu)`.
    ///
    /// What is verified: this Poisson-ratio-only ratio of Mindlin-type
    /// contact theory holds for a CIRCULAR (3D point) contact at zero
    /// traction; the reviewer derived it from lecture notes citing Mindlin
    /// 1949 and Mindlin and Deresiewicz 1953 (normal stiffness `2 a E*`,
    /// tangential `8 a G*`). It is carried over to this LINE contact as a
    /// disclosed approximation, not confirmed against a line-contact
    /// source. Read Mindlin and Deresiewicz 1953, or Johnson's Contact
    /// Mechanics sections 3.4 and 3.5, before trusting it anywhere friction
    /// decides the result (the sand angle of repose, #28).
    pub fn tangential_ratio(&self) -> f32 {
        let nu = self.elastic.poisson_ratio;
        2.0 * (1.0 - nu) / (2.0 - nu)
    }
}

/// The largest stable step of one contact of tangent stiffness `stiffness`
/// between bodies of reduced mass `reduced_mass`, damped at `damping_ratio`:
/// `omega dt <= 2 (sqrt(1 + zeta^2) - zeta)`, the explicit (symplectic
/// Euler) limit of a damped oscillator, `omega = sqrt(k / m)`.
pub fn critical_step(stiffness: f32, reduced_mass: f32, damping_ratio: f32) -> f32 {
    if stiffness <= 0.0 || reduced_mass <= 0.0 {
        return f32::INFINITY;
    }
    let omega = (stiffness / reduced_mass).sqrt();
    2.0 * ((1.0 + damping_ratio * damping_ratio).sqrt() - damping_ratio) / omega
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUARTZ_LIKE: DiscElastic = DiscElastic {
        young_modulus: 1.0e6,
        poisson_ratio: 0.2,
    };

    fn disc(radius: f32) -> ContactSide {
        ContactSide::Disc {
            radius,
            elastic: QUARTZ_LIKE,
        }
    }

    /// Eq. 56 is eq. 55 with the half-width substituted: on a rigid wall,
    /// `delta = P / (pi E') (ln(4 R pi E' / P) - 1 - nu')`.
    #[test]
    fn a_disc_on_a_wall_follows_gerl_and_zippelius_eq_56() {
        let radius = 2.0f32;
        let e = QUARTZ_LIKE.plane_strain_modulus();
        let nu = QUARTZ_LIKE.plane_strain_poisson();
        for load in [1.0e-2f32, 1.0, 1.0e2] {
            let eq56 = load / (std::f32::consts::PI * e)
                * ((4.0 * radius * std::f32::consts::PI * e / load).ln() - 1.0 - nu);
            let delta = approach(load, disc(radius), ContactSide::RigidWall);
            assert!(
                ((delta - eq56) / eq56).abs() < 1.0e-5,
                "P {load}: {delta} against eq. 56 {eq56}"
            );
        }
    }

    /// Two equal discs share one disc-on-wall's half-width, so they approach
    /// by twice as much under the same load.
    #[test]
    fn two_equal_discs_approach_twice_a_disc_on_a_wall() {
        for load in [1.0e-2f32, 1.0, 1.0e2] {
            let pair = approach(load, disc(1.5), disc(1.5));
            let wall = approach(load, disc(1.5), ContactSide::RigidWall);
            assert!(
                ((pair - 2.0 * wall) / pair).abs() < 1.0e-5,
                "P {load}: {pair} against twice {wall}"
            );
        }
    }

    /// Norden's line contact: per contact, `2 P lambda [C + ln(2R/b)]` with
    /// `lambda = (1 - nu^2)/(pi E)` and `b^2 = 4 R lambda P` on a rigid
    /// plane. The same law has the same logarithm's coefficient: over a
    /// thousandfold range of loads, `delta / (2 P lambda) - ln(2R/b)` stays
    /// one constant (ours is `ln 2 - (1 + nu')/2`, where Norden's authors
    /// have 0.33 to 0.41 for anvils).
    #[test]
    fn the_logarithm_matches_nordens_line_contact() {
        let radius = 2.0f32;
        let lambda = 1.0 / (std::f32::consts::PI * QUARTZ_LIKE.plane_strain_modulus());
        let constant = |load: f32| {
            let b = (4.0 * radius * lambda * load).sqrt();
            let delta = approach(load, disc(radius), ContactSide::RigidWall);
            delta / (2.0 * load * lambda) - (2.0 * radius / b).ln()
        };
        let expected = 2.0f32.ln() - 0.5 * (1.0 + QUARTZ_LIKE.plane_strain_poisson());
        for load in [1.0e-2f32, 1.0e-1, 1.0, 1.0e1] {
            assert!(
                (constant(load) - expected).abs() < 1.0e-4,
                "P {load}: constant {} against {expected}",
                constant(load)
            );
        }
    }

    /// The force inverts the approach, and the stiffness is its slope.
    #[test]
    fn the_force_inverts_the_approach() {
        for overlap in [1.0e-7f32, 1.0e-5, 1.0e-3, 1.0e-1] {
            for (first, second) in [
                (disc(1.0), disc(1.0)),
                (disc(0.5), disc(2.0)),
                (disc(1.0), ContactSide::RigidWall),
            ] {
                let (load, stiffness) = force_and_stiffness(overlap, first, second);
                let back = approach(load, first, second);
                assert!(
                    ((back - overlap) / overlap).abs() < 1.0e-4,
                    "overlap {overlap}: force {load} approaches by {back}"
                );
                let h = 1.0e-3 * load;
                let slope = (approach(load + h, first, second) - approach(load - h, first, second))
                    / (2.0 * h);
                assert!(
                    ((stiffness * slope) - 1.0).abs() < 2.0e-3,
                    "overlap {overlap}: stiffness {stiffness} against 1/{slope}"
                );
            }
        }
    }

    /// Eq. 58: at a vanishing overlap the force tends to
    /// `pi E' delta / ln(4R/delta)` on a wall, slowly (its correction is of
    /// order `ln ln / ln^2`): nearer at a smaller overlap.
    #[test]
    fn a_small_overlap_tends_to_eq_58() {
        let radius = 1.0f32;
        let e = QUARTZ_LIKE.plane_strain_modulus();
        let ratio = |overlap: f32| {
            let (load, _) = force_and_stiffness(overlap, disc(radius), ContactSide::RigidWall);
            load / (std::f32::consts::PI * e * overlap / (4.0 * radius / overlap).ln())
        };
        let (coarse, fine) = (ratio(1.0e-3), ratio(1.0e-7));
        assert!(
            (fine - 1.0).abs() < (coarse - 1.0).abs(),
            "ratio to eq. 58: {coarse} at 1e-3, {fine} at 1e-7"
        );
    }

    /// Quartz (2650 kg/m3) at 1 mm cells against water's reference density:
    /// a grain weighs what MPM particles of its area weigh, and its modulus
    /// takes the stress conversion of a `mass_from` scene.
    #[test]
    fn the_si_contract_shares_the_particles_units() {
        let quartz = crate::Elastic {
            e_pa: 95.6e9,
            nu: 0.08,
            rho_kg_m3: 2650.0,
        };
        let config = crate::SimConfig::earth(64, 1.0e-3, 1.0e-3);
        let grain = crate::matter::particle::Grain::from_si(
            glam::Vec2::splat(32.0),
            2.0e-3,
            &quartz,
            &config,
        );
        assert!(
            (grain.radius - 2.0).abs() < 1.0e-6,
            "radius {} cells",
            grain.radius
        );
        let spacing = 0.5f32;
        let particle_mass = crate::SpawnRegion {
            spacing,
            ..crate::SpawnRegion::for_sim(&config)
        }
        .mass_from(&quartz, &config)
        .mass_override
        .expect("mass_from sets the mass");
        let same_area = particle_mass * std::f32::consts::PI * 4.0 / (spacing * spacing);
        assert!(
            ((grain.mass - same_area) / same_area).abs() < 1.0e-5,
            "grain {} against particles of its area {same_area}",
            grain.mass
        );
        let elastic = DiscElastic::from_si(&quartz, &config);
        let expected = 95.6e9 / (config.reference_density_kg_m3 * 1.0e-6);
        assert!(((elastic.young_modulus - expected) / expected).abs() < 1.0e-5);
    }

    /// Schwager and Poschel eq. 21 at its limits: no damping keeps all the
    /// speed; their eq. 22 gives `1 / (4 zeta^2)` at heavy damping; and it is
    /// continuous through critical damping. Inverting it gives back the ratio.
    #[test]
    fn restitution_follows_schwager_and_poschel() {
        assert_eq!(restitution_for_damping_ratio(0.0), 1.0);
        let heavy = restitution_for_damping_ratio(100.0);
        assert!(((heavy * 4.0e4) - 1.0).abs() < 1.0e-3, "zeta 100: {heavy}");
        let (below, above) = (
            restitution_for_damping_ratio(1.0 - 1.0e-6),
            restitution_for_damping_ratio(1.0 + 1.0e-6),
        );
        assert!((below - above).abs() < 1.0e-3, "{below} {above}");
        for e in [0.05f32, 0.1, 0.5, 0.9, 0.99] {
            let zeta = damping_ratio_for_restitution(e);
            let back = restitution_for_damping_ratio(f64::from(zeta)) as f32;
            assert!((back - e).abs() < 1.0e-5, "e {e}: zeta {zeta} gives {back}");
        }
    }

    /// Past the theory's range the force stays finite and keeps rising.
    #[test]
    fn a_deep_overlap_is_continued_linearly() {
        let (a, ka) = force_and_stiffness(0.5, disc(1.0), ContactSide::RigidWall);
        let (b, kb) = force_and_stiffness(0.6, disc(1.0), ContactSide::RigidWall);
        assert!(a.is_finite() && b > a && ka == kb, "{a} {b} {ka} {kb}");
    }
}
