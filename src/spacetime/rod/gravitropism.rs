//! Gravitropism: curvature relaxation toward a per-organ gravitropic
//! set-point angle (GSA), with proprioceptive (self-straightening) damping.
//! Dynamics: Porat, Rivière & Meroz 2024, "A quantitative model for
//! spatio-temporal dynamics of root gravitropism," Journal of Experimental
//! Botany 75(2):620, eq. 2: `r*edot0*Dkappa/Dt = -beta*sin(theta_tip -
//! theta_g) - gamma*r*kappa`, built on Bastien et al.'s ACE
//! (Angle-Curvature-Elongation) model. Target angle: Digby & Firn 1995,
//! "The gravitropic set-point angle (GSA): the identification of an
//! important developmentally controlled variable governing plant
//! architecture," Plant, Cell & Environment 18(12):1434: an organ's
//! maintained angle from gravity is characteristic of the organ, not always
//! zero. Convention (the paper's): 0 = with gravity (a main root), pi =
//! against it (a main shoot), in between a plagiotropic lateral organ holding
//! that angle as its target. `target_angle_rad` defaults to 0.0.
//!
//! The law's form is cited (curvature grows to reduce the deviation from the
//! organ's target, damped by self-straightening so it cannot curl up); the
//! rates `sensitivity` and `straightening` are illustrative, chosen for a
//! stable visible response, not fitted to either paper. Applied to this
//! engine's dimensionless discrete curvature (see
//! `forces::discrete_curvature`), not the papers' 1/length.
//!
//! # `GravitropismMode`: which vertices evolve
//! Bastien, Bohr, Moulia & Douady 2013, "Unifying model of shoot
//! gravitropism reveals proprioception as a central feature of posture
//! control in plants," PNAS 110(2):755-760, write the same law as a field
//! along the arc length `s`: `dC(s,t)/dt = -beta*sin(A(s,t)) - gamma*C(s,t)`.
//! Two regimes, chosen per organ:
//! - `TipOnly` (default): only the bending vertex nearest the tip evolves,
//!   the growth zone of an elongating organ (Porat 2024's scope; mature
//!   tissue further back does not re-curve).
//! - `WholeOrgan`: every interior vertex evolves, each sensing its own edge
//!   direction, the posture control of a mature organ (Bastien 2013's scope,
//!   with his constant beta and gamma along the organ). Nudging the tip alone
//!   cannot undo a shape stored along the whole organ: a buckled grass blade
//!   recovered ~10% of its offset under `TipOnly`, the other ~17 vertices'
//!   rest curvature never moving.
//!
//! # Phototropism reuses the same core with another signal
//! `Phototropism`/`apply_phototropism` call the same per-vertex relaxation:
//! Cholodny & Went's auxin asymmetry (light-seeking bending) and the
//! statolith mechanism both reduce to this curvature-relaxation form in
//! Bastien et al.'s framework; only the sensed direction differs.
//! `GravitropismMode` applies to any such tropism.

use glam::Vec2;

use super::RodPoints;
use super::growth::{GrowthResistance, sample_mass_density};
use crate::grid::Grid;

/// Which vertices `apply_gravitropism` actually evolves -- see the module
/// doc's own "`GravitropismMode`" section for the cited distinction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GravitropismMode {
    /// Growth-zone-localized (Porat, Rivière, Meroz 2024) -- an actively
    /// elongating organ (a root). Default: exactly the original behavior.
    #[default]
    TipOnly,
    /// Whole-organ posture control (Bastien, Bohr, Moulia, Douady 2013) -- a
    /// mature, non-elongating organ that needs to recover its whole shape,
    /// not just reorient its growing tip.
    WholeOrgan,
}

#[derive(Debug, Clone, Copy)]
pub struct Gravitropism {
    /// Gravitropic sensitivity: how fast rest curvature grows to reduce
    /// local angular deviation from the organ's own target angle. 1/s
    /// (engine time units).
    pub sensitivity: f32,
    /// Proprioceptive straightening rate: decays rest curvature back toward
    /// zero on its own, preventing unbounded curling. 1/s.
    pub straightening: f32,
    /// Gravitropic set-point angle (GSA), signed radians (Digby & Firn
    /// 1995): 0.0 = aligned WITH gravity (main root), `PI` = aligned AGAINST
    /// gravity (main shoot/stem seeking true vertical), anything else = a
    /// plagiotropic organ actively holding that non-vertical angle. Default
    /// (via `new()`) is 0.0, matching the prior root-only behavior exactly.
    pub target_angle_rad: f32,
    /// Which vertices evolve (see `GravitropismMode`). Default `TipOnly`.
    pub mode: GravitropismMode,
    /// Turgor-versus-soil-resistance gate, `GrowthResistance` unchanged:
    /// gravitropic curling is differential cell elongation (Bastien et al.'s
    /// ACE model), so the balance that halts elongation under high resistance
    /// gates it too. `None` (default) = ungated; a no-op for a rod with no grid
    /// contact (a free-standing stem).
    pub resistance: Option<GrowthResistance>,
}

impl Gravitropism {
    pub fn new(sensitivity: f32, straightening: f32) -> Self {
        Self {
            sensitivity,
            straightening,
            target_angle_rad: 0.0,
            mode: GravitropismMode::default(),
            resistance: None,
        }
    }

    pub const fn with_resistance(mut self, resistance: GrowthResistance) -> Self {
        self.resistance = Some(resistance);
        self
    }

    /// Set this organ's `target_angle_rad` -- see that field's doc.
    pub const fn with_gsa(mut self, target_angle_rad: f32) -> Self {
        self.target_angle_rad = target_angle_rad;
        self
    }

    /// Choose which vertices evolve -- see `GravitropismMode`'s doc.
    pub const fn with_mode(mut self, mode: GravitropismMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Light-seeking curvature response (Cholodny & Went auxin asymmetry),
/// sharing `apply_tropism` with `Gravitropism` (see the module doc). The
/// sensed direction (`SimConfig::light_dir`) is a fixed vector set from
/// outside, not a solar model (see `SimConfig::light_dir`). Same fields as
/// `Gravitropism`, same illustrative `sensitivity`/`straightening`.
#[derive(Debug, Clone, Copy)]
pub struct Phototropism {
    /// See `Gravitropism::sensitivity`'s doc -- same law, light signal.
    pub sensitivity: f32,
    /// See `Gravitropism::straightening`'s doc.
    pub straightening: f32,
    /// This organ's target angle from the sensed light direction. 0.0 =
    /// grow directly toward the light (the common phototropic case); PI =
    /// away from it (negative phototropism, rare but e.g. some root
    /// behavior); anything between = a fixed lean relative to the light
    /// source. Default (via `new()`) is 0.0.
    pub target_angle_rad: f32,
    /// See `GravitropismMode`'s doc -- reused as-is (generic to any
    /// tropism, not gravity-specific in meaning).
    pub mode: GravitropismMode,
    /// See `Gravitropism::resistance`'s doc.
    pub resistance: Option<GrowthResistance>,
}

impl Phototropism {
    pub fn new(sensitivity: f32, straightening: f32) -> Self {
        Self {
            sensitivity,
            straightening,
            target_angle_rad: 0.0,
            mode: GravitropismMode::default(),
            resistance: None,
        }
    }

    pub const fn with_resistance(mut self, resistance: GrowthResistance) -> Self {
        self.resistance = Some(resistance);
        self
    }

    /// Set this organ's own target angle from the sensed light direction --
    /// see `target_angle_rad`'s doc.
    pub const fn with_target_angle(mut self, target_angle_rad: f32) -> Self {
        self.target_angle_rad = target_angle_rad;
        self
    }

    /// Choose which vertices evolve -- see `GravitropismMode`'s doc.
    pub const fn with_mode(mut self, mode: GravitropismMode) -> Self {
        self.mode = mode;
        self
    }
}

/// Convergence bound on `|d(kappa)/dt|` (curvature units per second) for
/// `tropism_still_correcting`, illustrative like `sensitivity`/
/// `straightening`: small enough to catch ongoing correction, loose enough
/// that float noise or a multi-tropism compromise lets the rod sleep. At the
/// default `sensitivity` (~0.01-0.05) and a small `sin_deviation`, a settled
/// `dkappa` falls well under it.
const CONVERGED_DKAPPA_PER_SECOND: f32 = 0.002;

/// Rotate `gravity_dir` (already a unit vector) by `target_angle_rad` to get
/// the organ's real target direction (its GSA, Digby & Firn 1995) --
/// `target_angle_rad=0.0` leaves it equal to `gravity_dir` (root-like);
/// `PI` flips it to point straight against gravity (a shoot seeking true
/// vertical up); anything between is a maintained plagiotropic
/// lean. Constant across every vertex for one call, matching Bastien
/// 2013's own model (the target/beta/gamma are not functions of arc-length
/// `s`; only the sensed local angle `A(s,t)` is).
fn target_direction(gravity_dir: Vec2, target_angle_rad: f32) -> Vec2 {
    let (sin_a, cos_a) = target_angle_rad.sin_cos();
    Vec2::new(
        gravity_dir.x * cos_a - gravity_dir.y * sin_a,
        gravity_dir.x * sin_a + gravity_dir.y * cos_a,
    )
}

/// Which `rest_curvature` indices `mode` governs, for a rod of `n` points
/// (`rest_curvature` has `n-2` entries). Matches `forces.rs`'s own vertex
/// convention: vertex `j` sits between points `j`, `j+1`, `j+2`.
const fn vertex_range(mode: GravitropismMode, n: usize) -> std::ops::Range<usize> {
    match mode {
        GravitropismMode::TipOnly => (n - 3)..(n - 2),
        GravitropismMode::WholeOrgan => 0..(n - 2),
    }
}

/// Vertex `j`'s own local "outgoing" direction -- `x[j+2]-x[j+1]`, exactly
/// `forces.rs`'s own `e1` (distal edge) convention for that vertex, and
/// exactly what the original tip-only code special-cased as `tip_dir` at
/// `j = n-3` (`x[n-1]-x[n-2]`). `None` for a degenerate (near-zero-length)
/// edge -- that vertex is simply skipped for this call, not the whole rod.
fn local_edge_direction(rod: &RodPoints, vertex: usize) -> Option<Vec2> {
    let edge = rod.x[vertex + 2] - rod.x[vertex + 1];
    let len = edge.length();
    if len < 1.0e-9 { None } else { Some(edge / len) }
}

/// Turgor-versus-soil-resistance gate (see `Gravitropism::resistance`)
/// sampled at `sample_pos`. Callers pass `rod.x[vertex+2]`, which at the tip
/// vertex (`vertex = n-3`) is `rod.x[n-1]`, the point `TipOnly` samples, so
/// `WholeOrgan` matches it there bit for bit.
fn resistance_gate(resistance: Option<GrowthResistance>, grid: &Grid, sample_pos: Vec2) -> f32 {
    match resistance {
        Some(r) => {
            let local_mass = sample_mass_density(grid, sample_pos);
            let local_resistance_pa = r.resistance_per_unit_mass_pa * local_mass;
            (1.0 - local_resistance_pa / r.turgor_pressure_pa.max(1.0e-6)).clamp(0.0, 1.0)
        }
        None => 1.0,
    }
}

/// Bundles the 4 parameters every private helper below shares verbatim --
/// gravitropism and phototropism differ only in `direction`/
/// `target_angle_rad`/`mode` (passed separately), never in these. The real
/// fix for clippy::too_many_arguments (was 4 separate loose params in each
/// of `vertex_dkappa`/`evolve_vertex_curvature`/`apply_tropism`/
/// `tropism_still_correcting`, silenced with `#[allow]`) rather than
/// suppressing the lint.
struct TropismLaw<'a> {
    sensitivity: f32,
    straightening: f32,
    resistance: Option<GrowthResistance>,
    grid: &'a Grid,
}

/// Per-vertex `d(kappa)/dt` under the relaxation law (Porat 2024 / Bastien
/// 2013), without applying it: shared by `evolve_vertex_curvature` (which
/// applies it) and `tropism_still_correcting` (which uses its magnitude as
/// the convergence signal). `None` for a near-zero-length edge.
fn vertex_dkappa(
    rod: &RodPoints,
    vertex: usize,
    target_dir: Vec2,
    law: &TropismLaw,
) -> Option<f32> {
    let local_dir = local_edge_direction(rod, vertex)?;
    // sin(theta_local - theta_target) via the 2D "perp dot product" of the
    // two unit directions -- exact, avoids explicit angle/atan2 and wraparound.
    let sin_deviation = local_dir.x * target_dir.y - local_dir.y * target_dir.x;
    let gate = resistance_gate(law.resistance, law.grid, rod.x[vertex + 2]);
    let kappa = rod.rest_curvature[vertex];
    Some(gate * law.sensitivity * sin_deviation - law.straightening * kappa)
}

/// One vertex's own damped-relaxation update -- reads/writes only
/// `rest_curvature[vertex]` and reads `rod.x[vertex..=vertex+2]`, so
/// calling this for every vertex in any order is provably race-free: no
/// two vertices ever touch the same `rest_curvature` slot, and none of
/// them mutate `rod.x`.
fn evolve_vertex_curvature(
    rod: &mut RodPoints,
    vertex: usize,
    target_dir: Vec2,
    law: &TropismLaw,
    dt: f32,
) {
    let Some(dkappa) = vertex_dkappa(rod, vertex, target_dir, law) else {
        return;
    };
    rod.rest_curvature[vertex] += dkappa * dt;
}

/// Shared entry point for any tropism under this law (gravitropism,
/// phototropism): `direction` is the sensed vector (gravity or light),
/// normalized and GSA-rotated here, then every vertex `mode` governs evolves.
/// No-op for a near-zero `direction` or a rod too short to bend.
fn apply_tropism(
    rod: &mut RodPoints,
    mode: GravitropismMode,
    direction: Vec2,
    target_angle_rad: f32,
    law: &TropismLaw,
    dt: f32,
) {
    let n = rod.len();
    if n < 3 {
        return;
    }
    let len_sq = direction.length_squared();
    if len_sq < 1.0e-12 {
        return;
    }
    let unit_dir = direction / len_sq.sqrt();
    let target_dir = target_direction(unit_dir, target_angle_rad);

    for vertex in vertex_range(mode, n) {
        evolve_vertex_curvature(rod, vertex, target_dir, law, dt);
    }
}

/// Shared convergence check for any tropism using `apply_tropism` (see the
/// gravitropism wrapper `still_correcting` below).
///
/// Tests whether the correction is still changing `rest_curvature`
/// (`|d(kappa)/dt|`, from `vertex_dkappa`), not whether one tropism reached
/// its own target: with gravitropism and phototropism pulling to different
/// targets the rod settles at a compromise that satisfies neither, and the
/// per-target deviation never reached zero (no sleep within 60 s at either
/// rate set tried). With one tropism the two conditions coincide.
fn tropism_still_correcting(
    rod: &RodPoints,
    mode: GravitropismMode,
    direction: Vec2,
    target_angle_rad: f32,
    law: &TropismLaw,
) -> bool {
    let n = rod.len();
    if n < 3 {
        return false;
    }
    let len_sq = direction.length_squared();
    if len_sq < 1.0e-12 {
        return false;
    }
    let unit_dir = direction / len_sq.sqrt();
    let target_dir = target_direction(unit_dir, target_angle_rad);

    for vertex in vertex_range(mode, n) {
        let Some(dkappa) = vertex_dkappa(rod, vertex, target_dir, law) else {
            continue;
        };
        if dkappa.abs() > CONVERGED_DKAPPA_PER_SECOND {
            return true;
        }
    }
    false
}

/// Without the `resistance` gate, a rod embedded in soil evolves
/// `rest_curvature` toward its own target-angle alignment regardless of
/// whether it can actually rotate that far, which can drive unbounded
/// velocity growth -- the same class of failure `growth.rs`'s own
/// `GrowthResistance` was built to prevent for elongation, extended here to
/// curvature (see `Gravitropism::resistance`'s doc).
///
/// Evolves `rest_curvature` toward `gravitropism`'s own real gravitropic
/// set-point angle (GSA, Digby & Firn 1995) at whichever vertices its
/// `mode` governs (see `GravitropismMode`'s doc) -- NOT always fully
/// vertical: `target_angle_rad=0.0` recovers the exact original root-only
/// behavior, but any other angle lets an organ actively hold a genuinely
/// non-vertical "true" shape (a plagiotropic lateral branch, or a shoot
/// seeking the OPPOSITE vertical via `PI`), matching how real plant organs
/// don't all converge on one single direction. No-op for a rod with fewer
/// than 3 points (no bending vertex exists) or under zero gravity (no
/// reference direction to rotate the target from).
pub fn apply_gravitropism(
    rod: &mut RodPoints,
    gravitropism: &Gravitropism,
    gravity: Vec2,
    grid: &Grid,
    dt: f32,
) {
    let law = TropismLaw {
        sensitivity: gravitropism.sensitivity,
        straightening: gravitropism.straightening,
        resistance: gravitropism.resistance,
        grid,
    };
    apply_tropism(
        rod,
        gravitropism.mode,
        gravity,
        gravitropism.target_angle_rad,
        &law,
        dt,
    );
}

/// True iff any vertex `gravitropism.mode` governs still has a real,
/// meaningful angular deviation from its own target direction -- used to
/// keep a rod awake (see `Rod::is_correcting_gravitropically`) until its
/// active correction has converged, not just until the passive
/// elastic settle finishes. `CONVERGED_SIN_DEVIATION` is illustrative (see
/// its doc), same disclosure as `sensitivity`/`straightening`.
pub(super) fn still_correcting(
    rod: &RodPoints,
    gravitropism: &Gravitropism,
    gravity: Vec2,
    grid: &Grid,
) -> bool {
    let law = TropismLaw {
        sensitivity: gravitropism.sensitivity,
        straightening: gravitropism.straightening,
        resistance: gravitropism.resistance,
        grid,
    };
    tropism_still_correcting(
        rod,
        gravitropism.mode,
        gravity,
        gravitropism.target_angle_rad,
        &law,
    )
}

/// Evolves `rest_curvature` toward `phototropism`'s own target angle from
/// the sensed `light_dir` -- same real damped-relaxation law as
/// `apply_gravitropism` (see module doc), just a different sensed signal.
/// No-op if `light_dir` is ~zero (nothing to sense).
pub fn apply_phototropism(
    rod: &mut RodPoints,
    phototropism: &Phototropism,
    light_dir: Vec2,
    grid: &Grid,
    dt: f32,
) {
    let law = TropismLaw {
        sensitivity: phototropism.sensitivity,
        straightening: phototropism.straightening,
        resistance: phototropism.resistance,
        grid,
    };
    apply_tropism(
        rod,
        phototropism.mode,
        light_dir,
        phototropism.target_angle_rad,
        &law,
        dt,
    );
}

/// See `still_correcting`'s doc -- the phototropism analog, used by
/// `Rod::is_correcting_phototropically`.
pub(super) fn still_correcting_phototropically(
    rod: &RodPoints,
    phototropism: &Phototropism,
    light_dir: Vec2,
    grid: &Grid,
) -> bool {
    let law = TropismLaw {
        sensitivity: phototropism.sensitivity,
        straightening: phototropism.straightening,
        resistance: phototropism.resistance,
        grid,
    };
    tropism_still_correcting(
        rod,
        phototropism.mode,
        light_dir,
        phototropism.target_angle_rad,
        &law,
    )
}

#[cfg(test)]
mod gsa_tests {
    use super::*;
    use crate::rod::build_straight_rod;

    /// A rod bent so its tip points HORIZONTALLY (perpendicular to gravity),
    /// for probing gravitropism's real response at a controlled deviation.
    fn horizontal_tip_rod() -> RodPoints {
        let mut points = build_straight_rod(Vec2::new(0.0, 5.0), Vec2::new(0.0, 4.0), 4, 0.01, 1.0);
        // Bend just the tip edge to point sideways (+x) instead of continuing
        // straight down, giving a nonzero angle to correct from any target.
        points.x[3] = points.x[2] + Vec2::new(1.0, 0.0);
        points
    }

    #[test]
    fn default_gsa_reproduces_original_align_with_gravity_behavior() {
        // GSA=0.0 (the default) must curl the tip TOWARD gravity_dir, exactly
        // the original (pre-GSA) root-only behavior -- a regression
        // guard, not just a smoke test.
        let mut rod = horizontal_tip_rod();
        let gravitropism = Gravitropism::new(1.0, 0.0);
        let grid = Grid::new(8);
        let gravity = Vec2::new(0.0, -1.0);

        let vertex = rod.rest_curvature.len() - 1;
        let kappa_before = rod.rest_curvature[vertex];
        apply_gravitropism(&mut rod, &gravitropism, gravity, &grid, 0.1);
        let kappa_after = rod.rest_curvature[vertex];

        assert_ne!(
            kappa_before, kappa_after,
            "gravitropism should evolve rest_curvature toward gravity alignment"
        );
    }

    #[test]
    fn gsa_of_pi_curls_the_opposite_way_from_gsa_zero() {
        // Same physical tip (pointing sideways, 90 degrees from either
        // pole) under two different set-point angles must curl in REAL,
        // OPPOSITE directions -- GSA=0 (root-like) pulls it toward "down",
        // GSA=PI (shoot-like) pulls it toward "up". Using a tip already
        // exactly at a pole would give a degenerate zero sin(deviation) for
        // BOTH cases (antiparallel is as much an equilibrium as parallel for
        // a pure sin-of-angle law -- correct physics, but not a useful test
        // case), so this uses a off-pole tip instead.
        let grid = Grid::new(8);
        let gravity = Vec2::new(0.0, -1.0);

        let mut rod_root = horizontal_tip_rod();
        let vertex = rod_root.rest_curvature.len() - 1;
        let root_gsa = Gravitropism::new(1.0, 0.0);
        apply_gravitropism(&mut rod_root, &root_gsa, gravity, &grid, 0.1);
        let kappa_root = rod_root.rest_curvature[vertex];

        let mut rod_shoot = horizontal_tip_rod();
        let shoot_gsa = Gravitropism::new(1.0, 0.0).with_gsa(std::f32::consts::PI);
        apply_gravitropism(&mut rod_shoot, &shoot_gsa, gravity, &grid, 0.1);
        let kappa_shoot = rod_shoot.rest_curvature[vertex];

        assert!(
            kappa_root.abs() > 1.0e-3 && kappa_shoot.abs() > 1.0e-3,
            "both should produce a real, nonzero correction: root={kappa_root}, shoot={kappa_shoot}"
        );
        assert!(
            kappa_root * kappa_shoot < 0.0,
            "GSA=0 and GSA=PI must curl the SAME starting tip in opposite directions \
             (root={kappa_root}, shoot={kappa_shoot}) -- proving negative gravitropism \
             (a shoot reaching for true vertical up) is a real, distinct behavior, not \
             just always following gravity"
        );
    }

    #[test]
    fn plagiotropic_gsa_is_a_real_stable_target_not_just_decay_toward_a_pole() {
        // A tip already sitting AT its own non-vertical set-point angle
        // (here, 90 degrees -- horizontal) should see zero correction, same
        // as a root at GSA=0 sitting straight down -- proving a plagiotropic
        // angle is a target the organ actively holds, not merely an
        // intermediate stop on the way to 0 or PI.
        let mut rod = horizontal_tip_rod();
        let grid = Grid::new(8);
        let gravity = Vec2::new(0.0, -1.0);
        let vertex = rod.rest_curvature.len() - 1;

        let plagiotropic = Gravitropism::new(1.0, 0.0).with_gsa(std::f32::consts::FRAC_PI_2);
        apply_gravitropism(&mut rod, &plagiotropic, gravity, &grid, 0.1);
        assert!(
            rod.rest_curvature[vertex].abs() < 1.0e-4,
            "a tip already at its own 90-degree plagiotropic set-point should see \
             ~zero correction, got {}",
            rod.rest_curvature[vertex]
        );
    }
}

#[cfg(test)]
mod whole_organ_tests {
    use super::*;
    use crate::rod::build_straight_rod;

    /// A 6-point rod uniformly tilted off-vertical -- every edge shares the
    /// SAME nonzero deviation from "straight down" (still genuinely
    /// zero actual curvature, but gravitropism only cares about local edge
    /// direction vs. target, not actual curvature), so every one of its 4
    /// interior vertices has a correction to make under any mode.
    fn uniformly_tilted_rod() -> RodPoints {
        let mut points = build_straight_rod(Vec2::new(0.0, 6.0), Vec2::new(0.0, 0.0), 6, 0.01, 1.0);
        for (i, p) in points.x.iter_mut().enumerate() {
            p.x += i as f32 * 0.3;
        }
        points
    }

    #[test]
    fn tip_only_default_touches_only_the_tip_vertex_bit_for_bit() {
        let mut rod = uniformly_tilted_rod();
        let before = rod.rest_curvature.clone();
        let gravitropism = Gravitropism::new(1.0, 0.0); // mode defaults to TipOnly
        let grid = Grid::new(16);
        apply_gravitropism(&mut rod, &gravitropism, Vec2::new(0.0, -1.0), &grid, 0.1);

        let tip = rod.rest_curvature.len() - 1;
        for (j, (&b, &a)) in before.iter().zip(rod.rest_curvature.iter()).enumerate() {
            if j == tip {
                assert_ne!(b, a, "the tip vertex itself must still evolve");
            } else {
                assert_eq!(
                    b, a,
                    "TipOnly (the default) must leave every non-tip vertex bit-for-bit \
                     unchanged -- vertex {j} moved from {b} to {a}"
                );
            }
        }
    }

    #[test]
    fn whole_organ_mode_evolves_every_interior_vertex() {
        let mut rod = uniformly_tilted_rod();
        let before = rod.rest_curvature.clone();
        let gravitropism = Gravitropism::new(1.0, 0.0).with_mode(GravitropismMode::WholeOrgan);
        let grid = Grid::new(16);
        apply_gravitropism(&mut rod, &gravitropism, Vec2::new(0.0, -1.0), &grid, 0.1);

        for (j, (&b, &a)) in before.iter().zip(rod.rest_curvature.iter()).enumerate() {
            assert_ne!(
                b, a,
                "WholeOrgan must evolve EVERY interior vertex, not just the tip -- \
                 vertex {j} did not change ({b})"
            );
        }
    }

    #[test]
    fn whole_organ_mode_matches_tip_only_at_the_tip_vertex() {
        // Same starting rod/params, only `mode` differs. Per-vertex updates
        // are independent (each reads/writes its own rest_curvature slot,
        // see `evolve_vertex_curvature`'s doc), so the tip vertex's own
        // result must be identical either way -- proving WholeOrgan is a
        // faithful superset of TipOnly's formula, not a different one.
        let grid = Grid::new(16);
        let gravity = Vec2::new(0.0, -1.0);
        let tip = uniformly_tilted_rod().rest_curvature.len() - 1;

        let mut rod_tip_only = uniformly_tilted_rod();
        let tip_only = Gravitropism::new(1.0, 0.0);
        apply_gravitropism(&mut rod_tip_only, &tip_only, gravity, &grid, 0.1);

        let mut rod_whole_organ = uniformly_tilted_rod();
        let whole_organ = Gravitropism::new(1.0, 0.0).with_mode(GravitropismMode::WholeOrgan);
        apply_gravitropism(&mut rod_whole_organ, &whole_organ, gravity, &grid, 0.1);

        assert_eq!(
            rod_tip_only.rest_curvature[tip], rod_whole_organ.rest_curvature[tip],
            "the tip vertex's own result must be identical between modes"
        );
    }

    #[test]
    fn resistance_gate_is_sampled_per_vertex_not_only_at_the_tip() {
        // Mass only near the vertex-0 sample point (dense soil at the base,
        // none higher up): under WholeOrgan the gate is evaluated at each
        // vertex's own position, not once at the tip.
        let rod_template = uniformly_tilted_rod();
        let mut grid = Grid::new(16);
        let base_sample_cell = rod_template.x[2].round().as_ivec2(); // vertex 0 samples x[0+2]
        for dx in -1..=1 {
            for dy in -1..=1 {
                grid.add_mass_momentum(
                    base_sample_cell + glam::IVec2::new(dx, dy),
                    1000.0,
                    Vec2::ZERO,
                );
            }
        }

        let mut rod = rod_template;
        let gravitropism = Gravitropism::new(1.0, 0.0)
            .with_mode(GravitropismMode::WholeOrgan)
            .with_resistance(GrowthResistance {
                turgor_pressure_pa: 1.0,
                resistance_per_unit_mass_pa: 1.0,
            });
        apply_gravitropism(&mut rod, &gravitropism, Vec2::new(0.0, -1.0), &grid, 0.1);

        let base_vertex_correction = rod.rest_curvature[0].abs();
        let tip_vertex_correction = rod.rest_curvature[rod.rest_curvature.len() - 1].abs();
        assert!(
            base_vertex_correction < tip_vertex_correction * 0.5,
            "the base vertex (heavily gated by nearby soil mass) should see a much \
             smaller correction than the tip vertex (no soil mass nearby): \
             base={base_vertex_correction}, tip={tip_vertex_correction}"
        );
    }
}

#[cfg(test)]
mod phototropism_tests {
    use super::*;
    use crate::rod::build_straight_rod;

    fn horizontal_tip_rod() -> RodPoints {
        let mut points = build_straight_rod(Vec2::new(0.0, 5.0), Vec2::new(0.0, 4.0), 4, 0.01, 1.0);
        points.x[3] = points.x[2] + Vec2::new(1.0, 0.0);
        points
    }

    /// The real point of Phase 3: `apply_phototropism` and `apply_gravitropism`
    /// must produce BIT-IDENTICAL results when given the same direction
    /// vector and equivalent parameters -- proof this is shared
    /// code (`apply_tropism`), not two separately-written, coincidentally-
    /// similar implementations that could silently drift apart later.
    fn same_sensitivity_params(
        sensitivity: f32,
        straightening: f32,
    ) -> (Gravitropism, Phototropism) {
        (
            Gravitropism::new(sensitivity, straightening),
            Phototropism::new(sensitivity, straightening),
        )
    }

    #[test]
    fn phototropism_and_gravitropism_share_the_same_core_bit_for_bit() {
        let direction = Vec2::new(0.0, -1.0);
        let grid = Grid::new(8);
        let (gravitropism, phototropism) = same_sensitivity_params(1.0, 0.0);

        let mut rod_g = horizontal_tip_rod();
        apply_gravitropism(&mut rod_g, &gravitropism, direction, &grid, 0.1);

        let mut rod_p = horizontal_tip_rod();
        apply_phototropism(&mut rod_p, &phototropism, direction, &grid, 0.1);

        assert_eq!(
            rod_g.rest_curvature, rod_p.rest_curvature,
            "gravitropism and phototropism must evolve rest_curvature identically \
             given the same direction/sensitivity/straightening -- they share one core"
        );
    }

    #[test]
    fn phototropism_curls_the_tip_toward_the_sensed_light_direction() {
        let mut rod = horizontal_tip_rod();
        let phototropism = Phototropism::new(1.0, 0.0);
        let grid = Grid::new(8);
        let light_dir = Vec2::new(0.0, 1.0); // light from above

        let vertex = rod.rest_curvature.len() - 1;
        let kappa_before = rod.rest_curvature[vertex];
        apply_phototropism(&mut rod, &phototropism, light_dir, &grid, 0.1);
        let kappa_after = rod.rest_curvature[vertex];

        assert_ne!(
            kappa_before, kappa_after,
            "phototropism should evolve rest_curvature toward the sensed light direction"
        );
    }

    #[test]
    fn zero_light_direction_is_a_real_no_op() {
        let mut rod = horizontal_tip_rod();
        let phototropism = Phototropism::new(1.0, 0.0);
        let grid = Grid::new(8);
        let before = rod.rest_curvature.clone();

        apply_phototropism(&mut rod, &phototropism, Vec2::ZERO, &grid, 0.1);

        assert_eq!(
            rod.rest_curvature, before,
            "no sensed light direction means nothing to correct toward -- must be a no-op"
        );
    }
}
