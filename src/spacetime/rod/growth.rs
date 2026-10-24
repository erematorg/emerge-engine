//! Elongation growth: the tip segment's `rest_edge_length` grows by the
//! logistic law (Verhulst 1838, `dL/dt = r·L·(1−L/K)`), the equation
//! `ScalarDiffusionField`'s resource regrowth uses
//! (`src/energy/thermodynamics/scalar_field.rs`, `resource_field.wgsl`),
//! applied to length. Growth as a kinematic quantity separate from elastic
//! strain is Rodriguez, Hoger & McCulloch 1994, "Stress-dependent finite
//! growth in soft elastic tissues," *Journal of Biomechanics* 27(4):455–467
//! (general growth theory, also used for tumours and vessels).
//!
//! Only the tip edge grows: apical-meristem elongation happens at the tip,
//! mature tissue does not keep stretching.
//!
//! **Point insertion (cell division)**: once the tip edge matures
//! (`0.99*max_segment_length_m`, `Rod::is_growing`'s threshold), a new point
//! is inserted beyond it, so growth keeps making new material instead of
//! stretching one segment to its `max_segment_length_m` ceiling. The new edge
//! starts at `NEW_SEGMENT_FRACTION` of `max_segment_length_m` (illustrative,
//! not a measured division size) and is the edge the logistic law grows next.
//!
//! **Force-balance gate** (`GrowthResistance`): root penetration resistance
//! is a distinct force-balance term (cavity expansion plus interfacial
//! friction; Bengough & Mullins 1990, *J. Soil Science* 41:341–358; 1997,
//! *European J. Soil Science*), and in Lockhart's (1965) framework elongation
//! proceeds only once turgor exceeds wall plus soil resistance. Soil
//! resistance is sensed here from the grid's mass density near the tip, a
//! simplification of Bengough & Mullins' cavity-expansion force, which needs
//! grain contact data the continuum grid does not have.
//! `turgor_pressure_pa`/`resistance_per_unit_mass_pa` are illustrative (the
//! ~0.1-1 MPa turgor range comes from secondary sources), like
//! `Gravitropism`'s rates.
//!
//! **Finite resource budget**: without soil, `resistance` gates nothing and
//! logistic growth plus point insertion elongates without limit. Before
//! photosynthesis a seedling's root elongation runs on finite seed reserves
//! (Deleens, Gregory & Bourdu 1984, "Transition between seed reserve use and
//! photosynthetic supply during development of maize seedlings," *Plant
//! Science Letters*: maize roots draw no autotrophic carbon until day
//! 10-14). `Growth::resource_budget_m` is the total length the reserves can
//! still fund, decremented by each call's elongation (stretching and newly
//! inserted length) and halting growth at zero, their "growth crisis".
//! `None` (default) = unlimited.
//!
//! **Light-driven growth rate**: `LightResponse` ties `Growth::rate` to light
//! exposure through a photosynthesis-irradiance curve, the ongoing supply the
//! budget stands in for; both can be used together (budget = reserves left,
//! light = how fast they are spent and replenished).

use glam::Vec2;

use super::RodPoints;
use crate::grid::Grid;
use crate::grid::kernel::quadratic_weights;

/// Photosynthesis-driven growth rate (see the module doc): the rectangular
/// hyperbola (Θ = 0) case of the photosynthesis-irradiance family in Ye et
/// al. 2019, *Scientific Reports*, "A general non-rectangular hyperbola
/// equation for photosynthetic light response curve of rice at various leaf
/// ages". `Growth::rate` becomes the light-saturated rate and the effective
/// rate is `rate * exposure/(half_saturation_exposure + exposure)`: toward
/// `rate` in bright light, toward zero in the dark.
#[derive(Debug, Clone, Copy)]
pub struct LightResponse {
    /// Exposure (see `apply_growth`'s doc for how it's measured) at
    /// which the effective rate reaches half of `Growth::rate`.
    /// Dimensionless, same units as `apply_growth`'s own Lambertian
    /// exposure term (0-1) -- illustrative, same disclosed-calibration
    /// status as this file's other rate constants.
    pub half_saturation_exposure: f32,
}

impl LightResponse {
    pub const fn new(half_saturation_exposure: f32) -> Self {
        Self {
            half_saturation_exposure,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct GrowthResistance {
    /// Internal turgor driving pressure, Pa.
    pub turgor_pressure_pa: f32,
    /// Conversion from local grid mass density to an equivalent soil
    /// resistance pressure, Pa per unit mass -- a but simplified,
    /// stand-in for Bengough & Mullins' particle-scale cavity-expansion
    /// force (see module doc).
    pub resistance_per_unit_mass_pa: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct Growth {
    /// Logistic growth rate `r`, 1/s.
    pub rate: f32,
    /// Carrying capacity `K` -- the maximum length the growing tip
    /// segment approaches, meters. A segment starts well below this and
    /// approaches it asymptotically (real sigmoidal growth curve), never
    /// exceeding it.
    pub max_segment_length_m: f32,
    /// Turgor-versus-soil-resistance growth gate (see module doc). `None`
    /// (default) = ungated logistic growth.
    pub resistance: Option<GrowthResistance>,
    /// Finite seed/storage-reserve budget (Deleens, Gregory & Bourdu 1984,
    /// see module doc): the total length, meters, the reserves can still
    /// fund. `None` (default) = unlimited.
    pub resource_budget_m: Option<f32>,
    /// Light-response growth rate (see `LightResponse`). `None` (default) =
    /// fixed rate.
    pub light_response: Option<LightResponse>,
}

impl Growth {
    pub const fn new(rate: f32, max_segment_length_m: f32) -> Self {
        Self {
            rate,
            max_segment_length_m,
            resistance: None,
            resource_budget_m: None,
            light_response: None,
        }
    }

    pub const fn with_resistance(mut self, resistance: GrowthResistance) -> Self {
        self.resistance = Some(resistance);
        self
    }

    /// Opt into a finite growth budget (see module doc) -- without
    /// this, growth (combined with point insertion) has no total-length
    /// limit at all, unrealistic for any plant given enough real time.
    pub const fn with_resource_budget(mut self, budget_m: f32) -> Self {
        self.resource_budget_m = Some(budget_m.max(0.0));
        self
    }

    /// Opt into real photosynthesis-driven growth rate (see `LightResponse`'s
    /// doc) -- without this, growth proceeds at the fixed `rate`
    /// regardless of light, exactly the prior behavior.
    pub const fn with_light_response(mut self, light_response: LightResponse) -> Self {
        self.light_response = Some(light_response);
        self
    }
}

/// Quadratic-B-spline-weighted local mass density near `pos` (the kernel of
/// every other grid sample, not a single-cell lookup). `pub(super)`:
/// `gravitropism.rs` uses it too, since gravitropic curling is itself
/// differential cell elongation (Bastien et al.'s ACE model, see
/// gravitropism.rs), gated by the same turgor-versus-resistance balance.
pub(super) fn sample_mass_density(grid: &Grid, pos: Vec2) -> f32 {
    let weights = quadratic_weights(pos);
    let mut mass = 0.0f32;
    for gx in 0..3usize {
        for gy in 0..3usize {
            let weight = weights.wx[gx] * weights.wy[gy];
            if weight <= 0.0 {
                continue;
            }
            let cell_pos = weights.base_cell + glam::IVec2::new(gx as i32 - 1, gy as i32 - 1);
            mass += weight * grid.mass_at(cell_pos);
        }
    }
    mass
}

/// Fraction of `max_segment_length_m` a new tip edge starts at
/// ("freshly divided cells start small"), illustrative like this file's
/// rate constants, not a measured division size.
const NEW_SEGMENT_FRACTION: f32 = 0.1;

/// Same maturity threshold `Rod::is_growing`'s doc already uses.
const MATURITY_FRACTION: f32 = 0.99;

/// Apical-meristem cell division: inserts a new point beyond the tip,
/// splitting the growing edge into a mature edge (unchanged) and a small new
/// tip edge that `apply_growth` grows next. Every per-point, per-edge and
/// per-vertex array is extended: `rest_curvature`/
/// `accumulated_plastic_curvature` get an unstrained, undamaged interior
/// vertex; `ea`/`ei` inherit the preceding edge's and vertex's stiffness;
/// mass is re-lumped between the old tip (now interior) and the new end with
/// the rod's `linear_density_kg_per_m` (extended with the parent edge's
/// current value, possibly thickened by secondary growth). `budget_cap_m`,
/// if set, caps the new segment at the remaining reserve (see the module
/// doc). Returns the length seeded, for the caller to deduct.
fn insert_tip_point(rod: &mut RodPoints, dx_meters: f32, budget_cap_m: Option<f32>) -> f32 {
    let old_tip = rod.x.len() - 1;
    let old_tip_edge = rod.rest_edge_length.len() - 1;
    let old_edge_length_m = rod.rest_edge_length[old_tip_edge].max(1.0e-9);

    // Linear density read from the tip edge's `linear_density_kg_per_m`, not
    // re-derived from the tip's lumped mass, which was exact only for a
    // uniform density; `secondary_growth` can thicken one edge.
    let linear_density = rod.linear_density_kg_per_m[old_tip_edge];

    let mut new_edge_length_m = (NEW_SEGMENT_FRACTION * old_edge_length_m).max(1.0e-6);
    if let Some(cap) = budget_cap_m {
        new_edge_length_m = new_edge_length_m.min(cap.max(1.0e-6));
    }
    let new_edge_mass = linear_density * new_edge_length_m;

    // New growth continues straight out along the old tip edge's tangent.
    let tangent = (rod.x[old_tip] - rod.x[old_tip - 1]).normalize_or_zero();
    let new_point_x = rod.x[old_tip] + tangent * (new_edge_length_m / dx_meters);

    rod.x.push(new_point_x);
    rod.v.push(rod.v[old_tip]); // real velocity continuity, not spawned at rest
    rod.pinned.push(0); // the growing tip is never pinned
    rod.contact_group.push(rod.contact_group[old_tip]);
    rod.position_compensation.push(Vec2::ZERO);

    // Re-lump mass: the old tip becomes interior (half of EACH adjacent
    // edge), the new point becomes the endpoint (half of its own edge only)
    // -- the same real lumped-mass convention `build_straight_rod` uses.
    rod.mass[old_tip] = 0.5 * linear_density * old_edge_length_m + 0.5 * new_edge_mass;
    rod.mass.push(0.5 * new_edge_mass);

    rod.rest_edge_length.push(new_edge_length_m);
    // Freshly divided tissue starts with the parent edge's current density
    // (possibly thickened by secondary growth), not the construction value.
    rod.linear_density_kg_per_m.push(linear_density);
    if !rod.ea.is_empty() {
        rod.ea.push(rod.ea[old_tip_edge]);
    }

    // A new interior vertex now exists at the old tip (it sits between the
    // old mature edge and the new tiny edge) -- unstrained, undamaged.
    rod.rest_curvature.push(0.0);
    rod.accumulated_plastic_curvature.push(0.0);
    if !rod.ei.is_empty() {
        rod.ei.push(*rod.ei.last().unwrap());
    }
    new_edge_length_m
}

/// Evolves the tip edge's own `rest_edge_length` via real logistic growth,
/// gated by the turgor-vs-resistance force balance when
/// `growth.resistance` is set, by real photosynthesis/light exposure when
/// `growth.light_response` is set, and by the finite
/// `resource_budget_m` when set (see module doc), then inserts a new point
/// (real cell division) once that edge matures -- unless reserves are
/// exhausted, in which case cell division halts too (Deleens et al. 1984's
/// own real "growth crisis"). No-op for a rod with fewer than 2 points (no
/// edge exists).
///
/// `light_dir` feeds the light-response gate only (see
/// `LightResponse`'s doc) -- exposure is measured as a Lambertian
/// cosine (Lambert's cosine law) between the growing tip's own local
/// tangent direction and `light_dir` (both normalized), clamped to `[0,
/// ∞)`. Disclosed simplification: this substitutes local growth-direction
/// alignment for true photon flux/leaf-area interception, not a full
/// radiative-transfer light model -- the same honest scope this engine's
/// other real-mechanism-illustrative-magnitude constants already carry.
pub fn apply_growth(
    rod: &mut RodPoints,
    growth: &mut Growth,
    grid: &Grid,
    light_dir: Vec2,
    dx_meters: f32,
    dt: f32,
) {
    let n = rod.rest_edge_length.len();
    if n == 0 {
        return;
    }
    let tip_edge = n - 1;
    let tip_point = rod.x.len() - 1;

    let gate = match growth.resistance {
        Some(r) => {
            let local_mass = sample_mass_density(grid, rod.x[tip_point]);
            let local_resistance_pa = r.resistance_per_unit_mass_pa * local_mass;
            (1.0 - local_resistance_pa / r.turgor_pressure_pa.max(1.0e-6)).clamp(0.0, 1.0)
        }
        None => 1.0,
    };

    let light_factor = match growth.light_response {
        Some(lr) => {
            let tangent = (rod.x[tip_point] - rod.x[tip_point - 1]).normalize_or_zero();
            let exposure = tangent.dot(light_dir.normalize_or_zero()).max(0.0);
            exposure / (lr.half_saturation_exposure.max(1.0e-6) + exposure)
        }
        None => 1.0,
    };

    let k = growth.max_segment_length_m.max(1.0e-6);
    let l = rod.rest_edge_length[tip_edge];
    let requested_dl = (gate * light_factor * growth.rate * l * (1.0 - l / k) * dt).max(0.0);
    let actual_dl = match &mut growth.resource_budget_m {
        Some(budget) => {
            let spent = requested_dl.min(*budget);
            *budget -= spent;
            spent
        }
        None => requested_dl,
    };
    rod.rest_edge_length[tip_edge] = (l + actual_dl).max(1.0e-6);

    if rod.rest_edge_length[tip_edge] >= MATURITY_FRACTION * k {
        // Growth crisis (Deleens et al. 1984): with the reserves exhausted,
        // cell division stops too, not only elongation.
        let can_divide = match growth.resource_budget_m {
            Some(remaining) => remaining > 1.0e-9,
            None => true,
        };
        if can_divide {
            let seeded = insert_tip_point(rod, dx_meters, growth.resource_budget_m);
            if let Some(budget) = &mut growth.resource_budget_m {
                *budget -= seeded;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rod::build_straight_rod;

    #[test]
    fn insertion_extends_every_array_to_the_correct_new_length_consistently() {
        let mut rod = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 6, 0.02, 1.0);
        rod.ea = vec![1.0e5; 5];
        rod.ei = vec![2.0; 4];
        let n_before = rod.x.len();

        insert_tip_point(&mut rod, 1.0, None);

        assert_eq!(rod.x.len(), n_before + 1);
        assert_eq!(rod.v.len(), n_before + 1);
        assert_eq!(rod.mass.len(), n_before + 1);
        assert_eq!(rod.pinned.len(), n_before + 1);
        assert_eq!(rod.contact_group.len(), n_before + 1);
        assert_eq!(rod.position_compensation.len(), n_before + 1);
        assert_eq!(rod.rest_edge_length.len(), n_before); // was n-1 edges, now n edges (n_before+1-1)
        assert_eq!(rod.linear_density_kg_per_m.len(), n_before);
        assert_eq!(rod.ea.len(), n_before);
        assert_eq!(rod.rest_curvature.len(), n_before - 1); // now (n_before+1)-2
        assert_eq!(rod.accumulated_plastic_curvature.len(), n_before - 1);
        assert_eq!(rod.ei.len(), n_before - 1);
        assert_eq!(
            *rod.pinned.last().unwrap(),
            0,
            "a new growing tip must never be pinned"
        );
    }

    #[test]
    fn insertion_adds_real_new_mass_not_just_redistributes_existing_mass() {
        // Growth adds biomass (a new cell), it does not reshuffle a fixed mass.
        let mut rod = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 5.0), 4, 0.02, 1.0);
        let mass_before: f32 = rod.mass.iter().sum();
        let old_edge_length_m = rod.rest_edge_length[rod.rest_edge_length.len() - 1];
        let linear_density = *rod.linear_density_kg_per_m.last().unwrap();
        let expected_new_mass = linear_density * NEW_SEGMENT_FRACTION * old_edge_length_m;

        insert_tip_point(&mut rod, 1.0, None);

        let mass_after: f32 = rod.mass.iter().sum();
        assert!(
            mass_after > mass_before,
            "insertion must add real new mass: before={mass_before} after={mass_after}"
        );
        assert!(
            (mass_after - mass_before - expected_new_mass).abs() < 1.0e-6,
            "added mass should match one real new segment's worth: added={} expected={expected_new_mass}",
            mass_after - mass_before
        );
    }

    #[test]
    fn apply_growth_inserts_a_point_once_the_tip_edge_matures() {
        let mut rod = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.02, 0.0), 2, 0.02, 1.0);
        let grid = Grid::new(16);
        // Fast rate, already-mature starting length -- matures on the first call.
        let mut growth = Growth::new(5.0, 0.02);
        rod.rest_edge_length[0] = 0.0199; // already at 99.5% of K

        assert_eq!(rod.x.len(), 2);
        apply_growth(&mut rod, &mut growth, &grid, Vec2::Y, 1.0, 0.01);
        assert_eq!(
            rod.x.len(),
            3,
            "an already-mature tip edge must insert a new point on the very next call"
        );
    }

    #[test]
    fn light_response_stalls_growth_in_the_dark_and_allows_it_in_full_light() {
        // Tip points straight up (+Y) in both cases -- only `light_dir`
        // differs, isolating the light-exposure gate from everything
        // else (gate/budget/rate all identical).
        let grid = Grid::new(16);
        let mut lit = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.05), 2, 0.02, 1.0);
        let mut dark = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 0.05), 2, 0.02, 1.0);
        lit.rest_edge_length[0] = 0.05;
        dark.rest_edge_length[0] = 0.05;

        let mut growth_lit = Growth::new(0.5, 1.0).with_light_response(LightResponse::new(0.5));
        let mut growth_dark = Growth::new(0.5, 1.0).with_light_response(LightResponse::new(0.5));

        for _ in 0..200 {
            // Light comes from straight above -- full exposure against the
            // tip's own upward tangent.
            apply_growth(&mut lit, &mut growth_lit, &grid, Vec2::Y, 1.0, 0.05);
            // Light comes from straight below -- the tip's upward tangent
            // gets a negative dot product, clamped to zero exposure.
            apply_growth(&mut dark, &mut growth_dark, &grid, -Vec2::Y, 1.0, 0.05);
        }

        assert!(
            lit.rest_edge_length[0] > 0.06,
            "a fully-lit tip must genuinely elongate: got {}",
            lit.rest_edge_length[0]
        );
        assert!(
            (dark.rest_edge_length[0] - 0.05).abs() < 1.0e-6,
            "a tip with zero light exposure must not grow at all: got {}",
            dark.rest_edge_length[0]
        );
    }

    #[test]
    fn light_response_half_saturation_halves_the_effective_rate() {
        // At exposure == half_saturation_exposure, the rectangular
        // hyperbola gives EXACTLY 0.5 -- a direct precision check of the
        // real equation, not just "more light grows more."
        let grid = Grid::new(16);
        let l0 = 0.05_f32;
        let k = 1.0_f32;
        let rate = 0.5_f32;
        let dt = 0.001_f32; // small step: compare instantaneous rates, not the integrated curve

        let mut unrestricted =
            build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, l0), 2, 0.02, 1.0);
        let mut half_lit =
            build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, l0), 2, 0.02, 1.0);
        unrestricted.rest_edge_length[0] = l0;
        half_lit.rest_edge_length[0] = l0;

        let mut growth_full = Growth::new(rate, k);
        // Tip tangent is +Y; light_dir = +Y gives exposure = 1.0. Setting
        // half_saturation_exposure = 1.0 makes THIS specific exposure sit
        // exactly at the curve's own half-saturation point.
        let mut growth_half = Growth::new(rate, k).with_light_response(LightResponse::new(1.0));

        apply_growth(&mut unrestricted, &mut growth_full, &grid, Vec2::Y, 1.0, dt);
        apply_growth(&mut half_lit, &mut growth_half, &grid, Vec2::Y, 1.0, dt);

        let dl_full = unrestricted.rest_edge_length[0] - l0;
        let dl_half = half_lit.rest_edge_length[0] - l0;

        assert!(
            (dl_half / dl_full - 0.5).abs() < 1.0e-4,
            "at exposure == half_saturation_exposure the effective rate must be \
             EXACTLY half the light-saturated rate: dl_full={dl_full:.8} dl_half={dl_half:.8} \
             ratio={:.6}",
            dl_half / dl_full
        );
    }

    /// Without a budget, sustained growth with no soil resistance keeps
    /// elongating; a finite budget halts it.
    #[test]
    fn resource_budget_genuinely_halts_growth_once_exhausted() {
        let grid = Grid::new(64);
        let drive = |growth: &mut Growth, steps: u32| -> f32 {
            let mut rod =
                build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.002, 0.0), 2, 0.02, 1.0);
            for _ in 0..steps {
                apply_growth(&mut rod, growth, &grid, Vec2::Y, 1.0, 1.0);
            }
            rod.rest_edge_length.iter().sum::<f32>()
        };

        let mut unlimited = Growth::new(0.5, 0.005);
        let total_unlimited = drive(&mut unlimited, 2000);

        let mut budgeted = Growth::new(0.5, 0.005).with_resource_budget(0.01);
        let total_budgeted = drive(&mut budgeted, 2000);

        assert!(
            total_unlimited > 10.0 * total_budgeted,
            "with no budget, sustained growth over many steps must vastly exceed a real, \
             finite 0.01m budget's own total (unlimited={total_unlimited:.4}, \
             budgeted={total_budgeted:.4}) -- otherwise the budget isn't really constraining \
             anything"
        );
        assert!(
            total_budgeted <= 0.012, // real 0.01m budget + the rod's own initial 0.002m length
            "a real, finite 0.01m budget must actually cap total growth, not just slow it: \
             got {total_budgeted:.4}"
        );
    }
}
