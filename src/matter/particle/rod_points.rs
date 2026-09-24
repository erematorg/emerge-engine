use glam::Vec2;

/// A single rod's centerline state — own SoA, independent of `Particles`.
/// `x`/`v` are in grid-cell units (same convention as `Particle::x`/`v`);
/// `mass` is real, unscaled kilograms (matches `Particle::mass`'s own
/// convention, confirmed via `Elastic::particle_mass` returning real kg).
///
/// Pure kinematic state, no dynamics of its own (moved out of
/// `spacetime::rod` 2026-08-05 to match the exact precedent `Grain` set
/// earlier the same session: `Particle` lives in `matter::particle`, its own
/// P2G/G2P dynamics live in `spacetime`; `RodPoints` now follows the same
/// split). `RodMaterial` stays behind in `spacetime::rod` — two of its own
/// methods (`modal_critical_damping`, `fundamental_period_s`) take
/// `&RodPoints` directly, a genuine solver-side coupling `RodPoints` itself
/// never has.
#[derive(Debug, Clone)]
pub struct RodPoints {
    pub x: Vec<Vec2>,
    pub v: Vec<Vec2>,
    /// Real kilograms per point.
    pub mass: Vec<f32>,
    /// Dirichlet anchor — identical semantics to `Particle::pinned`: G2P/the
    /// rod's own gather forces `v=0` for a pinned point instead of gathering,
    /// and its position is left completely untouched (not re-clamped to
    /// itself, avoiding float drift), while it still scatters mass/momentum
    /// normally so other bodies push against a real immovable anchor.
    pub pinned: Vec<u32>,
    /// Rest length of edge i (between points i, i+1). Length N-1. Meters.
    pub rest_edge_length: Vec<f32>,
    /// Rest discrete curvature at interior vertex i (points i, i+1, i+2).
    /// Length N-2. DIMENSIONLESS (collapses to the turning angle for small
    /// bends — see `forces::discrete_curvature`'s own doc) — 0.0 for a
    /// straight rod. NOT 1/meters; the per-length normalization lives in the
    /// bending-force formula's own Voronoi-length division, not here.
    pub rest_curvature: Vec<f32>,
    /// Per-edge axial stiffness `E*A`, Newtons. Length N-1. Real prior art:
    /// `network::NetworkEdge::ea` already does this for a branching
    /// `RodNetwork` (different branches are genuinely different
    /// thicknesses) — this ports the same pattern to a plain single-chain
    /// `Rod`, e.g. a stem stiffer at its base than its growing tip.
    /// Uninitialized (empty) when returned by `build_straight_rod` — filled
    /// with `RodMaterial::ea` at every index by `Rod::new`, so a normal
    /// construction is bit-identical to the prior uniform-material behavior;
    /// only a caller that explicitly overwrites entries after construction
    /// gets real non-uniform stiffness.
    pub ea: Vec<f32>,
    /// Per-vertex bending stiffness `E*I`, N·m². Length N-2. Same fill
    /// convention as `ea` above (filled from `RodMaterial::ei` by
    /// `Rod::new`).
    pub ei: Vec<f32>,
    /// Kahan (compensated) summation residual for `x`'s position integration
    /// in `coupling::gather_grid_to_rod`. Needed because a rod's own
    /// CFL-bound `dt` is extremely small (~1e-6s, set by its axial stiffness)
    /// while `x` sits at an ordinary grid-coordinate magnitude (e.g. 32.0,
    /// offset from the domain origin). Each individual `x[i] += v*dt`
    /// increment (velocity*dt ~ 1e-8 to 1e-9 at typical wind-driven speeds)
    /// falls BELOW f32's representable precision at that magnitude (ULP at
    /// 32.0 is ~3.8e-6) — naive accumulation silently rounds every substep's
    /// contribution away to nothing even though the underlying velocity is
    /// sustained and correct. Kahan summation (Kahan 1965, standard
    /// floating-point technique) fixes this by tracking the rounding error
    /// each addition drops and folding it back in next time, without needing
    /// f64 storage.
    pub position_compensation: Vec<Vec2>,
    /// Real multi-field frictional contact opt-in — identical semantics to
    /// `Particle::contact_group` (Bardenhagen 2001 + Nairn, Hammerquist,
    /// Smith 2020 normal fit): 0 = ordinary (sticks to whatever it touches,
    /// the MPM default), nonzero = a genuine slip/stick interface against
    /// everything else, resolved via `SimConfig::contact_friction`. Real
    /// measured root-soil friction coefficients (McKenzie et al. 2013,
    /// *Plant, Cell & Environment*) span ~0.02-0.31 depending on surface —
    /// set `contact_friction` to a value in that real range for a root
    /// scene, rather than relying on default MPM stick contact.
    pub contact_group: Vec<u32>,
    /// Real accumulated PLASTIC curvature magnitude at interior vertex i
    /// (see `plasticity` module doc) -- length N-2, same shape as
    /// `rest_curvature`. Monotonically non-decreasing: every time
    /// `plasticity::apply_bending_plasticity` absorbs an elastic excess into
    /// `rest_curvature`, that excess's magnitude adds here too. Deliberately
    /// SEPARATE from `rest_curvature` itself -- `rest_curvature` is also
    /// driven by gravitropism/growth for entirely non-mechanical (biological)
    /// reasons, so it alone can't distinguish "reshaped by active growth"
    /// from "permanently deformed by overload"; this field tracks only the
    /// latter. Real prior art: exactly the role `Particle::friction_
    /// hardening` plays for `VonMisesMaterial`'s own isotropic hardening
    /// (`sigma_y(kappa) = yield_stress + H*kappa`) -- hardening state lives
    /// on the thing being deformed, not on the material/law describing how.
    /// Always 0.0 and inert when no `Rod::plasticity` is attached.
    pub accumulated_plastic_curvature: Vec<f32>,
    /// Real per-edge linear mass density, kg/m. Length N-1. Authoritative
    /// source of truth for "how much has this edge's cross-section actually
    /// thickened" -- `secondary_growth::apply_secondary_growth` grows this
    /// in lockstep with `ea`'s own real fractional growth (`EA=E*A`, `E`
    /// held constant, so `d(area)/area = d(ea)/ea` exactly; mass ∝ area at
    /// fixed length/material density, same real derivation, no separate
    /// invented mechanism). `insert_tip_point` (`growth.rs`) reads this
    /// directly instead of re-deriving density from a possibly-already-
    /// non-uniform lumped point mass. Uniform-fill convention matches `ea`/
    /// `ei` -- `build_straight_rod` fills every edge with the same value;
    /// only secondary growth (or a caller building non-uniform density on
    /// purpose) makes it diverge per edge.
    pub linear_density_kg_per_m: Vec<f32>,
}

impl RodPoints {
    pub const fn len(&self) -> usize {
        self.x.len()
    }

    pub const fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}
