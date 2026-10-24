use glam::Vec2;

/// A single rod's centerline state -- own SoA, independent of `Particles`.
/// `x`/`v` are in grid-cell units (same convention as `Particle::x`/`v`);
/// `mass` is unscaled kilograms (as `Particle::mass`, see
/// `Elastic::particle_mass`).
///
/// Pure kinematic state, no dynamics: like `Particle` and `Grain`, the state
/// lives in `matter::particle` and the dynamics in `spacetime`.
/// `RodMaterial` stays in `spacetime::rod` because two of its methods
/// (`modal_critical_damping`, `fundamental_period_s`) take `&RodPoints`, a
/// solver-side coupling `RodPoints` itself does not have.
#[derive(Debug, Clone)]
pub struct RodPoints {
    pub x: Vec<Vec2>,
    pub v: Vec<Vec2>,
    /// Kilograms per point.
    pub mass: Vec<f32>,
    /// Dirichlet anchor -- identical semantics to `Particle::pinned`: G2P/the
    /// rod's own gather forces `v=0` for a pinned point instead of gathering,
    /// and its position is left completely untouched (not re-clamped to
    /// itself, avoiding float drift), while it still scatters mass/momentum
    /// normally so other bodies push against an immovable anchor.
    pub pinned: Vec<u32>,
    /// Rest length of edge i (between points i, i+1). Length N-1. Meters.
    pub rest_edge_length: Vec<f32>,
    /// Rest discrete curvature at interior vertex i (points i, i+1, i+2).
    /// Length N-2. DIMENSIONLESS (collapses to the turning angle for small
    /// bends -- see `forces::discrete_curvature`'s doc) -- 0.0 for a
    /// straight rod. NOT 1/meters; the per-length normalization lives in the
    /// bending-force formula's own Voronoi-length division, not here.
    pub rest_curvature: Vec<f32>,
    /// Per-edge axial stiffness `E*A`, Newtons. Length N-1. Real prior art:
    /// `network::NetworkEdge::ea` already does this for a branching
    /// `RodNetwork` (different branches are different
    /// thicknesses) -- this ports the same pattern to a plain single-chain
    /// `Rod`, e.g. a stem stiffer at its base than its growing tip.
    /// Uninitialized (empty) when returned by `build_straight_rod` -- filled
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
    /// (`spacetime::integration::advance_position`, used by every rod stepper).
    /// Needed because a rod's own
    /// CFL-bound `dt` is extremely small (~1e-6s, set by its axial stiffness)
    /// while `x` sits at an ordinary grid-coordinate magnitude (e.g. 32.0,
    /// offset from the domain origin). Each individual `x[i] += v*dt`
    /// increment (velocity*dt ~ 1e-8 to 1e-9 at typical wind-driven speeds)
    /// falls BELOW f32's representable precision at that magnitude (ULP at
    /// 32.0 is ~3.8e-6) -- naive accumulation silently rounds every substep's
    /// contribution away to nothing even though the underlying velocity is
    /// sustained and correct. Kahan summation (Kahan 1965, standard
    /// floating-point technique) fixes this by tracking the rounding error
    /// each addition drops and folding it back in next time, without needing
    /// f64 storage.
    pub position_compensation: Vec<Vec2>,
    /// Multi-field frictional contact opt-in, same semantics as
    /// `Particle::contact_group` (Bardenhagen 2001 + Nairn, Hammerquist,
    /// Smith 2020 normal fit): 0 = ordinary (sticks to whatever it touches,
    /// the MPM default), nonzero = a slip/stick interface against everything
    /// else, resolved via `SimConfig::contact_friction`. Measured root-soil
    /// friction coefficients span ~0.02-0.31 depending on surface (McKenzie
    /// et al. 2013, *Plant, Cell & Environment*); set `contact_friction` in
    /// that range for a root scene.
    pub contact_group: Vec<u32>,
    /// Accumulated plastic curvature magnitude at interior vertex i (see the
    /// `plasticity` module), length N-2 like `rest_curvature`. Non-decreasing:
    /// whenever `plasticity::apply_bending_plasticity` absorbs an elastic
    /// excess into `rest_curvature`, its magnitude adds here. Kept separate
    /// from `rest_curvature`, which gravitropism and growth also change for
    /// biological reasons, so this field alone records permanent deformation
    /// from overload. Same role as `Particle::friction_hardening` for
    /// `VonMisesMaterial`'s isotropic hardening (`sigma_y(kappa) =
    /// yield_stress + H*kappa`): hardening state lives on the deformed body,
    /// not on the law. Always 0.0 and inert when no `Rod::plasticity` is
    /// attached.
    pub accumulated_plastic_curvature: Vec<f32>,
    /// Per-edge linear mass density, kg/m, length N-1: the authoritative
    /// record of how much each edge's cross-section has thickened.
    /// `secondary_growth::apply_secondary_growth` grows it in step with `ea`
    /// (`EA = E*A` with `E` constant, so `d(area)/area = d(ea)/ea`, and mass
    /// is proportional to area at fixed length and material density).
    /// `insert_tip_point` (`growth.rs`) reads it directly instead of deriving
    /// density from a possibly non-uniform lumped point mass.
    /// `build_straight_rod` fills every edge with the same value, like
    /// `ea`/`ei`; only secondary growth (or a caller on purpose) makes it vary.
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
