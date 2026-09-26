//! A genuine 1D discrete elastic rod (Cosserat-rod family) -- a real,
//! dimensionally-reduced continuum solver for slender (length >> width)
//! bodies, sibling to `spacetime::diff` (a second, narrower, self-contained
//! solver living under `spacetime/`, not shoehorned into `solver/`).
//!
//! # Why this exists
//! Real engineering practice does not use full volumetric FEM/MPM for a
//! fishing rod, a cable, or a blade of grass -- it uses beam/rod theory, a
//! rigorous 1D reduction of the SAME continuum elasticity MPM's own 2D
//! materials already are relative to full 3D. Pure 2D volumetric MPM shows
//! genuine self-weight Euler/Greenhill buckling for a slender cantilever --
//! pushing a thin volumetric blade taller destabilizes it. `EI` (bending
//! stiffness) is an emergent property of carved cross-section width in 2D
//! MPM; here it is a direct, exact input parameter -- the real advantage of
//! the right dimensional reduction for this class of body.
//!
//! # Real physics: discrete elastic rods, specialized to 2D
//! Bergou, Wardetzky, Robinson, Audoly, Grinspun 2008, SIGGRAPH, "Discrete
//! Elastic Rods" -- the modern discrete form of classical Cosserat rod theory
//! (Cosserat brothers, 1909). A rod is `N` control points along a centerline;
//! `N-1` edges carry axial (stretch) elastic energy; `N-2` interior vertices
//! carry bending elastic energy from the discrete curvature between adjacent
//! edges. In 3D the curvature is a vector (the discrete binormal) with a
//! separate twist DOF about the centerline; in 2D the binormal direction is
//! fixed to the plane's own normal, so curvature collapses to a signed
//! scalar and twist has no real DOF at all (2D genuinely has one fewer
//! curvature dimension than 3D -- a real dimensional fact, not a cut corner).
//! See `forces.rs` for the actual formulas.
//!
//! # Coupling to the shared MPM grid
//! `Grid` (`spacetime::grid`) is a fully source-agnostic mass/momentum
//! accumulator -- every mutator is keyed purely by `IVec2` cell position,
//! nothing in it references `Particle`/`Particles`. `coupling.rs` scatters
//! rod points into that SAME grid using the identical quadratic-B-spline
//! weights `transfer::p2g`/`transfer::g2p` already use, so a rod and ordinary
//! MPM particles (fluid, sand, fire) genuinely exchange momentum through one
//! shared mechanism -- not an isolated parallel system. See `Simulation::rods`
//! and its `do_substep` insertion points for the real coupling (Phase 2).
//!
//! `RodPoints` is a new, independent SoA -- NOT grafted onto `Particle`
//! (which is `repr(C)`/`Pod`/128-byte GPU-layout-locked, append-only after a
//! real past corruption bug) or `Particles`. Same "independent store,
//! cross-talk only through the shared `Grid`" shape `Particles` itself
//! already is relative to `Grid`. Lives in `matter::particle` (moved there
//! 2026-08-05, re-exported here) -- pure kinematic state is a Matter
//! concern, same as `Particle`; everything in this module is the dynamics
//! that evolves it.
//!
//! # Scope (explicit, disclosed)
//! CPU-first (matches this engine's own "CPU correctness first, GPU port
//! second" rule -- GPU port is real future work, not attempted here: WGPU bind-group layouts are already
//! at the 4-group WebGPU baseline limit). 2D only. True branching topology
//! exists via `network::RodNetwork` (a real graph, not a single chain); a
//! plain `Rod` itself stays a single unbranched chain. No twist DOF (none
//! exists in 2D). `SimSnapshot::rods` covers per-rod count/sleeping/speed/tip
//! aggregates; `RodNetwork` isn't wired into that aggregate yet.

pub mod coupling;
pub mod forces;
pub mod gravitropism;
pub mod growth;
pub mod implicit;
pub mod integrator;
pub mod network;
pub mod plasticity;
pub mod secondary_growth;

use glam::Vec2;

pub use crate::matter::materials::RodMaterial;
pub use crate::matter::particle::RodPoints;
pub use coupling::{
    RodForceParams, apply_rod_internal_and_wind_forces, gather_grid_to_rod, scatter_rod_to_grid,
};
pub use forces::{
    RodRestState, compute_internal_forces, discrete_curvature, discrete_curvature_gradient,
};
pub use gravitropism::{
    Gravitropism, GravitropismMode, Phototropism, apply_gravitropism, apply_phototropism,
};
pub use growth::{Growth, GrowthResistance, apply_growth};
pub use implicit::{RodImplicitStepParams, step_rod_implicit};
pub use integrator::{apply_mass_scaling_for_target_dt, rod_cfl_dt, step_rod};
pub use network::{
    NetworkBendingVertex, NetworkEdge, RodNetwork, YBranchSpec, build_y_branch,
    compute_network_internal_forces, network_cfl_dt, step_network,
};
pub use plasticity::{RodPlasticity, apply_bending_plasticity};
pub use secondary_growth::{SecondaryGrowth, apply_secondary_growth};

/// A rod embedded in a `Simulation` -- points plus the material they share.
/// No separate gravity field: gravity comes from the single
/// `SimConfig::gravity` value the whole simulation already shares (a second
/// gravity knob would be exactly the kind of drift-prone duplication
/// a "derive, don't store" discipline warns against). Wind
/// and push are per-rod, mutable state instead -- set fresh by the caller
/// before each `step()` call, held constant for that call (which may cover
/// several substeps). Both default to zero/off, so embedding a rod with no
/// wind or push logic is exactly zero-cost.
#[derive(Debug, Clone)]
pub struct Rod {
    pub points: RodPoints,
    pub material: RodMaterial,
    pub wind_velocity: Vec2,
    pub wind_drag_coeff: f32,
    /// Read fresh every substep, same as `wind_velocity` -- a one-shot
    /// `rod.points.v[i] +=` applied before `step()` gets fully absorbed by
    /// critical damping within the substep loop before the frame even
    /// returns, since `step()` can run thousands of internal substeps.
    pub push_center: Option<Vec2>,
    pub push_strength: f32,
    pub push_radius: f32,
    /// Mirrors `Particle::sleeping` at rod granularity: a rod's points are
    /// elastically coupled (bending/stretch energy between neighbors), so
    /// sleep is an all-or-nothing property of the whole rod, not individual
    /// points, unlike independent MPM particles. A sleeping rod skips
    /// scatter/gather/internal-force integration AND `rod_cfl_dt` entirely --
    /// the real fix for many-simultaneous-rods cost (a grass field): most
    /// blades settle to near-zero velocity and should stop paying their own
    /// (expensive, stiff) CFL bound every substep once they have. Woken by
    /// the same grid-activity-overlap test `wake_particle` already uses, or
    /// immediately when a caller sets an active push.
    pub sleeping: bool,
    /// Sleep-scoring must NOT fire on the instant `max_speed_sq < threshold_sq`:
    /// a freshly-constructed rod trivially satisfies that (`v = Vec2::ZERO` at
    /// construction) before gravity/grid coupling gets a chance to act within
    /// that first substep's tiny `sub_dt`, and would then skip its own gravity
    /// while "asleep" until a neighboring disturbance woke it -- receiving the
    /// full, undamped gravitational transient it should have absorbed gradually
    /// as a large unphysical velocity spike. Every major real-time physics
    /// engine (Box2D, Bullet, PhysX) requires a body to stay below its sleep
    /// threshold for a minimum REAL DURATION, not one instant, to avoid this --
    /// this tracks that duration (real seconds), reset to 0 the moment speed
    /// exceeds the threshold, checked in `step.rs`'s sleep-scoring pass.
    pub below_threshold_time: f32,
    /// Real root gravitropism (Porat, Rivière, Meroz 2024 -- see
    /// `gravitropism` module doc). `None` (default) = no gravitropic
    /// response, zero cost -- a plain stem/blade doesn't grow toward
    /// gravity, only a root does.
    pub gravitropism: Option<Gravitropism>,
    /// Real phototropism (Cholodny & Went auxin-asymmetry theory -- see
    /// `gravitropism` module doc's own "Phototropism reuses the SAME core"
    /// section). `None` (default) = no light-seeking response, zero cost.
    pub phototropism: Option<Phototropism>,
    /// Real elongation growth (see `growth` module doc). `None` (default) =
    /// fixed length, zero cost -- most bodies aren't actively growing every
    /// frame of their existence.
    pub growth: Option<Growth>,
    /// Real stress-driven secondary growth / thigmomorphogenesis (Jaffe
    /// 1973, Mattheck & Kübler 1995 -- see `secondary_growth` module doc).
    /// `None` (default) = fixed stiffness, zero cost -- requires Phase 1's
    /// per-vertex `RodPoints::ea`/`ei` to already be filled (`Rod::new`
    /// does this).
    pub secondary_growth: Option<SecondaryGrowth>,
    /// Real elastic-perfectly-plastic bending (see `plasticity` module doc).
    /// `None` (default) = purely elastic, zero cost -- most bodies don't
    /// permanently deform under load; a wire/branch/cable that should stay
    /// bent after enough force opts in.
    pub plasticity: Option<RodPlasticity>,
    /// Implicit (backward Euler) integration, opt-in (Baraff & Witkin 1998;
    /// see `implicit` module doc). `false` (default) = the explicit path,
    /// unchanged. When `true`, solved ONCE per `Simulation::step()` at the
    /// full frame `dt`, unconditionally stable regardless of stiffness
    /// (1298->1 substeps/frame measured on `rod_blade_of_grass_gui.rs`'s
    /// blade), excluded from the substep loop's CFL scan.
    ///
    /// Gravity/wind/push stay INSIDE this solve, not on the shared grid --
    /// an implicit rod has no CFL ceiling, so an explicit `v += g*dt` at the
    /// full frame dt is unconditionally unstable (tried once, reverted).
    /// Scope: not grid-coupled -- no contact with sand/particles yet.
    pub use_implicit_integration: bool,
    /// Backward Euler is unconditionally STABLE at any `dt`, but at a large
    /// `dt` relative to the rod's own natural bending period it also
    /// introduces artificial numerical damping that can swamp the
    /// physically-tuned damping (`RodMaterial::axial_damping`/
    /// `bending_damping`), making an underdamped sway look like a smooth,
    /// "instant" glide to rest instead. Splitting the frame `dt` into more
    /// implicit substeps (this field) recovers the correct oscillatory
    /// behavior at the same physical damping ratio. Default `1` = one step
    /// at the full frame `dt`, zero change for any rod that doesn't opt in.
    /// Only meaningful when `use_implicit_integration` is `true`.
    pub implicit_substeps: u32,
}

impl Rod {
    /// Fills `points.ea`/`points.ei` to the correct per-edge/per-vertex
    /// length using `material`'s scalar values whenever they don't already
    /// match (the normal case: `build_straight_rod` leaves them empty) --
    /// makes a uniform-material rod bit-identical to the prior
    /// single-scalar-material behavior. A caller who wants real non-uniform
    /// stiffness fills `points.ea`/`ei` explicitly BEFORE calling `Rod::new`
    /// (matching length), and this fill is skipped.
    pub fn new(points: RodPoints, material: RodMaterial) -> Self {
        let mut points = points;
        let n_edges = points.x.len().saturating_sub(1);
        let n_bend = points.x.len().saturating_sub(2);
        if points.ea.len() != n_edges {
            points.ea = vec![material.ea; n_edges];
        }
        if points.ei.len() != n_bend {
            points.ei = vec![material.ei; n_bend];
        }
        Self {
            points,
            material,
            wind_velocity: Vec2::ZERO,
            wind_drag_coeff: 0.0,
            push_center: None,
            push_strength: 0.0,
            push_radius: 0.0,
            sleeping: false,
            below_threshold_time: 0.0,
            gravitropism: None,
            phototropism: None,
            growth: None,
            secondary_growth: None,
            plasticity: None,
            use_implicit_integration: false,
            implicit_substeps: 1,
        }
    }

    /// Immediate self-weight buckling check (Euler/Greenhill, see
    /// `RodMaterial::greenhill_critical_height_m`'s own doc). Returns
    /// `Some(human-readable message)` if this rod's length exceeds its own
    /// critical height (it will NEVER stand straight under gravity alone,
    /// regardless of damping -- that's the correct physics, not a numerical
    /// bug), `None` if it's safely below. Call this once right after
    /// construction and `eprintln!` the result to catch this class of
    /// mistake immediately.
    ///
    /// For a NON-uniform rod (per-vertex `points.ei`, see `RodPoints::ei`'s
    /// own doc): `greenhill_critical_height_m` is a closed-form result for a
    /// UNIFORM column, so there is no single exact non-uniform
    /// generalization here. Uses the WEAKEST (minimum) `ei` entry as the
    /// conservative bound instead -- a non-uniform rod buckles first at its
    /// most slender point, so checking the whole rod's length against that
    /// point's own critical height cannot UNDER-warn (it may warn slightly
    /// early for a rod that's stiffer everywhere else, never miss a real
    /// risk).
    pub fn buckling_warning(&self, gravity_m_s2: f32) -> Option<String> {
        let length_m: f32 = self.points.rest_edge_length.iter().sum();
        let total_mass_kg: f32 = self.points.mass.iter().sum();
        if length_m <= 0.0 || total_mass_kg <= 0.0 {
            return None;
        }
        let mu = total_mass_kg / length_m;
        // `self.material.ei` is only the FILL VALUE `Rod::new` used at
        // construction -- once `SecondaryGrowth` (or any caller) grows
        // `points.ei` beyond it, `material.ei` itself is never updated and
        // must NOT be folded in here, or a genuinely-stiffened rod would
        // incorrectly keep reporting its own stale original weakness.
        let weakest_ei = if self.points.ei.is_empty() {
            self.material.ei
        } else {
            self.points.ei.iter().copied().fold(f32::INFINITY, f32::min)
        };
        let h_crit = RodMaterial::greenhill_critical_height_m(weakest_ei, mu, gravity_m_s2);
        if length_m > h_crit {
            Some(format!(
                "rod is {length_m:.4}m tall but its own real Euler/Greenhill self-weight \
                 buckling critical height is only {h_crit:.4}m (weakest EI={weakest_ei:.4e} \
                 N*m^2, mu={mu:.4} kg/m, g={gravity_m_s2:.2} m/s^2) -- this rod will \
                 genuinely, physically NOT stand straight under gravity alone, no matter the \
                 damping. Either shorten it below {h_crit:.4}m or stiffen it (increase E or the \
                 cross-section's I)."
            ))
        } else {
            None
        }
    }

    pub const fn with_wind(mut self, wind_velocity: Vec2, wind_drag_coeff: f32) -> Self {
        self.wind_velocity = wind_velocity;
        self.wind_drag_coeff = wind_drag_coeff;
        self
    }

    pub const fn with_gravitropism(mut self, gravitropism: Gravitropism) -> Self {
        self.gravitropism = Some(gravitropism);
        self
    }

    pub const fn with_phototropism(mut self, phototropism: Phototropism) -> Self {
        self.phototropism = Some(phototropism);
        self
    }

    pub const fn with_growth(mut self, growth: Growth) -> Self {
        self.growth = Some(growth);
        self
    }

    pub const fn with_plasticity(mut self, plasticity: RodPlasticity) -> Self {
        self.plasticity = Some(plasticity);
        self
    }

    /// True while `growth` is still meaningfully lengthening the tip edge --
    /// real guard against a genuine sleep/growth interaction bug: sleep
    /// scoring (`step.rs`) only sees `rod.points.v`, but a critically-damped
    /// rod's elastic response reaches quasi-static equilibrium (near-zero
    /// velocity) on a MUCH faster timescale than logistic growth itself
    /// (milliseconds vs. tens of seconds), so a velocity-only sleep check
    /// would put the rod to sleep mid-growth and silently freeze it there --
    /// once `sleeping=true`, `apply_growth` is skipped entirely alongside
    /// everything else. 99% of `max_segment_length_m` is the real, standard
    /// cutoff for an asymptotic logistic curve that mathematically never
    /// exactly reaches its carrying capacity. `false` (safe to sleep) for a
    /// rod with no `growth` at all.
    pub fn is_growing(&self) -> bool {
        match &self.growth {
            Some(g) => match self.points.rest_edge_length.last() {
                Some(&l) => l < 0.99 * g.max_segment_length_m,
                None => false,
            },
            None => false,
        }
    }

    /// True while `gravitropism` still has a real, meaningful angular
    /// deviation left to correct -- the same class of sleep/growth
    /// interaction `is_growing`'s own doc describes: sleep scoring only
    /// sees `rod.points.v`, but gravitropism reshapes `rest_curvature`
    /// (not velocity directly), so a rod can settle to near-zero velocity
    /// from its LAST push, go to sleep, and then never wake again -- freezing
    /// gravitropism forever with no external event left to rouse it (a
    /// sleeping rod is skipped entirely at both `apply_gravitropism` call
    /// sites in `step.rs`). `false` (safe to sleep) for a rod with no
    /// `gravitropism` at all, or once it's genuinely converged.
    pub fn is_correcting_gravitropically(&self, gravity: Vec2, grid: &crate::grid::Grid) -> bool {
        self.gravitropism
            .as_ref()
            .is_some_and(|g| gravitropism::still_correcting(&self.points, g, gravity, grid))
    }

    /// See `is_correcting_gravitropically`'s own doc -- the phototropism
    /// analog, same sleep-freeze-prevention purpose.
    pub fn is_correcting_phototropically(&self, light_dir: Vec2, grid: &crate::grid::Grid) -> bool {
        self.phototropism.as_ref().is_some_and(|p| {
            gravitropism::still_correcting_phototropically(&self.points, p, light_dir, grid)
        })
    }
}

/// Build a straight rod from `start` to `end`, `n_points` control points,
/// uniform `linear_density_kg_per_m`, zero rest curvature (straight rest
/// shape). `dx_meters` converts the real-meters spacing into grid-cell units
/// for `x` -- same SI-to-grid convention `gravity_to_grid`/`lame_from_si`
/// already use elsewhere in this codebase.
pub fn build_straight_rod(
    start: Vec2,
    end: Vec2,
    n_points: usize,
    linear_density_kg_per_m: f32,
    dx_meters: f32,
) -> RodPoints {
    assert!(n_points >= 2, "a rod needs at least 2 points");
    let total_length_m = (end - start).length() * dx_meters;
    let segment_length_m = total_length_m / (n_points as f32 - 1.0);
    let point_mass = linear_density_kg_per_m * segment_length_m;

    let mut x = Vec::with_capacity(n_points);
    let mut v = Vec::with_capacity(n_points);
    let mut mass = Vec::with_capacity(n_points);
    let mut pinned = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let t = i as f32 / (n_points as f32 - 1.0);
        x.push(start.lerp(end, t));
        v.push(Vec2::ZERO);
        // Endpoints carry half a segment's mass (standard lumped-mass
        // discretization), interior points carry a full segment.
        let m = if i == 0 || i == n_points - 1 {
            point_mass * 0.5
        } else {
            point_mass
        };
        mass.push(m);
        pinned.push(0);
    }

    RodPoints {
        x,
        v,
        mass,
        pinned,
        rest_edge_length: vec![segment_length_m; n_points - 1],
        rest_curvature: vec![0.0; n_points.saturating_sub(2)],
        // Left empty -- `Rod::new` fills these from `RodMaterial::ea`/`ei`
        // (this function has no material to fill them with yet).
        ea: Vec::new(),
        ei: Vec::new(),
        position_compensation: vec![Vec2::ZERO; n_points],
        contact_group: vec![0; n_points],
        accumulated_plastic_curvature: vec![0.0; n_points.saturating_sub(2)],
        linear_density_kg_per_m: vec![linear_density_kg_per_m; n_points - 1],
    }
}

#[cfg(test)]
mod root_cause_fixes_tests {
    use super::*;

    /// Real, permanent regression guard for the modal-damping root-cause
    /// fix: `modal_critical_damping` must give a substantially LARGER
    /// bending value than the old, disclosed-as-too-small
    /// `critical_damping` for a real multi-point cantilever -- confirmed
    /// empirically to be a two-to-three-orders-of-magnitude gap for a
    /// 20-point blade.
    #[test]
    fn modal_critical_damping_exceeds_local_reference_substantially() {
        let start = Vec2::new(9.0, 4.0);
        let height_m = 0.10;
        let dx_meters = 0.01;
        let points = build_straight_rod(
            start,
            Vec2::new(start.x, start.y + height_m / dx_meters),
            20,
            0.01,
            dx_meters,
        );
        let young_modulus = 1.0e7_f32;
        let ea = young_modulus * 0.003 * 0.001;
        let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;
        let l0 = height_m / 19.0;
        let point_mass = 0.01 * l0;
        let (_, old_bending) = RodMaterial::critical_damping(l0, point_mass, ea, ei);
        let (_, new_bending) = RodMaterial::modal_critical_damping(&points, ea, ei);
        assert!(
            new_bending > old_bending * 100.0,
            "modal_critical_damping ({new_bending:.6e}) should exceed the local reference \
             ({old_bending:.6e}) by at least 100x for a 20-point cantilever -- if this ever \
             shrinks close to 1x, the two formulas may have been (wrongly) unified without \
             re-verifying against this real measurement"
        );
    }

    /// Real cross-check: `fundamental_period_s` must be the exact reciprocal
    /// of the frequency `energy::acoustics::modal::cantilever_rod_modes`
    /// computes -- both use the SAME real beta_1 eigenvalue and omega
    /// formula, so any divergence between them is a real bug in one or the
    /// other, not just numerical noise.
    #[cfg(feature = "experimental")]
    #[test]
    fn fundamental_period_matches_acoustics_module_frequency() {
        let start = Vec2::new(0.0, 0.0);
        let height_m = 0.10;
        let points =
            build_straight_rod(start, Vec2::new(start.x, start.y + height_m), 20, 0.01, 1.0);
        let young_modulus = 1.0e7_f32;
        let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;
        let material = RodMaterial::new(young_modulus * 0.003 * 0.001, ei, 0.0, 0.0);
        let rod = Rod::new(points, material);

        let period = RodMaterial::fundamental_period_s(&rod.points, ei);
        let modes = crate::acoustics::cantilever_rod_modes(&rod, 1);
        let freq_hz = modes[0].frequency_hz;

        let rel_err = (period - 1.0 / freq_hz).abs() / (1.0 / freq_hz);
        assert!(
            rel_err < 1.0e-4,
            "fundamental_period_s ({period:.6}s) should be the exact reciprocal of \
             cantilever_rod_modes's own frequency ({freq_hz:.4} Hz -> period {:.6}s) -- \
             rel_err={rel_err:.6}",
            1.0 / freq_hz
        );
    }

    /// Real, permanent regression guard: blade B's real parameters
    /// (E=5e6, height=0.10m) must trigger a buckling warning; blade A's
    /// (E=1e7, same height) must not.
    #[test]
    fn buckling_warning_matches_expected_critical_height() {
        let start = Vec2::new(9.0, 4.0);
        let height_m = 0.10;
        let dx_meters = 0.01;
        let make_rod = |young_modulus: f32| -> Rod {
            let points = build_straight_rod(
                start,
                Vec2::new(start.x, start.y + height_m / dx_meters),
                20,
                0.01,
                dx_meters,
            );
            let ea = young_modulus * 0.003 * 0.001;
            let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;
            Rod::new(points, RodMaterial::new(ea, ei, 0.0, 0.0))
        };

        let blade_a = make_rod(1.0e7);
        let blade_b = make_rod(5.0e6);

        assert!(
            blade_a.buckling_warning(9.81).is_none(),
            "blade A (E=1e7) should be safely below its own critical height -- got a warning: {:?}",
            blade_a.buckling_warning(9.81)
        );
        assert!(
            blade_b.buckling_warning(9.81).is_some(),
            "blade B (E=5e6) is the real, confirmed buckling case -- should warn"
        );
    }

    /// Real, permanent regression guard for the 2026-07-27 sleep-freeze fix:
    /// a rod with a genuine, uncorrected gravitropic deviation must report
    /// `is_correcting_gravitropically() == true` (blocking sleep), while one
    /// already aligned with its own target must NOT (so ordinary sleep still
    /// works once gravitropism has nothing real left to do).
    #[test]
    fn gravitropism_prevents_premature_sleep_until_converged() {
        let dx_meters = 0.01;
        let make_rod = |tip_offset_x: f32| -> Rod {
            let start = Vec2::new(9.0, 4.0);
            let mut points = build_straight_rod(
                start,
                Vec2::new(start.x, start.y + 0.10 / dx_meters),
                6,
                0.01,
                dx_meters,
            );
            let last = points.x.len() - 1;
            points.x[last].x += tip_offset_x;
            let mut rod = Rod::new(points, RodMaterial::new(1.0, 1.0e-6, 0.0, 0.0));
            rod.gravitropism = Some(Gravitropism::new(0.05, 0.005));
            rod
        };

        let gravity = Vec2::new(0.0, -1.0);
        let grid = crate::grid::Grid::new(16);
        let misaligned = make_rod(2.0); // tip pushed sideways -- real, unconverged deviation
        let aligned = make_rod(0.0); // straight down -- matches GSA=0.0's target exactly

        assert!(
            misaligned.is_correcting_gravitropically(gravity, &grid),
            "a rod with a real, uncorrected angular deviation must report still-correcting"
        );
        assert!(
            !aligned.is_correcting_gravitropically(gravity, &grid),
            "a rod already aligned with its own target must NOT block sleep"
        );
    }

    /// Same sleep-freeze-prevention guard as
    /// `gravitropism_prevents_premature_sleep_until_converged`, for the
    /// phototropism analog -- proves `is_correcting_phototropically` isn't
    /// a no-op stub.
    #[test]
    fn phototropism_prevents_premature_sleep_until_converged() {
        let dx_meters = 0.01;
        let make_rod = |tip_offset_x: f32| -> Rod {
            let start = Vec2::new(9.0, 4.0);
            let mut points = build_straight_rod(
                start,
                Vec2::new(start.x, start.y + 0.10 / dx_meters),
                6,
                0.01,
                dx_meters,
            );
            let last = points.x.len() - 1;
            points.x[last].x += tip_offset_x;
            let mut rod = Rod::new(points, RodMaterial::new(1.0, 1.0e-6, 0.0, 0.0));
            rod.phototropism = Some(Phototropism::new(0.05, 0.005));
            rod
        };

        let light_dir = Vec2::new(0.0, 1.0);
        let grid = crate::grid::Grid::new(16);
        let misaligned = make_rod(2.0);
        let aligned = make_rod(0.0);

        assert!(
            misaligned.is_correcting_phototropically(light_dir, &grid),
            "a rod with a real, uncorrected angular deviation from the light direction \
             must report still-correcting"
        );
        assert!(
            !aligned.is_correcting_phototropically(light_dir, &grid),
            "a rod already aligned with the light direction must NOT block sleep"
        );
    }
}

#[cfg(test)]
mod per_vertex_stiffness_tests {
    use super::*;

    /// Real backward-compat guard: `Rod::new` must fill `points.ea`/`ei` to
    /// the material's OWN scalar at every index -- a uniform-material rod's
    /// internal forces must be bit-identical to what the single-scalar
    /// `RodMaterial` path always produced, not just "close".
    #[test]
    fn rod_new_fills_uniform_stiffness_from_material_bit_identical() {
        let points = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 4.0), 5, 0.01, 1.0);
        let material = RodMaterial::new(123.0, 4.5e-3, 0.0, 0.0);
        let rod = Rod::new(points, material);

        assert_eq!(rod.points.ea.len(), 4, "N-1 edges for 5 points");
        assert_eq!(rod.points.ei.len(), 3, "N-2 bending vertices for 5 points");
        for &ea in &rod.points.ea {
            assert_eq!(
                ea, 123.0,
                "every edge must get the material's own EA exactly"
            );
        }
        for &ei in &rod.points.ei {
            assert_eq!(
                ei, 4.5e-3,
                "every bending vertex must get the material's own EI exactly"
            );
        }
    }

    /// A caller who explicitly sets non-uniform `points.ea`/`ei` BEFORE
    /// `Rod::new` gets real non-uniform stiffness -- `Rod::new` must not
    /// overwrite an already-correctly-sized array.
    #[test]
    fn rod_new_preserves_an_explicitly_set_non_uniform_array() {
        let mut points = build_straight_rod(Vec2::new(0.0, 0.0), Vec2::new(0.0, 4.0), 5, 0.01, 1.0);
        points.ea = vec![10.0, 20.0, 30.0, 40.0];
        points.ei = vec![1.0, 2.0, 3.0];
        let rod = Rod::new(points, RodMaterial::new(999.0, 999.0, 0.0, 0.0));

        assert_eq!(rod.points.ea, vec![10.0, 20.0, 30.0, 40.0]);
        assert_eq!(rod.points.ei, vec![1.0, 2.0, 3.0]);
    }

    /// The real, observable effect non-uniform stiffness should produce: a
    /// horizontal cantilever with a SOFT base half must deflect more at its
    /// arc-length midpoint under a real, constant transverse tip load than
    /// an equivalent uniformly-stiff rod carrying the exact same load -- no
    /// gravity/buckling involved (a straight rod under pure axial gravity
    /// has zero bending moment by symmetry, real physics, not useful for
    /// this comparison), a real transverse point load instead, same style
    /// `tests/accuracy.rs`'s own cantilever-deflection test already uses.
    #[test]
    fn softer_base_bends_more_than_a_uniformly_stiff_rod() {
        let length_m = 1.0;
        let n_points = 11usize;
        let ea = 1.0e5_f32;
        let ei = 50.0_f32;
        let tip_load_n = 0.02_f32;

        let make_rod = |ei_override: Option<Vec<f32>>| -> Rod {
            let mut points = build_straight_rod(
                Vec2::new(0.0, 0.0),
                Vec2::new(length_m, 0.0),
                n_points,
                0.1,
                1.0,
            );
            points.pinned[0] = 1;
            points.pinned[1] = 1;
            if let Some(ei_arr) = ei_override {
                points.ei = ei_arr;
            }
            let l0 = length_m / (n_points as f32 - 1.0);
            let point_mass = 0.1 * l0;
            let (axial_damping, bending_damping) =
                RodMaterial::critical_damping(l0, point_mass, ea, ei);
            Rod::new(
                points,
                RodMaterial::new(ea, ei, axial_damping, bending_damping),
            )
        };

        let n_bend = n_points - 2;
        let soft_ei: Vec<f32> = (0..n_bend)
            .map(|i| if i < n_bend / 2 { ei * 0.1 } else { ei })
            .collect();

        let mut uniform = make_rod(None);
        let mut soft_base = make_rod(Some(soft_ei));

        let dt = rod_cfl_dt(&uniform.points, &uniform.material, 0.4).min(rod_cfl_dt(
            &soft_base.points,
            &soft_base.material,
            0.4,
        ));
        assert!(dt.is_finite() && dt > 0.0);

        let n = n_points;
        for _ in 0..200_000 {
            for rod in [&mut uniform, &mut soft_base] {
                let mut internal = compute_internal_forces(
                    &rod.points.x,
                    &rod.points.v,
                    RodRestState {
                        rest_edge_length: &rod.points.rest_edge_length,
                        rest_curvature: &rod.points.rest_curvature,
                        ea: &rod.points.ea,
                        ei: &rod.points.ei,
                    },
                    &rod.material,
                    1.0,
                );
                internal[n - 1] += Vec2::new(0.0, tip_load_n);
                for (i, internal_force) in internal.iter().enumerate() {
                    if rod.points.pinned[i] != 0 {
                        rod.points.v[i] = Vec2::ZERO;
                        continue;
                    }
                    let a = *internal_force / rod.points.mass[i].max(1.0e-9);
                    rod.points.v[i] += a * dt;
                }
                for i in 0..n {
                    if rod.points.pinned[i] == 0 {
                        rod.points.x[i] += rod.points.v[i] * dt;
                    }
                }
            }
        }

        // Vertical deflection at the arc-length midpoint -- the real,
        // softer-base rod should have deflected further than the uniform
        // reference under the identical tip load by this point.
        let mid = n / 2;
        let uniform_deflection = (uniform.points.x[mid].y).abs();
        let soft_base_deflection = (soft_base.points.x[mid].y).abs();
        assert!(
            soft_base_deflection > uniform_deflection * 1.2,
            "a softer base should deflect measurably more at the midpoint than a uniform rod \
             under the same tip load: soft_base={soft_base_deflection:.6} uniform={uniform_deflection:.6}"
        );
    }
}

#[cfg(test)]
mod secondary_growth_integration_tests {
    use super::*;

    /// The real, concrete proof this whole chain was previously blocked on:
    /// a rod built ABOVE its own Greenhill critical height (`buckling_warning`
    /// reports genuine risk) that experiences real, sustained bending moment under its own
    /// self-weight (a tiny initial tilt breaks the perfectly-straight
    /// symmetric case, which has zero moment by construction -- same real
    /// lesson as the buckling investigation above) should,
    /// given `SecondaryGrowth`, genuinely stiffen enough over real time to
    /// raise its own critical height back above its actual height.
    #[test]
    fn sustained_bending_stress_raises_greenhill_height_above_actual_height() {
        let dx_meters = 0.01;
        let height_m = 0.10;
        let start = Vec2::new(0.0, 0.0);
        let end = Vec2::new(0.0, height_m / dx_meters);
        let n_points = 12;
        let young_modulus = 5.0e6_f32; // deliberately over-critical, same order as "blade B"
        let ea = young_modulus * 0.003 * 0.001;
        let ei = young_modulus * 0.003_f32.powi(3) * 0.001 / 12.0;

        let mut points = build_straight_rod(start, end, n_points, 0.01, dx_meters);
        points.pinned[0] = 1;
        points.pinned[1] = 1;
        // Real, genuinely CURVED shape (quadratic in index, not a rigid
        // linear tilt -- three colinear points have zero discrete
        // curvature regardless of overall angle, so a rigid tilt alone
        // would give secondary growth no real moment to respond to).
        // Modest magnitude (max ~0.15 grid cells = 1.5mm at the tip,
        // comparable to the rod's own ~9mm segment length) -- a real,
        // sustained bend a mature stem could plausibly hold under wind
        // load, not a violent distortion that would itself dominate the
        // dynamics. This test verifies `SecondaryGrowth`'s OWN response to
        // a sustained real moment directly (`x` held fixed, no `step_rod`
        // dynamics) -- the genuine buckling INSTABILITY's own real-seconds-
        // scale dynamics are already covered by
        // `tests/rod_gravitropism_whole_organ.rs`'s negative-control test.
        for (i, p) in points.x.iter_mut().enumerate() {
            let t = i as f32 / (n_points - 1) as f32;
            p.x += t * t * 0.15;
        }
        let material = RodMaterial::new(ea, ei, 0.0, 0.0);
        let rod = Rod::new(points, material);

        assert!(
            rod.buckling_warning(9.81).is_some(),
            "this rod must start genuinely over its own critical height"
        );

        let secondary_growth = SecondaryGrowth::new(1.0e-2, 1.0e-7, 0.0, 1.0e9);
        let dt = 0.05_f32;
        let mut rod = rod;
        let mut raised = false;
        for _ in 0..20_000 {
            apply_secondary_growth(&mut rod.points, &secondary_growth, dx_meters, dt);
            if rod.buckling_warning(9.81).is_none() {
                raised = true;
                break;
            }
        }

        assert!(
            raised,
            "sustained bending stress under SecondaryGrowth must eventually raise this rod's \
             own Greenhill critical height above its actual height -- final weakest ei={:?}",
            rod.points.ei.iter().cloned().fold(f32::INFINITY, f32::min)
        );
    }
}
