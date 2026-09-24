//! Taxonomy for how physics plugs into the MLS-MPM substep, and one real,
//! generic registration surface over it.
//!
//! `do_substep`'s phase order (P2G / grid update / G2P / a separate diffuse
//! step) is not a software choice -- it's the actual discretization of the
//! method (Stomakhin et al. 2013; Jiang, Schroeder, Selle, Teran, Stomakhin
//! 2015/2016, APIC), so [`Stage`] is fixed, not reorderable. It's also, in
//! the numerical-PDE sense, an operator split (Strang/Lie splitting: advance
//! a multi-physics system by applying each process's own operator in
//! sequence rather than solving one coupled system) -- this engine already
//! names its diffusion pass this way (`step.rs`'s accumulate-then-flush
//! comment).
//!
//! Inside that split there are two real, different kinds of operator, not
//! one:
//!
//! - [`StageOp`] -- a stateless correction or source term, for a NEW kind of
//!   physics that doesn't fit any existing mechanism's shape yet. NOT what
//!   `BoundaryCondition`/`Field` become: those are already correctly shaped
//!   per-cell/per-particle operators dispatched by their own existing loops,
//!   and forcing them through a whole-state `apply` would cost real cache
//!   locality for no real duplication removed. `StageOp` is a genuine,
//!   currently-empty extension point.
//! - [`CoupledBody`] -- a coupled sub-solver with its OWN persistent degrees
//!   of freedom (rod points, grain center/spin) and its OWN dynamics (the
//!   discrete elastic rod PDE -- Bergou et al. 2008; DEM contact -- Cundall
//!   & Strack 1979), exchanging momentum with the grid only at scatter/
//!   gather. Structurally the same pattern as partitioned fluid-structure
//!   interaction / the immersed boundary method (Peskin): a structure solve
//!   and a fluid solve, each independently integrated, exchanging boundary
//!   data.
//!
//! [`IntoSimEntry`] unifies REGISTRATION only -- every existing
//! mechanism keeps its own real formulas, storage, and dispatch loop
//! untouched; only how it gets attached to a [`Simulation`] collapses from
//! six-plus separately-named methods to `Simulation::add`/`with`.

use crate::boundary::BoundaryCondition;
use crate::fields::Field;
use crate::grains::population::GrainPopulation;
use crate::grid::Grid;
use crate::particle::Particles;
use crate::rod::Rod;
use crate::solver::Simulation;
use crate::solver::config::SimConfig;
use crate::thermodynamics::{ScalarDiffusionField, ThermalDiffusion};

/// Real, physics-fixed stage of the MLS-MPM operator split. See module doc
/// for why this is fixed rather than a reorderable graph.
///
/// `Diffuse` has a different real cadence from the other four, verified
/// against `step.rs` before this was written: `Scatter`/`GridCorrect`/
/// `Gather`/`PostGather` run once per substep, inside `do_substep`.
/// `Diffuse` accumulates dt across every substep and fires once per
/// `step()` call, after the substep loop ends -- diffusion's stability
/// limit is far laxer than the acoustic CFL driving substep size, so
/// per-substep application was pure waste (a real, already-fixed cost: see
/// `step.rs`'s own accumulate-then-flush comment).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Scatter,
    GridCorrect,
    Gather,
    PostGather,
    Diffuse,
}

/// Shared mutable access a [`StageOp`]/[`CoupledBody`] needs. Same trust
/// model `BoundaryCondition`/`Field` already use: implementors stay within
/// their documented contract, not enforced by the type system.
pub struct OperatorCtx<'a> {
    pub grid: &'a mut Grid,
    pub particles: &'a mut Particles,
    pub config: &'a SimConfig,
}

/// A NEW kind of physics that doesn't fit an existing mechanism's shape.
/// See module doc for why `BoundaryCondition`/`Field` don't implement this.
pub trait StageOp: Send + Sync {
    fn stage(&self) -> Stage;
    fn apply(&mut self, ctx: &mut OperatorCtx, dt: f32);
    /// Same role `MaterialModel::timestep_bound`/`rod_cfl_dt` play. Default
    /// = doesn't constrain the timestep.
    fn stable_dt(&self, _cfl_coefficient: f32) -> f32 {
        f32::INFINITY
    }
}

/// A coupled sub-solver with its own persistent state and dynamics. See
/// module doc for the partitioned-FSI grounding.
pub trait CoupledBody: Send + Sync {
    fn stages(&self) -> &'static [Stage];
    fn scatter(&mut self, _ctx: &mut OperatorCtx, _dt: f32) {}
    fn gather(&mut self, _ctx: &mut OperatorCtx, _dt: f32) {}
    fn post_gather(&mut self, _ctx: &mut OperatorCtx, _dt: f32) {}
    fn stable_dt(&self, _cfl_coefficient: f32) -> f32 {
        f32::INFINITY
    }
}

/// One real registration entry point for every existing mechanism plus
/// [`StageOp`]/[`CoupledBody`]. Each impl below delegates to that
/// mechanism's own existing method -- no internal storage or dispatch
/// changes for `BoundaryCondition`, `Field`, `ThermalDiffusion`, or
/// `ScalarDiffusionField`.
pub trait IntoSimEntry {
    /// `add_rod`/`add_grain_population` return a real index existing
    /// callers use; an associated type (not `()`) so `Simulation::add`
    /// doesn't silently drop that for the two mechanisms where it matters.
    type Output;
    fn register(self, sim: &mut Simulation) -> Self::Output;
}

impl IntoSimEntry for Box<dyn BoundaryCondition> {
    type Output = ();
    fn register(self, sim: &mut Simulation) {
        sim.add_boundary_condition(self);
    }
}

impl IntoSimEntry for Box<dyn Field> {
    type Output = ();
    fn register(self, sim: &mut Simulation) {
        sim.add_force_field(self);
    }
}

impl IntoSimEntry for ThermalDiffusion {
    type Output = ();
    fn register(self, sim: &mut Simulation) {
        sim.set_thermal(self);
    }
}

impl IntoSimEntry for ScalarDiffusionField {
    type Output = ();
    fn register(self, sim: &mut Simulation) {
        sim.attach_scalar_field(self);
    }
}

impl IntoSimEntry for Rod {
    type Output = usize;
    fn register(self, sim: &mut Simulation) -> usize {
        sim.add_rod(self)
    }
}

impl IntoSimEntry for crate::rod::RodNetwork {
    type Output = usize;
    fn register(self, sim: &mut Simulation) -> usize {
        sim.add_rod_network(self)
    }
}

impl IntoSimEntry for GrainPopulation {
    type Output = usize;
    fn register(self, sim: &mut Simulation) -> usize {
        sim.add_grain_population(self)
    }
}

impl IntoSimEntry for Box<dyn StageOp> {
    type Output = ();
    fn register(self, sim: &mut Simulation) {
        sim.stage_ops.push(self);
    }
}

impl IntoSimEntry for Box<dyn CoupledBody> {
    type Output = ();
    fn register(self, sim: &mut Simulation) {
        sim.coupled_bodies.push(self);
    }
}

/// Bevy-style bundling: a `Plugin` doesn't add a new capability, it just
/// calls `Simulation::add`/`with` (or other existing setup, e.g.
/// `add_phase_rule`) however many times a self-contained feature needs --
/// so a caller ships it as one `add_plugin(...)` instead of knowing the
/// internal call count.
///
/// NOT dynamically loaded/hot-swappable -- unlike "plugin" in most other
/// software contexts (browser extensions, VST, a DLL loaded at runtime),
/// this is ordinary, statically-linked, compile-time Rust. The name matches
/// Bevy's own specific, narrower usage (a struct that bundles registration
/// calls) only -- said explicitly so the name doesn't imply more than it is.
pub trait Plugin {
    fn build(self: Box<Self>, sim: &mut Simulation);
}

impl Simulation {
    /// Mutator form, mirrors `add_rod`/`add_grain_population` -- returns
    /// whatever that mechanism's own registration returns (an index for
    /// rod/grain, `()` otherwise).
    pub fn add<T: IntoSimEntry>(&mut self, x: T) -> T::Output {
        x.register(self)
    }

    /// Chainable builder form, mirrors `with_boundary`/`with_rod` --
    /// discards the return value, same as those existing methods.
    pub fn with<T: IntoSimEntry>(mut self, x: T) -> Self {
        let _ = self.add(x);
        self
    }

    pub fn add_plugin(&mut self, plugin: impl Plugin + 'static) {
        Box::new(plugin).build(self);
    }
}

#[cfg(test)]
mod into_sim_entry_tests {
    use super::*;
    use crate::boundary::SlipBoundary;
    use crate::materials::solid::granular::grain_contact_law::ContactLawConfig;
    use crate::particle::{Grain, Particle};
    use crate::rod::{RodMaterial, build_straight_rod};
    use crate::solver::SimConfig;
    use crate::thermodynamics::{ScalarDiffusionConfig, ThermalConfig};
    use glam::Vec2;

    fn contact_config() -> ContactLawConfig {
        ContactLawConfig {
            normal_stiffness: 1.0e5,
            tangential_stiffness: 0.8e5,
            rolling_stiffness: 5.0e3,
            normal_damping: 50.0,
            tangential_damping: 50.0,
            rolling_damping: 50.0,
            friction: 0.5,
            rolling_friction: 0.1,
        }
    }

    fn make_rod(dx_meters: f32) -> Rod {
        let points = build_straight_rod(
            Vec2::new(9.0, 4.0),
            Vec2::new(9.0, 10.0),
            5,
            0.01,
            dx_meters,
        );
        Rod::new(points, RodMaterial::new(1.0e3, 1.0e-3, 0.0, 0.0))
    }

    /// `sim.add(x)` on `Box<dyn BoundaryCondition>` must reach the exact
    /// same field `add_boundary_condition` does -- real parity check, not
    /// "compiles".
    #[test]
    fn add_boundary_matches_add_boundary_condition() {
        let mut sim = Simulation::empty(SimConfig::earth(32, 0.01, 0.02));
        let before = sim.boundaries.len();
        sim.add(Box::new(SlipBoundary::new(2)) as Box<dyn BoundaryCondition>);
        assert_eq!(sim.boundaries.len(), before + 1);
    }

    /// `sim.add(rod)` must return the SAME real index `add_rod` returns,
    /// not `()` -- the whole point of `IntoSimEntry::Output`.
    #[test]
    fn add_rod_returns_the_real_index() {
        let mut sim = Simulation::empty(SimConfig::earth(32, 0.01, 0.02));
        let idx = sim.add(make_rod(0.01));
        assert_eq!(idx, 0);
        assert_eq!(sim.rods().len(), 1);
        let idx2 = sim.add(make_rod(0.01));
        assert_eq!(idx2, 1);
    }

    /// Same real-index check for grains.
    #[test]
    fn add_grain_population_returns_the_real_index() {
        let mut sim = Simulation::empty(SimConfig::earth(32, 0.01, 0.02));
        let grain = Grain::new(Vec2::new(10.0, 10.0), 1.0, 1.0);
        let idx = sim.add(GrainPopulation::new(vec![grain], contact_config()));
        assert_eq!(idx, 0);
        assert_eq!(sim.grain_populations().len(), 1);
    }

    #[test]
    fn add_thermal_and_scalar_field_reach_their_real_fields() {
        let mut sim = Simulation::empty(SimConfig::earth(32, 0.01, 0.02));
        assert!(sim.thermal_config_mut().is_none());
        sim.add(ThermalDiffusion::new(ThermalConfig::default(), 32));
        assert!(sim.thermal_config_mut().is_some());

        fn get(p: &Particle) -> f32 {
            p.scalar_field
        }
        fn set(p: &mut Particle, v: f32) {
            p.scalar_field += v;
        }
        assert_eq!(sim.scalar_fields.len(), 0);
        sim.add(ScalarDiffusionField::new(
            ScalarDiffusionConfig::default(),
            get,
            set,
            32,
        ));
        assert_eq!(sim.scalar_fields.len(), 1);
    }

    /// The chainable `with` form must discard the index (matching
    /// `with_rod`'s existing behavior) but still register correctly.
    #[test]
    fn with_chains_and_discards_the_index() {
        let sim = Simulation::empty(SimConfig::earth(32, 0.01, 0.02))
            .with(make_rod(0.01))
            .with(Box::new(SlipBoundary::new(2)) as Box<dyn BoundaryCondition>);
        assert_eq!(sim.rods().len(), 1);
        assert_eq!(sim.boundaries.len(), 2); // default SlipBoundary + the added one
    }

    struct TwoRodsPlugin;
    impl Plugin for TwoRodsPlugin {
        fn build(self: Box<Self>, sim: &mut Simulation) {
            sim.add(make_rod(0.01));
            sim.add(make_rod(0.01));
        }
    }

    /// `add_plugin` must bundle multiple real `add` calls behind one entry
    /// point -- the actual "less LOC, fewer calls" claim, checked, not
    /// assumed.
    #[test]
    fn add_plugin_bundles_multiple_registrations() {
        let mut sim = Simulation::empty(SimConfig::earth(32, 0.01, 0.02));
        sim.add_plugin(TwoRodsPlugin);
        assert_eq!(sim.rods().len(), 2);
    }
}
