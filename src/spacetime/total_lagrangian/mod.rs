//! An implicit step for an elastic solid that keeps its energy and both
//! momenta exactly, at any substep (measured by the conserving gates of
//! `gates.rs`).
//!
//! Source, read on the document itself: Love and Sulsky, *An
//! Unconditionally Stable, Energy-Momentum Consistent Implementation of
//! the Material-Point Method*, HPC@UNM technical report 2004-002 (the
//! report of CMAME 195:3903-3925, 2006, which was not read). Their
//! algorithm, equation by equation:
//!
//! - The grid velocity solves the consistent-mass projection `M v_n = sum_p
//!   N_A(x_p) m_p v_p` (eq. 28), `M_AB = sum_p m_p N_A N_B` (eq. 24), or
//!   the effective mass `(1 - eps) M + eps M_lumped` (eq. 26), positive
//!   definite for `eps > 0` where the consistent mass can be singular
//!   (their remark 3.1.4).
//! - The grid moves by an implicit midpoint step, `M (v_{n+1} - v_n) / dt +
//!   F_int - F_ext = 0` with `u = dt v_{n+1/2}` (eq. 29), the internal
//!   force `F_int,A = sum_p V0_p F_{n+1/2} S F_n^T grad_n N_A(x_p)` (eq.
//!   30a), each particle's `F_{n+1} = (I + grad_n u) F_n` (eq. 31).
//! - The stress `S` is the discrete gradient of the stored energy, `S =
//!   S~ + 2 (W(C_{n+1}) - W(C_n) - S~ : dC / 2) dC / |dC|^2` (eq. 54),
//!   with `S~` the stress at the average strain (eq. 56), so that `S : dC
//!   / 2 = W(C_{n+1}) - W(C_n)` exactly (eq. 50).
//! - The particles take the grid's changes only: `x += u(x_p)`, `v +=
//!   (v_{n+1} - v_n)(x_p)` (eq. 32).
//!
//! With `eps = 0` the particles' total energy and both momenta are then
//! kept exactly (their corollaries 3.9, 3.11, 3.13); with `eps > 0` the
//! energy falls by `1/2 dv . (M_eps - M) dv` per step (eq. 68).
//!
//! Choices of this engine, not of the source:
//! - The grid is collocated, every node on a cell corner carrying both
//!   velocity components (`lattices`), as the source's, with the engine's
//!   quadratic B-spline (`transfer`) where the source uses bilinear
//!   elements; the mass matrix is block-diagonal by component.
//! - The nonlinear equations are solved by Newton's method with a
//!   Jacobian-free GMRES, as the source's PETSc solve does (quasi-Newton
//!   with matrix-free GMRES, their section 4), here with central
//!   differences, a banded preconditioner and every quantity in f64.
//! - The energy is the engine's elastic solid, the corotated shear energy
//!   of the isochoric deformation and `K / 2 (J - 1)^2`, written as a
//!   function of `C` through `F = sqrt(C)`.
//! - The grid stays on the bodies' reference configuration (the total
//!   Lagrangian form of the same momentum balance, `Div P + b = rho0
//!   u_tt`), and walls and bodies meet by contact: the law of
//!   `forces::boundary::signorini_coulomb`, held over each step on the
//!   velocity (`step_total_lagrangian_contact`). The updated Lagrangian
//!   step, whose walls held grid faces, is retired: those walls made the
//!   bounce blow up on both grids (measured), and the two steps
//!   discretized one balance twice.

use crate::forces::boundary::signorini_coulomb::{ContactBranch, contact_branch, contact_force};
use crate::matter::materials::solid::j2_hencky::{HenckyJ2, PlasticMetric};
use crate::matter::materials::viscous_stress::{newtonian_dissipation, newtonian_viscous_stress};
use glam::{DMat2, DVec2};
use rayon::prelude::*;

mod band;
mod basis;
mod contact;
#[cfg(all(test, feature = "experimental"))]
mod diagnostics;
#[cfg(all(test, feature = "experimental"))]
mod dirichlet;
mod elastic;
/// A one-solve local error estimate, measured beside the controller's
/// doubling, never deciding a step.
#[cfg(all(test, feature = "experimental"))]
pub(crate) mod estimate;
#[cfg(all(test, feature = "experimental"))]
mod galerkin2;
/// Probe hooks, off unless a probe turns them on: a per-thread trace of
/// every solve's Newton iterations, and a Newton homotopy (`R(u) - (1 -
/// lambda) R(0)` from `u = 0`, lambda from 0 to 1) that tells whether the
/// step's system has an admissible solution the solve failed to reach.
#[cfg(all(test, feature = "experimental"))]
pub(crate) mod hooks;
mod krylov;
mod modes;
mod multipliers;
mod relaxation;
mod solve;
mod step;
mod step_control;
#[cfg(all(test, feature = "experimental"))]
mod tests;
mod weighted_gap;

pub(crate) use band::*;
pub(crate) use basis::*;
pub use contact::*;
#[cfg(all(test, feature = "experimental"))]
pub(crate) use diagnostics::*;
#[cfg(all(test, feature = "experimental"))]
pub(crate) use dirichlet::*;
pub use elastic::*;
#[cfg(all(test, feature = "experimental"))]
pub(crate) use galerkin2::*;
pub(crate) use krylov::*;
pub(crate) use modes::*;
pub(crate) use multipliers::*;
pub use relaxation::Relaxation;
pub(crate) use relaxation::{CarriedMemory, Relaxed};
pub(crate) use solve::*;
pub use step::*;
pub use step_control::*;
pub(crate) use weighted_gap::*;

/// The grid the step lives on: `nx` by `ny` cells of width `dx`, cell
/// `(0, 0)` at the origin of the engine's grid coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GridLayout {
    pub nx: usize,
    pub ny: usize,
    pub dx: f32,
}

impl GridLayout {
    pub fn new(nx: usize, ny: usize, dx: f32) -> Self {
        assert!(nx > 0 && ny > 0, "a grid needs at least one cell");
        assert!(dx > 0.0, "cell width must be positive");
        Self { nx, ny, dx }
    }
}

/// The step's parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConservingSettings {
    /// `eps` of the effective mass (eq. 26): 0 is the consistent mass.
    pub epsilon: f64,
    /// Newton stops once the residual's largest entry is below this
    /// fraction of the inertial term's (`|2 M v_n / dt|`), or of the
    /// external force's when that is larger.
    pub newton_tolerance: f64,
    pub max_newton: u32,
    /// GMRES stops at this fraction of the Newton residual.
    pub gmres_tolerance: f64,
    pub gmres_restart: usize,
    pub max_gmres: u32,
    /// The occupation parameter `C_c` of the extended B-splines (`Grid`);
    /// `None` keeps the original B-splines.
    pub occupation: Option<f64>,
    /// How far a contact may end beyond its plane or segment when the
    /// multipliers' (Uzawa) iteration stops, as a fraction of its own
    /// relative motion over the step (never below the positions'
    /// round-off). The contact's work over a step is the force times that
    /// remaining gap, so this is the fraction of the contact's own work
    /// left as error, whatever the step's length.
    pub contact_tolerance: f64,
    /// Hold a contact from the step in which its free-flight gap closes
    /// (`step_total_lagrangian_contact`), not only once it is crossed.
    pub forecast_activation: bool,
}

impl Default for ConservingSettings {
    fn default() -> Self {
        Self {
            epsilon: 0.0,
            newton_tolerance: 1e-10,
            max_newton: 30,
            gmres_tolerance: 1e-6,
            gmres_restart: 60,
            max_gmres: 2000,
            // Kakouris et al.'s value in their one-dimensional examples.
            occupation: Some(0.75),
            // An absolute gap of 1e-8 cells took 1.2e-5 of a bar's energy
            // over 49 contact steps and 1e-12 took 2.9e-9 (measured,
            // `probe_bar_impact`): 1e-12 was 1.5e-9 of that step's own
            // motion. Being relative, it holds at any step length.
            contact_tolerance: 1e-9,
            forecast_activation: false,
        }
    }
}

/// Wall time of a step's parts, in seconds, accumulated: always on (one
/// clock read per part, tens of nanoseconds against the work it brackets),
/// so a profile of the path is a run, not new code.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Timing {
    /// The residual's evaluations: Newton's, the line search's, the starts'.
    pub residual: f64,
    /// The Jacobian-vector products (the internal force's derivative and
    /// the push's), inside GMRES.
    pub jacobian: f64,
    /// Building the particles' linearisations the products use, once per
    /// Newton iterate (`solve`'s `Linearised`).
    pub linearisation: f64,
    /// Of `jacobian`, the push's derivative (central differences of the
    /// walls' push with the active set frozen).
    pub push_derivative: f64,
    /// Of `jacobian`, the products the multipliers' responses took (their
    /// own GMRES solves, `responses`).
    pub responses_jacobian: f64,
    /// Building and factoring the preconditioner, and applying it.
    pub preconditioner_build: f64,
    pub preconditioner_solve: f64,
    /// GMRES in all (Newton's and the responses'), its products and
    /// preconditioner included: the algebra alone is this less those two.
    pub gmres: f64,
    /// The multipliers' response solves (`Response`), their GMRES
    /// included (counted in `gmres` too).
    pub responses: f64,
    /// The contact passes outside the solves: geometry, Uzawa's updates,
    /// the conditions joining.
    pub contact: f64,
    /// The multipliers' semismooth Newton steps.
    pub multipliers: f64,
    /// Building the grid, and the particles' update after the solve.
    pub grid: f64,
    pub finishing: f64,
    /// The step controller outside its tries' steps (the estimate, the
    /// states' copies).
    pub controller: f64,
    /// How many times the internal force was evaluated (one pass over the
    /// particles each): the residual's and the Jacobian products' alike,
    /// the products by central differences taking two each. A count, kept
    /// with the times it explains.
    pub internal_evaluations: u64,
    /// The Jacobian band factors built (`Factors::jacobian`), each an
    /// `approximate_jacobian`.
    pub factorizations: u64,
}

impl Timing {
    /// Adds `other`'s parts to these.
    pub fn add(&mut self, other: &Timing) {
        self.residual += other.residual;
        self.jacobian += other.jacobian;
        self.linearisation += other.linearisation;
        self.push_derivative += other.push_derivative;
        self.responses_jacobian += other.responses_jacobian;
        self.preconditioner_build += other.preconditioner_build;
        self.preconditioner_solve += other.preconditioner_solve;
        self.gmres += other.gmres;
        self.responses += other.responses;
        self.contact += other.contact;
        self.multipliers += other.multipliers;
        self.grid += other.grid;
        self.finishing += other.finishing;
        self.controller += other.controller;
        self.internal_evaluations += other.internal_evaluations;
        self.factorizations += other.factorizations;
    }
}

/// How a step ended.
#[derive(Clone, Copy, Debug, Default)]
pub struct StepReport {
    pub newton: u32,
    pub gmres: u32,
    /// Largest residual entry over the tolerance's scale when Newton
    /// stopped.
    pub residual: f64,
    /// The most energy the step's unconverged residual can move,
    /// `sum_k |u_k R_k(u)|`: the midpoint step's identity is `dE = u .
    /// R(u)` (the discrete gradient makes the internal force's work the
    /// stored energy's change exactly), so its equations held to Newton's
    /// tolerance bound the step's energy error by this.
    pub residual_work: f64,
    /// The unconverged residual's net force per axis, N/m: the internal
    /// forces sum to zero (the basis's partition of unity), so the step's
    /// momentum change differs from the external and wall impulses by `dt`
    /// times this, up to `force_round_off`.
    pub residual_force: DVec2,
    /// The round-off of the internal forces' sum, N/m: `16 eps` times the
    /// sum over unknowns of the stiffness's size there (the residual floor's
    /// terms, `solve`, summed instead of maximised).
    pub force_round_off: f64,
    /// The most work the contacts' normal forces can do over the step,
    /// `sum_v lambda_v a_v` with `a_v` each condition's tolerance on the
    /// gap it leaves (`step_total_lagrangian_contact`): held in velocity,
    /// they would do none.
    pub contact_work_bound: f64,
    pub converged: bool,
    /// Newton stopped on a residual that no longer decreased (`solve`):
    /// not converged, the equations held to that residual's noise.
    pub stagnated: bool,
    /// The energy the effective mass took out of the step, `1/2 dv .
    /// (M_eps - M) dv` (Love and Sulsky eq. 68), per unit density.
    pub stabilisation_loss: f64,
    /// Solves the walls' release logic took (`step`).
    pub passes: u32,
    /// Of those passes, the ones whose multipliers took a semismooth
    /// Newton step (`newton_on_multipliers`) instead of Uzawa's.
    pub multiplier_steps: u32,
    /// The grid velocity's projection (eq. 28) residual over its right-hand
    /// side: the step keeps the particles' energy only as far as it solves.
    pub projection: f64,
    /// The step's energy ledger, per unit density: the particles' kinetic
    /// energy change, the grid's (`1/2 v M v`), and the stored energy's
    /// change (`sum V0 W`); without external forces the three cancel.
    pub ledger: (f64, f64, f64),
    /// The most particles held on a wall's plane at once (`step`).
    pub contacts: usize,
    /// The step's work, per unit density: the walls' push on the held
    /// particles and the weight's. With the ledger, the step's energy
    /// balance: the kinetic and stored changes sum to these two.
    pub work: (f64, f64),
    /// Wall time spent, in seconds: building the grid, the projection, the
    /// preconditioner, Newton's iterations, the particles' update (every
    /// solve of the walls' passes summed).
    pub phases: [f64; 5],
    /// The same time by part, finer (`Timing`).
    pub timing: Timing,
    /// Newton's start, the better of its two, as its largest residual entry
    /// over the tolerance's scale (`solve`).
    pub start_residual: f64,
    /// The largest residual entry Newton's stop guarantees, N/m: the
    /// tolerance times the residual's scale, or the round-off floor when
    /// that is larger (`solve`).
    pub residual_allowed: f64,
    /// The largest grid velocity the projection gave (eq. 28), cells/s: on
    /// a near singular mass it can exceed every particle's speed.
    pub grid_speed: f64,
    /// The walls' total push at the step's end, the sum of their
    /// multipliers, per unit density (`step_total_lagrangian_contact`).
    pub push: f64,
    /// The heat friction made over the step, the opposite of its work,
    /// per unit density (`step_total_lagrangian_contact`).
    pub friction: f64,
    /// The energy the solids' Kelvin-Voigt dashpots dissipated over the
    /// step, turned into heat with friction's
    /// (`Elastic::viscous_dissipation_over`).
    pub viscous: f64,
    /// The energy the solids' plastic flow dissipated over the step (Love
    /// and Sulsky eq. 74 times each particle's volume), turned into heat with
    /// friction's (`Elastic::committed`).
    pub plastic: f64,
    /// The heat the solids' relaxation branches made over the step (Liu and
    /// Guan's `eta |dGamma|^2 / (2 dt)` times each particle's volume,
    /// `Relaxation::committed`), per unit density.
    pub relaxation: f64,
    /// The largest Green strain `|C - I| / 2` a particle with branches
    /// ended the step at, and how many such particles ended it beyond their
    /// record's validity limit (`RelaxationSpectrum::strain_limit`): the
    /// branches are applied there outside their measured data, reported,
    /// never silently. Only the strain is checkable: the band a record was
    /// measured over is not, a step's frequency content being unknown.
    pub relaxation_strain: f64,
    pub relaxation_beyond: u32,
    /// The most iterations Norton's return took at a particle of the kept
    /// step (`j2_hencky::norton_return`; zero for any other law).
    pub norton_iterations: u32,
    /// The particle-steps of the kept step whose Norton flow left what its
    /// record covers (`j2_hencky::Beyond`), counted where the flow's heat
    /// exceeds the step's residual work: slower than the record's rates,
    /// faster, or under a tensile mean stress.
    pub norton_slower: u32,
    pub norton_faster: u32,
    pub norton_tension: u32,
}

/// The particles' state the step reads and writes.
pub struct Particles<'a> {
    pub x: &'a mut [DVec2],
    pub v: &'a mut [DVec2],
    pub f: &'a mut [DMat2],
    pub mass: &'a [f64],
    pub v0: &'a [f64],
}

/// Grid faces held at a prescribed zero velocity, `Some(inward normal
/// sign)` along their own component: the Dirichlet boundary of the
/// manufactured-solution probes. Not a wall: walls are contacts
/// (`ContactSet`).
pub type Held<'a> = &'a dyn Fn(usize, usize, usize) -> Option<f64>;
