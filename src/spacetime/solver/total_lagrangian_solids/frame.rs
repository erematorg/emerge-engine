//! What a frame on the total Lagrangian path reports to its caller.

use crate::spacetime::total_lagrangian::Timing;
use glam::DVec2;

/// What the last frame did, for the caller's checks: the controlled steps
/// it took and rejected, the steps taken at the estimator's floor (its
/// failure, not an accuracy reached), the solids' energy (kinetic, stored,
/// gravity's) at its end, the heat friction made since the start, the
/// energy at the start, and the balance `energy + heat - start`; every
/// energy and work in J per metre of depth.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TotalLagrangianFrame {
    pub steps: u32,
    pub rejected: u32,
    pub at_floor: u32,
    pub energy: f64,
    /// The heat friction made, counted in `f64` when made. The particles'
    /// temperatures lag it by their `f32` debt below the explicit path's
    /// smallest applied rise: the ledger is exact, the temperature is not
    /// yet all of it.
    pub heat: f64,
    pub start_energy: f64,
    /// The energy the step tolerance is relative to (`tolerance_energy`),
    /// J/m: invariant under a translation of the scene, unlike
    /// `start_energy`, whose gravity part is measured from the grid's
    /// corner.
    pub tolerance_energy: f64,
    pub balance: f64,
    /// This frame's work of the friction and of the contacts' normal push
    /// (the steps' own accounts, to compare with the heat deposited and the
    /// energy's change).
    pub friction_work: f64,
    pub contact_work: f64,
    /// This frame's energy dissipated by the solids' Kelvin-Voigt
    /// dashpots, J/m, counted in `heat`.
    pub dissipated: f64,
    /// The energy's parts at the frame's end: kinetic, stored, gravity's.
    pub parts: [f64; 3],
    /// Newton's iterations, GMRES's iterations, and the kept steps Newton
    /// left unconverged.
    pub newton: u32,
    pub gmres: u32,
    /// Kept steps whose Newton stagnated on its residual's noise: a
    /// reported failure (`StepReport::stagnated`).
    pub stagnated: u32,
    pub unconverged: u32,
    /// Tries rejected because Newton did not converge, and kept steps
    /// where a point reached a wall (Acary's order-0 estimate).
    pub rejected_unconverged: u32,
    pub reaching: u32,
    /// The kept steps' contact passes (Uzawa's and the multipliers'
    /// Newton's) and the most contacts any of them held.
    pub passes: u32,
    pub contacts: usize,
    /// The frame's bound on the energy its steps' solves can leave
    /// unbalanced, J/m: their residuals' (`StepReport::residual_work`) and
    /// their contacts' normal work (`StepReport::contact_work_bound`). The
    /// balance can drift by their sum per frame and no more.
    pub residual_work: f64,
    pub contact_work_bound: f64,
    /// The frame's wall time, s, and its parts summed over every step's
    /// tries (`Timing`): a profile per frame, always on.
    pub wall: f64,
    pub timing: Timing,
    /// The frame's plastic dissipation, J/m (turned into heat).
    pub plastic_work: f64,
    /// The frame's heat from the relaxation branches, J/m
    /// (`StepReport::relaxation`); the largest Green strain a particle with
    /// branches ended a kept step at, and the particle-steps beyond their
    /// record's validity limit (`StepReport::relaxation_beyond`).
    pub relaxation_heat: f64,
    pub relaxation_strain: f64,
    pub relaxation_beyond: u32,
    /// The most iterations Norton's return took at a particle in the
    /// frame's kept steps (`j2_hencky::norton_return`).
    pub norton_iterations: u32,
    /// The frame's particle-steps whose Norton flow left what its record
    /// covers (`StepReport::norton_slower` and the others): slower than its
    /// measured rates, faster, or under a tensile mean stress.
    pub norton_slower: u32,
    pub norton_faster: u32,
    pub norton_tension: u32,
    /// The particles' angular momentum about their centre of mass at the
    /// frame's end, kg m^2/s per metre of depth, from the path's own `f64`
    /// state; and their moment of inertia about it, kg m^2 per metre.
    pub angular_momentum: f64,
    pub moment_of_inertia: f64,
    /// The particles' momentum at the frame's end and at the start, kg m/s
    /// per metre of depth, from the path's own `f64` state.
    pub momentum: DVec2,
    pub start_momentum: DVec2,
    /// What this frame's solves can leave of the momentum unbalanced, per
    /// axis, N s/m: each kept step's `dt |StepReport::residual_force|`;
    /// and their round-off, `dt StepReport::force_round_off`. The momentum
    /// can drift by their sum, beyond the external and wall impulses.
    pub residual_impulse: DVec2,
    pub impulse_round_off: f64,
    /// Whether the frame was a flight carried by the body's modes, with no
    /// step (`TotalLagrangianConfig::modal_flight`); and then its energy's
    /// change under the material's own law, J/m, which the linear modes do
    /// not keep (the law's nonlinearity; zero in a stepped frame).
    pub modal: bool,
    pub modal_energy_change: f64,
    /// Since the start: the frames flown by the modes, and the flights
    /// ended at a frame the body could touch a wall (handed to the step).
    pub modal_frames: u64,
    pub hand_backs: u64,
    /// Whether the frame was a rest carried by the constrained modes, with
    /// no step (`TotalLagrangianConfig::modal_rest`; its energy change in
    /// `modal_energy_change`); and since the start, the rest frames and the
    /// rests ended (handed to the step).
    pub modal_rest: bool,
    pub modal_rest_frames: u64,
    pub rest_hand_backs: u64,
    /// Since the start, the frames a lone body was free to fly
    /// (`TotalLagrangianConfig::modal_flight`) but had no modes to fly by; the
    /// step took the frame. The sum of the three causes below.
    pub modal_refused: u64,
    /// Of those, a law the modes do not carry: a dashpot, Norton's flow,
    /// J2 with relaxation branches, relaxation branches that do not
    /// decouple on them (two spectra in one body, a record whose shear and
    /// P weights differ).
    pub modal_refused_law: u64,
    /// Of those, a J2 body that has yielded (its stress-free configuration
    /// is no longer the modes' reference).
    pub modal_refused_yielded: u64,
    /// Of those, a flight whose strain bound could pass a damped record's
    /// strain limit or a J2 particle's yield (`HenckyJ2::elastic_within`).
    pub modal_refused_bound: u64,
    /// Since the start, the damped frames whose computed propagator
    /// lengthened a mode's state (the exact one never does; `flight`), handed
    /// to the step.
    pub modal_check_failed: u64,
    /// Since the start, the rests a body with relaxation branches was refused
    /// (`TotalLagrangianConfig::modal_rest`): branches the constrained modes
    /// do not carry, or a strain that could pass the record's limit.
    pub modal_rest_refused: u64,
}
