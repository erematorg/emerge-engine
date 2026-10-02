//! A discrete-grain population: rigid circular bodies (position and planar
//! spin) integrated by semi-implicit Euler, with contacts resolved by
//! `contact_law`'s force law.
//!
//! This type integrates the grains themselves. Coupling to the shared MPM
//! grid lives in `coupling.rs` and the packing-fraction oracle that decides
//! where grains are needed in `oracle.rs`.
//!
//! Contacts are a flat, growable `Vec<ActiveContact>` rebuilt each substep
//! by brute-force O(n^2) neighbour detection, the flat contact list shape of
//! GeoTaichi's `cplist` without its fixed GPU capacity. Brute force suits a
//! thin enrichment layer of grains; revisit if profiling shows it is the
//! bottleneck.

use glam::{Mat2, Vec2};

use crate::forces::boundary::BoundaryCondition;
use crate::forces::fields::GrainField;
use crate::matter::materials::granular::disc_contact::{self, DiscContactConfig};
use crate::matter::materials::granular::grain_contact_law::{
    ContactLawConfig, ContactSpring, GrainContactState, HertzianContactConfig,
    critical_timestep_hertzian, resolve_contact_pair, resolve_contact_pair_disc,
    resolve_contact_pair_hertzian, resolve_wall_contact, resolve_wall_contact_disc,
    resolve_wall_contact_hertzian,
};
use crate::matter::particle::Grain;
use crate::spacetime::integration::advance_position;

/// Which real contact force law a `GrainPopulation` resolves every contact
/// through -- `Linear` (Cundall & Strack 1979, constant stiffness, the
/// original and still-default model, right for granular/sand material) or
/// `Hertzian` (nonlinear, contact-patch-dependent stiffness, right for
/// smooth hard bodies -- see `HertzianContactConfig`'s doc). A real,
/// additive capability, not a breaking change: `GrainPopulation::new` keeps
/// its exact original signature and wraps its `ContactLawConfig` as
/// `Linear` internally, so every existing call site across this codebase
/// compiles unchanged.
#[derive(Clone, Copy, Debug)]
pub enum ContactModel {
    Linear(ContactLawConfig),
    Hertzian(HertzianContactConfig),
    /// The 2D grain contract: discs of unit depth in line contact (see
    /// `disc_contact`), for grains built with `Grain::from_si`.
    Disc2D(DiscContactConfig),
}

/// One currently-active contact pair, with its own persistent elastic
/// spring history. `i < j` always (canonical ordering -- avoids storing the
/// same pair twice or ever comparing a grain against itself).
#[derive(Clone, Copy, Debug)]
struct ActiveContact {
    i: usize,
    j: usize,
    spring: ContactSpring,
}

/// A discrete-grain population. See the module doc for its scope.
pub struct GrainPopulation {
    pub grains: Vec<Grain>,
    contacts: Vec<ActiveContact>,
    /// Persistent per-grain wall-contact spring state (real elastic-plastic
    /// memory, same role as `contacts` above but for grain-vs-boundary
    /// contact instead of grain-grain -- see `resolve_wall_contact_forces`'s
    /// doc). Resized lazily to match `grains.len()` rather than kept in
    /// sync at every push site -- indices beyond the current length are
    /// just treated as a fresh (zeroed) spring the first time they're used.
    wall_springs: Vec<ContactSpring>,
    /// Persistent per-grain terrain-contact spring state -- same real
    /// role as `wall_springs` above, but for the dynamic MPM
    /// terrain surface estimated by `terrain_contact::terrain_grain_
    /// contact` instead of a static `BoundaryCondition`. See
    /// `resolve_terrain_contact_forces`'s doc for why this exists as
    /// a separate mechanism.
    terrain_springs: Vec<ContactSpring>,
    /// Per-grain rounding residual of the position update
    /// (`spacetime::integration::advance_position`), resized lazily like
    /// `wall_springs`. A contact's stable step moves a slow grain by less
    /// than the f32 spacing at its coordinate, and without this the lost
    /// part keeps the grain from converging as the step shrinks.
    pub(super) position_compensation: Vec<Vec2>,
    pub config: ContactModel,
    /// Sweeps of `resolve_contact_forces`'s iterative relaxation (see that
    /// function). `1` via `new`/`new_hertzian` is the single pass; raise it
    /// with `with_contact_iterations` for simultaneous multi-body contact
    /// chains (several balls released together in a cradle row).
    pub contact_iterations: usize,
    /// External body forces (drag, wind, anything shaped like `GrainField`)
    /// applied on top of gravity and contact forces every `step`; empty by
    /// default. A standalone population bypasses `Simulation`'s `Field`
    /// pipeline, so this is its own hook (see `GrainField` for why it is a
    /// separate trait).
    pub grain_fields: Vec<Box<dyn GrainField>>,
    /// Running accumulator for the discrete-to-continuum stress
    /// (Christoffersen, Mehrabadi & Nemat-Nasser 1981; Bagi 1996:
    /// `sigma_ij = (1/A) * sum_contacts f_i^c * l_j^c`, branch vector `l`
    /// centre to centre), summed in `resolve_contact_forces`'s pair loop.
    /// `reset_stress_accum` starts a fresh averaging window, e.g. once a
    /// population has settled and its impact contacts should not count.
    stress_accum: Mat2,
    /// Active-pair contributions folded into `stress_accum` since the last
    /// reset, per substep and contact (not distinct pairs): the "enough
    /// contact data yet" gate of `effective_friction_angle_deg`.
    stress_accum_samples: usize,
    /// Terrain-contact configuration: `None` (via `new`/`new_hertzian`)
    /// never attempts terrain contact. `Some((reference_mass_per_cell,
    /// surface_threshold))`, set with `with_terrain_contact`, both derived by
    /// the caller from its scene (the terrain's per-particle mass; the
    /// packing-fraction cutoff of `grains::oracle::needs_discrete_treatment`).
    /// See `resolve_terrain_contact_forces`.
    terrain_contact_config: Option<(f32, f32)>,
    /// How many contact sub-steps the last grid-coupled substep took (see
    /// `grains::coupling::apply_grain_contact_forces`); 0 before any.
    last_contact_substeps: usize,
}

/// Outer product `a (x) b` as a 2x2 matrix (`M*v = a*(b.dot(v))`): the
/// per-contact term of the discrete stress sum.
fn outer_product(a: Vec2, b: Vec2) -> Mat2 {
    Mat2::from_cols(a * b.x, a * b.y)
}

/// Closed-form eigenvalues of a symmetric 2x2 matrix `[[a,b],[b,d]]`, as
/// `(largest, smallest)`: trace/2 +/- sqrt(((a-d)/2)^2 + b^2).
fn symmetric_2x2_eigenvalues(a: f32, b: f32, d: f32) -> (f32, f32) {
    let mean = (a + d) * 0.5;
    let half_diff = (a - d) * 0.5;
    let radius = (half_diff * half_diff + b * b).sqrt();
    (mean + radius, mean - radius)
}

impl GrainPopulation {
    pub const fn new(grains: Vec<Grain>, config: ContactLawConfig) -> Self {
        Self {
            grains,
            contacts: Vec::new(),
            wall_springs: Vec::new(),
            position_compensation: Vec::new(),
            terrain_springs: Vec::new(),
            config: ContactModel::Linear(config),
            contact_iterations: 1,
            grain_fields: Vec::new(),
            stress_accum: Mat2::ZERO,
            stress_accum_samples: 0,
            terrain_contact_config: None,
            last_contact_substeps: 0,
        }
    }

    /// Hertzian (nonlinear) contact model -- see `ContactModel`/
    /// `HertzianContactConfig`.
    pub const fn new_hertzian(grains: Vec<Grain>, config: HertzianContactConfig) -> Self {
        Self {
            grains,
            contacts: Vec::new(),
            wall_springs: Vec::new(),
            position_compensation: Vec::new(),
            terrain_springs: Vec::new(),
            config: ContactModel::Hertzian(config),
            contact_iterations: 1,
            grain_fields: Vec::new(),
            stress_accum: Mat2::ZERO,
            stress_accum_samples: 0,
            terrain_contact_config: None,
            last_contact_substeps: 0,
        }
    }

    /// The 2D grain contract's entry point: grains from `Grain::from_si`,
    /// contacts through `DiscContactConfig`'s line contact.
    pub const fn new_disc(grains: Vec<Grain>, config: DiscContactConfig) -> Self {
        Self {
            grains,
            contacts: Vec::new(),
            wall_springs: Vec::new(),
            position_compensation: Vec::new(),
            terrain_springs: Vec::new(),
            config: ContactModel::Disc2D(config),
            contact_iterations: 1,
            grain_fields: Vec::new(),
            stress_accum: Mat2::ZERO,
            stress_accum_samples: 0,
            terrain_contact_config: None,
            last_contact_substeps: 0,
        }
    }

    /// The largest step this population's contacts stay stable at: its
    /// contact model's stiffest contact on its lightest grain, through
    /// `disc_contact::critical_step` (the explicit limit of a damped
    /// oscillator, `omega dt <= 2 (sqrt(1 + zeta^2) - zeta)`). Linear: the
    /// normal, tangential and rolling springs with their own dashpots.
    /// Hertzian: `critical_timestep_hertzian`, its own worst-case bound. 2D
    /// contract: the line contact's tangent stiffness never exceeds
    /// `1 / sum_i (1 / (pi E_i'))` (see `disc_contact`), `pi E' / 2` between
    /// two discs on their reduced mass `m / 2` and `pi E'` against a wall on
    /// `m`, one frequency `sqrt(pi E' / m)`; the tangential spring is
    /// `tangential_ratio` of it. `INFINITY` without grains.
    pub fn contact_step_limit(&self) -> f32 {
        let lightest = self
            .grains
            .iter()
            .map(|g| g.mass)
            .fold(f32::INFINITY, f32::min);
        if !lightest.is_finite() {
            return f32::INFINITY;
        }
        let smallest = self
            .grains
            .iter()
            .map(|g| g.radius)
            .fold(f32::INFINITY, f32::min);
        let least_inertia = self
            .grains
            .iter()
            .map(Grain::moment_of_inertia)
            .fold(f32::INFINITY, f32::min);
        // A spring with its dashpot, on an effective mass (or inertia).
        let channel = |stiffness: f32, damping: f32, mass: f32| {
            if stiffness <= 0.0 {
                return f32::INFINITY;
            }
            let zeta = damping / (2.0 * (stiffness * mass).sqrt());
            disc_contact::critical_step(stiffness, mass, zeta)
        };
        match &self.config {
            ContactModel::Linear(cfg) => {
                let m_eff = 0.5 * lightest;
                channel(cfg.normal_stiffness, cfg.normal_damping, m_eff)
                    .min(channel(
                        cfg.tangential_stiffness,
                        cfg.tangential_damping,
                        m_eff,
                    ))
                    .min(channel(
                        cfg.rolling_stiffness,
                        cfg.rolling_damping,
                        0.5 * least_inertia,
                    ))
            }
            ContactModel::Hertzian(cfg) => {
                critical_timestep_hertzian(0.5 * lightest, smallest, cfg).min(channel(
                    cfg.rolling_stiffness,
                    cfg.rolling_damping,
                    0.5 * least_inertia,
                ))
            }
            ContactModel::Disc2D(cfg) => {
                let stiffness = std::f32::consts::PI
                    * cfg.elastic.plane_strain_modulus()
                    * cfg.tangential_ratio().max(1.0);
                disc_contact::critical_step(stiffness, lightest, cfg.damping_ratio).min(channel(
                    cfg.rolling_stiffness,
                    cfg.rolling_damping,
                    0.5 * least_inertia,
                ))
            }
        }
    }

    /// How many contact sub-steps the last grid-coupled substep took.
    pub const fn last_contact_substeps(&self) -> usize {
        self.last_contact_substeps
    }

    pub(crate) const fn set_last_contact_substeps(&mut self, substeps: usize) {
        self.last_contact_substeps = substeps;
    }

    /// Opts this population into K real iterative-relaxation sweeps per
    /// substep for `resolve_contact_forces` -- see that function's doc.
    /// `k=1` (the default) is a no-op (bit-identical to not calling this).
    pub fn with_contact_iterations(mut self, contact_iterations: usize) -> Self {
        self.contact_iterations = contact_iterations;
        self
    }

    /// Opt-in grain-vs-continuum-terrain contact (see
    /// `resolve_terrain_contact_forces`; separate from grain-vs-boundary
    /// contact). The caller derives both values from its scene:
    /// `reference_mass_per_cell` from the terrain's per-particle mass
    /// (`grains::oracle::reference_mass_per_cell`), `surface_threshold` from
    /// the packing-fraction cutoff `grains::oracle::needs_discrete_treatment`
    /// takes. No default is picked here.
    pub fn with_terrain_contact(
        mut self,
        reference_mass_per_cell: f32,
        surface_threshold: f32,
    ) -> Self {
        self.terrain_contact_config = Some((reference_mass_per_cell, surface_threshold));
        self
    }

    /// Adds one real external body force (e.g. `LinearDragField`) applied
    /// every `step`, on top of gravity and contact forces -- see
    /// `grain_fields`'s doc.
    pub fn with_grain_field(mut self, field: impl GrainField + 'static) -> Self {
        self.grain_fields.push(Box::new(field));
        self
    }

    /// Number of currently resolved contacts (diagnostics, tests).
    pub const fn active_contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// Active-pair contributions in the stress accumulator since the last
    /// `reset_stress_accum`, so callers can tell a near-empty accumulator
    /// from one worth reading.
    pub const fn stress_accum_sample_count(&self) -> usize {
        self.stress_accum_samples
    }

    /// Clears the running stress accumulator (Bagi/Christoffersen contact
    /// sum + sample count) to start a fresh averaging window -- e.g. once a
    /// population has visibly settled and older, pre-settling transient
    /// contact data should not pollute a "current state" reading.
    pub fn reset_stress_accum(&mut self) {
        self.stress_accum = Mat2::ZERO;
        self.stress_accum_samples = 0;
    }

    /// Discrete-to-continuum effective internal friction angle (degrees)
    /// from this population's accumulated contact forces: the area-averaged
    /// stress of Christoffersen, Mehrabadi & Nemat-Nasser 1981 / Bagi 1996
    /// (`stress_accum` over total grain area times sample count, grain area
    /// standing in for the representative area as `Particle::volume` does
    /// elsewhere), symmetrized (a contact sum need not be symmetric per
    /// contact, the averaged Cauchy stress is), eigen-decomposed, then
    /// Mohr-Coulomb `sin(phi) = (sigma1-sigma3)/(sigma1+sigma3)`. Compression
    /// is positive: contact normal forces push outward along the branch
    /// vector.
    ///
    /// `None` with too few sampled contacts (a near-empty accumulator is
    /// noise) or a stress state that is not compressive (`sigma1 + sigma3 <=
    /// 0`).
    pub fn effective_friction_angle_deg(&self) -> Option<f32> {
        let (sigma1, sigma3) = self.principal_stresses()?;
        let denom = sigma1 + sigma3;
        if denom <= 0.0 {
            return None;
        }
        let sin_phi = ((sigma1 - sigma3) / denom).clamp(-1.0, 1.0);
        Some(sin_phi.asin().to_degrees())
    }

    /// The raw (major, minor) principal stresses behind
    /// `effective_friction_angle_deg`, before clamp and `asin`: a reported
    /// 90 degrees (`sin_phi` saturating its clamp) can be a near-uniaxial
    /// stress state or a ratio overshooting 1 because `sigma3` came out
    /// slightly negative from discrete-sum noise; this tells which. `None`
    /// under the same conditions as `effective_friction_angle_deg`.
    pub fn principal_stresses(&self) -> Option<(f32, f32)> {
        const MIN_SAMPLES: usize = 20;
        if self.stress_accum_samples < MIN_SAMPLES {
            return None;
        }
        let total_area: f32 = self
            .grains
            .iter()
            .map(|g| std::f32::consts::PI * g.radius * g.radius)
            .sum();
        if total_area <= 0.0 {
            return None;
        }
        let norm = total_area * self.stress_accum_samples as f32;
        let sigma = self.stress_accum * (1.0 / norm);
        // Symmetrize: (sigma + sigma^T) / 2.
        let a = sigma.x_axis.x;
        let d = sigma.y_axis.y;
        let b = 0.5 * (sigma.x_axis.y + sigma.y_axis.x);
        Some(symmetric_2x2_eigenvalues(a, b, d))
    }

    /// Per-grain active contact count this substep: the coordination number
    /// (how many neighbours each grain touches). Tests in
    /// `grains_grid_coupling.rs` and `grains_repose_angle.rs` correlate it
    /// with per-grain energy.
    pub fn contact_count_per_grain(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.grains.len()];
        for c in &self.contacts {
            counts[c.i] += 1;
            counts[c.j] += 1;
        }
        counts
    }

    /// Detects contacts (carrying over persistent spring history for pairs
    /// that were ALREADY in contact last substep, matched by index -- a
    /// new pair starts with a fresh, zeroed spring, real DEM
    /// convention per `contact_law`'s doc) and resolves every pair's
    /// force/moment, returning per-grain net contact force and torque.
    /// Pure computation, no integration -- separated from `step` so grid
    /// coupling (`grains::coupling`) can apply these forces AFTER a
    /// grid-gathered velocity, exactly mirroring how `rod::advance_rod`
    /// applies a rod's internal forces after `gather_grid_to_rod`, not
    /// instead of it.
    pub fn resolve_contact_forces(&mut self, dt: f32) -> (Vec<Vec2>, Vec<f32>) {
        let n = self.grains.len();
        let k = self.contact_iterations.max(1);
        let sub_dt = dt / k as f32;

        struct Pair {
            i: usize,
            j: usize,
            spring: ContactSpring,
            active: bool,
        }
        let mut pairs: Vec<Pair> = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let gi = self.grains[i].contact_state();
                let gj = self.grains[j].contact_state();
                // Cheap reject before the contact-law geometry check --
                // avoids allocating/looking up spring history for pairs that
                // are nowhere near each other. Determined ONCE from
                // start-of-substep positions -- positions never move inside
                // this function (only velocity/spin do, see below), so this
                // stays valid across every sweep.
                let max_dist = gi.radius + gj.radius;
                if (gj.x - gi.x).length_squared() > max_dist * max_dist {
                    continue;
                }
                let spring = self
                    .contacts
                    .iter()
                    .find(|c| c.i == i && c.j == j)
                    .map(|c| c.spring)
                    .unwrap_or_default();
                pairs.push(Pair {
                    i,
                    j,
                    spring,
                    active: false,
                });
            }
        }

        let v0: Vec<Vec2> = self.grains.iter().map(|g| g.v).collect();
        let spin0: Vec<f32> = self.grains.iter().map(|g| g.spin).collect();

        // Iterative relaxation, Jacobi per sweep, K sweeps: each sweep
        // resolves every pair against velocities frozen at the sweep's start,
        // then commits all deltas together. A momentum handoff
        // (grain0 -> grain1 -> grain2) can then cross more than one contact
        // within a substep, which one sweep cannot at any stiffness
        // (`diag_newtons_cradle_two_ball_release_real_conservation_check`).
        // Same family as Box2D's sequential impulses (Catto); Jacobi rather
        // than Gauss-Seidel so `contact_iterations = 1` is exactly the single
        // pass.
        for _ in 0..k {
            let mut dv = vec![Vec2::ZERO; n];
            let mut dspin = vec![0.0f32; n];
            for pair in &mut pairs {
                let gi = self.grains[pair.i].contact_state();
                let gj = self.grains[pair.j].contact_state();
                let resolution = match &self.config {
                    ContactModel::Linear(cfg) => {
                        resolve_contact_pair(&gi, &gj, &mut pair.spring, cfg, sub_dt)
                    }
                    ContactModel::Hertzian(cfg) => {
                        resolve_contact_pair_hertzian(&gi, &gj, &mut pair.spring, cfg, sub_dt)
                    }
                    ContactModel::Disc2D(cfg) => {
                        resolve_contact_pair_disc(&gi, &gj, &mut pair.spring, cfg, sub_dt)
                    }
                };
                pair.active = resolution.is_some();
                if let Some(resolution) = resolution {
                    let force_on_j = resolution.normal_force * (gj.x - gi.x).normalize()
                        + resolution.tangential_force;
                    // Discrete-to-continuum stress term for this contact
                    // (see `stress_accum`): branch vector centre to centre,
                    // the force already computed above.
                    self.stress_accum += outer_product(force_on_j, gj.x - gi.x);
                    self.stress_accum_samples += 1;
                    dv[pair.j] += force_on_j / gj.mass * sub_dt;
                    dv[pair.i] -= force_on_j / gi.mass * sub_dt;
                    // Rolling moment: an action-reaction pair on spin.
                    // Friction torque is a separate source: the tangential
                    // force at the contact point, whose moment arm is each
                    // grain's own radius, so not an action-reaction pair.
                    // See `ContactResolution`.
                    let moi_i = self.grains[pair.i].moment_of_inertia();
                    let moi_j = self.grains[pair.j].moment_of_inertia();
                    dspin[pair.j] += (resolution.rolling_moment + resolution.friction_torque_on_j)
                        / moi_j
                        * sub_dt;
                    dspin[pair.i] += (-resolution.rolling_moment + resolution.friction_torque_on_i)
                        / moi_i
                        * sub_dt;
                }
            }
            for i in 0..n {
                self.grains[i].v += dv[i];
                self.grains[i].spin += dspin[i];
            }
        }

        // Convert the total, K-sweep velocity/spin change back into an
        // equivalent net force/torque so the caller (`grains::coupling`)
        // keeps integrating with its own existing `v += F/m*dt` unchanged
        // -- identical external contract, only the internal resolution
        // algorithm changed. This function never leaves grain state mutated
        // as a side effect (same contract as before), so restore v0/spin0
        // after reading off the equivalent force.
        let mut forces = vec![Vec2::ZERO; n];
        let mut torques = vec![0.0f32; n];
        for i in 0..n {
            forces[i] = self.grains[i].mass * (self.grains[i].v - v0[i]) / dt;
            torques[i] = self.grains[i].moment_of_inertia() * (self.grains[i].spin - spin0[i]) / dt;
            self.grains[i].v = v0[i];
            self.grains[i].spin = spin0[i];
        }

        self.contacts = pairs
            .into_iter()
            .filter(|p| p.active)
            .map(|p| ActiveContact {
                i: p.i,
                j: p.j,
                spring: p.spring,
            })
            .collect();

        (forces, torques)
    }

    /// Per-grain normal-velocity correction against a touching boundary.
    /// The grid's boundary correction (`BoundaryCondition::
    /// apply_to_grid_velocity`) acts per cell, but a grain's momentum spreads
    /// over a 3x3 kernel, several cells of which sit above the terrain and
    /// are never corrected; the gathered velocity is a blend, and given that
    /// blend `resolve_wall_contact` rolled a grain the wrong way.
    ///
    /// This sets the normal component from the exact local normal before any
    /// contact force, so friction and rolling torque see a clean velocity.
    /// The per-cell correction still runs; both push the same way and this
    /// one runs last.
    pub fn clean_wall_normal_velocity(
        &mut self,
        boundaries: &[Box<dyn BoundaryCondition>],
        grid_res: usize,
    ) {
        for grain in &mut self.grains {
            for boundary in boundaries {
                let contact = boundary.grain_contact(grain.x, grain.radius, grid_res);
                if let Some((normal, overlap)) = contact
                    && overlap > 0.0
                {
                    let v_n = grain.v.dot(normal);
                    if v_n < 0.0 {
                        grain.v -= v_n * normal;
                    }
                }
            }
        }
    }

    /// Grain-vs-boundary contact forces and torques (see
    /// `BoundaryCondition::grain_contact`): without them a grain resting on
    /// the ground never starts rolling, since only grain-grain contact gives
    /// torque. Separate from `resolve_contact_forces` because a boundary is
    /// not a `Grain` (no index, mass or spin).
    ///
    /// Elastic memory per grain (`wall_springs`): a fresh spring whenever the
    /// grain touches no boundary this substep, carried forward otherwise.
    /// One spring per grain, not per grain and boundary, so a grain in a
    /// corner shares one spring between two walls.
    pub fn resolve_wall_contact_forces(
        &mut self,
        boundaries: &[Box<dyn BoundaryCondition>],
        grid_res: usize,
        dt: f32,
    ) -> (Vec<Vec2>, Vec<f32>) {
        let n = self.grains.len();
        let mut forces = vec![Vec2::ZERO; n];
        let mut torques = vec![0.0f32; n];
        if self.wall_springs.len() < n {
            self.wall_springs.resize(n, ContactSpring::default());
        }

        for i in 0..n {
            let grain = self.grains[i];
            let gi: GrainContactState = grain.contact_state();
            let mut touched = false;
            for boundary in boundaries {
                let Some((normal, overlap)) = boundary.grain_contact(gi.x, gi.radius, grid_res)
                else {
                    continue;
                };
                let resolution = match &self.config {
                    ContactModel::Linear(cfg) => resolve_wall_contact(
                        &gi,
                        normal,
                        overlap,
                        &mut self.wall_springs[i],
                        cfg,
                        dt,
                    ),
                    ContactModel::Hertzian(cfg) => resolve_wall_contact_hertzian(
                        &gi,
                        normal,
                        overlap,
                        &mut self.wall_springs[i],
                        cfg,
                        dt,
                    ),
                    ContactModel::Disc2D(cfg) => resolve_wall_contact_disc(
                        &gi,
                        normal,
                        overlap,
                        &mut self.wall_springs[i],
                        cfg,
                        dt,
                    ),
                };
                if let Some(resolution) = resolution {
                    // Only the tangential friction force and its torque are
                    // applied, not `resolution.normal_force`: the grid's
                    // position/velocity clamp owns the normal direction, and
                    // the spring's normal force only sets the Coulomb cap.
                    // Applied as a force too, it refills every step against
                    // a clamp that keeps resetting the overlap (measured: in
                    // the hundreds every substep) and launches the grain.
                    forces[i] += resolution.tangential_force;
                    torques[i] += resolution.rolling_moment + resolution.friction_torque_on_j;
                    touched = true;
                }
            }
            if !touched {
                self.wall_springs[i] = ContactSpring::default();
            }
        }
        (forces, torques)
    }

    /// Grain-vs-continuum-terrain contact. `resolve_wall_contact_forces`
    /// only fires against a `BoundaryCondition`, so a grain resting on an MPM
    /// terrain (sharing the grid, not a boundary) got no rolling resistance
    /// from it, and that base layer sets a poured pile's footprint. See
    /// `terrain_contact`'s module doc for the normal/overlap estimate from
    /// the terrain's packing-fraction field.
    ///
    /// Like `resolve_wall_contact_forces`, applies only the tangential
    /// friction force and its torque: the grid's P2G/G2P exchange with the
    /// terrain already provides the normal repulsion, and a second normal
    /// spring would count it twice.
    pub fn resolve_terrain_contact_forces(
        &mut self,
        grid: &crate::grid::Grid,
        dt: f32,
    ) -> (Vec<Vec2>, Vec<f32>) {
        let n = self.grains.len();
        let mut forces = vec![Vec2::ZERO; n];
        let mut torques = vec![0.0f32; n];
        // Opt-in gate -- see `terrain_contact_config`.
        let Some((reference_mass_per_cell, surface_threshold)) = self.terrain_contact_config else {
            return (forces, torques);
        };
        if self.terrain_springs.len() < n {
            self.terrain_springs.resize(n, ContactSpring::default());
        }

        for i in 0..n {
            let grain = self.grains[i];
            let gi: GrainContactState = grain.contact_state();
            let Some((normal, overlap)) = super::terrain_contact::terrain_grain_contact(
                grid,
                reference_mass_per_cell,
                surface_threshold,
                gi.x,
                gi.radius,
            ) else {
                self.terrain_springs[i] = ContactSpring::default();
                continue;
            };
            let resolution = match &self.config {
                ContactModel::Linear(cfg) => resolve_wall_contact(
                    &gi,
                    normal,
                    overlap,
                    &mut self.terrain_springs[i],
                    cfg,
                    dt,
                ),
                ContactModel::Hertzian(cfg) => resolve_wall_contact_hertzian(
                    &gi,
                    normal,
                    overlap,
                    &mut self.terrain_springs[i],
                    cfg,
                    dt,
                ),
                ContactModel::Disc2D(cfg) => resolve_wall_contact_disc(
                    &gi,
                    normal,
                    overlap,
                    &mut self.terrain_springs[i],
                    cfg,
                    dt,
                ),
            };
            if let Some(resolution) = resolution {
                forces[i] += resolution.tangential_force;
                torques[i] += resolution.rolling_moment + resolution.friction_torque_on_j;
            } else {
                self.terrain_springs[i] = ContactSpring::default();
            }
        }
        (forces, torques)
    }

    /// One standalone semi-implicit Euler substep (gravity + contact
    /// forces + any `grain_fields` integrated directly into velocity/spin,
    /// then position integrated from the new velocity) -- for a
    /// `GrainPopulation` NOT coupled to the shared MPM grid (real,
    /// standard, more stable than explicit-Euler for stiff contact springs,
    /// same rationale this engine's own MLS-MPM P2G/G2P cycle already
    /// follows for the grid velocity update). Grid-coupled use goes through
    /// `grains::coupling` instead, which applies `resolve_contact_forces`'s
    /// output after a grid-gathered velocity rather than through this
    /// method (and does not yet apply `grain_fields` -- out of scope until
    /// a grid-coupled scene actually needs one).
    pub fn step(&mut self, gravity: Vec2, dt: f32) {
        let (forces, torques) = self.resolve_contact_forces(dt);
        self.position_compensation
            .resize(self.grains.len(), Vec2::ZERO);
        for (idx, (grain, compensation)) in self
            .grains
            .iter_mut()
            .zip(self.position_compensation.iter_mut())
            .enumerate()
        {
            let mut accel = gravity + forces[idx] / grain.mass;
            for field in &self.grain_fields {
                accel += field.acceleration(grain);
            }
            grain.v += accel * dt;
            let angular_accel = torques[idx] / grain.moment_of_inertia();
            grain.spin += angular_accel * dt;
            grain.orientation += grain.spin * dt;
            advance_position(&mut grain.x, compensation, grain.v * dt);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> ContactLawConfig {
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

    #[test]
    fn single_grain_free_falls_under_gravity_exactly() {
        let mut pop = GrainPopulation::new(vec![Grain::new(Vec2::ZERO, 0.5, 1.0)], config());
        let gravity = Vec2::new(0.0, -9.8);
        let dt = 0.001;
        for _ in 0..100 {
            pop.step(gravity, dt);
        }
        let expected_v = gravity.y * dt * 100.0;
        assert!(
            (pop.grains[0].v.y - expected_v).abs() < 1e-3,
            "v={} expected={}",
            pop.grains[0].v.y,
            expected_v
        );
        assert_eq!(pop.active_contact_count(), 0);
    }

    /// `effective_friction_angle_deg`'s math (eigen-decomposition and
    /// Mohr-Coulomb) against a hand-computed value, with an injected
    /// accumulator instead of the contact pipeline. The raw accumulator is
    /// asymmetric on purpose (`x_axis.y = 2`, `y_axis.x = 0`) and symmetrizes
    /// to `[[2,1],[1,2]]`: eigenvalues 3 and 1, `sin(phi) = 0.5`, `phi = 30`
    /// degrees. The raw matrix would give (2, 2) and 0 degrees, so a missing
    /// symmetrization fails here.
    #[test]
    fn effective_friction_angle_matches_hand_computed_mohr_coulomb_value() {
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::ZERO, 1.0, 1.0),
                Grain::new(Vec2::new(3.0, 0.0), 1.0, 1.0),
            ],
            config(),
        );
        let total_area = 2.0 * std::f32::consts::PI * 1.0 * 1.0; // two r=1 grains
        let samples = 20usize;
        let norm = total_area * samples as f32;
        // Raw sigma (pre-symmetrize) = [[2,0],[2,2]] (x_axis=(2,2), y_axis=(0,2)).
        pop.stress_accum = Mat2::from_cols(Vec2::new(2.0, 2.0), Vec2::new(0.0, 2.0)) * norm;
        pop.stress_accum_samples = samples;

        let phi = pop.effective_friction_angle_deg().expect(
            "20 samples at MIN_SAMPLES=20 and a genuinely compressive state must yield Some",
        );
        assert!((phi - 30.0).abs() < 1.0e-2, "phi={phi}, expected 30.0deg");

        let (sigma1, sigma3) = pop
            .principal_stresses()
            .expect("same gate as effective_friction_angle_deg -- must also yield Some here");
        assert!(
            (sigma1 - 3.0).abs() < 1.0e-3,
            "sigma1={sigma1}, expected 3.0"
        );
        assert!(
            (sigma3 - 1.0).abs() < 1.0e-3,
            "sigma3={sigma3}, expected 1.0"
        );
    }

    #[test]
    fn effective_friction_angle_is_none_below_the_minimum_sample_floor() {
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::ZERO, 1.0, 1.0),
                Grain::new(Vec2::new(3.0, 0.0), 1.0, 1.0),
            ],
            config(),
        );
        pop.stress_accum = Mat2::from_cols(Vec2::new(100.0, 0.0), Vec2::new(0.0, 100.0));
        pop.stress_accum_samples = 19; // one below MIN_SAMPLES=20
        assert_eq!(pop.effective_friction_angle_deg(), None);
    }

    #[test]
    fn effective_friction_angle_is_none_for_a_non_compressive_state() {
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::ZERO, 1.0, 1.0),
                Grain::new(Vec2::new(3.0, 0.0), 1.0, 1.0),
            ],
            config(),
        );
        // sigma1+sigma3 <= 0 -- e.g. a population that never actually
        // loaded any compressive contact, only recorded tension/noise.
        pop.stress_accum = Mat2::from_cols(Vec2::new(-1.0, 0.0), Vec2::new(0.0, -1.0));
        pop.stress_accum_samples = 20;
        assert_eq!(pop.effective_friction_angle_deg(), None);
    }

    #[test]
    fn effective_friction_angle_is_none_for_zero_total_area() {
        let mut pop = GrainPopulation::new(vec![], config());
        pop.stress_accum = Mat2::from_cols(Vec2::new(1.0, 0.0), Vec2::new(0.0, 1.0));
        pop.stress_accum_samples = 20;
        assert_eq!(pop.effective_friction_angle_deg(), None);
    }

    #[test]
    fn reset_stress_accum_zeroes_both_the_tensor_and_the_sample_count() {
        let mut pop = GrainPopulation::new(vec![Grain::new(Vec2::ZERO, 1.0, 1.0)], config());
        pop.stress_accum = Mat2::from_cols(Vec2::new(5.0, 0.0), Vec2::new(0.0, 5.0));
        pop.stress_accum_samples = 42;
        pop.reset_stress_accum();
        assert_eq!(pop.stress_accum, Mat2::ZERO);
        assert_eq!(pop.stress_accum_sample_count(), 0);
    }

    #[test]
    fn light_grain_resting_on_a_pinned_floor_reaches_the_real_predicted_equilibrium_overlap() {
        // A uniform force on both grains gives no relative acceleration, and a
        // huge mass alone still free-falls, so the floor grain is re-clamped
        // after every step (as `Particle::pinned` anchors a particle). Both
        // grains sit at order-1 coordinates: a 1e6 offset would lose the
        // 1e-4 signal to f32.
        //
        // At equilibrium kn*overlap = m*g, so overlap = m*g/kn.
        let cfg = config();
        let m = 1.0;
        let g = 9.8;
        let expected_overlap = m * g / cfg.normal_stiffness;
        let floor_anchor = Vec2::new(0.0, -1.0);
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.5), 0.5, m),
                Grain::new(floor_anchor, 0.5, 1.0e6),
            ],
            cfg,
        );
        let gravity = Vec2::new(0.0, -g);
        let dt = 0.0002;
        let mut max_speed: f32 = 0.0;
        for _ in 0..20_000 {
            pop.step(gravity, dt);
            // Pin the floor grain: real Dirichlet-anchor technique, not a
            // contact-law shortcut (see the comment above).
            pop.grains[1].x = floor_anchor;
            pop.grains[1].v = Vec2::ZERO;
            pop.grains[1].spin = 0.0;
            max_speed = max_speed.max(pop.grains[0].v.length());
        }
        assert!(
            max_speed < 50.0,
            "light grain never settled, exploded instead: max_speed={max_speed}"
        );
        let dist = (pop.grains[0].x - pop.grains[1].x).length();
        let overlap = 1.0 - dist;
        assert!(
            (overlap - expected_overlap).abs() < expected_overlap * 0.5,
            "expected overlap near {expected_overlap} (kn*overlap=m*g), got {overlap}"
        );
    }

    /// End-to-end wiring: settling dynamics (a light grain compressing onto a
    /// pinned floor grain under gravity, as in the equilibrium test above)
    /// populate `stress_accum` through `resolve_contact_forces`. No exact
    /// angle is asserted (it depends on the settling trajectory), only a
    /// sane reading once contact is sustained.
    #[test]
    fn effective_friction_angle_reads_real_nonzero_data_after_real_settling_contact() {
        let cfg = config();
        let m = 1.0;
        let g = 9.8;
        let floor_anchor = Vec2::new(0.0, -1.0);
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.5), 0.5, m),
                Grain::new(floor_anchor, 0.5, 1.0e6),
            ],
            cfg,
        );
        let gravity = Vec2::new(0.0, -g);
        let dt = 0.0002;
        for _ in 0..20_000 {
            pop.step(gravity, dt);
            pop.grains[1].x = floor_anchor;
            pop.grains[1].v = Vec2::ZERO;
            pop.grains[1].spin = 0.0;
        }
        assert!(
            pop.stress_accum_sample_count() > 0,
            "real settling contact never accumulated any stress samples -- wiring gap"
        );
        let phi = pop
            .effective_friction_angle_deg()
            .expect("a real, sustained compressive contact must yield Some");
        // A single vertical contact with no sliding is degenerate for
        // Mohr-Coulomb: the force is along the branch vector, so `outer(f, l)`
        // has rank 1, zero lateral stress, and `sin(phi) = 1`, 90 degrees.
        // Correct but uninformative, so this checks the wiring only (a
        // finite value in [0, 90]); a friction-angle gate needs a pile with
        // contacts in several directions.
        assert!(
            (0.0..=90.0).contains(&phi),
            "phi={phi}deg is outside any physically sane range"
        );
    }

    #[test]
    fn small_pile_under_gravity_settles_without_exploding() {
        // A handful of grains dropped onto a floor settle into a bounded,
        // finite configuration instead of diverging. Not a repose-angle
        // check.
        let mut grains = Vec::new();
        for row in 0..3 {
            for col in 0..4 {
                let x = col as f32 * 1.05 + (row % 2) as f32 * 0.5;
                let y = row as f32 * 1.0 + 3.0;
                grains.push(Grain::new(Vec2::new(x, y), 0.5, 1.0));
            }
        }
        // A large, heavy "floor" grain well below everything else.
        let mut floor = Grain::new(Vec2::new(2.0, -50.0), 50.0, 1.0e8);
        floor.mass = 1.0e8;
        grains.push(floor);

        let mut pop = GrainPopulation::new(grains, config());
        let gravity = Vec2::new(0.0, -9.8);
        let dt = 0.0002;
        for _ in 0..5000 {
            pop.step(gravity, dt);
        }
        for (idx, g) in pop.grains.iter().enumerate() {
            assert!(
                g.x.is_finite() && g.v.is_finite() && g.spin.is_finite(),
                "grain {idx} diverged: x={:?} v={:?} spin={}",
                g.x,
                g.v,
                g.spin
            );
            assert!(
                g.v.length() < 100.0,
                "grain {idx} velocity exploded: {:?}",
                g.v
            );
        }
    }

    #[test]
    fn two_free_grains_total_mechanical_energy_never_grows_without_an_external_driver() {
        // The two grains start overlapping by 1% of the radius (radii sum 1.0,
        // separation 0.99), so the normal spring starts with PE_n0 =
        // 0.5*kn*overlap0^2 = 5.0, more than the initial KE (4.0). Releasing
        // it raises KE (measured peak 6.571), which is correct, so kinetic
        // energy alone is not the invariant. For a damped contact with no
        // driver, total mechanical energy (KE plus spring PE) never increases:
        // d(KE+PE)/dt = -normal_damping*v_n^2 - tangential_damping*v_t^2 <= 0.
        // Measured over 2,000,000 steps, KE + spring PE + dissipated energy
        // stays within 0.58% of its start (semi-implicit Euler error for a
        // stiff damped spring).
        let mut cfg = config();
        cfg.friction = 1.0e6; // cap should never engage except right at separation
        cfg.rolling_stiffness = 0.0; // isolate normal+tangential only
        cfg.rolling_friction = 0.0;
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.0), 0.5, 1.0),
                Grain::new(Vec2::new(0.99, 0.0), 0.5, 1.0),
            ],
            cfg,
        );
        pop.grains[0].v = Vec2::new(0.0, 2.0);
        pop.grains[1].v = Vec2::new(0.0, -2.0);

        // Total mechanical energy: KE (translational and rotational) plus the
        // elastic PE in the active contact's normal and tangential springs.
        let mechanical_energy = |pop: &GrainPopulation| -> f32 {
            let ke: f32 = pop
                .grains
                .iter()
                .map(|g| {
                    0.5 * g.mass * g.v.length_squared()
                        + 0.5 * g.moment_of_inertia() * g.spin * g.spin
                })
                .sum();
            let gi = pop.grains[0];
            let gj = pop.grains[1];
            let overlap = (gi.radius + gj.radius - (gj.x - gi.x).length()).max(0.0);
            let ContactModel::Linear(cfg) = pop.config else {
                panic!("this energy-conservation test is built around the linear model")
            };
            let normal_pe = 0.5 * cfg.normal_stiffness * overlap * overlap;
            let spring = pop
                .contacts
                .iter()
                .find(|c| c.i == 0 && c.j == 1)
                .map(|c| c.spring)
                .unwrap_or_default();
            let tangential_pe = 0.5 * cfg.tangential_stiffness * spring.tangential.length_squared();
            ke + normal_pe + tangential_pe
        };

        let e0 = mechanical_energy(&pop);
        let mut max_e = e0;
        let dt = 0.0000001;
        for step in 0..2_000_000 {
            pop.step(Vec2::ZERO, dt);
            let e = mechanical_energy(&pop);
            max_e = max_e.max(e);
            assert!(
                pop.grains
                    .iter()
                    .all(|g| g.v.is_finite() && g.spin.is_finite()),
                "diverged at step {step}"
            );
        }
        // Measured peak 0.5786% overshoot; 1% leaves room for explicit Euler
        // error and still catches an energy injection (the 64% KE-only
        // reading would fail it).
        assert!(
            max_e <= e0 * 1.01,
            "total mechanical energy grew without an external driver: e0={e0:.6} max_e={max_e:.6} \
             -- with zero external driver and only dissipative (damped) contact forces, KE plus \
             energy stored in the contact springs must be monotonically non-increasing (up to \
             bounded semi-implicit-Euler numerical error), even though raw KE alone is free to \
             rise as preloaded spring PE legitimately converts into it"
        );
    }

    #[test]
    fn sliding_grain_on_a_pinned_floor_converges_toward_rolling_not_away_from_it() {
        // Friction-induced rolling (`ContactResolution::friction_torque_on_
        // i/j`): a grain sliding on a fixed floor must evolve toward rolling
        // without slipping, its contact-point slip |v_t| decaying as spin
        // builds up (measured 3.0 -> 0.9). This checks the torque's sign and
        // formula on one pair. Fixed 1% overlap and no gravity, so the contact
        // is engaged from the first step.
        let floor_anchor = Vec2::new(0.0, -0.495);
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.5), 0.5, 1.0),
                Grain::new(floor_anchor, 0.5, 1.0e6),
            ],
            config(),
        );
        pop.grains[0].v = Vec2::new(3.0, 0.0); // real sliding velocity, no spin yet
        let gravity = Vec2::ZERO;
        let dt = 0.000001;

        let slip_speed = |pop: &GrainPopulation| -> f32 {
            let g = &pop.grains[0];
            let floor = &pop.grains[1];
            let n = (g.x - floor.x).normalize();
            let t = Vec2::new(-n.y, n.x);
            let v_rel = g.v - floor.v;
            (v_rel.dot(t) - (g.radius * g.spin + floor.radius * floor.spin)).abs()
        };

        // Measured over the window real contact stays engaged
        // (checked directly: this pair separates vertically -- the normal
        // spring's own bounce -- a bit after step ~4500, at which point
        // velocity/spin freeze and any further "slip" reading is pure
        // separated-body geometric drift, not real contact physics; this
        // window is entirely within the engaged-contact regime).
        let slip_early = slip_speed(&pop);
        for _ in 0..4000 {
            pop.step(gravity, dt);
            pop.grains[1].x = floor_anchor;
            pop.grains[1].v = Vec2::ZERO;
            pop.grains[1].spin = 0.0;
        }
        let slip_late = slip_speed(&pop);

        assert!(
            pop.grains[0].v.is_finite() && pop.grains[0].spin.is_finite(),
            "diverged: v={:?} spin={}",
            pop.grains[0].v,
            pop.grains[0].spin
        );
        assert!(
            slip_late < slip_early,
            "expected slip speed to DECAY toward rolling (early={slip_early:.4} late={slip_late:.4}) \
             -- growth here means the friction torque is signed wrong, actively driving slip instead of resisting it"
        );
    }

    /// `Grain::orientation` accumulates `integral(spin dt)`, not just a
    /// nonzero spin that is never integrated. Same scenario as
    /// `sliding_grain_on_a_pinned_floor_converges_toward_rolling_not_away_
    /// from_it` (sliding on a pinned floor, friction-induced spin-up).
    #[test]
    fn grain_orientation_genuinely_accumulates_real_rotation_while_rolling() {
        let floor_anchor = Vec2::new(0.0, -0.495);
        let mut pop = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.5), 0.5, 1.0),
                Grain::new(floor_anchor, 0.5, 1.0e6),
            ],
            config(),
        );
        pop.grains[0].v = Vec2::new(3.0, 0.0);
        let gravity = Vec2::ZERO;
        let dt = 0.000001;

        assert_eq!(
            pop.grains[0].orientation, 0.0,
            "must start at zero rotation"
        );
        println!(
            "step=0 orientation={:.6} rad spin={:.4} rad/s",
            pop.grains[0].orientation, pop.grains[0].spin
        );
        for step in 1..=4000 {
            pop.step(gravity, dt);
            pop.grains[1].x = floor_anchor;
            pop.grains[1].v = Vec2::ZERO;
            pop.grains[1].spin = 0.0;
            if step % 500 == 0 {
                println!(
                    "step={step} orientation={:.6} rad ({:.2} deg) spin={:.4} rad/s",
                    pop.grains[0].orientation,
                    pop.grains[0].orientation.to_degrees(),
                    pop.grains[0].spin
                );
            }
        }
        let final_orientation = pop.grains[0].orientation;
        // At dt = 1e-6 s, 4000 steps are 4 ms, with spin ramping from 0 to
        // ~-2.4 rad/s, so the integral is about -0.005 rad. The claim is a
        // nonzero, correctly signed, finite rotation matching the spin
        // history, checked by the trapezoidal cross-check below.
        assert!(
            final_orientation.is_finite() && final_orientation != 0.0,
            "expected real, nonzero accumulated rotation from 4000 steps of \
             real friction-induced spin-up, got {final_orientation}"
        );
        // Cross-check: `orientation` must match the numerical integral
        // of `spin`, not just be "some nonzero number" -- reruns the exact
        // same physics while independently trapezoidal-integrating spin by
        // hand, then compares against the engine's own bookkeeping.
        let mut pop2 = GrainPopulation::new(
            vec![
                Grain::new(Vec2::new(0.0, 0.5), 0.5, 1.0),
                Grain::new(floor_anchor, 0.5, 1.0e6),
            ],
            config(),
        );
        pop2.grains[0].v = Vec2::new(3.0, 0.0);
        let mut independent_integral = 0.0f32;
        for _ in 0..4000 {
            let spin_before = pop2.grains[0].spin;
            pop2.step(gravity, dt);
            pop2.grains[1].x = floor_anchor;
            pop2.grains[1].v = Vec2::ZERO;
            pop2.grains[1].spin = 0.0;
            let spin_after = pop2.grains[0].spin;
            independent_integral += 0.5 * (spin_before + spin_after) * dt;
        }
        let rel_err = (pop2.grains[0].orientation - independent_integral).abs()
            / independent_integral.abs().max(1e-6);
        println!(
            "engine orientation={:.6} independent trapezoidal integral={:.6} rel_err={:.4}",
            pop2.grains[0].orientation, independent_integral, rel_err
        );
        assert!(
            rel_err < 0.01,
            "engine's own orientation bookkeeping doesn't match an independent integral of spin: \
             engine={:.6} independent={:.6}",
            pop2.grains[0].orientation,
            independent_integral
        );
    }

    /// Discs of the 2D grain contract at 1 mm cells, 1 mm radius, of a soft
    /// solid (100 kPa, nu 0.45, 1100 kg/m3): soft enough that its contact
    /// overlaps, a few 1e-4 cells, are resolved by f32 positions near the
    /// origin. Quartz under its own weight overlaps by 1.3e-9 cells, below
    /// the spacing of f32 values anywhere past a hundredth of a cell (#47).
    fn soft_disc_contract(restitution: f32) -> (Grain, DiscContactConfig) {
        use crate::matter::materials::granular::disc_contact::DiscElastic;
        let soft = crate::Elastic {
            e_pa: 1.0e5,
            nu: 0.45,
            rho_kg_m3: 1100.0,
        };
        let sim = crate::SimConfig::earth(64, 1.0e-3, 1.0e-3);
        let grain = Grain::from_si(Vec2::ZERO, 1.0e-3, &soft, &sim);
        let config = DiscContactConfig::new(
            DiscElastic::from_si(&soft, &sim),
            restitution,
            0.5,
            0.0,
            0.0,
            0.0,
        );
        (grain, config)
    }

    /// A disc resting on a pinned disc under gravity settles where the line
    /// contact carries its weight: at `approach(m g)` of two discs (the
    /// pinned one compresses too), within 1 percent.
    #[test]
    fn a_disc_contract_grain_rests_at_the_line_contacts_overlap() {
        use crate::matter::materials::granular::disc_contact::{self, ContactSide};
        let (grain, config) = soft_disc_contract(0.1);
        let g = 9810.0; // 9.81 m/s^2 at 1 mm cells
        let bottom = Grain {
            x: Vec2::new(0.0, -grain.radius),
            ..grain
        };
        let top = Grain {
            x: Vec2::new(0.0, grain.radius),
            ..grain
        };
        let mut pop = GrainPopulation::new_disc(vec![bottom, top], config);
        let side = ContactSide::Disc {
            radius: grain.radius,
            elastic: config.elastic,
        };
        let expected = disc_contact::approach(grain.mass * g, side, side);
        let (_, stiffness) = disc_contact::force_and_stiffness(expected, side, side);
        let dt = 0.1 * disc_contact::critical_step(stiffness, grain.mass, config.damping_ratio);
        for _ in 0..20_000 {
            pop.step(Vec2::new(0.0, -g), dt);
            pop.grains[0].x = bottom.x;
            pop.grains[0].v = Vec2::ZERO;
        }
        let overlap = 2.0 * grain.radius - (pop.grains[1].x.y - bottom.x.y);
        assert!(
            ((overlap - expected) / expected).abs() < 0.01,
            "overlap {overlap:e} cells against the line contact's {expected:e}"
        );
    }

    /// The restitution two equal discs part with, from `v0` cells/s, for a
    /// configured `e`; and their momentum ratio.
    fn disc_contract_collision(e: f32, v0: f32) -> (f32, f32) {
        use crate::matter::materials::granular::disc_contact::{self, ContactSide};
        let (grain, config) = soft_disc_contract(e);
        let side = ContactSide::Disc {
            radius: grain.radius,
            elastic: config.elastic,
        };
        let m_eff = 0.5 * grain.mass;
        // The kinetic energy bounds the deepest overlap and its stiffness.
        let (_, stiffness) = disc_contact::force_and_stiffness(0.1, side, side);
        let dt = 0.02 * disc_contact::critical_step(stiffness, m_eff, config.damping_ratio);
        let gap = 1.0e-3;
        let left = Grain {
            x: Vec2::new(-grain.radius - 0.5 * gap, 0.0),
            v: Vec2::new(v0, 0.0),
            ..grain
        };
        let right = Grain {
            x: Vec2::new(grain.radius + 0.5 * gap, 0.0),
            ..grain
        };
        let mut pop = GrainPopulation::new_disc(vec![left, right], config);
        let mut touched = false;
        for _ in 0..10_000_000 {
            pop.step(Vec2::ZERO, dt);
            let gap = pop.grains[1].x.x - pop.grains[0].x.x - 2.0 * grain.radius;
            touched |= gap < 0.0;
            if touched && gap > 0.0 {
                break;
            }
        }
        assert!(touched, "the discs never met");
        let (a, b) = (pop.grains[0].v.x, pop.grains[1].v.x);
        ((b - a) / v0, (a + b) / v0)
    }

    /// Two equal discs meeting head on keep momentum exactly and part with
    /// the restitution they were given within 5 percent, at 1 cm/s. The
    /// damping inverts Schwager and Poschel's eq. 21, exact for a linear
    /// spring; what is left is the line contact's logarithm. Measured
    /// (`disc_contract_restitution_table`, at 1, 10 and 100 cells/s): e 0.1
    /// gives 0.097, 0.096, 0.094; 0.3 gives 0.294, 0.292, 0.288; 0.5 gives
    /// 0.491, 0.492, 0.489; 0.9 gives 0.899, 0.898, 0.896.
    #[test]
    fn a_disc_contract_collision_keeps_momentum_and_restitution() {
        for e in [0.1f32, 0.3, 0.5, 0.9] {
            let (restitution, momentum) = disc_contract_collision(e, 10.0);
            assert!((momentum - 1.0).abs() < 1.0e-4, "momentum ratio {momentum}");
            assert!(
                (restitution - e).abs() < 0.05 * e,
                "restitution {restitution} for {e}"
            );
        }
    }

    /// Diagnostic: the restitution the line contact's damping gives, over
    /// configured values and impact speeds.
    #[test]
    #[ignore = "diagnostic: run with --ignored --nocapture"]
    fn disc_contract_restitution_table() {
        for e in [0.1f32, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99] {
            let row: Vec<String> = [1.0f32, 10.0, 100.0]
                .iter()
                .map(|&v0| {
                    let (restitution, momentum) = disc_contract_collision(e, v0);
                    assert!((momentum - 1.0).abs() < 1.0e-4, "momentum ratio {momentum}");
                    format!("{restitution:.4}")
                })
                .collect();
            println!("e {e}: at 1, 10, 100 cells/s {}", row.join(", "));
        }
    }
}
