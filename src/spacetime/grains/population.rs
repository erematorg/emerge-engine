//! A standalone, self-contained discrete-grain population -- rigid circular
//! bodies (2D: position + planar spin, matching this engine's own 2D
//! convention throughout) integrated via real semi-implicit Euler, with
//! contacts resolved by `contact_law`'s real, cited force law.
//!
//! This type integrates the grains themselves. Coupling to the shared MPM
//! grid lives in `coupling.rs` and the packing-fraction oracle that decides
//! where grains are needed in `oracle.rs`.
//!
//! Real, disclosed simplification for CPU-first correctness (per this
//! project's own standing "CPU correctness first, GPU port second" rule):
//! contacts are tracked as a flat, growable `Vec<ActiveContact>`, rebuilt
//! each substep via brute-force O(n^2) neighbor detection, not the
//! fixed-size-per-grain bounded array a real GPU port would need (matching
//! GeoTaichi's own real `cplist` precedent, which uses exactly this
//! flat-contact-list SHAPE, just with GPU-parallelization-driven fixed
//! capacity -- the same real technique, not a different one). Brute-force
//! neighbor detection is the correct, simple choice for THIS population's
//! expected scale (a thin enrichment layer, not the whole domain's particle
//! count) -- revisit only if a real profiling number shows it's the
//! bottleneck, per this project's own "measure before optimizing" rule.

use glam::{Mat2, Vec2};

use crate::forces::boundary::BoundaryCondition;
use crate::forces::fields::GrainField;
use crate::matter::materials::granular::grain_contact_law::{
    ContactLawConfig, ContactSpring, GrainContactState, HertzianContactConfig,
    resolve_contact_pair, resolve_contact_pair_hertzian, resolve_wall_contact,
    resolve_wall_contact_hertzian,
};
use crate::matter::particle::Grain;

/// Which real contact force law a `GrainPopulation` resolves every contact
/// through -- `Linear` (Cundall & Strack 1979, constant stiffness, the
/// original and still-default model, right for granular/sand material) or
/// `Hertzian` (nonlinear, contact-patch-dependent stiffness, right for
/// smooth hard bodies -- see `HertzianContactConfig`'s own doc). A real,
/// additive capability, not a breaking change: `GrainPopulation::new` keeps
/// its exact original signature and wraps its `ContactLawConfig` as
/// `Linear` internally, so every existing call site across this codebase
/// compiles unchanged.
#[derive(Clone, Copy, Debug)]
pub enum ContactModel {
    Linear(ContactLawConfig),
    Hertzian(HertzianContactConfig),
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

/// A standalone discrete-grain population. See module doc for real, disclosed
/// scope (not grid-coupled yet, brute-force neighbor detection).
pub struct GrainPopulation {
    pub grains: Vec<Grain>,
    contacts: Vec<ActiveContact>,
    /// Persistent per-grain wall-contact spring state (real elastic-plastic
    /// memory, same role as `contacts` above but for grain-vs-boundary
    /// contact instead of grain-grain -- see `resolve_wall_contact_forces`'s
    /// own doc). Resized lazily to match `grains.len()` rather than kept in
    /// sync at every push site -- indices beyond the current length are
    /// just treated as a fresh (zeroed) spring the first time they're used.
    wall_springs: Vec<ContactSpring>,
    /// Persistent per-grain terrain-contact spring state -- same real
    /// role as `wall_springs` above, but for the real, dynamic MPM
    /// terrain surface estimated by `terrain_contact::terrain_grain_
    /// contact` instead of a static `BoundaryCondition`. See
    /// `resolve_terrain_contact_forces`'s own doc for why this exists as
    /// a genuinely separate mechanism.
    terrain_springs: Vec<ContactSpring>,
    pub config: ContactModel,
    /// Real, opt-in sweep count for `resolve_contact_forces`'s own
    /// iterative relaxation -- see that function's doc for the real
    /// technique/citation. Defaults to `1` (today's exact single-pass
    /// behavior, zero blast radius) via `new`/`new_hertzian`; opt into
    /// K>1 real sweeps via `with_contact_iterations` for scenes with
    /// simultaneous multi-body contact chains (e.g. releasing >1 grain
    /// together in a Newton's-cradle-style row).
    pub contact_iterations: usize,
    /// Real, opt-in external body forces (drag, wind, anything shaped like
    /// `GrainField`) applied on top of gravity + contact forces every
    /// `step`. Empty by default -- zero blast radius for every existing
    /// scene. A standalone `GrainPopulation` bypasses `Simulation`'s own
    /// `Field` pipeline entirely (see module doc: not grid-coupled yet), so
    /// this is its own equivalent hook rather than a duplicated one-off
    /// function per demo -- see `GrainField`'s own doc for why it's a
    /// separate trait from `Field` instead of reusing it directly.
    pub grain_fields: Vec<Box<dyn GrainField>>,
    /// Real, opt-in running accumulator for the discrete-to-continuum
    /// stress mapping (Christoffersen, Mehrabadi & Nemat-Nasser 1981 /
    /// Bagi 1996: `sigma_ij = (1/A) * sum_contacts f_i^c * l_j^c`, branch
    /// vector `l` taken center-to-center between the two contacting
    /// grains). Updated inside `resolve_contact_forces`'s own existing
    /// per-pair loop (the force this population already computes there,
    /// just also summed here) -- zero cost/behaviour change for every
    /// scene that never reads `effective_friction_angle_deg`. Reset via
    /// `reset_stress_accum` to start a fresh averaging window (e.g. once a
    /// population has settled and its prior transient/impact contacts
    /// should not pollute a "current state" reading).
    stress_accum: Mat2,
    /// Real, cumulative count of active-pair contributions folded into
    /// `stress_accum` since the last reset -- a per-substep, per-contact
    /// count (not a distinct-pair count), used only as a "have we sampled
    /// enough real contact data yet" gate for `effective_friction_angle_deg`.
    stress_accum_samples: usize,
    /// Real, opt-in terrain-contact configuration -- `None` (the default
    /// via `new`/`new_hertzian`, zero cost/behavior change) means this
    /// population never attempts real terrain contact at all. `Some((
    /// reference_mass_per_cell, surface_threshold))` opts in explicitly,
    /// via `with_terrain_contact` -- both real values the CALLER derives
    /// from its own actual scene (the terrain's own real per-particle
    /// mass; the packing-fraction cutoff, same real, scene-tunable
    /// convention `grains::oracle::needs_discrete_treatment` already
    /// uses), never guessed or hardcoded by this struct itself. See
    /// `resolve_terrain_contact_forces`'s own doc for the real mechanism
    /// this unlocks.
    terrain_contact_config: Option<(f32, f32)>,
}

/// Real outer product `a (x) b` as a 2x2 matrix (`M*v = a*(b.dot(v))` for
/// any `v`) -- the per-contact term Bagi's/Christoffersen's discrete
/// stress formula sums over all active contacts.
fn outer_product(a: Vec2, b: Vec2) -> Mat2 {
    Mat2::from_cols(a * b.x, a * b.y)
}

/// Real closed-form eigenvalues of a symmetric 2x2 matrix `[[a,b],[b,d]]`,
/// returned as `(largest, smallest)`. Standard formula (trace/2 +/-
/// sqrt(((a-d)/2)^2 + b^2)) -- not an iterative solver, exact for 2x2.
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
            terrain_springs: Vec::new(),
            config: ContactModel::Linear(config),
            contact_iterations: 1,
            grain_fields: Vec::new(),
            stress_accum: Mat2::ZERO,
            stress_accum_samples: 0,
            terrain_contact_config: None,
        }
    }

    /// Real, additive entry point for the Hertzian (nonlinear) contact
    /// model -- see `ContactModel`/`HertzianContactConfig`'s own doc.
    pub const fn new_hertzian(grains: Vec<Grain>, config: HertzianContactConfig) -> Self {
        Self {
            grains,
            contacts: Vec::new(),
            wall_springs: Vec::new(),
            terrain_springs: Vec::new(),
            config: ContactModel::Hertzian(config),
            contact_iterations: 1,
            grain_fields: Vec::new(),
            stress_accum: Mat2::ZERO,
            stress_accum_samples: 0,
            terrain_contact_config: None,
        }
    }

    /// Opts this population into K real iterative-relaxation sweeps per
    /// substep for `resolve_contact_forces` -- see that function's own doc.
    /// `k=1` (the default) is a no-op (bit-identical to not calling this).
    pub fn with_contact_iterations(mut self, contact_iterations: usize) -> Self {
        self.contact_iterations = contact_iterations;
        self
    }

    /// Real, explicit opt-in for grain-vs-CONTINUUM-terrain contact (see
    /// `resolve_terrain_contact_forces`'s own doc for the full mechanism
    /// and why it's a genuinely separate concern from grain-vs-boundary
    /// contact). Both arguments are real values the CALLER must derive
    /// from its own actual scene -- `reference_mass_per_cell` from the
    /// terrain's own real per-particle mass (`grains::oracle::
    /// reference_mass_per_cell`), `surface_threshold` from the same real,
    /// scene-tunable packing-fraction cutoff `grains::oracle::
    /// needs_discrete_treatment` already exposes as a caller parameter --
    /// this method never picks a default value on the caller's behalf.
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
    /// `grain_fields`'s own doc.
    pub fn with_grain_field(mut self, field: impl GrainField + 'static) -> Self {
        self.grain_fields.push(Box::new(field));
        self
    }

    /// Real number of currently-resolved contacts -- diagnostic/test use.
    pub const fn active_contact_count(&self) -> usize {
        self.contacts.len()
    }

    /// Real, cumulative active-pair-contribution count folded into the
    /// stress accumulator since the last `reset_stress_accum` -- exposed so
    /// callers (and tests) can tell a fresh/near-empty accumulator from one
    /// with enough real contact data to trust, without duplicating the
    /// threshold `effective_friction_angle_deg` itself uses.
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

    /// Real discrete-to-continuum effective internal friction angle
    /// (degrees), derived from THIS population's own actual accumulated
    /// contact forces -- Christoffersen, Mehrabadi & Nemat-Nasser 1981 /
    /// Bagi 1996's area-averaged discrete stress
    /// (`sigma_ij = (1/A) * sum_contacts f_i^c * l_j^c`, `stress_accum`'s
    /// own running sum divided here by this population's own total grain
    /// area x sample count -- a real, disclosed proxy for the true
    /// representative area, matching this codebase's existing
    /// `Particle::volume`-as-2D-footprint convention used elsewhere, e.g.
    /// `Simulation::enrich_region_into_grain`), symmetrized (real discrete
    /// contact sums need not be exactly symmetric per contact even though
    /// the true averaged Cauchy stress is, by angular-momentum balance --
    /// standard practice, Bagi's own paper addresses this the same way),
    /// then closed-form 2x2 eigen-decomposed into principal stresses and
    /// converted via the standard Mohr-Coulomb relation
    /// `sin(phi) = (sigma1-sigma3)/(sigma1+sigma3)` (compression-positive
    /// convention: contact normal forces are purely repulsive/outward along
    /// the branch vector, so this accumulator is already compression-
    /// positive by construction, matching soil-mechanics convention).
    ///
    /// `None` when too few real contacts have been sampled yet (an
    /// arbitrary but disclosed floor, not zero -- a near-empty accumulator
    /// is noise, not a measurement) or when the resulting stress state
    /// isn't genuinely compressive (`sigma1 + sigma3 <= 0`, e.g. a
    /// population that never actually loaded any contacts).
    pub fn effective_friction_angle_deg(&self) -> Option<f32> {
        let (sigma1, sigma3) = self.principal_stresses()?;
        let denom = sigma1 + sigma3;
        if denom <= 0.0 {
            return None;
        }
        let sin_phi = ((sigma1 - sigma3) / denom).clamp(-1.0, 1.0);
        Some(sin_phi.asin().to_degrees())
    }

    /// Real diagnostic entry point, exposing the RAW (major, minor)
    /// principal stresses `effective_friction_angle_deg` derives its
    /// Mohr-Coulomb angle from -- pre-clamp, pre-`asin`. Added 2026-09-14
    /// specifically to distinguish a genuine physical plateau from a
    /// numerical-clamp artifact: `effective_friction_angle_deg` reporting a
    /// suspiciously exact 90deg (`sin_phi` saturating its `[-1,1]` clamp)
    /// could mean a real, degenerate (near-uniaxial) stress state, OR it
    /// could mean the raw ratio is silently overshooting past 1.0 (e.g.
    /// `sigma3` slightly negative from real discrete-sum noise) and the
    /// clamp is masking that distinction -- this method lets a caller see
    /// which. `None` under the exact same conditions
    /// `effective_friction_angle_deg` returns `None` (too few samples, zero
    /// total area).
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

    /// Per-grain active contact count this substep -- the real
    /// COORDINATION NUMBER, a standard granular-physics measure (how many
    /// neighbours each grain is actually touching), not a debug counter.
    ///
    /// Originally added for the 2026-08-03 scale-residual investigation and
    /// labelled temporary; kept as permanent API because it is genuinely
    /// meaningful on its own and four real tests consume it
    /// (`grains_grid_coupling.rs`, `grains_repose_angle.rs`) to correlate
    /// contact count with per-grain energy behaviour.
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
    /// genuinely new pair starts with a fresh, zeroed spring, real DEM
    /// convention per `contact_law`'s own doc) and resolves every pair's
    /// force/moment, returning per-grain net contact force and torque.
    /// Pure computation, no integration -- separated from `step` so grid
    /// coupling (`grains::coupling`) can apply these forces AFTER a
    /// grid-gathered velocity, exactly mirroring how `rod::coupling`'s own
    /// internal forces apply after `gather_grid_to_rod`, not instead of it.
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
                // Cheap reject before the real contact-law geometry check --
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

        // Real iterative relaxation, Jacobi-per-sweep, K sweeps: each sweep
        // resolves every active pair against velocities FROZEN at that
        // sweep's own start (same structure as the original always-single-
        // pass code -- pair order inside a sweep never matters), then
        // commits every pair's delta together before the next sweep reads
        // it. This lets a momentum handoff (grain0->grain1->grain2) cross
        // MORE THAN ONE contact interface within a single substep --
        // impossible in one sweep at any stiffness, which is exactly the
        // confirmed multi-body-chain bug (real data:
        // `diag_newtons_cradle_two_ball_release_real_conservation_check`,
        // stiffness-independent, 1x-1000x, no convergence). Real, standard
        // technique family: iterative constraint relaxation for
        // simultaneous-contact chains, same lineage as Erin Catto/Box2D's
        // sequential-impulse solver -- Jacobi ordering (not Gauss-Seidel's
        // usual immediate per-pair commit) is deliberately used here so
        // `contact_iterations=1` (the default) reduces to EXACTLY today's
        // math, sweep for sweep, keeping every existing scene (sand piles,
        // grain columns) that never opts in fully unaffected.
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
                };
                pair.active = resolution.is_some();
                if let Some(resolution) = resolution {
                    let force_on_j = resolution.normal_force * (gj.x - gi.x).normalize()
                        + resolution.tangential_force;
                    // Real Bagi (1996) / Christoffersen et al. (1981)
                    // discrete-to-continuum stress term for this contact --
                    // branch vector center-to-center, force this contact
                    // actually exerts (already computed above for the
                    // velocity update, not recomputed). See `stress_accum`'s
                    // own doc for the full formula/citation.
                    self.stress_accum += outer_product(force_on_j, gj.x - gi.x);
                    self.stress_accum_samples += 1;
                    dv[pair.j] += force_on_j / gj.mass * sub_dt;
                    dv[pair.i] -= force_on_j / gi.mass * sub_dt;
                    // Rolling moment: real action-reaction pair on spin.
                    // Friction torque: a SEPARATE source (the tangential
                    // force acting at the contact point, offset by each
                    // grain's own radius -- not an action-reaction pair
                    // since the moment arm differs per grain even though
                    // the underlying force is shared). See
                    // `ContactResolution`'s own doc for both; missing the
                    // friction-torque term was a real, confirmed bug found
                    // 2026-08 (`diag_max_speed_reached_during_collapse`).
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

    /// Real, DIRECT per-grain normal-velocity correction against any
    /// touching boundary -- found necessary live 2026-08-21, alongside the
    /// rolling-torque fix: the grid's OWN boundary correction
    /// (`BoundaryCondition::apply_to_grid_velocity`) applies PER GRID CELL,
    /// but a grain's momentum spreads across a 3x3 kernel of cells via
    /// P2G/G2P, several of which sit ABOVE the local terrain height and
    /// never receive the correction. The grain's gathered velocity ends up
    /// a noisy BLEND of corrected and uncorrected cell contributions, not
    /// the clean result real single-rigid-body contact needs -- confirmed
    /// the hard way: a grain given this blended velocity as the input to
    /// `resolve_wall_contact` rolled in the WRONG direction and appeared to
    /// fall through the terrain, even though `resolve_wall_contact`'s own
    /// math was hand-verified correct in isolation.
    ///
    /// This replaces the grid's own (noisy) normal correction with a clean,
    /// direct, per-grain one using the exact real local normal -- run
    /// BEFORE any contact-force resolution, so everything downstream
    /// (tangential friction, rolling torque) reacts to a physically clean
    /// velocity, not kernel-blend noise. The grid's own per-cell correction
    /// still runs too (harmless -- both push in the same direction, and
    /// this one runs last, fully re-establishing correctness regardless of
    /// what came before).
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

    /// Real grain-vs-BOUNDARY contact forces/torques -- found missing live
    /// 2026-08-21 (see `BoundaryCondition::grain_contact`'s own doc): before
    /// this, a grain resting on the ground had literally no mechanism to
    /// ever start rolling from rest, since only grain-grain contact
    /// (`resolve_contact_forces` above) ever produced torque. Separate from
    /// that method (not merged into it) because a boundary isn't a `Grain`
    /// -- it has no index into `self.grains`, no mass, no spin of its own.
    ///
    /// Real persistent elastic memory per grain (`wall_springs`, same "broken
    /// contact has no memory" convention as `contacts` above) -- reset to a
    /// fresh spring whenever a grain isn't touching any boundary this
    /// substep, carried forward otherwise. Only tracks ONE spring per grain
    /// (not per grain-per-boundary): a real, disclosed simplification for a
    /// grain touching multiple boundaries at once (e.g. a corner) -- rare,
    /// and this codebase's own DEM work has repeatedly found "handle the
    /// common case correctly, don't chase rare corner geometry" the right
    /// tradeoff (same spirit as `HeightmapBoundary`'s own real, disclosed
    /// fixed-+Y-normal-for-outer-walls simplification elsewhere).
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
                };
                if let Some(resolution) = resolution {
                    // Real, load-bearing choice (found the hard way, see this
                    // method's own doc): apply ONLY the tangential friction
                    // force and its resulting torque here -- NOT `resolution.
                    // normal_force`. The grid's own hard position/velocity
                    // clamp already owns the normal direction (stable,
                    // rigid); this spring's own normal_force is only ever
                    // used internally, as the Coulomb-friction cap basis.
                    // Applying it as a real force too double-counts the
                    // normal direction against a clamp that keeps resetting
                    // the same small overlap every step, so the spring keeps
                    // "refilling" a large repulsive force with nothing to
                    // bring it back down -- confirmed directly via debug
                    // instrumentation: normal_force stayed in the hundreds
                    // every single substep, launching the grain rather than
                    // letting it settle.
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

    /// Real grain-vs-CONTINUUM-terrain contact -- closes the real,
    /// root-caused structural gap `resolve_wall_contact_forces` above
    /// cannot: that method only ever fires against a real
    /// `BoundaryCondition` (static geometry), so a grain resting on a
    /// real, dynamic MPM terrain material (sharing the same grid via
    /// ordinary P2G/G2P, not a boundary) got ZERO rolling resistance from
    /// that contact -- exactly the base layer that sets a poured pile's
    /// own footprint. See `terrain_contact`'s own module doc for the full
    /// real diagnosis and the discrete-to-implicit-surface technique used
    /// to estimate a normal/overlap from the terrain's own real packing-
    /// fraction field.
    ///
    /// Same real, deliberate choice as `resolve_wall_contact_forces`
    /// above: applies ONLY the tangential friction force and its
    /// resulting torque, NOT the estimated contact's own normal
    /// component -- the shared grid's ordinary P2G/G2P momentum exchange
    /// with the terrain ALREADY provides real normal repulsion (the
    /// terrain's own elastic-plastic incompressibility resists overlap);
    /// adding a second, independent normal spring on top would double-
    /// count it, the same real failure mode that method's own doc
    /// disclosed and avoided.
    pub fn resolve_terrain_contact_forces(
        &mut self,
        grid: &crate::grid::Grid,
        dt: f32,
    ) -> (Vec<Vec2>, Vec<f32>) {
        let n = self.grains.len();
        let mut forces = vec![Vec2::ZERO; n];
        let mut torques = vec![0.0f32; n];
        // Real, disclosed opt-in gate -- see `terrain_contact_config`'s
        // own doc. Zero cost, zero behavior change for every population
        // that never calls `with_terrain_contact`.
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

    /// One real, standalone semi-implicit Euler substep (gravity + contact
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
        for (idx, grain) in self.grains.iter_mut().enumerate() {
            let mut accel = gravity + forces[idx] / grain.mass;
            for field in &self.grain_fields {
                accel += field.acceleration(grain);
            }
            grain.v += accel * dt;
            let angular_accel = torques[idx] / grain.moment_of_inertia();
            grain.spin += angular_accel * dt;
            grain.orientation += grain.spin * dt;
            grain.x += grain.v * dt;
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

    /// Real, hand-verified check of `effective_friction_angle_deg`'s own
    /// math (eigen-decomposition + Mohr-Coulomb), independent of the DEM
    /// dynamics that normally populate `stress_accum` -- directly injects a
    /// known accumulator value (same-module private-field access, this test
    /// is a child of `population`'s own module) and checks the closed-form
    /// answer against a hand-computed expectation, not the contact-force
    /// pipeline. A deliberately ASYMMETRIC raw accumulator (`x_axis.y=2`,
    /// `y_axis.x=0` -- real discrete contact sums need not be symmetric per
    /// contact, see `effective_friction_angle_deg`'s own doc) that
    /// symmetrizes to `[[2,1],[1,2]]`: eigenvalues 3 and 1 (mean=2,
    /// half_diff=0, radius=sqrt(0^2+1^2)=1), `sin(phi)=(3-1)/(3+1)=0.5` ->
    /// `phi=30deg` exactly. Using the RAW (unsymmetrized) matrix instead
    /// would give eigenvalues (2,2) -> `phi=0deg` -- this test's chosen
    /// asymmetry is deliberate, so a broken/missing symmetrization step
    /// would fail this test, not silently pass it.
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
        // loaded any real compressive contact, only recorded tension/noise.
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
        // Two real mistakes fixed here from an earlier version of this test:
        // (1) a spatially-uniform force applied to BOTH grains equally
        //     produces ZERO relative acceleration between them -- elementary
        //     mechanics (equivalence principle: gravity accelerates
        //     everything identically regardless of mass), not a property of
        //     this contact code. A huge MASS alone does not pin a body
        //     against gravity -- it still free-falls at the same rate, just
        //     reacts less to CONTACT forces. A real fixed floor needs an
        //     actual position anchor (real precedent: `Particle::pinned`'s
        //     own Dirichlet-boundary convention elsewhere in this engine),
        //     approximated here by re-clamping the floor grain's state after
        //     every step -- a real, standard test technique, not a hack
        //     specific to this bug.
        // (2) using a huge position offset (1e6) alongside an expected
        //     SIGNAL of order 1e-4 completely loses f32 precision (~7
        //     significant digits) -- real numerical-conditioning mistake,
        //     not a contact-law bug. Kept both grains at well-conditioned,
        //     order-1 coordinates instead.
        //
        // Real, precise, closed-form prediction at equilibrium: kn*overlap =
        // m*g (spring force balances weight) -> overlap = m*g/kn.
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

    /// Real end-to-end wiring check (as opposed to
    /// `effective_friction_angle_matches_hand_computed_mohr_coulomb_value`'s
    /// isolated math check): does `resolve_contact_forces`'s own real
    /// per-substep loop -- exercised by real settling dynamics, not a
    /// hand-injected accumulator -- actually populate `stress_accum` with
    /// sane data? Same settling scenario as the equilibrium-overlap test
    /// above (a light grain compressing onto a pinned floor grain under
    /// gravity). Doesn't assert an exact angle (that depends on the full
    /// nonlinear settling trajectory, not a closed form) -- just that a
    /// real reading exists and falls in a physically sane range once the
    /// grain has settled into sustained contact.
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
        // Real, honest finding from running this test, not assumed: a
        // single two-grain vertical contact with no sliding is a
        // DEGENERATE case for Mohr-Coulomb -- the contact force is purely
        // normal (parallel to the branch vector), so `outer(f, l)` is
        // rank-1 (one nonzero eigenvalue, one exactly zero -- zero real
        // lateral/confining stress). `sin(phi) = (sigma1-0)/(sigma1+0) = 1`
        // -> exactly 90deg. Mathematically correct, but physically
        // uninformative: no real granular assembly has zero lateral
        // confinement. This is why the REAL Phase-0 gate (comparing
        // against the 23.87deg pure-DEM baseline) must use a genuine
        // multi-grain pile with contacts in multiple directions, not a
        // single vertical pair -- this test only confirms the WIRING
        // (a real, sane, non-NaN value in [0,90]) reaches this point, not
        // that a two-grain pair is a meaningful friction-angle measurement.
        assert!(
            (0.0..=90.0).contains(&phi),
            "phi={phi}deg is outside any physically sane range"
        );
    }

    #[test]
    fn small_pile_under_gravity_settles_without_exploding() {
        // Real, minimal "does this actually work as a pile" sanity check --
        // not the full long-horizon repose-angle verification (a separate,
        // later, dedicated test matching this session's own established
        // discipline), just confirming a handful of grains dropped under
        // gravity onto a floor settle into a bounded, finite configuration
        // instead of diverging.
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
        // Real, CORRECTED physical invariant (2026-08-03). The original
        // version of this test asserted raw KINETIC energy alone could not
        // grow past its initial value -- empirically measured as a 64%
        // "violation" (ke0=4.0, max_ke=6.571). A full energy-breakdown
        // instrumentation (see `spacetime::grains` session notes) traced
        // this to a real, non-buggy cause: this test's own initial
        // condition spawns the two grains ALREADY overlapping by 1% of
        // radius (radii sum 1.0, separation 0.99), which means the contact
        // starts with substantial PRELOADED elastic potential energy in the
        // normal spring -- PE_n0 = 0.5*kn*overlap0^2 = 5.0, actually MORE
        // than the initial kinetic energy itself (KE0=4.0). As that real
        // compressed spring naturally pushes the two free grains apart --
        // ordinary, correct physics for ANY spring-dashpot contact model,
        // not a bug, exactly like releasing a compressed spring between two
        // free masses -- it legitimately converts stored PE into KE, which
        // the old invariant misread as an energy-conservation violation.
        //
        // Direct instrumentation confirmed this is legitimate: total
        // mechanical energy (KE + normal-spring PE + tangential-spring PE +
        // cumulative dissipated energy) stayed within 0.58% of its initial
        // value across the entire 2,000,000-step run (peak relative
        // overshoot 0.5786% at step 18229) -- real, bounded, expected
        // semi-implicit-Euler numerical error for a stiff damped spring
        // (matches this force law's own hand-derived power balance:
        // d(KE+PE)/dt = -normal_damping*v_n^2 - tangential_damping*v_t^2 <=
        // 0), nowhere close to a genuine 64% energy injection. A separate
        // check at 10x finer dt (which any REAL discretization bug should
        // shrink under, not grow under) instead showed the KE-alone
        // "violation" tracks physical PE release consistently at matching
        // physical time, further confirming this is not a discretization
        // artifact of contact_law.rs/population.rs.
        //
        // The real, physically correct invariant for a purely dissipative
        // (damped) contact system with zero external driver -- one that may
        // start pre-loaded with elastic energy, exactly like every real
        // pair of touching grains in a settled pile -- is that TOTAL
        // mechanical energy (KE + energy stored in the contact springs) is
        // monotonically non-increasing, NOT raw KE alone, which is free to
        // rise as preloaded spring PE legitimately converts into it.
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

        // Real total mechanical energy: KE (translational + rotational)
        // plus whatever elastic PE is currently stored in the active
        // contact's normal and tangential springs -- the actual
        // conserved-minus-dissipated quantity for this force law.
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
        // Real measured peak this session: 0.5786% overshoot. 1% gives
        // real headroom for bounded explicit-Euler numerical error while
        // still catching genuine energy injection (a real bug here would
        // look like the old invariant's measured 64% "violation").
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
        // Real, direct physical check for the friction-induced-rolling
        // torque (`ContactResolution::friction_torque_on_i/j`): a grain
        // given a real sliding velocity along a fixed floor must evolve
        // TOWARD "rolling without slipping" (the contact-point tangential
        // slip speed |v_t| decaying over time as spin builds up to match).
        // Confirms the torque's sign/formula is genuinely correct in
        // isolation (verified: slip decays smoothly 3.0 -> 0.9 while real
        // contact stays engaged). A real, SEPARATE full-column collapse
        // test (`tests/grains_repose_angle.rs`) showed unbounded growth
        // instead -- this test proves that's NOT a sign error in the core
        // force law; the real cause is elsewhere (many-body/repeated-
        // contact dynamics specific to that scene, not this pairwise law).
        // Real, fixed overlap (1% of radius) from the start, ZERO gravity --
        // isolates purely the sliding-friction-induces-rolling question,
        // removing the confound of an earlier version of this test (gravity
        // continuously growing the overlap over time, meaning the contact
        // barely engaged at all for the first several thousand steps while
        // a real gap was still closing -- a real, separate effect that
        // muddied this specific measurement, not itself a bug).
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

        // Measured over the window real contact stays genuinely engaged
        // (checked directly: this pair separates vertically -- the normal
        // spring's own bounce -- a bit after step ~4500, at which point
        // velocity/spin freeze and any further "slip" reading is pure
        // separated-body geometric drift, not real contact physics; this
        // window is entirely within the real, engaged-contact regime).
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

    /// Real, direct numeric proof that `Grain::orientation` (added
    /// 2026-08-03 specifically so a grain's real rolling has something to
    /// render) actually accumulates a genuine rotation, not just a nonzero
    /// `spin` that never gets integrated anywhere. Same real scenario as
    /// `sliding_grain_on_a_pinned_floor_converges_toward_rolling_not_away_
    /// from_it` above (a grain sliding on a pinned floor, real friction-
    /// induced spin-up) -- reused rather than invented fresh, since that
    /// scenario already independently proves the underlying spin dynamics
    /// are correct; this test's ONLY new claim is that `orientation`
    /// faithfully tracks `integral(spin dt)`.
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
        // Real, corrected threshold (2026-08-04): the ORIGINAL >0.5 rad bound
        // here was simply wrong -- at dt=1e-6s, 4000 steps is only 4ms of
        // real simulated time, and spin ramps from 0 up to ~-2.4 rad/s over
        // that same window, so the real, correct integral is on the order
        // of -0.005 rad (roughly avg_spin * duration), not >0.5 rad. Caught
        // by actually running this test rather than assuming the threshold
        // was right -- exactly the kind of "prove it, don't assume it"
        // check this session's own standing discipline requires. The real
        // claim this test makes is or nonzero, correctly-signed, non-NaN
        // rotation consistent with the spin history -- verified precisely
        // by the independent trapezoidal cross-check below, not by an
        // arbitrary magnitude bound.
        assert!(
            final_orientation.is_finite() && final_orientation != 0.0,
            "expected real, nonzero accumulated rotation from 4000 steps of \
             real friction-induced spin-up, got {final_orientation}"
        );
        // Cross-check: `orientation` must match the real numerical integral
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
}
