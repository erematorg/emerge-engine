//! Elastic solids moved by the total Lagrangian step, inside `Simulation`.
//!
//! An opt-in path (`SimConfig::total_lagrangian`): each frame advances the
//! solids by the total Lagrangian conserving step (`spacetime::
//! total_lagrangian`), walls and bodies meeting by the contact law of
//! `forces::boundary::signorini_coulomb` held over each step on the
//! velocity, the steps' lengths chosen by Acary's error estimate. Energy is
//! kept exactly in every step, friction's share turned into heat.
//!
//! The state that counts is kept here in `f64` and in SI, per metre of
//! depth (positions m, velocities m/s, masses kg/m, areas m^2, Lame
//! parameters Pa, energies J/m), converted from and to the particles' grid
//! units in one place (`grid_units::GridUnits`); the particles'
//! `f32` copies are written each frame for the renderer, the queries and
//! the GPU. Round-tripping through `f32` would cost about `1e-7` relative a
//! frame, a slow drift of the energy the step keeps exactly.
//!
//! What this path does not do yet it refuses, with the reason, instead of
//! ignoring: materials other than the corotated elastic solid (another law
//! would be a substitution until it has its own `f64` form), force fields
//! other than gravity, boundaries without a contact form, phase rules,
//! sleeping, rods, grains, thermal or scalar diffusion, granular fluidity,
//! Cosserat fields, the contact grip, and switching paths in a run (the
//! particles' APIC affine velocity is zeroed here, as this path does not
//! use it, and the explicit path's position residuals are not this one's).

use glam::{DMat2, DVec2, Mat2, Vec2};

use super::Simulation;
use super::grid_units::GridUnits;
use crate::energy::thermodynamics::temperature_rise_from_dissipation;
use crate::matter::materials::solid::corotated::CorotatedMaterial;
use crate::matter::materials::solid::elastic::NeoHookeanMaterial;
use crate::matter::materials::solid::von_mises::VonMisesMaterial;
use crate::spacetime::solver::config::SpawnRegion;
use crate::spacetime::solver::config::SpawnShape;
use crate::spacetime::total_lagrangian::{
    ConservingSettings, ContactKey, ContactSet, Elastic, Estimator, GridLayout, LagrangianCache,
    Modes, Particles, Plane, StepControl, body_modes, contact_grid, controlled_step,
    start_velocity_field, tolerance_energy,
};
use crate::spacetime::transfer::friction_heat::MIN_APPLIED_TEMPERATURE_RISE_K;

/// The path's settings: the step controller's tolerance (relative: the
/// error's energy per step is at most `(0.9 tolerance)^2` of the scene's
/// `tolerance_energy`, its kinetic and stored energy at the start plus each
/// body's `m g L` over its own extent) and the step's own settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TotalLagrangianConfig {
    pub tolerance: f64,
    pub settings: ConservingSettings,
    /// Carry a lone body's flights between contacts by its modes (`flight`)
    /// instead of stepping them. Off by default.
    pub modal_flight: bool,
    /// Carry a lone body's rest on frictionless walls by its constrained
    /// modes (`rest`) while its held contacts provably stay compressive.
    /// Off by default.
    pub modal_rest: bool,
}

impl TotalLagrangianConfig {
    /// The path with the step controller's `tolerance` and the step's
    /// default settings.
    pub fn new(tolerance: f64) -> Self {
        Self {
            tolerance,
            settings: ConservingSettings::default(),
            modal_flight: false,
            modal_rest: false,
        }
    }

    /// The same, a lone body's rest carried by its constrained modes
    /// (`rest`).
    pub fn with_modal_rest(mut self) -> Self {
        self.modal_rest = true;
        self
    }

    /// The same, a lone body's flights carried by its modes (`flight`).
    pub fn with_modal_flight(mut self) -> Self {
        self.modal_flight = true;
        self
    }
}

mod frame;
pub use frame::TotalLagrangianFrame;

/// The path's state, kept between frames.
#[cfg_attr(all(test, feature = "experimental"), derive(Clone))]
pub(crate) struct TotalLagrangianState {
    /// The solids' particles first, the bodies' contact points after them.
    reference: Vec<DVec2>,
    x: Vec<DVec2>,
    v: Vec<DVec2>,
    f: Vec<DMat2>,
    mass: Vec<f64>,
    v0: Vec<f64>,
    solids: Vec<Elastic>,
    penalty: Vec<f64>,
    /// Each body's contact points, by index, counter-clockwise.
    outlines: Vec<Vec<usize>>,
    /// Per outline, the indices (into it) of its corners.
    corners: Vec<Vec<usize>>,
    /// How many of the entries are the solids' particles.
    particles: usize,
    cache: Option<LagrangianCache>,
    step_length: f64,
    heat_debt: Vec<f32>,
    heat: f64,
    start_energy: f64,
    /// The step tolerance's energy (`tolerance_energy`), fixed at the start.
    tolerance_energy: f64,
    /// The particles' momentum at the start.
    start_momentum: DVec2,
    /// The time since the start, s.
    time: f64,
    /// The lone body's modes once a flight needed them (`None` inside when
    /// its law has none), and the flight under way.
    modes: Option<Option<Modes>>,
    flight: Option<Flight>,
    /// `TotalLagrangianFrame::modal_frames` and `hand_backs`.
    modal_frames: u64,
    hand_backs: u64,
    /// The rest under way, the held set the stepped frame before the last
    /// ended on, and `TotalLagrangianFrame::modal_rest_frames` and
    /// `rest_hand_backs`.
    rest: Option<Rest>,
    /// The last held set's rest basis (`rest::Basis`), kept so a refused
    /// rest is not rebuilt at every frame on the same set.
    rest_basis: Option<rest::Basis>,
    last_held: Vec<ContactKey>,
    modal_rest_frames: u64,
    rest_hand_backs: u64,
    modal_refused: [u64; 3],
    modal_check_failed: u64,
    modal_rest_refused: u64,
    pub(crate) frame: TotalLagrangianFrame,
}

/// A body's contact outline: its points and the indices of its corners
/// (`spawn_outline`).
pub(crate) type SpawnOutline = (Vec<Vec2>, Vec<usize>);

/// What the path records of a body at its spawn: its outline, its
/// material, and the tag its particles carry (`add_body`'s; 0 for the
/// initial spawn's).
pub(crate) struct SpawnedBody {
    pub(crate) outline: SpawnOutline,
    pub(crate) material: u32,
    pub(crate) tag: u32,
}

/// A body's surface as contact points, counter-clockwise, one every half
/// cell: the spawn region's box or disk; and the indices of its corners,
/// the points where the exact shape's normal jumps (a box's four, a disk's
/// none), where its contact conditions split (`weighted_gap`).
pub(crate) fn spawn_outline(spawn: &SpawnRegion) -> SpawnOutline {
    let spacing = 0.5;
    match spawn.shape {
        SpawnShape::Box => {
            let half = spawn.box_size.as_vec2() * 0.5;
            let (min, max) = (spawn.box_center - half, spawn.box_center + half);
            let corners = [min, Vec2::new(max.x, min.y), max, Vec2::new(min.x, max.y)];
            let (mut points, mut at) = (Vec::new(), Vec::new());
            for (k, &a) in corners.iter().enumerate() {
                let b = corners[(k + 1) % 4];
                let count = ((b - a).length() / spacing).ceil().max(1.0) as usize;
                at.push(points.len());
                points.extend((0..count).map(|i| a + (b - a) * (i as f32 / count as f32)));
            }
            (points, at)
        }
        SpawnShape::Disk { radius } => {
            let count = (std::f32::consts::TAU * radius / spacing).ceil().max(3.0) as usize;
            let points = (0..count)
                .map(|i| {
                    let angle = std::f32::consts::TAU * i as f32 / count as f32;
                    spawn.box_center + radius * Vec2::new(angle.cos(), angle.sin())
                })
                .collect();
            (points, Vec::new())
        }
    }
}

impl Simulation {
    /// Refuses, with the reason, what this path does not do yet.
    fn assert_total_lagrangian_supported(&self) {
        let refuse = |what: &str| {
            panic!(
                "the total Lagrangian solid path (SimConfig::total_lagrangian) does not support {what} yet; \
                 remove it or leave the path off"
            )
        };
        if !self.force_fields.is_empty() {
            refuse("force fields other than gravity");
        }
        if self.thermal.is_some() || !self.scalar_fields.is_empty() {
            refuse("thermal or scalar diffusion");
        }
        if self.granular_fluidity.is_some() || self.cosserat.is_some() {
            refuse("granular fluidity or Cosserat fields");
        }
        if !self.phase_rules.is_empty() {
            refuse("phase rules");
        }
        if !self.rods.is_empty() || !self.grain_populations.is_empty() {
            refuse("rods or grains");
        }
        if self.contact_grip.is_some() {
            refuse("the directional contact grip");
        }
        if self.config.sleep_threshold > 0.0 {
            refuse("sleeping particles");
        }
        if self.boundaries.iter().any(|b| b.domain_walls().is_none()) {
            refuse(
                "this boundary (only slip and Coulomb friction domain walls have a contact form)",
            );
        }
    }

    /// The elastic law of particle `p`'s material on this path.
    fn total_lagrangian_solid(&self, p: usize) -> Elastic {
        self.total_lagrangian_material(self.particles.material_id[p])
    }

    /// The elastic law of material `id` on this path.
    fn total_lagrangian_material(&self, id: u32) -> Elastic {
        let model = self.materials.get(id);
        let any = model.as_any();
        // The corotated material's own law (`matter`, in `f64`). Any other
        // law would be a substitution: refused until it has its own `f64`
        // form for the step's discrete gradient.
        // A ductile material's coefficients (`VonMisesMaterial`: its Lame
        // moduli, yield stress, hardening and dashpot) read into this path's
        // own law, J2 on Hencky elasticity (Love and Sulsky 2005; `matter`'s
        // `HenckyJ2`). The material's yield condition is `|dev tau|_F >=
        // yield_stress + H_m kappa`, the in-plane deviator's Frobenius norm,
        // `yield_stress = sqrt(2/3)` of the uniaxial stress
        // (`VonMisesMaterial::from_physical`), `kappa` the accumulated plastic
        // strain, `H_m` the radius's slope per unit of it. The law's: `|dev
        // tau| >= sqrt(2/3) (Y + H alpha)`, `alpha = sqrt(2/3) gamma`. In pure
        // shear the in-plane and 3D deviators coincide, so the onsets agree
        // when `Y = sqrt(3/2) yield_stress`, and the radii's slopes per unit
        // plastic strain when `(2/3) H = H_m` (`H = 3/2 H_m`); measured:
        // `j2_hencky::tests::the_von_mises_mapping_matches_in_pure_shear`.
        // Adaptation, declared: the dashpot reads the material's
        // `elastic_viscosity` as the textbook shear viscosity (Pa s) of
        // `matter`'s `newtonian_viscous_stress`, no bulk viscosity; the
        // explicit path still reads it with its half-factor convention (the
        // open engine-wide dashpot item).
        if let Some(m) = any.downcast_ref::<VonMisesMaterial>() {
            let units = GridUnits::of(&self.config);
            let law = Elastic::hencky_j2(
                units.stress(m.lambda),
                units.stress(m.mu),
                (1.5f64).sqrt() * units.stress(m.yield_stress),
                1.5 * units.stress(m.hardening_modulus),
                0.0,
            );
            // Norton's flow is SI already (Pa, 1/s), as the path is.
            let law = match m.norton {
                Some(norton) => law.with_norton(norton),
                None => law,
            };
            return law.with_viscosity(units.stress(m.elastic_viscosity), 0.0);
        }
        // Liquid water's record is SI (Pa), as the path is.
        if let Some(m) = any.downcast_ref::<crate::LiquidWaterMaterial>() {
            return Elastic::barotropic(m.bulk_modulus());
        }
        match any.downcast_ref::<CorotatedMaterial>() {
            Some(m) => {
                // Viscosities in Pa s: time is the second on both sides,
                // so they convert as stresses do.
                let units = GridUnits::of(&self.config);
                let solid = Elastic::corotated(units.stress(m.lambda), units.stress(m.mu))
                    .with_viscosity(
                        units.stress(m.elastic_viscosity),
                        units.stress(m.elastic_bulk_viscosity),
                    );
                // The relaxation spectrum's frequencies are per second, the
                // path's time unit; its weights are ratios.
                match &m.relaxation {
                    None => solid,
                    Some(spectrum) => solid.with_relaxation(spectrum).unwrap_or_else(|e| {
                        panic!("material {id}: its relaxation spectrum is refused: {e}")
                    }),
                }
            }
            None if any.downcast_ref::<NeoHookeanMaterial>().is_some() => panic!(
                "neo-Hookean on the total Lagrangian solid path: not yet (the step needs its own law in f64);                  material {id}"
            ),
            None => panic!(
                "the total Lagrangian solid path supports the corotated elastic solid only; material {id} is {model:?}"
            ),
        }
    }

    /// The step's grid layout: the simulation's grid, in metres.
    fn total_lagrangian_layout(&self) -> GridLayout {
        GridLayout::new(
            self.config.grid_res,
            self.config.grid_res,
            self.config.dx_meters,
        )
    }

    /// Builds the path's state from the particles as they are now, which
    /// becomes the reference configuration; `settings` are the step's. With
    /// `modal_flight`, a lone body's modes are computed here, at its spawn,
    /// on the grid the step will keep, not inside its first flight.
    fn build_total_lagrangian_state(
        &self,
        (settings, modal_flight): (&ConservingSettings, bool),
    ) -> TotalLagrangianState {
        let n = self.particles.len();
        let units = GridUnits::of(&self.config);
        let mut reference: Vec<DVec2> = self
            .particles
            .x
            .iter()
            .map(|&q| units.position(q))
            .collect();
        let mut v: Vec<DVec2> = self
            .particles
            .v
            .iter()
            .map(|&q| units.velocity(q))
            .collect();
        let mut f: Vec<DMat2> = self
            .particles
            .deformation_gradient
            .iter()
            .map(|&m| units.deformation(m))
            .collect();
        let mut mass: Vec<f64> = self.particles.mass.iter().map(|&m| units.mass(m)).collect();
        // Each particle's reference area is its mass over its material's
        // density, `V0 = m / rho` (what the particle stands for), not the
        // explicit path's packing estimate: that one grows up to 2.56 times
        // the lattice area on a body's free surface, and a thin body's
        // stiffness with it (measured: a 2-cell-thick bar's mean 0.36 cell^2
        // against its lattice 0.25, its first mode 18 percent above `pi c /
        // L`; issue #41). A material built from grid units declares no
        // density: `SimConfig::grid_density` then.
        let mut v0: Vec<f64> = (0..n)
            .map(|p| {
                let density = self
                    .materials
                    .get(self.particles.material_id[p])
                    .rest_density()
                    .unwrap_or(self.config.grid_density);
                units.area(self.particles.mass[p] / density)
            })
            .collect();
        let mut solids: Vec<Elastic> = (0..n).map(|p| self.total_lagrangian_solid(p)).collect();
        let mut penalty = vec![0.0; n];
        let (mut outlines, mut corners) = (Vec::new(), Vec::new());
        // Each body's mass (its tagged particles') and its outline's
        // reference points, for the tolerance's energy.
        let mut bodies = Vec::new();
        for body in &self.total_lagrangian_outlines {
            let (points, at) = &body.outline;
            corners.push(at.clone());
            let stiffness = &self.total_lagrangian_material(body.material);
            let start = reference.len();
            for &q in points {
                reference.push(units.position(q));
                v.push(DVec2::ZERO);
                f.push(DMat2::IDENTITY);
                mass.push(0.0);
                v0.push(0.0);
                solids.push(*stiffness);
                // The multipliers' penalty, an iteration parameter: the
                // P wave's stiffness (the step adds the inertia's).
                penalty.push(stiffness.bulk + stiffness.mu);
            }
            outlines.push((start..reference.len()).collect());
            let body_mass: f64 = (0..n)
                .filter(|&p| self.particles.user_tag[p] == body.tag)
                .map(|p| mass[p])
                .sum();
            bodies.push((body_mass, start..reference.len()));
        }
        // The contact points carry no mass, so no velocity of their own:
        // they start at the body's velocity field, the one the step's
        // increments then move (`start_velocity_field`). Started at rest,
        // they lagged a body spawned moving by its spawn velocity, read by
        // the contact tolerance's motion scale.
        let field = start_velocity_field(
            &self.total_lagrangian_layout(),
            &reference,
            (&mass, &v0, &v),
            settings,
        );
        v[n..].copy_from_slice(&field[n..]);
        let x = reference.clone();
        let mut state = TotalLagrangianState {
            reference,
            x,
            v,
            f,
            mass,
            v0,
            solids,
            penalty,
            outlines,
            corners,
            particles: n,
            cache: None,
            step_length: f64::from(self.config.dt),
            heat_debt: vec![0.0; n],
            heat: 0.0,
            start_energy: 0.0,
            tolerance_energy: 0.0,
            start_momentum: DVec2::ZERO,
            time: 0.0,
            modes: None,
            flight: None,
            modal_frames: 0,
            hand_backs: 0,
            rest: None,
            rest_basis: None,
            last_held: Vec::new(),
            modal_rest_frames: 0,
            rest_hand_backs: 0,
            modal_refused: [0; 3],
            modal_check_failed: 0,
            modal_rest_refused: 0,
            frame: TotalLagrangianFrame::default(),
        };
        state.start_momentum = (0..n).fold(DVec2::ZERO, |sum, p| sum + state.mass[p] * state.v[p]);
        if modal_flight && state.outlines.len() == 1 {
            let grid = contact_grid(
                &self.total_lagrangian_layout(),
                &state.reference,
                (&state.mass, &state.v0),
                settings,
            );
            state.modes = Some(body_modes(
                &grid,
                (&state.mass, &state.v0, &state.solids),
                (state.particles, state.x.len()),
            ));
        }
        state.start_energy = self.total_lagrangian_energy(&state);
        let kinetic_and_stored: f64 = (0..n)
            .map(|p| {
                0.5 * state.mass[p] * state.v[p].length_squared()
                    + state.v0[p] * state.solids[p].energy(state.f[p])
            })
            .sum();
        state.tolerance_energy = tolerance_energy(
            kinetic_and_stored,
            self.total_lagrangian_gravity(),
            bodies
                .iter()
                .map(|(m, points)| (*m, &state.reference[points.clone()])),
        );
        state
    }

    /// Gravity, m/s^2.
    fn total_lagrangian_gravity(&self) -> DVec2 {
        GridUnits::of(&self.config).velocity(self.config.gravity)
    }

    /// The solids' energy, J/m: kinetic, stored (the step's law) and
    /// gravity's.
    fn total_lagrangian_energy(&self, state: &TotalLagrangianState) -> f64 {
        let gravity = self.total_lagrangian_gravity();
        (0..state.particles)
            .map(|p| {
                0.5 * state.mass[p] * state.v[p].length_squared()
                    + state.v0[p] * state.solids[p].energy(state.f[p])
                    - state.mass[p] * gravity.dot(state.x[p])
            })
            .sum()
    }

    /// The domain's walls as contact planes and their friction coefficient
    /// (one for every contact, walls and bodies alike: declared).
    fn total_lagrangian_walls(&self) -> (Vec<Plane>, f64) {
        let cell = f64::from(self.config.dx_meters);
        let res = self.config.grid_res as f64 * cell;
        let mut planes = Vec::new();
        let mut friction = None::<f32>;
        for wall in self.boundaries.iter().filter_map(|b| b.domain_walls()) {
            let t = wall.thickness as f64 * cell;
            for axis in 0..2 {
                planes.push(Plane {
                    axis,
                    at: t,
                    inward: 1.0,
                });
                planes.push(Plane {
                    axis,
                    at: res - t,
                    inward: -1.0,
                });
            }
            match friction {
                Some(mu) if mu != wall.friction => panic!(
                    "the total Lagrangian solid path takes one friction coefficient for every contact; \
                     the walls give {mu} and {}",
                    wall.friction
                ),
                _ => friction = Some(wall.friction),
            }
        }
        (planes, f64::from(friction.unwrap_or(0.0)))
    }

    /// One frame on this path: controlled steps until the frame's time is
    /// covered, the heat into the particles' temperatures, the particles'
    /// `f32` copies written.
    pub(crate) fn step_total_lagrangian_solids(&mut self, config: TotalLagrangianConfig) {
        self.assert_total_lagrangian_supported();
        // The explicit path's state (its APIC affine velocity, its
        // position residuals) is not this path's, nor the reverse: a run
        // stays on one path from its first frame.
        assert!(
            self.total_lagrangian.is_some() || self.frame_index == 0,
            "the total Lagrangian solid path cannot be switched on after the explicit path has run"
        );
        // The step's reference configuration is each body's start, and
        // its state (`f64` positions, deformation, multipliers) lives here:
        // rebuilt from the particles mid-run, the bodies already moving
        // would take their deformed shape as their reference and lose that
        // state. A body added later is refused, not silently re-referenced.
        match &self.total_lagrangian {
            None => {
                self.total_lagrangian =
                    Some(self.build_total_lagrangian_state((&config.settings, config.modal_flight)))
            }
            Some(state) => assert!(
                state.particles == self.particles.len(),
                "the total Lagrangian solid path does not support adding bodies after its first frame (its reference configuration is the bodies' start)"
            ),
        }
        let (planes, friction) = self.total_lagrangian_walls();
        let layout = self.total_lagrangian_layout();
        let control = StepControl {
            tolerance: config.tolerance,
            safety: 0.9,
            growth: 10.0,
            largest: f64::from(self.config.dt),
            substeps: [1, 2, 3],
            estimator: Estimator::Acary,
        };
        let gravity = self.total_lagrangian_gravity();
        let mut state = self.total_lagrangian.take().expect("built above");
        let scale = state.tolerance_energy.max(f64::MIN_POSITIVE).sqrt();
        let mut frame = TotalLagrangianFrame::default();
        let framing = std::time::Instant::now();
        let length = f64::from(self.config.dt);
        frame.modal = config.modal_flight
            && Self::fly_total_lagrangian_frame(&mut state, (&planes, gravity), length);
        // A damped flight's heat (`Flight::carry_branches`), deposited like
        // the step's.
        if frame.modal
            && let Some(flight) = state.flight.as_mut()
        {
            let made = std::mem::take(&mut flight.heat);
            frame.relaxation_heat = made.iter().sum();
            frame.relaxation_strain = flight.strain;
            self.deposit_total_lagrangian_heat(&mut state, &made);
        }
        let now = state.time;
        frame.modal_rest = !frame.modal
            && config.modal_rest
            && rest::rest_frame(&mut state, (&planes, friction, gravity), (now, length));
        // A damped rest's heat (`Rest::carry_branches`), deposited like the
        // step's.
        if frame.modal_rest
            && let Some(rest) = state.rest.as_mut()
        {
            let made = std::mem::take(&mut rest.heat);
            frame.relaxation_heat = made.iter().sum();
            frame.relaxation_strain = rest.strain;
            self.deposit_total_lagrangian_heat(&mut state, &made);
        }
        // A modal frame takes no step; a stepped one first records the held
        // set it starts from (a rest needs two stepped frames ending on the
        // same set).
        if !frame.modal && !frame.modal_rest {
            state.last_held = rest::held_keys(&state);
        }
        let mut remaining = if frame.modal || frame.modal_rest {
            0.0
        } else {
            length
        };
        while remaining > 4.0 * f64::EPSILON * f64::from(self.config.dt) {
            // A step cut short to end on the frame's time says nothing
            // about the step the solution allows: kept at its first try,
            // the next proposal is never below the one it was cut from
            // (else each frame climbs back from its remainder, `growth`
            // per step).
            let proposed = state.step_length;
            let cut = proposed > remaining;
            state.step_length = proposed.min(remaining);
            let step = controlled_step(
                &layout,
                Particles {
                    x: &mut state.x,
                    v: &mut state.v,
                    f: &mut state.f,
                    mass: &state.mass,
                    v0: &state.v0,
                },
                (&state.reference, &mut state.solids),
                (gravity, &mut state.step_length),
                (
                    ContactSet {
                        planes: &planes,
                        pairs: &[],
                        outlines: &state.outlines,
                        corners: &state.corners,
                        friction,
                    },
                    &state.penalty,
                    &config.settings,
                ),
                &mut state.cache,
                (&control, scale),
            );
            remaining -= step.dt;
            if cut && step.rejected == 0 {
                state.step_length = state.step_length.max(proposed);
            }
            let substep = step.dt / step.kept.len().max(1) as f64;
            for kept in &step.kept {
                frame.residual_impulse += substep * kept.residual_force.abs();
                frame.impulse_round_off += substep * kept.force_round_off;
                frame.newton += kept.newton;
                frame.passes += kept.passes;
                frame.residual_work += kept.residual_work;
                frame.contact_work_bound += kept.contact_work_bound;
                frame.contacts = frame.contacts.max(kept.contacts);
                frame.gmres += kept.gmres;
                frame.unconverged += u32::from(!kept.converged);
                frame.friction_work += kept.friction;
                frame.contact_work += kept.work.0;
                frame.dissipated += kept.viscous;
                frame.plastic_work += kept.plastic;
                frame.relaxation_heat += kept.relaxation;
                frame.relaxation_strain = frame.relaxation_strain.max(kept.relaxation_strain);
                frame.relaxation_beyond += kept.relaxation_beyond;
                frame.norton_iterations = frame.norton_iterations.max(kept.norton_iterations);
                frame.norton_slower += kept.norton_slower;
                frame.norton_faster += kept.norton_faster;
                frame.norton_tension += kept.norton_tension;
            }
            frame.steps += 1;
            frame.timing.add(&step.timing);
            frame.rejected += step.rejected;
            frame.rejected_unconverged += step.unconverged;
            frame.reaching += u32::from(step.reaching);
            frame.at_floor += u32::from(step.at_floor);
            frame.stagnated += step.stagnated;
        }
        let heat = state
            .cache
            .as_mut()
            .map(LagrangianCache::take_heat)
            .unwrap_or_default();
        self.deposit_total_lagrangian_heat(&mut state, &heat);
        frame.energy = self.total_lagrangian_energy(&state);
        // The linearisation's residual: the energy's change less the heat a
        // damped flight made (none otherwise, the heat's change then zero).
        if frame.modal || frame.modal_rest {
            frame.modal_energy_change =
                frame.energy - state.frame.energy + (state.heat - state.frame.heat);
        }
        state.time += length;
        frame.parts = (0..state.particles).fold([0.0; 3], |[k, e, g], p| {
            [
                k + 0.5 * state.mass[p] * state.v[p].length_squared(),
                e + state.v0[p] * state.solids[p].energy(state.f[p]),
                g - state.mass[p] * gravity.dot(state.x[p]),
            ]
        });
        let total: f64 = state.mass[..state.particles].iter().sum();
        if total > 0.0 {
            let (centre, momentum) =
                (0..state.particles).fold((DVec2::ZERO, DVec2::ZERO), |(c, m), p| {
                    (
                        c + state.mass[p] * state.x[p],
                        m + state.mass[p] * state.v[p],
                    )
                });
            frame.momentum = momentum;
            let (centre, velocity) = (centre / total, momentum / total);
            for p in 0..state.particles {
                let arm = state.x[p] - centre;
                frame.angular_momentum += state.mass[p] * arm.perp_dot(state.v[p] - velocity);
                frame.moment_of_inertia += state.mass[p] * arm.length_squared();
            }
        }
        frame.heat = state.heat;
        frame.wall = framing.elapsed().as_secs_f64();
        frame.start_energy = state.start_energy;
        frame.start_momentum = state.start_momentum;
        (frame.modal_frames, frame.hand_backs) = (state.modal_frames, state.hand_backs);
        (frame.modal_rest_frames, frame.rest_hand_backs) =
            (state.modal_rest_frames, state.rest_hand_backs);
        [
            frame.modal_refused_law,
            frame.modal_refused_yielded,
            frame.modal_refused_bound,
        ] = state.modal_refused;
        frame.modal_refused = state.modal_refused.iter().sum();
        frame.modal_check_failed = state.modal_check_failed;
        frame.modal_rest_refused = state.modal_rest_refused;
        frame.tolerance_energy = state.tolerance_energy;
        frame.balance = frame.energy + frame.heat - frame.start_energy;
        state.frame = frame;
        let units = GridUnits::of(&self.config);
        for p in 0..state.particles {
            let f = state.f[p];
            self.particles.x[p] = units.position_to_grid(state.x[p]);
            self.particles.position_residual[p] = Vec2::ZERO;
            self.particles.v[p] = units.velocity_to_grid(state.v[p]);
            self.particles.velocity_gradient[p] = Mat2::ZERO;
            self.particles.deformation_gradient[p] = f.as_mat2();
            let volume = state.v0[p] * f.determinant();
            self.particles.volume[p] = units.area_to_grid(volume);
            if volume > 0.0 {
                self.particles.density[p] = units.density_to_grid(state.mass[p] / volume);
            }
        }
        self.total_lagrangian = Some(state);
        self.frame_index = self.frame_index.saturating_add(1);
    }

    /// One frame of a lone body's flight carried by its modes, when it may
    /// fly (`flight`): no contact held at the last step, no turning, and
    /// every contact point provably clear of the walls to the frame's end.
    /// False hands the frame to the step, the flight (if any) ended there.
    fn fly_total_lagrangian_frame(
        state: &mut TotalLagrangianState,
        (planes, gravity): (&[Plane], DVec2),
        length: f64,
    ) -> bool {
        let (t0, t1) = (state.time, state.time + length);
        if state.flight.is_none() {
            let free = state.outlines.len() == 1
                && state.cache.as_ref().is_some_and(|c| c.contacts.is_empty());
            if !free {
                return false;
            }
            if state.modes.is_none() {
                let grid = &state.cache.as_ref().expect("checked above").grid;
                let modes = body_modes(
                    grid,
                    (&state.mass, &state.v0, &state.solids),
                    (state.particles, state.x.len()),
                );
                state.modes = Some(modes);
            }
            let Some(Some(modes)) = &state.modes else {
                state.modal_refused[0] += 1;
                return false;
            };
            let started = Flight::start(state, modes, (t0, length));
            match started {
                Err(flight::Refusal::Yielded) => state.modal_refused[1] += 1,
                Err(flight::Refusal::Bound) => state.modal_refused[2] += 1,
                _ => {}
            }
            state.flight = started.ok();
        }
        let Some(mut flight) = state.flight.take() else {
            return false;
        };
        if !flight.clear(gravity, planes, (t0, t1)) {
            state.hand_backs += 1;
            return false;
        }
        let Some(Some(modes)) = state.modes.take() else {
            return false;
        };
        let start = (state.x.clone(), state.v.clone(), state.f.clone());
        if !flight.rebuild(&modes, gravity, t1, state) {
            state.modes = Some(Some(modes));
            state.modal_check_failed += 1;
            state.hand_backs += 1;
            return false;
        }
        // A frame whose rebuilt state does not keep the angular momentum is
        // stepped instead, from the frame's start (`flight`).
        if !flight.keeps_angular_momentum(state) {
            (state.x, state.v, state.f) = start;
            state.modes = Some(Some(modes));
            state.hand_backs += 1;
            return false;
        }
        flight.carry_branches(&modes, state, &start.2);
        state.modes = Some(Some(modes));
        state.flight = Some(flight);
        state.modal_frames += 1;
        true
    }

    /// Friction's heat into the particles' temperatures, banked until a
    /// rise survives `f32` (the explicit path's `transfer::friction_heat`
    /// rule), counted in the energy balance whether or not a material
    /// declares a heat capacity.
    fn deposit_total_lagrangian_heat(&mut self, state: &mut TotalLagrangianState, heat: &[f64]) {
        for (p, &made) in heat.iter().enumerate().take(state.particles) {
            // Every share counted, whatever its sign: the ledger is the
            // heat the step made, not the part a temperature can show.
            state.heat += made;
            if made <= 0.0 || state.mass[p] <= 0.0 {
                continue;
            }
            // J/m over kg/m: J/kg, SI already.
            let specific_si = (made / state.mass[p]) as f32;
            let c_p = self
                .materials
                .get(self.particles.material_id[p])
                .specific_heat_j_kg_k();
            let rise = temperature_rise_from_dissipation(specific_si, c_p);
            if rise <= 0.0 {
                continue;
            }
            state.heat_debt[p] += rise;
            if state.heat_debt[p] >= MIN_APPLIED_TEMPERATURE_RISE_K {
                self.particles.temperature[p] += state.heat_debt[p];
                state.heat_debt[p] = 0.0;
            }
        }
    }

    /// The last frame on the total Lagrangian solid path, when it is on.
    pub fn total_lagrangian_frame(&self) -> Option<TotalLagrangianFrame> {
        self.total_lagrangian.as_ref().map(|s| s.frame)
    }
}

mod flight;
use flight::Flight;

mod rest;
use rest::Rest;

#[cfg(all(test, feature = "experimental"))]
mod cap;

#[cfg(all(test, feature = "experimental"))]
mod vibration;

#[cfg(all(test, feature = "experimental"))]
mod modal_flight;

#[cfg(all(test, feature = "experimental"))]
mod impact;

#[cfg(all(test, feature = "experimental"))]
mod hard_frame;

#[cfg(test)]
mod outline_velocity_tests;

#[cfg(all(test, feature = "experimental"))]
mod at_rest_tests;

#[cfg(all(test, feature = "experimental"))]
mod damped_rest_tests;

#[cfg(all(test, feature = "experimental"))]
mod probes;

#[cfg(all(test, feature = "experimental"))]
mod effectivity;

#[cfg(all(test, feature = "experimental"))]
mod water;
