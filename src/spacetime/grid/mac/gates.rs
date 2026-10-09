//! Stop criteria for the MAC projection, written before any of its code and
//! frozen: they do not move once results have been seen.
//!
//! # What is under test
//!
//! The standard formulation (module doc) on the engine's own particle state
//! and units, apart from `Simulation::step`. How it couples to the nodal
//! MLS-MPM grid is the next step, with its own sources read first.
//!
//! # Fixed setup
//!
//! - Gravity is the only thing a scene changes between runs of scenes 1
//!   and 2: 0.38, 1 and 2.5 times 9.81 m/s^2. Scenes 3 and 4 run at 1.
//! - Cells of 1 cm unless stated. No equation of state, no viscosity, no
//!   surface tension: the projection alone holds the fluid.
//! - Frame 1/60 s. Substeps are chosen so a particle moving at the largest
//!   speed, plus what gravity adds over the substep, travels at most one
//!   cell (Zhu and Bridson 4.2.5). Particles keep the velocity they
//!   gather. Positions: `x += v dt`, as the engine's G2P does, for the
//!   first two gate runs; from the third, the midpoint rule through the
//!   projected velocity (Bridson and Muller-Fischer 3.1), a declared
//!   change of setup with its reason at `Scene::substep`. The criteria
//!   below did not move.
//! - Four particles per cell, each placed at random inside its quarter of
//!   the cell; those within one cell of a free surface are moved along its
//!   normal to half a cell from it (Zhu and Bridson 4.2.1). Particle radius
//!   in the level set: the particle spacing; kernel radius: twice that
//!   (their section 5).
//! - J is the fluid's own: its logarithm advances by `dt div v` each
//!   substep through `advance_log_volume_ratio`, with the engine's bounds
//!   [0.5, 2.0]. For the first three gate runs `div v` was `tr C`, `C` the
//!   velocity gradient gathered from the MAC grid; from the fourth, the
//!   divergence of the liquid cells at the particle, a declared change of
//!   setup with its reason at `Scene::substep`.
//!   The fluid's volume is `sum(V0 J)`.
//! - Named settings, each with its reason at its definition: the ghost
//!   fraction floor (0.01, as `apic2d`), the conjugate gradient tolerance
//!   and iteration cap, the MIC(0) parameter, the extrapolation depth, the
//!   travel per substep.
//! - Not allowed: a relaxation factor on the correction, a velocity clamp,
//!   damping, pushing particles back out of walls, or a J at its bounds.
//!
//! # Scene 1: column at rest
//!
//! Water filling a tank 0.40 m wide from wall to wall, 0.30 m deep, open
//! top, at rest, 5 s.
//!
//! - At every frame and every fluid cell of the central column, pressure
//!   within half a cell of head of `g (h - y)`, `h` the fill height.
//! - At 0.5 cm cells, at 1 g, the largest pressure error, in metres of
//!   head, is not larger than at 1 cm.
//! - No particle ever more than one cell from where it started.
//! - The surface, where the level set crosses zero on the central column,
//!   within half a cell of its height at the start.
//!
//! # Scene 2: droplet in free fall
//!
//! Disc of radius 5 cm, at rest, falling 0.3 s in a domain 1.44 m tall
//! that it never touches, at 2.5 g included. Exact answer: zero pressure,
//! translation at g, unchanged shape.
//!
//! - Largest pressure magnitude at most a hundredth of one cell of head.
//! - Centre-of-mass acceleration over each frame equal to g within 0.1
//!   percent.
//! - Every particle's speed relative to the centre of mass at most 0.1
//!   percent of `g t`.
//! - Radius of gyration within 0.5 percent of its start.
//! - Every J within 0.001 of 1.
//!
//! # Scene 3: dam break
//!
//! Column 0.20 m wide and 0.40 m high against the left wall of a closed
//! tank 0.64 m by 0.64 m, 2 s.
//!
//! - No NaN, and no particle outside the tank.
//! - Volume, `sum(V0 J)`, within 2 percent of its start at every frame.
//! - Kinetic plus potential energy never above its start by more than 1
//!   percent: the projection only removes kinetic energy (Bridson and
//!   Muller-Fischer eq. 4.41) and still walls do no work.
//! - Recorded, not judged: the front position over time, compared with an
//!   experiment only once its paper has been read, and the number of
//!   particles per interior fluid cell, a volume measure independent of J.
//!
//! # Scene 4: drop into a pool
//!
//! Disc of radius 5 cm released at rest with its centre 0.40 m above the
//! floor, over water 0.15 m deep filling the same closed tank, 2 s. Same
//! criteria as scene 3.
//!
//! # Each component alone
//!
//! The staggered transfer, the level set, the solid weights, the ghost
//! fluid assembly, the solver and the extrapolation each have their own
//! tests, with no state shared between them. The solver takes a system and
//! returns a solution.
//!
//! # Measured and reported, not judged
//!
//! Substeps per frame, conjugate gradient iterations per substep and their
//! largest value, solves stopped by the iteration cap, and milliseconds per
//! simulated second (debug build, so only for comparison between scenes).
//!
//! # Counting failures
//!
//! A failure is a full run of the four scenes in which any criterion above
//! is missed, once the implementation is believed complete. After the
//! first: one diagnosis, one fix, labelled real fix or declared
//! approximation; a crutch is not allowed. After the second: stop, record
//! it in `KNOWN_LIMITATIONS.md` and the plan, and phase 7 goes first.
//!
//! # Compressible projection (step A1), criteria written before its code
//!
//! The generalized Chorin projection of Stomakhin, Schroeder, Jiang, Chai,
//! Teran and Selle, *Augmented MPM for phase-change and varied materials*,
//! 2014, eqs. 14 to 18: `dp/dt = -K div v` taken implicitly with the
//! pressure update, so the pressure system gains `1 / (c^2 dt)` on its
//! diagonal and `q_n / (c^2 dt)` on its right-hand side (`q = p / rho`,
//! `c^2 = K / rho`, `q_n` from the particles' J through the linear
//! equation of state `q = -c^2 (J - 1)`, their eq. 10 without plasticity).
//! As `c` grows the system becomes scenes 1 to 4's. Same rules as above:
//! no clamp, no damping, no relaxation; positions and J as declared above.
//!
//! - Scene 5, compressible column at rest: scene 1's tank, 1 cm cells,
//!   1 g, `c = 10 m/s`, 5 s. Pressure within half a cell of head of
//!   `g (h - y)` with `h` the surface height of that frame, from t = 1 s
//!   (amended with the user's approval after the second run: a column
//!   released uncompressed rings while it settles, and no initial state met
//!   both this criterion at every frame and the surface one below); the
//!   mean J of
//!   the particles in each 5-cell band within 0.005 of `1 - g (h - y) /
//!   c^2` over the last second; the surface, averaged over the last second,
//!   lowered from its start by `g h^2 / (2 c^2)` within 0.15 cell.
//! - Scene 6, sound speed: a closed tube 200 cells long and 8 high, full,
//!   no gravity, `c = 10 m/s`, the 10 cells at one end starting at J =
//!   0.99, substeps of `dx / c`. The pressure peak reaches the cell 150
//!   from that end at `(150 - 5) dx / c` within 5 percent.
//! - Scene 7, real water: scenes 3 and 4 with `K = 2.2e9 Pa`, the same
//!   criteria as there, and substeps per frame at most 1.1 times those of
//!   the incompressible runs: the real bulk modulus must not bring back an
//!   acoustic time step.
//! - Scenes 1 to 4 keep passing unchanged.
//!
//! Failures counted as above: two failed full runs stop the step.
//!
//! # Elastic solid on the staggered grid (step A2), criteria first
//!
//! Stomakhin et al. 2014's split (their eqs. 7, 8, 13): the shear part of
//! the fixed corotated energy taken on the isochoric deformation,
//! `mu |J^(-1/d) F - R|^2`, gives a deviatoric Kirchhoff stress applied as
//! forces on the faces, explicitly; the volume part `lambda / 2 (J - 1)^2`
//! is the compressible projection of step A1 with `c^2 = lambda / rho`.
//! Plane strain, `lambda` and `mu` from `E` and `nu`. Same rules: no clamp,
//! no damping, no relaxation. Since step M3 the volume part acts through
//! the particles instead, with the shear part (M3 below).
//!
//! - Scene 8, confined elastic column at rest: scene 5's tank filled with
//!   the solid, `nu = 0.3`, `E` such that `(lambda + mu) / rho = (10
//!   m/s)^2`, the uniaxial-strain stiffness of this split in 2D; starting
//!   unstressed, 5 s. Over the last second, the mean J of each 5-cell band
//!   within 0.005 of `1 - g (h - y) rho / (lambda + mu)`; no particle out
//!   of the tank; no NaN. Amended with the material matrix (step M1): that
//!   stiffness was the split's own, with `lambda` as its volume modulus;
//!   real plane-strain elasticity has `lambda + 2 mu`, which the volume part
//!   now gives (`Scene::with_lame`), and the expected J is `1 - g (h - y)
//!   rho / (lambda + 2 mu)`, read from the scene's own coefficients.
//! - Scene 9, plane shear wave (amended after run 2, user's approval: the
//!   first version was a free strip 16 cells high timed by the peak of its
//!   velocity; a free surface along the strip forces `sigma_xy = 0`, which
//!   no plane shear wave satisfies, so the strip bends instead, and the
//!   peak at 150 cells was 1 to 6 percent of the kick and moved from 14
//!   percent early to 50 percent late with the height; it measured no
//!   speed). Now a free block 100 cells long and 208 high, no gravity,
//!   `nu = 0.3`, `E` such that `c_s = sqrt(mu / rho) = 5 m/s`, the 10
//!   cells at one end starting with an upward velocity `v0` = 0.05 m/s
//!   over the whole height, substeps of `dx / (2 c_s)`. Motion uniform
//!   along y obeys the 1D wave equation for `v_y` at `c_s`, so by
//!   d'Alembert a front of height `v0 / 2` leaves the kicked slab. Within 4
//!   cells of mid-height, a free edge's disturbance needs `(104 - 4) dx /
//!   c_p` (`c_p / c_s = sqrt(2 (1 - nu) / (1 - 2 nu))`, 53 `dx / c_s`) to
//!   arrive, after the window below. The mean vertical velocity of the
//!   particles within one cell of `x = 50` cells from that end, and within
//!   4 cells of mid-height, first reaches `v0 / 4` (half the front) at
//!   `40 dx / c_s` within 5 percent: the front leaves from the slab's inner
//!   edge, 10 cells from the end, 40 cells before the probe.
//! - Scene 10, elastic block dropped: a 10 by 10 cell block, `nu = 0.3`,
//!   `c_s = 5 m/s`, released at rest with its base 20 cells above the
//!   floor of scenes 3 and 4's closed tank, 1 g, 2 s. Kinetic, potential
//!   and elastic energy never above the start by more than 1 percent; no
//!   particle out; no NaN; at 2 s the block's radius of gyration about its
//!   centre within 5 percent of its start (it springs back).
//! - Scenes 1 to 7 keep passing unchanged.
//!
//! Failures counted as above: two failed full runs stop the step.
//!
//! Amended for the third attempt, with the user's approval: a particle
//! found beyond a wall after its step is moved to its mirror point (`x ->
//! 2 w - x`), the map the walls' image construction implies
//! (`extrapolate::constrain_to_solids`; Zhu and Bridson 4.2.5 push
//! particles back out of solids). Why: a wall is an invariant line of the
//! mirrored flow, where the normal velocity is zero, and where the liquid
//! leaves the wall it is an unstable one. The image continues the same
//! line, `u = a (x - w)`, behind the wall, so a particle put an ulp beyond
//! it by rounding is carried away as `exp(a t)`. Measured in the second
//! attempt's run 2: a dam-break particle gliding on the left wall plane
//! went 4 ulps beyond it with the f32 noise of the gathered normal velocity
//! (1.6e-5 cells/s), then 4.45 cells into the wall with `a = 26 1/s`. So
//! that the reflection cannot hide a real crossing, every reflection is
//! counted with its depth before it, and every scene with walls requires
//! the deepest at most 4 ulps of the wall's coordinate: one rounding of
//! the position's sum is half an ulp, and the noise step, `1.6e-5 x 3 ms`,
//! a tenth of one, with a declared factor of 4 over that.
//!
//! Amended again, by the user's criteria (precision, speed, every
//! material): that budget left out the pressure solve's own tolerance.
//! The solve stops at a residual divergence `r` (1/s), which leaves a face
//! velocity off by about `r dx`, a step of `dt dx r` towards a wall.
//! Measured with MIC(0) on the resting column at 2.5 g: `r` near 4e-4 1/s
//! at substeps of 16.7 ms gives 6.8e-6 cell, 14 ulps at the floor; the
//! crossing was 13.5. The multigrid cycle overshoots the tolerance and had
//! hidden the term. Each reflection's depth is now held to `4 ulp + 4 dt
//! dx r`, `r` the residual that substep's solve actually reached, the same
//! declared factor of 4 over both terms. A crossing at a physical
//! velocity stays orders of magnitude above it.

//!
//! # The material matrix (step M1), criteria first
//!
//! Every material runs the same scenes, set up and judged from its
//! measured coefficients alone, `rho`, Lame `lambda` and `mu` (plane
//! strain): no scene, tolerance or expected value is written for one
//! material. A row is a material; adding one adds no code. Rows: water
//! (`K = 2.2e9 Pa`, `rho = 1000 kg/m^3`, `mu = 0`), and the elastic part
//! of the very soft and medium clays of `examples/cpu/vonmises_clay_scene`
//! (FHWA NHI-06-088 and NAVFAC DM 7.01: `E = 300 c`, `c = q_u / 2`, `q_u`
//! 20 and 75 kPa, `nu = 0.45`, `rho = 1700 kg/m^3`).
//!
//! - P wave: a free block, no gravity, its first 10 cells kicked along x
//!   at 0.05 m/s over the whole height. The mean x velocity within one
//!   cell of 50 cells from the end, and 4 cells of mid-height, first
//!   reaches a quarter of the kick at `40 dx / c_p` within 5 percent, `c_p
//!   = sqrt((lambda + 2 mu) / rho)` (d'Alembert, as scene 9). The block is
//!   high enough that its free edges' disturbance, at most `c_p`, arrives
//!   after the window.
//! - S wave, for `mu > 0`: scene 9 with the row's coefficients, the height
//!   from the row's own `c_p / c_s`; `c_s = sqrt(mu / rho)`.
//! - Every scene: no NaN, no particle beyond a wall past its budget, and
//!   its wall time per simulated second against real time, reported.
//!
//! Added, criteria first (step M2): the scenes at the substep each material
//! runs at in a game, where its cost against real time is the verdict.
//!
//! - Rest: a column 40 cells wide and 30 deep filling a tank of slip
//!   walls, 1 cm cells, released unstressed under 1 g, 2 s. Over the last
//!   second the mean J of each 5-cell band within `0.05 rho g h / M + a T`
//!   of `1 - rho g (h - y) / M`, `M = lambda + 2 mu` (uniaxial strain),
//!   where `a T` is the drift the pressure solve's absolute tolerance `a`
//!   (1/s) allows J over the run's `T` seconds.
//! - Drop: a 10 by 10 cell block of the material released at rest 20
//!   cells above the floor of a closed tank, 1 g, 1 s. Kinetic,
//!   potential, shear and volume energy never above the start by more than
//!   1 percent; for `mu > 0`, the radius of gyration about the centre
//!   within 5 percent of the start at the end (it springs back); for `mu =
//!   0`, the volume within 2 percent (scenes 3 and 4's bound).
//! - Both: the wall time per simulated second, reported against real time
//!   (1000 ms); no criterion yet, as A2's explicit shear bounds the
//!   substep by the shear wave (the P wave since step M3), so the stiffer
//!   the solid the slower.
//!
//! Failures counted as above: two failed full runs stop the step.
//!
//! # A free body keeps its momentum (step M3), criteria first
//!
//! A body with no external force keeps its linear and angular momentum,
//! whatever its material. Measured before this step: a free block of the
//! very soft clay, vibrating, accelerated its centre to 0.27 m/s in 1 s
//! at 1 cm cells and to 0.45 m/s, spinning at 6.3 rad/s, at 0.5 cm, the
//! free surface's pressure giving it a net force (tanks hid it: their walls
//! take any net force).
//!
//! - Free body: every row, a 10 by 10 cell block at the centre of a 72
//!   cell tank, no gravity, never reaching a wall, started with the pure
//!   strain rate `v = s (y - c_y, x - c_x)` whose largest particle speed is
//!   0.5 m/s, the same for every row, at 1 and 0.5 cm cells, 1 s. (Set
//!   before any run: a strain over a sound crossing gave water 37 m/s.) The centre's velocity stays within 1 percent of the
//!   start's largest particle speed, and the angular momentum about the
//!   centre within 1 percent of `I s` (the start's inertia times the strain
//!   rate), at every frame; no NaN; energy never above the start by more
//!   than 1 percent.
//! - Every scene of steps A1, A2, M1 and M2 keeps passing.
//!
//! Failures counted as above: two failed full runs stop the step.
//!
//! Amendment, approved by the user after the second failed attempt: a row
//! with no shear stiffness (`mu = 0`) cannot hold a strain rate, so its
//! block stretched into the walls within the second, whose forces then
//! changed its momentum (the decomposition below did not telescope to
//! zero for water alone).
//! Such a row starts in pure translation instead: the block 4 cells from
//! two walls, moving at 0.5 m/s along the tank's diagonal, which keeps it
//! 14 cells from the far walls after 1 s. The bounds are unchanged, the
//! centre's velocity read against its start, the angular momentum's scale
//! `I s` with `s` the largest start speed over the half diagonal.
//!
//! What the attempts measured (probes and runs, every change reverted but
//! the last, backups kept outside the tree):
//! - The projection changes the faces' momentum by exactly `sum_f (m_f -
//!   rho V_f) du_f`, `V_f` the volume the kinematic update assumes of face
//!   `f`. On the free very soft clay the terms came from the bulk's sampled
//!   mass, from the surface's kernel-spread mass against the level set's
//!   `theta`, and from the faces past the surface that take the continued
//!   pressure.
//! - First attempt, the surface's `theta` from the face mass and no
//!   continued pressure: the clays' drift unchanged, the resting column of
//!   scene 1 broke.
//! - Second, the solved pressure applied as each particle's stress: the
//!   clay's centre held, water lost about 10 percent of its volume, the
//!   solved pressure belonging to the MAC operator, not to that one.
//! - Third, approved: Newton's law on every face, `du = dt F / m_f`. A
//!   resting liquid convected, as Ding, Shinar and Schroeder (MAC APIC, JCP
//!   408, 2020, 2.4.3) report of grid masses in the projection, "an
//!   initially stationary pool of water would develop currents in it", and
//!   the clays blew up on faces of vanishing mass.
//! - Fourth, approved: the power weights of Qu, Li, de Goes and Jiang (The
//!   Power Particle-In-Cell Method, TOG 41(4) 118, 2022), face masses exact
//!   to the transport plan's tolerance. Water clean, the clay still drifted:
//!   a compressed solid's density really varies, `rho0 / J`, which a
//!   density-blind projection cannot conserve.
//!
//! Result: a solid is not projected. Its whole elastic stress, the shear
//! part and the volume part `K J (J - 1) I` with `J = det F` and `K =
//! lambda + mu` the plane-strain bulk modulus, acts through its particles
//! (`transfer::stress_to_faces`, the engine's MPM force). The kernel's
//! gradients sum to zero over the faces, so a particle's stress moves no
//! momentum in or out whatever the stress or the sampling, and a symmetric
//! stress no angular momentum. Liquids keep the projection. A solid's
//! substep follows its P wave, `c_p^2 = (K + mu) / rho`: 3.3 times the
//! substeps on the matrix's clays (`nu = 0.45`, `c_p / c_s = sqrt(11)`). A
//! face on a wall plane takes no normal velocity, the grid's own wall
//! condition, which the projection's variational form held before. M3
//! then passes, the centre at 0.0000 of the start speed and the angular
//! momentum within 0.0004 of `I s`, and every gate of A1, A2, M1 and M2
//! passes with the second amendment. Lost against the split: the clays' P
//! wave arrives 4.2 percent late, 1.1 with the split.
//!
//! Second amendment, approved by the user with this change: scene 9 fixed
//! its substep at half a cell per shear wave crossing, which a solid
//! carrying its P wave crosses at 0.94 cell (limit 0.577), and blew up. Its
//! substep now follows the faster wave, as the matrix's plane waves set
//! it; the arrival is 3.3 percent late (3.7 with the split).

use std::time::Instant;

use glam::{IVec2, Mat2, Vec2};

use super::elastic::{deformation_step, shear_energy, shear_kirchhoff};
use super::extrapolate::fold_onto_images;
use super::field::{Field2, MacLayout};
use super::level_set::{SurfaceSettings, liquid_phi};
use super::solid::{
    FaceWeights, box_container, box_container_image, face_weights, sample_centres, sample_corners,
};
use super::transfer::{faces_to_particles, particles_to_faces, stress_to_faces};
use super::{Liquid, ProjectionSettings, Walls, project, travel_limited_dt};
use crate::diagnostics::{OCCUPANCY_BANDS, scene_map};
use crate::fields::EARTH_GRAVITY_M_S2;
use crate::materials::utils::advance_log_volume_ratio;
use crate::particle::{Particle, Particles};
use crate::solver::LcgRng;

const FRAME: f32 = 1.0 / 60.0;
/// The fluid material's own J bounds (`NewtonianFluidMaterial::update_particle`).
const J_BOUNDS: (f32, f32) = (0.5, 2.0);
const GRAVITIES: [f32; 3] = [0.38, 1.0, 2.5];

/// Particles and the still tank they sit in, stepped by the standard
/// projection alone. Lengths are in cells (`dx = 1`); `cell_m` gives the
/// metres per cell.
struct Scene {
    layout: MacLayout,
    cell_m: f32,
    /// Cells per second squared.
    gravity: Vec2,
    tank: (Vec2, Vec2),
    weights: FaceWeights,
    solid_centres: Field2,
    surface: SurfaceSettings,
    settings: ProjectionSettings,
    x: Vec<Vec2>,
    v: Vec<Vec2>,
    c: Vec<Mat2>,
    mass: Vec<f32>,
    v0: Vec<f32>,
    log_j: Vec<f32>,
    j_at_bounds: usize,
    time: f32,
    probe: Probe,
    /// `c^2` in cells^2/s^2 for the compressible projection (step A1);
    /// `None` is the incompressible limit.
    sound_speed2: Option<f32>,
    /// A substep length that replaces the travel limit (scene 6).
    fixed_dt: Option<f32>,
    /// `mu / rho` in cells^2/s^2 for an elastic solid (step A2); `None`
    /// for a liquid.
    shear: Option<f32>,
    /// Each particle's deformation gradient, for the shear stress.
    deformation: Vec<Mat2>,
}

/// Where things happen, for diagnosis only: no criterion reads it.
#[derive(Default)]
struct Probe {
    /// J reaching its bounds, by where the particle was: in a liquid cell
    /// of the tank, in an air cell of the tank, inside a wall.
    j_at_bounds_by_place: [usize; 3],
    /// Particles moved back from beyond a wall (`substep`), and the
    /// deepest of them before the move, in ulps of the wall's coordinate.
    reflections: u32,
    deepest_reflection_ulps: f32,
    /// The largest reflection's depth over its budget (criteria, second
    /// amendment), and the budget's solver term of the current substep,
    /// `4 dt dx r`.
    worst_reflection_ratio: f32,
    solve_drift: f32,
    /// Particles leaving the tank: time, position before, velocity,
    /// distance to the nearest wall before.
    crossings: Vec<(f32, Vec2, Vec2, f32)>,
    /// Wall-clock time per phase of the substep, in microseconds: P2G and
    /// gravity, level set, assembly for the frame record, projection
    /// (assembly, solve, update, extrapolation, walls), the two gathers,
    /// the particle update.
    phase_us: [u128; 6],
    /// Cell whose pressure every substep records, with the time (scene 6).
    watch: Option<usize>,
    watch_history: Vec<(f32, f32)>,
    /// Particles within one cell of this x and between these two heights:
    /// their mean vertical velocity every substep (scene 9).
    watch_x: Option<(f32, f32, f32)>,
    /// The watch reads the x velocity instead of y (a P wave along x).
    watch_longitudinal: bool,
    watch_x_history: Vec<(f32, f32)>,
    /// Where and when the deepest reflection happened, and the particle's
    /// position before the step that took it there.
    deepest_reflection_at: Option<(f32, Vec2)>,
}

/// What one frame's last substep left, and what the frame cost.
struct Frame {
    pressure: Vec<f32>,
    active: Vec<bool>,
    phi: Field2,
    /// Face velocity the last substep's particles gathered from.
    vel: super::field::MacVelocity,
    substeps: u32,
    iterations: Vec<u32>,
    capped: u32,
    largest_travel: f32,
}

impl Scene {
    fn new(
        (nx, ny): (usize, usize),
        cell_m: f32,
        g_fraction: f32,
        tank: (Vec2, Vec2),
        x: Vec<Vec2>,
    ) -> Self {
        let layout = MacLayout::new(nx, ny, 1.0);
        let solid = box_container(tank.0, tank.1);
        let corners = sample_corners(&layout, solid);
        let spacing = 0.5 * layout.dx;
        let n = x.len();
        Self {
            layout,
            cell_m,
            gravity: Vec2::new(0.0, -g_fraction * EARTH_GRAVITY_M_S2 / cell_m),
            tank,
            weights: face_weights(&layout, &corners),
            solid_centres: sample_centres(&layout, solid),
            surface: SurfaceSettings::from_spacing(spacing),
            settings: ProjectionSettings::default(),
            x,
            v: vec![Vec2::ZERO; n],
            c: vec![Mat2::ZERO; n],
            mass: vec![spacing * spacing; n],
            v0: vec![spacing * spacing; n],
            log_j: vec![0.0; n],
            j_at_bounds: 0,
            time: 0.0,
            probe: Probe::default(),
            sound_speed2: None,
            fixed_dt: None,
            shear: None,
            deformation: vec![Mat2::IDENTITY; n],
        }
    }

    fn g(&self) -> f32 {
        self.gravity.length()
    }

    /// A plane-strain elastic solid of Young's modulus over density
    /// `e_over_rho` (m^2/s^2) and Poisson's ratio `nu` (`with_lame`).
    fn with_elastic(self, e_over_rho: f32, nu: f32) -> Self {
        let mu = e_over_rho / (2.0 * (1.0 + nu));
        let lambda = e_over_rho * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        self.with_lame(lambda, mu)
    }

    /// A material by its plane-strain Lame coefficients over its density,
    /// `lambda / rho` and `mu / rho` in m^2/s^2. With `mu > 0` a solid: its
    /// whole stress, shear and volume, through its particles (step M3);
    /// with `mu = 0` a liquid, its volume in the compressible projection.
    fn with_lame(mut self, lambda_over_rho: f32, mu_over_rho: f32) -> Self {
        let cells2 = 1.0 / (self.cell_m * self.cell_m);
        self.shear = (mu_over_rho > 0.0).then_some(mu_over_rho * cells2);
        // The shear part acts on the isochoric deformation only, so all
        // the volume stiffness is the volume part's (the particles' for a
        // solid since step M3, the projection's for a liquid): the
        // plane-strain bulk modulus `kappa = lambda + mu`. With it, `2 mu dev(e) + kappa tr(e)
        // I = 2 mu e + lambda tr(e) I`, Lame's law; with `lambda` alone, as
        // Stomakhin et al.'s split writes the volume energy, the P wave ran
        // at `sqrt((lambda + mu) / rho)`: 6.0 percent late on the clays of
        // the matrix (`nu = 0.45`), against water's 1.0.
        self.sound_speed2 = Some((lambda_over_rho + mu_over_rho) * cells2);
        self
    }

    /// The compressible projection at `c_m_s`.
    fn with_sound_speed(mut self, c_m_s: f32) -> Self {
        let c = c_m_s / self.cell_m;
        self.sound_speed2 = Some(c * c);
        self
    }

    /// Mass-weighted J at the cell centres, bilinear, and the pressure the
    /// linear equation of state gives it, `q = -c^2 (J - 1)`; zero where no
    /// particle reaches.
    fn cell_pressure_from_j(&self, sound_speed2: f32) -> Vec<f32> {
        let (nx, ny) = (self.layout.nx, self.layout.ny);
        let mut sum_j = vec![0.0f32; nx * ny];
        let mut sum_w = vec![0.0f32; nx * ny];
        for p in 0..self.x.len() {
            let t = self.x[p] / self.layout.dx - Vec2::splat(0.5);
            let (i0, j0) = (t.x.floor() as i32, t.y.floor() as i32);
            let (fx, fy) = (t.x - i0 as f32, t.y - j0 as f32);
            let j_p = self.j(p);
            for (di, dj, w) in [
                (0, 0, (1.0 - fx) * (1.0 - fy)),
                (1, 0, fx * (1.0 - fy)),
                (0, 1, (1.0 - fx) * fy),
                (1, 1, fx * fy),
            ] {
                let (i, j) = (i0 + di, j0 + dj);
                if i < 0 || j < 0 || i as usize >= nx || j as usize >= ny {
                    continue;
                }
                let c = i as usize + nx * j as usize;
                sum_w[c] += w * self.mass[p];
                sum_j[c] += w * self.mass[p] * j_p;
            }
        }
        (0..nx * ny)
            .map(|c| {
                if sum_w[c] > 0.0 {
                    -sound_speed2 * (sum_j[c] / sum_w[c] - 1.0)
                } else {
                    0.0
                }
            })
            .collect()
    }

    /// The liquid level set with the tank's walls folded in (`apic2d`).
    fn phi(&self) -> Field2 {
        let mut phi = liquid_phi(&self.layout, &self.x, &self.surface);
        for (value, &solid) in phi.data_mut().iter_mut().zip(self.solid_centres.data()) {
            *value = value.min(solid);
        }
        phi
    }

    fn substep(&mut self, dt: f32, frame: &mut Frame) {
        let mut clock = Instant::now();
        let mut lap = |probe: &mut Probe, k: usize| {
            probe.phase_us[k] += clock.elapsed().as_micros();
            clock = Instant::now();
        };
        let (mut vel, face_mass) =
            particles_to_faces(&self.layout, &self.x, &self.v, &self.c, &self.mass);
        if let Some(mu) = self.shear {
            // The whole elastic stress on the particles (module doc, step
            // M3): the shear part on the isochoric deformation and the
            // volume part `K J (J - 1) I`, `J = det F`, with `K` the
            // plane-strain bulk modulus `with_lame` keeps.
            let bulk = self.sound_speed2.unwrap_or(0.0);
            let tau_v0: Vec<Mat2> = (0..self.x.len())
                .map(|p| {
                    let f = self.deformation[p];
                    let j = f.determinant();
                    let volume = bulk * j * (j - 1.0) * Mat2::IDENTITY;
                    self.v0[p] * (shear_kirchhoff(f, mu) + volume)
                })
                .collect();
            let mut force = stress_to_faces(&self.layout, &self.x, &tau_v0);
            let mut lent = super::field::MacVelocity {
                u: face_mass.u.clone(),
                v: face_mass.v.clone(),
            };
            let image = box_container_image(self.tank.0, self.tank.1);
            fold_onto_images(&self.layout, &mut force, &self.weights, &image, true);
            fold_onto_images(&self.layout, &mut lent, &self.weights, &image, false);
            for (field, f, m) in [
                (&mut vel.u, &force.u, &lent.u),
                (&mut vel.v, &force.v, &lent.v),
            ] {
                for ((value, &f), &m) in field.data_mut().iter_mut().zip(f.data()).zip(m.data()) {
                    if m > 0.0 {
                        *value += dt * f / m;
                    }
                }
            }
        }
        for value in vel.u.data_mut() {
            *value += self.gravity.x * dt;
        }
        for value in vel.v.data_mut() {
            *value += self.gravity.y * dt;
        }
        lap(&mut self.probe, 0);
        let phi = self.phi();
        lap(&mut self.probe, 1);
        let system = super::pressure::assemble(
            &self.layout,
            dt,
            &vel,
            &self.weights,
            super::pressure::Surface {
                phi: &phi,
                theta_floor: self.settings.theta_floor,
            },
        );
        lap(&mut self.probe, 2);
        let image = box_container_image(self.tank.0, self.tank.1);
        let solution = if self.shear.is_some() {
            let mut valid = super::field::FaceFlags {
                u: face_mass.u.data().iter().map(|&m| m > 0.0).collect(),
                v: face_mass.v.data().iter().map(|&m| m > 0.0).collect(),
            };
            // A solid is not projected (module doc, step M3). The wall
            // condition the projection's variational form holds (`pressure`
            // module doc) is then the grid's own, as standard MPM sets it:
            // a face on a wall plane has no normal velocity and keeps none
            // through the extension. Without it a settling column crossed
            // the floor by 3704 ulps in its first frame.
            let (lo, hi) = self.tank;
            let on_plane = |a: f32, b: f32| (a - b).abs() <= 1e-6 * self.layout.dx;
            for j in 0..self.layout.ny {
                for i in 0..=self.layout.nx {
                    let q = self.layout.u_position(i, j);
                    if on_plane(q.x, lo.x) || on_plane(q.x, hi.x) {
                        vel.u.set(i, j, 0.0);
                        valid.u[vel.u.index(i, j)] = true;
                    }
                }
            }
            for j in 0..=self.layout.ny {
                for i in 0..self.layout.nx {
                    let q = self.layout.v_position(i, j);
                    if on_plane(q.y, lo.y) || on_plane(q.y, hi.y) {
                        vel.v.set(i, j, 0.0);
                        valid.v[vel.v.index(i, j)] = true;
                    }
                }
            }
            super::extrapolate::extrapolate_velocity(
                &mut vel,
                &mut valid,
                self.settings.extrapolation_layers,
            );
            super::extrapolate::constrain_to_solids(&self.layout, &mut vel, &self.weights, &image);
            super::pcg::Solution {
                pressure: vec![0.0; self.layout.nx * self.layout.ny],
                iterations: 0,
                residual: 0.0,
                converged: true,
            }
        } else {
            let q_before = self
                .sound_speed2
                .map(|c2| (c2, self.cell_pressure_from_j(c2)));
            let material: Vec<bool> = self.solid_centres.data().iter().map(|&d| d > 0.0).collect();
            project(
                &self.layout,
                dt,
                &mut vel,
                Walls {
                    weights: &self.weights,
                    image: &image,
                    open: &material,
                },
                Liquid {
                    phi: &phi,
                    face_mass: &face_mass,
                },
                &self.settings,
                q_before
                    .as_ref()
                    .map(|(c2, q)| (*c2, q.as_slice(), material.as_slice())),
            )
        };
        lap(&mut self.probe, 3);
        faces_to_particles(&self.layout, &vel, &self.x, &mut self.v, &mut self.c);
        // Positions advance by the midpoint rule (RK2) through the projected
        // face velocity, as Bridson and Muller-Fischer 3.1 recommend over
        // forward Euler for trajectories. Against a wall the normal velocity
        // falls linearly to zero, `v = -a d`: forward Euler moves `d` to `d (1
        // - a dt)`, which crosses the wall once `a dt > 1`, while the midpoint
        // rule gives `d (1 - a dt + (a dt)^2 / 2)`, positive for every step.
        // After the corner fix, 71 of the 72 remaining crossings were water
        // decelerating against the lid within one cell. The particle keeps
        // the velocity and `C` gathered at its start, as before.
        let midpoint: Vec<Vec2> = (0..self.x.len())
            .map(|p| self.x[p] + 0.5 * dt * self.v[p])
            .collect();
        let mut v_mid = vec![Vec2::ZERO; self.x.len()];
        let mut c_mid = vec![Mat2::ZERO; self.x.len()];
        faces_to_particles(&self.layout, &vel, &midpoint, &mut v_mid, &mut c_mid);
        lap(&mut self.probe, 4);
        if self.shear.is_some() {
            for p in 0..self.x.len() {
                self.deformation[p] = deformation_step(dt * self.c[p]) * self.deformation[p];
            }
        }
        if let Some((x0, y_lo, y_hi)) = self.probe.watch_x {
            let (sum, n) = (0..self.x.len())
                .filter(|&p| {
                    (self.x[p].x - x0).abs() < self.layout.dx && (y_lo..y_hi).contains(&self.x[p].y)
                })
                .fold((0.0f32, 0u32), |(s, n), p| {
                    let v = if self.probe.watch_longitudinal {
                        self.v[p].x
                    } else {
                        self.v[p].y
                    };
                    (s + v, n + 1)
                });
            self.probe
                .watch_x_history
                .push((self.time + dt, sum / n.max(1) as f32));
        }
        let ln_bounds = (J_BOUNDS.0.ln(), J_BOUNDS.1.ln());
        // J advances by the divergence the projection holds: the cells'
        // divergence interpolated bilinearly at the particle, over liquid
        // cells only. `tr C` reads the same in the bulk (the bulk divergence
        // probe: 1.32e-5 against 1.31e-5 1/s), but near the free surface the
        // gather reaches the velocity extrapolated into the air, which no
        // solve made divergence free (Bridson and Muller-Fischer 6.3's
        // constant extension is not either), and J drifted 5 % there while
        // the particles per interior cell, a volume measure that does not
        // read J, held at 4.01. A particle with no liquid cell around it is
        // in free flight and reads zero.
        let (nx, ny) = (self.layout.nx as i32, self.layout.ny as i32);
        let liquid = |i: i32, j: i32| {
            i >= 0
                && j >= 0
                && i < nx
                && j < ny
                && phi.get(i as usize, j as usize) < 0.0
                && self.solid_centres.get(i as usize, j as usize) > 0.0
        };
        let cell_div = |i: i32, j: i32| {
            let (i, j) = (i as usize, j as usize);
            (vel.u.get(i + 1, j) - vel.u.get(i, j) + vel.v.get(i, j + 1) - vel.v.get(i, j))
                / self.layout.dx
        };
        for (p, v_step) in v_mid.iter().enumerate() {
            let dt_div = {
                let t = self.x[p] / self.layout.dx - Vec2::splat(0.5);
                let (i0, j0) = (t.x.floor() as i32, t.y.floor() as i32);
                let (fx, fy) = (t.x - i0 as f32, t.y - j0 as f32);
                let (mut sum, mut weight) = (0.0f32, 0.0f32);
                for (di, dj, w) in [
                    (0, 0, (1.0 - fx) * (1.0 - fy)),
                    (1, 0, fx * (1.0 - fy)),
                    (0, 1, (1.0 - fx) * fy),
                    (1, 1, fx * fy),
                ] {
                    if liquid(i0 + di, j0 + dj) {
                        sum += w * cell_div(i0 + di, j0 + dj);
                        weight += w;
                    }
                }
                if weight > 0.0 { dt * sum / weight } else { 0.0 }
            };
            let advanced = self.log_j[p] + dt_div;
            let step = *v_step * dt;
            let was_inside = self.inside_tank(p);
            if advanced <= ln_bounds.0 || advanced >= ln_bounds.1 {
                self.j_at_bounds += 1;
                let cell = self.x[p].floor().as_ivec2();
                let wet = phi.get_signed(cell.x, cell.y).is_some_and(|f| f < 0.0);
                let place = if !was_inside {
                    2
                } else if wet {
                    0
                } else {
                    1
                };
                self.probe.j_at_bounds_by_place[place] += 1;
            }
            self.log_j[p] =
                advance_log_volume_ratio(self.log_j[p], dt_div, J_BOUNDS.0, J_BOUNDS.1).0;
            frame.largest_travel = frame.largest_travel.max(step.length());
            let before = self.x[p];
            self.x[p] += step;
            if was_inside && !self.inside_tank(p) {
                let (lo, hi) = (before - self.tank.0, self.tank.1 - before);
                let wall = lo.x.min(lo.y).min(hi.x).min(hi.y);
                self.probe
                    .crossings
                    .push((self.time, before, self.v[p], wall));
            }
        }
        self.probe.solve_drift = 4.0 * dt * self.layout.dx * solution.residual;
        for p in 0..self.x.len() {
            self.reflect_off_walls(p);
        }
        lap(&mut self.probe, 5);
        self.time += dt;
        frame.substeps += 1;
        frame.iterations.push(solution.iterations);
        frame.capped += u32::from(!solution.converged);
        if let Some(c) = self.probe.watch {
            self.probe
                .watch_history
                .push((self.time, solution.pressure[c]));
        }
        frame.pressure = solution.pressure;
        frame.active = system.active;
        frame.phi = phi;
        frame.vel = vel;
    }

    /// Fixed substeps (`fixed_dt`) until `t_end`, for scenes shorter than
    /// a frame.
    fn run_until(&mut self, t_end: f32) {
        let dt = self.fixed_dt.expect("run_until needs a fixed substep");
        let mut frame = Frame {
            pressure: Vec::new(),
            active: Vec::new(),
            phi: self.layout.cells(0.0),
            vel: super::field::MacVelocity::zeros(&self.layout),
            substeps: 0,
            iterations: Vec::new(),
            capped: 0,
            largest_travel: 0.0,
        };
        while self.time < t_end {
            self.substep(dt.min(t_end - self.time).max(1e-9), &mut frame);
        }
    }

    fn frame(&mut self) -> Frame {
        let mut frame = Frame {
            pressure: Vec::new(),
            active: Vec::new(),
            phi: self.layout.cells(0.0),
            vel: super::field::MacVelocity::zeros(&self.layout),
            substeps: 0,
            iterations: Vec::new(),
            capped: 0,
            largest_travel: 0.0,
        };
        let mut left = FRAME;
        while left > 1e-7 {
            let speed = self.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
            // Half a cell per crossing of the fastest wave a solid's
            // particles carry, the P wave `c_p^2 = (K + mu) / rho`, under
            // the explicit limit `2 / sqrt(12) = 0.577` the transfer's
            // stiffest mode allows (`transfer::the_stiffest_mode_stays_
            // bounded_...`).
            let shear_limit = self.shear.map_or(f32::INFINITY, |mu| {
                0.5 * self.layout.dx / (mu + self.sound_speed2.unwrap_or(0.0)).sqrt()
            });
            let limit = self.fixed_dt.unwrap_or_else(|| {
                shear_limit.min(travel_limited_dt(
                    speed,
                    self.g(),
                    self.layout.dx,
                    self.settings.max_cells_per_substep,
                ))
            });
            let dt = limit.min(left);
            self.substep(dt, &mut frame);
            left -= dt;
        }
        frame
    }

    /// Moves particle `p`, if beyond a wall, to its mirror point across it,
    /// and counts it (criteria, third attempt's amendment).
    fn reflect_off_walls(&mut self, p: usize) {
        let ulp = |w: f32| f32::from_bits(w.abs().to_bits() + 1) - w.abs();
        let (lo, hi) = self.tank;
        let mut q = self.x[p];
        let drift = self.probe.solve_drift;
        let (mut deepest, mut ratio) = (0.0f32, 0.0f32);
        for (coord, low, high) in [(&mut q.x, lo.x, hi.x), (&mut q.y, lo.y, hi.y)] {
            let wall = if *coord < low {
                low
            } else if *coord > high {
                high
            } else {
                continue;
            };
            let depth = (*coord - wall).abs();
            deepest = deepest.max(depth / ulp(wall));
            ratio = ratio.max(depth / (4.0 * ulp(wall) + drift));
            *coord = 2.0 * wall - *coord;
        }
        if deepest > 0.0 {
            self.x[p] = q;
            self.probe.reflections += 1;
            if deepest > self.probe.deepest_reflection_ulps {
                self.probe.deepest_reflection_at = Some((self.time, self.x[p]));
            }
            self.probe.deepest_reflection_ulps = self.probe.deepest_reflection_ulps.max(deepest);
            self.probe.worst_reflection_ratio = self.probe.worst_reflection_ratio.max(ratio);
        }
    }

    /// The reflection criterion (third attempt's amendment).
    fn check_reflections(&self, label: &str, report: &mut Report) {
        let (n, deepest) = (self.probe.reflections, self.probe.deepest_reflection_ulps);
        let ratio = self.probe.worst_reflection_ratio;
        println!(
            "{label}: {n} reflections off a wall, deepest {deepest:.2} ulps before it, worst {ratio:.2} of its budget"
        );
        if let Some((t, at)) = self.probe.deepest_reflection_at {
            println!("{label}: deepest reflection at t = {t:.4} s, mirrored to {at:?}");
        }
        report.check(ratio <= 1.0, || {
            format!("{label}: a particle went {ratio:.2} times its budget beyond a wall ({deepest:.1} ulps)")
        });
    }

    /// Which cells have their centre outside every solid.
    fn open_centres(&self) -> Vec<bool> {
        self.solid_centres.data().iter().map(|&d| d > 0.0).collect()
    }

    /// A particle's volume ratio: a solid's is its own `det F`, the `J` its
    /// stress reads; a liquid's integrates the cells' divergence.
    fn j(&self, p: usize) -> f32 {
        if self.shear.is_some() {
            self.deformation[p].determinant()
        } else {
            self.log_j[p].exp()
        }
    }

    fn volume(&self) -> f32 {
        (0..self.x.len()).map(|p| self.v0[p] * self.j(p)).sum()
    }

    fn centre_of_mass(&self) -> (Vec2, Vec2) {
        let total: f32 = self.mass.iter().sum();
        let x: Vec2 = (0..self.x.len()).map(|p| self.x[p] * self.mass[p]).sum();
        let v: Vec2 = (0..self.x.len()).map(|p| self.v[p] * self.mass[p]).sum();
        (x / total, v / total)
    }

    /// Kinetic plus potential energy, the floor of the tank as zero.
    fn energy(&self) -> f32 {
        let floor = self.tank.0.y;
        let elastic = |p: usize| match (self.shear, self.sound_speed2) {
            (Some(mu), Some(lambda)) => {
                let j = self.j(p);
                self.v0[p]
                    * (shear_energy(self.deformation[p], mu) + 0.5 * lambda * (j - 1.0).powi(2))
            }
            _ => 0.0,
        };
        (0..self.x.len())
            .map(|p| {
                let kinetic = 0.5 * self.v[p].length_squared();
                self.mass[p] * (kinetic + self.g() * (self.x[p].y - floor)) + elastic(p)
            })
            .sum()
    }

    /// In the tank or on its walls: out means beyond a wall. A particle
    /// gliding along a wall approaches it geometrically (the normal velocity
    /// falls linearly to zero there, `substep` doc) and, in f32, ends on
    /// it: measured in scene 4 after step A2's changes, a particle reached
    /// `x = 4.0000005`, one ulp from the wall at 4, with `v_x = -1.6e-5`
    /// cells/s, sat exactly on the wall for one frame and went back in,
    /// never beyond it. The strict test counted it out.
    fn inside_tank(&self, p: usize) -> bool {
        let q = self.x[p];
        q.x >= self.tank.0.x && q.x <= self.tank.1.x && q.y >= self.tank.0.y && q.y <= self.tank.1.y
    }

    /// Particles per cell over cells whose eight neighbours also hold
    /// particles: a volume measure that does not read J.
    fn interior_particles_per_cell(&self) -> f32 {
        let (nx, ny) = (self.layout.nx as i32, self.layout.ny as i32);
        let mut count = vec![0u32; (nx * ny) as usize];
        for q in &self.x {
            let c = q.floor().as_ivec2();
            if c.x >= 0 && c.y >= 0 && c.x < nx && c.y < ny {
                count[(c.x + nx * c.y) as usize] += 1;
            }
        }
        let (mut sum, mut cells) = (0u32, 0u32);
        for j in 1..ny - 1 {
            for i in 1..nx - 1 {
                let full = (-1..=1)
                    .all(|dj| (-1..=1).all(|di| count[(i + di + nx * (j + dj)) as usize] > 0));
                if full {
                    sum += count[(i + nx * j) as usize];
                    cells += 1;
                }
            }
        }
        if cells == 0 {
            0.0
        } else {
            sum as f32 / cells as f32
        }
    }

    /// A text picture of the particles, printed when `EMERGE_GATE_MAPS` is
    /// set.
    fn print_map(&self, label: &str) {
        if crate::diagnostics::research_switch("EMERGE_GATE_MAPS").is_none() {
            return;
        }
        let mut particles = Particles::default();
        for &x in &self.x {
            particles.push(Particle {
                x,
                ..bytemuck::Zeroable::zeroed()
            });
        }
        let size = Vec2::new(self.layout.nx as f32, self.layout.ny as f32);
        let cols = self.layout.nx.min(72);
        let rows = (cols as f32 * size.y / size.x / 2.0).ceil() as usize;
        println!("{label} t={:.3}s", self.time);
        let map = scene_map(
            &particles,
            (Vec2::ZERO, size),
            cols,
            rows.max(1),
            |_| 1.0,
            &OCCUPANCY_BANDS,
        );
        for row in map {
            println!("|{row}|");
        }
    }
}

/// Where the level set crosses zero going up column `i` from the tank
/// floor, by linear interpolation between cell centres.
fn surface_height(phi: &Field2, i: usize, floor_cell: usize) -> Option<f32> {
    (floor_cell + 1..phi.nj()).find_map(|j| {
        let (below, above) = (phi.get(i, j - 1), phi.get(i, j));
        (below < 0.0 && above >= 0.0).then(|| (j as f32 - 0.5) + below / (below - above))
    })
}

/// Four particles per cell over `cells`, each at a random point of its
/// quarter of the cell, kept where `inside` holds; those within one cell
/// of a free surface (`surface` gives the distance and outward normal)
/// moved along the normal to the particle spacing from it (Zhu and
/// Bridson 4.2.1).
fn seed(
    cells: (IVec2, IVec2),
    inside: impl Fn(Vec2) -> bool,
    surface: impl Fn(Vec2) -> (f32, Vec2),
    rng: &mut LcgRng,
) -> Vec<Vec2> {
    let spacing = 0.5;
    let mut x = Vec::new();
    for j in cells.0.y..cells.1.y {
        for i in cells.0.x..cells.1.x {
            for (a, b) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let quarter = Vec2::new(0.25 + 0.5 * a as f32, 0.25 + 0.5 * b as f32);
                let jitter = Vec2::new(rng.next_f32() - 0.5, rng.next_f32() - 0.5) * spacing;
                let mut q = Vec2::new(i as f32, j as f32) + quarter + jitter;
                if !inside(q) {
                    continue;
                }
                let (distance, normal) = surface(q);
                if distance < 1.0 {
                    q += normal * (distance - spacing);
                }
                x.push(q);
            }
        }
    }
    x
}

/// Distance to the nearest of a box's free sides and that side's normal.
fn box_free_sides(q: Vec2, top: Option<f32>, right: Option<f32>) -> (f32, Vec2) {
    let mut best = (f32::INFINITY, Vec2::ZERO);
    if let Some(y) = top
        && y - q.y < best.0
    {
        best = (y - q.y, Vec2::Y);
    }
    if let Some(x) = right
        && x - q.x < best.0
    {
        best = (x - q.x, Vec2::X);
    }
    best
}

fn disc_side(q: Vec2, centre: Vec2, radius: f32) -> (f32, Vec2) {
    let d = q - centre;
    let r = d.length();
    (radius - r, if r > 0.0 { d / r } else { Vec2::Y })
}

/// Every missed criterion of a run, printed as it is found and asserted
/// together at the end, so one run shows them all.
struct Report {
    failures: Vec<String>,
}

impl Report {
    fn new() -> Self {
        Self {
            failures: Vec::new(),
        }
    }

    fn check(&mut self, ok: bool, what: impl FnOnce() -> String) {
        if !ok {
            let line = what();
            println!("  FAIL {line}");
            self.failures.push(line);
        }
    }

    fn finish(self) {
        assert!(
            self.failures.is_empty(),
            "{} criteria missed:\n{}",
            self.failures.len(),
            self.failures.join("\n")
        );
    }
}

/// Totals over a run, for the cost lines.
#[derive(Default)]
struct Cost {
    frames: u32,
    substeps: u32,
    max_substeps: u32,
    iterations: u64,
    solves: u64,
    max_iterations: u32,
    capped: u32,
    largest_travel: f32,
}

impl Cost {
    fn add(&mut self, frame: &Frame) {
        self.frames += 1;
        self.substeps += frame.substeps;
        self.max_substeps = self.max_substeps.max(frame.substeps);
        self.iterations += frame.iterations.iter().map(|&i| i as u64).sum::<u64>();
        self.solves += frame.iterations.len() as u64;
        let most = frame.iterations.iter().copied().max().unwrap_or(0);
        self.max_iterations = self.max_iterations.max(most);
        self.capped += frame.capped;
        self.largest_travel = self.largest_travel.max(frame.largest_travel);
    }

    fn print(&self, label: &str, wall_s: f32, simulated_s: f32) {
        println!(
            "  cost {label}: {:.1} substeps/frame (max {}), CG {:.1} it/solve (max {}), \
             {} capped, largest travel {:.3} cell, {:.0} ms per simulated s ({})",
            self.substeps as f32 / self.frames.max(1) as f32,
            self.max_substeps,
            self.iterations as f32 / self.solves.max(1) as f32,
            self.max_iterations,
            self.capped,
            self.largest_travel,
            1000.0 * wall_s / simulated_s,
            if cfg!(debug_assertions) {
                "debug"
            } else {
                "release"
            },
        );
    }
}

/// Scene 1 at `k` cells per centimetre and one gravity. Returns the
/// largest pressure error, in metres of head.
fn column_at_rest(k: usize, g_fraction: f32, report: &mut Report) -> f32 {
    let (wall, width, depth) = (4 * k, 40 * k, 30 * k);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18 * k);
    let floor = wall as f32;
    let h = floor + depth as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let water = (
        IVec2::new(wall as i32, wall as i32),
        IVec2::new((wall + width) as i32, (wall + depth) as i32),
    );
    let x = seed(
        water,
        |_| true,
        |q| box_free_sides(q, Some(h), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01 / k as f32, g_fraction, tank, x);
    let start = scene.x.clone();
    let centre_i = nx / 2;
    let surface_start = surface_height(&scene.phi(), centre_i, wall).expect("no surface");
    let g = scene.g();
    let label = format!("column {} cm, {g_fraction} g", 1.0 / k as f32);
    let mut cost = Cost::default();
    let (mut worst_p, mut worst_move, mut worst_surface) = (0.0f32, 0.0f32, 0.0f32);
    let wall_clock = Instant::now();
    for _ in 0..(5.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        cost.add(&frame);
        for j in wall..ny {
            let c = centre_i + nx * j;
            let y = j as f32 + 0.5;
            if y >= h || !frame.active[c] || scene.solid_centres.get(centre_i, j) <= 0.0 {
                continue;
            }
            worst_p = worst_p.max((frame.pressure[c] - g * (h - y)).abs());
        }
        for (a, b) in scene.x.iter().zip(&start) {
            worst_move = worst_move.max((*a - *b).length());
        }
        if let Some(s) = surface_height(&frame.phi, centre_i, wall) {
            worst_surface = worst_surface.max((s - surface_start).abs());
        }
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    scene.print_map(&label);
    let head_m = worst_p / g * scene.cell_m;
    println!(
        "{label}: pressure error {:.3} cell of head ({:.2} mm), largest move {:.3} cell, \
         surface drift {:.3} cell (start {:.3}, true {h}), J at bounds {}",
        worst_p / g,
        head_m * 1000.0,
        worst_move,
        worst_surface,
        surface_start,
        scene.j_at_bounds
    );
    scene.check_reflections(&label, report);
    cost.print(&label, wall_s, 5.0);
    report.check(worst_p <= 0.5 * g, || {
        format!("{label}: pressure off by {:.3} cell of head", worst_p / g)
    });
    report.check(worst_move <= 1.0, || {
        format!("{label}: a particle moved {worst_move:.3} cell")
    });
    report.check(worst_surface <= 0.5, || {
        format!("{label}: surface drifted {worst_surface:.3} cell")
    });
    report.check(scene.j_at_bounds == 0, || {
        format!("{label}: {} J at bounds", scene.j_at_bounds)
    });
    head_m
}

#[test]
#[ignore = "projection gate, scene 1: long; run with --ignored --nocapture"]
fn gate_column_at_rest() {
    let mut report = Report::new();
    let mut coarse = 0.0;
    for &g in &GRAVITIES {
        let head = column_at_rest(1, g, &mut report);
        if g == 1.0 {
            coarse = head;
        }
    }
    let fine = column_at_rest(2, 1.0, &mut report);
    println!(
        "pressure error at 1 g: {:.2} mm at 1 cm, {:.2} mm at 0.5 cm",
        coarse * 1000.0,
        fine * 1000.0
    );
    report.check(fine <= coarse, || {
        format!(
            "finer cells: error {:.2} mm, not below {:.2} mm",
            fine * 1000.0,
            coarse * 1000.0
        )
    });
    report.finish();
}

fn droplet_in_free_fall(g_fraction: f32, report: &mut Report) {
    let (nx, ny) = (32, 144);
    let (centre, radius) = (Vec2::new(16.0, 136.0), 5.0);
    let tank = (Vec2::ZERO, Vec2::new(nx as f32, ny as f32));
    let mut rng = LcgRng::new(11);
    let x = seed(
        (IVec2::new(10, 130), IVec2::new(22, 142)),
        |q| (q - centre).length() < radius,
        |q| disc_side(q, centre, radius),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, g_fraction, tank, x);
    let g = scene.g();
    let label = format!("droplet {g_fraction} g");
    let gyration = |s: &Scene| -> f32 {
        let (com, _) = s.centre_of_mass();
        let total: f32 = s.mass.iter().sum();
        let second: f32 = (0..s.x.len())
            .map(|p| s.mass[p] * (s.x[p] - com).length_squared())
            .sum();
        (second / total).sqrt()
    };
    let gyration_start = gyration(&scene);
    let mut cost = Cost::default();
    let mut worst = [0.0f32; 5];
    let mut v_before = scene.centre_of_mass().1;
    let wall_clock = Instant::now();
    for _ in 0..(0.3 / FRAME).round() as u32 {
        let frame = scene.frame();
        cost.add(&frame);
        for (c, &p) in frame.pressure.iter().enumerate() {
            if frame.active[c] {
                worst[0] = worst[0].max(p.abs() / g);
            }
        }
        let (_, v_com) = scene.centre_of_mass();
        let accel = (v_com - v_before) / FRAME;
        worst[1] = worst[1].max((accel - scene.gravity).length() / g);
        v_before = v_com;
        for p in 0..scene.x.len() {
            worst[2] = worst[2].max((scene.v[p] - v_com).length() / (g * scene.time));
            worst[4] = worst[4].max((scene.j(p) - 1.0).abs());
        }
        worst[3] = worst[3].max((gyration(&scene) / gyration_start - 1.0).abs());
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    scene.print_map(&label);
    let lowest = scene.x.iter().fold(f32::INFINITY, |m, q| m.min(q.y));
    println!(
        "{label}: |p| {:.2e} cell of head, acceleration error {:.2e} of g, relative speed \
         {:.2e} of g t, gyration {:.2e}, |J-1| {:.2e}, lowest particle y={lowest:.1}",
        worst[0], worst[1], worst[2], worst[3], worst[4]
    );
    scene.check_reflections(&label, report);
    cost.print(&label, wall_s, 0.3);
    report.check(lowest > 1.0, || {
        format!("{label}: the droplet reached the floor")
    });
    let limits = [0.01, 1e-3, 1e-3, 5e-3, 1e-3];
    let names = [
        "pressure, cells of head",
        "acceleration error, of g",
        "relative speed, of g t",
        "gyration change",
        "|J - 1|",
    ];
    for k in 0..5 {
        report.check(worst[k] <= limits[k], || {
            format!(
                "{label}: {} {:.3e} over {:.0e}",
                names[k], worst[k], limits[k]
            )
        });
    }
}

#[test]
#[ignore = "projection gate, scene 2: run with --ignored --nocapture"]
fn gate_droplet_in_free_fall() {
    let mut report = Report::new();
    for &g in &GRAVITIES {
        droplet_in_free_fall(g, &mut report);
    }
    report.finish();
}

/// Scenes 3 and 4 share their tank, their duration and their criteria.
fn violent_scene(label: &str, x: Vec<Vec2>, report: &mut Report) -> f32 {
    violent_scene_at(label, x, None, report)
}

/// Scenes 3 and 4, and scene 7 with `sound_speed` in m/s. Returns the mean
/// substeps per frame.
fn violent_scene_at(
    label: &str,
    x: Vec<Vec2>,
    sound_speed: Option<f32>,
    report: &mut Report,
) -> f32 {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    if let Some(c) = sound_speed {
        scene = scene.with_sound_speed(c);
    }
    let (volume_start, energy_start) = (scene.volume(), scene.energy());
    let mut cost = Cost::default();
    let (mut worst_volume, mut worst_energy) = (0.0f32, f32::NEG_INFINITY);
    let (mut outside, mut nan) = (0usize, false);
    scene.print_map(label);
    let wall_clock = Instant::now();
    for f in 1..=(2.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        cost.add(&frame);
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        let out = (0..scene.x.len())
            .filter(|&p| !scene.inside_tank(p))
            .count();
        outside = outside.max(out);
        worst_volume = worst_volume.max((scene.volume() / volume_start - 1.0).abs());
        worst_energy = worst_energy.max(scene.energy() / energy_start - 1.0);
        if f % 12 == 0 {
            let front = scene
                .x
                .iter()
                .filter(|q| q.y < tank.0.y + 2.0)
                .fold(f32::NEG_INFINITY, |m, q| m.max(q.x));
            println!(
                "  {label} t={:.2}s front {:.1} cells, volume {:+.4}, energy {:+.4}, \
                 {:.2} particles per interior cell",
                scene.time,
                front - tank.0.x,
                scene.volume() / volume_start - 1.0,
                scene.energy() / energy_start - 1.0,
                scene.interior_particles_per_cell()
            );
        }
        if f % 30 == 0 {
            scene.print_map(label);
        }
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    println!(
        "{label}: NaN {nan}, most outside {outside}, volume change {worst_volume:.4}, \
         energy rise {worst_energy:+.4}, J at bounds {}",
        scene.j_at_bounds
    );
    scene.check_reflections(label, report);
    cost.print(label, wall_s, 2.0);
    report.check(!nan, || format!("{label}: NaN"));
    report.check(outside == 0, || {
        format!("{label}: {outside} particles left the tank")
    });
    report.check(worst_volume <= 0.02, || {
        format!("{label}: volume off by {worst_volume:.4}")
    });
    report.check(worst_energy <= 0.01, || {
        format!("{label}: energy rose {worst_energy:+.4}")
    });
    report.check(scene.j_at_bounds == 0, || {
        format!("{label}: {} J at bounds", scene.j_at_bounds)
    });
    cost.substeps as f32 / cost.frames.max(1) as f32
}

#[test]
#[ignore = "projection gate, scene 3: run with --ignored --nocapture"]
fn gate_dam_break() {
    let mut report = Report::new();
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    violent_scene("dam break", x, &mut report);
    report.finish();
}

#[test]
#[ignore = "projection gate, scene 4: run with --ignored --nocapture"]
fn gate_drop_into_pool() {
    let mut report = Report::new();
    let mut rng = LcgRng::new(5);
    let mut x = seed(
        (IVec2::new(4, 4), IVec2::new(68, 19)),
        |_| true,
        |q| box_free_sides(q, Some(19.0), None),
        &mut rng,
    );
    let (centre, radius) = (Vec2::new(36.0, 44.0), 5.0);
    x.extend(seed(
        (IVec2::new(30, 38), IVec2::new(42, 50)),
        |q| (q - centre).length() < radius,
        |q| disc_side(q, centre, radius),
        &mut rng,
    ));
    violent_scene("drop into pool", x, &mut report);
    report.finish();
}

/// Where the dam break loses particles through its walls and where J
/// reaches its bounds. A probe, kept for the record; no criterion.
#[test]
#[ignore = "diagnostic probe for the first gate failure: run with --ignored --nocapture"]
fn probe_dam_break_walls_and_bounds() {
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    for _ in 0..(2.0 / FRAME).round() as u32 {
        scene.frame();
    }
    let [liquid, air, wall] = scene.probe.j_at_bounds_by_place;
    println!("J at bounds: {liquid} in liquid cells, {air} in air cells, {wall} inside walls");
    let crossings = &scene.probe.crossings;
    println!("{} crossings", crossings.len());
    let first = crossings.first().map_or(0.0, |c| c.0);
    println!("first at t={first:.3}s");
    let mut by_side = [0usize; 4];
    for &(_, before, _, _) in crossings {
        let (lo, hi) = (before - tank.0, tank.1 - before);
        let d = [lo.x, hi.x, lo.y, hi.y];
        let k = (0..4).min_by(|&a, &b| d[a].total_cmp(&d[b])).unwrap_or(0);
        by_side[k] += 1;
    }
    println!(
        "by wall: left {}, right {}, floor {}, lid {}",
        by_side[0], by_side[1], by_side[2], by_side[3]
    );
    let n = crossings.len().max(1) as f32;
    let mean_gap = crossings.iter().map(|c| c.3).sum::<f32>() / n;
    let widest_gap = crossings.iter().fold(0.0f32, |m, c| m.max(c.3));
    println!(
        "distance to the wall just before: mean {mean_gap:.3} cell, widest {widest_gap:.3} cell"
    );
    for &(t, before, v, gap) in crossings.iter().take(8) {
        println!(
            "  t={t:.3}s at ({:.2},{:.2}) v=({:.1},{:.1}) gap {gap:.3}",
            before.x, before.y, v.x, v.y
        );
    }
    let bins = [0.1f32, 0.5, 1.0, 1.5, 2.0];
    for w in bins.windows(2) {
        let count = crossings
            .iter()
            .filter(|c| c.3 >= w[0] && c.3 < w[1])
            .count();
        println!("  gap in [{}, {}): {count}", w[0], w[1]);
    }
    let close = crossings.iter().filter(|c| c.3 < 0.1).count();
    println!("  gap below 0.1: {close}");
    // Within two cells of a second wall as well: a tank corner.
    let corner = crossings
        .iter()
        .filter(|&&(_, before, _, _)| {
            let (lo, hi) = (before - tank.0, tank.1 - before);
            let mut d = [lo.x, hi.x, lo.y, hi.y];
            d.sort_by(f32::total_cmp);
            d[1] < 2.0
        })
        .count();
    println!(
        "  within two cells of a corner: {corner} of {}",
        crossings.len()
    );
}

/// Whether the volume change and the packing come from the particles that
/// touched a wall: J and bound hits split by that history. A probe.
#[test]
#[ignore = "diagnostic probe for the first gate failure: run with --ignored --nocapture"]
fn probe_dam_break_volume_away_from_walls() {
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    let n = scene.x.len();
    let mut touched = vec![false; n];
    let near_wall = |q: Vec2| {
        let (lo, hi) = (q - tank.0, tank.1 - q);
        lo.x.min(lo.y).min(hi.x).min(hi.y) < 1.0
    };
    for f in 1..=(2.0 / FRAME).round() as u32 {
        scene.frame();
        for (hit, &q) in touched.iter_mut().zip(&scene.x) {
            *hit |= near_wall(q);
        }
        if f % 24 == 0 {
            let split = |want: bool| {
                let ids: Vec<usize> = (0..n).filter(|&p| touched[p] == want).collect();
                let mean = ids.iter().map(|&p| scene.j(p)).sum::<f32>() / ids.len().max(1) as f32;
                let low = ids.iter().filter(|&&p| scene.j(p) < 0.9).count();
                let high = ids.iter().filter(|&&p| scene.j(p) > 1.1).count();
                (ids.len(), mean, low, high)
            };
            let (a, b) = (split(false), split(true));
            println!(
                "t={:.2}s never near a wall: {} particles, mean J {:.4}, {} below 0.9, {} above 1.1 |                  touched a wall: {}, mean J {:.4}, {} below 0.9, {} above 1.1 | {:.2} per interior cell",
                scene.time,
                a.0,
                a.1,
                a.2,
                a.3,
                b.0,
                b.1,
                b.2,
                b.3,
                scene.interior_particles_per_cell()
            );
        }
    }
}

/// Deep in the liquid, is the divergence J reads (`tr C` of the gather in
/// use) the divergence the grid keeps after the solve, or larger? Beside
/// it, the cells' divergence interpolated bilinearly, which is what a
/// gather quadratic along each component and linear across it would read.
/// A probe: it measured the first gate run's quadratic gather at 1000 to
/// 3000 times the grid's divergence, which is why J integrates the cells'
/// divergence.
#[test]
#[ignore = "diagnostic probe for the first gate failure: run with --ignored --nocapture"]
fn probe_dam_break_bulk_divergence() {
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    for f in 1..=60u32 {
        let frame = scene.frame();
        if f % 10 != 0 {
            continue;
        }
        let (nx, ny) = (72i32, 72i32);
        let liquid = |i: i32, j: i32| {
            i >= 0
                && j >= 0
                && i < nx
                && j < ny
                && frame.phi.get(i as usize, j as usize) < 0.0
                && scene.solid_centres.get(i as usize, j as usize) > 0.0
        };
        let cell_div = |i: i32, j: i32| {
            let (i, j) = (i as usize, j as usize);
            frame.vel.u.get(i + 1, j) - frame.vel.u.get(i, j) + frame.vel.v.get(i, j + 1)
                - frame.vel.v.get(i, j)
        };
        let (mut n, mut quad, mut conforming, mut kept) = (0usize, 0.0f32, 0.0f32, 0.0f32);
        for p in 0..scene.x.len() {
            let q = scene.x[p];
            let c = q.floor().as_ivec2();
            if !(-2..=2).all(|dj| (-2..=2).all(|di| liquid(c.x + di, c.y + dj))) {
                continue;
            }
            n += 1;
            quad += (scene.c[p].x_axis.x + scene.c[p].y_axis.y).abs();
            // Bilinear interpolation of the cell divergence at the particle.
            let t = q - Vec2::splat(0.5);
            let (i0, j0) = (t.x.floor() as i32, t.y.floor() as i32);
            let (fx, fy) = (t.x - i0 as f32, t.y - j0 as f32);
            let d = cell_div(i0, j0) * (1.0 - fx) * (1.0 - fy)
                + cell_div(i0 + 1, j0) * fx * (1.0 - fy)
                + cell_div(i0, j0 + 1) * (1.0 - fx) * fy
                + cell_div(i0 + 1, j0 + 1) * fx * fy;
            conforming += d.abs();
            let mut local = 0.0f32;
            for dj in -1..=1 {
                for di in -1..=1 {
                    local = local.max(cell_div(c.x + di, c.y + dj).abs());
                }
            }
            kept += local;
        }
        let m = n.max(1) as f32;
        println!(
            "t={:.2}s {n} bulk particles: |tr C| {:.3e} 1/s, bilinear cell divergence              {:.3e} 1/s, largest cell divergence nearby {:.3e} 1/s",
            scene.time,
            quad / m,
            conforming / m,
            kept / m
        );
    }
}

/// After the second gate run: the particles that never came near a wall
/// still lost volume. Split them by whether they ever sat within two cells
/// of an air cell, to see whether the loss is at the free surface, where
/// the gather reads velocities extrapolated into the air. A probe.
#[test]
#[ignore = "diagnostic probe for the second gate failure: run with --ignored --nocapture"]
fn probe_dam_break_volume_at_the_free_surface() {
    let mut rng = LcgRng::new(3);
    let x = seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    );
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
    let n = scene.x.len();
    let (mut near_wall, mut near_air) = (vec![false; n], vec![false; n]);
    for f in 1..=(2.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        for p in 0..n {
            let q = scene.x[p];
            let (lo, hi) = (q - tank.0, tank.1 - q);
            near_wall[p] |= lo.x.min(lo.y).min(hi.x).min(hi.y) < 1.0;
            let c = q.floor().as_ivec2();
            near_air[p] |= (-2..=2).any(|dj| {
                (-2..=2).any(|di| {
                    let (i, j) = (c.x + di, c.y + dj);
                    let inside = scene
                        .solid_centres
                        .get_signed(i, j)
                        .is_some_and(|s| s > 0.0);
                    inside && frame.phi.get_signed(i, j).is_some_and(|f| f >= 0.0)
                })
            });
        }
        if f % 30 == 0 {
            let group = |wall: bool, air: bool| {
                let ids: Vec<usize> = (0..n)
                    .filter(|&p| near_wall[p] == wall && near_air[p] == air)
                    .collect();
                let m = ids.len().max(1) as f32;
                let mean = ids.iter().map(|&p| scene.j(p)).sum::<f32>() / m;
                (ids.len(), mean)
            };
            let deep = group(false, false);
            let surface = group(false, true);
            println!(
                "t={:.2}s away from walls: never near air {} particles, mean J {:.4} | \
                 near air at some point {} particles, mean J {:.4}",
                scene.time, deep.0, deep.1, surface.0, surface.1
            );
        }
    }
}

/// The dam break and the drop into a pool, their particle positions every
/// fourth frame written as JSON to the path in `EMERGE_GATE_DUMP` (tenths
/// of a cell), and the time each phase of the substep took. A probe.
#[test]
#[ignore = "visual dump and phase timings: run with --ignored --nocapture"]
fn probe_dump_and_phase_costs() {
    let path = std::env::var("EMERGE_GATE_DUMP").ok();
    let mut json = String::from("{\"scenes\":[");
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    for (k, name) in ["dam break", "drop into pool"].into_iter().enumerate() {
        let mut rng = LcgRng::new(if k == 0 { 3 } else { 5 });
        let x = if k == 0 {
            seed(
                (IVec2::new(4, 4), IVec2::new(24, 44)),
                |_| true,
                |q| box_free_sides(q, Some(44.0), Some(24.0)),
                &mut rng,
            )
        } else {
            let mut x = seed(
                (IVec2::new(4, 4), IVec2::new(68, 19)),
                |_| true,
                |q| box_free_sides(q, Some(19.0), None),
                &mut rng,
            );
            let (centre, radius) = (Vec2::new(36.0, 44.0), 5.0);
            x.extend(seed(
                (IVec2::new(30, 38), IVec2::new(42, 50)),
                |q| (q - centre).length() < radius,
                |q| disc_side(q, centre, radius),
                &mut rng,
            ));
            x
        };
        let n = x.len();
        let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
        if k > 0 {
            json.push(',');
        }
        json.push_str(&format!(
            "{{\"name\":\"{name}\",\"grid\":72,\"frame_dt\":{FRAME},\"every\":4,\"boxes\":[],\"tags\":[{}],\"frames\":[",
            vec!["0"; n].join(",")
        ));
        let frames = (2.0 / FRAME).round() as u32;
        let wall = Instant::now();
        for f in 0..=frames {
            if f % 4 == 0 {
                if f > 0 {
                    json.push(',');
                }
                let xs: Vec<String> = scene
                    .x
                    .iter()
                    .flat_map(|q| [(q.x * 10.0).round() as i32, (q.y * 10.0).round() as i32])
                    .map(|v| v.to_string())
                    .collect();
                json.push('[');
                json.push_str(&xs.join(","));
                json.push(']');
            }
            if f < frames {
                scene.frame();
            }
        }
        json.push_str("]}");
        let total = wall.elapsed().as_micros().max(1) as f64;
        let names = [
            "P2G",
            "level set",
            "assembly (record)",
            "projection",
            "two gathers",
            "particle update",
        ];
        println!(
            "{name}: {} particles, {:.0} ms for 2 s simulated",
            n,
            total / 1e3
        );
        for (label, us) in names.iter().zip(scene.probe.phase_us) {
            println!(
                "  {label:<18} {:6.0} ms  {:5.1} %",
                us as f64 / 1e3,
                100.0 * us as f64 / total
            );
        }
    }
    json.push_str("]}");
    if let Some(path) = path {
        std::fs::write(path, json).expect("write dump");
    }
}

/// The seeds of scenes 3 and 4.
fn dam_break_seed() -> Vec<Vec2> {
    let mut rng = LcgRng::new(3);
    seed(
        (IVec2::new(4, 4), IVec2::new(24, 44)),
        |_| true,
        |q| box_free_sides(q, Some(44.0), Some(24.0)),
        &mut rng,
    )
}

fn drop_into_pool_seed() -> Vec<Vec2> {
    let mut rng = LcgRng::new(5);
    let mut x = seed(
        (IVec2::new(4, 4), IVec2::new(68, 19)),
        |_| true,
        |q| box_free_sides(q, Some(19.0), None),
        &mut rng,
    );
    let (centre, radius) = (Vec2::new(36.0, 44.0), 5.0);
    x.extend(seed(
        (IVec2::new(30, 38), IVec2::new(42, 50)),
        |q| (q - centre).length() < radius,
        |q| disc_side(q, centre, radius),
        &mut rng,
    ));
    x
}

#[test]
#[ignore = "compressible projection gate, scene 5: run with --ignored --nocapture"]
fn gate_compressible_column() {
    let mut report = Report::new();
    let c_m_s = 10.0;
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let floor = wall as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let h0 = floor + depth as f32;
    let x = seed(
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h0), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x).with_sound_speed(c_m_s);
    let c2 = scene.sound_speed2.unwrap_or(0.0);
    let g = scene.g();
    let centre_i = nx / 2;
    let surface_start = surface_height(&scene.phi(), centre_i, wall).expect("no surface");
    let expected_drop = g * (depth as f32).powi(2) / (2.0 * c2);
    let frames = (5.0 / FRAME).round() as u32;
    let last_second = frames - (1.0 / FRAME).round() as u32;
    let (mut worst_p, mut surface_sum, mut surface_n) = (0.0f32, 0.0f32, 0u32);
    let bands = depth / 5;
    let mut band_j = vec![(0.0f64, 0u64); bands];
    let mut band_h = 0.0f32;
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    for f in 1..=frames {
        let frame = scene.frame();
        cost.add(&frame);
        let Some(h) = surface_height(&frame.phi, centre_i, wall) else {
            report.check(false, || "column: no surface".to_string());
            break;
        };
        for j in wall..ny {
            let c = centre_i + nx * j;
            let y = j as f32 + 0.5;
            if scene.time < 1.0
                || y >= h - 0.5
                || !frame.active[c]
                || scene.solid_centres.get(centre_i, j) <= 0.0
            {
                continue;
            }
            worst_p = worst_p.max((frame.pressure[c] - g * (h - y)).abs());
        }
        if f > last_second {
            surface_sum += h;
            surface_n += 1;
            band_h += h;
            for p in 0..scene.x.len() {
                let k = ((scene.x[p].y - floor) / 5.0).floor();
                if k >= 0.0 && (k as usize) < bands {
                    let entry = &mut band_j[k as usize];
                    entry.0 += f64::from(scene.j(p));
                    entry.1 += 1;
                }
            }
        }
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let surface = surface_sum / surface_n.max(1) as f32;
    let h_mean = band_h / surface_n.max(1) as f32;
    let drop = surface_start - surface;
    println!(
        "compressible column: pressure error {:.3} cell of head, surface drop {drop:.3} cell against {expected_drop:.3}",
        worst_p / g
    );
    report.check(worst_p <= 0.5 * g, || {
        format!("column: pressure off by {:.3} cell of head", worst_p / g)
    });
    report.check((drop - expected_drop).abs() <= 0.15, || {
        format!("column: surface dropped {drop:.3} cell, expected {expected_drop:.3}")
    });
    for (k, &(sum, n)) in band_j.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let mean = (sum / n as f64) as f32;
        let y = floor + 5.0 * k as f32 + 2.5;
        let expected = 1.0 - g * (h_mean - y) / c2;
        println!("  band {k}: mean J {mean:.5}, expected {expected:.5}");
        report.check((mean - expected).abs() <= 0.005, || {
            format!("column: band {k} mean J {mean:.5}, expected {expected:.5}")
        });
    }
    scene.check_reflections("compressible column", &mut report);
    cost.print("compressible column", wall_s, 5.0);
    report.finish();
}

#[test]
#[ignore = "compressible projection gate, scene 6: run with --ignored --nocapture"]
fn gate_sound_speed() {
    let mut report = Report::new();
    let c_m_s = 10.0;
    let (wall, length, height) = (4usize, 200usize, 8usize);
    let (nx, ny) = (length + 2 * wall, height + 2 * wall);
    let tank = (
        Vec2::splat(wall as f32),
        Vec2::new((wall + length) as f32, (wall + height) as f32),
    );
    let mut rng = LcgRng::new(11);
    let x = seed(
        (
            IVec2::splat(wall as i32),
            IVec2::new((wall + length) as i32, (wall + height) as i32),
        ),
        |_| true,
        |_| (f32::INFINITY, Vec2::Y),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 0.0, tank, x).with_sound_speed(c_m_s);
    let c = c_m_s / scene.cell_m;
    scene.fixed_dt = Some(scene.layout.dx / c);
    for p in 0..scene.x.len() {
        if scene.x[p].x < wall as f32 + 10.0 {
            scene.log_j[p] = 0.99f32.ln();
        }
    }
    let watch_i = wall + 150;
    scene.probe.watch = Some(watch_i + nx * (wall + height / 2));
    let expected = (150.0 - 5.0) / c;
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    while scene.time < 1.5 * expected {
        let frame = scene.frame();
        cost.add(&frame);
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let (arrival, peak) = scene
        .probe
        .watch_history
        .iter()
        .filter(|(t, _)| *t < 1.5 * expected)
        .fold((0.0f32, f32::NEG_INFINITY), |m, &(t, q)| {
            if q > m.1 { (t, q) } else { m }
        });
    let error = (arrival - expected).abs() / expected;
    println!(
        "sound speed: peak {peak:.1} at t={arrival:.4}s, expected {expected:.4}s, off by {:.1} %",
        100.0 * error
    );
    report.check(error <= 0.05, || {
        format!("sound speed: arrival off by {:.1} %", 100.0 * error)
    });
    scene.check_reflections("sound speed", &mut report);
    cost.print("sound speed", wall_s, scene.time);
    report.finish();
}

#[test]
#[ignore = "compressible projection gate, scene 7: run with --ignored --nocapture"]
fn gate_real_water() {
    let mut report = Report::new();
    let water = 1483.0;
    for (name, seed_fn) in [
        ("dam break", dam_break_seed as fn() -> Vec<Vec2>),
        ("drop into pool", drop_into_pool_seed as fn() -> Vec<Vec2>),
    ] {
        let mut quiet = Report::new();
        let incompressible = violent_scene_at(name, seed_fn(), None, &mut quiet);
        let label = format!("{name}, real water");
        let compressible = violent_scene_at(&label, seed_fn(), Some(water), &mut report);
        println!(
            "{label}: {compressible:.2} substeps/frame against {incompressible:.2} incompressible"
        );
        report.check(compressible <= 1.1 * incompressible, || {
            format!("{label}: {compressible:.2} substeps/frame, over 1.1 x {incompressible:.2}")
        });
    }
    report.finish();
}

/// Diagnosis of A1's first run: the sound-speed arrival at substeps of
/// `dx / c`, half and a quarter of it (a time-discretisation error shrinks
/// with the step; a wrong `c` does not), and the compressible column's
/// largest pressure error per half second. A probe.
#[test]
#[ignore = "diagnostic probe for A1's first run: run with --ignored --nocapture"]
fn probe_a1_first_run() {
    let c_m_s = 10.0;
    for fraction in [1.0f32, 0.5, 0.25] {
        let (wall, length, height) = (4usize, 200usize, 8usize);
        let (nx, ny) = (length + 2 * wall, height + 2 * wall);
        let tank = (
            Vec2::splat(wall as f32),
            Vec2::new((wall + length) as f32, (wall + height) as f32),
        );
        let mut rng = LcgRng::new(11);
        let x = seed(
            (
                IVec2::splat(wall as i32),
                IVec2::new((wall + length) as i32, (wall + height) as i32),
            ),
            |_| true,
            |_| (f32::INFINITY, Vec2::Y),
            &mut rng,
        );
        let mut scene = Scene::new((nx, ny), 0.01, 0.0, tank, x).with_sound_speed(c_m_s);
        let c = c_m_s / scene.cell_m;
        scene.fixed_dt = Some(fraction * scene.layout.dx / c);
        for p in 0..scene.x.len() {
            if scene.x[p].x < wall as f32 + 10.0 {
                scene.log_j[p] = 0.99f32.ln();
            }
        }
        scene.probe.watch = Some(wall + 150 + nx * (wall + height / 2));
        let expected = (150.0 - 5.0) / c;
        while scene.time < 1.5 * expected {
            scene.frame();
        }
        let (arrival, peak) = scene
            .probe
            .watch_history
            .iter()
            .filter(|(t, _)| *t < 1.5 * expected)
            .fold((0.0f32, f32::NEG_INFINITY), |m, &(t, q)| {
                if q > m.1 { (t, q) } else { m }
            });
        println!(
            "dt = {fraction} dx/c: peak {peak:.1} at {arrival:.4} s, expected {expected:.4} s ({:+.1} %)",
            100.0 * (arrival - expected) / expected
        );
    }
    // Column: largest pressure error per half second.
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let floor = wall as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let h0 = floor + depth as f32;
    let x = seed(
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h0), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x).with_sound_speed(c_m_s);
    let g = scene.g();
    let centre_i = nx / 2;
    let mut worst = 0.0f32;
    let mut worst_y = 0.0f32;
    for f in 1..=(5.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        let h = surface_height(&frame.phi, centre_i, wall).unwrap_or(0.0);
        for j in wall..ny {
            let c = centre_i + nx * j;
            let y = j as f32 + 0.5;
            if y >= h - 0.5 || !frame.active[c] || scene.solid_centres.get(centre_i, j) <= 0.0 {
                continue;
            }
            let e = (frame.pressure[c] - g * (h - y)).abs();
            if e > worst {
                worst = e;
                worst_y = y;
            }
        }
        if f % 30 == 0 {
            let outside = (0..scene.x.len())
                .filter(|&p| !scene.inside_tank(p))
                .count();
            let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
            let geometric = scene.interior_particles_per_cell();
            let mean_j = (0..scene.x.len()).map(|p| scene.j(p)).sum::<f32>() / scene.x.len() as f32;
            println!(
                "column t={:.1}s: largest pressure error {:.3} cell of head (at y={worst_y:.1}), surface {h:.3}, outside {outside}, lowest y {lowest:.3}, {geometric:.3} per interior cell, mean J {mean_j:.5}",
                scene.time,
                worst / g
            );
            worst = 0.0;
        }
    }
}

/// `E / rho` (m^2/s^2) giving plane-strain shear speed `c_s` at `nu`.
fn e_over_rho_for_shear_speed(c_s: f32, nu: f32) -> f32 {
    2.0 * (1.0 + nu) * c_s * c_s
}

#[test]
#[ignore = "staggered elastic solid gate, scene 8: run with --ignored --nocapture"]
fn gate_elastic_column() {
    let mut report = Report::new();
    let nu = 0.3f32;
    // (lambda + mu) / rho = E / rho * (nu / ((1 + nu)(1 - 2 nu)) + 1 / (2 (1 + nu))).
    let stiffness_per_e = nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) + 0.5 / (1.0 + nu);
    let e_over_rho = 100.0 / stiffness_per_e;
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let floor = wall as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let h0 = floor + depth as f32;
    let x = seed(
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h0), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x).with_elastic(e_over_rho, nu);
    let g = scene.g();
    let cp2 = scene.shear.unwrap_or(0.0) + scene.sound_speed2.unwrap_or(0.0);
    let centre_i = nx / 2;
    let frames = (5.0 / FRAME).round() as u32;
    let last_second = frames - (1.0 / FRAME).round() as u32;
    let bands = depth / 5;
    let mut band_j = vec![(0.0f64, 0u64); bands];
    let (mut h_sum, mut h_n, mut nan, mut outside) = (0.0f32, 0u32, false, 0usize);
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    for f in 1..=frames {
        let frame = scene.frame();
        cost.add(&frame);
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        outside = outside.max(
            (0..scene.x.len())
                .filter(|&p| !scene.inside_tank(p))
                .count(),
        );
        if f > last_second {
            if let Some(h) = surface_height(&frame.phi, centre_i, wall) {
                h_sum += h;
                h_n += 1;
            }
            for p in 0..scene.x.len() {
                let k = ((scene.x[p].y - floor) / 5.0).floor();
                if k >= 0.0 && (k as usize) < bands {
                    let entry = &mut band_j[k as usize];
                    entry.0 += f64::from(scene.j(p));
                    entry.1 += 1;
                }
            }
        }
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let h = h_sum / h_n.max(1) as f32;
    report.check(!nan, || "elastic column: NaN".to_string());
    report.check(outside == 0, || {
        format!("elastic column: {outside} particles out")
    });
    for (k, &(sum, n)) in band_j.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let mean = (sum / n as f64) as f32;
        let y = floor + 5.0 * k as f32 + 2.5;
        let expected = 1.0 - g * (h - y) / cp2;
        println!("  elastic band {k}: mean J {mean:.5}, expected {expected:.5}");
        report.check((mean - expected).abs() <= 0.005, || {
            format!("elastic column: band {k} mean J {mean:.5}, expected {expected:.5}")
        });
    }
    scene.check_reflections("elastic column", &mut report);
    cost.print("elastic column", wall_s, 5.0);
    report.finish();
}

#[test]
#[ignore = "staggered elastic solid gate, scene 9: run with --ignored --nocapture"]
fn gate_shear_wave() {
    let mut report = Report::new();
    let (nu, c_s) = (0.3f32, 5.0f32);
    let (length, height) = (100usize, 208usize);
    let (nx, ny) = (length + 8, height + 8);
    let (x0, y0) = (4.0f32, 4.0f32);
    let tank = (
        Vec2::splat(1.0),
        Vec2::new(nx as f32 - 1.0, ny as f32 - 1.0),
    );
    let mut rng = LcgRng::new(13);
    let top = y0 + height as f32;
    let right = x0 + length as f32;
    let x = seed(
        (
            IVec2::new(x0 as i32, y0 as i32),
            IVec2::new(right as i32, top as i32),
        ),
        |_| true,
        |q| {
            let d = [
                (q.y - y0, Vec2::NEG_Y),
                (top - q.y, Vec2::Y),
                (q.x - x0, Vec2::NEG_X),
                (right - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 0.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    let c = c_s / scene.cell_m;
    scene.fixed_dt = Some(0.5 * scene.layout.dx / c);
    // Step M3's amendment: the solid's particles carry the P wave now, so
    // the explicit step follows it, as `plane_wave` sets it.
    let c_p = (scene.sound_speed2.unwrap_or(0.0) + scene.shear.unwrap_or(0.0)).sqrt();
    scene.fixed_dt = Some(0.5 * scene.layout.dx / c_p.max(c));
    let kick = 0.05 / scene.cell_m;
    for p in 0..scene.x.len() {
        if scene.x[p].x < x0 + 10.0 {
            scene.v[p].y = kick;
        }
    }
    let mid = y0 + 0.5 * height as f32;
    scene.probe.watch_x = Some((x0 + 50.0, mid - 4.0, mid + 4.0));
    let expected = 40.0 / c;
    // The window: the front's plateau ends at 50 dx / c_s, the edges'
    // disturbance arrives at 53 (criteria).
    let window = 50.0 / c;
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    while scene.time < window {
        let frame = scene.frame();
        cost.add(&frame);
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let history = &scene.probe.watch_x_history;
    let half = 0.25 * kick;
    let arrival = history.windows(2).find_map(|pair| {
        let ((t0, v0), (t1, v1)) = (pair[0], pair[1]);
        (v0 < half && v1 >= half).then(|| t0 + (t1 - t0) * (half - v0) / (v1 - v0))
    });
    let plateau = history
        .iter()
        .filter(|(t, _)| *t > expected && *t < window)
        .map(|&(_, v)| v)
        .fold(f32::NEG_INFINITY, f32::max);
    let error = arrival.map_or(f32::INFINITY, |t| (t - expected).abs() / expected);
    println!(
        "shear wave: v0/4 reached at t={:?}s, expected {expected:.4}s, off by {:.1} %; largest velocity after it {:.3} of v0 (d'Alembert: 0.5)",
        arrival,
        100.0 * error,
        plateau / kick
    );
    report.check(error <= 0.05, || {
        format!("shear wave: arrival off by {:.1} %", 100.0 * error)
    });
    scene.check_reflections("shear wave", &mut report);
    cost.print("shear wave", wall_s, scene.time);
    report.finish();
}

#[test]
#[ignore = "staggered elastic solid gate, scene 10: run with --ignored --nocapture"]
fn gate_elastic_drop() {
    let mut report = Report::new();
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    let gyration = |scene: &Scene| {
        let (com, _) = scene.centre_of_mass();
        let n = scene.x.len() as f32;
        (scene
            .x
            .iter()
            .map(|q| (*q - com).length_squared())
            .sum::<f32>()
            / n)
            .sqrt()
    };
    let (energy_start, gyration_start) = (scene.energy(), gyration(&scene));
    let (mut worst_energy, mut nan, mut outside) = (f32::NEG_INFINITY, false, 0usize);
    let mut cost = Cost::default();
    let wall_clock = Instant::now();
    for f in 1..=(2.0 / FRAME).round() as u32 {
        let frame = scene.frame();
        cost.add(&frame);
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        outside = outside.max(
            (0..scene.x.len())
                .filter(|&p| !scene.inside_tank(p))
                .count(),
        );
        worst_energy = worst_energy.max(scene.energy() / energy_start - 1.0);
        if f % 12 == 0 {
            println!(
                "  elastic drop t={:.2}s: energy {:+.4}, gyration {:.3} (start {gyration_start:.3}), lowest y {:.2}",
                scene.time,
                scene.energy() / energy_start - 1.0,
                gyration(&scene),
                scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min)
            );
        }
    }
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let change = (gyration(&scene) / gyration_start - 1.0).abs();
    println!(
        "elastic drop: NaN {nan}, most outside {outside}, energy rise {worst_energy:+.4}, gyration change {change:.4}"
    );
    report.check(!nan, || "elastic drop: NaN".to_string());
    report.check(outside == 0, || {
        format!("elastic drop: {outside} particles out")
    });
    report.check(worst_energy <= 0.01, || {
        format!("elastic drop: energy rose {worst_energy:+.4}")
    });
    report.check(change <= 0.05, || {
        format!("elastic drop: gyration changed {change:.4}")
    });
    scene.check_reflections("elastic drop", &mut report);
    cost.print("elastic drop", wall_s, 2.0);
    report.finish();
}

/// Diagnosis of A2's first run: scene 10's energy split into its parts
/// every frame. A probe.
#[test]
#[ignore = "diagnostic probe for A2's first run: run with --ignored --nocapture"]
fn probe_a2_energy_parts() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    // EMERGE_A2_SCENARIO: "drop" (scene 10), "rest" (on the floor, at
    // rest), "spin" (no gravity, no contact, spinning at 2 rad/s).
    let scenario = std::env::var("EMERGE_A2_SCENARIO").unwrap_or_else(|_| "drop".into());
    let base = if scenario == "rest" { 4.0 } else { 24.0 };
    let scenario = scenario.as_str();
    let (lo, hi) = (Vec2::new(31.0, base), Vec2::new(41.0, base + 10.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let spinning = scenario.starts_with("spin");
    let g_fraction = if spinning { 0.0 } else { 1.0 };
    let mut scene = Scene::new((72, 72), 0.01, g_fraction, tank, x);
    if scenario != "spin_liquid" {
        scene = scene.with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    }
    if spinning {
        let centre = 0.5 * (lo + hi);
        for p in 0..scene.x.len() {
            let r = scene.x[p] - centre;
            let w = std::env::var("EMERGE_A2_OMEGA")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(2.0);
            if std::env::var_os("EMERGE_A2_TRANSLATE").is_some() {
                scene.v[p] = Vec2::new(w * 5.0, 0.0);
            } else {
                scene.v[p] = w * Vec2::new(-r.y, r.x);
                scene.c[p] = Mat2::from_cols(Vec2::new(0.0, w), Vec2::new(-w, 0.0));
            }
        }
    }
    let (mu, lambda) = (
        scene.shear.unwrap_or(0.0),
        scene.sound_speed2.unwrap_or(0.0),
    );
    let parts = |s: &Scene| {
        let floor = s.tank.0.y;
        let mut out = [0.0f64; 5];
        for p in 0..s.x.len() {
            out[0] += f64::from(s.mass[p] * 0.5 * s.v[p].length_squared());
            out[1] += f64::from(s.mass[p] * s.g() * (s.x[p].y - floor));
            out[2] += f64::from(s.v0[p] * shear_energy(s.deformation[p], mu));
            out[3] += f64::from(s.v0[p] * 0.5 * lambda * (s.j(p) - 1.0).powi(2));
            out[4] += f64::from(s.deformation[p].determinant() / s.j(p) - 1.0).abs();
        }
        out[4] /= s.x.len() as f64;
        out
    };
    let start = parts(&scene);
    let total0 = (start[0] + start[1] + start[2] + start[3]).max(1.0);
    for f in 1..=72u32 {
        scene.frame();
        if f <= 2 && spinning {
            for p in [0usize, scene.x.len() / 2] {
                let fm = scene.deformation[p];
                let r = super::elastic::rotation(fm);
                let fh = fm * fm.determinant().sqrt().recip();
                println!(
                    "  particle {p} at {:?}: F {:?} (det {:.5}), angle of R {:.4}, |F^ - R| {:.5}, C {:?}",
                    scene.x[p],
                    fm,
                    fm.determinant(),
                    r.x_axis.y.atan2(r.x_axis.x),
                    (fh - r).x_axis.length() + (fh - r).y_axis.length(),
                    scene.c[p]
                );
            }
            // Strain rate |sym C| of the particles within one cell of the
            // block's edge against those deeper in, after this frame.
            let (lo_now, hi_now) = scene.x.iter().fold(
                (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
                |(a, b), q| (a.min(*q), b.max(*q)),
            );
            let (mut edge, mut ne, mut inner, mut ni) = (0.0f32, 0u32, 0.0f32, 0u32);
            for p in 0..scene.x.len() {
                let q = scene.x[p];
                let d = (q - lo_now).min(hi_now - q).min_element();
                let c = scene.c[p];
                let sym = 0.5 * (c + c.transpose());
                let rate = (sym.x_axis.length_squared() + sym.y_axis.length_squared()).sqrt();
                if d < 1.0 {
                    edge += rate;
                    ne += 1;
                } else if d > 2.0 {
                    inner += rate;
                    ni += 1;
                }
            }
            println!(
                "  strain rate after frame {f}: edge {:.3} 1/s ({ne} particles), interior {:.3} 1/s ({ni}); rotation rate 2",
                edge / ne.max(1) as f32,
                inner / ni.max(1) as f32
            );
        }
        let e = parts(&scene);
        let total = e[0] + e[1] + e[2] + e[3];
        println!(
            "f{f:3} t={:.3}: total {:+.4} | kinetic {:.4} potential {:.4} shear {:.4} volume {:.4} (fractions of start) | mean |det F / J - 1| {:.5}",
            scene.time,
            total / total0 - 1.0,
            e[0] / total0,
            e[1] / total0,
            e[2] / total0,
            e[3] / total0,
            e[4]
        );
    }
}

/// Diagnosis of A2's first run: one substep of a rigidly spinning liquid
/// block, no gravity, and the velocity gradient at an edge particle after
/// each stage. A probe.
#[test]
#[ignore = "diagnostic probe for A2's first run: run with --ignored --nocapture"]
fn probe_a2_one_substep_rotation() {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |_| (f32::INFINITY, Vec2::Y),
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 0.0, tank, x);
    let centre = 0.5 * (lo + hi);
    for p in 0..scene.x.len() {
        let r = scene.x[p] - centre;
        scene.v[p] = 2.0 * Vec2::new(-r.y, r.x);
        scene.c[p] = Mat2::from_cols(Vec2::new(0.0, 2.0), Vec2::new(-2.0, 0.0));
    }
    let edge = (0..scene.x.len())
        .min_by(|&a, &b| scene.x[a].y.total_cmp(&scene.x[b].y))
        .unwrap_or(0);
    let dt = 1.0e-3;
    let (mut vel, face_mass) =
        particles_to_faces(&scene.layout, &scene.x, &scene.v, &scene.c, &scene.mass);
    let read = |vel: &super::field::MacVelocity| {
        let mut v = [Vec2::ZERO];
        let mut c = [Mat2::ZERO];
        faces_to_particles(&scene.layout, vel, &[scene.x[edge]], &mut v, &mut c);
        (v[0], c[0])
    };
    println!(
        "edge particle at {:?}, exact v {:?}",
        scene.x[edge], scene.v[edge]
    );
    println!("after P2G: {:?}", read(&vel));
    let phi = scene.phi();
    for j in 21..27usize {
        let i = 38usize;
        let q = scene.layout.u_position(i, j);
        let g = phi.bilinear_gradient(q - Vec2::splat(0.5));
        println!(
            "  u face ({i},{j}) at y={:.1}: u {:.3} (exact {:.3}), phi left/right {:.3}/{:.3}, grad phi {:?}",
            q.y,
            vel.u.get(i, j),
            -2.0 * (q.y - centre.y),
            phi.get(i - 1, j),
            phi.get(i, j),
            g
        );
    }
    let image = box_container_image(tank.0, tank.1);
    let mut projected = vel.clone();
    let solution = project(
        &scene.layout,
        dt,
        &mut projected,
        Walls {
            weights: &scene.weights,
            image: &image,
            open: &scene.open_centres(),
        },
        Liquid {
            phi: &phi,
            face_mass: &face_mass,
        },
        &scene.settings,
        None,
    );
    println!(
        "after projection + extension: {:?}, largest pressure {}",
        read(&projected),
        solution.pressure.iter().fold(0.0f32, |m, p| m.max(p.abs()))
    );
    vel = projected;
    let _ = vel;
}

/// Diagnosis of A2's second run: the transfer round trip alone (scatter,
/// projection, gather, particles held still, no gravity) on a resting
/// column with small random velocities. Kinetic energy must never grow.
/// A probe.
#[test]
#[ignore = "diagnostic probe for A2's second run: run with --ignored --nocapture"]
fn probe_a2_round_trip_energy() {
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let h = (wall + depth) as f32;
    let tank = (
        Vec2::new(wall as f32, wall as f32),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let x = seed(
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 0.0, tank, x);
    let mut noise = LcgRng::new(3);
    for p in 0..scene.x.len() {
        scene.v[p] = Vec2::new(noise.next_f32() - 0.5, noise.next_f32() - 0.5);
    }
    let phi = scene.phi();
    let image = box_container_image(scene.tank.0, scene.tank.1);
    let kinetic = |s: &Scene| -> f64 {
        (0..s.x.len())
            .map(|p| f64::from(0.5 * s.mass[p] * s.v[p].length_squared()))
            .sum()
    };
    let mut previous = kinetic(&scene);
    let mut rises = 0;
    for round in 1..=200 {
        let (mut vel, face_mass) =
            particles_to_faces(&scene.layout, &scene.x, &scene.v, &scene.c, &scene.mass);
        project(
            &scene.layout,
            1.0e-3,
            &mut vel,
            Walls {
                weights: &scene.weights,
                image: &image,
                open: &scene.open_centres(),
            },
            Liquid {
                phi: &phi,
                face_mass: &face_mass,
            },
            &scene.settings,
            None,
        );
        faces_to_particles(&scene.layout, &vel, &scene.x, &mut scene.v, &mut scene.c);
        let now = kinetic(&scene);
        if now > previous * (1.0 + 1.0e-6) {
            rises += 1;
        }
        if round <= 5 || round % 40 == 0 {
            println!(
                "round {round}: kinetic {now:.6e} ({:+.3e} of the round before)",
                now / previous - 1.0
            );
        }
        previous = now;
    }
    println!("rounds where kinetic energy rose: {rises} of 200");
}

/// Diagnosis of the new A2 attempt's first run: scene 10's block, resting
/// on the floor at 1 s, went to NaN before 1.2 s. Substep by substep from
/// 0.95 s: the largest `|C|`, speed, `|F^ - R^|` and J range, until the
/// first value that is not finite. A probe.
#[test]
#[ignore = "diagnostic probe for the new A2 attempt: run with --ignored --nocapture"]
fn probe_a2_drop_first_nan() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    while scene.time < 0.95 {
        scene.frame();
    }
    let mu = scene.shear.unwrap_or(0.0);
    let dt = 0.5 * scene.layout.dx / mu.sqrt();
    let mut frame = scene.frame();
    for step in 0..20000 {
        scene.substep(dt, &mut frame);
        scene.time += dt;
        let bad: Vec<usize> = (0..scene.x.len())
            .filter(|&p| {
                !(scene.x[p].is_finite()
                    && scene.v[p].is_finite()
                    && scene.c[p].is_finite()
                    && scene.deformation[p].is_finite()
                    && scene.log_j[p].is_finite())
            })
            .collect();
        let (mut c_max, mut c_at, mut speed, mut dev, mut j_lo, mut j_hi) =
            (0.0f32, 0usize, 0.0f32, 0.0f32, f32::INFINITY, 0.0f32);
        for p in 0..scene.x.len() {
            let c = scene.c[p]
                .to_cols_array()
                .iter()
                .fold(0.0f32, |m, e| m.max(e.abs()));
            if c > c_max {
                c_max = c;
                c_at = p;
            }
            speed = speed.max(scene.v[p].length());
            dev = dev.max(shear_energy(scene.deformation[p], 1.0).sqrt());
            j_lo = j_lo.min(scene.j(p));
            j_hi = j_hi.max(scene.j(p));
        }
        if step % 200 == 0 || !bad.is_empty() || c_max > 500.0 {
            println!(
                "t={:.5} step {step}: |C| max {c_max:.3} at {:?} (F det {:.4}, J {:.4}), speed {speed:.3}, |F^-R^| max {dev:.4}, J {j_lo:.4}..{j_hi:.4}",
                scene.time,
                scene.x[c_at],
                scene.deformation[c_at].determinant(),
                scene.j(c_at)
            );
        }
        if !bad.is_empty() {
            for &p in bad.iter().take(5) {
                println!(
                    "  bad {p}: x {:?} v {:?} F {:?} log J {}",
                    scene.x[p], scene.v[p], scene.deformation[p], scene.log_j[p]
                );
            }
            break;
        }
    }
}

/// Diagnosis of the new A2 attempt's first run: when do particles of
/// scene 10's elastic block lose every neighbour within one cell (an
/// elastic body must never shed a piece)? A probe.
#[test]
#[ignore = "diagnostic probe for the new A2 attempt: run with --ignored --nocapture"]
fn probe_a2_drop_detached() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    for f in 1..=66u32 {
        scene.frame();
        let n = scene.x.len();
        let mut lonely = Vec::new();
        for p in 0..n {
            let nearest = (0..n)
                .filter(|&q| q != p)
                .map(|q| (scene.x[q] - scene.x[p]).length())
                .fold(f32::INFINITY, f32::min);
            if nearest > 1.0 {
                lonely.push((p, nearest));
            }
        }
        let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
        let speed = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
        println!(
            "f{f} t={:.3}: {} particles with no neighbour within 1 cell, lowest y {lowest:.2}, top speed {speed:.1}",
            scene.time,
            lonely.len()
        );
        for &(p, d) in lonely.iter().take(3) {
            println!(
                "   {p} at {:?}, nearest {d:.2}, v {:?}, F det {:.3}, J {:.3}",
                scene.x[p],
                scene.v[p],
                scene.deformation[p].determinant(),
                scene.j(p)
            );
        }
    }
}

/// Diagnosis of the new A2 attempt's first run: the mirror wall stands for
/// an image body, so a block hitting the floor must move exactly as one of
/// two identical blocks hitting each other head on, with no wall near.
/// No gravity; kinetic plus elastic energy and the top speed of each case
/// (`EMERGE_A2_CASE` = wall or pair). A probe.
#[test]
#[ignore = "diagnostic probe for the new A2 attempt: run with --ignored --nocapture"]
fn probe_a2_wall_against_image_pair() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let case = std::env::var("EMERGE_A2_CASE").unwrap_or_else(|_| "wall".into());
    let tank = (Vec2::splat(4.0), Vec2::new(68.0, 106.0));
    let block = |lo: Vec2, rng: &mut LcgRng| {
        let hi = lo + Vec2::splat(10.0);
        seed(
            (lo.as_ivec2(), hi.as_ivec2()),
            |_| true,
            |q| {
                let d = [
                    (q.y - lo.y, Vec2::NEG_Y),
                    (hi.y - q.y, Vec2::Y),
                    (q.x - lo.x, Vec2::NEG_X),
                    (hi.x - q.x, Vec2::X),
                ];
                d.into_iter().fold(
                    (f32::INFINITY, Vec2::Y),
                    |m, c| if c.0 < m.0 { c } else { m },
                )
            },
            rng,
        )
    };
    let mut rng = LcgRng::new(17);
    // Wall: base 6 cells above the floor (y = 4). Pair: the same gap to the
    // plane y = 46 on both sides.
    let (mut x, below) = if case == "pair" {
        let lower = block(Vec2::new(31.0, 30.0), &mut rng);
        let n = lower.len();
        let upper: Vec<Vec2> = lower.iter().map(|q| Vec2::new(q.x, 92.0 - q.y)).collect();
        let mut all = upper;
        all.extend(lower);
        (all, n)
    } else {
        (block(Vec2::new(31.0, 10.0), &mut rng), 0)
    };
    x.shrink_to_fit();
    let mut scene = Scene::new((72, 110), 0.01, 0.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    let speed0 = 2.0 / scene.cell_m;
    let n = scene.x.len();
    for p in 0..n {
        // The upper block (all of the wall case, the first half of the
        // pair) moves down, the lower block of the pair up.
        let down = case != "pair" || p < n - below;
        scene.v[p].y = if down { -speed0 } else { speed0 };
    }
    let (mu, lambda) = (
        scene.shear.unwrap_or(0.0),
        scene.sound_speed2.unwrap_or(0.0),
    );
    let energy = |s: &Scene| -> f64 {
        (0..s.x.len())
            .map(|p| {
                f64::from(
                    s.mass[p] * 0.5 * s.v[p].length_squared()
                        + s.v0[p]
                            * (shear_energy(s.deformation[p], mu)
                                + 0.5 * lambda * (s.j(p) - 1.0).powi(2)),
                )
            })
            .sum::<f64>()
            / s.x.len() as f64
    };
    let e0 = energy(&scene);
    if std::env::var_os("EMERGE_A2_TRACE").is_some() {
        let start: f32 = std::env::var("EMERGE_A2_TRACE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.25);
        while scene.time < start {
            scene.frame();
        }
        let mut frame = scene.frame();
        for step in 0..3000 {
            // The frame's own choice of substep (`Scene::frame`).
            let speed = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
            let dt = (0.5 * scene.layout.dx / mu.sqrt()).min(travel_limited_dt(
                speed,
                scene.g(),
                scene.layout.dx,
                scene.settings.max_cells_per_substep,
            ));
            scene.substep(dt, &mut frame);
            scene.time += dt;
            let (mut c_max, mut at) = (0.0f32, 0usize);
            for p in 0..scene.x.len() {
                let c = scene.c[p].to_cols_array().iter().fold(0.0f32, |m, e| {
                    if e.is_finite() {
                        m.max(e.abs())
                    } else {
                        f32::NAN
                    }
                });
                if c.is_nan() {
                    c_max = c;
                    at = p;
                    break;
                }
                if c > c_max {
                    c_max = c;
                    at = p;
                }
            }
            let f = scene.deformation[at];
            let fh = f / f.determinant().abs().sqrt();
            let tau = shear_kirchhoff(f, mu);
            let phi = scene.phi();
            let cell = scene.x[at].floor().as_ivec2();
            let near = (0..scene.x.len())
                .filter(|&q| (scene.x[q] - scene.x[at]).length() < 1.0)
                .count();
            if step % 5 == 0 || c_max > 300.0 || !c_max.is_finite() {
                println!(
                    "trace t={:.4}: |C| {c_max:.1} at {:?} phi {:.2}, {near} within 1 cell, det F {:.3}, J {:.3}, |F^| cols {:.3} {:.3}, |tau|/mu {:.3}, C {:?}",
                    scene.time,
                    scene.x[at],
                    phi.get_signed(cell.x, cell.y).unwrap_or(f32::NAN),
                    f.determinant(),
                    scene.j(at),
                    fh.x_axis.length(),
                    fh.y_axis.length(),
                    (tau.x_axis.length() + tau.y_axis.length()) / mu,
                    scene.c[at]
                );
            }
            if !c_max.is_finite() {
                break;
            }
        }
        return;
    }
    for f in 1..=180u32 {
        scene.frame();
        let top = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
        if f % 3 == 0 || !top.is_finite() {
            let (com, _) = scene.centre_of_mass();
            let dx2 = scene.layout.dx * scene.layout.dx;
            let (mut l_orbit, mut l_spin, mut p_tot) = (0.0f64, 0.0f64, Vec2::ZERO);
            let (mut lo, mut hi) = (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY));
            for q in 0..scene.x.len() {
                let r = scene.x[q] - com;
                l_orbit += f64::from(scene.mass[q] * (r.x * scene.v[q].y - r.y * scene.v[q].x));
                l_spin += f64::from(
                    scene.mass[q] * 0.25 * dx2 * (scene.c[q].x_axis.y - scene.c[q].y_axis.x),
                );
                p_tot += scene.mass[q] * scene.v[q];
                lo = lo.min(scene.x[q]);
                hi = hi.max(scene.x[q]);
            }
            let m: f32 = scene.mass.iter().sum();
            println!(
                "{case} f{f} t={:.3}: energy {:+.4}, top speed {:.2} of impact, L orbit {:.3e} spin {:.3e} (scale {:.3e}), momentum {:?} of impact, extent {:?}..{:?}",
                scene.time,
                energy(&scene) / e0 - 1.0,
                top / speed0,
                l_orbit,
                l_spin,
                f64::from(m * speed0 * 5.0),
                p_tot / (m * speed0),
                lo,
                hi
            );
        }
        if !top.is_finite() {
            break;
        }
    }
}

/// Where scene 4's drop into a pool loses a particle through a wall, after
/// the A2 transfer changes. A probe.
#[test]
#[ignore = "diagnostic probe for the new A2 attempt: run with --ignored --nocapture"]
fn probe_pool_crossings() {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, drop_into_pool_seed());
    let mut deepest = (0.0f32, 0usize, 0u32);
    for f in 0..(2.0 / FRAME).round() as u32 {
        scene.frame();
        for p in 0..scene.x.len() {
            let q = scene.x[p];
            let beyond = (tank.0 - q).max(q - tank.1).max_element();
            if beyond >= 0.0 && beyond >= deepest.0 {
                deepest = (beyond, p, f);
            }
        }
        let out = (0..scene.x.len())
            .filter(|&p| !scene.inside_tank(p))
            .count();
        if out > 0 {
            println!("frame {f}: {out} out");
        }
    }
    println!(
        "deepest beyond a wall: {:e} cell (particle {}, frame {})",
        deepest.0, deepest.1, deepest.2
    );
    for &(t, before, v, gap) in &scene.probe.crossings {
        println!(
            "crossing t={t:.4}s from ({:.7},{:.4}) v=({:e},{:.1}) gap {gap:e}",
            before.x, before.y, v.x, v.y
        );
    }
}

/// Where the dam break (scene 3, and scene 7's real water with
/// `EMERGE_PROBE_WATER`) loses particles after the ghost pressure: how far
/// beyond which wall, when, and how fast. A probe.
#[test]
#[ignore = "diagnostic probe for the third A2 attempt: run with --ignored --nocapture"]
fn probe_dam_break_crossings() {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, dam_break_seed());
    if std::env::var_os("EMERGE_PROBE_WATER").is_some() {
        scene = scene.with_sound_speed(1483.0);
    }
    let mut seen = std::collections::BTreeMap::new();
    for f in 0..(2.0 / FRAME).round() as u32 {
        scene.frame();
        for p in 0..scene.x.len() {
            let q = scene.x[p];
            let beyond = (tank.0 - q).max(q - tank.1).max_element();
            if beyond > 0.0 {
                let entry = seen.entry(p).or_insert((f, 0.0f32, q, scene.v[p], 0u32));
                entry.4 += 1;
                if beyond > entry.1 {
                    entry.1 = beyond;
                    entry.2 = q;
                    entry.3 = scene.v[p];
                }
            }
        }
    }
    println!("{} particles ever beyond a wall", seen.len());
    println!(
        "{} reflections, deepest {:.2} ulps",
        scene.probe.reflections, scene.probe.deepest_reflection_ulps
    );
    for (p, (first, depth, q, v, frames)) in &seen {
        println!(
            "particle {p}: first frame {first}, {frames} frames out, deepest {depth:e} cell at ({:.4},{:.4}), v ({:.1},{:.1})",
            q.x, q.y, v.x, v.y
        );
    }
}

/// Particle 478 of the real-water dam break, the first to pass through
/// the left wall in the third A2 attempt: its position and the velocity it
/// gathers every substep around the moment it enters the wall. A probe.
#[test]
#[ignore = "diagnostic probe for the third A2 attempt: run with --ignored --nocapture"]
fn probe_dam_break_particle_through_wall() {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene =
        Scene::new((72, 72), 0.01, 1.0, tank, dam_break_seed()).with_sound_speed(1483.0);
    let p = 478usize;
    for f in 0..48u32 {
        if f < 40 {
            scene.frame();
            continue;
        }
        let before = scene.x[p];
        scene.frame();
        println!(
            "frame {f}: x ({:.6},{:.6}) -> ({:.6},{:.6}), v ({:.3},{:.3}), C {:?}",
            before.x, before.y, scene.x[p].x, scene.x[p].y, scene.v[p].x, scene.v[p].y, scene.c[p]
        );
    }
}

/// Where the projection's time goes on scene 10's explicit control: each
/// piece of `project` timed alone, 200 times, on the state at 0.5 s. A
/// probe.
#[test]
#[ignore = "cost probe for real time: run with --ignored --nocapture"]
fn probe_projection_pieces() {
    use super::{extrapolate, pcg, pressure};
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |_| (f32::INFINITY, Vec2::Y),
        &mut rng,
    );
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_elastic(e_over_rho_for_shear_speed(c_s, nu), nu);
    while scene.time < 0.5 {
        scene.frame();
    }
    let dt = 1.0e-3;
    let (vel, face_mass) =
        particles_to_faces(&scene.layout, &scene.x, &scene.v, &scene.c, &scene.mass);
    let phi = scene.phi();
    let image = box_container_image(scene.tank.0, scene.tank.1);
    let open = scene.open_centres();
    let surface = pressure::Surface {
        phi: &phi,
        theta_floor: scene.settings.theta_floor,
    };
    let reps = 200u32;
    let time = |name: &str, f: &mut dyn FnMut()| {
        let t = Instant::now();
        for _ in 0..reps {
            f();
        }
        println!(
            "PIECE {name:<22} {:8.1} us",
            t.elapsed().as_secs_f64() * 1e6 / f64::from(reps)
        );
    };
    let system = pressure::assemble(&scene.layout, dt, &vel, &scene.weights, surface);
    time("assemble", &mut || {
        std::hint::black_box(pressure::assemble(
            &scene.layout,
            dt,
            &vel,
            &scene.weights,
            surface,
        ));
    });
    let solution = pcg::solve(&system, &scene.settings.solver);
    println!(
        "unknowns {}, iterations {}",
        system.unknowns(),
        solution.iterations
    );
    time("pcg solve", &mut || {
        std::hint::black_box(pcg::solve(&system, &scene.settings.solver));
    });
    let mut mic = scene.settings.solver;
    mic.preconditioner = super::pcg::Preconditioner::Mic0;
    let with_mic = pcg::solve(&system, &mic);
    println!("MIC(0): iterations {}", with_mic.iterations);
    time("pcg solve, MIC(0)", &mut || {
        std::hint::black_box(pcg::solve(&system, &mic));
    });
    let mut v = vel.clone();
    time("clone velocity", &mut || {
        v = std::hint::black_box(vel.clone());
    });
    let mut valid = pressure::apply_pressure(
        &scene.layout,
        dt,
        &solution.pressure,
        &scene.weights,
        surface,
        &mut v,
    );
    time("apply pressure", &mut || {
        let mut w = vel.clone();
        std::hint::black_box(pressure::apply_pressure(
            &scene.layout,
            dt,
            &solution.pressure,
            &scene.weights,
            surface,
            &mut w,
        ));
    });
    let ghost = extrapolate::ghost_pressure(
        &scene.layout,
        dt,
        &phi,
        &solution.pressure,
        &open,
        (
            scene.settings.theta_floor,
            scene.settings.extrapolation_layers,
        ),
    );
    time("ghost pressure", &mut || {
        std::hint::black_box(extrapolate::ghost_pressure(
            &scene.layout,
            dt,
            &phi,
            &solution.pressure,
            &open,
            (
                scene.settings.theta_floor,
                scene.settings.extrapolation_layers,
            ),
        ));
    });
    time("keep reached faces", &mut || {
        let (mut w, mut f) = (v.clone(), valid.clone());
        extrapolate::keep_reached_faces(&mut w, &vel, &mut f, &face_mass, &scene.weights, &ghost);
    });
    extrapolate::keep_reached_faces(&mut v, &vel, &mut valid, &face_mass, &scene.weights, &ghost);
    time("extrapolate velocity", &mut || {
        let (mut w, mut f) = (v.clone(), valid.clone());
        extrapolate::extrapolate_velocity(&mut w, &mut f, scene.settings.extrapolation_layers);
    });
    time("constrain to solids", &mut || {
        let mut w = v.clone();
        extrapolate::constrain_to_solids(&scene.layout, &mut w, &scene.weights, &image);
    });
    time("level set", &mut || {
        std::hint::black_box(scene.phi());
    });
}

/// One row of the material matrix: measured coefficients only.
struct MaterialRow {
    name: &'static str,
    /// kg/m^3.
    rho: f32,
    /// Plane-strain Lame coefficients, Pa.
    lambda: f32,
    mu: f32,
}

impl MaterialRow {
    fn from_young(name: &'static str, rho: f32, e: f32, nu: f32) -> Self {
        Self {
            name,
            rho,
            lambda: e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)),
            mu: e / (2.0 * (1.0 + nu)),
        }
    }

    fn c_p(&self) -> f32 {
        ((self.lambda + 2.0 * self.mu) / self.rho).sqrt()
    }

    fn c_s(&self) -> f32 {
        (self.mu / self.rho).sqrt()
    }
}

/// The rows (criteria, step M1).
fn material_rows() -> Vec<MaterialRow> {
    let clay = |name, q_u: f32| MaterialRow::from_young(name, 1700.0, 300.0 * q_u / 2.0, 0.45);
    vec![
        MaterialRow {
            name: "water",
            rho: 1000.0,
            lambda: 2.2e9,
            mu: 0.0,
        },
        clay("very soft clay", 20.0e3),
        clay("medium clay", 75.0e3),
    ]
}

/// A plane wave through a free block of the row's material: longitudinal
/// (P) or transverse (S). Returns the arrival's relative error and the wall
/// time per simulated second, in ms.
fn plane_wave(row: &MaterialRow, longitudinal: bool, report: &mut Report) -> (f32, f32) {
    let kind = if longitudinal { "P" } else { "S" };
    let label = format!("{}, {kind} wave", row.name);
    let speed = if longitudinal { row.c_p() } else { row.c_s() };
    // Edges' disturbances travel at most `c_p`; the window ends when the
    // kick's back has passed the probe, 50 cells, and they must arrive
    // from half the height later.
    let height = (2.0 * (50.0 * row.c_p() / speed + 4.0)).ceil() as usize + 4;
    let length = 100usize;
    let (nx, ny) = (length + 8, height + 8);
    let (x0, y0) = (4.0f32, 4.0f32);
    let tank = (
        Vec2::splat(1.0),
        Vec2::new(nx as f32 - 1.0, ny as f32 - 1.0),
    );
    let mut rng = LcgRng::new(13);
    let (top, right) = (y0 + height as f32, x0 + length as f32);
    let x = seed(
        (
            IVec2::new(x0 as i32, y0 as i32),
            IVec2::new(right as i32, top as i32),
        ),
        |_| true,
        |q| {
            let d = [
                (q.y - y0, Vec2::NEG_Y),
                (top - q.y, Vec2::Y),
                (q.x - x0, Vec2::NEG_X),
                (right - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene =
        Scene::new((nx, ny), 0.01, 0.0, tank, x).with_lame(row.lambda / row.rho, row.mu / row.rho);
    let c = speed / scene.cell_m;
    scene.fixed_dt = Some(0.5 * scene.layout.dx / (row.c_p().max(row.c_s()) / scene.cell_m));
    let kick = 0.05 / scene.cell_m;
    for p in 0..scene.x.len() {
        if scene.x[p].x < x0 + 10.0 {
            if longitudinal {
                scene.v[p].x = kick;
            } else {
                scene.v[p].y = kick;
            }
        }
    }
    let mid = y0 + 0.5 * height as f32;
    scene.probe.watch_x = Some((x0 + 50.0, mid - 4.0, mid + 4.0));
    scene.probe.watch_longitudinal = longitudinal;
    let expected = 40.0 / c;
    let window = 50.0 / c;
    let mut nan = false;
    let wall_clock = Instant::now();
    scene.run_until(window);
    nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
    let wall_s = wall_clock.elapsed().as_secs_f32();
    let history = &scene.probe.watch_x_history;
    let half = 0.25 * kick;
    let arrival = history.windows(2).find_map(|pair| {
        let ((t0, v0), (t1, v1)) = (pair[0], pair[1]);
        (v0 < half && v1 >= half).then(|| t0 + (t1 - t0) * (half - v0) / (v1 - v0))
    });
    let error = arrival.map_or(f32::INFINITY, |t| (t - expected) / expected);
    let ms = 1000.0 * wall_s / scene.time.max(f32::MIN_POSITIVE);
    println!(
        "{label}: {} m/s expected, arrival {:+.1} % ({} particles, {height} cells high), {ms:.0} ms per simulated s",
        speed,
        100.0 * error,
        scene.x.len()
    );
    report.check(!nan, || format!("{label}: NaN"));
    report.check(error.abs() <= 0.05, || {
        format!("{label}: arrival off by {:+.1} %", 100.0 * error)
    });
    scene.check_reflections(&label, report);
    (error, ms)
}

#[test]
#[ignore = "material matrix, P waves: run with --ignored --nocapture"]
fn gate_matrix_p_waves() {
    let mut report = Report::new();
    for row in material_rows() {
        plane_wave(&row, true, &mut report);
    }
    report.finish();
}

#[test]
#[ignore = "material matrix, S waves: run with --ignored --nocapture"]
fn gate_matrix_s_waves() {
    let mut report = Report::new();
    for row in material_rows().iter().filter(|r| r.mu > 0.0) {
        plane_wave(row, false, &mut report);
    }
    report.finish();
}

/// Kinetic, potential, shear and volume energy, the floor as zero: the
/// drop's measure for any row.
fn row_energy(scene: &Scene) -> f64 {
    let floor = scene.tank.0.y;
    (0..scene.x.len())
        .map(|p| {
            let shear = scene
                .shear
                .map_or(0.0, |mu| shear_energy(scene.deformation[p], mu));
            let volume = scene
                .sound_speed2
                .map_or(0.0, |kappa| 0.5 * kappa * (scene.j(p) - 1.0).powi(2));
            f64::from(
                scene.mass[p]
                    * (0.5 * scene.v[p].length_squared() + scene.g() * (scene.x[p].y - floor))
                    + scene.v0[p] * (shear + volume),
            )
        })
        .sum()
}

/// The rest scene for one row (criteria, step M2). Returns the wall time
/// per simulated second, in ms.
fn rest_column(row: &MaterialRow, report: &mut Report) -> f32 {
    let label = format!("{}, at rest", row.name);
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let floor = wall as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let h0 = floor + depth as f32;
    let x = seed(
        (
            IVec2::new(wall as i32, wall as i32),
            IVec2::new((wall + width) as i32, (wall + depth) as i32),
        ),
        |_| true,
        |q| box_free_sides(q, Some(h0), None),
        &mut rng,
    );
    let mut scene =
        Scene::new((nx, ny), 0.01, 1.0, tank, x).with_lame(row.lambda / row.rho, row.mu / row.rho);
    let g = scene.g();
    let m_over_rho = (row.lambda + 2.0 * row.mu) / row.rho / (scene.cell_m * scene.cell_m);
    let seconds = 2.0f32;
    let frames = (seconds / FRAME).round() as u32;
    let last_second = frames - (1.0 / FRAME).round() as u32;
    let bands = depth / 5;
    let mut band_j = vec![(0.0f64, 0u64); bands];
    let mut nan = false;
    let wall_clock = Instant::now();
    for f in 1..=frames {
        scene.frame();
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        if f > last_second {
            for p in 0..scene.x.len() {
                let k = ((scene.x[p].y - floor) / 5.0).floor();
                if k >= 0.0 && (k as usize) < bands {
                    let entry = &mut band_j[k as usize];
                    entry.0 += f64::from(scene.j(p));
                    entry.1 += 1;
                }
            }
        }
    }
    let ms = 1000.0 * wall_clock.elapsed().as_secs_f32() / seconds;
    let drift = scene.settings.solver.absolute_tolerance * seconds;
    let full_strain = g * depth as f32 / m_over_rho;
    let mut worst = 0.0f32;
    for (k, &(sum, n)) in band_j.iter().enumerate() {
        if n == 0 {
            continue;
        }
        let mean = (sum / n as f64) as f32;
        let y = floor + 5.0 * k as f32 + 2.5;
        let expected = 1.0 - g * (h0 - y) / m_over_rho;
        let bound = 0.05 * full_strain + drift;
        worst = worst.max((mean - expected).abs() / bound);
        report.check((mean - expected).abs() <= bound, || {
            format!("{label}: band {k} mean J {mean:.6}, expected {expected:.6}, bound {bound:.1e}")
        });
    }
    println!(
        "{label}: strain at the floor {full_strain:.2e}, worst band {worst:.2} of its bound, {ms:.0} ms per simulated s ({:.2}x real time)",
        1000.0 / ms
    );
    report.check(!nan, || format!("{label}: NaN"));
    scene.check_reflections(&label, report);
    ms
}

/// The drop for one row (criteria, step M2). Returns the wall time per
/// simulated second, in ms.
fn drop_block(row: &MaterialRow, report: &mut Report) -> f32 {
    let label = format!("{}, dropped", row.name);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene =
        Scene::new((72, 72), 0.01, 1.0, tank, x).with_lame(row.lambda / row.rho, row.mu / row.rho);
    let gyration = |scene: &Scene| {
        let (com, _) = scene.centre_of_mass();
        let n = scene.x.len() as f32;
        (scene
            .x
            .iter()
            .map(|q| (*q - com).length_squared())
            .sum::<f32>()
            / n)
            .sqrt()
    };
    let (energy_start, gyration_start, volume_start) =
        (row_energy(&scene), gyration(&scene), scene.volume());
    let seconds = 1.0f32;
    let (mut worst_energy, mut nan) = (f64::NEG_INFINITY, false);
    let wall_clock = Instant::now();
    for _ in 1..=(seconds / FRAME).round() as u32 {
        scene.frame();
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        worst_energy = worst_energy.max(row_energy(&scene) / energy_start - 1.0);
    }
    let ms = 1000.0 * wall_clock.elapsed().as_secs_f32() / seconds;
    let change = (gyration(&scene) / gyration_start - 1.0).abs();
    let volume_change = (scene.volume() / volume_start - 1.0).abs();
    // Reported, not judged yet: what the run dissipated, a realism defect
    // the criteria do not bound (the very soft clay lost 60 percent in 1 s,
    // bouncing).
    let energy_end = row_energy(&scene) / energy_start - 1.0;
    println!(
        "{label}: highest energy {worst_energy:+.4} of the start, at the end {energy_end:+.4}, gyration change {change:.4}, volume change {volume_change:.4}, {ms:.0} ms per simulated s ({:.2}x real time)",
        1000.0 / ms
    );
    report.check(!nan, || format!("{label}: NaN"));
    report.check(worst_energy <= 0.01, || {
        format!("{label}: energy rose {worst_energy:+.4}")
    });
    if row.mu > 0.0 {
        report.check(change <= 0.05, || {
            format!("{label}: gyration changed {change:.4}")
        });
    } else {
        report.check(volume_change <= 0.02, || {
            format!("{label}: volume changed {volume_change:.4}")
        });
    }
    scene.check_reflections(&label, report);
    ms
}

#[test]
#[ignore = "material matrix, rest: run with --ignored --nocapture"]
fn gate_matrix_rest() {
    let mut report = Report::new();
    for row in material_rows() {
        rest_column(&row, &mut report);
    }
    report.finish();
}

#[test]
#[ignore = "material matrix, drop: run with --ignored --nocapture"]
fn gate_matrix_drop() {
    let mut report = Report::new();
    for row in material_rows() {
        drop_block(&row, &mut report);
    }
    report.finish();
}

/// Check of step M2's first run: the very soft clay's drop kept its energy
/// to 1e-4. Does it reach the floor and bounce? Lowest particle, energy
/// and centre speed every tenth of a second. A probe.
#[test]
#[ignore = "diagnostic probe for step M2: run with --ignored --nocapture"]
fn probe_m2_clay_drop_trajectory() {
    let row = &material_rows()[1];
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |_| (f32::INFINITY, Vec2::Y),
        &mut rng,
    );
    let mut scene =
        Scene::new((72, 72), 0.01, 1.0, tank, x).with_lame(row.lambda / row.rho, row.mu / row.rho);
    let e0 = row_energy(&scene);
    for f in 1..=60u32 {
        scene.frame();
        if f % 6 == 0 {
            let (_, v) = scene.centre_of_mass();
            let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
            println!(
                "{} t={:.2}: lowest y {lowest:.3}, centre vy {:.2} m/s, energy {:+.5}",
                row.name,
                scene.time,
                v.y * scene.cell_m,
                row_energy(&scene) / e0 - 1.0
            );
        }
    }
}

/// The free body of step M3 for one row at `k` cells per centimetre.
fn free_body(row: &MaterialRow, k: usize, report: &mut Report) {
    let kf = k as f32;
    let label = format!("{}, free at {} cm", row.name, 1.0 / kf);
    let tank = (Vec2::splat(4.0 * kf), Vec2::splat(68.0 * kf));
    let mut rng = LcgRng::new(17);
    // Without shear stiffness the block translates (criteria amendment).
    let translate = row.mu == 0.0;
    let corner = if translate { 8.0 } else { 31.0 };
    let (lo, hi) = (Vec2::splat(corner) * kf, Vec2::splat(corner + 10.0) * kf);
    let x = seed(
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |q| {
            let d = [
                (q.y - lo.y, Vec2::NEG_Y),
                (hi.y - q.y, Vec2::Y),
                (q.x - lo.x, Vec2::NEG_X),
                (hi.x - q.x, Vec2::X),
            ];
            d.into_iter().fold(
                (f32::INFINITY, Vec2::Y),
                |m, c| if c.0 < m.0 { c } else { m },
            )
        },
        &mut rng,
    );
    let mut scene = Scene::new((72 * k, 72 * k), 0.01 / kf, 0.0, tank, x)
        .with_lame(row.lambda / row.rho, row.mu / row.rho);
    let centre = 0.5 * (lo + hi);
    // Largest speed 0.5 m/s at the block's corners, `s |r|_max`.
    let s = 0.5 / scene.cell_m / (5.0 * kf * std::f32::consts::SQRT_2);
    for p in 0..scene.x.len() {
        let r = scene.x[p] - centre;
        if translate {
            scene.v[p] = 0.5 / scene.cell_m * Vec2::ONE.normalize();
        } else {
            scene.v[p] = s * Vec2::new(r.y, r.x);
            scene.c[p] = Mat2::from_cols(Vec2::new(0.0, s), Vec2::new(s, 0.0));
        }
    }
    let largest = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
    let (com0, vcom0) = scene.centre_of_mass();
    let inertia: f32 = (0..scene.x.len())
        .map(|p| scene.mass[p] * (scene.x[p] - com0).length_squared())
        .sum();
    let angular = |scene: &Scene| -> f32 {
        let (com, vcom) = scene.centre_of_mass();
        (0..scene.x.len())
            .map(|p| {
                let (r, v) = (scene.x[p] - com, scene.v[p] - vcom);
                scene.mass[p] * (r.x * v.y - r.y * v.x)
            })
            .sum()
    };
    let (l0, e0) = (angular(&scene), row_energy(&scene));
    let (mut worst_v, mut worst_l, mut worst_e, mut nan) =
        (0.0f32, 0.0f32, f64::NEG_INFINITY, false);
    for _ in 0..(1.0 / FRAME).round() as u32 {
        scene.frame();
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        let (_, vcom) = scene.centre_of_mass();
        worst_v = worst_v.max((vcom - vcom0).length() / largest);
        worst_l = worst_l.max((angular(&scene) - l0).abs() / (inertia * s));
        worst_e = worst_e.max(row_energy(&scene) / e0 - 1.0);
    }
    println!(
        "{label}: centre speed up to {worst_v:.4} of the largest start speed, angular momentum off by {worst_l:.4} of I s, energy up to {worst_e:+.4}"
    );
    report.check(!nan, || format!("{label}: NaN"));
    report.check(worst_v <= 0.01, || {
        format!("{label}: centre reached {worst_v:.4} of the start speed")
    });
    report.check(worst_l <= 0.01, || {
        format!("{label}: angular momentum off by {worst_l:.4}")
    });
    report.check(worst_e <= 0.01, || {
        format!("{label}: energy rose {worst_e:+.4}")
    });
}

#[test]
#[ignore = "material matrix, free body (step M3): run with --ignored --nocapture"]
fn gate_matrix_free_body() {
    let mut report = Report::new();
    for row in material_rows() {
        for k in [1usize, 2] {
            free_body(&row, k, &mut report);
        }
    }
    report.finish();
}

/// Step M3's free body frame by frame for each row: substeps, largest
/// speed, the centre's speed and the energy against the start, stopping
/// after two minutes of wall time. The resolution is `EMERGE_PROBE_K`
/// cells per centimetre. A probe.
#[test]
#[ignore = "diagnostic probe for step M3: run with --ignored --nocapture"]
fn probe_free_body_frames() {
    let k: usize = std::env::var("EMERGE_PROBE_K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1);
    let kf = k as f32;
    let only = std::env::var("EMERGE_PROBE_ROW").ok();
    for row in material_rows()
        .into_iter()
        .filter(|r| only.as_deref().is_none_or(|o| r.name == o))
    {
        let tank = (Vec2::splat(4.0 * kf), Vec2::splat(68.0 * kf));
        let mut rng = LcgRng::new(17);
        let (lo, hi) = (Vec2::new(31.0, 31.0) * kf, Vec2::new(41.0, 41.0) * kf);
        let x = seed(
            (lo.as_ivec2(), hi.as_ivec2()),
            |_| true,
            |_| (f32::INFINITY, Vec2::Y),
            &mut rng,
        );
        let mut scene = Scene::new((72 * k, 72 * k), 0.01 / kf, 0.0, tank, x)
            .with_lame(row.lambda / row.rho, row.mu / row.rho);
        let centre = 0.5 * (lo + hi);
        let s = 0.5 / scene.cell_m / (5.0 * kf * std::f32::consts::SQRT_2);
        for p in 0..scene.x.len() {
            let r = scene.x[p] - centre;
            scene.v[p] = s * Vec2::new(r.y, r.x);
            scene.c[p] = Mat2::from_cols(Vec2::new(0.0, s), Vec2::new(s, 0.0));
        }
        let e0 = row_energy(&scene);
        let clock = Instant::now();
        for f in 1..=60u32 {
            let frame = scene.frame();
            let top = scene.v.iter().map(|v| v.length()).fold(0.0f32, |m, s| {
                if s.is_nan() || m.is_nan() {
                    f32::NAN
                } else {
                    m.max(s)
                }
            });
            let (_, vcom) = scene.centre_of_mass();
            let late = clock.elapsed().as_secs() > 120;
            if f <= 3 || f % 6 == 0 || !top.is_finite() || late {
                println!(
                    "{} frame {f}: {} substeps, largest speed {:.3} m/s, centre {:.4} m/s, energy {:.4} of the start, {:.1} s wall",
                    row.name,
                    frame.substeps,
                    top * scene.cell_m,
                    vcom.length() * scene.cell_m,
                    row_energy(&scene) / e0,
                    clock.elapsed().as_secs_f32()
                );
            }
            if !top.is_finite() || late {
                break;
            }
        }
    }
}

/// Scene 1's column at 1 cm and one gravity, frame by frame for the first
/// second: largest particle speed, largest pressure error in cells of head
/// and where, largest move. A probe.
#[test]
#[ignore = "diagnostic probe for step M3: run with --ignored --nocapture"]
fn probe_column_frames() {
    let (wall, width, depth) = (4usize, 40usize, 30usize);
    let (nx, ny) = (width + 2 * wall, wall + depth + 18);
    let floor = wall as f32;
    let h = floor + depth as f32;
    let tank = (
        Vec2::new(wall as f32, floor),
        Vec2::new((wall + width) as f32, 1.0e6),
    );
    let mut rng = LcgRng::new(7);
    let water = (
        IVec2::new(wall as i32, wall as i32),
        IVec2::new((wall + width) as i32, (wall + depth) as i32),
    );
    let x = seed(
        water,
        |_| true,
        |q| box_free_sides(q, Some(h), None),
        &mut rng,
    );
    let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x);
    let start = scene.x.clone();
    let g = scene.g();
    let clock = Instant::now();
    for f in 1..=60u32 {
        let frame = scene.frame();
        let top = scene.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
        let moved = scene
            .x
            .iter()
            .zip(&start)
            .fold(0.0f32, |m, (a, b)| m.max((*a - *b).length()));
        let mut worst = (0.0f32, 0usize, 0usize);
        for j in wall..ny {
            for i in wall..wall + width {
                let c = i + nx * j;
                let y = j as f32 + 0.5;
                if y >= h || !frame.active[c] {
                    continue;
                }
                let e = (frame.pressure[c] - g * (h - y)).abs() / g;
                if e > worst.0 {
                    worst = (e, i, j);
                }
            }
        }
        if f <= 6 || f % 6 == 0 {
            println!(
                "frame {f}: {} substeps, largest speed {:.3} cells/s, pressure off by {:.3} cell of head at ({}, {}), largest move {:.3} cell",
                frame.substeps, top, worst.0, worst.1, worst.2, moved
            );
        }
    }
    println!("{:.2} s wall for 1 s", clock.elapsed().as_secs_f32());
}
