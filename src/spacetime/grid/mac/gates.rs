//! Stop criteria for the MAC projection and the solids built on it, each
//! written before its code and frozen: a criterion moves only by a declared
//! amendment, with its reason.
//!
//! # What is under test
//!
//! The standard formulation (module doc) on the engine's own particle state
//! and units, apart from `Simulation::step`. How it couples to the nodal
//! MLS-MPM grid is the next step, with its own sources read first.
//!
//! # Fixed setup
//!
//! - Gravity is the only thing that changes between runs of the column at
//!   rest and the droplet in free fall: 0.38, 1 and 2.5 times 9.81 m/s^2.
//!   The dam break and the drop into a pool run at 1.
//! - Cells of 1 cm unless stated. No equation of state, no viscosity, no
//!   surface tension: the projection alone holds the fluid.
//! - Frame 1/60 s. Substeps are chosen so a particle moving at the largest
//!   speed, plus what gravity adds over the substep, travels at most one
//!   cell (Zhu and Bridson 4.2.5). Particles keep the velocity they
//!   gather. Positions advance by the midpoint rule through the projected
//!   velocity (Bridson and Muller-Fischer 3.1) instead of the engine's
//!   `x += v dt`, a declared change of setup with its reason at
//!   `Scene::substep`. The criteria below did not move with it.
//! - Four particles per cell, each placed at random inside its quarter of
//!   the cell; those within one cell of a free surface are moved along its
//!   normal to half a cell from it (Zhu and Bridson 4.2.1). Particle radius
//!   in the level set: the particle spacing; kernel radius: twice that
//!   (their section 5).
//! - J is the fluid's own: its logarithm advances by `dt div v` each
//!   substep through `advance_log_volume_ratio`, with the engine's bounds
//!   [0.5, 2.0], `div v` the divergence of the liquid cells at the particle
//!   rather than the trace of the gathered velocity gradient, a declared
//!   change of setup with its reason at `Scene::substep`.
//!   The fluid's volume is `sum(V0 J)`.
//! - Named settings, each with its reason at its definition: the ghost
//!   fraction floor (0.01, as `apic2d`), the conjugate gradient tolerance
//!   and iteration cap, the MIC(0) parameter, the extrapolation depth, the
//!   travel per substep.
//! - Not allowed: a relaxation factor on the correction, a velocity clamp,
//!   damping, pushing particles back out of walls, or a J at its bounds.
//!
//! # Column at rest
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
//! # Droplet in free fall
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
//! # Dam break
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
//! # Drop into a pool
//!
//! Disc of radius 5 cm released at rest with its centre 0.40 m above the
//! floor, over water 0.15 m deep filling the same closed tank, 2 s. Same
//! criteria as the dam break.
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
//! A failure is a full run of the scenes in which any criterion above is
//! missed, once the implementation is believed complete. After the first:
//! one diagnosis, one fix, labelled real fix or declared approximation; a
//! crutch is not allowed. After the second: stop and record it in
//! `KNOWN_LIMITATIONS.md`.
//!
//! # Compressible projection, criteria written before its code
//!
//! The generalized Chorin projection of Stomakhin, Schroeder, Jiang, Chai,
//! Teran and Selle, *Augmented MPM for phase-change and varied materials*,
//! 2014, eqs. 14 to 18: `dp/dt = -K div v` taken implicitly with the
//! pressure update, so the pressure system gains `1 / (c^2 dt)` on its
//! diagonal and `q_n / (c^2 dt)` on its right-hand side (`q = p / rho`,
//! `c^2 = K / rho`, `q_n` from the particles' J through the linear
//! equation of state `q = -c^2 (J - 1)`, their eq. 10 without plasticity).
//! As `c` grows the system becomes the incompressible one above. Same
//! rules as above: no clamp, no damping, no relaxation; positions and J as
//! declared above.
//!
//! - Compressible column at rest: the column at rest's tank, 1 cm cells,
//!   1 g, `c = 10 m/s`, 5 s. Pressure within half a cell of head of
//!   `g (h - y)` with `h` the surface height of that frame, from t = 1 s
//!   (amended: a column released uncompressed rings while it settles, and
//!   no initial state met both this criterion at every frame and the
//!   surface one below); the mean J of the particles in each 5-cell band
//!   within 0.005 of `1 - g (h - y) / c^2` over the last second; the
//!   surface, averaged over the last second, lowered from its start by
//!   `g h^2 / (2 c^2)` within 0.15 cell.
//! - Sound speed: a closed tube 200 cells long and 8 high, full, no
//!   gravity, `c = 10 m/s`, the 10 cells at one end starting at J = 0.99,
//!   substeps of `dx / c`. The pressure peak reaches the cell 150 from that
//!   end at `(150 - 5) dx / c` within 5 percent.
//! - Real water: the dam break and the drop into a pool with `K = 2.2e9
//!   Pa`, the same criteria as there, and substeps per frame at most 1.1
//!   times those of the incompressible runs: the real bulk modulus must not
//!   bring back an acoustic time step.
//! - The incompressible scenes keep passing unchanged.
//!
//! Failures counted as above.
//!
//! # Elastic solid on the staggered grid, criteria first
//!
//! Stomakhin et al. 2014's split (their eqs. 7, 8, 13): the shear part of
//! the fixed corotated energy taken on the isochoric deformation,
//! `mu |J^(-1/d) F - R|^2`, gives a deviatoric Kirchhoff stress applied as
//! forces on the faces, explicitly; the volume part `lambda / 2 (J - 1)^2`
//! is the compressible projection above with `c^2 = lambda / rho`. Plane
//! strain, `lambda` and `mu` from `E` and `nu`. Same rules: no clamp, no
//! damping, no relaxation. Since the free-body change below, the volume
//! part acts through the particles instead, with the shear part.
//!
//! - Confined elastic column at rest: the compressible column's tank
//!   filled with the solid, `nu = 0.3`, `E` such that `(lambda + mu) / rho
//!   = (10 m/s)^2`, the uniaxial-strain stiffness of this split in 2D;
//!   starting unstressed, 5 s. Over the last second, the mean J of each
//!   5-cell band within 0.005 of `1 - g (h - y) rho / (lambda + mu)`; no
//!   particle out of the tank; no NaN. Amended with the material matrix
//!   below: that stiffness was the split's own, with `lambda` as its volume
//!   modulus; real plane-strain elasticity has `lambda + 2 mu`, which the
//!   volume part now gives (`Scene::with_lame`), and the expected J is `1 -
//!   g (h - y) rho / (lambda + 2 mu)`, read from the scene's own
//!   coefficients.
//! - Plane shear wave (amended: a free strip 16 cells high timed by the
//!   peak of its velocity measured no speed. A free surface along the strip
//!   forces `sigma_xy = 0`, which no plane shear wave satisfies, so the
//!   strip bends instead, and the peak at 150 cells was 1 to 6 percent of
//!   the kick and moved from 14 percent early to 50 percent late with the
//!   height). Now a free block 100 cells long and 208 high, no gravity,
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
//! - Elastic block dropped: a 10 by 10 cell block, `nu = 0.3`, `c_s = 5
//!   m/s`, released at rest with its base 20 cells above the floor of the
//!   dam break's closed tank, 1 g, 2 s. Kinetic, potential and elastic
//!   energy never above the start by more than 1 percent; no particle out;
//!   no NaN; at 2 s the block's radius of gyration about its centre within
//!   5 percent of its start (it springs back).
//! - The incompressible and compressible scenes keep passing unchanged.
//!
//! Failures counted as above.
//!
//! Amended: a particle found beyond a wall after its step is moved to its
//! mirror point (`x -> 2 w - x`), the map the walls' image construction
//! implies (`extrapolate::constrain_to_solids`; Zhu and Bridson 4.2.5 push
//! particles back out of solids). Why: a wall is an invariant line of the
//! mirrored flow, where the normal velocity is zero, and where the liquid
//! leaves the wall it is an unstable one. The image continues the same
//! line, `u = a (x - w)`, behind the wall, so a particle put an ulp beyond
//! it by rounding is carried away as `exp(a t)`. Measured: a dam-break
//! particle gliding on the left wall plane went 4 ulps beyond it with the
//! f32 noise of the gathered normal velocity (1.6e-5 cells/s), then 4.45
//! cells into the wall with `a = 26 1/s`. So that the reflection cannot
//! hide a real crossing, every reflection is counted with its depth before
//! it, and every scene with walls requires the deepest at most 4 ulps of
//! the wall's coordinate: one rounding of the position's sum is half an
//! ulp, and the noise step, `1.6e-5 x 3 ms`, a tenth of one, with a
//! declared factor of 4 over that.
//!
//! Amended again, for precision, speed and every material: that budget
//! left out the pressure solve's own tolerance. The solve stops at a
//! residual divergence `r` (1/s), which leaves a face velocity off by about
//! `r dx`, a step of `dt dx r` towards a wall. Measured with MIC(0) on the
//! resting column at 2.5 g: `r` near 4e-4 1/s at substeps of 16.7 ms gives
//! 6.8e-6 cell, 14 ulps at the floor; the crossing was 13.5. The multigrid
//! cycle overshoots the tolerance and had hidden the term. Each
//! reflection's depth is now held to `4 ulp + 4 dt dx r`, `r` the residual
//! that substep's solve actually reached, the same declared factor of 4
//! over both terms. A crossing at a physical velocity stays orders of
//! magnitude above it.
//!
//! # The material matrix, criteria first
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
//!   = sqrt((lambda + 2 mu) / rho)` (d'Alembert, as the plane shear wave).
//!   The block is high enough that its free edges' disturbance, at most
//!   `c_p`, arrives after the window.
//! - S wave, for `mu > 0`: the plane shear wave with the row's
//!   coefficients, the height from the row's own `c_p / c_s`; `c_s =
//!   sqrt(mu / rho)`.
//! - Every scene: no NaN, no particle beyond a wall past its budget, and
//!   its wall time per simulated second against real time, reported.
//!
//! Added, criteria first: the scenes at the substep each material runs at
//! in a game, where its cost against real time is the verdict.
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
//!   0`, the volume within 2 percent (the dam break's bound).
//! - Both: the wall time per simulated second, reported against real time
//!   (1000 ms); no criterion yet, as the explicit shear bounds the substep
//!   by the shear wave (by the P wave since the free-body change below), so
//!   the stiffer the solid the slower.
//!
//! Failures counted as above.
//!
//! # A free body keeps its momentum, criteria first
//!
//! A body with no external force keeps its linear and angular momentum,
//! whatever its material. Measured before this change: a free block of the
//! very soft clay, vibrating, accelerated its centre to 0.27 m/s in 1 s
//! at 1 cm cells and to 0.45 m/s, spinning at 6.3 rad/s, at 0.5 cm, the
//! free surface's pressure giving it a net force (tanks hid it: their walls
//! take any net force).
//!
//! - Free body: every row, a 10 by 10 cell block at the centre of a 72
//!   cell tank, no gravity, never reaching a wall, started with the pure
//!   strain rate `v = s (y - c_y, x - c_x)` whose largest particle speed is
//!   0.5 m/s, the same for every row, at 1 and 0.5 cm cells, 1 s. (Set
//!   before any run: a strain over a sound crossing gave water 37 m/s.) The
//!   centre's velocity stays within 1 percent of the start's largest
//!   particle speed, and the angular momentum about the centre within 1
//!   percent of `I s` (the start's inertia times the strain rate), at every
//!   frame; no NaN; energy never above the start by more than 1 percent.
//! - Every scene above keeps passing.
//!
//! Failures counted as above.
//!
//! Amended: a row with no shear stiffness (`mu = 0`) cannot hold a strain
//! rate, so its block stretched into the walls within the second, whose
//! forces then changed its momentum (the decomposition below did not
//! telescope to zero for water alone). Such a row starts in pure
//! translation instead: the block 4 cells from two walls, moving at 0.5
//! m/s along the tank's diagonal, which keeps it 14 cells from the far
//! walls after 1 s. The bounds are unchanged, the centre's velocity read
//! against its start, the angular momentum's scale `I s` with `s` the
//! largest start speed over the half diagonal.
//!
//! Measured on the way (probes and runs, every change reverted but the
//! last):
//! - The projection changes the faces' momentum by exactly `sum_f (m_f -
//!   rho V_f) du_f`, `V_f` the volume the kinematic update assumes of face
//!   `f`. On the free very soft clay the terms came from the bulk's sampled
//!   mass, from the surface's kernel-spread mass against the level set's
//!   `theta`, and from the faces past the surface that take the continued
//!   pressure.
//! - The surface's `theta` from the face mass and no continued pressure:
//!   the clays' drift unchanged, the column at rest broke.
//! - The solved pressure applied as each particle's stress: the clay's
//!   centre held, water lost about 10 percent of its volume, the solved
//!   pressure belonging to the MAC operator, not to that one.
//! - Newton's law on every face, `du = dt F / m_f`. A resting liquid
//!   convected, as Ding, Shinar and Schroeder (MAC APIC, JCP 408, 2020,
//!   2.4.3) report of grid masses in the projection, "an initially
//!   stationary pool of water would develop currents in it", and the clays
//!   blew up on faces of vanishing mass.
//! - The power weights of Qu, Li, de Goes and Jiang (The Power
//!   Particle-In-Cell Method, TOG 41(4) 118, 2022), face masses exact to
//!   the transport plan's tolerance. Water clean, the clay still drifted: a
//!   compressed solid's density really varies, `rho0 / J`, which a
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
//! condition, which the projection's variational form held before. The
//! free body then passes, the centre at 0.0000 of the start speed and the
//! angular momentum within 0.0005 of `I s` (0.0004 before solids were
//! seeded without the surface pull, `seed_body`), and every scene above
//! passes with the plane shear wave's substep amended below. Lost against
//! the split: the clays' P wave arrives 4.2 percent late, 1.1 with the
//! split.
//!
//! Measured since, not a criterion: the free body's energy bound catches a
//! rise only, and the explicit path loses most of the free block's energy
//! within the second (95 to 99.95 percent, the conserving step below). The
//! conserving step has its own gate with the bound either way
//! (`gate_conserving_free_vibration`).
//!
//! Amended with this change: the plane shear wave fixed its substep at
//! half a cell per shear wave crossing, which a solid carrying its P wave
//! crosses at 0.94 cell (limit 0.577), and blew up. Its substep now follows
//! the faster wave, as the matrix's plane waves set it; the arrival is 3.3
//! percent late (3.7 with the split).
//!
//! # A solid keeps its energy at large substeps, criteria first
//!
//! Real time for a solid needs substeps far longer than the explicit P
//! wave limit, 273 to 529 per frame on the matrix's clays at 1 cm, and no
//! loss of precision means no numerical damping. Measured before this
//! change: the free body's clay block ends its second with 95 to 99.95
//! percent less energy, all of it taken by the method
//! (`probe_free_body_frames`: 99.7 and 99.95 percent at 1 cm for the very
//! soft and medium clays, 95 at 0.5 cm for the very soft one). Love and
//! Sulsky (An Unconditionally Stable, Energy-Momentum Consistent
//! Implementation of the Material-Point Method, HPC@UNM technical report
//! 2004-002, the report of CMAME 195, 2006) prove both the cause and the
//! remedy for a hyperelastic solid. A lumped mass loses
//! `1/2 dv . (M_lumped - M) dv` each step (their eq. 68), and a transfer
//! that overwrites the particles' velocity loses the velocity it cannot
//! carry (their eq. 49).
//! With the consistent mass `M_AB = sum_p m_p N_A N_B` (eq. 24), a
//! particle update that adds only the grid's change (eq. 32), and an
//! implicit midpoint step on the grid whose stress is the discrete
//! gradient of the stored energy (eqs. 29, 54, 56), energy and both
//! momenta are kept exactly, at any substep.
//!
//! - Free vibration: every solid row, the free body's block with its
//!   strain start, no gravity, 1 s, at 1 and 0.5 cm, at three substeps: the
//!   explicit limit `0.5 dx / c_p`, four times it, and one frame (1/60 s,
//!   or the travel limit if shorter). The total energy, kinetic and
//!   stored, within 1 percent of the start at every frame; the free body's
//!   momentum bounds; no NaN.
//! - Bounce: the matrix's drop for every solid row, 2 s, at one frame's
//!   substep. The total energy, kinetic, gravitational and stored, within 1
//!   percent of the start at every frame; the radius of gyration back
//!   within 5 percent at the end (the drop's bound); nothing out of the
//!   tank.
//! - Rest: the matrix's rest for every solid row at one frame's substep,
//!   within its bounds.
//! - Waves: the matrix's P and S waves keep their bounds at the substeps
//!   the matrix sets. Past one crossing per substep a step cannot carry the
//!   wave, so there only the energy bound above holds.
//! - Every scene above keeps passing.
//! - Cost: the wall time per simulated second of the matrix's drop for
//!   each solid row, against real time and against the explicit run before
//!   this change (16.5 and 32.0 s per simulated second at 1 cm, release).
//!   Reported, not a criterion: reaching real time is the next stage, kept
//!   apart so that a loss of energy and a loss of speed are never confused.
//!
//! Amendments after the bounce first failed (both clays NaN at their
//! impact), each measured before the next run (`gate_conserving_bounce`):
//! - Walls: the tank holds a face on or beyond a wall's plane while the
//!   force the wall exerts on it is compressive and lets it go once it
//!   would pull, the contact condition itself (no penetration, no pull;
//!   Bardenhagen et al. 2001, `total_lagrangian::step`). Held faces keep the
//!   step's energy balance exactly: the drop's measured loss equalled the
//!   mass stabilisation's own cost to five digits.
//! - Mass stabilisation: the extended B-splines (`conserving`'s grid) in
//!   place of the ghost penalty, which took 26 to 46 percent of the drop's
//!   energy through its impacts (measured, `gamma` 0.05 down to 0.002); the
//!   extended basis adds nothing to the mass and took none.
//! - The energy bound is two-sided, as written: `|E / E_0 - 1|` within 1
//!   percent at every frame.
//!
//! Measured since, not an amendment: at one frame per substep the drop
//! keeps its energy to 1.5e-6 and bounces when the block reaches the floor
//! at about 0.66 m/s (1.1 cells per frame at 1 cm), and breaks at its
//! first contact at 2 m/s (3.3 cells per frame) for every stiffness from 3
//! MPa to 10 GPa (`probe_conserving_spread`): a step that carries the block
//! further than about a cell into its contact has no admissible solution
//! on the grid. The bounce at the drop's 2 m/s is therefore open at one
//! frame per substep; the contact refinement (`Scene::contact_substep`), a
//! declared stepping stone, resolves it at the P wave step.
//!
//! Failures counted as above.

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
use crate::matter::kinematics::volume_ratio::advance_log_volume_ratio;
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
    /// `c^2` in cells^2/s^2 for the compressible projection;
    /// `None` is the incompressible limit.
    sound_speed2: Option<f32>,
    /// A substep length that replaces the travel limit (the sound-speed
    /// tube).
    fixed_dt: Option<f32>,
    /// `mu / rho` in cells^2/s^2 for an elastic solid; `None`
    /// for a liquid.
    shear: Option<f32>,
    /// A solid moves by the energy-momentum conserving implicit step
    /// (`conserving`) instead of the explicit one.
    conserving: Option<crate::spacetime::total_lagrangian::ConservingSettings>,
    /// One material per particle for the conserving step, in place of the
    /// scene's one (a probe's several bodies).
    solids: Option<Vec<crate::spacetime::total_lagrangian::Elastic>>,
    /// The conserving step's contact refinement, a declared stepping stone
    /// (`contact_substep`): its substep in P wave steps, `0.5 dx / c_p`;
    /// set by a probe.
    refine_contact: Option<f32>,
    /// Each particle's deformation gradient, for the shear stress.
    deformation: Vec<Mat2>,
    /// The conserving step's reference configuration (taken at its first
    /// step) and its grid, kept across steps (`conserving_substep`).
    lagrangian: Option<(
        Vec<glam::DVec2>,
        Option<crate::spacetime::total_lagrangian::LagrangianCache>,
    )>,
    /// The first of the bodies' contact points (`with_contact_outline`):
    /// only they meet the walls. `None`: every particle does.
    contact_from: Option<usize>,
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
    /// Cell whose pressure every substep records, with the time (the
    /// sound-speed tube).
    watch: Option<usize>,
    watch_history: Vec<(f32, f32)>,
    /// Particles within one cell of this x and between these two heights:
    /// their mean vertical velocity every substep (the plane shear wave).
    watch_x: Option<(f32, f32, f32)>,
    /// The watch reads the x velocity instead of y (a P wave along x).
    watch_longitudinal: bool,
    watch_x_history: Vec<(f32, f32)>,
    /// Where and when the deepest reflection happened, and the particle's
    /// position before the step that took it there.
    deepest_reflection_at: Option<(f32, Vec2)>,
    /// The conserving step's implicit steps: Newton and GMRES iterations
    /// summed, the most GMRES iterations in one step, the worst final residual
    /// and the steps that did not converge.
    conserving: (u64, u64, u32, f64, u32),
    /// Conserving steps rejected and halved (step-size control).
    rejected: u32,
    /// The stabilised mass's energy loss summed over the conserving steps.
    stabilisation_loss: f64,
    /// The worst relative residual of the conserving steps' projection.
    worst_projection: f64,
    /// The last conserving step's energy ledger and work
    /// (`total_lagrangian::StepReport`).
    ledger: (f64, f64, f64),
    work: (f64, f64),
    /// The most particles the conserving step held on a wall's plane at
    /// once.
    contacts: usize,
    /// The conserving steps' wall time by phase (`total_lagrangian::StepReport`)
    /// and how many steps.
    phases: [f64; 5],
    steps: u32,
    /// The conserving steps' solves summed (the walls' passes).
    passes: u64,
    /// The local error estimate of Zienkiewicz and Xie (1991), as Vetter,
    /// Wittel, Stoop and Herrmann write it for the Newmark family (arXiv
    /// 1111.5128, eq. 64), `|beta - 1/6| dt^2 |a_{n+1} - a_n|_inf / u_ref`,
    /// here with the midpoint's `beta = 1/4` and each step's average
    /// acceleration `(v_{n+1} - v_n) / dt` per particle in place of their
    /// predicted one, an adaptation of this engine's. Measured when
    /// `estimator_reference` (`u_ref`, cells) is positive: the last step's
    /// accelerations and the largest estimate since it was last reset.
    estimator_reference: f32,
    last_acceleration: Vec<Vec2>,
    largest_estimate: f32,
    /// The bodies whose mass conditioning each conserving step measures
    /// before it moves (`total_lagrangian::mass_conditioning`): the union's and
    /// each body's own, the smallest since last reset; empty, none.
    conditioning_bodies: Vec<std::ops::Range<usize>>,
    conditioning: (f64, Vec<f64>),
    /// Below this conditioning a body's basis is dissected
    /// (`total_lagrangian::basis_report`), the worst since last reset kept with
    /// its conditioning; zero, never.
    basis_threshold: f64,
    basis_worst: Option<(f64, String)>,
    /// Since last reset: the largest grid velocity of a projection over the
    /// fastest particle's speed, and Newton's worst start
    /// (`total_lagrangian::StepReport`).
    grid_speed_ratio: f64,
    start_residual: f64,
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
            conserving: None,
            solids: None,
            refine_contact: None,
            deformation: vec![Mat2::IDENTITY; n],
            lagrangian: None,
            contact_from: None,
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
    /// whole stress, shear and volume, through its particles (module doc:
    /// a solid is not projected);
    /// with `mu = 0` a liquid, its volume in the compressible projection.
    /// Adds contact points of no mass and no volume on the bodies'
    /// surface, which alone meet the walls in the conserving step. A
    /// contact holds a body's surface, as the node-to-surface contact of
    /// Laursen and Chawla, IJNME 40 (1997), sec. 3, holds its slave
    /// surface's nodes: a particle inside the body held where it crossed a
    /// wall would be held there while its neighbours move (measured: the
    /// very soft clay's second, corner-first impact blew up with every
    /// particle held).
    fn with_contact_outline(mut self, points: Vec<Vec2>) -> Self {
        self.contact_from = Some(self.x.len());
        for q in points {
            self.x.push(q);
            self.v.push(Vec2::ZERO);
            self.c.push(Mat2::ZERO);
            self.mass.push(0.0);
            self.v0.push(0.0);
            self.log_j.push(0.0);
            self.deformation.push(Mat2::IDENTITY);
        }
        self
    }

    fn with_lame(mut self, lambda_over_rho: f32, mu_over_rho: f32) -> Self {
        let cells2 = 1.0 / (self.cell_m * self.cell_m);
        self.shear = (mu_over_rho > 0.0).then_some(mu_over_rho * cells2);
        // The shear part acts on the isochoric deformation only, so all
        // the volume stiffness is the volume part's (the particles' for a
        // solid, the projection's for a liquid): the
        // plane-strain bulk modulus `kappa = lambda + mu`. With it, `2 mu dev(e) + kappa tr(e)
        // I = 2 mu e + lambda tr(e) I`, Lame's law; with `lambda` alone, as
        // Stomakhin et al.'s split writes the volume energy, the P wave ran
        // at `sqrt((lambda + mu) / rho)`: 6.0 percent late on the clays of
        // the matrix (`nu = 0.45`), against water's 1.0.
        self.sound_speed2 = Some((lambda_over_rho + mu_over_rho) * cells2);
        // Probe switch: a solid moves by the conserving implicit step, at
        // one frame per substep (`frame`) or at the explicit P wave limit
        // (`p`), when `EMERGE_PROBE_CONSERVING` names one.
        if let (Some(mu), Ok(step)) = (self.shear, std::env::var("EMERGE_PROBE_CONSERVING")) {
            let defaults = crate::spacetime::total_lagrangian::ConservingSettings::default();
            self.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
                occupation: std::env::var("EMERGE_PROBE_OCCUPATION")
                    .ok()
                    .and_then(|c| c.parse().ok())
                    .or(defaults.occupation),
                ..defaults
            });
            let c_p = (self.sound_speed2.unwrap_or(0.0) + mu).sqrt();
            let step = Substep::parse(&step).unwrap_or(Substep::Frame);
            self.fixed_dt = Some(step.length(self.layout.dx, c_p));
        }
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
        if let (Some(mu), Some(settings)) = (self.shear, self.conserving) {
            self.conserving_substep(mu, settings, dt, frame);
            return;
        }
        let mut clock = Instant::now();
        let mut lap = |probe: &mut Probe, k: usize| {
            probe.phase_us[k] += clock.elapsed().as_micros();
            clock = Instant::now();
        };
        let (mut vel, face_mass) =
            particles_to_faces(&self.layout, &self.x, &self.v, &self.c, &self.mass);
        if let Some(mu) = self.shear {
            // The whole elastic stress on the particles (module doc, a solid is
            // not projected): the shear part on the isochoric deformation and
            // the volume part `K J (J - 1) I`, `J = det F`, with `K` the
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
            // A solid is not projected (module doc). The wall
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
        let contact = self.contact_substep();
        while left > 1e-7 {
            let speed = self.v.iter().fold(0.0f32, |m, v| m.max(v.length()));
            // Half a cell per crossing of the fastest wave a solid's
            // particles carry, the P wave `c_p^2 = (K + mu) / rho`, under
            // the explicit limit `2 / sqrt(12) = 0.577` the transfer's
            // stiffest mode allows (`transfer::the_stiffest_mode_stays_
            // bounded_...`).
            let shear_limit = match (self.shear, self.conserving) {
                (Some(mu), None) => {
                    0.5 * self.layout.dx / (mu + self.sound_speed2.unwrap_or(0.0)).sqrt()
                }
                _ => f32::INFINITY,
            };
            let limit = contact.or(self.fixed_dt).unwrap_or_else(|| {
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

    /// The conserving step's contact refinement, a stepping stone toward
    /// resolving impacts only where they happen, declared as such: a frame
    /// during which any particle is, or would come, within reach of a wall
    /// moves at the P wave step `0.5 dx / c_p` instead of one step. An impact
    /// lasts about `2 L / c_p` (Love and Sulsky's contact time, their section
    /// 4.1), far under a frame, and the elastodynamic contact problem keeps its
    /// energy through it (Lebeau and Schatzman 1984, as Dabaghi et al., arXiv
    /// 1601.00778, theorem 2.1, state it), which one frame-long step cannot
    /// reproduce (measured: a body's centre 10 mm off the resolved one after a
    /// 0.66 m/s impact, and 76 percent of the energy lost at a 2 m/s one with
    /// the contact held at the step's end). The reach is the quadratic
    /// B-spline's support, 1.5 cells, plus the half cell a particle stands for
    /// (the region BFEMP, arXiv 2108.03349, notes a particle represents); the
    /// prediction is each particle's own trajectory over the frame with its
    /// velocity and the weight, the linear motion continuous collision
    /// detection assumes over a step. Its cost: a body resting on a wall stays
    /// refined. `None` keeps the scene's own substep.
    fn contact_substep(&self) -> Option<f32> {
        let multiple = self.refine_contact?;
        let (mu, _) = (self.shear?, self.conserving?);
        let c_p = (self.sound_speed2.unwrap_or(0.0) + mu).sqrt();
        let (lo, hi) = self.tank;
        let reach = 2.0 * self.layout.dx;
        let distance = |q: Vec2| (q.x - lo.x).min(hi.x - q.x).min(q.y - lo.y).min(hi.y - q.y);
        let near = self.x.iter().zip(&self.v).any(|(&q, &v)| {
            let ahead = q + v * FRAME + 0.5 * self.gravity * FRAME * FRAME;
            distance(q).min(distance(ahead)) < reach
        });
        near.then_some(multiple * 0.5 * self.layout.dx / c_p)
    }

    /// One implicit, energy-momentum conserving step of the solid: the
    /// total Lagrangian step (`total_lagrangian::step_total_lagrangian_contact`),
    /// its grid on the configuration of its first step, the tank's walls
    /// held by the contact condition in velocity (route C), frictionless.
    fn conserving_substep(
        &mut self,
        mu: f32,
        settings: crate::spacetime::total_lagrangian::ConservingSettings,
        dt: f32,
        frame: &mut Frame,
    ) {
        use glam::{DMat2, DVec2};
        let n = self.x.len();
        let mut x: Vec<DVec2> = self.x.iter().map(|q| q.as_dvec2()).collect();
        let mut v: Vec<DVec2> = self.v.iter().map(|q| q.as_dvec2()).collect();
        let mut f: Vec<DMat2> = self.deformation.iter().map(|m| m.as_dmat2()).collect();
        let mass: Vec<f64> = self.mass.iter().map(|&m| f64::from(m)).collect();
        let v0: Vec<f64> = self.v0.iter().map(|&m| f64::from(m)).collect();
        let (lo, hi) = (self.tank.0.as_dvec2(), self.tank.1.as_dvec2());
        // The probes' grid reports see no wall: walls are contacts now.
        let held = |_: usize, _: usize, _: usize| -> Option<f64> { None };
        let solids = self.solids.clone().unwrap_or_else(|| {
            vec![
                crate::spacetime::total_lagrangian::Elastic::isochoric(
                    f64::from(mu),
                    f64::from(self.sound_speed2.unwrap_or(0.0)),
                );
                n
            ]
        });
        if !self.probe.conditioning_bodies.is_empty() {
            let volume: Vec<f64> = (0..n).map(|p| v0[p] * f[p].determinant()).collect();
            let at = |r: std::ops::Range<usize>| {
                crate::spacetime::total_lagrangian::mass_conditioning(
                    &self.layout,
                    &x[r.clone()],
                    (&mass[r.clone()], &volume[r]),
                    &held,
                    (settings.occupation, settings.epsilon),
                )
            };
            let union = at(0..n);
            let each: Vec<f64> = self
                .probe
                .conditioning_bodies
                .iter()
                .map(|r| at(r.clone()))
                .collect();
            // The worst body below the threshold, dissected.
            let threshold = self.probe.basis_threshold;
            let worst = each
                .iter()
                .enumerate()
                .filter(|(_, c)| **c < threshold)
                .min_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, &c)| (i, c));
            if let Some((i, c)) = worst
                && self.probe.basis_worst.as_ref().is_none_or(|(w, _)| c < *w)
            {
                let r = self.probe.conditioning_bodies[i].clone();
                let report = crate::spacetime::total_lagrangian::basis_report(
                    &self.layout,
                    &x[r.clone()],
                    (&mass[r.clone()], &volume[r]),
                    &held,
                    (settings.occupation, settings.epsilon),
                );
                self.probe.basis_worst = Some((c, format!("body {i}: {report}")));
            }
            let (low, lows) = &mut self.probe.conditioning;
            *low = low.min(union);
            if lows.len() != each.len() {
                *lows = vec![f64::INFINITY; each.len()];
            }
            for (l, e) in lows.iter_mut().zip(each) {
                *l = l.min(e);
            }
        }
        let fastest = v.iter().fold(0.0f64, |m, q| m.max(q.length()));
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut v,
            f: &mut f,
            mass: &mass,
            v0: &v0,
        };
        let plane =
            |axis, at, inward| crate::spacetime::total_lagrangian::Plane { axis, at, inward };
        let planes = [
            plane(0, lo.x, 1.0),
            plane(0, hi.x, -1.0),
            plane(1, lo.y, 1.0),
            plane(1, hi.y, -1.0),
        ];
        // The penalty, the Uzawa iteration's parameter: the P wave's
        // stiffness (factor 1, `block_slide_run`'s measured choice).
        let from = self.contact_from.unwrap_or(0);
        let penalty: Vec<f64> = solids
            .iter()
            .enumerate()
            .map(|(p, s)| if p >= from { s.bulk + s.mu } else { 0.0 })
            .collect();
        if self.lagrangian.as_ref().is_none_or(|(r, _)| r.len() != n) {
            self.lagrangian = Some((self.x.iter().map(|q| q.as_dvec2()).collect(), None));
        }
        let (reference, cache) = self.lagrangian.as_mut().expect("set above");
        let report = crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
            &self.layout,
            particles,
            reference,
            &solids,
            (
                f64::from(dt),
                crate::spacetime::total_lagrangian::ContactMode::Velocity,
                self.gravity.as_dvec2(),
            ),
            (
                crate::spacetime::total_lagrangian::ContactSet {
                    planes: &planes,
                    pairs: &[],
                    outlines: &[],
                    corners: &[],
                    friction: 0.0,
                },
                &penalty,
                &settings,
            ),
            cache,
        );
        if std::env::var_os("EMERGE_PROBE_SCENE_TRACE").is_some() {
            println!(
                "  step {}: lowest {:.3} (floor {:.3}), Newton {} residual {:.2e} (start {:.2e}) converged {}, contacts {}, push {:.3e}, passes {}, fastest {:.3e}",
                self.probe.steps,
                self.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min),
                lo.y,
                report.newton,
                report.residual,
                report.start_residual,
                report.converged,
                report.contacts,
                report.push,
                report.passes,
                fastest
            );
        }
        self.probe.contacts = self.probe.contacts.max(report.contacts);
        for (total, part) in self.probe.phases.iter_mut().zip(report.phases) {
            *total += part;
        }
        self.probe.steps += 1;
        self.probe.passes += u64::from(report.passes);
        self.probe.stabilisation_loss += report.stabilisation_loss;
        self.probe.grid_speed_ratio = self
            .probe
            .grid_speed_ratio
            .max(report.grid_speed / fastest.max(f64::MIN_POSITIVE));
        self.probe.start_residual = self.probe.start_residual.max(report.start_residual);
        self.probe.worst_projection = self.probe.worst_projection.max(report.projection);
        self.probe.ledger = report.ledger;
        self.probe.work = report.work;
        let stats = &mut self.probe.conserving;
        stats.0 += u64::from(report.newton);
        stats.1 += u64::from(report.gmres);
        stats.2 = stats.2.max(report.gmres);
        stats.3 = stats.3.max(report.residual);
        stats.4 += u32::from(!report.converged);
        if self.probe.estimator_reference > 0.0 {
            let acceleration: Vec<Vec2> =
                (0..n).map(|p| (v[p].as_vec2() - self.v[p]) / dt).collect();
            if self.probe.last_acceleration.len() == n {
                let jump = acceleration
                    .iter()
                    .zip(&self.probe.last_acceleration)
                    .fold(0.0f32, |m, (a, b)| m.max((*a - *b).abs().max_element()));
                let estimate = (0.25 - 1.0 / 6.0) * dt * dt * jump / self.probe.estimator_reference;
                self.probe.largest_estimate = self.probe.largest_estimate.max(estimate);
            }
            self.probe.last_acceleration = acceleration;
        }
        for p in 0..n {
            let before = self.x[p];
            self.x[p] = x[p].as_vec2();
            self.v[p] = v[p].as_vec2();
            self.c[p] = Mat2::ZERO;
            self.deformation[p] = f[p].as_mat2();
            frame.largest_travel = frame.largest_travel.max((self.x[p] - before).length());
        }
        self.probe.solve_drift = 0.0;
        // Counted, not moved: the contact holds the walls, and a moved
        // particle would leave the grid's reference configuration (`x = X +
        // u(X)`); the depth beyond a wall stays a measured criterion.
        for p in 0..n {
            self.beyond_walls(p, false);
        }
        self.time += dt;
        frame.substeps += 1;
        frame.phi = self.phi();
    }

    /// Moves particle `p`, if beyond a wall, to its mirror point across it,
    /// and counts it (the reflection amendment, module doc).
    fn reflect_off_walls(&mut self, p: usize) {
        self.beyond_walls(p, true);
    }

    /// Counts particle `p` if it is beyond a wall, by how far, and with
    /// `mirror` moves it to its mirror point across it.
    fn beyond_walls(&mut self, p: usize, mirror: bool) {
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
            if mirror {
                self.x[p] = q;
            }
            self.probe.reflections += 1;
            if deepest > self.probe.deepest_reflection_ulps {
                self.probe.deepest_reflection_at = Some((self.time, self.x[p]));
            }
            self.probe.deepest_reflection_ulps = self.probe.deepest_reflection_ulps.max(deepest);
            self.probe.worst_reflection_ratio = self.probe.worst_reflection_ratio.max(ratio);
        }
    }

    /// The reflection criterion (module doc).
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

    /// In the tank or on its walls: out means beyond a wall. A particle gliding
    /// along a wall approaches it geometrically (the normal velocity falls
    /// linearly to zero there, `substep` doc) and, in f32, ends on it: measured
    /// in the drop into a pool once the elastic solid's transfer was in, a
    /// particle reached `x = 4.0000005`, one ulp from the wall at 4, with `v_x
    /// = -1.6e-5` cells/s, sat exactly on the wall for one frame and went back
    /// in, never beyond it. The strict test counted it out.
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

/// `seed` for a body: a liquid's particles pulled to the particle spacing
/// from its free surface (Zhu and Bridson's technique, `seed`, which
/// smooths the surface their level set rebuilds), a solid's left where they
/// fall. A solid's particles are the quadrature points of its volume, and
/// the pull lays the two outer rows of a box onto one line (the particles a
/// quarter and three quarters of a spacing from a face both end half a
/// spacing from it), a degenerate quadrature (measured: at one frame per
/// substep Newton fails on it within four frames and a particle inverts).
fn seed_body(
    solid: bool,
    cells: (IVec2, IVec2),
    inside: impl Fn(Vec2) -> bool,
    surface: impl Fn(Vec2) -> (f32, Vec2),
    rng: &mut LcgRng,
) -> Vec<Vec2> {
    if solid {
        seed(cells, inside, |_| (f32::INFINITY, Vec2::Y), rng)
    } else {
        seed(cells, inside, surface, rng)
    }
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

/// The column at rest at `k` cells per centimetre and one gravity. Returns the
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
#[ignore = "projection gate, column at rest: long; run with --ignored --nocapture"]
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
#[ignore = "projection gate, droplet in free fall: run with --ignored --nocapture"]
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

/// The dam break and the drop into a pool, and real water with
/// `sound_speed` in m/s. Returns the mean
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
#[ignore = "projection gate, dam break: run with --ignored --nocapture"]
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
#[ignore = "projection gate, drop into a pool: run with --ignored --nocapture"]
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

/// The seeds of the dam break and the drop into a pool.
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
#[ignore = "compressible projection gate, column at rest: run with --ignored --nocapture"]
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
#[ignore = "compressible projection gate, sound speed: run with --ignored --nocapture"]
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
#[ignore = "compressible projection gate, real water: run with --ignored --nocapture"]
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

/// Diagnosis of the compressible projection: the sound-speed arrival at
/// substeps of `dx / c`, half and a quarter of it (a time-discretisation error
/// shrinks with the step; a wrong `c` does not), and the compressible column's
/// largest pressure error per half second. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_sound_speed_arrival() {
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
#[ignore = "staggered elastic solid gate, confined column at rest: run with --ignored --nocapture"]
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
    let x = seed_body(
        true,
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
#[ignore = "staggered elastic solid gate, plane shear wave: run with --ignored --nocapture"]
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
    let x = seed_body(
        true,
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
    // Amended with the free body (module doc): the solid's particles carry the
    // P wave now, so the explicit step follows it, as `plane_wave` sets it.
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
#[ignore = "staggered elastic solid gate, dropped block: run with --ignored --nocapture"]
fn gate_elastic_drop() {
    let mut report = Report::new();
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed_body(
        true,
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

/// Diagnosis of the dropped elastic block: its energy split into its
/// parts every frame. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_elastic_drop_energy_parts() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    // EMERGE_PROBE_SCENARIO: "drop" (the dropped block), "rest" (on the floor,
    // at rest), "spin" (no gravity, no contact, spinning at 2 rad/s).
    let scenario = std::env::var("EMERGE_PROBE_SCENARIO").unwrap_or_else(|_| "drop".into());
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
            let w = std::env::var("EMERGE_PROBE_OMEGA")
                .ok()
                .and_then(|v| v.parse::<f32>().ok())
                .unwrap_or(2.0);
            if std::env::var_os("EMERGE_PROBE_TRANSLATE").is_some() {
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

/// Diagnosis of the staggered elastic solid: one substep of a rigidly spinning
/// liquid block, no gravity, and the velocity gradient at an edge particle
/// after each stage. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_rigid_spin_one_substep() {
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

/// Diagnosis of the staggered elastic solid: the transfer round trip alone
/// (scatter, projection, gather, particles held still, no gravity) on a resting
/// column with small random velocities. Kinetic energy must never grow. A
/// probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_transfer_round_trip_energy() {
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

/// Diagnosis of the dropped elastic block, which, resting on the floor at
/// 1 s, went to NaN before 1.2 s. Substep by substep from
/// 0.95 s: the largest `|C|`, speed, `|F^ - R^|` and J range, until the
/// first value that is not finite. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_elastic_drop_first_nan() {
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

/// Diagnosis of the dropped elastic block: when do its particles lose
/// every neighbour within one cell (an
/// elastic body must never shed a piece)? A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_elastic_drop_detached() {
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

/// Diagnosis of the dropped elastic block: the mirror wall stands for
/// an image body, so a block hitting the floor must move exactly as one of
/// two identical blocks hitting each other head on, with no wall near.
/// No gravity; kinetic plus elastic energy and the top speed of each case
/// (`EMERGE_PROBE_WALL_CASE` = wall or pair). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_wall_against_image_pair() {
    let (nu, c_s) = (0.3f32, 5.0f32);
    let case = std::env::var("EMERGE_PROBE_WALL_CASE").unwrap_or_else(|_| "wall".into());
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
    if std::env::var_os("EMERGE_PROBE_TRACE").is_some() {
        let start: f32 = std::env::var("EMERGE_PROBE_TRACE")
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

/// Where the drop into a pool loses a particle through a wall, after the
/// staggered elastic solid's transfer changes. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
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

/// Where the dam break (and the real water one with
/// `EMERGE_PROBE_WATER`) loses particles after the ghost pressure: how far
/// beyond which wall, when, and how fast. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
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

/// Particle 478 of the real-water dam break, the first to pass through the left
/// wall with an earlier version of the staggered solid's transfer: its position
/// and the velocity it gathers every substep around the moment it enters the
/// wall. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
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

/// Where the projection's time goes on the dropped elastic block's explicit
/// control: each piece of `project` timed alone, 200 times, on the state at 0.5
/// s. A probe.
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

/// The rows (criteria: the material matrix).
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
    let x = seed_body(
        row.mu > 0.0,
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

/// The rest scene for one row (criteria: the material matrix). Returns the wall
/// time per simulated second, in ms.
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
    let x = seed_body(
        row.mu > 0.0,
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

/// The drop for one row (criteria: the material matrix). Returns the wall time
/// per simulated second, in ms.
fn drop_block(row: &MaterialRow, report: &mut Report) -> f32 {
    let label = format!("{}, dropped", row.name);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed_body(
        row.mu > 0.0,
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
    // the criteria do not bound (the very soft clay lost 42 percent in 1 s
    // seeded as a solid, `seed_body`, 57 with the surface pull,
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

/// The conserving step's bounce for one solid row (criteria and their
/// amendments, module doc): the matrix's drop, 2 s, one frame per substep, the
/// tank's walls held by contact (`Scene::conserving_substep`). Lift-off is reported, not judged. Returns the
/// wall time per simulated second, in ms.
fn conserving_bounce(row: &MaterialRow, report: &mut Report) -> f32 {
    let label = format!("{}, bounce", row.name);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed_body(
        true,
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
    // The block's surface, a contact point every half cell around it.
    let mut outline = Vec::new();
    let steps = (2.0 * (hi.x - lo.x)) as usize;
    for k in 0..steps {
        let t = k as f32 / steps as f32;
        outline.push(lo + t * Vec2::new(hi.x - lo.x, 0.0));
        outline.push(Vec2::new(hi.x, lo.y) + t * Vec2::new(0.0, hi.y - lo.y));
        outline.push(hi - t * Vec2::new(hi.x - lo.x, 0.0));
        outline.push(Vec2::new(lo.x, hi.y) - t * Vec2::new(0.0, hi.y - lo.y));
    }
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
        .with_lame(row.lambda / row.rho, row.mu / row.rho)
        .with_contact_outline(outline);
    scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
        ..Default::default()
    });
    scene.fixed_dt = Some(FRAME);
    // Over the particles with mass: the contact points carry none.
    let gyration = |scene: &Scene| {
        let (com, _) = scene.centre_of_mass();
        let total: f32 = scene.mass.iter().sum();
        (scene
            .x
            .iter()
            .zip(&scene.mass)
            .map(|(q, m)| m * (*q - com).length_squared())
            .sum::<f32>()
            / total)
            .sqrt()
    };
    let floor = tank.0.y;
    let contact_time = 2.0 * (hi.y - lo.y) * scene.cell_m / row.c_p();
    let (energy_start, gyration_start) = (row_energy(&scene), gyration(&scene));
    let seconds = 2.0f32;
    let (mut worst_energy, mut nan) = (0.0f64, false);
    let (mut touched, mut lifted): (Option<f32>, Option<(f32, f32)>) = (None, None);
    let wall_clock = Instant::now();
    for _ in 1..=(seconds / FRAME).round() as u32 {
        scene.frame();
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        let change = row_energy(&scene) / energy_start - 1.0;
        nan |= !change.is_finite();
        worst_energy = worst_energy.max(change.abs());
        let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
        if touched.is_none() && lowest <= floor + 1.0 {
            touched = Some(scene.time);
        }
        if let (Some(t0), None) = (touched, lifted)
            && scene.time >= t0 + 2.0 * contact_time
        {
            let (_, v) = scene.centre_of_mass();
            lifted = Some((lowest - floor, v.y * scene.cell_m));
        }
    }
    let ms = 1000.0 * wall_clock.elapsed().as_secs_f32() / seconds;
    let change = (gyration(&scene) / gyration_start - 1.0).abs();
    let s = scene.probe.conserving;
    println!(
        "{label}: worst energy change {worst_energy:.6} of the start, gyration change {change:.4}, lift-off {lifted:?} (cells above the floor, m/s; contact time {:.2} ms), Newton {} GMRES {} unconverged {} rejected {}, {ms:.0} ms per simulated s ({:.4}x real time)",
        1000.0 * contact_time,
        s.0,
        s.1,
        s.4,
        scene.probe.rejected,
        1000.0 / ms
    );
    report.check(!nan, || format!("{label}: NaN"));
    report.check(worst_energy <= 0.01, || {
        format!("{label}: energy changed {worst_energy:.6}")
    });
    report.check(change <= 0.05, || {
        format!("{label}: gyration changed {change:.4}")
    });
    scene.check_reflections(&label, report);
    ms
}

#[test]
#[ignore = "conserving step, bounce: run with --ignored --nocapture"]
fn gate_conserving_bounce() {
    let mut report = Report::new();
    for row in material_rows().iter().filter(|r| r.mu > 0.0) {
        conserving_bounce(row, &mut report);
    }
    report.finish();
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

/// Check of the matrix's drop: the very soft clay's drop kept its energy
/// to 1e-4. Does it reach the floor and bounce? Lowest particle, energy
/// and centre speed every tenth of a second. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_soft_clay_drop_trajectory() {
    let row = &material_rows()[std::env::var("EMERGE_PROBE_ROW")
        .ok()
        .and_then(|r| r.parse().ok())
        .unwrap_or(1)];
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut rng = LcgRng::new(17);
    let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
    let x = seed_body(
        true,
        (lo.as_ivec2(), hi.as_ivec2()),
        |_| true,
        |_| (f32::INFINITY, Vec2::Y),
        &mut rng,
    );
    let mut scene =
        Scene::new((72, 72), 0.01, 1.0, tank, x).with_lame(row.lambda / row.rho, row.mu / row.rho);
    let e0 = row_energy(&scene);
    let frames = std::env::var("EMERGE_PROBE_FRAMES")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(60u32);
    for f in 1..=frames {
        scene.frame();
        let every = if scene.conserving.is_some() { 1 } else { 6 };
        if f % every == 0 {
            let (_, v) = scene.centre_of_mass();
            let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
            let s = scene.probe.conserving;
            println!(
                "{} t={:.3}: lowest y {lowest:.3}, centre vy {:.2} m/s, energy {:+.5}, newton {} gmres {} worst residual {:.1e} unconverged {} reflections {} rejected {}, stabilisation took {:.5}",
                row.name,
                scene.time,
                v.y * scene.cell_m,
                row_energy(&scene) / e0 - 1.0,
                s.0,
                s.1,
                s.3,
                s.4,
                scene.probe.reflections,
                scene.probe.rejected,
                scene.probe.stabilisation_loss / e0
            );
        }
    }
}

/// The conserving step's bounce with the floor a body: the drop scene's clay
/// block falls on a slab of an elastic solid `EMERGE_PROBE_FLOOR` times stiffer
/// than the medium clay (default 1000, same density), 8 cells thick on the
/// tank's floor, all of it moved by the conserving step at one frame per
/// substep (`EMERGE_PROBE_CONSERVING=frame`, walls `held`). Energy every frame,
/// kinetic, weight and stored, against the block's drop energy onto the slab. A
/// probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_conserving_drop_on_slab() {
    use crate::spacetime::total_lagrangian::Elastic;
    let stiffer: f32 = std::env::var("EMERGE_PROBE_FLOOR")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(1000.0);
    let frames: u32 = std::env::var("EMERGE_PROBE_FRAMES")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(60);
    let rows = material_rows();
    let floor = &rows[2];
    for row in &rows[1..] {
        let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
        let mut rng = LcgRng::new(17);
        let top = 12.0f32;
        let slab = seed_body(
            true,
            (IVec2::new(4, 4), IVec2::new(68, top as i32)),
            |_| true,
            |q| (top - q.y, Vec2::Y),
            &mut rng,
        );
        let (lo, hi) = (Vec2::new(31.0, 24.0), Vec2::new(41.0, 34.0));
        let block = seed_body(
            true,
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
        let (n_slab, n_block) = (slab.len(), block.len());
        let x: Vec<Vec2> = slab.into_iter().chain(block).collect();
        let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x)
            .with_lame(row.lambda / row.rho, row.mu / row.rho);
        let cells2 = 1.0 / f64::from(scene.cell_m).powi(2);
        let elastic = |r: &MaterialRow, k: f32| {
            Elastic::isochoric(
                f64::from(k * r.mu / r.rho) * cells2,
                f64::from(k * (r.lambda + r.mu) / r.rho) * cells2,
            )
        };
        let solids: Vec<Elastic> = (0..n_slab)
            .map(|_| elastic(floor, stiffer))
            .chain((0..n_block).map(|_| elastic(row, 1.0)))
            .collect();
        scene.solids = Some(solids.clone());
        let energy = |scene: &Scene| -> (f64, f64) {
            let g = f64::from(scene.g());
            let mut parts = (0.0, 0.0);
            for (p, solid) in solids.iter().enumerate() {
                let m = f64::from(scene.mass[p]);
                let e = m
                    * (0.5 * f64::from(scene.v[p].length_squared()) + g * f64::from(scene.x[p].y))
                    + f64::from(scene.v0[p]) * solid.energy(scene.deformation[p].as_dmat2());
                if p < n_slab {
                    parts.0 += e;
                } else {
                    parts.1 += e;
                }
            }
            parts
        };
        let start = energy(&scene);
        let block_mass: f64 = scene.mass[n_slab..].iter().map(|&m| f64::from(m)).sum();
        let drop = block_mass * f64::from(scene.g()) * f64::from(lo.y - top);
        let clock = std::time::Instant::now();
        let mut worst = 0.0f64;
        for _ in 0..frames {
            scene.frame();
            let now = energy(&scene);
            let change = (now.0 + now.1 - start.0 - start.1) / drop;
            worst = worst.max(change.abs());
            let lowest = scene.x[n_slab..]
                .iter()
                .map(|q| q.y)
                .fold(f32::INFINITY, f32::min);
            let vy = scene.v[n_slab..].iter().map(|v| v.y).sum::<f32>() / n_block as f32;
            let s = scene.probe.conserving;
            println!(
                "{} on slab x{stiffer} t={:.3}: block lowest {lowest:.3}, block vy {:.3} m/s, energy change {change:+.5} of the drop (slab {:+.5}, block {:+.5}), newton {} gmres {} unconverged {} worst residual {:.1e} rejected {}",
                row.name,
                scene.time,
                vy * scene.cell_m,
                (now.0 - start.0) / drop,
                (now.1 - start.1) / drop,
                s.0,
                s.1,
                s.4,
                s.3,
                scene.probe.rejected
            );
        }
        println!(
            "{} on slab x{stiffer}: worst energy change {worst:.5} of the drop, {:.1} s wall for {:.2} s simulated",
            row.name,
            clock.elapsed().as_secs_f64(),
            scene.time
        );
    }
}

/// A conserving run's fixed substep: the explicit P wave limit `0.5 dx /
/// c_p`, four times it, or one frame.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Substep {
    PWave,
    FourPWaves,
    Frame,
}

impl Substep {
    /// From a probe's setting: `p`, `4p` or `frame`.
    fn parse(name: &str) -> Option<Self> {
        match name {
            "p" => Some(Self::PWave),
            "4p" => Some(Self::FourPWaves),
            "frame" => Some(Self::Frame),
            _ => None,
        }
    }

    /// Its length in seconds on cells of `dx` for a P wave of `c_p`.
    fn length(self, dx: f32, c_p: f32) -> f32 {
        match self {
            Self::PWave => 0.5 * dx / c_p,
            Self::FourPWaves => 2.0 * dx / c_p,
            Self::Frame => FRAME,
        }
    }
}

impl std::fmt::Display for Substep {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::PWave => "0.5 dx / c_p",
            Self::FourPWaves => "2 dx / c_p",
            Self::Frame => "one frame",
        })
    }
}

/// How a conserving run steps: always at one substep, or at one frame per
/// substep refined to a multiple of the P wave step while near a wall
/// (`Scene::contact_substep`).
#[derive(Clone, Copy, Debug)]
enum Stepping {
    Fixed(Substep),
    RefinedAtContact(f32),
}

impl std::fmt::Display for Stepping {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fixed(substep) => write!(f, "at a substep of {substep}"),
            Self::RefinedAtContact(multiple) => write!(
                f,
                "at one frame per substep, refined at contact to {multiple} P wave steps"
            ),
        }
    }
}

/// One configuration of the conserving step's spread
/// (`probe_conserving_spread`): a square elastic block of `size` cells at `k`
/// cells per centimetre, Young's modulus `e` (Pa), Poisson's ratio `nu`,
/// density `rho`, either vibrating freely (no gravity, the free body's shear
/// start with corner speed `speed`) or dropped (1 g, starting 2 cells above the
/// floor at `speed` downwards), moved at one frame per substep with the
/// extended B-splines and the walls holding only while they push. Prints the
/// energy, the convergence, lift-off and the cost; returns, per frame, the
/// energy's change, the kinetic energy over the start's total, the centre of
/// mass's height (m) and vertical speed (m/s).
#[derive(Clone, Copy)]
struct SpreadCase {
    drop: bool,
    e: f32,
    nu: f32,
    rho: f32,
    size: f32,
    speed: f32,
    k: usize,
    seconds: f32,
    stepping: Stepping,
    /// Prints the error estimate of each frame (`Probe::estimator_reference`,
    /// the block's side as `u_ref`).
    estimate: bool,
}

fn spread_case(case: SpreadCase) -> Vec<[f64; 4]> {
    let SpreadCase {
        drop,
        e,
        nu,
        rho,
        size,
        speed,
        k,
        seconds,
        stepping,
        estimate,
    } = case;
    let kf = k as f32;
    let n = 72 * k;
    let tank = (Vec2::splat(4.0 * kf), Vec2::splat(68.0 * kf));
    let mut rng = LcgRng::new(17);
    let side = size * kf;
    let centre = Vec2::new(
        36.0 * kf,
        if drop {
            4.0 * kf + 2.0 * kf + 0.5 * side
        } else {
            36.0 * kf
        },
    );
    let (lo, hi) = (
        centre - Vec2::splat(0.5 * side),
        centre + Vec2::splat(0.5 * side),
    );
    let x = seed_body(
        true,
        (lo.floor().as_ivec2(), hi.ceil().as_ivec2()),
        |q| q.cmpge(lo).all() && q.cmple(hi).all(),
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
    let g = if drop { 1.0 } else { 0.0 };
    let mut scene = Scene::new((n, n), 0.01 / kf, g, tank, x).with_elastic(e / rho, nu);
    // `EMERGE_PROBE_OCCUPATION=none` keeps the original B-splines.
    let occupation = std::env::var("EMERGE_PROBE_OCCUPATION").ok();
    scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
        occupation: match occupation.as_deref() {
            Some("none") => None,
            other => Some(other.and_then(|c| c.parse().ok()).unwrap_or(0.75)),
        },
        // `eps` of the effective mass (`ConservingSettings::epsilon`): 1 is
        // the lumped mass.
        epsilon: std::env::var("EMERGE_PROBE_EPSILON")
            .ok()
            .and_then(|c| c.parse().ok())
            .unwrap_or(0.0),
        ..Default::default()
    });
    let c_p = (scene.sound_speed2.unwrap_or(0.0) + scene.shear.unwrap_or(0.0)).sqrt();
    let (substep, refine) = match stepping {
        Stepping::Fixed(substep) => (substep, None),
        Stepping::RefinedAtContact(multiple) => (Substep::Frame, Some(multiple)),
    };
    scene.refine_contact = refine;
    scene.fixed_dt = Some(substep.length(scene.layout.dx, c_p));
    let cells = speed / scene.cell_m;
    if drop {
        scene.v.iter_mut().for_each(|v| *v = Vec2::new(0.0, -cells));
    } else {
        let s = cells / (0.5 * side * std::f32::consts::SQRT_2);
        for p in 0..scene.x.len() {
            let r = scene.x[p] - centre;
            scene.v[p] = s * Vec2::new(r.y, r.x);
            scene.c[p] = Mat2::from_cols(Vec2::new(0.0, s), Vec2::new(s, 0.0));
        }
    }
    let e0 = row_energy(&scene);
    let floor = tank.0.y;
    let (mut worst, mut nan) = (0.0f64, false);
    let (mut touched, mut lift): (bool, Option<f32>) = (false, None);
    let mut series = Vec::new();
    let clock = Instant::now();
    let verbose = std::env::var("EMERGE_PROBE_VERBOSE").is_ok();
    let mut substeps = 0u32;
    if estimate {
        scene.probe.estimator_reference = size * kf;
    }
    // `EMERGE_PROBE_CONDITIONING`: the body's mass conditioning every step.
    if std::env::var_os("EMERGE_PROBE_CONDITIONING").is_some() {
        scene.probe.conditioning_bodies = std::iter::once(0..scene.x.len()).collect();
        scene.probe.conditioning = (f64::INFINITY, Vec::new());
    }
    let mut estimates = Vec::new();
    for _ in 0..(seconds / FRAME).round() as u32 {
        substeps += scene.frame().substeps;
        estimates.push(std::mem::take(&mut scene.probe.largest_estimate));
        let change = row_energy(&scene) / e0 - 1.0;
        let kinetic: f64 = (0..scene.x.len())
            .map(|p| f64::from(0.5 * scene.mass[p] * scene.v[p].length_squared()))
            .sum();
        let (com, com_v) = scene.centre_of_mass();
        series.push([
            change,
            kinetic / e0,
            f64::from(com.y * scene.cell_m),
            f64::from(com_v.y * scene.cell_m),
        ]);
        if verbose {
            let st = scene.probe.conserving;
            let j = (0..scene.x.len())
                .map(|p| scene.j(p))
                .fold(f32::INFINITY, f32::min);
            println!(
                "  t={:.3}: energy {change:+.2e}, Newton {} GMRES {} worst residual {:.1e} unconverged {}, smallest J {j:.4}, ledger particles {:+.3e} grid {:+.3e} stored {:+.3e}, work contact {:+.3e} weight {:+.3e}",
                scene.time,
                st.0,
                st.1,
                st.3,
                st.4,
                scene.probe.ledger.0,
                scene.probe.ledger.1,
                scene.probe.ledger.2,
                scene.probe.work.0,
                scene.probe.work.1
            );
        }
        nan |= !change.is_finite() || scene.x.iter().any(|q| !q.is_finite());
        if nan {
            break;
        }
        worst = worst.max(change.abs());
        if drop {
            let lowest = scene.x.iter().map(|q| q.y).fold(f32::INFINITY, f32::min);
            touched |= lowest <= floor + kf;
            let (_, v) = scene.centre_of_mass();
            if touched && lowest > floor + kf && v.y > 0.0 {
                lift = Some(lift.unwrap_or(0.0).max(v.y * scene.cell_m));
            }
        }
    }
    let end = row_energy(&scene) / e0 - 1.0;
    let st = scene.probe.conserving;
    let wall = clock.elapsed().as_secs_f32() / seconds;
    println!(
        "{} {stepping}: E {e:.2e} nu {nu} size {size} cm speed {speed} m/s at {:.1} cm: worst energy {worst:.2e}, end {end:+.2e}{}, unconverged {} of {} Newton, worst residual {:.1e}, projection {:.1e}, reflections {}, held on a plane at most {}, {} substeps, {}{wall:.1} s per simulated s",
        if drop { "drop" } else { "vibration" },
        1.0 / kf,
        if nan { " NaN" } else { "" },
        st.4,
        st.0,
        st.3,
        scene.probe.worst_projection,
        scene.probe.reflections,
        scene.probe.contacts,
        substeps,
        if drop {
            format!("lift-off {:?} m/s, ", lift)
        } else {
            String::new()
        }
    );
    let per = |k: usize| 1000.0 * scene.probe.phases[k] / f64::from(scene.probe.steps.max(1));
    println!(
        "  per conserving step ({} steps, {:.2} solves, {:.2} Newton, {:.1} GMRES): grid {:.2} ms, projection {:.2}, preconditioner {:.2}, Newton {:.2}, update {:.2}",
        scene.probe.steps,
        scene.probe.passes as f64 / f64::from(scene.probe.steps.max(1)),
        scene.probe.conserving.0 as f64 / f64::from(scene.probe.steps.max(1)),
        scene.probe.conserving.1 as f64 / f64::from(scene.probe.steps.max(1)),
        per(0),
        per(1),
        per(2),
        per(3),
        per(4)
    );
    println!(
        "  mass stabilisation's loss {:.2e} of the start, grid speed over the fastest particle at most {:.2}, Newton start at worst {:.1e}{}",
        scene.probe.stabilisation_loss / e0,
        scene.probe.grid_speed_ratio,
        scene.probe.start_residual,
        if scene.probe.conditioning_bodies.is_empty() {
            String::new()
        } else {
            format!(
                ", mass conditioning at worst {:.1e}",
                scene.probe.conditioning.0
            )
        }
    );
    if estimate {
        let by_frame: Vec<String> = estimates.iter().map(|e| format!("{e:.1e}")).collect();
        println!("  error estimate by frame: {}", by_frame.join(" "));
    }
    series
}

/// The conserving step across states: stiffness, size, speed, Poisson's
/// ratio and resolution, one at a time around a base (medium clay's 11.25
/// MPa, 0.45, 1700 kg/m^3, 10 cm, 2 m/s, 1 cm), for free vibration
/// (`EMERGE_PROBE_SPREAD=vibration`) or drops (`drop`). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_conserving_spread() {
    let drop = std::env::var("EMERGE_PROBE_SPREAD").as_deref() == Ok("drop");
    let seconds = if drop { 0.5 } else { 1.0 };
    let base = (11.25e6f32, 0.45f32, 1700.0f32, 10.0f32, 2.0f32, 1usize);
    let mut cases = vec![base];
    for e in [3.0e6, 1.0e8, 1.0e9, 1.0e10] {
        cases.push((e, base.1, base.2, base.3, base.4, base.5));
    }
    for size in [5.0, 20.0] {
        cases.push((base.0, base.1, base.2, size, base.4, base.5));
    }
    for speed in [0.2, 5.0] {
        cases.push((base.0, base.1, base.2, base.3, speed, base.5));
    }
    for nu in [0.3, 0.49] {
        cases.push((base.0, nu, base.2, base.3, base.4, base.5));
    }
    cases.push((base.0, base.1, base.2, base.3, base.4, 2));
    let only: Option<usize> = std::env::var("EMERGE_PROBE_CASE")
        .ok()
        .and_then(|c| c.parse().ok());
    for (i, &(e, nu, rho, size, speed, k)) in cases.iter().enumerate() {
        if only.is_none_or(|c| c == i) {
            spread_case(SpreadCase {
                drop,
                e,
                nu,
                rho,
                size,
                speed,
                k,
                seconds,
                stepping: Stepping::Fixed(Substep::Frame),
                estimate: false,
            });
        }
    }
}

/// The conserving step at several steppings on the spread's base block
/// (medium clay, 11.25 MPa, 0.45, 10 cm at 1 cm), a probe, not a gate.
/// Each run of `EMERGE_PROBE_RUNS` (default `p,frame,refined:1`, the first
/// the reference: `p`, `4p` or `frame` a fixed substep, `refined:m` one
/// frame per substep refined at contact to `m` P wave steps,
/// `Scene::contact_substep`) is compared frame by frame with the first: the
/// kinetic energy's share (the vibration's phase and amplitude) and the
/// centre of mass's height. `EMERGE_PROBE_WHICH`: `vibration` (the free
/// body's shear start, corner speed `EMERGE_PROBE_SPEED` or 2 m/s, 1 s) or a
/// drop speed in m/s (0.5 s).
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_contact_step_convergence() {
    let which = std::env::var("EMERGE_PROBE_WHICH").unwrap_or_else(|_| "vibration".into());
    let drop_speed: Option<f32> = which.parse().ok();
    let runs = std::env::var("EMERGE_PROBE_RUNS").unwrap_or_else(|_| "p,frame,refined:1".into());
    let mut reference: Option<Vec<[f64; 4]>> = None;
    for run in runs.split(',') {
        let stepping = match run.split_once(':') {
            Some(("refined", multiple)) => {
                Stepping::RefinedAtContact(multiple.parse().unwrap_or(1.0))
            }
            _ => {
                Stepping::Fixed(Substep::parse(run).unwrap_or_else(|| panic!("unknown run {run}")))
            }
        };
        let series = spread_case(SpreadCase {
            drop: drop_speed.is_some(),
            e: 11.25e6,
            nu: 0.45,
            rho: 1700.0,
            size: 10.0,
            speed: drop_speed.unwrap_or_else(|| {
                std::env::var("EMERGE_PROBE_SPEED")
                    .ok()
                    .and_then(|c| c.parse().ok())
                    .unwrap_or(2.0)
            }),
            k: 1,
            seconds: if drop_speed.is_some() { 0.5 } else { 1.0 },
            stepping,
            estimate: false,
        });
        let finite = series
            .iter()
            .take_while(|s| s.iter().all(|e| e.is_finite()))
            .count();
        let (low, high) = series[..finite]
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), s| {
                (l.min(s[0]), h.max(s[0]))
            });
        let against = reference.as_ref().map(|r| {
            let frames = finite.min(r.len());
            let (mut square, mut kinetic, mut height) = (0.0f64, 0.0f64, 0.0f64);
            for (s, q) in series.iter().zip(r).take(frames) {
                square += (s[1] - q[1]).powi(2);
                kinetic = kinetic.max((s[1] - q[1]).abs());
                height = height.max((s[2] - q[2]).abs());
            }
            format!(
                ", against the reference over {frames} frames: kinetic share off by {:.4} rms, {kinetic:.4} at worst; centre height off by {:.3} mm at worst",
                (square / frames.max(1) as f64).sqrt(),
                1000.0 * height
            )
        });
        // The centre of mass after its lowest point: how high it rises
        // again and how fast it moves up (the rebound).
        let lowest = (0..finite).min_by(|&a, &b| series[a][2].total_cmp(&series[b][2]));
        let rebound = lowest.map(|k| {
            let after = &series[k..finite];
            let rise = after.iter().fold(f64::NEG_INFINITY, |m, s| m.max(s[2])) - series[k][2];
            let up = after.iter().fold(f64::NEG_INFINITY, |m, s| m.max(s[3]));
            format!(
                ", centre lowest {:.2} mm at frame {}, then rises {:.2} mm, up to {:.3} m/s upward",
                1000.0 * series[k][2],
                k + 1,
                1000.0 * rise,
                up
            )
        });
        println!(
            "  {run}: {finite} of {} frames finite, energy from {low:+.2e} to {high:+.2e}{}{}",
            series.len(),
            against.unwrap_or_default(),
            rebound.unwrap_or_default()
        );
        if reference.is_none() {
            reference = Some(series[..finite].to_vec());
        }
    }
}

/// A square solid block of `side` cells centred on `centre`, seeded
/// (`seed_body`) from `seed_value`: the per-body probes' bodies.
fn clay_block(centre: Vec2, side: f32, seed_value: u32) -> Vec<Vec2> {
    let (lo, hi) = (
        centre - Vec2::splat(0.5 * side),
        centre + Vec2::splat(0.5 * side),
    );
    seed_body(
        true,
        (lo.floor().as_ivec2(), hi.ceil().as_ivec2()),
        |q| q.cmpge(lo).all() && q.cmple(hi).all(),
        |_| (f32::INFINITY, Vec2::Y),
        &mut LcgRng::new(seed_value),
    )
}

/// One body's energy (`row_energy` over its particles `r`), kinetic
/// energy, centre height in m and vertical centre speed in m/s.
fn body_measure(scene: &Scene, r: std::ops::Range<usize>) -> [f64; 4] {
    let floor = scene.tank.0.y;
    let mu = scene.shear.unwrap_or(0.0);
    let kappa = scene.sound_speed2.unwrap_or(0.0);
    let (mut energy, mut kinetic, mut mass) = (0.0f64, 0.0f64, 0.0f64);
    let (mut y, mut vy) = (0.0f64, 0.0f64);
    for p in r {
        let m = f64::from(scene.mass[p]);
        let k = 0.5 * m * f64::from(scene.v[p].length_squared());
        let stored = f64::from(
            scene.v0[p]
                * (shear_energy(scene.deformation[p], mu)
                    + 0.5 * kappa * (scene.j(p) - 1.0).powi(2)),
        );
        energy += k + m * f64::from(scene.g() * (scene.x[p].y - floor)) + stored;
        kinetic += k;
        mass += m;
        y += m * f64::from(scene.x[p].y);
        vy += m * f64::from(scene.v[p].y);
    }
    let cell = f64::from(scene.cell_m);
    [energy, kinetic, y / mass * cell, vy / mass * cell]
}

/// Per-body steps, a first measured test, nothing built into the engine: two
/// medium clay blocks in one tank, 22 cells apart so no grid node is shared, A
/// dropped onto the floor at 2 m/s and B thrown up at 2 m/s while vibrating
/// (the free body's shear start, corners at 2 m/s), out of every wall's reach
/// for the whole 0.4 s. Run together with the contact refinement as it is,
/// global (`Scene::contact_substep` refines the whole scene), then each block
/// alone in its own scene with its own refinement. A takes the same steps in
/// both, so A together against A alone is what the bodies' shared solve
/// couples; B alone keeps frame steps where together it followed A's
/// refinement. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_per_body_steps() {
    use std::ops::Range;
    let (e, nu, rho, side) = (11.25e6f32, 0.45f32, 1700.0f32, 10.0f32);
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let block = |centre: Vec2, seed_value: u32| clay_block(centre, side, seed_value);
    let (centre_a, centre_b) = (
        Vec2::new(20.0, 4.0 + 2.0 + 0.5 * side),
        Vec2::new(52.0, 36.0),
    );
    let (seed_a, seed_b) = (block(centre_a, 17), block(centre_b, 23));
    // Each body's start: positions, velocities, APIC matrices.
    let start = |body: usize| -> (Vec<Vec2>, Vec<Vec2>, Vec<Mat2>) {
        if body == 0 {
            let n = seed_a.len();
            (
                seed_a.clone(),
                vec![Vec2::new(0.0, -200.0); n],
                vec![Mat2::ZERO; n],
            )
        } else {
            let s = 200.0 / (0.5 * side * std::f32::consts::SQRT_2);
            let v = seed_b
                .iter()
                .map(|&q| {
                    let r = q - centre_b;
                    s * Vec2::new(r.y, r.x) + Vec2::new(0.0, 200.0)
                })
                .collect();
            let c = Mat2::from_cols(Vec2::new(0.0, s), Vec2::new(s, 0.0));
            (seed_b.clone(), v, vec![c; seed_b.len()])
        }
    };
    // The scene holding `bodies`, with each one's particle range.
    let build = |bodies: &[usize], tolerance: f64| -> (Scene, Vec<Range<usize>>) {
        let (mut x, mut v, mut c, mut ranges) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &b in bodies {
            let (xb, vb, cb) = start(b);
            ranges.push(x.len()..x.len() + xb.len());
            x.extend(xb);
            v.extend(vb);
            c.extend(cb);
        }
        let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x).with_elastic(e / rho, nu);
        scene.v = v;
        scene.c = c;
        scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
            occupation: Some(0.75),
            epsilon: 0.0,
            newton_tolerance: tolerance,
            ..Default::default()
        });
        scene.refine_contact = Some(1.0);
        scene.fixed_dt = Some(FRAME);
        (scene, ranges)
    };
    let measure = |scene: &Scene, r: &Range<usize>| body_measure(scene, r.clone());
    let frames = 24;
    // One run: per body and frame, the measures and the positions; the
    // substeps and frames refined; the wall time.
    let run = |label: &str, bodies: &[usize], tolerance: f64| {
        let (mut scene, ranges) = build(bodies, tolerance);
        let mut series: Vec<Vec<[f64; 4]>> = vec![Vec::new(); ranges.len()];
        let mut positions: Vec<Vec<Vec<Vec2>>> = vec![Vec::new(); ranges.len()];
        let (mut substeps, mut refined) = (0u32, 0u32);
        let clock = Instant::now();
        for _ in 0..frames {
            refined += u32::from(scene.contact_substep().is_some());
            substeps += scene.frame().substeps;
            for (b, r) in ranges.iter().enumerate() {
                series[b].push(measure(&scene, r));
                positions[b].push(scene.x[r.clone()].to_vec());
            }
        }
        let wall = clock.elapsed().as_secs_f64();
        let st = scene.probe.conserving;
        println!(
            "{label}: {substeps} substeps, {refined} of {frames} frames refined, {:.1} s wall ({:.1} s per simulated s), unconverged {} of {} Newton, worst residual {:.1e}",
            wall,
            wall / (f64::from(frames) * f64::from(FRAME)),
            st.4,
            st.0,
            st.3
        );
        (series, positions, wall)
    };
    // One body in two runs: the measures' worst gaps, and the particles'
    // largest gap at a few frames.
    let compare = |label: &str,
                   s: &[[f64; 4]],
                   q: &[[f64; 4]],
                   xs: &[Vec<Vec2>],
                   xq: &[Vec<Vec2>]| {
        let e0 = q[0][0];
        let (mut energy, mut kinetic, mut height, mut speed) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let (mut drift_s, mut drift_q) = (0.0f64, 0.0f64);
        for (a, b) in s.iter().zip(q) {
            energy = energy.max((a[0] - b[0]).abs() / e0);
            kinetic = kinetic.max((a[1] - b[1]).abs() / e0);
            height = height.max((a[2] - b[2]).abs());
            speed = speed.max((a[3] - b[3]).abs());
            drift_s = drift_s.max((a[0] / e0 - 1.0).abs());
            drift_q = drift_q.max((b[0] / e0 - 1.0).abs());
        }
        let gap = |f: usize| {
            xs[f]
                .iter()
                .zip(&xq[f])
                .fold(0.0f32, |m, (a, b)| m.max((*a - *b).length()))
        };
        let by_frame: Vec<String> = [1, 2, 3, 4, 8, 16, 24]
            .iter()
            .map(|&f| format!("{f}: {:.1e}", gap(f - 1)))
            .collect();
        println!(
            "  {label}: energy off by {energy:.2e} of its start at worst (energy change {drift_s:.2e} and {drift_q:.2e}), kinetic off by {kinetic:.2e}, centre height off by {:.4} mm, centre speed off by {:.2e} m/s; particles off by at most, in cells, after frame {}",
            1000.0 * height,
            speed,
            by_frame.join(", ")
        );
    };
    let default =
        crate::spacetime::total_lagrangian::ConservingSettings::default().newton_tolerance;
    let (together, together_x, together_wall) =
        run("together, refinement global", &[0, 1], default);
    let (a_alone, a_x, a_wall) = run("body A alone", &[0], default);
    compare(
        "A together against alone",
        &together[0],
        &a_alone[0],
        &together_x[0],
        &a_x[0],
    );
    let (b_alone, b_x, b_wall) = run("body B alone", &[1], default);
    compare(
        "B together against alone",
        &together[1],
        &b_alone[0],
        &together_x[1],
        &b_x[0],
    );
    println!(
        "wall time: together {together_wall:.1} s, the two alone {:.1} s ({:.2}x)",
        a_wall + b_wall,
        together_wall / (a_wall + b_wall)
    );
    // The control: A alone again with a Newton tolerance a hundred times
    // tighter, the size of perturbation the shared solve brings.
    let (a_tight, a_tight_x, _) = run("body A alone, Newton tolerance 1e-12", &[0], 1e-12);
    compare(
        "A tight against default, both alone",
        &a_tight[0],
        &a_alone[0],
        &a_tight_x[0],
        &a_x[0],
    );
    compare(
        "A together against A alone with the tight tolerance",
        &together[0],
        &a_tight[0],
        &together_x[0],
        &a_tight_x[0],
    );
}

/// One group of the per-body stress probe: bodies stepped together, the
/// scene holding them and each one's particles.
struct Group {
    scene: Scene,
    bodies: Vec<usize>,
    ranges: Vec<std::ops::Range<usize>>,
}

impl Group {
    /// Body `i`'s positions and velocities.
    fn motion(&self, i: usize) -> (&[Vec2], &[Vec2]) {
        let r = self.ranges[i].clone();
        (&self.scene.x[r.clone()], &self.scene.v[r])
    }
}

/// Whether two bodies, positions and velocities, come within `reach` cells
/// of each other along both axes at once (the max norm, a quadratic
/// B-spline's support being square) at the frame's start or at its
/// predicted end, each particle on its own trajectory with the weight
/// (`Scene::contact_substep`'s prediction).
fn bodies_within(a: (&[Vec2], &[Vec2]), b: (&[Vec2], &[Vec2]), gravity: Vec2, reach: f32) -> bool {
    let ends = |(x, v): (&[Vec2], &[Vec2])| -> Vec<(Vec2, Vec2)> {
        x.iter()
            .zip(v)
            .map(|(&q, &w)| (q, q + w * FRAME + 0.5 * gravity * FRAME * FRAME))
            .collect()
    };
    let (ea, eb) = (ends(a), ends(b));
    let bounds = |e: &[(Vec2, Vec2)]| {
        e.iter().fold(
            (Vec2::splat(f32::INFINITY), Vec2::splat(f32::NEG_INFINITY)),
            |(lo, hi), &(q, w)| (lo.min(q).min(w), hi.max(q).max(w)),
        )
    };
    let ((alo, ahi), (blo, bhi)) = (bounds(&ea), bounds(&eb));
    if (alo - Vec2::splat(reach)).cmpgt(bhi).any() || (blo - Vec2::splat(reach)).cmpgt(ahi).any() {
        return false;
    }
    let apart = |p: Vec2, q: Vec2| (p - q).abs().max_element();
    ea.iter().any(|&(qa, wa)| {
        eb.iter()
            .any(|&(qb, wb)| apart(qa, qb) < reach || apart(wa, wb) < reach)
    })
}

/// Per-body steps under stress, a probe, nothing built into the engine: up
/// to six medium clay blocks in a 1.36 m tank over 0.6 s
/// (`EMERGE_PROBE_BODIES`, default `0,1,2,3,4,5`; `EMERGE_PROBE_FRAMES`,
/// default 36). 0 is dropped at 2 m/s, 1 and 2 fall from rest from two
/// heights, 3 is thrown up at 2 m/s while vibrating, 4 and 5 fly at each
/// other at 1 m/s each and meet in the air. Run once as one scene, refined
/// whenever any body nears a wall or another body (the contact refinement,
/// global), then as groups that each take their own steps: a group refines
/// when one of its bodies nears a wall or another of its bodies, and two
/// groups merge as soon as their bodies come within 5 cells along both
/// axes, before they could share a node (a particle's support reaches 1.5
/// cells each way). A guard counts the frames ending with two groups within
/// 2 cells of a common cell. The approach between bodies refining is this
/// probe's: the refinement as it is only knows the walls.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_per_body_stress() {
    // The largest of two, NaN when either is: `f64::max` would drop a NaN and
    // a broken run would read as a small gap.
    let worst = |m: f64, x: f64| {
        if m.is_nan() || x.is_nan() {
            f64::NAN
        } else {
            m.max(x)
        }
    };
    let (e, nu, rho, side) = (11.25e6f32, 0.45f32, 1700.0f32, 10.0f32);
    let (nx, ny) = (144usize, 72usize);
    let tank = (Vec2::splat(4.0), Vec2::new(140.0, 68.0));
    let reach = 5.0f32;
    // `EMERGE_PROBE_CONDITIONING`: each group of several bodies measures its
    // mass conditioning every step, the union's and each body's own.
    let conditioning = std::env::var_os("EMERGE_PROBE_CONDITIONING").is_some();
    // `EMERGE_PROBE_EPSILON`: the effective mass's `eps` (Love and Sulsky eq.
    // 26), 0 the consistent mass; `EMERGE_PROBE_NO_REFINE`: every substep
    // one frame, no refinement near walls or bodies; `EMERGE_PROBE_BASIS`:
    // the conditioning below which a body's basis is dissected.
    let epsilon: f64 = std::env::var("EMERGE_PROBE_EPSILON")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let no_refine = std::env::var_os("EMERGE_PROBE_NO_REFINE").is_some();
    let basis_threshold: f64 = std::env::var("EMERGE_PROBE_BASIS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let chosen: Vec<usize> = std::env::var("EMERGE_PROBE_BODIES")
        .unwrap_or_else(|_| "0,1,2,3,4,5".into())
        .split(',')
        .filter_map(|b| b.trim().parse().ok())
        .collect();
    let frames: usize = std::env::var("EMERGE_PROBE_FRAMES")
        .ok()
        .and_then(|f| f.parse().ok())
        .unwrap_or(36);
    // Each body's start: positions, velocities, APIC matrices.
    let start = |body: usize| -> (Vec<Vec2>, Vec<Vec2>, Vec<Mat2>) {
        let shake = 200.0 / (0.5 * side * std::f32::consts::SQRT_2);
        let (centre, push, vibrating) = match body {
            0 => (Vec2::new(13.0, 11.0), Vec2::new(0.0, -200.0), false),
            1 => (Vec2::new(31.0, 30.0), Vec2::ZERO, false),
            2 => (Vec2::new(49.0, 50.0), Vec2::ZERO, false),
            3 => (Vec2::new(67.0, 26.0), Vec2::new(0.0, 200.0), true),
            4 => (Vec2::new(87.0, 56.0), Vec2::new(100.0, 0.0), false),
            _ => (Vec2::new(127.0, 56.0), Vec2::new(-100.0, 0.0), false),
        };
        let x = clay_block(centre, side, 17 + body as u32);
        let n = x.len();
        if vibrating {
            let v = x
                .iter()
                .map(|&q| {
                    let r = q - centre;
                    shake * Vec2::new(r.y, r.x) + push
                })
                .collect();
            let c = Mat2::from_cols(Vec2::new(0.0, shake), Vec2::new(shake, 0.0));
            (x, v, vec![c; n])
        } else {
            (x, vec![push; n], vec![Mat2::ZERO; n])
        }
    };
    let starts: Vec<_> = (0..6).map(start).collect();
    // The scene holding `bodies` at their start, with each one's particles.
    let build = |bodies: &[usize]| -> Group {
        let (mut x, mut v, mut c, mut ranges) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &b in bodies {
            let (xb, vb, cb) = &starts[b];
            ranges.push(x.len()..x.len() + xb.len());
            x.extend_from_slice(xb);
            v.extend_from_slice(vb);
            c.extend_from_slice(cb);
        }
        let mut scene = Scene::new((nx, ny), 0.01, 1.0, tank, x).with_elastic(e / rho, nu);
        scene.v = v;
        scene.c = c;
        scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
            occupation: Some(0.75),
            epsilon,
            ..Default::default()
        });
        scene.refine_contact = (!no_refine).then_some(1.0);
        scene.fixed_dt = Some(FRAME);
        Group {
            scene,
            bodies: bodies.to_vec(),
            ranges,
        }
    };
    // Whether any two of a group's bodies near, or a body of each of two.
    let near = |a: &Group, b: Option<&Group>| -> bool {
        let other = b.unwrap_or(a);
        (0..a.bodies.len()).any(|i| {
            let from = if b.is_some() { 0 } else { i + 1 };
            (from..other.bodies.len())
                .any(|j| bodies_within(a.motion(i), other.motion(j), a.scene.gravity, reach))
        })
    };
    // The cells within 2 of a group's particles.
    let cells = |g: &Group| -> Vec<bool> {
        let mut out = vec![false; nx * ny];
        for q in &g.scene.x {
            let (i0, j0) = (
                (q.x - 2.0).floor().max(0.0) as usize,
                (q.y - 2.0).floor().max(0.0) as usize,
            );
            let (i1, j1) = (
                ((q.x + 2.0).floor() as usize).min(nx - 1),
                ((q.y + 2.0).floor() as usize).min(ny - 1),
            );
            for j in j0..=j1 {
                for i in i0..=i1 {
                    out[i + nx * j] = true;
                }
            }
        }
        out
    };
    let c_p = {
        let g = build(&chosen[..1]);
        (g.scene.sound_speed2.unwrap_or(0.0) + g.scene.shear.unwrap_or(0.0)).sqrt()
    };
    let p_step = 0.5 / c_p;
    // One run, per body: the measures and positions each frame, the frames
    // it was refined; and the run's totals.
    let run = |label: &str, per_body: bool| {
        let mut groups: Vec<Group> = if per_body {
            chosen.iter().map(|&b| build(&[b])).collect()
        } else {
            vec![build(&chosen)]
        };
        let index = |b: usize| chosen.iter().position(|&c| c == b).expect("chosen");
        let mut series = vec![Vec::new(); chosen.len()];
        let mut positions = vec![Vec::new(); chosen.len()];
        let mut refined = vec![0u32; chosen.len()];
        let (mut particle_steps, mut flags) = (0u64, 0u32);
        let mut merges = Vec::new();
        // Newton, unconverged, worst residual and phases of groups merged
        // away.
        let mut carried = (0u64, 0u32, 0.0f64, [0.0f64; 5]);
        let e0: f64 = groups
            .iter()
            .flat_map(|g| {
                g.ranges
                    .iter()
                    .map(|r| body_measure(&g.scene, r.clone())[0])
            })
            .sum();
        let mut drift = 0.0f64;
        let mut first_broken: Option<usize> = None;
        let clock = Instant::now();
        for frame in 0..frames {
            // Groups merge before their bodies could share a node.
            while let Some((a, b)) = per_body
                .then(|| {
                    (0..groups.len())
                        .flat_map(|a| (a + 1..groups.len()).map(move |b| (a, b)))
                        .find(|&(a, b)| near(&groups[a], Some(&groups[b])))
                })
                .flatten()
            {
                let second = groups.remove(b);
                let first = groups.remove(a);
                let bodies: Vec<usize> =
                    first.bodies.iter().chain(&second.bodies).copied().collect();
                let mut merged = build(&bodies);
                for (old, offset) in [(&first, 0), (&second, first.bodies.len())] {
                    for (i, r) in old.ranges.iter().enumerate() {
                        let to = merged.ranges[offset + i].clone();
                        let (s, m) = (&old.scene, &mut merged.scene);
                        m.x[to.clone()].copy_from_slice(&s.x[r.clone()]);
                        m.v[to.clone()].copy_from_slice(&s.v[r.clone()]);
                        m.c[to.clone()].copy_from_slice(&s.c[r.clone()]);
                        m.deformation[to.clone()].copy_from_slice(&s.deformation[r.clone()]);
                        m.log_j[to].copy_from_slice(&s.log_j[r.clone()]);
                    }
                    let st = old.scene.probe.conserving;
                    carried.0 += st.0;
                    carried.1 += st.4;
                    carried.2 = carried.2.max(st.3);
                    for (t, part) in carried.3.iter_mut().zip(old.scene.probe.phases) {
                        *t += part;
                    }
                }
                merged.scene.time = first.scene.time;
                merges.push(format!("{bodies:?} at frame {}", frame + 1));
                groups.push(merged);
            }
            let mut stepped = Vec::new();
            for g in &mut groups {
                let bodies_near = !no_refine && near(g, None);
                g.scene.fixed_dt = Some(if bodies_near { p_step } else { FRAME });
                if bodies_near || g.scene.contact_substep().is_some() {
                    for &b in &g.bodies {
                        refined[index(b)] += 1;
                    }
                }
                let measure = conditioning && g.bodies.len() > 1;
                if measure {
                    g.scene.probe.conditioning_bodies = g.ranges.clone();
                    g.scene.probe.conditioning = (f64::INFINITY, Vec::new());
                    g.scene.probe.basis_threshold = basis_threshold;
                    g.scene.probe.basis_worst = None;
                }
                g.scene.probe.grid_speed_ratio = 0.0;
                g.scene.probe.start_residual = 0.0;
                let substeps = g.scene.frame().substeps;
                particle_steps += u64::from(substeps) * g.scene.x.len() as u64;
                let st = g.scene.probe.conserving;
                let measured = if measure {
                    let (union, each) = &g.scene.probe.conditioning;
                    let each: Vec<String> = each.iter().map(|c| format!("{c:.1e}")).collect();
                    format!(
                        ", mass conditioning union {union:.1e}, each body {}",
                        each.join(" ")
                    )
                } else {
                    String::new()
                };
                stepped.push(format!(
                    "{:?} {substeps} (solves {}, held at most {}, Newton {}, GMRES {} and at most {} in a step, worst residual {:.1e}, unconverged {}{measured}, grid speed over the fastest particle {:.2}, Newton start {:.1e}, mass stabilisation's loss so far {:.2e} of the run's start)",
                    g.bodies,
                    g.scene.probe.passes,
                    g.scene.probe.contacts,
                    st.0,
                    st.1,
                    st.2,
                    st.3,
                    st.4,
                    g.scene.probe.grid_speed_ratio,
                    g.scene.probe.start_residual,
                    g.scene.probe.stabilisation_loss / e0
                ));
                if let Some((c, report)) = &g.scene.probe.basis_worst {
                    stepped.push(format!("\n      basis at conditioning {c:.1e}, {report}"));
                }
            }
            println!(
                "  {label} frame {}: substeps by group {}, {:.0} s",
                frame + 1,
                stepped.join(", "),
                clock.elapsed().as_secs_f64()
            );
            if per_body {
                let marks: Vec<Vec<bool>> = groups.iter().map(cells).collect();
                let shared = (0..marks.len()).any(|a| {
                    (a + 1..marks.len())
                        .any(|b| marks[a].iter().zip(&marks[b]).any(|(&p, &q)| p && q))
                });
                flags += u32::from(shared);
            }
            let mut total = 0.0;
            for g in &groups {
                for (i, &b) in g.bodies.iter().enumerate() {
                    let m = body_measure(&g.scene, g.ranges[i].clone());
                    total += m[0];
                    series[index(b)].push(m);
                    positions[index(b)].push(g.scene.x[g.ranges[i].clone()].to_vec());
                }
            }
            drift = worst(drift, (total / e0 - 1.0).abs());
            if first_broken.is_none()
                && groups
                    .iter()
                    .any(|g| g.scene.x.iter().any(|q| !q.is_finite()))
            {
                first_broken = Some(frame + 1);
            }
        }
        let wall = clock.elapsed().as_secs_f64();
        let (mut newton, mut unconverged, mut residual, mut phases) = carried;
        for g in &groups {
            let st = g.scene.probe.conserving;
            newton += st.0;
            unconverged += st.4;
            residual = residual.max(st.3);
            for (t, part) in phases.iter_mut().zip(g.scene.probe.phases) {
                *t += part;
            }
        }
        println!(
            "{label}: {} body-frames refined of {}, {:.2}e6 particle-steps, {wall:.1} s wall ({:.1} s per simulated s), phases grid {:.1} s, projection {:.1}, preconditioner {:.1}, Newton {:.1}, update {:.1}; unconverged {unconverged} of {newton} Newton, worst residual {residual:.1e}; total energy change {drift:.2e} at worst; first frame with a position not finite {first_broken:?}; merges {merges:?}; frames ending with groups within reach of a common cell {flags}",
            refined.iter().sum::<u32>(),
            chosen.len() * frames,
            particle_steps as f64 * 1e-6,
            wall / (frames as f64 * f64::from(FRAME)),
            phases[0],
            phases[1],
            phases[2],
            phases[3],
            phases[4]
        );
        (series, positions, refined, wall)
    };
    let (one, one_x, one_refined, one_wall) = run("one scene, refinement global", false);
    let (each, each_x, each_refined, each_wall) = run("groups, each its own steps", true);
    println!(
        "wall time: one scene {one_wall:.1} s, groups {each_wall:.1} s ({:.2}x)",
        one_wall / each_wall
    );
    for (i, &b) in chosen.iter().enumerate() {
        let (s, q) = (&one[i], &each[i]);
        let e0 = q[0][0];
        let (mut kinetic, mut height, mut speed) = (0.0f64, 0.0f64, 0.0f64);
        for (a, c) in s.iter().zip(q) {
            kinetic = worst(kinetic, (a[1] - c[1]).abs() / e0);
            height = worst(height, (a[2] - c[2]).abs());
            speed = worst(speed, (a[3] - c[3]).abs());
        }
        let up = |r: &[[f64; 4]]| r.iter().fold(f64::NEG_INFINITY, |m, f| worst(m, f[3]));
        let gap = |f: usize| {
            one_x[i][f]
                .iter()
                .zip(&each_x[i][f])
                .fold(0.0f64, |m, (a, c)| worst(m, f64::from((*a - *c).length())))
        };
        let marks: Vec<String> = [1, frames / 4, frames / 2, 3 * frames / 4, frames]
            .iter()
            .map(|&f| format!("{f}: {:.1e}", gap(f.max(1) - 1)))
            .collect();
        println!(
            "  body {b}: refined {} frames in one scene, {} alone or grouped; centre height off by {:.3} mm at worst, centre speed off by {speed:.2e} m/s, kinetic off by {kinetic:.2e} of its start; fastest upward centre speed {:.3} and {:.3} m/s; particles off by at most, in cells, after frame {}",
            one_refined[i],
            each_refined[i],
            1000.0 * height,
            up(s),
            up(q),
            marks.join(", ")
        );
    }
}

/// The error estimate of Zienkiewicz and Xie (`Probe::estimator_reference`)
/// on the conserving step, frame by frame: the spread's base block (medium
/// clay, 10 cm at 1 cm) vibrating freely at one frame per substep and at
/// the P wave step, and dropped at 0.66 and 2 m/s with the contact
/// refinement, so that its values at an impact stand against those of a
/// vibration a frame's step does not resolve. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_error_estimator() {
    let case = |drop: bool, speed: f32, seconds: f32, stepping: Stepping| SpreadCase {
        drop,
        e: 11.25e6,
        nu: 0.45,
        rho: 1700.0,
        size: 10.0,
        speed,
        k: 1,
        seconds,
        stepping,
        estimate: true,
    };
    spread_case(case(false, 2.0, 1.0, Stepping::Fixed(Substep::Frame)));
    spread_case(case(false, 2.0, 0.05, Stepping::Fixed(Substep::PWave)));
    spread_case(case(true, 0.2, 0.1, Stepping::RefinedAtContact(1.0)));
    spread_case(case(true, 2.0, 0.1, Stepping::RefinedAtContact(1.0)));
}

/// One run of the manufactured solution (`probe_manufactured_solution`).
#[derive(Clone, Copy)]
struct Manufactured {
    /// The grid kept on the reference configuration
    /// (`total_lagrangian::step_total_lagrangian`), or rebuilt at every step
    /// (`total_lagrangian::step_dirichlet`).
    lagrangian: bool,
    substep: Substep,
    /// The body's side `L` in cells (`L = 10 cm`, so `h = L / cells`).
    cells: usize,
    /// `G / L`.
    amplitude: f64,
    /// `omega dt` at one frame's step, the same `omega` at every substep.
    omega_frame: f64,
    newton_tolerance: f64,
    /// The longest a P wave step run simulates, s.
    cap: f64,
    /// `sin^2` in place of `sin` along each axis: zero with a zero slope
    /// on the boundary, so the held faces beyond it are what the exact
    /// field's spline needs there to second order.
    squared: bool,
}

/// The manufactured solution of Gong (PhD, UNM 2015, section 4.7, eq.
/// 4.41) after Sadeghirad et al. 2011, with its frequency free: on the
/// reference square `[0, L]^2`, `U = (G sin(pi X / L) sin(w t), G sin(pi Y
/// / L) sin(w t))`. The body force making it exact is derived here for this
/// engine's energy (`total_lagrangian::Elastic`): `B = A - Div P` per unit
/// reference mass (density 1 in the step's units), `A = -w^2 U`, `P = tau
/// F^-T` the step's own stress (`Elastic::first_pk`), its divergence by a
/// central difference along `X` and `Y` of the exact `P` (`P` is diagonal
/// for this diagonal `F`, so only `dP11/dX` and `dP22/dY` remain). The
/// rollers `v1 = 0` at `X = 0, L` and `v2 = 0` at `Y = 0, L` are exact for
/// it; the faces on and beyond them are held at zero both ways. Errors are
/// the relative L2 norms over the particles of displacement, velocity and
/// `P`, each over the norm of its exact field at its peak phase, at the end
/// and at worst over the frames' ends; realism is one less the error.
fn manufactured_run(case: Manufactured) -> String {
    use glam::{DMat2, DVec2};
    let Manufactured {
        lagrangian,
        substep,
        cells,
        amplitude,
        omega_frame,
        newton_tolerance,
        cap,
        squared,
    } = case;
    let side = cells as f64;
    let cell_m = 0.1 / side;
    // Medium clay; `EMERGE_PROBE_MS_NU` overrides its Poisson ratio.
    let nu: f64 = std::env::var("EMERGE_PROBE_MS_NU")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.45);
    let (e, rho) = (11.25e6f64, 1700.0f64);
    let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu);
    let c_p = (solid.bulk + solid.mu).sqrt();
    // The body's corner on the grid: on a face (default) or between faces.
    let corner: f64 = std::env::var("EMERGE_PROBE_MS_CORNER")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4.0);
    let layout = MacLayout::new(cells + 8, cells + 8, 1.0);
    let omega = omega_frame / f64::from(FRAME);
    let g = amplitude * side;
    let k = std::f64::consts::PI / side;
    let period = 2.0 * std::f64::consts::PI / omega;
    let frame = f64::from(FRAME);
    // `EMERGE_PROBE_MS_PERIODS` periods (default one).
    let periods: f64 = std::env::var("EMERGE_PROBE_MS_PERIODS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let frames = (periods * period / frame).round().max(1.0) as usize;
    let per_frame = match substep {
        Substep::Frame => 1,
        _ => (frame / f64::from(substep.length(1.0, c_p as f32))).ceil() as usize,
    };
    let dt = frame / per_frame as f64;
    let mut steps = frames * per_frame;
    if !matches!(substep, Substep::Frame) {
        steps = steps.min((cap / dt).round().max(1.0) as usize);
    }
    // The profile along one axis and its slope.
    let profile = |s: f64| {
        let a = k * (s - corner);
        if squared {
            (a.sin() * a.sin(), k * (2.0 * a).sin())
        } else {
            (a.sin(), k * a.cos())
        }
    };
    // The exact fields at a reference point and a time.
    let u =
        |q: DVec2, t: f64| DVec2::new(g * profile(q.x).0, g * profile(q.y).0) * (omega * t).sin();
    let v = |q: DVec2, t: f64| {
        DVec2::new(g * profile(q.x).0, g * profile(q.y).0) * omega * (omega * t).cos()
    };
    let grad = |q: DVec2, t: f64| {
        DMat2::from_diagonal(DVec2::new(
            1.0 + g * profile(q.x).1 * (omega * t).sin(),
            1.0 + g * profile(q.y).1 * (omega * t).sin(),
        ))
    };
    let stress = |q: DVec2, t: f64| solid.first_pk(grad(q, t));
    // With `EMERGE_PROBE_ZETA` a Kelvin-Voigt viscous stress
    // (`total_lagrangian::Elastic::viscous_stress`, stiffness-proportional) whose
    // damping ratio is `zeta` at 1 kHz, `beta = 2 zeta / (2 pi 1000 Hz)`:
    // the clay's small-strain damping, measured in the geotechnical band
    // (0.05-80 Hz) and by bender elements from 1 kHz, used here at the
    // body's unresolved modes (an extrapolation, declared). The step
    // evaluates it at the end-of-step velocity; the exact body force
    // includes it, so the manufactured motion stays exact.
    let zeta: f64 = std::env::var("EMERGE_PROBE_ZETA")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.0);
    let beta = 2.0 * zeta / (2.0 * std::f64::consts::PI * 1000.0);
    crate::spacetime::total_lagrangian::hooks::set_viscosity(beta);
    let grad_rate = |q: DVec2, t: f64| {
        DMat2::from_diagonal(DVec2::new(
            g * profile(q.x).1 * omega * (omega * t).cos(),
            g * profile(q.y).1 * omega * (omega * t).cos(),
        ))
    };
    let total_stress = |q: DVec2, t: f64| {
        let f = grad(q, t);
        if beta > 0.0 {
            solid.first_pk(f) + f * solid.viscous_stress(f, grad_rate(q, t), beta)
        } else {
            solid.first_pk(f)
        }
    };
    let divergence = |q: DVec2, t: f64, delta: f64| {
        let dx = DVec2::new(delta, 0.0);
        let dy = DVec2::new(0.0, delta);
        DVec2::new(
            (total_stress(q + dx, t).x_axis.x - total_stress(q - dx, t).x_axis.x) / (2.0 * delta),
            (total_stress(q + dy, t).y_axis.y - total_stress(q - dy, t).y_axis.y) / (2.0 * delta),
        )
    };
    let delta = 1e-3;
    let body = |q: DVec2, t: f64| -omega * omega * u(q, t) - divergence(q, t, delta);
    // The quadrature points of the reference block: particles two by two
    // per cell, or with `EMERGE_PROBE_MS_QUADRATURE=gauss` the reference
    // domain's own integrals, a Gauss-Legendre rule of
    // `EMERGE_PROBE_MS_GAUSS` points (default 3) per direction on every
    // half cell, where both lattices' B-splines are polynomials.
    let gauss = std::env::var("EMERGE_PROBE_MS_QUADRATURE").as_deref() == Ok("gauss");
    let order: usize = std::env::var("EMERGE_PROBE_MS_GAUSS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let (nodes, node_weights) = if gauss {
        gauss_legendre(order)
    } else {
        (vec![0.5], vec![1.0])
    };
    let (reference, v0): (Vec<DVec2>, Vec<f64>) = (0..2 * cells)
        .flat_map(|j| (0..2 * cells).map(move |i| (i, j)))
        .flat_map(|(i, j)| {
            let (nodes, node_weights) = (&nodes, &node_weights);
            (0..nodes.len()).flat_map(move |b| {
                (0..nodes.len()).map(move |a| {
                    (
                        DVec2::new(
                            corner + (i as f64 + nodes[a]) * 0.5,
                            corner + (j as f64 + nodes[b]) * 0.5,
                        ),
                        0.25 * node_weights[a] * node_weights[b],
                    )
                })
            })
        })
        .unzip();
    let n = reference.len();
    // Density 1 in the step's units.
    let mass = v0.clone();
    let solids = vec![solid; n];
    let mut x = reference.clone();
    let mut vel: Vec<DVec2> = reference.iter().map(|&q| v(q, 0.0)).collect();
    let mut f = vec![DMat2::IDENTITY; n];
    let (lo, hi) = (corner, corner + side);
    let held = |a: usize, i: usize, j: usize| -> Option<f64> {
        let at = if a == 0 { i as f64 } else { j as f64 };
        if at <= lo + 1e-9 {
            Some(1.0)
        } else if at >= hi - 1e-9 {
            Some(-1.0)
        } else {
            None
        }
    };
    let settings = crate::spacetime::total_lagrangian::ConservingSettings {
        newton_tolerance,
        occupation: match std::env::var("EMERGE_PROBE_OCCUPATION").as_deref() {
            Ok("none") => None,
            _ => crate::spacetime::total_lagrangian::ConservingSettings::default().occupation,
        },
        ..Default::default()
    };
    // L2 norms over the quadrature points, each weighted by its volume
    // (equal weights for the particles: the plain sums).
    let norm = |values: &mut dyn Iterator<Item = f64>| values.sum::<f64>().sqrt();
    let peak = 0.5 * std::f64::consts::PI / omega;
    let u_scale = norm(&mut (0..n).map(|p| v0[p] * u(reference[p], peak).length_squared()));
    let v_scale = norm(&mut (0..n).map(|p| v0[p] * v(reference[p], 0.0).length_squared()));
    let p_scale = norm(&mut (0..n).map(|p| {
        let s = stress(reference[p], peak);
        v0[p] * (s.x_axis.length_squared() + s.y_axis.length_squared())
    }));
    // With `EMERGE_PROBE_THETA=cell`, `Theta` constant per grid cell (the
    // cell's mean J, Gonzalez 2000 eq. 53): the step's volumetric stress
    // takes the cell's pressure, and the stored energy is `sum V0 W_dev +
    // sum_cells V U(Theta)` (`total_lagrangian::Elastic::mixed_stress`).
    let cell_theta = std::env::var("EMERGE_PROBE_THETA").as_deref() == Ok("cell");
    let cell_of: Vec<usize> = reference
        .iter()
        .map(|q| q.x.floor() as usize + layout.nx * q.y.floor() as usize)
        .collect();
    let cell_count = layout.nx * layout.ny;
    crate::spacetime::total_lagrangian::hooks::set_cells(
        cell_theta.then(|| (cell_of.clone(), cell_count)),
    );
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        let kinetic: f64 = (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared())
            .sum();
        if cell_theta {
            let mut sums = vec![(0.0f64, 0.0f64); cell_count];
            for p in 0..n {
                sums[cell_of[p]].0 += v0[p];
                sums[cell_of[p]].1 += v0[p] * f[p].determinant();
            }
            let volumetric: f64 = sums
                .iter()
                .filter(|s| s.0 > 0.0)
                .map(|&(volume, j)| volume * 0.5 * solid.bulk * (j / volume - 1.0).powi(2))
                .sum();
            kinetic
                + (0..n)
                    .map(|p| v0[p] * solid.deviatoric_energy(f[p]))
                    .sum::<f64>()
                + volumetric
        } else {
            kinetic + (0..n).map(|p| v0[p] * solid.energy(f[p])).sum::<f64>()
        }
    };
    // Every particle, and the interior: at least two cells from the
    // boundary, out of reach of the held faces beyond it.
    let inside: Vec<bool> = reference
        .iter()
        .map(|q| {
            let d = (q.x - corner)
                .min(corner + side - q.x)
                .min(q.y - corner)
                .min(corner + side - q.y);
            d >= 2.0
        })
        .collect();
    let scales = |set: &dyn Fn(usize) -> bool| -> [f64; 3] {
        [
            norm(
                &mut (0..n)
                    .filter(|&p| set(p))
                    .map(|p| v0[p] * u(reference[p], peak).length_squared()),
            ),
            norm(
                &mut (0..n)
                    .filter(|&p| set(p))
                    .map(|p| v0[p] * v(reference[p], 0.0).length_squared()),
            ),
            norm(&mut (0..n).filter(|&p| set(p)).map(|p| {
                let s = stress(reference[p], peak);
                v0[p] * (s.x_axis.length_squared() + s.y_axis.length_squared())
            })),
        ]
    };
    let all_scales = [u_scale, v_scale, p_scale];
    // Every frame's errors, and `F`'s own, when asked.
    let history = std::env::var("EMERGE_PROBE_MS_HISTORY").is_ok();
    let f_scale = norm(&mut (0..n).map(|p| {
        let d = grad(reference[p], peak) - DMat2::IDENTITY;
        v0[p] * (d.x_axis.length_squared() + d.y_axis.length_squared())
    }));
    let inner_scales = scales(&|p| inside[p]);
    let edge_scales = scales(&|p| !inside[p]);
    let errors = |x: &[DVec2], vel: &[DVec2], f: &[DMat2], t: f64| -> [f64; 9] {
        let part = |set: &dyn Fn(usize) -> bool, scale: [f64; 3]| -> [f64; 3] {
            let du = norm(
                &mut (0..n)
                    .filter(|&p| set(p))
                    .map(|p| v0[p] * (x[p] - reference[p] - u(reference[p], t)).length_squared()),
            );
            let dv = norm(
                &mut (0..n)
                    .filter(|&p| set(p))
                    .map(|p| v0[p] * (vel[p] - v(reference[p], t)).length_squared()),
            );
            let dp = norm(&mut (0..n).filter(|&p| set(p)).map(|p| {
                let d = solid.first_pk(f[p]) - stress(reference[p], t);
                v0[p] * (d.x_axis.length_squared() + d.y_axis.length_squared())
            }));
            [du / scale[0], dv / scale[1], dp / scale[2]]
        };
        let all = part(&|_| true, all_scales);
        let inner = part(&|p| inside[p], inner_scales);
        let edge = part(&|p| !inside[p], edge_scales);
        [
            all[0], all[1], all[2], inner[0], inner[1], inner[2], edge[0], edge[1], edge[2],
        ]
    };
    // The divergence's difference step against half of it, at the peak.
    let step_check = reference
        .iter()
        .step_by(7)
        .map(|&q| {
            let a = divergence(q, peak, delta);
            let b = divergence(q, peak, 0.5 * delta);
            (a - b).length() / b.length().max(f64::MIN_POSITIVE)
        })
        .fold(0.0f64, f64::max);
    let e0 = energy(&vel, &f);
    let (mut work, mut largest_energy) = (0.0f64, e0);
    let mut worst = [0.0f64; 9];
    let (mut newton, mut gmres, mut most, mut unconverged, mut cost) =
        (0u64, 0u64, 0u32, 0u32, 0.0f64);
    let mut strain = 0.0f64;
    // With `EMERGE_PROBE_MS_BOUNDARY=web` the rollers are the weight of
    // the web-splines (`LagrangianCache::weighted`) instead of held faces:
    // per component, the product of the distances to the two planes the
    // held faces hold it on, over the side (slope one at each plane).
    let web = std::env::var("EMERGE_PROBE_MS_BOUNDARY").as_deref() == Ok("web");
    if web && !lagrangian {
        return "grid rebuilt: skipped, the weighted basis is built on the reference grid".into();
    }
    let weight = |a: usize, q: DVec2| -> (f64, DVec2) {
        let s = if a == 0 { q.x } else { q.y };
        let slope = (hi + lo - 2.0 * s) / side;
        let along = if a == 0 { DVec2::X } else { DVec2::Y };
        ((s - lo) * (hi - s) / side, slope * along)
    };
    let free = |_: usize, _: usize, _: usize| -> Option<f64> { None };
    let walls: crate::spacetime::total_lagrangian::Held<'_> = if web { &free } else { &held };
    let mut cache = web.then(|| {
        crate::spacetime::total_lagrangian::LagrangianCache::weighted(
            &layout,
            &reference,
            (&mass, &v0),
            settings.occupation,
            &weight,
        )
    });
    // With `EMERGE_PROBE_MS_TIME=eg2` the step in time is Gross's eG(2)
    // (`total_lagrangian::step_total_lagrangian_eg2`), on the reference grid only.
    let eg2 = std::env::var("EMERGE_PROBE_MS_TIME").as_deref() == Ok("eg2");
    if eg2 && !lagrangian {
        return "grid rebuilt: skipped, eG(2) is built on the reference grid".into();
    }
    let gauss_times = [0.5 - 3.0f64.sqrt() / 6.0, 0.5 + 3.0f64.sqrt() / 6.0];
    // With `EMERGE_PROBE_MS_STOP` an unconverged step is not committed:
    // the run stops there and says so.
    let stop = std::env::var("EMERGE_PROBE_MS_STOP").is_ok();
    let averaged = std::env::var("EMERGE_PROBE_MS_LOAD").as_deref() == Ok("average");
    // With `EMERGE_PROBE_DG=trapezoid` the discrete gradient's base stress
    // is the two ends' average (`total_lagrangian::hooks::set_trapezoid`).
    let gradient = std::env::var("EMERGE_PROBE_DG").unwrap_or_default();
    let trapezoid = !gradient.is_empty();
    crate::spacetime::total_lagrangian::hooks::set_trapezoid(
        gradient == "trapezoid"
            || gradient == "split_trapezoid"
            || gradient == "cofactor_trapezoid",
    );
    crate::spacetime::total_lagrangian::hooks::set_cofactor(gradient.starts_with("cofactor"));
    // `EMERGE_PROBE_VOLUMETRIC_BASE=midpoint|trapezoid` overrides the
    // split's volumetric base alone.
    crate::spacetime::total_lagrangian::hooks::set_volumetric_base(
        match std::env::var("EMERGE_PROBE_VOLUMETRIC_BASE").as_deref() {
            Ok("midpoint") => 1,
            Ok("trapezoid") => 2,
            _ => 0,
        },
    );
    crate::spacetime::total_lagrangian::hooks::set_trapezoid_p(gradient == "trapezoid_p");
    crate::spacetime::total_lagrangian::hooks::set_split(gradient.starts_with("split"));
    let mut stopped = None;
    // The heat the viscous stress books, in total and its least step.
    let (mut heat, mut least_heat) = (0.0f64, f64::INFINITY);
    // With `EMERGE_PROBE_MS_PATTERN` every frame's velocity error is also
    // described: its correlation with the previous frame's, the share of
    // its half-cell averages in the checkerboard and in the two stripe
    // patterns, its share left after a four-neighbour smoothing, and its
    // RMS by distance to the boundary in fractions of the side.
    let pattern = std::env::var("EMERGE_PROBE_MS_PATTERN").is_ok();
    let mut previous: Option<Vec<DVec2>> = None;
    let mut t = 0.0;
    // With `EMERGE_PROBE_MS_RESET` every step starts from the exact state
    // (positions, velocities, deformation gradients): each step's error is
    // then that step's own, its local truncation error.
    let reset = std::env::var("EMERGE_PROBE_MS_RESET").is_ok();
    for step in 0..steps {
        if reset {
            for p in 0..n {
                x[p] = reference[p] + u(reference[p], t);
                vel[p] = v(reference[p], t);
                f[p] = grad(reference[p], t);
            }
        }
        // The load over the step: sampled at its midpoint, or with
        // `EMERGE_PROBE_MS_LOAD=average` the mean of its two ends, as
        // Gonzalez, CMAME 190 (2000), eq. 10 writes `b_{n+1/2}` (the work
        // stays the load times the step's displacement).
        let b: Vec<DVec2> = if averaged {
            reference
                .iter()
                .map(|&q| 0.5 * (body(q, t) + body(q, t + dt)))
                .collect()
        } else {
            reference.iter().map(|&q| body(q, t + 0.5 * dt)).collect()
        };
        let before = x.clone();
        let kept = stop.then(|| (vel.clone(), f.clone()));
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut vel,
            f: &mut f,
            mass: &mass,
            v0: &v0,
        };
        let mut step_work = None;
        let report = if eg2 {
            let at = gauss_times.map(|xi| {
                reference
                    .iter()
                    .map(|&q| body(q, t + xi * dt))
                    .collect::<Vec<_>>()
            });
            let (report, done) = crate::spacetime::total_lagrangian::step_total_lagrangian_eg2(
                &layout,
                particles,
                &reference,
                &solids,
                ([&at[0], &at[1]], dt),
                (walls, &settings),
                &mut cache,
            );
            step_work = Some(done);
            report
        } else if lagrangian {
            crate::spacetime::total_lagrangian::step_total_lagrangian(
                &layout,
                particles,
                &reference,
                &solids,
                (DVec2::ZERO, Some(&b), dt),
                (walls, &settings),
                &mut cache,
            )
        } else {
            crate::spacetime::total_lagrangian::step_dirichlet(
                &layout,
                particles,
                &solids,
                (&b, dt),
                (&held, &settings),
            )
        };
        if let Some((kept_v, kept_f)) = kept.filter(|_| !report.converged) {
            x = before;
            vel = kept_v;
            f = kept_f;
            newton += u64::from(report.newton);
            gmres += u64::from(report.gmres);
            most = most.max(report.gmres);
            unconverged += 1;
            cost += report.phases.iter().sum::<f64>();
            stopped = Some((
                step + 1,
                report.residual,
                report.newton,
                report.start_residual,
            ));
            break;
        }
        t += dt;
        if beta > 0.0 {
            let booked = crate::spacetime::total_lagrangian::hooks::viscous_work();
            heat += booked;
            least_heat = least_heat.min(booked);
        }
        work += step_work.unwrap_or_else(|| {
            (0..n)
                .map(|p| mass[p] * b[p].dot(x[p] - before[p]))
                .sum::<f64>()
        });
        newton += u64::from(report.newton);
        gmres += u64::from(report.gmres);
        most = most.max(report.gmres);
        unconverged += u32::from(!report.converged);
        cost += report.phases.iter().sum::<f64>();
        if (step + 1) % per_frame == 0 || step + 1 == steps {
            let now = errors(&x, &vel, &f, t);
            if pattern {
                let error: Vec<DVec2> = (0..n).map(|p| vel[p] - v(reference[p], t)).collect();
                let size = |e: &[DVec2]| e.iter().map(|d| d.length_squared()).sum::<f64>().sqrt();
                let correlation = previous.as_ref().map_or(f64::NAN, |old: &Vec<DVec2>| {
                    let dot: f64 = error.iter().zip(old).map(|(a, b)| a.dot(*b)).sum();
                    dot / (size(&error) * size(old)).max(f64::MIN_POSITIVE)
                });
                let halves = 2 * cells;
                let mut cell = vec![DVec2::ZERO; halves * halves];
                let mut count = vec![0usize; halves * halves];
                for (p, e) in error.iter().enumerate() {
                    let q = reference[p] - DVec2::splat(corner);
                    let (i, j) = (
                        ((q.x / 0.5).floor() as usize).min(halves - 1),
                        ((q.y / 0.5).floor() as usize).min(halves - 1),
                    );
                    cell[i + halves * j] += *e;
                    count[i + halves * j] += 1;
                }
                for (c, &k) in cell.iter_mut().zip(&count) {
                    *c /= k.max(1) as f64;
                }
                let total: f64 = cell.iter().map(|c| c.length_squared()).sum();
                let share = |sign: &dyn Fn(usize, usize) -> f64| {
                    let mut sum = DVec2::ZERO;
                    for j in 0..halves {
                        for i in 0..halves {
                            sum += sign(i, j) * cell[i + halves * j];
                        }
                    }
                    sum.length_squared() / (halves * halves) as f64 / total.max(f64::MIN_POSITIVE)
                };
                let parity = |k: usize| if k.is_multiple_of(2) { 1.0 } else { -1.0 };
                let checker = share(&|i, j| parity(i + j));
                let stripes = (share(&|i, _| parity(i)), share(&|_, j| parity(j)));
                let mut rough = 0.0;
                for j in 0..halves {
                    for i in 0..halves {
                        let mut mean = DVec2::ZERO;
                        let mut k = 0.0;
                        for (di, dj) in [(-1i64, 0i64), (1, 0), (0, -1), (0, 1)] {
                            let (a, c) = (i as i64 + di, j as i64 + dj);
                            if a >= 0 && c >= 0 && (a as usize) < halves && (c as usize) < halves {
                                mean += cell[a as usize + halves * c as usize];
                                k += 1.0;
                            }
                        }
                        rough += (cell[i + halves * j] - mean / k).length_squared();
                    }
                }
                let bands = [0.0, 1.0 / 16.0, 1.0 / 8.0, 1.0 / 4.0, f64::INFINITY];
                let band = |lo: f64, hi: f64| {
                    let (mut sum, mut k) = (0.0, 0.0);
                    for (p, e) in error.iter().enumerate() {
                        let q = reference[p] - DVec2::splat(corner);
                        let d = q.x.min(side - q.x).min(q.y).min(side - q.y) / side;
                        if d >= lo && d < hi {
                            sum += v0[p] * e.length_squared();
                            k += v0[p];
                        }
                    }
                    (sum / f64::max(k, f64::MIN_POSITIVE)).sqrt()
                        / (v_scale / v0.iter().sum::<f64>().sqrt())
                };
                println!(
                    "  pattern t {t:.4}: vel error {:.3e}, correlation with the last frame {correlation:+.3}, half-cell checkerboard {:.3}, stripes x {:.3} y {:.3}, left after smoothing {:.3}; RMS by distance to the boundary (side fractions 0-1/16, 1/16-1/8, 1/8-1/4, >1/4) {:.3e} {:.3e} {:.3e} {:.3e}",
                    now[1],
                    checker,
                    stripes.0,
                    stripes.1,
                    rough / total.max(f64::MIN_POSITIVE),
                    band(bands[0], bands[1]),
                    band(bands[1], bands[2]),
                    band(bands[2], bands[3]),
                    band(bands[3], bands[4])
                );
                previous = Some(error);
            }
            if history {
                let f_error = norm(&mut (0..n).map(|p| {
                    let d = f[p] - grad(reference[p], t);
                    v0[p] * (d.x_axis.length_squared() + d.y_axis.length_squared())
                })) / f_scale;
                let mean = (0..n)
                    .map(|p| x[p] - reference[p] - u(reference[p], t))
                    .sum::<DVec2>()
                    / n as f64;
                // The first column's error along x, against the middle
                // column's: the slip at the boundary carried inside.
                let column = |at: f64| {
                    let set: Vec<usize> = (0..n)
                        .filter(|&p| (reference[p].x - corner - at).abs() < 0.1)
                        .collect();
                    set.iter()
                        .map(|&p| (x[p] - reference[p] - u(reference[p], t)).x)
                        .sum::<f64>()
                        / set.len().max(1) as f64
                };
                let (edge, middle) = (column(0.25), column(0.5 * side - 0.25));
                println!(
                    "  t {t:.4} (w t {:.3}): disp {:.3e} vel {:.3e} P {:.3e} F {f_error:.3e}, interior disp {:.3e} vel {:.3e}; mean displacement error ({:+.2e}, {:+.2e}) of the scale {:.2e}; x error first column {edge:+.2e}, middle column {middle:+.2e}",
                    omega * t,
                    now[0],
                    now[1],
                    now[2],
                    now[3],
                    now[4],
                    mean.x,
                    mean.y,
                    u_scale / (n as f64).sqrt()
                );
            }
            for (w, e) in worst.iter_mut().zip(now) {
                *w = if e.is_nan() || w.is_nan() {
                    f64::NAN
                } else {
                    w.max(e)
                };
            }
            largest_energy = largest_energy.max(energy(&vel, &f));
            for m in &f {
                let d = *m - DMat2::IDENTITY;
                strain = strain.max((d.x_axis.length_squared() + d.y_axis.length_squared()).sqrt());
            }
        }
    }
    let end = errors(&x, &vel, &f, t);
    let balance = (energy(&vel, &f) - e0 - work + heat) / largest_energy;
    let realism = |e: f64| if e.is_nan() { 0.0 } else { (1.0 - e).max(0.0) };
    let taken = stopped.map_or(steps, |(k, ..)| k);
    let per = taken.max(1) as f64;
    let ran = match stopped {
        Some((k, residual, newton, start)) => format!(
            "STOPPED at step {k} of {steps} (unconverged, not committed: residual {residual:.2e} after {newton} Newton from {start:.2e}), {} committed",
            k - 1
        ),
        None => format!("{steps} steps"),
    };
    let quadrature = if gauss {
        let (nodes, node_weights) = gauss_legendre(order);
        // The rule against the monomials it must integrate exactly.
        let check = (0..2 * order as i32)
            .map(|d| {
                let sum: f64 = nodes
                    .iter()
                    .zip(&node_weights)
                    .map(|(t, w)| w * t.powi(d))
                    .sum();
                (sum - 1.0 / f64::from(d + 1)).abs()
            })
            .fold(0.0f64, f64::max);
        format!(", Gauss {order}x{order} per half cell ({n} points, rule check {check:.1e})")
    } else {
        format!(", particles 2x2 per cell ({n})")
    };
    let quadrature = if beta > 0.0 {
        format!(
            "{quadrature}, Kelvin-Voigt zeta {zeta} at 1 kHz (heat {:.3e} of the largest energy, least step {:+.2e})",
            heat / largest_energy,
            least_heat / largest_energy
        )
    } else {
        quadrature
    };
    format!(
        "{}{}{}{}{quadrature} at {substep}, h = L/{cells}, G/L {amplitude}, w dt_frame {omega_frame} (divergence step check {step_check:.1e}): {ran} to t = {t:.4} s ({:.2} periods); error end disp {:.2e} vel {:.2e} P {:.2e}, worst disp {:.2e} vel {:.2e} P {:.2e}; realism at worst disp {:.1} % vel {:.1} % P {:.1} %; interior (2 cells from the boundary) error end disp {:.2e} vel {:.2e} P {:.2e}, realism at worst disp {:.1} % vel {:.1} % P {:.1} %; edge (within 2 cells) realism at worst disp {:.1} % vel {:.1} % P {:.1} %; energy balance {balance:+.2e}; Newton {:.2} per step, GMRES {:.1} per step and at most {most}, unconverged {unconverged}; {:.1} s per simulated s; largest |F - I| {strain:.3}",
        if lagrangian {
            "reference grid"
        } else {
            "grid rebuilt"
        },
        match (squared, cell_theta) {
            (true, true) => ", sin^2 profile, Theta per cell",
            (true, false) => ", sin^2 profile",
            (false, true) => ", Theta per cell",
            (false, false) => "",
        },
        if web {
            ", weighted walls"
        } else {
            ", held walls"
        },
        if eg2 {
            ", eG(2) in time"
        } else {
            match (averaged, trapezoid) {
                (true, true) if gradient == "trapezoid_p" => {
                    ", load averaged over the step's ends, trapezoid gradient in P"
                }
                (true, true) if gradient == "split" => {
                    ", load averaged over the step's ends, split gradient"
                }
                (true, true) if gradient == "split_trapezoid" => {
                    ", load averaged over the step's ends, split trapezoid gradient"
                }
                (true, true) if gradient == "cofactor" => {
                    ", load averaged over the step's ends, split cofactor gradient"
                }
                (true, true) if gradient == "cofactor_trapezoid" => {
                    ", load averaged over the step's ends, split cofactor trapezoid gradient"
                }
                (true, true) => ", load averaged over the step's ends, trapezoid gradient",
                (true, false) => ", load averaged over the step's ends",
                (false, true) => ", trapezoid gradient",
                (false, false) => "",
            }
        },
        t / period,
        end[0],
        end[1],
        end[2],
        worst[0],
        worst[1],
        worst[2],
        100.0 * realism(worst[0]),
        100.0 * realism(worst[1]),
        100.0 * realism(worst[2]),
        end[3],
        end[4],
        end[5],
        100.0 * realism(worst[3]),
        100.0 * realism(worst[4]),
        100.0 * realism(worst[5]),
        100.0 * realism(worst[6]),
        100.0 * realism(worst[7]),
        100.0 * realism(worst[8]),
        newton as f64 / per,
        gmres as f64 / per,
        cost / t
    )
}

/// The Gauss-Legendre rule of `n` points (1 to 5) on `[0, 1]`: nodes and
/// weights, from the closed forms of the roots of the Legendre polynomial
/// `P_n` on `[-1, 1]` mapped by `t = (x + 1) / 2` (weights halved). Exact
/// for polynomials of degree up to `2 n - 1`, which the probe checks.
fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let (x, w): (Vec<f64>, Vec<f64>) = match n {
        1 => (vec![0.0], vec![2.0]),
        2 => {
            let a = (1.0f64 / 3.0).sqrt();
            (vec![-a, a], vec![1.0, 1.0])
        }
        3 => {
            let a = (3.0f64 / 5.0).sqrt();
            (vec![-a, 0.0, a], vec![5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
        }
        4 => {
            let r = 2.0 / 7.0 * (6.0f64 / 5.0).sqrt();
            let (a, b) = ((3.0 / 7.0 - r).sqrt(), (3.0 / 7.0 + r).sqrt());
            let s = 30.0f64.sqrt();
            let (wa, wb) = ((18.0 + s) / 36.0, (18.0 - s) / 36.0);
            (vec![-b, -a, a, b], vec![wb, wa, wa, wb])
        }
        _ => {
            let r = 2.0 * (10.0f64 / 7.0).sqrt();
            let (a, b) = ((5.0 - r).sqrt() / 3.0, (5.0 + r).sqrt() / 3.0);
            let s = 13.0 * 70.0f64.sqrt();
            let (wa, wb) = ((322.0 + s) / 900.0, (322.0 - s) / 900.0);
            (vec![-b, -a, 0.0, a, b], vec![wb, wa, 128.0 / 225.0, wa, wb])
        }
    };
    (
        x.iter().map(|x| 0.5 * (x + 1.0)).collect(),
        w.iter().map(|w| 0.5 * w).collect(),
    )
}

/// The manufactured solution (`manufactured_run`) over its sweep, a probe:
/// `EMERGE_PROBE_MS_METHODS` (`rebuilt,reference`), `_STEPS` (`frame,p`),
/// `_CELLS` (`8,16,32`), `_AMPLITUDES` (`0.001,0.05,0.1`), `_OMEGAS`
/// (`0.1,0.3,1,3,10`, `omega dt` at one frame), `_CAP` (the longest a P
/// wave run simulates, s, default 0.2), `_PROFILE=square` for the `sin^2`
/// profile, `_HISTORY` to print every frame's errors, `_CORNER` the body's
/// corner in cells (default 4, on a face), `_QUADRATURE=gauss` and
/// `_GAUSS` (points per direction per half cell, default 3) for the
/// reference domain's own integrals, `_BOUNDARY=web` for the rollers as
/// web-spline weights, `_STOP` to stop at an unconverged step without
/// committing it, `_TIME=eg2` for the eG(2) step in time, `_LOAD=average`
/// for the load averaged over each step's ends, `_PATTERN` to describe the
/// velocity error's pattern in space and time, `_PERIODS` the periods run
/// (default 1); `EMERGE_PROBE_DG=trapezoid` for the trapezoid base of the
/// discrete gradient (`trapezoid_p` the same in `P`, a diagnostic; `split`
/// and `split_trapezoid` Gonzalez's deviatoric/volumetric split, `cofactor`
/// and `cofactor_trapezoid` the same with the volumetric part `p cof(F_mid)`);
/// `EMERGE_PROBE_MS_NU` the Poisson ratio (default 0.45);
/// `EMERGE_PROBE_THETA=cell` for `Theta` constant per cell;
/// `EMERGE_PROBE_MS_RESET` to start every step from the exact state;
/// `EMERGE_PROBE_VOLUMETRIC_BASE` the split's volumetric base on its own;
/// `EMERGE_PROBE_ZETA` a Kelvin-Voigt damping ratio at 1 kHz;
/// `EMERGE_PROBE_NEWTON_TOLERANCE` (default the step's). First, the checks
/// of the body force: `P` against the energy's gradient by central
/// differences, and the divergence's difference step against half of it.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_manufactured_solution() {
    use glam::{DMat2, DVec2};
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    let numbers = |name: &str, default: &str| -> Vec<f64> {
        list(name, default)
            .iter()
            .filter_map(|s| s.parse().ok())
            .collect()
    };
    // The stress against the energy's gradient, on the strained states the
    // sweep reaches (medium clay at 1 cm: the stress scales, the check does
    // not depend on it).
    {
        let (e, nu, rho, cell_m) = (11.25e6f64, 0.45f64, 1700.0f64, 0.01f64);
        let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
        let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
        let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu);
        let mut worst = 0.0f64;
        for (a, b) in [(1.3, 0.8), (0.75, 1.2), (1.05, 1.31), (0.69, 0.71)] {
            let f = DMat2::from_diagonal(DVec2::new(a, b));
            let p = solid.first_pk(f);
            let eta = 1e-6;
            for col in 0..2 {
                for row in 0..2 {
                    let mut plus = f;
                    let mut minus = f;
                    plus.col_mut(col)[row] += eta;
                    minus.col_mut(col)[row] -= eta;
                    let fd = (solid.energy(plus) - solid.energy(minus)) / (2.0 * eta);
                    let scale = p.x_axis.length().max(p.y_axis.length());
                    worst = worst.max((fd - p.col(col)[row]).abs() / scale);
                }
            }
        }
        println!("check: P = dW/dF by central differences, largest relative gap {worst:.1e}");
    }
    let tolerance: f64 = std::env::var("EMERGE_PROBE_NEWTON_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(
            crate::spacetime::total_lagrangian::ConservingSettings::default().newton_tolerance,
        );
    let cap = numbers("EMERGE_PROBE_MS_CAP", "0.2")[0];
    let squared = std::env::var("EMERGE_PROBE_MS_PROFILE").as_deref() == Ok("square");
    for method in list("EMERGE_PROBE_MS_METHODS", "rebuilt,reference") {
        for step in list("EMERGE_PROBE_MS_STEPS", "frame,p") {
            for cells in numbers("EMERGE_PROBE_MS_CELLS", "8,16,32") {
                for amplitude in numbers("EMERGE_PROBE_MS_AMPLITUDES", "0.001,0.05,0.1") {
                    for omega_frame in numbers("EMERGE_PROBE_MS_OMEGAS", "0.1,0.3,1,3,10") {
                        println!(
                            "{}",
                            manufactured_run(Manufactured {
                                lagrangian: method == "reference",
                                substep: Substep::parse(&step).unwrap_or(Substep::Frame),
                                cells: cells as usize,
                                amplitude,
                                omega_frame,
                                newton_tolerance: tolerance,
                                cap,
                                squared,
                            })
                        );
                    }
                }
            }
        }
    }
}

/// The manufactured solution's failing steps, traced (`total_lagrangian::hooks`):
/// for each case `cells:G/L:w dt_frame` of `EMERGE_PROBE_FAIL_CASES`
/// (default `8:0.1:1,8:0.05:3,8:0.05:10`, the reference grid at one
/// frame), every Newton iteration of the first failing solve and of the
/// solve before it (residual, what GMRES reached, the line search's step,
/// the least `det F`), and the homotopy from `u = 0` that tells whether the
/// step's system has an admissible solution; `EMERGE_PROBE_HOMOTOPY` sets
/// its first load increments (default 4), `EMERGE_PROBE_FOLD` also watches
/// the Jacobian's least eigenvalue along it.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_failing_step() {
    let increments: u32 = std::env::var("EMERGE_PROBE_HOMOTOPY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let cases = std::env::var("EMERGE_PROBE_FAIL_CASES")
        .unwrap_or_else(|_| "8:0.1:1,8:0.05:3,8:0.05:10".into());
    for case in cases.split(',') {
        let parts: Vec<f64> = case
            .split(':')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let [cells, amplitude, omega_frame] = parts[..] else {
            continue;
        };
        crate::spacetime::total_lagrangian::hooks::start(increments);
        crate::spacetime::total_lagrangian::hooks::watch_fold(
            std::env::var("EMERGE_PROBE_FOLD").is_ok(),
        );
        let line = manufactured_run(Manufactured {
            lagrangian: true,
            substep: Substep::Frame,
            cells: cells as usize,
            amplitude,
            omega_frame,
            newton_tolerance: crate::spacetime::total_lagrangian::ConservingSettings::default()
                .newton_tolerance,
            cap: 0.2,
            squared: false,
        });
        let solves = crate::spacetime::total_lagrangian::hooks::finish();
        println!("{line}");
        let failed = |s: &Vec<String>| {
            s.last()
                .is_some_and(|l| l.starts_with("end: converged false"))
        };
        let first = solves.iter().position(failed);
        for (k, solve) in solves.iter().enumerate() {
            let shown = match first {
                Some(f) => k + 1 == f || k == f || (k > f && failed(solve) && k <= f + 1),
                None => k == 0,
            };
            if shown {
                println!("  solve {} of {}:", k + 1, solves.len());
                for l in solve {
                    println!("    {l}");
                }
            } else {
                let homotopy = solve.iter().find(|l| l.starts_with("homotopy"));
                let end = solve.last();
                println!(
                    "  solve {}: {} | {}",
                    k + 1,
                    homotopy.map_or("", String::as_str),
                    end.map_or("", String::as_str)
                );
            }
        }
    }
}

/// One run of the generalized vortex (`probe_generalized_vortex`).
#[derive(Clone, Copy)]
struct Vortex {
    /// The grid kept on the reference configuration
    /// (`total_lagrangian::step_total_lagrangian`), or rebuilt at every step
    /// (`total_lagrangian::step_dirichlet`).
    lagrangian: bool,
    substep: Substep,
    /// The ring's width `Ro - Ri` in cells.
    cells: usize,
    /// `T0`, s: the run lasts `T0`.
    period: f64,
    newton_tolerance: f64,
}

/// The generalized vortex as the TLMPM review states it (de Vaucorbeil,
/// Nguyen, Sinaie, Wu, Adv. Appl. Mech. 53 (2020), section 6.3, eqs.
/// 6.13-6.16, after Kamojjala et al. 2015): the ring `Ri = 0.75 m <= R <=
/// Ro = 1.25 m` turns by `alpha = g(t) h(R)` about its centre, `x = Q(alpha)
/// X`, `h = 1 - 8 s^2 + 16 s^4` with `s = (R - R_bar) / (Ri - Ro)`, `g = G
/// sin(pi t / T0)`, `G = 1`, `E = 1e3 Pa`, `nu = 0.3`, `rho0 = 1000`. The
/// motion is isochoric, `h` and `h'` vanish on both circles (fixed and free
/// of traction), and the faces outside the ring are held at zero both ways,
/// as the review sets the nodes outside the ring. The review's body force
/// is for its material; this engine's is derived the same way as the
/// manufactured solution's (`manufactured_run`): `B = A - Div P` with the
/// step's own `P` (`Elastic::first_pk`), the full divergence by central
/// differences. The review also states a zero initial velocity, which its
/// own `g` contradicts (`g'(0) = pi G / T0`): the run starts from the
/// motion's velocity, the only start the motion is exact for. Errors as in
/// the manufactured solution, over the norms of the exact fields at their
/// peak phase (`t = T0 / 2` for displacement and `P`, `t = 0` for velocity).
fn vortex_run(case: Vortex) -> String {
    use glam::{DMat2, DVec2};
    let Vortex {
        lagrangian,
        substep,
        cells,
        period,
        newton_tolerance,
    } = case;
    let cell_m = 0.5 / cells as f64;
    let (e, nu, rho) = (1e3f64, 0.3f64, 1000.0f64);
    let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu);
    let c_p = (solid.bulk + solid.mu).sqrt();
    let (inner, outer) = (0.75 / cell_m, 1.25 / cell_m);
    let middle = 0.5 * (inner + outer);
    let side = (2.0 * outer).ceil() as usize + 8;
    let layout = MacLayout::new(side, side, 1.0);
    let centre = DVec2::splat(0.5 * side as f64);
    let pi = std::f64::consts::PI;
    let frame = f64::from(FRAME);
    let frames = (period / frame).round().max(1.0) as usize;
    let per_frame = match substep {
        Substep::Frame => 1,
        _ => (frame / f64::from(substep.length(1.0, c_p as f32))).ceil() as usize,
    };
    let dt = period / (frames * per_frame) as f64;
    let steps = frames * per_frame;
    let rate = pi / period;
    // g = G sin(pi t / T0) and its first two time derivatives;
    // `EMERGE_PROBE_VORTEX_G` sets G (default 1, the review's).
    let amplitude: f64 = std::env::var("EMERGE_PROBE_VORTEX_G")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1.0);
    let g = |t: f64| amplitude * (rate * t).sin();
    let g1 = |t: f64| amplitude * rate * (rate * t).cos();
    let g2 = |t: f64| -amplitude * rate * rate * (rate * t).sin();
    // h and h' at a radius.
    let shape = |r: f64| {
        let s = (r - middle) / (inner - outer);
        let w = 1.0 - 4.0 * s * s;
        (w * w, -16.0 * s * w / (inner - outer))
    };
    let turn =
        |a: f64| DMat2::from_cols(DVec2::new(a.cos(), a.sin()), DVec2::new(-a.sin(), a.cos()));
    let perp = |d: DVec2| DVec2::new(-d.y, d.x);
    let u = |q: DVec2, t: f64| {
        let r = q - centre;
        turn(g(t) * shape(r.length()).0) * r - r
    };
    let v = |q: DVec2, t: f64| {
        let r = q - centre;
        let h = shape(r.length()).0;
        g1(t) * h * perp(turn(g(t) * h) * r)
    };
    let acceleration = |q: DVec2, t: f64| {
        let r = q - centre;
        let h = shape(r.length()).0;
        let turned = turn(g(t) * h) * r;
        g2(t) * h * perp(turned) - (g1(t) * h).powi(2) * turned
    };
    let grad = |q: DVec2, t: f64| {
        let r = q - centre;
        let length = r.length();
        let (h, dh) = shape(length);
        let q_turn = turn(g(t) * h);
        let a = perp(q_turn * r);
        let b = r * (g(t) * dh / length);
        q_turn + DMat2::from_cols(a * b.x, a * b.y)
    };
    let stress = |q: DVec2, t: f64| solid.first_pk(grad(q, t));
    let divergence = |q: DVec2, t: f64, delta: f64| {
        let dx = DVec2::new(delta, 0.0);
        let dy = DVec2::new(0.0, delta);
        (stress(q + dx, t).x_axis - stress(q - dx, t).x_axis + stress(q + dy, t).y_axis
            - stress(q - dy, t).y_axis)
            / (2.0 * delta)
    };
    let delta = 1e-3;
    let body = |q: DVec2, t: f64| acceleration(q, t) - divergence(q, t, delta);
    // Every step's solve, when asked.
    let progress = std::env::var("EMERGE_PROBE_VORTEX_PROGRESS").is_ok();
    // No face held: the motion is fixed and free of traction on both
    // circles, so it is exact for a free ring too.
    // `EMERGE_PROBE_VORTEX_BOUNDARY`: `held` (default, the faces outside
    // the ring held), `free` (none held) or `web` (both components weighted
    // by `w = (R - Ri)(Ro - R)/(Ro - Ri)`, zero on both circles, the
    // web-splines of Hollig and Reif 2003 for the clamped ring; no face held).
    let boundary = std::env::var("EMERGE_PROBE_VORTEX_BOUNDARY").unwrap_or_else(|_| {
        if std::env::var("EMERGE_PROBE_VORTEX_FREE").is_ok() {
            "free".into()
        } else {
            "held".into()
        }
    });
    let free = boundary != "held";
    let web = boundary == "web";
    // Quadrature points: particles two by two per cell (those in the ring),
    // or with `EMERGE_PROBE_VORTEX_QUADRATURE=gauss` the ring's own
    // integrals (`ring_quadrature`, refinement depth
    // `EMERGE_PROBE_VORTEX_DEPTH`, default 4).
    let gauss = std::env::var("EMERGE_PROBE_VORTEX_QUADRATURE").as_deref() == Ok("gauss");
    let depth: u32 = std::env::var("EMERGE_PROBE_VORTEX_DEPTH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);
    let (reference, v0): (Vec<DVec2>, Vec<f64>) = if gauss {
        ring_quadrature(centre, (inner, outer), side, depth)
    } else {
        (0..2 * side)
            .flat_map(|j| {
                (0..2 * side)
                    .map(move |i| DVec2::new((i as f64 + 0.5) * 0.5, (j as f64 + 0.5) * 0.5))
            })
            .filter(|&q| {
                let r = (q - centre).length();
                (inner..=outer).contains(&r)
            })
            .map(|q| (q, 0.25))
            .unzip()
    };
    let n = reference.len();
    // The ring's area against the quadrature's: the geometric error.
    let area_error =
        v0.iter().sum::<f64>() / (std::f64::consts::PI * (outer * outer - inner * inner)) - 1.0;
    let mass = v0.clone();
    let solids = vec![solid; n];
    let mut x = reference.clone();
    let mut vel: Vec<DVec2> = reference.iter().map(|&q| v(q, 0.0)).collect();
    let mut f = vec![DMat2::IDENTITY; n];
    let held = |a: usize, i: usize, j: usize| -> Option<f64> {
        let at =
            DVec2::new(i as f64, j as f64) + crate::spacetime::total_lagrangian::lattice_offset(a);
        let r = (at - centre).length();
        (!free && (r < inner || r > outer)).then_some(1.0)
    };
    let settings = crate::spacetime::total_lagrangian::ConservingSettings {
        newton_tolerance,
        occupation: match std::env::var("EMERGE_PROBE_OCCUPATION").as_deref() {
            Ok("none") => None,
            _ => crate::spacetime::total_lagrangian::ConservingSettings::default().occupation,
        },
        ..Default::default()
    };
    // L2 norms weighted by each point's volume.
    let norm = |values: &mut dyn Iterator<Item = f64>| values.sum::<f64>().sqrt();
    let peak = 0.5 * period;
    let u_scale = norm(&mut (0..n).map(|p| v0[p] * u(reference[p], peak).length_squared()));
    let v_scale = norm(&mut (0..n).map(|p| v0[p] * v(reference[p], 0.0).length_squared()));
    let p_scale = norm(&mut (0..n).map(|p| {
        let s = stress(reference[p], peak);
        v0[p] * (s.x_axis.length_squared() + s.y_axis.length_squared())
    }));
    let exact_strain = reference
        .iter()
        .map(|&q| {
            let d = grad(q, peak) - DMat2::IDENTITY;
            (d.x_axis.length_squared() + d.y_axis.length_squared()).sqrt()
        })
        .fold(0.0f64, f64::max);
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0[p] * solid.energy(f[p]))
            .sum()
    };
    let errors = |x: &[DVec2], vel: &[DVec2], f: &[DMat2], t: f64| -> [f64; 3] {
        let du = norm(
            &mut (0..n)
                .map(|p| v0[p] * (x[p] - reference[p] - u(reference[p], t)).length_squared()),
        );
        let dv = norm(&mut (0..n).map(|p| v0[p] * (vel[p] - v(reference[p], t)).length_squared()));
        let dp = norm(&mut (0..n).map(|p| {
            let d = solid.first_pk(f[p]) - stress(reference[p], t);
            v0[p] * (d.x_axis.length_squared() + d.y_axis.length_squared())
        }));
        [du / u_scale, dv / v_scale, dp / p_scale]
    };
    // The divergence's difference step against half of it, at the peak.
    let step_check = reference
        .iter()
        .step_by(7)
        .map(|&q| {
            let a = divergence(q, peak, delta);
            let b = divergence(q, peak, 0.5 * delta);
            (a - b).length() / b.length().max(f64::MIN_POSITIVE)
        })
        .fold(0.0f64, f64::max);
    let e0 = energy(&vel, &f);
    let (mut work, mut largest_energy) = (0.0f64, e0);
    let mut worst = [0.0f64; 3];
    let (mut newton, mut gmres, mut most, mut unconverged, mut cost) =
        (0u64, 0u64, 0u32, 0u32, 0.0f64);
    let mut strain = 0.0f64;
    let mut cache = web.then(|| {
        crate::spacetime::total_lagrangian::LagrangianCache::weighted(
            &layout,
            &reference,
            (&mass, &v0),
            settings.occupation,
            &|_, q| {
                let r = q - centre;
                let length = r.length();
                let w = (length - inner) * (outer - length) / (outer - inner);
                let slope = (outer + inner - 2.0 * length) / (outer - inner);
                (w, r * (slope / length.max(f64::MIN_POSITIVE)))
            },
        )
    });
    // The least det F over the run, the whole ring's and the middle half's
    // (|s| <= 1/4), and an unconverged step stops the run uncommitted with
    // `EMERGE_PROBE_MS_STOP`.
    let (mut least_all, mut least_middle) = (f64::INFINITY, f64::INFINITY);
    // The least det F within a tenth of the width of each circle, and the
    // filled fraction of each point's half cell (its points' volume over
    // the half cell's): where a cut cell sits against the defect.
    let (mut least_inner, mut least_outer) = (f64::INFINITY, f64::INFINITY);
    let half_cell =
        |q: DVec2| (q.x / 0.5).floor() as usize + 2 * side * (q.y / 0.5).floor() as usize;
    let mut filled = vec![0.0f64; 4 * side * side];
    for p in 0..n {
        filled[half_cell(reference[p])] += v0[p] / 0.25;
    }
    let stop = std::env::var("EMERGE_PROBE_MS_STOP").is_ok();
    let mut stopped = None;
    let mut t = 0.0;
    for step in 0..steps {
        let b: Vec<DVec2> = reference.iter().map(|&q| body(q, t + 0.5 * dt)).collect();
        let before = x.clone();
        let kept = stop.then(|| (vel.clone(), f.clone()));
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut vel,
            f: &mut f,
            mass: &mass,
            v0: &v0,
        };
        let report = if lagrangian {
            crate::spacetime::total_lagrangian::step_total_lagrangian(
                &layout,
                particles,
                &reference,
                &solids,
                (DVec2::ZERO, Some(&b), dt),
                (
                    if web {
                        &|_: usize, _: usize, _: usize| -> Option<f64> { None }
                    } else {
                        &held
                    },
                    &settings,
                ),
                &mut cache,
            )
        } else {
            crate::spacetime::total_lagrangian::step_dirichlet(
                &layout,
                particles,
                &solids,
                (&b, dt),
                (&held, &settings),
            )
        };
        if let Some((kept_v, kept_f)) = kept.filter(|_| !report.converged) {
            x = before;
            vel = kept_v;
            f = kept_f;
            unconverged += 1;
            stopped = Some(step + 1);
            break;
        }
        t += dt;
        for p in 0..n {
            let det = f[p].determinant();
            least_all = least_all.min(det);
            let s_p = ((reference[p] - centre).length() - middle) / (outer - inner);
            if s_p <= -0.4 {
                least_inner = least_inner.min(det);
            }
            if s_p >= 0.4 {
                least_outer = least_outer.min(det);
            }
            if s_p.abs() <= 0.25 {
                least_middle = least_middle.min(det);
            }
        }
        work += (0..n)
            .map(|p| mass[p] * b[p].dot(x[p] - before[p]))
            .sum::<f64>();
        newton += u64::from(report.newton);
        gmres += u64::from(report.gmres);
        most = most.max(report.gmres);
        unconverged += u32::from(!report.converged);
        cost += report.phases.iter().sum::<f64>();
        if progress {
            let (least, at) = f
                .iter()
                .enumerate()
                .map(|(p, m)| (m.determinant(), p))
                .fold((f64::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a });
            let s_at = ((reference[at] - centre).length() - middle) / (outer - inner);
            // The least det F in the ring's middle half, |s| <= 1/4, the
            // same band of the body at every resolution, and where it sits.
            let (inside, inside_at) = (0..n)
                .filter(|&p| {
                    let r = (reference[p] - centre).length();
                    ((r - middle) / (outer - inner)).abs() <= 0.25
                })
                .map(|p| (f[p].determinant(), p))
                .fold((f64::INFINITY, 0), |a, b| if b.0 < a.0 { b } else { a });
            let s_inside = ((reference[inside_at] - centre).length() - middle) / (outer - inner);
            let edge = |near: &dyn Fn(f64) -> bool| {
                (0..n)
                    .filter(|&p| {
                        near(((reference[p] - centre).length() - middle) / (outer - inner))
                    })
                    .map(|p| f[p].determinant())
                    .fold(f64::INFINITY, f64::min)
            };
            let (edge_in, edge_out) = (edge(&|s| s <= -0.4), edge(&|s| s >= 0.4));
            let fill = filled[half_cell(reference[at])];
            let offset = reference[at] - centre;
            let angle = offset.y.atan2(offset.x).to_degrees();
            println!(
                "  step {} t {t:.4}: Newton {} GMRES {} converged {} residual {:.2e} start residual {:.2e}, least det F {least:.4} at s = {s_at:+.3} (the circles at -0.5 and +0.5), {inside:.4} at s = {s_inside:+.3} in the middle half; edges inner {edge_in:.4} outer {edge_out:.4}; least point's half cell filled {fill:.3}, angle {angle:.0} deg",
                step + 1,
                report.newton,
                report.gmres,
                report.converged,
                report.residual,
                report.start_residual
            );
        }
        if (step + 1) % per_frame == 0 || step + 1 == steps {
            let now = errors(&x, &vel, &f, t);
            for (w, e) in worst.iter_mut().zip(now) {
                *w = if e.is_nan() || w.is_nan() {
                    f64::NAN
                } else {
                    w.max(e)
                };
            }
            largest_energy = largest_energy.max(energy(&vel, &f));
            for m in &f {
                let d = *m - DMat2::IDENTITY;
                strain = strain.max((d.x_axis.length_squared() + d.y_axis.length_squared()).sqrt());
            }
        }
    }
    let end = errors(&x, &vel, &f, t);
    let balance = (energy(&vel, &f) - e0 - work) / largest_energy;
    let realism = |e: f64| if e.is_nan() { 0.0 } else { (1.0 - e).max(0.0) };
    let per = steps.max(1) as f64;
    let boundary_label = format!(", {boundary} boundary");
    let quadrature_label = if gauss {
        format!(", ring quadrature depth {depth} ({n} points, area error {area_error:+.1e})")
    } else {
        format!(", particles ({n}, area error {area_error:+.1e})")
    };
    let stopped_label = stopped.map_or(String::new(), |k| {
        format!(" STOPPED at step {k} (unconverged, not committed);")
    });
    format!(
        "vortex G {amplitude}, {}{boundary_label}{quadrature_label} at {substep}, h = width/{cells}, T0 {period} s (w dt {:.3}, {n} particles, divergence step check {step_check:.1e}): {steps} steps to t = {t:.4} s; error end disp {:.2e} vel {:.2e} P {:.2e}, worst disp {:.2e} vel {:.2e} P {:.2e}; realism at worst disp {:.1} % vel {:.1} % P {:.1} %; energy balance {balance:+.2e}; Newton {:.2} per step, GMRES {:.1} per step and at most {most}, unconverged {unconverged}; {:.1} s per simulated s; largest |F - I| {strain:.3} (exact {exact_strain:.3}); least det F over the run {least_all:.4}, middle half {least_middle:.4}, inner edge {least_inner:.4}, outer edge {least_outer:.4} (exact 1);{stopped_label}",
        if lagrangian {
            "reference grid"
        } else {
            "grid rebuilt"
        },
        rate * dt,
        end[0],
        end[1],
        end[2],
        worst[0],
        worst[1],
        worst[2],
        100.0 * realism(worst[0]),
        100.0 * realism(worst[1]),
        100.0 * realism(worst[2]),
        newton as f64 / per,
        gmres as f64 / per,
        cost / t
    )
}

/// Quadrature points of the ring `Ri <= R <= Ro` about `centre`: every half
/// cell of the grid (both lattices' B-splines are polynomials on it) wholly
/// inside the ring takes a tensor Gauss-Legendre rule (3x3 by default,
/// `EMERGE_PROBE_VORTEX_GAUSS`); a half
/// cell the circles cut is split in four, down to `depth` levels, and the
/// rule's points of the last level are kept where they lie in the ring.
/// Composite Gauss quadrature over an adaptive subdivision of the cut
/// cells; its geometric error is the run's measured area error.
fn ring_quadrature(
    centre: glam::DVec2,
    (inner, outer): (f64, f64),
    side: usize,
    depth: u32,
) -> (Vec<glam::DVec2>, Vec<f64>) {
    use glam::DVec2;
    // `EMERGE_PROBE_VORTEX_GAUSS` points per direction (default 3).
    let order: usize = std::env::var("EMERGE_PROBE_VORTEX_GAUSS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(3);
    let (nodes, weights) = gauss_legendre(order);
    let (mut points, mut volumes) = (Vec::new(), Vec::new());
    let mut stack: Vec<(DVec2, f64, u32)> = (0..2 * side)
        .flat_map(|j| (0..2 * side).map(move |i| (DVec2::new(i as f64, j as f64) * 0.5, 0.5, 0)))
        .collect();
    while let Some((lo, size, level)) = stack.pop() {
        let nearest = DVec2::new(
            centre.x.clamp(lo.x, lo.x + size),
            centre.y.clamp(lo.y, lo.y + size),
        );
        let near = (nearest - centre).length();
        let far = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
            .iter()
            .map(|&(a, b)| (lo + size * DVec2::new(a, b) - centre).length())
            .fold(0.0f64, f64::max);
        if far <= inner || near >= outer {
            continue;
        }
        let whole = near >= inner && far <= outer;
        if whole || level >= depth {
            for (b, wb) in nodes.iter().zip(&weights) {
                for (a, wa) in nodes.iter().zip(&weights) {
                    let q = lo + size * DVec2::new(*a, *b);
                    let r = (q - centre).length();
                    if whole || (inner..=outer).contains(&r) {
                        points.push(q);
                        volumes.push(size * size * wa * wb);
                    }
                }
            }
        } else {
            let half = 0.5 * size;
            for (a, b) in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)] {
                stack.push((lo + half * DVec2::new(a, b), half, level + 1));
            }
        }
    }
    (points, volumes)
}

/// The generalized vortex (`vortex_run`) over its sweep, a probe:
/// `EMERGE_PROBE_VORTEX_METHODS` (`reference,rebuilt`), `_STEPS` (`frame`),
/// `_CELLS` (`8,16,32`, cells across the ring), `_PERIODS` (`1,0.1,0.05`,
/// `T0` in s, whole frames), `_PROGRESS` to print every step's solve,
/// `_FREE` to hold no face;
/// `EMERGE_PROBE_NEWTON_TOLERANCE` (default the step's);
/// `EMERGE_PROBE_OCCUPATION=none` for the basis without the occupation
/// threshold; `EMERGE_PROBE_VORTEX_QUADRATURE=gauss` (`_DEPTH`) for the ring's
/// own integrals, `EMERGE_PROBE_VORTEX_BOUNDARY=held|free|web`,
/// `EMERGE_PROBE_VORTEX_G` the amplitude G (default 1),
/// `EMERGE_PROBE_MS_STOP` to stop at an unconverged step.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_generalized_vortex() {
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    let tolerance: f64 = std::env::var("EMERGE_PROBE_NEWTON_TOLERANCE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(
            crate::spacetime::total_lagrangian::ConservingSettings::default().newton_tolerance,
        );
    for method in list("EMERGE_PROBE_VORTEX_METHODS", "reference,rebuilt") {
        for step in list("EMERGE_PROBE_VORTEX_STEPS", "frame") {
            for cells in list("EMERGE_PROBE_VORTEX_CELLS", "8,16,32") {
                for period in list("EMERGE_PROBE_VORTEX_PERIODS", "1,0.1,0.05") {
                    let (Ok(cells), Ok(period)) = (cells.parse(), period.parse()) else {
                        continue;
                    };
                    println!(
                        "{}",
                        vortex_run(Vortex {
                            lagrangian: method == "reference",
                            substep: Substep::parse(&step).unwrap_or(Substep::Frame),
                            cells,
                            period,
                            newton_tolerance: tolerance,
                        })
                    );
                }
            }
        }
    }
}

/// The free body for one row at `k` cells per centimetre.
fn free_body(row: &MaterialRow, k: usize, report: &mut Report) {
    free_body_at(row, k, None, report);
}

/// The free body, moved by the conserving step at `conserving`'s substep
/// when it names one, and then judged by the conserving step's bounds: the
/// energy within 1 percent either way.
fn free_body_at(row: &MaterialRow, k: usize, conserving: Option<Substep>, report: &mut Report) {
    let kf = k as f32;
    let label = match conserving {
        None => format!("{}, free at {} cm", row.name, 1.0 / kf),
        Some(step) => format!(
            "{}, conserving free at {} cm, substep {step}",
            row.name,
            1.0 / kf
        ),
    };
    let tank = (Vec2::splat(4.0 * kf), Vec2::splat(68.0 * kf));
    let mut rng = LcgRng::new(17);
    // Without shear stiffness the block translates (criteria amendment).
    let translate = row.mu == 0.0;
    let corner = if translate { 8.0 } else { 31.0 };
    let (lo, hi) = (Vec2::splat(corner) * kf, Vec2::splat(corner + 10.0) * kf);
    let x = seed_body(
        row.mu > 0.0,
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
    if let Some(step) = conserving {
        scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings::default());
        let c_p = (scene.sound_speed2.unwrap_or(0.0) + scene.shear.unwrap_or(0.0)).sqrt();
        scene.fixed_dt = Some(step.length(scene.layout.dx, c_p));
    }
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
    let mut lowest_e = f64::INFINITY;
    let clock = Instant::now();
    for _ in 0..(1.0 / FRAME).round() as u32 {
        scene.frame();
        nan |= scene.x.iter().chain(&scene.v).any(|q| !q.is_finite());
        let (_, vcom) = scene.centre_of_mass();
        worst_v = worst_v.max((vcom - vcom0).length() / largest);
        worst_l = worst_l.max((angular(&scene) - l0).abs() / (inertia * s));
        let change = row_energy(&scene) / e0 - 1.0;
        worst_e = worst_e.max(change);
        lowest_e = lowest_e.min(change);
    }
    let st = scene.probe.conserving;
    println!(
        "{label}: centre speed up to {worst_v:.4} of the largest start speed, angular momentum off by {worst_l:.4} of I s, energy from {lowest_e:+.4} to {worst_e:+.4}{}",
        if conserving.is_some() {
            format!(
                ", Newton {} unconverged {}, {:.1} s wall",
                st.0,
                st.4,
                clock.elapsed().as_secs_f32()
            )
        } else {
            String::new()
        }
    );
    if conserving.is_some() {
        report.check(lowest_e >= -0.01, || {
            format!("{label}: energy fell {lowest_e:+.4}")
        });
    }
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

/// The conserving step's free vibration (criteria in the module doc):
/// every solid row, the free body's block, 1 and 0.5 cm, at the explicit
/// P wave limit, four times it and one frame; the energy within 1 percent
/// either way at every frame, the free body's momentum bounds, no NaN.
#[test]
#[ignore = "conserving step, free vibration: run with --ignored --nocapture"]
fn gate_conserving_free_vibration() {
    let mut report = Report::new();
    for row in material_rows().iter().filter(|r| r.mu > 0.0) {
        for k in [1usize, 2] {
            for step in [Substep::PWave, Substep::FourPWaves, Substep::Frame] {
                free_body_at(row, k, Some(step), &mut report);
            }
        }
    }
    report.finish();
}

#[test]
#[ignore = "material matrix, free body: run with --ignored --nocapture"]
fn gate_matrix_free_body() {
    let mut report = Report::new();
    for row in material_rows() {
        for k in [1usize, 2] {
            free_body(&row, k, &mut report);
        }
    }
    report.finish();
}

/// The free body frame by frame for each row: substeps, largest
/// speed, the centre's speed and the energy against the start, stopping
/// after two minutes of wall time. The resolution is `EMERGE_PROBE_K`
/// cells per centimetre. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
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

/// The column at rest at 1 cm and one gravity, frame by frame for the first
/// second: largest particle speed, largest pressure error in cells of head and
/// where, largest move. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
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

/// The free body's block of each solid row moved by the conserving
/// implicit step, at the substep `EMERGE_PROBE_STEP` (`p`, `4p` or `frame`,
/// `Substep::parse`), frame
/// by frame: energy against the start, the centre's speed, Newton and
/// GMRES work, wall time. Stops after five minutes. A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_conserving_free_vibration() {
    let which = std::env::var("EMERGE_PROBE_STEP").unwrap_or_else(|_| "frame".into());
    for row in material_rows().into_iter().filter(|r| r.mu > 0.0) {
        let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
        let mut rng = LcgRng::new(17);
        let (lo, hi) = (Vec2::new(31.0, 31.0), Vec2::new(41.0, 41.0));
        let x = seed(
            (lo.as_ivec2(), hi.as_ivec2()),
            |_| true,
            |_| (f32::INFINITY, Vec2::Y),
            &mut rng,
        );
        let mut scene = Scene::new((72, 72), 0.01, 0.0, tank, x)
            .with_lame(row.lambda / row.rho, row.mu / row.rho);
        let epsilon = std::env::var("EMERGE_PROBE_EPSILON")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0.0);
        scene.conserving = Some(crate::spacetime::total_lagrangian::ConservingSettings {
            epsilon,
            ..Default::default()
        });
        let c_p = (scene.sound_speed2.unwrap_or(0.0) + scene.shear.unwrap_or(0.0)).sqrt();
        let substep = Substep::parse(&which).unwrap_or(Substep::Frame);
        scene.fixed_dt = Some(substep.length(scene.layout.dx, c_p));
        let centre = 0.5 * (lo + hi);
        let s = 0.5 / scene.cell_m / (5.0 * std::f32::consts::SQRT_2);
        for p in 0..scene.x.len() {
            let r = scene.x[p] - centre;
            scene.v[p] = s * Vec2::new(r.y, r.x);
        }
        let e0 = row_energy(&scene);
        let clock = Instant::now();
        for f in 1..=60u32 {
            let frame = scene.frame();
            let (_, vcom) = scene.centre_of_mass();
            let late = clock.elapsed().as_secs() > 900;
            let e = row_energy(&scene) / e0;
            if f <= 3 || f % 6 == 0 || !e.is_finite() || late {
                let st = scene.probe.conserving;
                println!(
                    "{} [{which}] frame {f}: {} substeps, energy {:.6} of the start, centre {:.5} m/s, Newton {} GMRES {} (max {}), residual {:.1e}, {} unconverged, {:.1} s wall, stabilisation took {:.6}",
                    row.name,
                    frame.substeps,
                    e,
                    vcom.length() * scene.cell_m,
                    st.0,
                    st.1,
                    st.2,
                    st.3,
                    st.4,
                    clock.elapsed().as_secs_f32(),
                    scene.probe.stabilisation_loss / e0
                );
            }
            if !e.is_finite() || late {
                break;
            }
        }
    }
}

/// The impact of an elastic bar on a rigid ground, the first benchmark of
/// Doyen, Ern and Piperno, *Time-integration schemes for the finite
/// element dynamic Signorini problem*, SIAM J. Sci. Comput. 33 (2011),
/// section 3.1: a bar of length `L`, undeformed, falls at `v0` onto the
/// ground with no gravity, stays on it `2 L / c0` while a shock crosses it
/// twice, and leaves undeformed at `v0` (their eqs. 26 and 27: the
/// displacement, and the contact pressure `E v0 / c0` throughout the
/// contact). Here a column `width` cells wide with Poisson's ratio 0
/// (`lambda = 0`), whose free sides then carry no lateral stress: the 1D
/// bar, uniaxial, with `E = 2 mu` and `c0^2 = E / rho`. At the small strain
/// `v0 / c0` (`EMERGE_PROBE_BAR_STRAIN`, default 1e-3) the engine's
/// nonlinear energy departs from linear elasticity by about that strain, a
/// declared difference.
///
/// The step is the total Lagrangian conserving one with walls in contact
/// (`total_lagrangian::step_total_lagrangian_contact`), Gauss 3x3 per half cell,
/// the ground a wall at the bottom face's contact points (`width + 2` of
/// them: its ends and the middles of its knot spans, as many as the
/// normal component's coefficients there). By default the contact
/// boundary is massless: those coefficients, whose support holds only the
/// bottom knot span, carry no inertia (Doyen, Ern and Piperno section 7);
/// `EMERGE_PROBE_BAR_MASS=standard` keeps the standard mass, their
/// midpoint with implicit contact (Discretization 4.4).
///
/// Measured against the exact solution at every step's end: the
/// displacement's RMS error over the bar, over `v0 tau_w` (the largest
/// contact displacement, `tau_w = L / c0`); the velocity's, over `v0`; the
/// mean push over the contact steps against `E v0 / c0`; the contact's
/// steps against `2 tau_w`; the take-off velocity (momentum over mass)
/// against `v0`; the energy (`1/2 v M* v` plus the stored energy) at the
/// end against the start, the walls' work, and the vibration left
/// (energy less the centre of mass's kinetic energy; exact: none).
/// Realism is one less each relative error.
fn bar_impact_run(courant: Option<f64>, cells: usize) -> String {
    use glam::{DMat2, DVec2};
    let float = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    // Medium clay's stiffness and density, a 1 m bar.
    let (e, rho) = (
        float("EMERGE_PROBE_BAR_E", 11.25e6),
        float("EMERGE_PROBE_BAR_RHO", 1700.0),
    );
    let length = float("EMERGE_PROBE_BAR_LENGTH", 1.0);
    let width = float("EMERGE_PROBE_BAR_WIDTH", 2.0) as usize;
    let strain = float("EMERGE_PROBE_BAR_STRAIN", 1e-3);
    // `EMERGE_PROBE_BAR_CONTACT=position|velocity|recovered`
    // (`total_lagrangian::ContactMode`); the last two use the standard mass.
    let mode = match std::env::var("EMERGE_PROBE_BAR_CONTACT").as_deref() {
        Ok("velocity") => crate::spacetime::total_lagrangian::ContactMode::Velocity,
        Ok("recovered") => crate::spacetime::total_lagrangian::ContactMode::Recovered,
        _ => crate::spacetime::total_lagrangian::ContactMode::Position,
    };
    let standard = std::env::var("EMERGE_PROBE_BAR_MASS").as_deref() == Ok("standard")
        || mode != crate::spacetime::total_lagrangian::ContactMode::Position;
    let cell_m = length / cells as f64;
    // Per unit density, cells and seconds; `lambda = 0`, so `bulk = mu`.
    let c0 = (e / rho).sqrt() / cell_m;
    let mu = 0.5 * c0 * c0;
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, mu);
    let side = cells as f64;
    let tau = side / c0;
    let v0 = strain * c0;
    let frame = f64::from(FRAME);
    let dt = courant.map_or(frame, |nu| nu / c0);
    // The bottom face on a knot of the normal component's lattice (its
    // coefficients sit on the faces `y = j`, its knots half way), the
    // first impact inside the third step, at the fraction
    // `EMERGE_PROBE_BAR_PHASE` (default 0.37) of it.
    // `EMERGE_PROBE_BAR_OFFSET=dx,dy`: the whole scene, bar and ground,
    // moved by a fraction of a cell (the answer's dependence on where the
    // bar sits on the grid; the exact one has none).
    let offset = std::env::var("EMERGE_PROBE_BAR_OFFSET")
        .ok()
        .and_then(|v| {
            let mut parts = v.split(',').map(|s| s.trim().parse::<f64>().ok());
            Some((parts.next()??, parts.next()??))
        })
        .unwrap_or((0.0, 0.0));
    let (x0, bottom) = (4.0 + offset.0, 4.5 + offset.1);
    let t1 = (2.0 + float("EMERGE_PROBE_BAR_PHASE", 0.37)) * dt;
    let gap = v0 * t1;
    let floor = bottom - gap;
    let t2 = t1 + 2.0 * tau;
    let end = t2 + tau.max(3.0 * dt);
    let steps = (end / dt).ceil() as usize;
    let layout = MacLayout::new(width + 8, cells + 8, 1.0);
    let (nodes, node_weights) = gauss_legendre(3);
    let (mut reference, mut v0s): (Vec<DVec2>, Vec<f64>) = (0..2 * cells)
        .flat_map(|j| (0..2 * width).map(move |i| (i, j)))
        .flat_map(|(i, j)| {
            let (nodes, node_weights) = (&nodes, &node_weights);
            (0..3).flat_map(move |b| {
                (0..3).map(move |a| {
                    (
                        DVec2::new(
                            x0 + (i as f64 + nodes[a]) * 0.5,
                            bottom + (j as f64 + nodes[b]) * 0.5,
                        ),
                        0.25 * node_weights[a] * node_weights[b],
                    )
                })
            })
        })
        .unzip();
    let body = reference.len();
    // The contact points, of no mass and no volume.
    let mut along = vec![x0];
    along.extend((0..width).map(|i| x0 + i as f64 + 0.5));
    along.push(x0 + width as f64);
    for &xc in &along {
        reference.push(DVec2::new(xc, bottom));
        v0s.push(0.0);
    }
    let n = reference.len();
    let mass = v0s.clone();
    let mut solids = vec![solid; n];
    // The normal component's coefficients on the face `y = 4`: support
    // `[2.5, 5.5]`, of which the bar holds only the knot span `[4.5, 5.5]`.
    let row = (bottom - 0.5) as usize;
    crate::spacetime::total_lagrangian::hooks::set_massless((!standard).then(|| {
        (x0 as usize - 1..=x0 as usize + width)
            .map(|i| (1, i, row))
            .collect()
    }));
    // `EMERGE_PROBE_CONTACT_TOL`: the contacts' gap tolerance, cells.
    let mut settings = crate::spacetime::total_lagrangian::ConservingSettings::default();
    settings.contact_tolerance = float("EMERGE_PROBE_CONTACT_TOL", settings.contact_tolerance);
    // `EMERGE_PROBE_FORECAST`: hold contacts from the step their gap closes.
    settings.forecast_activation = std::env::var_os("EMERGE_PROBE_FORECAST").is_some();
    let share = crate::spacetime::total_lagrangian::massive_share(
        &layout,
        &reference,
        (&mass, &v0s),
        settings.occupation,
    );
    let mut x = reference.clone();
    // Uniform `-v0` on the massive coefficients.
    let mut vel: Vec<DVec2> = share.iter().map(|s| DVec2::new(0.0, -v0 * s[1])).collect();
    let mut f = vec![DMat2::IDENTITY; n];
    let planes = [crate::spacetime::total_lagrangian::Plane {
        axis: 1,
        at: floor,
        inward: 1.0,
    }];
    // The augmented Lagrangian's penalty on the contact points, an
    // iteration parameter: `EMERGE_PROBE_BARS_PENALTY` (default 100) times
    // the column's P-wave stiffness over one cell, so each Uzawa update
    // closes most of the gap left.
    let factor = float("EMERGE_PROBE_BARS_PENALTY", 100.0);
    let penalty: Vec<f64> = (0..n)
        .map(|p| if p < body { 0.0 } else { factor * 2.0 * mu })
        .collect();
    let mut cache = None;
    // The exact solution (their eq. 26) at height `s` above the bottom.
    let exact = |s: f64, t: f64| -> (f64, f64) {
        if t <= t1 {
            (gap - v0 * t, -v0)
        } else if t <= t2 {
            let r = t - t1;
            let reach = tau - (r - tau).abs();
            if s / c0 < reach {
                (-v0 * s / c0, 0.0)
            } else {
                (-v0 * reach, if r < tau { -v0 } else { v0 })
            }
        } else {
            (v0 * (t - t2), v0)
        }
    };
    let volume: f64 = v0s.iter().sum();
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p]))
            .sum()
    };
    let e0 = energy(&vel, &f);
    // No gravity: the step tolerance's energy is the start's kinetic and
    // stored energy alone.
    let tolerance = crate::spacetime::total_lagrangian::tolerance_energy(
        e0,
        DVec2::ZERO,
        std::iter::empty::<(f64, &[DVec2])>(),
    );
    let (mut worst_u, mut worst_v) = (0.0f64, 0.0f64);
    let (mut first, mut last) = (None::<f64>, None::<f64>);
    let (mut push_sum, mut push_sq, mut pushes) = (0.0f64, 0.0f64, 0usize);
    let (mut work, mut least_work, mut updates) = (0.0f64, 0.0f64, 0.0f64);
    let (mut newton, mut gmres, mut passes, mut most_passes, mut unconverged, mut cost) =
        (0u64, 0u64, 0u64, 0u32, 0u32, 0.0f64);
    let mut deepest = 0.0f64;
    let pressure = c0 * v0;
    // `EMERGE_PROBE_BAR_ADAPTIVE=tol`: the step chosen by its error
    // (`total_lagrangian::controlled_step`), from the frame's step at most.
    let control = std::env::var("EMERGE_PROBE_BAR_ADAPTIVE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(
            |tolerance| crate::spacetime::total_lagrangian::StepControl {
                tolerance,
                safety: 0.9,
                growth: 10.0,
                largest: f64::from(FRAME),
                // `EMERGE_PROBE_SUBSTEPS=2,4,6`: the tries' counts, KSD's 1, 2, 3
                // by default.
                substeps: match std::env::var("EMERGE_PROBE_SUBSTEPS").as_deref() {
                    Ok("2,4,6") => [2, 4, 6],
                    _ => [1, 2, 3],
                },
                // `EMERGE_PROBE_ESTIMATOR=acary`: Acary's estimate.
                estimator: match std::env::var("EMERGE_PROBE_ESTIMATOR").as_deref() {
                    Ok("acary") => crate::spacetime::total_lagrangian::Estimator::Acary,
                    _ => crate::spacetime::total_lagrangian::Estimator::Klapproth,
                },
            },
        );
    let contact_set = crate::spacetime::total_lagrangian::ContactSet {
        planes: &planes,
        pairs: &[],
        outlines: &[],
        corners: &[],
        friction: 0.0,
    };
    let (mut step_length, mut accepted, mut rejected, mut solves) = (dt, 0u32, 0u32, 0u32);
    let (mut after_release, mut shortest, mut solved, mut push_time) =
        (0u32, f64::INFINITY, 0usize, 0.0f64);
    let mut t = 0.0;
    let mut taken = 0usize;
    // Chatter: changes of the number of active contacts from one kept
    // substep to the next, and sign flips of the contact points' normal
    // velocity from one kept step to the next while they are beyond the
    // ground.
    let (mut last_active, mut set_changes) = (0usize, 0u32);
    let mut last_sign = vec![0.0f64; n];
    // The gap the bottom face keeps above the ground while the ground
    // pushes it (a forecast contact's standoff).
    let mut standoff = 0.0f64;
    let mut flips = 0u32;
    while if control.is_some() {
        t < end - 1e-12
    } else {
        taken < steps
    } {
        taken += 1;
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut vel,
            f: &mut f,
            mass: &mass,
            v0: &v0s,
        };
        let (reports, step) = match &control {
            Some(control) => {
                let r = crate::spacetime::total_lagrangian::controlled_step(
                    &layout,
                    particles,
                    (&reference, &mut solids),
                    (DVec2::ZERO, &mut step_length),
                    (contact_set, &penalty, &settings),
                    &mut cache,
                    (control, tolerance.sqrt()),
                );
                accepted += 1;
                rejected += r.rejected;
                solves += r.solves;
                shortest = shortest.min(r.dt / r.kept.len() as f64);
                if !r.contact && first.is_some() {
                    after_release += 1;
                }
                (r.kept, r.dt)
            }
            None => (
                vec![
                    crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
                        &layout,
                        particles,
                        &reference,
                        &solids,
                        (dt, mode, DVec2::ZERO),
                        (contact_set, &penalty, &settings),
                        &mut cache,
                    ),
                ],
                dt,
            ),
        };
        let sub = step / reports.len() as f64;
        let pushed = reports.iter().any(|r| r.push > 0.0);
        for report in reports {
            t += sub;
            solved += 1;
            newton += u64::from(report.newton);
            gmres += u64::from(report.gmres);
            passes += u64::from(report.passes);
            most_passes = most_passes.max(report.passes);
            unconverged += u32::from(!report.converged);
            cost += report.phases.iter().sum::<f64>();
            work += report.work.0;
            updates += report.work.1;
            least_work = least_work.min(report.work.0);
            if report.contacts != last_active {
                set_changes += 1;
                last_active = report.contacts;
            }
            if report.push > 0.0 {
                first.get_or_insert(t - sub);
                last = Some(t);
                let p = report.push / width as f64;
                push_sum += p * sub;
                push_sq += p * p * sub;
                push_time += sub;
                pushes += 1;
            }
        }
        for q in &x[body..] {
            deepest = deepest.max(floor - q.y);
        }
        if pushed {
            let lowest = x[body..].iter().map(|q| q.y).fold(f64::INFINITY, f64::min);
            standoff = standoff.max(lowest - floor);
        }
        for p in body..n {
            let sign = vel[p].y.signum();
            if x[p].y < floor && last_sign[p] != 0.0 && sign != last_sign[p] {
                flips += 1;
            }
            last_sign[p] = sign;
        }
        let (mut du, mut dv) = (0.0, 0.0);
        for p in 0..body {
            let (u, v) = exact(reference[p].y - bottom, t);
            du += v0s[p] * (x[p].y - reference[p].y - (u - gap)).powi(2)
                + v0s[p] * (x[p].x - reference[p].x).powi(2);
            dv += v0s[p] * ((vel[p].y - v).powi(2) + vel[p].x.powi(2));
        }
        worst_u = worst_u.max((du / volume).sqrt() / (v0 * tau));
        worst_v = worst_v.max((dv / volume).sqrt() / v0);
    }
    let momentum: f64 = (0..body).map(|p| mass[p] * vel[p].y).sum();
    // A uniform velocity on the massive coefficients: its momentum and its
    // kinetic energy per unit velocity and squared velocity.
    let carried: f64 = (0..body).map(|p| mass[p] * share[p][1]).sum();
    let rigid: f64 = (0..body).map(|p| mass[p] * share[p][1] * share[p][1]).sum();
    let take_off = momentum / carried;
    let e_end = energy(&vel, &f);
    let vibration = (e_end - 0.5 * rigid * take_off * take_off) / e0;
    let contact = match (first, last) {
        (Some(a), Some(b)) => b - a,
        _ => 0.0,
    };
    let mean_push = push_sum / push_time.max(f64::MIN_POSITIVE);
    let spread = (push_sq / push_time.max(f64::MIN_POSITIVE) - mean_push * mean_push)
        .max(0.0)
        .sqrt();
    let realism = |error: f64| 100.0 * (1.0 - error).max(0.0);
    crate::spacetime::total_lagrangian::hooks::set_massless(None);
    format!(
        "bar impact, {mode:?} contact, {} mass, h = L/{cells} (L {length} m, width {width} cells), {}, c0 {:.1} m/s, strain {strain:e}: contact {:.2} steps exact; {steps} steps to t = {t:.4} s\n  contact {:.4} s for {:.4} exact (realism {:.1} %, resolution one step {:.4} s); push mean {:.4} of E v0/c0 (realism {:.1} %), spread {:.3} of it, over {pushes} steps; take-off {:.5} of v0 (realism {:.1} %)\n  energy end {:.6} of the start (realism {:.2} %), walls' work {:+.3e} of it (least step {:+.2e}), velocity updates {:+.3e}, balance {:+.2e}; vibration left {:.4} of the start\n  worst displacement error {:.3e} of v0 tau_w (realism {:.1} %), worst velocity error {:.3e} of v0 (realism {:.1} %); deepest end below the ground {deepest:.2e} cells, largest standoff while pushed {standoff:.2e} cells\n  Newton {:.2}, GMRES {:.1}, passes {:.2} (at most {most_passes}) per step, unconverged {unconverged}; {:.2} s per simulated s",
        if standard {
            "standard"
        } else {
            "massless contact boundary"
        },
        match courant {
            Some(nu) => format!("Courant {nu} (dt {dt:.3e} s)"),
            None => format!("frame step (Courant {:.1})", c0 * dt),
        },
        c0 * cell_m,
        2.0 * tau / dt,
        contact,
        2.0 * tau,
        realism((contact - 2.0 * tau).abs() / (2.0 * tau)),
        dt,
        mean_push / pressure,
        realism((mean_push / pressure - 1.0).abs()),
        spread / pressure,
        take_off / v0,
        realism((take_off / v0 - 1.0).abs()),
        e_end / e0,
        realism((e_end / e0 - 1.0).abs()),
        work / e0,
        least_work / e0,
        updates / e0,
        (e_end - e0 - work - updates) / e0,
        vibration,
        worst_u,
        realism(worst_u),
        worst_v,
        realism(worst_v),
        newton as f64 / solved as f64,
        gmres as f64 / solved as f64,
        passes as f64 / solved as f64,
        cost / t
    ) + &match control {
        Some(c) => format!(
            "\n  controlled, tol {} (error energy at most {:.1e} of the start): {accepted} steps kept, {rejected} rejected, {solves} solves, {after_release} after the release, shortest {shortest:.3e} s ({:.1} per contact); active set changed {set_changes} times over the kept substeps, contact points' normal velocity flipped {flips} times beyond the ground",
            c.tolerance,
            (c.safety * c.tolerance).powi(2),
            2.0 * tau / shortest
        ),
        None => String::new(),
    }
}

/// The bar's impact (`bar_impact_run`) over `EMERGE_PROBE_BAR_COURANT`
/// (comma separated Courant numbers `c0 dt / h`, or `frame` for the 60 Hz
/// frame's step; default `1.5,frame`) and `EMERGE_PROBE_BAR_CELLS`
/// (default 16). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_bar_impact() {
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    for cells in list("EMERGE_PROBE_BAR_CELLS", "16") {
        for step in list("EMERGE_PROBE_BAR_COURANT", "1.5,frame") {
            let Ok(cells) = cells.parse() else { continue };
            let courant = if step == "frame" {
                None
            } else {
                match step.parse() {
                    Ok(nu) => Some(nu),
                    Err(_) => continue,
                }
            };
            println!("{}", bar_impact_run(courant, cells));
        }
    }
}

/// Two identical elastic bars in longitudinal impact, the body-body
/// counterpart of `bar_impact_run`, held by the contact condition in
/// velocity between two bodies (Laursen and Chawla, IJNME 40 (1997), eqs.
/// 46-48 and 51-52: the gap rate from both bodies' midpoint velocities;
/// `total_lagrangian::step_total_lagrangian_contact` with pairs). Bar A, above,
/// falls at `v0` onto bar B, below. `EMERGE_PROBE_BARS_CASE`:
/// - `rest` (default): B at rest. Exact for identical bars (Laursen and
///   Chawla sec. 6.1, Cirak and West sec. 6.1): after a contact of `2 L /
///   c0`, A stops and B leaves at `v0`, both undeformed.
/// - `symmetric`: B rises at `v0`. By symmetry the interface stays still,
///   so each bar behaves as `bar_impact_run`'s bar on a wall: the two must
///   agree, a check of the pair contact against the wall contact.
///
/// `EMERGE_PROBE_BARS_RATIO` (default 1) makes A that many times B's length,
/// B at rest. For equal impedances the solution follows from the
/// characteristics (derived, checked against Kakouris, Chatzis and
/// Triantafyllou, arXiv 2403.13534, sec. 4.2, who state the short striker's
/// case citing Wriggers 2006, not read): the interface moves at `v0 / 2`
/// while the push lasts, `2 min(L_A, L_B) / c0`. A longer striker leaves
/// B rigid at `v0` and keeps the mean `v0 (L_A - L_B) / L_A`; a shorter one
/// stops dead and leaves B the mean `v0 L_A / L_B` (the two stay touching
/// without force until `2 L_B / c0`).
///
/// The two bodies' reference configurations are apart (their B-splines
/// share no coefficient), and A starts displaced next to B: on the
/// reference grid they meet only through the contact multipliers. Same
/// material, sizes and units as `bar_impact_run` (`EMERGE_PROBE_BAR_*`),
/// standard mass, the impact inside the third step at
/// `EMERGE_PROBE_BAR_PHASE`.
fn two_bar_impact_run(courant: Option<f64>, cells: usize) -> String {
    use glam::{DMat2, DVec2};
    let float = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let (e, rho) = (
        float("EMERGE_PROBE_BAR_E", 11.25e6),
        float("EMERGE_PROBE_BAR_RHO", 1700.0),
    );
    let length = float("EMERGE_PROBE_BAR_LENGTH", 1.0);
    let width = float("EMERGE_PROBE_BAR_WIDTH", 2.0) as usize;
    let strain = float("EMERGE_PROBE_BAR_STRAIN", 1e-3);
    let symmetric = std::env::var("EMERGE_PROBE_BARS_CASE").as_deref() == Ok("symmetric");
    let ratio = float("EMERGE_PROBE_BARS_RATIO", 1.0);
    assert!(
        !symmetric || ratio == 1.0,
        "the symmetric case needs equal bars"
    );
    let cells_a = (cells as f64 * ratio).round() as usize;
    let cell_m = length / cells as f64;
    let c0 = (e / rho).sqrt() / cell_m;
    let mu = 0.5 * c0 * c0;
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, mu);
    let side = cells as f64;
    let side_a = cells_a as f64;
    // The push lasts `2 min(L_A, L_B) / c0`.
    let tau = side.min(side_a) / c0;
    let v0 = strain * c0;
    let frame = f64::from(FRAME);
    let dt = courant.map_or(frame, |nu| nu / c0);
    // Closing speed: v0, or 2 v0 when B rises too.
    let closing = if symmetric { 2.0 * v0 } else { v0 };
    let t1 = (2.0 + float("EMERGE_PROBE_BAR_PHASE", 0.37)) * dt;
    let gap = closing * t1;
    let end = t1 + 2.0 * side.max(side_a) / c0 + tau.max(3.0 * dt);
    let steps = (end / dt).ceil() as usize;
    // B's reference above the grid's bottom, A's reference `apart` cells
    // above B's; A starts shifted down to `gap` above B.
    let (x0, bottom, apart) = (4.0, 4.5, 6.0);
    let top_b = bottom + side;
    let bottom_a = top_b + apart;
    let layout = MacLayout::new(width + 8, cells + cells_a + apart as usize + 9, 1.0);
    let (nodes, node_weights) = gauss_legendre(3);
    let block = |base: f64, rows: usize| -> Vec<(DVec2, f64)> {
        let mut out = Vec::new();
        for j in 0..2 * rows {
            for i in 0..2 * width {
                for b in 0..3 {
                    for a in 0..3 {
                        out.push((
                            DVec2::new(
                                x0 + (i as f64 + nodes[a]) * 0.5,
                                base + (j as f64 + nodes[b]) * 0.5,
                            ),
                            0.25 * node_weights[a] * node_weights[b],
                        ));
                    }
                }
            }
        }
        out
    };
    let (mut reference, mut v0s): (Vec<DVec2>, Vec<f64>) = block(bottom, cells).into_iter().unzip();
    let body_b = reference.len();
    let (ra, va): (Vec<DVec2>, Vec<f64>) = block(bottom_a, cells_a).into_iter().unzip();
    reference.extend(ra);
    v0s.extend(va);
    let body_a = reference.len();
    // The contact points: B's top face and A's bottom face, at the same
    // abscissae (the bars stay aligned), of no mass and no volume.
    let mut along = vec![x0];
    along.extend((0..width).map(|i| x0 + i as f64 + 0.5));
    along.push(x0 + width as f64);
    let first_b = reference.len();
    for &xc in &along {
        reference.push(DVec2::new(xc, top_b));
        v0s.push(0.0);
    }
    let first_a = reference.len();
    for &xc in &along {
        reference.push(DVec2::new(xc, bottom_a));
        v0s.push(0.0);
    }
    let n = reference.len();
    let in_a = |p: usize| (body_b..body_a).contains(&p) || p >= first_a;
    let shift = DVec2::new(0.0, gap - apart);
    let mut x: Vec<DVec2> = (0..n)
        .map(|p| reference[p] + if in_a(p) { shift } else { DVec2::ZERO })
        .collect();
    let mass = v0s.clone();
    let solids = vec![solid; n];
    let mut vel: Vec<DVec2> = (0..n)
        .map(|p| {
            if in_a(p) {
                DVec2::new(0.0, -v0)
            } else if symmetric {
                DVec2::new(0.0, v0)
            } else {
                DVec2::ZERO
            }
        })
        .collect();
    let mut f = vec![DMat2::IDENTITY; n];
    crate::spacetime::total_lagrangian::hooks::set_massless(None);
    let settings = crate::spacetime::total_lagrangian::ConservingSettings::default();
    // A above B: their separation `y_A - y_B` stays non-negative.
    let pairs: Vec<crate::spacetime::total_lagrangian::Pair> = (0..along.len())
        .map(|i| {
            (
                first_a + i,
                first_b + i,
                crate::spacetime::total_lagrangian::Plane {
                    axis: 1,
                    at: 0.0,
                    inward: 1.0,
                },
            )
        })
        .collect();
    // The augmented Lagrangian's penalty, an iteration parameter (the
    // converged multipliers do not depend on it): `EMERGE_PROBE_BARS_PENALTY`
    // times the P-wave stiffness over one cell (default 10: at 100 the
    // Newton floor of the Jacobian-free solve reached the tolerance on 1-2
    // steps, which stopped Uzawa early and left work in the contact;
    // measured, 2026-10-13).
    let factor = float("EMERGE_PROBE_BARS_PENALTY", 10.0);
    let penalty: Vec<f64> = (0..n)
        .map(|p| if p < first_b { 0.0 } else { factor * 2.0 * mu })
        .collect();
    let mut cache = None;
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p]))
            .sum()
    };
    let momentum = |vel: &[DVec2]| -> DVec2 { (0..n).map(|p| mass[p] * vel[p]).sum() };
    let mean = |vel: &[DVec2], range: std::ops::Range<usize>| -> f64 {
        let total: f64 = range.clone().map(|p| mass[p]).sum();
        range.map(|p| mass[p] * vel[p].y).sum::<f64>() / total
    };
    let e0 = energy(&vel, &f);
    let p0 = momentum(&vel);
    let scale = (0..n).map(|p| mass[p] * vel[p].length()).sum::<f64>();
    let (mut first, mut last) = (None::<f64>, None::<f64>);
    let (mut work, mut deepest, mut newton, mut gmres, mut unconverged, mut cost) =
        (0.0f64, 0.0f64, 0u64, 0u64, 0u32, 0.0f64);
    let mut worst_momentum = 0.0f64;
    let history = std::env::var("EMERGE_PROBE_BARS_HISTORY").is_ok();
    let (mut released, mut recontacts, mut late_impulse) = (false, 0usize, 0.0f64);
    let mut t = 0.0;
    // With `EMERGE_PROBE_BARS_TRACE` an unconverged step's Newton
    // iterations are printed, solve by solve (`total_lagrangian::hooks`).
    let trace = std::env::var("EMERGE_PROBE_BARS_TRACE").is_ok();
    for _ in 0..steps {
        if trace {
            crate::spacetime::total_lagrangian::hooks::start(0);
        }
        let report = crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
            &layout,
            crate::spacetime::total_lagrangian::Particles {
                x: &mut x,
                v: &mut vel,
                f: &mut f,
                mass: &mass,
                v0: &v0s,
            },
            &reference,
            &solids,
            (
                dt,
                crate::spacetime::total_lagrangian::ContactMode::Velocity,
                DVec2::ZERO,
            ),
            (
                crate::spacetime::total_lagrangian::ContactSet {
                    planes: &[],
                    pairs: &pairs,
                    outlines: &[],
                    corners: &[],
                    friction: 0.0,
                },
                &penalty,
                &settings,
            ),
            &mut cache,
        );
        if trace {
            let solves = crate::spacetime::total_lagrangian::hooks::finish();
            if !report.converged {
                for (k, lines) in solves.iter().enumerate() {
                    println!("    solve {k}:");
                    for line in lines {
                        println!("      {line}");
                    }
                }
            }
        }
        if !report.converged {
            println!(
                "  unconverged at t {:.4}: residual {:.2e} after {} Newton, {} GMRES, {} passes, {} contacts, start residual {:.2e}",
                t + dt,
                report.residual,
                report.newton,
                report.gmres,
                report.passes,
                report.contacts,
                report.start_residual
            );
        }
        t += dt;
        newton += u64::from(report.newton);
        gmres += u64::from(report.gmres);
        unconverged += u32::from(!report.converged);
        cost += report.phases.iter().sum::<f64>();
        work += report.work.0;
        // The first contiguous run of pushing steps is the impact; any later
        // push is a re-contact, counted apart with its impulse.
        if report.push > 0.0 {
            if first.is_none() {
                first = Some(t);
            }
            if released {
                recontacts += 1;
                late_impulse += report.push * dt;
            } else {
                last = Some(t);
            }
        } else if first.is_some() {
            released = true;
        }
        // With `EMERGE_PROBE_BARS_HISTORY`, every step's push (over the
        // exact `E v0 / (2 c0)` times the width) and the bars' mean speeds.
        if history {
            println!(
                "  t {t:.4}: push {:.4}, A {:+.4} v0, B {:+.4} v0, {} contacts, {} passes",
                report.push / (0.5 * c0 * v0 * width as f64),
                mean(&vel, body_b..body_a) / v0,
                mean(&vel, 0..body_b) / v0,
                report.contacts,
                report.passes
            );
        }
        for &(p, q, _) in &pairs {
            deepest = deepest.max(x[q].y - x[p].y);
        }
        worst_momentum = worst_momentum.max((momentum(&vel) - p0).length() / scale);
    }
    let contact = match (first, last) {
        (Some(a), Some(b)) => b - a + dt,
        _ => 0.0,
    };
    let (va_end, vb_end) = (mean(&vel, body_b..body_a), mean(&vel, 0..body_b));
    let (va_exact, vb_exact) = if symmetric {
        (v0, -v0)
    } else if side_a >= side {
        (-v0 * (side_a - side) / side_a, -v0)
    } else {
        (0.0, -v0 * side_a / side)
    };
    let e_end = energy(&vel, &f);
    let mass_a: f64 = (body_b..body_a).map(|p| mass[p]).sum();
    let mass_b: f64 = (0..body_b).map(|p| mass[p]).sum();
    let rigid = 0.5 * (mass_a * va_end * va_end + mass_b * vb_end * vb_end);
    let realism = |error: f64| 100.0 * (1.0 - error).max(0.0);
    format!(
        "two bars, {} case, L_A/L_B {ratio}, h = L_B/{cells} (L_B {length} m, width {width} cells), {}, c0 {:.1} m/s, strain {strain:e}: push {:.2} steps exact; {steps} steps to t = {t:.4} s\n  push {:.4} s for {:.4} exact (realism {:.1} %); {recontacts} re-contact steps after it, impulse {:.2e} of the exact push's\n  A leaves at {:+.5} v0 (exact {:+.4}), B at {:+.5} v0 (exact {:+.4}); transfer realism {:.1} %\n  energy end {:.6} of the start, contact work {:+.3e} of it, balance {:+.2e}; vibration left {:.4} of the start; momentum error at worst {:.2e} of the total |m v|\n  deepest crossing {deepest:.2e} cells ({:.2e} m); Newton {:.2}, GMRES {:.1} per step, unconverged {unconverged}; {:.2} s per simulated s",
        if symmetric { "symmetric" } else { "rest" },
        match courant {
            Some(nu) => format!("Courant {nu} (dt {dt:.3e} s)"),
            None => format!("frame step (Courant {:.1})", c0 * dt),
        },
        c0 * cell_m,
        2.0 * tau / dt,
        contact,
        2.0 * tau,
        realism((contact - 2.0 * tau).abs() / (2.0 * tau)),
        late_impulse / (0.5 * c0 * v0 * width as f64 * 2.0 * tau),
        va_end / v0,
        va_exact / v0,
        vb_end / v0,
        vb_exact / v0,
        realism(((va_end - va_exact).abs() + (vb_end - vb_exact).abs()) / (2.0 * v0)),
        e_end / e0,
        work / e0,
        (e_end - e0 - work) / e0,
        (e_end - rigid) / e0,
        worst_momentum,
        deepest * cell_m,
        newton as f64 / steps as f64,
        gmres as f64 / steps as f64,
        cost / t
    )
}

/// The two bars' impact (`two_bar_impact_run`) over
/// `EMERGE_PROBE_BAR_COURANT` (default `1.5,frame`) and
/// `EMERGE_PROBE_BAR_CELLS` (default 16). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_two_bar_impact() {
    let list = |name: &str, default: &str| -> Vec<String> {
        std::env::var(name)
            .unwrap_or_else(|_| default.into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect()
    };
    for cells in list("EMERGE_PROBE_BAR_CELLS", "16") {
        for step in list("EMERGE_PROBE_BAR_COURANT", "1.5,frame") {
            let Ok(cells) = cells.parse() else { continue };
            let courant = if step == "frame" {
                None
            } else {
                match step.parse() {
                    Ok(nu) => Some(nu),
                    Err(_) => continue,
                }
            };
            println!("{}", two_bar_impact_run(courant, cells));
        }
    }
}

/// Two identical elastic discs in a frictionless, off-centre collision,
/// held by the contact condition in velocity against each other's outline
/// (Laursen and Chawla, IJNME 40 (1997), eqs. 40-48: normal and closest
/// point in the step's midpoint configuration, which keeps both momenta;
/// `total_lagrangian::step_total_lagrangian_contact` with outlines). Disc A moves
/// at `v0` along `x` into disc B at rest, its centre `b` above B's line of
/// motion; A may spin (`EMERGE_PROBE_DISC_SPIN`, `omega R / v0`).
///
/// No closed form for a deformable 2D impact is used. What is checked is
/// what frictionless contact between circles implies (the push along the
/// line of centres, no torque about either centre): each disc keeps its
/// velocity component across that line and its spin; both momenta are
/// conserved exactly (the step's own claim). The normal restitution is
/// reported (to compare later with Gerl and Zippelius, arXiv
/// cond-mat/9808258, Fig. 1, which tends to 1 as `v0 / c -> 0`).
///
/// Parameters (`EMERGE_PROBE_DISC_*`): `RADIUS` cells (default 8), `B`
/// the offset over the diameter (default 0.5), `NU` Poisson's ratio
/// (default 0.3), `STRAIN` `v0` over the P-wave speed (default 1e-3),
/// `OUTLINE` points per cell of circumference (default 2), `PENALTY` (as
/// the bars', default 10), `MU` Coulomb's friction coefficient (default
/// 0), `CONTACT` (`off`: no contact, a control).
/// Medium clay's modulus and density, a 0.2 m disc.
fn disc_impact_run(courant: Option<f64>) -> String {
    use glam::{DMat2, DVec2};
    let float = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let radius = float("EMERGE_PROBE_DISC_RADIUS", 8.0);
    let offset = float("EMERGE_PROBE_DISC_B", 0.5);
    let nu = float("EMERGE_PROBE_DISC_NU", 0.3);
    let strain = float("EMERGE_PROBE_DISC_STRAIN", 1e-3);
    let spin_ratio = float("EMERGE_PROBE_DISC_SPIN", 0.0);
    let per_cell = float("EMERGE_PROBE_DISC_OUTLINE", 2.0);
    let factor = float("EMERGE_PROBE_DISC_PENALTY", 10.0);
    let friction = float("EMERGE_PROBE_DISC_MU", 0.0);
    let (e, rho) = (11.25e6f64, 1700.0f64);
    let cell_m = 0.1 / radius;
    let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
    let solid = crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu);
    let c_p = (lambda + 2.0 * mu).sqrt();
    let v0 = strain * c_p;
    let frame = f64::from(FRAME);
    let dt = courant.map_or(frame, |nu| nu / c_p);
    // Reference centres apart; A starts displaced left of B.
    let margin = 4.0;
    let centre_b = DVec2::splat(radius + margin);
    let centre_a_ref = DVec2::new(3.0 * radius + 3.0 * margin, radius + margin);
    let side = (4.0 * radius + 4.0 * margin).ceil() as usize;
    let layout = MacLayout::new(
        side + 2,
        (2.0 * radius + 2.0 * margin).ceil() as usize + 2,
        1.0,
    );
    let rise = 2.0 * radius * offset;
    let reach = (4.0 * radius * radius - rise * rise).sqrt();
    let t1 = (2.0 + float("EMERGE_PROBE_BAR_PHASE", 0.37)) * dt;
    let centre_a = DVec2::new(centre_b.x - reach - v0 * t1, centre_b.y + rise);
    let shift = centre_a - centre_a_ref;
    let (mut reference, mut v0s) = ring_quadrature(centre_b, (0.0, radius), side, 1);
    let body_b = reference.len();
    let (ra, va) = ring_quadrature(centre_a_ref, (0.0, radius), side, 1);
    reference.extend(ra);
    v0s.extend(va);
    let body_a = reference.len();
    // The outlines: points on each circle, counter-clockwise, of no mass.
    let count = (2.0 * std::f64::consts::PI * radius * per_cell).ceil() as usize;
    // `EMERGE_PROBE_DISC_INSET` cells: the outline drawn inside the circle
    // (a diagnostic, default 0).
    let inset = float("EMERGE_PROBE_DISC_INSET", 0.0);
    let outline = |centre: DVec2, start: usize| -> (Vec<DVec2>, Vec<usize>) {
        (0..count)
            .map(|k| {
                let angle = 2.0 * std::f64::consts::PI * k as f64 / count as f64;
                (
                    centre + (radius - inset) * DVec2::new(angle.cos(), angle.sin()),
                    start + k,
                )
            })
            .unzip()
    };
    let (points_b, outline_b) = outline(centre_b, reference.len());
    reference.extend(points_b);
    let first_outline_a = reference.len();
    let (points_a, outline_a) = outline(centre_a_ref, first_outline_a);
    reference.extend(points_a);
    v0s.resize(reference.len(), 0.0);
    let n = reference.len();
    let in_a = |p: usize| (body_b..body_a).contains(&p) || p >= first_outline_a;
    let mut x: Vec<DVec2> = (0..n)
        .map(|p| reference[p] + if in_a(p) { shift } else { DVec2::ZERO })
        .collect();
    let omega = spin_ratio * v0 / radius;
    // `EMERGE_PROBE_DISC_SHEAR`: A's velocity also carries a nonuniform,
    // asymmetric part (a diagnostic of the step's own angular momentum).
    let shear = float("EMERGE_PROBE_DISC_SHEAR", 0.0);
    let mut vel: Vec<DVec2> = (0..n)
        .map(|p| {
            if in_a(p) {
                let r = x[p] - centre_a;
                DVec2::new(v0, 0.0)
                    + omega * DVec2::new(-r.y, r.x)
                    + shear * v0 * DVec2::new(0.0, r.x * (r.x + r.y) / (radius * radius))
            } else {
                DVec2::ZERO
            }
        })
        .collect();
    let mass = v0s.clone();
    let solids = vec![solid; n];
    let mut f = vec![DMat2::IDENTITY; n];
    crate::spacetime::total_lagrangian::hooks::set_massless(None);
    // `NEWTON` the Newton tolerance, `OCCUPATION` the extended B-splines'
    // `C_c` (`none`: the original B-splines), both as the defaults if unset.
    let mut settings = crate::spacetime::total_lagrangian::ConservingSettings::default();
    settings.newton_tolerance = float("EMERGE_PROBE_DISC_NEWTON", settings.newton_tolerance);
    match std::env::var("EMERGE_PROBE_DISC_OCCUPATION").as_deref() {
        Ok("none") => settings.occupation = None,
        Ok(value) => settings.occupation = value.parse().ok().or(settings.occupation),
        Err(_) => {}
    }
    let penalty: Vec<f64> = (0..n)
        .map(|p| if p < body_a { 0.0 } else { factor * c_p * c_p })
        .collect();
    // `EMERGE_PROBE_DISC_CONTACT=off`: no outlines, the discs pass
    // through each other (a control: the step's own momenta, no contact).
    let outlines = if std::env::var("EMERGE_PROBE_DISC_CONTACT").as_deref() == Ok("off") {
        Vec::new()
    } else {
        vec![outline_b.clone(), outline_a.clone()]
    };
    let energy = |vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p]))
            .sum()
    };
    let linear = |vel: &[DVec2]| -> DVec2 { (0..n).map(|p| mass[p] * vel[p]).sum() };
    let angular = |x: &[DVec2], vel: &[DVec2]| -> f64 {
        (0..n).map(|p| mass[p] * x[p].perp_dot(vel[p])).sum()
    };
    // A disc's centre, centre velocity and spin.
    let state =
        |x: &[DVec2], vel: &[DVec2], range: std::ops::Range<usize>| -> (DVec2, DVec2, f64) {
            let total: f64 = range.clone().map(|p| mass[p]).sum();
            let centre: DVec2 = range.clone().map(|p| mass[p] * x[p]).sum::<DVec2>() / total;
            let speed: DVec2 = range.clone().map(|p| mass[p] * vel[p]).sum::<DVec2>() / total;
            let (mut moment, mut inertia) = (0.0, 0.0);
            for p in range {
                let r = x[p] - centre;
                moment += mass[p] * r.perp_dot(vel[p] - speed);
                inertia += mass[p] * r.length_squared();
            }
            (centre, speed, moment / inertia)
        };
    let e0 = energy(&vel, &f);
    let (l0, j0) = (linear(&vel), angular(&x, &vel));
    let disc_mass: f64 = (0..body_b).map(|p| mass[p]).sum();
    let (l_scale, j_scale) = (disc_mass * v0, disc_mass * v0 * radius);
    let (_, _, spin_a0) = state(&x, &vel, body_b..body_a);
    let mut cache = None;
    // `EMERGE_PROBE_DISC_TRACE`: each step's change of the angular momentum
    // and how its solve ended.
    let trace = std::env::var("EMERGE_PROBE_DISC_TRACE").is_ok();
    let mut j_last = j0;
    crate::spacetime::total_lagrangian::hooks::watch_torque(trace);
    let (mut worst_l, mut worst_j, mut work, mut deepest) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let (mut first, mut last) = (None::<f64>, None::<f64>);
    let (mut newton, mut gmres, mut unconverged, mut cost, mut steps) =
        (0u64, 0u64, 0u32, 0.0f64, 0usize);
    let quiet = (4.0 * radius / c_p).max(3.0 * dt);
    let cap = t1 + 60.0 * radius / c_p + 10.0 * dt;
    let mut t = 0.0;
    while t < cap {
        let report = crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
            &layout,
            crate::spacetime::total_lagrangian::Particles {
                x: &mut x,
                v: &mut vel,
                f: &mut f,
                mass: &mass,
                v0: &v0s,
            },
            &reference,
            &solids,
            (
                dt,
                crate::spacetime::total_lagrangian::ContactMode::Velocity,
                DVec2::ZERO,
            ),
            (
                crate::spacetime::total_lagrangian::ContactSet {
                    planes: &[],
                    pairs: &[],
                    outlines: &outlines,
                    corners: &[],
                    friction,
                },
                &penalty,
                &settings,
            ),
            &mut cache,
        );
        t += dt;
        steps += 1;
        newton += u64::from(report.newton);
        gmres += u64::from(report.gmres);
        unconverged += u32::from(!report.converged);
        cost += report.phases.iter().sum::<f64>();
        work += report.work.0;
        if report.push > 0.0 {
            first.get_or_insert(t);
            last = Some(t);
        }
        worst_l = worst_l.max((linear(&vel) - l0).length() / l_scale);
        worst_j = worst_j.max((angular(&x, &vel) - j0).abs() / j_scale);
        if trace {
            let j_now = angular(&x, &vel);
            let (step_moment, push_moment) =
                crate::spacetime::total_lagrangian::hooks::take_torque().unwrap_or_default();
            println!(
                "  step {steps}: dJ {:+.3e} of M v0 R (from m x_mid x dv {:+.3e}, the push's share {:+.3e}; total {:+.3e}), push {:.3e}, Newton {} residual {:.2e} converged {}, projection {:.2e}, passes {}",
                (j_now - j_last) / j_scale,
                step_moment / j_scale,
                push_moment / j_scale,
                (j_now - j0) / j_scale,
                report.push,
                report.newton,
                report.residual,
                report.converged,
                report.projection,
                report.passes
            );
            j_last = j_now;
        }
        // Overlap: the outline points of A inside B's circle, and back.
        let (cb, _, _) = state(&x, &vel, 0..body_b);
        let (ca, _, _) = state(&x, &vel, body_b..body_a);
        for &p in &outline_a {
            deepest = deepest.max(radius - (x[p] - cb).length());
        }
        for &p in &outline_b {
            deepest = deepest.max(radius - (x[p] - ca).length());
        }
        if last.is_some_and(|l| t - l > quiet) {
            break;
        }
    }
    let normal = DVec2::new(reach, -rise) / (2.0 * radius);
    let tangent = DVec2::new(rise, reach) / (2.0 * radius);
    let (_, va_end, spin_a) = state(&x, &vel, body_b..body_a);
    let (_, vb_end, spin_b) = state(&x, &vel, 0..body_b);
    let restitution = (vb_end - va_end).dot(normal) / v0 / normal.x;
    let e_end = energy(&vel, &f);
    let rigid = 0.5 * disc_mass * (va_end.length_squared() + vb_end.length_squared());
    let contact = match (first, last) {
        (Some(a), Some(b)) => b - a + dt,
        _ => 0.0,
    };
    format!(
        "discs, R {radius} cells, b/2R {offset}, nu {nu}, spin {spin_ratio} v0/R, {}, c_p {:.1} m/s, strain {strain:e}, outline {count} points: {steps} steps to t = {t:.4} s, contact {:.4} s ({:.1} steps)\n  A across the line of centres {:+.5} of its exact {:+.4} v0, B {:+.2e} v0 (exact 0); normal restitution {:.5}; spin A {:+.3e} (start {:+.3e}), B {:+.3e} v0/R\n  linear momentum error at worst {worst_l:.2e} of M v0, angular {worst_j:.2e} of M v0 R; energy end {:.6} of the start, contact work {:+.3e} of it, balance {:+.2e}; vibration left {:.4}\n  deepest overlap {deepest:.2e} cells; Newton {:.2}, GMRES {:.1} per step, unconverged {unconverged}; {:.2} s per simulated s",
        match courant {
            Some(nu) => format!("Courant {nu} (dt {dt:.3e} s)"),
            None => format!("frame step (Courant {:.1})", c_p * dt),
        },
        c_p * cell_m,
        contact,
        contact / dt,
        va_end.dot(tangent) / (v0 * tangent.x),
        tangent.x,
        vb_end.dot(tangent) / v0,
        restitution,
        spin_a * radius / v0,
        spin_a0 * radius / v0,
        spin_b * radius / v0,
        e_end / e0,
        work / e0,
        (e_end - e0 - work) / e0,
        (e_end - rigid - 0.0) / e0,
        newton as f64 / steps.max(1) as f64,
        gmres as f64 / steps.max(1) as f64,
        cost / t
    )
}

/// The discs' collision (`disc_impact_run`) over `EMERGE_PROBE_BAR_COURANT`
/// (default `1.5,frame`). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_disc_impact() {
    let steps = std::env::var("EMERGE_PROBE_BAR_COURANT").unwrap_or_else(|_| "1.5,frame".into());
    for step in steps.split(',').map(str::trim) {
        let courant = if step == "frame" {
            None
        } else {
            match step.parse() {
                Ok(nu) => Some(nu),
                Err(_) => continue,
            }
        };
        println!("{}", disc_impact_run(courant));
    }
}

/// A stiff block sliding on a rigid floor under gravity, Coulomb friction
/// on the contact condition in velocity (`total_lagrangian::step_total_lagrangian_contact`
/// with `ContactSet::friction`). The floor may be an incline of slope
/// `TILT` (`tan theta`, default 0), as gravity turned by `theta` against
/// a level floor, down the slope along `+x`. Its rigid limit is exact: the
/// block slows at `d = g (mu cos theta - sin theta)`, `v(t) = v0 - d t`;
/// with `d > 0` it stops at `t_s = v0 / d`, `v0^2 / (2 d)` from its start,
/// and then stays (static friction holds it, `mu >= tan theta`); with `d
/// < 0` it speeds up. The heat friction makes is its force times the path,
/// `mu g cos theta M s` (Acary and Collins-Craft, JTCAM 2025, eq. 73: the
/// friction's work is the mechanical energy lost, exactly); on the level,
/// `1/2 M v0^2`.
///
/// The block is `WIDTH` by `HEIGHT` cells (defaults 8 by 4) of 1 cm, Gauss
/// 3x3 per half cell as the bar's, with massless contact points on its
/// bottom face as the bar's (its ends and the middles of its knot spans);
/// the floor's plane a hair under that face, so it holds from the first
/// step. `E` (default 1 GPa, the rigid limit's stand-in), `RHO` (1700),
/// `NU` (0.3), `MU` the friction coefficient (0.3), `V0` m/s (1),
/// `PENALTY` (times `c_p^2`, default 1) under `EMERGE_PROBE_SLIDE_*`;
/// `STEPS` the steps per frame to run, a list (default `1,4`).
///
/// Measured: the centre's velocity against the exact one at every step
/// (largest error over the reference speed, `v0` or the exact end speed if
/// larger), the stop time against `t_s`, the path against the exact one,
/// the creep after the stop, the heat against the exact heat, the energy
/// balance (kinetic, stored, gravity's, heat) over `1/2 M` the reference
/// speed squared, the steps where friction made energy (its work
/// positive).
fn block_slide_run(per_frame: usize) -> String {
    use glam::{DMat2, DVec2};
    let float = |name: &str, default: f64| -> f64 {
        std::env::var(name)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    };
    let width = float("EMERGE_PROBE_SLIDE_WIDTH", 8.0) as usize;
    let height = float("EMERGE_PROBE_SLIDE_HEIGHT", 4.0) as usize;
    let (e, rho, nu) = (
        float("EMERGE_PROBE_SLIDE_E", 1e9),
        float("EMERGE_PROBE_SLIDE_RHO", 1700.0),
        float("EMERGE_PROBE_SLIDE_NU", 0.3),
    );
    let friction = float("EMERGE_PROBE_SLIDE_MU", 0.3);
    let cell_m = 0.01;
    let g = 9.81 / cell_m;
    let v0 = float("EMERGE_PROBE_SLIDE_V0", 1.0) / cell_m;
    // At 10 GMRES could not go below 1e-3 while the block stuck (the
    // penalty's stiffness over the inertia, measured at 1 GPa); 1 solves.
    let factor = float("EMERGE_PROBE_SLIDE_PENALTY", 1.0);
    let mu = e / (2.0 * (1.0 + nu)) / rho / (cell_m * cell_m);
    let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)) / rho / (cell_m * cell_m);
    // `EMERGE_PROBE_SLIDE_LAW=corotated`: the corotated material's law.
    let solid = if std::env::var("EMERGE_PROBE_SLIDE_LAW").as_deref() == Ok("corotated") {
        crate::spacetime::total_lagrangian::Elastic::corotated(lambda, mu)
    } else {
        crate::spacetime::total_lagrangian::Elastic::isochoric(mu, lambda + mu)
    };
    let c_p = (lambda + 2.0 * mu).sqrt();
    let dt = f64::from(FRAME) / per_frame as f64;
    let tilt = float("EMERGE_PROBE_SLIDE_TILT", 0.0);
    let (sine, cosine) = (
        tilt / (1.0 + tilt * tilt).sqrt(),
        1.0 / (1.0 + tilt * tilt).sqrt(),
    );
    let gravity = g * DVec2::new(sine, -cosine);
    let slowing = g * (friction * cosine - sine);
    // With `d > 0` and a start speed the block stops, then stays: a
    // quarter second more of it; otherwise 0.6 s.
    let stops = slowing > 0.0 && v0 > 0.0;
    let stop = if slowing > 0.0 {
        v0 / slowing
    } else {
        f64::INFINITY
    };
    // `DROP` cells of fall onto the floor first (default 0: resting on it
    // from the start) and `END` seconds of run (default as above).
    let drop = float("EMERGE_PROBE_SLIDE_DROP", 0.0);
    let end = float(
        "EMERGE_PROBE_SLIDE_END",
        if stops { stop + 0.25 } else { 0.6 },
    );
    let speed = |t: f64| -> f64 {
        if slowing > 0.0 {
            (v0 - slowing * t).max(0.0)
        } else {
            v0 - slowing * t
        }
    };
    let path = |t: f64| -> f64 {
        let t = t.min(stop);
        v0 * t - 0.5 * slowing * t * t
    };
    // At rest on a slope it holds, the speed it would reach with no
    // friction sets the scale.
    let drop_speed = (2.0 * g * float("EMERGE_PROBE_SLIDE_DROP", 0.0)).sqrt();
    let reference_speed = v0.max(speed(end).abs()).max(g * sine * end).max(drop_speed);
    let travel = path(end);
    let (x0, floor) = (4.0, 4.0);
    let bottom = floor + drop;
    let layout = MacLayout::new(
        (x0 + width as f64 + travel + 8.0).ceil() as usize,
        (height as f64 + drop + 8.0).ceil() as usize,
        1.0,
    );
    let (nodes, node_weights) = gauss_legendre(3);
    let (mut reference, mut v0s): (Vec<DVec2>, Vec<f64>) = (0..2 * height)
        .flat_map(|j| (0..2 * width).map(move |i| (i, j)))
        .flat_map(|(i, j)| {
            let (nodes, node_weights) = (&nodes, &node_weights);
            (0..3).flat_map(move |b| {
                (0..3).map(move |a| {
                    (
                        DVec2::new(
                            x0 + (i as f64 + nodes[a]) * 0.5,
                            bottom + (j as f64 + nodes[b]) * 0.5,
                        ),
                        0.25 * node_weights[a] * node_weights[b],
                    )
                })
            })
        })
        .unzip();
    let body = reference.len();
    let mut along = vec![x0];
    along.extend((0..width).map(|i| x0 + i as f64 + 0.5));
    along.push(x0 + width as f64);
    for &xc in &along {
        reference.push(DVec2::new(xc, bottom));
        v0s.push(0.0);
    }
    let n = reference.len();
    let mass = v0s.clone();
    let mut solids = vec![solid; n];
    crate::spacetime::total_lagrangian::hooks::set_massless(None);
    // `EMERGE_PROBE_CONTACT_TOL`: the contacts' gap tolerance, cells.
    let mut settings = crate::spacetime::total_lagrangian::ConservingSettings::default();
    settings.contact_tolerance = float("EMERGE_PROBE_CONTACT_TOL", settings.contact_tolerance);
    // `EMERGE_PROBE_FORECAST`: hold contacts from the step their gap closes.
    settings.forecast_activation = std::env::var_os("EMERGE_PROBE_FORECAST").is_some();
    let share = crate::spacetime::total_lagrangian::massive_share(
        &layout,
        &reference,
        (&mass, &v0s),
        settings.occupation,
    );
    let mut x = reference.clone();
    let mut vel: Vec<DVec2> = share.iter().map(|s| DVec2::new(v0 * s[0], 0.0)).collect();
    let mut f = vec![DMat2::IDENTITY; n];
    let planes = [crate::spacetime::total_lagrangian::Plane {
        axis: 1,
        at: if drop > 0.0 { floor } else { bottom + 1e-9 },
        inward: 1.0,
    }];
    let penalty: Vec<f64> = (0..n)
        .map(|p| if p < body { 0.0 } else { factor * c_p * c_p })
        .collect();
    let total: f64 = mass.iter().sum();
    let centre = |x: &[DVec2], vel: &[DVec2]| -> (DVec2, DVec2) {
        (
            (0..n).map(|p| mass[p] * x[p]).sum::<DVec2>() / total,
            (0..n).map(|p| mass[p] * vel[p]).sum::<DVec2>() / total,
        )
    };
    let energy = |x: &[DVec2], vel: &[DVec2], f: &[DMat2]| -> f64 {
        (0..n)
            .map(|p| {
                0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p])
                    - mass[p] * gravity.dot(x[p])
            })
            .sum()
    };
    let gyration = |x: &[DVec2]| -> f64 {
        let (c, _) = centre(x, x);
        ((0..body)
            .map(|p| mass[p] * (x[p] - c).length_squared())
            .sum::<f64>()
            / total)
            .sqrt()
    };
    let gyration0 = gyration(&x);
    let (mut worst_gyration, mut nan) = (0.0f64, false);
    let (mut impact, mut rebound, mut touched) = (0.0f64, 0.0f64, false);
    let (start, start_v) = centre(&x, &vel);
    let start_x = start.x;
    let _ = start_v;
    let kinetic0 = 0.5 * total * reference_speed * reference_speed;
    let e0 = energy(&x, &vel, &f);
    // The step tolerance's energy: the start's kinetic and stored energy
    // and gravity's work over the block's own extent (`tolerance_energy`),
    // never `e0`, whose potential is measured from the layout's corner.
    let tolerance = crate::spacetime::total_lagrangian::tolerance_energy(
        (0..n)
            .map(|p| 0.5 * mass[p] * vel[p].length_squared() + v0s[p] * solid.energy(f[p]))
            .sum(),
        gravity,
        [(total, &x[..n])],
    );
    let mut cache = None;
    let (mut heat, mut worst_v, mut made, mut worst_made) = (0.0f64, 0.0f64, 0usize, 0.0f64);
    let (mut stopped, mut stop_x, mut creep) = (None::<f64>, 0.0, 0.0f64);
    let (mut newton, mut unconverged, mut cost, mut steps) = (0u64, 0u32, 0.0f64, 0usize);
    let mut lowest = f64::INFINITY;
    let mut residual = 0.0f64;
    let mut t = 0.0;
    // `EMERGE_PROBE_SLIDE_NEWTON=k`: every Newton iteration of step `k`.
    let watched: Option<usize> = std::env::var("EMERGE_PROBE_SLIDE_NEWTON")
        .ok()
        .and_then(|v| v.parse().ok());
    // `EMERGE_PROBE_SLIDE_ADAPTIVE=tol`: the step chosen by its error
    // (`total_lagrangian::controlled_step`), at most a frame, ending on every
    // frame's time.
    let control = std::env::var("EMERGE_PROBE_SLIDE_ADAPTIVE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .map(
            |tolerance| crate::spacetime::total_lagrangian::StepControl {
                tolerance,
                safety: 0.9,
                growth: 10.0,
                largest: f64::from(FRAME),
                // `EMERGE_PROBE_SUBSTEPS=2,4,6`: the tries' counts, KSD's 1, 2, 3
                // by default.
                substeps: match std::env::var("EMERGE_PROBE_SUBSTEPS").as_deref() {
                    Ok("2,4,6") => [2, 4, 6],
                    _ => [1, 2, 3],
                },
                // `EMERGE_PROBE_ESTIMATOR=acary`: Acary's estimate.
                estimator: match std::env::var("EMERGE_PROBE_ESTIMATOR").as_deref() {
                    Ok("acary") => crate::spacetime::total_lagrangian::Estimator::Acary,
                    _ => crate::spacetime::total_lagrangian::Estimator::Klapproth,
                },
            },
        );
    // `EMERGE_PROBE_SLIDE_DUMP=path`: every frame's state as JSON, for a
    // replay (positions in metres).
    let dump = std::env::var("EMERGE_PROBE_SLIDE_DUMP").ok();
    let mut frames: Vec<String> = Vec::new();
    let mut next_frame = f64::from(FRAME);
    let (mut step_length, mut kept, mut rejected, mut shortest) = (dt, 0u32, 0u32, f64::INFINITY);
    let (mut at_floor, mut contact_work) = (0u32, 0.0f64);
    let contact_set = crate::spacetime::total_lagrangian::ContactSet {
        planes: &planes,
        pairs: &[],
        outlines: &[],
        corners: &[],
        friction,
    };
    while t < end - 1e-12 {
        if watched == Some(steps + 1) {
            crate::spacetime::total_lagrangian::hooks::start(0);
        }
        let particles = crate::spacetime::total_lagrangian::Particles {
            x: &mut x,
            v: &mut vel,
            f: &mut f,
            mass: &mass,
            v0: &v0s,
        };
        let (report, taken) = match &control {
            Some(control) => {
                // A frame's time within round-off counts as reached: no
                // remainder step of a few `eps` (measured: 3.9e-15 s).
                while next_frame - t <= 1e-12 {
                    next_frame += f64::from(FRAME);
                }
                step_length = step_length.min(next_frame - t);
                let r = crate::spacetime::total_lagrangian::controlled_step(
                    &layout,
                    particles,
                    (&reference, &mut solids),
                    (gravity, &mut step_length),
                    (contact_set, &penalty, &settings),
                    &mut cache,
                    (control, tolerance.sqrt()),
                );
                kept += 1;
                rejected += r.rejected;
                shortest = shortest.min(r.dt / r.kept.len() as f64);
                at_floor += u32::from(r.at_floor);
                let work = r.kept.iter().map(|k| k.work.0).sum::<f64>();
                contact_work += work;
                // `EMERGE_PROBE_SLIDE_WORK=threshold`: every kept step whose
                // contact work exceeds `threshold` of 1/2 M v^2.
                if let Some(threshold) = std::env::var("EMERGE_PROBE_SLIDE_WORK")
                    .ok()
                    .and_then(|v| v.parse::<f64>().ok())
                    && (work / kinetic0).abs() > threshold
                {
                    println!(
                        "  t {t:.6}: dt {:.3e}, contact work {:+.3e}, reaching {}, at the floor {}, error {:.2e}, rejected {}",
                        r.dt,
                        work / kinetic0,
                        r.reaching,
                        r.at_floor,
                        r.error,
                        r.rejected
                    );
                }
                // The kept substeps as one report: their sums, their
                // worst, whether any pushed.
                let mut all = crate::spacetime::total_lagrangian::StepReport {
                    converged: true,
                    ..Default::default()
                };
                for k in &r.kept {
                    all.newton += k.newton;
                    all.gmres += k.gmres;
                    all.converged &= k.converged;
                    all.residual = all.residual.max(k.residual);
                    all.friction += k.friction;
                    all.push = all.push.max(k.push);
                    all.passes += k.passes;
                    all.contacts = all.contacts.max(k.contacts);
                    for (a, b) in all.phases.iter_mut().zip(k.phases) {
                        *a += b;
                    }
                }
                (all, r.dt)
            }
            None => (
                crate::spacetime::total_lagrangian::step_total_lagrangian_contact(
                    &layout,
                    particles,
                    &reference,
                    &solids,
                    (
                        dt,
                        crate::spacetime::total_lagrangian::ContactMode::Velocity,
                        gravity,
                    ),
                    (contact_set, &penalty, &settings),
                    &mut cache,
                ),
                dt,
            ),
        };
        if watched == Some(steps + 1) {
            for (solve, lines) in crate::spacetime::total_lagrangian::hooks::finish()
                .iter()
                .enumerate()
            {
                for line in lines {
                    println!("  solve {solve}: {line}");
                }
            }
        }
        t += taken;
        steps += 1;
        newton += u64::from(report.newton);
        unconverged += u32::from(!report.converged);
        residual = residual.max(report.residual);
        cost += report.phases.iter().sum::<f64>();
        heat += report.friction;
        if report.friction < 0.0 {
            made += 1;
            worst_made = worst_made.max(-report.friction / kinetic0);
        }
        let (c, v) = centre(&x, &vel);
        let exact = speed(t);
        if std::env::var_os("EMERGE_PROBE_SLIDE_TRACE").is_some() {
            println!(
                "  t {t:.4}: v {:+.5} (exact {:+.5}) v0, heat {:+.3e}, push {:.3e}, Newton {} residual {:.2e} converged {}, passes {}, contacts {}",
                v.x / reference_speed,
                exact / reference_speed,
                report.friction / kinetic0,
                report.push,
                report.newton,
                report.residual,
                report.converged,
                report.passes,
                report.contacts
            );
        }
        worst_v = worst_v.max((v.x - exact).abs() / reference_speed);
        if stops && stopped.is_none() && v.x <= 1e-3 * v0 {
            stopped = Some(t);
            stop_x = c.x;
        }
        if stopped.is_some() {
            creep = creep.max((c.x - stop_x).abs());
        }
        for q in &x[body..] {
            lowest = lowest.min(q.y - floor);
        }
        nan |= x.iter().chain(&vel).any(|q| !q.is_finite());
        worst_gyration = worst_gyration.max((gyration(&x) / gyration0 - 1.0).abs());
        if report.push > 0.0 && !touched {
            touched = true;
        } else if !touched {
            impact = v.y;
        }
        if touched {
            rebound = rebound.max(v.y);
        }
        if dump.is_some() && t >= next_frame - 1e-9 {
            next_frame += f64::from(FRAME);
            let metres = |q: DVec2| format!("[{:.6},{:.6}]", q.x * cell_m, q.y * cell_m);
            let body_points: Vec<String> = x[..body].iter().map(|&q| metres(q)).collect();
            let speeds: Vec<String> = vel[..body]
                .iter()
                .map(|q| format!("{:.5}", q.length() * cell_m))
                .collect();
            let outline: Vec<String> = x[body..].iter().map(|&q| metres(q)).collect();
            let now = energy(&x, &vel, &f);
            frames.push(format!(
                "{{\"time\":{t:.6},\"positions\":[{}],\"speed_m_s\":[{}],\"contact_points\":[{}],\"energy\":{:.9e},\"balance\":{:.3e}}}",
                body_points.join(","),
                speeds.join(","),
                outline.join(","),
                now,
                (now + heat - e0) / kinetic0
            ));
        }
    }
    if let Some(path) = &dump {
        let header = format!(
            "{{\"dx_m\":{cell_m},\"material\":\"elastic solid (corotated, Gonzalez discrete gradient)\",\"E_pa\":{e},\"nu\":{nu},\"rho_kg_m3\":{rho},\"friction\":{friction},\"drop_m\":{:.4},\"impact_speed_m_s\":{:.4},\"floor_y_m\":{:.4},\"floor_x_m\":[0,{:.4}],\"step\":\"{}\",\"precomputed\":true}}",
            drop * cell_m,
            (2.0 * g * drop).sqrt() * cell_m,
            floor * cell_m,
            f64::from(layout.nx as u32) * cell_m,
            match &control {
                Some(c) => format!(
                    "controlled, tol {} (Klapproth, Schiela, Deuflhard 2011, mean-velocity estimator)",
                    c.tolerance
                ),
                None => format!("fixed, {per_frame} per frame (route C)"),
            }
        );
        let text = format!(
            "{{\"header\":{header},\"frames\":[\n{}\n]}}\n",
            frames.join(",\n")
        );
        if let Some(dir) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(dir).expect("dump folder");
        }
        std::fs::write(path, text).expect("dump written");
    }
    let (c, v) = centre(&x, &vel);
    let e_end = energy(&x, &vel, &f);
    let bounce = if drop > 0.0 {
        format!(
            "
  drop {drop} cells: impact {:+.4} m/s, highest rebound {:+.4} m/s (ratio {:.4}), gyration change at worst {worst_gyration:.2e}, NaN {nan}",
            impact * cell_m,
            rebound * cell_m,
            rebound / impact.abs().max(f64::MIN_POSITIVE)
        )
    } else {
        String::new()
    };
    let moved = c.x - start_x;
    let exact_heat = friction * g * cosine * total * travel;
    format!(
        "block {width} x {height} cells of 1 cm, E {e:e} Pa, nu {nu}, mu {friction}, slope {tilt}, v0 {:.2} m/s, c_p {:.0} m/s, {per_frame} steps per frame (dt {dt:.3e} s, Courant {:.0}): {steps} steps to {t:.3} s
  centre velocity error at worst {worst_v:.2e} of {:.3} m/s; stop at {} s (exact {}); path {:.5} m (exact {:.5}, realism {:.4}); creep after the stop {:.2e} m; end speed {:+.3e} m/s (exact {:+.3e})
  heat {:.6} of the exact; energy balance (kinetic, stored, gravity, heat) {:+.2e} of 1/2 M v^2; friction made energy on {made} steps, at worst {worst_made:.2e}; lowest contact point {lowest:+.2e} cells
  Newton {:.2} per step, unconverged {unconverged} (worst residual {residual:.1e} of the tolerance's scale); {:.2} s per simulated s{}",
        v0 * cell_m,
        c_p * cell_m,
        c_p * dt,
        reference_speed * cell_m,
        stopped.map_or("never".into(), |s| format!("{s:.4}")),
        if stops { format!("{stop:.4}") } else { "never".into() },
        moved * cell_m,
        travel * cell_m,
        1.0 - (moved - travel).abs()
            / if travel > 0.0 {
                travel
            } else {
                0.5 * g * sine * end * end
            },
        creep * cell_m,
        v.x * cell_m,
        speed(t) * cell_m,
        if exact_heat > 0.0 { heat / exact_heat } else { heat / kinetic0 },
        (e_end + heat - e0) / kinetic0,
        newton as f64 / steps.max(1) as f64,
        cost / t,
        match &control {
            Some(c) => format!("\n  controlled, tol {}: {kept} steps kept, {rejected} rejected, {at_floor} at the estimator's floor, shortest {shortest:.3e} s, contact work {:+.3e} of 1/2 M v^2", c.tolerance, contact_work / kinetic0),
            None => String::new(),
        }
    ) + &bounce
}

/// The block's slide (`block_slide_run`) at `EMERGE_PROBE_SLIDE_STEPS`
/// steps per frame (default `1,4`). A probe.
#[test]
#[ignore = "diagnostic probe: run with --ignored --nocapture"]
fn probe_block_slide() {
    let steps = std::env::var("EMERGE_PROBE_SLIDE_STEPS").unwrap_or_else(|_| "1,4".into());
    for per_frame in steps.split(',').filter_map(|s| s.trim().parse().ok()) {
        println!("{}", block_slide_run(per_frame));
    }
}
