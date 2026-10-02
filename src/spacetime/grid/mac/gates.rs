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
//!   cell (Zhu and Bridson 4.2.5). Particles move with the velocity they
//!   gather, `x += v dt`, as the engine's G2P does.
//! - Four particles per cell, each placed at random inside its quarter of
//!   the cell; those within one cell of a free surface are moved along its
//!   normal to half a cell from it (Zhu and Bridson 4.2.1). Particle radius
//!   in the level set: the particle spacing; kernel radius: twice that
//!   (their section 5).
//! - J is the fluid's own: its logarithm advances by `dt tr(C)` each
//!   substep through `advance_log_volume_ratio`, with the engine's bounds
//!   [0.5, 2.0]. `C` is the velocity gradient gathered from the MAC grid.
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

use std::time::Instant;

use glam::{IVec2, Mat2, Vec2};

use super::field::{Field2, MacLayout};
use super::level_set::{SurfaceSettings, liquid_phi};
use super::solid::{FaceWeights, box_container, face_weights, sample_centres, sample_corners};
use super::transfer::{faces_to_particles, particles_to_faces};
use super::{ProjectionSettings, project, travel_limited_dt};
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
    corners: Field2,
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
}

/// Where things happen, for diagnosis only: no criterion reads it.
#[derive(Default)]
struct Probe {
    /// J reaching its bounds, by where the particle was: in a liquid cell
    /// of the tank, in an air cell of the tank, inside a wall.
    j_at_bounds_by_place: [usize; 3],
    /// Particles leaving the tank: time, position before, velocity,
    /// distance to the nearest wall before.
    crossings: Vec<(f32, Vec2, Vec2, f32)>,
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
            corners,
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
        }
    }

    fn g(&self) -> f32 {
        self.gravity.length()
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
        let (mut vel, _) = particles_to_faces(&self.layout, &self.x, &self.v, &self.c, &self.mass);
        for value in vel.u.data_mut() {
            *value += self.gravity.x * dt;
        }
        for value in vel.v.data_mut() {
            *value += self.gravity.y * dt;
        }
        let phi = self.phi();
        let system = super::pressure::assemble(
            &self.layout,
            dt,
            &vel,
            &self.weights,
            &phi,
            self.settings.theta_floor,
        );
        let solution = project(
            &self.layout,
            dt,
            &mut vel,
            &self.weights,
            &self.corners,
            &phi,
            &self.settings,
        );
        faces_to_particles(&self.layout, &vel, &self.x, &mut self.v, &mut self.c);
        let ln_bounds = (J_BOUNDS.0.ln(), J_BOUNDS.1.ln());
        for p in 0..self.x.len() {
            let dt_div = dt * (self.c[p].x_axis.x + self.c[p].y_axis.y);
            let advanced = self.log_j[p] + dt_div;
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
            let step = self.v[p] * dt;
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
        self.time += dt;
        frame.substeps += 1;
        frame.iterations.push(solution.iterations);
        frame.capped += u32::from(!solution.converged);
        frame.pressure = solution.pressure;
        frame.active = system.active;
        frame.phi = phi;
        frame.vel = vel;
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
            let limit = travel_limited_dt(
                speed,
                self.g(),
                self.layout.dx,
                self.settings.max_cells_per_substep,
            );
            let dt = limit.min(left);
            self.substep(dt, &mut frame);
            left -= dt;
        }
        frame
    }

    fn j(&self, p: usize) -> f32 {
        self.log_j[p].exp()
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
        (0..self.x.len())
            .map(|p| {
                let kinetic = 0.5 * self.v[p].length_squared();
                self.mass[p] * (kinetic + self.g() * (self.x[p].y - floor))
            })
            .sum()
    }

    fn inside_tank(&self, p: usize) -> bool {
        let q = self.x[p];
        q.x > self.tank.0.x && q.x < self.tank.1.x && q.y > self.tank.0.y && q.y < self.tank.1.y
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
             {} capped, largest travel {:.3} cell, {:.0} ms per simulated s (debug)",
            self.substeps as f32 / self.frames.max(1) as f32,
            self.max_substeps,
            self.iterations as f32 / self.solves.max(1) as f32,
            self.max_iterations,
            self.capped,
            self.largest_travel,
            1000.0 * wall_s / simulated_s
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
fn violent_scene(label: &str, x: Vec<Vec2>, report: &mut Report) {
    let tank = (Vec2::splat(4.0), Vec2::splat(68.0));
    let mut scene = Scene::new((72, 72), 0.01, 1.0, tank, x);
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
/// gather quadratic along each component and linear across it reads. A
/// probe: it measured the first gate run's quadratic gather at 1000 to
/// 3000 times the grid's divergence.
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
