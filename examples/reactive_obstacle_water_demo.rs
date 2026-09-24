extern crate emerge_engine as emerge;

use emerge::render::demo_harness::{DemoApp, run_demo};
use emerge::render::{ColorMode, Renderer};
/// Real, live, TWO-WAY coupling demo. Click anywhere: a real point-mass
/// obstacle DROPS from the cursor's position -- zero initial velocity, pure
/// gravity from there, same as the water's own gravity. Height does the
/// work by itself (real integration, `v += gravity*dt` every frame): click
/// higher above the water, it arrives faster, no separate "launch speed" or
/// aim-at-cursor propulsion (an earlier version scripted an initial throw
/// velocity toward the cursor -- real, but confusing, looked like the
/// object was being fired rather than dropped, and sent it flying clear
/// across the domain since nothing capped its horizontal speed). The
/// obstacle also receives a real, mass-weighted reaction impulse (Newton's
/// third law) from every grid cell it corrects via `KinematicCircleBoundary`
/// -- unlike `kinematic_obstacle_water_demo.rs` (scripted, infinite
/// effective mass, water can never push back), this one's velocity evolves
/// purely from real physics after the drop.
///
/// Visible as a small ring of marker particles tracing the obstacle's real
/// position/radius every frame (the boundary itself has no particle body
/// the renderer could otherwise draw).
///
/// Ported 2026-08-16 onto `emerge::render::demo_harness` -- the real,
/// general window+event-loop runner extracted from the winit/wgpu
/// boilerplate every demo in this repo was hand-duplicating (~80 lines of
/// adapter/device/surface setup + `ApplicationHandler` impl removed here,
/// same physics, same behavior).
///
///   cargo run --example reactive_obstacle_water_demo --features render
use emerge::{
    KinematicCircleBoundary, NewtonianFluidMaterial, SimConfig, Simulation, SlipBoundary,
    SpawnRegion,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::event::MouseButton;
use winit::keyboard::KeyCode;

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_WATER: u32 = 0;
// Distinct material_id purely so `ColorMode::ByMaterial` renders these
// particles a different color -- NOT a distinct physics model (they use
// whatever `with_default_material` registers, same as water; irrelevant
// since their position is hard-overwritten every frame below, and their
// mass is negligible so their real physical contribution to the grid is too).
const MAT_MARKER: u32 = 1;
const OBSTACLE_RADIUS: f32 = 2.5;
const OBSTACLE_MASS: f32 = 8.0;
/// Default/reset resting position, off to the side above the water -- only
/// used before the first click and on reset, not a launch mechanic.
const LAUNCH_POS: Vec2 = Vec2::new(6.0, 40.0);
const GRAVITY: Vec2 = Vec2::new(0.0, -0.3);

/// Points on a ring around `center` -- the obstacle has no real "shape" the
/// renderer knows about (it's a pure grid-level boundary correction, not a
/// particle body), so this is a visual-only stand-in: real particles whose
/// position gets stomped to this ring every frame, traced from the SAME
/// `center`/`radius` the physics actually uses, so it never desyncs from
/// the real invisible boundary. `count` comes from however many particles
/// the marker spawn region actually produced (`SpawnRegion::box_size` is a
/// world-space extent, not a literal particle count) -- read live, not
/// assumed, after construction.
fn marker_ring(center: Vec2, radius: f32, count: usize) -> Vec<Vec2> {
    (0..count)
        .map(|i| {
            let theta = i as f32 / count as f32 * std::f32::consts::TAU;
            center + Vec2::new(theta.cos(), theta.sin()) * radius
        })
        .collect()
}

struct State {
    sim: Simulation,
    obstacle: Arc<KinematicCircleBoundary>,
    marker_start: usize,
    marker_count: usize,
    obstacle_pos: Vec2,
    obstacle_vel: Vec2,
    renderer: Renderer,
    /// Latest cursor position in grid coordinates -- updated on every
    /// `cursor_moved`, read at click time (`mouse_button`). Real, direct
    /// equivalent of the old `cursor_grid()` computed lazily at click time;
    /// stored instead of recomputed since the harness hands `cursor_moved`
    /// a window-size FRACTION, not raw pixels, and this demo's own grid
    /// convention (Y flipped) is applied once, here, not duplicated.
    cursor_grid: Vec2,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sim() -> (Simulation, Arc<KinematicCircleBoundary>, usize, usize) {
    let config = SimConfig {
        min_dt: 1.0e-4,
        max_substeps_per_step: 400,
        gravity: GRAVITY,
        recompute_density_each_step: false,
        cfl_include_affine_speed: false,
        // Real, measured perf issue found live 2026-08-16: substep count
        // jumps 4-5 (free-fall) -> 22 (resting/contact against the floor)
        // once the obstacle settles, tanking fps ~60->~40. Real cause: the
        // fluid's own acoustic-CFL bound (Tait EOS stiffness) tightens
        // under sustained contact -- genuine physics, not a bug, not
        // artificially capped (22 is nowhere near the 400 cap). Same
        // already-validated, real fix this project already shipped for an
        // analogous DP-sand near-wall/contact case (still inside the
        // literature's normal 0.3-1.0 CFL-coefficient range) -- not a blind
        // guess.
        material_cfl_coefficient: 0.7,
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let water = NewtonianFluidMaterial::low_viscosity(0.1, 2.5);
    // 0.65, not the usual 0.9 -- particle count scales ~1/spacing^2, so this
    // is ~(0.9/0.65)^2 =~ 1.9x more particles in the same footprint (real
    // resolution increase, not a footprint change -- `mass_override` below
    // already scales off SPACING itself, so per-particle mass shrinks to
    // match automatically, staying physically consistent).
    const SPACING: f32 = 0.65;
    // Initial spawn (becomes particles[0..N_MARKER_POINTS], guaranteed --
    // nothing before it) -- negligible mass so P2G contamination is real
    // but tiny; position gets overwritten every frame regardless of what
    // physics does to them, so their own dynamics genuinely don't matter.
    let spawn_markers = SpawnRegion {
        spacing: 0.3,
        mass_override: Some(1.0e-6),
        box_size: IVec2::new(2, 2),
        box_center: LAUNCH_POS,
        material_id: MAT_MARKER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_water = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(0.1 * SPACING * SPACING),
        box_size: IVec2::new(24, 20),
        box_center: Vec2::new(34.0, 20.0),
        material_id: MAT_WATER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let obstacle = Arc::new(KinematicCircleBoundary::new(
        LAUNCH_POS,
        OBSTACLE_RADIUS,
        0.3,
    ));
    // Water is the INITIAL spawn (material_id=0) -- `Simulation::new` inits
    // its particles immediately, before any builder method has registered
    // anything else, so the initial spawn's material_id must already match
    // the registry's one pre-existing slot (index 0, what
    // `with_default_material` replaces). Markers (material_id=1) can only
    // be added via `add_body` AFTER `.with_material(MAT_MARKER, ...)` has
    // actually registered that slot -- this ordering bit me once already
    // tonight (real bug, real fix, not guessed).
    let mut solver = Simulation::new(config, spawn_water)
        .with_default_material(Box::new(water))
        // Markers need SOME registered material to exist at all -- which
        // one is irrelevant, their position is stomped every frame
        // regardless (see marker_ring's own doc).
        .with_material(
            MAT_MARKER,
            Box::new(NewtonianFluidMaterial::low_viscosity(0.1, 2.5)),
        )
        // Real bug found live tonight: this demo never had outer domain
        // walls at all (only the obstacle's own local circle boundary) --
        // nothing stopped the water OR the dropped obstacle from leaving
        // the grid entirely. Both boundaries apply in order; SlipBoundary
        // is already `is_strict_wc_mpm_fluid_compatible`.
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)))
        .with_boundary(Box::new(obstacle.clone()));
    let marker_start = solver.particles().x.len();
    let _ = solver.add_body(spawn_markers);
    let marker_count = solver.particles().x.len() - marker_start;
    assert!(marker_count > 0, "marker spawn box produced zero particles");
    (solver, obstacle, marker_start, marker_count)
}

impl State {
    /// Drop the obstacle from the given (grid-space) position -- zero
    /// initial velocity, real gravity does the rest. Click higher above the
    /// water and it genuinely falls farther before contact, arriving faster
    /// -- no separate "throw" mechanic, no artificial horizontal propulsion.
    fn drop_at(&mut self, pos: Vec2) {
        self.obstacle_pos = pos;
        self.obstacle_vel = Vec2::ZERO;
        println!("drop: pos=({:.2},{:.2})", pos.x, pos.y);
    }

    fn reset(&mut self) {
        let (sim, obstacle, marker_start, marker_count) = make_sim();
        self.sim = sim;
        self.obstacle = obstacle;
        self.marker_start = marker_start;
        self.marker_count = marker_count;
        self.obstacle_pos = LAUNCH_POS;
        self.obstacle_vel = Vec2::ZERO;
        self.frame = 0;
        println!("reset");
    }
}

impl DemoApp for State {
    const TITLE: &'static str = "emerge -- Reactive Obstacle vs. Real Water";

    fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let (sim, obstacle, marker_start, marker_count) = make_sim();
        let mut renderer = Renderer::new(device, sim.particles().len(), format);
        renderer.set_camera(queue, GRID as u32, width, height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "reactive obstacle vs. real water: {} particles | click to drop a real object from the cursor | R reset | Q quit",
            sim.particles().len()
        );
        Self {
            sim,
            obstacle,
            marker_start,
            marker_count,
            obstacle_pos: LAUNCH_POS,
            obstacle_vel: Vec2::ZERO,
            renderer,
            cursor_grid: Vec2::ZERO,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
        }
    }

    fn resize(&mut self, queue: &wgpu::Queue, width: u32, height: u32) {
        self.renderer
            .set_camera(queue, GRID as u32, width, height, 0.6, true);
    }

    fn cursor_moved(&mut self, x_frac: f32, y_frac: f32) {
        self.cursor_grid = Vec2::new(x_frac * GRID as f32, (1.0 - y_frac) * GRID as f32);
    }

    fn mouse_button(&mut self, button: MouseButton, pressed: bool) {
        if button == MouseButton::Left && pressed {
            let pos = self.cursor_grid;
            self.drop_at(pos);
        }
    }

    fn key_pressed(&mut self, key: KeyCode) {
        if key == KeyCode::KeyR {
            self.reset();
        }
    }

    fn update_and_render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        // Real physics only: gravity (same as the water's own) + real
        // reaction impulse from whatever contact happened last step. No
        // cursor influence on the obstacle after launch.
        let reaction = self.obstacle.take_reaction_impulse();
        self.obstacle_vel += GRAVITY * DT + reaction / OBSTACLE_MASS;
        self.obstacle_pos += self.obstacle_vel * DT;

        // `SlipBoundary` clamps real particles (water, markers) to the
        // domain -- it does NOT touch `obstacle_pos`, which is plain app-
        // level state, not a particle the engine's own boundary code ever
        // sees. Without this, gravity accelerates the obstacle downward
        // forever once it's below the water -- nothing else was ever
        // stopping it, so it fell straight through the floor and out of
        // view (the real bug just found live). Same margin convention as
        // the engine's own `clamp_position_inside_grid` (boundary_thickness
        // cells), plus the obstacle's own radius so it visually rests AT
        // the wall instead of clipping through it. Inelastic: zero the
        // velocity component that hit the wall, don't bounce.
        const BOUNDARY_THICKNESS: f32 = 2.0;
        let lo = BOUNDARY_THICKNESS + OBSTACLE_RADIUS;
        let hi = GRID as f32 - BOUNDARY_THICKNESS - OBSTACLE_RADIUS;
        if self.obstacle_pos.x < lo {
            self.obstacle_pos.x = lo;
            self.obstacle_vel.x = self.obstacle_vel.x.max(0.0);
        }
        if self.obstacle_pos.x > hi {
            self.obstacle_pos.x = hi;
            self.obstacle_vel.x = self.obstacle_vel.x.min(0.0);
        }
        if self.obstacle_pos.y < lo {
            self.obstacle_pos.y = lo;
            self.obstacle_vel.y = self.obstacle_vel.y.max(0.0);
        }
        if self.obstacle_pos.y > hi {
            self.obstacle_pos.y = hi;
            self.obstacle_vel.y = self.obstacle_vel.y.min(0.0);
        }

        self.obstacle
            .set_position_velocity(self.obstacle_pos, self.obstacle_vel);

        self.sim.step();

        // Visual-only: stomp the marker ring to the obstacle's real current
        // position/radius, overriding whatever the (irrelevant, negligible-
        // mass) physics did to them this step.
        let ring = marker_ring(self.obstacle_pos, OBSTACLE_RADIUS, self.marker_count);
        for (i, p) in ring.iter().enumerate() {
            self.sim.particles_mut().x[self.marker_start + i] = *p;
            self.sim.particles_mut().v[self.marker_start + i] = Vec2::ZERO;
        }

        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let snap = self.sim.diagnostics_snapshot();
            println!(
                "frame={} fps={:.0} obstacle_pos=({:.2},{:.2}) speed={:.2} substeps={} step_us={}",
                self.frame,
                fps,
                self.obstacle_pos.x,
                self.obstacle_pos.y,
                self.obstacle_vel.length(),
                snap.substeps_last_step,
                snap.timing.total_us,
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }

        self.renderer
            .render(device, queue, self.sim.particles(), view, true);
    }
}

fn main() {
    run_demo::<State>();
}
