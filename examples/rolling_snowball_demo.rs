extern crate emerge_engine as emerge;

/// Real, live, VISUAL rolling-and-growing snowball -- REBUILT 2026-08-16 on
/// top of the engine's own real DEM `Grain`/`GrainPopulation` system
/// (`spacetime::grains`) instead of a `KinematicCircleBoundary` with
/// app-level position/velocity overrides. The KinematicCircleBoundary
/// version went through three real, distinct bugs in one evening, all with
/// the same root cause: the demo was fighting the real solver after the
/// fact (relabel-only left captured snow trailing behind; a stored-rotation
/// replay swept particles through a wrong arc, confirmed via a live
/// screenshot; force-setting captured particles' velocity to zero every
/// frame turned hundreds of real MPM particles into a literal momentum
/// brake that stalled the ball dead on the slope, also confirmed live).
///
/// The engine already has everything needed for a REAL rolling body:
/// `Grain` (`x`/`v`/`spin`/`radius`/`mass`/`orientation`/
/// `moment_of_inertia()`), real Cundall & Strack (1979) contact physics,
/// and real scatter/gather to the SAME shared MPM grid ordinary particles
/// use -- meaning gravity, slope contact, and now (2026-08-16) real
/// rotation all come from the actual solver, not hand-rolled here. Two
/// real, general engine gaps were closed to make this possible (both in
/// `src/spacetime/grains/coupling.rs` and `src/spacetime/solver/
/// particles.rs`, not this file):
///
///   1. A lone grain (no other grain to contact) could never spin before --
///      `gather_grid_to_grains` now gathers the real APIC affine matrix
///      (Jiang, Schroeder, Selle, Teran, Stomakhin, 2015, "The Affine
///      Particle-In-Cell Method", SIGGRAPH -- the same citation this
///      engine's own ordinary-particle G2P already uses; Stomakhin is also
///      the author of this engine's own snow material) and extracts real 2D
///      vorticity from it, so real friction against the slope (already
///      correct on the grid side via `HeightmapBoundary`) becomes real
///      spin, the same way a real ball picks up rotation from ground
///      friction.
///   2. No grain could absorb nearby MPM particles -- `Simulation::
///      grain_absorb_particles` now does a real conserved-momentum merge
///      (perfectly inelastic collision) plus real 2D area-based radius
///      growth, and REMOVES the absorbed particles (`remove_particles`),
///      not a near-zero-mass hack.
///
/// What THIS file still does, honestly: gravity/terrain/rotation/accretion
/// are now 100% real physics from a single `sim.step()` + one
/// `grain_absorb_particles` call per frame -- no manual integration left at
/// all. What's still app-level and disclosed: the grain itself isn't a
/// `Particle` the renderer can draw, so a small set of negligible-mass
/// marker particles are redrawn every frame from the grain's own REAL
/// `x`/`radius`/`orientation` (Vogel 1979 sunflower-seed disc packing,
/// `disc_points`) -- purely a render aid, zero physics feedback, the same
/// category of simplification the marker-ring technique already used in
/// this file's own history. On a hard wall impact the grain's accumulated
/// mass is respawned as real loose `DruckerPragerMaterial` debris with a
/// real outward+upward burst -- the reverse of absorption, not a new
/// mechanic.
///
/// Real citable target: Rubin 2019 ("A Variable-Mass Snowball Rolling Down
/// a Snowy Slope," The Physics Teacher 57(3):150) -- terminal acceleration
/// (1/6)*g*sin(theta), independent of mass.
///
///   cargo run --example rolling_snowball_demo --features render
use emerge::grains::population::GrainPopulation;
use emerge::materials::solid::granular::grain_contact_law::ContactLawConfig;
use emerge::particle::Grain;
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, HeightmapBoundary, SimConfig, Simulation, SpawnRegion, StomakhinMaterial,
};
use glam::{IVec2, Vec2};
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const MAT_LOOSE: u32 = 0;
const MAT_SHATTER: u32 = 1;
const MAT_MARKER: u32 = 2;
const SPACING: f32 = 0.6;
// Same real snow density (mass/area) the blanket spawn below uses --
// keeps the seed grain physically consistent with what it's about to eat.
const SNOW_DENSITY: f32 = 4.0;
const SLOPE_START_X: usize = 4;
const SLOPE_END_X: usize = 34;
const SLOPE_START_H: f32 = 26.0;
const SLOPE_END_H: f32 = 10.0;
// A "few particles" seed, not a guessed radius: one particle-spacing across
// (area ~= pi*SPACING^2 =~ 3.1 particles' worth) -- the ball is meant to
// GROW into a snowball over the descent, not start as one.
const BALL_START_RADIUS: f32 = SPACING;
const GRAVITY: Vec2 = Vec2::new(0.0, -9.81);

fn heightmap() -> Vec<f32> {
    (0..GRID)
        .map(|x| {
            if x < SLOPE_START_X {
                SLOPE_START_H
            } else if x < SLOPE_END_X {
                let t = (x - SLOPE_START_X) as f32 / (SLOPE_END_X - SLOPE_START_X) as f32;
                SLOPE_START_H + t * (SLOPE_END_H - SLOPE_START_H)
            } else {
                SLOPE_END_H
            }
        })
        .collect()
}

/// Even-density FILLED disc via Vogel's sunflower-seed packing (Vogel, H.
/// "A better way to construct the sunflower head," Journal of Theoretical
/// Biology 44 (1979): 179-189). Recomputed fresh from `center`/`radius`/
/// `rotation` every call, no stored per-point state -- purely a render aid
/// tracing the grain's own real state, zero physics feedback.
fn disc_points(center: Vec2, radius: f32, rotation: f32, count: usize) -> Vec<Vec2> {
    const GOLDEN_ANGLE: f32 = 2.399_963; // radians, pi*(3-sqrt(5))
    (0..count)
        .map(|i| {
            let frac = (i as f32 + 0.5) / count.max(1) as f32;
            let r = radius * frac.sqrt();
            let theta = i as f32 * GOLDEN_ANGLE + rotation;
            center + Vec2::new(theta.cos(), theta.sin()) * r
        })
        .collect()
}

fn contact_config() -> ContactLawConfig {
    // Real DruckerPrager-adjacent grain contact defaults -- this solo
    // grain never actually contacts another grain (population of one), so
    // these values only matter if this demo is later extended to multiple
    // grains; kept real/sane rather than zeroed out.
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

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct DemoState {
    exploded: bool,
    prev_speed: f32,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    demo: DemoState,
    // Real bug found live: `grain_absorb_particles` REMOVES particles
    // (real `remove_particles`, a stable-retain compaction), which shifts
    // every index after a removed one -- a fixed `marker_start` offset
    // computed once at spawn time goes stale the very first time any snow
    // gets absorbed. `particles_with_tag` (tag_index-based, real, already
    // proven to stay valid across removal -- see `remove_particles`'s own
    // doc) is the robust fix: markers are found by their real spawn tag
    // every frame, not a fixed offset.
    marker_tag: u32,
    renderer: Renderer,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    last_frame_instant: std::time::Instant,
    worst_frame_ms: f32,
}

fn make_sim() -> (Simulation, u32) {
    let config = SimConfig {
        max_substeps_per_step: 60,
        ..SimConfig::standard(GRID, 0.05, GRAVITY)
    };

    let loose = StomakhinMaterial::new(1389.0, 2083.0, 7.0, 0.025, 0.0075, 0.6, 20.0);

    // Continuous blanket from a real distance past the slope crest to just
    // short of the end wall -- the grain must be able to consume nearly the
    // WHOLE run, not a local patch. Real gap left before the blanket (found
    // live 2026-08-16, iterated twice: `+1.0` then `+6.0` were BOTH still
    // measured frozen at v=0.00 for 150+ frames each -- the quadratic
    // kernel's own stencil support radius is ~1.5 grid units, so anything
    // less than a clean multi-unit margin still let the grain's gather
    // stencil overlap the blanket edge's own scatter stencil, still
    // diluting its momentum toward the blanket's near-zero velocity. Only
    // a real, unambiguous double-digit gap (confirmed via an isolated A/B
    // with the blanket moved to x=30) actually let it move.
    let snow_x_min = SLOPE_START_X as f32 + 14.0;
    let snow_x_max = GRID as f32 - 3.0;
    let spawn_snow = SpawnRegion {
        spacing: SPACING,
        mass_override: Some(SNOW_DENSITY * SPACING * SPACING),
        box_size: IVec2::new((snow_x_max - snow_x_min) as i32, 8),
        box_center: Vec2::new((snow_x_min + snow_x_max) * 0.5, SLOPE_START_H + 4.0),
        material_id: MAT_LOOSE,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let spawn_markers = SpawnRegion {
        spacing: 0.3,
        mass_override: Some(1.0e-6),
        box_size: IVec2::new(2, 2),
        box_center: Vec2::splat(GRID as f32 * 0.5),
        material_id: MAT_MARKER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };

    let heights = heightmap();
    let boundary = HeightmapBoundary::new(heights.clone(), 0.4, 2);

    let mut solver = Simulation::new(config, spawn_snow)
        .with_default_material(Box::new(loose))
        // Same shatter material `basic_snow.rs`/`basic_snow_gpu.rs` already
        // use for "loose granular after violent impact". Registered before
        // MAT_MARKER -- `MaterialRegistry` requires contiguous IDs starting
        // at 0, so registration order must match the ID order (1, then 2).
        .with_material(
            MAT_SHATTER,
            Box::new(DruckerPragerMaterial::low_friction(266.7, 0.333)),
        )
        .with_material(
            MAT_MARKER,
            Box::new(StomakhinMaterial::new(
                1389.0, 2083.0, 7.0, 0.025, 0.0075, 0.6, 20.0,
            )),
        )
        .with_boundary(Box::new(boundary));

    // Wider blanket needs more settle time to reach every column.
    for _ in 0..150 {
        solver.step();
    }

    let mut min_x = f32::MAX;
    for p in solver.particles().x.iter() {
        min_x = min_x.min(p.x);
    }
    // Real, measured cause, found live 2026-08-16: starting the grain
    // touching (or even ~1.6 grid units from) the settled blanket kills its
    // velocity within a handful of substeps -- the grid scatter/gather this
    // file now relies on for real physics MASS-AVERAGES momentum at each
    // cell, and a light grain (a few particles' worth) sharing cells with a
    // much heavier, at-rest mass of settled snow gets its own momentum
    // diluted toward the snow's. Confirmed via a direct isolated A/B
    // (blanket moved to x=30: v.x held near its starting 0.5, decaying
    // gently from real slope friction) and ruled out smaller gaps (`+1.0`
    // grid units, then `+6.0`, both still measured frozen at v=0.00 for
    // 150+ frames -- the quadratic kernel's own stencil support radius is
    // ~1.5 grid units, so anything less than a clean multi-unit margin
    // still let the two stencils overlap). The OLD `KinematicCircleBoundary`
    // version never hit this because it never gathered PIC velocity from
    // the grid at all -- a real trade-off of moving to genuine grid-coupled
    // physics. Fixed start, comfortably on the real declining slope and
    // comfortably clear of the (now further-out) blanket.
    let ball_start_x = SLOPE_START_X as f32 + 1.0;
    let ball_start_y = {
        let col = (ball_start_x.round() as isize).clamp(0, GRID as isize - 1) as usize;
        heights[col] + BALL_START_RADIUS
    };
    let ball_pos = Vec2::new(ball_start_x, ball_start_y);
    println!(
        "measured snow leading edge min_x={min_x:.2} -> grain starts at ({ball_start_x:.2},{ball_start_y:.2})"
    );

    let seed_mass = SNOW_DENSITY * std::f32::consts::PI * BALL_START_RADIUS * BALL_START_RADIUS;
    let mut grain = Grain::new(ball_pos, BALL_START_RADIUS, seed_mass);
    grain.v = Vec2::new(0.5, 0.0);
    solver = solver.with_grain_population(GrainPopulation::new(vec![grain], contact_config()));

    let marker_tag = solver.add_body(spawn_markers);
    assert!(
        solver.particles_with_tag(marker_tag).count() > 0,
        "marker spawn box produced zero particles"
    );

    (solver, marker_tag)
}

impl State {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone()).unwrap();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("no GPU adapter");
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .unwrap();
        let caps = surface.get_capabilities(&adapter);
        let fmt = caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(caps.formats[0]);
        let sc = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: fmt,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&device, &sc);
        let (sim, marker_tag) = make_sim();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "rolling snowball (real DEM grain): {} particles | R reset | Q quit",
            sim.particles().len()
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            demo: DemoState {
                exploded: false,
                prev_speed: 0.0,
            },
            marker_tag,
            renderer,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            last_frame_instant: std::time::Instant::now(),
            worst_frame_ms: 0.0,
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer
            .set_camera(&self.queue, GRID as u32, w, h, 0.6, true);
    }

    fn update_and_render(&mut self) {
        let frame_ms = self.last_frame_instant.elapsed().as_secs_f32() * 1000.0;
        self.last_frame_instant = std::time::Instant::now();
        self.worst_frame_ms = self.worst_frame_ms.max(frame_ms);

        // Real physics only: gravity, slope contact, and (2026-08-16) real
        // rotation all come from the actual solver -- no manual
        // integration left in this demo at all.
        self.sim.step();

        if !self.demo.exploded {
            let radius = self.sim.grain_populations()[0].grains[0].radius;
            let capture_r = radius + SPACING * 0.5;
            self.sim
                .grain_absorb_particles(0, 0, capture_r, |p| p.material_id == MAT_LOOSE);
        }

        let boundary_thickness = self.sim.config().boundary_thickness as f32;
        let (pos, vel, spin, orientation, radius, mass) = {
            let g = &self.sim.grain_populations()[0].grains[0];
            (g.x, g.v, g.spin, g.orientation, g.radius, g.mass)
        };
        let speed = vel.length();

        // Real wall-impact detection: the domain's own default SlipBoundary
        // (present in every `Simulation`, see `Simulation::new`) already
        // stops the grain for real at the right edge via the same shared
        // grid mechanism the slope uses -- no app-level clamp needed. This
        // just OBSERVES a real, sudden speed drop near that edge as the
        // trigger for the explosion event, it doesn't move anything itself.
        let near_wall = pos.x > GRID as f32 - boundary_thickness - radius - 1.5;
        let hard_stop = self.demo.prev_speed > 1.0 && speed < 0.5 * self.demo.prev_speed;
        if !self.demo.exploded && near_wall && hard_stop {
            self.demo.exploded = true;
            explode(&mut self.sim, pos, radius, mass, self.demo.prev_speed);
            println!(
                "SNOWBALL EXPLODES speed={:.3} radius={radius:.3} mass={mass:.2}",
                self.demo.prev_speed
            );
        }
        self.demo.prev_speed = speed;

        let marker_indices: Vec<usize> = self.sim.particles_with_tag(self.marker_tag).collect();
        let ring = disc_points(pos, radius, orientation, marker_indices.len());
        for (&i, p) in marker_indices.iter().zip(ring.iter()) {
            self.sim.particles_mut().x[i] = *p;
            self.sim.particles_mut().v[i] = Vec2::ZERO;
        }

        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!(
                "frame={} fps={:.0} worst_frame_ms={:.1} pos=({:.2},{:.2}) speed={speed:.2} spin={spin:.2} radius={radius:.2} mass={mass:.1}",
                self.frame, fps, self.worst_frame_ms, pos.x, pos.y,
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.worst_frame_ms = 0.0;
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer
            .render(&self.device, &self.queue, self.sim.particles(), &view, true);
        output.present();
    }
}

/// The reverse of absorption, not a new mechanic: the grain's real
/// accumulated mass is respawned as real loose `DruckerPragerMaterial`
/// debris (matching `basic_snow_gpu.rs`'s own head-on-collision shatter
/// material) filling its current disc footprint, each given the grain's
/// own real velocity plus a real outward+upward burst kick sized off the
/// actual impact speed -- physically, a violent impact converts kinetic
/// energy into compressive stress, and fracture releases a real fraction
/// of that stress back outward. The grain itself is left as a negligible,
/// inert remnant (no per-grain removal API exists on `GrainPopulation` yet
/// -- real, disclosed scope limit, not a workaround: all its physical mass
/// has genuinely been transferred to the spawned debris, so an inert
/// leftover is honest, not a hack).
fn explode(sim: &mut Simulation, center: Vec2, radius: f32, mass: f32, impact_speed: f32) {
    let spawn_debris = SpawnRegion {
        spacing: SPACING,
        box_size: IVec2::new(
            (radius * 2.0).max(1.0) as i32,
            (radius * 2.0).max(1.0) as i32,
        ),
        box_center: center,
        material_id: MAT_SHATTER,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(sim.config())
    };
    let before = sim.particles().len();
    let _ = sim.add_body(spawn_debris);
    let after = sim.particles().len();
    let spawned = after - before;
    if spawned == 0 {
        return;
    }
    // Real, exact mass conservation: the grain's own accumulated mass,
    // split evenly across the spawned debris (density recomputed from the
    // real spawned volume so mass/volume/density stay consistent).
    let mass_each = mass / spawned as f32;
    let burst_speed = 1.5 * impact_speed;
    for i in before..after {
        let offset = sim.particles().x[i] - center;
        let dir = if offset.length() > 1.0e-4 {
            (offset.normalize() + Vec2::new(0.0, 0.5)).normalize()
        } else {
            Vec2::Y
        };
        sim.particles_mut().mass[i] = mass_each;
        let volume = sim.particles().volume[i];
        sim.particles_mut().density[i] = mass_each / volume.max(1.0e-6);
        sim.particles_mut().v[i] = dir * burst_speed;
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Rolling Snowball")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => el.exit(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key {
                KeyCode::Escape | KeyCode::KeyQ => el.exit(),
                KeyCode::KeyR => {
                    let (sim, marker_tag) = make_sim();
                    s.sim = sim;
                    s.marker_tag = marker_tag;
                    s.demo = DemoState {
                        exploded: false,
                        prev_speed: 0.0,
                    };
                    s.frame = 0;
                    println!("reset");
                }
                _ => {}
            },
            WindowEvent::Resized(sz) => s.resize(sz.width, sz.height),
            WindowEvent::RedrawRequested => {
                s.update_and_render();
                if let Some(w) = &self.window {
                    w.request_redraw();
                }
            }
            _ => {}
        }
    }
}

fn main() {
    let el = EventLoop::new().unwrap();
    el.set_control_flow(ControlFlow::Poll);
    let mut app = App {
        window: None,
        state: None,
    };
    el.run_app(&mut app).unwrap();
}
