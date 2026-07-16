extern crate emerge_engine as emerge;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    CorotatedMaterial, Lnn, NeoHookeanMaterial, RatchetFrictionBoundary, SimConfig, Simulation,
    SpawnRegion,
};
use glam::{IVec2, Vec2};

/// First terrain-material demo: a creature (same real peristaltic body as
/// `basic_creature`) crawling through a field of grass blades. The point of
/// this demo is passive terrain interaction, not creature locomotion tuning --
/// grass blades are a soft `CorotatedMaterial` (large-rotation bending is
/// exactly what Corotated handles better than NeoHookean's small-strain
/// assumption), rooted only by their own weight + friction against the same
/// `RatchetFrictionBoundary` the creature's feet already use -- no special
/// "pin" mechanism, no per-material collision code. Both bodies share one
/// grid; bending emerges from ordinary MPM contact, nothing bespoke.
///
/// The creature's own locomotion has a known, documented, still-open bug (see
/// `combined_kirchhoff_stress` doc in `src/spacetime/transfer.rs`): net drift
/// stalls after ~650 sim-seconds even though the CPG stays alive. That's
/// unrelated to this demo's purpose (grass interaction) and left as-is -- the
/// creature still crawls for real distance before it settles, which is enough
/// to push through several blades.
///
/// Mouse controls (the same "classic controls" pattern as basic_jellies/
/// basic_fluids/basic_sand/basic_snow): left-click = push (radial impulse
/// outward from the cursor), right-click = pull (inward) -- lets you shove
/// the creature into the grass directly, independent of its own imperfect
/// autonomous crawl, to see the bending/sway interaction on demand.
///
///   cargo run --example grass_field --features "render"
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 96;
const DT: f32 = 0.1;
const MAT_BODY: u32 = 0;
const MUSCLE_GROUPS: u32 = 8;
const N_RINGS: usize = 2;
const N_PER_RING: usize = MUSCLE_GROUPS as usize / N_RINGS;
const RING_CROSS_COUPLING: f32 = 1.0;
const MUSCLE_AMPLITUDE: f32 = 0.9;
const CPG_BURN_IN_STEPS: usize = 600;

// Grass blade x-positions, all clear of the body's spawn footprint (it spans
// roughly x=36..60 at spawn and free-falls onto it from y=20 -- a blade placed
// under that would get crushed by the landing impact, not bent by a crawl).
// Real found bug (2026-07-11): an earlier layout placed blades directly under
// the spawn box and every one of them collapsed to J=0.000 (fully degenerate)
// on the very first drop, before the creature ever crawled anywhere. Starting
// clear of the landing zone means blades only ever get pushed by an actual
// crawl, not an initial fall.
//
// Moved closer (64 -> 62 start) the same day, after the viscosity fix: the
// creature's crawl is now correctly damped (real, sustained, ~0.2 units/step
// window, not the old undamped/eventually-collapsing rate), which is slower
// than an artificially undamped crawl -- a demo layout adjustment, not a
// physics change, so contact happens within a reasonable watch time.
const GRASS_X_POSITIONS: [f32; 8] = [62.0, 65.0, 68.0, 71.0, 74.0, 77.0, 80.0, 83.0];
const GRASS_BLADE_HEIGHT_CELLS: i32 = 12; // world height = cells * spacing = 6 units
const GRASS_SPACING: f32 = 0.5;

fn make_cpg() -> Lnn {
    let mut lnn = Lnn::coupled_traveling_wave(N_RINGS, N_PER_RING, 1.0, RING_CROSS_COUPLING);
    for _ in 0..CPG_BURN_IN_STEPS {
        lnn.step(DT);
    }
    lnn
}

fn make_sim() -> (
    Simulation,
    std::ops::Range<usize>,
    Arc<RatchetFrictionBoundary>,
) {
    // Real basic_creature.rs tuning -- see that file's history for why
    // (13,26,40,fiber=0.3,1.0) is the settled-on tradeoff. viscosity=150 +
    // the ORIGINAL (min_dt=0.01, max_substeps=64) config, not the initially-
    // tried viscosity=400/finer-timestep combination: that gave a perfectly
    // flat, non-decaying drift in a headless sweep, but caused real
    // interactive lag live (up to 512 substeps/frame in debug mode).
    // viscosity=150 at this cheaper config was independently verified to
    // sustain real drift far longer than untreated (15,000+ steps vs ~6,500),
    // at the SAME substep budget the pre-fix code already used.
    let mut mat = NeoHookeanMaterial::new(13.0, 26.0);
    mat.active_stress_coeff = 40.0;
    mat.viscosity = 150.0;
    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 64,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(48.0, 20.0);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(24, 6),
        box_center: body_center,
        material_id: MAT_BODY,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let ratchet = Arc::new(RatchetFrictionBoundary::new(4, 0.1, 0.95, Vec2::X));
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(Arc::clone(&ratchet)));

    let body_range = 0..solver.particles().len();
    let body_left = body_center.x - 12.0;
    let fiber_dir = Vec2::new(0.3, 1.0).normalize();
    {
        let particles = solver.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
            particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
            particles.activation_dir[i] = fiber_dir;
        }
    }

    // Grass: Corotated rather than NeoHookean -- bending is almost pure rotation
    // with little stretch, exactly the regime Corotated's polar-decomposition
    // formulation is built for (NeoHookean's small-strain assumption degrades
    // under large rotation). No activation, no muscle groups -- purely passive,
    // deformed only by gravity settling and creature contact.
    //
    // Stiffness (120, 240) -- NOT the initially-tried (1.5, 3.0). Real bug found
    // 2026-07-11 (user: "the grass falls flat"): a solo blade with NO creature
    // anywhere near it still buckled flat under its own weight alone at (1.5,3.0)
    // AND at (15,30) -- genuine Euler-style self-weight buckling of a thin, tall
    // column, unrelated to any creature contact. Verified via a real headless
    // sweep (solo blade, 6000-step horizon): (100,200) still buckles (delayed to
    // step ~1400), (120,240) stays upright the whole run, (150,300) also stable.
    // Picked 120/240 for the most bend under real contact while still passing the
    // no-creature stability check with margin. This ends up numerically STIFFER
    // than the creature's own tissue (13,26) -- not a contradiction: a thin/tall
    // column needs far more bending stiffness to resist self-buckling than a
    // squat/wide body does, regardless of which material "feels" softer.
    let grass_mat_id = solver.register_material(Box::new(CorotatedMaterial::new(120.0, 240.0)));
    for &x in GRASS_X_POSITIONS.iter() {
        let blade_center = Vec2::new(
            x,
            6.0 + GRASS_BLADE_HEIGHT_CELLS as f32 * GRASS_SPACING * 0.5,
        );
        let blade_spawn = SpawnRegion {
            spacing: GRASS_SPACING,
            box_size: IVec2::new(2, GRASS_BLADE_HEIGHT_CELLS),
            box_center: blade_center,
            material_id: grass_mat_id.0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(solver.config())
        };
        let _tag = solver.add_body(blade_spawn);
    }

    (solver, body_range, ratchet)
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    sim: Simulation,
    body_range: std::ops::Range<usize>,
    lnn: Lnn,
    paused: bool,
    ratchet: Arc<RatchetFrictionBoundary>,
    renderer: Renderer,
    frame: u64,
    telemetry_timer: std::time::Instant,
    spawn_centroid_x: f32,
    cursor_pos: [f32; 2],
    lmb: bool,
    rmb: bool,
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
        let (sim, body_range, ratchet) = make_sim();
        let n = body_range.len().max(1) as f32;
        let spawn_centroid_x: f32 = body_range
            .clone()
            .map(|i| sim.particles().x[i].x)
            .sum::<f32>()
            / n;
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "grass_field: {} particles ({} blades)  |  LMB push  RMB pull  Space pause  R reset  Q quit",
            sim.particles().len(),
            GRASS_X_POSITIONS.len()
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            body_range,
            lnn: make_cpg(),
            paused: false,
            ratchet,
            renderer,
            frame: 0,
            telemetry_timer: std::time::Instant::now(),
            cursor_pos: [0.0; 2],
            lmb: false,
            rmb: false,
            spawn_centroid_x,
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

    /// Same convention as basic_jellies/basic_fluids/basic_sand/basic_snow:
    /// screen pixels -> grid coordinates, y flipped (screen down = grid up).
    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn update_and_render(&mut self) {
        if !self.paused {
            self.lnn.step(DT);
            let activations: Vec<f32> = self.lnn.activations().collect();
            let body_range = self.body_range.clone();
            let particles = self.sim.particles_mut();
            for i in body_range {
                let group = particles.muscle_group_id[i] as usize;
                particles.activation[i] = (MUSCLE_AMPLITUDE * activations[group]).clamp(0.0, 1.0);
            }
            if self.lmb || self.rmb {
                let mag = if self.lmb { 3.0 } else { -3.0 };
                self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, mag);
            }
            self.sim.step();
            self.frame += 1;
        }
        if self.telemetry_timer.elapsed().as_secs_f32() >= 1.0 {
            // Light "sensing" hook: how much grass sits within a radius of the
            // creature right now -- a real, already-existing engine query
            // (`count_near`), not new infra. Demonstrates the sensing/cover use
            // case cheaply alongside the movement/bending one.
            let n = self.body_range.len().max(1) as f32;
            let particles = self.sim.particles();
            let centroid: Vec2 = self
                .body_range
                .clone()
                .map(|i| particles.x[i])
                .fold(Vec2::ZERO, |a, b| a + b)
                / n;
            // Radius 14, not 6: the body itself is ~12 units wide (half-width from
            // centroid to leading edge), so a tighter radius reads near-zero contact
            // even while the body is actively pushing into blades at its front edge
            // (found via a real headless check: blade x-position visibly shifted
            // +5.2 units under body contact while a radius-6 count still read 0).
            let nearby_grass = self.sim.count_near(centroid, 14.0, 1);
            let mut min_j = f32::INFINITY;
            let mut max_j = f32::NEG_INFINITY;
            let mut mean_speed = 0.0f32;
            for i in 0..particles.len() {
                let j = particles.deformation_gradient[i].determinant();
                min_j = min_j.min(j);
                max_j = max_j.max(j);
                mean_speed += particles.v[i].length();
            }
            mean_speed /= particles.len() as f32;
            println!(
                "frame {:<6} centroid=({:.3},{:.3})  drift_x={:+.3}  J=[{min_j:.3},{max_j:.3}] \
                 mean|v|={mean_speed:.4}  grass within 14 units: {nearby_grass}",
                self.frame,
                centroid.x,
                centroid.y,
                centroid.x - self.spawn_centroid_x
            );
            self.telemetry_timer = std::time::Instant::now();
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

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Grass field [terrain interaction]")
                    .with_inner_size(winit::dpi::LogicalSize::new(640u32, 480u32)),
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
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => s.lmb = state == ElementState::Pressed,
                MouseButton::Right => s.rmb = state == ElementState::Pressed,
                _ => {}
            },
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(key),
                        state,
                        ..
                    },
                ..
            } => {
                let pressed = state == ElementState::Pressed;
                match key {
                    KeyCode::Escape | KeyCode::KeyQ if pressed => el.exit(),
                    KeyCode::Space if pressed => {
                        s.paused = !s.paused;
                        println!("{}", if s.paused { "PAUSED" } else { "RUNNING" });
                    }
                    KeyCode::KeyR if pressed => {
                        let (sim, range, ratchet) = make_sim();
                        let n = range.len().max(1) as f32;
                        s.spawn_centroid_x =
                            range.clone().map(|i| sim.particles().x[i].x).sum::<f32>() / n;
                        s.sim = sim;
                        s.body_range = range;
                        s.ratchet = ratchet;
                        s.lnn = make_cpg();
                        s.frame = 0;
                        println!("reset");
                    }
                    _ => {}
                }
            }
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
