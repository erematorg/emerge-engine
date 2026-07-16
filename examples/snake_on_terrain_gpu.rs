extern crate emerge_engine as emerge;

/// Snake crawling on REAL granular sand terrain -- GPU path, zero-copy rendering.
///
/// GPU counterpart to `snake_on_terrain.rs` (CPU), built after the full GPU multi-field
/// contact port (P2G grip scatter, point-cloud gather, Newton-Raphson LR normal fit,
/// resolve_contact's Coulomb + velocity-floor Baumgarte correction, G2P routing) landed
/// and was verified end to end (`gpu_multi_field_contact_produces_real_coulomb_slip_and_stick`,
/// `tests/gpu.rs`) -- this is the first VISUAL look at that work, not just headless
/// assertions.
///
/// Real, disclosed limitation: GPU has no `DirectionalContactGrip` equivalent yet (see
/// `GpuDirectionalGripParams`'s doc, `src/systems/gpu/step_params.rs`) -- friction is
/// plain symmetric Coulomb at `SimConfig::contact_friction`, uploaded once, not
/// live-adjustable per direction. So unlike the CPU version, there is NO steering input
/// here (asymmetric grip is what makes net-directional crawling possible at all) --
/// this scene only proves the OTHER real, already-verified claim: real CPG-driven
/// muscle activity pushing against real sand terrain, rendered live, at real GPU frame
/// rates. Steering support is real future work once GPU's directional grip lands.
///
///   cargo run --example snake_on_terrain_gpu --features render
use std::sync::Arc;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, FixedStepController, GpuSimulation, Lnn, MaterialRegistry,
    NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 128;
// REAL BUG FOUND AND FIXED 2026-07-15, TWO ROUNDS:
//
// Round 1: the original single `DT=0.1` fed the physics solver 0.1 simulated
// seconds per rendered frame -- at a real ~60fps display frame (~0.0167s real
// time), that's the world running at 6x real-time speed (already a known,
// documented issue elsewhere in this codebase -- `gpu_grid_resolution_cost`'s own
// `REAL_TIME_DT` comment, tests/gpu.rs). Once the terrain was recalibrated to real
// sand stiffness, that 6x inflation became a real, measured performance problem
// (54/128 substeps/frame).
//
// Round 2 (a REAL BUG INTRODUCED BY ROUND 1's OWN FIX, found live): the first fix
// attempt split this into `PHYSICS_DT=1/60` (correct) and a separate `CPG_DT` left
// at the OLD `0.1`, reasoned as "preserving basic_creature.rs's tuning." That
// reasoning was backwards. The CPG steps ONCE PER PHYSICS FRAME by whatever DT it's
// given, with no awareness of what a "frame" represents in real time. Before, one
// frame = 0.1 real seconds and the CPG advanced 0.1 per frame -- matched, 1:1 real
// time. After the split, one frame = 1/60 real seconds (6x shorter) but the CPG
// STILL advanced by the old 0.1 every frame -- meaning the muscle now cycled 6x
// FASTER in real wall-clock time than it was ever tuned for. Confirmed live: violent
// "up/down perma movement" and a genuine, escalating instability (vmax climbing
// from ~2 to >20 over a long run) -- real muscle energy being pumped in far faster
// than the body/contact system was ever validated to absorb. There is only ONE real
// DT: whatever a physics frame actually represents in real time. Both the physics
// solver AND the CPG must step by that SAME value to preserve the ORIGINAL
// real-time gait rate -- splitting them was the actual bug, not fixing anything.
const DT: f32 = 1.0 / 60.0;
const MUSCLE_GROUPS: u32 = 8;
const N_RINGS: usize = 2;
const N_PER_RING: usize = MUSCLE_GROUPS as usize / N_RINGS;
const RING_CROSS_COUPLING: f32 = 0.5;
const MUSCLE_AMPLITUDE: f32 = 0.9;
const CPG_BURN_IN_STEPS: usize = 600;
const SNAKE_CONTACT_GROUP: u32 = 1;
const FIBER_DIAG: f32 = 3.0;
const BODY_LEN: f32 = 18.0;
const BODY_CENTER: Vec2 = Vec2::new(64.0, 20.0);

fn make_cpg() -> Lnn {
    let mut lnn = Lnn::coupled_traveling_wave(N_RINGS, N_PER_RING, 1.0, RING_CROSS_COUPLING);
    for _ in 0..CPG_BURN_IN_STEPS {
        lnn.step(DT);
    }
    lnn
}

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuSimulation,
    renderer: Renderer,
    snake_range: std::ops::Range<usize>,
    muscle_group_of: Vec<u32>,
    lnn: Lnn,
    paused: bool,
    wave_speed: f32,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    // REAL BUG FOUND AND FIXED 2026-07-15: the previous version called
    // `sim.step_frame()` exactly once per rendered frame, silently ASSUMING each
    // render call corresponds to exactly `DT` (1/60s) of real elapsed time.
    // Real frames don't take exactly that long (measured: 41-57fps, i.e. real frame
    // times of 0.0175-0.024s, not the assumed 0.0167s) -- the mismatch between
    // assumed and actual elapsed time is exactly what produced the reported jitter
    // and inconsistent/slow-feeling motion (physics pace silently tracked whatever
    // the render frame rate happened to be, not real wall-clock time). `emerge`
    // already ships the correct tool for this exact problem
    // (`runtime::FixedStepController`, an accumulator-based fixed-timestep stepper)
    // -- this was simply never wired up in this example. `last_instant` measures
    // REAL elapsed time each frame; `stepper` converts that into the correct
    // (possibly 0, 1, or more) number of physics steps to run this frame,
    // decoupling physics pacing from render frame-rate jitter entirely.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
}

fn make_sim_data(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
) -> (GpuSimulation, std::ops::Range<usize>, Vec<u32>) {
    let config = SimConfig {
        contact_friction: 0.5,
        // REAL BUG FOUND AND FIXED 2026-07-15: the previous `min_dt: 0.01` override
        // was harmless for the OLD, ~750x-softer terrain (E=133.3), whose own real
        // CFL-safe timestep was comfortably above 0.01 so this floor never actually
        // engaged. `cfl_bound` (src/spacetime/solver/step.rs) clamps the chosen
        // substep to be AT LEAST `min_dt` regardless of what the material's own
        // stability bound requires -- once the terrain was correctly recalibrated to
        // E=1e5 (see the real-sand fix below), its true safe timestep dropped well
        // below 0.01, and this override then forced an UNSAFE, too-large step every
        // substep, causing a real explosion (confirmed live: user watched the scene
        // blow up after the stiffness fix alone). Removed -- inherits
        // `SimConfig::default()`'s own safe `1.0e-3`. `max_substeps_per_step` raised
        // to give the adaptive scheme enough real headroom to actually reach a
        // smaller substep within one frame's DT budget.
        max_substeps_per_step: 128,
        project_invalid_state: true,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };

    let terrain_particles = build_particles(
        &config,
        SpawnRegion {
            spacing: 0.5,
            box_size: IVec2::new(100, 12),
            box_center: Vec2::new(64.0, 10.0),
            material_id: 0,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        },
    );

    // REAL FIX 2026-07-15: the original (133.3, 0.333) was ~750x softer than this
    // codebase's OWN validated real-sand reference
    // (`sand_angle_of_repose_is_physical`, tests/accuracy.rs, uses
    // `from_young_modulus(1.0e5, 0.2)` -- E=1e5 is also sparkl/wgsparkl's own
    // canonical demo value). A DP material this soft deforms continuously under any
    // load instead of holding a rigid granular structure before yielding -- the
    // direct, measured cause of "looks like fluid, not sand" (user observation,
    // real GPU visual scene). `cohesionless` is a thin wrapper over
    // `from_young_modulus` (same Lamé conversion), so this is a pure recalibration,
    // not a different construction path.
    let terrain_mat = DruckerPragerMaterial::cohesionless(1.0e5, 0.2);
    let registry = MaterialRegistry::with_default(Box::new(terrain_mat));
    let mut sim = GpuSimulation::with_device(device, queue, config, terrain_particles, registry);

    let mut snake_mat = NeoHookeanMaterial::new(13.0, 26.0);
    snake_mat.active_stress_coeff = 80.0;
    snake_mat.viscosity = 150.0;
    let snake_mat_id = sim.register_material(Box::new(snake_mat));
    let snake_spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(36, 4),
        box_center: BODY_CENTER,
        material_id: snake_mat_id.id(),
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(sim.config())
    };
    let snake_range = sim.spawn_region(snake_spawn);

    let body_left = BODY_CENTER.x - BODY_LEN / 2.0;
    let mut muscle_group_of = Vec::with_capacity(snake_range.len());
    {
        let particles = sim.particles_mut();
        for i in snake_range.clone() {
            particles[i].contact_group = SNAKE_CONTACT_GROUP;
            let t = ((particles[i].x.x - body_left) / BODY_LEN).clamp(0.0, 1.0);
            let group = ((t * MUSCLE_GROUPS as f32) as u32).min(MUSCLE_GROUPS - 1);
            particles[i].muscle_group_id = group;
            muscle_group_of.push(group);
            let local_y = particles[i].x.y - BODY_CENTER.y;
            let flip = if group % 2 == 1 { -1.0 } else { 1.0 };
            particles[i].activation_dir = if local_y >= 0.0 {
                Vec2::new(-FIBER_DIAG * flip, 1.0).normalize()
            } else {
                Vec2::new(FIBER_DIAG * flip, 1.0).normalize()
            };
        }
    }
    sim.mark_particles_dirty();
    (sim, snake_range, muscle_group_of)
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
        let (sim, snake_range, muscle_group_of) = make_sim_data(Arc::new(device), Arc::new(queue));
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), fmt);
        renderer.set_camera(sim.queue(), GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "snake_on_terrain_gpu: {} particles ({} snake)  |  up/down wave speed  Space pause  R reset  Q quit  (steering API now exists -- GpuSimulation::set_grip_direction/set_grip_friction -- but the underlying directional effect is measurably unstable run to run, not wired into this demo yet; see gpu_directional_grip_is_direction_aware's #[ignore] reason)",
            sim.particle_count(),
            snake_range.len()
        );
        Self {
            surface,
            surface_config: sc,
            sim,
            renderer,
            snake_range,
            muscle_group_of,
            lnn: make_cpg(),
            paused: false,
            wave_speed: 1.0,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            stepper: FixedStepController::standard(DT, 1.0 / DT),
            last_instant: std::time::Instant::now(),
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.surface_config.width = w;
        self.surface_config.height = h;
        self.surface
            .configure(self.sim.device(), &self.surface_config);
        self.renderer
            .set_camera(self.sim.queue(), GRID as u32, w, h, 0.6, true);
    }

    fn reset(&mut self) {
        let (device, queue) = (self.sim.device().clone(), self.sim.queue().clone());
        let (sim, snake_range, muscle_group_of) = make_sim_data(device, queue);
        self.sim = sim;
        self.snake_range = snake_range;
        self.muscle_group_of = muscle_group_of;
        self.lnn = make_cpg();
        self.frame = 0;
        // Real accumulated leftover time from before the reset must not leak into
        // the new run (would cause a stutter of "catch-up" steps right after reset).
        self.stepper.reset();
        self.last_instant = std::time::Instant::now();
        println!("reset");
    }

    fn update_and_render(&mut self) {
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        if !self.paused {
            let steps = self.stepper.steps_for_frame(frame_delta);
            for _ in 0..steps {
                self.lnn.step(DT * self.wave_speed);
                let activations: Vec<f32> = self.lnn.activations().collect();
                {
                    let particles = self.sim.particles_mut();
                    for (offset, i) in self.snake_range.clone().enumerate() {
                        let group = self.muscle_group_of[offset] as usize;
                        particles[i].activation =
                            (MUSCLE_AMPLITUDE * activations[group]).clamp(0.0, 1.0);
                    }
                }
                self.sim.mark_particles_dirty();
                self.sim.step_frame();
                self.frame += 1;
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 1.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            let snap = self.sim.diagnostics_snapshot();
            println!(
                "frame={} fps={:.0} sub={}/{} vmax={:.3}",
                self.frame,
                fps,
                snap.substeps_last_step,
                self.sim.config().max_substeps_per_step,
                snap.max_particle_speed,
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render_gpu(
            self.sim.device(),
            self.sim.queue(),
            self.sim.particle_buffer(),
            self.sim.particle_count(),
            &view,
            true,
        );
        output.present();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, el: &ActiveEventLoop) {
        let w = Arc::new(
            el.create_window(
                winit::window::WindowAttributes::default()
                    .with_title("emerge -- Snake on real terrain (GPU)")
                    .with_inner_size(winit::dpi::LogicalSize::new(640u32, 640u32)),
            )
            .unwrap(),
        );
        self.state = Some(pollster::block_on(State::new(w.clone())));
        self.window = Some(w);
    }

    fn window_event(&mut self, el: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(s) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => el.exit(),
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
                    KeyCode::KeyR if pressed => s.reset(),
                    KeyCode::ArrowUp if pressed => {
                        s.wave_speed = (s.wave_speed + 0.2).min(3.0);
                        println!("wave_speed={:.1}", s.wave_speed);
                    }
                    KeyCode::ArrowDown if pressed => {
                        s.wave_speed = (s.wave_speed - 0.2).max(0.1);
                        println!("wave_speed={:.1}", s.wave_speed);
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
