extern crate emerge_engine as emerge;

/// GPU snowballs colliding — Stomakhin 2013 snow plasticity, zero CPU readback.
///
///   Mat 0  soft powder (blue)  — low hardening, wide plastic limits
///   Mat 1  packed snow (gold)  — high hardening, tight limits
///   Mat 2  shatter     (cyan)  — loose granular after violent impact
///
///   cargo run --example basic_snow_gpu --features "render"
use std::sync::Arc;

use emerge::diagnostics::log_frame_gpu;
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, FixedStepController, GpuSimulation, MaterialRegistry, SimConfig,
    SpawnRegion, StomakhinMaterial, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_SOFT: u32 = 0;
const MAT_PACKED: u32 = 1;
const MAT_SHATTER: u32 = 2;
const BALL_R: f32 = 9.0;
const BALL_A: Vec2 = Vec2::new(16.0, 44.0);
const BALL_B: Vec2 = Vec2::new(48.0, 44.0);
const SPEED: f32 = 15.0;
const LABELS: &[(u32, &str)] = &[
    (MAT_SOFT, "soft"),
    (MAT_PACKED, "packed"),
    (MAT_SHATTER, "shatter"),
];

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuSimulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
    /// Real-time-decoupled stepping -- see `basic_fluids_gpu.rs`'s own field
    /// doc for the full real bug/fix writeup.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
    max_steps_seen: usize,
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        max_substeps_per_step: 20,
        // Deliberately weak, NOT real IRL gravity (real g_grid ~= 981 via
        // SimConfig::earth) -- tuned down for a calmer, more legible demo at
        // this grid scale. Disclosed, deferred: basic_snow_gui.rs's
        // gravity_fraction slider is the real-IRL-with-live-control
        // pattern, not yet ported to every plain example.
        gravity: Vec2::new(0.0, -0.08),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn_ball = |center: Vec2, mat: u32, seed: u32| SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new((BALL_R * 2.0) as i32, (BALL_R * 2.0) as i32),
        box_center: center,
        material_id: mat,
        precompute_initial_volumes: true,
        rng_seed: seed,
        ..SpawnRegion::for_sim(&config)
    };
    let mut particles = build_particles(&config, spawn_ball(BALL_A, MAT_SOFT, 1));
    for p in particles.iter_mut() {
        p.v.x = SPEED;
    }
    let mut right = build_particles(&config, spawn_ball(BALL_B, MAT_PACKED, 2));
    for p in right.iter_mut() {
        p.v.x = -SPEED;
    }
    particles.extend(right);

    let mut registry = MaterialRegistry::with_default(Box::new(StomakhinMaterial::new(
        1389.0, 2083.0, 7.0, 0.025, 0.0075, 0.6, 20.0,
    )));
    registry.insert(
        MAT_PACKED,
        Box::new(
            StomakhinMaterial::new(1389.0, 2083.0, 10.0, 0.012, 0.004, 0.6, 20.0)
                .with_cohesion(400.0),
        ),
    );
    registry.insert(
        MAT_SHATTER,
        Box::new(DruckerPragerMaterial::low_friction(266.7, 0.333)),
    );
    GpuSimulation::with_device(device, queue, config, particles, registry)
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
                required_limits: adapter.limits(), // use full hardware limits, not wgpu defaults
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
        let sim = make_sim_data(Arc::new(device), Arc::new(queue));
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), fmt);
        renderer.set_camera(sim.queue(), GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "snow GPU: {} particles  |  LMB push  R reset  Q quit",
            sim.particle_count()
        );
        Self {
            surface,
            surface_config: sc,
            sim,
            renderer,
            cursor_pos: [0.0; 2],
            lmb: false,
            frame: 0,
            fps_timer: std::time::Instant::now(),
            fps_frames: 0,
            stepper: FixedStepController::standard(DT, 60.0),
            last_instant: std::time::Instant::now(),
            max_steps_seen: 0,
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

    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn reset(&mut self) {
        let (device, queue) = (self.sim.device().clone(), self.sim.queue().clone());
        self.sim = make_sim_data(device, queue);
        self.frame = 0;
        self.stepper.reset();
        self.last_instant = std::time::Instant::now();
        println!("reset");
    }

    fn update_and_render(&mut self) {
        if self.lmb {
            self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, 10.0);
        }
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        let now = std::time::Instant::now();
        let frame_delta = (now - self.last_instant).as_secs_f32();
        self.last_instant = now;
        let steps = self.stepper.steps_for_frame(frame_delta);
        self.max_steps_seen = self.max_steps_seen.max(steps);
        for _ in 0..steps {
            self.sim.step_frame();
            self.frame += 1;
            // Gated, not evaluated every step -- `phase_transition` does a real,
            // BLOCKING GPU->CPU sync internally (`sync_particles_blocking`, see
            // its own doc), so calling it once per real simulation step (rather
            // than once per RENDER callback, the pre-2026-07-30 behavior) was a
            // genuine, self-inflicted cost multiplication once multiple steps
            // could happen per callback -- confirmed live: fps collapsed to
            // 13-16 with a chronic 4-steps-per-render pattern. Isolated timing
            // showed BOTH `step_frame()` (~6ms) and `phase_transition()` alone
            // (~1ms) are individually cheap -- the real cost is the sync POINT
            // itself breaking CPU/GPU pipelining in the live interleaved
            // render+compute loop (a real, well-known GPU perf pattern:
            // synchronization stalls cost far more in context than in
            // isolation). `is_multiple_of(15)`, matching `material_sandbox_
            // gpu`'s own already-proven gated-scan interval, not a fresh guess.
            if self.frame.is_multiple_of(15) {
                // Fracture trigger: real plastic compression (Jp), not raw
                // speed -- the old `v.length() > 5.0` fired at launch,
                // before any collision (already fixed in basic_snow.rs/
                // basic_snow_gui.rs 2026-08-16; this GPU sibling was missed
                // at the time and carried the same live regression until now).
                self.sim.phase_transition(
                    |p| p.material_id == MAT_PACKED && p.plastic_volume_ratio < 0.9,
                    MAT_SHATTER,
                );
            }
            if self.frame.is_multiple_of(60) {
                log_frame_gpu(self.frame, DT, self.sim.particles(), LABELS, 1);
                let snap = self.sim.diagnostics_snapshot();
                println!(
                    "  non_finite={}  out_of_bounds={}  max_speed={:.3}  sub={}  cfl={:.4}",
                    snap.non_finite_particle_values,
                    snap.out_of_bounds_particles,
                    snap.max_particle_speed,
                    snap.substeps_last_step,
                    snap.cfl_number,
                );
            }
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!(
                "frame={} fps={:.0} max_steps_per_render={}",
                self.frame, fps, self.max_steps_seen
            );
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
            self.max_steps_seen = 0;
        }
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
                    .with_title("emerge — Snow GPU [Stomakhin 2013: soft / packed / shatter]")
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
            WindowEvent::CursorMoved { position, .. } => {
                s.cursor_pos = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => s.lmb = state == ElementState::Pressed,
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
                KeyCode::KeyR => s.reset(),
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
