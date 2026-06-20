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
    DruckerPragerMaterial, GpuSimulation, MaterialRegistry, SimConfig, SpawnRegion,
    StomakhinMaterial, build_particles,
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
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        max_substeps_per_step: 20,
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
        self.sim.step_frame();
        self.sim.phase_transition(
            |p| p.material_id == MAT_PACKED && p.v.length() > 5.0,
            MAT_SHATTER,
        );
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!("frame={} fps={:.0}", self.frame, fps);
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
        }
        if self.frame % 60 == 0 {
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
            WindowEvent::MouseInput { state, button, .. } => match button {
                MouseButton::Left => s.lmb = state == ElementState::Pressed,
                _ => {}
            },
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
