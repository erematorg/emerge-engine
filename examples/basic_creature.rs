extern crate emerge_engine as emerge;

use emerge::render::{ColorMode, Renderer};
use emerge::{FrictionBoundary, NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};
use std::f32::consts::TAU;
/// CPU creature -- NeoHookean soft body with peristaltic muscle activation.
///
/// Traveling wave of vertical muscle contraction -- segments squats into floor
/// -> grips -> neighbors slide forward. Arrow keys steer / adjust wave speed.
/// Space pauses. R resets.
///
///   cargo run --example basic_creature --features "render"
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const MAT_BODY: u32 = 0;
const MUSCLE_GROUPS: u32 = 8;

// Per-segment colors matching the SoftZoo rainbow palette (ByMaterial slots 0""7).
// ColorMode::ByMaterial assigns color by material_id % 16, so we encode muscle group
// as material_id directly for rendering. Physics still uses MAT_BODY internally.
// For simplicity we render via ByMaterial which gives blue for all (one material).
// Advanced: override muscle group rendering via a custom color callback.

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
    sim_time: f32,
    paused: bool,
    wave_speed: f32,
    wave_amplitude: f32,
    steer: f32,
    renderer: Renderer,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sim() -> (Simulation, std::ops::Range<usize>) {
    let mut mat = NeoHookeanMaterial::new(5.0, 10.0);
    mat.active_stress_coeff = 25.0;
    let config = SimConfig {
        min_dt: 0.01,
        max_substeps_per_step: 8,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let body_center = Vec2::new(32.0, 20.0);
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(24, 6),
        box_center: body_center,
        material_id: MAT_BODY,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(mat))
        .with_boundary(Box::new(FrictionBoundary::new(4, 0.65)));

    let body_range = 0..solver.particles().len();
    let body_left = body_center.x - 12.0;
    {
        let particles = solver.particles_mut();
        for i in body_range.clone() {
            let t = ((particles.x[i].x - body_left) / 24.0).clamp(0.0, 1.0);
            particles.muscle_group_id[i] = (t * MUSCLE_GROUPS as f32) as u32;
            particles.activation_dir[i] = Vec2::Y;
        }
    }
    (solver, body_range)
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
            .request_device(&wgpu::DeviceDescriptor::default())
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
        let (sim, body_range) = make_sim();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByActivation);
        println!(
            "creature: {} particles  |  up/down wave speed  left/right steer  Space pause  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            body_range,
            sim_time: 0.0,
            paused: false,
            wave_speed: 1.0,
            wave_amplitude: 0.9,
            steer: 0.0,
            renderer,
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
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer
            .set_camera(&self.queue, GRID as u32, w, h, 0.6, true);
    }

    fn update_and_render(&mut self) {
        if !self.paused {
            self.sim_time += DT;
            let t = self.sim_time;
            let body_range = self.body_range.clone();
            let (wave_speed, wave_amplitude, steer) =
                (self.wave_speed, self.wave_amplitude, self.steer);
            let particles = self.sim.particles_mut();
            for i in body_range {
                let group = particles.muscle_group_id[i] as f32;
                let phase = group / MUSCLE_GROUPS as f32 * TAU;
                let wave = (TAU * wave_speed * t - phase + steer).sin();
                particles.activation[i] = wave_amplitude * (wave * 0.5 + 0.5);
            }
            self.sim.step();
            self.frame += 1;
        }
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!("frame={} fps={:.0}", self.frame, fps);
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
                    .with_title("emerge -- Creature [peristaltic locomotion]")
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
                        let (sim, range) = make_sim();
                        s.sim = sim;
                        s.body_range = range;
                        s.sim_time = 0.0;
                        s.frame = 0;
                        println!("reset");
                    }
                    KeyCode::ArrowUp if pressed => s.wave_speed = (s.wave_speed + 0.2).min(6.0),
                    KeyCode::ArrowDown if pressed => s.wave_speed = (s.wave_speed - 0.2).max(0.1),
                    KeyCode::ArrowLeft if pressed => s.steer = (s.steer - 0.3).max(-2.0),
                    KeyCode::ArrowRight if pressed => s.steer = (s.steer + 0.3).min(2.0),
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
