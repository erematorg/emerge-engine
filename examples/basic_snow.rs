extern crate emerge_engine as emerge;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, SimConfig, Simulation, SlipBoundary, SpawnRegion, StomakhinMaterial,
};
use glam::{IVec2, Vec2};
/// CPU snowballs colliding -- Stomakhin 2013 snow plasticity.
///
///   Mat 0  soft powder (blue)  -- low hardening, wide plastic limits
///   Mat 1  packed snow (gold)  -- high hardening, tight limits
///   Mat 2  shatter     (cyan)  -- loose granular after violent impact
///
///   cargo run --example basic_snow --features "render"
use std::sync::Arc;
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
    renderer: Renderer,
    cursor_pos: [f32; 2],
    lmb: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        max_substeps_per_step: 20,
        gravity: Vec2::new(0.0, -0.08),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(58, 58),
        rng_seed: 7,
        ..SpawnRegion::for_sim(&config)
    };
    let mut solver = Simulation::new(config, spawn)
        .with_default_material(Box::new(StomakhinMaterial::new(
            1389.0, 2083.0, 7.0, 0.025, 0.0075, 0.6, 20.0,
        )))
        .with_material(
            MAT_PACKED,
            Box::new(
                StomakhinMaterial::new(1389.0, 2083.0, 10.0, 0.012, 0.004, 0.6, 20.0)
                    .with_cohesion(400.0),
            ),
        )
        .with_material(
            MAT_SHATTER,
            Box::new(DruckerPragerMaterial::low_friction(266.7, 0.333)),
        )
        .with_boundary(Box::new(SlipBoundary::new(config.boundary_thickness)));

    solver.retain_particles(|p| {
        (p.x - BALL_A).length() <= BALL_R || (p.x - BALL_B).length() <= BALL_R
    });
    solver.particles_mut().for_each_mut(|p| {
        if (p.x - BALL_A).length() <= BALL_R {
            p.material_id = MAT_SOFT;
            p.v = Vec2::new(SPEED, 0.0);
        } else {
            p.material_id = MAT_PACKED;
            p.v = Vec2::new(-SPEED, 0.0);
        }
    });
    solver.recompute_initial_volumes();
    solver
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
        let sim = make_sim();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "snow: {} particles  |  LMB push  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
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
        self.surface.configure(&self.device, &self.surface_config);
        self.renderer
            .set_camera(&self.queue, GRID as u32, w, h, 0.6, true);
    }

    fn cursor_grid(&self) -> Vec2 {
        Vec2::new(
            self.cursor_pos[0] / self.surface_config.width as f32 * GRID as f32,
            (1.0 - self.cursor_pos[1] / self.surface_config.height as f32) * GRID as f32,
        )
    }

    fn update_and_render(&mut self) {
        if self.lmb {
            self.sim.apply_radial_impulse(self.cursor_grid(), 6.0, 10.0);
        }
        self.sim.step();
        // Fracture: packed snow hit hard > transitions to loose granular.
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
                    .with_title("emerge -- Snow [Soft Powder / Packed Snow collision]")
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
            } => {
                s.lmb = state == ElementState::Pressed;
            }
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
                    s.sim = make_sim();
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
