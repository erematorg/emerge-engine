extern crate emerge_engine as emerge;

use emerge::diagnostics::log_frame_gpu;
use emerge::gpu::GpuSimulation;
use emerge::render::{ColorMode, Renderer};
use emerge::{
    DruckerPragerMaterial, MaterialRegistry, NeoHookeanMaterial, NewtonianFluidMaterial, SimConfig,
    SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
/// GPU three-material showcase -- sand terrain, fluid pool, elastic blob.
///
///   Mat 0  NeoHookean elastic (blue)  -- creature body, arrow-key drive
///   Mat 1  Sand Drucker-Prager (gold) -- terrain
///   Mat 2  Newtonian fluid  (cyan)    -- water pool
///
///   arrow keys  drive elastic blob  |  LMB push  RMB pull  |  R reset  Q quit
///   cargo run --example basic_showcase_gpu --features "render,gpu"
use std::sync::Arc;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const ELASTIC_ID: u32 = 0;
const SAND_ID: u32 = 1;
const FLUID_ID: u32 = 2;
const SPACING: f32 = 0.7;
const LABELS: &[(u32, &str)] = &[
    (ELASTIC_ID, "elastic"),
    (SAND_ID, "sand"),
    (FLUID_ID, "fluid"),
];

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    sim: GpuSimulation,
    renderer: Renderer,
    cursor_pos: [f32; 2],
    physics_colors: bool,
    lmb: bool,
    rmb: bool,
    arrow_up: bool,
    arrow_down: bool,
    arrow_left: bool,
    arrow_right: bool,
    frame: u64,
    fps_timer: std::time::Instant,
    fps_frames: u64,
}

fn make_sim(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        min_dt: 0.005,
        max_substeps_per_step: 16,
        recompute_density_each_step: true,
        gravity: Vec2::new(0.0, -0.3),
        ..SimConfig::earth(GRID, 0.01, DT)
    };
    let mut p = build_particles(
        &config,
        SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(22, 14),
            box_center: Vec2::new(19.0, 9.0),
            material_id: SAND_ID,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        },
    );
    p.extend(build_particles(
        &config,
        SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(22, 14),
            box_center: Vec2::new(45.0, 9.0),
            material_id: FLUID_ID,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        },
    ));
    p.extend(build_particles(
        &config,
        SpawnRegion {
            spacing: SPACING,
            box_size: IVec2::new(12, 12),
            box_center: Vec2::new(32.0, 46.0),
            material_id: ELASTIC_ID,
            precompute_initial_volumes: true,
            ..SpawnRegion::for_sim(&config)
        },
    ));
    let elastic = NeoHookeanMaterial::new(40.0, 80.0);
    let sand = DruckerPragerMaterial::new(400.0, 200.0);
    // Real water: Cole 1948 Tait exponent (7.0) + real dynamic viscosity, not a
    // hand-picked 0.1/4.0 pair -- see NewtonianFluidMaterial::low_viscosity.
    let fluid = NewtonianFluidMaterial::low_viscosity(4.0, 10.0);
    let mut reg = MaterialRegistry::with_default(Box::new(elastic));
    reg.insert(SAND_ID, Box::new(sand));
    reg.insert(FLUID_ID, Box::new(fluid));
    GpuSimulation::with_device(device, queue, config, p, reg)
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
        let device = Arc::new(device);
        let queue = Arc::new(queue);
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
        let sim = make_sim(device.clone(), queue.clone());
        let mut renderer = Renderer::new(&device, sim.particle_count(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByPhysics);
        println!(
            "showcase_gpu: {} particles  |  arrow keys=drive blob  LMB/RMB push/pull  R reset  Q quit",
            sim.particle_count()
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
            physics_colors: true,
            rmb: false,
            arrow_up: false,
            arrow_down: false,
            arrow_left: false,
            arrow_right: false,
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
        // Arrow-key drive on elastic blob centroid.
        let mut dir = Vec2::ZERO;
        if self.arrow_up {
            dir.y += 1.0;
        }
        if self.arrow_down {
            dir.y -= 1.0;
        }
        if self.arrow_left {
            dir.x -= 1.0;
        }
        if self.arrow_right {
            dir.x += 1.0;
        }
        if dir != Vec2::ZERO {
            let impulse = dir.normalize() * 10.0;
            let particles = self.sim.particles();
            let (sum, n) = particles
                .iter()
                .filter(|p| p.material_id == ELASTIC_ID)
                .fold((Vec2::ZERO, 0usize), |(s, n), p| (s + p.x, n + 1));
            if n > 0 {
                let centroid = sum / n as f32;
                self.sim.apply_impulse(centroid, 12.0, impulse);
            }
        }

        if self.lmb || self.rmb {
            let mag = if self.lmb { 2.0 } else { -2.0 };
            self.sim.apply_radial_impulse(self.cursor_grid(), 5.0, mag);
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        self.sim.step_frame();
        self.frame += 1;
        self.fps_frames += 1;
        if self.fps_timer.elapsed().as_secs_f32() >= 2.0 {
            let fps = self.fps_frames as f32 / self.fps_timer.elapsed().as_secs_f32();
            println!("frame={} fps={:.0}", self.frame, fps);
            self.fps_timer = std::time::Instant::now();
            self.fps_frames = 0;
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
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.renderer.render_gpu(
            &self.device,
            &self.queue,
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
                    .with_title("emerge -- Showcase GPU [Sand / Fluid / Elastic]")
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
                    KeyCode::KeyG if pressed => {
                        s.physics_colors = !s.physics_colors;
                        s.renderer.set_color_mode(if s.physics_colors {
                            ColorMode::ByPhysics
                        } else {
                            ColorMode::ByMaterial
                        });
                    }
                    KeyCode::KeyR if pressed => {
                        s.sim = make_sim(s.device.clone(), s.queue.clone());
                        s.frame = 0;
                        println!("reset");
                    }
                    KeyCode::ArrowUp => s.arrow_up = pressed,
                    KeyCode::ArrowDown => s.arrow_down = pressed,
                    KeyCode::ArrowLeft => s.arrow_left = pressed,
                    KeyCode::ArrowRight => s.arrow_right = pressed,
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
