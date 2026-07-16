extern crate emerge_engine as emerge;

/// GPU day-night ambient thermal diffusion demo -- a real Fourier's-law heat equation
/// (∂T/∂t = α·∇²T) plus Newton cooling, GPU-ported 2026-07-16. A terrain-like slab
/// starts at a uniform temperature; `set_thermal_ambient` oscillates the ambient
/// temperature on a real sinusoidal day-night cycle, and the slab's own temperature
/// chases it via real diffusion + cooling -- watch it with ColorMode::ByThermal to see
/// warm (day) vs cool (night) spread visibly through the material, not just at a single
/// point.
///
///   cargo run --example day_night_thermal_gpu --features "render"
use std::sync::Arc;

use emerge::render::{ColorMode, Renderer};
use emerge::{
    GpuSimulation, MaterialRegistry, NeoHookeanMaterial, SimConfig, SpawnRegion, build_particles,
};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 48;
const DT: f32 = 0.1;
// Real air-ish thermal constants (see ThermalConfig's own doc for reference values):
// conductivity ~0.5 (between air 0.025 and water 0.6 -- a damp-earth-like slab),
// heat_capacity ~1000 J/(kg*K), grid_cell_size=1.0m (each cell is a real meter).
const CONDUCTIVITY: f32 = 0.5;
const HEAT_CAPACITY: f32 = 1000.0;
const GRID_CELL_SIZE_M: f32 = 1.0;
const COOLING_RATE: f32 = 0.05; // Newton cooling, 1/s
const DAY_AMBIENT: f32 = 35.0;
const NIGHT_AMBIENT: f32 = 5.0;
const CYCLE_SECONDS: f32 = 20.0; // one full day-night cycle, real sim seconds

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuSimulation,
    renderer: Renderer,
    elapsed: f32,
    frame: u64,
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        max_substeps_per_step: 8,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(32, 12),
        box_center: Vec2::new(24.0, 10.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let particles = build_particles(&config, spawn);
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(50.0, 100.0)));
    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    // Start at night_ambient -- the cycle's own first quarter-period will warm it.
    {
        let particles = sim.particles_mut();
        for p in particles.iter_mut() {
            p.temperature = NIGHT_AMBIENT;
        }
    }
    sim.mark_particles_dirty();
    sim.attach_thermal_gpu(
        CONDUCTIVITY,
        HEAT_CAPACITY,
        GRID_CELL_SIZE_M,
        NIGHT_AMBIENT,
        COOLING_RATE,
    );
    sim
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
        let sim = make_sim_data(Arc::new(device), Arc::new(queue));
        let mut renderer = Renderer::new(sim.device(), sim.particle_count(), fmt);
        renderer.set_camera(sim.queue(), GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByThermal);
        println!(
            "day_night_thermal_gpu: {} particles  |  {}s day-night cycle (ambient {}..{})  |  R reset  Q quit",
            sim.particle_count(),
            CYCLE_SECONDS,
            NIGHT_AMBIENT,
            DAY_AMBIENT
        );
        Self {
            surface,
            surface_config: sc,
            sim,
            renderer,
            elapsed: 0.0,
            frame: 0,
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
        self.sim = make_sim_data(device, queue);
        self.elapsed = 0.0;
        self.frame = 0;
        println!("reset");
    }

    fn update_and_render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };

        // Real sinusoidal day-night cycle -- midpoint + amplitude*sin, phase chosen so
        // t=0 starts at night_ambient (matches the slab's own initial temperature).
        self.elapsed += DT;
        let mid = (DAY_AMBIENT + NIGHT_AMBIENT) * 0.5;
        let amp = (DAY_AMBIENT - NIGHT_AMBIENT) * 0.5;
        let phase =
            2.0 * std::f32::consts::PI * self.elapsed / CYCLE_SECONDS - std::f32::consts::FRAC_PI_2;
        let ambient = mid + amp * phase.sin();
        self.sim.set_thermal_ambient(ambient);

        self.sim.step_frame();
        self.frame += 1;
        if self.frame.is_multiple_of(30) {
            self.sim.sync_particles_blocking();
            let avg_t: f32 = self
                .sim
                .particles()
                .iter()
                .map(|p| p.temperature)
                .sum::<f32>()
                / self.sim.particle_count() as f32;
            println!(
                "t={:.1}s ambient={:.1} avg_particle_temp={:.2}",
                self.elapsed, ambient, avg_t
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
                    .with_title("emerge -- Day-Night Thermal Diffusion [GPU]")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 360u32)),
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
