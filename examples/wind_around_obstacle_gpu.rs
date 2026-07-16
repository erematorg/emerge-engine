extern crate emerge_engine as emerge;

/// GPU potential-flow wind field demo -- real closed-form 2D potential flow around a
/// circular cylinder (uniform stream + doublet superposition), GPU-ported 2026-07-16
/// as `GpuFieldEntry::spatial_drag_potential_flow_cylinder`. A cloud of light dust
/// particles gets pushed by the real flow field and visibly deflects around an
/// (invisible, force-only -- no rigid obstacle particles) cylinder rather than
/// plowing straight through it.
///
///   cargo run --example wind_around_obstacle_gpu --features "render"
use std::sync::Arc;

use emerge::gpu::GpuFieldEntry;
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

const GRID: usize = 64;
const DT: f32 = 0.05;
const CYLINDER_CENTER: Vec2 = Vec2::new(32.0, 32.0);
const CYLINDER_RADIUS: f32 = 6.0;
const FREE_STREAM_U: f32 = 6.0;
const DRAG_K: f32 = 1.5;

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuSimulation,
    renderer: Renderer,
    frame: u64,
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        max_substeps_per_step: 8,
        ..SimConfig::standard(GRID, DT, Vec2::ZERO)
    };
    // A wide, thin band of light dust upstream (low x) of the cylinder -- the real
    // potential-flow field pushes it rightward and deflects it around the cylinder.
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(6, 40),
        box_center: Vec2::new(6.0, 32.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let particles = build_particles(&config, spawn);
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(2.0, 4.0)));
    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    sim.add_force_field_gpu(GpuFieldEntry::spatial_drag_potential_flow_cylinder(
        CYLINDER_CENTER,
        FREE_STREAM_U,
        CYLINDER_RADIUS,
        DRAG_K,
        GpuFieldEntry::ALL_MATERIALS,
    ));
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
        renderer.set_color_mode(ColorMode::ByVelocity);
        println!(
            "wind_around_obstacle_gpu: {} particles  |  potential flow around an invisible \
             cylinder (center={CYLINDER_CENTER:?} r={CYLINDER_RADIUS})  |  R reset  Q quit",
            sim.particle_count()
        );
        Self {
            surface,
            surface_config: sc,
            sim,
            renderer,
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
        self.frame = 0;
        println!("reset");
    }

    fn update_and_render(&mut self) {
        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        self.sim.step_frame();
        self.frame += 1;
        if self.frame.is_multiple_of(60) {
            self.sim.sync_particles_blocking();
            let particles = self.sim.particles();
            let centroid_x: f32 =
                particles.iter().map(|p| p.x.x).sum::<f32>() / particles.len() as f32;
            let past_cylinder = particles
                .iter()
                .filter(|p| p.x.x > CYLINDER_CENTER.x + CYLINDER_RADIUS)
                .count();
            println!(
                "frame={} centroid_x={:.2} particles_past_cylinder={past_cylinder}",
                self.frame, centroid_x
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
                    .with_title("emerge -- Wind Around an Obstacle [potential flow, GPU]")
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
