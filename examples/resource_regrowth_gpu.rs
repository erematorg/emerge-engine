extern crate emerge_engine as emerge;

/// GPU resource regrowth demo -- real logistic-growth PDE (Verhulst 1838,
/// GPU-ported 2026-07-17) combined with `saturating_uptake` consumption (same
/// composition CPU's `resource_field_depletes_near_consumer_then_regrows` proves).
/// A field of "grass" starts at full resource (bright); a stationary consumer
/// depletes nearby resource at a real, rate-limited pace via `particles_near` +
/// `saturating_uptake` (external, same as the trophic predation demo -- the GPU port
/// itself only owns the real regrowth PDE, not the consumption rule, matching CPU's
/// own scope split). Watch resource dim near the consumer, then hold Space to stop
/// consuming and watch it regrow back via the real PDE alone.
///
///   cargo run --example resource_regrowth_gpu --features "render"
use std::sync::Arc;

use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::saturating_uptake;
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
const RESOURCE_K: f32 = 1.0;
const RESOURCE_R: f32 = 0.5;
const CONSUMER_POS: Vec2 = Vec2::new(14.0, 10.0);
const SENSE_RADIUS: f32 = 6.0;
const MAX_CONSUMPTION_RATE: f32 = 0.3;
const HALF_SATURATION_DENSITY: f32 = 0.3;

struct App {
    window: Option<Arc<Window>>,
    state: Option<State>,
}

struct State {
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    sim: GpuSimulation,
    renderer: Renderer,
    consuming: bool,
    frame: u64,
}

fn make_sim_data(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> GpuSimulation {
    let config = SimConfig {
        max_substeps_per_step: 8,
        ..SimConfig::standard(GRID, DT, Vec2::new(0.0, -0.3))
    };
    let spawn = SpawnRegion {
        spacing: 0.5,
        box_size: IVec2::new(40, 12),
        box_center: Vec2::new(24.0, 10.0),
        material_id: 0,
        precompute_initial_volumes: true,
        ..SpawnRegion::for_sim(&config)
    };
    let particles = build_particles(&config, spawn);
    let registry = MaterialRegistry::with_default(Box::new(NeoHookeanMaterial::new(50.0, 100.0)));
    let mut sim = GpuSimulation::with_device(device, queue, config, particles, registry);
    {
        let particles = sim.particles_mut();
        for p in particles.iter_mut() {
            p.scalar_field = RESOURCE_K;
        }
    }
    sim.mark_particles_dirty();
    sim.attach_resource_field_gpu(0.0, RESOURCE_K, RESOURCE_R, RESOURCE_K);
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
        renderer.set_color_mode(ColorMode::ByScalarField);
        println!(
            "resource_regrowth_gpu: {} particles  |  real logistic regrowth + saturating_uptake \
             consumption  |  Space toggle consumer  R reset  Q quit",
            sim.particle_count()
        );
        Self {
            surface,
            surface_config: sc,
            sim,
            renderer,
            consuming: true,
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
        self.consuming = true;
        self.frame = 0;
        println!("reset");
    }

    fn update_and_render(&mut self) {
        if self.consuming {
            // Real consumption: Holling Type II / saturating_uptake applied per-particle
            // on THAT particle's own phi (same structure as CPU's own verified
            // `resource_field_depletes_near_consumer_then_regrows`, tests/solver.rs) --
            // rate naturally -> 0 as phi -> 0, so depletion decelerates near zero instead
            // of a flat/aggregate budget driving everything to a hard clamp.
            //
            // Skip the readback on frame 0: the CPU mirror already holds the correct
            // freshly-set scalar_field=RESOURCE_K (no step_frame has uploaded/evolved
            // anything yet) -- syncing here would instead download the GPU's stale
            // pre-upload buffer (spawn-time scalar_field=0.0) and clobber it before the
            // real upload (queued by `mark_particles_dirty` in `make_sim_data`) ever runs.
            if self.frame > 0 {
                self.sim.sync_particles_blocking();
            }
            let nearby: Vec<usize> = self
                .sim
                .particles_near(CONSUMER_POS, SENSE_RADIUS)
                .map(|(i, _)| i)
                .collect();
            if !nearby.is_empty() {
                let particles = self.sim.particles_mut();
                for &i in &nearby {
                    let phi = particles[i].scalar_field;
                    let rate =
                        saturating_uptake(phi, MAX_CONSUMPTION_RATE, HALF_SATURATION_DENSITY);
                    particles[i].scalar_field = (phi - rate * DT).max(0.0);
                }
                self.sim.mark_particles_dirty();
            }
        }

        let output = match self.surface.get_current_texture() {
            Ok(t) => t,
            Err(_) => return,
        };
        self.sim.step_frame();
        self.frame += 1;
        if self.frame.is_multiple_of(30) {
            self.sim.sync_particles_blocking();
            let particles = self.sim.particles();
            let near: Vec<f32> = self
                .sim
                .particles_near(CONSUMER_POS, SENSE_RADIUS)
                .map(|(i, _)| particles[i].scalar_field)
                .collect();
            let near_avg = if near.is_empty() {
                0.0
            } else {
                near.iter().sum::<f32>() / near.len() as f32
            };
            println!(
                "frame={} consuming={} near_consumer_avg={near_avg:.3}",
                self.frame, self.consuming
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
                    .with_title(
                        "emerge -- Resource Regrowth [logistic PDE + saturating_uptake, GPU]",
                    )
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
                KeyCode::Space => {
                    s.consuming = !s.consuming;
                    println!("consuming = {}", s.consuming);
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
