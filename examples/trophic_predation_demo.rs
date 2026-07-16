extern crate emerge_engine as emerge;

/// Real trophic/predation demo -- proves `saturating_uptake` (Holling Type II /
/// Michaelis-Menten / Monod, added 2026-07-16 to replace a hardcoded "eat everyone
/// within radius X, instantly" rule with a real, density-driven, continuously
/// rate-limited consumption law). Watch it visually: prey (green) near the predator
/// (red) convert to eaten (dark grey) gradually over many frames, never all at once,
/// because the conversion rate saturates with local prey density instead of being a
/// binary yes/no cutoff -- the same composition already proven in
/// `tests/solver.rs::trophic_predation_depletes_prey_near_predator`, here driven by a
/// real render loop instead of a fixed step count so you can actually watch the rate
/// limiting happen.
///
///   cargo run --example trophic_predation_demo --features "render"
use std::sync::Arc;

use emerge::render::{ColorMode, Renderer};
use emerge::thermodynamics::saturating_uptake;
use emerge::{NeoHookeanMaterial, SimConfig, Simulation, SpawnRegion};
use glam::{IVec2, Vec2};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

const GRID: usize = 64;
const DT: f32 = 0.1;
const PREY_ID: u32 = 0;
const PREDATOR_ID: u32 = 1;
const EATEN_ID: u32 = 2;
const SENSE_RADIUS: f32 = 6.0;
const MAX_CONSUMPTION_RATE: f32 = 8.0; // prey/s at saturating (high) local density
const HALF_SATURATION_DENSITY: f32 = 0.15; // prey per unit area, test-calibrated

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
    eat_budget: f32,
    frame: u64,
}

fn make_sim() -> Simulation {
    let config = SimConfig {
        gravity: Vec2::ZERO,
        ..SimConfig::standard(GRID, DT, Vec2::ZERO)
    };
    let prey_spawn = SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(40, 40),
        box_center: Vec2::new(32.0, 32.0),
        material_id: PREY_ID,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let predator_spawn = SpawnRegion {
        spacing: 0.6,
        box_size: IVec2::new(4, 4),
        box_center: Vec2::new(32.0, 32.0),
        material_id: PREDATOR_ID,
        initial_velocity_scale: 0.0,
        ..SpawnRegion::for_sim(&config)
    };
    let mut sim = Simulation::new(config, prey_spawn)
        .with_default_material(Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_material(PREDATOR_ID, Box::new(NeoHookeanMaterial::new(10.0, 20.0)))
        .with_material(EATEN_ID, Box::new(NeoHookeanMaterial::new(10.0, 20.0)));
    let _ = sim.add_body(predator_spawn);
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
        let sim = make_sim();
        let mut renderer = Renderer::new(&device, sim.particles().len(), fmt);
        renderer.set_camera(&queue, GRID as u32, size.width, size.height, 0.6, true);
        renderer.set_color_mode(ColorMode::ByMaterial);
        println!(
            "trophic_predation_demo: {} particles  |  saturating_uptake (Holling II) \
             predation, real rate-limited, watch prey (green) convert to eaten (grey) \
             gradually  |  R reset  Q quit",
            sim.particles().len()
        );
        Self {
            surface,
            surface_config: sc,
            device,
            queue,
            sim,
            renderer,
            eat_budget: 0.0,
            frame: 0,
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

    fn reset(&mut self) {
        self.sim = make_sim();
        self.eat_budget = 0.0;
        self.frame = 0;
        println!("reset");
    }

    fn update_and_render(&mut self) {
        // Real predation step: gather predator positions, compute local prey density,
        // convert that into a saturating (Holling II) consumption rate, accumulate a
        // budget, and convert only as many prey as the budget allows -- exactly the
        // composition tests/solver.rs proves, driven every frame here instead of a
        // fixed step count.
        let sense_area = std::f32::consts::PI * SENSE_RADIUS * SENSE_RADIUS;
        let predator_positions: Vec<Vec2> = self
            .sim
            .particles()
            .iter()
            .filter(|p| p.material_id == PREDATOR_ID)
            .map(|p| p.x)
            .collect();
        let mut nearby_prey: Vec<usize> = predator_positions
            .iter()
            .flat_map(|&pp| self.sim.particles_near(pp, SENSE_RADIUS))
            .filter(|&i| self.sim.particles().get(i).material_id == PREY_ID)
            .collect();
        nearby_prey.sort_unstable();
        nearby_prey.dedup();

        let local_density = nearby_prey.len() as f32 / sense_area;
        let rate = saturating_uptake(local_density, MAX_CONSUMPTION_RATE, HALF_SATURATION_DENSITY);
        self.eat_budget += rate * DT;
        let to_eat = (self.eat_budget.floor() as usize).min(nearby_prey.len());
        self.eat_budget -= to_eat as f32;
        {
            let particles = self.sim.particles_mut();
            for &i in nearby_prey.iter().take(to_eat) {
                particles.material_id[i] = EATEN_ID;
            }
        }

        self.sim.step();
        self.frame += 1;
        if self.frame.is_multiple_of(30) {
            let particles = self.sim.particles();
            let prey = particles
                .iter()
                .filter(|p| p.material_id == PREY_ID)
                .count();
            let eaten = particles
                .iter()
                .filter(|p| p.material_id == EATEN_ID)
                .count();
            println!(
                "frame={} prey_remaining={prey} eaten={eaten} local_density={local_density:.3} rate={rate:.3}/s",
                self.frame
            );
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
                    .with_title("emerge -- Trophic Predation [saturating_uptake / Holling II]")
                    .with_inner_size(winit::dpi::LogicalSize::new(480u32, 480u32)),
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
