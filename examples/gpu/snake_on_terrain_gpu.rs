extern crate emerge_engine as emerge;

#[path = "../scenes/snake_on_terrain.rs"]
mod scene;

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

use emerge::render::{ColorMode, GpuRenderParams, Renderer};
use emerge::{
    FixedStepController, GpuSimulation, Lnn, MaterialRegistry, SimConfig, build_particles,
};
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{Window, WindowId};

fn make_cpg() -> Lnn {
    let mut lnn = Lnn::coupled_traveling_wave(
        scene::N_RINGS,
        scene::N_PER_RING,
        1.0,
        scene::RING_CROSS_COUPLING,
    );
    for _ in 0..scene::CPG_BURN_IN_STEPS {
        lnn.step(scene::DT);
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
    // Converts real measured elapsed time into the correct number of physics
    // steps per frame -- calling `sim.step_frame()` once per render frame
    // assumes each frame takes exactly `DT` of real time, which it doesn't
    // (render frame rate varies), and produces jitter/inconsistent pacing.
    stepper: FixedStepController,
    last_instant: std::time::Instant,
}

fn make_sim_data(
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
) -> (GpuSimulation, std::ops::Range<usize>, Vec<u32>) {
    // GPU has no `DirectionalContactGrip` equivalent yet (see this file's own
    // top-of-file doc) -- `contact_friction` is the real, disclosed stand-in,
    // layered on top of the shared base config rather than in it, since CPU's
    // own config doesn't use this field at all.
    let config = SimConfig {
        contact_friction: 0.5,
        ..scene::base_config()
    };

    let terrain_particles = build_particles(&config, scene::terrain_spawn(&config));
    let registry = MaterialRegistry::with_default(Box::new(scene::terrain_material()));
    let mut sim = GpuSimulation::with_device(device, queue, config, terrain_particles, registry);

    let snake_mat_id = sim.register_material(Box::new(scene::snake_material()));
    let snake_spawn = scene::snake_spawn(sim.config(), snake_mat_id.id());
    let snake_range = sim.spawn_region(snake_spawn);

    let mut muscle_group_of = Vec::with_capacity(snake_range.len());
    {
        let particles = sim.particles_mut();
        for i in snake_range.clone() {
            particles[i].contact_group = scene::SNAKE_CONTACT_GROUP;
            let (group, dir) = scene::snake_particle_tag(particles[i].x);
            particles[i].muscle_group_id = group;
            muscle_group_of.push(group);
            particles[i].activation_dir = dir;
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
        renderer.set_camera(
            sim.queue(),
            scene::GRID as u32,
            size.width,
            size.height,
            0.6,
            true,
        );
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
            stepper: FixedStepController::standard(scene::DT, 1.0 / scene::DT),
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
            .set_camera(self.sim.queue(), scene::GRID as u32, w, h, 0.6, true);
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
                self.lnn.step(scene::DT * self.wave_speed);
                let activations: Vec<f32> = self.lnn.activations().collect();
                {
                    let particles = self.sim.particles_mut();
                    for (offset, i) in self.snake_range.clone().enumerate() {
                        let group = self.muscle_group_of[offset] as usize;
                        particles[i].activation =
                            (scene::MUSCLE_AMPLITUDE * activations[group]).clamp(0.0, 1.0);
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
            GpuRenderParams {
                particle_buf: self.sim.particle_buffer(),
                particle_count: self.sim.particle_count(),
                output_view: &view,
                clear: true,
                interp_alpha: 1.0,
            },
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
